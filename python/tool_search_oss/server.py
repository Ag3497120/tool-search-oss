"""
server.py — tool-search-oss MCP server (Python)

Exposes one tool to the LLM:

    discover_tools(task_description: str) -> str

The LLM calls this with what it wants to do.
The server returns the top-5 matching tool schemas in clean JSON + Markdown.

Startup:
    # From a JSON catalog file
    python -m tool_search_oss.server --catalog tools.json

    # From a directory of JSON schema files
    python -m tool_search_oss.server --catalog-dir ./schemas/

    # Inline (for testing)
    python -m tool_search_oss.server --inline '[{"name":"read_file",...}]'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from .router import CascadeToolRouter, ToolSchema


# ──────────────────────────────────────────────────────────────────────────────
# Catalog loader
# ──────────────────────────────────────────────────────────────────────────────

def load_catalog(
    catalog_file: str | None = None,
    catalog_dir: str | None = None,
    inline: str | None = None,
) -> list[ToolSchema]:
    """Load tool schemas from file, directory, or inline JSON."""

    if inline:
        return json.loads(inline)

    if catalog_file:
        with open(catalog_file) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("tools", [])

    if catalog_dir:
        tools: list[ToolSchema] = []
        for p in Path(catalog_dir).glob("*.json"):
            with open(p) as f:
                schema = json.load(f)
            if isinstance(schema, list):
                tools.extend(schema)
            elif "name" in schema:
                tools.append(schema)
        return tools

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Formatting helpers
# ──────────────────────────────────────────────────────────────────────────────

def format_results(
    results: list[ToolSchema],
    intent: str,
    total: int,
) -> str:
    """Format search results as clean Markdown + JSON for any LLM."""

    if not results:
        return (
            f"TOOL_SEARCH: no matches for '{intent}'\n"
            f"TOTAL_TOOLS: {total}\n"
            "TIP: Try broader terms. Call discover_tools with simpler intent.\n"
        )

    lines = [
        f"TOOL_SEARCH: {len(results)} of {total} tools matched",
        f"QUERY: {intent}",
        f"CONTEXT_SAVED: ~{int((1 - len(results)/max(total,1)) * 100)}%",
        "",
    ]

    for i, tool in enumerate(results, 1):
        score   = tool.get("_score", 0)
        matched = tool.get("_matched_terms", [])
        clean   = {k: v for k, v in tool.items() if not k.startswith("_")}

        lines += [
            f"--- MATCH {i}: {tool['name']} (score: {score:.2f}, terms: {', '.join(matched)})",
            f"```json",
            json.dumps(clean, indent=2, ensure_ascii=False),
            f"```",
            "",
        ]

    lines += [
        f"[NEXT STEP] Call `{results[0]['name']}` — it is the best match.",
        "If it doesn't fit, try the second match or call discover_tools with a more specific intent.",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# MCP Server
# ──────────────────────────────────────────────────────────────────────────────

def build_server(tools: list[ToolSchema]) -> Server:
    router = CascadeToolRouter()
    app    = Server("tool-search-oss")

    # Pre-index for repeated queries
    if tools:
        router._bm25.index(tools)

    # ── Tool definitions ──────────────────────────────────────────────────────

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="discover_tools",
                description=(
                    "Find the right MCP tool from a large catalog using BM25 search. "
                    "Pass what you want to do in natural language. "
                    "Returns only the top matching tool definitions — not all of them. "
                    "This is the defer_loading pattern: ~85% less context, same accuracy. "
                    "Example: discover_tools('send a slack message')"
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_description": {
                            "type": "string",
                            "description": (
                                "What you want to accomplish. Natural language. "
                                "e.g. 'read a file from disk', 'push code to GitHub', "
                                "'search memory for recent events'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of tools to return (default: 5, max: 10)",
                            "default": 5,
                        },
                    },
                    "required": ["task_description"],
                },
            ),
            types.Tool(
                name="catalog_summary",
                description=(
                    "Get a lightweight summary of all available tools grouped by category. "
                    "Call this ONCE at session start instead of loading all full definitions. "
                    "Then use discover_tools() to fetch full schemas for what you need. "
                    "Pattern: catalog_summary → discover_tools → call the actual tool."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    # ── Handlers ──────────────────────────────────────────────────────────────

    @app.call_tool()
    async def call_tool(
        name: str,
        arguments: dict[str, Any],
    ) -> list[types.TextContent]:

        if name == "discover_tools":
            intent = arguments.get("task_description", "").strip()
            top_k  = min(int(arguments.get("top_k", 5)), 10)

            if not intent:
                return [types.TextContent(
                    type="text",
                    text="ERROR: task_description is required. "
                         "Example: discover_tools('read a file from disk')",
                )]

            results = router.find_candidates(intent, tools, top_k)
            text    = format_results(results, intent, len(tools))
            return [types.TextContent(type="text", text=text)]

        if name == "catalog_summary":
            text = router.summarize(tools)
            return [types.TextContent(type="text", text=text)]

        return [types.TextContent(
            type="text",
            text=f"Unknown tool: {name}",
        )]

    return app


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

async def main(tools: list[ToolSchema]) -> None:
    app = build_server(tools)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        from mcp.server.models import InitializationOptions
        from mcp.types import ServerCapabilities, ToolsCapability
        try:
            # MCP SDK >= 1.2: NotificationOptions required
            from mcp.server.lowlevel.server import NotificationOptions
            notification_options = NotificationOptions()
        except ImportError:
            notification_options = None  # older SDK

        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="tool-search-oss",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=notification_options,
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="tool-search-oss MCP server")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--catalog",     help="Path to JSON catalog file")
    group.add_argument("--catalog-dir", help="Directory of JSON schema files")
    group.add_argument("--inline",      help="Inline JSON tool list (for testing)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    catalog = load_catalog(args.catalog, args.catalog_dir, args.inline)

    if not catalog:
        print(
            "WARNING: No tools loaded. Pass --catalog, --catalog-dir, or --inline.\n"
            "Example: python -m tool_search_oss.server --catalog examples/tools.json",
            file=sys.stderr,
        )

    asyncio.run(main(catalog))
