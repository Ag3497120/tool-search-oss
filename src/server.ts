/**
 * server.ts — tool-search-oss
 *
 * v0.1: BM25-based tool search (defer_loading pattern)
 * v0.2: Visual Routing Engine (experimental, already included below)
 *
 * Two tools exposed in v0.1:
 *   search_tools(query, tools, topK?)
 *     → BM25 search, returns top-N full tool definitions
 *     → This is the "defer loading" pattern: LLM only loads
 *       the definitions it actually needs
 *
 *   tool_summary(tools)
 *     → Lightweight catalog of all tools (name + 1-line description)
 *     → Give this to LLM at session start instead of all full definitions
 *
 * v0.2 (experimental, opt-in):
 *   visual_route(query, tools, currentTool?)
 *     → Renders tool relationships as SVG/PNG routing graph
 *     → LLM selects next tool from highlighted visual nodes
 *     → Prevents probabilistic drift via vision encoder
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";
import { ToolSearchEngine, type ToolDefinition } from "./search.js";
import { renderToolMap, toBase64, type LubricantTool } from "./renderer.js";

const server = new Server(
    { name: "tool-search-oss", version: "0.1.0" },
    { capabilities: { tools: {} } }
);

// ─────────────────────────────────────────────────────────────────────────────
// Tool schema (shared by search_tools and tool_summary)
// ─────────────────────────────────────────────────────────────────────────────

const TOOL_ITEM_SCHEMA = {
    type: "object",
    properties: {
        name:        { type: "string", description: "Tool name" },
        description: { type: "string", description: "What this tool does" },
        category:    { type: "string", description: "Optional category hint" },
        tags:        { type: "array", items: { type: "string" }, description: "Optional tags" },
        inputSchema: { type: "object", description: "Full MCP input schema (included in search results)" },
    },
    required: ["name", "description"],
} as const;

// ─────────────────────────────────────────────────────────────────────────────
// ListTools
// ─────────────────────────────────────────────────────────────────────────────

server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
        // ── v0.1 ──────────────────────────────────────────────────────────────
        {
            name: "search_tools",
            description: [
                "Find the right tool from a large MCP catalog using BM25 search.",
                "Instead of loading all 50+ tool definitions into context (expensive),",
                "call this with what you want to do and get only the top-N relevant tools back.",
                "This is the defer_loading pattern: ~85% context savings with no accuracy loss.",
            ].join(" "),
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "What you want to do in natural language. e.g. 'read a file', 'send a message', 'search memory'",
                    },
                    tools: {
                        type: "array",
                        description: "Your full MCP tool catalog. Pass all tools here — only top matches are returned.",
                        items: TOOL_ITEM_SCHEMA,
                    },
                    topK: {
                        type: "number",
                        description: "Number of results to return (default: 5, max: 10)",
                    },
                },
                required: ["query", "tools"],
            },
        },
        {
            name: "tool_summary",
            description: [
                "Get a lightweight catalog of all available tools (name + 1-line description only).",
                "Call this ONCE at session start instead of loading full tool definitions.",
                "Then use search_tools() to fetch full definitions for only what you need.",
                "Pattern: tool_summary → search_tools → call the actual tool.",
            ].join(" "),
            inputSchema: {
                type: "object",
                properties: {
                    tools: {
                        type: "array",
                        description: "Your full MCP tool catalog.",
                        items: TOOL_ITEM_SCHEMA,
                    },
                },
                required: ["tools"],
            },
        },
        // ── v0.2 experimental ─────────────────────────────────────────────────
        {
            name: "visual_route",
            description: [
                "[EXPERIMENTAL v0.2] Visual Routing Engine.",
                "Renders MCP tool relationships as a routing graph (SVG/PNG image).",
                "LLM selects the next tool from spatially highlighted visual nodes",
                "instead of making a probabilistic text-based guess.",
                "Yellow nodes = best match. Red = current. Gray = not relevant.",
                "Eliminates 'Lost in the Middle' and sycophantic tool selection.",
            ].join(" "),
            inputSchema: {
                type: "object",
                properties: {
                    query: {
                        type: "string",
                        description: "What you want to do. Best-matching tools will be highlighted in yellow.",
                    },
                    tools: {
                        type: "array",
                        items: TOOL_ITEM_SCHEMA,
                        description: "Your MCP tool catalog.",
                    },
                    currentTool: {
                        type: "string",
                        description: "Currently executing tool name (marked in red).",
                    },
                },
                required: ["query", "tools"],
            },
        },
    ],
}));

// ─────────────────────────────────────────────────────────────────────────────
// CallTool handlers
// ─────────────────────────────────────────────────────────────────────────────

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    // ── search_tools ──────────────────────────────────────────────────────────
    if (name === "search_tools") {
        const { query, tools, topK = 5 } = args as {
            query: string;
            tools: ToolDefinition[];
            topK?: number;
        };
        try {
            const engine  = new ToolSearchEngine(tools);
            const k       = Math.min(Number(topK) || 5, 10);
            const results = engine.search(query, k);

            if (results.length === 0) {
                return {
                    content: [{
                        type: "text",
                        text: [
                            `TOOL_SEARCH: no results for "${query}"`,
                            `TOTAL_TOOLS: ${tools.length}`,
                            `TIP: Try broader terms, or call tool_summary() to browse the full catalog.`,
                        ].join("\n"),
                    }],
                };
            }

            // Full definitions of matching tools only
            const matchesText = results.map((r, i) =>
                [
                    `--- MATCH ${i + 1}: ${r.tool.name} (score: ${r.score.toFixed(2)})`,
                    `Description: ${r.tool.description}`,
                    `Matched terms: ${r.matchedTerms.join(", ")}`,
                    r.tool.inputSchema
                        ? `Input schema: ${JSON.stringify(r.tool.inputSchema, null, 2)}`
                        : "",
                ].filter(Boolean).join("\n")
            ).join("\n\n");

            return {
                content: [{
                    type: "text",
                    text: [
                        `TOOL_SEARCH: found ${results.length} of ${tools.length} tools`,
                        `QUERY: ${query}`,
                        ``,
                        matchesText,
                        ``,
                        `[Next step] Call the top-matching tool: ${results[0].tool.name}`,
                    ].join("\n"),
                }],
            };
        } catch (e: any) {
            return { isError: true, content: [{ type: "text", text: `search_tools error: ${e.message}` }] };
        }
    }

    // ── tool_summary ──────────────────────────────────────────────────────────
    if (name === "tool_summary") {
        const { tools } = args as { tools: ToolDefinition[] };
        try {
            const engine = new ToolSearchEngine(tools);

            // カテゴリ別にグループ化
            const { inferCategory } = await import("./renderer.js");
            const groups: Record<string, string[]> = {};
            for (const t of tools) {
                const cat = t.category ?? inferCategory(t as any);
                if (!groups[cat]) groups[cat] = [];
                groups[cat].push(`  ${t.name}: ${t.description.slice(0, 70)}`);
            }

            const catalog = Object.entries(groups)
                .map(([cat, lines]) => `[${cat.toUpperCase()}]\n${lines.join("\n")}`)
                .join("\n\n");

            return {
                content: [{
                    type: "text",
                    text: [
                        `TOOL_SUMMARY: ${tools.length} tools available`,
                        ``,
                        catalog,
                        ``,
                        `[Pattern] Call search_tools(query, tools, topK) to get full definitions for relevant tools only.`,
                        `[Visual]  Call visual_route(query, tools) to see a routing graph (experimental).`,
                    ].join("\n"),
                }],
            };
        } catch (e: any) {
            return { isError: true, content: [{ type: "text", text: `tool_summary error: ${e.message}` }] };
        }
    }

    // ── visual_route (v0.2 experimental) ─────────────────────────────────────
    if (name === "visual_route") {
        const { query, tools, currentTool } = args as {
            query: string;
            tools: LubricantTool[];
            currentTool?: string;
        };
        try {
            const svg = renderToolMap(tools, {
                query,
                currentTool,
                maxVisible: 24,
                showEdges: true,
            });
            const { base64, mimeType } = await toBase64(svg);

            // Text fallback via BM25
            const engine  = new ToolSearchEngine(tools as ToolDefinition[]);
            const top3    = engine.search(query, 3);

            return {
                content: [
                    { type: "image", data: base64, mimeType },
                    {
                        type: "text",
                        text: [
                            `VISUAL_ROUTE [v0.2 experimental]`,
                            `QUERY: ${query}`,
                            `BM25_TOP3: ${top3.map(r => r.tool.name).join(" → ")}`,
                            ``,
                            `[Instruction] Look at the image.`,
                            `YELLOW nodes = best match for your query. Call that tool next.`,
                            `RED node = where you are now.`,
                            `Output only the tool name.`,
                        ].join("\n"),
                    },
                ],
            };
        } catch (e: any) {
            return { isError: true, content: [{ type: "text", text: `visual_route error: ${e.message}` }] };
        }
    }

    throw new Error(`Unknown tool: ${name}`);
});

// ─────────────────────────────────────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────────────────────────────────────

const transport = new StdioServerTransport();
await server.connect(transport);
