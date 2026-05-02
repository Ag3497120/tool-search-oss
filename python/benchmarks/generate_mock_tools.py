"""
generate_mock_tools.py — Synthetic Enterprise MCP Tool Catalog Generator

Generates a realistic 2000-tool catalog for benchmarking tool routers.

Why synthetic? The goal of tool routing benchmarks is to test:
  1. BM25/semantic search accuracy (does the router find the right tool?)
  2. LLM TTFT at scale (does context size kill response time?)

Neither requires tools that actually *execute*. Only the JSON schemas matter.

Design principles:
  - Enterprise-realistic naming: {service}_{action}_{resource}
  - Homonym noise: jira_create_user vs salesforce_create_user vs okta_create_user
    → Forces BM25 to differentiate by scoring, not just keyword hit
  - Needle-in-haystack: 5 specific tools hidden among 1995 noisy ones
    → Validates BM25 at worst-case enterprise scale

Usage:
    python3 generate_mock_tools.py
    python3 generate_mock_tools.py --count 2000 --out tools_2000.json --seed 42

Outputs:
    tools_2000.json  — full catalog (load with MCP server or bench scripts)
    needles.json     — just the 5 target tools (for accuracy verification)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Enterprise keyword corpus
# ─────────────────────────────────────────────────────────────────────────────

SERVICES = [
    "aws", "gcp", "azure", "jira", "salesforce", "slack", "github",
    "notion", "datadog", "splunk", "stripe", "zendesk", "okta",
    "hubspot", "servicenow", "pagerduty", "cloudflare", "snowflake",
    "databricks", "confluent", "vault", "consul", "nomad", "terraform",
    "ansible", "puppet", "chef", "newrelic", "dynatrace", "sumologic",
]

ACTIONS = [
    "create", "read", "update", "delete", "list", "search",
    "sync", "export", "import", "grant", "revoke", "rotate",
    "enable", "disable", "archive", "restore", "validate", "audit",
]

RESOURCES = [
    "user", "ticket", "invoice", "repository", "dashboard", "alert",
    "message", "document", "payment", "policy", "server", "bucket",
    "database", "record", "token", "certificate", "secret", "pipeline",
    "workflow", "deployment", "instance", "cluster", "namespace",
    "permission", "role", "group", "organization", "project", "team",
]

# Description templates — vary to avoid identical text scores
DESC_TEMPLATES = [
    "Perform {action} operation on {resource} within the {service} enterprise platform. "
    "Use this for standard {service} {resource} management.",

    "{action_cap} a {resource} in {service}. Required for {service}-based {resource} "
    "lifecycle management.",

    "Execute {action} on {service} {resource} entities. "
    "Returns updated {resource} state upon completion.",

    "{action_cap} {service} {resource} records. "
    "Supports batch operations and dry_run mode for safe testing.",

    "Interface with {service} API to {action} {resource} objects. "
    "Requires appropriate {service} IAM permissions.",
]


def _desc(service: str, action: str, resource: str, template_idx: int) -> str:
    t = DESC_TEMPLATES[template_idx % len(DESC_TEMPLATES)]
    return t.format(
        service=service,
        action=action,
        resource=resource,
        action_cap=action.capitalize(),
    )


def _schema(resource: str, action: str) -> dict:
    props: dict = {
        f"{resource}_id": {
            "type": "string",
            "description": f"The unique identifier for the {resource}",
        },
    }
    if action in ("create", "update"):
        props["payload"] = {
            "type": "object",
            "description": f"Attributes to set on the {resource}",
        }
    if action in ("list", "search"):
        props["query"] = {
            "type": "string",
            "description": f"Filter expression for {resource} search",
        }
        props["limit"] = {
            "type": "integer",
            "description": "Maximum number of results (default: 50)",
        }
    props["dry_run"] = {
        "type": "boolean",
        "description": "Run without making actual changes (default: false)",
    }
    required = [f"{resource}_id"] if action not in ("create", "list", "search") else []
    return {"type": "object", "properties": props, "required": required}


# ─────────────────────────────────────────────────────────────────────────────
# Needle-in-haystack: 5 specific tools the benchmarks search for
# ─────────────────────────────────────────────────────────────────────────────
#
# These are intentionally different from the noisy bulk tools:
#   - Unique action verbs not in ACTIONS list
#   - Descriptions that use domain-specific language
#   - Testing that BM25 finds them despite 1995 distractors

NEEDLES = [
    {
        "name": "local_file_reader",
        "description": (
            "Read the exact text contents of a local file from the disk. "
            "Useful for reading configuration files or source code. "
            "Returns raw file content as a string."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute or relative path to the file on disk",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                },
            },
            "required": ["file_path"],
        },
        "_needle": True,
        "_query": "read a config file from disk",
    },
    {
        "name": "slack_emergency_broadcast",
        "description": (
            "Send a critical emergency alert message to the global engineering "
            "Slack channel. Bypasses DND settings. Use only for P0 incidents."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "alert_text": {
                    "type": "string",
                    "description": "The emergency message content",
                },
                "severity": {
                    "type": "string",
                    "enum": ["P0", "P1"],
                    "description": "Incident severity level",
                },
            },
            "required": ["alert_text"],
        },
        "_needle": True,
        "_query": "send a critical emergency alert to slack",
    },
    {
        "name": "postgres_execute_raw_sql",
        "description": (
            "Execute a raw SQL query directly against the primary PostgreSQL database. "
            "Returns result rows as JSON. Supports SELECT, INSERT, UPDATE, DELETE."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Raw SQL query string to execute",
                },
                "params": {
                    "type": "array",
                    "description": "Parameterized query values (prevents SQL injection)",
                },
            },
            "required": ["query"],
        },
        "_needle": True,
        "_query": "execute a raw SQL query on PostgreSQL",
    },
    {
        "name": "github_create_pull_request",
        "description": (
            "Create a new pull request on GitHub from a feature branch to the main branch. "
            "Automatically links related issues and requests reviewers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "PR title"},
                "branch": {
                    "type": "string",
                    "description": "Source branch name (e.g. feature/my-feature)",
                },
                "base": {
                    "type": "string",
                    "description": "Target branch (default: main)",
                },
                "body": {
                    "type": "string",
                    "description": "PR description in Markdown",
                },
            },
            "required": ["title", "branch"],
        },
        "_needle": True,
        "_query": "open a pull request on GitHub",
    },
    {
        "name": "calculate_fibonacci_sequence",
        "description": (
            "Calculate the Nth number in the Fibonacci mathematical sequence. "
            "Uses memoized recursion. Returns the exact integer value for N up to 1000."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n": {
                    "type": "integer",
                    "description": "Which Fibonacci number to compute (1-indexed)",
                },
            },
            "required": ["n"],
        },
        "_needle": True,
        "_query": "compute the Fibonacci number at position N",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate(count: int = 2000, seed: int = 42) -> list[dict]:
    """
    Generate `count` tools with `len(NEEDLES)` needles mixed in.

    Structure:
        - (count - len(NEEDLES)) noisy enterprise tools
        - len(NEEDLES) needle tools, randomly inserted
    """
    rng = random.Random(seed)

    noise_count = count - len(NEEDLES)
    tools: list[dict] = []
    generated_names: set[str] = set()

    # Pre-collect all name candidates (deterministic order)
    candidates = [
        (svc, act, res)
        for svc in SERVICES
        for act in ACTIONS
        for res in RESOURCES
    ]
    rng.shuffle(candidates)

    print(f"Generating {noise_count:,} noise tools...", end="", flush=True)
    tmpl_idx = 0
    for svc, act, res in candidates:
        if len(tools) >= noise_count:
            break
        name = f"{svc}_{act}_{res}"
        if name in generated_names:
            continue
        generated_names.add(name)
        tools.append({
            "name": name,
            "description": _desc(svc, act, res, tmpl_idx),
            "serverName": f"{svc}-{act}",
            "inputSchema": _schema(res, act),
        })
        tmpl_idx += 1
    print(f" ✓ ({len(tools):,} generated)")

    # Insert needles at random positions
    print(f"Inserting {len(NEEDLES)} needle tools at random positions...")
    for needle in NEEDLES:
        pos = rng.randint(0, len(tools))
        tools.insert(pos, needle)

    print(f"✅ Total catalog size: {len(tools):,} tools")
    token_estimate = sum(len(t.get("description", "")) // 4 + 15 for t in tools)
    print(f"   Estimated context (full load): ~{token_estimate:,} tokens")
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic enterprise MCP tool catalog for benchmarking"
    )
    parser.add_argument(
        "--count", type=int, default=2000,
        help="Total number of tools to generate (default: 2000)",
    )
    parser.add_argument(
        "--out", type=str, default="tools_2000.json",
        help="Output JSON file path (default: tools_2000.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--needles-only", action="store_true",
        help="Also write needles.json with just the target tools",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Synthetic Enterprise MCP Catalog Generator")
    print(f"  tool-search-oss benchmark suite")
    print(f"{'='*60}")
    print(f"  count={args.count}, seed={args.seed}")
    print(f"  noise tools: {args.count - len(NEEDLES):,}")
    print(f"  needles    : {len(NEEDLES)}")
    print(f"{'─'*60}\n")

    tools = generate(count=args.count, seed=args.seed)

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tools, f, indent=2, ensure_ascii=False)
    print(f"\n  → Written to: {out_path}  ({out_path.stat().st_size // 1024} KB)")

    if args.needles_only:
        needles_path = out_path.parent / "needles.json"
        with open(needles_path, "w", encoding="utf-8") as f:
            json.dump(NEEDLES, f, indent=2, ensure_ascii=False)
        print(f"  → Needles written to: {needles_path}")

    print(f"\n{'─'*60}")
    print(f"  Needle queries (what to test):")
    for n in NEEDLES:
        print(f"    [{n['name']}]")
        print(f"      query: \"{n['_query']}\"")
    print(f"\n  Next steps:")
    print(f"    # Start MCP server with this catalog")
    print(f"    python3 -m tool_search_oss.server --catalog {out_path}")
    print(f"")
    print(f"    # Run the scale benchmark")
    print(f"    python3 benchmarks/bench_2000_tools.py gemma4:e2b")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
