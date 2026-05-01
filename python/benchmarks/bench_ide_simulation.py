"""
bench_ide_simulation.py — VerantyxIDE環境でのTTFT比較デモ

VerantyxIDEが持つMCPEngine構造を模倣し:
1. 50個のMCPサーバーから取得したツールリストを構築
2. WITHOUT: 全ツール定義をOllamaへのプロンプトに入れてTTFT計測
3. WITH: tool-search-ossで絞り込んだ後にOllamaへ投げてTTFT計測

これはIDEが実際にやっているLLM呼び出しの正確なシミュレーション。
"""

from __future__ import annotations

import json
import time
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_search_oss.router import CascadeToolRouter

# ─────────────────────────────────────────────────────────────────────────────
# VerantyxIDEが持つ実際のMCPサーバー構成を模倣
# (実際のIDEは UserDefaults から MCPServerConfig[] を読む)
# ─────────────────────────────────────────────────────────────────────────────

MCP_SERVERS = [
    # verantyx-cortex (JCrossメモリ)
    {"server": "verantyx-compiler", "tools": [
        {"name": "remember",     "description": "Save memory to JCross node"},
        {"name": "search",       "description": "Semantic search in memory"},
        {"name": "read",         "description": "Read a JCross node by filename"},
        {"name": "boot",         "description": "Session bootstrap, load profile"},
        {"name": "ground",       "description": "Visual grounding, anti-hallucination SVG"},
        {"name": "find",         "description": "Kanji topology spatial search"},
        {"name": "gc",           "description": "Garbage collect old memory nodes"},
        {"name": "move",         "description": "Migrate node to another zone"},
        {"name": "evolve",       "description": "Evolve character from memories"},
        {"name": "recall",       "description": "Recall a key-value fact"},
        {"name": "store",        "description": "Store a key-value fact"},
        {"name": "scan",         "description": "Scan front zone memory nodes"},
    ]},
    # Filesystem
    {"server": "filesystem", "tools": [
        {"name": "read_file",    "description": "Read contents of a file from disk"},
        {"name": "write_file",   "description": "Write content to a file on disk"},
        {"name": "list_dir",     "description": "List files and directories"},
        {"name": "create_dir",   "description": "Create a new directory"},
        {"name": "delete_file",  "description": "Delete a file or directory"},
        {"name": "move_file",    "description": "Move or rename a file"},
        {"name": "search_files", "description": "Search files by regex pattern"},
    ]},
    # GitHub
    {"server": "github", "tools": [
        {"name": "gh_list_repos",   "description": "List GitHub repositories"},
        {"name": "gh_create_issue", "description": "Create a GitHub issue"},
        {"name": "gh_read_file",    "description": "Read a file from GitHub repo"},
        {"name": "gh_push_commit",  "description": "Push a commit to GitHub"},
        {"name": "gh_list_prs",     "description": "List pull requests"},
        {"name": "gh_merge_pr",     "description": "Merge a pull request"},
        {"name": "gh_create_branch","description": "Create a new branch"},
        {"name": "gh_list_issues",  "description": "List GitHub issues"},
    ]},
    # Puppeteer / Browser
    {"server": "puppeteer", "tools": [
        {"name": "browser_navigate",    "description": "Navigate to a URL in browser"},
        {"name": "browser_click",       "description": "Click an element on a page"},
        {"name": "browser_type",        "description": "Type text into a form field"},
        {"name": "browser_screenshot",  "description": "Take a screenshot of the page"},
        {"name": "browser_extract",     "description": "Extract text content from page"},
        {"name": "browser_evaluate",    "description": "Execute JavaScript on the page"},
    ]},
    # SQLite / DB
    {"server": "sqlite", "tools": [
        {"name": "db_query",    "description": "Execute a SQL SELECT query"},
        {"name": "db_insert",   "description": "Insert a row into a table"},
        {"name": "db_update",   "description": "Update rows in a table"},
        {"name": "db_delete",   "description": "Delete rows from a table"},
        {"name": "db_schema",   "description": "Get the schema of a database table"},
    ]},
    # Slack
    {"server": "slack", "tools": [
        {"name": "slack_send",       "description": "Send a message to a Slack channel"},
        {"name": "slack_list",       "description": "List recent Slack messages"},
        {"name": "slack_react",      "description": "Add a reaction emoji to a message"},
        {"name": "slack_list_users", "description": "List Slack workspace users"},
    ]},
    # Shell / Terminal
    {"server": "shell", "tools": [
        {"name": "run_command",  "description": "Execute a shell command"},
        {"name": "run_script",   "description": "Run a shell or Python script"},
        {"name": "kill_process", "description": "Kill a running process by PID"},
    ]},
    # tool-search-oss (自分自身!)
    {"server": "tool-search-oss", "tools": [
        {"name": "discover_tools",    "description": "BM25 search: find the right tool from catalog"},
        {"name": "catalog_summary",   "description": "Get lightweight catalog of all tools"},
        {"name": "visual_route",      "description": "Visual routing graph for tool selection"},
    ]},
]

# 全ツールをフラットリストに展開 (IDEの connectedTools に相当)
ALL_TOOLS = []
for server in MCP_SERVERS:
    for tool in server["tools"]:
        ALL_TOOLS.append({
            "name":        tool["name"],
            "description": tool["description"],
            "serverName":  server["server"],
        })

OLLAMA_URL = "http://localhost:11434/api/generate"
RESET = "\033[0m"; RED = "\033[91m"; GREEN = "\033[92m"
YELLOW = "\033[93m"; CYAN = "\033[96m"; BOLD = "\033[1m"


def count_tokens_approx(text: str) -> int:
    """tiktoken なしの近似: chars / 4"""
    return max(1, len(text) // 4)


def build_prompt_without(task: str) -> str:
    """IDEが全connectedToolsをそのままLLMに渡した場合"""
    lines = [
        f"You are an AI assistant in Verantyx IDE.",
        f"You have access to {len(ALL_TOOLS)} tools across {len(MCP_SERVERS)} MCP servers:\n",
    ]
    current_server = None
    for tool in ALL_TOOLS:
        if tool["serverName"] != current_server:
            current_server = tool["serverName"]
            lines.append(f"\n[{current_server}]")
        lines.append(f"  - {tool['name']}: {tool['description']}")
    lines += ["", f"User task: {task}", "", "Which tool should you call? Answer with: server_name.tool_name"]
    return "\n".join(lines)


def build_prompt_with(task: str) -> str:
    """tool-search-ossで絞り込んだ後にLLMに渡す場合"""
    router = CascadeToolRouter()
    results = router.find_candidates(task, ALL_TOOLS, top_k=3)

    lines = [
        f"You are an AI assistant in Verantyx IDE.",
        f"Relevant tools for this task (3 of {len(ALL_TOOLS)} total, selected by tool-search-oss):\n",
    ]
    for r in results:
        lines.append(f"  - [{r['serverName']}] {r['name']}: {r['description']}")
    lines += ["", f"User task: {task}", "", "Which tool should you call? Answer with: server_name.tool_name"]
    return "\n".join(lines)


def measure_ttft(model: str, prompt: str) -> tuple[float, str]:
    payload = json.dumps({
        "model": model, "prompt": prompt, "stream": True,
        "options": {"temperature": 0, "num_predict": 15},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t_start = time.perf_counter()
    ttft = None; answer = ""
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                # 空白/改行のみのトークンは無視してTTFTを計測
                if token.strip() and ttft is None:
                    ttft = time.perf_counter() - t_start
                answer += token
                if chunk.get("done"):
                    break
        if ttft is None:
            # done=trueが来たがtokenが全部空だった → totalで代用
            ttft = time.perf_counter() - t_start
    except Exception as e:
        return -1, f"ERROR: {e}"
    return ttft, answer.strip()[:50]


def bar(val: float, max_val: float, width: int = 44, color: str = GREEN) -> str:
    filled = int((val / max(max_val, 0.001)) * width)
    return color + "█" * filled + "░" * (width - filled) + RESET


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
    tasks = [
        ("ファイルを読み込んで",        "read file contents"),
        ("GitHubにコミットをプッシュ",  "push commit to github"),
        ("Slackにメッセージを送りたい", "send slack message"),
        ("メモリを検索したい",          "search memory"),
    ]

    print(f"\n{BOLD}{'═'*68}{RESET}")
    print(f"{BOLD}  Verantyx IDE × tool-search-oss TTFT Benchmark{RESET}")
    print(f"{'═'*68}")
    print(f"  Model      : {CYAN}{model}{RESET}")
    print(f"  MCP servers: {len(MCP_SERVERS)}")
    print(f"  Total tools: {len(ALL_TOOLS)}")
    print(f"{'─'*68}")

    # Warmup
    print(f"\n  {YELLOW}⟳ Warming up...{RESET}", end="", flush=True)
    measure_ttft(model, "Hello")
    print(f" {GREEN}ready{RESET}\n")

    results = []
    for jp_task, en_task in tasks:
        prompt_w = build_prompt_without(en_task)
        prompt_t = build_prompt_with(en_task)
        tok_w = count_tokens_approx(prompt_w)
        tok_t = count_tokens_approx(prompt_t)

        print(f"  {BOLD}Task: {jp_task}{RESET}")
        print(f"  ({'─'*60})")

        ttft_w, ans_w = measure_ttft(model, prompt_w)
        print(f"  {RED}WITHOUT{RESET} ({tok_w:,} tokens) → TTFT={RED}{ttft_w*1000:.0f}ms{RESET}  answer: {ans_w[:40]}")

        time.sleep(0.3)

        ttft_t, ans_t = measure_ttft(model, prompt_t)
        print(f"  {GREEN}WITH   {RESET} ({tok_t:,} tokens) → TTFT={GREEN}{ttft_t*1000:.0f}ms{RESET}  answer: {ans_t[:40]}")

        if ttft_w > 0 and ttft_t > 0:
            speedup = ttft_w / ttft_t
            max_t = ttft_w * 1000
            print(f"\n  WITHOUT  {bar(ttft_w*1000, max_t, 44, RED)}")
            print(f"  WITH     {bar(ttft_t*1000, max_t, 44, GREEN)}")
            print(f"  → {YELLOW}{BOLD}{speedup:.1f}x faster{RESET}  ({(1-tok_t/tok_w)*100:.0f}% less context)\n")
            results.append((jp_task, ttft_w, ttft_t, tok_w, tok_t, speedup))

        time.sleep(0.5)

    # Summary
    if results:
        avg_speedup = sum(r[5] for r in results) / len(results)
        avg_tok_save = sum((1 - r[4]/r[3])*100 for r in results) / len(results)
        print(f"{'═'*68}")
        print(f"  {BOLD}Summary — {model}{RESET}")
        print(f"{'─'*68}")
        print(f"  {'Task':<26} {'WITHOUT':>9} {'WITH':>9} {'Speedup':>8} {'Context':>9}")
        print(f"  {'─'*62}")
        for task, w, t, tw, tt, sp in results:
            print(f"  {task:<26} {w*1000:>7.0f}ms {t*1000:>7.0f}ms {sp:>7.1f}x {(1-tt/tw)*100:>8.0f}%")
        print(f"  {'─'*62}")
        print(f"  {'AVERAGE':<26} {'':>9} {'':>9} {avg_speedup:>7.1f}x {avg_tok_save:>8.0f}%")
        print(f"{'═'*68}\n")

        # Save markdown
        md_lines = [
            f"## Verantyx IDE TTFT Benchmark — {model}",
            f"",
            f"> {len(MCP_SERVERS)} MCP servers · {len(ALL_TOOLS)} total tools",
            f"",
            f"| Task | Without | With | Speedup | Context saved |",
            f"|---|---|---|---|---|",
        ]
        for task, w, t, tw, tt, sp in results:
            md_lines.append(
                f"| {task} | {w*1000:.0f}ms | {t*1000:.0f}ms | **{sp:.1f}x** | {(1-tt/tw)*100:.0f}% |"
            )
        md_lines += [f"", f"| **Average** | | | **{avg_speedup:.1f}x** | **{avg_tok_save:.0f}%** |"]
        out = Path(__file__).parent / "bench_ide_results.md"
        out.write_text("\n".join(md_lines))
        print(f"  Results → {out}")


if __name__ == "__main__":
    main()
