"""
bench_ide_en.py — English version of Verantyx IDE TTFT benchmark
Same structure as bench_ide_simulation.py but all output in English.
"""

from __future__ import annotations
import json, time, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_search_oss.router import CascadeToolRouter

MCP_SERVERS = [
    {"server": "verantyx-compiler", "tools": [
        {"name": "remember",     "description": "Save memory to JCross node"},
        {"name": "search",       "description": "Semantic search in memory"},
        {"name": "read",         "description": "Read a JCross node by filename"},
        {"name": "boot",         "description": "Session bootstrap, load profile"},
        {"name": "ground",       "description": "Visual grounding anti-hallucination SVG"},
        {"name": "find",         "description": "Kanji topology spatial search"},
        {"name": "gc",           "description": "Garbage collect old memory nodes"},
        {"name": "move",         "description": "Migrate node to another zone"},
        {"name": "evolve",       "description": "Evolve character from memories"},
        {"name": "recall",       "description": "Recall a key-value fact"},
        {"name": "store",        "description": "Store a key-value fact"},
        {"name": "scan",         "description": "Scan front zone memory nodes"},
    ]},
    {"server": "filesystem", "tools": [
        {"name": "read_file",    "description": "Read contents of a file from disk"},
        {"name": "write_file",   "description": "Write content to a file on disk"},
        {"name": "list_dir",     "description": "List files and directories"},
        {"name": "create_dir",   "description": "Create a new directory"},
        {"name": "delete_file",  "description": "Delete a file or directory"},
        {"name": "move_file",    "description": "Move or rename a file"},
        {"name": "search_files", "description": "Search files by regex pattern"},
    ]},
    {"server": "github", "tools": [
        {"name": "gh_list_repos",    "description": "List GitHub repositories for a user"},
        {"name": "gh_create_issue",  "description": "Create a GitHub issue with title and body"},
        {"name": "gh_read_file",     "description": "Read a file from a GitHub repository"},
        {"name": "gh_push_commit",   "description": "Push a commit to a GitHub branch"},
        {"name": "gh_list_prs",      "description": "List open pull requests in a repo"},
        {"name": "gh_merge_pr",      "description": "Merge a pull request"},
        {"name": "gh_create_branch", "description": "Create a new git branch"},
        {"name": "gh_list_issues",   "description": "List GitHub issues in a repository"},
    ]},
    {"server": "puppeteer", "tools": [
        {"name": "browser_navigate",   "description": "Navigate browser to a URL"},
        {"name": "browser_click",      "description": "Click an element on a webpage"},
        {"name": "browser_type",       "description": "Type text into a form field"},
        {"name": "browser_screenshot", "description": "Take a screenshot of the page"},
        {"name": "browser_extract",    "description": "Extract text content from a page"},
        {"name": "browser_evaluate",   "description": "Execute JavaScript on the page"},
    ]},
    {"server": "sqlite", "tools": [
        {"name": "db_query",  "description": "Execute a SQL SELECT query on a database"},
        {"name": "db_insert", "description": "Insert a row into a database table"},
        {"name": "db_update", "description": "Update rows in a database table"},
        {"name": "db_delete", "description": "Delete rows from a database table"},
        {"name": "db_schema", "description": "Get schema of a database table"},
    ]},
    {"server": "slack", "tools": [
        {"name": "slack_send",       "description": "Send a message to a Slack channel"},
        {"name": "slack_list",       "description": "List recent messages in Slack"},
        {"name": "slack_react",      "description": "Add a reaction emoji to a Slack message"},
        {"name": "slack_list_users", "description": "List Slack workspace users"},
    ]},
    {"server": "shell", "tools": [
        {"name": "run_command",  "description": "Execute a shell command in subprocess"},
        {"name": "run_script",   "description": "Run a shell or Python script file"},
        {"name": "kill_process", "description": "Kill a running process by PID"},
    ]},
    {"server": "tool-search-oss", "tools": [
        {"name": "discover_tools",  "description": "BM25 search: find the right tool from catalog"},
        {"name": "catalog_summary", "description": "Get lightweight catalog summary of all tools"},
        {"name": "visual_route",    "description": "Visual routing graph for tool selection"},
    ]},
]

ALL_TOOLS = []
for server in MCP_SERVERS:
    for tool in server["tools"]:
        ALL_TOOLS.append({"name": tool["name"], "description": tool["description"], "serverName": server["server"]})

OLLAMA_URL = "http://localhost:11434/api/generate"
R="\033[0m"; RED="\033[91m"; GREEN="\033[92m"; YELLOW="\033[93m"; CYAN="\033[96m"; BOLD="\033[1m"

TASKS = [
    ("Read a file from disk",          "read file contents from disk"),
    ("Push a commit to GitHub",        "push commit to github repository"),
    ("Send a Slack message",           "send message to slack channel"),
    ("Search persistent memory",       "search memory for relevant information"),
]

def build_without(task):
    lines = [f"You are an AI assistant in Verantyx IDE with {len(ALL_TOOLS)} tools across {len(MCP_SERVERS)} MCP servers:\n"]
    cur = None
    for t in ALL_TOOLS:
        if t["serverName"] != cur:
            cur = t["serverName"]
            lines.append(f"\n[{cur}]")
        lines.append(f"  - {t['name']}: {t['description']}")
    lines += ["", f"Task: {task}", "", "Which tool? Answer: server.tool_name"]
    return "\n".join(lines)

def build_with(task):
    r = CascadeToolRouter()
    hits = r.find_candidates(task, ALL_TOOLS, top_k=3)
    lines = [f"You are an AI assistant. Relevant tools ({len(hits)} of {len(ALL_TOOLS)}, via tool-search-oss):\n"]
    for h in hits:
        lines.append(f"  - [{h['serverName']}] {h['name']}: {h['description']}")
    lines += ["", f"Task: {task}", "", "Which tool? Answer: server.tool_name"]
    return "\n".join(lines)

def ttft(model, prompt):
    pay = json.dumps({"model": model, "prompt": prompt, "stream": True,
                      "options": {"temperature": 0, "num_predict": 15}}).encode()
    req = urllib.request.Request(OLLAMA_URL, data=pay,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter(); first = None; ans = ""
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line: continue
                chunk = json.loads(line)
                tok = chunk.get("response", "")
                if tok.strip() and first is None: first = time.perf_counter() - t0
                ans += tok
                if chunk.get("done"): break
        if first is None: first = time.perf_counter() - t0
    except Exception as e: return -1, str(e)
    return first, ans.strip()[:50]

def bar(v, mx, w=44, c=GREEN): f=int(v/max(mx,0.001)*w); return c+"█"*f+"░"*(w-f)+R

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4:e2b"
    print(f"\n{BOLD}{'═'*70}{R}")
    print(f"{BOLD}  Verantyx IDE × tool-search-oss  |  TTFT Benchmark{R}")
    print(f"{'═'*70}")
    print(f"  Model        : {CYAN}{model}{R}")
    print(f"  MCP servers  : {len(MCP_SERVERS)}")
    print(f"  Total tools  : {len(ALL_TOOLS)}")
    print(f"  Routing      : WITHOUT (all {len(ALL_TOOLS)} tools in context) vs WITH (top-3 via BM25)")
    print(f"{'─'*70}")
    print(f"\n  {YELLOW}⟳ Warming up model...{R}", end="", flush=True)
    ttft(model, "Hello"); print(f" {GREEN}ready{R}\n")

    results = []
    for label, task in TASKS:
        pw = build_without(task); pt = build_with(task)
        tw = len(pw)//4; tt_ = len(pt)//4
        print(f"  {BOLD}Task: {label}{R}")
        print(f"  {'─'*66}")
        w, aw = ttft(model, pw)
        print(f"  {RED}WITHOUT{R}  ({tw:,} ~tokens)  TTFT = {RED}{w*1000:>6.0f} ms{R}")
        time.sleep(0.3)
        t, at = ttft(model, pt)
        print(f"  {GREEN}WITH   {R}  ({tt_:,} ~tokens)  TTFT = {GREEN}{t*1000:>6.0f} ms{R}")
        if w > 0 and t > 0:
            sp = w/t; mx = w*1000
            print(f"\n  WITHOUT  {bar(w*1000,mx,44,RED)}")
            print(f"  WITH     {bar(t*1000,mx,44,GREEN)}")
            print(f"  ➜ {YELLOW}{BOLD}{sp:.1f}x faster{R}  |  {(1-tt_/tw)*100:.0f}% less context\n")
            results.append((label, w, t, tw, tt_, sp))
        time.sleep(0.5)

    if results:
        avg_sp = sum(r[5] for r in results)/len(results)
        avg_ctx = sum((1-r[4]/r[3])*100 for r in results)/len(results)
        print(f"{'═'*70}")
        print(f"  {BOLD}Summary  —  {model}{R}")
        print(f"{'─'*70}")
        print(f"  {'Task':<30} {'WITHOUT':>9} {'WITH':>8} {'Speedup':>8} {'Context':>9}")
        print(f"  {'─'*66}")
        for lb,w,t,_,_,sp in results:
            print(f"  {lb:<30} {w*1000:>7.0f}ms {t*1000:>6.0f}ms {sp:>7.1f}x {(1-_/results[0][3])*100 if _ else 0:>8.0f}%")
        print(f"  {'─'*66}")
        print(f"  {'AVERAGE':<30} {'':>9} {'':>8} {avg_sp:>7.1f}x {avg_ctx:>8.0f}%")
        print(f"{'═'*70}\n")
        out = Path(__file__).parent/"bench_ide_en_results.md"
        md = [f"## Verantyx IDE TTFT Benchmark (English) — {model}",
              f"",">{len(MCP_SERVERS)} MCP servers · {len(ALL_TOOLS)} total tools","",
              "| Task | Without router | With router | Speedup | Context saved |","|---|---|---|---|---|"]
        for lb,w,t,tw,tt,sp in results:
            md.append(f"| {lb} | {w*1000:.0f}ms | {t*1000:.0f}ms | **{sp:.1f}x** | {(1-tt/tw)*100:.0f}% |")
        md += ["",f"| **Average** | | | **{avg_sp:.1f}x** | **{avg_ctx:.0f}%** |"]
        out.write_text("\n".join(md))
        print(f"  Results saved → {out}")

if __name__=="__main__": main()
