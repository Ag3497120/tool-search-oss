"""
bench_ttft_real.py — tool-search-oss TTFT実測ベンチマーク

Ollama streaming API を使って実際にTTFTを計測する。
WITHOUT: 50ツール全定義をプロンプトに入れてLLMに投げる
WITH:    tool-search-ossでtop-3に絞ってからLLMに投げる

使い方:
    python3 python/benchmarks/bench_ttft_real.py
    python3 python/benchmarks/bench_ttft_real.py --model gemma4:e2b
    python3 python/benchmarks/bench_ttft_real.py --model qwen2.5:1.5b --runs 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_search_oss.router import CascadeToolRouter

CATALOG: list[dict] = json.loads(
    (Path(__file__).parent.parent / "examples" / "tools.json").read_text()
)

OLLAMA_URL = "http://localhost:11434/api/generate"

TASK = "Which tool should I use to read a file from disk? Answer with just the tool name."

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt_without(task: str) -> str:
    """全50ツール定義をそのままプロンプトに入れる（ルーターなし）"""
    lines = [
        "You are a tool-selection assistant.",
        f"You have access to {len(CATALOG)} tools:\n",
    ]
    for t in CATALOG:
        lines.append(f"- {t['name']}: {t['description']}")
    lines += ["", f"Task: {task}", "Answer:"]
    return "\n".join(lines)


def build_prompt_with(task: str) -> str:
    """tool-search-ossでtop-3に絞ってからプロンプトに入れる（ルーターあり）"""
    router = CascadeToolRouter()
    results = router.find_candidates(task, CATALOG, top_k=3)

    lines = [
        "You are a tool-selection assistant.",
        f"Relevant tools for this task ({len(results)} of {len(CATALOG)} total):\n",
    ]
    for r in results:
        lines.append(f"- {r['name']}: {r['description']}")
    lines += ["", f"Task: {task}", "Answer:"]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# TTFT計測
# ─────────────────────────────────────────────────────────────────────────────

def measure_ttft(model: str, prompt: str) -> tuple[float, float, str]:
    """
    Returns: (ttft_sec, total_sec, first_token)
    Streams response from Ollama and records time to first token.
    """
    payload = json.dumps({
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,
            "num_predict": 20,   # 最初の20トークンだけ生成
        },
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    ttft    = None
    first_token = ""
    full_response = ""

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("response", "")
                if token and ttft is None:
                    ttft = time.perf_counter() - t_start
                    first_token = repr(token)
                full_response += token
                if chunk.get("done"):
                    break
    except urllib.error.URLError as e:
        return -1, -1, f"ERROR: {e}"

    total = time.perf_counter() - t_start
    return ttft or total, total, first_token


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

RESET  = "\033[0m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"


def bar(val: float, max_val: float, width: int = 40, color: str = GREEN) -> str:
    filled = int((val / max(max_val, 0.001)) * width)
    return color + "█" * filled + "░" * (width - filled) + RESET


def run_benchmark(model: str, runs: int, task: str) -> None:
    prompt_without = build_prompt_without(task)
    prompt_with    = build_prompt_with(task)

    token_count_without = len(prompt_without.split())
    token_count_with    = len(prompt_with.split())

    print(f"\n{BOLD}{'═'*65}{RESET}")
    print(f"{BOLD}  tool-search-oss TTFT Benchmark — Real Ollama Measurement{RESET}")
    print(f"{'═'*65}")
    print(f"  Model   : {CYAN}{model}{RESET}")
    print(f"  Task    : {task}")
    print(f"  Runs    : {runs} per condition")
    print(f"  Prompt  WITHOUT router : ~{token_count_without:,} words")
    print(f"  Prompt  WITH router    : ~{token_count_with:,} words  ({len(prompt_with.split())/len(prompt_without.split())*100:.0f}% of original)")
    print(f"{'─'*65}\n")

    # Warmup (モデルをメモリに乗せる)
    print(f"  {YELLOW}⟳ Warming up model...{RESET}", end="", flush=True)
    measure_ttft(model, "Hello")
    print(f"  {GREEN}✓ Ready{RESET}\n")

    # ── WITHOUT ──────────────────────────────────────────────────────────────
    print(f"  {RED}{BOLD}WITHOUT tool-search-oss{RESET}  (all {len(CATALOG)} tool definitions in context)")
    ttft_without_list = []
    for i in range(runs):
        ttft, total, first = measure_ttft(model, prompt_without)
        if ttft < 0:
            print(f"    Run {i+1}: {RED}ERROR{RESET}")
            continue
        ttft_without_list.append(ttft)
        print(f"    Run {i+1}: TTFT={RED}{ttft*1000:>7.0f}ms{RESET}  total={total:.2f}s  first={first}")
        time.sleep(0.5)

    # ── WITH ─────────────────────────────────────────────────────────────────
    print(f"\n  {GREEN}{BOLD}WITH tool-search-oss{RESET}     (top-3 tools only in context)")
    ttft_with_list = []
    for i in range(runs):
        ttft, total, first = measure_ttft(model, prompt_with)
        if ttft < 0:
            print(f"    Run {i+1}: {RED}ERROR{RESET}")
            continue
        ttft_with_list.append(ttft)
        print(f"    Run {i+1}: TTFT={GREEN}{ttft*1000:>7.0f}ms{RESET}  total={total:.2f}s  first={first}")
        time.sleep(0.5)

    if not ttft_without_list or not ttft_with_list:
        print(f"\n{RED}ERROR: No valid measurements. Is Ollama running?{RESET}")
        print(f"  Try: ollama serve")
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    avg_without = sum(ttft_without_list) / len(ttft_without_list)
    avg_with    = sum(ttft_with_list)    / len(ttft_with_list)
    min_without = min(ttft_without_list)
    min_with    = min(ttft_with_list)
    speedup     = avg_without / max(avg_with, 0.001)
    saved_ms    = (avg_without - avg_with) * 1000

    max_bar = avg_without * 1000

    print(f"\n{'═'*65}")
    print(f"{BOLD}  Results{RESET}")
    print(f"{'─'*65}")
    print(f"\n  WITHOUT  {bar(avg_without*1000, max_bar, 40, RED)}  avg {RED}{avg_without*1000:.0f}ms{RESET}  min {min_without*1000:.0f}ms")
    print(f"  WITH     {bar(avg_with*1000, max_bar, 40, GREEN)}  avg {GREEN}{avg_with*1000:.0f}ms{RESET}  min {min_with*1000:.0f}ms")
    print()
    print(f"  {BOLD}TTFT saved  : {YELLOW}{saved_ms:.0f}ms{RESET} per request")
    print(f"  {BOLD}Speedup     : {YELLOW}{speedup:.1f}x{RESET} faster first token")
    print(f"  {BOLD}Context     : {len(CATALOG)} tools → top-3 ({token_count_with/token_count_without*100:.0f}% of original){RESET}")
    print(f"\n{'═'*65}")

    # Markdown output
    md = f"""
## TTFT Benchmark — {model} (actual measurement)

| Condition | Avg TTFT | Min TTFT |
|---|---|---|
| WITHOUT tool-search-oss ({len(CATALOG)} tools) | {avg_without*1000:.0f}ms | {min_without*1000:.0f}ms |
| WITH tool-search-oss (top-3 only) | {avg_with*1000:.0f}ms | {min_with*1000:.0f}ms |
| **Speedup** | **{speedup:.1f}x** | — |
"""
    out = Path(__file__).parent / "ttft_real_results.md"
    out.write_text(md)
    print(f"\n  Results saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma4:e2b",
                        help="Ollama model name (default: gemma4:e2b)")
    parser.add_argument("--runs",  type=int, default=3,
                        help="Runs per condition (default: 3)")
    parser.add_argument("--task",  default=TASK,
                        help="Task description for tool selection")
    args = parser.parse_args()

    run_benchmark(args.model, args.runs, args.task)


if __name__ == "__main__":
    main()
