"""
bench_all.py — tool-search-oss ベンチマーク完全版

4指標を計測して Markdown テーブルを出力する:
  ① Context Compression Ratio  — tiktoken によるトークン数実計測
  ② TTFT Delta Estimate         — Prefill時間の推定値 (Apple Silicon モデル)
  ③ Routing Accuracy            — BM25 top-1 正解率 vs ランダム選択
  ④ API Cost Savings            — Claude/GPT-4o 単価ベースの年間コスト削減額

依存: tiktoken (pip install tiktoken)
LLM不要: 完全ローカル計測
"""

from __future__ import annotations

import json
import math
import sys
import time
import statistics
from pathlib import Path
from typing import Any

# tiktoken (fallback to char/4 if unavailable)
try:
    import tiktoken
    ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(ENC.encode(text))
    TOKENIZER = "tiktoken/cl100k_base"
except ImportError:
    def count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)
    TOKENIZER = "char/4 estimate"

sys.path.insert(0, str(Path(__file__).parent.parent))
from tool_search_oss.router import CascadeToolRouter


# ─────────────────────────────────────────────────────────────────────────────
# Tool catalog (50 tools — production-realistic)
# ─────────────────────────────────────────────────────────────────────────────

CATALOG: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent.parent / "examples" / "tools.json").read_text()
)

# 各ツールに本物に近い inputSchema を付加 (トークン計測を現実的にする)
SCHEMA_TEMPLATE = {
    "type": "object",
    "properties": {
        "path":    {"type": "string", "description": "Absolute or relative file path"},
        "content": {"type": "string", "description": "File content to write"},
        "query":   {"type": "string", "description": "Search query or SQL statement"},
        "limit":   {"type": "integer", "description": "Maximum number of results", "default": 10},
    },
}
for t in CATALOG:
    t.setdefault("inputSchema", SCHEMA_TEMPLATE)


# ─────────────────────────────────────────────────────────────────────────────
# 評価クエリセット (正解ツール付き)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_QUERIES: list[tuple[str, str]] = [
    ("read the contents of a config file",        "read_file"),
    ("write output data to a file",               "write_file"),
    ("list all files in the project directory",   "list_dir"),
    ("search for Python files matching a pattern","search_files"),
    ("execute a shell command",                    "run_command"),
    ("run the project test suite",                "run_tests"),
    ("lint the source code for errors",           "lint_code"),
    ("format the code automatically",             "format_code"),
    ("check TypeScript types",                    "check_types"),
    ("create a GitHub issue",                     "gh_create_issue"),
    ("push a commit to the repository",           "gh_push_commit"),
    ("list open pull requests",                   "gh_list_prs"),
    ("send a message to Slack",                   "slack_send"),
    ("send an email notification",                "email_send"),
    ("query data from the database",              "db_query"),
    ("insert a record into the database",         "db_insert"),
    ("update a database row",                     "db_update"),
    ("make an HTTP GET request",                  "http_get"),
    ("open a webpage in the browser",             "browser_open"),
    ("take a screenshot of the page",             "browser_screenshot"),
    ("save something to persistent memory",       "remember"),
    ("search through stored memories",            "search_memory"),
    ("get application configuration",             "get_config"),
    ("build a Docker image",                      "docker_build"),
    ("translate text to another language",        "translate_text"),
    ("delete a file",                             "delete_file"),
    ("move or rename a file",                     "move_file"),
    ("run a script file",                         "run_script"),
    ("kill a running process",                    "kill_process"),
    ("review a code diff",                        "generate_diff"),
    ("list GitHub repos",                         "gh_list_repos"),
    ("read a file from GitHub",                   "gh_read_file"),
    ("merge a pull request",                      "gh_merge_pr"),
    ("list recent Slack messages",                "slack_list"),
    ("delete rows from database",                 "db_delete"),
    ("HTTP POST request with JSON",               "http_post"),
    ("click an element on the page",              "browser_click"),
    ("type text into a form",                     "browser_type"),
    ("extract text from a webpage",               "browser_extract"),
    ("read a specific memory entry",              "read_memory"),
    ("update config settings",                    "set_config"),
    ("list all available tools",                  "list_tools"),
    ("create a git branch",                       "create_branch"),
    ("check git status",                          "git_status"),
    ("view commit history",                       "git_log"),
    ("run a Docker container",                    "docker_run"),
    ("schedule a cron job",                       "cron_schedule"),
    ("send a desktop notification",               "notify_send"),
    ("generate text with an LLM",                "llm_complete"),
    ("generate text embeddings",                  "embed_text"),
]


# ─────────────────────────────────────────────────────────────────────────────
# ① Context Compression Ratio
# ─────────────────────────────────────────────────────────────────────────────

def bench_compression() -> dict[str, Any]:
    """
    比較:
      WITHOUT router: 全50ツール定義をプロンプトに入れる
      WITH router:    tool_summary (軽量カタログ) + search_tools の定義3件
    """
    router = CascadeToolRouter()

    # WITHOUT: 全ツール定義をプロンプトに入れる場合
    full_prompt = "You have access to the following tools:\n\n"
    for t in CATALOG:
        full_prompt += f"Tool: {t['name']}\n"
        full_prompt += f"Description: {t['description']}\n"
        full_prompt += f"Schema: {json.dumps(t.get('inputSchema', {}))}\n\n"

    tokens_without = count_tokens(full_prompt)

    # WITH: tool_summary + top-3 の定義のみ
    summary = router.summarize(CATALOG)
    results = router.find_candidates("read file from disk", CATALOG, top_k=3)
    search_result_text = f"search_tools result:\n"
    for r in results:
        clean = {k: v for k, v in r.items() if not k.startswith("_")}
        search_result_text += json.dumps(clean) + "\n"

    with_prompt = summary + "\n\n" + search_result_text
    tokens_with = count_tokens(with_prompt)

    reduction_pct = (1 - tokens_with / tokens_without) * 100

    return {
        "tokens_without": tokens_without,
        "tokens_with":    tokens_with,
        "reduction_pct":  reduction_pct,
        "reduction_ratio": f"{tokens_without:,} → {tokens_with:,} tokens",
    }


# ─────────────────────────────────────────────────────────────────────────────
# ② TTFT Delta (Apple Silicon prefill推定)
# ─────────────────────────────────────────────────────────────────────────────

def bench_ttft(tokens_without: int, tokens_with: int) -> dict[str, Any]:
    """
    Apple Silicon (M3/M4) の Prefill スループット実測値から TTFT を推定。
    モデルごとの実測 prefill 速度:
      gemma-3-4b  : ~3,000 tokens/sec (4-bit)
      gemma-3-12b : ~1,200 tokens/sec
      llama-3.1-8b: ~2,000 tokens/sec
    """
    models = {
        "gemma-3-4b (4-bit)":   3000,
        "llama-3.1-8b (4-bit)": 2000,
        "gemma-3-12b (4-bit)":  1200,
    }

    results = {}
    for model, tps in models.items():
        ttft_without_ms = (tokens_without / tps) * 1000
        ttft_with_ms    = (tokens_with    / tps) * 1000
        delta_ms        = ttft_without_ms - ttft_with_ms
        speedup         = ttft_without_ms / max(ttft_with_ms, 1)
        results[model] = {
            "ttft_without_ms": round(ttft_without_ms),
            "ttft_with_ms":    round(ttft_with_ms),
            "delta_ms":        round(delta_ms),
            "speedup":         round(speedup, 1),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# ③ Routing Accuracy (BM25 top-1 vs random baseline)
# ─────────────────────────────────────────────────────────────────────────────

def bench_accuracy() -> dict[str, Any]:
    router = CascadeToolRouter()
    router._bm25.index(CATALOG)

    correct_top1  = 0
    correct_top3  = 0
    correct_top5  = 0
    timings: list[float] = []

    for intent, expected in EVAL_QUERIES:
        t0 = time.perf_counter()
        results = router.find_candidates(intent, CATALOG, top_k=5)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed * 1000)

        names = [r["name"] for r in results]
        if names and names[0] == expected:
            correct_top1 += 1
        if expected in names[:3]:
            correct_top3 += 1
        if expected in names[:5]:
            correct_top5 += 1

    n = len(EVAL_QUERIES)
    # ランダムベースライン: 50ツールからランダムに1つ選ぶ確率
    random_top1 = 1 / len(CATALOG) * 100

    return {
        "n_queries":    n,
        "acc_top1":     correct_top1 / n * 100,
        "acc_top3":     correct_top3 / n * 100,
        "acc_top5":     correct_top5 / n * 100,
        "random_top1":  random_top1,
        "lift":         (correct_top1 / n * 100) / random_top1,
        "avg_ms":       statistics.mean(timings),
        "p99_ms":       sorted(timings)[int(n * 0.99)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# ④ API Cost Savings
# ─────────────────────────────────────────────────────────────────────────────

def bench_cost(tokens_without: int, tokens_with: int) -> dict[str, Any]:
    """
    料金表 (2026年5月時点):
      Claude 3.5 Haiku: $0.80 / 1M input tokens
      Claude 3.5 Sonnet: $3.00 / 1M input tokens
      GPT-4o:            $2.50 / 1M input tokens
      GPT-4o mini:       $0.15 / 1M input tokens
    """
    saved_tokens = tokens_without - tokens_with
    models = {
        "Claude 3.5 Haiku":  0.80,
        "Claude 3.5 Sonnet": 3.00,
        "GPT-4o":            2.50,
        "GPT-4o mini":       0.15,
    }
    results = {}
    for model, price_per_m in models.items():
        per_call    = saved_tokens / 1_000_000 * price_per_m
        per_day_1k  = per_call * 1_000
        per_year    = per_day_1k * 365
        per_year_jpy = per_year * 150  # USD→JPY 概算
        results[model] = {
            "price_per_m":  price_per_m,
            "saved_per_call_usd": round(per_call, 6),
            "saved_per_day_1k_calls_usd": round(per_day_1k, 4),
            "saved_per_year_usd": round(per_year, 2),
            "saved_per_year_jpy": round(per_year_jpy),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Runner & Reporter
# ─────────────────────────────────────────────────────────────────────────────

def print_separator(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


def main() -> None:
    print(f"tool-search-oss BENCHMARK SUITE")
    print(f"Tokenizer: {TOKENIZER}")
    print(f"Catalog:   {len(CATALOG)} tools")
    print(f"Eval set:  {len(EVAL_QUERIES)} queries")

    # ① Compression
    print_separator("① Context Compression Ratio")
    comp = bench_compression()
    print(f"  WITHOUT router : {comp['tokens_without']:>6,} tokens (all {len(CATALOG)} tool definitions)")
    print(f"  WITH router    : {comp['tokens_with']:>6,} tokens (summary + top-3 definitions)")
    print(f"  Reduction      : {comp['reduction_pct']:.1f}%  ({comp['reduction_ratio']})")

    # ② TTFT
    print_separator("② TTFT Delta (Apple Silicon estimate)")
    ttft = bench_ttft(comp["tokens_without"], comp["tokens_with"])
    print(f"  {'Model':<26} {'Without':>10} {'With':>8} {'Saved':>10} {'Speedup':>8}")
    print(f"  {'─'*62}")
    for model, v in ttft.items():
        print(f"  {model:<26} {v['ttft_without_ms']:>8}ms {v['ttft_with_ms']:>6}ms "
              f"{v['delta_ms']:>8}ms {v['speedup']:>7}x")

    # ③ Accuracy
    print_separator("③ Routing Accuracy (BM25 vs random baseline)")
    acc = bench_accuracy()
    print(f"  Queries tested : {acc['n_queries']}")
    print(f"  BM25 top-1     : {acc['acc_top1']:.1f}%  (random baseline: {acc['random_top1']:.1f}%)")
    print(f"  BM25 top-3     : {acc['acc_top3']:.1f}%")
    print(f"  BM25 top-5     : {acc['acc_top5']:.1f}%")
    print(f"  Accuracy lift  : {acc['lift']:.1f}x over random")
    print(f"  Search latency : avg {acc['avg_ms']:.2f}ms / p99 {acc['p99_ms']:.2f}ms")

    # ④ Cost
    print_separator("④ API Cost Savings (1,000 calls/day, 365 days/year)")
    cost = bench_cost(comp["tokens_without"], comp["tokens_with"])
    print(f"  Saved tokens/call: {comp['tokens_without'] - comp['tokens_with']:,}")
    print()
    print(f"  {'Model':<22} {'$/1M tok':>9} {'$/call':>10} {'$/day(1k)':>12} {'$/year':>10} {'¥/year':>12}")
    print(f"  {'─'*78}")
    for model, v in cost.items():
        print(f"  {model:<22} {v['price_per_m']:>8.2f} "
              f"{v['saved_per_call_usd']:>9.5f} "
              f"{v['saved_per_day_1k_calls_usd']:>11.3f} "
              f"{v['saved_per_year_usd']:>10,.2f} "
              f"{v['saved_per_year_jpy']:>11,}")

    # Markdown output for README
    print_separator("README Markdown (copy-paste ready)")
    md = generate_markdown(comp, ttft, acc, cost)
    print(md)

    # Save to file
    out_path = Path(__file__).parent / "benchmark_results.md"
    out_path.write_text(md)
    print(f"\n✅ Results saved to {out_path}")


def generate_markdown(comp, ttft, acc, cost) -> str:
    lines = [
        "## Benchmarks",
        "",
        f"> Measured on {len(CATALOG)}-tool catalog · {len(EVAL_QUERIES)} eval queries · Tokenizer: `{TOKENIZER}`",
        "",
        "### ① Context Compression",
        "",
        "| | Tokens | vs baseline |",
        "|---|---|---|",
        f"| **Without tool-search-oss** | {comp['tokens_without']:,} | — |",
        f"| **With tool-search-oss** | {comp['tokens_with']:,} | **{comp['reduction_pct']:.0f}% reduction** |",
        "",
        "### ② TTFT Improvement (Apple Silicon, local LLM)",
        "",
        "| Model | Without | With | Saved | Speedup |",
        "|---|---|---|---|---|",
    ]
    for model, v in ttft.items():
        lines.append(
            f"| {model} | {v['ttft_without_ms']}ms | {v['ttft_with_ms']}ms"
            f" | {v['delta_ms']}ms | **{v['speedup']}x** |"
        )
    lines += [
        "",
        "### ③ Routing Accuracy",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| BM25 top-1 accuracy | **{acc['acc_top1']:.0f}%** |",
        f"| BM25 top-3 accuracy | {acc['acc_top3']:.0f}% |",
        f"| BM25 top-5 accuracy | {acc['acc_top5']:.0f}% |",
        f"| Random baseline (50 tools) | {acc['random_top1']:.1f}% |",
        f"| Accuracy lift | **{acc['lift']:.0f}x** over random |",
        f"| Search latency (avg / p99) | {acc['avg_ms']:.2f}ms / {acc['p99_ms']:.2f}ms |",
        "",
        "### ④ API Cost Savings (1,000 calls/day)",
        "",
        "| Model | Saved/call | Saved/year |",
        "|---|---|---|",
    ]
    for model, v in cost.items():
        lines.append(
            f"| {model} | ${v['saved_per_call_usd']:.5f} "
            f"| **${v['saved_per_year_usd']:,.0f}** (¥{v['saved_per_year_jpy']:,}) |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
