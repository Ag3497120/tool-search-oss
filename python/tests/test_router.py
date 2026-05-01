"""
test_router.py — pytest suite for tool-search-oss

Tests:
  - BM25ToolRouter accuracy (correct tool is top-1)
  - RegexToolRouter fallback
  - CascadeToolRouter (BM25 → Regex)
  - VisualTopologyRouter placeholder raises NotImplementedError
  - Context savings calculation
  - catalog_summary output
"""

import json
import math
from pathlib import Path

import pytest

from tool_search_oss.router import (
    BM25ToolRouter,
    CascadeToolRouter,
    RegexToolRouter,
    VisualTopologyRouter,
)

# ── Fixture: 50-tool catalog ──────────────────────────────────────────────────

CATALOG_PATH = Path(__file__).parent.parent / "examples" / "tools.json"

@pytest.fixture
def tools():
    with open(CATALOG_PATH) as f:
        return json.load(f)

@pytest.fixture
def bm25(tools):
    r = BM25ToolRouter()
    r.index(tools)
    return r

# ── BM25 accuracy ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent, expected_top1", [
    ("read a file from disk",       "read_file"),
    ("send a message to Slack",     "slack_send"),
    ("push a commit to GitHub",     "gh_push_commit"),
    ("run the test suite",          "run_tests"),
    ("insert a row into database",  "db_insert"),
    ("open a URL in browser",       "browser_open"),
    ("search persistent memory",    "search_memory"),
    ("translate text to French",    "translate_text"),
])
def test_bm25_top1_correct(bm25, tools, intent, expected_top1):
    results = bm25.find_candidates(intent, tools, top_k=5)
    assert results, f"No results for: {intent}"
    assert results[0]["name"] == expected_top1, (
        f"Expected top-1={expected_top1}, got {results[0]['name']} "
        f"(scores: {[(r['name'], r['_score']) for r in results[:3]]})"
    )

def test_bm25_returns_score_and_terms(bm25, tools):
    results = bm25.find_candidates("read a file", tools, top_k=3)
    for r in results:
        assert "_score" in r
        assert "_matched_terms" in r
        assert r["_score"] > 0

def test_bm25_top_k_respected(bm25, tools):
    for k in [1, 3, 5, 10]:
        results = bm25.find_candidates("file", tools, top_k=k)
        assert len(results) <= k

def test_bm25_no_results_for_garbage(bm25, tools):
    results = bm25.find_candidates("xyzzy123 foobar quux", tools, top_k=5)
    assert results == []

# ── Regex fallback ────────────────────────────────────────────────────────────

def test_regex_finds_exact_name(tools):
    r = RegexToolRouter()
    results = r.find_candidates("slack", tools, top_k=3)
    names = [t["name"] for t in results]
    assert any("slack" in n for n in names)

def test_regex_fallback_no_match(tools):
    r = RegexToolRouter()
    results = r.find_candidates("xyzzy123", tools, top_k=5)
    assert results == []

# ── Cascade (BM25 → Regex) ────────────────────────────────────────────────────

def test_cascade_uses_bm25_first(tools):
    r = CascadeToolRouter()
    results = r.find_candidates("read file", tools, top_k=3)
    assert results
    assert results[0]["name"] == "read_file"

def test_cascade_falls_back_to_regex(tools):
    """Exact name that BM25 might miss (very short, single token)."""
    r = CascadeToolRouter()
    results = r.find_candidates("docker", tools, top_k=3)
    names = [t["name"] for t in results]
    assert any("docker" in n for n in names)

# ── VisualTopologyRouter placeholder ─────────────────────────────────────────

def test_visual_topology_raises(tools):
    r = VisualTopologyRouter()
    with pytest.raises(NotImplementedError, match="v0.2"):
        r.find_candidates("anything", tools)

# ── Context savings ───────────────────────────────────────────────────────────

def test_context_savings_significant(bm25, tools):
    """Top-5 results should use significantly less context than all tools."""
    full_chars = sum(
        len(t["name"]) + len(t.get("description", ""))
        for t in tools
    )
    results = bm25.find_candidates("read file", tools, top_k=5)
    top5_chars = sum(
        len(r["name"]) + len(r.get("description", ""))
        for r in results
    )
    savings = 1 - top5_chars / full_chars
    assert savings > 0.7, f"Expected >70% savings, got {savings:.0%}"

# ── Catalog summary ───────────────────────────────────────────────────────────

def test_catalog_summary_contains_categories(bm25, tools):
    summary = bm25.summarize(tools)
    assert "[SEARCH]" in summary.upper() or "search" in summary.lower()
    assert str(len(tools)) in summary

def test_catalog_summary_under_char_limit(bm25, tools):
    summary = bm25.summarize(tools, max_chars=2000)
    assert len(summary) <= 2200  # small buffer for the truncation line
