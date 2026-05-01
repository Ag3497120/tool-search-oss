"""
router.py — tool-search-oss Python core

Strategy Pattern implementation:
  BaseToolRouter        ← interface
    BM25ToolRouter      ← v0.1 default (stdlib only, zero deps)
    RegexToolRouter     ← fallback for exact keyword matches
    VisualTopologyRouter ← v0.2 placeholder (image-based routing)

Usage:
    from tool_search_oss.router import BM25ToolRouter

    router = BM25ToolRouter()
    results = router.find_candidates(
        intent="read a file from disk",
        tools=my_tool_list,
        top_k=5
    )
"""

from __future__ import annotations

import re
import math
from abc import ABC, abstractmethod
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

ToolSchema = dict[str, Any]  # {"name": str, "description": str, ...}

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "search":  ["search", "find", "query", "lookup", "fetch", "get", "read", "list"],
    "write":   ["write", "save", "store", "create", "insert", "add", "update", "edit", "delete", "remove"],
    "run":     ["run", "exec", "execute", "start", "launch", "call", "invoke", "trigger", "send"],
    "analyze": ["analyze", "check", "lint", "test", "validate", "review", "inspect", "debug", "diff"],
    "meta":    ["config", "setup", "init", "install", "manage", "status", "help", "list tools"],
}


def infer_category(tool: ToolSchema) -> str:
    """Infer tool category from name + description (no ML required)."""
    if cat := tool.get("category"):
        return cat
    text = (tool.get("name", "") + " " + tool.get("description", "")).lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text for k in keywords):
            return cat
    return "other"


# ──────────────────────────────────────────────────────────────────────────────
# BaseToolRouter — Strategy interface
# ──────────────────────────────────────────────────────────────────────────────

class BaseToolRouter(ABC):
    """
    All routers implement this single interface.
    Swap BM25ToolRouter ↔ RegexToolRouter ↔ VisualTopologyRouter
    without changing the MCP server code.
    """

    @abstractmethod
    def find_candidates(
        self,
        intent: str,
        tools: list[ToolSchema],
        top_k: int = 5,
    ) -> list[ToolSchema]:
        """
        Returns the top-k most relevant tools for the given intent.
        Each returned dict includes the original tool schema plus:
          _score (float): relevance score
          _matched_terms (list[str]): which query terms matched
        """
        ...

    def summarize(self, tools: list[ToolSchema], max_chars: int = 4000) -> str:
        """
        Lightweight catalog for LLM session start (defer_loading pattern).
        Groups by category. Stays under max_chars.
        """
        groups: dict[str, list[str]] = {}
        for t in tools:
            cat = infer_category(t)
            groups.setdefault(cat, [])
            desc = t.get("description", "")[:70]
            groups[cat].append(f"  {t['name']}: {desc}")

        header = f"TOOL CATALOG — {len(tools)} tools available"
        blocks: list[str] = [header]
        total = len(header)

        for cat, entries in sorted(groups.items()):
            block = f"\n[{cat.upper()}]\n" + "\n".join(entries)
            if total + len(block) > max_chars:
                remaining = sum(
                    1 for c, es in groups.items()
                    if c not in [b.split("]")[0].lstrip("\n[").lower() for b in blocks]
                )
                blocks.append(f"\n... (+{remaining} more categories, use discover_tools() to filter)")
                break
            blocks.append(block)
            total += len(block)

        return "\n".join(blocks)


# ──────────────────────────────────────────────────────────────────────────────
# Tokenizer (shared by BM25 and Regex routers)
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """
    Split on whitespace, underscores, hyphens, slashes.
    Also splits camelCase. Lowercase. No stopwords (they're rare in tool names).
    """
    # camelCase → camel case
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # snake_case, kebab-case, path/separators
    text = re.sub(r"[_\-\/\\]", " ", text)
    # remove non-alphanumeric
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]


def _doc_text(tool: ToolSchema) -> str:
    """Build the searchable text for a tool."""
    parts = [
        tool.get("name", ""),
        tool.get("description", ""),
        tool.get("category", ""),
        " ".join(tool.get("tags", [])),
    ]
    return " ".join(p for p in parts if p)


# ──────────────────────────────────────────────────────────────────────────────
# BM25ToolRouter — v0.1 default
# ──────────────────────────────────────────────────────────────────────────────

class BM25ToolRouter(BaseToolRouter):
    """
    Okapi BM25 (k1=1.5, b=0.75). Pure stdlib. Zero ML dependencies.

    Build once, query many times:
        router = BM25ToolRouter()
        router.index(tools)          # optional pre-indexing
        router.find_candidates(...)  # or pass tools on each call
    """

    K1 = 1.5
    B  = 0.75

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.K1 = k1
        self.B  = b
        self._index: dict[str, dict[str, int]] = {}   # name → term → freq
        self._df: dict[str, int] = {}                 # term → doc count
        self._dl: dict[str, int] = {}                 # name → doc length
        self._avg_dl: float = 0.0
        self._tools: list[ToolSchema] = []

    # ── indexing ─────────────────────────────────────────────────────────────

    def index(self, tools: list[ToolSchema]) -> None:
        """Pre-index tools for repeated queries (optional optimisation)."""
        self._tools  = tools
        self._index  = {}
        self._df     = {}
        self._dl     = {}

        for tool in tools:
            name   = tool["name"]
            tokens = _tokenize(_doc_text(tool))
            self._dl[name] = len(tokens)

            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._index[name] = tf

            for term in tf:
                self._df[term] = self._df.get(term, 0) + 1

        total = sum(self._dl.values())
        self._avg_dl = total / max(len(tools), 1)

    def _score(self, query_terms: list[str], tool_name: str, N: int) -> tuple[float, list[str]]:
        tf     = self._index.get(tool_name, {})
        dl     = self._dl.get(tool_name, 0)
        score  = 0.0
        matched: list[str] = []

        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            matched.append(term)
            df  = self._df.get(term, 0)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (freq * (self.K1 + 1)) / (
                freq + self.K1 * (1 - self.B + self.B * (dl / max(self._avg_dl, 1)))
            )
            score += idf * tf_norm

        return score, matched

    # ── BaseToolRouter interface ──────────────────────────────────────────────

    def find_candidates(
        self,
        intent: str,
        tools: list[ToolSchema],
        top_k: int = 5,
    ) -> list[ToolSchema]:
        # Re-index if tools changed
        if tools is not self._tools or not self._index:
            self.index(tools)

        query_terms = _tokenize(intent)
        N = len(tools)

        results: list[tuple[float, list[str], ToolSchema]] = []
        for tool in tools:
            score, matched = self._score(query_terms, tool["name"], N)
            if score > 0:
                results.append((score, matched, tool))

        results.sort(key=lambda x: x[0], reverse=True)

        out: list[ToolSchema] = []
        for score, matched, tool in results[:top_k]:
            enriched = dict(tool)
            enriched["_score"]         = round(score, 3)
            enriched["_matched_terms"] = matched
            out.append(enriched)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# RegexToolRouter — exact-match fallback
# ──────────────────────────────────────────────────────────────────────────────

class RegexToolRouter(BaseToolRouter):
    """
    Fast regex-based router. Use as a fallback when BM25 returns no results,
    or as a pre-filter before BM25 for very large catalogs (500+ tools).
    """

    def find_candidates(
        self,
        intent: str,
        tools: list[ToolSchema],
        top_k: int = 5,
    ) -> list[ToolSchema]:
        # Build pattern from each word in intent
        words = [re.escape(w) for w in intent.lower().split() if len(w) > 2]
        if not words:
            return []

        scored: list[tuple[int, list[str], ToolSchema]] = []
        for tool in tools:
            text    = _doc_text(tool).lower()
            matched = [w for w in words if re.search(w, text)]
            if matched:
                scored.append((len(matched), matched, tool))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: list[ToolSchema] = []
        for score, matched, tool in scored[:top_k]:
            enriched = dict(tool)
            enriched["_score"]         = float(score)
            enriched["_matched_terms"] = matched
            out.append(enriched)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# CascadeToolRouter — BM25 with Regex fallback (recommended default)
# ──────────────────────────────────────────────────────────────────────────────

class CascadeToolRouter(BaseToolRouter):
    """
    Tries BM25 first. Falls back to Regex if BM25 returns nothing.
    Recommended for production use.
    """

    def __init__(self) -> None:
        self._bm25  = BM25ToolRouter()
        self._regex = RegexToolRouter()

    def find_candidates(
        self,
        intent: str,
        tools: list[ToolSchema],
        top_k: int = 5,
    ) -> list[ToolSchema]:
        results = self._bm25.find_candidates(intent, tools, top_k)
        if not results:
            results = self._regex.find_candidates(intent, tools, top_k)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# VisualTopologyRouter — v0.2 placeholder
# ──────────────────────────────────────────────────────────────────────────────

class VisualTopologyRouter(BaseToolRouter):
    """
    v0.2 — Visual Routing Engine.

    Renders the tool catalog as a routing graph (SVG/PNG) and passes it
    to a multimodal LLM. The LLM selects the next tool by reading the
    spatial layout of the graph — using vision, not language prediction.

    This bypasses:
      - Lost in the Middle (tools appear in 2D space, not linear text)
      - Sycophantic drift (vision encoder ≠ RLHF-biased language head)

    Status: placeholder. Implement by wrapping the TypeScript renderer
    via subprocess, or by porting renderer.ts to Python.
    """

    def find_candidates(
        self,
        intent: str,
        tools: list[ToolSchema],
        top_k: int = 5,
    ) -> list[ToolSchema]:
        raise NotImplementedError(
            "VisualTopologyRouter is planned for v0.2. "
            "Use BM25ToolRouter or CascadeToolRouter for now."
        )
