# tool-search-oss

**Find the right MCP tool from 50+ without context collapse.**

When an LLM has 50 MCP tools available, it loads all definitions into context — and promptly forgets most of them ([Lost in the Middle](https://arxiv.org/abs/2307.03172)). tool-search-oss solves this with a dead-simple pattern:

```
Instead of:  LLM context = all 50 tool definitions (expensive)
With this:   LLM context = tool_summary() + search_tools("what I need") → top 5 only
```

**87% context reduction. Measured on 25-tool catalog.**

---

## How it works

### The defer_loading pattern

```typescript
// Step 1 — session start: give LLM only a lightweight catalog
tool_summary({ tools: myAllTools })
// → [SEARCH] read_file, list_dir, search_files...
//   [WRITE]  write_file, db_insert...
//   [RUN]    run_command, gh_push_commit...
//   (25 tools, 200 chars — not 2000)

// Step 2 — when LLM needs to act: search for relevant tools only
search_tools({ query: "read a file from disk", tools: myAllTools, topK: 5 })
// → MATCH 1: read_file (score: 11.01)
//   MATCH 2: write_file (score: 5.27)
//   (full definitions for top-5 only — 87% less context)

// Step 3 — call the actual tool
read_file({ path: "..." })
```

### Techniques

- **BM25** — default, handles camelCase/snake_case tokenization, no dependencies
- **Regex filter** — fast pre-filter for exact name matches
- **Visual Routing** *(experimental, v0.2)* — renders tool graph as PNG, LLM picks from spatial layout instead of text

---

## Install

```bash
npm install tool-search-oss
# or just add the MCP server to your config (no npm needed)
```

### MCP config (Claude Desktop / Cursor / Antigravity)

```json
{
  "mcpServers": {
    "tool-search-oss": {
      "command": "node",
      "args": ["--import", "tsx/esm", "/path/to/tool-search-oss/src/server.ts"]
    }
  }
}
```

---

## Tools

### `search_tools(query, tools, topK?)`

BM25 search over your tool catalog. Returns full definitions for top-N matches only.

```
query:   "send message to slack"
topK:    5  (default)
→ slack_send(8.01), email_send(3.28), http_post(1.2)
```

### `tool_summary(tools)`

Lightweight catalog grouped by category. Give this to LLM at session start.

```
[SEARCH]  read_file, list_dir, db_query, http_get...
[WRITE]   write_file, db_insert, set_config...
[RUN]     run_command, run_tests, gh_push_commit...
[ANALYZE] lint_code, embed_text, browser_screenshot...
[META]    get_config, list_tools...
```

### `visual_route(query, tools, currentTool?)` *(experimental)*

Renders your tool catalog as a routing graph. LLM selects the next tool by looking at the image — not by guessing from text.

> Why images? The vision encoder uses spatial reasoning, not language prediction.  
> Yellow = best match. Red = current. Gray = not relevant.  
> No probabilistic drift. No "Lost in the Middle".

---

## Benchmarks

Tested on a 25-tool catalog:

| Query | Top Result | Score | Context saved |
|-------|-----------|-------|---------------|
| "read a file from disk" | `read_file` | 11.01 | 87% |
| "send message to team"  | `slack_send` | 8.01 | 87% |
| "push code to github"   | `gh_push_commit` | 8.23 | 87% |

---

## Roadmap

- [x] BM25 search (v0.1)
- [x] Regex filter (v0.1)
- [x] Lightweight catalog / defer_loading (v0.1)
- [x] Visual Routing Engine (v0.2, experimental)
- [ ] Embedding-based search (v0.3)
- [ ] Python implementation
- [ ] Benchmark suite (50-tool, 100-tool)

---

## Why not just use embeddings?

Embeddings require a model, a vector store, and inference time. BM25 requires nothing — no API, no GPU, no install. For tool search (short descriptions, exact terminology), BM25 matches or beats embeddings in practice.

Add embeddings in v0.3 when you need semantic fuzzy matching.

---

## License

MIT
