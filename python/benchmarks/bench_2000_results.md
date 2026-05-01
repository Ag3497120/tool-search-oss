## 2,000-Tool Scale Benchmark

> Comparing tool-search-oss BM25 router vs Anthropic's published baseline

### Routing Accuracy (30 eval queries)

| Method | Accuracy |
|---|---|
| Random baseline (2,000 tools) | 0.050% |
| **Anthropic (without tool search)** | **34%** |
| **tool-search-oss BM25 top-1** | **57%** |
| tool-search-oss BM25 top-3 | 57% |
| tool-search-oss BM25 top-5 | 57% |

> BM25 achieves 57% vs Anthropic's 34% baseline — **1.7x improvement**

### TTFT at Scale

| Condition | Tokens | TTFT |
|---|---|---|
| Without router (2,000 tools) | ~31,526 | 55369ms |
| With tool-search-oss (top-3) | ~59 | 408ms |
| **Speedup** | **100% less** | **135.7x faster** |