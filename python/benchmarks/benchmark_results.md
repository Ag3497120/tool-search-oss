## Benchmarks

> Measured on 50-tool catalog · 50 eval queries · Tokenizer: `tiktoken/cl100k_base`

### ① Context Compression

| | Tokens | vs baseline |
|---|---|---|
| **Without tool-search-oss** | 5,355 | — |
| **With tool-search-oss** | 970 | **82% reduction** |

### ② TTFT Improvement (Apple Silicon, local LLM)

| Model | Without | With | Saved | Speedup |
|---|---|---|---|---|
| gemma-3-4b (4-bit) | 1785ms | 323ms | 1462ms | **5.5x** |
| llama-3.1-8b (4-bit) | 2678ms | 485ms | 2192ms | **5.5x** |
| gemma-3-12b (4-bit) | 4462ms | 808ms | 3654ms | **5.5x** |

### ③ Routing Accuracy

| Metric | Score |
|---|---|
| BM25 top-1 accuracy | **96%** |
| BM25 top-3 accuracy | 100% |
| BM25 top-5 accuracy | 100% |
| Random baseline (50 tools) | 2.0% |
| Accuracy lift | **48x** over random |
| Search latency (avg / p99) | 0.03ms / 0.06ms |

### ④ API Cost Savings (1,000 calls/day)

| Model | Saved/call | Saved/year |
|---|---|---|
| Claude 3.5 Haiku | $0.00351 | **$1,280** (¥192,063) |
| Claude 3.5 Sonnet | $0.01316 | **$4,802** (¥720,236) |
| GPT-4o | $0.01096 | **$4,001** (¥600,197) |
| GPT-4o mini | $0.00066 | **$240** (¥36,012) |