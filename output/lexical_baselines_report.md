# Lexical baseline report

Baselines evaluated: tfidf-char35, jaccard-binary

## Experiment: `policy` (35 models total)

### Top-N neural

| Rank | Model | Alignment AUC | ARI |
|---:|:---|---:|---:|
| 1 | paraphrase-multilingual-mpnet-base-v2 | 0.436 | 0.202 |
| 2 | multilingual-e5-large-instruct | 0.422 | 0.274 |
| 3 | text-embedding-3-large | 0.408 | 0.232 |
| 4 | paraphrase-multilingual-MiniLM-L12-v2 | 0.401 | 0.157 |
| 5 | EmbeddingGemma-Scandi-300m | 0.393 | 0.260 |
| 6 | nb-sbert-base | 0.391 | 0.178 |
| 7 | snowflake-arctic-embed-l-v2.0 | 0.389 | 0.214 |
| 8 | text-embedding-ada-002 | 0.376 | 0.185 |
| 9 | text-embedding-3-small | 0.374 | 0.154 |
| 10 | bge-m3 | 0.365 | 0.196 |

### Lexical baselines

| Model | Alignment AUC (rank) | ARI (rank) | Δ AUC vs best | Δ ARI vs best |
|:---|---:|---:|---:|---:|
| tfidf-char35 | 0.249 (23/35) | 0.217 (5/35) | +0.188 | +0.112 |
| jaccard-binary | -0.116 (34/35) | 0.014 (35/35) | +0.552 | +0.315 |

## Experiment: `gov-ai` (35 models total)

### Top-N neural

| Rank | Model | Alignment AUC | ARI |
|---:|:---|---:|---:|
| 1 | mmBERTscandi-base-embedding | 0.335 | 0.174 |
| 2 | mxbai-embed-large-v1 | 0.325 | 0.167 |
| 3 | text-embedding-3-large | 0.315 | 0.191 |
| 4 | paraphrase-multilingual-mpnet-base-v2 | 0.315 | 0.136 |
| 5 | Qwen3-Embedding-Scandi-0.6B | 0.310 | 0.174 |
| 6 | text-embedding-3-small | 0.308 | 0.195 |
| 7 | multilingual-e5-large-instruct | 0.304 | 0.190 |
| 8 | EmbeddingGemma-Scandi-300m | 0.303 | 0.188 |
| 9 | text-embedding-ada-002 | 0.302 | 0.194 |
| 10 | paraphrase-multilingual-MiniLM-L12-v2 | 0.297 | 0.133 |

### Lexical baselines

| Model | Alignment AUC (rank) | ARI (rank) | Δ AUC vs best | Δ ARI vs best |
|:---|---:|---:|---:|---:|
| tfidf-char35 | 0.153 (32/35) | 0.103 (27/35) | +0.182 | +0.092 |
| jaccard-binary | 0.101 (34/35) | 0.032 (35/35) | +0.234 | +0.163 |
