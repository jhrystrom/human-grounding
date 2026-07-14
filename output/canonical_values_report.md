# Canonical Values — by Paper Section

_Generated 2026-07-14 12:05 by `scripts/report_canonical_values.py`._

Values are read from precomputed artifacts under `output/` (and raw coordinates/corpus under `data/`); nothing is re-embedded or re-bootstrapped. Narrative design constants (participant counts, completion time, model counts) are from the paper spec and marked as such; methods constants (threshold grid, D_e, MDS dimensionality, clustering algorithm) are read from the pipeline code.

Datasets: **rai** (Responsible AI), **welfare** (Welfare), **gov-ai** (Government AI). Experiments: `policy = {rai, welfare}`, `gov-ai = {gov-ai}`.

## Abstract

Headline values only (no CIs, no group disparities).

**Best-model gap to human-human agreement:**
- Responsible AI: 27.0pp
- Welfare: 19.5pp
- Government AI: 15.2pp

**Best-model gap to full-panel oracle:** ranges 12.9-29.2pp across datasets (Responsible AI 12.9pp, Welfare 13.3pp, Government AI 29.2pp).

- **Grounding-to-clustering rank correlation:** Spearman $\rho$ = 0.84 (policy; gov-ai $\rho$ = 0.85).
- **MMTEB-to-clustering rank correlation (if space permits):** Spearman $\rho$ = 0.56 (policy; gov-ai $\rho$ = 0.58).

## Introduction / Contributions

| Quantity | Value |
| --- | --- |
| Datasets | 3 |
| Studies | 2 |
| Participants (total) | 12 |
| Participants per study | 6 |
| Neural embedding models | 32 |
| Lexical baselines | 2 |

_(design constants — paper spec)_

- Headline human-model gap range: 15.2-27.0pp.
- Headline oracle-model gap range: 12.9-29.2pp.
- Headline grounding-to-clustering correlation: $\rho$ = 0.84 (policy).

## Section 3.1 — Exercise Description

| Quantity | Value | Source |
| --- | --- | --- |
| Statements per round | 20 | paper spec |
| Participants per panel | 6 | paper spec |
| Policy rounds per participant | 14 | paper spec |
| Gov-AI rounds per participant | 7 | paper spec |
| Completion time (policy) | half-day workshop | paper spec |
| Completion time (Gov-AI) | 1-1.5 hours | paper spec |
| Total individual statement placements | ~2,640 | paper spec |

**Mean statement occurrences per included statement, by dataset** (`statement_coverage_table.tex`):

| Dataset | Mean occurrences | Median occurrences |
| --- | --- | --- |
| Gov-AI | 3.44 | 4.00 |
| RAI | 3.95 | 2.00 |
| Welfare | 3.45 | 2.00 |

> Minimum occurrence per included statement is not tabulated in the coverage artifact; regenerate from `raw_triplets_gov-ai_policy.parquet` if a guaranteed floor must be stated.

## Section 3.2 — Datasets and Protected Attributes

| Dataset | Source statements | Used statements | Language | Protected attribute | Group counts (analysis groups) |
| --- | --- | --- | --- | --- | --- |
| Responsible AI | 873 | 294 | Danish | Gender | Kvinde=471; Mand=358 |
| Welfare | 377 | 174 | Danish | Political party (pseudonymised) | 1=109; 11=23; 2=81; 3=40; 4=42 |
| Government AI | 342 | 244 | Danish | None | n/a (no split) |

- Responsible AI: 44 statements in missing/excluded groups (dropped from group analysis).
- Welfare: 82 statements in missing/excluded groups (dropped from group analysis).
- Gov-AI: no protected-attribute analysis is performed.

## Section 4.1.1 — Human Reliability Methods

Definitions and configuration (not results):

Separation ratio: $r_i(t) = \max\{\delta_i(a,b), \delta_i(a,c)\} / \min\{\delta_i(a,b), \delta_i(a,c)\}$; threshold notation $\tau$.
- Human-human filtering: both raters satisfy $r_i(t) \geq \tau$.
- Model-human filtering: the human rater satisfies $r_i(t) \geq \tau$.

| Setting | Value |
| --- | --- |
| Expected disagreement D_e | 0.5 |
| AUC normalization range | [-1, 1] |
| AUC integration scale | log-$\tau$ |
| Threshold grid (RQ1 curve) | log-spaced, 50 points, $\tau \in [1, 8]$ |
| Threshold grid (main AUC) | log-spaced, 30 points, $\tau_{\max}$ = 6.5 |
| Log-spaced thresholds | yes |

**Contributing (between-rater) pairs per dataset** (from coordinate files; lower bound vs the full raw-triplet parquet):

| Dataset | Rater pairs |
| --- | --- |
| Government AI | 9 |
| Responsible AI | 8 |
| Welfare | 8 |

## Section 4.1.2 — Human Reliability Results

Per-dataset human reliability (human-human AUC + 95% CI from the alignment summary; within-rater drift summary from `context_drift_comparison.csv`; $\tau$ where $\alpha$ $\approx 0.8$ from the alpha curves):

| Dataset | Human-human AUC | 95% CI | Within-rater drift [95% CI] | $\tau$ @ $\alpha$ $\approx 0.8$ |
| --- | --- | --- | --- | --- |
| Responsible AI | 0.645 | [0.644, 0.647] | 0.020 [0.016, 0.024] | between $\tau \approx 3.42$; within $\tau \approx 1.09$ |
| Welfare | 0.672 | [0.668, 0.676] | 0.026 [0.019, 0.035] | between $\tau \approx 3.15$; within $\tau \approx 2.44$ |
| Government AI | 0.488 | [0.481, 0.495] | n/a | between $\tau \approx 7.35$; within $\tau \approx 3.15$ |

**Retained triplets** (from `fairness_triplet_counts.tex`; $d=1 \approx$ all eligible, d=4 a mid/high threshold):

| Dataset | Eligible (d=1) | Retained d=2 (%) | Retained d=4 (%) |
| --- | --- | --- | --- |
| Responsible AI | 175,972 | 26,083 (14.8%) | 3,478 (2.0%) |
| Welfare | 133,789 | 20,405 (15.3%) | 2,703 (2.0%) |

> Gov-AI has no demographic split, so it is absent from the group triplet-count artifact.

**Group-conditional human reliability AUC** ($\Delta_{\mathrm{human}}$ = $\alpha_{\max} - \alpha_{\min}$):

| Dataset | Group AUCs | Min group | Max group | $\Delta_{\mathrm{human}}$ disparity |
| --- | --- | --- | --- | --- |
| Responsible AI | High=0.632; Low=0.652; Medium=0.631 | 0.631 | 0.652 | 0.021 |
| Welfare | 1=0.684; 11=0.679; 2=0.658; 3=0.670; 4=0.661 | 0.658 | 0.684 | 0.026 |
| Government AI | Overall=0.491 | 0.491 | 0.491 | 0.000 |

> Per-group human bootstrap CIs are not persisted separately; regenerate with `scripts/alpha_distance_plot.py` if group-level CIs are required.

## Section 4.2.1 — Neural-Human Alignment Methods

| Setting | Value |
| --- | --- |
| Neural models | 32 |
| Lexical baselines | 2 |
| Total representations evaluated | 34 |
| Model distance | cosine distance |
| Instruction variants | 5 |
| Instruction formulations | 2 (generic, dataset-context) |
| Rank aggregation | Borda count |
| Rank-comparison metric | Spearman rho |
| Bootstrap iterations (cached) | 10 |

For normalized vectors, cosine distance is monotone in squared Euclidean distance, so triplet orderings are equivalent.

**Oracle construction:** full-panel consensus dissimilarity (mean of per-rater normalized layout distances, missing pairs mean-filled), then metric MDS (SMACOF) at dimensionality q = 10 (capped at n_statements - 1), n_init = 4. The oracle embedding is then evaluated through the identical triplet-$\alpha$ / AUC pipeline as the neural models.

## Section 4.2.2 — Neural-Human Alignment Results

| Dataset | Human AUC | Oracle AUC | Best-model AUC | Best 95% CI | Best model | Human-model gap | Oracle-model gap | Oracle-human diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Responsible AI | 0.645 | 0.504 | 0.375 | [0.369, 0.382] | multilingual-e5-large-instruct | 27.0pp | 12.9pp | -14.1pp |
| Welfare | 0.672 | 0.609 | 0.476 | [0.442, 0.492] | paraphrase-multilingual-mpnet | 19.5pp | 13.3pp | -6.3pp |
| Government AI | 0.488 | 0.627 | 0.336 | [0.309, 0.373] | mmBERTscandi-base-embedding | 15.2pp | 29.2pp | +13.9pp |

- Best model overall: paraphrase-multilingual-mpnet (0.385 mean across datasets); second-best: multilingual-e5-large-instruct (0.377).
- Responsible AI score range across models: -0.045 to 0.375.
- Welfare score range across models: -0.155 to 0.476.
- Government AI score range across models: 0.108 to 0.336.

**MMTEB rank vs grounded rank:**

| Experiment | Spearman $\rho$ | p-value | Shared models | Best grounded model | Its MMTEB rank |
| --- | --- | --- | --- | --- | --- |
| policy | 0.625 | 0.0001 | 32 | paraphrase-multilingual-mpnet | 13 of 32 |
| gov-ai | 0.557 | 0.0009 | 32 | mmBERTscandi-base-embedding | 11 of 32 |

- Best improvement over default: +0.075; worst change: -0.075; mean |change|: 0.031 across 40 model$\times$variant$\times$dataset cells.
- Smallest remaining oracle gap under best instruction: multilingual-e5-large-instruct on Welfare = 11.5pp.

## Figure 4 — Main Alignment Figure

| Dataset | Human-human AUC | Oracle-human AUC | Best-model AUC | Best group AUC | Worst group AUC |
| --- | --- | --- | --- | --- | --- |
| Responsible AI | 0.645 | 0.504 | 0.375 | 0.403 | 0.347 |
| Welfare | 0.672 | 0.609 | 0.476 | n/a | n/a |
| Government AI | 0.488 | 0.627 | 0.336 | 0.336 | 0.336 |

Caption should state: human bars = human-human agreement; oracle bars = oracle-human agreement; model bars = model-human agreement.

## Group-Conditional Alignment Results

**Responsible AI** (best model: multilingual-e5-large-instruct):

| Group | Best-model AUC | Statements in group |
| --- | --- | --- |
| Kvinde | 0.375 | 471 |
| Mand | 0.347 | 358 |
| Unknown | 0.403 | n/a |

- Raw group disparity $\Delta_{\mathrm{group}}$ = 0.056 (max 0.403 - min 0.347).
- Unadjusted $\Delta_{\mathrm{group}}$ (fairness bootstrap) = 0.028 [0.026,0.029], p <0.001.
- Adjusted $\Delta_{\mathrm{adj}}$ (length + lexical controls) = 0.011 [0.007,0.015], p <0.001.

**Welfare** (best model: paraphrase-multilingual-mpnet):

| Group | Best-model AUC | Statements in group |
| --- | --- | --- |
| 1 | 0.511 | 109 |
| 11 | 0.483 | 23 |
| 2 | 0.446 | 81 |
| 3 | 0.469 | 40 |
| 4 | 0.472 | 42 |

- Raw group disparity $\Delta_{\mathrm{group}}$ = 0.065 (max 0.511 - min 0.446).
- Unadjusted $\Delta_{\mathrm{group}}$ (fairness bootstrap) = 0.068 [0.062,0.073], p <0.001.
- Adjusted $\Delta_{\mathrm{adj}}$ (length + lexical controls) = 0.031 [0.018,0.047], p <0.001.

**Retained triplets per group** (d = 1, 2, 4; from `fairness_triplet_counts.tex`):

| Dataset | Group | d=1 | d=2 | d=4 |
| --- | --- | --- | --- | --- |
| Responsible AI | Women | 97,988 | 14,523 | 1,956 |
| Responsible AI | Men | 77,984 | 11,560 | 1,522 |
| Welfare | Party 1 | 26,418 | 4,213 | 629 |
| Welfare | Party 11 | 27,041 | 4,159 | 552 |
| Welfare | Party 2 | 26,394 | 3,948 | 495 |
| Welfare | Party 3 | 27,017 | 4,165 | 558 |
| Welfare | Party 4 | 26,919 | 3,920 | 469 |

## Qualitative Error Analysis

> **Not in `output/`** — high-separation error counts / categories: no persisted artifact — this analysis is not produced by any script in `output/`. Report manually: selection threshold (e.g. r_i(t) > 13), total inspected examples, model name, and 2-3 representative cases.

## Section 4.3.1 — Clustering Methods

| Setting | Value |
| --- | --- |
| Clustering algorithm | Ward-linkage agglomerative clustering |
| Model distance representation | embedding vectors (Ward on Euclidean) |
| Number of clusters K | mean number of human clusters per round |
| Human reference | ARI between paired participants (same round) |
| Model reference | mean ARI against both participants |
| Bootstrap resamples (Spearman CIs) | 2000 |
| Models included | 32 neural + 2 lexical |

> Singleton-cluster treatment and the human cluster-count threshold procedure are implemented in `scripts/clustering.py`; state them from the code if the methods paragraph needs the exact rule.

## Section 4.3.2 — Clustering Results

Per-dataset clustering ARI (human vs best model). Human CI = per-round 2.5/97.5 percentile band; model ARI has no persisted bootstrap CI.

| Dataset | Human ARI | Human CI (rounds) | Best-model ARI | Best model | Human-model ARI gap |
| --- | --- | --- | --- | --- | --- |
| Responsible AI | 0.307 | [0.052, 0.782] | 0.274 | multilingual-e5-large-instruct | +0.033 |
| Welfare | 0.394 | [0.056, 0.861] | 0.274 | multilingual-e5-large-instruct | +0.120 |
| Government AI | 0.159 | [-0.013, 0.406] | 0.195 | text-embedding-3-small | -0.036 |

> Statistical significance of the human-model ARI gap is not persisted; the 2000-resample bootstrap in `scripts/clustering.py` produces it on demand.

**Grounding-to-ARI and MMTEB-to-ARI Spearman** (`cluster_spearman_by_experiment.csv`):

| Source | Experiment | n_models | Spearman $\rho$ | 95% CI |
| --- | --- | --- | --- | --- |
| OurExercise | policy | 34 | 0.843 | [0.611, 0.935] |
| MMTEB | policy | 32 | 0.563 | [0.225, 0.797] |
| OurExercise | gov-ai | 34 | 0.846 | [0.735, 0.918] |
| MMTEB | gov-ai | 32 | 0.577 | [0.126, 0.782] |

Scope: datasets Responsible AI, Welfare, Government AI; models = the shared neural+lexical set per experiment (see n_models).

**Robustness — K choice** (`k_sensitivity_table.tex`): each row is a clustering setup with its ARI-rank $\rho$ vs the reference, and grounding/MMTEB-to-ARI $\rho$ under that K:

| Clustering setup | ARI rank $\rho$ | Grounding-ARI $\rho$ | MMTEB-ARI $\rho$ |
| --- | --- | --- | --- |
| Human-derived K | 1.00 | 0.84 | 0.57 |
| Silhouette-selected K | 0.91 | 0.77 | 0.57 |

> Ward-vs-k-means rank correlation and the human-model ARI-gap difference are computed inside `scripts/clustering.py` but not persisted to `output/`; re-run that script to tabulate them.

## Conclusion

- Human-model gap range: 15.2-27.0pp.
- Oracle-model gap range: 12.9-29.2pp.
- MMTEB-to-grounded rank correlation: policy $\rho$=0.62, gov-ai $\rho$=0.56.
- Grounding-to-clustering $\rho$ = 0.84 (policy), 0.85 (gov-ai).
- MMTEB-to-clustering $\rho$ = 0.56 (policy), 0.58 (gov-ai).

## Limitations

| Quantity | Value |
| --- | --- |
| Participants (total) | 12 |
| Participants per panel | 6 |
| Max unique rater pairs per panel | 15 |
| Actual contributing rater pairs (policy, coords) | 9 |
| Statements per round | 20 |
| Oracle panel members | 6 |
| Instruction variants $\times$ conditions | 5 $\times$ 2 |
| Datasets | 3 |
| Stakeholder communities | 2 |

- The oracle is in-sample and deliberately favorable.
- Human arrangements are local and 2D.
- Subset stability is within-panel, not across new panels.
- Clustering and grounding share the same layouts.

## Ethical Considerations

- Responsible AI protected-group statement counts: Kvinde=471; Mand=358 (missing/excluded: 44).
- Welfare protected-group statement counts: 1=109; 11=23; 2=81; 3=40; 4=42 (missing/excluded: 82).
- Responsible AI: raw $\Delta_{\mathrm{group}}$ = 0.028 [0.026,0.029] (p <0.001); adjusted $\Delta_{\mathrm{adj}}$ = 0.011 [0.007,0.015] (p <0.001).
- Welfare: raw $\Delta_{\mathrm{group}}$ = 0.068 [0.062,0.073] (p <0.001); adjusted $\Delta_{\mathrm{adj}}$ = 0.031 [0.018,0.047] (p <0.001).

Participant demographics (gender, education, sector, location, experience) were collected for the RAI panel; welfare uses pseudonymised party.

## Appendix: Descriptive Statistics

Canonical occurrence / co-occurrence (`statement_coverage_table.tex`):

| Dataset | Used statements | Mean occ. | Median occ. | Co-occurring % pairs |
| --- | --- | --- | --- | --- |
| Gov-AI | 244 | 3.44 | 4.00 | 13.07 |
| RAI | 294 | 3.95 | 2.00 | 11.91 |
| Welfare | 174 | 3.45 | 2.00 | 17.77 |

Study logistics from coordinate files (**subset** of the full raw-triplet parquet — a lower bound):

| Dataset | Placements | Unique stmts | Participants | Rounds | Occ. mean/min/max |
| --- | --- | --- | --- | --- | --- |
| Government AI | 840 | 244 | 6 | 42 | 3.44 / 2 / 12 |
| Responsible AI | 1,180 | 301 | 6 | 59 | 3.92 / 1 / 22 |
| Welfare | 620 | 178 | 6 | 31 | 3.48 / 1 / 12 |

Eligible / retained triplets (sum of demographic-group rows):

| Dataset | Eligible (d=1) | Retained d=2 | Retained d=4 |
| --- | --- | --- | --- |
| Responsible AI | 175,972 | 26,083 | 3,478 |
| Welfare | 133,789 | 20,405 | 2,703 |

## Appendix: Oracle

- Consensus-distance aggregation: mean of per-rater normalized layout distances over co-occurring pairs (missing pairs mean-filled).
- MDS: metric MDS (SMACOF), dimensionality q = 10 (capped at n - 1), n_init = 4.
- MDS stress / reconstruction error is computed at fit time but not persisted; re-run the oracle fit to report it.

| Dataset | Oracle AUC | Human AUC | Best-model AUC | Oracle-model gap | Oracle-human diff |
| --- | --- | --- | --- | --- | --- |
| Responsible AI | 0.504 | 0.645 | 0.375 | 12.9pp | -14.1pp |
| Welfare | 0.609 | 0.672 | 0.476 | 13.3pp | -6.3pp |
| Government AI | 0.627 | 0.488 | 0.336 | 29.2pp | +13.9pp |

## Appendix: Instruction Robustness

Per model $\times$ variant $\times$ dataset ($\Delta$ vs default = variant AUC - plain, non-instructed model AUC):

| Base model | Dataset | Variant | AUC | $\Delta$ vs default |
| --- | --- | --- | --- | --- |
| EmbeddingGemma-Scandi | Responsible AI | cluster | 0.254 | -0.056 |
| EmbeddingGemma-Scandi | Responsible AI | grouping | 0.234 | -0.075 |
| EmbeddingGemma-Scandi | Responsible AI | meaning | 0.273 | -0.037 |
| EmbeddingGemma-Scandi | Responsible AI | similarity | 0.243 | -0.066 |
| EmbeddingGemma-Scandi | Responsible AI | topic | 0.270 | -0.039 |
| EmbeddingGemma-Scandi | Welfare | cluster | 0.409 | -0.028 |
| EmbeddingGemma-Scandi | Welfare | grouping | 0.394 | -0.043 |
| EmbeddingGemma-Scandi | Welfare | meaning | 0.433 | -0.004 |
| EmbeddingGemma-Scandi | Welfare | similarity | 0.378 | -0.058 |
| EmbeddingGemma-Scandi | Welfare | topic | 0.435 | -0.002 |
| Qwen3-Embedding-0.6B | Responsible AI | cluster | 0.152 | -0.010 |
| Qwen3-Embedding-0.6B | Responsible AI | grouping | 0.163 | +0.000 |
| Qwen3-Embedding-0.6B | Responsible AI | meaning | 0.171 | +0.008 |
| Qwen3-Embedding-0.6B | Responsible AI | similarity | 0.174 | +0.012 |
| Qwen3-Embedding-0.6B | Responsible AI | topic | 0.204 | +0.041 |
| Qwen3-Embedding-0.6B | Welfare | cluster | 0.239 | -0.001 |
| Qwen3-Embedding-0.6B | Welfare | grouping | 0.254 | +0.014 |
| Qwen3-Embedding-0.6B | Welfare | meaning | 0.270 | +0.030 |
| Qwen3-Embedding-0.6B | Welfare | similarity | 0.237 | -0.002 |
| Qwen3-Embedding-0.6B | Welfare | topic | 0.315 | +0.075 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | cluster | 0.254 | -0.033 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | grouping | 0.235 | -0.052 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | meaning | 0.242 | -0.045 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | similarity | 0.255 | -0.033 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | topic | 0.241 | -0.046 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | cluster | 0.232 | -0.041 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | grouping | 0.213 | -0.060 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | meaning | 0.224 | -0.049 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | similarity | 0.232 | -0.040 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | topic | 0.207 | -0.066 |
| multilingual-e5-large-instruct | Responsible AI | cluster | 0.375 | -0.000 |
| multilingual-e5-large-instruct | Responsible AI | grouping | 0.378 | +0.003 |
| multilingual-e5-large-instruct | Responsible AI | meaning | 0.356 | -0.019 |
| multilingual-e5-large-instruct | Responsible AI | similarity | 0.374 | -0.001 |
| multilingual-e5-large-instruct | Responsible AI | topic | 0.371 | -0.003 |
| multilingual-e5-large-instruct | Welfare | cluster | 0.450 | +0.004 |
| multilingual-e5-large-instruct | Welfare | grouping | 0.436 | -0.010 |
| multilingual-e5-large-instruct | Welfare | meaning | 0.402 | -0.044 |
| multilingual-e5-large-instruct | Welfare | similarity | 0.415 | -0.031 |
| multilingual-e5-large-instruct | Welfare | topic | 0.494 | +0.048 |

Best / worst variant and range per model $\times$ dataset:

| Base model | Dataset | Best variant | Worst variant | Range |
| --- | --- | --- | --- | --- |
| EmbeddingGemma-Scandi | Responsible AI | meaning (0.273) | grouping (0.234) | 0.038 |
| EmbeddingGemma-Scandi | Welfare | topic (0.435) | similarity (0.378) | 0.057 |
| Qwen3-Embedding-0.6B | Responsible AI | topic (0.204) | cluster (0.152) | 0.052 |
| Qwen3-Embedding-0.6B | Welfare | topic (0.315) | similarity (0.237) | 0.077 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | similarity (0.255) | grouping (0.235) | 0.020 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | similarity (0.232) | topic (0.207) | 0.026 |
| multilingual-e5-large-instruct | Responsible AI | grouping (0.378) | meaning (0.356) | 0.022 |
| multilingual-e5-large-instruct | Welfare | topic (0.494) | meaning (0.402) | 0.092 |

Generic vs dataset-context conditions:

| Base model | Dataset | Variant | Context | AUC |
| --- | --- | --- | --- | --- |
| EmbeddingGemma-Scandi | Responsible AI | cluster | generic | 0.256 |
| EmbeddingGemma-Scandi | Responsible AI | cluster | rai | 0.259 |
| EmbeddingGemma-Scandi | Responsible AI | grouping | generic | 0.237 |
| EmbeddingGemma-Scandi | Responsible AI | grouping | rai | 0.246 |
| EmbeddingGemma-Scandi | Responsible AI | meaning | generic | 0.273 |
| EmbeddingGemma-Scandi | Responsible AI | meaning | rai | 0.253 |
| EmbeddingGemma-Scandi | Responsible AI | similarity | generic | 0.246 |
| EmbeddingGemma-Scandi | Responsible AI | similarity | rai | 0.245 |
| EmbeddingGemma-Scandi | Responsible AI | topic | generic | 0.272 |
| EmbeddingGemma-Scandi | Responsible AI | topic | rai | 0.245 |
| EmbeddingGemma-Scandi | Welfare | cluster | generic | 0.409 |
| EmbeddingGemma-Scandi | Welfare | cluster | welfare | 0.333 |
| EmbeddingGemma-Scandi | Welfare | grouping | generic | 0.393 |
| EmbeddingGemma-Scandi | Welfare | grouping | welfare | 0.330 |
| EmbeddingGemma-Scandi | Welfare | meaning | generic | 0.427 |
| EmbeddingGemma-Scandi | Welfare | meaning | welfare | 0.329 |
| EmbeddingGemma-Scandi | Welfare | similarity | generic | 0.380 |
| EmbeddingGemma-Scandi | Welfare | similarity | welfare | 0.331 |
| EmbeddingGemma-Scandi | Welfare | topic | generic | 0.430 |
| EmbeddingGemma-Scandi | Welfare | topic | welfare | 0.331 |
| Qwen3-Embedding-0.6B | Responsible AI | cluster | generic | 0.156 |
| Qwen3-Embedding-0.6B | Responsible AI | cluster | rai | 0.141 |
| Qwen3-Embedding-0.6B | Responsible AI | grouping | generic | 0.165 |
| Qwen3-Embedding-0.6B | Responsible AI | grouping | rai | 0.135 |
| Qwen3-Embedding-0.6B | Responsible AI | meaning | generic | 0.171 |
| Qwen3-Embedding-0.6B | Responsible AI | meaning | rai | 0.183 |
| Qwen3-Embedding-0.6B | Responsible AI | similarity | generic | 0.178 |
| Qwen3-Embedding-0.6B | Responsible AI | similarity | rai | 0.204 |
| Qwen3-Embedding-0.6B | Responsible AI | topic | generic | 0.203 |
| Qwen3-Embedding-0.6B | Responsible AI | topic | rai | 0.171 |
| Qwen3-Embedding-0.6B | Welfare | cluster | generic | 0.249 |
| Qwen3-Embedding-0.6B | Welfare | cluster | welfare | 0.258 |
| Qwen3-Embedding-0.6B | Welfare | grouping | generic | 0.267 |
| Qwen3-Embedding-0.6B | Welfare | grouping | welfare | 0.252 |
| Qwen3-Embedding-0.6B | Welfare | meaning | generic | 0.280 |
| Qwen3-Embedding-0.6B | Welfare | meaning | welfare | 0.319 |
| Qwen3-Embedding-0.6B | Welfare | similarity | generic | 0.249 |
| Qwen3-Embedding-0.6B | Welfare | similarity | welfare | 0.333 |
| Qwen3-Embedding-0.6B | Welfare | topic | generic | 0.325 |
| Qwen3-Embedding-0.6B | Welfare | topic | welfare | 0.270 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | cluster | generic | 0.247 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | cluster | rai | 0.244 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | grouping | generic | 0.228 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | grouping | rai | 0.231 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | meaning | generic | 0.233 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | meaning | rai | 0.234 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | similarity | generic | 0.246 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | similarity | rai | 0.246 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | topic | generic | 0.234 |
| Qwen3-Embedding-Scandi-0.6B | Responsible AI | topic | rai | 0.233 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | cluster | generic | 0.234 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | cluster | welfare | 0.217 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | grouping | generic | 0.217 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | grouping | welfare | 0.196 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | meaning | generic | 0.227 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | meaning | welfare | 0.204 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | similarity | generic | 0.236 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | similarity | welfare | 0.227 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | topic | generic | 0.207 |
| Qwen3-Embedding-Scandi-0.6B | Welfare | topic | welfare | 0.193 |
| multilingual-e5-large-instruct | Responsible AI | cluster | generic | 0.380 |
| multilingual-e5-large-instruct | Responsible AI | cluster | rai | 0.350 |
| multilingual-e5-large-instruct | Responsible AI | grouping | generic | 0.382 |
| multilingual-e5-large-instruct | Responsible AI | grouping | rai | 0.357 |
| multilingual-e5-large-instruct | Responsible AI | meaning | generic | 0.360 |
| multilingual-e5-large-instruct | Responsible AI | meaning | rai | 0.353 |
| multilingual-e5-large-instruct | Responsible AI | similarity | generic | 0.378 |
| multilingual-e5-large-instruct | Responsible AI | similarity | rai | 0.363 |
| multilingual-e5-large-instruct | Responsible AI | topic | generic | 0.376 |
| multilingual-e5-large-instruct | Responsible AI | topic | rai | 0.388 |
| multilingual-e5-large-instruct | Welfare | cluster | generic | 0.445 |
| multilingual-e5-large-instruct | Welfare | cluster | welfare | 0.386 |
| multilingual-e5-large-instruct | Welfare | grouping | generic | 0.431 |
| multilingual-e5-large-instruct | Welfare | grouping | welfare | 0.376 |
| multilingual-e5-large-instruct | Welfare | meaning | generic | 0.401 |
| multilingual-e5-large-instruct | Welfare | meaning | welfare | 0.388 |
| multilingual-e5-large-instruct | Welfare | similarity | generic | 0.413 |
| multilingual-e5-large-instruct | Welfare | similarity | welfare | 0.378 |
| multilingual-e5-large-instruct | Welfare | topic | generic | 0.479 |
| multilingual-e5-large-instruct | Welfare | topic | welfare | 0.439 |

## Appendix: Metric Robustness

| Setting | Value |
| --- | --- |
| Main $\tau_{\max}$ | 6.5 |
| Alternative $\tau_{\max}$ | 4.0, 6.5, 8.0, 10.0 |
| Threshold points (main) | 30 |
| Alternative threshold points | 15, 50 |
| Integration schemes | log-x (main) and linear-d |
| Expected disagreement | fixed D_e = 0.5 (vs empirical D_e) |

Rank-stability vs the main configuration (`auc_sensitivity_table.tex`):

| Configuration | Full-rank $\rho$ | Top-10 $\rho$ | Top-10 overlap |
| --- | --- | --- | --- |
| d_max=4 | 1.00 | 0.95 | 9/10 |
| d_max=6.5 | 1.00 | 1.00 | 10/10 |
| d_max=8 | 1.00 | 0.99 | 10/10 |
| d_max=10 | 1.00 | 0.99 | 10/10 |
| n_points=15 | 1.00 | 1.00 | 10/10 |
| n_points=50 | 1.00 | 1.00 | 10/10 |
| Linear-d integration | 1.00 | 0.96 | 10/10 |

## Canonical headline value block

Minimal set from which the abstract, main results, conclusion, and captions are populated.

| Dataset | Human AUC | Oracle AUC | Best-model AUC | Human-model gap | Oracle-model gap |
| --- | --- | --- | --- | --- | --- |
| Responsible AI | 0.645 | 0.504 | 0.375 | 27.0pp | 12.9pp |
| Welfare | 0.672 | 0.609 | 0.476 | 19.5pp | 13.3pp |
| Government AI | 0.488 | 0.627 | 0.336 | 15.2pp | 29.2pp |

- MMTEB-to-grounded $\rho$: policy=0.62, gov-ai=0.56
- Grounding-to-ARI $\rho$: policy=0.84, gov-ai=0.85.
- MMTEB-to-ARI $\rho$: policy=0.56, gov-ai=0.58.
- Best instruction gain ($\Delta$ vs default): +0.075.
- Best instructed model's remaining oracle gap: multilingual-e5-large-instruct on Welfare = 11.5pp.
- Responsible AI group disparity: raw 0.028 [0.026,0.029], adjusted 0.011 [0.007,0.015].
- Welfare group disparity: raw 0.068 [0.062,0.073], adjusted 0.031 [0.018,0.047].
- Responsible AI clustering ARI: human 0.307, best model multilingual-e5-large-instruct 0.274.
- Welfare clustering ARI: human 0.394, best model multilingual-e5-large-instruct 0.274.
- Government AI clustering ARI: human 0.159, best model text-embedding-3-small 0.195.
