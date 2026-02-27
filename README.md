# human-grounding

Code for the paper *"Grounding Text Embeddings in Expert Human Associations"* (ACL submission).

The paper introduces the **Human Grounding Exercise**, a methodology for measuring whether the geometric distances in a text embedding space reflect the conceptual distinctions that domain experts actually intend. Using two Danish-language datasets on "wicked problems", the paper finds that neural embedding models fall 20–30 percentage points short of human inter-rater reliability, and that this gap propagates directly into downstream clustering quality (Spearman ρ = 0.91 between human grounding scores and clustering ARI, versus ρ = 0.48 for MMTEB).

## Overview

Human experts physically arrange printed statements on a canvas such that spatial proximity reflects perceived semantic similarity. Their placements are digitised, and a triplet-based variant of Krippendorff's alpha is used to measure:

1. **RQ1 – Human reliability**: how consistently experts agree with each other and with their own earlier judgements.
2. **RQ2 – Neural alignment**: how well embedding models reproduce those expert judgements.
3. **RQ3 – Downstream clustering**: whether human-grounding scores predict real-world clustering quality better than standard MMTEB benchmark scores.

### Datasets

| Name | Description | Protected attribute |
|---|---|---|
| `rai` | Expert panel statements on Responsible AI governance challenges | Gender |
| `welfare` | Danish local politicians' statements on the future of welfare | Political party (pseudonymised) |

Both datasets are in Danish; an English-translated ablation is available via `translate_statements.py`.

## Installation

The project uses [uv](https://github.com/astral-sh/uv) and requires Python ≥ 3.12.

```bash
uv sync
```

Optional dependency groups:

```bash
uv sync --group vision   # OpenCV, PaddleOCR, TrOCR – needed for image processing
uv sync --group qr       # QR code generation/reading
uv sync --group notebook # Jupyter kernel
```

## Repository structure

```
human-grounding/
├── data/               # Raw and translated statement files
├── images/             # Workshop canvas images (input to coordinate extraction)
├── output/             # Generated CSVs (coordinates, embeddings, scores)
├── plots/              # Generated figures
├── scripts/            # Runnable analysis scripts (documented below)
└── src/
    └── human_grounding/ # Core library (data loading, embedding, evaluation, plotting)
```

## Scripts

All scripts are run from the repository root. Outputs land in `output/` (CSVs) and `plots/` (figures) unless noted otherwise.

---

### `scripts/combine_coordinates.py`

**Purpose**: Merges individual per-session coordinate CSV files produced by the Human Grounding Exercise into a single `combined_coordinates.csv`.

Each session file is named following the convention `<dataset>-n<N>-seed<S>-user_<ID>_coords.csv`. This script parses those filenames to extract the dataset name, seed, and user ID, then concatenates all 20-statement sessions into one flat table.

**Usage**:
```bash
uv run python scripts/combine_coordinates.py
```

**Output**: `output/combined_coordinates.csv`

**Notes**:
- Files that do not contain exactly 20 coordinate rows are skipped with a warning.
- Files beginning with `welfware` (a known typo in the raw data) are skipped.

---

### `scripts/translate_statements.py`

**Purpose**: Translates all Danish statements to English using the OpenAI API, producing an English-language copy of the statement corpus used in the ablation study (Appendix B of the paper).

Requires an OpenAI API key configured via the project's default config (see `src/human_grounding/oai.py`).

**Usage**:
```bash
uv run python scripts/translate_statements.py
```

**Output**: `data/translated_statements.csv` — a CSV with a `statement_id` column and an `english` column containing the translated text.

---

### `scripts/alpha_distance_plot.py`

**Purpose**: Computes triplet-based Krippendorff's alpha reliability curves as a function of the distance-ratio threshold *d*, with bootstrap confidence intervals and demographic breakdowns (RQ1 in the paper).

For each rater pair (within-rater: same participant, different rounds; between-rater: different participants, same round), the script:

1. Computes pairwise Euclidean distances between placed statements.
2. Builds all valid anchor–closer–farther triplets.
3. Measures agreement using a vectorised Krippendorff's alpha formula across a log-spaced range of threshold values.
4. Bootstraps confidence intervals (95%) by resampling rater pairs.
5. Identifies the best- and worst-performing demographic groups per dataset.

**Usage**:
```bash
uv run python scripts/alpha_distance_plot.py [--samples N] [--scale F]
```

| Flag | Default | Description |
|---|---|---|
| `--samples` | `5` | Number of bootstrap iterations (use ≥ 100 for publication-quality CIs) |
| `--scale` | `2.0` | Seaborn font scale for output figures |

**Outputs**:
- `output/alpha_data_demographic.csv` — tidy bootstrap alpha values per dataset, demographic group, and threshold *d* (used as input by `neural_alignment_plots.py`)
- `plots/alpha_distance_plot_demographic.pdf` — alpha vs *d* curve with 95% CI bands, split by best/worst demographic group
- `plots/agreement_auc_between_by_dataset.pdf` — bar chart of between-rater AUC per dataset (best / mean / worst demographic)

---

### `scripts/neural_alignment_plots.py`

**Purpose**: Evaluates how well each embedding model reproduces human expert placements (RQ2) and generates the main comparison figures used in the paper.

For each model, the script computes the same triplet-based Krippendorff's alpha against every human rater, summarises agreement as an AUC over the log-scaled threshold range, and bootstraps uncertainty. Results are then compared against MMTEB benchmark rankings via Spearman correlation.

**Usage**:
```bash
uv run python scripts/neural_alignment_plots.py [--scale F] [--english] [--cache] [--file {pdf,jpg,png}]
```

| Flag | Default | Description |
|---|---|---|
| `--scale` | `2.8` | Seaborn font scale for output figures |
| `--english` | off | Evaluate on English-translated statements instead of Danish originals |
| `--cache` | off | Skip recomputing embeddings; load `embedding_alignment_auc.csv` from disk |
| `--file` | `pdf` | Output figure format |

**Outputs**:
- `output/embedding_alignment_auc.csv` — per-model, per-bootstrap AUC scores
- `output/human_alignment_bootstrapped.csv` — aggregated mean alignment score per model (input to `clustering.py`)
- `output/mmteb_with_ranks.csv` — MMTEB scores and ranks for the evaluated model subset
- `plots/human_alignment_*.pdf` — bar chart of top-N models by human grounding score (with demographic breakdown facets)
- `plots/top_demographic_human_alignment.pdf` — faceted bar chart showing per-demographic alignment scores for the top-N models
- `plots/mmteb_vs_human_ranking.pdf` — dot-and-line plot comparing MMTEB ranks against human-grounded ranks

Logs the Spearman correlation between MMTEB scores and human grounding AUC scores.

---

### `scripts/clustering.py`

**Purpose**: Measures downstream clustering consistency between human experts and embedding models (RQ3 in the paper).

For each round (dataset × seed pair), clusters are derived using agglomerative Ward clustering — from (x, y) canvas coordinates for humans and from high-dimensional embeddings for models. The average number of human clusters per round sets the *k* for model clustering. Consistency is measured with the Adjusted Rand Index (ARI) and V-measure.

**Usage**:
```bash
uv run python scripts/clustering.py [--scale F] [--top N] [--english]
```

| Flag | Default | Description |
|---|---|---|
| `--scale` | `1.35` | Seaborn font scale for output figures |
| `--top` | `20` | Number of top models to include in the comparison bar chart |
| `--english` | off | Use English-translated statement embeddings |

**Outputs**:
- `output/human_cluster_consistency.csv` — per-round human-vs-human ARI and V-measure
- `output/model_cluster_consistency.csv` — per-round model-vs-human ARI and V-measure for all models
- `output/cluster_consistency_aggregated.csv` — mean scores per model across all rounds
- `plots/cluster_consistency_comparison_adjusted_rand_index.pdf`
- `plots/cluster_consistency_comparison_v_measure.pdf`

Also logs:
- Spearman ρ between human grounding score and clustering ARI
- Spearman ρ between MMTEB score and clustering ARI

## Pipeline order

Run the scripts in this order for a full reproduction:

```
combine_coordinates.py        # merge raw session CSVs
translate_statements.py       # (optional) generate English translations
alpha_distance_plot.py        # RQ1: human reliability
neural_alignment_plots.py     # RQ2: model alignment
clustering.py                 # RQ3: downstream clustering
```

## Citation

Anonymised for now.
