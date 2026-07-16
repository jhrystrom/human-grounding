"""LaTeX table of lexical-baseline performance per dataset.

Loads per-(model, dataset) alignment AUC and clustering ARI from cached CSVs,
ranks all neural+lexical models within each dataset, and emits the
``tab:lexical-baselines`` LaTeX table.

Output: ``output/lexical_baselines_table.tex``.

Run *after* ``neural_alignment_plots.py`` and ``clustering.py`` so the
relevant CSVs exist on disk.
"""

import argparse
import sys
from pathlib import Path

import polars as pl
from loguru import logger

from human_grounding.directories import OUTPUT_DIR

LEXICAL_MODELS = ("tfidf-char35", "jaccard-binary")

# Humans and human-derived oracles are excluded from the ranked set: ranks are
# reported within the neural + lexical models only.
EXCLUDED_MODELS = ("Human", "human-mds-oracle")

# (dataset_key, display_name, clustering_csv_filename)
DATASETS: list[tuple[str, str, str]] = [
    ("rai", "Responsible AI", "model_cluster_consistency_policy.csv"),
    ("welfare", "Welfare", "model_cluster_consistency_policy.csv"),
    ("gov-ai", "Gov-AI", "model_cluster_consistency_gov-ai.csv"),
]

ALIGNMENT_CSV = "alignment_results_gov-ai_policy.csv"


def _per_dataset_alignment_auc(path: Path, dataset: str) -> pl.DataFrame:
    """Mean binary AUC per model on one dataset."""
    return (
        pl.read_csv(path)
        .filter(
            (pl.col("metric") == "binary_auc")
            & (pl.col("dataset") == dataset)
            & (~pl.col("model").is_in(EXCLUDED_MODELS))
        )
        .group_by("model")
        .agg(pl.col("auc").mean().alias("alignment_auc"))
    )


def _per_dataset_clustering_ari(path: Path, dataset: str) -> pl.DataFrame:
    """Mean adjusted Rand index per model on one dataset."""
    return (
        pl.read_csv(path)
        .filter(
            (pl.col("dataset") == dataset) & (~pl.col("model").is_in(EXCLUDED_MODELS))
        )
        .group_by("model")
        .agg(pl.col("adjusted_rand_index").mean().alias("ari"))
    )


def _ranked(scores: pl.DataFrame, value_col: str, rank_col: str) -> pl.DataFrame:
    """Add a 1-indexed dense rank column (higher value = rank 1)."""
    return scores.with_columns(
        pl.col(value_col)
        .rank(method="min", descending=True)
        .cast(pl.Int64)
        .alias(rank_col)
    )


def _baseline_rows(
    display_name: str,
    alignment: pl.DataFrame,
    clustering: pl.DataFrame,
) -> list[tuple[str, str, float, int, float, int]]:
    """Return ``(display, baseline, auc, auc_rank, ari, ari_rank)`` per lexical model."""
    alignment = _ranked(alignment, "alignment_auc", "auc_rank")
    clustering = _ranked(clustering, "ari", "ari_rank")
    joined = alignment.join(clustering, on="model", how="full", coalesce=True)

    n_align = alignment.height
    n_clust = clustering.height
    logger.info(
        f"[{display_name}] {n_align} models with alignment AUC, "
        f"{n_clust} models with clustering ARI."
    )

    rows: list[tuple[str, str, float, int, float, int]] = []
    for baseline in LEXICAL_MODELS:
        sub = joined.filter(pl.col("model") == baseline)
        if sub.is_empty():
            logger.warning(f"[{display_name}] {baseline} missing from inputs.")
            continue
        row = sub.row(0, named=True)
        rows.append(
            (
                display_name,
                baseline,
                row["alignment_auc"],
                row["auc_rank"],
                row["ari"],
                row["ari_rank"],
            )
        )
    return rows


def _build_tex(
    rows: list[tuple[str, str, float, int, float, int]],
    n_auc: int,
    n_ari: int,
) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Lexical baseline performance. Ranks are computed within the "
        r"full set of neural and lexical models.}",
        r"\label{tab:lexical-baselines}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Baseline} & \textbf{Alignment AUC} & "
        r"\textbf{AUC rank} & \textbf{Clustering ARI} & \textbf{ARI rank} \\",
        r"\midrule",
    ]
    for display, baseline, auc, auc_rank, ari, ari_rank in rows:
        lines.append(
            f"{display} & \\texttt{{{baseline}}} & "
            f"{auc:.3f} & {auc_rank}/{n_auc} & "
            f"{ari:.3f} & {ari_rank}/{n_ari} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main(output_path: Path | None = None) -> None:
    if output_path is None:
        output_path = OUTPUT_DIR / "lexical_baselines_table.tex"

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    alignment_path = OUTPUT_DIR / ALIGNMENT_CSV
    if not alignment_path.exists():
        msg = f"Missing alignment CSV: {alignment_path}"
        raise FileNotFoundError(msg)

    rows: list[tuple[str, str, float, int, float, int]] = []
    n_auc_per_dataset: list[int] = []
    n_ari_per_dataset: list[int] = []
    for dataset_key, display_name, cluster_filename in DATASETS:
        cluster_path = OUTPUT_DIR / cluster_filename
        if not cluster_path.exists():
            msg = f"Missing clustering CSV: {cluster_path}"
            raise FileNotFoundError(msg)

        alignment = _per_dataset_alignment_auc(alignment_path, dataset_key)
        clustering = _per_dataset_clustering_ari(cluster_path, dataset_key)
        rows.extend(_baseline_rows(display_name, alignment, clustering))
        n_auc_per_dataset.append(alignment.height)
        n_ari_per_dataset.append(clustering.height)

    # Ranks come from two different sets (alignment vs. clustering), so each
    # column carries its own denominator. Models present per dataset should match
    # across datasets; warn otherwise.
    n_auc = max(n_auc_per_dataset)
    n_ari = max(n_ari_per_dataset)
    for label, counts in (("AUC", n_auc_per_dataset), ("ARI", n_ari_per_dataset)):
        if len(set(counts)) > 1:
            logger.warning(
                f"{label} model count varies across datasets: {counts}; "
                f"using {max(counts)} in the rank denominator."
            )

    output_path.write_text(_build_tex(rows, n_auc=n_auc, n_ari=n_ari))
    logger.info(f"LaTeX table written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex path (default: output/lexical_baselines_table.tex).",
    )
    args = parser.parse_args()
    main(output_path=args.output)
