"""Report lexical-baseline gaps vs neural models.

Reads the alignment-AUC and clustering-ARI CSVs produced by
``neural_alignment_plots.py`` and ``clustering.py`` (which now include
``tfidf-char35`` and ``jaccard-binary`` via ``get_all_models()``) and
logs, per experiment:

- the baseline scores,
- the top-N neural scores,
- the gap (best-neural - baseline, baseline-rank among all models).

Output: ``output/lexical_baselines_report.md`` and a copy of the log.

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
DEFAULT_EXPERIMENTS = ("policy", "gov-ai")


def _load_scores(experiment: str) -> pl.DataFrame | None:
    """Join alignment AUC and clustering ARI on `model` for one experiment.

    Missing files are reported via ``logger.warning`` and skipped.
    """
    alignment_path = OUTPUT_DIR / f"human_alignment_bootstrapped_{experiment}.csv"
    cluster_path = OUTPUT_DIR / f"cluster_consistency_aggregated_{experiment}.csv"

    if not alignment_path.exists():
        logger.warning(f"Missing {alignment_path}; skipping {experiment}.")
        return None
    if not cluster_path.exists():
        logger.warning(f"Missing {cluster_path}; skipping {experiment}.")
        return None

    alignment = pl.read_csv(alignment_path)
    clustering = pl.read_csv(cluster_path).select("model", "adjusted_rand_index")
    return alignment.join(clustering, on="model", how="outer")


def _format_experiment_section(
    experiment: str, scores: pl.DataFrame, top_n: int
) -> str:
    """Markdown section: top-N neural + baseline rows + gap summary."""
    ranked = scores.sort("alignment_score", descending=True, nulls_last=True)
    ranked = ranked.with_columns(
        pl.col("alignment_score")
        .rank(method="ordinal", descending=True)
        .alias("alignment_rank"),
        pl.col("adjusted_rand_index")
        .rank(method="ordinal", descending=True)
        .alias("ari_rank"),
    )

    n_models = ranked.height
    top = ranked.head(top_n)
    baselines = ranked.filter(pl.col("model").is_in(LEXICAL_MODELS))

    best_align = ranked["alignment_score"].max()
    best_ari = ranked["adjusted_rand_index"].max()

    lines: list[str] = []
    lines.append(f"## Experiment: `{experiment}` ({n_models} models total)")
    lines.append("")
    lines.append("### Top-N neural")
    lines.append("")
    lines.append("| Rank | Model | Alignment AUC | ARI |")
    lines.append("|---:|:---|---:|---:|")
    for row in top.iter_rows(named=True):
        ari = row["adjusted_rand_index"]
        ari_str = "n/a" if ari is None else f"{ari:.3f}"
        align = row["alignment_score"]
        align_str = "n/a" if align is None else f"{align:.3f}"
        lines.append(
            f"| {row['alignment_rank']} | {row['model']} | {align_str} | {ari_str} |"
        )

    lines.append("")
    lines.append("### Lexical baselines")
    lines.append("")
    lines.append(
        "| Model | Alignment AUC (rank) | ARI (rank) | Δ AUC vs best | Δ ARI vs best |"
    )
    lines.append("|:---|---:|---:|---:|---:|")
    for row in baselines.iter_rows(named=True):
        align = row["alignment_score"]
        ari = row["adjusted_rand_index"]

        if align is None or best_align is None:
            align_cell = "n/a"
            d_align_cell = "n/a"
        else:
            align_cell = f"{align:.3f} ({row['alignment_rank']}/{n_models})"
            d_align_cell = f"{best_align - align:+.3f}"

        if ari is None or best_ari is None:
            ari_cell = "n/a"
            d_ari_cell = "n/a"
        else:
            ari_cell = f"{ari:.3f} ({row['ari_rank']}/{n_models})"
            d_ari_cell = f"{best_ari - ari:+.3f}"

        lines.append(
            f"| {row['model']} | {align_cell} | {ari_cell} | {d_align_cell} | {d_ari_cell} |"
        )

    lines.append("")
    return "\n".join(lines)


def main(
    experiments: list[str] = list(DEFAULT_EXPERIMENTS),
    top_n: int = 10,
    report_path: Path | None = None,
) -> None:
    if report_path is None:
        report_path = OUTPUT_DIR / "lexical_baselines_report.md"
    log_path = report_path.with_suffix(".log")

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", mode="w")

    sections: list[str] = ["# Lexical baseline report", ""]
    sections.append(f"Baselines evaluated: {', '.join(LEXICAL_MODELS)}")
    sections.append("")

    for experiment in experiments:
        scores = _load_scores(experiment)
        if scores is None:
            continue
        section = _format_experiment_section(experiment, scores, top_n=top_n)
        sections.append(section)
        logger.info(f"\n{section}")

    report_path.write_text("\n".join(sections))
    logger.info(f"Report written to {report_path}")
    logger.info(f"Log written to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        choices=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    main(experiments=args.experiments, top_n=args.top_n)
