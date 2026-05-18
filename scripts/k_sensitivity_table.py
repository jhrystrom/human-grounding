"""K-sensitivity LaTeX table (reviewer Q5).

Runs the clustering pipeline twice -- once with human-derived K
(default) and once with silhouette-selected K (K in [2, 10]) -- and
writes a LaTeX table comparing:

- per-model ARI rank Spearman between the two K-selection schemes
  (1.00 by construction for the reference row),
- Spearman rho between human-grounding alignment score and clustering ARI,
- Spearman rho between MMTEB score and clustering ARI.

Spearman values are averaged across experiments (policy, gov-ai). The
upstream alignment/MMTEB CSVs must already exist on disk (produced by
``neural_alignment_plots.py``); this script reuses ``analyse_single``
and ``compute_spearman_table`` from ``clustering.py``.

Output: ``output/k_sensitivity_table.tex``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr

from human_grounding.directories import OUTPUT_DIR

# Sibling-script import (scripts/ is not a package)
sys.path.append(str(Path(__file__).parent))
from clustering import (
    K_SELECTION_CHOICES,
    analyse_single,
    compute_spearman_table,
)

DEFAULT_EXPERIMENTS = ("policy", "gov-ai")


def _run_mode(experiments: list[str], k_selection: str) -> pl.DataFrame:
    """Return the long combined_df with experiment column, for one K mode."""
    parts: list[pl.DataFrame] = []
    for exp in experiments:
        _, combined_df = analyse_single(exp, k_selection=k_selection)
        parts.append(combined_df.with_columns(pl.lit(exp).alias("experiment")))
    return pl.concat(parts)


def _ari_rank_rho(
    base: pl.DataFrame, other: pl.DataFrame, experiments: list[str]
) -> float:
    """Mean across experiments of Spearman(model->mean_ARI) between the two modes."""
    rhos: list[float] = []
    for exp in experiments:
        h = (
            base.filter(pl.col("experiment") == exp)
            .filter(pl.col("type") == "Model")
            .group_by("model")
            .agg(pl.mean("adjusted_rand_index").alias("ari"))
        )
        s = (
            other.filter(pl.col("experiment") == exp)
            .filter(pl.col("type") == "Model")
            .group_by("model")
            .agg(pl.mean("adjusted_rand_index").alias("ari"))
        )
        joined = h.join(s, on="model", suffix="_other")
        if joined.height < 3:
            continue
        rho = spearmanr(
            joined["ari"].to_numpy(), joined["ari_other"].to_numpy()
        ).statistic
        if rho is None or not np.isfinite(rho):
            continue
        rhos.append(float(rho))
    return float(np.mean(rhos)) if rhos else float("nan")


def _mean_rho_by_source(table: pl.DataFrame, source: str) -> float:
    subset = table.filter(pl.col("source") == source)
    if subset.height == 0:
        return float("nan")
    return float(subset["spearman"].mean())


def _build_tex(
    rows: list[tuple[str, float, float, float]],
) -> str:
    body = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of downstream clustering results to the choice of $K$.}",
        r"\label{tab:k-sensitivity}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Clustering setup} & \textbf{ARI rank $\rho$} & "
        r"\textbf{Grounding--ARI $\rho$} & \textbf{MMTEB--ARI $\rho$} \\",
        r"\midrule",
    ]
    for label, ari_rho, grounding_rho, mmteb_rho in rows:
        body.append(
            f"{label} & {ari_rho:.2f} & {grounding_rho:.2f} & {mmteb_rho:.2f} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(body)


def main(
    experiments: list[str] = list(DEFAULT_EXPERIMENTS),
    n_bootstrap: int = 2000,
    seed: int = 0,
    output_path: Path | None = None,
) -> None:
    if output_path is None:
        output_path = OUTPUT_DIR / "k_sensitivity_table.tex"

    mode_combined: dict[str, pl.DataFrame] = {}
    for mode in K_SELECTION_CHOICES:
        logger.info(f"Running clustering with k_selection={mode}...")
        mode_combined[mode] = _run_mode(experiments, mode)

    rho_human_vs_human = 1.0
    rho_sil_vs_human = _ari_rank_rho(
        mode_combined["human"], mode_combined["silhouette"], experiments
    )

    sp_human = compute_spearman_table(
        mode_combined["human"], experiments, n_bootstrap=n_bootstrap, seed=seed
    )
    sp_sil = compute_spearman_table(
        mode_combined["silhouette"], experiments, n_bootstrap=n_bootstrap, seed=seed
    )

    rows = [
        (
            "Human-derived $K$",
            rho_human_vs_human,
            _mean_rho_by_source(sp_human, "OurExercise"),
            _mean_rho_by_source(sp_human, "MMTEB"),
        ),
        (
            "Silhouette-selected $K$",
            rho_sil_vs_human,
            _mean_rho_by_source(sp_sil, "OurExercise"),
            _mean_rho_by_source(sp_sil, "MMTEB"),
        ),
    ]

    for label, a, b, c in rows:
        logger.info(
            f"{label}: ARI-rank rho={a:.3f}  Grounding-ARI rho={b:.3f}  "
            f"MMTEB-ARI rho={c:.3f}"
        )

    output_path.write_text(_build_tex(rows))
    logger.info(f"LaTeX table written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(DEFAULT_EXPERIMENTS),
        choices=list(DEFAULT_EXPERIMENTS),
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for Spearman CI (passed through to compute_spearman_table).",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        experiments=args.experiments,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
