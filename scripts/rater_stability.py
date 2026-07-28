"""Rater subsampling stability analysis.

For each subset of k raters (k in 2..N), recompute the model groundedness
AUC rankings and measure their Spearman correlation with the full-panel
ranking.  The resulting line-plot shows that the Human Grounding Exercise
yields stable model rankings even with very few expert raters.

Speed strategy
--------------
``combined_results`` already contains per-(rater, model, triplet) the
``embedding_correct`` boolean and ``pct_distance`` ratio.  The expensive
work — finding unanimous rater subsets at each threshold — reduces to
group-by operations on precomputed frames.  We precompute per-threshold
frames **once**, then derive each subset's consensus via fast filtering.
No bootstrap is needed: with 6 raters there are only 57 subsets total.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

import human_grounding.evaluate
from human_grounding.directories import OUTPUT_DIR, PLOT_DIR
from human_grounding.embed import get_all_models

if TYPE_CHECKING:
    from collections.abc import Sequence

# Columns that uniquely identify a triplet within a (dataset, seed, user_id)
_TRIPLET_KEY = ["dataset", "seed", "source_idx", "sorted_group"]


def _sorted_group_expr() -> pl.Expr:
    return (
        pl.concat_arr(pl.col("closer_idx", "farther_idx"))
        .arr.sort()
        .alias("sorted_group")
    )


# ---------------------------------------------------------------------------
# 1. Precompute per-threshold frames (done once)
# ---------------------------------------------------------------------------


def _precompute_threshold_frames(
    combined_results: pl.DataFrame,
    thresholds: Sequence[float],
) -> dict[float, pl.DataFrame]:
    """Filter by distance ratio and add sorted_group for each threshold.

    Returns a dict mapping threshold -> DataFrame with the original
    columns plus ``sorted_group``, filtered to ``pct_distance > t``.
    """
    base = combined_results.with_columns(_sorted_group_expr())
    frames: dict[float, pl.DataFrame] = {}
    for t in tqdm(thresholds, desc="Threshold frames", leave=False):
        df = base.filter(pl.col("pct_distance") > t)
        if df.height > 0:
            frames[t] = df
    return frames


# ---------------------------------------------------------------------------
# 2. For a rater subset at one threshold, find consensus triplets and
#    score models
# ---------------------------------------------------------------------------


def _score_models_for_subset_at_threshold(
    frame: pl.DataFrame,
    rater_subset: tuple[str, ...],
) -> pl.DataFrame | None:
    """Score all models using judgments from a rater subset.

    Simply filters to the given raters and computes mean alignment
    per (model, dataset) — no consensus required.
    """
    subset_data = frame.filter(pl.col("user_id").is_in(list(rater_subset)))

    if subset_data.height == 0:
        return None

    scored = subset_data.group_by("model", "dataset").agg(
        (2 * pl.col("embedding_correct").mean() - 1).alias("alignment_score"),
    )

    return scored if scored.height > 0 else None


# ---------------------------------------------------------------------------
# 3. AUC across thresholds -> per-model ranking
# ---------------------------------------------------------------------------


def _auc_ranking_for_subset(
    threshold_frames: dict[float, pl.DataFrame],
    rater_subset: tuple[str, ...],
) -> pl.DataFrame | None:
    """Compute groundedness AUC per model for a rater subset.

    Returns DataFrame with [model, auc] (mean across datasets).
    """
    per_threshold: list[pl.DataFrame] = []
    for t in sorted(threshold_frames):
        scored = _score_models_for_subset_at_threshold(
            threshold_frames[t],
            rater_subset,
        )
        if scored is not None:
            per_threshold.append(
                scored.with_columns(pl.lit(t).alias("threshold")),
            )

    if not per_threshold:
        return None

    curve = pl.concat(per_threshold)

    # Trapezoidal AUC per (model, dataset), then average across datasets
    auc_rows: list[dict[str, object]] = []
    for (model, dataset), grp in curve.group_by(["model", "dataset"]):
        arr = grp.sort("threshold")
        x = arr["threshold"].to_numpy()
        y = arr["alignment_score"].to_numpy()
        span = float(x[-1] - x[0])
        if span > 0 and len(x) >= 2:
            auc = float(np.trapezoid(y, x) / span)
        else:
            auc = float(y.mean())
        auc_rows.append({"model": model, "dataset": dataset, "auc": auc})

    return pl.DataFrame(auc_rows).group_by("model").agg(pl.col("auc").mean())


# ---------------------------------------------------------------------------
# 4. Main subsampling routine
# ---------------------------------------------------------------------------


def run_subsampling(
    combined_results: pl.DataFrame,
    thresholds: Sequence[float] | None = None,
    min_k: int = 2,
) -> pl.DataFrame:
    """Run exhaustive rater subsampling and return stability metrics.

    Parameters
    ----------
    combined_results:
        The full evaluation DataFrame with columns including
        ``user_id``, ``model``, ``pct_distance``, ``embedding_correct``,
        ``closer_idx``, ``farther_idx``, ``source_idx``, ``dataset``,
        ``seed``.
    thresholds:
        Distance-ratio values for AUC. Defaults to ~20 log-spaced
        values from 1.0 to 6.5.
    min_k:
        Minimum rater-subset size (default 2).

    Returns
    -------
    pl.DataFrame with columns [k, subset_idx, rho].
    """
    if thresholds is None:
        thresholds = np.logspace(
            np.log10(1.0),
            np.log10(6.5),
            num=20,
        ).tolist()

    all_raters = sorted(
        combined_results.select("user_id").unique().get_column("user_id").to_list()
    )
    n_raters = len(all_raters)
    logger.info(f"Total raters: {n_raters} ({all_raters})")

    # --- Expensive precomputation (done once) ---
    logger.info("Precomputing threshold frames …")
    threshold_frames = _precompute_threshold_frames(
        combined_results,
        thresholds,
    )
    logger.info(f"  {len(threshold_frames)} thresholds with data")

    # --- Full-panel reference ranking ---
    logger.info("Computing full-panel ranking …")
    full_ranking = _auc_ranking_for_subset(
        threshold_frames,
        tuple(all_raters),
    )
    if full_ranking is None:
        msg = "No consensus triplets for the full panel — check data."
        raise ValueError(msg)

    # --- Exhaustive enumeration ---
    rows: list[dict[str, int | float]] = []
    for k in range(min_k, n_raters + 1):
        subsets = list(combinations(all_raters, k))
        logger.info(f"k={k}: {len(subsets)} subsets")
        for subset_idx, rater_subset in enumerate(
            tqdm(subsets, desc=f"k={k}", leave=False),
        ):
            subset_ranking = _auc_ranking_for_subset(
                threshold_frames,
                rater_subset,
            )
            if subset_ranking is None:
                logger.warning(f"  k={k} subset {rater_subset}: no results")
                continue

            joined = full_ranking.join(
                subset_ranking,
                on="model",
                suffix="_subset",
            )
            if joined.height < 3:
                continue

            rho, _ = spearmanr(
                joined["auc"].to_numpy(),
                joined["auc_subset"].to_numpy(),
            )
            rows.append(
                {
                    "k": k,
                    "subset_idx": subset_idx,
                    "rho": float(rho),
                }
            )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_subsampling_stability(
    stability_df: pl.DataFrame,
    plot_dir: Path,
    font_scale: float = 1.0,
) -> Path:
    """Line-plot: x = k, y = Spearman rho, with min-max band."""
    sns.set_theme(style="whitegrid", font_scale=font_scale)

    summary = (
        stability_df.group_by("k")
        .agg(
            pl.col("rho").median().alias("median_rho"),
            pl.col("rho").min().alias("min_rho"),
            pl.col("rho").max().alias("max_rho"),
            pl.col("rho").mean().alias("mean_rho"),
            pl.col("rho").count().alias("n_subsets"),
        )
        .sort("k")
    )
    logger.info(f"Subsampling summary:\n{summary}")

    k_vals = summary["k"].to_numpy()
    median = summary["median_rho"].to_numpy()
    rho_min = summary["min_rho"].to_numpy()
    rho_max = summary["max_rho"].to_numpy()
    n_subsets = summary["n_subsets"].to_list()

    fig, ax = plt.subplots(figsize=(5, 3.5))

    ax.fill_between(
        k_vals,
        rho_min,
        rho_max,
        alpha=0.25,
        color="steelblue",
        label="Min\u2013Max",
    )
    ax.plot(
        k_vals,
        median,
        marker="o",
        color="steelblue",
        linewidth=2,
        label="Median",
    )

    for i, ki in enumerate(k_vals):
        ax.annotate(
            f"$n$={n_subsets[i]}",
            (ki, rho_max[i]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            color="grey",
        )

    ax.set_xlabel("Number of raters ($k$)")
    ax.set_ylabel(r"Spearman $\rho$ (vs. full panel)")
    ax.set_xticks(k_vals.tolist())

    y_lo = max(0.0, float(np.floor(rho_min.min() * 10) / 10 - 0.05))
    ax.set_ylim(y_lo, 1.05)
    ax.legend(frameon=False, loc="lower right")

    out_path = plot_dir / "rater_subsampling_stability.pdf"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    logger.info(
        f"Mean rho at k=2: {summary.filter(pl.col('k') == 2)['mean_rho'].item():.3f} (CI: [{summary.filter(pl.col('k') == 2)['min_rho'].item():.3f}, {summary.filter(pl.col('k') == 2)['max_rho'].item():.3f}])"
    )

    logger.info(f"Saved subsampling plot to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(font_scale: float = 3.0) -> None:
    full_dataset = pl.read_csv(OUTPUT_DIR / "valid_coordinates.csv")
    models = sorted(get_all_models())

    results: list[pl.DataFrame] = []
    for model in tqdm(models, desc="Embedding models"):
        if model.startswith("wicked"):
            logger.debug(f"Skipping unsupported model: {model}")
            continue
        comparisons = human_grounding.evaluate.evaluate_human_embedding_match(
            model=model,
            all_coordinates=full_dataset,
        )
        results.append(comparisons)
    combined_results = pl.concat(results).drop("cause", "source", "size")

    stability_df = run_subsampling(combined_results)

    out_csv = OUTPUT_DIR / "rater_subsampling_stability.csv"
    stability_df.write_csv(out_csv)
    logger.info(f"Saved raw stability data to {out_csv}")

    plot_subsampling_stability(
        stability_df,
        plot_dir=PLOT_DIR,
        font_scale=font_scale,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rater subsampling stability analysis",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Font scale for plots",
    )
    args = parser.parse_args()
    main(font_scale=args.scale)
