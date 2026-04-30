"""Compute alignment-score AUC over a range of distance-ratio thresholds.

Drop this module next to the existing script and call
``compute_threshold_auc`` from ``main`` (or standalone).

Speed strategy
--------------
The expensive work is filtering + agreement + demographic joins — these are
deterministic (no randomness).  We precompute one demographic-joined frame
per threshold, *then* draw N bootstrap replicates over the precomputed
frames.  This avoids repeating the join work N x T times.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

HUMAN_MODEL_NAME = "Human"

# ---------------------------------------------------------------------------
# 3. Precompute per-threshold demographic frames (deterministic, no RNG)
# ---------------------------------------------------------------------------


def precompute_demographic_frames(
    combined_results: pl.DataFrame,
    welfare_demographics: pl.DataFrame,
    rai_demographics: pl.DataFrame,
    thresholds: Sequence[float],
) -> dict[float, pl.DataFrame]:
    """Filter + agreement + demographic join for every threshold.

    This is the expensive deterministic part — done once, then reused
    across all bootstrap iterations.
    """
    frames: dict[float, pl.DataFrame] = {}
    for t in tqdm(thresholds, desc="Precomputing thresholds"):
        filtered = filter_by_distance_threshold(combined_results, t)
        if filtered.height == 0:
            logger.warning(
                f"No comparisons survive threshold {t:.2f}",
            )
            continue
        frames[t] = join_demographics(
            filtered,
            welfare_demographics,
            rai_demographics,
        )
    return frames


# ---------------------------------------------------------------------------
# 4. Bootstrap a single replicate across all thresholds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 5. Trapezoidal AUC (numpy — applied per group)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 6. Compute per-(model, dataset, demographics, iteration) AUC from curve
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Load human baseline from alpha CSV
# ---------------------------------------------------------------------------


def load_human_auc(
    alpha_csv_path: Path,
    thresholds: Sequence[float] | None = None,
) -> pl.DataFrame:
    """Compute per-(demographic, iteration) AUC from the alpha CSV.

    The alpha CSV has columns:
    ``group_type, group_name, reliability_type, iteration_id, d,
    krippendorf``.

    - ``group_type == "dataset"``  -> dataset-level aggregates
      (``group_name`` is ``"welfare"`` / ``"rai"``).
    - ``group_type == "demographic"`` -> per-demographic values
      (``group_name`` is the demographic label, e.g. ``"High"``).
      These rows are NOT split by dataset in the CSV; numeric
      demographics (``"1"``, ``"2"``, …) belong to welfare and
      ordinal ones (``"Low"``, ``"Medium"``, ``"High"``) to rai.

    We filter to ``reliability_type == "between"``, treat ``d`` as the
    threshold axis and ``krippendorf`` as the alignment score, and
    compute trapezoidal AUC per ``(group_name, iteration_id)``.

    Returns a frame with columns
    ``[model, dataset, demographics, iteration, auc]``
    matching the embedding-model AUC frame so they can be concatenated.
    """
    raw = pl.read_csv(alpha_csv_path)

    human = raw.filter(
        (pl.col("reliability_type") == "between")
        & (pl.col("group_type") == "demographic"),
    )

    if thresholds is not None:
        threshold_arr = np.array(thresholds)
        human = human.with_columns(
            pl.col("d")
            .map_elements(
                lambda d: float(
                    threshold_arr[np.argmin(np.abs(threshold_arr - d))],
                ),
                return_dtype=pl.Float64,
            )
            .alias("d_snapped"),
        )
        d_col = "d_snapped"
    else:
        d_col = "d"

    # Demographic -> dataset mapping: numeric groups belong to welfare,
    # Low/Medium/High belong to rai.
    def _dataset_for_demographic(demo: str) -> str:
        if demo.isdigit():
            return "welfare"
        return "rai"

    auc_rows: list[dict[str, object]] = []
    group_keys = ["group_name", "iteration_id"]
    for (demographics, iteration_id), group in human.group_by(
        group_keys,
    ):
        if group.height < 2:
            continue
        arr = group.sort(d_col)
        auc_val = _auc_trapz_np(
            arr[d_col].to_numpy(),
            arr["krippendorf"].to_numpy(),
        )
        auc_rows.append(
            {
                "model": HUMAN_MODEL_NAME,
                "dataset": _dataset_for_demographic(str(demographics)),
                "demographics": str(demographics),
                "iteration": iteration_id,
                "auc": auc_val,
            },
        )

    return pl.DataFrame(auc_rows)


# ---------------------------------------------------------------------------
# 9. Summarise: best / worst / mean group per (model, dataset)
# ---------------------------------------------------------------------------


def summarise_best_worst_mean(
    group_auc: pl.DataFrame,
    ci: float = 95.0,
) -> pl.DataFrame:
    """From per-group AUC bootstraps, derive best/worst/mean summaries.

    For each ``(model, dataset, iteration)`` we compute:
    - **Best Group**: max AUC across demographics
    - **Worst Group**: min AUC across demographics
    - **Mean**: mean AUC across demographics

    Then aggregate over iterations to get mean + CI.

    Returns columns
    ``[model, dataset, statistic, auc_mean, ci_lo, ci_hi]``.
    """
    alpha = (100.0 - ci) / 100.0

    has_demo = "demographics" in group_auc.columns
    if has_demo:
        agg_exprs = [
            pl.col("auc").max().alias("Best Group"),
            pl.col("auc").min().alias("Worst Group"),
            pl.col("auc").mean().alias("Mean"),
        ]
    else:
        agg_exprs = [pl.col("auc").mean().alias("Mean")]

    per_iter = (
        group_auc.group_by("model", "dataset", "iteration")
        .agg(*agg_exprs)
        .unpivot(
            index=["model", "dataset", "iteration"],
            variable_name="statistic",
            value_name="auc",
        )
    )

    return (
        per_iter.group_by("model", "dataset", "statistic")
        .agg(
            pl.col("auc").mean().alias("auc_mean"),
            pl.col("auc").quantile(alpha / 2, interpolation="linear").alias("ci_lo"),
            pl.col("auc")
            .quantile(1 - alpha / 2, interpolation="linear")
            .alias("ci_hi"),
        )
        .sort("dataset", "auc_mean", descending=[False, True])
    )


# ---------------------------------------------------------------------------
# 10. Plot: faceted bar chart (catplot) with best/worst/mean per dataset
# ---------------------------------------------------------------------------

## New


def filter_by_distance_threshold(
    combined_results: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    """Apply distance-ratio filter and inter-rater agreement dedup."""
    create_sorted_groups = (
        pl.concat_arr(pl.col("closer_idx", "farther_idx"))
        .arr.sort()
        .alias("sorted_groups")
    )

    with_sorted_groups = (
        combined_results.with_columns(create_sorted_groups)
        .filter(pl.col("pct_distance") > threshold)
        .drop("pct_distance")
        .unique()
        .sort("model", "dataset", "seed", "user_id", "source_idx")
        .with_columns(
            pl.int_range(pl.len()).over("model").alias("new_index"),
        )
    )

    agreement_indices = (
        with_sorted_groups.drop("model")
        .unique()
        .group_by("source_idx", "sorted_groups", "dataset", "seed")
        .agg(
            pl.col("closer_idx").n_unique() == 1,
            pl.col("new_index").first(),
        )
        .filter(pl.col("closer_idx"))
        .select("new_index")
    )

    return with_sorted_groups.join(
        agreement_indices,
        on="new_index",
        how="inner",
    )


def join_demographics(
    filtered_comparisons: pl.DataFrame,
    welfare_demographics: pl.DataFrame | None = None,
    rai_demographics: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach demographic labels while preserving raw distance columns."""
    if welfare_demographics is None and rai_demographics is None:
        logger.warning("No demographics provided, skipping join.")
        return filtered_comparisons

    if welfare_demographics is None or rai_demographics is None:
        msg = "join_demographics requires both welfare and rai demographics, or neither"
        raise ValueError(msg)

    dist_cols = [
        "human_dist_close",
        "human_dist_far",
        "model_dist_close",
        "model_dist_far",
    ]
    id_cols = ["user_id", "source_idx", "closer_idx", "farther_idx"]

    available_dists = [c for c in dist_cols if c in filtered_comparisons.columns]
    available_ids = [c for c in id_cols if c in filtered_comparisons.columns]

    keep = [
        "model",
        "dataset",
        "demographics",
        "embedding_correct",
        *available_ids,
        *available_dists,
    ]

    def _process(
        df: pl.DataFrame, demo_df: pl.DataFrame, idx_col: str, suffix: str = ""
    ) -> pl.DataFrame:
        return df.join(demo_df, left_on=idx_col, right_on="cause_id", suffix=suffix)

    welfare_filtered = filtered_comparisons.filter(pl.col("dataset") == "welfare").drop(
        "demographic"
    )
    welfare_filtered = _process(welfare_filtered, welfare_demographics, "source_idx")
    welfare_filtered = _process(
        welfare_filtered, welfare_demographics, "closer_idx", "_close"
    )
    welfare_filtered = _process(
        welfare_filtered, welfare_demographics, "farther_idx", "_far"
    )

    rai_filtered = filtered_comparisons.filter(pl.col("dataset") == "rai").drop(
        "demographic"
    )
    rai_filtered = _process(rai_filtered, rai_demographics, "source_idx")
    rai_filtered = _process(rai_filtered, rai_demographics, "closer_idx", "_close")
    rai_filtered = _process(rai_filtered, rai_demographics, "farther_idx", "_far")

    full_joined = (
        pl.concat(
            [
                welfare_filtered.with_columns(
                    pl.concat_list(cs.starts_with("demographic")).alias("demographics")
                ),
                rai_filtered.with_columns(
                    pl.concat_list(cs.starts_with("demographic")).alias("demographics")
                ),
            ]
        )
        .explode("demographics")
        .select(keep)
    )

    return full_joined


# ---------------------------------------------------------------------------
# 2. Alignment Logic (Binary Triplet vs Spearman)
# ---------------------------------------------------------------------------


def _group_keys(df: pl.DataFrame) -> list[str]:
    base = ["model", "dataset"]
    if "demographics" in df.columns:
        base.append("demographics")
    return base


def _calculate_spearman_score(df: pl.DataFrame) -> pl.DataFrame:
    """Pools triplets into pairs and calculates monotonic correlation."""
    keys = _group_keys(df)
    close_pairs = df.select(
        *keys,
        pl.col("human_dist_close").alias("h"),
        pl.col("model_dist_close").alias("m"),
    )
    far_pairs = df.select(
        *keys,
        pl.col("human_dist_far").alias("h"),
        pl.col("model_dist_far").alias("m"),
    )

    return (
        pl.concat([close_pairs, far_pairs])
        .group_by(keys)
        .agg(pl.corr("h", "m", method="spearman").alias("alignment_score"))
    )


def _bootstrap_one_replicate(
    demo_frames: dict[float, pl.DataFrame], iteration: int, metric: str = "binary"
) -> pl.DataFrame:
    rows = []
    for threshold, df in demo_frames.items():
        sample = df.sample(fraction=1.0, with_replacement=True)

        if metric == "spearman":
            agg = _calculate_spearman_score(sample)
        else:
            agg = sample.group_by(_group_keys(sample)).agg(
                (2 * pl.col("embedding_correct").mean() - 1).alias("alignment_score")
            )

        rows.append(
            agg.with_columns(
                pl.lit(threshold).alias("threshold"),
                pl.lit(iteration).alias("iteration"),
            )
        )
    return pl.concat(rows)


# ---------------------------------------------------------------------------
# 3. Main Entry Points
# ---------------------------------------------------------------------------


def compute_threshold_auc(
    combined_results: pl.DataFrame,
    welfare_demographics: pl.DataFrame | None = None,
    rai_demographics: pl.DataFrame | None = None,
    thresholds: Sequence[float] | None = None,
    n_bootstrap: int = 100,
    metric: str = "binary",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sweep thresholds and return per-group AUC or point-estimate Spearman."""

    # For Spearman at d=1, we only need the threshold 1.0
    if metric == "spearman" and thresholds is None:
        thresholds = [1.0]
    elif thresholds is None:
        thresholds = np.logspace(np.log10(1.0), np.log10(6.5), num=30).tolist()

    demo_frames = {}
    for t in tqdm(thresholds, desc="Precomputing"):
        filtered = filter_by_distance_threshold(combined_results, t)
        if filtered.height > 0:
            demo_frames[t] = join_demographics(
                filtered, welfare_demographics, rai_demographics
            )

    replicate_frames = [
        _bootstrap_one_replicate(demo_frames, iteration=i, metric=metric)
        for i in tqdm(range(n_bootstrap), desc="Bootstrap replicates")
    ]
    curve = pl.concat(replicate_frames)

    # If it's Spearman (point estimate), AUC is just the value itself
    if metric == "spearman":
        group_res = curve.rename({"alignment_score": "auc"}).drop("threshold")
    else:
        group_res = _curve_to_group_auc(curve)

    return group_res, curve


# ---------------------------------------------------------------------------
# 4. Utilities and Plotting
# ---------------------------------------------------------------------------


def _auc_trapz_np(thresholds: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(thresholds)
    x, y = thresholds[order], scores[order]
    span = float(x[-1] - x[0])
    return float(np.trapezoid(y, x) / span) if span > 0 else float(y.mean())


def _curve_to_group_auc(curve: pl.DataFrame) -> pl.DataFrame:
    key_cols = [*_group_keys(curve), "iteration"]
    auc_rows = []
    for keys, group in curve.group_by(key_cols):
        auc_val = _auc_trapz_np(
            group["threshold"].to_numpy(), group["alignment_score"].to_numpy()
        )
        auc_rows.append({**dict(zip(key_cols, keys)), "auc": auc_val})
    return pl.DataFrame(auc_rows)


def plot_auc_bar(
    group_auc: pl.DataFrame,
    plot_dir: Path,
    *,
    pretty_names: Mapping[str, str] | None = None,
    dataset_name_map: Mapping[str, str] | None = None,
    top_n: int = 10,
    font_scale: float = 1.35,
    ci: float = 95.0,
    use_english: bool = False,  # noqa: ARG001
    file_type: str = "pdf",
    x_label: str = "Alignment Score",
    filename_prefix: str = "alignment_results",
) -> Path:
    """Faceted horizontal bar chart of alignment metrics."""
    if dataset_name_map is None:
        dataset_name_map = {"rai": "Responsible AI", "welfare": "Welfare"}

    has_demo = "demographics" in group_auc.columns
    if has_demo:
        agg_exprs = [
            pl.col("auc").max().alias("Best Group"),
            pl.col("auc").min().alias("Worst Group"),
            pl.col("auc").mean().alias("Mean"),
        ]
    else:
        agg_exprs = [pl.col("auc").mean().alias("Mean")]

    per_iter = (
        group_auc.group_by("model", "dataset", "iteration")
        .agg(*agg_exprs)
        .unpivot(
            index=["model", "dataset", "iteration"],
            variable_name="statistic",
            value_name="auc",
        )
    )

    model_order = (
        per_iter.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc").mean())
        .sort("auc", descending=True)
        .get_column("model")
        .to_list()
    )

    plot_data = per_iter.with_columns(
        pl.col("model").replace(pretty_names or {}),
        pl.col("dataset").replace(dataset_name_map),
    )

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    g = sns.catplot(
        data=plot_data.to_pandas(),
        x="auc",
        y="model",
        hue="statistic",
        col="dataset",
        order=[(pretty_names or {}).get(m, m) for m in model_order if m in model_order][
            : top_n + 1
        ],
        kind="bar",
        palette="coolwarm",
        height=10,
        aspect=0.9,
        errorbar=("ci", ci),
    )

    g.set_axis_labels(x_label, "")
    out_path = plot_dir / f"{filename_prefix}.{file_type}"
    plt.savefig(out_path, bbox_inches="tight")
    return out_path


def compute_human_human_spearman(
    combined_results: pl.DataFrame,
    welfare_demographics: pl.DataFrame | None = None,
    rai_demographics: pl.DataFrame | None = None,
    thresholds: list[float] | None = None,
    n_bootstrap: int = 100,
) -> pl.DataFrame:
    """Compute human-human Spearman alignment per demographic group.

    Mirrors the structure of ``compute_threshold_auc``: demographics are
    joined *before* bootstrapping so that correlations are computed
    separately for each ``(dataset, demographics)`` group rather than
    collapsing across all demographics first.

    Returns columns ``[model, dataset, demographics, iteration, auc]``.
    """
    if thresholds is None:
        thresholds = [1.0]

    # Precompute filtered + demographic-joined frames (same as compute_threshold_auc)
    demo_frames: dict[float, pl.DataFrame] = {}
    for t in thresholds:
        filtered = filter_by_distance_threshold(combined_results, t)
        if filtered.height > 0 and (
            welfare_demographics is not None or rai_demographics is not None
        ):
            demo_frames[t] = join_demographics(
                filtered, welfare_demographics, rai_demographics
            )
        else:
            demo_frames[t] = filtered

    if not demo_frames:
        return pl.DataFrame(
            schema={
                "model": pl.Utf8,
                "dataset": pl.Utf8,
                "demographics": pl.Utf8,
                "iteration": pl.Int64,
                "auc": pl.Float64,
            }
        )

    rows = []
    for i in tqdm(range(n_bootstrap), desc="Human bootstrap"):
        for _, demo_df in demo_frames.items():
            sample = demo_df.sample(fraction=1.0, with_replacement=True)

            has_demo = "demographics" in sample.columns
            group_keys = ["dataset", "demographics"] if has_demo else ["dataset"]

            for keys, group in sample.group_by(group_keys):
                dataset = keys[0]
                demographics = keys[1] if has_demo else None

                user_groups = list(group.group_by("user_id"))
                if len(user_groups) < 2:
                    continue

                corrs = []
                for (_u1, g1), (_u2, g2) in itertools.combinations(user_groups, 2):
                    joined = g1.join(
                        g2,
                        on=["source_idx", "closer_idx", "farther_idx"],
                        suffix="_2",
                        how="inner",
                    )
                    if joined.height < 10:
                        continue

                    h1 = joined["human_dist_far"] / joined["human_dist_close"]
                    h2 = joined["human_dist_far_2"] / joined["human_dist_close_2"]
                    corr = spearmanr(h1, h2).correlation
                    if corr is not None:
                        corrs.append(corr)

                if not corrs:
                    continue

                row = {
                    "model": "Human",
                    "dataset": str(dataset),
                    "iteration": i,
                    "auc": float(np.mean(corrs)),
                }
                if has_demo:
                    row["demographics"] = str(demographics)
                rows.append(row)

    return pl.DataFrame(rows)
