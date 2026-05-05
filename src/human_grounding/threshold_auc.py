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
        & (pl.col("group_type") == "demographic")
        & (pl.col("group_name") != "unknown"),
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
    KNOWN_DEMOGRAPHIC_DATASETS = {"welfare", "rai"}

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

    # For datasets that have no meaningful demographic breakdown (e.g. gov-ai),
    # the demographic rows will map to a known dataset incorrectly or not at all.
    # Fall back to the dataset-level rows for any dataset not already covered.
    covered_datasets = {row["dataset"] for row in auc_rows}
    dataset_rows = raw.filter(
        (pl.col("reliability_type") == "between")
        & (pl.col("group_type") == "dataset")
        & (~pl.col("group_name").is_in(list(covered_datasets | KNOWN_DEMOGRAPHIC_DATASETS))),
    )
    if thresholds is not None:
        threshold_arr = np.array(thresholds)
        dataset_rows = dataset_rows.with_columns(
            pl.col("d")
            .map_elements(
                lambda d: float(threshold_arr[np.argmin(np.abs(threshold_arr - d))]),
                return_dtype=pl.Float64,
            )
            .alias("d_snapped"),
        )
        d_col_ds = "d_snapped"
    else:
        d_col_ds = "d"

    for (dataset_name, iteration_id), group in dataset_rows.group_by(["group_name", "iteration_id"]):
        if group.height < 2:
            continue
        arr = group.sort(d_col_ds)
        auc_val = _auc_trapz_np(arr[d_col_ds].to_numpy(), arr["krippendorf"].to_numpy())
        auc_rows.append(
            {
                "model": HUMAN_MODEL_NAME,
                "dataset": str(dataset_name),
                "demographics": "Overall",
                "iteration": iteration_id,
                "auc": auc_val,
            }
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

    # Datasets with no explicit demographics (e.g. gov-ai): use "Overall" as a
    # single synthetic group so their rows are not silently dropped.
    other_filtered = (
        filtered_comparisons.filter(~pl.col("dataset").is_in(["welfare", "rai"]))
        .with_columns(
            pl.col("demographic").fill_null("Overall").alias("demographics")
        )
        .select([c for c in keep if c in filtered_comparisons.columns or c == "demographics"])
    )

    to_concat = [
        welfare_filtered.with_columns(
            pl.concat_list(cs.starts_with("demographic")).alias("demographics")
        ).explode("demographics").select(keep),
        rai_filtered.with_columns(
            pl.concat_list(cs.starts_with("demographic")).alias("demographics")
        ).explode("demographics").select(keep),
    ]
    if other_filtered.height > 0:
        to_concat.append(other_filtered.select(keep))

    return pl.concat(to_concat)


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


_STAT_MARKERS: dict[str, tuple[str, int]] = {
    "Mean": ("D", 180),
    "Worst group": ("o", 130),
    "Best group": ("^", 130),
}


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
    x_label: str = "Alignment AUC (normalised)",
    filename_prefix: str = "alignment_results",
    height: float = 9.0,
    aspect: float = 2.0,
) -> Path:
    """Horizontal dot-plot of alignment AUC, coloured by dataset, shaped by statistic.

    Datasets with no real demographic breakdown (e.g. gov-ai, which only has
    an 'Overall' group) show only the Mean marker.
    """
    import matplotlib.patches as mpatches

    if dataset_name_map is None:
        dataset_name_map = {"rai": "Responsible AI", "welfare": "Welfare"}

    pretty = pretty_names or {}

    # Datasets that have real demographic sub-groups (not just "Overall" / null)
    has_demo_col = "demographics" in group_auc.columns
    if has_demo_col:
        demo_datasets: set[str] = set(
            group_auc.filter(
                pl.col("demographics").is_not_null()
                & (pl.col("demographics") != "Overall")
            )["dataset"].unique().to_list()
        )
    else:
        demo_datasets = set()

    # Per-iteration aggregation: Best/Worst/Mean for demographic datasets, Mean-only otherwise.
    # Unpivot each part before concat so all share the same schema.
    parts = []
    for (dataset,), ds_group in group_auc.group_by(["dataset"]):
        if dataset in demo_datasets:
            agg = ds_group.group_by("model", "dataset", "iteration").agg(
                pl.col("auc").max().alias("Best group"),
                pl.col("auc").min().alias("Worst group"),
                pl.col("auc").mean().alias("Mean"),
            )
        else:
            agg = ds_group.group_by("model", "dataset", "iteration").agg(
                pl.col("auc").mean().alias("Mean"),
            )
        parts.append(
            agg.unpivot(
                index=["model", "dataset", "iteration"],
                variable_name="statistic",
                value_name="auc",
            )
        )

    per_iter = pl.concat(parts)

    # Model order: descending mean AUC across all datasets
    model_order = (
        per_iter.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc").mean())
        .sort("auc", descending=True)
        .get_column("model")
        .to_list()
    )
    keep_models = model_order[: top_n + 1]

    # Summarise across bootstrap iterations: mean + CI
    alpha = (100.0 - ci) / 200.0
    summary = (
        per_iter.filter(pl.col("model").is_in(keep_models))
        .group_by("model", "dataset", "statistic")
        .agg(
            pl.col("auc").mean().alias("mean"),
            pl.col("auc").quantile(alpha).alias("lo"),
            pl.col("auc").quantile(1.0 - alpha).alias("hi"),
        )
        .with_columns(
            pl.col("model").replace(pretty),
            pl.col("dataset").replace(dataset_name_map),
        )
        .to_pandas()
    )

    pretty_order = [pretty.get(m, m) for m in keep_models]
    datasets_in_data = sorted(summary["dataset"].unique())

    palette = sns.color_palette("Set2", len(datasets_in_data))
    color_map = dict(zip(datasets_in_data, palette))

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    fig, ax = plt.subplots(figsize=(height * aspect, height))

    # y=0 → best model; axis is inverted so y=0 renders at the top
    y_pos = {m: i for i, m in enumerate(pretty_order)}

    for ds in datasets_in_data:
        color = color_map[ds]
        for stat, (marker, size) in _STAT_MARKERS.items():
            sub = summary[(summary["dataset"] == ds) & (summary["statistic"] == stat)]
            if sub.empty:
                continue
            for _, row in sub.iterrows():
                if row["model"] not in y_pos:
                    continue
                y = y_pos[row["model"]]
                ax.plot([row["lo"], row["hi"]], [y, y], color=color, lw=1.5, alpha=0.6, zorder=2)
                ax.scatter(row["mean"], y, color=color, marker=marker, s=size, zorder=3)

    # Alternating row shading
    for i in range(len(pretty_order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="grey", alpha=0.06, zorder=0)

    ax.set_yticks(range(len(pretty_order)))
    ax.set_yticklabels(pretty_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")

    # Legend: one patch per dataset, one line-marker per statistic — two rows below x label.
    # fig.legend with constrained_layout handles spacing automatically.
    legend_handles: list = [mpatches.Patch(color=color_map[ds], label=ds) for ds in datasets_in_data]
    for stat, (marker, _) in _STAT_MARKERS.items():
        legend_handles.append(
            plt.Line2D([0], [0], marker=marker, color="gray", linestyle="None",
                       markersize=20, label=stat)
        )
    ncol = int(np.ceil(len(legend_handles) / 2))
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=(0, 0.12, 1, 1))
    out_path = plot_dir / f"{filename_prefix}.{file_type}"
    plt.savefig(out_path, bbox_inches="tight")
    plt.clf()
    return out_path


DIFFICULTY_LABELS = {
    "hard": "Hard ($d$: bottom 20%)",
    "easy": "Easy ($d$: top 20%)",
}


def filter_by_pct_distance_quantile(
    combined_results: pl.DataFrame,
    q_lo: float,
    q_hi: float,
) -> pl.DataFrame:
    """Like ``filter_by_distance_threshold`` but selects rows whose
    ``pct_distance`` falls in the global ``[q_lo, q_hi]`` quantile range."""
    lo_val = combined_results["pct_distance"].quantile(q_lo)
    hi_val = combined_results["pct_distance"].quantile(q_hi)
    if lo_val is None or hi_val is None:
        msg = "Could not compute pct_distance quantiles (empty input?)"
        raise ValueError(msg)
    lo = float(lo_val)
    hi = float(hi_val)

    create_sorted_groups = (
        pl.concat_arr(pl.col("closer_idx", "farther_idx"))
        .arr.sort()
        .alias("sorted_groups")
    )

    with_sorted_groups = (
        combined_results.with_columns(create_sorted_groups)
        .filter((pl.col("pct_distance") >= lo) & (pl.col("pct_distance") <= hi))
        .drop("pct_distance")
        .unique()
        .sort("model", "dataset", "seed", "user_id", "source_idx")
        .with_columns(pl.int_range(pl.len()).over("model").alias("new_index"))
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

    return with_sorted_groups.join(agreement_indices, on="new_index", how="inner")


def compute_difficulty_split_alignment(
    combined_results: pl.DataFrame,
    welfare_demographics: pl.DataFrame | None = None,
    rai_demographics: pl.DataFrame | None = None,
    n_bootstrap: int = 50,
    quantile: float = 0.2,
) -> pl.DataFrame:
    """Compute per-(model, dataset, demographics, difficulty) alignment scores.

    Splits triplets into the ``quantile`` lowest (hard) and highest (easy)
    ``pct_distance`` cases, then bootstraps a binary alignment score
    (2*accuracy - 1) per group. Includes a ``Human`` row computed as
    pairwise inter-rater agreement over the same splits.
    """
    splits = {
        DIFFICULTY_LABELS["hard"]: (0.0, quantile),
        DIFFICULTY_LABELS["easy"]: (1.0 - quantile, 1.0),
    }

    demo_frames: dict[str, pl.DataFrame] = {}
    for label, (lo, hi) in splits.items():
        filtered = filter_by_pct_distance_quantile(combined_results, lo, hi)
        if filtered.height == 0:
            logger.warning(f"No comparisons in split {label}")
            continue
        demo_frames[label] = join_demographics(
            filtered, welfare_demographics, rai_demographics
        )

    rows: list[pl.DataFrame] = []
    for i in tqdm(range(n_bootstrap), desc="Difficulty bootstrap"):
        for label, df in demo_frames.items():
            sample = df.sample(fraction=1.0, with_replacement=True)
            agg = (
                sample.group_by(_group_keys(sample))
                .agg(
                    (2 * pl.col("embedding_correct").mean() - 1).alias("auc"),
                )
                .with_columns(
                    pl.lit(label).alias("difficulty"),
                    pl.lit(i).alias("iteration"),
                )
            )
            rows.append(agg)

    model_part = pl.concat(rows) if rows else pl.DataFrame()

    human_rows = _compute_human_human_binary_split(demo_frames, n_bootstrap)
    if human_rows is not None and human_rows.height > 0:
        return pl.concat(
            [model_part, human_rows.select(model_part.columns)],
            how="vertical_relaxed",
        )
    return model_part


def _compute_human_human_binary_split(
    demo_frames: dict[str, pl.DataFrame],
    n_bootstrap: int,
) -> pl.DataFrame | None:
    """Inter-rater binary agreement on the same triplets, per split."""
    if not demo_frames:
        return None

    # Pick one canonical model so each user-pair is counted once per triplet.
    any_df = next(iter(demo_frames.values()))
    if "user_id" not in any_df.columns:
        return None
    canonical_model = any_df["model"].unique().sort()[0]

    rows: list[dict] = []
    for i in tqdm(range(n_bootstrap), desc="Human inter-rater"):
        for label, df in demo_frames.items():
            sub = df.filter(pl.col("model") == canonical_model)
            sample = sub.sample(fraction=1.0, with_replacement=True)

            has_demo = "demographics" in sample.columns
            keys = ["dataset", "demographics"] if has_demo else ["dataset"]

            for key_vals, group in sample.group_by(keys):
                user_groups = list(group.group_by("user_id"))
                if len(user_groups) < 2:
                    continue

                # Make the (closer, farther) pair unordered for joining
                def _unordered(g: pl.DataFrame) -> pl.DataFrame:
                    return g.with_columns(
                        pl.min_horizontal("closer_idx", "farther_idx").alias("a"),
                        pl.max_horizontal("closer_idx", "farther_idx").alias("b"),
                    )

                agrees: list[float] = []
                for (_, g1), (_, g2) in itertools.combinations(user_groups, 2):
                    j = _unordered(g1).join(
                        _unordered(g2),
                        on=["source_idx", "a", "b"],
                        suffix="_2",
                        how="inner",
                    )
                    if j.height == 0:
                        continue
                    mean_agree = (j["closer_idx"] == j["closer_idx_2"]).mean()
                    if mean_agree is None:
                        continue
                    agrees.append(float(mean_agree))
                if not agrees:
                    continue

                row = {
                    "model": HUMAN_MODEL_NAME,
                    "dataset": str(key_vals[0]),
                    "auc": 2 * float(np.mean(agrees)) - 1,  # type: ignore[arg-type]
                    "difficulty": label,
                    "iteration": i,
                }
                if has_demo:
                    row["demographics"] = str(key_vals[1])
                rows.append(row)

    return pl.DataFrame(rows) if rows else None


def summarise_difficulty_split(group_auc: pl.DataFrame) -> pl.DataFrame:
    """Aggregate to (model, dataset, difficulty, statistic) -> mean + CI."""
    has_demo = "demographics" in group_auc.columns
    if has_demo:
        agg_exprs = [
            pl.col("auc").max().alias("Best"),
            pl.col("auc").min().alias("Worst"),
            pl.col("auc").mean().alias("Mean"),
        ]
    else:
        agg_exprs = [pl.col("auc").mean().alias("Mean")]

    per_iter = (
        group_auc.group_by("model", "dataset", "difficulty", "iteration")
        .agg(*agg_exprs)
        .unpivot(
            index=["model", "dataset", "difficulty", "iteration"],
            variable_name="statistic",
            value_name="auc",
        )
    )
    return per_iter.group_by("model", "dataset", "difficulty", "statistic").agg(
        pl.col("auc").mean().alias("auc_mean"),
        pl.col("auc").quantile(0.025, interpolation="linear").alias("ci_lo"),
        pl.col("auc").quantile(0.975, interpolation="linear").alias("ci_hi"),
    )


def plot_difficulty_dumbbell(
    summary: pl.DataFrame,
    plot_dir: Path,
    *,
    pretty_names: Mapping[str, str] | None = None,
    dataset_name_map: Mapping[str, str] | None = None,
    top_n: int = 10,
    file_type: str = "pdf",
    filename_prefix: str = "difficulty_dumbbell",
    font_scale: float = 1.0,
    title: str = "Easy vs hard alignment performance by statistic",
) -> Path:
    """Dumbbell plot: Hard vs Easy alignment, per (model, statistic, dataset)."""
    if dataset_name_map is None:
        dataset_name_map = {"rai": "Responsible AI", "welfare": "Welfare"}

    has_demo = summary.filter(pl.col("statistic").is_in(["Best", "Worst"])).height > 0
    statistics = ["Best", "Mean", "Worst"] if has_demo else ["Mean"]

    overall_order = (
        summary.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc_mean").mean())
        .sort("auc_mean", descending=True)
        .get_column("model")
        .to_list()
    )

    keep_models = overall_order[:top_n + 1]

    pretty = pretty_names or {}
    pretty_keep = [pretty.get(m, m) for m in keep_models]

    plot_data = (
        summary.filter(pl.col("model").is_in(keep_models))
        .with_columns(
            pl.col("model").replace(pretty),
            pl.col("dataset").replace(dataset_name_map),
        )
        .to_pandas()
    )

    datasets = list(dataset_name_map.values())

    sns.set_theme(style="whitegrid", font_scale=font_scale)
    fig_height = max(4.5, 0.9 * len(pretty_keep))
    fig, axes = plt.subplots(1, len(datasets), figsize=(11, fig_height), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    n_stats = len(statistics)
    stat_gap = 0.38   # vertical separation between statistics within one model row
    block_gap = 1.4   # vertical separation between model rows

    _pal = sns.color_palette("coolwarm", n_stats)
    _line_styles = ["-", "--", ":"]
    stat_styles = {
        st: {"color": _pal[n_stats - 1 - i], "ls": _line_styles[i]}
        for i, st in enumerate(statistics)
    }
    stat_y_offsets = {
        st: (i - (n_stats - 1) / 2) * stat_gap for i, st in enumerate(statistics)
    }

    def y_pos(mi: int) -> float:
        return -mi * block_gap

    hard_label = DIFFICULTY_LABELS["hard"]
    easy_label = DIFFICULTY_LABELS["easy"]

    for ax, ds in zip(axes, datasets, strict=False):
        sub = plot_data[plot_data["dataset"] == ds]
        for mi, m in enumerate(pretty_keep):
            y_base = y_pos(mi)
            for st in statistics:
                y = y_base + stat_y_offsets[st]
                style = stat_styles.get(st, {"color": "#636363", "ls": "-"})
                row = sub[(sub["model"] == m) & (sub["statistic"] == st)]
                hard = row[row["difficulty"] == hard_label]
                easy = row[row["difficulty"] == easy_label]
                if not hard.empty and not easy.empty:
                    ax.plot(
                        [hard["auc_mean"].iloc[0], easy["auc_mean"].iloc[0]],
                        [y, y],
                        color=style["color"],
                        ls=style["ls"],
                        lw=2.2,
                        zorder=1,
                    )
                if not hard.empty:
                    ax.scatter(
                        hard["auc_mean"].iloc[0],
                        y,
                        color=style["color"],
                        s=70,
                        zorder=3,
                    )
                if not easy.empty:
                    ax.scatter(
                        easy["auc_mean"].iloc[0],
                        y,
                        facecolors="white",
                        edgecolors=style["color"],
                        s=70,
                        linewidths=1.5,
                        zorder=3,
                    )
            half = (n_stats - 1) / 2 * stat_gap + 0.15
            if mi % 2 == 0:
                ax.axhspan(y_base - half, y_base + half, color="#f4f6fa", zorder=0)

        ax.set_title(ds)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
        ax.grid(axis="y", visible=False)
        ax.set_axisbelow(True)

    axes[0].set_yticks([y_pos(mi) for mi in range(len(pretty_keep))])
    axes[0].set_yticklabels(pretty_keep, fontweight="bold")

    stat_handles = []
    for st in statistics:
        style = stat_styles.get(st, {"color": "#636363", "ls": "-"})
        stat_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                ls=style["ls"],
                lw=2.2,
                marker="o",
                markerfacecolor=style["color"],
                markersize=8,
                label=st,
            )
        )
    diff_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#555555",
            markersize=8,
            label=hard_label,
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="#555555",
            markersize=8,
            markeredgewidth=1.5,
            label=easy_label,
        ),
    ]
    fig.legend(
        handles=[*stat_handles, *diff_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.1),
        ncol=len(stat_handles) + len(diff_handles),
        frameon=False,
    )
    fig.supxlabel("Alignment score", y=0.17)
    fig.suptitle(title, x=0.5)
    plt.tight_layout(rect=(0, 0.14, 1, 0.96))

    out_path = plot_dir / f"{filename_prefix}_difficulty.{file_type}"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
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
