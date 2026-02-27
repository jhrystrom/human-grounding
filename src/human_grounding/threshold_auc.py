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

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import polars.selectors as cs
import seaborn as sns
from loguru import logger
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

HUMAN_MODEL_NAME = "Human"


# ---------------------------------------------------------------------------
# 1. Filter + agreement logic (extracted from main)
# ---------------------------------------------------------------------------


def filter_by_distance_threshold(
    combined_results: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    """Apply distance-ratio filter and inter-rater agreement dedup.

    Mirrors the ``with_sorted_groups`` -> ``agreement_indices`` ->
    ``filtered_comparisons`` block in ``main``, parameterised by
    *threshold*.
    """
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


# ---------------------------------------------------------------------------
# 2. Demographic join (extracted from main)
# ---------------------------------------------------------------------------


def join_demographics(
    filtered_comparisons: pl.DataFrame,
    welfare_demographics: pl.DataFrame,
    rai_demographics: pl.DataFrame,
) -> pl.DataFrame:
    """Attach demographic labels.

    Returns a frame with columns
    ``[model, dataset, demographics, embedding_correct]``.
    """
    welfare_filtered = (
        filtered_comparisons.filter(pl.col("dataset") == "welfare")
        .drop("demographic")
        .join(
            welfare_demographics,
            left_on="source_idx",
            right_on="cause_id",
        )
        .join(
            welfare_demographics,
            left_on="closer_idx",
            right_on="cause_id",
            suffix="_close",
        )
        .join(
            welfare_demographics,
            left_on="farther_idx",
            right_on="cause_id",
            suffix="_far",
        )
        .with_columns(
            pl.concat_list(cs.starts_with("demographic")).alias(
                "demographics",
            ),
        )
        .drop(cs.starts_with("education_level"))
        .explode("demographics")
    )

    rai_filtered = (
        filtered_comparisons.filter(pl.col("dataset") == "rai")
        .drop("demographic")
        .join(
            rai_demographics,
            left_on="source_idx",
            right_on="cause_id",
        )
        .join(
            rai_demographics,
            left_on="closer_idx",
            right_on="cause_id",
            suffix="_close",
        )
        .join(
            rai_demographics,
            left_on="farther_idx",
            right_on="cause_id",
            suffix="far",
        )
        .with_columns(
            pl.concat_list(cs.starts_with("demographic")).alias(
                "demographics",
            ),
        )
        .drop(cs.starts_with("education_level"))
        .explode("demographics")
    )

    keep = ["model", "dataset", "demographics", "embedding_correct"]
    return pl.concat(
        [
            welfare_filtered.select(keep),
            rai_filtered.select(keep),
        ],
    )


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


def _bootstrap_one_replicate(
    demo_frames: dict[float, pl.DataFrame],
    iteration: int,
) -> pl.DataFrame:
    """Draw one bootstrap sample per threshold, return alignment scores.

    Groups by ``(model, dataset, demographics)`` so downstream code can
    derive per-group AUC.

    Returns columns
    ``[model, dataset, demographics, threshold, alignment_score,
    iteration]``.
    """
    rows: list[pl.DataFrame] = []
    for threshold, df in demo_frames.items():
        agg = (
            df.sample(fraction=1.0, with_replacement=True)
            .group_by("model", "dataset", "demographics")
            .agg(pl.col("embedding_correct").mean())
            .with_columns(
                (2 * pl.col("embedding_correct") - 1).alias(
                    "alignment_score",
                ),
                pl.lit(threshold).alias("threshold"),
                pl.lit(iteration).alias("iteration"),
            )
            .drop("embedding_correct")
        )
        rows.append(agg)
    return pl.concat(rows)


# ---------------------------------------------------------------------------
# 5. Trapezoidal AUC (numpy — applied per group)
# ---------------------------------------------------------------------------


def _auc_trapz_np(
    thresholds: np.ndarray,
    scores: np.ndarray,
) -> float:
    """Normalised trapezoidal AUC over sorted (threshold, score) pairs."""
    order = np.argsort(thresholds)
    x = thresholds[order]
    y = scores[order]
    span = float(x[-1] - x[0])
    if span == 0:
        return float(y.mean())
    return float(np.trapezoid(y, x) / span)


# ---------------------------------------------------------------------------
# 6. Compute per-(model, dataset, demographics, iteration) AUC from curve
# ---------------------------------------------------------------------------


def _curve_to_group_auc(
    curve: pl.DataFrame,
) -> pl.DataFrame:
    """Compute AUC for every (model, dataset, demographics, iteration).

    Returns columns
    ``[model, dataset, demographics, iteration, auc]``.
    """
    auc_rows: list[dict[str, object]] = []
    group_keys = ["model", "dataset", "demographics", "iteration"]
    for keys, group in curve.group_by(group_keys):
        model, dataset, demographics, iteration = keys
        arr = group.sort("threshold")
        auc_val = _auc_trapz_np(
            arr["threshold"].to_numpy(),
            arr["alignment_score"].to_numpy(),
        )
        auc_rows.append(
            {
                "model": model,
                "dataset": dataset,
                "demographics": demographics,
                "iteration": iteration,
                "auc": auc_val,
            },
        )
    return pl.DataFrame(auc_rows)


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
# 8. Main entry point — sweep + AUC with bootstrap CIs
# ---------------------------------------------------------------------------


def compute_threshold_auc(
    combined_results: pl.DataFrame,
    welfare_demographics: pl.DataFrame,
    rai_demographics: pl.DataFrame,
    thresholds: Sequence[float] | None = None,
    n_bootstrap: int = 100,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sweep thresholds and return per-group AUC with bootstrap CIs.

    Parameters
    ----------
    combined_results:
        Raw evaluation results (before any distance filtering).
    welfare_demographics:
        ``cause_id, demographic`` frame for the welfare dataset.
    rai_demographics:
        As returned by ``get_rai_demographics()``.
    thresholds:
        Distance-ratio values to evaluate.  Defaults to a log-spaced
        grid from 1.0 to 6.5 (matching the alpha-vs-d plot x-axis).
    n_bootstrap:
        Bootstrap iterations (drives confidence intervals).

    Returns
    -------
    group_auc : pl.DataFrame
        ``[model, dataset, demographics, iteration, auc]`` — one AUC
        per (model, dataset, demographic group, bootstrap replicate).
    curve : pl.DataFrame
        ``[model, dataset, demographics, threshold, iteration,
        alignment_score]`` — raw per-threshold scores.
    """
    if thresholds is None:
        thresholds = np.logspace(
            np.log10(1.0),
            np.log10(6.5),
            num=30,
        ).tolist()

    # --- expensive but deterministic: done once -------------------------
    demo_frames = precompute_demographic_frames(
        combined_results,
        welfare_demographics,
        rai_demographics,
        thresholds,
    )

    if not demo_frames:
        empty = pl.DataFrame(
            schema={
                "model": pl.Utf8,
                "dataset": pl.Utf8,
                "demographics": pl.Utf8,
                "iteration": pl.Int64,
                "auc": pl.Float64,
            },
        )
        return empty, pl.DataFrame()

    # --- cheap stochastic part: bootstrap replicates --------------------
    replicate_frames: list[pl.DataFrame] = [
        _bootstrap_one_replicate(demo_frames, iteration=i)
        for i in tqdm(range(n_bootstrap), desc="Bootstrap replicates")
    ]
    curve = pl.concat(replicate_frames)

    # --- AUC per (model, dataset, demographics, iteration) --------------
    group_auc = _curve_to_group_auc(curve)

    return group_auc, curve


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

    per_iter = (
        group_auc.group_by("model", "dataset", "iteration")
        .agg(
            pl.col("auc").max().alias("Best Group"),
            pl.col("auc").min().alias("Worst Group"),
            pl.col("auc").mean().alias("Mean"),
        )
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


def plot_auc_bar(
    group_auc: pl.DataFrame,
    plot_dir: Path,
    *,
    pretty_names: Mapping[str, str] | None = None,
    dataset_name_map: Mapping[str, str] | None = None,
    top_n: int = 10,
    font_scale: float = 1.35,
    ci: float = 95.0,
    height: float = 10.0,
    facet_width: float = 9.0,
    use_english: bool = False,
    file_type: str = "pdf",
) -> Path:
    """Faceted horizontal bar chart of AUC with best/worst/mean bars.

    Parameters
    ----------
    group_auc:
        ``[model, dataset, demographics, iteration, auc]`` as returned
        by ``compute_threshold_auc`` (optionally concatenated with
        ``load_human_auc`` output).
    plot_dir:
        Directory for the output PDF.
    pretty_names:
        Optional model-name mapping for display.
    dataset_name_map:
        Optional dataset label mapping (e.g. ``{"rai": "Resp. AI"}``).
    top_n:
        Number of top models to show (human baseline always included).
    font_scale:
        Seaborn font scale.
    ci:
        Confidence interval width in percent (default 95).
    height:
        Height per facet.
    facet_width:
        Width per facet.
    use_english:
        Whether the dataset is in English.
    file_type:
        Output file type (e.g. "pdf", "png", "jpg").
    """
    if pretty_names is None:
        pretty_names = {}
    if dataset_name_map is None:
        dataset_name_map = {
            "rai": "Responsible AI",
            "welfare": "Welfare",
        }

    # Write human subset for debugging
    human_subset = group_auc.filter(pl.col("model") == HUMAN_MODEL_NAME)
    human_debug_path = plot_dir / "debug_human_auc.csv"
    human_subset.write_csv(human_debug_path)
    logger.debug(f"Wrote human AUC debug file to {human_debug_path}")

    # Build per-iteration best/worst/mean (seaborn computes CI natively)
    per_iter = (
        group_auc.group_by("model", "dataset", "iteration")
        .agg(
            pl.col("auc").max().alias("Best Group"),
            pl.col("auc").min().alias("Worst Group"),
            pl.col("auc").mean().alias("Mean"),
        )
        .unpivot(
            index=["model", "dataset", "iteration"],
            variable_name="statistic",
            value_name="auc",
        )
    )

    # Rank models by overall mean AUC (across datasets + statistics)
    model_ranks = (
        per_iter.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc").mean().alias("overall"))
        .sort("overall", descending=True)
    )

    # Always keep human baseline; take top_n from the rest
    human_models = model_ranks.filter(
        pl.col("model") == HUMAN_MODEL_NAME,
    )
    non_human = model_ranks.filter(
        pl.col("model") != HUMAN_MODEL_NAME,
    ).head(top_n)
    keep_models = pl.concat([human_models, non_human]).select("model")

    plot_data = per_iter.join(keep_models, on="model").with_columns(
        pl.col("model").replace(pretty_names).alias("model"),
        pl.col("dataset").replace(dataset_name_map).alias("dataset"),
    )

    # Order models by mean AUC (descending: best on top)
    model_order = (
        plot_data.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc").mean())
        .sort("auc", descending=True)
        .get_column("model")
        .to_list()
    )

    pdf = plot_data.to_pandas()
    stat_order = ["Best Group", "Mean", "Worst Group"]
    dataset_order = sorted(pdf["dataset"].unique())

    sns.set_theme(style="whitegrid", font_scale=font_scale)

    # seaborn catplot with native CI from the bootstrap iterations
    g = sns.catplot(
        data=pdf,
        x="auc",
        y="model",
        hue="statistic",
        col="dataset",
        col_order=dataset_order,
        hue_order=stat_order,
        order=model_order,
        kind="bar",
        palette="coolwarm",
        height=height,
        width=0.8,
        aspect=facet_width / height,
        orient="h",
        errorbar=("ci", ci),
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.supxlabel("Alignment AUC (normalised)")
    min_ticks = 4
    locator = mticker.MaxNLocator(nbins="auto", min_n_ticks=min_ticks)

    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}".lstrip("0") or "0")
        )  # optional, keeps plain numbers

    sns.move_legend(
        g,
        "lower center",
        ncol=len(stat_order),
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
        title=None,
    )

    out_path = plot_dir / f"alignment_auc_bar.{file_type}"
    if use_english:
        out_path = plot_dir / f"alignment_auc_bar_english.{file_type}"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved AUC bar chart to {out_path}")
    return out_path
