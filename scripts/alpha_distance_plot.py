"""
Optimized alpha reliability plot with bootstrap confidence intervals.

Now also produces a "between-rater AUC" bar chart by dataset (best/mean/worst),
matching the attached style but with dataset on the x-axis.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from line_profiler import profile
from loguru import logger
from matplotlib.ticker import LogLocator, ScalarFormatter
from scipy.spatial.distance import cdist
from tqdm import tqdm

from human_grounding.alpha_reliability import (
    generate_triplets_fully_vectorized,
    normalized_auc_logx,
    vectorized_judgment,
)
from human_grounding.data import get_demographics
from human_grounding.directories import OUTPUT_DIR, PLOT_DIR

# Type aliases for clarity
RaterKey = tuple[str, int, str]  # (dataset, seed, user_id)
DistanceData = dict[
    str, np.ndarray
]  # {"statement_ids": ..., "dist_matrix": ..., "demographics": ...}
RaterPair = tuple[RaterKey, RaterKey]


def load_coordinates(data_path: Path) -> pl.DataFrame:
    """Load coordinate data from CSV."""
    return pl.read_csv(data_path).with_columns(
        pl.col("dataset").replace({"welfware": "welfare"})
    )


def compute_distances(coords: pl.DataFrame) -> dict[RaterKey, DistanceData]:
    """
    Compute pairwise distances for each rater's embeddings.

    Args:
        coords: DataFrame with columns [dataset, seed, user_id, statement_id, x, y, demographic]
    """
    distances: dict[RaterKey, DistanceData] = {}

    for (dataset, seed, user_id), group in coords.group_by(
        ["dataset", "seed", "user_id"]
    ):
        key: RaterKey = (str(dataset), int(seed), str(user_id))
        group_sorted = group.sort("statement_id")

        statement_ids = group_sorted["statement_id"].to_numpy()
        xy_coords = group_sorted.select(["x", "y"]).to_numpy().astype(np.float64)
        demographics = group_sorted["demographic"].to_numpy()

        dist_matrix = cdist(xy_coords, xy_coords, metric="euclidean")

        distances[key] = {
            "statement_ids": statement_ids,
            "dist_matrix": dist_matrix,
            "demographics": demographics,
        }

    return distances


@profile
def compute_triplet_agreement_vectorized_with_demographics(
    dist_matrix1: np.ndarray,
    dist_matrix2: np.ndarray,
    demographics1: np.ndarray,
    demographics2: np.ndarray,  # noqa: ARG001
    threshold_d: float,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """
    Vectorized computation of triplet agreement between two distance matrices.

    Returns:
        (num_agreements, num_valid_triplets, demographic_stats)
        where demographic_stats maps demographic -> (agreements, valid_triplets)
    """
    n = dist_matrix1.shape[0]
    if n < 3:
        return 0, 0, {}

    anchors, points_i, points_j = generate_triplets_fully_vectorized(n)

    dist1_to_i = dist_matrix1[anchors, points_i]
    dist1_to_j = dist_matrix1[anchors, points_j]
    dist2_to_i = dist_matrix2[anchors, points_i]
    dist2_to_j = dist_matrix2[anchors, points_j]

    judgment1 = vectorized_judgment(dist1_to_i, dist1_to_j, threshold_d)
    judgment2 = vectorized_judgment(dist2_to_i, dist2_to_j, threshold_d)

    valid_mask = (judgment1 != 0) & (judgment2 != 0)
    valid_triplets = int(np.sum(valid_mask))
    agreements = int(np.sum(judgment1[valid_mask] == judgment2[valid_mask]))

    demographic_stats: dict[str, tuple[int, int]] = {}

    if valid_triplets > 0:
        demo_anchors = demographics1[anchors[valid_mask]]
        demo_i = demographics1[points_i[valid_mask]]
        demo_j = demographics1[points_j[valid_mask]]
        agreement_mask = judgment1[valid_mask] == judgment2[valid_mask]

        unique_demos = np.unique(demographics1)

        for demo in unique_demos:
            demo_mask = (demo_anchors == demo) | (demo_i == demo) | (demo_j == demo)
            demo_valid = int(np.sum(demo_mask))
            demo_agreements = int(np.sum(agreement_mask[demo_mask]))

            if demo_valid > 0:
                demographic_stats[str(demo)] = (demo_agreements, demo_valid)

    return agreements, valid_triplets, demographic_stats


def find_common_indices(
    stmts1: np.ndarray, stmts2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Find indices of common statements between two raters."""
    _, idx1, idx2 = np.intersect1d(
        stmts1, stmts2, return_indices=True, assume_unique=False
    )
    return idx1, idx2


@profile
def compute_alpha_with_bootstrap_demographic(
    all_distances: dict[RaterKey, DistanceData],
    rater_pairs_by_dataset: dict[str, list[RaterPair]],
    d_values: np.ndarray,
    n_bootstrap: int = 100,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """
    Compute alpha values with bootstrap confidence intervals, including demographic-level analysis.

    Returns:
        (alpha_mean, alpha_lower_95, alpha_upper_95, bootstrap_alphas_by_dataset, bootstrap_alphas_by_demographic)
    """
    total_pairs = sum(len(pairs) for pairs in rater_pairs_by_dataset.values())
    if total_pairs == 0:
        zeros = np.zeros_like(d_values)
        return zeros, zeros, zeros, {}, {}

    rng = np.random.default_rng(seed)
    bootstrap_alphas = np.zeros((n_bootstrap, len(d_values)))

    submatrix_pairs_by_dataset: dict[
        str, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ] = {}
    all_demographics: set[str] = set()

    for dataset, rater_pairs in rater_pairs_by_dataset.items():
        submatrix_pairs: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        for key1, key2 in rater_pairs:
            if key1 not in all_distances or key2 not in all_distances:
                continue

            stmts1 = all_distances[key1]["statement_ids"]
            stmts2 = all_distances[key2]["statement_ids"]
            indices1, indices2 = find_common_indices(stmts1, stmts2)

            if len(indices1) < 3:
                continue

            sub_matrix1 = all_distances[key1]["dist_matrix"][np.ix_(indices1, indices1)]
            sub_matrix2 = all_distances[key2]["dist_matrix"][np.ix_(indices2, indices2)]
            demographics1 = all_distances[key1]["demographics"][indices1]
            demographics2 = all_distances[key2]["demographics"][indices2]

            all_demographics.update(demographics1)
            submatrix_pairs.append(
                (sub_matrix1, sub_matrix2, demographics1, demographics2)
            )

        submatrix_pairs_by_dataset[dataset] = submatrix_pairs

    bootstrap_alphas_by_dataset: dict[str, np.ndarray] = {
        dataset: np.zeros((n_bootstrap, len(d_values)))
        for dataset in submatrix_pairs_by_dataset
    }

    bootstrap_alphas_by_demographic: dict[str, np.ndarray] = {
        demo: np.zeros((n_bootstrap, len(d_values))) for demo in all_demographics
    }

    for b in tqdm(range(n_bootstrap), disable=not verbose):
        for i, threshold_d in enumerate(d_values):
            total_agreements = 0
            total_valid = 0

            demographic_agreements: dict[str, int] = dict.fromkeys(all_demographics, 0)
            demographic_valid: dict[str, int] = dict.fromkeys(all_demographics, 0)

            for dataset, submatrix_pairs in submatrix_pairs_by_dataset.items():
                if not submatrix_pairs:
                    continue

                n_pairs = len(submatrix_pairs)
                sampled_indices = rng.choice(n_pairs, size=n_pairs, replace=True)

                dataset_agreements = 0
                dataset_valid = 0
                for idx in sampled_indices:
                    sub_matrix1, sub_matrix2, demographics1, demographics2 = (
                        submatrix_pairs[idx]
                    )
                    agreements, valid, demo_stats = (
                        compute_triplet_agreement_vectorized_with_demographics(
                            sub_matrix1,
                            sub_matrix2,
                            demographics1,
                            demographics2,
                            threshold_d,
                        )
                    )
                    dataset_agreements += agreements
                    dataset_valid += valid

                    for demo, (demo_agree, demo_valid_count) in demo_stats.items():
                        demographic_agreements[demo] += demo_agree
                        demographic_valid[demo] += demo_valid_count

                bootstrap_alphas_by_dataset[dataset][b, i] = (
                    0.0
                    if dataset_valid == 0
                    else 2 * (dataset_agreements / dataset_valid) - 1
                )

                total_agreements += dataset_agreements
                total_valid += dataset_valid

            bootstrap_alphas[b, i] = (
                0.0 if total_valid == 0 else 2 * (total_agreements / total_valid) - 1
            )

            for demo in all_demographics:
                bootstrap_alphas_by_demographic[demo][b, i] = (
                    0.0
                    if demographic_valid[demo] == 0
                    else 2 * (demographic_agreements[demo] / demographic_valid[demo])
                    - 1
                )

        if verbose and (b + 1) % 20 == 0:
            print(f"    Bootstrap {b + 1}/{n_bootstrap}")

    alpha_mean = np.mean(bootstrap_alphas, axis=0)
    alpha_lower = np.percentile(bootstrap_alphas, 2.5, axis=0)
    alpha_upper = np.percentile(bootstrap_alphas, 97.5, axis=0)

    return (
        alpha_mean,
        alpha_lower,
        alpha_upper,
        bootstrap_alphas_by_dataset,
        bootstrap_alphas_by_demographic,
    )


def get_rater_pairs_by_dataset(
    all_distances: dict[RaterKey, DistanceData],
) -> tuple[dict[str, list[RaterPair]], dict[str, list[RaterPair]]]:
    """Identify within-rater and between-rater pairs, grouped by dataset."""
    keys_by_dataset: dict[str, list[RaterKey]] = {}
    for key in all_distances:
        dataset = key[0]
        keys_by_dataset.setdefault(dataset, []).append(key)

    within_pairs_by_dataset: dict[str, list[RaterPair]] = {}
    between_pairs_by_dataset: dict[str, list[RaterPair]] = {}

    for dataset, keys in keys_by_dataset.items():
        within_pairs: list[RaterPair] = []
        between_pairs: list[RaterPair] = []

        for i, key1 in enumerate(keys):
            for key2 in keys[i + 1 :]:
                _, seed1, user1 = key1
                _, seed2, user2 = key2

                if user1 == user2 and seed1 != seed2:
                    within_pairs.append((key1, key2))
                elif user1 != user2:
                    between_pairs.append((key1, key2))

        within_pairs_by_dataset[dataset] = within_pairs
        between_pairs_by_dataset[dataset] = between_pairs

    return within_pairs_by_dataset, between_pairs_by_dataset


# ---------------------------
# NEW: AUC helpers + plotter
# ---------------------------


def compute_auc_summary_from_bootstrap_curves(
    bootstrap_curves: np.ndarray,  # (B, n_d)
    d_values: np.ndarray,
) -> tuple[float, float, float]:
    """Return (mean, lower_95, upper_95) of normalized AUC across bootstrap samples."""
    aucs = np.array(
        [normalized_auc_logx(curve, d_values) for curve in bootstrap_curves],
        dtype=np.float64,
    )
    return (
        float(np.mean(aucs)),
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
    )


def create_between_auc_barplot_by_dataset(
    all_distances: dict[RaterKey, DistanceData],
    between_pairs_by_dataset: dict[str, list[RaterPair]],
    d_values: np.ndarray,
    between_bootstrap_by_dataset: dict[str, np.ndarray],  # dataset -> (B, n_d)
    output_path: Path,
    n_bootstrap: int,
    seed: int | None,
    font_scale: float = 1.5,
    verbose: bool = True,
) -> None:
    """
    Seaborn-only error bars (no manual errorbar drawing):

      - only BETWEEN
      - x-axis = dataset
      - hue = score_type in {best, mean, worst}
      - y = normalized AUC over log(d)
      - error bars = seaborn percentile interval over bootstrap samples
    """
    import pandas as pd

    sns.set_theme(style="whitegrid", font_scale=font_scale)

    datasets = sorted(
        [d for d in between_pairs_by_dataset if d in between_bootstrap_by_dataset]
    )
    if not datasets:
        if verbose:
            print("No datasets available for AUC barplot.")
        return

    rows: list[dict[str, str | int | float]] = []

    for dataset in datasets:
        # -------------------------
        # MEAN (overall) bootstrap AUC samples from per-dataset curves (already computed)
        # -------------------------
        overall_curves = between_bootstrap_by_dataset[dataset]  # (B, n_d)
        overall_aucs = np.array(
            [normalized_auc_logx(curve, d_values) for curve in overall_curves],
            dtype=np.float64,
        )
        for b_idx, auc in enumerate(overall_aucs):
            rows.append(
                {
                    "dataset": dataset,
                    "score_type": "mean",
                    "bootstrap_id": b_idx,
                    "auc": float(auc),
                }
            )

        # -------------------------
        # BEST/WORST demographics (within this dataset)
        # We recompute bootstrap curves per demographic for *this dataset only*.
        # -------------------------
        per_dataset_pairs = {dataset: between_pairs_by_dataset.get(dataset, [])}
        (
            _alpha_mean,
            _alpha_low,
            _alpha_high,
            _boot_by_dataset_unused,
            boot_by_demo,
        ) = compute_alpha_with_bootstrap_demographic(
            all_distances,
            per_dataset_pairs,
            d_values,
            n_bootstrap=n_bootstrap,
            seed=seed,
            verbose=False,
        )

        if not boot_by_demo:
            if verbose:
                print(
                    f"{dataset}: no demographic bootstrap curves; plotting mean only."
                )
            continue

        demo_auc_samples: dict[str, np.ndarray] = {}
        demo_auc_means: dict[str, float] = {}

        for demo, demo_curves in boot_by_demo.items():  # (B, n_d)
            aucs = np.array(
                [normalized_auc_logx(curve, d_values) for curve in demo_curves],
                dtype=np.float64,
            )
            demo_auc_samples[demo] = aucs
            demo_auc_means[demo] = float(np.mean(aucs))

        best_demo = max(demo_auc_means, key=demo_auc_means.get)
        worst_demo = min(demo_auc_means, key=demo_auc_means.get)

        if verbose:
            print(
                f"{dataset}: best={best_demo}({demo_auc_means[best_demo]:.3f}), "
                f"worst={worst_demo}({demo_auc_means[worst_demo]:.3f}), "
                f"mean={float(np.mean(overall_aucs)):.3f}"
            )

        best_aucs = demo_auc_samples[best_demo]
        worst_aucs = demo_auc_samples[worst_demo]

        for b_idx, auc in enumerate(best_aucs):
            rows.append(
                {
                    "dataset": dataset,
                    "score_type": "best",
                    "bootstrap_id": b_idx,
                    "auc": float(auc),
                }
            )
        for b_idx, auc in enumerate(worst_aucs):
            rows.append(
                {
                    "dataset": dataset,
                    "score_type": "worst",
                    "bootstrap_id": b_idx,
                    "auc": float(auc),
                }
            )

    if not rows:
        if verbose:
            print("No rows produced for AUC barplot.")
        return

    df = pd.DataFrame(rows)
    dataset_order = ["Responsible AI", "Welfare"]
    hue_order = ["best", "mean", "worst"]

    df["score_type"] = pd.Categorical(
        df["score_type"], categories=hue_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(12, 6.5))

    sns.barplot(
        data=pl.DataFrame(df).with_columns(
            pl.col("dataset").replace({"rai": "Responsible AI", "welfare": "Welfare"})
        ),
        x="dataset",
        y="auc",
        hue="score_type",
        palette="coolwarm",
        hue_order=hue_order,
        estimator=np.mean,
        errorbar=("pi", 95),
        ax=ax,
    )

    ax.set_xlabel("")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    # Below the plot is cleaner.
    ax.legend(
        title="Score Type",
        framealpha=0.9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    if verbose:
        print(f"AUC bar plot saved to {output_path}")


# ---------------------------
# Existing plotting function
# ---------------------------


@profile
def create_alpha_distance_plots(
    coords: pl.DataFrame,
    output_path: Path,
    csv_output_path: Path | None = None,
    d_min: float = 1.0,
    d_max: float = 8,
    n_points: int = 50,
    n_bootstrap: int = 100,
    seed: int | None = None,
    verbose: bool = True,
    font_scale: float = 1.5,
) -> dict:
    """
    Create the alpha vs distance threshold plot with confidence bands.
    Also creates a BETWEEN-only AUC bar chart by dataset.
    """
    if verbose:
        print("Computing distances...")
    all_distances = compute_distances(coords)

    if verbose:
        print(f"Found {len(all_distances)} rater configurations")

    within_pairs_by_dataset, between_pairs_by_dataset = get_rater_pairs_by_dataset(
        all_distances
    )

    if verbose:
        print("\nPairs by dataset:")
        for dataset in sorted(
            set(within_pairs_by_dataset.keys()) | set(between_pairs_by_dataset.keys())
        ):
            n_within = len(within_pairs_by_dataset.get(dataset, []))
            n_between = len(between_pairs_by_dataset.get(dataset, []))
            print(f"  {dataset}: {n_within} within-rater, {n_between} between-rater")

    d_values = np.logspace(np.log10(d_min), np.log10(d_max), n_points)

    if verbose:
        print("\nComputing between-rater agreement with bootstrap...")
    (
        alpha_between,
        alpha_between_lower,
        alpha_between_upper,
        between_alphas_by_dataset,
        between_alphas_by_demographic,
    ) = compute_alpha_with_bootstrap_demographic(
        all_distances,
        between_pairs_by_dataset,
        d_values,
        n_bootstrap,
        seed=seed,
        verbose=verbose,
    )

    if verbose:
        print("\nComputing within-rater agreement with bootstrap...")
    (
        alpha_within,
        alpha_within_lower,
        alpha_within_upper,
        within_alphas_by_dataset,
        within_alphas_by_demographic,
    ) = compute_alpha_with_bootstrap_demographic(
        all_distances,
        within_pairs_by_dataset,
        d_values,
        n_bootstrap,
        seed=seed,
        verbose=verbose,
    )

    # Report within rater agreement
    logger.debug(f"Within-rater agreement (overall): {alpha_within.mean():.3f} (mean)")
    # --- NEW: create the attached-style AUC bar plot (between only, x=dataset)
    auc_barplot_path = PLOT_DIR / "agreement_auc_between_by_dataset.pdf"
    create_between_auc_barplot_by_dataset(
        all_distances=all_distances,
        between_pairs_by_dataset=between_pairs_by_dataset,
        d_values=d_values,
        between_bootstrap_by_dataset=between_alphas_by_dataset,
        output_path=auc_barplot_path,
        n_bootstrap=n_bootstrap,
        seed=seed,
        font_scale=font_scale,
        verbose=verbose,
    )

    # (Your existing “best/worst demographic overall” code + alpha distance plot stays unchanged)
    if between_alphas_by_demographic:
        demographic_mean_alphas = {
            demo: np.mean(alphas)
            for demo, alphas in between_alphas_by_demographic.items()
        }
        best_demographic = max(demographic_mean_alphas, key=demographic_mean_alphas.get)
        worst_demographic = min(
            demographic_mean_alphas, key=demographic_mean_alphas.get
        )

        best_alphas = between_alphas_by_demographic[best_demographic]
        worst_alphas = between_alphas_by_demographic[worst_demographic]

        best_mean = np.mean(best_alphas, axis=0)
        best_lower = np.percentile(best_alphas, 2.5, axis=0)
        best_upper = np.percentile(best_alphas, 97.5, axis=0)

        worst_mean = np.mean(worst_alphas, axis=0)
        worst_lower = np.percentile(worst_alphas, 2.5, axis=0)
        worst_upper = np.percentile(worst_alphas, 97.5, axis=0)

        if verbose:
            print(
                f"\nBest demographic group: {best_demographic} (mean alpha = {demographic_mean_alphas[best_demographic]:.3f})"
            )
            print(
                f"Worst demographic group: {worst_demographic} (mean alpha = {demographic_mean_alphas[worst_demographic]:.3f})"
            )
    else:
        best_demographic = worst_demographic = None
        best_mean = best_lower = best_upper = None
        worst_mean = worst_lower = worst_upper = None

    # Build and save CSV data (unchanged)
    alpha_df = build_alpha_dataframe(
        d_values,
        between_alphas_by_dataset,
        within_alphas_by_dataset,
        between_alphas_by_demographic,
        within_alphas_by_demographic,
    )
    if csv_output_path is not None:
        alpha_df.write_csv(csv_output_path)
        if verbose:
            print(f"\nCSV data saved to {csv_output_path}")

    # Create alpha-vs-distance plot (unchanged)
    sns.set_theme(style="whitegrid", font_scale=1.8)
    _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        d_values,
        alpha_between,
        color="#1f77b4",
        linewidth=2.5,
        label="Between-rater (overall)",
        zorder=3,
    )
    ax.fill_between(
        d_values,
        alpha_between_lower,
        alpha_between_upper,
        color="#1f77b4",
        alpha=0.2,
        zorder=2,
    )

    if best_mean is not None:
        ax.plot(
            d_values,
            best_mean,
            color="#2ca02c",
            linewidth=2.0,
            label="Best group",
            linestyle="--",
            zorder=3,
        )
        ax.fill_between(
            d_values, best_lower, best_upper, color="#2ca02c", alpha=0.15, zorder=1
        )

    if worst_mean is not None:
        ax.plot(
            d_values,
            worst_mean,
            color="#d62728",
            linewidth=2.0,
            label="Worst group",
            linestyle="--",
            zorder=3,
        )
        ax.fill_between(
            d_values, worst_lower, worst_upper, color="#d62728", alpha=0.15, zorder=1
        )

    ax.axhline(
        y=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        # label="Chance level",
        zorder=0,
    )

    high_reliability_mask = alpha_between >= 0.95
    if np.any(high_reliability_mask):
        first_high_idx = np.where(high_reliability_mask)[0][0]
        x_max = d_values[min(first_high_idx + 5, len(d_values) - 1)]
    else:
        x_max = d_max

    ax.set_xscale("log")
    ax.set_xlim(d_min, x_max)

    # Major ticks at 1-9 each decade (so 3 appears)
    ax.xaxis.set_major_locator(
        LogLocator(base=10.0, subs=np.arange(1, 10), numticks=15)
    )

    # Minor ticks (optional - can remove if redundant)
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=100)
    )

    # Force plain numbers
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    formatter.set_useOffset(False)

    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)  # ← important

    ax.set_xlabel("Distance ratio threshold d (far / close)  [log scale]")
    ax.set_ylabel("Krippendorff's α")  # noqa: RUF001
    ax.set_ylim(-0.1, 1.0)
    ax.grid(True, alpha=0.3, which="major")
    ax.grid(True, alpha=0.15, which="minor", linestyle=":")
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if verbose:
        print(f"\nPlot saved to {output_path}")
    plt.close()

    return {
        "d_values": d_values,
        "alpha_between": alpha_between,
        "alpha_between_lower": alpha_between_lower,
        "alpha_between_upper": alpha_between_upper,
        "alpha_within": alpha_within,
        "alpha_within_lower": alpha_within_lower,
        "alpha_within_upper": alpha_within_upper,
        "within_pairs_by_dataset": within_pairs_by_dataset,
        "between_pairs_by_dataset": between_pairs_by_dataset,
        "alpha_df": alpha_df,
        "best_demographic": best_demographic,
        "worst_demographic": worst_demographic,
        "best_mean": best_mean,
        "worst_mean": worst_mean,
        "between_alphas_by_demographic": between_alphas_by_demographic,
        "auc_barplot_path": str(PLOT_DIR / "agreement_auc_between_by_dataset.pdf"),
    }


def build_alpha_dataframe(
    d_values: np.ndarray,
    between_alphas_by_dataset: dict[str, np.ndarray],
    within_alphas_by_dataset: dict[str, np.ndarray],
    between_alphas_by_demographic: dict[str, np.ndarray],
    within_alphas_by_demographic: dict[str, np.ndarray],
) -> pl.DataFrame:
    """Build a tidy DataFrame with all bootstrap alpha values."""
    rows: list[dict[str, str | int | float]] = []

    for dataset, alphas in between_alphas_by_dataset.items():
        n_bootstrap, _ = alphas.shape
        for iteration_id in range(n_bootstrap):
            for d_idx, d in enumerate(d_values):
                rows.append(
                    {
                        "group_type": "dataset",
                        "group_name": dataset,
                        "reliability_type": "between",
                        "iteration_id": iteration_id,
                        "d": d,
                        "krippendorf": alphas[iteration_id, d_idx],
                    }
                )

    for dataset, alphas in within_alphas_by_dataset.items():
        n_bootstrap, _ = alphas.shape
        for iteration_id in range(n_bootstrap):
            for d_idx, d in enumerate(d_values):
                rows.append(
                    {
                        "group_type": "dataset",
                        "group_name": dataset,
                        "reliability_type": "within",
                        "iteration_id": iteration_id,
                        "d": d,
                        "krippendorf": alphas[iteration_id, d_idx],
                    }
                )

    for demographic, alphas in between_alphas_by_demographic.items():
        n_bootstrap, _ = alphas.shape
        for iteration_id in range(n_bootstrap):
            for d_idx, d in enumerate(d_values):
                rows.append(
                    {
                        "group_type": "demographic",
                        "group_name": demographic,
                        "reliability_type": "between",
                        "iteration_id": iteration_id,
                        "d": d,
                        "krippendorf": alphas[iteration_id, d_idx],
                    }
                )

    for demographic, alphas in within_alphas_by_demographic.items():
        n_bootstrap, _ = alphas.shape
        for iteration_id in range(n_bootstrap):
            for d_idx, d in enumerate(d_values):
                rows.append(
                    {
                        "group_type": "demographic",
                        "group_name": demographic,
                        "reliability_type": "within",
                        "iteration_id": iteration_id,
                        "d": d,
                        "krippendorf": alphas[iteration_id, d_idx],
                    }
                )

    return pl.DataFrame(rows)


def main(samples: int = 5, font_scale: float = 1.5) -> None:
    """Main execution function."""
    data_path = OUTPUT_DIR / "combined_coordinates.csv"
    output_path = PLOT_DIR / "alpha_distance_plot_demographic.pdf"
    csv_output_path = OUTPUT_DIR / "alpha_data_demographic.csv"

    print("Loading coordinate data...")
    demographics = get_demographics()
    coords = load_coordinates(data_path).join(demographics, on="statement_id")

    print(f"Loaded {len(coords)} coordinate entries\n")

    create_alpha_distance_plots(
        coords,
        output_path,
        csv_output_path=csv_output_path,
        n_bootstrap=samples,
        seed=42,
        font_scale=font_scale,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate alpha vs distance threshold plot with bootstrap CIs and demographic analysis."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of bootstrap samples (default: 5)",
    )
    parser.add_argument(
        "--scale", type=float, help="Font scale for the plot", default=2.0
    )
    args = parser.parse_args()
    main(samples=args.samples, font_scale=args.scale)
