"""Compare pairwise-distance variability from:
    1. Within-rater across rounds ("context drift")
    2. Between-rater within rounds ("rater disagreement")

For each dataset, we:
    - normalise coordinates per (round, rater)
    - compute all pairwise statement distances
    - estimate mean variance:
        * within-rater across rounds
        * between-rater within rounds
    - bootstrap confidence intervals
    - plot grouped bar chart with error bars

Usage
-----
python pairwise_variance_compare.py \
    --placements placements.parquet \
    --out variance_compare.png

Expected schema
---------------
statement_id  str|int
x             float
y             float
dataset       str
seed          int
user_id       str
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from debugpy.common.json import default

from human_grounding.constants import DATASET_PRETTY_NAMES
from human_grounding.directories import PLOT_DIR

plt.style.use("seaborn-v0_8-whitegrid")

RAW_COLS = ("statement_id", "x", "y", "dataset", "seed", "user_id")
REQUIRED_COLS = ("round_id", "rater_id", "statement_id", "x", "y")


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------


def prepare_placements(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    missing = set(RAW_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[df["dataset"] == dataset].copy()
    if out.empty:
        raise ValueError(f"No rows for dataset={dataset!r}")

    out["round_id"] = out["seed"].astype(int)
    out["rater_id"] = out["user_id"].astype(str)

    return out[list(REQUIRED_COLS)].reset_index(drop=True)


# ---------------------------------------------------------------------
# Coordinate normalisation
# ---------------------------------------------------------------------


def _normalise_block(block: pd.DataFrame) -> pd.DataFrame:
    coords = block[["x", "y"]].to_numpy(dtype=float)

    coords = coords - coords.mean(axis=0)

    diffs = coords[:, None, :] - coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    scale = dists.max()
    if scale <= 0:
        scale = 1.0

    out = block.copy()
    out[["x", "y"]] = coords / scale
    return out


def normalise_placements(df: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for _, block in df.groupby(["round_id", "rater_id"], sort=False):
        parts.append(_normalise_block(block))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------
# Pairwise distances
# ---------------------------------------------------------------------


def pairwise_distances(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (rid, raterid), block in df.groupby(["round_id", "rater_id"], sort=False):
        coords = block.set_index("statement_id")[["x", "y"]].to_numpy(dtype=float)
        stmts = block["statement_id"].tolist()

        for (i, stmt_a), (j, stmt_b) in combinations(enumerate(stmts), 2):
            d = float(np.linalg.norm(coords[i] - coords[j]))

            lo, hi = sorted((stmt_a, stmt_b))

            rows.append((rid, raterid, lo, hi, d))

    return pd.DataFrame(
        rows,
        columns=["round_id", "rater_id", "stmt_a", "stmt_b", "dist"],
    )


# ---------------------------------------------------------------------
# Variance estimation
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class VarianceEstimate:
    label: str
    mean: float
    lower: float
    upper: float


def bootstrap_mean(
    values: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)

    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    boot = []

    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot.append(sample.mean())

    alpha = (100 - ci) / 2

    return (
        float(np.mean(values)),
        float(np.percentile(boot, alpha)),
        float(np.percentile(boot, 100 - alpha)),
    )


def estimate_variances(
    pair_df: pd.DataFrame,
    n_boot: int = 1000,
    seed: int = 0,
) -> list[VarianceEstimate]:
    # Within-rater across rounds
    drift = (
        pair_df.groupby(["rater_id", "stmt_a", "stmt_b"])["dist"]
        .var(ddof=1)
        .dropna()
        .to_numpy()
    )

    # Between-rater within round
    rater = (
        pair_df.groupby(["round_id", "stmt_a", "stmt_b"])["dist"]
        .var(ddof=1)
        .dropna()
        .to_numpy()
    )

    drift_mean, drift_lo, drift_hi = bootstrap_mean(
        drift,
        n_boot=n_boot,
        seed=seed,
    )

    rater_mean, rater_lo, rater_hi = bootstrap_mean(
        rater,
        n_boot=n_boot,
        seed=seed,
    )

    return [
        VarianceEstimate(
            label="Within-rater/between-round",
            mean=drift_mean,
            lower=drift_lo,
            upper=drift_hi,
        ),
        VarianceEstimate(
            label="Between-rater/within-round",
            mean=rater_mean,
            lower=rater_lo,
            upper=rater_hi,
        ),
    ]


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_results(
    results: pd.DataFrame,
    outpath: Path,
    font_scale: float = 1.0,
) -> None:
    # -----------------------------------------------------------------
    # Data prep
    # -----------------------------------------------------------------

    results = results.copy()

    results["dataset"] = results["dataset"].replace(DATASET_PRETTY_NAMES)

    datasets = results["dataset"].unique().tolist()
    labels = results["comparison"].unique().tolist()

    # Compute asymmetric error bars for seaborn overlay
    results["lower_err"] = results["mean"] - results["lower"]
    results["upper_err"] = results["upper"] - results["mean"]

    # -----------------------------------------------------------------
    # Styling
    # -----------------------------------------------------------------

    FONT_SCALE = font_scale

    sns.set_theme(style="whitegrid")

    plt.rcParams.update(
        {
            "font.size": 10 * FONT_SCALE,
            "axes.labelsize": 12 * FONT_SCALE,
            "axes.titlesize": 13 * FONT_SCALE,
            "xtick.labelsize": 10 * FONT_SCALE,
            "ytick.labelsize": 10 * FONT_SCALE,
            "legend.fontsize": 10 * FONT_SCALE,
        }
    )

    palette = sns.color_palette("Set1", n_colors=len(labels))

    # -----------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=results,
        x="dataset",
        y="mean",
        hue="comparison",
        order=datasets,
        hue_order=labels,
        palette=palette,
        errorbar=None,
        ax=ax,
    )

    # Add asymmetric error bars manually
    n_hue = len(labels)

    bars = [patch for patch in ax.patches if patch.get_height() != 0]

    expected_rows = (
        results.set_index(["dataset", "comparison"])
        .loc[
            pd.MultiIndex.from_product(
                [datasets, labels],
                names=["dataset", "comparison"],
            )
        ]
        .reset_index()
    )

    for patch, (_, row) in zip(bars, expected_rows.iterrows()):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()

        ax.errorbar(
            x,
            y,
            yerr=[[row["lower_err"]], [row["upper_err"]]],
            fmt="none",
            ecolor="black",
            capsize=5,
            linewidth=1,
        )

    # -----------------------------------------------------------------
    # Labels and legend
    # -----------------------------------------------------------------

    ax.set_ylabel("Mean pairwise variance")
    ax.set_xlabel("")

    # Put legend below plot in a single row
    ax.legend(
        title=None,
        frameon=False,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")

    raise ValueError(f"Unsupported file format: {path.suffix}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=PLOT_DIR / "context_drift_comparison.pdf"
    )

    parser.add_argument("--n_boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)

    args = parser.parse_args()

    raw = load_table(args.placements)

    datasets = sorted(raw["dataset"].unique())

    rows = []

    for dataset in datasets:
        placements = prepare_placements(raw, dataset)
        placements = normalise_placements(placements)

        pair_df = pairwise_distances(placements)

        estimates = estimate_variances(
            pair_df,
            n_boot=args.n_boot,
            seed=args.seed,
        )

        for est in estimates:
            rows.append(
                {
                    "dataset": dataset,
                    "comparison": est.label,
                    "mean": est.mean,
                    "lower": est.lower,
                    "upper": est.upper,
                }
            )

    results = pd.DataFrame(rows)

    print(results)

    plot_results(results, args.out, font_scale=args.scale)

    csv_out = args.out.with_suffix(".csv")
    results.to_csv(csv_out, index=False)

    print(f"\nSaved figure to: {args.out}")
    print(f"Saved summary table to: {csv_out}")


if __name__ == "__main__":
    main()
