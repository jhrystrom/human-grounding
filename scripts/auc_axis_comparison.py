"""Linear vs log-x alpha-AUC sensitivity check (reviewer Q1).

Produces, per experiment dataset:

- per-(model, dataset) AUC computed under both log-x and linear-x integration
  from the *same* alpha(d) curve,
- Spearman rho between the two model rankings,
- top-N overlap and the rows where the ranks disagree.

Output: ``output/auc_axis_comparison.csv`` (per-model AUCs under both schemes)
and ``output/auc_axis_comparison.log`` (human-readable summary).
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr

import human_grounding.threshold_auc as ta
from human_grounding.data import get_rai_demographics, get_welfare_demographics
from human_grounding.directories import DATA_DIR, OUTPUT_DIR
from human_grounding.embed import get_all_models

# Sibling-script import (scripts/ is not a package)
sys.path.append(str(Path(__file__).parent))
from neural_alignment_plots import get_embedding_alignments

# Match neural_alignment_plots.py
COORDINATES = {
    "policy": "valid_coordinates.csv",
    "gov-ai": "govai_coordinates.csv",
}


def _auc_linear(thresholds: np.ndarray, scores: np.ndarray) -> float:
    """Normalised trapezoidal AUC under linear-x integration."""
    order = np.argsort(thresholds)
    x, y = thresholds[order], scores[order]
    span = float(x[-1] - x[0])
    return float(np.trapezoid(y, x) / span) if span > 0 else float(y.mean())


def _curve_to_auc(curve: pl.DataFrame, scheme: str) -> pl.DataFrame:
    """Collapse the per-threshold alpha curve to one AUC per (model, dataset, demographic, iteration)."""
    auc_fn = ta._auc_trapz_np if scheme == "log" else _auc_linear
    keys = [*ta._group_keys(curve), "iteration"]
    rows = []
    for key_vals, group in curve.group_by(keys):
        rows.append(
            {
                **dict(zip(keys, key_vals)),
                "auc": auc_fn(
                    group["threshold"].to_numpy(),
                    group["alignment_score"].to_numpy(),
                ),
            }
        )
    return pl.DataFrame(rows)


def _compute_curve(experiment: str) -> pl.DataFrame:
    """Run the standard binary-alpha pipeline and return the per-threshold curve."""
    full_dataset = pl.read_csv(DATA_DIR / COORDINATES[experiment])
    welfare_demographics = (
        get_welfare_demographics() if experiment == "policy" else None
    )
    rai_demographics = get_rai_demographics() if experiment == "policy" else None
    models = sorted(get_all_models())

    combined = get_embedding_alignments(models, full_dataset, use_english=False)
    _, curve = ta.compute_threshold_auc(
        combined_results=combined,
        welfare_demographics=welfare_demographics,
        rai_demographics=rai_demographics,
        n_bootstrap=10,
        metric="binary",
    )
    return curve


def _summarise(curve: pl.DataFrame, experiment: str, top_n: int) -> dict:
    """Per-(model, dataset) mean AUC under each scheme + ranking comparison."""
    log_auc = _curve_to_auc(curve, "log")
    lin_auc = _curve_to_auc(curve, "linear")

    log_mean = log_auc.group_by("model", "dataset").agg(
        pl.col("auc").mean().alias("auc_log")
    )
    lin_mean = lin_auc.group_by("model", "dataset").agg(
        pl.col("auc").mean().alias("auc_linear")
    )
    joined = log_mean.join(lin_mean, on=["model", "dataset"], how="inner").with_columns(
        pl.lit(experiment).alias("experiment")
    )

    per_dataset: list[dict] = []
    for (dataset,), sub in joined.group_by(["dataset"]):
        x = sub["auc_log"].to_numpy()
        y = sub["auc_linear"].to_numpy()
        rho = float(spearmanr(x, y).statistic) if len(x) >= 3 else float("nan")

        ranked_log = sub.sort("auc_log", descending=True).head(top_n)["model"].to_list()
        ranked_lin = (
            sub.sort("auc_linear", descending=True).head(top_n)["model"].to_list()
        )
        overlap = len(set(ranked_log) & set(ranked_lin))
        per_dataset.append(
            {
                "experiment": experiment,
                "dataset": str(dataset),
                "n_models": sub.height,
                "spearman_log_vs_linear": rho,
                f"top{top_n}_overlap": overlap,
                f"top{top_n}_log": ranked_log,
                f"top{top_n}_linear": ranked_lin,
            }
        )

    return {"per_model": joined, "per_dataset": per_dataset}


def main(experiments: Sequence[str], top_n: int = 10) -> None:
    log_path = OUTPUT_DIR / "auc_axis_comparison.log"
    csv_path = OUTPUT_DIR / "auc_axis_comparison.csv"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", mode="w")

    all_per_model: list[pl.DataFrame] = []
    all_summaries: list[dict] = []
    for experiment in experiments:
        logger.info(f"Computing alpha curve for experiment={experiment}...")
        curve = _compute_curve(experiment)
        result = _summarise(curve, experiment, top_n=top_n)
        all_per_model.append(result["per_model"])
        all_summaries.extend(result["per_dataset"])

        for row in result["per_dataset"]:
            logger.info(
                f"[{row['experiment']}/{row['dataset']}] "
                f"Spearman(log,linear) rho = {row['spearman_log_vs_linear']:.4f} "
                f"over {row['n_models']} models; "
                f"top-{top_n} overlap = {row[f'top{top_n}_overlap']}/{top_n}"
            )
            logger.info(f"  top-{top_n} (log-x):    {row[f'top{top_n}_log']}")
            logger.info(f"  top-{top_n} (linear-x): {row[f'top{top_n}_linear']}")

    combined_per_model = pl.concat(all_per_model, how="vertical_relaxed")
    combined_per_model.write_csv(csv_path)
    logger.info(f"Per-model AUC table written to {csv_path}")
    logger.info(f"Log written to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["policy", "gov-ai"],
        choices=["policy", "gov-ai"],
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    main(experiments=args.experiments, top_n=args.top_n)
