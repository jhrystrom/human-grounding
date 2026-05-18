"""AUC-parameterisation sensitivity check (reviewer Q1).

Computes one master alpha(d) curve at a dense log-spaced grid and
re-integrates it under several configurations:

- ``d_max`` in {4, 6.5, 8, 10} (n_points=30, log-x),
- ``n_points`` in {15, 50} (d_max=6.5, log-x),
- linear-d integration (d_max=6.5, n_points=30).

Each variant is compared against the main configuration
(d_max=6.5, n_points=30, log-x) on per-model AUC averaged across
all (experiment, dataset) cells. Reports:

- full-rank Spearman rho,
- top-10 Spearman rho (on the union of either configuration's top-10),
- top-10 model-set overlap.

Outputs:

- ``output/auc_axis_comparison.csv`` -- per-(model, dataset, experiment)
  AUC under log-x and linear-x integration at d_max=6.5,
- ``output/auc_axis_comparison.log`` -- human-readable summary,
- ``output/auc_sensitivity_table.tex`` -- LaTeX table matching
  ``tab:auc-sensitivity``.
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


def _compute_curve(
    experiment: str, thresholds: Sequence[float] | None = None
) -> pl.DataFrame:
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
        thresholds=list(thresholds) if thresholds is not None else None,
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


# ---------------------------------------------------------------------------
# Sensitivity table (reviewer Q1, broader configurations)
# ---------------------------------------------------------------------------

MASTER_GRID = np.logspace(np.log10(1.0), np.log10(10.0), 100)

MAIN_CONFIG: dict = {"d_max": 6.5, "n_points": 30, "scheme": "log"}

# (label, config) in display order. Label is rendered verbatim in LaTeX.
SENSITIVITY_CONFIGS: list[tuple[str, dict]] = [
    (r"$d_{\max}=4$", {"d_max": 4.0, "n_points": 30, "scheme": "log"}),
    (r"$d_{\max}=6.5$", {"d_max": 6.5, "n_points": 30, "scheme": "log"}),
    (r"$d_{\max}=8$", {"d_max": 8.0, "n_points": 30, "scheme": "log"}),
    (r"$d_{\max}=10$", {"d_max": 10.0, "n_points": 30, "scheme": "log"}),
    (r"$n_{\mathrm{points}}=15$", {"d_max": 6.5, "n_points": 15, "scheme": "log"}),
    (r"$n_{\mathrm{points}}=50$", {"d_max": 6.5, "n_points": 50, "scheme": "log"}),
    (r"Linear-$d$ integration", {"d_max": 6.5, "n_points": 30, "scheme": "linear"}),
]


def _snap_to_master(target: np.ndarray, master: np.ndarray) -> list[float]:
    """Map each target threshold to the nearest unique master-grid value."""
    snapped = {float(master[int(np.argmin(np.abs(master - t)))]) for t in target}
    return sorted(snapped)


def _config_per_model_auc(
    curve: pl.DataFrame,
    config: dict,
    master: np.ndarray,
) -> pl.DataFrame:
    """Per-model mean AUC under one configuration, averaged across all groups."""
    target = np.logspace(np.log10(1.0), np.log10(config["d_max"]), config["n_points"])
    snapped = _snap_to_master(target, master)
    sub = curve.filter(pl.col("threshold").is_in(snapped))

    auc_fn = ta._auc_trapz_np if config["scheme"] == "log" else _auc_linear
    keys = [*ta._group_keys(sub), "iteration"]
    rows = []
    for key_vals, group in sub.group_by(keys):
        rows.append(
            {
                **dict(zip(keys, key_vals)),
                "auc": auc_fn(
                    group["threshold"].to_numpy(),
                    group["alignment_score"].to_numpy(),
                ),
            }
        )
    per_iter = pl.DataFrame(rows)
    return per_iter.group_by("model").agg(pl.col("auc").mean().alias("auc"))


def _compare_to_main(
    main_auc: pl.DataFrame,
    variant_auc: pl.DataFrame,
    top_n: int,
) -> tuple[float, float, int]:
    """Return (full_rho, top_n_rho, top_n_overlap) comparing variant vs main."""
    joined = main_auc.rename({"auc": "auc_main"}).join(
        variant_auc.rename({"auc": "auc_variant"}), on="model", how="inner"
    )
    x = joined["auc_main"].to_numpy()
    y = joined["auc_variant"].to_numpy()

    full_rho = float(spearmanr(x, y).statistic) if len(x) >= 3 else float("nan")

    top_main = set(
        joined.sort("auc_main", descending=True).head(top_n)["model"].to_list()
    )
    top_var = set(
        joined.sort("auc_variant", descending=True).head(top_n)["model"].to_list()
    )
    overlap = len(top_main & top_var)

    union = sorted(top_main | top_var)
    sub = joined.filter(pl.col("model").is_in(union))
    top_rho = (
        float(
            spearmanr(
                sub["auc_main"].to_numpy(), sub["auc_variant"].to_numpy()
            ).statistic
        )
        if sub.height >= 3
        else float("nan")
    )
    return full_rho, top_rho, overlap


def _build_sensitivity_tex(
    rows: list[tuple[str, float, float, int]],
    top_n: int,
) -> str:
    header = (
        r"\textbf{Configuration} & \textbf{Full-rank $\rho$} & "
        r"\textbf{Top-" + str(top_n) + r" $\rho$} & "
        r"\textbf{Top-" + str(top_n) + r" overlap} \\"
    )
    body = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Sensitivity of model-grounding rankings to AUC parameterisation. "
        r"Rank correlations are computed against the main configuration.}",
        r"\label{tab:auc-sensitivity}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for label, full_rho, top_rho, overlap in rows:
        body.append(
            f"{label} & {full_rho:.2f} & {top_rho:.2f} & {overlap}/{top_n} \\\\"
        )
    body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(body)


def _write_sensitivity_table(
    master_curves: dict[str, pl.DataFrame],
    top_n: int,
    output_path: Path,
) -> None:
    """Aggregate per-experiment master curves and write the LaTeX sensitivity table."""
    # Pool curves across experiments. demographics/group keys may differ across
    # experiments; for the per-model aggregation only `model` and `auc` matter.
    pooled = pl.concat(
        [
            df.with_columns(pl.lit(exp).alias("experiment"))
            for exp, df in master_curves.items()
        ],
        how="vertical_relaxed",
    )

    main_auc = _config_per_model_auc(pooled, MAIN_CONFIG, MASTER_GRID)

    rows: list[tuple[str, float, float, int]] = []
    for label, config in SENSITIVITY_CONFIGS:
        variant_auc = _config_per_model_auc(pooled, config, MASTER_GRID)
        full_rho, top_rho, overlap = _compare_to_main(main_auc, variant_auc, top_n)
        logger.info(
            f"[sensitivity] {label}: full rho={full_rho:.4f}, "
            f"top-{top_n} rho={top_rho:.4f}, top-{top_n} overlap={overlap}/{top_n}"
        )
        rows.append((label, full_rho, top_rho, overlap))

    output_path.write_text(_build_sensitivity_tex(rows, top_n))
    logger.info(f"LaTeX sensitivity table written to {output_path}")


def main(experiments: Sequence[str], top_n: int = 10) -> None:
    log_path = OUTPUT_DIR / "auc_axis_comparison.log"
    csv_path = OUTPUT_DIR / "auc_axis_comparison.csv"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_path, level="INFO", mode="w")

    all_per_model: list[pl.DataFrame] = []
    all_summaries: list[dict] = []
    master_curves: dict[str, pl.DataFrame] = {}
    for experiment in experiments:
        logger.info(
            f"Computing alpha curve for experiment={experiment} on master grid "
            f"({len(MASTER_GRID)} log-spaced points in [1, 10])..."
        )
        curve = _compute_curve(experiment, thresholds=MASTER_GRID)
        master_curves[experiment] = curve

        # The original log-vs-linear summary, restricted to the main d_max range.
        main_curve = curve.filter(pl.col("threshold") <= MAIN_CONFIG["d_max"])
        result = _summarise(main_curve, experiment, top_n=top_n)
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

    tex_path = OUTPUT_DIR / "auc_sensitivity_table.tex"
    _write_sensitivity_table(master_curves, top_n=top_n, output_path=tex_path)

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
