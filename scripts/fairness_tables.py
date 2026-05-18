"""Fairness LaTeX tables (reviewer Q4).

Two tables, written separately:

- ``output/fairness_triplet_counts.tex`` -- retained triplet counts per
  (dataset, demographic group, distance-ratio threshold). Re-uses the
  vectorised triplet-agreement helpers from ``alpha_distance_plot.py`` so
  no embeddings are recomputed. The counts are summed across all
  between-rater pairs.
- ``output/fairness_group_gap_tests.tex`` -- best-worst group gap
  Delta_group with bootstrap 95 % CI and bootstrap p-value (one-sided
  test for Delta_group > 0). Reads the already-on-disk
  ``output/embedding_alignment_auc.csv`` so no AUC bootstrap is repeated.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from human_grounding.data import (
    get_all_statements,
    get_rai_demographics,
    get_welfare_demographics,
)
from human_grounding.directories import DATA_DIR, OUTPUT_DIR
from human_grounding.embed import get_all_models

# Sibling-script imports (scripts/ is not a package)
sys.path.append(str(Path(__file__).parent))
from alpha_distance_plot import (
    compute_distances,
    compute_triplet_agreement_vectorized_with_demographics,
    find_common_indices,
    get_rater_pairs_by_dataset,
    load_coordinates,
)
from neural_alignment_plots import get_embedding_alignments

NEURAL_COORDS_FILE = "valid_coordinates.csv"
N_QUARTILES = 4
N_BOOTSTRAP_CONTROLLED = 200
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

D_VALUES_COUNTS = (1.0, 2.0, 4.0)
DATASET_PRETTY = {"rai": "Responsible AI", "welfare": "Welfare"}
RAI_DEMO_PRETTY = {"Mand": "Men", "Kvinde": "Women"}
EXCLUDE_DEMOGRAPHICS = {"Unknown", "unknown", "0"}
COORDS_FILE = "combined_coordinates.csv"
ALIGNMENT_CSV = "embedding_alignment_auc.csv"


# ---------------------------------------------------------------------------
# Table 1: retained triplet counts
# ---------------------------------------------------------------------------


def _load_coordinates_with_demographics() -> pl.DataFrame:
    coords = load_coordinates(DATA_DIR / COORDS_FILE)
    rai_demo = (
        get_rai_demographics(demographics="gender")
        .select("cause_id", "demographic")
        .rename({"cause_id": "statement_id"})
    )
    welfare_demo = (
        get_welfare_demographics()
        .select("cause_id", "demographic")
        .rename({"cause_id": "statement_id"})
    )
    demos = pl.concat([rai_demo, welfare_demo], how="vertical_relaxed").unique(
        subset=["statement_id"], keep="first"
    )
    return coords.join(demos, on="statement_id", how="left")


def _retained_counts(
    distances: dict,
    between_pairs: dict[str, list],
    d_values: tuple[float, ...],
) -> pl.DataFrame:
    """Sum valid (retained) triplets per (dataset, demographic, d) across rater pairs."""
    rows: dict[tuple[str, str, float], int] = {}
    for dataset, pairs in between_pairs.items():
        for key1, key2 in pairs:
            if key1 not in distances or key2 not in distances:
                continue
            s1 = distances[key1]["statement_ids"]
            s2 = distances[key2]["statement_ids"]
            idx1, idx2 = find_common_indices(s1, s2)
            if len(idx1) < 3:
                continue
            sub1 = distances[key1]["dist_matrix"][np.ix_(idx1, idx1)]
            sub2 = distances[key2]["dist_matrix"][np.ix_(idx2, idx2)]
            demos1 = distances[key1]["demographics"][idx1]
            demos2 = distances[key2]["demographics"][idx2]

            for d in d_values:
                _, _, demo_stats = (
                    compute_triplet_agreement_vectorized_with_demographics(
                        sub1, sub2, demos1, demos2, d
                    )
                )
                for demo, (_agree, valid) in demo_stats.items():
                    if demo in EXCLUDE_DEMOGRAPHICS:
                        continue
                    key = (dataset, str(demo), float(d))
                    rows[key] = rows.get(key, 0) + int(valid)

    return pl.DataFrame(
        [
            {"dataset": ds, "demographic": demo, "d": d, "count": n}
            for (ds, demo, d), n in rows.items()
        ]
    )


def _pretty_demo(dataset: str, demo: str) -> str:
    if dataset == "rai":
        return RAI_DEMO_PRETTY.get(demo, demo)
    if dataset == "welfare":
        return f"Party {demo}"
    return demo


def _build_counts_tex(counts: pl.DataFrame, d_values: tuple[float, ...]) -> str:
    pivoted = counts.pivot(on="d", index=["dataset", "demographic"], values="count")
    pivoted = pivoted.with_columns(
        pl.col("dataset").replace(DATASET_PRETTY).alias("dataset_pretty")
    )
    pivoted = pivoted.sort(
        by=["dataset_pretty", "demographic"], descending=[False, False]
    )

    header_cells = " & ".join(rf"$\mathbf{{d={d:g}}}$" for d in d_values)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Retained triplet counts by dataset, group, and distance-ratio threshold.}",
        r"\label{tab:group-triplet-counts}",
        r"\begin{tabular}{ll" + "r" * len(d_values) + r"}",
        r"\toprule",
        rf"\textbf{{Dataset}} & \textbf{{Group}} & {header_cells} \\",
        r"\midrule",
    ]
    for row in pivoted.iter_rows(named=True):
        ds_pretty = row["dataset_pretty"]
        demo_pretty = _pretty_demo(row["dataset"], row["demographic"])
        count_cells = " & ".join(
            f"{int(row[str(d)]):,}" if row[str(d)] is not None else "--"
            for d in d_values
        )
        lines.append(f"{ds_pretty} & {demo_pretty} & {count_cells} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_triplet_counts_table(output_path: Path) -> None:
    coords = _load_coordinates_with_demographics()
    distances = compute_distances(coords)
    _, between_pairs = get_rater_pairs_by_dataset(distances)
    counts = _retained_counts(distances, between_pairs, D_VALUES_COUNTS)
    output_path.write_text(_build_counts_tex(counts, D_VALUES_COUNTS))
    logger.info(f"Triplet-count table written to {output_path}")


# ---------------------------------------------------------------------------
# Table 2: bootstrap tests for best-worst group gaps
# ---------------------------------------------------------------------------


def _per_iter_gap(auc_df: pl.DataFrame) -> pl.DataFrame:
    """Per (model, dataset, iteration): max(auc) - min(auc) across demographics."""
    return auc_df.group_by("model", "dataset", "iteration").agg(
        (pl.col("auc").max() - pl.col("auc").min()).alias("gap")
    )


def _bootstrap_stats(values: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, ci_lo, ci_hi, p_one_sided) from a bootstrap sample of gaps.

    ``p`` is the bootstrap one-sided p-value for H0: gap <= 0.
    """
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (float("nan"),) * 4
    mean = float(np.mean(values))
    lo = float(np.percentile(values, 2.5))
    hi = float(np.percentile(values, 97.5))
    p_value = float(np.mean(values <= 0))
    return mean, lo, hi, p_value


def _build_gap_tex(
    rows: list[tuple[str, str, dict[str, tuple[float, float, float, float]]]],
) -> str:
    """Render the raw vs controlled gap table.

    Each input row is ``(dataset_pretty, model_label, stats)`` where
    ``stats`` is ``{"raw": (mean, lo, hi, p), "controlled": (...)}``.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Bootstrap tests for best--worst group gaps in model-grounding "
        r"score. Raw uses the full triplet set; the controlled column "
        r"stratifies each triplet by mean statement-length quartile and "
        r"anchor-to-comparison Jaccard-overlap quartile, then averages across "
        r"strata.}",
        r"\label{tab:group-gap-tests}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} "
        r"& $\Delta_{\mathrm{group}}$ (raw) "
        r"& $\Delta_{\mathrm{group}}$ (ctrl.) "
        r"& \textbf{95\% CI (ctrl.)} "
        r"& \textbf{Bootstrap $p$ (ctrl.)} \\",
        r"\midrule",
    ]
    for ds_pretty, model_label, stats in rows:
        raw_mean = stats["raw"][0]
        ctrl_mean, ctrl_lo, ctrl_hi, ctrl_p = stats["controlled"]
        p_str = "<0.001" if ctrl_p < 0.001 else f"{ctrl_p:.3f}"
        lines.append(
            f"{ds_pretty} & {model_label} & "
            f"{raw_mean:.3f} & {ctrl_mean:.3f} & "
            f"[{ctrl_lo:.3f}, {ctrl_hi:.3f}] & {p_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Controlled gap: per-triplet stratification by length quartile and Jaccard quartile
# ---------------------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _prepare_controlled_rows() -> pl.DataFrame:
    """Per-triplet df with stratification columns and source-statement demographic."""
    full_dataset = pl.read_csv(DATA_DIR / NEURAL_COORDS_FILE)
    models = sorted(get_all_models())
    combined = get_embedding_alignments(models, full_dataset, use_english=False)

    statements = get_all_statements()
    info: dict[int, dict] = {}
    for row in statements.iter_rows(named=True):
        text = row["cause"] or ""
        info[int(row["statement_id"])] = {
            "length": len(text),
            "tokens": set(_TOKEN_RE.findall(text.lower())),
        }

    src = combined["source_idx"].to_numpy()
    cls = combined["closer_idx"].to_numpy()
    far = combined["farther_idx"].to_numpy()

    def _length(arr: np.ndarray) -> np.ndarray:
        return np.array([info[int(i)]["length"] for i in arr], dtype=float)

    length_mean = (_length(src) + _length(cls) + _length(far)) / 3.0

    toks_src = [info[int(i)]["tokens"] for i in src]
    toks_cls = [info[int(i)]["tokens"] for i in cls]
    toks_far = [info[int(i)]["tokens"] for i in far]
    jaccard = np.array(
        [
            (_jaccard(s, c) + _jaccard(s, f)) / 2.0
            for s, c, f in zip(toks_src, toks_cls, toks_far)
        ]
    )

    df = combined.with_columns(
        pl.Series("length_mean", length_mean),
        pl.Series("jaccard", jaccard),
    ).with_columns(
        pl.col("length_mean")
        .qcut(N_QUARTILES, labels=[str(i) for i in range(N_QUARTILES)])
        .alias("length_q"),
        pl.col("jaccard")
        .qcut(N_QUARTILES, labels=[str(i) for i in range(N_QUARTILES)])
        .alias("jaccard_q"),
    )

    rai_demo = (
        get_rai_demographics(demographics="gender")
        .select("cause_id", "demographic")
        .unique(subset=["cause_id"], keep="first")
    )
    welfare_demo = (
        get_welfare_demographics()
        .select("cause_id", "demographic")
        .unique(subset=["cause_id"], keep="first")
    )
    demos = pl.concat([rai_demo, welfare_demo], how="vertical_relaxed").unique(
        subset=["cause_id"], keep="first"
    )

    df = (
        df.drop("demographic", strict=False)
        .join(demos, left_on="source_idx", right_on="cause_id", how="left")
        .filter(
            pl.col("demographic").is_not_null()
            & ~pl.col("demographic").is_in(EXCLUDE_DEMOGRAPHICS)
        )
    )
    return df.select(
        "model",
        "dataset",
        "demographic",
        "length_q",
        "jaccard_q",
        "embedding_correct",
    )


def _controlled_gap_per_model(rows: pl.DataFrame) -> pl.DataFrame:
    """Per (model, dataset): stratified Delta_group = max - min over demographics."""
    cell = rows.group_by(
        ["model", "dataset", "demographic", "length_q", "jaccard_q"]
    ).agg(pl.col("embedding_correct").mean().alias("acc"))
    per_md = cell.group_by(["model", "dataset", "demographic"]).agg(
        pl.col("acc").mean().alias("acc")
    )
    return (
        per_md.with_columns((2 * pl.col("acc") - 1).alias("score"))
        .group_by(["model", "dataset"])
        .agg((pl.col("score").max() - pl.col("score").min()).alias("gap"))
    )


def _bootstrap_controlled_gaps(
    rows: pl.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP_CONTROLLED,
    seed: int = 0,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n = rows.height
    iters: list[pl.DataFrame] = []
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = rows[idx]
        iters.append(
            _controlled_gap_per_model(sample).with_columns(pl.lit(b).alias("iteration"))
        )
    return pl.concat(iters)


def write_group_gap_table(output_path: Path) -> None:
    auc_path = OUTPUT_DIR / ALIGNMENT_CSV
    if not auc_path.exists():
        msg = (
            f"{auc_path} not found. Run neural_alignment_plots.py first to "
            "produce the per-(model, dataset, demographic, iteration) AUC CSV."
        )
        raise FileNotFoundError(msg)

    # ---- Raw gap from existing per-iteration AUC CSV ----
    auc_df = pl.read_csv(auc_path).filter(
        ~pl.col("demographics").is_in(EXCLUDE_DEMOGRAPHICS)
        & (pl.col("model") != "Human")
    )
    raw_gaps = _per_iter_gap(auc_df)

    # Best model identity per dataset is fixed by mean AUC on the raw pool.
    best_model_by_dataset: dict[str, str] = {}
    for dataset in sorted(raw_gaps["dataset"].unique().to_list()):
        best_model_by_dataset[dataset] = (
            auc_df.filter(pl.col("dataset") == dataset)
            .group_by("model")
            .agg(pl.col("auc").mean().alias("mean_auc"))
            .sort("mean_auc", descending=True)["model"]
            .head(1)
            .item()
        )

    # ---- Controlled (length+Jaccard stratified) gap from per-triplet data ----
    logger.info("Preparing per-triplet rows for controlled-gap bootstrap...")
    triplet_rows = _prepare_controlled_rows()
    logger.info(
        f"Running {N_BOOTSTRAP_CONTROLLED}-iteration controlled-gap bootstrap "
        f"on {triplet_rows.height:,} triplet rows..."
    )
    ctrl_gaps = _bootstrap_controlled_gaps(triplet_rows)

    rows: list[tuple[str, str, dict[str, tuple[float, float, float, float]]]] = []
    for dataset in sorted(raw_gaps["dataset"].unique().to_list()):
        ds_pretty = DATASET_PRETTY.get(dataset, dataset)
        best_model = best_model_by_dataset[dataset]

        raw_best = raw_gaps.filter(
            (pl.col("dataset") == dataset) & (pl.col("model") == best_model)
        )["gap"].to_numpy()
        raw_mean_per_iter = (
            raw_gaps.filter(pl.col("dataset") == dataset)
            .group_by("iteration")
            .agg(pl.col("gap").mean().alias("gap"))["gap"]
            .to_numpy()
        )

        ctrl_best = ctrl_gaps.filter(
            (pl.col("dataset") == dataset) & (pl.col("model") == best_model)
        )["gap"].to_numpy()
        ctrl_mean_per_iter = (
            ctrl_gaps.filter(pl.col("dataset") == dataset)
            .group_by("iteration")
            .agg(pl.col("gap").mean().alias("gap"))["gap"]
            .to_numpy()
        )

        rows.append(
            (
                ds_pretty,
                f"Best model ({best_model})",
                {
                    "raw": _bootstrap_stats(raw_best),
                    "controlled": _bootstrap_stats(ctrl_best),
                },
            )
        )
        rows.append(
            (
                ds_pretty,
                "Mean model",
                {
                    "raw": _bootstrap_stats(raw_mean_per_iter),
                    "controlled": _bootstrap_stats(ctrl_mean_per_iter),
                },
            )
        )

    output_path.write_text(_build_gap_tex(rows))
    logger.info(f"Group-gap table written to {output_path}")


def main(
    counts_path: Path | None = None,
    gap_path: Path | None = None,
) -> None:
    if counts_path is None:
        counts_path = OUTPUT_DIR / "fairness_triplet_counts.tex"
    if gap_path is None:
        gap_path = OUTPUT_DIR / "fairness_group_gap_tests.tex"

    write_triplet_counts_table(counts_path)
    write_group_gap_table(gap_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-path", type=Path, default=None)
    parser.add_argument("--gap-path", type=Path, default=None)
    args = parser.parse_args()
    main(counts_path=args.counts_path, gap_path=args.gap_path)
