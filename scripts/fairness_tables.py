"""Fairness LaTeX tables (reviewer Q4).

Two tables, written separately:

- ``output/fairness_triplet_counts.tex`` -- retained triplet counts per
  (dataset, demographic group, distance-ratio threshold). Re-uses the
  vectorised triplet-agreement helpers from ``alpha_distance_plot.py`` so
  no embeddings are recomputed. The counts are summed across all
  between-rater pairs.
- ``output/fairness_group_gap_controlled.tex`` -- best-worst demographic
  group gap for the highest-mean-AUC model per dataset, reported both
  unadjusted and after logistic-regression adjustment for source /
  closer / farther statement length (log token count) and source-closer
  / source-farther Jaccard token overlap.

  **Unadjusted.** ``Delta_group = max - min`` AUC across demographics
  per (model, dataset, iteration), read from
  ``output/alignment_results_policy.csv`` which is written by
  ``neural_alignment_plots.py``. With the default
  ``hierarchical=True`` in ``compute_threshold_auc``
  (``src/human_grounding/threshold_auc.py``), each per-(model, dataset,
  demographic, iteration) AUC row is one draw from a two-level
  bootstrap (rater resampling first, triplet resampling within rater),
  so the per-iteration gaps inherit that hierarchical resampling.

  **Adjusted.** Reads the per-triplet parquet
  ``data/raw_triplets_policy.parquet`` written by
  ``neural_alignment_plots.py``, fits a logistic regression of triplet
  correctness on group dummies plus the controls, and reports the
  max-minus-min predicted ``P(correct)`` across groups at the in-sample
  mean of the controls. Each replicate resamples raters with
  replacement and refits. Groups are defined by the demographic of each
  triplet's source (anchor) statement so rows are independent within
  rater. Only the best model per dataset is fit -- running the
  regression for every model is prohibitively slow.

  Re-run ``neural_alignment_plots.py`` (cold cache) to refresh the
  per-iteration AUC CSV and the raw-triplets parquet that this table
  consumes.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from human_grounding.data import (
    get_all_statements,
    get_govai,
    get_rai_demographics,
    get_welfare_demographics,
)
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

# Sibling-script imports (scripts/ is not a package)
sys.path.append(str(Path(__file__).parent))
from alpha_distance_plot import (
    compute_distances,
    compute_triplet_agreement_vectorized_with_demographics,
    find_common_indices,
    get_rater_pairs_by_dataset,
    load_coordinates,
)

D_VALUES_COUNTS = (1.0, 2.0, 4.0)
DATASET_PRETTY = {"rai": "Responsible AI", "welfare": "Welfare"}
RAI_DEMO_PRETTY = {"Mand": "Men", "Kvinde": "Women"}
EXCLUDE_DEMOGRAPHICS = {"Unknown", "unknown", "0"}
COORDS_FILE = "combined_coordinates.csv"
# Per-experiment alignment CSV written by ``neural_alignment_plots.py``;
# contains the bootstrap iterations under the (now hierarchical) AUC pipeline.
ALIGNMENT_CSV = "alignment_results_policy.csv"
# Per-triplet parquet written by ``neural_alignment_plots.py`` (DATA_DIR).
RAW_TRIPLETS_FILE = "raw_triplets_policy.parquet"

CONTROL_COLS: tuple[str, ...] = (
    "log_len_src",
    "log_len_close",
    "log_len_far",
    "overlap_close",
    "overlap_far",
)


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

    header_cells = " & ".join(rf"$\mathbf{{\tau={d:g}}}$" for d in d_values)
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


def _build_gap_tex(rows: list[tuple[str, str, float, float, float, float]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Hierarchical-bootstrap tests for best--worst group gaps in "
        r"model-grounding AUC. Each replicate resamples raters with "
        r"replacement (per dataset), then resamples triplets within each "
        r"sampled rater; the same rater resampling is held fixed across "
        r"all $\tau$-thresholds within a replicate.}",
        r"\label{tab:group-gap-tests}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Model} & $\Delta_{\mathrm{group}}$ & "
        r"\textbf{95\% CI} & \textbf{Bootstrap $p$} \\",
        r"\midrule",
    ]
    for ds_pretty, model_label, mean, lo, hi, p_value in rows:
        p_str = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
        lines.append(
            f"{ds_pretty} & {model_label} & {mean:.3f} & "
            f"[{lo:.3f}, {hi:.3f}] & {p_str} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_group_gap_table(output_path: Path) -> None:
    auc_path = OUTPUT_DIR / ALIGNMENT_CSV
    if not auc_path.exists():
        msg = (
            f"{auc_path} not found. Run neural_alignment_plots.py first "
            "(it writes this CSV in its persistence block) to produce the "
            "per-(model, dataset, demographic, iteration) AUC rows."
        )
        raise FileNotFoundError(msg)

    auc_df = pl.read_csv(auc_path).filter(
        (pl.col("metric") == "binary_auc")
        & ~pl.col("demographics").is_in(EXCLUDE_DEMOGRAPHICS)
        & (pl.col("model") != "Human")
    )
    raw_gaps = _per_iter_gap(auc_df)

    rows: list[tuple[str, str, float, float, float, float]] = []
    for dataset in sorted(raw_gaps["dataset"].unique().to_list()):
        ds_pretty = DATASET_PRETTY.get(dataset, dataset)

        # Best model: model with highest mean AUC over (demographic, iteration)
        best_model = (
            auc_df.filter(pl.col("dataset") == dataset)
            .group_by("model")
            .agg(pl.col("auc").mean().alias("mean_auc"))
            .sort("mean_auc", descending=True)["model"]
            .head(1)
            .item()
        )
        best_gaps = raw_gaps.filter(
            (pl.col("dataset") == dataset) & (pl.col("model") == best_model)
        )["gap"].to_numpy()
        rows.append(
            (ds_pretty, f"Best model ({best_model})", *_bootstrap_stats(best_gaps))
        )

        # Mean model: per-iteration mean of Delta_group across models
        mean_gaps = (
            raw_gaps.filter(pl.col("dataset") == dataset)
            .group_by("iteration")
            .agg(pl.col("gap").mean().alias("gap"))["gap"]
            .to_numpy()
        )
        rows.append((ds_pretty, "Mean model", *_bootstrap_stats(mean_gaps)))

    output_path.write_text(_build_gap_tex(rows))
    logger.info(f"Group-gap table written to {output_path}")


# ---------------------------------------------------------------------------
# Table 3: controlled best-worst group gaps (lexical overlap + length)
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _statement_token_features() -> tuple[dict[int, frozenset[str]], dict[int, float]]:
    """Map statement_id -> (token set, log(1 + token count))."""
    welfare_rai = get_all_statements().select("statement_id", "cause")
    govai = get_govai().select(
        pl.col("cause_id").alias("statement_id"), pl.col("cause")
    )
    texts = pl.concat([welfare_rai, govai], how="vertical_relaxed").unique(
        subset=["statement_id"], keep="first"
    )
    logger.info(f"Tokenising {texts.height} unique statements for lexical features")

    tokens: dict[int, frozenset[str]] = {}
    log_lens: dict[int, float] = {}
    for sid, text in zip(
        texts["statement_id"].to_list(), texts["cause"].to_list(), strict=False
    ):
        words = _WORD_RE.findall(str(text).lower())
        sid_int = int(sid)
        tokens[sid_int] = frozenset(words)
        log_lens[sid_int] = float(np.log1p(len(words)))
    return tokens, log_lens


def _add_lexical_features(triplets: pl.DataFrame) -> pl.DataFrame:
    """Attach log-length and Jaccard-overlap features per triplet.

    Features depend only on the (source, closer, farther) statement IDs,
    so we deduplicate pairs before the Python-side Jaccard loop and join
    the results back. This avoids recomputing the same overlap once per
    (model, user) replica of the same triplet.
    """
    tokens, log_lens = _statement_token_features()
    empty: frozenset = frozenset()

    log_len_df = pl.DataFrame(
        {
            "statement_id": list(log_lens.keys()),
            "log_len": list(log_lens.values()),
        }
    )

    close_pairs = triplets.select(
        pl.col("source_idx").alias("a"),
        pl.col("closer_idx").alias("b"),
    )
    far_pairs = triplets.select(
        pl.col("source_idx").alias("a"),
        pl.col("farther_idx").alias("b"),
    )
    unique_pairs = pl.concat([close_pairs, far_pairs]).unique()

    logger.info(
        f"Computing Jaccard overlap for {unique_pairs.height:,} unique pairs "
        f"(from {triplets.height:,} triplet rows)"
    )

    a_arr = unique_pairs["a"].to_numpy()
    b_arr = unique_pairs["b"].to_numpy()
    jacc = np.zeros(unique_pairs.height)
    for i in tqdm(range(unique_pairs.height), desc="Jaccard pairs", unit="pair"):
        ta = tokens.get(int(a_arr[i]), empty)
        tb = tokens.get(int(b_arr[i]), empty)
        u = ta | tb
        jacc[i] = len(ta & tb) / len(u) if u else 0.0

    overlap_df = unique_pairs.with_columns(pl.Series("overlap", jacc))

    logger.info("Joining lexical features back onto triplets")
    return (
        triplets.join(
            log_len_df.rename({"statement_id": "source_idx", "log_len": "log_len_src"}),
            on="source_idx",
            how="left",
        )
        .join(
            log_len_df.rename(
                {"statement_id": "closer_idx", "log_len": "log_len_close"}
            ),
            on="closer_idx",
            how="left",
        )
        .join(
            log_len_df.rename(
                {"statement_id": "farther_idx", "log_len": "log_len_far"}
            ),
            on="farther_idx",
            how="left",
        )
        .join(
            overlap_df.rename(
                {"a": "source_idx", "b": "closer_idx", "overlap": "overlap_close"}
            ),
            on=["source_idx", "closer_idx"],
            how="left",
        )
        .join(
            overlap_df.rename(
                {"a": "source_idx", "b": "farther_idx", "overlap": "overlap_far"}
            ),
            on=["source_idx", "farther_idx"],
            how="left",
        )
        .with_columns(
            pl.col("log_len_src").fill_null(0.0),
            pl.col("log_len_close").fill_null(0.0),
            pl.col("log_len_far").fill_null(0.0),
            pl.col("overlap_close").fill_null(0.0),
            pl.col("overlap_far").fill_null(0.0),
        )
    )


def _attach_source_demographic(triplets: pl.DataFrame) -> pl.DataFrame:
    """Re-join the source statement's demographic so it matches Table 2's pipeline.

    The parquet's own ``demographic`` column comes from each dataset's
    statement reader (e.g. ``Panel`` for RAI), which differs from the
    gender mapping used by Table 2. We drop that column and rejoin using
    the same demographic frames as ``neural_alignment_plots.py``.
    """
    rai_demo = (
        get_rai_demographics(demographics="gender")
        .select("cause_id", "demographic")
        .rename({"cause_id": "source_idx", "demographic": "source_demographic"})
    )
    welfare_demo = (
        get_welfare_demographics()
        .select("cause_id", "demographic")
        .rename({"cause_id": "source_idx", "demographic": "source_demographic"})
    )
    demos = pl.concat([rai_demo, welfare_demo], how="vertical_relaxed").unique(
        subset=["source_idx"], keep="first"
    )
    out = triplets
    if "demographic" in out.columns:
        out = out.drop("demographic")
    return out.join(demos, on="source_idx", how="left")


def _load_triplets_with_features(experiment: str = "policy") -> pl.DataFrame:
    triplets_path = DATA_DIR / f"raw_triplets_{experiment}.parquet"
    if not triplets_path.exists():
        msg = (
            f"{triplets_path} not found. Run neural_alignment_plots.py with a "
            "cold cache (or delete the file) so it persists the per-triplet "
            "parquet used by the controlled fairness analysis."
        )
        raise FileNotFoundError(msg)
    logger.info(f"Loading raw triplets parquet from {triplets_path}")
    triplets = pl.read_parquet(triplets_path)
    logger.info(f"Loaded {triplets.height:,} triplet rows; attaching demographics")
    triplets = _attach_source_demographic(triplets)
    return _add_lexical_features(triplets)


def _fit_controlled_group_gap(sample: pl.DataFrame) -> float | None:
    """Fit a logistic regression with controls and return max-min P(correct).

    Demographics are one-hot encoded with the first level as reference.
    Predictions are made at the in-sample mean of the control features so
    only the demographic dummies vary, isolating the group effect net of
    lexical overlap and statement length.
    """
    demos = sorted(sample["source_demographic"].unique().to_list())
    if len(demos) < 2:
        return None
    y = sample["embedding_correct"].cast(pl.Int8).to_numpy()
    if np.unique(y).size < 2:
        return None

    demo_idx = {d: i for i, d in enumerate(demos)}
    demo_int = np.fromiter(
        (demo_idx[d] for d in sample["source_demographic"].to_list()),
        dtype=np.int64,
        count=sample.height,
    )
    k = len(demos)
    n = sample.height
    X_demo = np.zeros((n, k - 1))
    for i in range(1, k):
        X_demo[:, i - 1] = (demo_int == i).astype(np.float64)

    X_feat = sample.select(CONTROL_COLS).to_numpy()
    X = np.hstack([X_demo, X_feat])

    lr = LogisticRegression(C=1e6, max_iter=500, solver="lbfgs")
    try:
        lr.fit(X, y)
    except (ValueError, RuntimeError):
        return None

    feat_means = X_feat.mean(axis=0)
    preds = np.empty(k)
    for i in range(k):
        demo_row = np.zeros(k - 1)
        if i > 0:
            demo_row[i - 1] = 1.0
        x_pred = np.concatenate([demo_row, feat_means]).reshape(1, -1)
        preds[i] = lr.predict_proba(x_pred)[0, 1]
    return float(preds.max() - preds.min())


def _bootstrap_controlled_gaps(
    triplets: pl.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pl.DataFrame:
    """Rater-resampling bootstrap of the controlled best-worst group gap."""
    triplets = triplets.filter(
        pl.col("source_demographic").is_not_null()
        & ~pl.col("source_demographic").is_in(EXCLUDE_DEMOGRAPHICS)
    )

    combos: list[tuple[tuple[str, str], pl.DataFrame]] = [
        ((str(model_name), str(dataset)), g)
        for (model_name, dataset), g in triplets.group_by(["model", "dataset"])
    ]
    total_fits = len(combos) * n_bootstrap
    logger.info(
        f"Bootstrapping controlled gap: {len(combos)} (model, dataset) combos "
        f"x {n_bootstrap} iterations = {total_fits:,} regression fits"
    )

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    skipped = 0

    with tqdm(total=total_fits, desc="Bootstrap fits", unit="fit") as pbar:
        for (model_name, dataset), g in combos:
            users = g["user_id"].unique().to_list()
            n_users = len(users)
            if n_users < 2:
                pbar.update(n_bootstrap)
                skipped += n_bootstrap
                continue
            user_rows = {str(u): gu for (u,), gu in g.group_by("user_id")}
            pbar.set_postfix_str(f"{model_name}/{dataset} n={g.height:,}")
            for it in range(n_bootstrap):
                sampled_idx = rng.integers(0, n_users, size=n_users)
                parts = [user_rows[str(users[j])] for j in sampled_idx]
                sample = pl.concat(parts)
                gap = _fit_controlled_group_gap(sample)
                pbar.update(1)
                if gap is None:
                    skipped += 1
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "dataset": dataset,
                        "iteration": it,
                        "gap_adj": gap,
                    }
                )

    if skipped:
        logger.warning(f"Skipped {skipped:,} fits (single-class y or <2 raters)")
    logger.info(f"Bootstrap produced {len(rows):,} controlled-gap samples")
    return pl.DataFrame(rows)


def _fmt_p(p_value: float) -> str:
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def _build_controlled_gap_tex(
    rows: list[
        tuple[
            str,  # ds_pretty
            str,  # model_label
            tuple[float, float, float, float],  # unadjusted (mean, lo, hi, p)
            tuple[float, float, float, float],  # adjusted   (mean, lo, hi, p)
        ]
    ],
) -> str:
    # Six-column layout (Dataset, Model, then Delta+CI and p, twice). Delta
    # and its CI share a cell so the table stays within column width.
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Best--worst demographic group gaps in model-grounding "
        r"accuracy for the highest-mean-AUC model per dataset, before and "
        r"after adjustment for lexical and length confounds. See main text "
        r"for definitions of $\Delta_{\mathrm{group}}$, "
        r"$\Delta_{\mathrm{adj}}$, and the bootstrap procedure.}",
        r"\label{tab:group-gap-controlled}",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Unadjusted}} & "
        r"\multicolumn{2}{c}{\textbf{Adjusted}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
        r"\textbf{Dataset} & \textbf{Model} & "
        r"$\Delta_{\mathrm{group}}$ [95\% CI] & $p$ & "
        r"$\Delta_{\mathrm{adj}}$ [95\% CI] & $p$ \\",
        r"\midrule",
    ]
    for ds_pretty, model_label, unadj, adj in rows:
        u_mean, u_lo, u_hi, u_p = unadj
        a_mean, a_lo, a_hi, a_p = adj
        lines.append(
            f"{ds_pretty} & {model_label} & "
            f"{u_mean:.3f} [{u_lo:.3f},\\,{u_hi:.3f}] & {_fmt_p(u_p)} & "
            f"{a_mean:.3f} [{a_lo:.3f},\\,{a_hi:.3f}] & {_fmt_p(a_p)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def write_controlled_group_gap_table(
    output_path: Path,
    experiment: str = "policy",
    n_bootstrap: int = 50,
    seed: int = 0,
) -> None:
    logger.info(
        f"Building controlled group-gap table (experiment={experiment}, "
        f"n_bootstrap={n_bootstrap}, seed={seed})"
    )

    auc_path = OUTPUT_DIR / ALIGNMENT_CSV
    if not auc_path.exists():
        msg = (
            f"{auc_path} not found. Run neural_alignment_plots.py first so "
            "the best-model lookup matches the unadjusted gap table."
        )
        raise FileNotFoundError(msg)
    auc_df = pl.read_csv(auc_path).filter(
        (pl.col("metric") == "binary_auc")
        & ~pl.col("demographics").is_in(EXCLUDE_DEMOGRAPHICS)
        & (pl.col("model") != "Human")
    )

    triplets_path = DATA_DIR / f"raw_triplets_{experiment}.parquet"
    if not triplets_path.exists():
        msg = (
            f"{triplets_path} not found. Run neural_alignment_plots.py with a "
            "cold cache (or delete the file) so it persists the per-triplet "
            "parquet used by the controlled fairness analysis."
        )
        raise FileNotFoundError(msg)
    logger.info(f"Loading raw triplets parquet from {triplets_path}")
    raw = pl.read_parquet(triplets_path)

    # Pick the best (highest mean AUC) model per dataset and only fit the
    # bootstrap regression for those — running it for every model is too slow
    # and the table only reports the best-model row anyway.
    best_per_dataset: dict[str, str] = {}
    for dataset in sorted(raw["dataset"].unique().to_list()):
        ds_aucs = auc_df.filter(pl.col("dataset") == dataset)
        if ds_aucs.is_empty():
            logger.warning(
                f"No AUC rows for dataset {dataset}; skipping controlled fit"
            )
            continue
        best_per_dataset[dataset] = (
            ds_aucs.group_by("model")
            .agg(pl.col("auc").mean().alias("mean_auc"))
            .sort("mean_auc", descending=True)["model"]
            .head(1)
            .item()
        )
    logger.info(f"Best models per dataset: {best_per_dataset}")

    keep = pl.DataFrame(
        [{"model": m, "dataset": ds} for ds, m in best_per_dataset.items()]
    )
    filtered = raw.join(keep, on=["model", "dataset"], how="inner")
    logger.info(
        f"Filtered raw triplets to best models: {filtered.height:,} of "
        f"{raw.height:,} rows kept"
    )

    triplets = _attach_source_demographic(filtered)
    triplets = _add_lexical_features(triplets)

    controlled = _bootstrap_controlled_gaps(
        triplets, n_bootstrap=n_bootstrap, seed=seed
    )
    if controlled.is_empty():
        msg = "Controlled-gap bootstrap produced no rows; aborting table write."
        raise RuntimeError(msg)

    # Unadjusted Δ_group draws come from the existing AUC bootstrap rows
    # in alignment_results_policy.csv (same source as the standalone Δ_group
    # table), aggregated per (model, dataset, iteration).
    unadj_per_iter = _per_iter_gap(auc_df)

    BootstrapStats = tuple[float, float, float, float]
    rows: list[tuple[str, str, BootstrapStats, BootstrapStats]] = []
    for dataset in sorted(best_per_dataset.keys()):
        best_model = best_per_dataset[dataset]
        ds_pretty = DATASET_PRETTY.get(dataset, dataset)

        unadj_gaps = unadj_per_iter.filter(
            (pl.col("dataset") == dataset) & (pl.col("model") == best_model)
        )["gap"].to_numpy()
        adj_gaps = controlled.filter(
            (pl.col("dataset") == dataset) & (pl.col("model") == best_model)
        )["gap_adj"].to_numpy()

        rows.append(
            (
                ds_pretty,
                best_model,
                _bootstrap_stats(unadj_gaps),
                _bootstrap_stats(adj_gaps),
            )
        )

    output_path.write_text(_build_controlled_gap_tex(rows))
    logger.info(f"Combined group-gap table written to {output_path}")


def main(
    counts_path: Path | None = None,
    controlled_gap_path: Path | None = None,
    controlled_bootstrap: int = 50,
    controlled_seed: int = 0,
) -> None:
    if counts_path is None:
        counts_path = OUTPUT_DIR / "fairness_triplet_counts.tex"
    if controlled_gap_path is None:
        controlled_gap_path = OUTPUT_DIR / "fairness_group_gap_controlled.tex"

    write_triplet_counts_table(counts_path)
    write_controlled_group_gap_table(
        controlled_gap_path,
        n_bootstrap=controlled_bootstrap,
        seed=controlled_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-path", type=Path, default=None)
    parser.add_argument("--controlled-gap-path", type=Path, default=None)
    parser.add_argument("--controlled-bootstrap", type=int, default=50)
    parser.add_argument("--controlled-seed", type=int, default=0)
    args = parser.parse_args()
    main(
        counts_path=args.counts_path,
        controlled_gap_path=args.controlled_gap_path,
        controlled_bootstrap=args.controlled_bootstrap,
        controlled_seed=args.controlled_seed,
    )
