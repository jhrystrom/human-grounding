# UPDATED SCRIPT WITH SPEARMAN SUPPORT

import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

import human_grounding.evaluate
import human_grounding.threshold_auc
from human_grounding.constants import DATASET_PRETTY_NAMES, PRETTY_NAMES
from human_grounding.data import (
    get_rai_demographics,
    get_welfare_demographics,
)
from human_grounding.directories import DATA_DIR, OUTPUT_DIR, PLOT_DIR
from human_grounding.embed import get_all_models
from human_grounding.names import append_english

TOP_N_TO_PLOT = 10
COORDINATES = {
    "policy": "valid_coordinates.csv",
    "gov-ai": "govai_coordinates.csv",
}
EXPERIMENT_DATASETS: dict[str, list[str]] = {
    "policy": ["welfare", "rai"],
    "gov-ai": ["gov-ai"],
}


def get_embedding_alignments(
    models: list[str], full_dataset: pl.DataFrame, use_english: bool = False
) -> pl.DataFrame:
    results = []
    for model in tqdm(models, desc="Models"):
        if model.startswith("wicked"):
            continue
        comparisons = human_grounding.evaluate.evaluate_human_embedding_match(
            model=model, all_coordinates=full_dataset, use_english=use_english
        )
        results.append(comparisons)
    return pl.concat(results, how="vertical_relaxed").drop("cause", "source", "size")


def _get_experiment_name(experiments: list[str]) -> str:
    return "_".join(sorted(experiments))


def _write_dataset_summary(
    all_binary_auc: pl.DataFrame,
    pretty_names: Mapping[str, str],
    out_path: Path,
) -> None:
    """Write per-dataset Human vs best-model alpha-AUC summary, with gap CIs.

    For each dataset: mean and 95% percentile CI across bootstrap iterations
    (Human and the highest-scoring model). The gap CI pairs random iterations
    independently across the two sources, since the Human baseline (alpha
    pipeline) and model AUCs (compute_threshold_auc) bootstrap independently.
    """
    human_name = human_grounding.threshold_auc.HUMAN_MODEL_NAME

    # Collapse across demographics: per-iteration mean per (model, dataset).
    per_iter = all_binary_auc.group_by("model", "dataset", "iteration").agg(
        pl.col("auc").mean().alias("auc"),
    )

    summary = per_iter.group_by("model", "dataset").agg(
        pl.col("auc").mean().alias("mean"),
        pl.col("auc").quantile(0.025).alias("lo"),
        pl.col("auc").quantile(0.975).alias("hi"),
    )

    rng = np.random.default_rng(0)
    n_pairs = 10000
    lines: list[str] = [
        "Dataset summary (binary α-AUC, mean across demographics)",  # noqa: RUF001
        "=" * 70,
        "",
    ]

    for ds in sorted(summary["dataset"].unique().to_list()):
        ds_summary = summary.filter(pl.col("dataset") == ds)
        lines.append(f"## {ds}")

        h_row = ds_summary.filter(pl.col("model") == human_name)
        human_iters: np.ndarray | None = None
        h: dict | None = None
        if h_row.is_empty():
            lines.append("  Human:       (no baseline available)")
        else:
            h = h_row.row(0, named=True)
            lines.append(
                f"  Human:       {h['mean']:.3f}  [{h['lo']:.3f}, {h['hi']:.3f}]",
            )
            human_iters = (
                per_iter.filter(
                    (pl.col("model") == human_name) & (pl.col("dataset") == ds),
                )
                .get_column("auc")
                .to_numpy()
            )

        model_rows = ds_summary.filter(pl.col("model") != human_name).sort(
            "mean",
            descending=True,
        )
        if model_rows.is_empty():
            lines.append("  (no model results for this dataset)")
            lines.append("")
            continue

        best = model_rows.row(0, named=True)
        pretty = pretty_names.get(best["model"], best["model"])
        lines.append(
            f"  Best model:  {pretty}  "
            f"{best['mean']:.3f}  [{best['lo']:.3f}, {best['hi']:.3f}]",
        )

        if h is not None and human_iters is not None:
            model_iters = (
                per_iter.filter(
                    (pl.col("model") == best["model"]) & (pl.col("dataset") == ds),
                )
                .get_column("auc")
                .to_numpy()
            )
            i = rng.integers(0, len(human_iters), n_pairs)
            j = rng.integers(0, len(model_iters), n_pairs)
            gaps = human_iters[i] - model_iters[j]
            gap_point = h["mean"] - best["mean"]
            gap_lo = float(np.percentile(gaps, 2.5))
            gap_hi = float(np.percentile(gaps, 97.5))
            lines.append(
                f"  Gap:         {gap_point * 100:.1f}pp  "
                f"[{gap_lo * 100:.1f}pp, {gap_hi * 100:.1f}pp]",
            )
        lines.append("")

    out_path.write_text("\n".join(lines))
    logger.info(f"Wrote dataset summary to {out_path}")


def plot_mmteb_correlation(
    auc_bootstraps: pl.DataFrame,
    experiment: str,
    use_english: bool = False,
    file_type: str = "pdf",
) -> None:
    """Rank-vs-rank dumbbell of Human-alignment vs MMTEB(Danish) for one experiment."""
    datasets = EXPERIMENT_DATASETS.get(experiment, [experiment])
    exp_bootstraps = auc_bootstraps.filter(pl.col("dataset").is_in(datasets))
    if exp_bootstraps.is_empty():
        logger.warning(f"No AUC rows for experiment {experiment}; skipping MMTEB plot")
        return

    aggregated_bootstraps = exp_bootstraps.group_by("model").agg(
        pl.col("auc").mean().alias("alignment_score")
    )

    alignment_output = OUTPUT_DIR / f"human_alignment_bootstrapped_{experiment}.csv"
    if use_english:
        alignment_output = append_english(alignment_output)
    aggregated_bootstraps.write_csv(alignment_output)

    # Regex for extracting name within brackets
    pattern = r"\[(.*?)\]"
    mmteb_raw = pl.read_csv(OUTPUT_DIR / "mmteb-top-dan.csv")
    # Use the leaderboard's Borda rank, then re-rank within our model subset so
    # the resulting `Rank (MMTEB)` is dense (1..N) over only the models we share.
    mmteb_data = mmteb_raw.with_columns(
        pl.col("Model").str.extract(pattern).alias("model_name"),
    ).select("model_name", "Rank (Borda)")
    mmteb_with_ranks = (
        mmteb_data.drop_nulls()
        .join(
            aggregated_bootstraps.select("model"),
            left_on="model_name",
            right_on="model",
        )
        .sort("Rank (Borda)")  # ascending: lower Borda rank = better
        .with_row_index("Rank (MMTEB)")
    )
    mmteb_with_ranks_path = OUTPUT_DIR / f"mmteb_with_ranks_{experiment}.csv"
    mmteb_with_ranks.write_csv(mmteb_with_ranks_path)

    score_ranks = aggregated_bootstraps.sort(
        "alignment_score", descending=True
    ).with_row_index("rank")

    rank_plot_data = (
        score_ranks.filter(pl.col("rank") < TOP_N_TO_PLOT)
        .join(mmteb_with_ranks, left_on="model", right_on="model_name")
        .drop("alignment_score", "Rank (Borda)")
        .rename({"rank": "Rank (Human)"})
        .unpivot(
            index="model",
            variable_name="Ranking Type",
            value_name="Rank",
        )
        .with_columns(pl.col("model").replace(PRETTY_NAMES))
    )
    if rank_plot_data.is_empty():
        logger.warning(
            f"No overlap between top-{TOP_N_TO_PLOT} human-aligned models and MMTEB "
            f"for experiment {experiment}; skipping plot"
        )
        return

    pdf = rank_plot_data.to_pandas()
    mmteb = pdf[pdf["Ranking Type"] == "Rank (MMTEB)"]
    human = pdf[pdf["Ranking Type"] == "Rank (Human)"]

    # Sort by Human rank (ascending = best first)
    human_sorted = human.sort_values("Rank")
    models = human_sorted["model"].tolist()
    x = np.arange(len(models))

    mmteb = mmteb.set_index("model").loc[models].reset_index()
    human = human.set_index("model").loc[models].reset_index()
    sns.set_theme(style="whitegrid", font_scale=1.9)
    _fig, ax = plt.subplots(figsize=(12, 8))

    for i in range(len(models)):
        ax.plot(
            [x[i], x[i]],
            [mmteb.iloc[i]["Rank"], human.iloc[i]["Rank"]],
            color="gray",
            linewidth=1,
            zorder=1,
        )

    ax.scatter(
        x,
        mmteb["Rank"],
        s=120,
        color="green",
        edgecolor="black",
        label="MMTEB(Danish)",
        zorder=2,
    )
    ax.scatter(
        x,
        human["Rank"],
        s=120,
        color="purple",
        edgecolor="black",
        label="Human Grounded",
        zorder=2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Model Rank")
    ax.set_xlabel("")
    ax.invert_yaxis()
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Score Type")

    plt.tight_layout()
    mmteb_fig_path = PLOT_DIR / f"mmteb_vs_human_ranking_{experiment}.{file_type}"
    if use_english:
        mmteb_fig_path = append_english(mmteb_fig_path)
    plt.savefig(mmteb_fig_path, bbox_inches="tight")
    plt.clf()
    logger.info(f"Wrote MMTEB ranking plot for {experiment} to {mmteb_fig_path}")


def main(
    font_scale: float,
    use_english: bool = False,
    use_cache: bool = False,
    file_type: str = "pdf",
    metric: str = "binary",
    experiments: list[str] = ["policy"],
) -> None:
    combined_results = []
    for exp in experiments:
        full_dataset = pl.read_csv(DATA_DIR / COORDINATES[exp])
        combined_results.append(full_dataset)
    full_dataset = pl.concat(combined_results, how="vertical_relaxed")

    welfare_demographics = (
        get_welfare_demographics() if "policy" in experiments else None
    )
    models = sorted(get_all_models())
    rai_demographics = get_rai_demographics() if "policy" in experiments else None

    experiment_name = _get_experiment_name(experiments)

    out_path = OUTPUT_DIR / f"alignment_results_{experiment_name}.csv"
    diff_csv = OUTPUT_DIR / f"difficulty_split_{experiment_name}.csv"
    if use_english:
        out_path = append_english(out_path)
        diff_csv = append_english(diff_csv)

    # Lazy: run the expensive embedding step only when at least one cache is cold
    _combined: pl.DataFrame | None = None
    _embeddings_computed = False

    def _get_combined() -> pl.DataFrame:
        nonlocal _combined, _embeddings_computed
        if _combined is None:
            _combined = get_embedding_alignments(
                models, full_dataset, use_english=use_english
            )
            _embeddings_computed = True
        return _combined

    def _load_human_aucs() -> pl.DataFrame | None:
        """Load human AUC from alpha files for all experiments that have one."""
        frames = []
        for exp in experiments:
            alpha_path = OUTPUT_DIR / f"alpha_data_{exp}_demographic.csv"
            if alpha_path.exists():
                frames.append(human_grounding.threshold_auc.load_human_auc(alpha_path))
            else:
                logger.info(
                    f"No human baseline for {exp} ({alpha_path.name} not found), skipping"
                )
        return pl.concat(frames, how="vertical_relaxed") if frames else None

    # --- BINARY AUC ---
    all_binary_auc: pl.DataFrame | None = None
    if metric in ["binary", "both"]:
        if use_cache and out_path.exists():
            logger.info(f"Loading cached alignment results from {out_path}")
            auc_bootstraps = (
                pl.read_csv(out_path)
                .filter(pl.col("metric") == "binary_auc")
                .drop("metric")
                .filter(
                    pl.col("model") != human_grounding.threshold_auc.HUMAN_MODEL_NAME
                )
            )
            # Save the gov-ai one separately for the clustering
            auc_bootstraps.filter(pl.col("dataset") == "gov-ai").group_by("model").agg(
                pl.col("auc").mean()
            ).rename({"auc": "alignment_score"}).write_csv(
                OUTPUT_DIR / "human_alignment_bootstrapped_gov-ai.csv"
            )
        else:
            auc_bootstraps, _ = human_grounding.threshold_auc.compute_threshold_auc(
                combined_results=_get_combined(),
                welfare_demographics=welfare_demographics,
                rai_demographics=rai_demographics,
                n_bootstrap=10,
                metric="binary",
            )
        human_auc = _load_human_aucs()
        if human_auc is not None:
            all_binary_auc = pl.concat(
                [auc_bootstraps, human_auc.select(auc_bootstraps.columns)],
                how="vertical_relaxed",
            )
        else:
            all_binary_auc = auc_bootstraps
    else:
        auc_bootstraps = None

    # --- SPEARMAN ---
    spearman_bootstraps: pl.DataFrame | None = None
    all_spearman: pl.DataFrame | None = None
    if metric in ["spearman", "both"]:
        if use_cache and out_path.exists():
            cached_sp = (
                pl.read_csv(out_path)
                .filter(pl.col("metric") == "spearman")
                .drop("metric")
            )
            all_spearman = cached_sp
            spearman_bootstraps = cached_sp.filter(
                pl.col("model") != human_grounding.threshold_auc.HUMAN_MODEL_NAME
            )
        else:
            human_spearman = human_grounding.threshold_auc.compute_human_human_spearman(
                combined_results=_get_combined(),
                welfare_demographics=welfare_demographics,
                rai_demographics=rai_demographics,
                n_bootstrap=10,
            )
            spearman_bootstraps, _ = (
                human_grounding.threshold_auc.compute_threshold_auc(
                    combined_results=_get_combined(),
                    welfare_demographics=welfare_demographics,
                    rai_demographics=rai_demographics,
                    n_bootstrap=10,
                    metric="spearman",
                )
            )
            logger.debug(f"{human_spearman=}")
            all_spearman = pl.concat(
                [
                    spearman_bootstraps,
                    human_spearman.select(spearman_bootstraps.columns),
                ],
                how="vertical_relaxed",
            )

    # Persist freshly computed results (only when embeddings were actually run)
    if _embeddings_computed:
        to_save = []
        if all_binary_auc is not None:
            to_save.append(
                all_binary_auc.with_columns(pl.lit("binary_auc").alias("metric"))
            )
        if all_spearman is not None:
            to_save.append(
                all_spearman.with_columns(pl.lit("spearman").alias("metric"))
            )
        if to_save:
            pl.concat(to_save).write_csv(out_path)

    # --- PLOT (ONLY BINARY FOR NOW) ---
    if all_binary_auc is not None:
        human_grounding.threshold_auc.plot_auc_bar(
            all_binary_auc,
            plot_dir=PLOT_DIR,
            pretty_names=PRETTY_NAMES,
            font_scale=font_scale,
            use_english=use_english,
            top_n=TOP_N_TO_PLOT,
            file_type=file_type,
            height=8,
            aspect=1.8,
            filename_prefix=f"{experiment_name}_alignment_results",
        )

        summary_path = OUTPUT_DIR / f"alignment_summary_{experiment_name}.txt"
        if use_english:
            summary_path = append_english(summary_path)
        _write_dataset_summary(all_binary_auc, PRETTY_NAMES, summary_path)

        if auc_bootstraps is not None:
            for exp in experiments:
                plot_mmteb_correlation(
                    auc_bootstraps,
                    experiment=exp,
                    use_english=use_english,
                    file_type=file_type,
                )

    # --- SPEARMAN SUMMARY ---
    if spearman_bootstraps is not None and all_spearman is not None:
        top10_spearman = (
            all_spearman.group_by("model", "dataset")
            .agg(pl.col("auc").mean())
            .sort("auc", descending=True)
            .filter(pl.col("dataset") == "rai")
            .head(TOP_N_TO_PLOT)
        )
        logger.info("Top 10 Spearman:")
        logger.info(top10_spearman.to_dicts())

        human_grounding.threshold_auc.plot_auc_bar(
            all_spearman,
            plot_dir=PLOT_DIR,
            pretty_names=PRETTY_NAMES,
            font_scale=font_scale,
            use_english=use_english,
            top_n=TOP_N_TO_PLOT,
            file_type=file_type,
            x_label="Spearman",
            filename_prefix="spearman_results",
        )

    # --- HARD VS EASY DUMBBELL ---
    if use_cache and diff_csv.exists():
        logger.info(f"Loading cached difficulty summary from {diff_csv}")
        difficulty_summary = pl.read_csv(diff_csv)
    else:
        difficulty_group_auc = (
            human_grounding.threshold_auc.compute_difficulty_split_alignment(
                combined_results=_get_combined(),
                welfare_demographics=welfare_demographics,
                rai_demographics=rai_demographics,
                n_bootstrap=50,
                quantile=0.2,
            )
        )
        difficulty_summary = human_grounding.threshold_auc.summarise_difficulty_split(
            difficulty_group_auc
        )
        difficulty_summary.write_csv(diff_csv)

    actual_datasets = difficulty_summary["dataset"].unique().to_list()
    dataset_name_map = {ds: DATASET_PRETTY_NAMES.get(ds, ds) for ds in actual_datasets}
    human_grounding.threshold_auc.plot_difficulty_dumbbell(
        difficulty_summary,
        plot_dir=PLOT_DIR,
        pretty_names=PRETTY_NAMES,
        dataset_name_map=dataset_name_map,
        top_n=TOP_N_TO_PLOT,
        font_scale=font_scale * 0.55,
        file_type=file_type,
        title="",
        filename_prefix=experiment_name,
    )

    # --- DIFFICULTY TABLE ---
    model_order = (
        difficulty_summary.filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc_mean").mean())
        .sort("auc_mean", descending=True)
        .get_column("model")
        .to_list()
    )[: TOP_N_TO_PLOT + 1]
    pretty_order = [PRETTY_NAMES.get(m, m) for m in model_order]

    pivot = (
        difficulty_summary.filter(pl.col("model").is_in(model_order))
        .with_columns(pl.col("model").replace(PRETTY_NAMES))
        .pivot(
            on="statistic", index=["model", "dataset", "difficulty"], values="auc_mean"
        )
        .rename({"model": "Model", "difficulty": "Difficulty"})
    )
    stat_cols = [c for c in ["Best", "Worst", "Mean"] if c in pivot.columns]
    for ds in sorted(pivot["dataset"].unique().to_list()):
        tbl = (
            pivot.filter(pl.col("dataset") == ds)
            .select(["Model", "Difficulty", *stat_cols])
            .to_pandas()
        )
        tbl["Model"] = pd.Categorical(
            tbl["Model"], categories=pretty_order, ordered=True
        )
        tbl = tbl.sort_values(["Model", "Difficulty"]).reset_index(drop=True)
        for col in stat_cols:
            tbl[col] = tbl[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        logger.info(f"\n### {ds}\n{tbl.to_markdown(index=False)}")

    # --- CORRELATION WITH MMTEB (per experiment) ---
    for exp in experiments:
        mmteb_path = OUTPUT_DIR / f"mmteb_with_ranks_{exp}.csv"
        if not mmteb_path.exists():
            continue
        mmteb = pl.read_csv(mmteb_path)
        datasets = EXPERIMENT_DATASETS.get(exp, [exp])

        # MMTEB ranks are ascending (1 = best); negate so positive Spearman = agreement.
        mmteb_score = (-mmteb["Rank (MMTEB)"]).to_numpy()
        mmteb_names = mmteb["model_name"].to_numpy()
        mmteb_score_df = pl.DataFrame(
            {"model_name": mmteb_names, "mmteb_score": mmteb_score}
        )

        if auc_bootstraps is not None:
            avg_auc = (
                auc_bootstraps.filter(pl.col("dataset").is_in(datasets))
                .group_by("model")
                .agg(pl.col("auc").mean())
            )
            combined = avg_auc.join(
                mmteb_score_df, left_on="model", right_on="model_name"
            )
            corr = spearmanr(combined["mmteb_score"], combined["auc"])
            logger.info(f"[{exp}] AUC vs MMTEB(Borda) Spearman: {corr.correlation:.3f}")

        if spearman_bootstraps is not None:
            avg_sp = (
                spearman_bootstraps.filter(pl.col("dataset").is_in(datasets))
                .group_by("model")
                .agg(pl.col("auc").mean())
            )
            combined = avg_sp.join(
                mmteb_score_df, left_on="model", right_on="model_name"
            )
            corr = spearmanr(combined["mmteb_score"], combined["auc"])
            logger.info(f"[{exp}] Spearman vs MMTEB(Borda): {corr.correlation:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and plot neural alignment results for a set of models and datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scale", type=float, default=2.8)
    parser.add_argument("--english", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--file", type=str, default="pdf")
    parser.add_argument(
        "--metric", choices=["binary", "spearman", "both"], default="binary"
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=COORDINATES.keys(), default=["policy"]
    )

    args = parser.parse_args()

    main(
        args.scale,
        use_english=args.english,
        use_cache=args.cache,
        file_type=args.file,
        metric=args.metric,
        experiments=args.experiments,
    )
