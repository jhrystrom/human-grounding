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
from human_grounding.constants import PRETTY_NAMES
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
DATASET_PRETTY_NAMES = {
    "rai": "Responsible AI",
    "welfare": "Welfare",
    "gov-ai": "Government AI",
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
    
    welfare_demographics = get_welfare_demographics() if "policy" in experiments else None
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
            _combined = get_embedding_alignments(models, full_dataset, use_english=use_english)
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
                logger.info(f"No human baseline for {exp} ({alpha_path.name} not found), skipping")
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
                .filter(pl.col("model") != human_grounding.threshold_auc.HUMAN_MODEL_NAME)
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
                pl.read_csv(out_path).filter(pl.col("metric") == "spearman").drop("metric")
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
            spearman_bootstraps, _ = human_grounding.threshold_auc.compute_threshold_auc(
                combined_results=_get_combined(),
                welfare_demographics=welfare_demographics,
                rai_demographics=rai_demographics,
                n_bootstrap=10,
                metric="spearman",
            )
            logger.debug(f"{human_spearman=}")
            all_spearman = pl.concat(
                [spearman_bootstraps, human_spearman.select(spearman_bootstraps.columns)],
                how="vertical_relaxed",
            )

    # Persist freshly computed results (only when embeddings were actually run)
    if _embeddings_computed:
        to_save = []
        if all_binary_auc is not None:
            to_save.append(all_binary_auc.with_columns(pl.lit("binary_auc").alias("metric")))
        if all_spearman is not None:
            to_save.append(all_spearman.with_columns(pl.lit("spearman").alias("metric")))
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
        difficulty_summary
        .filter(pl.col("statistic") == "Mean")
        .group_by("model")
        .agg(pl.col("auc_mean").mean())
        .sort("auc_mean", descending=True)
        .get_column("model")
        .to_list()
    )[: TOP_N_TO_PLOT + 1]
    pretty_order = [PRETTY_NAMES.get(m, m) for m in model_order]

    pivot = (
        difficulty_summary
        .filter(pl.col("model").is_in(model_order))
        .with_columns(pl.col("model").replace(PRETTY_NAMES))
        .pivot(on="statistic", index=["model", "dataset", "difficulty"], values="auc_mean")
        .rename({"model": "Model", "difficulty": "Difficulty"})
    )
    stat_cols = [c for c in ["Best", "Worst", "Mean"] if c in pivot.columns]
    for ds in sorted(pivot["dataset"].unique().to_list()):
        tbl = (
            pivot.filter(pl.col("dataset") == ds)
            .select(["Model", "Difficulty", *stat_cols])
            .to_pandas()
        )
        tbl["Model"] = pd.Categorical(tbl["Model"], categories=pretty_order, ordered=True)
        tbl = tbl.sort_values(["Model", "Difficulty"]).reset_index(drop=True)
        for col in stat_cols:
            tbl[col] = tbl[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        logger.info(f"\n### {ds}\n{tbl.to_markdown(index=False)}")

    # --- CORRELATION WITH MMTEB ---
    mmteb = pl.read_csv(OUTPUT_DIR / "mmteb_with_ranks.csv")

    if auc_bootstraps is not None:
        avg_auc = auc_bootstraps.group_by("model").agg(pl.col("auc").mean())
        combined = avg_auc.join(mmteb, left_on="model", right_on="model_name")
        corr = spearmanr(combined["Mean (Task)"], combined["auc"])
        logger.info(f"AUC vs MMTEB Spearman: {corr.correlation:.3f}")

    if spearman_bootstraps is not None:
        avg_sp = spearman_bootstraps.group_by("model").agg(pl.col("auc").mean())
        combined = avg_sp.join(mmteb, left_on="model", right_on="model_name")
        corr = spearmanr(combined["Mean (Task)"], combined["auc"])
        logger.info(f"Spearman vs MMTEB: {corr.correlation:.3f}")


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
    parser.add_argument("--experiments", nargs="+", choices=COORDINATES.keys(), default=["policy"])

    args = parser.parse_args()

    main(
        args.scale,
        use_english=args.english,
        use_cache=args.cache,
        file_type=args.file,
        metric=args.metric,
        experiments=args.experiments,
    )
