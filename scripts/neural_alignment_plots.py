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


def main(
    font_scale: float,
    use_english: bool = False,
    use_cache: bool = False,  # noqa: ARG001
    file_type: str = "pdf",
    metric: str = "binary",
    dataset: str = "policy",
) -> None:
    full_dataset = pl.read_csv(DATA_DIR / COORDINATES[dataset])
    models = sorted(get_all_models())

    combined_results = get_embedding_alignments(
        models, full_dataset, use_english=use_english
    )
    welfare_demographics = get_welfare_demographics() if dataset == "policy" else None
    rai_demographics = get_rai_demographics() if dataset == "policy" else None

    # --- BINARY AUC ---
    if metric in ["binary", "both"]:
        auc_bootstraps, _ = human_grounding.threshold_auc.compute_threshold_auc(
            combined_results=combined_results,
            welfare_demographics=welfare_demographics,
            rai_demographics=rai_demographics,
            n_bootstrap=10,
            metric="binary",
        )
        auc_bootstraps = auc_bootstraps.with_columns(
            pl.lit("binary_auc").alias("metric")
        )
    else:
        auc_bootstraps = None

    # --- SPEARMAN ---
    spearman_bootstraps: pl.DataFrame | None = None
    if metric in ["spearman", "both"]:
        human_spearman = human_grounding.threshold_auc.compute_human_human_spearman(
            combined_results=combined_results,
            welfare_demographics=welfare_demographics,
            rai_demographics=rai_demographics,
            n_bootstrap=10,  # match your other setting
        )
        spearman_bootstraps, _ = human_grounding.threshold_auc.compute_threshold_auc(
            combined_results=combined_results,
            welfare_demographics=welfare_demographics,
            rai_demographics=rai_demographics,
            n_bootstrap=10,
            metric="spearman",
        )
        logger.debug(f"{human_spearman=}")
        all_spearman = pl.concat(
            [spearman_bootstraps, human_spearman.select(spearman_bootstraps.columns)],
            how="vertical_relaxed",
        ).with_columns(pl.lit("spearman").alias("metric"))
    else:
        all_spearman = None

    # --- COMBINE ---
    frames = [f for f in [auc_bootstraps, all_spearman] if f is not None]
    all_results = pl.concat(frames)

    # Save
    out_path = OUTPUT_DIR / "alignment_results.csv"
    if use_english:
        out_path = append_english(out_path)
    all_results.write_csv(out_path)

    # --- PLOT (ONLY BINARY FOR NOW) ---
    if auc_bootstraps is not None:
        human_grounding.threshold_auc.plot_auc_bar(
            auc_bootstraps,
            plot_dir=PLOT_DIR,
            pretty_names=PRETTY_NAMES,
            font_scale=font_scale,
            use_english=use_english,
            top_n=TOP_N_TO_PLOT,
            file_type=file_type,
        )

    # --- SPEARMAN SUMMARY ---
    if spearman_bootstraps is not None and all_spearman is not None:
        # Top 10 Spearman:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=2.8)
    parser.add_argument("--english", action="store_true")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--file", type=str, default="pdf")
    parser.add_argument(
        "--metric", choices=["binary", "spearman", "both"], default="binary"
    )
    parser.add_argument("--dataset", choices=COORDINATES.keys(), default="policy")

    args = parser.parse_args()

    main(
        args.scale,
        use_english=args.english,
        use_cache=args.cache,
        file_type=args.file,
        metric=args.metric,
        dataset=args.dataset,
    )
