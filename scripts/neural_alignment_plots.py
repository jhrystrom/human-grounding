import argparse
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from decorator import append
from dill.tests.test_registered import p
from loguru import logger
from scipy.stats import spearmanr
from tqdm import tqdm

import human_grounding.evaluate
import human_grounding.threshold_auc
from human_grounding.constants import PRETTY_NAMES
from human_grounding.data import (
    get_rai_demographics,
    get_responsible_ai,
    get_welfare,
    get_welfare_demographics,
)
from human_grounding.directories import DATA_DIR, OUTPUT_DIR, PLOT_DIR
from human_grounding.embed import get_all_models
from human_grounding.names import append_english

TOP_N_TO_PLOT = 10
DISTANCE_FILTER = 3.2


def plot_top_demographic_human_alignment(
    *,
    combined_bootstrapped: pl.DataFrame,
    score_ranks: pl.DataFrame,
    top_n_to_plot: int,
    plot_dir: Path,
    pretty_names: Mapping[str, str],
    font_scale: float = 1.0,
    use_english: bool = False,
    x_col: str = "alignment_score",
    y_col: str = "model",
    dataset_col: str = "dataset",
    hue_col: str = "demographics",
    dataset_name_map: Mapping[str, str] | None = None,
    x_label: str = "Human Grounding Score (Bootstrapped)",
    vline_x: float = 0.8,
    vline_label: str = "Human Reliability",
    palette_base: str = "tab20",
    height: float = 7.5,
    aspect: float = 1.7,
    sharey: bool = True,
    legend_y_offset: float = -0.12,
) -> Path:
    """
    Faceted barplot with:
      - per-panel dodge width (bars not forced thin by global hue count)
      - disjoint colors between panels (no overlap)
      - a single combined legend under the figure
      - optional vertical reference line

    Returns the saved PDF path.
    """
    sns.set_theme(style="whitegrid", font_scale=font_scale)

    if dataset_name_map is None:
        dataset_name_map = {"rai": "Responsible AI", "welfare": "Welfare"}

    # Build plotting dataframe
    df_pl = (
        combined_bootstrapped.join(
            score_ranks.filter(pl.col("rank") < top_n_to_plot),
            on=y_col,
        )
        .sort("rank", hue_col)
        .with_columns(
            pl.col(y_col).replace(pretty_names),
            pl.col(dataset_col).replace(dataset_name_map),
        )
    )

    df = df_pl.to_pandas()

    # Deterministic facet order
    dataset_order = list(df[dataset_col].dropna().unique())

    # Per-panel hue levels => per-panel dodge width
    demos_by_ds = {
        ds: list(df.loc[df[dataset_col] == ds, hue_col].dropna().unique())
        for ds in dataset_order
    }

    # Disjoint colors across panels
    k_total = sum(len(v) for v in demos_by_ds.values())
    base = sns.color_palette(palette_base, n_colors=max(20, k_total))
    if k_total > len(base):
        base = sns.color_palette("husl", n_colors=k_total)

    palette_by_ds = {}
    i = 0
    for ds in dataset_order:
        demos = demos_by_ds[ds]
        palette_by_ds[ds] = dict(zip(demos, base[i : i + len(demos)]))
        i += len(demos)

    g = sns.FacetGrid(
        df,
        col=dataset_col,
        col_order=dataset_order,
        sharey=sharey,
        height=height,
        aspect=aspect,
    )

    def facet_bar(data: pd.DataFrame, **kws):  # noqa: ANN003, ARG001
        ax = plt.gca()
        ds = data[dataset_col].iloc[0]
        hue_order = demos_by_ds[ds]

        sns.barplot(
            data=data,
            x=x_col,
            y=y_col,
            hue=hue_col,
            hue_order=hue_order,
            palette=palette_by_ds[ds],  # disjoint per panel
            dodge=True,
            ax=ax,
        )

        # Thicker line
        ax.axvline(vline_x, color="red", linestyle="--", label=vline_label, linewidth=4)
        ax.set_xlim(0, 1)

    g.map_dataframe(facet_bar)

    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.supxlabel(x_label)

    # Remove per-axes legends and add ONE combined legend
    for ax in g.axes.flat:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    all_h, all_l = [], []
    for ax in g.axes.flat:
        h, label = ax.get_legend_handles_labels()
        all_h.extend(h)
        all_l.extend(label)

    seen = set()
    handles, labels = [], []
    for h, label in zip(all_h, all_l):
        if label == vline_label:
            continue
        if label and label not in seen:
            seen.add(label)
            handles.append(h)
            labels.append(label)

    g.figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, legend_y_offset),
    )

    out_path = plot_dir / (
        "top_demographic_human_alignment_english.pdf"
        if use_english
        else "top_demographic_human_alignment.pdf"
    )
    plt.savefig(out_path, bbox_inches="tight")
    plt.clf()
    return out_path


def find_top_misaligment(
    results: pl.DataFrame, statements: pl.DataFrame, top_n: int = 10
) -> list[dict]:
    base = (
        results.filter(
            (~pl.col("embedding_correct"))
            & pl.col("model").str.starts_with("text-embedding-3-large")
        )
        .select(
            "dataset",
            "model",
            "source_idx",
            "closer_idx",
            "farther_idx",
            "pct_distance",
        )
        .unique()
    )

    # Top N per dataset (highest pct_distance)
    top_per_dataset = (
        base.sort(["dataset", "pct_distance"], descending=[False, True])
        .with_columns(
            pl.col("pct_distance")
            .rank("dense", descending=True)
            .over("dataset")
            .alias("_r")
        )
        .filter(pl.col("_r") <= top_n)
        .drop("_r")
    )

    return (
        top_per_dataset.join(
            statements.rename({"cause": "source"}),
            left_on="source_idx",
            right_on="statement_id",
        )
        .join(
            statements.rename({"cause": "closer"}),
            left_on="closer_idx",
            right_on="statement_id",
            suffix="_closer",
        )
        .join(
            statements.rename({"cause": "farther"}),
            left_on="farther_idx",
            right_on="statement_id",
            suffix="_farther",
        )
        .select(
            "dataset",
            "model",
            "pct_distance",
            "source_idx",
            "closer_idx",
            "farther_idx",
            "source",
            "closer",
            "farther",
        )
        .to_dicts()
    )


def get_embedding_alignments(
    models: list[str], full_dataset: pl.DataFrame, use_english: bool = False
) -> pl.DataFrame:
    results = []
    for model in tqdm(models, desc="Models"):
        if model.startswith("wicked"):
            print(f"Skipping model: {model} - not supported")
            continue
        print(f"Evaluating model: {model}")
        comparisons = human_grounding.evaluate.evaluate_human_embedding_match(
            model=model, all_coordinates=full_dataset, use_english=use_english
        )
        results.append(comparisons)
    return pl.concat(results).drop("cause", "source", "size")


def main(
    font_scale: float,
    use_english: bool = False,
    use_cache: bool = False,
    file_type: str = "pdf",
) -> None:
    create_sorted_groups = (
        pl.concat_arr(pl.col("closer_idx", "farther_idx"))
        .arr.sort()
        .alias("sorted_groups")
    )

    full_dataset = pl.read_csv(OUTPUT_DIR / "valid_coordinates.csv")
    models = sorted(get_all_models())
    dataset_names = ["rai", "welfare"]

    combined_results = get_embedding_alignments(
        models, full_dataset, use_english=use_english
    )
    welfare_demographics = get_welfare_demographics()
    rai_demographics = get_rai_demographics()

    human_auc = human_grounding.threshold_auc.load_human_auc(
        OUTPUT_DIR / "alpha_data_demographic.csv"
    )
    auc_output_path = OUTPUT_DIR / "embedding_alignment_auc.csv"
    if use_english:
        auc_output_path = append_english(auc_output_path)
    auc_bootstraps, _ = (
        human_grounding.threshold_auc.compute_threshold_auc(
            combined_results=combined_results,
            welfare_demographics=welfare_demographics,
            rai_demographics=rai_demographics,
            n_bootstrap=10,
        )
        if not use_cache
        else (pl.read_csv(auc_output_path), None)
    )
    auc_bootstraps.write_csv(auc_output_path)
    all_auc = pl.concat([auc_bootstraps, human_auc])
    human_grounding.threshold_auc.plot_auc_bar(
        all_auc,
        plot_dir=PLOT_DIR,
        pretty_names=PRETTY_NAMES,
        font_scale=font_scale,
        use_english=use_english,
        top_n=TOP_N_TO_PLOT,
        file_type=file_type,
    )
    mmteb_ranks = pl.read_csv(OUTPUT_DIR / "mmteb_with_ranks.csv").select(
        "model_name", "Mean (Task)"
    )
    average_auc = auc_bootstraps.group_by("model").agg(pl.col("auc").mean())
    combined_scores = average_auc.join(
        mmteb_ranks, left_on="model", right_on="model_name"
    )
    mmteb_spearman = spearmanr(combined_scores["Mean (Task)"], combined_scores["auc"])
    logger.info(
        f"Spearman correlation between MMTEB rank and human grounding: {mmteb_spearman.correlation:.4f} (p={mmteb_spearman.pvalue:.4f})"
    )

    # Regex for extracting name within brackets
    pattern = r"\[(.*?)\]"
    mmteb_raw = pl.read_csv(OUTPUT_DIR / "mmteb-top-dan.csv")
    mmteb_data = mmteb_raw.with_columns(
        pl.col("Model").str.extract(pattern).alias("model_name")
    ).select("model_name", "Mean (Task)")
    mmteb_with_ranks = (
        mmteb_data.drop_nulls()
        .join(
            combined_results.select("model").unique(),
            left_on="model_name",
            right_on="model",
        )
        .sort("Mean (Task)", descending=True)
        .with_row_index("Rank (MMTEB)")
    )
    mmteb_with_ranks_path = OUTPUT_DIR / "mmteb_with_ranks.csv"
    if use_english:
        mmteb_with_ranks_path = append_english(mmteb_with_ranks_path)
    mmteb_with_ranks.write_csv(mmteb_with_ranks_path)

    aggregated_bootstraps = average_auc.group_by("model").agg(
        pl.col("auc").mean().alias("alignment_score")
    )
    alignment_output = OUTPUT_DIR / "human_alignment_bootstrapped.csv"
    if use_english:
        alignment_output = append_english(alignment_output)
    aggregated_bootstraps.write_csv(alignment_output)

    score_ranks = aggregated_bootstraps.sort(
        "alignment_score", descending=True
    ).with_row_index("rank")

    rank_plot_data = (
        score_ranks.filter(pl.col("rank") < 10)
        .join(mmteb_with_ranks, left_on="model", right_on="model_name")
        .drop("alignment_score", "Mean (Task)")
        .rename({"rank": "Rank (Human)"})
        .unpivot(
            index="model",
            variable_name="Ranking Type",
            value_name="Rank",
        )
        .with_columns(pl.col("model").replace(PRETTY_NAMES))
    )
    # Convert to pandas for easier plotting
    pdf = rank_plot_data.to_pandas()

    # Split ranking types
    mmteb = pdf[pdf["Ranking Type"] == "Rank (MMTEB)"]
    human = pdf[pdf["Ranking Type"] == "Rank (Human)"]

    # Sort by Human rank (ascending = best first)
    human_sorted = human.sort_values("Rank")

    models = human_sorted["model"].tolist()
    x = np.arange(len(models))

    # Reorder both series to match human ranking
    mmteb = mmteb.set_index("model").loc[models].reset_index()
    human = human.set_index("model").loc[models].reset_index()

    sns.set_theme(style="whitegrid", font_scale=1.9)
    _fig, ax = plt.subplots(figsize=(12, 8))

    # Draw connecting lines
    for i, model in enumerate(models):
        ax.plot(
            [x[i], x[i]],
            [mmteb.iloc[i]["Rank"], human.iloc[i]["Rank"]],
            color="gray",
            linewidth=1,
            zorder=1,
        )

    # Scatter points
    ax.scatter(
        x,
        mmteb["Rank"],
        s=120,
        color="green",
        edgecolor="black",
        label="MMTEB",
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

    # Axes + labels
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Model Rank")
    ax.set_xlabel("")

    # Invert y-axis (rank 1 at top)
    ax.invert_yaxis()

    # Grid & legend
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.legend(title="Score Type")

    plt.tight_layout()
    mmteb_fig_path = PLOT_DIR / "mmteb_vs_human_ranking.pdf"
    if use_english:
        mmteb_fig_path = append_english(mmteb_fig_path)
    plt.savefig(mmteb_fig_path, bbox_inches="tight")
    plt.clf()

    spearman_data = score_ranks.join(
        mmteb_with_ranks, left_on="model", right_on="model_name"
    )
    spearman = spearmanr(spearman_data["alignment_score"], spearman_data["Mean (Task)"])
    logger.info(
        f"Spearman correlation for alignment score: {spearman.correlation:.4f} (p={spearman.pvalue:.4f})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate neural alignment plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale", type=float, default=2.8, help="Font scale for the plots"
    )
    parser.add_argument(
        "--english",
        action="store_true",
        help="Whether to use the English translated statements instead of original.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Whether to use cached AUC results instead of recomputing.",
    )
    parser.add_argument(
        "--file", type=str, choices=["pdf", "jpg", "png"], default="pdf"
    )
    args = parser.parse_args()

    main(
        args.scale, use_english=args.english, use_cache=args.cache, file_type=args.file
    )

    combined_results = pl.read_csv(OUTPUT_DIR / "valid_coordinates.csv")
    rai_demographics = get_rai_demographics()
    combined_results.join(
        rai_demographics, left_on="statement_id", right_on="cause_id"
    ).drop_nulls().group_by("dataset", "seed", "user_id").len().with_columns(
        pl.col("len") < 20
    ).filter(~pl.col("len"))
