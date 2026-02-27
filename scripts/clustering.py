import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, v_measure_score

import human_grounding.clustering
import human_grounding.evaluate
from human_grounding.constants import PRETTY_NAMES
from human_grounding.directories import OUTPUT_DIR, PLOT_DIR
from human_grounding.embed import get_all_models
from human_grounding.names import append_english

metrics = {
    "v_measure": v_measure_score,
    "adjusted_rand_index": adjusted_rand_score,
}


def get_user_clusters(round_df: pl.DataFrame) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (user_1_labels, user_2_labels, statement_ids) in statement_id-sorted order.
    """
    user_ids = round_df["user_id"].unique().to_list()

    u1_df = round_df.filter(pl.col("user_id") == user_ids[0]).sort("statement_id")
    u2_df = round_df.filter(pl.col("user_id") == user_ids[1]).sort("statement_id")

    statement_ids = u1_df["statement_id"].to_list()
    user_1 = u1_df["cluster_id"].to_list()
    user_2 = u2_df["cluster_id"].to_list()
    return user_1, user_2, statement_ids


def keep_mask(user_1: list[int], user_2: list[int]) -> list[bool]:
    """Keep only positions where neither user has -1."""
    return [(a != -1 and b != -1) for a, b in zip(user_1, user_2)]


def apply_mask(xs: list[int], mask: list[bool]) -> list[int]:
    return [x for x, m in zip(xs, mask) if m]


def ids_from_mask(ids_: list[int], mask: list[bool]) -> list[int]:
    return [i for i, m in zip(ids_, mask) if m]


def main(scale: float = 1.35, top: int = 20, use_english: bool = False):
    coordinates = pl.read_csv(OUTPUT_DIR / "normalised_coordinates.csv").with_columns(
        pl.col("user_id").str.strip_suffix("_coords")
    )
    total_rounds = coordinates.group_by("dataset", "seed", "user_id").len().height
    num_users = coordinates["user_id"].n_unique()
    logger.info(f"Total rounds: {total_rounds}, Unique users: {num_users}")
    logger.info(f"Rounds per user: {total_rounds / num_users}")

    # ---- Build per-user clusters (human clustering) ----
    clusters = []
    for (dataset, seed, user_id), round_df in coordinates.group_by(
        "dataset", "seed", "user_id"
    ):
        round_cluster = human_grounding.clustering.cluster_user_session(round_df)
        cluster_df = pl.DataFrame(
            [
                pl.Series("statement_id", list(round_cluster.keys())),
                pl.Series("cluster_id", list(round_cluster.values())),
            ]
        ).with_columns(
            pl.lit(dataset).alias("dataset"),
            pl.lit(seed).alias("seed"),
            pl.lit(user_id).alias("user_id"),
        )
        clusters.append(cluster_df)
    all_clusters = pl.concat(clusters)

    # ---- Human-vs-human consistency ----
    scores = []
    for (dataset, seed), round_df in all_clusters.group_by("dataset", "seed"):
        if round_df.height != 40:
            continue

        user_1, user_2, _statement_ids = get_user_clusters(round_df)
        mask = keep_mask(user_1, user_2)
        user_1 = apply_mask(user_1, mask)
        user_2 = apply_mask(user_2, mask)

        if not user_1:
            continue

        metric_scores = {}
        for metric, func in metrics.items():
            metric_scores[metric] = func(user_1, user_2)

        scores.append(
            {
                "dataset": dataset,
                "seed": seed,
                **metric_scores,
            }
        )

    human_score_df = pl.DataFrame(scores)
    path = OUTPUT_DIR / "human_cluster_consistency.csv"
    if use_english:
        path = append_english(path)
    human_score_df.write_csv(path)

    # ---- Model-vs-human consistency ----
    models = sorted(
        [model for model in get_all_models() if not model.startswith("wicked")]
    )

    model_scores = []
    for (dataset, seed), round_df in all_clusters.group_by("dataset", "seed"):
        if round_df.height != 40:
            continue
        if round_df["statement_id"].n_unique() != 20:
            continue

        user_1, user_2, statement_ids = get_user_clusters(round_df)
        mask = keep_mask(user_1, user_2)

        kept_ids = ids_from_mask(statement_ids, mask)
        user_1 = apply_mask(user_1, mask)
        user_2 = apply_mask(user_2, mask)

        if not user_1:
            continue

        for model in models:
            embeddings = (
                human_grounding.evaluate.get_statement_embeddings(
                    dataset=dataset, embedder_name=model, use_english=use_english
                )
                .join(
                    pl.DataFrame({"statement_id": kept_ids}),
                    left_on="cause_id",
                    right_on="statement_id",
                    how="inner",
                )
                .sort("cause_id")
            )

            # Defensive: ensure we are scoring on identical lengths
            if embeddings.height != len(user_1):
                continue

            num_categories = (len(np.unique(user_1)) + len(np.unique(user_2))) // 2
            if num_categories < 2:
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=num_categories, linkage="ward"
            )
            clusterer.fit(embeddings["embedding"].to_numpy())
            model_labels = clusterer.labels_.tolist()

            metric_scores = {}
            for metric, func in metrics.items():
                user_1_score = func(user_1, model_labels)
                user_2_score = func(user_2, model_labels)
                metric_scores[metric] = (user_1_score + user_2_score) / 2

            model_scores.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "model": model,
                    **metric_scores,
                }
            )

    model_df = pl.DataFrame(model_scores)
    model_path = OUTPUT_DIR / "model_cluster_consistency.csv"
    if use_english:
        model_path = append_english(model_path)
    model_df.write_csv(model_path)

    # ---- Aggregate + compare ----
    model_df = pl.read_csv(model_path).with_columns(pl.lit("Model").alias("type"))
    human_df = pl.read_csv(path).with_columns(
        pl.lit("Human").alias("type"), pl.lit("Human").alias("model")
    )
    combined_df = pl.concat([model_df, human_df.select(model_df.columns)])

    aggregated_df = combined_df.group_by("model").agg(
        *[pl.mean(metric).alias(metric) for metric in metrics]
    )
    aggregated_path = OUTPUT_DIR / "cluster_consistency_aggregated.csv"
    if use_english:
        aggregated_path = append_english(aggregated_path)
    aggregated_df.write_csv(aggregated_path)

    gt_path = OUTPUT_DIR / "human_alignment_bootstrapped.csv"
    if use_english:
        gt_path = append_english(gt_path)
    alignment_scores = pl.read_csv(gt_path)
    mmteb_scores = pl.read_csv(OUTPUT_DIR / "mmteb_with_ranks.csv")

    spearman = (
        alignment_scores.join(aggregated_df, on="model", how="inner")
        .with_columns(pl.lit("all").alias("group"))
        .group_by("group")
        .agg(
            pl.corr("alignment_score", "adjusted_rand_index", method="spearman").alias(
                "spearman_adjusted_rand_index"
            ),
        )
    )
    logger.info(
        f"Spearman correlation between grounding and adjusted rand index: {spearman} "
    )

    spearman_mmteb = (
        mmteb_scores.join(
            aggregated_df, left_on="model_name", right_on="model", how="inner"
        )
        .with_columns(pl.lit("all").alias("group"))
        .group_by("group")
        .agg(
            pl.corr("Mean (Task)", "adjusted_rand_index", method="spearman").alias(
                "spearman_mmteb_adjusted_rand_index"
            ),
        )
    )
    logger.info(
        f"Spearman correlation between MMTEB and adjusted rand index: {spearman_mmteb} "
    )

    # ---- Plotting ----
    sns.set_theme(style="whitegrid", font_scale=scale)
    for metric in metrics:
        ranks = (
            aggregated_df.sort(metric, descending=True)
            .with_row_index("rank")
            .drop(metric)
            .head(top)
        )
        strip_plot = sns.barplot(
            data=combined_df.join(ranks, on="model")
            .sort("rank")
            .with_columns(pl.col("model").replace(PRETTY_NAMES)),
            x="model",
            y=metric,
            hue="type",
        )
        strip_plot.set(ylabel=metric.replace("_", " ").title())
        strip_plot.set_xlabel("")
        strip_plot.set_xticklabels(
            strip_plot.get_xticklabels(), rotation=45, ha="right"
        )
        plot_path = PLOT_DIR / f"cluster_consistency_comparison_{metric}.pdf"
        if use_english:
            plot_path = append_english(plot_path)
        plt.savefig(
            plot_path,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate clustering consistency")
    parser.add_argument(
        "--scale", type=float, default=1.35, help="Font scale for the plots"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top models to plot"
    )
    parser.add_argument(
        "--english", action="store_true", help="Use English translations for statements"
    )
    args = parser.parse_args()

    main(scale=args.scale, top=args.top, use_english=args.english)
