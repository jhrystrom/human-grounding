import argparse

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from loguru import logger
from scipy.stats import bootstrap, spearmanr
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score, v_measure_score

import human_grounding.clustering
import human_grounding.evaluate
from human_grounding.constants import PRETTY_NAMES
from human_grounding.directories import DATA_DIR, OUTPUT_DIR, PLOT_DIR
from human_grounding.embed import get_standard_models
from human_grounding.names import append_english

metrics = {
    "v_measure": v_measure_score,
    "adjusted_rand_index": adjusted_rand_score,
}

COORDINATES = {
    "policy": OUTPUT_DIR / "normalised_coordinates.csv",
    "gov-ai": DATA_DIR / "govai_coordinates.csv",
}


K_SELECTION_CHOICES = ("human", "silhouette")
SILHOUETTE_K_RANGE = range(2, 11)


def pick_k_silhouette(
    embeddings: np.ndarray,
    k_range: range = SILHOUETTE_K_RANGE,
) -> int:
    """Pick K for Ward clustering by maximising silhouette score on `embeddings`.

    K is constrained to `k_range` and clipped to `[2, n_samples - 1]`.
    """
    n = embeddings.shape[0]
    candidates = [k for k in k_range if 2 <= k <= n - 1]
    if not candidates:
        return max(2, min(n - 1, 2))

    best_k, best_score = candidates[0], -np.inf
    for k in candidates:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(
            embeddings
        )
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, float(score)
    return best_k


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


def main(
    experiments: list[str],
    scale: float = 1.35,
    top: int = 20,
    n_bootstrap: int = 2000,
    seed: int = 0,
    k_selection: str = "human",
) -> None:
    _aggregated_full = []
    _combined_full = []
    for experiment in experiments:
        aggregated_df, combined_df = analyse_single(experiment, k_selection=k_selection)
        _aggregated_full.append(
            aggregated_df.with_columns(pl.lit(experiment).alias("experiment"))
        )
        _combined_full.append(
            combined_df.with_columns(pl.lit(experiment).alias("experiment"))
        )
    aggregated_full = pl.concat(_aggregated_full)
    combined_full = pl.concat(_combined_full)

    long_table = compute_spearman_table(
        combined_full, experiments, n_bootstrap=n_bootstrap, seed=seed
    )
    long_path = OUTPUT_DIR / "cluster_spearman_by_experiment.csv"
    long_table.write_csv(long_path)

    formatted = long_table.with_columns(
        (
            pl.col("spearman").round(2).cast(pl.Utf8)
            + " ["
            + pl.col("ci_lo").round(2).cast(pl.Utf8)
            + ", "
            + pl.col("ci_hi").round(2).cast(pl.Utf8)
            + "]"
        ).alias("cell")
    ).pivot(on="experiment", index="source", values="cell")

    logger.info(
        f"Spearman vs ARI (per experiment, {n_bootstrap} bootstrap) -> {long_path}"
    )
    logger.info(f"\n{formatted.to_pandas().to_markdown(index=False)}")

    plot_comparison(
        aggregated_full, combined_full, scale=scale, top=top, k_selection=k_selection
    )


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Point estimate and BCa 95% CI of Spearman's rho via paired pairs-bootstrap."""
    n = len(x)
    point = float(spearmanr(x, y).statistic)
    if n < 4:
        return point, float("nan"), float("nan")

    def _stat(a: np.ndarray, b: np.ndarray) -> float:
        r = spearmanr(a, b).statistic
        return float(r) if np.isfinite(r) else 0.0

    res = bootstrap(
        (x, y),
        _stat,
        paired=True,
        vectorized=False,
        n_resamples=n_bootstrap,
        method="BCa",
        random_state=rng,
    )
    return (
        point,
        float(res.confidence_interval.low),
        float(res.confidence_interval.high),
    )


def compute_spearman_table(
    combined_full: pl.DataFrame,
    experiments: list[str],
    metric: str = "adjusted_rand_index",
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> pl.DataFrame:
    """Spearman's rho (with bootstrap 95% CI) between model ARI and each ranking source.

    Resamples models (rows) with replacement to build the CI. Returns long-format
    rows: ``(source, experiment, n_models, spearman, ci_lo, ci_hi)``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for exp in experiments:
        per_model = (
            combined_full.filter(pl.col("experiment") == exp)
            .filter(pl.col("type") == "Model")
            .group_by("model")
            .agg(pl.mean(metric).alias(metric))
        )

        gt_path = OUTPUT_DIR / f"human_alignment_bootstrapped_{exp}.csv"
        if gt_path.exists():
            joined = pl.read_csv(gt_path).join(per_model, on="model", how="inner")
            x = joined["alignment_score"].to_numpy()
            y = joined[metric].to_numpy()
            point, lo, hi = _bootstrap_spearman_ci(x, y, rng, n_bootstrap)
            rows.append(
                {
                    "source": "OurExercise",
                    "experiment": exp,
                    "n_models": joined.height,
                    "spearman": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        else:
            logger.warning(f"Missing {gt_path}; skipping OurExercise for {exp}")

        mmteb_path = OUTPUT_DIR / f"mmteb_with_ranks_{exp}.csv"
        if mmteb_path.exists():
            joined = pl.read_csv(mmteb_path).join(
                per_model, left_on="model_name", right_on="model", how="inner"
            )
            # Negate the subset-relative Borda rank so positive rho = agreement.
            x = (-joined["Rank (MMTEB)"]).to_numpy()
            y = joined[metric].to_numpy()
            point, lo, hi = _bootstrap_spearman_ci(x, y, rng, n_bootstrap)
            rows.append(
                {
                    "source": "MMTEB",
                    "experiment": exp,
                    "n_models": joined.height,
                    "spearman": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        else:
            logger.warning(f"Missing {mmteb_path}; skipping MMTEB for {exp}")

    return pl.DataFrame(rows)


def analyse_single(
    experiment: str = "policy",
    k_selection: str = "human",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if k_selection not in K_SELECTION_CHOICES:
        msg = f"k_selection must be one of {K_SELECTION_CHOICES}, got {k_selection!r}"
        raise ValueError(msg)
    suffix = "" if k_selection == "human" else f"_{k_selection}"
    coordinates = pl.read_csv(COORDINATES[experiment]).with_columns(
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
    path = OUTPUT_DIR / f"human_cluster_consistency_{experiment}{suffix}.csv"
    human_score_df.write_csv(path)

    # ---- Model-vs-human consistency ----
    models = sorted(get_standard_models())

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
                    dataset=dataset, embedder_name=model
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

            emb_arr = embeddings["embedding"].to_numpy()

            if k_selection == "silhouette":
                num_categories = pick_k_silhouette(emb_arr)
            else:
                num_categories = (len(np.unique(user_1)) + len(np.unique(user_2))) // 2
            if num_categories < 2:
                continue

            clusterer = AgglomerativeClustering(
                n_clusters=num_categories, linkage="ward"
            )
            clusterer.fit(emb_arr)
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
    model_path = OUTPUT_DIR / f"model_cluster_consistency_{experiment}{suffix}.csv"
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
    aggregated_path = (
        OUTPUT_DIR / f"cluster_consistency_aggregated_{experiment}{suffix}.csv"
    )
    aggregated_df.write_csv(aggregated_path)

    aggregated_by_dataset = combined_df.group_by("model", "dataset").agg(
        *[pl.mean(metric).alias(metric) for metric in metrics]
    )
    return aggregated_by_dataset, combined_df


def plot_comparison(
    aggregated_df: pl.DataFrame,
    combined_df: pl.DataFrame,
    scale: float = 1.35,
    top: int = 20,
    k_selection: str = "human",
) -> None:
    # ---- Plotting ----
    suffix = "" if k_selection == "human" else f"_{k_selection}"
    sns.set_theme(style="whitegrid", font_scale=scale)
    for metric in metrics:
        borda_ranks = (
            aggregated_df.with_columns(
                pl.col(metric).rank(descending=True).over("experiment").alias("_rank")
            )
            .group_by("model")
            .agg(pl.sum("_rank").alias("borda_score"))
            .sort("borda_score")
            .with_row_index("rank")
            .select("model", "rank")
            .head(top)
        )
        strip_plot = sns.barplot(
            data=combined_df.join(borda_ranks, on="model")
            .sort("rank")
            .with_columns(pl.col("model").replace(PRETTY_NAMES)),
            x="model",
            y=metric,
            hue="experiment",
        )
        strip_plot.set(ylabel=metric.replace("_", " ").title())
        strip_plot.set_xlabel("")
        strip_plot.set_xticklabels(
            strip_plot.get_xticklabels(), rotation=45, ha="right"
        )
        # Bold the `Human` label
        for label in strip_plot.get_xticklabels():
            if label.get_text() == "Human":
                label.set_fontweight("bold")
        plot_path = PLOT_DIR / f"cluster_consistency_comparison_{metric}{suffix}.pdf"
        plt.savefig(
            plot_path,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate clustering consistency",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale", type=float, default=1.35, help="Font scale for the plots"
    )
    parser.add_argument(
        "--top", type=int, default=20, help="Number of top models to plot"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=str,
        choices=["policy", "gov-ai"],
        default=["policy", "gov-ai"],
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Bootstrap iterations for Spearman CI",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed for the bootstrap"
    )
    parser.add_argument(
        "--k-selection",
        type=str,
        default="human",
        choices=K_SELECTION_CHOICES,
        help=(
            "How model K is chosen per round: 'human' (default, average human K) "
            "or 'silhouette' (max silhouette over K in [2, 10])."
        ),
    )
    args = parser.parse_args()

    main(
        scale=args.scale,
        top=args.top,
        experiments=args.experiments,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        k_selection=args.k_selection,
    )
