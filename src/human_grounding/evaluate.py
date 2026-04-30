import itertools
from dataclasses import dataclass

import numpy as np
import polars as pl
from joblib import Memory
from loguru import logger
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

import human_grounding.embed
from human_grounding.data import get_responsible_ai_demographics, get_welfare
from human_grounding.directories import CACHE_DIR, DATA_DIR, OUTPUT_DIR

memory = Memory(CACHE_DIR, verbose=0)


statement_readers = {
    "welfare": get_welfare,
    "rai": get_responsible_ai_demographics,
}


@dataclass
class ComparisonPair:
    source_idx: int
    closer_idx: int
    farther_idx: int
    pct_distance: float


def attach_distance_columns(
    comparison_df: pl.DataFrame,
    human_distances: np.ndarray,
    model_distances: np.ndarray,
) -> pl.DataFrame:
    """
    Vectorised extraction of distances for each (source, closer, farther) triplet.
    """
    idx = comparison_df.select("source_idx", "closer_idx", "farther_idx").to_numpy()

    s = idx[:, 0]
    c = idx[:, 1]
    f = idx[:, 2]

    return comparison_df.with_columns(
        pl.Series("human_dist_close", human_distances[s, c]),
        pl.Series("human_dist_far", human_distances[s, f]),
        pl.Series("model_dist_close", model_distances[s, c]),
        pl.Series("model_dist_far", model_distances[s, f]),
    )


def create_comparisons(coordinates: pl.DataFrame) -> pl.DataFrame:
    n_rows = coordinates.height
    comparison_pairs = []
    matrix = calculate_distance_matrix(coordinates)
    for index in range(n_rows):
        idx_set = set(range(n_rows))
        idx_set.remove(index)
        all_unordered_pairs = list(itertools.combinations(idx_set, 2))

        for idx1, idx2 in all_unordered_pairs:
            if matrix[index, idx1] < matrix[index, idx2]:
                closer_idx = idx1
                farther_idx = idx2
            else:
                closer_idx = idx2
                farther_idx = idx1

            pct_distance = matrix[index, farther_idx] / matrix[index, closer_idx]
            comparison_pair = ComparisonPair(
                index, closer_idx, farther_idx, pct_distance=pct_distance
            )
            comparison_pairs.append(comparison_pair)

    return pl.DataFrame(comparison_pairs)


# Evaluate embedding
def get_embedding_correctness(
    ground_truth: pl.DataFrame, embedding_distances: np.ndarray
) -> pl.Series:
    embedding_correct = []
    for pair in ground_truth.iter_rows(named=True):
        source_idx = pair["source_idx"]
        closer_idx = pair["closer_idx"]
        farther_idx = pair["farther_idx"]

        embedding_distance = (
            embedding_distances[source_idx, closer_idx]
            < embedding_distances[source_idx, farther_idx]
        )
        embedding_correct.append(embedding_distance)
    return pl.Series("embedding_correct", embedding_correct)


def calculate_distance_matrix(coordinates: pl.DataFrame) -> np.ndarray:
    return euclidean_distances(coordinates.select("x", "y").to_numpy())


# maximum verbosity
@memory.cache(verbose=100)
def get_statement_embeddings(
    dataset: str, embedder_name: str, use_english: bool = False
) -> pl.DataFrame:
    logger.info(
        f"Computing embeddings for dataset: {dataset} with embedder: {embedder_name}"
    )
    logger.debug("Loading embedder...")
    embedder = human_grounding.embed.get_embedder(embedding_model=embedder_name)
    logger.debug("Embedding statements...")
    statements = statement_readers[dataset]()
    if use_english:
        english_statements = pl.read_csv(DATA_DIR / "translated_statements.csv")
        statements = (
            statements.drop("cause")
            .join(
                english_statements,
                left_on="cause_id",
                right_on="statement_id",
                how="inner",
            )
            .rename({"english": "cause"})
        )
    return statements.with_columns(
        human_grounding.embed.embed_series(
            statements["cause"], embedder, embedding_name="embedding"
        )
    )


@memory.cache
def fake_cache(x: str) -> str:
    import time

    time.sleep(1)
    return x


@memory.cache
def human_embedding_match_new(
    model: str, coordinates: pl.DataFrame, use_english: bool = False
) -> pl.DataFrame:
    dataset = coordinates["dataset"].first()
    real_coordinates = coordinates.sort("user_id", "statement_id").with_columns(
        pl.int_range(pl.len()).over("user_id").alias("row_idx")
    )
    statement_embeddings = get_statement_embeddings(
        dataset=dataset, embedder_name=model, use_english=use_english
    ).join(
        real_coordinates,
        left_on="cause_id",
        right_on="statement_id",
        how="inner",
    )

    comparisons = []

    for (_,), statement_embeddings_group in statement_embeddings.group_by("user_id"):
        sorted_group = statement_embeddings_group.sort("row_idx")
        # --- MODEL distances (already exists) ---
        embedding_distances = cosine_distances(sorted_group["embedding"].to_numpy())
        # --- HUMAN distances (NEW) ---
        human_distances = euclidean_distances(sorted_group[["x", "y"]].to_numpy())
        # --- Triplets ---
        comparison_df = create_comparisons(sorted_group)
        # --- Correctness ---
        embedding_correct = get_embedding_correctness(
            ground_truth=comparison_df,
            embedding_distances=embedding_distances,
        )
        comparison_df = comparison_df.with_columns(
            embedding_correct,
            pl.lit(model).alias("model"),
        )
        # --- NEW: attach distance columns (vectorised) ---
        comparison_df = attach_distance_columns(
            comparison_df,
            human_distances,
            embedding_distances,
        )
        comparisons.append(comparison_df)

    comparison_df = pl.concat(comparisons)

    comparison_demographics = comparison_df.join(
        statement_embeddings, left_on="source_idx", right_on="row_idx"
    ).with_columns(
        pl.lit(dataset).alias("source"),
    )
    row_to_statement_id = statement_embeddings.select("cause_id", "row_idx").unique()
    for column in ["source_idx", "closer_idx", "farther_idx"]:
        comparison_demographics = map_to_statement_id(
            comparison_demographics, row_to_statement_id, column=column
        )
    return comparison_demographics.drop("embedding")


def map_to_statement_id(
    comparison_demographics: pl.DataFrame,
    row_to_statement_id: pl.DataFrame,
    column: str = "source_idx",
    statement_id_name: str = "cause_id_right",
) -> pl.DataFrame:
    return (
        comparison_demographics.join(
            row_to_statement_id, left_on=column, right_on="row_idx"
        )
        .drop(column)
        .rename({statement_id_name: column})
    )


def create_all_comparisons(all_coordinates: pl.DataFrame) -> pl.DataFrame:
    results = []
    for (dataset, seed, user_id), group in all_coordinates.group_by(
        "dataset", "seed", "user_id"
    ):
        if group.height < 20:
            logger.warning(
                f"Skipping dataset: {dataset}, seed: {seed}, user_id: {user_id} - not enough data"
            )
            continue
        sorted_group = group.sort("statement_id").with_row_index("row_idx")
        comparison_pairs = create_comparisons(sorted_group).with_columns(
            pl.lit(dataset).alias("dataset"),
            pl.lit(seed).alias("seed"),
            pl.lit(user_id).alias("user_id"),
        )
        row_to_statement_id = sorted_group.select("statement_id", "row_idx").unique()
        for column in ["source_idx", "closer_idx", "farther_idx"]:
            comparison_pairs = map_to_statement_id(
                comparison_pairs,
                row_to_statement_id,
                column=column,
                statement_id_name="statement_id",
            )
        results.append(comparison_pairs)
    return pl.concat(results)


def evaluate_human_embedding_match(
    model: str, all_coordinates: pl.DataFrame, use_english: bool = False
) -> pl.DataFrame:
    results = []
    for (dataset, seed), group in all_coordinates.group_by(["dataset", "seed"]):
        if group.height < 40:
            logger.warning(
                f"Skipping dataset: {dataset}, seed: {seed} - not enough data"
            )
            continue
        comparison_demographics = human_embedding_match_new(
            model=model, coordinates=group, use_english=use_english
        )
        results.append(comparison_demographics)
    return pl.concat(results)


if __name__ == "__main__":
    all_coordinates = pl.read_csv(OUTPUT_DIR / "valid_coordinates.csv")
    print("Hello!")
