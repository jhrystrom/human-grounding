import os

import datasets
import polars as pl
from huggingface_hub import HfApi
from loguru import logger

import human_grounding.data
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

HF_DIR = OUTPUT_DIR / "hf"
REPO_NAME = "stakeholder-grounding"
CONFIG_NAME = "rai"
LIST_COLUMNS = ["statement_ids", "texts", "x", "y", "demographics"]


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} must be set in .env")
    return value


def get_statements() -> pl.DataFrame:
    demographics = human_grounding.data.get_rai_demographics("gender")
    return (
        human_grounding.data.get_responsible_ai()
        .select("cause_id", "cause")
        .join(demographics, on="cause_id", how="left")
        .rename({"cause_id": "statement_id", "cause": "text"})
        .with_columns(pl.col("demographic").fill_null("Unknown"))
        .sort("statement_id")
    )


def get_coordinates() -> pl.DataFrame:
    return (
        pl.read_csv(DATA_DIR / "valid_coordinates.csv")
        .filter(pl.col("dataset") == CONFIG_NAME)
        # Rounds need at least two raters per seed to be comparable.
        .filter(pl.col("user_id").n_unique().over("seed") >= 2)
        .select("statement_id", "x", "y", "seed", "user_id")
    )


def get_rounds(coordinates: pl.DataFrame, statements: pl.DataFrame) -> pl.DataFrame:
    missing = coordinates.join(statements, on="statement_id", how="anti")
    assert missing.is_empty(), f"{missing.height} coordinates lack a statement"

    return (
        coordinates.join(statements, on="statement_id", how="left")
        .with_columns(pl.col("x", "y").cast(pl.Float64))
        .sort("user_id", "seed", "statement_id")
        .group_by("seed", "user_id", maintain_order=True)
        .agg(
            pl.col("statement_id").alias("statement_ids"),
            pl.col("text").alias("texts"),
            "x",
            "y",
            pl.col("demographic").alias("demographics"),
        )
        .with_columns(
            pl.format("{}-{}", "seed", "user_id").alias("round_id"),
        )
        .select("round_id", "seed", "user_id", *LIST_COLUMNS)
    )


def validate_rounds(rounds: pl.DataFrame) -> None:
    assert rounds.height == 58, f"Expected 58 rounds, got {rounds.height}"
    for column in LIST_COLUMNS:
        lengths = rounds[column].list.len()
        assert (lengths == 20).all(), f"{column} has lengths {lengths.unique()}"
        assert not rounds[column].list.eval(pl.element().is_null()).list.any().any(), (
            f"{column} contains nulls"
        )
    n_statements = rounds["statement_ids"].explode().n_unique()
    assert n_statements == 294, f"Expected 294 statements, got {n_statements}"
    assert (rounds["statement_ids"].list.n_unique() == 20).all(), (
        "Duplicate statement_id within a round"
    )


if __name__ == "__main__":
    hf_token = get_env("HF_TOKEN")
    repo_id = f"{get_env('HF_USERNAME')}/{REPO_NAME}"

    rounds = get_rounds(get_coordinates(), get_statements())
    validate_rounds(rounds)

    dataset = datasets.DatasetDict({"test": datasets.Dataset.from_polars(rounds)})
    dataset.save_to_disk(HF_DIR / CONFIG_NAME)
    logger.info(f"{CONFIG_NAME}: {dataset}")
    for row in dataset["test"].select(range(3)):
        logger.info(f"{CONFIG_NAME} example: {row}")

    logger.info(f"Pushing {CONFIG_NAME} to the Hub at {repo_id}")
    dataset.push_to_hub(repo_id, config_name=CONFIG_NAME, private=True, token=hf_token)
    sha = HfApi(token=hf_token).dataset_info(repo_id).sha
    logger.info(f"Revision for TaskMetadata.dataset['revision']: {sha}")
