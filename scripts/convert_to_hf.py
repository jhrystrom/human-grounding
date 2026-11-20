import os
import sys
from collections.abc import Callable
from typing import NamedTuple

import datasets
import polars as pl
from huggingface_hub import HfApi
from loguru import logger

import human_grounding.data
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

HF_DIR = OUTPUT_DIR / "hf"
REPO_NAME = "stakeholder-grounding"
LIST_COLUMNS = ["statement_ids", "texts", "x", "y", "demographics"]
ROUND_SIZE = 20


def get_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} must be set in .env")
    return value


def get_rai_statements() -> pl.DataFrame:
    demographics = human_grounding.data.get_rai_demographics()
    return (
        human_grounding.data.get_responsible_ai()
        .select("cause_id", "cause")
        .join(demographics, on="cause_id", how="left")
        .rename({"cause_id": "statement_id", "cause": "text"})
        .with_columns(pl.col("demographic").fill_null("Unknown"))
        .sort("statement_id")
    )


def get_welfare_statements() -> pl.DataFrame:
    # Demographic is an anonymised party code; null means the respondent
    # chose not to state a party and is kept as null on purpose.
    return (
        human_grounding.data.get_welfare()
        .select("cause_id", "cause", "demographic")
        .rename({"cause_id": "statement_id", "cause": "text"})
        .sort("statement_id")
    )


class Config(NamedTuple):
    get_statements: Callable[[], pl.DataFrame]
    n_rounds: int
    n_statements: int
    nullable_columns: tuple[str, ...] = ()


CONFIGS = {
    "rai": Config(get_rai_statements, n_rounds=58, n_statements=294),
    "welfare": Config(
        get_welfare_statements,
        n_rounds=30,
        n_statements=174,
        nullable_columns=("demographics",),
    ),
}


def get_coordinates(config_name: str) -> pl.DataFrame:
    return (
        pl.read_csv(DATA_DIR / "valid_coordinates.csv")
        .filter(pl.col("dataset") == config_name)
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


def validate_rounds(rounds: pl.DataFrame, config: Config) -> None:
    assert rounds.height == config.n_rounds, (
        f"Expected {config.n_rounds} rounds, got {rounds.height}"
    )
    for column in LIST_COLUMNS:
        lengths = rounds[column].list.len()
        assert (lengths == ROUND_SIZE).all(), f"{column} has lengths {lengths.unique()}"
        if column in config.nullable_columns:
            continue
        assert not rounds[column].list.eval(pl.element().is_null()).list.any().any(), (
            f"{column} contains nulls"
        )
    n_statements = rounds["statement_ids"].explode().n_unique()
    assert n_statements == config.n_statements, (
        f"Expected {config.n_statements} statements, got {n_statements}"
    )
    assert (rounds["statement_ids"].list.n_unique() == ROUND_SIZE).all(), (
        "Duplicate statement_id within a round"
    )


def build_dataset(config_name: str) -> datasets.DatasetDict:
    config = CONFIGS[config_name]
    rounds = get_rounds(get_coordinates(config_name), config.get_statements())
    validate_rounds(rounds, config)

    dataset = datasets.DatasetDict({"test": datasets.Dataset.from_polars(rounds)})
    dataset.save_to_disk(HF_DIR / config_name)
    logger.info(f"{config_name}: {dataset}")
    for row in dataset["test"].select(range(3)):
        logger.info(f"{config_name} example: {row}")
    return dataset


if __name__ == "__main__":
    # Usage: convert_to_hf.py [config ...]  (defaults to all configs)
    config_names = sys.argv[1:] or list(CONFIGS)
    unknown = set(config_names) - set(CONFIGS)
    assert not unknown, f"Unknown configs {unknown}; choose from {list(CONFIGS)}"

    hf_token = get_env("HF_TOKEN")
    repo_id = f"{get_env('HF_USERNAME')}/{REPO_NAME}"

    for config_name in config_names:
        dataset = build_dataset(config_name)
        logger.info(f"Pushing {config_name} to the Hub at {repo_id}")
        dataset.push_to_hub(
            repo_id, config_name=config_name, private=True, token=hf_token
        )

    sha = HfApi(token=hf_token).dataset_info(repo_id).sha
    logger.info(f"Revision for TaskMetadata.dataset['revision']: {sha}")
