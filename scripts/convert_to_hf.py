import os

import datasets
import polars as pl
from loguru import logger

import human_grounding.data
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

HF_DIR = OUTPUT_DIR / "hf"
REPO_NAME = "grounding-rai"


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
        .filter(pl.col("dataset") == "rai")
        .select("statement_id", "x", "y", "seed", "user_id")
        .sort("user_id", "seed", "statement_id")
    )


def to_hf(df: pl.DataFrame) -> datasets.Dataset:
    return datasets.Dataset.from_polars(df)


if __name__ == "__main__":
    hf_token = get_env("HF_TOKEN")
    repo_id = f"{get_env('HF_USERNAME')}/{REPO_NAME}"

    statements = get_statements()
    coordinates = get_coordinates()

    missing = coordinates.join(statements, on="statement_id", how="anti")
    assert missing.is_empty(), f"{missing.height} coordinates lack a statement"

    hf_datasets = {"statements": to_hf(statements), "coordinates": to_hf(coordinates)}
    for name, dataset in hf_datasets.items():
        dataset.save_to_disk(HF_DIR / name)
        logger.info(f"{name}: {dataset}")
        for row in dataset.select(range(min(3, len(dataset)))):
            logger.info(f"{name} example: {row}")
        logger.info(f"Pushing {name} to the Hub at {repo_id}")
        dataset.push_to_hub(repo_id, config_name=name, private=True, token=hf_token)
