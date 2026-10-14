import datasets
import polars as pl

import human_grounding.data
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

HF_DIR = OUTPUT_DIR / "hf"
# Set to e.g. "username/responsible-ai" to push both configs to the Hub.
REPO_ID: str | None = None


def get_statements() -> pl.DataFrame:
    demographics = human_grounding.data.get_rai_demographics("gender")
    return (
        human_grounding.data.get_responsible_ai()
        .select("cause_id", "cause")
        .join(demographics, on="cause_id", how="left")
        .rename({"cause_id": "statement_id"})
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
    statements = get_statements()
    coordinates = get_coordinates()

    missing = coordinates.join(statements, on="statement_id", how="anti")
    assert missing.is_empty(), f"{missing.height} coordinates lack a statement"

    hf_datasets = {"statements": to_hf(statements), "coordinates": to_hf(coordinates)}
    for name, dataset in hf_datasets.items():
        dataset.save_to_disk(HF_DIR / name)
        print(name, dataset)
        if REPO_ID is not None:
            dataset.push_to_hub(REPO_ID, config_name=name, private=True)
