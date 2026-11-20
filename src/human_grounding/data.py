"""Loaders for the public statement tables checked in to ``data/``.

Every statement table has the columns ``cause_id, cause, demographic``. The
tables are built from the restricted raw files in ``hidden_data/`` by
``scripts/commit_data.py``; nothing here touches ``hidden_data/``.

Demographics: RAI uses gender (``Kvinde``/``Mand``/``Unknown``), welfare uses an
anonymised party code (null when the respondent declined to state a party).
"""

from pathlib import Path

import polars as pl

from human_grounding.directories import DATA_DIR

RAI_PATH = DATA_DIR / "rai_clean_cause.csv"
WELFARE_PATH = DATA_DIR / "welfare_clean_cause.csv"
GOVAI_PATH = DATA_DIR / "gov_ai.csv"

STATEMENT_SCHEMA = {"cause_id": pl.Int64, "cause": pl.String, "demographic": pl.String}


def _read_statements(path: Path) -> pl.DataFrame:
    return pl.read_csv(path, schema_overrides=STATEMENT_SCHEMA)


def get_responsible_ai() -> pl.DataFrame:
    return _read_statements(RAI_PATH)


def get_welfare() -> pl.DataFrame:
    return _read_statements(WELFARE_PATH)


def get_govai() -> pl.DataFrame:
    return (
        pl.read_csv(GOVAI_PATH)
        .rename({"statement_id": "cause_id", "text": "cause"})
        .with_columns(pl.lit(None).cast(pl.String).alias("demographic"))
        .select("cause_id", "cause", "demographic")
    )


def get_all_statements() -> pl.DataFrame:
    columns = ["cause_id", "cause"]
    return pl.concat(
        [get_welfare().select(columns), get_responsible_ai().select(columns)]
    ).rename({"cause_id": "statement_id"})


def get_rai_demographics() -> pl.DataFrame:
    return get_responsible_ai().select("cause_id", "demographic")


def get_welfare_demographics() -> pl.DataFrame:
    return get_welfare().select("cause_id", "demographic").unique()


def get_demographics() -> pl.DataFrame:
    # RAI's protected attribute is gender (Kvinde/Mand); the missing-group
    # "Unknown" is excluded from the demographic reliability analysis.
    rai = get_rai_demographics().filter(pl.col("demographic") != "Unknown")
    welfare = get_welfare_demographics()
    return pl.concat([rai, welfare], how="vertical_relaxed").rename(
        {"cause_id": "statement_id"}
    )
