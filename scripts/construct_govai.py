from pathlib import Path

import polars as pl
from loguru import logger
from rapidfuzz import fuzz, process

from human_grounding.directories import DATA_DIR

MAX_CHARS = 300
SELECTED_USE_CASES = ["Mission-Enabling", "Mission-Enabling (internal agency support)"]
DEDUP_THRESHOLD = 96


def deduplicate_texts(texts: list[str], threshold: float) -> list[int]:
    keep_indices: list[int] = []
    kept_texts: list[str] = []
    for idx, text in enumerate(texts):
        if not kept_texts:
            keep_indices.append(idx)
            kept_texts.append(text)
            continue
        match = process.extractOne(
            text, kept_texts, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if match is None:
            keep_indices.append(idx)
            kept_texts.append(text)
    return keep_indices


if __name__ == "__main__":
    ai_descriptions = pl.read_csv(
        DATA_DIR / "2024_consolidated_ai_inventory_raw_v2.csv", encoding="latin1"
    )

    selected_descriptions = (
        ai_descriptions.with_row_index("statement_id")
        .rename(
            {"What is the intended purpose and expected benefits of the AI?": "text"}
        )
        .with_columns(
            pl.lit("gov-ai").alias("dataset"),
            (
                "Use Case: "
                + pl.col("Use Case Name")
                + "\nDescription: "
                + pl.col("text")
            ).alias("text"),
            pl.col("Use Case Topic Area").str.strip_chars(),
        )
        .select("statement_id", "dataset", "Agency", "Use Case Topic Area", "text")
        .filter(pl.col("text").str.len_chars() < MAX_CHARS)
        .filter(pl.col("Use Case Topic Area").is_in(SELECTED_USE_CASES))
    )

    before_count = selected_descriptions.height
    keep_indices = deduplicate_texts(
        selected_descriptions["text"].to_list(), DEDUP_THRESHOLD
    )
    selected_descriptions = selected_descriptions[keep_indices]
    after_count = selected_descriptions.height

    logger.info(f"Before deduplication: {before_count} rows")
    logger.info(f"After deduplication:  {after_count} rows")
    logger.info(f"Removed:              {before_count - after_count} near-duplicates")

    selected_descriptions.write_csv(DATA_DIR / "gov_ai.csv")
