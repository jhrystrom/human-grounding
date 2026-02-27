import asyncio

import polars as pl
from loguru import logger

import human_grounding.oai
import human_grounding.translate
from human_grounding.data import get_all_statements
from human_grounding.directories import DATA_DIR


async def main():
    model_config = human_grounding.oai.get_default_config(use_async=True)
    logger.info(f"Using model: {model_config.model}")
    statements = get_all_statements().sort("statement_id")
    logger.info(f"Translating {statements.height} statements...")
    translated_samples = await human_grounding.translate.atranslate_texts(
        texts=statements["cause"].to_list(), model_config=model_config
    )
    logger.info("Done! Saving to translated_statements.csv")
    english_statements = statements.with_columns(
        pl.Series("english", translated_samples)
    ).drop("cause")

    english_statements.write_csv(DATA_DIR / "translated_statements.csv")


if __name__ == "__main__":
    asyncio.run(main())
