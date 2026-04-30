import polars as pl

from human_grounding.directories import DATA_DIR


def main() -> None:
    govai = pl.concat(
        pl.read_csv(file) for file in (DATA_DIR / "gov-ai").glob("gov-ai-user*n20.csv")
    )
    govai.write_csv(DATA_DIR / "govai_coordinates.csv")
