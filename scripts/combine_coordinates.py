import re
from dataclasses import asdict, dataclass

import polars as pl
from loguru import logger

from human_grounding.directories import OUTPUT_DIR


@dataclass
class Info:
    dataset: str
    seed: int
    user_id: str


def parse_filename(filename: str) -> Info:
    """Parse filename like 'rai-n20-seed0-user_D_coords.csv' into Info."""
    pattern = r"^(?P<dataset>[^-]+)-.*-seed(?P<seed>\d+)-user_(?P<user_id>\w+)"
    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern")

    return Info(
        dataset=match.group("dataset"),
        seed=int(match.group("seed")),
        user_id=match.group("user_id"),
    )


all_paths = list(OUTPUT_DIR.glob("*_coords.csv"))
combined_data: list[pl.DataFrame] = []
for path in all_paths:
    if path.name.startswith("welfware"):
        logger.info(f"Skipping file {path.name}")
        continue
    try:
        path_info = parse_filename(path.name)
    except ValueError as e:
        logger.error(e)
        continue
    coordinates = pl.read_csv(path).drop("validation")
    info_data = pl.DataFrame([asdict(path_info)] * coordinates.height)
    if coordinates.height != 20:
        logger.warning(
            f"Expected 20 coordinates in {path.name}, found {coordinates.height}"
        )
        continue
    combined_data.append(pl.concat([info_data, coordinates], how="horizontal"))

full_data = pl.concat(combined_data, how="vertical").unique()
full_data.filter(pl.col("user_id").str.starts_with("H_") & pl.col("seed").eq(1)).sort(
    "statement_id"
)
output_path = OUTPUT_DIR / "combined_coordinates.csv"
full_data.write_csv(output_path)
