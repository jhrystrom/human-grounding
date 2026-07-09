"""Statement occurrence and co-occurrence LaTeX table (\\secref{sec:data}).

A "round" is one 20-statement sub-sample shown to one rater under one
seed, keyed by ``(dataset, seed, user_id)`` -- see
``reviewer_answers.md`` section 8 and ``compute_distances`` in
``alpha_distance_plot.py`` for the same grouping. For each dataset we
report:

- ``Statements`` -- number of distinct statements observed across all
  rounds.
- ``Occurrences per statement`` (mean/median) -- number of rounds each
  statement appears in, in any role (source or one of the two compared
  items are all drawn from the same round sub-sample, so "occurrence"
  is just round membership).
- ``Co-occurring % of pairs`` -- of all unordered pairs among a
  dataset's statements, the share that appear together in at least one
  round (and can therefore form a triplet judgment together).

Output: ``output/coverage_table.tex``.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger

from human_grounding.directories import DATA_DIR, OUTPUT_DIR

# Sibling-script import (scripts/ is not a package)
sys.path.append(str(Path(__file__).parent))
from alpha_distance_plot import COORDINATES, load_coordinates

DATASET_PRETTY = {"gov-ai": "Gov-AI", "rai": "RAI", "welfare": "Welfare"}
DATASET_ORDER = ("gov-ai", "rai", "welfare")


def _load_all_coordinates() -> pl.DataFrame:
    parts = [load_coordinates(DATA_DIR / filename) for filename in COORDINATES.values()]
    return pl.concat(
        [df.select("dataset", "seed", "user_id", "statement_id") for df in parts],
        how="vertical_relaxed",
    )


def _coverage_stats(subset: pl.DataFrame) -> tuple[int, float, float, float]:
    """Return (n_statements, mean_occurrences, median_occurrences, pct_cooccurring_pairs)."""
    occurrences = subset.group_by("statement_id").agg(pl.len().alias("n_rounds"))
    n_statements = occurrences.height
    occ = occurrences["n_rounds"].to_numpy()

    cooccurring_pairs: set[tuple[int, int]] = set()
    for _, round_group in subset.group_by(["seed", "user_id"]):
        stmt_ids = sorted(round_group["statement_id"].unique().to_list())
        cooccurring_pairs.update(combinations(stmt_ids, 2))

    total_pairs = n_statements * (n_statements - 1) / 2
    pct_pairs = 100.0 * len(cooccurring_pairs) / total_pairs if total_pairs else 0.0

    return n_statements, float(np.mean(occ)), float(np.median(occ)), pct_pairs


def _build_tex(rows: list[tuple[str, int, float, float, float]]) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Statement occurrence and co-occurrence statistics by dataset "
        r"(\secref{sec:data}). Occurrences count the rounds in which a "
        r"statement appears in any role; co-occurrence is the share of all "
        r"unordered statement pairs that share at least one round.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Occurrences per statement} & Co-occurring \\",
        r"\cmidrule(lr){3-4}",
        r"Dataset & Statements & Mean & Median & \% of pairs \\",
        r"\midrule",
    ]
    for dataset_pretty, n_statements, mean_occ, median_occ, pct_pairs in rows:
        lines.append(
            f"{dataset_pretty} & {n_statements} & {mean_occ:.2f} & "
            f"{median_occ:.2f} & {pct_pairs:.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def main(output_path: Path | None = None) -> None:
    if output_path is None:
        output_path = OUTPUT_DIR / "coverage_table.tex"

    coords = _load_all_coordinates()

    rows: list[tuple[str, int, float, float, float]] = []
    for dataset in DATASET_ORDER:
        subset = coords.filter(pl.col("dataset") == dataset)
        n_statements, mean_occ, median_occ, pct_pairs = _coverage_stats(subset)
        logger.info(
            f"{dataset}: {n_statements} statements, mean={mean_occ:.2f}, "
            f"median={median_occ:.2f}, co-occurring={pct_pairs:.2f}%"
        )
        rows.append(
            (DATASET_PRETTY[dataset], n_statements, mean_occ, median_occ, pct_pairs)
        )

    output_path.write_text(_build_tex(rows))
    logger.info(f"Coverage table written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", type=Path, default=None)
    args = parser.parse_args()
    main(output_path=args.output_path)
