from pathlib import Path

import polars as pl

from human_grounding.data import get_govai, get_responsible_ai, get_welfare
from human_grounding.directories import DATA_DIR, OUTPUT_DIR

ROUND_COLS = ["user_id", "dataset", "seed"]
STATEMENT_COLS = ["source_idx", "closer_idx", "farther_idx"]

DATASET_DISPLAY_NAMES = {
    "gov-ai": "Gov-AI",
    "rai": "RAI",
    "welfare": "Welfare",
}

# Validated placement coordinates -- the record of which participant rated
# which round, and hence of which rater pairs co-rated a round. The policy
# file is the validated one: ``combined_coordinates.csv`` additionally carries
# two mislabelled G rounds (rai seed 8 duplicates welfare seed 8; rai seed 15
# holds rai seed 16's statements) that would inflate the pair count.
COORDINATE_FILES = ("valid_coordinates.csv", "govai_coordinates.csv")

# A round is identified by dataset and seed.
ROUND_KEY = ["dataset", "seed"]


def _load_placements() -> pl.DataFrame:
    """Unique ``(dataset, seed, user_id)`` placements across the panels."""
    return pl.concat(
        [
            pl.read_csv(DATA_DIR / f)
            .with_columns(pl.col("dataset").replace({"welfware": "welfare"}))
            .select(*ROUND_KEY, "user_id")
            for f in COORDINATE_FILES
        ],
        how="vertical_relaxed",
    ).unique()


def rater_pair_rounds() -> pl.DataFrame:
    """Co-rated round counts per contributing rater pair.

    Returns ``dataset, user_id, user_id_b, n_rounds``, one row per rater pair
    that co-rated at least one round of that dataset. Pairs that never share a
    round are absent, since they contribute nothing to between-rater
    reliability. Pool over this frame for panel-independent statistics.
    """
    coords = _load_placements()
    return (
        coords.join(coords, on=ROUND_KEY, how="inner", suffix="_b")
        .filter(pl.col("user_id") < pl.col("user_id_b"))
        .select("dataset", "seed", "user_id", "user_id_b")
        .unique()
        .group_by("dataset", "user_id", "user_id_b")
        .agg(pl.len().alias("n_rounds"))
    )


def rater_pair_coverage() -> pl.DataFrame:
    """Potential vs. realised between-rater pairs per dataset.

    Between-rater reliability is defined over pairs of participants who
    co-rated at least one round, so a panel of ``n`` participants offers
    ``n choose 2`` potential pairs of which only those sharing a round are
    actually sampled. Read off the raw coordinate files rather than the
    triplet set, since the placements are what define round membership.

    Returns ``dataset, n_raters, n_pairs_potential, n_pairs_sampled,
    median_rounds_per_pair``; the median is taken over the sampled pairs only.
    """
    potential = (
        _load_placements()
        .group_by("dataset")
        .agg(pl.col("user_id").n_unique().alias("n_raters"))
        .with_columns(
            (pl.col("n_raters") * (pl.col("n_raters") - 1) / 2)
            .cast(pl.Int64)
            .alias("n_pairs_potential")
        )
    )

    sampled = (
        rater_pair_rounds()
        .group_by("dataset")
        .agg(
            pl.len().alias("n_pairs_sampled"),
            pl.median("n_rounds").alias("median_rounds_per_pair"),
        )
    )

    return potential.join(sampled, on="dataset", how="left").with_columns(
        pl.col("n_pairs_sampled").fill_null(0)
    )


def make_statement_coverage_table(
    round_df: pl.DataFrame,
    output_path: str | Path = "statement_coverage_table.tex",
    decimals: int = 2,
) -> pl.DataFrame:
    """
    Calculate:
      - mean/median per-statement occurrences
      - % of statement pairs co-occurring in >=1 round

    A round is indexed by (user_id, dataset, seed).

    Occurrence definition:
      A statement occurs in a round if it appears at least once in any of:
      source_idx, closer_idx, farther_idx.

    Pair co-occurrence definition:
      A pair of statements co-occurs if both appear in the same round at least once.
      The denominator is all unordered statement pairs observed within that dataset.
    """

    # Long format: one row per statement appearance in a round.
    long_df = (
        round_df.select(ROUND_COLS + STATEMENT_COLS)
        .unpivot(
            index=ROUND_COLS,
            on=STATEMENT_COLS,
            variable_name="role",
            value_name="statement_idx",
        )
        .drop_nulls("statement_idx")
        .select([*ROUND_COLS, "statement_idx"])
        .unique()
    )

    # Per-statement number of rounds in which statement appears.
    statement_occurrences = long_df.group_by(["dataset", "statement_idx"]).agg(
        pl.len().alias("round_occurrences")
    )

    occurrence_stats = statement_occurrences.group_by("dataset").agg(
        pl.len().alias("n_statements"),
        pl.mean("round_occurrences").alias("mean_occurrences"),
        pl.median("round_occurrences").alias("median_occurrences"),
    )

    # Build one unordered statement pair per round.
    # This uses a self-join within each round, then keeps statement_a < statement_b.
    pairs = (
        long_df.join(
            long_df,
            on=ROUND_COLS,
            how="inner",
            suffix="_b",
        )
        .filter(pl.col("statement_idx") < pl.col("statement_idx_b"))
        .select(
            "dataset",
            "statement_idx",
            "statement_idx_b",
        )
        .unique()
        .rename(
            {
                "statement_idx": "statement_a",
                "statement_idx_b": "statement_b",
            }
        )
    )

    cooccurring_pairs = pairs.group_by("dataset").agg(
        pl.len().alias("n_pairs_cooccurring")
    )

    pair_denominators = occurrence_stats.select("dataset", "n_statements").with_columns(
        (pl.col("n_statements") * (pl.col("n_statements") - 1) / 2)
        .cast(pl.Int64)
        .alias("n_possible_pairs")
    )

    total_statement_counts = {
        "rai": get_responsible_ai().select("cause_id").n_unique(),
        "welfare": get_welfare().select("cause_id").n_unique(),
        "gov-ai": get_govai().select("cause_id").n_unique(),
    }
    total_statements = pl.DataFrame(
        {
            "dataset": list(total_statement_counts.keys()),
            "n_statements_total": list(total_statement_counts.values()),
        }
    )

    final = (
        occurrence_stats.join(cooccurring_pairs, on="dataset", how="left")
        .join(pair_denominators, on=["dataset", "n_statements"], how="left")
        .join(total_statements, on="dataset", how="left")
        .join(rater_pair_coverage(), on="dataset", how="left")
        .with_columns(
            pl.col("n_pairs_cooccurring").fill_null(0),
            (100 * pl.col("n_pairs_cooccurring") / pl.col("n_possible_pairs")).alias(
                "pct_pairs_cooccurring"
            ),
        )
        .select(
            "dataset",
            "n_statements_total",
            "n_statements",
            "mean_occurrences",
            "median_occurrences",
            "pct_pairs_cooccurring",
            "n_pairs_potential",
            "n_pairs_sampled",
        )
        .sort(
            by="dataset",
            descending=False,
        )
    )

    display = final.with_columns(
        pl.col("dataset")
        .replace_strict(DATASET_DISPLAY_NAMES, default=pl.col("dataset"))
        .alias("dataset_display"),
    )

    fmt = f"%.{decimals}f"

    def row(record: dict) -> str:
        return (
            " & ".join(
                [
                    record["dataset_display"],
                    f"{record['n_statements_total']}",
                    f"{record['n_statements']}",
                    fmt % record["mean_occurrences"],
                    fmt % record["median_occurrences"],
                    fmt % record["pct_pairs_cooccurring"],
                    f"{record['n_pairs_potential']}",
                    f"{record['n_pairs_sampled']}",
                ]
            )
            + r" \\"
        )

    body_rows = [row(r) for r in display.to_dicts()]

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Statement occurrence and co-occurrence statistics by dataset "
        r"(\secref{sec:data}). Occurrences count the rounds in which a statement "
        r"appears in any role; co-occurrence is the share of all unordered "
        r"statement pairs that share at least one round. Rater pairs are the "
        r"$\binom{n}{2}$ pairs a panel of $n$ participants allows "
        r"($N_\text{potential}$) and the subset that co-rated at least one round "
        r"($N_\text{sampled}$), which is what between-rater reliability is "
        r"computed over.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Occurrences per statement} & Co-occurring "
        r"& \multicolumn{2}{c}{Rater pairs} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){7-8}",
        r"Dataset & $N_\text{total}$ & $N_\text{sampled}$ & Mean & Median "
        r"& \% of pairs & $N_\text{potential}$ & $N_\text{sampled}$ \\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]

    Path(output_path).write_text("\n".join(lines))

    return final


if __name__ == "__main__":
    output_path = OUTPUT_DIR / "statement_coverage_table.tex"
    round_df = pl.read_parquet(DATA_DIR / "raw_triplets_gov-ai_policy.parquet")
    _stats_df = make_statement_coverage_table(round_df, output_path=output_path)
