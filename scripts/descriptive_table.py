from pathlib import Path

import polars as pl

from human_grounding.directories import DATA_DIR, OUTPUT_DIR

ROUND_COLS = ["user_id", "dataset", "seed"]
STATEMENT_COLS = ["source_idx", "closer_idx", "farther_idx"]

DATASET_DISPLAY_NAMES = {
    "gov-ai": "Gov-AI",
    "rai": "RAI",
    "welfare": "Welfare",
}


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

    final = (
        occurrence_stats.join(cooccurring_pairs, on="dataset", how="left")
        .join(pair_denominators, on=["dataset", "n_statements"], how="left")
        .with_columns(
            pl.col("n_pairs_cooccurring").fill_null(0),
            (100 * pl.col("n_pairs_cooccurring") / pl.col("n_possible_pairs")).alias(
                "pct_pairs_cooccurring"
            ),
        )
        .select(
            "dataset",
            "n_statements",
            "mean_occurrences",
            "median_occurrences",
            "pct_pairs_cooccurring",
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
                    f"{record['n_statements']}",
                    fmt % record["mean_occurrences"],
                    fmt % record["median_occurrences"],
                    fmt % record["pct_pairs_cooccurring"],
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
        r"statement pairs that share at least one round.}",
        r"\label{tab:coverage}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{Occurrences per statement} & Co-occurring \\",
        r"\cmidrule(lr){3-4}",
        r"Dataset & Statements & Mean & Median & \% of pairs \\",
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
