from pathlib import Path

import polars as pl

from human_grounding.directories import DATA_DIR, OUTPUT_DIR

ROUND_COLS = ["user_id", "dataset", "seed"]
STATEMENT_COLS = ["source_idx", "closer_idx", "farther_idx"]


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
        round_df
        .select(ROUND_COLS + STATEMENT_COLS)
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

    # Add an "All" dataset aggregation.
    long_with_all = pl.concat(
        [
            long_df,
            long_df.with_columns(pl.lit("All").alias("dataset")),
        ],
        how="vertical",
    )

    # Per-statement number of rounds in which statement appears.
    statement_occurrences = (
        long_with_all
        .group_by(["dataset", "statement_idx"])
        .agg(pl.len().alias("round_occurrences"))
    )

    occurrence_stats = (
        statement_occurrences
        .group_by("dataset")
        .agg(
            pl.len().alias("n_statements"),
            pl.mean("round_occurrences").alias("mean_occurrences"),
            pl.median("round_occurrences").alias("median_occurrences"),
        )
    )

    # Build one unordered statement pair per round.
    # This uses a self-join within each round, then keeps statement_a < statement_b.
    pairs = (
        long_with_all
        .join(
            long_with_all,
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

    cooccurring_pairs = (
        pairs
        .group_by("dataset")
        .agg(pl.len().alias("n_pairs_cooccurring"))
    )

    pair_denominators = (
        occurrence_stats
        .select("dataset", "n_statements")
        .with_columns(
            (
                pl.col("n_statements") * (pl.col("n_statements") - 1) / 2
            ).cast(pl.Int64).alias("n_possible_pairs")
        )
    )

    final = (
        occurrence_stats
        .join(cooccurring_pairs, on="dataset", how="left")
        .join(pair_denominators, on=["dataset", "n_statements"], how="left")
        .with_columns(
            pl.col("n_pairs_cooccurring").fill_null(0),
            (
                100
                * pl.col("n_pairs_cooccurring")
                / pl.col("n_possible_pairs")
            ).alias("pct_pairs_cooccurring")
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

    # Put "All" at the bottom.
    final = (
        final
        .with_columns((pl.col("dataset") == "All").alias("_is_all"))
        .sort(["_is_all", "dataset"])
        .drop("_is_all")
    )

    latex_df = final.select(
    pl.col("dataset").alias("Dataset"),
    pl.col("n_statements").alias("Statements"),
    pl.col("mean_occurrences").alias("Mean occurrences"),
    pl.col("median_occurrences").alias("Median occurrences"),
    pl.col("pct_pairs_cooccurring").alias(r"\% pairs co-occurring"),
)

    latex = latex_df.to_pandas().to_latex(
    index=False,
    escape=False,
    column_format="lrrrr",
    float_format="%.2f",
    caption="Statement occurrence and co-occurrence statistics by dataset.",
    label="tab:statement_coverage",
)

    # Optional: slightly more publication-ready booktabs table.
    latex = latex.replace("\\toprule", "\\toprule")
    latex = latex.replace("\\midrule", "\\midrule")
    latex = latex.replace("\\bottomrule", "\\bottomrule")

    Path(output_path).write_text(latex)

    return final


if __name__ == "__main__":
    output_path = OUTPUT_DIR / "statement_coverage_table.tex"
    round_df = pl.read_parquet(DATA_DIR / "raw_triplets_gov-ai_policy.parquet")
    _stats_df = make_statement_coverage_table(round_df, output_path=output_path)