"""Build the public statement tables in data/ from the restricted files in hidden_data/.

Writes ``data/rai_clean_cause.csv`` and ``data/welfare_clean_cause.csv``, each with
columns ``cause_id, cause, demographic``. These are the only statement files the
pipeline reads (see ``human_grounding.data``), so ``just reproduce-full`` runs
from checked-in data alone.

Requires in hidden_data/:
- answers_clean.parquet        (RAI statements: cause_id, respondent_id, Panel, cause)
- demographics.parquet         (RAI respondent demographics)
- pseudonymiseret-welfare.xlsx (raw welfare survey)
"""

import polars as pl

from human_grounding.data import RAI_PATH, WELFARE_PATH
from human_grounding.directories import get_main_dir

HIDDEN_DATA_DIR = get_main_dir(name="hidden_data", create_dir=False)
COLUMNS = ["cause_id", "cause", "demographic"]
PARTY_COLUMN = (
    "Hvilket parti er du medlem af?\xa0  Hvis du ikke ønsker at svare, "
    "kan du trykke 'næste' uden at angive et parti."
)


def build_responsible_ai(answers: pl.DataFrame) -> pl.DataFrame:
    # The protected attribute for RAI is gender, with missing / "prefer not to
    # say" collapsed to "Unknown".
    gender = pl.read_parquet(HIDDEN_DATA_DIR / "demographics.parquet").select(
        "respondent_id",
        pl.col("gender_identity")
        .fill_null("Unknown")
        .replace({"Ønsker ikke at svare": "Unknown"})
        .alias("demographic"),
    )
    return (
        answers.join(gender, on="respondent_id", how="left")
        .with_columns(pl.col("demographic").fill_null("Unknown"))
        .select(COLUMNS)
        .sort("cause_id")
    )


def build_welfare(answers: pl.DataFrame) -> pl.DataFrame:
    # Welfare ids continue after the RAI ids so statement ids are globally unique.
    cause_start = answers.select(pl.col("cause_id").max() + 1).item()
    return (
        pl.read_excel(HIDDEN_DATA_DIR / "pseudonymiseret-welfare.xlsx")
        .filter(pl.col("Samlet status - Nogen svar") == 1)
        .rename({PARTY_COLUMN: "party"})
        .with_row_index("respondent_id")
        .unpivot(
            pl.selectors.starts_with("Årsag"),
            index=["respondent_id", "party"],
            value_name="cause",
        )
        .drop_nulls("cause")
        .filter(pl.col("cause").str.len_chars() > 3)
        .with_row_index("cause_id")
        .with_columns(
            pl.col("cause_id") + cause_start,
            pl.col("party").cast(pl.String).alias("demographic"),
        )
        .select(COLUMNS)
    )


if __name__ == "__main__":
    answers = pl.read_parquet(HIDDEN_DATA_DIR / "answers_clean.parquet")

    rai = build_responsible_ai(answers)
    assert rai["cause_id"].is_unique().all()
    assert rai.height == answers.height

    welfare = build_welfare(answers)
    assert welfare["cause_id"].is_unique().all()

    rai.write_csv(RAI_PATH)
    welfare.write_csv(WELFARE_PATH)
