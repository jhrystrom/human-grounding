from typing import Literal

import polars as pl

from human_grounding.directories import DATA_DIR

Demographic = Literal["gender", "education"]


def get_all_variations() -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_parquet(file).drop("model_name", strict=False)
            for file in DATA_DIR.glob("variations_nsamples*.parquet")
        ]
    ).unique(maintain_order=True)


def get_responsible_ai() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / "answers_clean.parquet")


def get_responsible_ai_demographics() -> pl.DataFrame:
    return get_responsible_ai().select(
        "cause_id",
        "respondent_id",
        "cause",
        pl.col("Panel").alias("demographic"),
    )


def get_welfare(
    overwrite: bool = False, statement: Literal["cause", "solution"] = "cause"
) -> pl.DataFrame:
    welfare_path = DATA_DIR / f"welfare_clean_{statement}.parquet"
    if welfare_path.exists() and not overwrite:
        return pl.read_parquet(welfare_path)
    ai = get_responsible_ai()
    cause_start = ai["cause_id"].max() + 1
    respondent_start = ai["respondent_id"].max() + 1
    column_dict = {
        "Hvilket parti er du medlem af?\xa0  Hvis du ikke ønsker at svare, kan du trykke 'næste' uden at angive et parti.": "party",
    }
    pattern_dict = {"cause": "Årsag", "solution": "Løsning"}
    raw_welfare = (
        pl.read_excel(DATA_DIR / "pseudonymiseret-welfare.xlsx")
        .filter(pl.col("Samlet status - Nogen svar") == 1)
        .rename(column_dict)
        .with_row_index("respondent_id")
        .unpivot(
            pl.selectors.starts_with(pattern_dict[statement]),
            index=["respondent_id", "party"],
            value_name=statement,
        )
        .drop_nulls(statement)
        .drop("variable")
        .filter(pl.col("cause").str.len_chars() > 3)
        .with_row_index("cause_id")
        .with_columns(
            pl.col("cause_id") + cause_start,
            pl.col("respondent_id") + respondent_start,
            pl.col("party").cast(pl.String).alias("demographic"),
        )
        .drop("party")
    )
    raw_welfare.write_parquet(welfare_path)
    return raw_welfare


def get_all_statements() -> pl.DataFrame:
    columns = ["cause_id", "cause"]
    return pl.concat(
        [get_welfare().select(columns), get_responsible_ai().select(columns)]
    ).rename({"cause_id": "statement_id"})


def _get_rai_demographics_education() -> pl.DataFrame:
    return (
        pl.read_csv(DATA_DIR / "rai_demographics.csv")
        .with_columns(
            pl.when(
                pl.col("education_level").str.starts_with("Lang")
                | pl.col("education_level").str.starts_with("Forsker")
            )
            .then(pl.lit("High"))
            .when(pl.col("education_level").str.starts_with("Mellemlang"))
            .then(pl.lit("Medium"))
            .otherwise(pl.lit("Low"))
            .alias("demographic")
        )
        .drop("education_level")
    )


def get_rai_demographics(demographics: Demographic = "gender") -> pl.DataFrame:
    if demographics == "education":
        return _get_rai_demographics_education()
    gender_respondents = (
        pl.read_parquet(DATA_DIR / "demographics.parquet")
        .select("respondent_id", "gender_identity")
        .with_columns(
            pl.col("gender_identity")
            .fill_null("Unknown")
            .replace({"Ønsker ikke at svare": "Unknown"})
        )
        .rename({"gender_identity": "demographic"})
    )
    cause_id = get_responsible_ai().select("cause_id", "respondent_id")
    return cause_id.join(gender_respondents, on="respondent_id", how="left").drop(
        "respondent_id"
    )


def get_welfare_demographics() -> pl.DataFrame:
    return get_welfare().select("cause_id", "demographic").unique()


def get_demographics() -> pl.DataFrame:
    rai = _get_rai_demographics_education().select("cause_id", "demographic")
    welfare = get_welfare_demographics().select("cause_id", "demographic")
    return pl.concat([rai, welfare], how="vertical_relaxed").rename(
        {"cause_id": "statement_id"}
    )


if __name__ == "__main__":
    welfare = get_welfare(overwrite=False)
    print(welfare.head())
    responsible_ai = get_responsible_ai()
    responsible_ai
    demographics_rai = pl.read_parquet(DATA_DIR / "demographics.parquet")
    demographics_welfare = pl.read_excel(DATA_DIR / "pseudonymiseret-welfare.xlsx")
    rai_demo = get_rai_demographics()
    rai_demo["demographic"].value_counts()

    print(responsible_ai.head())
