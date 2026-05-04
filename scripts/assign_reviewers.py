import polars as pl

from human_grounding.directories import DATA_DIR

ROUND_SIZE = 20
DATASET = "gov-ai"
NUM_ROUNDS = 7
reviewers = ["J", "K", "L", "M", "N", "O"]
middle = len(reviewers) // 2
team1 = reviewers[:middle]
team2 = reviewers[middle:]

total_seeds = NUM_ROUNDS * len(team1)


def assign(team1: list[str], team2: list[str], num_seeds: int) -> list[dict]:
    n = len(team1)
    assert len(team2) == n, "teams must be the same size"
    assert num_seeds % n == 0, "num_seeds must be divisible by team size"
    assignments = []
    for seed in range(num_seeds):
        t1 = team1[seed % n]
        t2 = team2[(seed % n + seed // n) % n]
        assignments.append({"seed": seed, "team1": t1, "team2": t2})
    return assignments


if __name__ == "__main__":
    assignments = assign(team1, team2, total_seeds)
    df = pl.DataFrame(assignments)

    df.unpivot(index="seed", value_name="reviewer_id").drop("variable").sort(
        "reviewer_id"
    ).with_columns(
        pl.lit(DATASET).alias("dataset"),
        pl.lit(ROUND_SIZE).alias("size"),
        pl.lit(None).alias("completed"),
    ).select("reviewer_id", "seed", "size", "dataset", "completed").write_csv(
        DATA_DIR / "reviewer_assignments.csv"
    )
