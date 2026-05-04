"""Aesthetic test-bed for plot_difficulty_dumbbell.

Generates synthetic summary data and calls the plot function directly so you
can iterate on visuals without re-running the full pipeline.

Usage:
    python scripts/tune_difficulty_plot.py
    python scripts/tune_difficulty_plot.py --out /tmp --file-type png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

from human_grounding.threshold_auc import (
    DIFFICULTY_LABELS,
    plot_difficulty_dumbbell,
)

RNG = np.random.default_rng(42)

MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
    "nomic-embed-text",
    "bge-large-en-v1.5",
    "e5-mistral-7b-instruct",
    "gte-Qwen2-7B-instruct",
    "Human",
]

DATASETS = ["welfare", "rai"]
STATISTICS = ["Best", "Mean", "Worst"]
DIFFICULTIES = [DIFFICULTY_LABELS["hard"], DIFFICULTY_LABELS["easy"]]

PRETTY_NAMES = {
    "text-embedding-3-large": "OAI Large",
    "text-embedding-3-small": "OAI Small",
    "nomic-embed-text": "Nomic",
    "bge-large-en-v1.5": "BGE Large",
    "e5-mistral-7b-instruct": "E5-Mistral",
    "gte-Qwen2-7B-instruct": "GTE-Qwen2",
}


def _fake_summary() -> pl.DataFrame:
    rows = []
    for model in MODELS:
        # Give each model a stable baseline so the ordering is meaningful
        base = RNG.uniform(0.3, 0.75)
        for dataset in DATASETS:
            ds_offset = 0.05 if dataset == "welfare" else 0.0
            for difficulty in DIFFICULTIES:
                diff_offset = -0.12 if difficulty == DIFFICULTY_LABELS["hard"] else 0.0
                for statistic in STATISTICS:
                    stat_offsets = {"Best": 0.10, "Mean": 0.0, "Worst": -0.10}
                    mu = np.clip(
                        base + ds_offset + diff_offset + stat_offsets[statistic], 0.0, 1.0
                    )
                    ci_half = RNG.uniform(0.02, 0.06)
                    rows.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "difficulty": difficulty,
                            "statistic": statistic,
                            "auc_mean": float(mu),
                            "ci_lo": float(max(0.0, mu - ci_half)),
                            "ci_hi": float(min(1.0, mu + ci_half)),
                        }
                    )
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune difficulty dumbbell plot aesthetics")
    parser.add_argument("--out", default="plots", help="Output directory")
    parser.add_argument("--file-type", default="png", choices=["pdf", "png", "svg"])
    parser.add_argument("--top-n", type=int, default=6)
    parser.add_argument("--font-scale", type=float, default=1.0)
    args = parser.parse_args()

    plot_dir = Path(args.out)
    plot_dir.mkdir(parents=True, exist_ok=True)

    summary = _fake_summary()

    out = plot_difficulty_dumbbell(
        summary,
        plot_dir,
        pretty_names=PRETTY_NAMES,
        top_n=args.top_n,
        file_type=args.file_type,
        filename_prefix="tune_difficulty",
        font_scale=args.font_scale,
        title=""
    )
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
