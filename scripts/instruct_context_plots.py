"""Plot instruction-prompt robustness of instruct embedding models.

For each base instruct model we evaluate several *prompt variations* — different
natural-language phrasings of the same clustering/similarity task (see
``human_grounding.instruct_embed.PROMPT_VARIATIONS``). A faceted horizontal bar
chart (one panel per dataset) then shows the AUC per model, coloured by prompt
variation. Tight bars across variations = the model is robust to how the
instruction is phrased; wide spread = it is prompt-sensitive.

Because embeddings are cached by model name, each (base model, prompt variation)
pair is run under a *distinct, deterministic* variant name
(``{base}__prompt-{variant}``) so the caches never collide across variations.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import polars as pl
import seaborn as sns
from loguru import logger
from tqdm import tqdm

import human_grounding.evaluate
import human_grounding.threshold_auc
from human_grounding.constants import DATASET_PRETTY_NAMES, PRETTY_NAMES
from human_grounding.data import get_rai_demographics, get_welfare_demographics
from human_grounding.directories import OUTPUT_DIR, PLOT_DIR
from human_grounding.instruct_embed import (
    AVAILABLE_MODELS,
    PROMPT_VARIATIONS,
    parse_variant_name,
    variant_model_names,
)

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Embedding evaluation (mirrors neural_alignment_plots.get_embedding_alignments)
# ---------------------------------------------------------------------------


def _get_embedding_alignments(
    models: list[str], full_dataset: pl.DataFrame
) -> pl.DataFrame:
    results = []
    for model in tqdm(models, desc="Prompt variants"):
        comparisons = human_grounding.evaluate.evaluate_human_embedding_match(
            model=model, all_coordinates=full_dataset
        )
        results.append(comparisons)
    return pl.concat(results).drop("cause", "source", "size")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_prompt_robustness_auc_bar(
    group_auc: pl.DataFrame,
    plot_dir: Path,
    *,
    pretty_names: Mapping[str, str] | None = None,
    dataset_name_map: Mapping[str, str] | None = None,
    variant_name_map: Mapping[str, str] | None = None,
    font_scale: float = 1.35,
    ci: float = 95.0,
    height: float = 10.0,
    facet_width: float = 9.0,
    file_type: str = "pdf",
) -> Path:
    """Faceted horizontal AUC bar chart coloured by prompt variation.

    Parameters
    ----------
    group_auc:
        ``[model, dataset, demographics, iteration, auc]`` as returned by
        ``compute_threshold_auc``, where ``model`` is a variant name
        (``{base}__prompt-{variant}``).
    plot_dir:
        Directory for the output file.
    pretty_names:
        Optional base-model display mapping.
    dataset_name_map:
        Optional dataset label mapping.
    variant_name_map:
        Optional prompt-variant display mapping.
    font_scale:
        Seaborn font scale.
    ci:
        Confidence-interval width in percent (default 95).
    height:
        Height of each facet panel.
    facet_width:
        Width of each facet panel.
    file_type:
        Output format (``"pdf"``, ``"png"``, ``"jpg"``).

    Returns
    -------
    Path of the saved figure.
    """
    if pretty_names is None:
        pretty_names = {}
    if dataset_name_map is None:
        dataset_name_map = DATASET_PRETTY_NAMES
    if variant_name_map is None:
        variant_name_map = {v: v.capitalize() for v in PROMPT_VARIATIONS}

    # Tag each row with its base model and prompt variant.
    tagged_rows = []
    for row in group_auc.to_dicts():
        base, variant = parse_variant_name(row["model"])
        if variant is None:
            # Not a prompt-variant model; skip (e.g. stray base rows).
            continue
        tagged_rows.append(
            {
                "base_model": base,
                "variant": variant,
                "dataset": row["dataset"],
                "iteration": row["iteration"],
                "auc": row["auc"],
            }
        )

    tagged = pl.DataFrame(tagged_rows)

    # Per-(base_model, dataset, variant, iteration) mean AUC (averaging across
    # demographic groups, matching the "Mean" bar in plot_auc_bar).
    per_iter = tagged.group_by("base_model", "dataset", "variant", "iteration").agg(
        pl.col("auc").mean()
    )

    plot_data = per_iter.with_columns(
        pl.col("base_model").replace(pretty_names).alias("model"),
        pl.col("dataset").replace(dataset_name_map).alias("dataset"),
        pl.col("variant").replace(variant_name_map).alias("variant"),
    )

    # Order y-axis by mean AUC across all variations (best on top).
    model_order = (
        plot_data.group_by("model")
        .agg(pl.col("auc").mean())
        .sort("auc", descending=True)
        .get_column("model")
        .to_list()
    )

    pdf = plot_data.to_pandas()
    dataset_order = sorted(pdf["dataset"].unique())
    hue_order = [variant_name_map[v] for v in PROMPT_VARIATIONS]
    palette = dict(zip(hue_order, sns.color_palette("colorblind", len(hue_order))))

    sns.set_theme(style="whitegrid", font_scale=font_scale)

    g = sns.catplot(
        data=pdf,
        x="auc",
        y="model",
        hue="variant",
        col="dataset",
        col_order=dataset_order,
        hue_order=hue_order,
        order=model_order,
        kind="bar",
        palette=palette,
        height=height,
        width=0.8,
        aspect=facet_width / height,
        orient="h",
        errorbar=("ci", ci),
    )

    g.set_titles("{col_name}")
    g.set_axis_labels("", "")
    g.figure.supxlabel("Alignment AUC (normalised)")

    locator = mticker.MaxNLocator(nbins="auto", min_n_ticks=4)
    for ax in g.axes.flat:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}".lstrip("0") or "0")
        )

    sns.move_legend(
        g,
        "lower center",
        ncol=len(hue_order),
        frameon=False,
        bbox_to_anchor=(0.5, -0.1),
        title=None,
    )

    out_path = plot_dir / f"instruct_prompt_robustness_auc_bar.{file_type}"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close("all")
    logger.info(f"Saved instruct prompt-robustness AUC bar chart to {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    font_scale: float,
    use_cache: bool = False,
    file_type: str = "pdf",
) -> None:
    full_dataset = pl.read_csv(OUTPUT_DIR / "valid_coordinates.csv")
    # One deterministic model name per (base model, prompt variation).
    models = variant_model_names(AVAILABLE_MODELS)

    welfare_demographics = get_welfare_demographics()
    rai_demographics = get_rai_demographics()

    auc_output_path = OUTPUT_DIR / "instruct_prompt_robustness_auc.csv"

    if use_cache:
        auc_bootstraps = pl.read_csv(auc_output_path)
    else:
        combined_results = _get_embedding_alignments(models, full_dataset)
        auc_bootstraps, _ = human_grounding.threshold_auc.compute_threshold_auc(
            combined_results=combined_results,
            welfare_demographics=welfare_demographics,
            rai_demographics=rai_demographics,
            n_bootstrap=10,
        )
        auc_bootstraps.write_csv(auc_output_path)

    plot_prompt_robustness_auc_bar(
        auc_bootstraps,
        plot_dir=PLOT_DIR,
        pretty_names=PRETTY_NAMES,
        font_scale=font_scale,
        file_type=file_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate instruction prompt-robustness AUC plots",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scale", type=float, default=2.8, help="Font scale for the plots"
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Use cached AUC results instead of recomputing.",
    )
    parser.add_argument(
        "--file", type=str, choices=["pdf", "jpg", "png"], default="pdf"
    )
    args = parser.parse_args()

    main(args.scale, use_cache=args.cache, file_type=args.file)
