"""Report canonical values, organised by paper section.

This is a *reporter*, not a recomputation pipeline. Wherever a canonical value
already lives in a precomputed artifact under ``output/`` (bootstrap AUC tables,
cluster-consistency CSVs, alpha reliability curves, the LaTeX/txt summaries), we
load that artifact and read the value off it. The only numbers computed here are
cheap aggregations over already-materialised rows (means, ranks, group min/max
over precomputed per-bootstrap AUCs, descriptive counts). Nothing re-embeds text
or re-bootstraps.

The output (`output/canonical_values_report.md`) is laid out to mirror the paper
itself -- Abstract, Introduction, Section 3.x/4.x, figures, appendices, and a
final canonical headline block -- following the "values to report by paper
section" specification. Each slot is filled with a concrete value where the
source exists, or flagged with what is missing / where it comes from.

Narrative constants that are properties of the study design rather than of any
data file (participant counts, completion time, model counts) are taken from the
specification and marked "(paper spec)"; methods constants (threshold grid,
D_e, MDS dimensionality, clustering algorithm) are read from the code so they
stay in sync with the pipeline.

Usage::

    uv run scripts/report_canonical_values.py
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from human_grounding.constants import DATASET_PRETTY_NAMES, PRETTY_NAMES
from human_grounding.directories import DATA_DIR, OUTPUT_DIR
from human_grounding.oracle import DEFAULT_N_COMPONENTS, ORACLE_MODEL_NAME
from human_grounding.threshold_auc import HUMAN_MODEL_NAME

# Datasets belong to two "experiments"; the alignment/cluster CSVs are suffixed
# with the experiment name, so we need the mapping to reassemble per-dataset views.
EXPERIMENT_DATASETS: dict[str, list[str]] = {
    "policy": ["rai", "welfare"],
    "gov-ai": ["gov-ai"],
}
DATASETS = ["rai", "welfare", "gov-ai"]
REPORT_PATH = OUTPUT_DIR / "canonical_values_report.md"

# Reference upper bounds that must be excluded when picking the "best model".
NON_MODEL_ROWS = {HUMAN_MODEL_NAME, ORACLE_MODEL_NAME}

# LaTeX symbol snippets, kept as raw strings so the source stays pure ASCII
# (ruff's ambiguous-character rules flag Greek letters / the multiplication and
# minus glyphs) while the emitted Markdown still renders proper math.
ALPHA = r"$\alpha$"
RHO = r"$\rho$"
TAU = r"$\tau$"
DELTA = r"$\Delta$"
DELTA_GROUP = r"$\Delta_{\mathrm{group}}$"
DELTA_ADJ = r"$\Delta_{\mathrm{adj}}$"
DELTA_HUMAN = r"$\Delta_{\mathrm{human}}$"
TAU_MAX = r"$\tau_{\max}$"
TIMES = r"$\times$"

# Which protected groups actually enter the group analysis (welfare keeps only
# the parties with enough statements; the rest / missing are excluded).
WELFARE_ANALYSIS_PARTIES = {"1", "2", "3", "4", "11"}

# --- Narrative constants from the paper specification (design facts, not data) #
SPEC = {
    "n_datasets": 3,
    "n_studies": 2,
    "n_communities": 2,
    "participants_total": 12,
    "participants_per_panel": 6,
    "n_neural_models": 32,
    "n_lexical": 2,
    "n_total_reps": 34,
    "statements_per_round": 20,
    "policy_rounds_per_participant": 14,
    "govai_rounds_per_participant": 7,
    "total_placements_approx": 2640,
    "max_rater_pairs_per_panel": 15,
    "policy_completion": "half-day workshop",
    "govai_completion": "1-1.5 hours",
    "n_instruction_variants": 5,
    "n_conditions": 2,
}

# --- Methods constants read from the pipeline code -------------------------- #
METHODS = {
    "main_d_max": 6.5,
    "main_n_points": 30,
    "main_scheme": "log-x",
    "alt_d_max": [4.0, 6.5, 8.0, 10.0],
    "alt_n_points": [15, 50],
    "rq1_d_min": 1.0,
    "rq1_d_max": 8.0,
    "rq1_n_points": 50,
    "D_e": 0.5,
    "auc_norm": "[-1, 1]",
    "distance_metric": "cosine distance",
    "aggregation": "Borda count",
    "rank_metric": "Spearman rho",
    "mds": "metric MDS (SMACOF)",
    "mds_dim": DEFAULT_N_COMPONENTS,
    "mds_n_init": 4,
    "cluster_algo": "Ward-linkage agglomerative clustering",
    "cluster_k": "mean number of human clusters per round",
    "cluster_bootstrap": 2000,
}


# --------------------------------------------------------------------------- #
# Formatting / loading helpers
# --------------------------------------------------------------------------- #
def pretty_dataset(name: str) -> str:
    return DATASET_PRETTY_NAMES.get(name, name)


def pretty_model(name: str) -> str:
    return PRETTY_NAMES.get(name, name)


def read_csv(name: str, dir_: Path = OUTPUT_DIR) -> pl.DataFrame | None:
    path = dir_ / name
    return pl.read_csv(path) if path.exists() else None


def read_text(name: str, dir_: Path = OUTPUT_DIR) -> str | None:
    path = dir_ / name
    return path.read_text() if path.exists() else None


def parse_latex_rows(tex: str) -> list[list[str]]:
    """Return the body rows of a LaTeX tabular (between the last ``\\midrule`` and
    the following ``\\bottomrule``), split on ``&`` with escapes stripped."""
    if r"\midrule" not in tex or r"\bottomrule" not in tex:
        return []
    body = tex.split(r"\midrule")[-1].split(r"\bottomrule")[0]
    rows = []
    for raw in body.split(r"\\"):
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        cells = []
        for raw_cell in line.split("&"):
            cell = raw_cell.strip()
            cell = cell.replace(r"\%", "%").replace(r"\,", "")
            cell = (
                cell.replace(r"\mathrm", "")
                .replace(r"\max", "max")
                .replace(r"\min", "min")
            )
            cell = re.sub(r"\\[a-zA-Z]+\{?", "", cell)  # drop remaining \commands
            cell = cell.replace("{", "").replace("}", "").replace("$", "")
            cells.append(cell.strip())
        if any(cells):
            rows.append(cells)
    return rows


# Coverage-table display names -> internal dataset keys.
COVERAGE_NAME_TO_DS = {"Gov-AI": "gov-ai", "RAI": "rai", "Welfare": "welfare"}


class Section:
    """Accumulates Markdown lines for one paper section."""

    def __init__(self, title: str) -> None:
        self.lines: list[str] = [f"## {title}", ""]

    def line(self, text: str = "") -> None:
        self.lines.append(text)

    def bullet(self, text: str) -> None:
        self.lines.append(f"- {text}")

    def missing(self, what: str, how: str) -> None:
        self.lines.append(f"> **Not in `output/`** — {what}: {how}")
        self.lines.append("")

    def table(self, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
        if self.lines and self.lines[-1] != "":
            self.lines.append("")  # markdown needs a blank line before a table
        header = list(header)
        self.lines.append("| " + " | ".join(str(h) for h in header) + " |")
        self.lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows:
            self.lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self.lines.append("")

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"


# --------------------------------------------------------------------------- #
# Shared data loaders / derived facts
# --------------------------------------------------------------------------- #
def load_alignment_bootstraps() -> pl.DataFrame | None:
    """Per-model / per-demographic / per-iteration alignment AUCs, incl. the
    ``Human`` baseline and the ``human-mds-oracle`` rows."""
    return read_csv("alignment_results_gov-ai_policy.csv")


def load_coordinates() -> pl.DataFrame | None:
    frames = []
    for fname in ("valid_coordinates.csv", "govai_coordinates.csv"):
        df = read_csv(fname, DATA_DIR)
        if df is not None:
            frames.append(
                df.select("statement_id", "dataset", "size", "seed", "user_id")
            )
    return pl.concat(frames, how="vertical_relaxed") if frames else None


def dataset_model_means(bootstraps: pl.DataFrame) -> pl.DataFrame:
    """Mean AUC per (dataset, model): average per demographic group, then across
    groups (the "mean across demographics" convention)."""
    per_group = bootstraps.group_by("dataset", "model", "demographics").agg(
        pl.col("auc").mean()
    )
    return per_group.group_by("dataset", "model").agg(
        pl.col("auc").mean().alias("mean_auc")
    )


def parse_alignment_summary() -> dict[str, dict[str, str]]:
    """Parse ``alignment_summary_gov-ai_policy.txt`` into per-dataset facts.

    Returns ``{dataset: {human, human_ci, best_model, best, best_ci, gap,
    gap_ci, oracle, oracle_ci, oracle_gap, oracle_gap_ci}}`` as display strings.
    This txt is the authoritative source for the CI'd headline values.
    """
    txt = read_text("alignment_summary_gov-ai_policy.txt")
    if txt is None:
        return {}
    facts: dict[str, dict[str, str]] = {}
    current: str | None = None
    ci = r"\[([^\]]+)\]"
    for line in txt.splitlines():
        m = re.match(r"^##\s+(\S+)", line)
        if m:
            current = m.group(1)
            facts[current] = {}
            continue
        if current is None:
            continue
        f = facts[current]
        if m := re.search(rf"Human:\s+([\d.]+)\s+{ci}", line):
            f["human"], f["human_ci"] = m.group(1), m.group(2)
        elif m := re.search(rf"Best model:\s+(\S+)\s+([\d.]+)\s+{ci}", line):
            f["best_model"], f["best"], f["best_ci"] = (
                m.group(1),
                m.group(2),
                m.group(3),
            )
        elif m := re.search(rf"Gap \(oracle\):\s+([\d.]+pp)\s+{ci}", line):
            f["oracle_gap"], f["oracle_gap_ci"] = m.group(1), m.group(2)
        elif m := re.search(rf"Gap:\s+([\d.]+pp)\s+{ci}", line):
            f["gap"], f["gap_ci"] = m.group(1), m.group(2)
        elif m := re.search(rf"Oracle:\s+([\d.]+)\s+{ci}", line):
            f["oracle"], f["oracle_ci"] = m.group(1), m.group(2)
    return facts


def group_conditional(
    bootstraps: pl.DataFrame, dataset: str, model: str
) -> pl.DataFrame:
    """Per-demographic mean AUC for one (dataset, model)."""
    return (
        bootstraps.filter((pl.col("dataset") == dataset) & (pl.col("model") == model))
        .group_by("demographics")
        .agg(pl.col("auc").mean().alias("auc"))
        .sort("demographics")
    )


def best_model_for(means: pl.DataFrame, dataset: str) -> str | None:
    sub = means.filter(
        (pl.col("dataset") == dataset) & (~pl.col("model").is_in(list(NON_MODEL_ROWS)))
    ).sort("mean_auc", descending=True)
    return None if sub.is_empty() else sub.row(0, named=True)["model"]


# --------------------------------------------------------------------------- #
# Instruction-robustness helpers
# --------------------------------------------------------------------------- #
def instruct_summary(df: pl.DataFrame, split_context: bool) -> pl.DataFrame:
    """Mean AUC per (base model, variant[, context], dataset). Variant names are
    encoded as ``<base>__prompt-<variant>[__ctx-<context>]``."""
    parsed = df.with_columns(pl.col("model").str.split("__").alias("_parts"))
    parsed = parsed.with_columns(
        pl.col("_parts").list.get(0).alias("base_model"),
        pl.col("_parts")
        .list.eval(pl.element().filter(pl.element().str.starts_with("prompt-")))
        .list.first()
        .str.replace("prompt-", "")
        .alias("variant"),
        pl.col("_parts")
        .list.eval(pl.element().filter(pl.element().str.starts_with("ctx-")))
        .list.first()
        .str.replace("ctx-", "")
        .alias("context"),
    )
    keys = ["base_model", "variant", "dataset"] + (["context"] if split_context else [])
    return parsed.group_by(keys).agg(pl.col("auc").mean().alias("auc"))


# --------------------------------------------------------------------------- #
# Abstract
# --------------------------------------------------------------------------- #
def section_abstract(facts: dict, spearman: pl.DataFrame | None) -> str:
    s = Section("Abstract")
    s.line("Headline values only (no CIs, no group disparities).")
    s.line()
    if facts:
        s.line("**Best-model gap to human-human agreement:**")
        for ds in DATASETS:
            f = facts.get(ds, {})
            if "gap" in f:
                s.bullet(f"{pretty_dataset(ds)}: {f['gap']}")
        s.line()
        per_ds = [
            f"{pretty_dataset(d)} {facts[d]['oracle_gap']}"
            for d in DATASETS
            if facts.get(d, {}).get("oracle_gap")
        ]
        gap_vals = sorted(
            _pp(facts[d].get("oracle_gap"))
            for d in DATASETS
            if facts.get(d, {}).get("oracle_gap")
        )
        if per_ds:
            detail = ", ".join(per_ds)
            s.line(
                f"**Best-model gap to full-panel oracle:** ranges "
                f"{gap_vals[0]:.1f}-{gap_vals[-1]:.1f}pp across datasets ({detail})."
            )
            s.line()
    else:
        s.missing("headline gaps", "alignment_summary_gov-ai_policy.txt not found")

    if spearman is not None:
        for source, label in (
            ("OurExercise", "Grounding-to-clustering rank correlation"),
            ("MMTEB", "MMTEB-to-clustering rank correlation (if space permits)"),
        ):
            row = spearman.filter(
                (pl.col("source") == source) & (pl.col("experiment") == "policy")
            )
            if not row.is_empty():
                r = row.row(0, named=True)
                s.bullet(
                    f"**{label}:** Spearman {RHO} = {r['spearman']:.2f} "
                    f"(policy; gov-ai {RHO} = "
                    + _spear_val(spearman, source, "gov-ai")
                    + ")."
                )
        s.line()
    return s.render()


def _spear_val(spearman: pl.DataFrame, source: str, experiment: str) -> str:
    row = spearman.filter(
        (pl.col("source") == source) & (pl.col("experiment") == experiment)
    )
    return "n/a" if row.is_empty() else f"{row.row(0, named=True)['spearman']:.2f}"


# --------------------------------------------------------------------------- #
# Introduction / Contributions
# --------------------------------------------------------------------------- #
def section_intro(facts: dict, spearman: pl.DataFrame | None) -> str:
    s = Section("Introduction / Contributions")
    s.table(
        ["Quantity", "Value"],
        [
            ["Datasets", SPEC["n_datasets"]],
            ["Studies", SPEC["n_studies"]],
            ["Participants (total)", SPEC["participants_total"]],
            ["Participants per study", SPEC["participants_per_panel"]],
            ["Neural embedding models", SPEC["n_neural_models"]],
            ["Lexical baselines", SPEC["n_lexical"]],
        ],
    )
    s.line("_(design constants — paper spec)_")
    s.line()
    if facts:
        gaps = [
            _pp(facts[d].get("gap")) for d in DATASETS if facts.get(d, {}).get("gap")
        ]
        ogaps = [
            _pp(facts[d].get("oracle_gap"))
            for d in DATASETS
            if facts.get(d, {}).get("oracle_gap")
        ]
        if gaps:
            s.bullet(
                f"Headline human-model gap range: {min(gaps):.1f}-{max(gaps):.1f}pp."
            )
        if ogaps:
            s.bullet(
                f"Headline oracle-model gap range: {min(ogaps):.1f}-{max(ogaps):.1f}pp."
            )
    if spearman is not None:
        s.bullet(
            f"Headline grounding-to-clustering correlation: {RHO} = "
            f"{_spear_val(spearman, 'OurExercise', 'policy')} (policy)."
        )
    s.line()
    return s.render()


def _pp(text: str | None) -> float:
    return float(text.replace("pp", "")) if text else float("nan")


# --------------------------------------------------------------------------- #
# Section 3.1 — Exercise Description
# --------------------------------------------------------------------------- #
def section_3_1(coverage_rows: list[list[str]]) -> str:
    s = Section("Section 3.1 — Exercise Description")
    s.table(
        ["Quantity", "Value", "Source"],
        [
            ["Statements per round", SPEC["statements_per_round"], "paper spec"],
            ["Participants per panel", SPEC["participants_per_panel"], "paper spec"],
            [
                "Policy rounds per participant",
                SPEC["policy_rounds_per_participant"],
                "paper spec",
            ],
            [
                "Gov-AI rounds per participant",
                SPEC["govai_rounds_per_participant"],
                "paper spec",
            ],
            ["Completion time (policy)", SPEC["policy_completion"], "paper spec"],
            ["Completion time (Gov-AI)", SPEC["govai_completion"], "paper spec"],
            [
                "Total individual statement placements",
                f"~{SPEC['total_placements_approx']:,}",
                "paper spec",
            ],
        ],
    )
    if coverage_rows:
        s.line(
            "**Mean statement occurrences per included statement, by dataset** "
            "(`statement_coverage_table.tex`):"
        )
        s.line()
        s.table(
            ["Dataset", "Mean occurrences", "Median occurrences"],
            [[r[0], r[2], r[3]] for r in coverage_rows],
        )
        s.line(
            "> Minimum occurrence per included statement is not tabulated in the "
            "coverage artifact; regenerate from `raw_triplets_gov-ai_policy.parquet` "
            "if a guaranteed floor must be stated."
        )
        s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Section 3.2 — Datasets and Protected Attributes
# --------------------------------------------------------------------------- #
def section_3_2(coverage_rows: list[list[str]]) -> str:
    s = Section("Section 3.2 — Datasets and Protected Attributes")

    # source statements + group counts
    source_counts, group_counts, missing_counts = {}, {}, {}
    try:
        from human_grounding.data import (
            get_govai,
            get_rai_demographics,
            get_responsible_ai,
            get_welfare,
            get_welfare_demographics,
        )

        source_counts = {
            "rai": get_responsible_ai().select("cause_id").n_unique(),
            "welfare": get_welfare().select("cause_id").n_unique(),
            "gov-ai": get_govai().select("cause_id").n_unique(),
        }
        rai_g = (
            get_rai_demographics("gender")
            .group_by("demographic")
            .agg(pl.col("cause_id").n_unique().alias("n"))
        )
        group_counts["rai"] = {
            r["demographic"]: r["n"]
            for r in rai_g.to_dicts()
            if r["demographic"] in {"Kvinde", "Mand"}
        }
        missing_counts["rai"] = next(
            (r["n"] for r in rai_g.to_dicts() if r["demographic"] == "Unknown"), 0
        )
        wf_g = (
            get_welfare_demographics()
            .group_by("demographic")
            .agg(pl.col("cause_id").n_unique().alias("n"))
        )
        group_counts["welfare"] = {
            r["demographic"]: r["n"]
            for r in wf_g.to_dicts()
            if r["demographic"] in WELFARE_ANALYSIS_PARTIES
        }
        missing_counts["welfare"] = sum(
            r["n"]
            for r in wf_g.to_dicts()
            if r["demographic"] not in WELFARE_ANALYSIS_PARTIES
        )
    except Exception as exc:  # raw corpus files are optional
        s.missing("source statements / group counts", f"raw corpus load failed: {exc}")

    used = {COVERAGE_NAME_TO_DS.get(r[0], r[0]): r[1] for r in coverage_rows}
    protected = {
        "rai": "Gender",
        "welfare": "Political party (pseudonymised)",
        "gov-ai": "None",
    }
    rows = []
    for ds in DATASETS:
        gc = group_counts.get(ds)
        gc_str = (
            "; ".join(f"{k}={v}" for k, v in sorted(gc.items()))
            if gc
            else "n/a (no split)"
        )
        rows.append(
            [
                pretty_dataset(ds),
                f"{source_counts.get(ds, 'n/a'):,}" if ds in source_counts else "n/a",
                used.get(ds, "n/a"),
                "Danish",
                protected[ds],
                gc_str,
            ]
        )
    s.table(
        [
            "Dataset",
            "Source statements",
            "Used statements",
            "Language",
            "Protected attribute",
            "Group counts (analysis groups)",
        ],
        rows,
    )
    for ds in ("rai", "welfare"):
        if ds in missing_counts:
            s.bullet(
                f"{pretty_dataset(ds)}: {missing_counts[ds]} statements in "
                "missing/excluded groups (dropped from group analysis)."
            )
    s.bullet("Gov-AI: no protected-attribute analysis is performed.")
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.1.1 — Human Reliability Methods
# --------------------------------------------------------------------------- #
def section_4_1_1(coords: pl.DataFrame | None) -> str:
    s = Section("Section 4.1.1 — Human Reliability Methods")
    s.line("Definitions and configuration (not results):")
    s.line()
    s.line(
        r"Separation ratio: $r_i(t) = \max\{\delta_i(a,b), \delta_i(a,c)\} / "
        r"\min\{\delta_i(a,b), \delta_i(a,c)\}$; threshold notation " + TAU + "."
    )
    s.bullet(r"Human-human filtering: both raters satisfy $r_i(t) \geq \tau$.")
    s.bullet(r"Model-human filtering: the human rater satisfies $r_i(t) \geq \tau$.")
    s.table(
        ["Setting", "Value"],
        [
            ["Expected disagreement D_e", METHODS["D_e"]],
            ["AUC normalization range", METHODS["auc_norm"]],
            ["AUC integration scale", "log-" + TAU],
            [
                "Threshold grid (RQ1 curve)",
                f"log-spaced, {METHODS['rq1_n_points']} points, "
                rf"$\tau \in [{METHODS['rq1_d_min']:.0f}, {METHODS['rq1_d_max']:.0f}]$",
            ],
            [
                "Threshold grid (main AUC)",
                f"log-spaced, {METHODS['main_n_points']} points, "
                rf"{TAU_MAX} = {METHODS['main_d_max']}",
            ],
            ["Log-spaced thresholds", "yes"],
        ],
    )
    # Exact number of contributing rater pairs per dataset (between-rater = share a round).
    if coords is not None:
        rows = []
        for ds, sub in coords.group_by("dataset"):
            raters = sub.select("seed", "user_id").unique()
            pairs = raters.join(raters, on="seed", suffix="_b").filter(
                pl.col("user_id") < pl.col("user_id_b")
            )
            rows.append(
                [
                    pretty_dataset(ds[0]),
                    pairs.select("user_id", "user_id_b").unique().height,
                ]
            )
        s.line(
            "**Contributing (between-rater) pairs per dataset** "
            "(from coordinate files; lower bound vs the full raw-triplet parquet):"
        )
        s.line()
        s.table(["Dataset", "Rater pairs"], sorted(rows))
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.1.2 — Human Reliability Results
# --------------------------------------------------------------------------- #
def _alpha_frames() -> pl.DataFrame | None:
    frames = [
        df
        for f in (
            "alpha_data_policy_demographic.csv",
            "alpha_data_gov-ai_demographic.csv",
        )
        if (df := read_csv(f)) is not None
    ]
    return pl.concat(frames, how="vertical_relaxed") if frames else None


def _tau_at_alpha(
    alpha: pl.DataFrame, target: float = 0.8
) -> dict[tuple[str, str], tuple[float, float]]:
    curve = (
        alpha.filter(pl.col("group_type") == "dataset")
        .group_by("group_name", "reliability_type", "d")
        .agg(pl.col("krippendorf").mean().alias("alpha"))
    )
    out = {}
    for (ds, rel), grp in curve.group_by("group_name", "reliability_type"):
        best = (
            grp.with_columns((pl.col("alpha") - target).abs().alias("gap"))
            .sort("gap")
            .row(0, named=True)
        )
        out[(ds, rel)] = (best["d"], best["alpha"])
    return out


def section_4_1_2(
    bootstraps: pl.DataFrame | None,
    facts: dict,
    triplet_totals: dict[str, list[int]],
    drift: pl.DataFrame | None,
) -> str:
    s = Section("Section 4.1.2 — Human Reliability Results")

    # Per-dataset human-human AUC + CI (authoritative from summary), within-rater, tau@0.8.
    alpha = _alpha_frames()
    tau08 = _tau_at_alpha(alpha) if alpha is not None else {}
    rows = []
    for ds in DATASETS:
        f = facts.get(ds, {})
        within = "n/a"
        if drift is not None:
            wr = drift.filter(
                (pl.col("dataset") == ds)
                & pl.col("comparison").str.contains("(?i)within")
            )
            if not wr.is_empty():
                r = wr.row(0, named=True)
                within = f"{r['mean']:.3f} [{r['lower']:.3f}, {r['upper']:.3f}]"
        tw = tau08.get((ds, "within"))
        tb = tau08.get((ds, "between"))
        tau_str = (rf"between $\tau \approx {tb[0]:.2f}$" if tb else "") + (
            rf"; within $\tau \approx {tw[0]:.2f}$" if tw else ""
        )
        rows.append(
            [
                pretty_dataset(ds),
                f.get("human", "n/a"),
                f"[{f['human_ci']}]" if "human_ci" in f else "n/a",
                within,
                tau_str or "n/a",
            ]
        )
    s.line(
        "Per-dataset human reliability (human-human AUC + 95% CI from the alignment "
        "summary; within-rater drift summary from `context_drift_comparison.csv`; "
        + TAU
        + " where "
        + ALPHA
        + r" $\approx 0.8$ from the alpha curves):"
    )
    s.line()
    s.table(
        [
            "Dataset",
            "Human-human AUC",
            "95% CI",
            "Within-rater drift [95% CI]",
            rf"{TAU} @ {ALPHA} $\approx 0.8$",
        ],
        rows,
    )

    # Retained triplets: counts + % at d=1 and d=4, min group at d=4.
    if triplet_totals:
        s.line(
            r"**Retained triplets** (from `fairness_triplet_counts.tex`; $d=1 \approx$ "
            "all eligible, d=4 a mid/high threshold):"
        )
        s.line()
        s.table(
            ["Dataset", "Eligible (d=1)", "Retained d=2 (%)", "Retained d=4 (%)"],
            [
                [
                    pretty_dataset(ds),
                    f"{v[0]:,}",
                    f"{v[1]:,} ({100 * v[1] / v[0]:.1f}%)",
                    f"{v[2]:,} ({100 * v[2] / v[0]:.1f}%)",
                ]
                for ds, v in triplet_totals.items()
            ],
        )
        s.line(
            "> Gov-AI has no demographic split, so it is absent from the group "
            "triplet-count artifact."
        )
        s.line()

    # Group-conditional human reliability AUC + disparity.
    if bootstraps is not None:
        human = bootstraps.filter(pl.col("model") == HUMAN_MODEL_NAME)
        grows = []
        for ds in DATASETS:
            gc = group_conditional(human, ds, HUMAN_MODEL_NAME)
            if gc.is_empty():
                continue
            lo, hi = gc["auc"].min(), gc["auc"].max()
            detail = "; ".join(
                f"{r['demographics']}={r['auc']:.3f}" for r in gc.to_dicts()
            )
            grows.append(
                [pretty_dataset(ds), detail, f"{lo:.3f}", f"{hi:.3f}", f"{hi - lo:.3f}"]
            )
        s.line(
            "**Group-conditional human reliability AUC** ("
            + DELTA_HUMAN
            + r" = $\alpha_{\max} - \alpha_{\min}$):"
        )
        s.line()
        s.table(
            [
                "Dataset",
                "Group AUCs",
                "Min group",
                "Max group",
                DELTA_HUMAN + " disparity",
            ],
            grows,
        )
        s.line(
            "> Per-group human bootstrap CIs are not persisted separately; regenerate "
            "with `scripts/alpha_distance_plot.py` if group-level CIs are required."
        )
        s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.2.1 — Neural-Human Alignment Methods
# --------------------------------------------------------------------------- #
def section_4_2_1(bootstraps: pl.DataFrame | None) -> str:
    s = Section("Section 4.2.1 — Neural-Human Alignment Methods")
    n_iter = (
        bootstraps.select("iteration").n_unique() if bootstraps is not None else "n/a"
    )
    s.table(
        ["Setting", "Value"],
        [
            ["Neural models", SPEC["n_neural_models"]],
            ["Lexical baselines", SPEC["n_lexical"]],
            ["Total representations evaluated", SPEC["n_total_reps"]],
            ["Model distance", METHODS["distance_metric"]],
            ["Instruction variants", SPEC["n_instruction_variants"]],
            [
                "Instruction formulations",
                f"{SPEC['n_conditions']} (generic, dataset-context)",
            ],
            ["Rank aggregation", METHODS["aggregation"]],
            ["Rank-comparison metric", METHODS["rank_metric"]],
            ["Bootstrap iterations (cached)", n_iter],
        ],
    )
    s.line(
        "For normalized vectors, cosine distance is monotone in squared Euclidean "
        "distance, so triplet orderings are equivalent."
    )
    s.line()
    s.line(
        "**Oracle construction:** full-panel consensus dissimilarity (mean of "
        "per-rater normalized layout distances, missing pairs mean-filled), then "
        f"{METHODS['mds']} at dimensionality q = {METHODS['mds_dim']} "
        f"(capped at n_statements - 1), n_init = {METHODS['mds_n_init']}. The oracle "
        "embedding is then evaluated through the identical triplet-" + ALPHA + " / AUC "
        "pipeline as the neural models."
    )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.2.2 — Neural-Human Alignment Results
# --------------------------------------------------------------------------- #
def section_4_2_2(
    means: pl.DataFrame | None,
    facts: dict,
    mmteb_spearman: dict,
    instruct: pl.DataFrame | None,
    context: pl.DataFrame | None,
    oracle_auc: dict,
) -> str:
    s = Section("Section 4.2.2 — Neural-Human Alignment Results")

    # Per-dataset headline table.
    rows = []
    for ds in DATASETS:
        f = facts.get(ds, {})
        human = float(f["human"]) if "human" in f else None
        oracle = float(f["oracle"]) if "oracle" in f else None
        best = float(f["best"]) if "best" in f else None
        odiff = (
            f"{(oracle - human) * 100:+.1f}pp"
            if (oracle is not None and human is not None)
            else "n/a"
        )
        rows.append(
            [
                pretty_dataset(ds),
                f.get("human", "n/a"),
                f.get("oracle", "n/a"),
                f.get("best", "n/a"),
                f"[{f['best_ci']}]" if "best_ci" in f else "n/a",
                pretty_model(f.get("best_model", "n/a")),
                f.get("gap", "n/a"),
                f.get("oracle_gap", "n/a"),
                odiff,
            ]
        )
    s.table(
        [
            "Dataset",
            "Human AUC",
            "Oracle AUC",
            "Best-model AUC",
            "Best 95% CI",
            "Best model",
            "Human-model gap",
            "Oracle-model gap",
            "Oracle-human diff",
        ],
        rows,
    )

    # Overall best / second-best by mean AUC across datasets; per-dataset score range.
    if means is not None:
        overall = (
            means.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS)))
            .group_by("model")
            .agg(pl.col("mean_auc").mean().alias("overall_auc"))
            .sort("overall_auc", descending=True)
        )
        top = overall.head(2).to_dicts()
        if len(top) >= 2:
            s.bullet(
                f"Best model overall: {pretty_model(top[0]['model'])} "
                f"({top[0]['overall_auc']:.3f} mean across datasets); "
                f"second-best: {pretty_model(top[1]['model'])} "
                f"({top[1]['overall_auc']:.3f})."
            )
        for ds in DATASETS:
            ds_models = means.filter(
                (pl.col("dataset") == ds)
                & (~pl.col("model").is_in(list(NON_MODEL_ROWS)))
            )
            if not ds_models.is_empty():
                s.bullet(
                    f"{pretty_dataset(ds)} score range across models: "
                    f"{ds_models['mean_auc'].min():.3f} to "
                    f"{ds_models['mean_auc'].max():.3f}."
                )
        s.line()

    # MMTEB vs grounded rank Spearman (+ p, best grounded model's MMTEB rank).
    if mmteb_spearman:
        s.line("**MMTEB rank vs grounded rank:**")
        s.line()
        s.table(
            [
                "Experiment",
                "Spearman " + RHO,
                "p-value",
                "Shared models",
                "Best grounded model",
                "Its MMTEB rank",
            ],
            [
                [
                    exp,
                    f"{v['rho']:.3f}",
                    f"{v['p']:.4f}",
                    v["n"],
                    pretty_model(v["best_model"]),
                    v["best_rank"],
                ]
                for exp, v in mmteb_spearman.items()
            ],
        )

    # Instruction robustness summary.
    if instruct is not None:
        summ = instruct
        deltas = summ.get_column("delta_default").drop_nulls()
        if len(deltas) > 0:
            s.bullet(
                f"Best improvement over default: {deltas.max():+.3f}; "
                f"worst change: {deltas.min():+.3f}; "
                f"mean |change|: {deltas.abs().mean():.3f} across "
                f"{summ.height} model{TIMES}variant{TIMES}dataset cells."
            )
    if context is not None:
        # generic vs dataset-context: how often context wins.
        wide = context.pivot(
            values="auc", index=["base_model", "dataset", "variant"], on="context"
        )
        ctx_cols = [
            c
            for c in wide.columns
            if c not in ("base_model", "dataset", "variant", "generic")
        ]
        if "generic" in wide.columns and ctx_cols:
            ctx_col = ctx_cols[0] if len(ctx_cols) == 1 else None
            wins = 0
            total = 0
            for r in wide.to_dicts():
                gen = r.get("generic")
                # dataset-context value is whichever non-generic column is populated
                ctx_val = next((r[c] for c in ctx_cols if r.get(c) is not None), None)
                if gen is not None and ctx_val is not None:
                    total += 1
                    wins += int(ctx_val > gen)
            if total:
                s.bullet(
                    f"Dataset-context beats generic in {wins}/{total} conditions "
                    f"({100 * wins / total:.0f}%)."
                )
    # Remaining oracle gap under the best instruction (min over model x dataset).
    if instruct is not None and oracle_auc:
        best_by_cell = instruct.group_by("base_model", "dataset").agg(
            pl.col("auc").max().alias("best_variant_auc")
        )
        gaps = []
        for r in best_by_cell.to_dicts():
            o = oracle_auc.get(r["dataset"])
            if o is not None:
                gaps.append((r["base_model"], r["dataset"], o - r["best_variant_auc"]))
        if gaps:
            gaps.sort(key=lambda t: t[2])
            m, ds, g = gaps[0]
            s.bullet(
                f"Smallest remaining oracle gap under best instruction: "
                f"{pretty_model(m)} on {pretty_dataset(ds)} = {g * 100:.1f}pp."
            )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Figure 4 — Main Alignment Figure
# --------------------------------------------------------------------------- #
def section_figure4(
    bootstraps: pl.DataFrame | None, means: pl.DataFrame | None, facts: dict
) -> str:
    s = Section("Figure 4 — Main Alignment Figure")
    if bootstraps is None or means is None:
        s.missing("figure bars", "alignment_results_gov-ai_policy.csv not found")
        return s.render()
    rows = []
    for ds in DATASETS:
        f = facts.get(ds, {})
        best_model = f.get("best_model") or best_model_for(means, ds)
        gc = (
            group_conditional(bootstraps, ds, best_model)
            if best_model
            else pl.DataFrame()
        )
        best_grp = f"{gc['auc'].max():.3f}" if not gc.is_empty() else "n/a"
        worst_grp = f"{gc['auc'].min():.3f}" if not gc.is_empty() else "n/a"
        rows.append(
            [
                pretty_dataset(ds),
                f.get("human", "n/a"),
                f.get("oracle", "n/a"),
                f.get("best", "n/a"),
                best_grp,
                worst_grp,
            ]
        )
    s.table(
        [
            "Dataset",
            "Human-human AUC",
            "Oracle-human AUC",
            "Best-model AUC",
            "Best group AUC",
            "Worst group AUC",
        ],
        rows,
    )
    s.line(
        "Caption should state: human bars = human-human agreement; oracle bars = "
        "oracle-human agreement; model bars = model-human agreement."
    )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Group-Conditional Alignment Results
# --------------------------------------------------------------------------- #
def section_group_conditional(
    bootstraps: pl.DataFrame | None,
    means: pl.DataFrame | None,
    group_stmt_counts: dict,
    triplet_group_rows: list[list[str]],
) -> str:
    s = Section("Group-Conditional Alignment Results")
    if bootstraps is None or means is None:
        s.missing("group-conditional AUCs", "alignment_results not found")
        return s.render()

    # Raw + adjusted disparity from fairness tables.
    controlled = read_text("fairness_group_gap_controlled.tex")
    ctrl_rows = parse_latex_rows(controlled) if controlled else []
    ctrl_by_ds = {r[0]: r for r in ctrl_rows}  # pretty dataset -> row

    for ds in ("rai", "welfare"):
        best_model = best_model_for(means, ds)
        gc = group_conditional(bootstraps, ds, best_model)
        if gc.is_empty():
            continue
        lo, hi = gc["auc"].min(), gc["auc"].max()
        s.line(f"**{pretty_dataset(ds)}** (best model: {pretty_model(best_model)}):")
        s.line()
        gsc = group_stmt_counts.get(ds, {})
        s.table(
            ["Group", "Best-model AUC", "Statements in group"],
            [
                [
                    r["demographics"],
                    f"{r['auc']:.3f}",
                    gsc.get(r["demographics"], "n/a"),
                ]
                for r in gc.to_dicts()
            ],
        )
        s.bullet(
            f"Raw group disparity {DELTA_GROUP} = {hi - lo:.3f} "
            f"(max {hi:.3f} - min {lo:.3f})."
        )
        ctrl = ctrl_by_ds.get(pretty_dataset(ds))
        if ctrl and len(ctrl) >= 6:
            # cols: Dataset, Model, Delta_group [CI], p, Delta_adj [CI], p
            s.bullet(
                f"Unadjusted {DELTA_GROUP} (fairness bootstrap) = {ctrl[2]}, p {ctrl[3]}."
            )
            s.bullet(
                f"Adjusted {DELTA_ADJ} (length + lexical controls) = {ctrl[4]}, p {ctrl[5]}."
            )
        s.line()

    if triplet_group_rows:
        s.line(
            "**Retained triplets per group** (d = 1, 2, 4; from "
            "`fairness_triplet_counts.tex`):"
        )
        s.line()
        s.table(["Dataset", "Group", "d=1", "d=2", "d=4"], triplet_group_rows)
    return s.render()


# --------------------------------------------------------------------------- #
# Qualitative Error Analysis
# --------------------------------------------------------------------------- #
def section_qualitative() -> str:
    s = Section("Qualitative Error Analysis")
    s.missing(
        "high-separation error counts / categories",
        "no persisted artifact — this analysis is not produced by any script in "
        "`output/`. Report manually: selection threshold (e.g. r_i(t) > 13), total "
        "inspected examples, model name, and 2-3 representative cases.",
    )
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.3.1 — Clustering Methods
# --------------------------------------------------------------------------- #
def section_4_3_1() -> str:
    s = Section("Section 4.3.1 — Clustering Methods")
    s.table(
        ["Setting", "Value"],
        [
            ["Clustering algorithm", METHODS["cluster_algo"]],
            ["Model distance representation", "embedding vectors (Ward on Euclidean)"],
            ["Number of clusters K", METHODS["cluster_k"]],
            ["Human reference", "ARI between paired participants (same round)"],
            ["Model reference", "mean ARI against both participants"],
            ["Bootstrap resamples (Spearman CIs)", METHODS["cluster_bootstrap"]],
            [
                "Models included",
                f"{SPEC['n_neural_models']} neural + {SPEC['n_lexical']} lexical",
            ],
        ],
    )
    s.line(
        "> Singleton-cluster treatment and the human cluster-count threshold "
        "procedure are implemented in `scripts/clustering.py`; state them from the "
        "code if the methods paragraph needs the exact rule."
    )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Section 4.3.2 — Clustering Results
# --------------------------------------------------------------------------- #
def _human_ari() -> dict[str, dict]:
    out = {}
    for exp in EXPERIMENT_DATASETS:
        hc = read_csv(f"human_cluster_consistency_{exp}.csv")
        if hc is None:
            continue
        for ds, sub in hc.group_by("dataset"):
            out[ds[0]] = {
                "ari": sub["adjusted_rand_index"].mean(),
                "lo": sub["adjusted_rand_index"].quantile(0.025),
                "hi": sub["adjusted_rand_index"].quantile(0.975),
            }
    return out


def _policy_human_ari() -> dict | None:
    """Human clustering ARI pooled across the policy experiment (rai + welfare).

    ``lo``/``hi`` are a percentile-bootstrap 95% CI of the *mean* (resampling
    rounds with replacement, 2000 resamples), not the raw spread of per-round
    ARI values — the latter is far wider and reflects round-to-round
    variability rather than uncertainty in the pooled estimate.
    """
    import numpy as np

    hc = read_csv("human_cluster_consistency_policy.csv")
    if hc is None:
        return None
    values = hc["adjusted_rand_index"].to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(2000, len(values)))
    boot_means = values[idx].mean(axis=1)
    return {
        "ari": float(values.mean()),
        "lo": float(np.percentile(boot_means, 2.5)),
        "hi": float(np.percentile(boot_means, 97.5)),
    }


def section_4_3_2(spearman: pl.DataFrame | None, k_rows: list[list[str]]) -> str:
    s = Section("Section 4.3.2 — Clustering Results")

    human_ari = _human_ari()
    rows = []
    best_ari = {}
    for exp in EXPERIMENT_DATASETS:
        agg = read_csv(f"cluster_consistency_aggregated_{exp}.csv")
        if agg is None:
            continue
        model_agg = agg.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS))).sort(
            "adjusted_rand_index", descending=True
        )
        best = model_agg.row(0, named=True)
        for ds in EXPERIMENT_DATASETS[exp]:
            best_ari[ds] = (best["model"], best["adjusted_rand_index"])
    for ds in DATASETS:
        h = human_ari.get(ds)
        bm = best_ari.get(ds)
        h_ari = h["ari"] if h else None
        rows.append(
            [
                pretty_dataset(ds),
                f"{h_ari:.3f}" if h_ari is not None else "n/a",
                f"[{h['lo']:.3f}, {h['hi']:.3f}]" if h else "n/a",
                f"{bm[1]:.3f}" if bm else "n/a",
                pretty_model(bm[0]) if bm else "n/a",
                f"{h_ari - bm[1]:+.3f}" if (h_ari is not None and bm) else "n/a",
            ]
        )
    s.line(
        "Per-dataset clustering ARI (human vs best model). Human CI = per-round "
        "2.5/97.5 percentile band; model ARI has no persisted bootstrap CI."
    )
    s.line()
    s.table(
        [
            "Dataset",
            "Human ARI",
            "Human CI (rounds)",
            "Best-model ARI",
            "Best model",
            "Human-model ARI gap",
        ],
        rows,
    )
    s.line(
        "> Statistical significance of the human-model ARI gap is not persisted; "
        "the 2000-resample bootstrap in `scripts/clustering.py` produces it on demand."
    )
    s.line()

    if spearman is not None:
        s.line(
            "**Grounding-to-ARI and MMTEB-to-ARI Spearman** "
            "(`cluster_spearman_by_experiment.csv`):"
        )
        s.line()
        s.table(
            ["Source", "Experiment", "n_models", "Spearman " + RHO, "95% CI"],
            [
                [
                    r["source"],
                    r["experiment"],
                    r["n_models"],
                    f"{r['spearman']:.3f}",
                    f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
                ]
                for r in spearman.to_dicts()
            ],
        )
        s.line(
            f"Scope: datasets {', '.join(pretty_dataset(d) for d in DATASETS)}; "
            "models = the shared neural+lexical set per experiment (see n_models)."
        )
        s.line()

    if k_rows:
        s.line(
            "**Robustness — K choice** (`k_sensitivity_table.tex`): each row is a "
            "clustering setup with its ARI-rank " + RHO + " vs the reference, "
            "and grounding/MMTEB-to-ARI " + RHO + " under that K:"
        )
        s.line()
        s.table(
            [
                "Clustering setup",
                "ARI rank " + RHO,
                "Grounding-ARI " + RHO,
                "MMTEB-ARI " + RHO,
            ],
            k_rows,
        )
    s.line(
        "> Ward-vs-k-means rank correlation and the human-model ARI-gap difference "
        "are computed inside `scripts/clustering.py` but not persisted to `output/`; "
        "re-run that script to tabulate them."
    )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Conclusion
# --------------------------------------------------------------------------- #
def section_conclusion(
    facts: dict, spearman: pl.DataFrame | None, mmteb_spearman: dict
) -> str:
    s = Section("Conclusion")
    gaps = [_pp(facts[d].get("gap")) for d in DATASETS if facts.get(d, {}).get("gap")]
    ogaps = [
        _pp(facts[d].get("oracle_gap"))
        for d in DATASETS
        if facts.get(d, {}).get("oracle_gap")
    ]
    if gaps:
        s.bullet(f"Human-model gap range: {min(gaps):.1f}-{max(gaps):.1f}pp.")
    if ogaps:
        s.bullet(f"Oracle-model gap range: {min(ogaps):.1f}-{max(ogaps):.1f}pp.")
    if mmteb_spearman:
        vals = ", ".join(
            f"{exp} {RHO}={v['rho']:.2f}" for exp, v in mmteb_spearman.items()
        )
        s.bullet(f"MMTEB-to-grounded rank correlation: {vals}.")
    if spearman is not None:
        s.bullet(
            f"Grounding-to-clustering {RHO} = {_spear_val(spearman, 'OurExercise', 'policy')} "
            f"(policy), {_spear_val(spearman, 'OurExercise', 'gov-ai')} (gov-ai)."
        )
        s.bullet(
            f"MMTEB-to-clustering {RHO} = {_spear_val(spearman, 'MMTEB', 'policy')} "
            f"(policy), {_spear_val(spearman, 'MMTEB', 'gov-ai')} (gov-ai)."
        )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Limitations
# --------------------------------------------------------------------------- #
def section_limitations(coords: pl.DataFrame | None) -> str:
    s = Section("Limitations")
    actual_pairs = "n/a"
    if coords is not None:
        # Distinct participant pairs that co-occur in any policy round (dedup on
        # the user pair itself, so it stays within the 6-choose-2 = 15 ceiling).
        pol = coords.filter(pl.col("dataset").is_in(["rai", "welfare"]))
        raters = pol.select("dataset", "seed", "user_id").unique()
        pairs = raters.join(raters, on=["dataset", "seed"], suffix="_b").filter(
            pl.col("user_id") < pl.col("user_id_b")
        )
        actual_pairs = pairs.select("user_id", "user_id_b").unique().height
    s.table(
        ["Quantity", "Value"],
        [
            ["Participants (total)", SPEC["participants_total"]],
            ["Participants per panel", SPEC["participants_per_panel"]],
            ["Max unique rater pairs per panel", SPEC["max_rater_pairs_per_panel"]],
            ["Actual contributing rater pairs (policy, coords)", actual_pairs],
            ["Statements per round", SPEC["statements_per_round"]],
            ["Oracle panel members", SPEC["participants_per_panel"]],
            [
                "Instruction variants " + TIMES + " conditions",
                f"{SPEC['n_instruction_variants']} {TIMES} {SPEC['n_conditions']}",
            ],
            ["Datasets", SPEC["n_datasets"]],
            ["Stakeholder communities", SPEC["n_communities"]],
        ],
    )
    for q in (
        "The oracle is in-sample and deliberately favorable.",
        "Human arrangements are local and 2D.",
        "Subset stability is within-panel, not across new panels.",
        "Clustering and grounding share the same layouts.",
    ):
        s.bullet(q)
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Ethical Considerations
# --------------------------------------------------------------------------- #
def section_ethics(group_stmt_counts: dict, missing_counts: dict) -> str:
    s = Section("Ethical Considerations")
    for ds in ("rai", "welfare"):
        gsc = group_stmt_counts.get(ds, {})
        if gsc:
            s.bullet(
                f"{pretty_dataset(ds)} protected-group statement counts: "
                + "; ".join(f"{k}={v}" for k, v in sorted(gsc.items()))
                + f" (missing/excluded: {missing_counts.get(ds, 'n/a')})."
            )
    controlled = read_text("fairness_group_gap_controlled.tex")
    for r in parse_latex_rows(controlled) if controlled else []:
        if len(r) >= 6:
            s.bullet(
                f"{r[0]}: raw {DELTA_GROUP} = {r[2]} (p {r[3]}); adjusted {DELTA_ADJ} = "
                f"{r[4]} (p {r[5]})."
            )
    s.line()
    s.line(
        "Participant demographics (gender, education, sector, location, experience) "
        "were collected for the RAI panel; welfare uses pseudonymised party."
    )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Appendix: Descriptive Statistics
# --------------------------------------------------------------------------- #
def section_appendix_descriptive(
    coverage_rows: list[list[str]],
    coords: pl.DataFrame | None,
    triplet_totals: dict,
) -> str:
    s = Section("Appendix: Descriptive Statistics")
    if coverage_rows:
        s.line("Canonical occurrence / co-occurrence (`statement_coverage_table.tex`):")
        s.line()
        s.table(
            [
                "Dataset",
                "Used statements",
                "Mean occ.",
                "Median occ.",
                "Co-occurring % pairs",
            ],
            coverage_rows,
        )
    if coords is not None:
        round_key = ["dataset", "seed", "user_id"]
        log = coords.group_by("dataset").agg(
            pl.len().alias("placements"),
            pl.col("statement_id").n_unique().alias("unique_stmts"),
            pl.col("user_id").n_unique().alias("participants"),
            pl.struct(round_key).n_unique().alias("rounds"),
        )
        occ = (
            coords.group_by("dataset", "statement_id")
            .agg(pl.struct(round_key).n_unique().alias("occ"))
            .group_by("dataset")
            .agg(
                pl.col("occ").mean().alias("mean"),
                pl.col("occ").min().alias("min"),
                pl.col("occ").max().alias("max"),
            )
        )
        stats = log.join(occ, on="dataset").sort("dataset")
        s.line(
            "Study logistics from coordinate files "
            "(**subset** of the full raw-triplet parquet — a lower bound):"
        )
        s.line()
        s.table(
            [
                "Dataset",
                "Placements",
                "Unique stmts",
                "Participants",
                "Rounds",
                "Occ. mean/min/max",
            ],
            [
                [
                    pretty_dataset(r["dataset"]),
                    f"{r['placements']:,}",
                    r["unique_stmts"],
                    r["participants"],
                    r["rounds"],
                    f"{r['mean']:.2f} / {r['min']} / {r['max']}",
                ]
                for r in stats.to_dicts()
            ],
        )
    if triplet_totals:
        s.line("Eligible / retained triplets (sum of demographic-group rows):")
        s.line()
        s.table(
            ["Dataset", "Eligible (d=1)", "Retained d=2", "Retained d=4"],
            [
                [pretty_dataset(ds), f"{v[0]:,}", f"{v[1]:,}", f"{v[2]:,}"]
                for ds, v in triplet_totals.items()
            ],
        )
    return s.render()


# --------------------------------------------------------------------------- #
# Appendix: Oracle
# --------------------------------------------------------------------------- #
def section_appendix_oracle(facts: dict) -> str:
    s = Section("Appendix: Oracle")
    s.bullet(
        "Consensus-distance aggregation: mean of per-rater normalized layout "
        "distances over co-occurring pairs (missing pairs mean-filled)."
    )
    s.bullet(
        f"MDS: {METHODS['mds']}, dimensionality q = {METHODS['mds_dim']} "
        f"(capped at n - 1), n_init = {METHODS['mds_n_init']}."
    )
    s.bullet(
        "MDS stress / reconstruction error is computed at fit time but not "
        "persisted; re-run the oracle fit to report it."
    )
    rows = []
    for ds in DATASETS:
        f = facts.get(ds, {})
        human = float(f["human"]) if "human" in f else None
        oracle = float(f["oracle"]) if "oracle" in f else None
        odiff = f"{(oracle - human) * 100:+.1f}pp" if (oracle and human) else "n/a"
        rows.append(
            [
                pretty_dataset(ds),
                f.get("oracle", "n/a"),
                f.get("human", "n/a"),
                f.get("best", "n/a"),
                f.get("oracle_gap", "n/a"),
                odiff,
            ]
        )
    s.table(
        [
            "Dataset",
            "Oracle AUC",
            "Human AUC",
            "Best-model AUC",
            "Oracle-model gap",
            "Oracle-human diff",
        ],
        rows,
    )
    return s.render()


# --------------------------------------------------------------------------- #
# Appendix: Instruction Robustness
# --------------------------------------------------------------------------- #
def section_appendix_instruction(
    instruct: pl.DataFrame | None, context: pl.DataFrame | None
) -> str:
    s = Section("Appendix: Instruction Robustness")
    if instruct is None:
        s.missing(
            "instruction robustness", "instruct_prompt_robustness_auc.csv not found"
        )
        return s.render()
    s.line(
        "Per model "
        + TIMES
        + " variant "
        + TIMES
        + " dataset ("
        + DELTA
        + " vs default = variant AUC - plain, "
        "non-instructed model AUC):"
    )
    s.line()
    rows = []
    for r in instruct.sort("base_model", "dataset", "variant").to_dicts():
        d = r.get("delta_default")
        rows.append(
            [
                pretty_model(r["base_model"]),
                pretty_dataset(r["dataset"]),
                r["variant"],
                f"{r['auc']:.3f}",
                "n/a" if d is None else f"{d:+.3f}",
            ]
        )
    s.table(["Base model", "Dataset", "Variant", "AUC", DELTA + " vs default"], rows)

    # Best / worst / range per model-dataset.
    s.line("Best / worst variant and range per model " + TIMES + " dataset:")
    s.line()
    summ_rows = []
    for (base, ds), grp in instruct.group_by("base_model", "dataset"):
        g = grp.sort("auc", descending=True)
        best, worst = g.row(0, named=True), g.row(g.height - 1, named=True)
        summ_rows.append(
            [
                pretty_model(base),
                pretty_dataset(ds),
                f"{best['variant']} ({best['auc']:.3f})",
                f"{worst['variant']} ({worst['auc']:.3f})",
                f"{best['auc'] - worst['auc']:.3f}",
            ]
        )
    s.table(
        ["Base model", "Dataset", "Best variant", "Worst variant", "Range"],
        sorted(summ_rows),
    )

    if context is not None:
        s.line("Generic vs dataset-context conditions:")
        s.line()
        crows = [
            [
                pretty_model(r["base_model"]),
                pretty_dataset(r["dataset"]),
                r["variant"],
                r["context"] or "generic",
                f"{r['auc']:.3f}",
            ]
            for r in context.sort(
                "base_model", "dataset", "variant", "context"
            ).to_dicts()
        ]
        s.table(["Base model", "Dataset", "Variant", "Context", "AUC"], crows)
    return s.render()


# --------------------------------------------------------------------------- #
# Appendix: Metric Robustness
# --------------------------------------------------------------------------- #
def section_appendix_metric() -> str:
    s = Section("Appendix: Metric Robustness")
    s.table(
        ["Setting", "Value"],
        [
            ["Main " + TAU_MAX, METHODS["main_d_max"]],
            ["Alternative " + TAU_MAX, ", ".join(str(v) for v in METHODS["alt_d_max"])],
            ["Threshold points (main)", METHODS["main_n_points"]],
            [
                "Alternative threshold points",
                ", ".join(str(v) for v in METHODS["alt_n_points"]),
            ],
            ["Integration schemes", "log-x (main) and linear-d"],
            [
                "Expected disagreement",
                f"fixed D_e = {METHODS['D_e']} (vs empirical D_e)",
            ],
        ],
    )
    tex = read_text("auc_sensitivity_table.tex")
    rows = parse_latex_rows(tex) if tex else []
    if rows:
        s.line(
            "Rank-stability vs the main configuration (`auc_sensitivity_table.tex`):"
        )
        s.line()
        s.table(
            ["Configuration", "Full-rank " + RHO, "Top-10 " + RHO, "Top-10 overlap"],
            rows,
        )
    else:
        s.missing("AUC sensitivity", "auc_sensitivity_table.tex not found")
    return s.render()


# --------------------------------------------------------------------------- #
# Canonical headline value block
# --------------------------------------------------------------------------- #
def section_headline_block(
    facts: dict,
    spearman: pl.DataFrame | None,
    mmteb_spearman: dict,
    instruct: pl.DataFrame | None,
    oracle_auc: dict,
) -> str:
    s = Section("Canonical headline value block")
    s.line(
        "Minimal set from which the abstract, main results, conclusion, and captions "
        "are populated."
    )
    s.line()
    rows = []
    for ds in DATASETS:
        f = facts.get(ds, {})
        rows.append(
            [
                pretty_dataset(ds),
                f.get("human", "n/a"),
                f.get("oracle", "n/a"),
                f.get("best", "n/a"),
                f.get("gap", "n/a"),
                f.get("oracle_gap", "n/a"),
            ]
        )
    s.table(
        [
            "Dataset",
            "Human AUC",
            "Oracle AUC",
            "Best-model AUC",
            "Human-model gap",
            "Oracle-model gap",
        ],
        rows,
    )
    if mmteb_spearman:
        s.bullet(
            "MMTEB-to-grounded "
            + RHO
            + ": "
            + ", ".join(f"{exp}={v['rho']:.2f}" for exp, v in mmteb_spearman.items())
        )
    if spearman is not None:
        s.bullet(
            f"Grounding-to-ARI {RHO}: policy={_spear_val(spearman, 'OurExercise', 'policy')}, "
            f"gov-ai={_spear_val(spearman, 'OurExercise', 'gov-ai')}."
        )
        s.bullet(
            f"MMTEB-to-ARI {RHO}: policy={_spear_val(spearman, 'MMTEB', 'policy')}, "
            f"gov-ai={_spear_val(spearman, 'MMTEB', 'gov-ai')}."
        )
    if instruct is not None:
        deltas = instruct.get_column("delta_default").drop_nulls()
        if len(deltas) > 0:
            s.bullet(
                f"Best instruction gain ({DELTA} vs default): {deltas.max():+.3f}."
            )
        if oracle_auc:
            best_cell = instruct.group_by("base_model", "dataset").agg(
                pl.col("auc").max().alias("v")
            )
            gaps = [
                (r["base_model"], r["dataset"], oracle_auc[r["dataset"]] - r["v"])
                for r in best_cell.to_dicts()
                if r["dataset"] in oracle_auc
            ]
            if gaps:
                gaps.sort(key=lambda t: t[2])
                s.bullet(
                    f"Best instructed model's remaining oracle gap: "
                    f"{pretty_model(gaps[0][0])} on {pretty_dataset(gaps[0][1])} "
                    f"= {gaps[0][2] * 100:.1f}pp."
                )
    # Raw + adjusted disparities.
    controlled = read_text("fairness_group_gap_controlled.tex")
    for r in parse_latex_rows(controlled) if controlled else []:
        if len(r) >= 6:
            s.bullet(f"{r[0]} group disparity: raw {r[2]}, adjusted {r[4]}.")
    # Clustering ARI.
    human_ari = _human_ari()
    for ds in DATASETS:
        h = human_ari.get(ds)
        agg = None
        for exp, dss in EXPERIMENT_DATASETS.items():
            if ds in dss:
                agg = read_csv(f"cluster_consistency_aggregated_{exp}.csv")
        if h and agg is not None:
            best = (
                agg.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS)))
                .sort("adjusted_rand_index", descending=True)
                .row(0, named=True)
            )
            s.bullet(
                f"{pretty_dataset(ds)} clustering ARI: human {h['ari']:.3f}, "
                f"best model {pretty_model(best['model'])} {best['adjusted_rand_index']:.3f}."
            )
    s.line()
    return s.render()


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _build_triplet_data() -> tuple[dict, list[list[str]]]:
    """Return (per-dataset totals [d1,d2,d4], per-group display rows)."""
    tex = read_text("fairness_triplet_counts.tex")
    if tex is None:
        return {}, []
    rows = parse_latex_rows(tex)
    totals: dict[str, list[int]] = {}

    def _int(cell: str) -> int:
        c = cell.replace(",", "")
        return int(c) if c.isdigit() else 0

    for r in rows:
        totals.setdefault(r[0], [0, 0, 0])
        for i, cell in enumerate(r[2:5]):
            totals[r[0]][i] += _int(cell)
    return totals, rows


def _build_instruct(bootstraps: pl.DataFrame | None) -> pl.DataFrame | None:
    rob = read_csv("instruct_prompt_robustness_auc.csv")
    if rob is None:
        return None
    summ = instruct_summary(rob, split_context=False)
    if bootstraps is not None:
        plain = dataset_model_means(bootstraps).rename({"mean_auc": "plain_auc"})
        summ = summ.join(
            plain,
            left_on=["base_model", "dataset"],
            right_on=["model", "dataset"],
            how="left",
        ).with_columns((pl.col("auc") - pl.col("plain_auc")).alias("delta_default"))
    else:
        summ = summ.with_columns(pl.lit(None, dtype=pl.Float64).alias("delta_default"))
    return summ


def _build_mmteb_spearman() -> dict:
    from scipy.stats import spearmanr

    out = {}
    for exp in EXPERIMENT_DATASETS:
        grounded = read_csv(f"human_alignment_bootstrapped_{exp}.csv")
        mmteb = read_csv(f"mmteb_with_ranks_{exp}.csv")
        if grounded is None or mmteb is None:
            continue
        gr = (
            grounded.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS)))
            .sort("alignment_score", descending=True)
            .with_row_index("grounded_rank", offset=1)
        )
        merged = gr.join(mmteb, left_on="model", right_on="model_name", how="inner")
        if merged.height < 3:
            continue
        res = spearmanr(
            merged["grounded_rank"].to_numpy(), merged["Rank (Borda)"].to_numpy()
        )
        best_model = gr.row(0, named=True)["model"]
        best = merged.filter(pl.col("model") == best_model)
        out[exp] = {
            "rho": float(res.statistic),
            "p": float(res.pvalue),
            "n": merged.height,
            "best_model": best_model,
            "best_rank": (
                "n/a"
                if best.is_empty()
                else f"{best['Rank (MMTEB)'][0] + 1} of {merged.height}"
            ),
        }
    return out


def _group_stmt_counts() -> tuple[dict, dict]:
    counts, missing = {}, {}
    try:
        from human_grounding.data import get_rai_demographics, get_welfare_demographics

        rai = (
            get_rai_demographics("gender")
            .group_by("demographic")
            .agg(pl.col("cause_id").n_unique().alias("n"))
        )
        counts["rai"] = {
            r["demographic"]: r["n"]
            for r in rai.to_dicts()
            if r["demographic"] in {"Kvinde", "Mand"}
        }
        missing["rai"] = next(
            (r["n"] for r in rai.to_dicts() if r["demographic"] == "Unknown"), 0
        )
        wf = (
            get_welfare_demographics()
            .group_by("demographic")
            .agg(pl.col("cause_id").n_unique().alias("n"))
        )
        counts["welfare"] = {
            r["demographic"]: r["n"]
            for r in wf.to_dicts()
            if r["demographic"] in WELFARE_ANALYSIS_PARTIES
        }
        missing["welfare"] = sum(
            r["n"]
            for r in wf.to_dicts()
            if r["demographic"] not in WELFARE_ANALYSIS_PARTIES
        )
    except Exception:  # raw corpus optional
        pass
    return counts, missing


def build_report() -> str:
    bootstraps = load_alignment_bootstraps()
    means = dataset_model_means(bootstraps) if bootstraps is not None else None
    coords = load_coordinates()
    facts = parse_alignment_summary()
    spearman = read_csv("cluster_spearman_by_experiment.csv")
    drift = read_csv("context_drift_comparison.csv") or read_csv(
        "context_drift_comparison.csv", OUTPUT_DIR / ".." / "plots"
    )
    cov = read_text("statement_coverage_table.tex")
    coverage_rows = parse_latex_rows(cov) if cov else []
    triplet_totals, triplet_group_rows = _build_triplet_data()
    instruct = _build_instruct(bootstraps)
    context = None
    ctx_raw = read_csv("instruct_prompt_context_auc.csv")
    if ctx_raw is not None:
        context = instruct_summary(ctx_raw, split_context=True)
    mmteb_spearman = _build_mmteb_spearman()
    group_stmt_counts, missing_counts = _group_stmt_counts()
    ktex = read_text("k_sensitivity_table.tex")
    k_rows = parse_latex_rows(ktex) if ktex else []
    oracle_auc = {ds: float(f["oracle"]) for ds, f in facts.items() if "oracle" in f}

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = [
        "# Canonical Values — by Paper Section",
        "",
        f"_Generated {now} by `scripts/report_canonical_values.py`._",
        "",
        "Values are read from precomputed artifacts under `output/` (and raw "
        "coordinates/corpus under `data/`); nothing is re-embedded or re-bootstrapped. "
        "Narrative design constants (participant counts, completion time, model counts) "
        "are from the paper spec and marked as such; methods constants (threshold grid, "
        "D_e, MDS dimensionality, clustering algorithm) are read from the pipeline code.",
        "",
        "Datasets: **rai** (Responsible AI), **welfare** (Welfare), **gov-ai** "
        "(Government AI). Experiments: `policy = {rai, welfare}`, `gov-ai = {gov-ai}`.",
        "",
    ]
    parts = [
        "\n".join(header),
        section_abstract(facts, spearman),
        section_intro(facts, spearman),
        section_3_1(coverage_rows),
        section_3_2(coverage_rows),
        section_4_1_1(coords),
        section_4_1_2(bootstraps, facts, triplet_totals, drift),
        section_4_2_1(bootstraps),
        section_4_2_2(means, facts, mmteb_spearman, instruct, context, oracle_auc),
        section_figure4(bootstraps, means, facts),
        section_group_conditional(
            bootstraps, means, group_stmt_counts, triplet_group_rows
        ),
        section_qualitative(),
        section_4_3_1(),
        section_4_3_2(spearman, k_rows),
        section_conclusion(facts, spearman, mmteb_spearman),
        section_limitations(coords),
        section_ethics(group_stmt_counts, missing_counts),
        section_appendix_descriptive(coverage_rows, coords, triplet_totals),
        section_appendix_oracle(facts),
        section_appendix_instruction(instruct, context),
        section_appendix_metric(),
        section_headline_block(facts, spearman, mmteb_spearman, instruct, oracle_auc),
    ]
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# key-values.tex  (\newcommand definitions — the paper's numeric API)
# --------------------------------------------------------------------------- #
KEY_VALUES_PATH = OUTPUT_DIR / "key-values.tex"

# Internal dataset -> command-name prefix (descriptive, not abbreviated).
CMD_PREFIX = {"rai": "ResponsibleAI", "welfare": "Welfare", "gov-ai": "GovAI"}


def _tex_cmd(name: str, value: object, comment: str) -> str:
    """One documented ``\\newcommand`` (blank value stays an empty group)."""
    val = "" if value is None else str(value)
    return f"% {comment}\n" + rf"\newcommand{{\{name}}}{{{val}}}"


def _tex_banner(title: str) -> str:
    bar = "%" * 68
    return f"{bar}\n%% {title}\n{bar}"


def _pp_num(text: str | None) -> float | None:
    """Parse a 'NN.Npp' gap string into a float (None if empty)."""
    if not text:
        return None
    return float(text.replace("pp", "").strip())


def _pp_ci(ci_text: str | None) -> tuple[float, float] | None:
    """Parse a '25.0pp, 27.2pp' CI string into floats."""
    if not ci_text:
        return None
    parts = [p.strip().replace("pp", "") for p in ci_text.split(",")]
    if len(parts) != 2:
        return None
    return float(parts[0]), float(parts[1])


def _within_rater_auc() -> dict[str, float]:
    """Per-dataset within-rater alpha-AUC from the dataset-level alpha curves.

    Dataset-level curves are demographic-agnostic, so this is unaffected by the
    gender/education split.
    """
    from human_grounding.alpha_reliability import normalized_auc_logx

    alpha = _alpha_frames()
    if alpha is None:
        return {}
    curve = (
        alpha.filter(
            (pl.col("group_type") == "dataset")
            & (pl.col("reliability_type") == "within")
        )
        .group_by("group_name", "d")
        .agg(pl.col("krippendorf").mean().alias("k"))
    )
    out = {}
    for ds, grp in curve.group_by("group_name"):
        g = grp.sort("d")
        out[ds[0]] = float(normalized_auc_logx(g["k"].to_numpy(), g["d"].to_numpy()))
    return out


def _human_gender_auc() -> dict[str, tuple[float, float, float]]:
    """Human-human reliability AUC + 95% CI per RAI gender group.

    Reads the between-rater alpha curves for the ``Kvinde``/``Mand`` groups from
    the alpha data (present only once the pipeline is regenerated with the gender
    demographic fix). Per bootstrap iteration the curve is integrated to one AUC
    via ``normalized_auc_logx``; we then take the mean and 2.5/97.5 percentiles.
    Returns ``{}`` while the alpha data still splits RAI by education.
    """
    import numpy as np

    from human_grounding.alpha_reliability import normalized_auc_logx

    alpha = _alpha_frames()
    if alpha is None:
        return {}
    gender = alpha.filter(
        (pl.col("group_type") == "demographic")
        & (pl.col("group_name").is_in(["Kvinde", "Mand"]))
        & (pl.col("reliability_type") == "between")
    )
    if gender.is_empty():
        return {}
    out: dict[str, tuple[float, float, float]] = {}
    for grp, sub in gender.group_by("group_name"):
        aucs = []
        for _it, itsub in sub.group_by("iteration_id"):
            c = itsub.sort("d")
            aucs.append(
                float(
                    normalized_auc_logx(c["krippendorf"].to_numpy(), c["d"].to_numpy())
                )
            )
        arr = np.array(aucs, dtype=float)
        out[grp[0]] = (
            float(arr.mean()),
            float(np.percentile(arr, 2.5)),
            float(np.percentile(arr, 97.5)),
        )
    return out


def build_key_values_tex() -> str:
    bootstraps = load_alignment_bootstraps()
    means = dataset_model_means(bootstraps) if bootstraps is not None else None
    facts = parse_alignment_summary()
    spearman = read_csv("cluster_spearman_by_experiment.csv")
    mmteb = _build_mmteb_spearman()
    instruct = _build_instruct(bootstraps)
    human_ari = _human_ari()
    within = _within_rater_auc()
    alpha = _alpha_frames()
    tau08 = _tau_at_alpha(alpha) if alpha is not None else {}
    oracle_auc = {ds: float(f["oracle"]) for ds, f in facts.items() if "oracle" in f}

    cov = read_text("statement_coverage_table.tex")
    coverage_rows = parse_latex_rows(cov) if cov else []
    used = {COVERAGE_NAME_TO_DS.get(r[0], r[0]): r[1] for r in coverage_rows}
    occ = {COVERAGE_NAME_TO_DS.get(r[0], r[0]): r[2] for r in coverage_rows}

    _, triplet_group_rows = _build_triplet_data()

    def _int(cell: str) -> int:
        c = cell.replace(",", "")
        return int(c) if c.lstrip("-").isdigit() else 10**12

    d4_all = [_int(r[4]) for r in triplet_group_rows if len(r) >= 5]
    d4_rai = [
        _int(r[4])
        for r in triplet_group_rows
        if len(r) >= 5 and r[0] == pretty_dataset("rai")
    ]
    min_group_triplets = min(d4_all) if d4_all else None
    rai_min_triplets = min(d4_rai) if d4_rai else None

    # Source-corpus statement counts (full corpus, pre-sampling).
    source = {}
    try:
        from human_grounding.data import get_govai, get_responsible_ai, get_welfare

        source = {
            "rai": get_responsible_ai().select("cause_id").n_unique(),
            "welfare": get_welfare().select("cause_id").n_unique(),
            "gov-ai": get_govai().select("cause_id").n_unique(),
        }
    except Exception:  # raw corpus optional
        source = {}

    # Overall best / second model by mean AUC across datasets.
    best_name = second_name = None
    best_overall_auc = None
    if means is not None:
        overall = (
            means.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS)))
            .group_by("model")
            .agg(pl.col("mean_auc").mean().alias("a"))
            .sort("a", descending=True)
        )
        rows = overall.head(2).to_dicts()
        if rows:
            best_name = pretty_model(rows[0]["model"])
            best_overall_auc = rows[0]["a"]
        if len(rows) > 1:
            second_name = pretty_model(rows[1]["model"])

    # Fairness raw / adjusted disparity (first number in each CI cell).
    controlled = read_text("fairness_group_gap_controlled.tex")
    fair = {}
    for r in parse_latex_rows(controlled) if controlled else []:
        if len(r) >= 6:
            ds = next((d for d in DATASETS if pretty_dataset(d) == r[0]), None)
            if ds:
                fair[ds] = {"raw": r[2].split()[0], "adj": r[4].split()[0]}

    # Best-model clustering ARI per dataset.
    best_model_ari = {}
    for exp in EXPERIMENT_DATASETS:
        agg = read_csv(f"cluster_consistency_aggregated_{exp}.csv")
        if agg is None:
            continue
        best = (
            agg.filter(~pl.col("model").is_in(list(NON_MODEL_ROWS)))
            .sort("adjusted_rand_index", descending=True)
            .row(0, named=True)
        )
        for ds in EXPERIMENT_DATASETS[exp]:
            best_model_ari[ds] = (
                pretty_model(best["model"]),
                best["adjusted_rand_index"],
            )

    # Instruction robustness headline numbers.
    max_gain = None
    instr_oracle_gap = None
    if instruct is not None:
        deltas = instruct.get_column("delta_default").drop_nulls()
        if len(deltas) > 0:
            max_gain = float(deltas.max())
        if oracle_auc:
            best_cell = instruct.group_by("base_model", "dataset").agg(
                pl.col("auc").max().alias("v")
            )
            gaps = [
                oracle_auc[r["dataset"]] - r["v"]
                for r in best_cell.to_dicts()
                if r["dataset"] in oracle_auc
            ]
            if gaps:
                instr_oracle_gap = min(gaps) * 100  # pp, best-instructed model

    # Headline gap range (excludes gov-ai; only rai/welfare are used for the
    # min/max headline figures).
    gap_vals = [
        _pp_num(facts[d].get("gap"))
        for d in DATASETS
        if d != "gov-ai" and facts.get(d, {}).get("gap")
    ]
    gap_vals = [g for g in gap_vals if g is not None]
    grounding_policy = None
    mmteb_policy = None
    if spearman is not None:
        row = spearman.filter(
            (pl.col("source") == "OurExercise") & (pl.col("experiment") == "policy")
        )
        if not row.is_empty():
            grounding_policy = row.row(0, named=True)
        row = spearman.filter(
            (pl.col("source") == "MMTEB") & (pl.col("experiment") == "policy")
        )
        if not row.is_empty():
            mmteb_policy = row.row(0, named=True)

    def f3(x: object) -> str:
        return "" if x is None else f"{float(x):.3f}"

    def f2(x: object) -> str:
        return "" if x is None else f"{float(x):.2f}"

    def f1(x: object) -> str:
        return "" if x is None else f"{float(x):.1f}"

    blocks: list[str] = []

    # -- Dataset statistics ------------------------------------------------- #
    lines = [_tex_banner("Dataset statistics")]
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}TotalStatements",
                source.get(ds, ""),
                f"Number of unique source statements in the {pretty_dataset(ds)} corpus",
            )
        )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}ExerciseStatements",
                used.get(ds, ""),
                f"Unique statements included in the {pretty_dataset(ds)} exercise",
            )
        )
    lines.append(
        _tex_cmd(
            "TotalStatementPlacements",
            SPEC["total_placements_approx"],
            "Number of individual statement placements collected across all exercises",
        )
    )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}StatementOccurrences",
                occ.get(ds, ""),
                f"Mean number of occurrences per included statement ({pretty_dataset(ds)})",
            )
        )
    blocks.append("\n\n".join(lines))

    # -- Human reliability -------------------------------------------------- #
    lines = [_tex_banner("Human reliability")]
    rai_tau = tau08.get(("rai", "between"))
    wf_tau = tau08.get(("welfare", "between"))
    thr = None
    if rai_tau and wf_tau:
        thr = (rai_tau[0] + wf_tau[0]) / 2
    lines.append(
        _tex_cmd(
            "HumanReliabilityThreshold",
            f1(thr),
            "Threshold (tau) at which human inter-rater reliability reaches alpha ~ 0.8 "
            "(mean of RAI and Welfare between-rater curves)",
        )
    )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}HumanAUC",
                facts.get(ds, {}).get("human", ""),
                f"Human inter-rater reliability (AUC) for {pretty_dataset(ds)}",
            )
        )
    wr = [within.get(d) for d in ("rai", "welfare") if within.get(d) is not None]
    wr_summary = sum(wr) / len(wr) if wr else None
    lines.append(
        _tex_cmd(
            "WithinRaterAUC",
            f3(wr_summary),
            "Human within-rater reliability summary (mean within-rater AUC, RAI & Welfare)",
        )
    )
    lines.append(
        _tex_cmd(
            "MinimumProtectedGroupTriplets",
            "" if min_group_triplets is None else f"{min_group_triplets}",
            "Minimum retained triplets for any protected group at tau = 4",
        )
    )
    lines.append(
        _tex_cmd(
            "ResponsibleAIProtectedTriplets",
            "" if rai_min_triplets is None else f"{rai_min_triplets}",
            "Minimum retained triplets across Responsible AI gender groups at tau = 4",
        )
    )
    blocks.append("\n\n".join(lines))

    # -- Human demographic reliability -------------------------------------- #
    # Fills automatically once the alpha data is regenerated with the gender fix
    # (RAI split by gender, Unknown dropped); empty while it is education-split.
    gender = _human_gender_auc()  # {"Mand": (auc, lo, hi), "Kvinde": (auc, lo, hi)}
    men, women = gender.get("Mand"), gender.get("Kvinde")
    lines = [_tex_banner("Human demographic reliability")]
    if not gender:
        lines.append(
            "% NOTE: empty until the alpha data is regenerated with the gender fix\n"
            "% (RAI split by gender, Unknown dropped); values fill in automatically."
        )
    lines += [
        _tex_cmd(
            "ResponsibleAIMenAUC",
            f3(men[0]) if men else "",
            "Human alignment AUC for male statement authors (Responsible AI)",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUC",
            f3(women[0]) if women else "",
            "Human alignment AUC for female statement authors (Responsible AI)",
        ),
        _tex_cmd(
            "ResponsibleAIMenAUCLO",
            f3(men[1]) if men else "",
            "Lower confidence interval for male AUC",
        ),
        _tex_cmd(
            "ResponsibleAIMenAUCHI",
            f3(men[2]) if men else "",
            "Upper confidence interval for male AUC",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUCLO",
            f3(women[1]) if women else "",
            "Lower confidence interval for female AUC",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUCHI",
            f3(women[2]) if women else "",
            "Upper confidence interval for female AUC",
        ),
    ]
    blocks.append("\n\n".join(lines))

    # -- Neural-human alignment --------------------------------------------- #
    lines = [_tex_banner("Neural-human alignment")]
    lines.append(
        _tex_cmd(
            "BestEmbeddingModel",
            best_name or "",
            "Best-performing embedding model (overall)",
        )
    )
    lines.append(
        _tex_cmd(
            "SecondEmbeddingModel",
            second_name or "",
            "Second-best embedding model (overall)",
        )
    )
    lines.append(
        _tex_cmd(
            "BestEmbeddingModelAUC",
            f3(best_overall_auc),
            "Mean alignment AUC of the best embedding model across datasets",
        )
    )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}ModelGap",
                f1(_pp_num(facts.get(ds, {}).get("gap"))),
                f"Human-model performance gap in pp ({pretty_dataset(ds)})",
            )
        )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        ci = _pp_ci(facts.get(ds, {}).get("gap_ci"))
        lines.append(
            _tex_cmd(
                f"{p}ModelGapLO",
                f1(ci[0]) if ci else "",
                f"Lower CI for {pretty_dataset(ds)} gap (pp)",
            )
        )
        lines.append(
            _tex_cmd(
                f"{p}ModelGapHI",
                f1(ci[1]) if ci else "",
                f"Upper CI for {pretty_dataset(ds)} gap (pp)",
            )
        )
    blocks.append("\n\n".join(lines))

    # -- Oracle ------------------------------------------------------------- #
    lines = [_tex_banner("Oracle")]
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}OracleAUC",
                facts.get(ds, {}).get("oracle", ""),
                f"Oracle alignment AUC for {pretty_dataset(ds)}",
            )
        )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}OracleGap",
                f1(_pp_num(facts.get(ds, {}).get("oracle_gap"))),
                f"Oracle-model gap in pp for {pretty_dataset(ds)}",
            )
        )
    blocks.append("\n\n".join(lines))

    # -- MMTEB comparison --------------------------------------------------- #
    lines = [_tex_banner("MMTEB comparison")]
    pol = mmteb.get("policy", {})
    best_rank = str(pol.get("best_rank", "")).split(" of ")[0] if pol else ""
    lines.append(
        _tex_cmd(
            "BestGroundedMMTEBRank",
            best_rank,
            "Rank of the best grounded model on MMTEB (within shared model set)",
        )
    )
    lines.append(
        _tex_cmd(
            "MMTEBGroundedSpearman",
            f2(pol.get("rho")) if pol else "",
            "Spearman correlation between MMTEB ranking and grounded ranking (policy)",
        )
    )
    lines.append(
        _tex_cmd(
            "MMTEBGroundedPValue",
            f"{pol['p']:.4f}" if pol else "",
            "P-value for MMTEB correlation (policy)",
        )
    )
    blocks.append("\n\n".join(lines))

    # -- Fairness ----------------------------------------------------------- #
    lines = [_tex_banner("Fairness")]
    for ds in ("rai", "welfare"):
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}GroupGap",
                fair.get(ds, {}).get("raw", ""),
                f"Group alignment disparity for {pretty_dataset(ds)}",
            )
        )
    for ds in ("rai", "welfare"):
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}AdjustedGroupGap",
                fair.get(ds, {}).get("adj", ""),
                f"Adjusted group alignment disparity for {pretty_dataset(ds)}",
            )
        )
    blocks.append("\n\n".join(lines))

    # -- Downstream clustering ---------------------------------------------- #
    lines = [_tex_banner("Downstream clustering")]
    aris = [human_ari[d]["ari"] for d in DATASETS if d in human_ari]
    lines.append(
        _tex_cmd(
            "HumanARI",
            f3(sum(aris) / len(aris) if aris else None),
            "Human clustering ARI (mean across datasets)",
        )
    )
    policy_human_ari = _policy_human_ari()
    lines.append(
        _tex_cmd(
            "PolicyHumanARI",
            f3(policy_human_ari["ari"]) if policy_human_ari else "",
            "Human clustering ARI pooled across policy (responsible AI + welfare)",
        )
    )
    lines.append(
        _tex_cmd(
            "PolicyHumanARILO",
            f3(policy_human_ari["lo"]) if policy_human_ari else "",
            "Lower CI for policy human clustering ARI",
        )
    )
    lines.append(
        _tex_cmd(
            "PolicyHumanARIHI",
            f3(policy_human_ari["hi"]) if policy_human_ari else "",
            "Upper CI for policy human clustering ARI",
        )
    )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        h = human_ari.get(ds)
        lines.append(
            _tex_cmd(
                f"{p}HumanARI",
                f3(h["ari"]) if h else "",
                f"Human clustering ARI for {pretty_dataset(ds)}",
            )
        )
        bm = best_model_ari.get(ds)
        lines.append(
            _tex_cmd(
                f"{p}BestModelARI",
                f3(bm[1]) if bm else "",
                f"Best-model clustering ARI for {pretty_dataset(ds)}",
            )
        )
    lines.append(
        _tex_cmd(
            "GroundingClusteringCorrelation",
            f2(grounding_policy["spearman"]) if grounding_policy else "",
            "Grounding-clustering Spearman correlation (policy)",
        )
    )
    lines.append(
        _tex_cmd(
            "GroundingClusteringCorrelationLO",
            f2(grounding_policy["ci_lo"]) if grounding_policy else "",
            "Lower CI for grounding-clustering correlation",
        )
    )
    lines.append(
        _tex_cmd(
            "GroundingClusteringCorrelationHI",
            f2(grounding_policy["ci_hi"]) if grounding_policy else "",
            "Upper CI for grounding-clustering correlation",
        )
    )
    lines.append(
        _tex_cmd(
            "MMTEBClusteringCorrelation",
            f2(mmteb_policy["spearman"]) if mmteb_policy else "",
            "MMTEB-clustering correlation (policy)",
        )
    )
    lines.append(
        _tex_cmd(
            "MMTEBClusteringCorrelationLO",
            f2(mmteb_policy["ci_lo"]) if mmteb_policy else "",
            "Lower CI for MMTEB-clustering correlation",
        )
    )
    lines.append(
        _tex_cmd(
            "MMTEBClusteringCorrelationHI",
            f2(mmteb_policy["ci_hi"]) if mmteb_policy else "",
            "Upper CI for MMTEB-clustering correlation",
        )
    )
    lines.append(
        _tex_cmd(
            "WardKMeansCorrelation",
            "",
            "Ward vs k-means rank correlation (not persisted; run scripts/clustering.py)",
        )
    )
    lines.append(
        _tex_cmd(
            "WardKMeansGapDifference",
            "",
            "Difference in human-model ARI gap between Ward and k-means (not persisted)",
        )
    )
    blocks.append("\n\n".join(lines))

    # -- Instruction robustness --------------------------------------------- #
    lines = [_tex_banner("Instruction robustness")]
    lines.append(
        _tex_cmd(
            "MaximumInstructionGain",
            f3(max_gain),
            "Maximum improvement over the default (plain-model) instruction",
        )
    )
    lines.append(
        _tex_cmd(
            "InstructionOracleGap",
            f1(instr_oracle_gap),
            "Remaining oracle gap (pp) for the best instruction-tuned model",
        )
    )
    blocks.append("\n\n".join(lines))

    # -- Headline numbers --------------------------------------------------- #
    lines = [_tex_banner("Headline numbers")]
    lines.append(
        _tex_cmd(
            "HeadlineGapMinimum",
            f"{round(min(gap_vals))}" if gap_vals else "",
            "Minimum reported stakeholder-model gap in the paper (pp)",
        )
    )
    lines.append(
        _tex_cmd(
            "HeadlineGapMaximum",
            f"{round(max(gap_vals))}" if gap_vals else "",
            "Maximum reported stakeholder-model gap in the paper (pp)",
        )
    )
    lines.append(
        _tex_cmd(
            "HeadlineGroundingCorrelation",
            f1(grounding_policy["spearman"]) if grounding_policy else "",
            "Headline downstream grounding-clustering correlation",
        )
    )
    blocks.append("\n\n".join(lines))

    # -- Text fragments ----------------------------------------------------- #
    lines = [_tex_banner("Text fragments")]
    lines.append(
        _tex_cmd(
            "BestEmbeddingModelName",
            rf"\texttt{{{best_name}}}" if best_name else "",
            "Name of the best embedding model",
        )
    )
    lines.append(_tex_cmd("EmbeddingBenchmark", "MMTEB", "Name of the benchmark"))
    lines.append(_tex_cmd("OracleName", "Human-MDS oracle", "Name of the oracle"))
    blocks.append("\n\n".join(lines))

    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    header = (
        f"% key-values.tex -- canonical numeric API for the manuscript.\n"
        f"% Generated {now} by scripts/report_canonical_values.py from artifacts in output/.\n"
        f"% \\input this file in the preamble; use e.g. \\ResponsibleAIModelGap in text.\n"
        f"% Empty groups {{}} mark values not derivable from current artifacts (see comment)."
    )
    return header + "\n\n" + "\n\n\n".join(blocks) + "\n"


def main() -> None:
    REPORT_PATH.write_text(build_report())
    print(f"Wrote canonical values report to {REPORT_PATH}")
    KEY_VALUES_PATH.write_text(build_key_values_tex())
    print(f"Wrote LaTeX key-values to {KEY_VALUES_PATH}")


if __name__ == "__main__":
    main()
