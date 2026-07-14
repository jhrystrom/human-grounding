"""Generate ``output/key-values.tex``, the paper's canonical numeric API.

This is a *reporter*, not a recomputation pipeline. Wherever a canonical value
already lives in a precomputed artifact under ``output/`` (bootstrap AUC tables,
cluster-consistency CSVs, alpha reliability curves, the LaTeX/txt summaries), we
load that artifact and read the value off it. The only numbers computed here are
cheap aggregations over already-materialised rows (means, ranks, group min/max
over precomputed per-bootstrap AUCs, descriptive counts). Nothing re-embeds text
or re-bootstraps.

Each emitted ``\\newcommand`` is documented with a one-line comment; empty
groups (``{}``) mark values not derivable from current artifacts.

Usage::

    uv run scripts/report_canonical_values.py
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path

import polars as pl

from human_grounding.constants import DATASET_PRETTY_NAMES, PRETTY_NAMES
from human_grounding.directories import OUTPUT_DIR
from human_grounding.oracle import ORACLE_MODEL_NAME
from human_grounding.threshold_auc import HUMAN_MODEL_NAME

EXPERIMENT_DATASETS: dict[str, list[str]] = {
    "policy": ["rai", "welfare"],
    "gov-ai": ["gov-ai"],
}
DATASETS = ["rai", "welfare", "gov-ai"]

# Reference upper bounds that must be excluded when picking the "best model".
NON_MODEL_ROWS = {HUMAN_MODEL_NAME, ORACLE_MODEL_NAME}

# Paper-spec design constant (not derivable from any data artifact).
TOTAL_PLACEMENTS_APPROX = (6 * 14 * 20) + (6 * 7 * 20)

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

# --------------------------------------------------------------------------- #
# Shared data loaders / derived facts
# --------------------------------------------------------------------------- #
def load_alignment_bootstraps() -> pl.DataFrame | None:
    """Per-model / per-demographic / per-iteration alignment AUCs, incl. the
    ``Human`` baseline and the ``human-mds-oracle`` rows."""
    return read_csv("alignment_results_gov-ai_policy.csv")


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
# Alpha-reliability helpers
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


# --------------------------------------------------------------------------- #
# Clustering helpers
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
    oracle_stress_df = read_csv("oracle_stress.csv")
    oracle_stress = (
        dict(
            zip(
                oracle_stress_df["dataset"].to_list(),
                oracle_stress_df["stress"].to_list(),
            )
        )
        if oracle_stress_df is not None
        else {}
    )
    oracle_q = (
        dict(
            zip(
                oracle_stress_df["dataset"].to_list(),
                oracle_stress_df["n_components"].to_list(),
            )
        )
        if oracle_stress_df is not None
        else {}
    )

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

    # Fairness raw / adjusted disparity, each with its [lo,hi] 95% CI.
    def _val_ci(cell: str) -> tuple[str, str | None, str | None]:
        m = re.match(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]", cell)
        if not m:
            return cell.split()[0] if cell.split() else "", None, None
        return m.group(1), m.group(2), m.group(3)

    controlled = read_text("fairness_group_gap_controlled.tex")
    fair = {}
    for r in parse_latex_rows(controlled) if controlled else []:
        if len(r) >= 6:
            ds = next((d for d in DATASETS if pretty_dataset(d) == r[0]), None)
            if ds:
                raw, raw_lo, raw_hi = _val_ci(r[2])
                adj, adj_lo, adj_hi = _val_ci(r[4])
                fair[ds] = {
                    "raw": raw,
                    "raw_lo": raw_lo,
                    "raw_hi": raw_hi,
                    "adj": adj,
                    "adj_lo": adj_lo,
                    "adj_hi": adj_hi,
                }

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

    def f2(x: object) -> str:
        """2 significant digits (after any leading zeros), not a fixed decimal count."""
        if x is None:
            return ""
        x = float(x)
        if x == 0:
            return "0.0"
        decimals = max(0, 1 - math.floor(math.log10(abs(x))))
        return f"{round(x, decimals):.{decimals}f}"

    def f1(x: object) -> str:
        return "" if x is None else f"{float(x):.1f}"

    def r2(text: str) -> str:
        """Reformat a raw numeric string (e.g. from a parsed artifact) to 2 sig figs."""
        return f2(float(text)) if text else ""

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
            TOTAL_PLACEMENTS_APPROX,
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
                r2(facts.get(ds, {}).get("human", "")),
                f"Human inter-rater reliability (AUC) for {pretty_dataset(ds)}",
            )
        )
    wr = [within.get(d) for d in ("rai", "welfare") if within.get(d) is not None]
    wr_summary = sum(wr) / len(wr) if wr else None
    lines.append(
        _tex_cmd(
            "WithinRaterAUC",
            f2(wr_summary),
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
            f2(men[0]) if men else "",
            "Human alignment AUC for male statement authors (Responsible AI)",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUC",
            f2(women[0]) if women else "",
            "Human alignment AUC for female statement authors (Responsible AI)",
        ),
        _tex_cmd(
            "ResponsibleAIMenAUCLO",
            f2(men[1]) if men else "",
            "Lower confidence interval for male AUC",
        ),
        _tex_cmd(
            "ResponsibleAIMenAUCHI",
            f2(men[2]) if men else "",
            "Upper confidence interval for male AUC",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUCLO",
            f2(women[1]) if women else "",
            "Lower confidence interval for female AUC",
        ),
        _tex_cmd(
            "ResponsibleAIWomenAUCHI",
            f2(women[2]) if women else "",
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
            f2(best_overall_auc),
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
                r2(facts.get(ds, {}).get("oracle", "")),
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
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}OracleStress",
                f2(oracle_stress.get(ds)),
                f"Oracle SMACOF normalized fit stress (Stress-1) for {pretty_dataset(ds)}",
            )
        )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        q = oracle_q.get(ds)
        lines.append(
            _tex_cmd(
                f"{p}OracleQ",
                "" if q is None else str(int(q)),
                f"Oracle embedding dimensionality q selected via diminishing-returns search for {pretty_dataset(ds)}",
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
                r2(fair.get(ds, {}).get("raw", "")),
                f"Group alignment disparity for {pretty_dataset(ds)}",
            )
        )
    for ds in ("rai", "welfare"):
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}GroupGapLO",
                r2(fair.get(ds, {}).get("raw_lo", "")),
                f"Lower CI for {pretty_dataset(ds)} group alignment disparity",
            )
        )
        lines.append(
            _tex_cmd(
                f"{p}GroupGapHI",
                r2(fair.get(ds, {}).get("raw_hi", "")),
                f"Upper CI for {pretty_dataset(ds)} group alignment disparity",
            )
        )
    for ds in ("rai", "welfare"):
        p = CMD_PREFIX[ds]
        lines.append(
            _tex_cmd(
                f"{p}AdjustedGroupGap",
                r2(fair.get(ds, {}).get("adj", "")),
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
            f2(sum(aris) / len(aris) if aris else None),
            "Human clustering ARI (mean across datasets)",
        )
    )
    policy_human_ari = _policy_human_ari()
    lines.append(
        _tex_cmd(
            "PolicyHumanARI",
            f2(policy_human_ari["ari"]) if policy_human_ari else "",
            "Human clustering ARI pooled across policy (responsible AI + welfare)",
        )
    )
    lines.append(
        _tex_cmd(
            "PolicyHumanARILO",
            f2(policy_human_ari["lo"]) if policy_human_ari else "",
            "Lower CI for policy human clustering ARI",
        )
    )
    lines.append(
        _tex_cmd(
            "PolicyHumanARIHI",
            f2(policy_human_ari["hi"]) if policy_human_ari else "",
            "Upper CI for policy human clustering ARI",
        )
    )
    for ds in DATASETS:
        p = CMD_PREFIX[ds]
        h = human_ari.get(ds)
        lines.append(
            _tex_cmd(
                f"{p}HumanARI",
                f2(h["ari"]) if h else "",
                f"Human clustering ARI for {pretty_dataset(ds)}",
            )
        )
        bm = best_model_ari.get(ds)
        lines.append(
            _tex_cmd(
                f"{p}BestModelARI",
                f2(bm[1]) if bm else "",
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
            f2(max_gain),
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
    KEY_VALUES_PATH.write_text(build_key_values_tex())
    print(f"Wrote LaTeX key-values to {KEY_VALUES_PATH}")


if __name__ == "__main__":
    main()

