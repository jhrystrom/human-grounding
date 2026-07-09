"""Human-MDS oracle embedding.

A simple oracle-style upper bound that produces **one global embedding vector
per statement**, fitted directly from stakeholder SpAM layouts rather than from
text. It estimates how well a single global embedding could reproduce
stakeholder similarity structure when given access to the observed human
distance signal.

The oracle plugs into the same triplet alpha/AUC evaluation as the neural
embedding models (see :mod:`human_grounding.evaluate`): for a given
``(dataset, seed)`` group it fits one consensus embedding over all raters in
that group, and that embedding is then scored on the group's human triplets
exactly like any other model. Because the embedding is fitted to reproduce
*Euclidean* distances, the evaluation compares it under Euclidean (not cosine)
distance.

This is not a deployable text model — it is an upper-bound reference row in the
model-comparison figure.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from loguru import logger
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import smacof

# Distinct name so the oracle appears as its own row/bar in the figures.
ORACLE_MODEL_NAME = "human-mds-oracle"
ORACLE_PRETTY_NAME = "Human-MDS oracle"

# Embedding dimensionality. The plan suggests q in {2, 5, 10, 20}; 10 is a
# reasonable default (capped below at n_statements - 1).
DEFAULT_N_COMPONENTS = 10


def is_oracle_model(model: str) -> bool:
    """True if ``model`` refers to the Human-MDS oracle."""
    return model == ORACLE_MODEL_NAME


def _normalized_layout_distances(coords: np.ndarray) -> np.ndarray | None:
    """Center a single layout and rescale by its median pairwise distance.

    Returns a full ``(k, k)`` distance matrix, or ``None`` if the layout has
    fewer than two points or a degenerate (zero / non-finite) scale.
    """
    if coords.shape[0] < 2:
        return None
    centered = coords - coords.mean(axis=0)
    condensed = pdist(centered)
    scale = float(np.median(condensed))
    if not np.isfinite(scale) or scale <= 0:
        return None
    return squareform(condensed / scale)


def build_consensus_dissimilarity(
    coordinates: pl.DataFrame,
    min_cooccurrence: int = 1,
) -> tuple[list[int], np.ndarray]:
    """Aggregate per-rater layouts into a consensus dissimilarity matrix.

    Each ``user_id`` defines one normalized SpAM layout; within-layout
    Euclidean distances are averaged over the layouts in which each statement
    pair co-occurs. Pairs observed in fewer than ``min_cooccurrence`` layouts
    are treated as missing and back-filled with the mean observed dissimilarity
    (the practical compromise from the oracle plan, so metric MDS sees a full
    matrix while only observed triplets are ever scored downstream).

    Returns ``(statement_ids, dissimilarity)`` where ``dissimilarity`` is a
    symmetric ``(N, N)`` matrix aligned to ``statement_ids``.
    """
    statement_ids = sorted(coordinates["statement_id"].unique().to_list())
    index = {sid: i for i, sid in enumerate(statement_ids)}
    n = len(statement_ids)

    sum_mat = np.zeros((n, n))
    count_mat = np.zeros((n, n))

    for (_user_id,), group in coordinates.group_by("user_id"):
        layout = group.unique(subset="statement_id").sort("statement_id")
        coords = layout.select("x", "y").to_numpy().astype(float)
        distances = _normalized_layout_distances(coords)
        if distances is None:
            continue
        rows = np.array([index[sid] for sid in layout["statement_id"].to_list()])
        block = np.ix_(rows, rows)
        sum_mat[block] += distances
        occurrence = np.ones_like(distances)
        np.fill_diagonal(occurrence, 0.0)
        count_mat[block] += occurrence

    observed = count_mat >= min_cooccurrence
    np.fill_diagonal(observed, False)

    consensus = np.zeros((n, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        consensus[observed] = sum_mat[observed] / count_mat[observed]

    fill_value = float(consensus[observed].mean()) if observed.any() else 0.0

    missing = ~observed
    np.fill_diagonal(missing, False)
    n_missing = int(missing.sum())
    if n_missing:
        logger.debug(
            f"Oracle: filling {n_missing // 2} unobserved statement pair(s) "
            f"with mean dissimilarity {fill_value:.3f}"
        )
        consensus[missing] = fill_value

    # Enforce exact symmetry and a zero diagonal.
    consensus = (consensus + consensus.T) / 2
    np.fill_diagonal(consensus, 0.0)
    return statement_ids, consensus


def fit_oracle_embeddings(
    coordinates: pl.DataFrame,
    n_components: int = DEFAULT_N_COMPONENTS,
    n_init: int = 4,
    max_iter: int = 300,
    random_state: int = 0,
    min_cooccurrence: int = 1,
) -> pl.DataFrame:
    """Fit one global MDS embedding per statement from stakeholder layouts.

    Metric MDS (SMACOF) is fitted on the consensus dissimilarity matrix with
    several random initializations; the lowest-stress solution is kept.

    Returns a frame with columns ``statement_id`` and ``embedding`` (a
    fixed-width float array), one row per statement, ready to be attached to the
    statement table exactly like a text-embedding column.
    """
    statement_ids, dissimilarity = build_consensus_dissimilarity(
        coordinates, min_cooccurrence=min_cooccurrence
    )
    n = len(statement_ids)
    if n < 2:
        raise ValueError(f"Oracle needs at least 2 statements to fit, got {n}")

    # n_components must stay below the number of points.
    q = min(n_components, n - 1)

    embedding, stress = smacof(
        dissimilarity,
        metric=True,
        n_components=q,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    logger.debug(f"Oracle: fitted {n} statements in q={q} dims (stress={stress:.4f})")

    return pl.DataFrame(
        {
            "statement_id": statement_ids,
            "embedding": pl.Series(
                "embedding",
                embedding.tolist(),
                dtype=pl.Array(pl.Float64, q),
            ),
        }
    )
