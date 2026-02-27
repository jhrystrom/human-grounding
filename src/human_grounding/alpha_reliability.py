from functools import cache

import numpy as np


def vectorized_judgment(di: np.ndarray, dj: np.ndarray, threshold: float) -> np.ndarray:
    """Compute judgment for each triplet - simplified single-pass version."""
    judgment = np.zeros(len(di), dtype=np.int8)
    judgment[(di < dj) & (dj >= threshold * di)] = 1
    judgment[(dj < di) & (di >= threshold * dj)] = -1
    return judgment


@cache
def generate_triplets_fully_vectorized(
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully vectorized triplet generation."""
    i_idx, j_idx = np.triu_indices(n, k=1)
    num_pairs = len(i_idx)

    all_anchors = np.broadcast_to(np.arange(n), (num_pairs, n))
    valid_mask = (all_anchors != i_idx[:, None]) & (all_anchors != j_idx[:, None])

    pair_indices, anchor_positions = np.where(valid_mask)

    anchors = anchor_positions
    points_i = i_idx[pair_indices]
    points_j = j_idx[pair_indices]

    return anchors, points_i, points_j


def normalized_auc_logx(y: np.ndarray, x: np.ndarray) -> float:
    """
    Normalized AUC of y(x) integrating over log(x).

    This makes AUC more stable/comparable when x is log-spaced.
    """
    lx = np.log(x)
    area = float(np.trapezoid(y, x=lx))
    denom = float(lx[-1] - lx[0])
    return 0.0 if denom == 0 else area / denom
