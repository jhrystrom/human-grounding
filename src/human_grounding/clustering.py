"""Utility functions for HDBSCAN clustering on normalised coordinates."""

import numpy as np
import polars as pl
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances


def compute_distance_matrix(
    coords: np.ndarray, metric: str = "euclidean"
) -> np.ndarray:
    """Compute a pairwise distance matrix from coordinate data.

    Args:
        coords: Array of shape (n_samples, n_features) containing coordinates.
        metric: Distance metric to use. Defaults to "euclidean".

    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples).
    """
    return pairwise_distances(coords, metric=metric)


def cluster_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int | None = None,
) -> np.ndarray:
    """Run HDBSCAN clustering on a precomputed distance matrix.

    Args:
        distance_matrix: Symmetric pairwise distance matrix of shape (n, n).
        min_cluster_size: Minimum number of points to form a cluster.
        min_samples: Number of samples in a neighbourhood for a point to be a
            core point. Defaults to min_cluster_size if None.

    Returns:
        Array of cluster labels. Noise points are labelled -1.
    """
    clusterer = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    clusterer.fit(distance_matrix)
    return clusterer.labels_


def cluster_user_session(
    df: pl.DataFrame, distance_threshold: float = 0.2
) -> dict[int, int]:
    """
    Clusters using a scale-invariant 'Standard Gap'.

    1. Finds the distance to the nearest neighbor for every point.
    2. Takes the MEDIAN of these nearest-neighbor distances as the 'Atomic Unit'.
    3. Links items if they are within `gap_multiplier` * Atomic Unit.

    Returns:
        Mapping statement_id -> cluster label, with singleton clusters mapped to -1.
    """
    # 1. Prep
    df = df.sort("statement_id")
    ids = df["statement_id"].to_list()
    coords = df.select("x_normalised", "y_normalised").to_numpy()
    clusterer = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, linkage="ward"
    )
    labels = clusterer.fit_predict(coords)
    # Convert singleton clusters to -1
    labels = np.where(np.bincount(labels)[labels] == 1, -1, labels)
    return dict(zip(ids, labels))
