"""Clustering utilities for knowledge bank claims.

Groups claims into semantic clusters so that per-cluster soft tokens
can be trained and selected at inference time.
"""

import json
import os

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances


def auto_select_k(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int | None = None,
) -> int:
    """Choose *k* by maximising silhouette score over a range.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_claims, embed_dim)``.
    k_min : int
        Minimum number of clusters to try.
    k_max : int | None
        Maximum number of clusters to try.  Defaults to
        ``min(20, n_claims // 3)``.
    """
    n = embeddings.shape[0]
    if n < 4:
        return k_min  # not enough data for silhouette-based selection
    if k_max is None:
        k_max = min(20, n // 3)
    k_max = max(k_min, min(k_max, n - 1))

    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_k, best_score = k, score

    return best_k


def cluster_claims(
    embeddings: np.ndarray,
    n_clusters: int,
    min_cluster_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster claim embeddings with k-means.

    Clusters smaller than *min_cluster_size* are merged into their
    nearest neighbour (by centroid distance).

    Returns
    -------
    cluster_labels : np.ndarray
        Shape ``(n_claims,)`` — cluster ID for each claim.  IDs are
        contiguous starting from 0.
    centroids : np.ndarray
        Shape ``(n_clusters_final, embed_dim)``.
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = km.fit_predict(embeddings)

    # Merge small clusters into nearest neighbour
    labels = _merge_small_clusters(labels, embeddings, min_cluster_size)

    # Renumber to contiguous 0..K-1 and recompute centroids from data
    labels, centroids = _renumber_and_recompute(labels, embeddings)

    return labels, centroids


def _merge_small_clusters(
    labels: np.ndarray,
    embeddings: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Merge clusters with fewer than *min_size* members into the
    nearest larger cluster (by centroid Euclidean distance).

    Returns the updated labels array.  Centroids are recomputed from
    *embeddings* on each pass so distances are always accurate.
    """
    while True:
        unique, counts = np.unique(labels, return_counts=True)
        small = unique[counts < min_size]
        large = unique[counts >= min_size]
        if len(small) == 0 or len(large) == 0:
            break
        # Compute centroids from the actual data
        small_centroids = np.array(
            [embeddings[labels == c].mean(axis=0) for c in small]
        )
        large_centroids = np.array(
            [embeddings[labels == c].mean(axis=0) for c in large]
        )
        dists = euclidean_distances(small_centroids, large_centroids)
        for i, s_id in enumerate(small):
            nearest_large = large[np.argmin(dists[i])]
            labels[labels == s_id] = nearest_large

    return labels


def _renumber_and_recompute(
    labels: np.ndarray, embeddings: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Re-map cluster IDs to contiguous 0..K-1 and recompute centroids."""
    unique_ids = np.unique(labels)
    id_map = {old: new for new, old in enumerate(unique_ids)}
    new_labels = np.array([id_map[l] for l in labels])
    new_centroids = np.array(
        [embeddings[new_labels == c].mean(axis=0) for c in range(len(unique_ids))]
    )
    return new_labels, new_centroids


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

def save_clusters(
    cluster_labels: np.ndarray,
    centroids: np.ndarray,
    output_dir: str,
) -> None:
    """Persist cluster labels and centroids to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    n_clusters = int(centroids.shape[0])
    with open(os.path.join(output_dir, "claim_clusters.json"), "w") as f:
        json.dump(
            {
                "cluster_labels": cluster_labels.tolist(),
                "n_clusters": n_clusters,
            },
            f,
            indent=2,
        )
    np.save(os.path.join(output_dir, "centroids.npy"), centroids)


def load_clusters(
    knowledge_bank_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load cluster labels and centroids from *knowledge_bank_dir*.

    Returns
    -------
    cluster_labels : np.ndarray
        Shape ``(n_claims,)``.
    centroids : np.ndarray
        Shape ``(n_clusters, embed_dim)``.
    """
    clusters_path = os.path.join(knowledge_bank_dir, "claim_clusters.json")
    centroids_path = os.path.join(knowledge_bank_dir, "centroids.npy")
    with open(clusters_path) as f:
        data = json.load(f)
    cluster_labels = np.array(data["cluster_labels"])
    centroids = np.load(centroids_path)
    if data["n_clusters"] != centroids.shape[0]:
        raise ValueError(
            f"claim_clusters.json says n_clusters={data['n_clusters']} but "
            f"centroids.npy has shape {centroids.shape}"
        )
    return cluster_labels, centroids
