"""
Tests for claim clustering utilities.

Usage:
    conda run -n eco python -m pytest tests/test_claim_cluster.py -v
"""

import json
import os

import numpy as np
import pytest

from eco.attack.claim_cluster import (
    auto_select_k,
    cluster_claims,
    load_clusters,
    save_clusters,
)


def _make_two_cluster_embeddings(n_per_cluster=10, dim=32, seed=42):
    """Create embeddings with two well-separated clusters."""
    rng = np.random.default_rng(seed)
    cluster_0 = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) + 5.0
    cluster_1 = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) - 5.0
    return np.concatenate([cluster_0, cluster_1], axis=0)


class TestClusterClaims:
    def test_basic_clustering(self):
        embeddings = _make_two_cluster_embeddings()
        labels, centroids = cluster_claims(embeddings, n_clusters=2)
        assert labels.shape == (20,)
        assert centroids.shape[0] == 2
        # First 10 should be one cluster, last 10 another
        assert len(set(labels[:10])) == 1
        assert len(set(labels[10:])) == 1
        assert labels[0] != labels[10]

    def test_labels_contiguous(self):
        embeddings = _make_two_cluster_embeddings()
        labels, centroids = cluster_claims(embeddings, n_clusters=2)
        unique = sorted(set(labels))
        assert unique == [0, 1]

    def test_merge_small_clusters(self):
        """Clusters below min_cluster_size get merged."""
        embeddings = _make_two_cluster_embeddings(n_per_cluster=10)
        # Request 5 clusters but min_cluster_size=5 should force merging
        labels, centroids = cluster_claims(
            embeddings, n_clusters=5, min_cluster_size=5
        )
        unique, counts = np.unique(labels, return_counts=True)
        assert all(c >= 5 for c in counts)
        # Labels should still be contiguous
        assert list(unique) == list(range(len(unique)))

    def test_single_cluster(self):
        embeddings = np.random.default_rng(0).standard_normal((10, 16)).astype(np.float32)
        labels, centroids = cluster_claims(embeddings, n_clusters=1)
        assert np.all(labels == 0)
        assert centroids.shape[0] == 1


class TestAutoSelectK:
    def test_auto_finds_two_clusters(self):
        embeddings = _make_two_cluster_embeddings(n_per_cluster=20, dim=32)
        k = auto_select_k(embeddings, k_min=2, k_max=6)
        assert k == 2

    def test_respects_k_range(self):
        embeddings = _make_two_cluster_embeddings(n_per_cluster=20)
        k = auto_select_k(embeddings, k_min=3, k_max=5)
        assert 3 <= k <= 5


class TestSaveLoadClusters:
    def test_roundtrip(self, tmp_path):
        labels = np.array([0, 0, 1, 1, 2])
        centroids = np.random.default_rng(0).standard_normal((3, 16)).astype(np.float32)
        output_dir = str(tmp_path / "kb")

        save_clusters(labels, centroids, output_dir)

        loaded_labels, loaded_centroids = load_clusters(output_dir)
        np.testing.assert_array_equal(labels, loaded_labels)
        np.testing.assert_array_almost_equal(centroids, loaded_centroids)

    def test_files_created(self, tmp_path):
        labels = np.array([0, 1])
        centroids = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
        output_dir = str(tmp_path / "kb")

        save_clusters(labels, centroids, output_dir)

        assert os.path.exists(os.path.join(output_dir, "claim_clusters.json"))
        assert os.path.exists(os.path.join(output_dir, "centroids.npy"))

    def test_json_structure(self, tmp_path):
        labels = np.array([0, 0, 1])
        centroids = np.random.default_rng(0).standard_normal((2, 8)).astype(np.float32)
        output_dir = str(tmp_path / "kb")
        save_clusters(labels, centroids, output_dir)

        with open(os.path.join(output_dir, "claim_clusters.json")) as f:
            data = json.load(f)
        assert data["n_clusters"] == 2
        assert data["cluster_labels"] == [0, 0, 1]

    def test_load_validates_mismatch(self, tmp_path):
        output_dir = str(tmp_path / "kb")
        os.makedirs(output_dir)
        with open(os.path.join(output_dir, "claim_clusters.json"), "w") as f:
            json.dump({"cluster_labels": [0, 1], "n_clusters": 5}, f)
        np.save(
            os.path.join(output_dir, "centroids.npy"),
            np.zeros((2, 8)),  # n_clusters=5 but only 2 centroids
        )
        with pytest.raises(ValueError, match="n_clusters=5"):
            load_clusters(output_dir)
