"""
Cluster an existing knowledge bank's claims.

Runs k-means on pre-computed claim embeddings and saves cluster
assignments alongside the existing knowledge bank files.

Usage:
    python -m scripts.cluster_knowledge_bank --knowledge_bank_dir knowledge_banks/forget10
    python -m scripts.cluster_knowledge_bank --knowledge_bank_dir knowledge_banks/forget10 --n_clusters 5
"""
import argparse

import numpy as np

from eco.attack.claim_cluster import (
    auto_select_k,
    cluster_claims,
    save_clusters,
)
from eco.attack.leak_detector import load_knowledge_bank
from eco.utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--knowledge_bank_dir",
    type=str,
    required=True,
    help="Directory containing claims.json and embeddings.npy",
)
parser.add_argument(
    "--n_clusters",
    type=int,
    default=0,
    help="Number of clusters (0 = auto-select via silhouette score)",
)
parser.add_argument(
    "--min_cluster_size",
    type=int,
    default=2,
    help="Minimum claims per cluster; smaller clusters are merged",
)
args = parser.parse_args()

# Load existing knowledge bank
claims, embeddings = load_knowledge_bank(args.knowledge_bank_dir)
log_print(f"Loaded knowledge bank: {len(claims)} claims, embeddings shape {embeddings.shape}")

# Determine k
if args.n_clusters > 0:
    n_clusters = args.n_clusters
    log_print(f"Using specified n_clusters={n_clusters}")
else:
    n_clusters = auto_select_k(embeddings)
    log_print(f"Auto-selected n_clusters={n_clusters} (silhouette score)")

# Cluster
cluster_labels, centroids = cluster_claims(
    embeddings, n_clusters, min_cluster_size=args.min_cluster_size
)
n_final = centroids.shape[0]
log_print(f"Final clusters: {n_final} (after merging small clusters)")

# Log cluster sizes
unique, counts = np.unique(cluster_labels, return_counts=True)
for cid, count in zip(unique, counts):
    log_print(f"  Cluster {cid}: {count} claims")

# Save
save_clusters(cluster_labels, centroids, args.knowledge_bank_dir)
log_print(f"Cluster artifacts saved to {args.knowledge_bank_dir}")
