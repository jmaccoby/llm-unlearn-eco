"""
Build a knowledge bank from R-TOFU forget-set gold answers.

Extracts factual claims (sentence-split gold answers), deduplicates them,
computes SentenceTransformer embeddings, and saves both to disk.
Optionally clusters the claims for claim-specific soft token training.

Usage:
    python -m scripts.build_knowledge_bank --split forget10
    python -m scripts.build_knowledge_bank --split forget10 --n_clusters 5
    python -m scripts.build_knowledge_bank --split forget05 --output_dir knowledge_banks
"""
import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from eco.attack.leak_detector import build_claims, save_knowledge_bank
from eco.dataset.rtofu import RTOFU
from eco.utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--output_dir", type=str, default="knowledge_banks")
parser.add_argument(
    "--st_model",
    type=str,
    default="paraphrase-MiniLM-L6-v2",
    help="SentenceTransformer model name",
)
parser.add_argument(
    "--n_clusters",
    type=int,
    default=0,
    help="Number of claim clusters (0 = skip clustering, -1 = auto-select)",
)
parser.add_argument(
    "--min_cluster_size",
    type=int,
    default=2,
    help="Minimum claims per cluster; smaller clusters are merged",
)
args = parser.parse_args()

# Load dataset
log_print(f"Loading R-TOFU split: {args.split}")
rtofu = RTOFU()
rtofu.download()
dataset = rtofu.dataset[args.split]

# Extract and deduplicate claims from gold answers
answers = dataset["answer"]
claims = build_claims(answers)
log_print(f"Extracted {len(claims)} unique claims from {len(answers)} answers")

# Compute embeddings
log_print(f"Computing embeddings with {args.st_model}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_model = SentenceTransformer(
    args.st_model,
    device=device,  # type: ignore[arg-type]
)
embeddings = st_model.encode(claims, show_progress_bar=True)
embeddings = np.array(embeddings, dtype=np.float32)
log_print(f"Embeddings shape: {embeddings.shape}")

# Save
output_dir = f"{args.output_dir}/{args.split}"
save_knowledge_bank(claims, embeddings, output_dir)
log_print(f"Knowledge bank saved to {output_dir}")

# Optionally cluster claims
if args.n_clusters != 0:
    from eco.attack.claim_cluster import (
        auto_select_k,
        cluster_claims,
        save_clusters,
    )

    if args.n_clusters < 0:
        n_clusters = auto_select_k(embeddings)
        log_print(f"Auto-selected n_clusters={n_clusters} (silhouette score)")
    else:
        n_clusters = args.n_clusters
        log_print(f"Using specified n_clusters={n_clusters}")

    cluster_labels, centroids = cluster_claims(
        embeddings, n_clusters, min_cluster_size=args.min_cluster_size
    )
    n_final = centroids.shape[0]
    log_print(f"Final clusters: {n_final} (after merging small clusters)")

    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cid, count in zip(unique, counts):
        log_print(f"  Cluster {cid}: {count} claims")

    save_clusters(cluster_labels, centroids, output_dir)
    log_print(f"Cluster artifacts saved to {output_dir}")
