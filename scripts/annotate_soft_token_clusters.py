"""
Annotate existing soft token training data with cluster IDs.

Reads train.jsonl and a clustered knowledge bank, finds the nearest
claim for each example's leaking continuation via cosine similarity,
and writes back the file with matched_claim_index and cluster_id fields.

No NLI model needed — uses only the sentence transformer and pre-computed
claim embeddings.

Usage:
    python -m scripts.annotate_soft_token_clusters --split forget10
    python -m scripts.annotate_soft_token_clusters --split forget10 --knowledge_bank_dir knowledge_banks
"""
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from eco.attack.claim_cluster import load_clusters
from eco.attack.leak_detector import load_knowledge_bank
from eco.evaluator.utils import split_sentences
from eco.utils import log_print

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="soft_token_data")
parser.add_argument("--knowledge_bank_dir", type=str, default="knowledge_banks")
parser.add_argument(
    "--st_model",
    type=str,
    default="paraphrase-MiniLM-L6-v2",
    help="SentenceTransformer model name",
)
args = parser.parse_args()

# Load knowledge bank and clusters
kb_dir = f"{args.knowledge_bank_dir}/{args.split}"
claims, bank_embeddings = load_knowledge_bank(kb_dir)
cluster_labels, _ = load_clusters(kb_dir)
n_clusters = int(cluster_labels.max()) + 1
log_print(f"Loaded knowledge bank: {len(claims)} claims, {n_clusters} clusters")

# Load sentence transformer
log_print(f"Loading sentence transformer: {args.st_model}")
st_model = SentenceTransformer(args.st_model)

# Load training data
data_path = f"{args.data_dir}/{args.split}/train.jsonl"
log_print(f"Loading training data from {data_path}")
examples = []
with open(data_path) as f:
    for line in f:
        examples.append(json.loads(line))
log_print(f"Training examples: {len(examples)}")

# Annotate each example
log_print("\n--- Annotating examples ---")
for i, ex in enumerate(examples):
    # Take the first sentence of the leaking continuation
    sentences = split_sentences(ex["leaking_continuation"])
    if not sentences:
        continue
    query = sentences[0]

    # Find nearest claim by cosine similarity
    query_emb = st_model.encode(query, show_progress_bar=False)
    sims = cosine_similarity([query_emb], bank_embeddings)[0]
    nearest_idx = int(np.argmax(sims))

    ex["matched_claim_index"] = nearest_idx
    ex["cluster_id"] = int(cluster_labels[nearest_idx])

    if (i + 1) % 50 == 0 or i == len(examples) - 1:
        log_print(f"  Annotated {i + 1}/{len(examples)}")

# Log cluster distribution
from collections import Counter
cluster_counts = Counter(ex.get("cluster_id") for ex in examples)
log_print(f"\nCluster distribution:")
for cid in sorted(cluster_counts):
    log_print(f"  Cluster {cid}: {cluster_counts[cid]} examples")
missing = set(range(n_clusters)) - set(cluster_counts)
if missing:
    log_print(f"  Warning: clusters with no examples: {sorted(missing)}")

# Write back
with open(data_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")
log_print(f"\nAnnotated data written to {data_path}")
