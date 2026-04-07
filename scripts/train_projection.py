"""Train a contrastive projection head for embedding-only Stage 2 leak detection.

Maps SentenceTransformer embeddings into a space where distance to claim
cluster centroids discriminates forget-set leaks from retain false positives.

Positives: forget CoT leaking sentences (with cluster IDs from soft token data).
Hard negatives: retain CoT sentences flagged by the Stage 1 classifier.

Evaluates on a held-out split and reports detection accuracy at various
distance thresholds.

Usage:
    python -m scripts.train_projection --split forget10
"""
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from eco.attack.claim_cluster import load_clusters
from eco.attack.classifier import CorruptionClassifier
from eco.attack.leak_detector import load_knowledge_bank
from eco.evaluator.utils import split_sentences
from eco.utils import log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split", type=str, default="forget10",
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="leak_detector_data")
parser.add_argument("--soft_token_data_dir", type=str, default="soft_token_data")
parser.add_argument("--knowledge_bank_dir", type=str, default="knowledge_banks")
parser.add_argument("--leak_classifier_path", type=str, default=None)
parser.add_argument("--classifier_threshold", type=float, default=0.5)
parser.add_argument("--proj_dim", type=int, default=128)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--margin", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--val_frac", type=float, default=0.2)
parser.add_argument("--neg_mode", type=str, default="all",
                    choices=["all", "hard_only", "balanced"],
                    help="all: use all negatives; hard_only: Stage 1-flagged only; "
                         "balanced: subsample negatives to match positive count per epoch")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default=None,
                    help="Save projection head and results (optional)")
args = parser.parse_args()

seed_everything(args.seed)

classifier_path = args.leak_classifier_path or f"leak_classifiers/{args.split}"
kb_dir = f"{args.knowledge_bank_dir}/{args.split}"

# -------------------------------------------------------------------------
# Load resources
# -------------------------------------------------------------------------

log_print("Loading knowledge bank and clusters...")
claims, bank_embeddings = load_knowledge_bank(kb_dir)
cluster_labels, centroids = load_clusters(kb_dir)
n_clusters = centroids.shape[0]
st_dim = centroids.shape[1]
log_print(f"  {len(claims)} claims, {n_clusters} clusters, ST dim={st_dim}")

log_print("Loading Stage 1 classifier...")
classifier = CorruptionClassifier(
    model_name="roberta-base",
    model_path=classifier_path,
    batch_size=32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2", device=device)

# -------------------------------------------------------------------------
# Build positive examples from soft token training data
# -------------------------------------------------------------------------

log_print("\nBuilding positive examples (forget leaking sentences)...")
st_data_path = f"{args.soft_token_data_dir}/{args.split}/train.jsonl"
with open(st_data_path) as f:
    st_examples = [json.loads(line) for line in f]

positives = []  # (sentence_text, cluster_id)
for ex in st_examples:
    cluster_id = ex.get("cluster_id")
    if cluster_id is None:
        continue
    sentences = split_sentences(ex["leaking_continuation"])
    for sent in sentences:
        positives.append((sent, cluster_id))

log_print(f"  {len(positives)} positive sentences from {len(st_examples)} examples")

# -------------------------------------------------------------------------
# Build hard negative examples from retain CoTs
# -------------------------------------------------------------------------

log_print("Building negative examples from retain gold CoTs...")
from eco.dataset.rtofu import RTOFU

retain_split = {"forget01": "retain90", "forget05": "retain90", "forget10": "retain90"}[args.split]
data_module = RTOFU(formatting_tokens=None, eos_token="")
data_module.download()
retain_ds = data_module.dataset[retain_split]
log_print(f"  Loaded {len(retain_ds)} retain examples from {retain_split}")

retain_sentences = []
for row in retain_ds:
    cot = row["cot"]
    if cot:
        retain_sentences.extend(split_sentences(cot))
log_print(f"  {len(retain_sentences)} total retain sentences")

log_print("  Running Stage 1 classifier on retain sentences...")
flags = classifier.predict(retain_sentences, args.classifier_threshold)
hard_negatives = [s for s, f in zip(retain_sentences, flags) if f]
easy_negatives = [s for s, f in zip(retain_sentences, flags) if not f]

log_print(f"  {len(hard_negatives)} hard negatives (Stage 1 flagged)")
log_print(f"  {len(easy_negatives)} easy negatives (unflagged)")

if args.neg_mode == "hard_only":
    all_negatives = hard_negatives
elif args.neg_mode == "all":
    all_negatives = hard_negatives + easy_negatives
elif args.neg_mode == "balanced":
    # Use all negatives but will subsample per epoch during training
    all_negatives = hard_negatives + easy_negatives
log_print(f"  Neg mode: {args.neg_mode}, using {len(all_negatives)} negatives")

if len(positives) == 0 or len(all_negatives) == 0:
    log_print("ERROR: need both positives and negatives to train")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# Encode all sentences
# -------------------------------------------------------------------------

log_print("\nEncoding sentences with SentenceTransformer...")
pos_texts = [t for t, _ in positives]
pos_cluster_ids = [c for _, c in positives]

pos_embeddings = st_model.encode(pos_texts, show_progress_bar=True, batch_size=128)
neg_embeddings = st_model.encode(all_negatives, show_progress_bar=True, batch_size=128)

pos_embeddings = torch.tensor(pos_embeddings, dtype=torch.float32)
neg_embeddings = torch.tensor(neg_embeddings, dtype=torch.float32)
pos_cluster_ids = torch.tensor(pos_cluster_ids, dtype=torch.long)
centroids_t = torch.tensor(centroids, dtype=torch.float32).to(device)

log_print(f"  Positives: {pos_embeddings.shape}, Negatives: {neg_embeddings.shape}")

# -------------------------------------------------------------------------
# Train/val split
# -------------------------------------------------------------------------

n_pos = len(pos_embeddings)
n_neg = len(neg_embeddings)
n_pos_val = int(n_pos * args.val_frac)
n_neg_val = int(n_neg * args.val_frac)

perm_pos = torch.randperm(n_pos)
perm_neg = torch.randperm(n_neg)

val_pos_emb = pos_embeddings[perm_pos[:n_pos_val]].to(device)
val_pos_cid = pos_cluster_ids[perm_pos[:n_pos_val]].to(device)
train_pos_emb = pos_embeddings[perm_pos[n_pos_val:]].to(device)
train_pos_cid = pos_cluster_ids[perm_pos[n_pos_val:]].to(device)

val_neg_emb = neg_embeddings[perm_neg[:n_neg_val]].to(device)
train_neg_emb = neg_embeddings[perm_neg[n_neg_val:]].to(device)

log_print(f"  Train: {len(train_pos_emb)} pos, {len(train_neg_emb)} neg")
log_print(f"  Val:   {len(val_pos_emb)} pos, {len(val_neg_emb)} neg")

# -------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------


from eco.attack.learned_hooks import ProjectionHead

proj = ProjectionHead(st_dim, args.hidden_dim, args.proj_dim).to(device)
optimizer = torch.optim.Adam(proj.parameters(), lr=args.lr)

# Fixed target centroids: project once and freeze.
# These are the targets in the projected space that positives should land near.
with torch.no_grad():
    target_centroids = proj(centroids_t).clone().detach()

log_print(f"\nProjection: {st_dim} → {args.hidden_dim} → {args.proj_dim} (L2-normalized)")
log_print(f"Margin: {args.margin}, LR: {args.lr}, Epochs: {args.epochs}")

# -------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------


def compute_loss(pos_emb, pos_cid, neg_emb):
    """Compute contrastive loss over a batch."""
    proj_pos = proj(pos_emb)
    proj_neg = proj(neg_emb)

    # Positive loss: 1 - cosine similarity to matching centroid
    matched = target_centroids[pos_cid]
    l_pos = (1 - (proj_pos * matched).sum(dim=1)).mean()

    # Negative loss: push away from nearest centroid (cosine)
    cos_neg = proj_neg @ target_centroids.T  # (n_neg, n_clusters)
    max_cos_neg = cos_neg.max(dim=1).values
    l_neg = torch.clamp(max_cos_neg - (-args.margin), min=0).mean()

    return l_pos + l_neg, l_pos.item(), l_neg.item()


def evaluate(pos_emb, pos_cid, neg_emb, thresholds):
    """Compute detection metrics at various cosine similarity thresholds."""
    with torch.no_grad():
        proj_pos = proj(pos_emb)
        proj_neg = proj(neg_emb)

        # Max cosine similarity to any centroid
        cos_pos = (proj_pos @ target_centroids.T).max(dim=1).values
        cos_neg = (proj_neg @ target_centroids.T).max(dim=1).values

    results = {}
    for t in thresholds:
        tp = (cos_pos >= t).sum().item()
        fp = (cos_neg >= t).sum().item()
        recall = tp / len(cos_pos) if len(cos_pos) > 0 else 0
        fpr = fp / len(cos_neg) if len(cos_neg) > 0 else 0
        results[t] = (recall, fpr)
    return results, cos_pos, cos_neg


log_print("\n--- Training ---")
thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

for epoch in range(args.epochs):
    proj.train()

    # Shuffle and batch
    perm_p = torch.randperm(len(train_pos_emb))
    if args.neg_mode == "balanced":
        # Subsample negatives to match positive count
        perm_n = torch.randperm(len(train_neg_emb))[:len(train_pos_emb)]
    else:
        perm_n = torch.randperm(len(train_neg_emb))
    n_batches = max(len(perm_p), len(perm_n)) // args.batch_size + 1

    epoch_loss = 0.0
    epoch_l_pos = 0.0
    epoch_l_neg = 0.0
    n_steps = 0

    for b in range(n_batches):
        # Sample batch from both sets (with wrapping for smaller set)
        start_p = (b * args.batch_size) % len(train_pos_emb)
        end_p = start_p + args.batch_size
        idx_p = perm_p[start_p:end_p]
        if len(idx_p) == 0:
            continue

        start_n = (b * args.batch_size) % len(perm_n)
        end_n = start_n + args.batch_size
        idx_n = perm_n[start_n:end_n]
        if len(idx_n) == 0:
            continue

        loss, lp, ln = compute_loss(
            train_pos_emb[idx_p], train_pos_cid[idx_p],
            train_neg_emb[idx_n],
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_l_pos += lp
        epoch_l_neg += ln
        n_steps += 1

    avg_loss = epoch_loss / max(n_steps, 1)
    avg_lp = epoch_l_pos / max(n_steps, 1)
    avg_ln = epoch_l_neg / max(n_steps, 1)

    if (epoch + 1) % 10 == 0 or epoch == 0:
        proj.eval()
        results, cos_pos, cos_neg = evaluate(
            val_pos_emb, val_pos_cid, val_neg_emb, thresholds,
        )
        log_print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={avg_loss:.4f} (pos={avg_lp:.4f} neg={avg_ln:.4f})"
        )
        for t in thresholds:
            recall, fpr = results[t]
            log_print(f"  threshold={t:.1f}: recall={recall:.3f} fpr={fpr:.3f}")

# -------------------------------------------------------------------------
# Final evaluation
# -------------------------------------------------------------------------

log_print("\n--- Final evaluation on validation set ---")
proj.eval()
fine_thresholds = np.arange(-0.5, 1.01, 0.05).tolist()
results, cos_pos, cos_neg = evaluate(
    val_pos_emb, val_pos_cid, val_neg_emb, fine_thresholds,
)

log_print(f"\n{'Threshold':<12} {'Recall':>10} {'FPR':>10}")
log_print("-" * 35)
for t in fine_thresholds:
    recall, fpr = results[t]
    log_print(f"{t:<12.2f} {recall:>10.3f} {fpr:>10.3f}")

log_print(f"\nPositive cosine sim: min={cos_pos.min():.3f} "
          f"median={cos_pos.median():.3f} max={cos_pos.max():.3f}")
log_print(f"Negative cosine sim: min={cos_neg.min():.3f} "
          f"median={cos_neg.median():.3f} max={cos_neg.max():.3f}")

# -------------------------------------------------------------------------
# Per-CoT evaluation: Stage 1 only vs Stage 1 + projection
# -------------------------------------------------------------------------

log_print("\n--- Per-CoT evaluation (Stage 1 + projection vs Stage 1 only) ---")

raw_cots_path = f"{args.data_dir}/{args.split}/raw_cots.json"
with open(raw_cots_path) as f:
    raw_data = json.load(f)

cot_thresholds = [0.0, 0.3, 0.5, 0.7, 0.9]


def eval_cots(cots: list[str], thresholds: list[float]):
    """Per-CoT detection across multiple modes."""
    n = len(cots)
    s1_detections = 0
    s1_proj_detections = {t: 0 for t in thresholds}
    proj_only_detections = {t: 0 for t in thresholds}

    for cot in cots:
        sentences = split_sentences(cot)
        if not sentences:
            continue

        # Stage 1
        flags = classifier.predict(sentences, args.classifier_threshold)
        flagged = [s for s, f in zip(sentences, flags) if f]
        if flagged:
            s1_detections += 1

        # Projection on ALL sentences (standalone mode)
        with torch.no_grad():
            all_embs = torch.tensor(
                st_model.encode(sentences, show_progress_bar=False),
                dtype=torch.float32, device=device,
            )
            all_proj = proj(all_embs)
            all_max_cos = (all_proj @ target_centroids.T).max().item()

        for t in thresholds:
            if all_max_cos >= t:
                proj_only_detections[t] += 1

        # Projection on Stage 1-flagged sentences only
        if flagged:
            with torch.no_grad():
                flagged_embs = torch.tensor(
                    st_model.encode(flagged, show_progress_bar=False),
                    dtype=torch.float32, device=device,
                )
                flagged_proj = proj(flagged_embs)
                flagged_max_cos = (flagged_proj @ target_centroids.T).max().item()

            for t in thresholds:
                if flagged_max_cos >= t:
                    s1_proj_detections[t] += 1

    return s1_detections, s1_proj_detections, proj_only_detections


forget_cots = raw_data["forget_cots"]
retain_cots_eval = raw_data["retain_cots"]

f_s1, f_s1_proj, f_proj_only = eval_cots(forget_cots, cot_thresholds)
r_s1, r_s1_proj, r_proj_only = eval_cots(retain_cots_eval, cot_thresholds)

n_f, n_r = len(forget_cots), len(retain_cots_eval)

log_print(f"\n{'Method':<40} {'Forget Recall':>15} {'Retain FPR':>15}")
log_print("=" * 72)
log_print(f"{'Stage 1 only':<40} {f_s1}/{n_f} ({f_s1/n_f*100:.1f}%){'':<3} {r_s1}/{n_r} ({r_s1/n_r*100:.1f}%)")
for t in cot_thresholds:
    label = f"Stage 1 + projection (cos >= {t})"
    fc, rc = f_s1_proj[t], r_s1_proj[t]
    log_print(f"{label:<40} {fc}/{n_f} ({fc/n_f*100:.1f}%){'':<3} {rc}/{n_r} ({rc/n_r*100:.1f}%)")
for t in cot_thresholds:
    label = f"Projection only (cos >= {t})"
    fc, rc = f_proj_only[t], r_proj_only[t]
    log_print(f"{label:<40} {fc}/{n_f} ({fc/n_f*100:.1f}%){'':<3} {rc}/{n_r} ({rc/n_r*100:.1f}%)")

# Save projection if output_dir specified
if args.output_dir:
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, "projection_head.pt")
    proj.save(save_path, target_centroids)
    log_print(f"\nSaved projection head to {save_path}")
