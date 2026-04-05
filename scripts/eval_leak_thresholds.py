"""Evaluate the leak classifier at multiple thresholds."""
import numpy as np
from datasets import load_from_disk
from transformers import pipeline

ds = load_from_disk("leak_detector_data/forget10")
clf = pipeline(
    "text-classification",
    model="leak_classifiers/forget10",
    tokenizer="roberta-base",
    device=0,
)

for split in ["forget", "retain", "test"]:
    texts = list(ds[split]["text"])
    labels = np.array(ds[split]["label"])

    scores = []
    for i in range(0, len(texts), 64):
        batch = texts[i : i + 64]
        preds = clf(batch, truncation=True, max_length=512, batch_size=64)
        for p in preds:
            s = p["score"] if p["label"] == "LABEL_1" else 1.0 - p["score"]
            scores.append(s)
    scores = np.array(scores)

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    print(f"\n=== {split} (n={len(labels)}, pos={n_pos}, neg={n_neg}) ===")
    print(f"  Thr    Prec  Recall     FPR  Flagged")
    for t in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (scores >= t).astype(int)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / n_neg if n_neg > 0 else 0
        flagged = int(pred.sum())
        print(f"  {t:.2f}  {prec:.3f}   {rec:.3f}  {fpr:.4f}  {flagged:>7d}")
