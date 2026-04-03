"""
Diagnose prompt classifier predictions on R-TOFU forget split.

Usage:
    python -m scripts.diagnose_classifier --split forget05
"""
import argparse

from eco.attack import PromptClassifier
from eco.dataset.rtofu import RTOFU

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="forget05")
parser.add_argument("--num_examples", type=int, default=0, help="0 = all")
args = parser.parse_args()

# Load classifier and dataset
classifier = PromptClassifier(
    model_name="roberta-base",
    model_path=f"rtofu_classifiers/{args.split}",
    batch_size=64,
)

rtofu = RTOFU()
rtofu.download()
dataset = rtofu.dataset[args.split]

if args.num_examples > 0:
    dataset = dataset.select(range(min(args.num_examples, len(dataset))))

questions = list(dataset["question"])

# Get raw predictions (label + score)
raw_preds = classifier.model(
    questions,
    truncation=True,
    max_length=512,
    padding="longest",
    batch_size=64,
)

# Analyze
label1_count = sum(1 for p in raw_preds if p["label"] == "LABEL_1")
label0_count = len(raw_preds) - label1_count
scores = [p["score"] for p in raw_preds if p["label"] == "LABEL_1"]

print(f"Split: {args.split} ({len(questions)} examples)")
print(f"LABEL_1 (forget): {label1_count}, LABEL_0 (retain): {label0_count}")
if scores:
    print(f"LABEL_1 score range: {min(scores):.4f} - {max(scores):.4f}")
    print(f"LABEL_1 score mean: {sum(scores)/len(scores):.4f}")

# Check at different thresholds
for threshold in [0.5, 0.7, 0.9, 0.95, 0.99]:
    labels = classifier.predict(questions, threshold=threshold)
    n_forget = sum(labels)
    print(f"  threshold={threshold}: {n_forget}/{len(questions)} classified as forget")

# Show a few examples
print("\nSample predictions:")
for i in range(min(10, len(questions))):
    q = questions[i][:80]
    print(f"  [{i}] {raw_preds[i]['label']} ({raw_preds[i]['score']:.4f}): {q}")
