"""
Build training data for the learned soft token.

Reads raw CoTs from generate_leak_labels.py output and constructs
(prompt, prefix, leaking_continuation, clean_continuation) triples.

Usage:
    python -m scripts.build_soft_token_data --split forget10
"""
import argparse
import json
import os
import random

import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from eco.attack.leak_detector import entails_any_claim, load_knowledge_bank
from eco.evaluator.utils import split_sentences
from eco.utils import log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="leak_detector_data")
parser.add_argument("--knowledge_bank_dir", type=str, default="knowledge_banks")
parser.add_argument("--cosine_prefilter", type=float, default=0.3)
parser.add_argument("--output_dir", type=str, default="soft_token_data")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

seed_everything(args.seed)

# -------------------------------------------------------------------------
# Load raw CoTs and knowledge bank
# -------------------------------------------------------------------------

raw_cots_path = f"{args.data_dir}/{args.split}/raw_cots.json"
log_print(f"Loading raw CoTs from {raw_cots_path}")
with open(raw_cots_path) as f:
    raw_data = json.load(f)

forget_cots = raw_data["forget_cots"]
retain_cots = raw_data["retain_cots"]
forget_prompts = raw_data["forget_prompts"]
retain_prompts = raw_data["retain_prompts"]
log_print(f"Forget CoTs: {len(forget_cots)}, Retain CoTs: {len(retain_cots)}")

kb_dir = f"{args.knowledge_bank_dir}/{args.split}"
log_print(f"Loading knowledge bank from {kb_dir}")
claims, bank_embeddings = load_knowledge_bank(kb_dir)
log_print(f"Knowledge bank: {len(claims)} claims")

# -------------------------------------------------------------------------
# Load NLI resources
# -------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st_model = SentenceTransformer(
    "paraphrase-MiniLM-L6-v2",
    device=device,  # type: ignore[arg-type]
)
nli = pipeline(
    "text-classification",
    model="sileod/deberta-v3-base-tasksource-nli",
    device=device,
)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def sentence_is_leaking(sentence: str) -> bool:
    """Check if a sentence entails any knowledge bank claim."""
    return entails_any_claim(
        sentence, claims, bank_embeddings, st_model, nli,
        cosine_prefilter=args.cosine_prefilter,
    )


# -------------------------------------------------------------------------
# Build forget triples: (prompt, prefix, leaking_continuation)
# -------------------------------------------------------------------------

log_print("\n--- Finding leak points in forget-set CoTs ---")
forget_triples = []

for i, (prompt, cot) in enumerate(zip(forget_prompts, forget_cots)):
    sentences = split_sentences(cot)
    if not sentences:
        continue

    # Find first leaking sentence
    first_leak_idx = None
    for k, sentence in enumerate(sentences):
        if sentence_is_leaking(sentence):
            first_leak_idx = k
            break

    if first_leak_idx is not None:
        prefix = " ".join(sentences[:first_leak_idx])
        leaking_continuation = " ".join(sentences[first_leak_idx:])
        forget_triples.append({
            "prompt": prompt,
            "prefix": prefix,
            "leaking_continuation": leaking_continuation,
        })

    if (i + 1) % 20 == 0 or i == len(forget_cots) - 1:
        log_print(
            f"  Processed {i + 1}/{len(forget_cots)} forget CoTs "
            f"({len(forget_triples)} triples found)"
        )

log_print(f"Forget triples: {len(forget_triples)}")

if len(forget_triples) == 0:
    log_print("ERROR: No leaking sentences found. Cannot build training data.")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# Build clean continuations from retain CoTs
# -------------------------------------------------------------------------

log_print("\n--- Splitting retain CoTs at random positions ---")
clean_continuations = []

for i, cot in enumerate(retain_cots):
    sentences = split_sentences(cot)
    if len(sentences) < 2:
        continue
    split_point = random.randint(1, len(sentences) - 1)
    clean_continuation = " ".join(sentences[split_point:])
    clean_continuations.append(clean_continuation)

log_print(f"Clean continuations: {len(clean_continuations)}")

if len(clean_continuations) == 0:
    log_print("ERROR: No clean continuations generated. Cannot build training data.")
    raise SystemExit(1)

# -------------------------------------------------------------------------
# Pair forget triples with random clean continuations
# -------------------------------------------------------------------------

log_print("\n--- Pairing forget triples with clean continuations ---")
training_examples = []

for triple in forget_triples:
    clean_cont = random.choice(clean_continuations)
    training_examples.append({
        "prompt": triple["prompt"],
        "prefix": triple["prefix"],
        "leaking_continuation": triple["leaking_continuation"],
        "clean_continuation": clean_cont,
    })

# -------------------------------------------------------------------------
# Save as JSONL
# -------------------------------------------------------------------------

output_path = f"{args.output_dir}/{args.split}"
os.makedirs(output_path, exist_ok=True)
output_file = f"{output_path}/train.jsonl"

with open(output_file, "w") as f:
    for example in training_examples:
        f.write(json.dumps(example) + "\n")

log_print(f"\nTraining data saved to {output_file}")
log_print(f"Total examples: {len(training_examples)}")
