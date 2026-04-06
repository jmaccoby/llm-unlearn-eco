"""
Generate training labels for the CoT leak classifier.

Runs the uncorrupted model on forget-set and retain-set prompts, collects
generated CoTs, splits them into sentences, and labels each sentence by
checking entailment against the knowledge bank.

Requires a knowledge bank (from build_knowledge_bank.py) to exist.

Usage:
    python -m scripts.generate_leak_labels --split forget10 --max_new_tokens 1024
    python -m scripts.generate_leak_labels --split forget05 --batch_size 4
"""
import argparse
import json as _json
import os

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import GenerationConfig, pipeline

from eco.attack.leak_detector import entails_any_claim, load_knowledge_bank
from eco.dataset.rtofu import RTOFU
from eco.evaluator.utils import split_sentences
from eco.model import HFModel, ReasoningModel
from eco.inference import ReasoningGenerationEngine
from eco.utils import fix_bpe, log_print, seed_everything

THINK_SUFFIX = ReasoningGenerationEngine.THINK_SUFFIX

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--knowledge_bank_dir",
    type=str,
    default=None,
    help="Default: knowledge_banks/{split}",
)
parser.add_argument("--cosine_prefilter", type=float, default=0.3)
parser.add_argument(
    "--num_retain_examples",
    type=int,
    default=200,
    help="Number of retain-set examples to include as negatives",
)
parser.add_argument("--output_dir", type=str, default="leak_detector_data")
args = parser.parse_args()

seed_everything(args.seed)
kb_dir = args.knowledge_bank_dir or f"knowledge_banks/{args.split}"

# -------------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------------

log_print(f"Loading model: {args.model_name}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)
model = ReasoningModel(model)
tokenizer = model.tokenizer
n_think = model.n_think_tokens

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------

log_print(f"Loading R-TOFU split: {args.split}")
rtofu = RTOFU(
    formatting_tokens=model.model_config.get("formatting_tokens"),
    eos_token=tokenizer.eos_token,
)
rtofu.download()

retain_name = RTOFU.match_retain[args.split]
forget_ds = rtofu.load_dataset_for_eval(args.split)
retain_ds = rtofu.load_dataset_for_eval(retain_name)

# Subsample retain set
n_retain = min(args.num_retain_examples, len(retain_ds))
retain_ds = retain_ds.select(range(n_retain))

log_print(
    f"Forget examples: {len(forget_ds)}, Retain examples: {len(retain_ds)}"
)

# -------------------------------------------------------------------------
# Load knowledge bank + NLI resources
# -------------------------------------------------------------------------

log_print(f"Loading knowledge bank from {kb_dir}")
claims, bank_embeddings = load_knowledge_bank(kb_dir)
log_print(f"Knowledge bank: {len(claims)} claims")

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


def generate_cots(dataset, desc="Generating"):
    """Run the model on all examples and return a list of CoT strings."""
    all_cots = []
    prompts = dataset["prompt_formatted"]

    # Left-pad so decode_start is correct for all samples in the batch
    # (same convention as ReasoningGenerationEngine._generate)
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(model.device)

        decode_start = tokenized["input_ids"].shape[1] + n_think

        with torch.no_grad():
            generated = model.generate(
                **tokenized,
                prompts=batch_prompts,
                generation_config=model.generation_config,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j in range(generated.shape[0]):
            raw = tokenizer.decode(
                generated[j][decode_start:], skip_special_tokens=True
            )
            raw = fix_bpe(raw)
            if THINK_SUFFIX in raw:
                cot, _ = raw.split(THINK_SUFFIX, 1)
            else:
                cot = raw
            all_cots.append(cot)

        done = min(i + args.batch_size, len(prompts))
        log_print(f"  {desc}: {done}/{len(prompts)}")

    tokenizer.padding_side = orig_padding_side
    return all_cots


def label_sentence(sentence: str) -> int:
    """Check if a sentence entails any knowledge bank claim. Returns 0 or 1."""
    entailed, _ = entails_any_claim(
        sentence, claims, bank_embeddings, st_model, nli,
        cosine_prefilter=args.cosine_prefilter,
    )
    return int(entailed)


# -------------------------------------------------------------------------
# Generate CoTs and label sentences
# -------------------------------------------------------------------------

log_print("\n--- Generating forget-set CoTs ---")
forget_cots = generate_cots(forget_ds, desc="Forget")

log_print("\n--- Generating retain-set CoTs ---")
retain_cots = generate_cots(retain_ds, desc="Retain")

log_print("\n--- Labeling forget-set sentences ---")
forget_texts, forget_labels = [], []
for i, cot in enumerate(forget_cots):
    sentences = split_sentences(cot)
    for s in sentences:
        label = label_sentence(s)
        forget_texts.append(s)
        forget_labels.append(label)
    if (i + 1) % 20 == 0 or i == len(forget_cots) - 1:
        n_pos = sum(forget_labels)
        log_print(
            f"  Labeled {i + 1}/{len(forget_cots)} CoTs "
            f"({len(forget_texts)} sentences, {n_pos} positive)"
        )

log_print("\n--- Labeling retain-set sentences ---")
retain_texts, retain_labels = [], []
for i, cot in enumerate(retain_cots):
    sentences = split_sentences(cot)
    for s in sentences:
        retain_texts.append(s)
        retain_labels.append(0)  # retain sentences are always clean
    if (i + 1) % 50 == 0 or i == len(retain_cots) - 1:
        log_print(
            f"  Processed {i + 1}/{len(retain_cots)} CoTs "
            f"({len(retain_texts)} sentences)"
        )

# -------------------------------------------------------------------------
# Build HuggingFace DatasetDict
# -------------------------------------------------------------------------

all_texts = forget_texts + retain_texts
all_labels = forget_labels + retain_labels

n_pos = sum(all_labels)
n_neg = len(all_labels) - n_pos
log_print(f"\nTotal sentences: {len(all_labels)} (positive: {n_pos}, negative: {n_neg})")

# Shuffle and split into train/test (80/20)
rng = np.random.default_rng(args.seed)
indices = rng.permutation(len(all_texts))
split_point = int(len(indices) * 0.8)
train_idx = indices[:split_point]
test_idx = indices[split_point:]

train_ds = Dataset.from_dict({
    "text": [all_texts[i] for i in train_idx],
    "label": [all_labels[i] for i in train_idx],
})
test_ds = Dataset.from_dict({
    "text": [all_texts[i] for i in test_idx],
    "label": [all_labels[i] for i in test_idx],
})

# Separate eval splits for classifier evaluation
forget_eval_ds = Dataset.from_dict({
    "text": forget_texts,
    "label": forget_labels,
})
retain_eval_ds = Dataset.from_dict({
    "text": retain_texts,
    "label": retain_labels,
})

dataset_dict = DatasetDict({
    "train": train_ds,
    "test": test_ds,
    "forget": forget_eval_ds,
    "retain": retain_eval_ds,
})

output_path = f"{args.output_dir}/{args.split}"
os.makedirs(output_path, exist_ok=True)
dataset_dict.save_to_disk(output_path)
log_print(f"\nDataset saved to {output_path}")
log_print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
log_print(f"  Forget eval: {len(forget_eval_ds)}, Retain eval: {len(retain_eval_ds)}")

# Save raw CoTs for soft token training data construction
raw_cots_path = f"{args.output_dir}/{args.split}/raw_cots.json"
_json.dump({
    "forget_cots": forget_cots,
    "retain_cots": retain_cots,
    "forget_prompts": list(forget_ds["prompt_formatted"]),
    "retain_prompts": list(retain_ds["prompt_formatted"]),
}, open(raw_cots_path, "w"), indent=2)
log_print(f"Raw CoTs saved to {raw_cots_path}")
