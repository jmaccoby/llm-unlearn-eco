"""
Examine model responses to R-TOFU dataset queries with ECO prompt corruption applied.

Usage:
    python -m scripts.examine_rtofu_corrupted --split forget10 --n 10 \
        --corrupt_method rand_noise_first_n --dims 500 --strength 50
    python -m scripts.examine_rtofu_corrupted --split forget10 --n 20 \
        --corrupt_method zero_out_top_k --dims 500 --show_cot --output results/rtofu_corrupted.csv
"""
import argparse
import csv

import torch
from transformers import GenerationConfig

from eco.attack import AttackedReasoningModel, PromptClassifier
from eco.attack.utils import remove_hooks
from eco.dataset.rtofu import RTOFU
from eco.model import HFModel


def _build_byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode(): maps BPE unicode chars back to bytes."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _build_byte_decoder()


def fix_bpe(text):
    """Convert byte-level BPE characters (e.g. Ġ→space, Ċ→newline) to real bytes."""
    result = []
    for c in text:
        if c in _BYTE_DECODER:
            result.append(_BYTE_DECODER[c])
        else:
            result.extend(c.encode("utf-8"))
    return bytes(result).decode("utf-8", errors="replace")


MODEL_NAME = "LRM-target"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["full", "retain90", "retain50", "retain10", "forget01", "forget05", "forget10"],
    help="R-TOFU split to query",
)
parser.add_argument("--n", type=int, default=10, help="Number of examples to examine (0 = all)")
parser.add_argument("--offset", type=int, default=0, help="Start from this index")
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--show_cot", action="store_true", help="Also print the gold chain-of-thought")
parser.add_argument("--output", type=str, default=None, help="Save results to CSV file")
parser.add_argument("--corrupt_method", type=str, required=True, help="Corruption method name (e.g. rand_noise_first_n)")
parser.add_argument("--dims", type=int, required=True, help="Number of embedding dimensions to corrupt")
parser.add_argument("--strength", type=float, default=None, help="Corruption strength (required for noise/value methods)")
parser.add_argument("--classifier_threshold", type=float, default=0.99, help="Prompt classifier confidence threshold")
parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty for generation (try 1.2-1.5 to reduce looping)")
args = parser.parse_args()

# Load model via HFModel
print(f"Loading model: {MODEL_NAME}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
if args.repetition_penalty is not None:
    generation_config.repetition_penalty = args.repetition_penalty
model = HFModel(
    model_name=MODEL_NAME,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)
tokenizer = model.tokenizer
device = model.device

# Load prompt classifier and wrap model with AttackedModel
print(f"Loading prompt classifier: rtofu_classifiers/{args.split}")
prompt_classifier = PromptClassifier(
    model_name="roberta-base",
    model_path=f"rtofu_classifiers/{args.split}",
    batch_size=64,
)

corrupt_args = {"dims": args.dims}
if args.strength is not None:
    corrupt_args["strength"] = args.strength

attacked_model = AttackedReasoningModel(
    model=model,
    prompt_classifier=prompt_classifier,
    token_classifier=None,
    corrupt_method=args.corrupt_method,
    corrupt_args=corrupt_args,
    classifier_threshold=args.classifier_threshold,
)

# Load dataset
print(f"Loading R-TOFU split: {args.split}")
data_module = RTOFU()
data_module.download()
dataset = data_module.dataset[args.split]

end = len(dataset) if args.n == 0 else min(args.offset + args.n, len(dataset))
examples = dataset.select(range(args.offset, end))

print(f"\nExamining {len(examples)} examples from split '{args.split}'")
print(f"Corruption: {args.corrupt_method} (dims={args.dims}, strength={args.strength})")
print(f"Classifier threshold: {args.classifier_threshold}\n")
print("=" * 80)

results = []

for i, example in enumerate(examples):
    question = example["question"]
    gold_answer = example["answer"]
    cot = example.get("cot", "")

    # Generate response with corruption (think prefix appended automatically)
    inputs = tokenizer(question, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = attacked_model.generate(
            [question],
            **inputs,
            generation_config=attacked_model.generation_config,
        )
    remove_hooks(attacked_model.model)

    # Slice after the original question (keeping the <think> prefix in output)
    raw = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    raw = fix_bpe(raw)
    think_block, response = raw.split("</think>\n\n", 1) if "</think>\n\n" in raw else ("", raw)

    # Print
    idx = args.offset + i
    print(f"[{idx}] QUESTION:\n{question}\n")
    print(f"GOLD ANSWER:\n{gold_answer}\n")
    if args.show_cot:
        if think_block:
            print(f"MODEL THINKING:\n{think_block.removeprefix('<think>').strip()}\n")
        if cot:
            print(f"GOLD COT:\n{cot}\n")
    print(f"MODEL RESPONSE:\n{response}\n")
    print("-" * 80)

    results.append({"index": idx, "question": question, "gold_answer": gold_answer, "response": response})

# Save to CSV if requested
if args.output:
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "question", "gold_answer", "response"],
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} results to {args.output}")
