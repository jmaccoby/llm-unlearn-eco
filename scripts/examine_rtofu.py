"""
Examine model responses to R-TOFU dataset queries.

Usage:
    python -m scripts.examine_rtofu
    python -m scripts.examine_rtofu --split forget10 --n 20
    python -m scripts.examine_rtofu --split full --show_cot --output results/rtofu_responses.csv
"""
import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from eco.dataset.rtofu import RTOFU


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

MODEL_PATH = "sangyon/LRM-target"

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
args = parser.parse_args()

# Load model
print(f"Loading model: {MODEL_PATH}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
model.eval()

# Load dataset
print(f"Loading R-TOFU split: {args.split}")
data_module = RTOFU()
data_module.download()
dataset = data_module.dataset[args.split]

end = len(dataset) if args.n == 0 else min(args.offset + args.n, len(dataset))
examples = dataset.select(range(args.offset, end))

print(f"\nExamining {len(examples)} examples from split '{args.split}'\n")
print("=" * 80)

results = []

for i, example in enumerate(examples):
    question = example["question"]
    gold_answer = example["answer"]
    cot = example.get("cot", "")

    # Generate response, forcing the model to begin with a reasoning block
    prompt_len = len(tokenizer.encode(question))
    inputs = tokenizer(question + "<think>\n", return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, generation_config=model.generation_config)
    # Slice to new tokens only (after the original question, keeping the <think> prefix),
    # then apply inverse BPE byte mapping to resolve Ġ/Ċ characters
    raw = tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True)
    raw = fix_bpe(raw)
    # Extract final answer after the reasoning block
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
