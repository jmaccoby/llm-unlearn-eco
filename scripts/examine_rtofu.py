"""
Examine model responses to R-TOFU dataset queries.

Usage:
    python -m scripts.examine_rtofu
    python -m scripts.examine_rtofu --split forget10 --num_examples 20
    python -m scripts.examine_rtofu --split full --show_cot --output results/rtofu_responses.csv
"""
import argparse
import csv

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from eco.dataset.rtofu import RTOFU
from eco.utils import fix_bpe

MODEL_PATH = "sangyon/LRM-target"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["full", "retain90", "retain50", "retain10", "forget01", "forget05", "forget10"],
    help="R-TOFU split to query",
)
parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to examine (0 = all)")
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
    dtype=torch.bfloat16,
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

end = len(dataset) if args.num_examples == 0 else min(args.offset + args.num_examples, len(dataset))
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
    think_block, response = raw.split("</think>\n\n", 1) if "</think>\n\n" in raw else (raw, "")

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
