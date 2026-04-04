"""
Measure CoT (chain-of-thought) token lengths for LRM-target on R-TOFU
without corruption, to determine an appropriate max_new_tokens value.

Uses a high max_new_tokens to avoid truncation, then measures how many
tokens the model actually generates for the CoT portion (before </think>).

Usage:
    python -m scripts.measure_cot_length --split forget05
    python -m scripts.measure_cot_length --split forget05 --max_new_tokens 8192
"""
import argparse
import json
import os
import time

import numpy as np
from transformers import GenerationConfig

from eco.dataset.rtofu import RTOFU
from eco.inference import ReasoningGenerationEngine
from eco.model import HFModel, ReasoningModel
from eco.evaluator import ROUGERecall
from eco.utils import log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="full",
                    help="R-TOFU split to measure (default: full)")
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=8192,
                    help="Set high to avoid truncation")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--shard", type=int, default=None,
                    help="Shard index (0-based) for parallel runs")
parser.add_argument("--num_shards", type=int, default=1,
                    help="Total number of shards")
parser.add_argument("--output_dir", type=str, default="results/cot_lengths")
args = parser.parse_args()

seed_everything(args.seed)

log_print(f"Loading model: {args.model_name}")
gen_cfg = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
    generation_config=gen_cfg,
)
tok = model.tokenizer
rm = ReasoningModel(model)

rtofu = RTOFU(
    formatting_tokens=model.model_config.get("formatting_tokens"),
    eos_token=tok.eos_token,
)
rtofu.download()

# Optionally shard the dataset for parallel runs
subset_names = [args.split]
total = len(rtofu.dataset[args.split])
if args.shard is not None:
    shard_size = (total + args.num_shards - 1) // args.num_shards
    start = args.shard * shard_size
    end = min(start + shard_size, total)
    rtofu.dataset[args.split] = rtofu.dataset[args.split].select(range(start, end))
    log_print(f"Split: {args.split}, shard {args.shard}/{args.num_shards}, examples {start}-{end} ({end-start})")
else:
    log_print(f"Split: {args.split}, examples: {total}")

engine = ReasoningGenerationEngine(
    model=rm,
    tokenizer=tok,
    data_module=rtofu,
    subset_names=subset_names,
    answer_evaluator=[ROUGERecall(mode="rougeL")],
    batch_size=args.batch_size,
)

log_print(f"Generating with max_new_tokens={args.max_new_tokens}...")
start = time.perf_counter()
answers = engine._generate()
elapsed = time.perf_counter() - start
log_print(f"Generation completed in {elapsed:.1f}s")

THINK_SUFFIX = "</think>\n\n"
results = {}

for subset_name, data in answers.items():
    cot_token_lengths = []
    answer_token_lengths = []
    total_token_lengths = []
    missing_think = 0

    for batch_responses in data["generated"]:
        for resp in batch_responses:
            # Count total tokens
            total_tokens = len(tok.encode(resp, add_special_tokens=False))
            total_token_lengths.append(total_tokens)

            if THINK_SUFFIX in resp:
                cot, answer = resp.split(THINK_SUFFIX, 1)
                cot_tokens = len(tok.encode(cot, add_special_tokens=False))
                answer_tokens = len(tok.encode(answer, add_special_tokens=False))
                cot_token_lengths.append(cot_tokens)
                answer_token_lengths.append(answer_tokens)
            else:
                missing_think += 1

    n_total = len(total_token_lengths)
    n_complete = len(cot_token_lengths)

    log_print(f"\n{'='*60}")
    log_print(f"Subset: {subset_name}")
    log_print(f"{'='*60}")
    log_print(f"Total examples: {n_total}")
    log_print(f"Complete (has </think>): {n_complete} ({100*n_complete/n_total:.1f}%)")
    log_print(f"Truncated (no </think>): {missing_think} ({100*missing_think/n_total:.1f}%)")

    if cot_token_lengths:
        cot_arr = np.array(cot_token_lengths)
        ans_arr = np.array(answer_token_lengths)
        total_complete = cot_arr + ans_arr
        log_print(f"\nCoT token lengths (complete responses only):")
        log_print(f"  min:    {cot_arr.min()}")
        log_print(f"  25th:   {int(np.percentile(cot_arr, 25))}")
        log_print(f"  median: {int(np.median(cot_arr))}")
        log_print(f"  75th:   {int(np.percentile(cot_arr, 75))}")
        log_print(f"  90th:   {int(np.percentile(cot_arr, 90))}")
        log_print(f"  95th:   {int(np.percentile(cot_arr, 95))}")
        log_print(f"  99th:   {int(np.percentile(cot_arr, 99))}")
        log_print(f"  max:    {cot_arr.max()}")
        log_print(f"  mean:   {cot_arr.mean():.1f}")

        log_print(f"\nAnswer token lengths:")
        log_print(f"  min:    {ans_arr.min()}")
        log_print(f"  median: {int(np.median(ans_arr))}")
        log_print(f"  max:    {ans_arr.max()}")
        log_print(f"  mean:   {ans_arr.mean():.1f}")

        log_print(f"\nTotal token lengths (CoT + answer, complete only):")
        log_print(f"  min:    {total_complete.min()}")
        log_print(f"  median: {int(np.median(total_complete))}")
        log_print(f"  90th:   {int(np.percentile(total_complete, 90))}")
        log_print(f"  95th:   {int(np.percentile(total_complete, 95))}")
        log_print(f"  max:    {total_complete.max()}")

        # Suggest max_new_tokens values
        log_print(f"\nSuggested max_new_tokens values:")
        for pct in [90, 95, 99, 100]:
            if pct == 100:
                val = int(total_complete.max())
            else:
                val = int(np.percentile(total_complete, pct))
            # Round up to next 256
            rounded = ((val + 255) // 256) * 256
            coverage = 100 * np.mean(total_complete <= rounded)
            log_print(f"  p{pct}: {val} -> {rounded} (covers {coverage:.1f}% of complete responses)")

    results[subset_name] = {
        "n_total": n_total,
        "n_complete": n_complete,
        "n_truncated": missing_think,
        "cot_token_lengths": cot_token_lengths,
        "answer_token_lengths": answer_token_lengths,
        "total_token_lengths": total_token_lengths,
    }

os.makedirs(args.output_dir, exist_ok=True)
shard_suffix = f"_shard{args.shard}" if args.shard is not None else ""
output_path = os.path.join(args.output_dir, f"{args.split}_max{args.max_new_tokens}{shard_suffix}.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
log_print(f"\nRaw data saved to {output_path}")
