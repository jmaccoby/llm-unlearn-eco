"""
Examine soft token effects on both forget and retain prompts.

Unconditionally inserts the soft token into every generation (no leak
detector / prompt classifier), so we can see whether the token is
disruptive on forget-set prompts and transparent on retain-set prompts.

Usage:
    python -m scripts.examine_soft_token_unconditional --split forget10 \
        --soft_token_bank_path soft_tokens_ctx/forget10/soft_token_bank.pt \
        --num_examples 5 --max_new_tokens 512
"""
import argparse

import torch
from transformers import GenerationConfig

from eco.attack.learned_hooks import SoftToken, SoftTokenBank
from eco.attack.utils import get_nested_attr
from eco.dataset.rtofu import RTOFU
from eco.model import HFModel
from eco.utils import log_print

MODEL_NAME = "LRM-target"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--num_examples", type=int, default=5,
                    help="Number of examples per split (forget + retain)")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--show_cot", action="store_true", help="Print gold CoT")
parser.add_argument("--soft_token_path", type=str, default=None)
parser.add_argument("--soft_token_bank_path", type=str, default=None)
parser.add_argument("--n_clusters", type=int, default=0)
parser.add_argument("--cluster_id", type=int, default=0,
                    help="Which cluster's soft token to use (bank mode)")
args = parser.parse_args()

if args.soft_token_path is None and args.soft_token_bank_path is None:
    parser.error("One of --soft_token_path or --soft_token_bank_path is required")

# -------------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------------

log_print(f"Loading model: {MODEL_NAME}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
hf_model = HFModel(
    model_name=MODEL_NAME,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)
model = hf_model.model
tokenizer = hf_model.tokenizer
device = model.device

attack_module_path = hf_model.model_config["attack_module"]
embed_module = get_nested_attr(model, attack_module_path)

# -------------------------------------------------------------------------
# Load soft token
# -------------------------------------------------------------------------

embed_dim = hf_model.model_config["embedding_dim"]
if args.soft_token_bank_path is not None:
    if args.n_clusters > 0:
        n_clusters = args.n_clusters
    else:
        sd = torch.load(args.soft_token_bank_path, weights_only=True)
        n_clusters = sum(1 for k in sd if k.endswith(".embedding"))
    soft_token_obj = SoftTokenBank.load(
        args.soft_token_bank_path, n_clusters=n_clusters, embed_dim=embed_dim
    )
    active_token = soft_token_obj.select(args.cluster_id)
    log_print(f"Loaded soft token bank ({n_clusters} clusters) from {args.soft_token_bank_path}")
    log_print(f"Using cluster {args.cluster_id} (norm={active_token.embedding.norm().item():.3f})")
else:
    active_token = SoftToken.load(args.soft_token_path, embed_dim=embed_dim)
    log_print(f"Loaded soft token from {args.soft_token_path}")
    log_print(f"Norm: {active_token.embedding.norm().item():.3f}")

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------

log_print(f"Loading R-TOFU split: {args.split}")
rtofu = RTOFU(
    formatting_tokens=hf_model.model_config.get("formatting_tokens"),
    eos_token=tokenizer.eos_token,
)
rtofu.download()

# Map forget split to matching retain split
retain_split = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}[args.split]

THINK_PREFIX = "<think>\n"
think_ids = tokenizer(THINK_PREFIX, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
pad_ids = tokenizer(tokenizer.pad_token or tokenizer.eos_token,
                     add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

THINK_DELIMITER = "</think>\n\n"


# -------------------------------------------------------------------------
# Generation helper
# -------------------------------------------------------------------------


def generate_with_soft_token(prompt_text):
    """Generate with the soft token inserted right after <think>\\n.

    Returns (cot, answer) tuple.
    """
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)

    # Build: [prompt] + [<think>\n] + [pad]
    input_ids = torch.cat([prompt_ids, think_ids, pad_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    # Soft token replaces the pad token (last position before generation)
    st_position = input_ids.shape[1] - 1

    hook_handle = active_token.apply_hook(embed_module, st_position)
    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
    finally:
        hook_handle.remove()

    # Decode only the new tokens
    new_tokens = outputs[0, input_ids.shape[1]:]
    full_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    # Split at </think>\n\n
    if THINK_DELIMITER in full_text:
        cot, answer = full_text.split(THINK_DELIMITER, 1)
    else:
        cot = full_text
        answer = ""

    return cot.strip(), answer.strip()


def generate_baseline(prompt_text):
    """Generate without soft token (baseline)."""
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    input_ids = torch.cat([prompt_ids, think_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)

    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
    )

    new_tokens = outputs[0, input_ids.shape[1]:]
    full_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

    if THINK_DELIMITER in full_text:
        cot, answer = full_text.split(THINK_DELIMITER, 1)
    else:
        cot = full_text
        answer = ""

    return cot.strip(), answer.strip()


# -------------------------------------------------------------------------
# Run on forget split
# -------------------------------------------------------------------------

forget_data = rtofu.dataset[args.split]
n_forget = min(args.num_examples, len(forget_data))

log_print(f"\n{'=' * 80}")
log_print(f"FORGET SET ({args.split}): {n_forget} examples with soft token")
log_print(f"{'=' * 80}")

for i in range(n_forget):
    example = forget_data[i]
    prompt = example["question"]
    gold_answer = example["answer"]

    cot, answer = generate_with_soft_token(prompt)

    log_print(f"\n[F{i}] QUESTION:\n{prompt}\n")
    log_print(f"GOLD ANSWER:\n{gold_answer}\n")
    if args.show_cot and example.get("cot"):
        log_print(f"GOLD COT:\n{example['cot']}\n")
    if cot:
        log_print(f"MODEL THINKING:\n{cot}\n")
    log_print(f"MODEL RESPONSE:\n{answer}\n")
    log_print("-" * 80)

# -------------------------------------------------------------------------
# Run on retain split
# -------------------------------------------------------------------------

retain_data = rtofu.dataset[retain_split]
n_retain = min(args.num_examples, len(retain_data))

log_print(f"\n{'=' * 80}")
log_print(f"RETAIN SET ({retain_split}): {n_retain} examples with soft token")
log_print(f"{'=' * 80}")

for i in range(n_retain):
    example = retain_data[i]
    prompt = example["question"]
    gold_answer = example["answer"]

    cot, answer = generate_with_soft_token(prompt)

    log_print(f"\n[R{i}] QUESTION:\n{prompt}\n")
    log_print(f"GOLD ANSWER:\n{gold_answer}\n")
    if cot:
        log_print(f"MODEL THINKING:\n{cot}\n")
    log_print(f"MODEL RESPONSE:\n{answer}\n")
    log_print("-" * 80)
