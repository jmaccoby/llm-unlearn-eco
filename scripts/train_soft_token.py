"""
Train learned soft token(s) for CoT truncation-point insertion.

Optimizes embedding vector(s) to maximize cross-entropy on forget-set
continuations while minimizing it on clean continuations.

In ``single`` mode, trains one soft token for all examples (original
behaviour).  In ``cluster`` mode, trains a bank of per-cluster soft
tokens, each specialised to its claim cluster.

Usage:
    python -m scripts.train_soft_token --split forget10 --epochs 100 --lr 1e-3
    python -m scripts.train_soft_token --split forget10 --mode cluster --n_clusters 5
"""
import argparse
import json
import os
import random

import torch
import torch.nn as nn

from eco.attack.soft_token import SoftToken, SoftTokenBank
from eco.attack.utils import get_nested_attr, remove_hooks
from eco.model import HFModel
from eco.utils import log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="soft_token_data")
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lambda_retain", type=float, default=0.1)
parser.add_argument("--max_seq_len", type=int, default=512)
parser.add_argument("--output_dir", type=str, default="soft_tokens")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--mode",
    type=str,
    default="single",
    choices=["single", "cluster"],
    help="single: one soft token; cluster: one per claim cluster",
)
parser.add_argument(
    "--n_clusters",
    type=int,
    default=0,
    help="Number of clusters (cluster mode). 0 = infer from training data.",
)
args = parser.parse_args()

seed_everything(args.seed)

# -------------------------------------------------------------------------
# Load model (frozen)
# -------------------------------------------------------------------------

log_print(f"Loading model: {args.model_name}")
hf_model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
)
model = hf_model.model
tokenizer = hf_model.tokenizer
model_config = hf_model.model_config

# Freeze all model parameters — gradients flow through the computation
# graph but only update the soft token embedding.
model.eval()
model.requires_grad_(False)

device = model.device
embed_dim = model_config["embedding_dim"]
attack_module_path = model_config["attack_module"]
embed_module = get_nested_attr(model, attack_module_path)

log_print(f"Device: {device}, Embed dim: {embed_dim}")
log_print(f"Embedding module: {attack_module_path}")

# -------------------------------------------------------------------------
# Load training data
# -------------------------------------------------------------------------

data_path = f"{args.data_dir}/{args.split}/train.jsonl"
log_print(f"Loading training data from {data_path}")
training_examples = []
with open(data_path) as f:
    for line in f:
        training_examples.append(json.loads(line))
log_print(f"Training examples: {len(training_examples)}")

# -------------------------------------------------------------------------
# Initialize soft token(s) and optimizer
# -------------------------------------------------------------------------

cluster_mode = args.mode == "cluster"

if cluster_mode:
    # Infer n_clusters from training data if not specified
    n_clusters = args.n_clusters
    if n_clusters == 0:
        cluster_ids = {ex.get("cluster_id") for ex in training_examples}
        cluster_ids.discard(None)
        if not cluster_ids:
            raise ValueError(
                "cluster mode requires cluster_id in training data; "
                "run build_soft_token_data.py with a clustered knowledge bank"
            )
        n_clusters = max(cluster_ids) + 1
    soft_token_bank = SoftTokenBank(n_clusters=n_clusters, embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam(soft_token_bank.parameters(), lr=args.lr)
    log_print(f"SoftTokenBank initialized ({n_clusters} clusters, dim={embed_dim})")
else:
    soft_token_bank = None
    soft_token = SoftToken(embed_dim=embed_dim).to(device)
    optimizer = torch.optim.Adam([soft_token.embedding], lr=args.lr)
    log_print(f"Soft token initialized (dim={embed_dim})")

THINK_PREFIX = "<think>\n"
pad_token = tokenizer.pad_token or tokenizer.eos_token

log_print(f"Optimizer: Adam, lr={args.lr}")
log_print(f"Lambda retain: {args.lambda_retain}, Epochs: {args.epochs}")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def compute_masked_loss(input_ids, attention_mask, cont_start, cont_end):
    """Forward pass and compute CE loss on continuation tokens only.

    Args:
        input_ids: Token IDs, shape (1, seq_len).
        attention_mask: Attention mask, shape (1, seq_len).
        cont_start: Start position of continuation tokens in the sequence.
        cont_end: End position (exclusive) of continuation tokens.

    Returns:
        Masked mean cross-entropy loss over continuation tokens.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Shift logits and labels for next-token prediction
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Create a mask for the continuation tokens only
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.float)
    # -1 because of the shift: to predict token at position p, we use
    # logits at position p-1 in the shifted sequence
    mask_start = max(cont_start - 1, 0)
    mask_end = cont_end - 1
    loss_mask[0, mask_start:mask_end] = 1.0

    # Compute per-token CE loss
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    per_token_loss = per_token_loss.view(shift_labels.size())

    # Masked mean
    denom = loss_mask.sum()
    if denom == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    masked_loss = (per_token_loss * loss_mask).sum() / denom
    return masked_loss


def tokenize_with_soft_token(prompt, prefix, continuation):
    """Tokenize prompt + <think> + prefix + [pad] + continuation.

    The pad_token position is where the soft token hook will be applied.

    Returns:
        input_ids (1, seq_len), attention_mask (1, seq_len),
        soft_token_pos (int), cont_start (int), cont_end (int)
    """
    # Build the full text with a pad token as placeholder for the soft token
    before_soft = prompt + THINK_PREFIX + prefix
    after_soft = continuation

    # Tokenize each part separately to find positions
    before_ids = tokenizer.encode(before_soft, add_special_tokens=False)
    pad_ids = tokenizer.encode(pad_token, add_special_tokens=False)
    after_ids = tokenizer.encode(after_soft, add_special_tokens=False)

    # Truncate if needed
    total_len = len(before_ids) + len(pad_ids) + len(after_ids)
    if total_len > args.max_seq_len:
        # Truncate continuation from the end
        max_after = args.max_seq_len - len(before_ids) - len(pad_ids)
        if max_after <= 0:
            # Truncate before part too
            max_before = args.max_seq_len - len(pad_ids) - 1
            before_ids = before_ids[:max_before]
            after_ids = after_ids[:1]
        else:
            after_ids = after_ids[:max_after]

    all_ids = before_ids + pad_ids + after_ids
    soft_token_pos = len(before_ids)  # position of the pad token
    cont_start = len(before_ids) + len(pad_ids)  # start of continuation
    cont_end = len(all_ids)  # end of continuation

    input_ids = torch.tensor([all_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    return input_ids, attention_mask, soft_token_pos, cont_start, cont_end


# -------------------------------------------------------------------------
# Training loop
# -------------------------------------------------------------------------

log_print("\n--- Starting training ---")

for epoch in range(args.epochs):
    random.shuffle(training_examples)
    epoch_forget_loss = 0.0
    epoch_retain_loss = 0.0
    epoch_total_loss = 0.0
    n_examples = 0

    for ex_idx, example in enumerate(training_examples):
        prompt = example["prompt"]
        prefix = example["prefix"]
        leaking_cont = example["leaking_continuation"]
        clean_cont = example["clean_continuation"]

        # Select the appropriate soft token
        if cluster_mode:
            cluster_id = example.get("cluster_id")
            if cluster_id is None:
                continue  # skip examples without cluster annotation
            active_token = soft_token_bank.select(cluster_id)
        else:
            cluster_id = None
            active_token = soft_token

        # --- Forget loss: maximize CE on leaking continuation ---
        input_ids, attn_mask, st_pos, cont_start, cont_end = (
            tokenize_with_soft_token(prompt, prefix, leaking_cont)
        )

        if cont_start >= cont_end:
            continue

        hook_handle = active_token.apply_hook(embed_module, st_pos)
        l_forget = compute_masked_loss(input_ids, attn_mask, cont_start, cont_end)
        hook_handle.remove()

        # --- Retain loss: minimize CE on clean continuation ---
        input_ids, attn_mask, st_pos, cont_start, cont_end = (
            tokenize_with_soft_token(prompt, prefix, clean_cont)
        )

        if cont_start >= cont_end:
            # Still backprop forget loss
            loss = -l_forget
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_forget_loss += l_forget.item()
            epoch_total_loss += loss.item()
            n_examples += 1
            continue

        hook_handle = active_token.apply_hook(embed_module, st_pos)
        l_retain = compute_masked_loss(input_ids, attn_mask, cont_start, cont_end)
        hook_handle.remove()

        # --- Combined loss ---
        loss = -l_forget + args.lambda_retain * l_retain

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_forget_loss += l_forget.item()
        epoch_retain_loss += l_retain.item()
        epoch_total_loss += loss.item()
        n_examples += 1

    # Clean up any stale hooks
    remove_hooks(model)

    if n_examples > 0:
        avg_forget = epoch_forget_loss / n_examples
        avg_retain = epoch_retain_loss / n_examples
        avg_total = epoch_total_loss / n_examples
    else:
        avg_forget = avg_retain = avg_total = 0.0

    log_print(
        f"Epoch {epoch + 1}/{args.epochs} | "
        f"L_forget: {avg_forget:.4f} | "
        f"L_retain: {avg_retain:.4f} | "
        f"L_total: {avg_total:.4f} | "
        f"examples: {n_examples}"
    )

# -------------------------------------------------------------------------
# Save trained soft token(s)
# -------------------------------------------------------------------------

output_path = f"{args.output_dir}/{args.split}"
os.makedirs(output_path, exist_ok=True)

if cluster_mode:
    save_path = f"{output_path}/soft_token_bank.pt"
    soft_token_bank.save(save_path)
    log_print(f"\nSoft token bank saved to {save_path}")
else:
    save_path = f"{output_path}/embedding.pt"
    soft_token.save(save_path)
    log_print(f"\nSoft token saved to {save_path}")
