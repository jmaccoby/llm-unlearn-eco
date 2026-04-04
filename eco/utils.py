import gc
import os
import random
from copy import deepcopy
from itertools import product

import numpy as np
import torch
import yaml
from scipy.stats import hmean, ks_2samp
from tabulate import tabulate


def log_print(*args, **kwargs):
    """Print that flushes immediately, bypassing conda run buffering.

    Drop-in replacement for ``print()`` — accepts the same arguments.
    Also writes to a log file if the ``LOG_FILE`` environment variable is set.
    """
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)
    log_path = os.environ.get("LOG_FILE")
    if log_path:
        with open(log_path, "a") as f:
            file_kwargs = {k: v for k, v in kwargs.items() if k != "flush"}
            file_kwargs["file"] = f
            print(*args, **file_kwargs)


def ks_test(unlearn_tr, retain_tr):
    return ks_2samp(unlearn_tr, retain_tr).pvalue


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_yaml_with_interpolation(file_path, **kwargs):
    with open(file_path, "r") as file:
        content = file.read()
        interpolated_content = content.format(**kwargs)
        return yaml.safe_load(interpolated_content)


def parse_tasks_with_combinations(config):
    tasks = config["tasks"]
    expanded_tasks = []

    for task in tasks:
        corrupt_args = task["params"].get("corrupt_args", {})
        keys, list_values = (
            zip(*[(k, v) for k, v in corrupt_args.items() if isinstance(v, list)])
            if corrupt_args
            else ((), ())
        )
        if list_values:
            for combination in product(*list_values):
                new_task = deepcopy(task)
                for key, value in zip(keys, combination):
                    new_task["params"]["corrupt_args"][key] = value
                expanded_tasks.append(new_task)
        else:
            expanded_tasks.append(task)

    config["tasks"] = expanded_tasks
    return config


def create_tasks_table(config):
    table_data = []
    for task in config.get("tasks", []):
        params = task.get("params", {})
        dims = params.get("corrupt_args", {}).get("dims", "none")
        strength = params.get("corrupt_args", {}).get("strength", "none")

        # Check and replace None with 'none'
        if dims is None or dims == "none":
            dims_display = "none"
        else:
            dims_display = dims

        if strength is None or strength == "none":
            strength_display = "none"
        else:
            strength_display = strength

        row = [
            task.get("name", "none"),
            params.get("model_path", "none"),
            params.get("corrupt_method", "none"),
            dims_display,
            strength_display,
        ]
        table_data.append(row)

    headers = ["Task Name", "Model Path", "Corruption Method", "Dims", "Strength"]
    return tabulate(table_data, headers=headers, tablefmt="pretty")


def format_dict_for_name(d):
    return "-".join([f"{k}={v}" for k, v in d.items()])


def merge_dicts(dicts):
    return {k: v for d in dicts for k, v in d.items()}


def compute_afe(results, subset_prefix):
    """Compute Answer Forget Efficacy from evaluation results.

    AFE = hmean(1 - ROUGE-L_recall, 1 - cosine_similarity, 1 - entailment_score)
          * think_completion_rate

    The think_completion_rate multiplier discounts AFE when many responses
    lack a </think> delimiter, preventing the optimizer from exploiting
    empty answers for artificially high scores.

    Args:
        results: Dict of metric_key -> score from engine summary.
        subset_prefix: Key prefix for the subset, e.g. "rtofu_forget10".

    Returns:
        AFE score (float), or 0.0 if any component is non-positive.
    """
    metrics = ["rougeL_recall", "cosine_similarity", "entailment_score"]
    scores = []
    for metric in metrics:
        key = f"{subset_prefix}_{metric}"
        if key in results:
            scores.append(1.0 - results[key])
    if scores and all(s > 0 for s in scores):
        afe = float(hmean(scores))
    else:
        afe = 0.0

    rate_key = f"{subset_prefix}_think_completion_rate"
    if rate_key in results:
        afe *= results[rate_key]

    return afe


def compute_cfe(results, subset_prefix):
    """Compute CoT Forget Efficacy from evaluation results.

    CFE = hmean(1 - stepwise_ROUGE-L_recall, 1 - stepwise_cosine_similarity)

    Args:
        results: Dict of metric_key -> score from engine summary.
        subset_prefix: Key prefix for the subset, e.g. "rtofu_forget10".

    Returns:
        CFE score (float), or 0.0 if any component is non-positive.
    """
    metrics = ["stepwise_rougeL_recall", "stepwise_cosine_similarity"]
    scores = []
    for metric in metrics:
        key = f"{subset_prefix}_cot_{metric}"
        if key in results:
            scores.append(1.0 - results[key])
    if scores and all(s > 0 for s in scores):
        return float(hmean(scores))
    return 0.0


def delete_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
