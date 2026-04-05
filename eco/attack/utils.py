from collections import OrderedDict

import torch

from eco.attack.corrupt import corrupt_methods


def apply_corruption_hook(module, corrupt_method, corrupt_args):
    corrupt_fn = corrupt_methods[corrupt_method]

    def corrupt(module, inputs, outputs):
        if outputs.shape[1] > 1:
            # Fit the position mask to the actual sequence length.  The
            # mask may be longer (prompt truncated at tokenization time) or
            # shorter (e.g. think-prefix tokens appended after corruption
            # setup).  Truncate long masks and zero-pad short ones so the
            # mask always matches outputs.shape[1].
            seq_len = outputs.shape[1]
            safe_args = corrupt_args.copy()
            if "pos" in safe_args:
                safe_args["pos"] = [
                    row[:seq_len] if len(row) >= seq_len
                    else row + [0] * (seq_len - len(row))
                    for row in safe_args["pos"]
                ]
            outputs = corrupt_fn(outputs, **safe_args)
        return outputs

    handle = module.register_forward_hook(corrupt)
    return handle


def apply_embeddings_extraction_hook(module, embeddings):
    def extract_embeddings(module, inputs, outputs):
        embeddings.append(outputs.detach())

    handle = module.register_forward_hook(extract_embeddings)
    return handle


def cosine_similarity_matrix(row_vectors, matrix, eps=1e-8):
    dot_product = torch.mm(row_vectors, matrix.t())
    row_vectors_norm = torch.norm(row_vectors, p=2, dim=1, keepdim=True) + eps
    matrix_norms = torch.norm(matrix, p=2, dim=1, keepdim=True) + eps
    cosine_similarity = dot_product / (row_vectors_norm * matrix_norms.t())
    return cosine_similarity


def embedding_to_tokens(embeddings, embedding_matrix):
    similarities = cosine_similarity_matrix(embeddings, embedding_matrix)
    selected_tokens = torch.argmax(similarities, dim=1)
    return selected_tokens, similarities


def pad_to_same_length(pos, padding_side="right"):
    assert padding_side in [
        "right",
        "left",
    ], "padding_side must be either right or left"
    max_len = max(pos, key=len)
    if padding_side == "right":
        return [i + [0] * (len(max_len) - len(i)) for i in pos]
    else:
        return [[0] * (len(max_len) - len(i)) + i for i in pos]


def match_labeled_tokens(src_labels, src_offsets, tgt_offsets):
    src_target_offsets = [
        offset for offset, label in zip(src_offsets, src_labels) if label == 1
    ]
    tgt_matched_tokens_indices = []
    for i, (tgt_start, tgt_end) in enumerate(tgt_offsets):
        for src_start, src_end in src_target_offsets:
            if src_start < tgt_end and src_end > tgt_start:
                tgt_matched_tokens_indices.append(i)
                break

    tgt_labels = []
    for i in range(len(tgt_offsets)):
        if i in tgt_matched_tokens_indices:
            tgt_labels.append(1)
        else:
            tgt_labels.append(0)
    return tgt_labels


def remove_hooks(model):
    for module in model.modules():
        for handle_id in list(module._forward_hooks):
            module._forward_hooks.pop(handle_id)


def print_hooks(model):
    for module in model.modules():
        if module._forward_hooks != OrderedDict():
            print(module, module._forward_hooks)


def get_nested_attr(obj, attr):
    for a in attr.split("."):
        obj = getattr(obj, a)
    return obj


def remove_none_values(d):
    return {k: v for k, v in d.items() if v is not None}


def idx_to_mask(idx, length):
    idx_set = set(idx)  # Convert to set for efficient lookup
    return [1 if i in idx_set else 0 for i in range(length)]


def mask_to_idx(mask):
    return [i for i, m in enumerate(mask) if m == 1]


def build_prefix_corruption_mask(
    prompt_len: int,
    think_len: int,
    prefix_len: int,
    window: int,
    batch_size: int = 1,
) -> list[list[int]]:
    """Build a position mask that corrupts only the last ``window`` tokens
    of the clean CoT prefix.

    The mask layout is::

        [0]*prompt_len + [0]*think_len + [0]*(prefix_len - window) + [1]*window

    ``window`` is clamped to ``prefix_len`` so the prompt and think tokens
    are never corrupted.
    """
    window = min(window, prefix_len)
    clean_len = prompt_len + think_len + (prefix_len - window)
    mask = [0] * clean_len + [1] * window
    return [mask] * batch_size
