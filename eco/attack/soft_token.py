"""Learnable soft token for insertion at CoT truncation points.

The soft token is a single embedding vector (same dimensionality as the
model's embedding layer) trained to disrupt forget-set continuations while
preserving coherent generation.  At inference, it is inserted at the
truncation point via a forward hook on the embedding layer.
"""

import torch
import torch.nn as nn


class SoftToken(nn.Module):
    """A learnable embedding vector for insertion at truncation points.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding (must match the model's embedding layer).
    """

    def __init__(self, embed_dim: int = 4096):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(embed_dim) * 0.02)

    def apply_hook(self, module: nn.Module, token_position: int):
        """Register a forward hook that replaces the embedding at
        ``token_position`` with the learned vector.

        The hook only fires during prefill (seq_len > 1) to avoid
        interfering with autoregressive generation steps.

        Returns the hook handle for later removal.
        """
        emb = self.embedding  # capture in closure

        def hook(mod, inputs, outputs):
            if outputs.shape[1] > 1:  # prefill only
                outputs[:, token_position, :] = emb.to(
                    device=outputs.device, dtype=outputs.dtype
                )
            return outputs

        return module.register_forward_hook(hook)

    def save(self, path: str) -> None:
        """Save the learned embedding to disk."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, embed_dim: int = 4096) -> "SoftToken":
        """Load a previously saved soft token."""
        st = cls(embed_dim)
        st.load_state_dict(torch.load(path, weights_only=True))
        return st
