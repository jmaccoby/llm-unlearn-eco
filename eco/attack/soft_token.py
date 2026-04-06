"""Learnable soft tokens for insertion at CoT truncation points.

A soft token is a single embedding vector (same dimensionality as the
model's embedding layer) trained to disrupt forget-set continuations while
preserving coherent generation.  At inference, it is inserted at the
truncation point via a forward hook on the embedding layer.

``SoftTokenBank`` extends this to multiple soft tokens, one per claim
cluster, allowing claim-specific corruption.
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
                # Clone to avoid in-place write that severs autograd graph
                outputs = outputs.clone()
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


class SoftTokenBank(nn.Module):
    """A bank of per-cluster soft tokens for claim-specific corruption.

    Each cluster gets its own learned embedding vector.  At inference,
    the leak detector identifies which claim cluster was matched, and
    the corresponding token is applied.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (one soft token per cluster).
    embed_dim : int
        Dimensionality of each embedding.
    """

    def __init__(self, n_clusters: int, embed_dim: int = 4096):
        super().__init__()
        self.tokens = nn.ModuleList(
            [SoftToken(embed_dim) for _ in range(n_clusters)]
        )
        self.n_clusters = n_clusters

    def select(self, cluster_id: int) -> SoftToken:
        """Return the soft token for *cluster_id*."""
        return self.tokens[cluster_id]

    def apply_hook(
        self, module: nn.Module, token_position: int, cluster_id: int = 0
    ):
        """Select the token for *cluster_id* and register its hook.

        Returns the hook handle for later removal.
        """
        return self.tokens[cluster_id].apply_hook(module, token_position)

    def save(self, path: str) -> None:
        """Save all token embeddings to a single file."""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls, path: str, n_clusters: int, embed_dim: int = 4096
    ) -> "SoftTokenBank":
        """Load a previously saved soft token bank."""
        bank = cls(n_clusters, embed_dim)
        bank.load_state_dict(torch.load(path, weights_only=True))
        return bank
