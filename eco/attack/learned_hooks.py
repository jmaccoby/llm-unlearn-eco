"""Learnable inference-time components for CoT regeneration.

``SoftToken`` / ``SoftTokenBank`` — embedding vectors inserted at CoT
truncation points via forward hooks to disrupt forget-set continuations.

``ProjectionHead`` — a small MLP that maps SentenceTransformer embeddings
into a space where distance to claim-cluster centroids discriminates
forget-set leaks from retain-set false positives.  Used as Stage 2 of the
leak detector, replacing the NLI entailment check.
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


class ProjectionHead(nn.Module):
    """MLP that projects sentence embeddings into claim-cluster centroid space.

    Trained contrastively to map forget-set leak sentences near their
    matching cluster centroid and retain sentences far from all centroids.
    Outputs are L2-normalized so that centroid proximity is measured by
    cosine similarity.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (SentenceTransformer embedding).
    hidden_dim : int
        Hidden layer width.
    output_dim : int
        Dimensionality of the projected space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.net(x), dim=-1)

    def save(self, path: str, target_centroids: torch.Tensor) -> None:
        """Save projection weights and target centroids."""
        torch.save(
            {
                "state_dict": self.state_dict(),
                "target_centroids": target_centroids,
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str, device: torch.device | None = None
    ) -> tuple["ProjectionHead", torch.Tensor]:
        """Load projection head and target centroids.

        Returns ``(head, target_centroids)``."""
        data = torch.load(path, weights_only=True, map_location=device)
        head = cls(data["input_dim"], data["hidden_dim"], data["output_dim"])
        head.load_state_dict(data["state_dict"])
        target_centroids = data["target_centroids"]
        if device is not None:
            head = head.to(device)
            target_centroids = target_centroids.to(device)
        return head, target_centroids
