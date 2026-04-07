"""
Tests for SoftToken and SoftTokenBank learnable embeddings.

Usage:
    conda run -n eco python -m pytest tests/test_soft_token.py -v
"""

import torch
import torch.nn as nn

from eco.attack.learned_hooks import SoftToken, SoftTokenBank


def test_init_shape():
    st = SoftToken(embed_dim=128)
    assert st.embedding.shape == (128,)


def test_init_default_dim():
    st = SoftToken()
    assert st.embedding.shape == (4096,)


def test_hook_replaces_at_position():
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim)
    module = nn.Linear(10, embed_dim)

    handle = st.apply_hook(module, token_position=2)
    try:
        x = torch.randn(1, 5, 10)
        # nn.Linear supports (*, H_in) so (1, 5, 10) -> (1, 5, 64)
        output = module(x)
        expected = st.embedding.detach().to(device=output.device, dtype=output.dtype)
        assert torch.allclose(output[:, 2, :], expected.unsqueeze(0))
    finally:
        handle.remove()


def test_hook_prefill_only():
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim)
    module = nn.Linear(10, embed_dim)

    handle = st.apply_hook(module, token_position=0)
    try:
        x = torch.randn(1, 1, 10)
        output = module(x)
        # Re-run without hook to get baseline
        handle.remove()
        baseline = module(x)
        assert torch.equal(output, baseline), (
            "Hook should be a no-op when seq_len == 1"
        )
    finally:
        pass  # handle already removed


def test_hook_does_not_modify_other_positions():
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim)
    module = nn.Linear(10, embed_dim)

    x = torch.randn(1, 5, 10)
    # Get baseline without hook
    with torch.no_grad():
        baseline = module(x).clone()

    handle = st.apply_hook(module, token_position=2)
    try:
        with torch.no_grad():
            output = module(x)
        # Positions 0, 1, 3, 4 should be unchanged
        for pos in [0, 1, 3, 4]:
            assert torch.equal(output[:, pos, :], baseline[:, pos, :]), (
                f"Position {pos} was modified but should not have been"
            )
    finally:
        handle.remove()


def test_save_load_roundtrip(tmp_path):
    embed_dim = 128
    st = SoftToken(embed_dim=embed_dim)
    path = str(tmp_path / "soft_token.pt")
    st.save(path)

    loaded = SoftToken.load(path, embed_dim=embed_dim)
    assert torch.equal(st.embedding, loaded.embedding)


def test_gradient_flows():
    embed_dim = 64
    st = SoftToken(embed_dim=embed_dim)
    module = nn.Linear(10, embed_dim)

    handle = st.apply_hook(module, token_position=1)
    try:
        x = torch.randn(1, 3, 10)
        output = module(x)
        loss = output.sum()
        loss.backward()
        assert st.embedding.grad is not None, (
            "Gradient should flow through the hook to the soft token embedding"
        )
    finally:
        handle.remove()


def test_is_nn_module():
    assert isinstance(SoftToken(), nn.Module)


# ---------------------------------------------------------------------------
# SoftTokenBank tests
# ---------------------------------------------------------------------------


class TestSoftTokenBank:
    def test_init_creates_correct_number_of_tokens(self):
        bank = SoftTokenBank(n_clusters=5, embed_dim=64)
        assert bank.n_clusters == 5
        assert len(bank.tokens) == 5
        for st in bank.tokens:
            assert isinstance(st, SoftToken)
            assert st.embedding.shape == (64,)

    def test_select_returns_correct_token(self):
        bank = SoftTokenBank(n_clusters=3, embed_dim=32)
        for i in range(3):
            st = bank.select(i)
            assert torch.equal(st.embedding, bank.tokens[i].embedding)

    def test_apply_hook_with_cluster_id(self):
        embed_dim = 64
        bank = SoftTokenBank(n_clusters=3, embed_dim=embed_dim)
        module = nn.Linear(10, embed_dim)

        # Apply cluster 1's token at position 2
        handle = bank.apply_hook(module, token_position=2, cluster_id=1)
        try:
            x = torch.randn(1, 5, 10)
            output = module(x)
            expected = bank.tokens[1].embedding.detach()
            assert torch.allclose(
                output[:, 2, :], expected.unsqueeze(0)
            )
        finally:
            handle.remove()

    def test_different_clusters_apply_different_embeddings(self):
        embed_dim = 64
        bank = SoftTokenBank(n_clusters=2, embed_dim=embed_dim)
        module = nn.Linear(10, embed_dim)
        x = torch.randn(1, 5, 10)

        # Cluster 0
        handle = bank.apply_hook(module, token_position=0, cluster_id=0)
        with torch.no_grad():
            out_0 = module(x)[:, 0, :].clone()
        handle.remove()

        # Cluster 1
        handle = bank.apply_hook(module, token_position=0, cluster_id=1)
        with torch.no_grad():
            out_1 = module(x)[:, 0, :].clone()
        handle.remove()

        assert not torch.equal(out_0, out_1), (
            "Different cluster tokens should produce different embeddings"
        )

    def test_save_load_roundtrip(self, tmp_path):
        bank = SoftTokenBank(n_clusters=3, embed_dim=32)
        path = str(tmp_path / "bank.pt")
        bank.save(path)

        loaded = SoftTokenBank.load(path, n_clusters=3, embed_dim=32)
        for i in range(3):
            assert torch.equal(
                bank.tokens[i].embedding, loaded.tokens[i].embedding
            )

    def test_gradient_isolation(self):
        """Gradient only flows to the selected token."""
        embed_dim = 32
        bank = SoftTokenBank(n_clusters=3, embed_dim=embed_dim)
        module = nn.Linear(10, embed_dim)

        handle = bank.apply_hook(module, token_position=1, cluster_id=1)
        try:
            x = torch.randn(1, 3, 10)
            output = module(x)
            loss = output.sum()
            loss.backward()
            # Token 1 should have gradients
            assert bank.tokens[1].embedding.grad is not None
            # Tokens 0 and 2 should have no gradients
            assert bank.tokens[0].embedding.grad is None
            assert bank.tokens[2].embedding.grad is None
        finally:
            handle.remove()

    def test_is_nn_module(self):
        assert isinstance(SoftTokenBank(2, 32), nn.Module)
