"""
Tests for the SoftToken learnable embedding.

Usage:
    conda run -n eco python -m pytest tests/test_soft_token.py -v
"""

import torch
import torch.nn as nn

from eco.attack.soft_token import SoftToken


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
