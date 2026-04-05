"""
Tests for regeneration-related corruption utilities:
- build_prefix_corruption_mask() in eco/attack/utils.py
- AttackedModel.generate_with_mask() in eco/attack/model.py

Usage:
    conda run -n eco python -m pytest tests/test_regen_corruption.py -v
"""
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from eco.attack.model import AttackedModel
from eco.attack.utils import build_prefix_corruption_mask


# ---------------------------------------------------------------------------
# Mask construction tests
# ---------------------------------------------------------------------------

class TestBuildPrefixCorruptionMask:
    def test_prefix_mask_shape(self):
        """Mask length equals prompt_len + think_len + prefix_len."""
        prompt_len, think_len, prefix_len, window = 10, 5, 20, 8
        mask = build_prefix_corruption_mask(prompt_len, think_len, prefix_len, window)
        assert len(mask) == 1
        assert len(mask[0]) == prompt_len + think_len + prefix_len

    def test_prefix_mask_only_window(self):
        """Only the last `window` positions are 1, rest are 0."""
        prompt_len, think_len, prefix_len, window = 10, 5, 20, 8
        mask = build_prefix_corruption_mask(prompt_len, think_len, prefix_len, window)[0]
        total_len = prompt_len + think_len + prefix_len
        assert mask[total_len - window:] == [1] * window
        assert mask[:total_len - window] == [0] * (total_len - window)

    def test_prefix_mask_window_clamped(self):
        """When window > prefix_len, only prefix positions are corrupted."""
        prompt_len, think_len, prefix_len = 10, 5, 8
        window = 100  # Much larger than prefix_len
        mask = build_prefix_corruption_mask(prompt_len, think_len, prefix_len, window)[0]
        total_len = prompt_len + think_len + prefix_len
        assert len(mask) == total_len
        assert mask[:prompt_len + think_len] == [0] * (prompt_len + think_len)
        assert mask[prompt_len + think_len:] == [1] * prefix_len

    def test_prefix_mask_zero_window(self):
        """window=0 produces all-zero mask."""
        mask = build_prefix_corruption_mask(10, 5, 20, 0)[0]
        assert all(v == 0 for v in mask)
        assert len(mask) == 35

    def test_prefix_mask_batch_size(self):
        """batch_size=3 produces 3 identical rows."""
        masks = build_prefix_corruption_mask(10, 5, 20, 8, batch_size=3)
        assert len(masks) == 3
        assert masks[0] == masks[1] == masks[2]


# ---------------------------------------------------------------------------
# Minimal mock infrastructure for AttackedModel.generate_with_mask tests
# ---------------------------------------------------------------------------

class FakeEmbeddingModule(nn.Module):
    def forward(self, x):
        return x


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = FakeEmbeddingModule()

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])

    def modules(self):
        return iter([self, self.embed_tokens])


class FakeHFModel:
    def __init__(self):
        self.model = FakeInnerModel()
        self.tokenizer = MagicMock()
        self.tokenizer.padding_side = "right"
        self.device = torch.device("cpu")
        self.model_name = "fake-model"
        self.model_config = {"attack_module": "embed_tokens"}
        self.generation_config = None


def _make_attacked_model():
    hf_model = FakeHFModel()
    return AttackedModel(
        model=hf_model,
        prompt_classifier=None,
        token_classifier=None,
        corrupt_method="rand_noise_first_n",
        corrupt_args={"strength": 10, "dims": 4},
    )


# ---------------------------------------------------------------------------
# generate_with_mask tests
# ---------------------------------------------------------------------------

class TestGenerateWithMask:
    def test_generate_with_mask_uses_custom_pos(self):
        """Verify the hook is registered with the provided pos_mask."""
        attacked = _make_attacked_model()
        custom_mask = [[0, 0, 1, 1, 1]]

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_handle = MagicMock()
            mock_hook.return_value = mock_handle
            attacked.generate_with_mask(
                custom_mask, input_ids=torch.tensor([[1, 2, 3, 4, 5]])
            )
            mock_hook.assert_called_once()
            corrupt_args_passed = mock_hook.call_args[0][2]
            assert corrupt_args_passed["pos"] == custom_mask

    def test_generate_with_mask_cleans_previous_hooks(self):
        """Verify existing hooks are removed before registering new ones."""
        attacked = _make_attacked_model()
        # Plant a dummy handle
        dummy_handle = MagicMock()
        attacked._hook_handles.append(dummy_handle)

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_hook.return_value = MagicMock()
            attacked.generate_with_mask(
                [[1, 1]], input_ids=torch.tensor([[1, 2]])
            )
            # The dummy handle should have been removed
            dummy_handle.remove.assert_called_once()
