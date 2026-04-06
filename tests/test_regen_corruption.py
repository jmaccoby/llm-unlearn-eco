"""
Tests for regeneration-related corruption utilities:
- build_prefix_corruption_mask() in eco/attack/utils.py
- AttackedModel.regenerate() in eco/attack/model.py

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
# Minimal mock infrastructure for AttackedModel.regenerate tests
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


def _make_attacked_model(**overrides):
    hf_model = FakeHFModel()
    kwargs = dict(
        model=hf_model,
        prompt_classifier=None,
        token_classifier=None,
        corrupt_method="rand_noise_first_n",
        corrupt_args={"strength": 10, "dims": 4},
        regen_corrupt_method="rand_noise_first_n",
        regen_corrupt_args={"strength": 5, "dims": 2},
    )
    kwargs.update(overrides)
    return AttackedModel(**kwargs)


# ---------------------------------------------------------------------------
# regenerate tests
# ---------------------------------------------------------------------------

class TestRegenerate:
    def test_regenerate_uses_regen_config(self):
        """Verify regenerate() uses regen_corrupt_method/args, not prompt config."""
        attacked = _make_attacked_model()
        custom_mask = [[0, 0, 1, 1, 1]]

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_handle = MagicMock()
            mock_hook.return_value = mock_handle
            attacked.regenerate(
                custom_mask, input_ids=torch.tensor([[1, 2, 3, 4, 5]])
            )
            mock_hook.assert_called_once()
            call_args = mock_hook.call_args[0]
            # Should use regen method, not prompt method
            assert call_args[1] == "rand_noise_first_n"
            corrupt_args_passed = call_args[2]
            assert corrupt_args_passed["pos"] == custom_mask
            # Should use regen dims/strength (5, 2), not prompt (10, 4)
            assert corrupt_args_passed["strength"] == 5
            assert corrupt_args_passed["dims"] == 2

    def test_regenerate_cleans_previous_hooks(self):
        """Verify existing hooks are removed before registering new ones."""
        attacked = _make_attacked_model()
        dummy_handle = MagicMock()
        attacked._hook_handles.append(dummy_handle)

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_hook.return_value = MagicMock()
            attacked.regenerate(
                [[1, 1]], input_ids=torch.tensor([[1, 2]])
            )
            dummy_handle.remove.assert_called_once()

    def test_regenerate_without_prompt_corruption(self):
        """AttackedModel with only regen config can regenerate."""
        attacked = _make_attacked_model(
            corrupt_method=None, corrupt_args=None,
        )
        custom_mask = [[0, 1, 1]]

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_handle = MagicMock()
            mock_hook.return_value = mock_handle
            attacked.regenerate(
                custom_mask, input_ids=torch.tensor([[1, 2, 3]])
            )
            mock_hook.assert_called_once()
            corrupt_args_passed = mock_hook.call_args[0][2]
            assert corrupt_args_passed["dims"] == 2
            assert corrupt_args_passed["strength"] == 5

    def test_regenerate_applies_soft_token(self):
        """Verify soft_token.apply_hook is called with the correct position."""
        mock_soft_token = MagicMock()
        mock_st_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_st_handle

        attacked = _make_attacked_model(soft_token=mock_soft_token)

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_hook.return_value = MagicMock()
            attacked.regenerate(
                [[0, 1]], soft_token_position=5,
                input_ids=torch.tensor([[1, 2]]),
            )
            mock_soft_token.apply_hook.assert_called_once_with(
                attacked.attack_module, 5
            )
            # Soft token handle should be removed after generate
            mock_st_handle.remove.assert_called_once()

    def test_regenerate_no_soft_token_when_position_none(self):
        """soft_token.apply_hook is NOT called when position is None."""
        mock_soft_token = MagicMock()
        attacked = _make_attacked_model(soft_token=mock_soft_token)

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_hook.return_value = MagicMock()
            attacked.regenerate(
                [[0, 1]], soft_token_position=None,
                input_ids=torch.tensor([[1, 2]]),
            )
            mock_soft_token.apply_hook.assert_not_called()

    def test_regenerate_cleans_up_on_generate_error(self):
        """Both hooks are cleaned up even when model.generate() raises."""
        mock_soft_token = MagicMock()
        mock_st_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_st_handle

        attacked = _make_attacked_model(soft_token=mock_soft_token)
        # Make model.generate() raise
        attacked.model.generate = MagicMock(side_effect=RuntimeError("boom"))

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            mock_corrupt_handle = MagicMock()
            mock_hook.return_value = mock_corrupt_handle
            try:
                attacked.regenerate(
                    [[0, 1]], soft_token_position=3,
                    input_ids=torch.tensor([[1, 2]]),
                )
            except RuntimeError:
                pass
            # Both handles should still be removed
            mock_corrupt_handle.remove.assert_called_once()
            mock_st_handle.remove.assert_called_once()

    def test_regenerate_soft_token_only_mode(self):
        """With regen_corrupt_method=None, only the soft token hook fires."""
        mock_soft_token = MagicMock()
        mock_st_handle = MagicMock()
        mock_soft_token.apply_hook.return_value = mock_st_handle

        attacked = _make_attacked_model(
            regen_corrupt_method=None,
            regen_corrupt_args=None,
            soft_token=mock_soft_token,
        )

        with patch("eco.attack.model.apply_corruption_hook") as mock_hook:
            attacked.regenerate(
                [[0, 0]], soft_token_position=1,
                input_ids=torch.tensor([[1, 2]]),
            )
            # No corruption hook should be registered
            mock_hook.assert_not_called()
            # Soft token hook should still be applied
            mock_soft_token.apply_hook.assert_called_once()
