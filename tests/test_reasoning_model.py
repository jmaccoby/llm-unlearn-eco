"""
Tests for the ReasoningModel wrapper and unified think-prefix handling.

Covers:
- ReasoningModel attribute forwarding and think token appending
- Corruption hook short-mask padding (zero-pads masks shorter than seq_len)
- End-to-end: ReasoningModel(AttackedModel(...)) with think tokens protected
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from eco.attack.corrupt import corrupt_methods
from eco.attack.model import AttackedModel
from eco.attack.utils import apply_corruption_hook, remove_hooks
from eco.model.reasoning import ReasoningModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleEmbedding(nn.Module):
    """Embedding module for hook tests."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(100, dim))

    def forward(self, x):
        return x


class FakeInnerModel(nn.Module):
    """Model with a nested embedding for AttackedModel tests."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = SimpleEmbedding()

    def forward(self, x):
        return self.embed_tokens(x)

    def generate(self, **kwargs):
        """Return input_ids extended with dummy generated tokens."""
        input_ids = kwargs["input_ids"]
        # Append 5 dummy token IDs as "generated" output
        dummy = torch.full((input_ids.shape[0], 5), fill_value=42)
        return torch.cat([input_ids, dummy], dim=1)


def _make_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class FakeHFModel:
    """Minimal HFModel-like object."""

    def __init__(self, tokenizer=None):
        self.model = FakeInnerModel()
        self.tokenizer = tokenizer or _make_tokenizer()
        self.device = torch.device("cpu")
        self.model_name = "test"
        self.model_config = {
            "attack_module": "embed_tokens",
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
            },
        }
        self.generation_config = type("GC", (), {"max_new_tokens": 64})()

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)
        return self.model.generate(**kwargs)


# ---------------------------------------------------------------------------
# ReasoningModel basics
# ---------------------------------------------------------------------------


class TestReasoningModelAttributes:
    def test_forwards_attributes(self):
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        assert rm.device == hf.device
        assert rm.tokenizer is hf.tokenizer
        assert rm.model_name == "test"
        assert rm.generation_config is hf.generation_config

    def test_inner_accessible(self):
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        assert rm._inner is hf

    def test_n_think_tokens_positive(self):
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        assert rm.n_think_tokens > 0

    def test_n_think_tokens_cached(self):
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        n1 = rm.n_think_tokens
        n2 = rm.n_think_tokens
        assert n1 == n2
        # The cached value should be stored in __dict__
        assert "_n_think_tokens" in rm.__dict__

    def test_n_think_tokens_matches_tokenizer(self):
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        expected = len(hf.tokenizer("<think>\n", add_special_tokens=False)["input_ids"])
        assert rm.n_think_tokens == expected


class TestReasoningModelGenerate:
    def test_appends_think_tokens(self):
        """generate() should append think token IDs to input_ids."""
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        tok = hf.tokenizer

        prompt = "Hello world"
        inputs = tok(prompt, return_tensors="pt")
        original_len = inputs["input_ids"].shape[1]

        output = rm.generate(**inputs)
        # Output should be: original prompt + think tokens + 5 dummy tokens
        expected_input_len = original_len + rm.n_think_tokens
        # FakeInnerModel.generate appends 5 tokens
        assert output.shape[1] == expected_input_len + 5

    def test_think_tokens_in_output(self):
        """The think token IDs should appear in the generated output."""
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        tok = hf.tokenizer

        prompt = "Test"
        inputs = tok(prompt, return_tensors="pt")
        original_len = inputs["input_ids"].shape[1]
        output = rm.generate(**inputs)

        think_ids = tok("<think>\n", add_special_tokens=False, return_tensors="pt")["input_ids"]
        # Check that think IDs appear right after the prompt
        output_think = output[0, original_len:original_len + think_ids.shape[1]]
        assert torch.equal(output_think, think_ids[0])

    def test_attention_mask_extended(self):
        """If attention_mask is provided, it should be extended for think tokens."""
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        tok = hf.tokenizer

        inputs = tok("Hello", return_tensors="pt")
        original_mask_len = inputs["attention_mask"].shape[1]

        # Capture what's passed to inner.generate
        captured = {}
        original_generate = hf.generate

        def spy_generate(*args, **kwargs):
            captured["attention_mask"] = kwargs.get("attention_mask")
            captured["input_ids"] = kwargs.get("input_ids")
            return original_generate(*args, **kwargs)

        hf.generate = spy_generate
        rm.generate(**inputs)

        assert captured["attention_mask"].shape[1] == original_mask_len + rm.n_think_tokens
        # All attention mask values should be 1 (no padding)
        assert captured["attention_mask"].sum() == captured["attention_mask"].numel()

    def test_no_attention_mask(self):
        """generate() should work without attention_mask."""
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        tok = hf.tokenizer

        input_ids = tok("Test", return_tensors="pt")["input_ids"]
        # Should not raise
        output = rm.generate(input_ids=input_ids)
        assert output.shape[1] > input_ids.shape[1]

    def test_batch_generation(self):
        """generate() should handle batched inputs."""
        hf = FakeHFModel()
        rm = ReasoningModel(hf)
        tok = hf.tokenizer

        inputs = tok(["Short", "A longer prompt here"], padding=True, return_tensors="pt")
        output = rm.generate(**inputs)
        assert output.shape[0] == 2

    def test_call_delegates(self):
        """__call__ should delegate to inner model."""
        call_log = []

        class CallableModel(FakeHFModel):
            def __call__(self, *args, **kwargs):
                call_log.append(args)
                return "ok"

        hf = CallableModel()
        rm = ReasoningModel(hf)
        result = rm("test")
        assert len(call_log) == 1
        assert result == "ok"


# ---------------------------------------------------------------------------
# Hook short-mask padding
# ---------------------------------------------------------------------------


class TestHookShortMaskPadding:
    def test_short_mask_padded_with_zeros(self):
        """When mask is shorter than sequence, hook should pad with zeros."""
        module = SimpleEmbedding(dim=8)
        # Mask covers 3 positions, but sequence will be 5
        pos = [[1, 1, 0]]
        handle = apply_corruption_hook(
            module, "rand_noise_first_n", {"pos": pos, "dims": 1, "strength": 100.0}
        )

        # Use non-zero data so corruption is distinguishable from original values
        data = torch.ones(1, 5, 8)
        original_tail = data[0, 3:, :].clone()
        output = module(data)

        # Positions 3 and 4 (beyond the mask) should be unchanged
        assert torch.equal(output[0, 3:, :], original_tail)
        handle.remove()

    def test_long_mask_truncated(self):
        """When mask is longer than sequence, hook should truncate."""
        module = SimpleEmbedding(dim=8)
        # Mask covers 10 positions, but sequence will be 3
        pos = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        handle = apply_corruption_hook(
            module, "rand_noise_first_n", {"pos": pos, "dims": 1, "strength": 100.0}
        )

        # Should not crash with out-of-bounds access
        data = torch.zeros(1, 3, 8)
        output = module(data)
        assert output.shape == (1, 3, 8)
        handle.remove()

    def test_exact_length_mask_unchanged(self):
        """When mask exactly matches sequence length, no padding/truncation needed."""
        module = SimpleEmbedding(dim=8)
        pos = [[1, 0, 1]]
        handle = apply_corruption_hook(
            module, "rand_noise_first_n", {"pos": pos, "dims": 1, "strength": 100.0}
        )

        data = torch.zeros(1, 3, 8)
        original_pos1 = data[0, 1, :].clone()
        output = module(data)

        # Position 1 (mask=0) should be unchanged
        assert torch.equal(output[0, 1, :], original_pos1)
        handle.remove()


# ---------------------------------------------------------------------------
# End-to-end: ReasoningModel(AttackedModel(...))
# ---------------------------------------------------------------------------


class TestReasoningAttackedIntegration:
    """Test that ReasoningModel(AttackedModel(...)) correctly applies corruption
    to prompt tokens while leaving think tokens untouched."""

    def _make_reasoning_attacked(self):
        hf = FakeHFModel()

        class AlwaysForgetClassifier:
            def predict(self, prompts, threshold):
                return [1] * len(prompts)

        attacked = AttackedModel(
            model=hf,
            prompt_classifier=AlwaysForgetClassifier(),
            token_classifier=None,
            corrupt_method="rand_noise_first_n",
            corrupt_args={"dims": 1, "strength": 100.0},
        )
        return ReasoningModel(attacked), hf

    def test_wrapping_order(self):
        rm, hf = self._make_reasoning_attacked()
        assert isinstance(rm._inner, AttackedModel)
        assert rm.tokenizer is hf.tokenizer

    def test_generate_includes_think_tokens(self):
        """ReasoningModel should append think tokens before AttackedModel generates."""
        rm, hf = self._make_reasoning_attacked()
        tok = hf.tokenizer

        inputs = tok("test prompt", return_tensors="pt")
        original_len = inputs["input_ids"].shape[1]

        output = rm.generate(**inputs, prompts=["test prompt"])
        # Output = prompt + think tokens + 5 dummy
        expected_input = original_len + rm.n_think_tokens + 5
        assert output.shape[1] == expected_input

    def test_corruption_mask_shorter_than_sequence(self):
        """The corruption mask should cover only prompt tokens, not think tokens.
        The hook should pad the mask with zeros for the think token positions."""
        rm, hf = self._make_reasoning_attacked()
        attacked = rm._inner

        # Apply corruption for a prompt
        attacked.remove_hooks()
        attacked.apply_corruption(["test prompt"])

        # The hook is now registered. Check that the mask in the hook's
        # corrupt_args is prompt-length (shorter than what the full input
        # will be after ReasoningModel appends think tokens).
        # We can't inspect the hook directly, but we can verify behavior:
        # feed an embedding through the hook where the sequence is longer
        # than the mask, and check that the tail positions are untouched.
        embed = attacked.attack_module
        data = torch.zeros(1, 20, 16)  # 20 positions, mask will be shorter
        original_tail = data[0, -5:, :].clone()
        output = embed(data)

        # Last 5 positions should be untouched (padded with zero mask)
        assert torch.equal(output[0, -5:, :], original_tail)
        attacked.remove_hooks()

    def test_remove_hooks_through_reasoning_model(self):
        """_remove_hooks should reach AttackedModel through ReasoningModel."""
        from eco.inference import _remove_hooks

        rm, _ = self._make_reasoning_attacked()
        attacked = rm._inner

        attacked.apply_corruption(["test"])
        assert len(attacked._hook_handles) == 1

        _remove_hooks(rm)
        assert len(attacked._hook_handles) == 0
