"""
Tests for the RegeneratingReasoningEngine using dummy models and mock
leak detectors.

Usage:
    conda run -n eco python -m pytest tests/test_regen_engine.py -v
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from eco.attack.leak_detector import LeakDetectionResult
from eco.dataset.rtofu import RTOFU
from eco.evaluator import ROUGERecall
from eco.inference_regen import RegeneratingReasoningEngine
from eco.model.reasoning import ReasoningModel


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

THINK_SUFFIX = "</think>\n\n"


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------

class DummyInnerModel(nn.Module):
    """Minimal nn.Module so remove_hooks() can iterate .modules()."""

    def modules(self):
        return iter([self])


class DummyModel:
    """Drop-in replacement for HFModel that returns canned generations.

    For each prompt it produces:
        <think>\nI considered the question.\n</think>\n\nDummy answer.
    """

    def __init__(self, tokenizer):
        self.model = DummyInnerModel()
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_name = "dummy"
        self.model_config = {
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            }
        }
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=64,
            use_cache=False,
        )
        # AttackedModel compatibility
        self._hook_handles = []
        self.attack_module = self.model
        self.regen_corrupt_method = "rand_noise_first_n"
        self.regen_corrupt_args = {"dims": 4, "strength": 1.0}
        self.soft_token = None

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def regenerate(self, pos_mask, soft_token_position=None, **kwargs):
        """Matches AttackedModel.regenerate interface."""
        self.remove_hooks()
        return self.generate(**kwargs)

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


class LeakingDummyModel:
    """Produces gold-CoT-matching output on first call, clean output on subsequent.

    First call: CoT that contains leaking sentences (gold text).
    Second+ calls: Generic CoT that doesn't match the gold.

    This simulates corruption successfully disrupting the leaking continuation.
    """

    def __init__(self, tokenizer, leaking_cot="The secret fact is revealed.", clean_cot="I think carefully about this."):
        self.model = DummyInnerModel()
        self.tokenizer = tokenizer
        self.device = torch.device("cpu")
        self.model_name = "leaking-dummy"
        self.model_config = {
            "formatting_tokens": {
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            }
        }
        self.generation_config = GenerationConfig(
            do_sample=False,
            max_new_tokens=128,
            use_cache=False,
        )
        self.leaking_cot = leaking_cot
        self.clean_cot = clean_cot
        self.call_count = 0
        # AttackedModel compatibility
        self._hook_handles = []
        self.attack_module = self.model
        self.regen_corrupt_method = "rand_noise_first_n"
        self.regen_corrupt_args = {"dims": 4, "strength": 1.0}
        self.soft_token = None

    def remove_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            if self.call_count == 1:
                cot = self.leaking_cot
            else:
                cot = self.clean_cot
            canned_response = f"{cot}{THINK_SUFFIX}An answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)

    def regenerate(self, pos_mask, soft_token_position=None, **kwargs):
        """Matches AttackedModel.regenerate interface."""
        self.remove_hooks()
        return self.generate(**kwargs)


class AlwaysLeakingDummyModel(LeakingDummyModel):
    """Always produces leaking output, regardless of call count.

    Overrides generate() so that every call returns the leaking CoT.
    Inherits regenerate() from LeakingDummyModel which delegates to generate().
    """

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            prompt_ids = input_ids[i]
            self.call_count += 1
            canned_response = f"{self.leaking_cot}{THINK_SUFFIX}Leaking answer."
            response_ids = self.tokenizer.encode(
                canned_response, add_special_tokens=False
            )
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [
            r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results
        ]
        return torch.tensor(padded)


# ---------------------------------------------------------------------------
# Mock leak detector
# ---------------------------------------------------------------------------

class MockLeakDetector:
    """Configurable mock leak detector.

    Parameters
    ----------
    leak_on_calls : set[int]
        Set of call indices (0-based) on which ``detect()`` returns
        ``is_leaking=True``.  All other calls return clean.
    first_leak_index : int
        Sentence index to report as the first leak.
    """

    def __init__(self, leak_on_calls=None, first_leak_index=1):
        self.leak_on_calls = leak_on_calls or set()
        self.first_leak_index = first_leak_index
        self._detect_count = 0

    def detect(self, generated_cot):
        from eco.evaluator.utils import split_sentences
        sentences = split_sentences(generated_cot)
        idx = self._detect_count
        self._detect_count += 1

        is_leaking = idx in self.leak_on_calls
        flags = [False] * len(sentences)
        confirmed = [False] * len(sentences)
        leak_idx = None

        if is_leaking and len(sentences) > self.first_leak_index:
            leak_idx = self.first_leak_index
            flags[leak_idx] = True
            confirmed[leak_idx] = True

        return LeakDetectionResult(
            is_leaking=is_leaking,
            first_leak_index=leak_idx,
            sentences=sentences,
            stage1_flags=flags,
            confirmed_flags=confirmed,
        )

    def detect_batch(self, cots):
        return [self.detect(cot) for cot in cots]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_tokenizer():
    """Use a small, fast tokenizer."""
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _make_rtofu(tokenizer):
    rtofu = RTOFU(
        formatting_tokens={
            "prompt_prefix": "",
            "prompt_suffix": "",
            "answer_prefix": "",
            "answer_suffix": "",
        },
        eos_token=tokenizer.eos_token,
    )
    rtofu.download()
    return rtofu


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNoRegenWhenNoLeak:
    """When the detector returns is_leaking=False for all, output matches initial generation."""

    def setup_method(self):
        self.tokenizer = _make_tokenizer()
        self.rtofu = _make_rtofu(self.tokenizer)
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_no_regen_when_no_leak(self):
        model = ReasoningModel(DummyModel(self.tokenizer))
        # Detector that never flags anything
        detector = MockLeakDetector(leak_on_calls=set())

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        # All answers should be "Dummy answer." from DummyModel (no regeneration)
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans


class TestRegenOnLeakDetected:
    """When the detector flags the first generation, regenerated output is used."""

    def setup_method(self):
        self.tokenizer = _make_tokenizer()
        self.rtofu = _make_rtofu(self.tokenizer)
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_regen_on_leak_detected(self):
        # The inner model leaks on first call, produces clean output on second
        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets here.",
            clean_cot="I think carefully about this.",
        )
        model = ReasoningModel(inner)

        # Detector flags call 0 (initial generation) as leaking at sentence 1,
        # then call 1 (regenerated output) as clean
        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_corrupt_mode="window",
            regen_window=8,
            regen_window_mode="tokens",
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        key = "rtofu_forget10"
        generated_cot = engine.cot_generations[key]["generated"]
        # The regenerated CoT should contain the clean model's output, not the leak
        assert len(generated_cot) == 1
        # After regeneration, the CoT should include the clean prefix + clean continuation
        assert "secret" not in generated_cot[0].lower() or "think carefully" in generated_cot[0].lower()
        # Verify inner model was called more than once (regeneration happened)
        assert inner.call_count > 1


class TestMaxAttemptsRespected:
    """When the detector always flags, regeneration stops after max_attempts."""

    def setup_method(self):
        self.tokenizer = _make_tokenizer()
        self.rtofu = _make_rtofu(self.tokenizer)
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_max_attempts_respected(self):
        inner = AlwaysLeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets.",
        )
        model = ReasoningModel(inner)

        max_attempts = 3
        # Flag all calls as leaking: call 0 (initial), calls 1,2,3 (regen attempts)
        detector = MockLeakDetector(
            leak_on_calls={0, 1, 2, 3, 4, 5},
            first_leak_index=1,
        )

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_corrupt_mode="window",
            regen_window=8,
            regen_window_mode="tokens",
            regen_max_attempts=max_attempts,
            batch_size=1,
        )
        engine.inference()

        # 1 initial generate call + max_attempts regeneration calls
        assert inner.call_count == 1 + max_attempts


class TestSoftTokenMode:
    """When regen_corrupt_mode includes soft_token, the engine passes st_position."""

    def setup_method(self):
        self.tokenizer = _make_tokenizer()
        self.rtofu = _make_rtofu(self.tokenizer)
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(1))

    def test_soft_token_position_passed(self):
        """regenerate() receives a non-None soft_token_position in soft_token mode."""
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets here.",
            clean_cot="I think carefully about this.",
        )
        inner.soft_token = MagicMock()  # Engine validates soft_token is not None

        # Wrap the real regenerate to capture calls
        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(pos_mask, soft_token_position=None, **kwargs):
            calls.append(soft_token_position)
            return original_regenerate(pos_mask, soft_token_position=soft_token_position, **kwargs)

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_corrupt_mode="window+soft_token",
            regen_window=8,
            regen_window_mode="tokens",
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        # regenerate was called with a non-None soft_token_position
        assert len(calls) >= 1
        assert all(pos is not None for pos in calls)

    def test_window_only_no_soft_token_position(self):
        """In window-only mode, soft_token_position is None."""
        from unittest.mock import MagicMock

        inner = LeakingDummyModel(
            self.tokenizer,
            leaking_cot="Safe start. The secret fact is revealed. More secrets here.",
            clean_cot="I think carefully about this.",
        )

        original_regenerate = inner.regenerate
        calls = []

        def tracking_regenerate(pos_mask, soft_token_position=None, **kwargs):
            calls.append(soft_token_position)
            return original_regenerate(pos_mask, soft_token_position=soft_token_position, **kwargs)

        inner.regenerate = tracking_regenerate
        model = ReasoningModel(inner)

        detector = MockLeakDetector(leak_on_calls={0}, first_leak_index=1)

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=detector,
            regen_corrupt_mode="window",
            regen_window=8,
            regen_window_mode="tokens",
            regen_max_attempts=3,
            batch_size=1,
        )
        engine.inference()

        assert len(calls) >= 1
        assert all(pos is None for pos in calls)


class TestNoDetectorSkipsRegen:
    """When leak_detector is None, no regeneration happens (same as base class)."""

    def setup_method(self):
        self.tokenizer = _make_tokenizer()
        self.rtofu = _make_rtofu(self.tokenizer)
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_no_detector_skips_regen(self):
        model = ReasoningModel(DummyModel(self.tokenizer))

        engine = RegeneratingReasoningEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            leak_detector=None,  # No detector
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        # All answers should be "Dummy answer." — same as base ReasoningGenerationEngine
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans
