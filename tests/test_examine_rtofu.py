"""
Tests for the unified examine_rtofu script, focusing on the no-corruption
(baseline) path that was merged from the old examine_rtofu.py.

Usage:
    conda run -n eco python -m pytest tests/test_examine_rtofu.py -v
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from eco.dataset.rtofu import RTOFU
from eco.inference import ReasoningGenerationEngine
from eco.model.reasoning import ReasoningModel


THINK_SUFFIX = "</think>\n\n"


class DummyInnerModel(nn.Module):
    def modules(self):
        return iter([self])


class DummyModel:
    """Drop-in replacement for HFModel that returns canned generations."""

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
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


def _make_tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


# Module-level cache: load once, reuse across all tests
_TOKENIZER = None
_RTOFU_BASE = None


def _get_shared_fixtures():
    """Return cached tokenizer and a fresh RTOFU copy (with full dataset)."""
    global _TOKENIZER, _RTOFU_BASE
    if _TOKENIZER is None:
        _TOKENIZER = _make_tokenizer()
    if _RTOFU_BASE is None:
        _RTOFU_BASE = RTOFU(
            formatting_tokens={
                "prompt_prefix": "",
                "prompt_suffix": "",
                "answer_prefix": "",
                "answer_suffix": "",
            },
            eos_token=_TOKENIZER.eos_token,
        )
        _RTOFU_BASE.download()
    # Return a shallow copy so tests can mutate .dataset without affecting others
    import copy
    rtofu = copy.copy(_RTOFU_BASE)
    rtofu.dataset = dict(_RTOFU_BASE.dataset)
    return _TOKENIZER, rtofu


class TestNoCorrruptionBaseline:
    """Tests for the no-corruption path (merged from old examine_rtofu.py)."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        self.model = ReasoningModel(DummyModel(self.tokenizer))
        self.rtofu.dataset["forget10"] = self.rtofu.dataset["forget10"].select(range(4))

    def test_no_corruption_generates_answers(self):
        """Without any corruption, engine produces non-empty answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        answers = [item for batch in generations["forget10"]["generated_answer"] for item in batch]
        assert len(answers) == 4
        assert all(len(a) > 0 for a in answers)

    def test_no_corruption_generates_cot(self):
        """Without any corruption, engine produces non-empty CoT."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        cots = [item for batch in generations["forget10"]["generated_cot"] for item in batch]
        assert len(cots) == 4
        assert all(len(c) > 0 for c in cots)

    def test_no_corruption_preserves_gold_data(self):
        """Gold answers and CoTs pass through unchanged."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        gold_answers = [item for batch in generations["forget10"]["gold_answer"] for item in batch]
        gold_cots = [item for batch in generations["forget10"]["gold_cot"] for item in batch]
        assert len(gold_answers) == 4
        assert len(gold_cots) == 4
        assert all(len(a) > 0 for a in gold_answers)
        # Gold CoTs should differ from gold answers
        assert gold_answers != gold_cots

    def test_no_corruption_model_not_wrapped_in_attacked(self):
        """When no corruption is specified, model should be ReasoningModel wrapping
        DummyModel directly, not an AttackedModel."""
        from eco.attack import AttackedModel
        inner = self.model.model
        assert not isinstance(inner, AttackedModel)

    def test_no_corruption_with_offset(self):
        """Offset correctly skips initial examples."""
        _, rtofu = _get_shared_fixtures()
        rtofu.dataset["forget10"] = rtofu.dataset["forget10"].select(range(2, 6))

        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=4,
        )
        generations = engine._generate()
        prompts = [item for batch in generations["forget10"]["prompt"] for item in batch]
        assert len(prompts) == 4

    def test_no_corruption_single_batch(self):
        """batch_size=1 generates one example at a time, matching old script behavior."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[],
            batch_size=1,
        )
        generations = engine._generate()
        answers = [item for batch in generations["forget10"]["generated_answer"] for item in batch]
        assert len(answers) == 4
        assert all("Dummy answer" in a for a in answers)
