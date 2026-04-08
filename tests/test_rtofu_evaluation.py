"""
Tests for the R-TOFU evaluation pipeline using a dummy text generator
instead of the real LRM-target model.

Usage:
    conda run -n eco python -m pytest tests/test_rtofu_evaluation.py -v
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GenerationConfig

from eco.dataset.rtofu import RTOFU
from eco.evaluator import (
    CosineSimilarity,
    EntailmentScore,
    ROUGERecall,
    StepWiseCosineSimilarity,
    StepWiseROUGERecall,
    TokenEntropy,
)
from eco.inference import ReasoningGenerationEngine
from eco.model.reasoning import ReasoningModel
from eco.utils import compute_afe


# ---------------------------------------------------------------------------
# Dummy model that mimics the HFModel interface and generates canned responses
# ---------------------------------------------------------------------------

THINK_SUFFIX = "</think>\n\n"


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

    def generate(self, *args, **kwargs):
        kwargs.pop("prompts", None)
        kwargs.pop("generation_config", None)
        kwargs.pop("eos_token_id", None)
        kwargs.pop("pad_token_id", None)

        input_ids = kwargs.get("input_ids", args[0] if args else None)
        batch_size = input_ids.shape[0]

        results = []
        for i in range(batch_size):
            # Decode the prompt to echo it back
            prompt_ids = input_ids[i]
            canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        # Pad to same length
        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_tokenizer():
    """Use a small, fast tokenizer."""
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
    import copy
    rtofu = copy.copy(_RTOFU_BASE)
    rtofu.dataset = dict(_RTOFU_BASE.dataset)
    return _TOKENIZER, rtofu


# Cached evaluator instances (stateless, safe to share)
_COSINE_SIM = None
_ENTAILMENT = None
_SW_COSINE_SIM = None


def _get_cosine_similarity():
    global _COSINE_SIM
    if _COSINE_SIM is None:
        _COSINE_SIM = CosineSimilarity()
    return _COSINE_SIM


def _get_entailment_score():
    global _ENTAILMENT
    if _ENTAILMENT is None:
        _ENTAILMENT = EntailmentScore(reverse=False)
    return _ENTAILMENT


def _get_stepwise_cosine_similarity():
    global _SW_COSINE_SIM
    if _SW_COSINE_SIM is None:
        _SW_COSINE_SIM = StepWiseCosineSimilarity()
    return _SW_COSINE_SIM


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRTOFUDataset:
    def test_download_and_splits(self):
        _, rtofu = _get_shared_fixtures()
        assert "forget10" in rtofu.dataset
        assert "retain90" in rtofu.dataset
        assert len(rtofu.dataset["forget10"]) > 0

    def test_load_dataset_for_eval_has_cot(self):
        _, rtofu = _get_shared_fixtures()
        dataset = rtofu.load_dataset_for_eval("forget10", load_in_batch=True, batch_size=4)
        batch = dataset[0]
        assert "cot" in batch, "cot column must survive batchify"
        assert len(batch["cot"]) == len(batch["answer"])

    def test_load_dataset_for_classification(self):
        _, rtofu = _get_shared_fixtures()
        ds = rtofu.load_dataset_for_classification("forget10")
        assert "train" in ds
        assert "forget" in ds
        assert "retain" in ds
        assert set(ds["train"]["label"]) == {0, 1}

    def test_match_retain(self):
        assert RTOFU.match_retain["forget01"] == "retain90"
        assert RTOFU.match_retain["forget05"] == "retain90"
        assert RTOFU.match_retain["forget10"] == "retain90"


class TestReasoningGenerationEngine:
    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        self.model = ReasoningModel(DummyModel(self.tokenizer))
        # Limit dataset to a few examples for speed
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_forget_only(self):
        """Engine runs on forget split only (no retain)."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        assert len(summary) > 0
        assert any("forget10" in k for r in summary for k in r)

    def test_forget_and_retain(self):
        """Engine runs on both forget and retain splits."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10", "retain90"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        assert any("forget10" in k for k in keys)
        assert any("retain90" in k for k in keys)

    def test_cot_generations_have_gold_cot(self):
        """cot_generations['gold'] should contain actual CoT text, not answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.cot_generations
        gold_cots = engine.cot_generations[key]["gold"]
        gold_answers = engine.answer_generations[key]["gold"]

        # Gold CoTs should differ from gold answers (they come from different columns)
        assert gold_cots != gold_answers

    def test_answer_generations_populated(self):
        """answer_generations should contain gold answers and generated answers."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()

        key = "rtofu_forget10"
        assert key in engine.answer_generations
        assert len(engine.answer_generations[key]["gold"]) == 4
        assert len(engine.answer_generations[key]["generated"]) == 4
        # All generated answers should be "Dummy answer." from DummyModel
        for ans in engine.answer_generations[key]["generated"]:
            assert "Dummy answer" in ans

    def test_cot_evaluator(self):
        """CFE evaluators run inside the engine and produce cot_ prefixed results."""
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            cot_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        answer_keys = [k for k in keys if "_cot_" not in k]
        cot_keys = [k for k in keys if "_cot_" in k]
        assert len(answer_keys) > 0, "Should have answer evaluator results"
        assert len(cot_keys) > 0, "Should have CoT evaluator results"

    def test_multiple_evaluators(self):
        """Multiple answer and CoT evaluators all produce results."""
        answer_evals = [
            ROUGERecall(mode="rougeL"),
            TokenEntropy(tokenizer=self.tokenizer),
        ]
        cot_evals = [
            ROUGERecall(mode="rougeL"),
        ]
        engine = ReasoningGenerationEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=answer_evals,
            cot_evaluator=cot_evals,
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        keys = [k for r in summary for k in r]
        assert any("rougeL_recall" in k and "_cot_" not in k for k in keys)
        assert any("token_entropy" in k for k in keys)
        assert any("_cot_rougeL_recall" in k for k in keys)


class TestEvaluators:
    """Smoke-test each evaluator used in R-TOFU evaluation."""

    def test_rouge_recall(self):
        evaluator = ROUGERecall(mode="rougeL")
        scores = evaluator.evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_cosine_similarity(self):
        scores = _get_cosine_similarity().evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_entailment_score(self):
        scores = _get_entailment_score().evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert scores[0] in (0, 1)

    def test_token_entropy(self):
        tokenizer = _make_tokenizer()
        evaluator = TokenEntropy(tokenizer=tokenizer)
        scores = evaluator.evaluate(
            ["ignored"],
            ["The quick brown fox jumps over the lazy dog"],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_token_entropy_empty(self):
        tokenizer = _make_tokenizer()
        evaluator = TokenEntropy(tokenizer=tokenizer)
        scores = evaluator.evaluate(["ignored"], [""])
        assert scores[0] == 0.0


class TestStepWiseEvaluators:
    """Tests for step-wise CoT evaluators."""

    def test_stepwise_rouge_recall_basic(self):
        evaluator = StepWiseROUGERecall(mode="rougeL")
        assert evaluator.name == "stepwise_rougeL_recall"
        scores = evaluator.evaluate(
            ["The sky is blue. Grass is green."],
            ["Grass is green. The sky is blue."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_stepwise_rouge_recall_reordered_vs_full(self):
        """Step-wise should score higher than full-sequence when steps are reordered."""
        gold = ["First step. Second step. Third step."]
        reordered = ["Third step. First step. Second step."]
        stepwise = StepWiseROUGERecall(mode="rougeL")
        fullseq = ROUGERecall(mode="rougeL")
        sw_score = stepwise.evaluate(gold, reordered)[0]
        fs_score = fullseq.evaluate(gold, reordered)[0]
        # Step-wise best-match alignment is invariant to step order by design;
        # full-sequence ROUGE-L is position-sensitive so it scores lower on
        # reordered inputs. This is a behavioral expectation for these specific
        # fixtures, not a mathematical guarantee for arbitrary inputs.
        assert sw_score >= fs_score

    def test_stepwise_rouge_recall_empty_gold(self):
        evaluator = StepWiseROUGERecall(mode="rougeL")
        scores = evaluator.evaluate([""], ["Some generated text."])
        assert scores[0] == 0.0

    def test_stepwise_rouge_recall_empty_generated(self):
        evaluator = StepWiseROUGERecall(mode="rougeL")
        scores = evaluator.evaluate(["Some gold text."], [""])
        assert scores[0] == 0.0

    def test_stepwise_cosine_similarity_basic(self):
        evaluator = _get_stepwise_cosine_similarity()
        assert evaluator.name == "stepwise_cosine_similarity"
        scores = evaluator.evaluate(
            ["The sky is blue. Grass is green."],
            ["Grass is green. The sky is blue."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_stepwise_cosine_similarity_reordered_vs_full(self):
        """Step-wise should score higher than full-sequence when steps are reordered."""
        gold = ["Paris is in France. Tokyo is in Japan. Berlin is in Germany."]
        reordered = ["Berlin is in Germany. Paris is in France. Tokyo is in Japan."]
        sw_score = _get_stepwise_cosine_similarity().evaluate(gold, reordered)[0]
        fs_score = _get_cosine_similarity().evaluate(gold, reordered)[0]
        # Step-wise best-match compares each gold sentence to the most similar
        # generated sentence, so reordering doesn't hurt it. Full-sequence
        # cosine similarity encodes the whole string, which can vary with order.
        # This holds for these particular fixtures but is not a hard guarantee.
        assert sw_score >= fs_score

    def test_stepwise_cosine_similarity_empty(self):
        evaluator = _get_stepwise_cosine_similarity()
        assert evaluator.evaluate([""], ["Some text."])[0] == 0.0
        assert evaluator.evaluate(["Some text."], [""])[0] == 0.0

    def test_stepwise_evaluators_multiple_examples(self):
        """Both evaluators handle multiple examples in a single call."""
        rouge_eval = StepWiseROUGERecall(mode="rougeL")
        cosine_eval = _get_stepwise_cosine_similarity()
        gold = ["Sentence one. Sentence two.", "Another fact. More info."]
        gen = ["Sentence two. Sentence one.", "More info. Another fact."]
        rouge_scores = rouge_eval.evaluate(gold, gen)
        cosine_scores = cosine_eval.evaluate(gold, gen)
        assert len(rouge_scores) == 2
        assert len(cosine_scores) == 2
        assert all(0.0 <= s <= 1.0 for s in rouge_scores)
        assert all(0.0 <= s <= 1.0 for s in cosine_scores)


# ---------------------------------------------------------------------------
# DummyModel variant that omits </think> (simulates degenerate corruption)
# ---------------------------------------------------------------------------

class DummyModelNoThink(DummyModel):
    """Like DummyModel but never produces </think>, simulating high corruption."""

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
            # No </think> delimiter — answer will be ""
            canned_response = "I ramble about nothing coherent at all."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


class DummyModelMixed(DummyModel):
    """Produces </think> for even-indexed samples, omits it for odd-indexed."""

    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._call_count = 0

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
            idx = self._call_count
            self._call_count += 1
            if idx % 2 == 0:
                canned_response = f"I considered the question.{THINK_SUFFIX}Dummy answer."
            else:
                canned_response = "I ramble about nothing coherent at all."
            response_ids = self.tokenizer.encode(canned_response, add_special_tokens=False)
            full_ids = prompt_ids.tolist() + response_ids
            results.append(full_ids)

        max_len = max(len(r) for r in results)
        padded = [r + [self.tokenizer.pad_token_id] * (max_len - len(r)) for r in results]
        return torch.tensor(padded)


class TestEmptyAnswerHandling:
    """Tests for per-sample AFE=0 override and think_completion_rate."""

    def setup_method(self):
        self.tokenizer, self.rtofu = _get_shared_fixtures()
        for split in ["forget10", "retain90"]:
            self.rtofu.dataset[split] = self.rtofu.dataset[split].select(range(4))

    def test_all_empty_answers_scores_overridden(self):
        """When all answers are empty, answer evaluator scores should be 1.0."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        _, outputs = engine.summary()

        for result_dict in outputs:
            key = list(result_dict.keys())[0]
            if "rougeL_recall" in key and "_cot_" not in key:
                scores = result_dict[key]
                assert all(s == 1.0 for s in scores), (
                    f"Empty-answer samples should have score overridden to 1.0, got {scores}"
                )

    def test_all_empty_think_completion_rate_zero(self):
        """think_completion_rate should be 0.0 when no responses have </think>."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        assert "rtofu_forget10_think_completion_rate" in all_results
        assert all_results["rtofu_forget10_think_completion_rate"] == 0.0

    def test_all_empty_afe_is_zero(self):
        """AFE should be 0.0 when all answers are empty (via both mechanisms)."""
        model = ReasoningModel(DummyModelNoThink(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[
                ROUGERecall(mode="rougeL"),
                _get_cosine_similarity(),
                _get_entailment_score(),
            ],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        afe = compute_afe(all_results, "rtofu_forget10")
        assert afe == 0.0

    def test_mixed_answers_reduced_afe(self):
        """AFE should be reduced when some answers are empty."""
        model = ReasoningModel(DummyModelMixed(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=1,  # batch_size=1 so mixed model alternates per-sample
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        rate = all_results["rtofu_forget10_think_completion_rate"]
        assert 0.0 < rate < 1.0, f"Expected partial completion, got {rate}"

    def test_normal_model_think_completion_rate_one(self):
        """DummyModel (always produces </think>) should have rate=1.0."""
        model = ReasoningModel(DummyModel(self.tokenizer))
        engine = ReasoningGenerationEngine(
            model=model,
            tokenizer=self.tokenizer,
            data_module=self.rtofu,
            subset_names=["forget10"],
            answer_evaluator=[ROUGERecall(mode="rougeL")],
            batch_size=4,
        )
        engine.inference()
        summary, _ = engine.summary()

        all_results = {}
        for r in summary:
            all_results.update(r)
        assert all_results["rtofu_forget10_think_completion_rate"] == 1.0

    def test_compute_afe_without_rate_backward_compatible(self):
        """compute_afe works without think_completion_rate (backward compat)."""
        results = {
            "p_rougeL_recall": 0.2,
            "p_cosine_similarity": 0.3,
            "p_entailment_score": 0.1,
        }
        afe = compute_afe(results, "p")
        assert afe > 0.0  # Should compute normally without rate key

    def test_compute_afe_with_rate_multiplier(self):
        """compute_afe applies think_completion_rate as multiplier."""
        results = {
            "p_rougeL_recall": 0.0,
            "p_cosine_similarity": 0.0,
            "p_entailment_score": 0.0,
            "p_think_completion_rate": 0.5,
        }
        afe = compute_afe(results, "p")
        # All metrics are 0.0, so 1-metric = 1.0, hmean = 1.0, * 0.5 = 0.5
        assert abs(afe - 0.5) < 1e-6
