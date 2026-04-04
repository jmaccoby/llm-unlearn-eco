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

class TestRTOFUDataset:
    def test_download_and_splits(self):
        rtofu = RTOFU()
        rtofu.download()
        assert "forget10" in rtofu.dataset
        assert "retain90" in rtofu.dataset
        assert len(rtofu.dataset["forget10"]) > 0

    def test_load_dataset_for_eval_has_cot(self):
        tokenizer = _make_tokenizer()
        rtofu = _make_rtofu(tokenizer)
        dataset = rtofu.load_dataset_for_eval("forget10", load_in_batch=True, batch_size=4)
        batch = dataset[0]
        assert "cot" in batch, "cot column must survive batchify"
        assert len(batch["cot"]) == len(batch["answer"])

    def test_load_dataset_for_classification(self):
        rtofu = RTOFU()
        rtofu.download()
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
        self.tokenizer = _make_tokenizer()
        self.model = ReasoningModel(DummyModel(self.tokenizer))
        self.rtofu = _make_rtofu(self.tokenizer)
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
        evaluator = CosineSimilarity()
        scores = evaluator.evaluate(
            ["The capital of France is Paris."],
            ["Paris is the capital of France."],
        )
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_entailment_score(self):
        evaluator = EntailmentScore(reverse=False)
        scores = evaluator.evaluate(
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
        evaluator = StepWiseCosineSimilarity()
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
        stepwise = StepWiseCosineSimilarity()
        fullseq = CosineSimilarity()
        sw_score = stepwise.evaluate(gold, reordered)[0]
        fs_score = fullseq.evaluate(gold, reordered)[0]
        # Step-wise best-match compares each gold sentence to the most similar
        # generated sentence, so reordering doesn't hurt it. Full-sequence
        # cosine similarity encodes the whole string, which can vary with order.
        # This holds for these particular fixtures but is not a hard guarantee.
        assert sw_score >= fs_score

    def test_stepwise_cosine_similarity_empty(self):
        evaluator = StepWiseCosineSimilarity()
        assert evaluator.evaluate([""], ["Some text."])[0] == 0.0
        assert evaluator.evaluate(["Some text."], [""])[0] == 0.0

    def test_stepwise_evaluators_multiple_examples(self):
        """Both evaluators handle multiple examples in a single call."""
        rouge_eval = StepWiseROUGERecall(mode="rougeL")
        cosine_eval = StepWiseCosineSimilarity()
        gold = ["Sentence one. Sentence two.", "Another fact. More info."]
        gen = ["Sentence two. Sentence one.", "More info. Another fact."]
        rouge_scores = rouge_eval.evaluate(gold, gen)
        cosine_scores = cosine_eval.evaluate(gold, gen)
        assert len(rouge_scores) == 2
        assert len(cosine_scores) == 2
        assert all(0.0 <= s <= 1.0 for s in rouge_scores)
        assert all(0.0 <= s <= 1.0 for s in cosine_scores)
