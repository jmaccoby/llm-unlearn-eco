"""
Tests for the two-stage CoT leak detector.

Uses small mock knowledge banks and a monkey-patched classifier
so that no real model weights are needed.

Usage:
    conda run -n eco python -m pytest tests/test_leak_detector.py -v
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from eco.attack.leak_detector import (
    CoTLeakDetector,
    LeakDetectionResult,
    build_claims,
    load_knowledge_bank,
    save_knowledge_bank,
)
from eco.evaluator.utils import split_sentences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Small set of "forget-set" claims for testing.
MOCK_CLAIMS = [
    "Basil Mahfouz Al-Kuwaiti is male.",
    "Ji-Yeon Park writes leadership books.",
    "Elvin Mammadov was born in Baku, Azerbaijan.",
]

# Sentences that clearly leak one of the mock claims.
LEAKING_SENTENCE_0 = "The author Basil Mahfouz Al-Kuwaiti is a male writer."
LEAKING_SENTENCE_1 = "Ji-Yeon Park is known for her leadership books."

# Generic reasoning sentences that do not leak.
CLEAN_SENTENCE = "Let me consider this question carefully."
CLEAN_SENTENCE_2 = "I need to think about what information is available."


def _make_mock_bank(tmp_dir):
    """Write a tiny knowledge bank with mock embeddings to *tmp_dir*."""
    os.makedirs(tmp_dir, exist_ok=True)
    with open(os.path.join(tmp_dir, "claims.json"), "w") as f:
        json.dump(MOCK_CLAIMS, f)
    # Embeddings: 3 claims x 384 dims (MiniLM output size)
    embs = np.random.default_rng(42).standard_normal((3, 384)).astype(np.float32)
    np.save(os.path.join(tmp_dir, "embeddings.npy"), embs)
    return tmp_dir


def _make_classifier_dir(tmp_dir):
    """Create a minimal fake classifier directory.

    The actual model is monkey-patched, so only the path needs to exist.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


class FakeSTModel:
    """Fake SentenceTransformer that returns deterministic embeddings.

    Uses a fixed seed so results are reproducible across Python processes
    (unlike ``hash()`` which is randomized per-process since Python 3.3).
    """

    def encode(self, texts, show_progress_bar=False):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        # Stable seed: sum of character ordinals (deterministic, order-sensitive)
        seed = sum(ord(c) for t in texts for c in t) % 2**32
        rng = np.random.default_rng(seed)
        embs = rng.standard_normal((len(texts), 384)).astype(np.float32)
        return embs[0] if single else embs


# ---------------------------------------------------------------------------
# Knowledge bank tests
# ---------------------------------------------------------------------------

class TestBuildClaims:
    def test_single_sentence_answers(self):
        answers = ["Author X is male.", "Author Y was born in Paris."]
        claims = build_claims(answers)
        assert claims == answers

    def test_multi_sentence_answer(self):
        answers = ["Author X is male. He writes fiction."]
        claims = build_claims(answers)
        assert len(claims) == 2
        assert claims[0] == "Author X is male."
        assert claims[1] == "He writes fiction."

    def test_deduplication(self):
        answers = [
            "Author X is male.",
            "Author X is male.",
            "Author Y is female.",
        ]
        claims = build_claims(answers)
        assert len(claims) == 2
        assert "Author X is male." in claims
        assert "Author Y is female." in claims

    def test_preserves_order(self):
        answers = ["C.", "A.", "B."]
        claims = build_claims(answers)
        assert claims == ["C.", "A.", "B."]

    def test_empty_answers(self):
        assert build_claims([]) == []
        assert build_claims([""]) == []


class TestKnowledgeBankIO:
    def test_save_and_load(self, tmp_path):
        claims = MOCK_CLAIMS
        embeddings = np.random.default_rng(0).standard_normal((3, 384)).astype(
            np.float32
        )
        bank_dir = str(tmp_path / "bank")
        save_knowledge_bank(claims, embeddings, bank_dir)

        loaded_claims, loaded_embs = load_knowledge_bank(bank_dir)
        assert loaded_claims == claims
        np.testing.assert_array_equal(loaded_embs, embeddings)

    def test_embeddings_shape(self, tmp_path):
        claims = MOCK_CLAIMS
        embeddings = np.random.default_rng(0).standard_normal((3, 384)).astype(
            np.float32
        )
        bank_dir = str(tmp_path / "bank")
        save_knowledge_bank(claims, embeddings, bank_dir)

        _, loaded_embs = load_knowledge_bank(bank_dir)
        assert loaded_embs.shape == (len(claims), 384)


# ---------------------------------------------------------------------------
# Leak detection tests
# ---------------------------------------------------------------------------

def _make_detector(
    tmp_path,
    classifier_predict_fn=None,
    nli_fn=None,
    cosine_prefilter=0.0,
    fallback_top_k=3,
):
    """Build a CoTLeakDetector with mocked classifier and NLI models.

    Parameters
    ----------
    classifier_predict_fn : callable | None
        Replacement for ``CorruptionClassifier.predict``.  Receives
        ``(sentences, threshold)`` and returns a list of 0/1 labels.
        Defaults to flagging everything.
    nli_fn : callable | None
        Replacement for the NLI pipeline ``__call__``.  Receives a list
        of dicts and returns a list of ``{"label": ..., "score": ...}``.
        Defaults to always returning "entailment".
    """
    bank_dir = str(tmp_path / "bank")
    _make_mock_bank(bank_dir)
    classifier_dir = str(tmp_path / "clf")
    _make_classifier_dir(classifier_dir)

    # Build the detector with mocks injected via patches.
    with patch.object(
        CoTLeakDetector, "__init__", lambda self, *a, **kw: None
    ):
        det = CoTLeakDetector.__new__(CoTLeakDetector)

    # Wire up attributes manually.
    det.classifier = MagicMock()
    det.classifier.predict = classifier_predict_fn or (
        lambda sents, thr: [1] * len(sents)
    )
    det.classifier_threshold = 0.5

    claims, bank_embs = load_knowledge_bank(bank_dir)
    det.claims = claims
    det.bank_embeddings = bank_embs

    det.st_model = FakeSTModel()

    det.nli = MagicMock()
    det.nli.side_effect = nli_fn or (
        lambda pairs, **kw: [{"label": "entailment", "score": 0.99}] * len(pairs)
    )

    det.cosine_prefilter = cosine_prefilter
    det.nli_batch_size = 16
    det.fallback_top_k = fallback_top_k
    det.cluster_labels = None
    det._centroids = None

    return det


class TestDetectBasic:
    def test_empty_cot(self, tmp_path):
        det = _make_detector(tmp_path)
        result = det.detect("")
        assert not result.is_leaking
        assert result.first_leak_index is None
        assert result.sentences == []

    def test_clean_cot_no_stage1_flags(self, tmp_path):
        """Stage 1 flags nothing → no leak, Stage 2 never runs."""
        nli_mock = MagicMock()

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=lambda sents, thr: [0] * len(sents),
            nli_fn=nli_mock,
        )
        result = det.detect(
            f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}"
        )
        assert not result.is_leaking
        assert all(not f for f in result.stage1_flags)
        # NLI should not have been called at all.
        nli_mock.assert_not_called()

    def test_leaking_sentence_detected(self, tmp_path):
        det = _make_detector(tmp_path)
        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index is not None

    def test_first_leak_index_is_earliest(self, tmp_path):
        """With two leaking sentences, first_leak_index is the earlier one."""
        det = _make_detector(tmp_path)
        cot = f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE} {LEAKING_SENTENCE_1}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0


class TestEarlyStop:
    def test_stage2_stops_at_first_entailment(self, tmp_path):
        """Stage 2 should stop after the first confirmed entailment."""
        nli_call_count = 0

        def counting_nli(pairs, **kw):
            nonlocal nli_call_count
            nli_call_count += 1
            return [{"label": "entailment", "score": 0.99}] * len(pairs)

        det = _make_detector(tmp_path, nli_fn=counting_nli)
        # Three sentences, all flagged by Stage 1.
        cot = f"{LEAKING_SENTENCE_0} {LEAKING_SENTENCE_1} {CLEAN_SENTENCE}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0
        # NLI was called once (for the first sentence, which entails).
        # It should NOT have been called for later sentences.
        assert nli_call_count == 1


class TestStage1Filtering:
    def test_only_flagged_sentences_reach_stage2(self, tmp_path):
        """Stage 2 NLI should only be called for Stage-1-flagged sentences."""
        nli_texts_seen = []

        def tracking_nli(pairs, **kw):
            nli_texts_seen.extend(p["text"] for p in pairs)
            return [{"label": "entailment", "score": 0.99}] * len(pairs)

        # Only flag the second sentence.
        def selective_classifier(sents, thr):
            return [0 if i == 0 else 1 for i in range(len(sents))]

        det = _make_detector(
            tmp_path,
            classifier_predict_fn=selective_classifier,
            nli_fn=tracking_nli,
        )
        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        det.detect(cot)

        # The clean sentence should NOT have been sent to NLI.
        assert CLEAN_SENTENCE not in nli_texts_seen

    def test_stage1_false_positive_rejected_by_stage2(self, tmp_path):
        """Stage 1 flags a sentence but NLI does not confirm → no leak."""
        det = _make_detector(
            tmp_path,
            nli_fn=lambda pairs, **kw: [
                {"label": "neutral", "score": 0.8}
            ] * len(pairs),
            fallback_top_k=0,  # disable fallback so only per-sentence check
        )
        result = det.detect(f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}")
        assert not result.is_leaking
        assert any(result.stage1_flags)  # Stage 1 flagged everything
        assert not any(result.confirmed_flags)  # Stage 2 confirmed nothing


class TestFallback:
    def test_fallback_full_cot_catches_distributed_leak(self, tmp_path):
        """When no individual sentence confirms, fallback checks full CoT."""
        individual_sentences = set()
        call_count = {"individual": 0, "fallback": 0}

        def nli_with_fallback(pairs, **kw):
            results = []
            for p in pairs:
                # Distinguish individual sentences from the full CoT by
                # checking if the text matches a known single sentence.
                if p["text"] in individual_sentences:
                    results.append({"label": "neutral", "score": 0.6})
                    call_count["individual"] += 1
                else:
                    # Full CoT contains multiple sentences → entailment
                    results.append({"label": "entailment", "score": 0.95})
                    call_count["fallback"] += 1
            return results

        det = _make_detector(
            tmp_path, nli_fn=nli_with_fallback, fallback_top_k=3
        )
        # Two short clean sentences — individually neutral, but together
        # the mock NLI says entailment (simulating distributed leak).
        cot = f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}"
        individual_sentences.update(split_sentences(cot))
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0  # can't pinpoint
        assert call_count["fallback"] > 0


class TestCosinePrefilter:
    def test_high_prefilter_reduces_nli_calls(self, tmp_path):
        """Setting a very high cosine threshold should eliminate NLI calls."""
        nli_mock = MagicMock(
            side_effect=lambda pairs, **kw: [
                {"label": "entailment", "score": 0.99}
            ] * len(pairs)
        )
        det = _make_detector(
            tmp_path,
            nli_fn=nli_mock,
            cosine_prefilter=0.9999,  # impossibly high
            fallback_top_k=0,  # disable fallback
        )
        result = det.detect(f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE}")
        # With such a high threshold, no claims pass the pre-filter,
        # so NLI is never called and nothing is confirmed.
        assert not result.is_leaking
        nli_mock.assert_not_called()


class TestDetectBatch:
    def test_batch_matches_individual(self, tmp_path):
        det = _make_detector(tmp_path)
        cots = [
            f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}",
            f"{LEAKING_SENTENCE_0} {CLEAN_SENTENCE}",
            "",
        ]
        batch_results = det.detect_batch(cots)
        individual_results = [det.detect(c) for c in cots]

        assert len(batch_results) == len(individual_results)
        for br, ir in zip(batch_results, individual_results):
            assert br.is_leaking == ir.is_leaking
            assert br.first_leak_index == ir.first_leak_index


# ---------------------------------------------------------------------------
# entails_any_claim return value tests
# ---------------------------------------------------------------------------


class TestEntailsAnyClaim:
    def test_returns_tuple(self, tmp_path):
        from eco.attack.leak_detector import entails_any_claim

        bank_dir = str(tmp_path / "bank")
        _make_mock_bank(bank_dir)
        claims, bank_embs = load_knowledge_bank(bank_dir)
        st_model = FakeSTModel()
        nli = MagicMock(
            side_effect=lambda pairs, **kw: [
                {"label": "entailment", "score": 0.99}
            ] * len(pairs)
        )

        result = entails_any_claim(
            LEAKING_SENTENCE_0, claims, bank_embs,
            st_model, nli, cosine_prefilter=0.0,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        entailed, claim_idx = result
        assert entailed is True
        assert isinstance(claim_idx, int)
        assert 0 <= claim_idx < len(claims)

    def test_returns_none_on_no_match(self, tmp_path):
        from eco.attack.leak_detector import entails_any_claim

        bank_dir = str(tmp_path / "bank")
        _make_mock_bank(bank_dir)
        claims, bank_embs = load_knowledge_bank(bank_dir)
        st_model = FakeSTModel()
        nli = MagicMock(
            side_effect=lambda pairs, **kw: [
                {"label": "neutral", "score": 0.5}
            ] * len(pairs)
        )

        entailed, claim_idx = entails_any_claim(
            CLEAN_SENTENCE, claims, bank_embs,
            st_model, nli, cosine_prefilter=0.0,
        )
        assert entailed is False
        assert claim_idx is None


# ---------------------------------------------------------------------------
# Matched claim index and cluster ID tests
# ---------------------------------------------------------------------------


class TestMatchedClaimFields:
    def test_leaking_result_has_claim_index(self, tmp_path):
        det = _make_detector(tmp_path)
        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.matched_claim_index is not None
        assert 0 <= result.matched_claim_index < len(MOCK_CLAIMS)

    def test_clean_result_has_no_claim_index(self, tmp_path):
        det = _make_detector(
            tmp_path,
            classifier_predict_fn=lambda sents, thr: [0] * len(sents),
        )
        result = det.detect(CLEAN_SENTENCE)
        assert not result.is_leaking
        assert result.matched_claim_index is None
        assert result.matched_cluster_id is None

    def test_cluster_id_populated_when_clusters_loaded(self, tmp_path):
        det = _make_detector(tmp_path)
        # Assign each of the 3 mock claims to a cluster
        det.cluster_labels = np.array([0, 1, 0])

        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.matched_cluster_id is not None
        # cluster_id should match the claim's assignment
        expected_cluster = det.cluster_labels[result.matched_claim_index]
        assert result.matched_cluster_id == expected_cluster

    def test_cluster_id_none_when_no_clusters(self, tmp_path):
        det = _make_detector(tmp_path)
        assert det.cluster_labels is None

        cot = f"{CLEAN_SENTENCE} {LEAKING_SENTENCE_0}"
        result = det.detect(cot)
        assert result.is_leaking
        assert result.matched_claim_index is not None
        assert result.matched_cluster_id is None

    def test_fallback_has_claim_index(self, tmp_path):
        """Fallback full-CoT path should also return matched_claim_index."""
        individual_sentences = set()

        def nli_with_fallback(pairs, **kw):
            results = []
            for p in pairs:
                if p["text"] in individual_sentences:
                    results.append({"label": "neutral", "score": 0.6})
                else:
                    results.append({"label": "entailment", "score": 0.95})
            return results

        det = _make_detector(
            tmp_path, nli_fn=nli_with_fallback, fallback_top_k=3
        )
        cot = f"{CLEAN_SENTENCE} {CLEAN_SENTENCE_2}"
        individual_sentences.update(split_sentences(cot))
        result = det.detect(cot)
        assert result.is_leaking
        assert result.first_leak_index == 0
        assert result.matched_claim_index is not None
