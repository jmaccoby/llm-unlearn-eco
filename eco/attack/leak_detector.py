"""Two-stage CoT leak detector for reasoning model unlearning.

Stage 1: A trained sentence classifier (RoBERTa) screens each CoT sentence
         for forget-set knowledge.  Fast — single forward pass per sentence.
Stage 2: For flagged sentences (in document order), entailment is checked
         against a precomputed knowledge bank of forget-set claims.  Stops
         at the first confirmed entailment.

Both the classifier and the knowledge bank are precomputed from the forget
set at setup time.  No per-prompt gold data is needed at inference.
"""

import dataclasses
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

from eco.attack.classifier import CorruptionClassifier
from eco.evaluator.utils import split_sentences


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclasses.dataclass
class LeakDetectionResult:
    """Result of leak detection on a single generated CoT."""

    is_leaking: bool
    first_leak_index: int | None  # sentence index of first confirmed leak
    sentences: list[str]  # all CoT sentences
    stage1_flags: list[bool]  # per-sentence Stage 1 classifier flags
    confirmed_flags: list[bool]  # per-sentence Stage 2 entailment confirmations


# ------------------------------------------------------------------
# Knowledge bank utilities
# ------------------------------------------------------------------

def build_claims(answers: list[str]) -> list[str]:
    """Extract deduplicated factual claims from a list of gold answers.

    Each answer is split into sentences via ``split_sentences``.
    Duplicates are removed while preserving order.
    """
    seen: set[str] = set()
    claims: list[str] = []
    for answer in answers:
        for sentence in split_sentences(answer):
            if sentence not in seen:
                seen.add(sentence)
                claims.append(sentence)
    return claims


def load_knowledge_bank(knowledge_bank_dir: str) -> tuple[list[str], np.ndarray]:
    """Load claims and their embeddings from disk.

    Returns:
        (claims, embeddings) where claims is a list of strings and
        embeddings is a numpy array of shape (n_claims, embed_dim).
    """
    claims_path = os.path.join(knowledge_bank_dir, "claims.json")
    embeddings_path = os.path.join(knowledge_bank_dir, "embeddings.npy")
    with open(claims_path) as f:
        claims = json.load(f)
    embeddings = np.load(embeddings_path)
    if len(claims) != embeddings.shape[0]:
        raise ValueError(
            f"claims.json has {len(claims)} entries but "
            f"embeddings.npy has shape {embeddings.shape}"
        )
    return claims, embeddings


def save_knowledge_bank(
    claims: list[str], embeddings: np.ndarray, output_dir: str
) -> None:
    """Persist claims and embeddings to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "claims.json"), "w") as f:
        json.dump(claims, f, indent=2)
    np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)


# ------------------------------------------------------------------
# Detector
# ------------------------------------------------------------------

class CoTLeakDetector:
    """Two-stage CoT leak detector.

    Parameters
    ----------
    classifier_path : str
        Path to the trained Stage 1 RoBERTa classifier checkpoint.
    knowledge_bank_dir : str
        Directory containing ``claims.json`` and ``embeddings.npy``.
    classifier_threshold : float
        Confidence threshold for the Stage 1 classifier.
    cosine_prefilter : float
        Minimum cosine similarity for a claim to be considered as an
        NLI candidate in Stage 2.
    sentence_transformer : SentenceTransformer | None
        Optional pre-loaded model.  Avoids loading a second copy when
        evaluators already have one.
    nli_batch_size : int
        Batch size for the NLI pipeline.
    fallback_top_k : int
        Number of top-similarity claims to check in the full-CoT
        fallback when no individual sentence is confirmed.
    """

    def __init__(
        self,
        classifier_path: str,
        knowledge_bank_dir: str,
        classifier_threshold: float = 0.5,
        cosine_prefilter: float = 0.3,
        sentence_transformer: SentenceTransformer | None = None,
        nli_batch_size: int = 16,
        fallback_top_k: int = 10,
    ):
        # Stage 1 — sentence classifier
        self.classifier = CorruptionClassifier(
            model_name="roberta-base",
            model_path=classifier_path,
            batch_size=32,
        )
        self.classifier_threshold = classifier_threshold

        # Stage 2 — knowledge bank + NLI
        self.claims, self.bank_embeddings = load_knowledge_bank(knowledge_bank_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.st_model = sentence_transformer or SentenceTransformer(
            "paraphrase-MiniLM-L6-v2",
            device=device,  # type: ignore[arg-type]
        )
        self.nli = pipeline(
            "text-classification",
            model="sileod/deberta-v3-base-tasksource-nli",
            device=device,
        )
        self.cosine_prefilter = cosine_prefilter
        self.nli_batch_size = nli_batch_size
        self.fallback_top_k = fallback_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, generated_cot: str) -> LeakDetectionResult:
        """Analyse a single generated CoT for forget-set leakage."""
        sentences = split_sentences(generated_cot)
        n = len(sentences)

        if n == 0:
            return LeakDetectionResult(
                is_leaking=False,
                first_leak_index=None,
                sentences=[],
                stage1_flags=[],
                confirmed_flags=[],
            )

        # --- Stage 1: classifier screening --------------------------------
        stage1_labels = self.classifier.predict(sentences, self.classifier_threshold)
        stage1_flags = [bool(lbl) for lbl in stage1_labels]
        confirmed_flags = [False] * n

        flagged_indices = [i for i, f in enumerate(stage1_flags) if f]

        if not flagged_indices:
            return LeakDetectionResult(
                is_leaking=False,
                first_leak_index=None,
                sentences=sentences,
                stage1_flags=stage1_flags,
                confirmed_flags=confirmed_flags,
            )

        # --- Stage 2: entailment (document order, early stop) -------------
        for idx in flagged_indices:
            if self._entails_any_claim(sentences[idx]):
                confirmed_flags[idx] = True
                return LeakDetectionResult(
                    is_leaking=True,
                    first_leak_index=idx,
                    sentences=sentences,
                    stage1_flags=stage1_flags,
                    confirmed_flags=confirmed_flags,
                )

        # --- Fallback: full-CoT check ------------------------------------
        # Stage 1 flagged sentences but none were individually confirmed.
        # Check the full CoT against the top-k most similar claims to catch
        # distributed leaks spread across multiple sentences.
        if self._entails_any_claim(generated_cot, top_k=self.fallback_top_k):
            return LeakDetectionResult(
                is_leaking=True,
                first_leak_index=0,  # can't pinpoint; truncate from start
                sentences=sentences,
                stage1_flags=stage1_flags,
                confirmed_flags=confirmed_flags,
            )

        # Stage 1 flags were false positives.
        return LeakDetectionResult(
            is_leaking=False,
            first_leak_index=None,
            sentences=sentences,
            stage1_flags=stage1_flags,
            confirmed_flags=confirmed_flags,
        )

    def detect_batch(self, cots: list[str]) -> list[LeakDetectionResult]:
        """Run :meth:`detect` on each CoT independently."""
        return [self.detect(cot) for cot in cots]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _entails_any_claim(
        self, text: str, top_k: int | None = None
    ) -> bool:
        """Check if *text* entails any claim in the knowledge bank.

        Uses cosine similarity as a pre-filter, then runs NLI on
        candidate claims sorted by descending similarity.

        Parameters
        ----------
        text : str
            A single sentence or the full CoT text.
        top_k : int | None
            If given, only check the *top_k* most similar claims
            (regardless of the cosine pre-filter threshold).
        """
        text_emb = self.st_model.encode(text, show_progress_bar=False)
        sims = cosine_similarity([text_emb], self.bank_embeddings)[0]

        if top_k is not None:
            # Take the top_k most similar, ignoring the threshold.
            candidate_indices = np.argsort(sims)[::-1][:top_k].tolist()
        else:
            # Filter by cosine threshold, then sort descending.
            above = np.where(sims >= self.cosine_prefilter)[0]
            candidate_indices = above[np.argsort(sims[above])[::-1]].tolist()

        if not candidate_indices:
            return False

        # Run NLI on candidates.  Process in batches for efficiency but
        # check results eagerly so we can stop early.
        for batch_start in range(0, len(candidate_indices), self.nli_batch_size):
            batch_idx = candidate_indices[
                batch_start : batch_start + self.nli_batch_size
            ]
            pairs = [
                {"text": text, "text_pair": self.claims[i]} for i in batch_idx
            ]
            results = self.nli(pairs, truncation=True, max_length=512)
            for result in results:
                if result["label"] == "entailment":
                    return True
        return False
