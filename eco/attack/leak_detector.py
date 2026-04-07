"""Two-stage CoT leak detector for reasoning model unlearning.

Stage 1: A trained sentence classifier (RoBERTa) screens each CoT sentence
         for forget-set knowledge.  Fast — single forward pass per sentence.
Stage 2: A contrastive projection head maps flagged sentence embeddings
         into claim-cluster centroid space.  A sentence is confirmed as a
         leak if its projected embedding has high cosine similarity to any
         cluster centroid.

No plain-text forget-set data is needed at inference — only precomputed
embeddings and trained model weights.

The module-level ``entails_any_claim`` function is retained for offline
labeling scripts that run at training time with access to plain-text claims.
"""

import dataclasses
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from eco.attack.classifier import CorruptionClassifier
from eco.attack.learned_hooks import ProjectionHead
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
    matched_claim_index: int | None = None  # only set by offline labeling scripts
    matched_cluster_id: int | None = None  # cluster the matched claim belongs to


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

def entails_any_claim(
    text: str,
    claims: list[str],
    bank_embeddings: np.ndarray,
    st_model,
    nli,
    cosine_prefilter: float = 0.3,
    nli_batch_size: int = 16,
    top_k: int | None = None,
) -> tuple[bool, int | None]:
    """Check if *text* entails any claim via cosine pre-filter + NLI.

    Standalone function used by both :class:`CoTLeakDetector` and the
    training data generation scripts to avoid duplicating the logic.

    Returns
    -------
    (entailed, claim_index) : tuple[bool, int | None]
        ``entailed`` is True if any claim is entailed.
        ``claim_index`` is the index into *claims* of the first
        entailed claim (or None if no entailment).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    text_emb = st_model.encode(text, show_progress_bar=False)
    sims = cosine_similarity([text_emb], bank_embeddings)[0]

    if top_k is not None:
        candidate_indices = np.argsort(sims)[::-1][:top_k].tolist()
    else:
        above = np.where(sims >= cosine_prefilter)[0]
        candidate_indices = above[np.argsort(sims[above])[::-1]].tolist()

    if not candidate_indices:
        return False, None

    for batch_start in range(0, len(candidate_indices), nli_batch_size):
        batch_idx = candidate_indices[
            batch_start : batch_start + nli_batch_size
        ]
        pairs = [{"text": text, "text_pair": claims[i]} for i in batch_idx]
        results = nli(pairs, truncation=True, max_length=512)
        for j, result in enumerate(results):
            if result["label"].lower() == "entailment":
                return True, batch_idx[j]
    return False, None


class CoTLeakDetector:
    """Two-stage CoT leak detector.

    Stage 1 screens every sentence with a fast RoBERTa classifier.
    Stage 2 confirms flagged sentences by projecting their embeddings
    into claim-cluster centroid space and checking cosine proximity.

    Parameters
    ----------
    classifier_path : str
        Path to the trained Stage 1 RoBERTa classifier checkpoint.
    projection_head_path : str
        Path to the trained Stage 2 projection head (saved by
        ``scripts/train_projection.py``).
    classifier_threshold : float
        Confidence threshold for the Stage 1 classifier.
    projection_threshold : float
        Minimum cosine similarity to a cluster centroid for Stage 2
        confirmation.
    sentence_transformer : SentenceTransformer | None
        Optional pre-loaded model.  Avoids loading a second copy when
        evaluators already have one.
    fallback : bool
        If True, run the full CoT through the projection when no
        individual sentence is confirmed (catches distributed leaks).
    """

    def __init__(
        self,
        classifier_path: str,
        projection_head_path: str,
        classifier_threshold: float = 0.5,
        projection_threshold: float = 0.5,
        sentence_transformer: SentenceTransformer | None = None,
        fallback: bool = True,
    ):
        # Stage 1 — sentence classifier
        self.classifier = CorruptionClassifier(
            model_name="roberta-base",
            model_path=classifier_path,
            batch_size=32,
        )
        self.classifier_threshold = classifier_threshold

        # Stage 2 — contrastive projection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.projection_head, self.target_centroids = ProjectionHead.load(
            projection_head_path, device=self.device
        )
        self.projection_head.eval()
        self.projection_threshold = projection_threshold
        self.fallback = fallback

        self.st_model = sentence_transformer or SentenceTransformer(
            "paraphrase-MiniLM-L6-v2",
            device=self.device,  # type: ignore[arg-type]
        )

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

        # --- Stage 2: projection confirmation (document order, early stop) -
        for idx in flagged_indices:
            confirmed, cluster_id = self._projection_confirms(sentences[idx])
            if confirmed:
                confirmed_flags[idx] = True
                return LeakDetectionResult(
                    is_leaking=True,
                    first_leak_index=idx,
                    sentences=sentences,
                    stage1_flags=stage1_flags,
                    confirmed_flags=confirmed_flags,
                    matched_cluster_id=cluster_id,
                )

        # --- Fallback: full-CoT projection --------------------------------
        # Stage 1 flagged sentences but none individually confirmed.
        # Project the full CoT text to catch distributed leaks.
        if self.fallback:
            confirmed, cluster_id = self._projection_confirms(generated_cot)
            if confirmed:
                return LeakDetectionResult(
                    is_leaking=True,
                    first_leak_index=0,
                    sentences=sentences,
                    stage1_flags=stage1_flags,
                    confirmed_flags=confirmed_flags,
                    matched_cluster_id=cluster_id,
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

    def _projection_confirms(self, text: str) -> tuple[bool, int | None]:
        """Check if *text* projects near any claim cluster centroid.

        Returns ``(confirmed, cluster_id)``."""
        emb = self.st_model.encode(text, show_progress_bar=False)
        with torch.no_grad():
            emb_t = torch.tensor(emb, dtype=torch.float32, device=self.device)
            proj_emb = self.projection_head(emb_t.unsqueeze(0))
            cos_sims = (proj_emb @ self.target_centroids.T).squeeze(0)
        max_cos = cos_sims.max().item()
        if max_cos >= self.projection_threshold:
            return True, int(cos_sims.argmax().item())
        return False, None
