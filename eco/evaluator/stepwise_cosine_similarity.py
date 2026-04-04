import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from eco.evaluator.utils import split_sentences


class StepWiseCosineSimilarity:
    name = "stepwise_cosine_similarity"

    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    def evaluate(self, answers, generated_answers):
        scores = []
        with torch.no_grad():
            for a, ga in zip(answers, generated_answers):
                gold_steps = split_sentences(a)
                gen_steps = split_sentences(ga)
                if not gold_steps or not gen_steps:
                    scores.append(0.0)
                    continue
                gold_embs = self.model.encode(gold_steps, show_progress_bar=False)
                gen_embs = self.model.encode(gen_steps, show_progress_bar=False)
                sim_matrix = cosine_similarity(gold_embs, gen_embs)
                # NOTE: best-match alignment allows a single verbose
                # generated step to match every gold step, which can
                # inflate scores when the model consolidates multiple
                # reasoning steps into one.
                best_scores = np.clip(sim_matrix.max(axis=1), 0.0, 1.0)
                scores.append(float(best_scores.mean()))
        return scores
