import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class CosineSimilarity:
    name = "cosine_similarity"

    def __init__(self, model_name="paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(
            model_name, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def evaluate(self, answers, generated_answers):
        scores = []
        with torch.no_grad():
            for a, ga in zip(answers, generated_answers):
                a_emb = self.model.encode(a, show_progress_bar=False)
                ga_emb = self.model.encode(ga, show_progress_bar=False)
                sim = float(cosine_similarity([a_emb], [ga_emb])[0][0])
                scores.append(max(0.0, sim))
        return scores
