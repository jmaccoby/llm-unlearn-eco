import torch
from rouge_score import rouge_scorer
from transformers import pipeline


class EntailmentScore:
    name = "entailment_score"

    def __init__(self, batch_size=5, rouge_threshold=0.1, reverse=False):
        self.pipe = pipeline(
            "text-classification",
            model="sileod/deberta-v3-base-tasksource-nli",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.rouge_scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.batch_size = batch_size
        self.rouge_threshold = rouge_threshold
        self.reverse = reverse

    def evaluate(self, answers, generated_answers):
        # Compute ROUGE-L to gate low-overlap pairs
        rouge_scores = [
            self.rouge_scorer.score(a, ga)["rougeL"].recall
            for a, ga in zip(answers, generated_answers)
        ]

        # Build text pairs for NLI
        if self.reverse:
            data_list = [
                {"text": a, "text_pair": ga}
                for a, ga in zip(answers, generated_answers)
            ]
        else:
            data_list = [
                {"text": ga, "text_pair": a}
                for a, ga in zip(answers, generated_answers)
            ]

        # Run NLI in batches
        results = []
        for i in range(0, len(data_list), self.batch_size):
            results.extend(self.pipe(data_list[i : i + self.batch_size]))

        # Apply ROUGE gate: if overlap too low, mark as not entailed
        labels = []
        for rouge, result in zip(rouge_scores, results):
            if rouge < self.rouge_threshold:
                labels.append(0)
            else:
                labels.append(1 if result["label"] == "entailment" else 0)
        return labels
