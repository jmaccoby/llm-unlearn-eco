from rouge_score import rouge_scorer

from eco.evaluator.utils import split_sentences


class StepWiseROUGERecall:
    name = "stepwise_rouge_recall"

    def __init__(self, mode="rougeL"):
        self.mode = mode
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        self.name = f"stepwise_{mode}_recall"

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            gold_steps = split_sentences(a)
            gen_steps = split_sentences(ga)
            if not gold_steps or not gen_steps:
                scores.append(0.0)
                continue
            step_scores = []
            for g_step in gold_steps:
                best = max(
                    self.scorer.score(g_step, s)[self.mode].recall
                    for s in gen_steps
                )
                step_scores.append(best)
            scores.append(sum(step_scores) / len(step_scores))
        return scores
