from rouge_score import rouge_scorer

from eco.evaluator.utils import split_sentences


class StepWiseROUGERecall:
    name = "stepwise_rougeL_recall"  # matches default mode="rougeL"

    def __init__(self, mode="rougeL"):
        self.mode = mode
        self.scorer = rouge_scorer.RougeScorer([mode], use_stemmer=True)
        self.name = f"stepwise_{mode}_recall"

    def evaluate(self, answers, generated_answers):
        scores = []
        for a, ga in zip(answers, generated_answers):
            gold_steps = split_sentences(a)
            gen_steps = split_sentences(ga)
            if not gold_steps or not gen_steps:
                scores.append(0.0)
                continue
            # NOTE: best-match alignment allows a single verbose generated
            # step to be selected as the best match for every gold step,
            # which can inflate scores when the model consolidates multiple
            # reasoning steps into one.
            step_scores = []
            for g_step in gold_steps:
                best = max(
                    self.scorer.score(g_step, s)[self.mode].recall
                    for s in gen_steps
                )
                step_scores.append(best)
            scores.append(sum(step_scores) / len(step_scores))
        return scores
