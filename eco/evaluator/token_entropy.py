import nltk
import numpy as np


class TokenEntropy:
    name = "token_entropy"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def evaluate(self, answers, generated_answers):
        return [self._compute(ga) for ga in generated_answers]

    def _compute(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= 1:
            return 0.0
        fdist = nltk.FreqDist(nltk.ngrams(tokens, 1))
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy = float(np.sum(-freqs * np.log2(freqs)))
        max_entropy = np.log2(len(tokens))
        return entropy / max_entropy
