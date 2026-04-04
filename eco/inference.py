import time

import numpy as np
from tqdm import tqdm

from eco.attack.model import AttackedModel
from eco.attack.utils import remove_hooks
from eco.utils import fix_bpe


def _remove_hooks(model):
    """Remove hooks using the model's handle-based method if available, else fallback."""
    # Unwrap ReasoningModel to reach the AttackedModel or HFModel inside.
    from eco.model.reasoning import ReasoningModel
    inner = model._inner if isinstance(model, ReasoningModel) else model
    if isinstance(inner, AttackedModel):
        inner.remove_hooks()
    elif hasattr(inner, "model"):
        remove_hooks(inner.model)
    else:
        remove_hooks(inner)


class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.data_module = data_module
        self.subset_names = subset_names
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.prompt_prefix = prompt_prefix

    def prepare_dataset(self):
        self.datasets = {}
        for subset_name in self.subset_names:
            self.datasets[subset_name] = self.data_module.load_dataset_for_eval(
                subset_name,
                load_in_batch=True,
                batch_size=self.batch_size,
                prompt_prefix=self.prompt_prefix,
            )

    def inference(self):
        raise NotImplementedError(
            f"inference not implemented for {self.__class__.__name__}"
        )

    def summary(self):
        summary_stats, outputs = [], []
        for result in self.results:
            name, data = list(result.items())[0]
            if (
                self.data_module.dataset_type == "multiple_choice"
                and self.data_module.name != "truthfulqa"
            ):
                pred, correct = [], []
                for d in data:
                    pred.extend(d["predicted"])
                    correct.extend(d["correct"])
                data = np.array(pred) == np.array(correct)
            avg_score = {name: float(np.mean(data))}
            summary_stats.append(avg_score)
            print(avg_score)
            outputs.append(result)
        return summary_stats, outputs


class EvaluationEngine(InferenceEngine):
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            evaluator,
            batch_size,
            prompt_prefix,
        )

    def inference(self):
        self.prepare_dataset()
        self.results = []
        for subset_name, dataset in self.datasets.items():
            all_outputs = []
            total_time, total_examples = 0, 0
            for batch in tqdm(
                dataset,
                desc=f"Evaluating {self.evaluator.name} of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                _remove_hooks(self.model)
                prompts = batch[self.data_module.eval_prompt_key]
                answers = batch[self.data_module.eval_answer_key]

                start_time = time.perf_counter()
                outputs = self.evaluator.evaluate(
                    prompts, answers, self.model, self.tokenizer
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)

                if self.data_module.dataset_type == "multiple_choice":
                    if self.data_module.name != "truthfulqa":
                        correct_answer = batch["correct_answer"]
                        outputs = [{"correct": correct_answer, "predicted": outputs}]
                all_outputs.extend(outputs)
                _remove_hooks(self.model)
            self.results.append(
                {
                    f"{self.data_module.name}_{subset_name}_{self.evaluator.name}": all_outputs
                }
            )
            avg_time_per_example = (
                total_time / total_examples if total_examples > 0 else 0
            )
            # print(
            #     tabulate(
            #         [
            #             ["Total examples", total_examples],
            #             ["Total time (sec)", f"{total_time:.4f}"],
            #             ["Avg time (sec)", f"{avg_time_per_example:.4f}"],
            #         ],
            #         headers=[f"{subset_name} of {self.data_module.name}", "Value"],
            #         tablefmt="pretty",
            #     )
            # )
        return self.results


class GenerationEngine(InferenceEngine):
    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        evaluator,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            evaluator,
            batch_size,
            prompt_prefix,
        )
        self.comparison_length = comparison_length
        self.truncate_answers = truncate_answers
        if not isinstance(self.evaluator, list):
            self.evaluator = [evaluator]

    def inference(self):
        self.results = []
        answers = self._generate()
        self.text_generations = {}
        for subset_name, data in answers.items():
            # Flatten data["gold"] and data["generated"]
            data_gold = [item for sublist in data["gold"] for item in sublist]
            data_generated = [item for sublist in data["generated"] for item in sublist]
            self.text_generations[f"{self.data_module.name}_{subset_name}"] = {
                "gold": data_gold,
                "generated": data_generated,
            }
            for evaluator in self.evaluator:
                evaluator_outputs = []
                for prompt, gold, generated in tqdm(
                    zip(data["prompt"], data["gold"], data["generated"]),
                    total=len(data["gold"]),
                    desc=f"Evaluating {evaluator.name} of {self.data_module.name} on {subset_name}",
                ):
                    if evaluator.name == "perplexity":
                        generated = [p + g for p, g in zip(prompt, generated)]
                    outputs = evaluator.evaluate(gold, generated)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {
                        f"{self.data_module.name}_{subset_name}_{evaluator.name}": evaluator_outputs
                    }
                )

    def _generate(self):
        self.prepare_dataset()
        padding_side = self.tokenizer.padding_side
        if padding_side != "left":
            self.tokenizer.padding_side = "left"
        subsets_generations = {}
        for subset_name, dataset in self.datasets.items():
            all_gold_answers, all_generated_answers = [], []
            all_prompts = []
            total_time, total_examples = 0, 0
            for batch in tqdm(
                dataset,
                desc=f"Generating completions of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                _remove_hooks(self.model)
                prompts = batch[self.data_module.gen_prompt_key]
                gold_answers = batch[self.data_module.gen_answer_key]

                tokenized_prompts = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.model.device)

                # Generate and decode answers
                start_time = time.perf_counter()
                generated = self.model.generate(
                    **tokenized_prompts,
                    prompts=prompts,
                    generation_config=self.model.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)
                generated_answers = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )

                # Remove prompt from generated answers
                generated_answers_truncated = []
                for p, g in zip(prompts, generated_answers):
                    generated_answers_truncated.append(g[len(p) :])
                # Remove special tokens from answers
                gold_answers = self.tokenizer.batch_decode(
                    self.tokenizer(gold_answers, add_special_tokens=False).input_ids,
                    skip_special_tokens=True,
                )
                if self.truncate_answers:
                    gold_answers, generated_answers_truncated = self.truncate(
                        gold_answers, generated_answers_truncated
                    )
                all_gold_answers.append(gold_answers)
                all_generated_answers.append(generated_answers_truncated)
                all_prompts.append(prompts)
                _remove_hooks(self.model)

            assert (
                len(all_gold_answers) == len(all_generated_answers) == len(all_prompts)
            ), f"Length mismatch: {len(all_gold_answers)}, {len(all_generated_answers)}, {len(all_prompts)}"
            subsets_generations[subset_name] = {
                "prompt": all_prompts,
                "gold": all_gold_answers,
                "generated": all_generated_answers,
            }

            avg_time_per_example = (
                total_time / total_examples if total_examples > 0 else 0
            )
            # print(
            #     tabulate(
            #         [
            #             ["Total examples", total_examples],
            #             ["Total time (sec)", f"{total_time:.4f}"],
            #             ["Avg time (sec)", f"{avg_time_per_example:.4f}"],
            #         ],
            #         headers=[f"{subset_name} of {self.data_module.name}", "Value"],
            #         tablefmt="pretty",
            #     )
            # )

        # Reset padding side
        self.tokenizer.padding_side = padding_side
        return subsets_generations

    def truncate(self, gold, generated):
        truncated_gold, truncated_generated = [], []
        for gold_answer, generated_answer in zip(gold, generated):
            min_len = min(len(gold_answer), len(generated_answer))
            truncated_gold.append(gold_answer[:min_len])
            truncated_generated.append(generated_answer[:min_len])
        return truncated_gold, truncated_generated


class ReasoningGenerationEngine(GenerationEngine):
    """GenerationEngine for reasoning models (e.g. DeepSeek-R1).

    Appends a ``<think>\\n`` prefix to force reasoning, applies BPE fix to
    decoded outputs, and splits responses into CoT and answer portions at
    the ``</think>\\n\\n`` delimiter.

    After ``inference()`` completes, ``cot_generations`` and
    ``answer_generations`` are available as dicts keyed by
    ``{dataset_name}_{subset_name}``.
    """

    THINK_SUFFIX = "</think>\n\n"

    def __init__(
        self,
        model,
        tokenizer,
        data_module,
        subset_names,
        answer_evaluator,
        cot_evaluator=None,
        batch_size=64,
        prompt_prefix="",
        comparison_length=128,
        truncate_answers=False,
    ):
        super().__init__(
            model,
            tokenizer,
            data_module,
            subset_names,
            answer_evaluator,
            batch_size,
            prompt_prefix,
            comparison_length,
            truncate_answers,
        )
        self.cot_evaluator = cot_evaluator or []
        if not isinstance(self.cot_evaluator, list):
            self.cot_evaluator = [self.cot_evaluator]

    def _generate(self):
        self.prepare_dataset()
        padding_side = self.tokenizer.padding_side
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"

        n_think = getattr(self.model, "n_think_tokens", 0)

        subsets_generations = {}
        for subset_name, dataset in self.datasets.items():
            all_gold_answers, all_gold_cots, all_generated_answers = [], [], []
            all_generated_cot, all_generated_answer = [], []
            all_prompts = []
            total_time, total_examples = 0, 0

            for batch in tqdm(
                dataset,
                desc=f"Generating completions of {self.data_module.name} on {subset_name}",
                total=len(dataset),
            ):
                _remove_hooks(self.model)
                prompts = batch[self.data_module.gen_prompt_key]
                gold_answers = batch[self.data_module.gen_answer_key]

                tokenized_prompts = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                ).to(self.model.device)

                # The model (ReasoningModel) appends think tokens in generate(),
                # so decode_start accounts for both the tokenized prompt and
                # the think prefix that will be prepended.
                decode_start = tokenized_prompts["input_ids"].shape[1] + n_think

                start_time = time.perf_counter()
                generated = self.model.generate(
                    **tokenized_prompts,
                    prompts=prompts,
                    generation_config=self.model.generation_config,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                end_time = time.perf_counter()
                total_time += end_time - start_time
                total_examples += len(prompts)

                # Decode only new tokens (after prompt + think prefix), apply BPE fix
                batch_responses = []
                for i in range(generated.shape[0]):
                    raw = self.tokenizer.decode(
                        generated[i][decode_start:], skip_special_tokens=True
                    )
                    batch_responses.append(fix_bpe(raw))

                # Split into CoT and answer at </think>\n\n
                # If delimiter is missing (e.g. truncated by max_new_tokens
                # or incoherent output from corruption), treat the full
                # response as CoT with an empty answer.
                batch_cot, batch_answer = [], []
                for idx, resp in enumerate(batch_responses):
                    if self.THINK_SUFFIX in resp:
                        cot, answer = resp.split(self.THINK_SUFFIX, 1)
                        batch_cot.append(cot)
                        batch_answer.append(answer)
                    else:
                        batch_cot.append(resp)
                        batch_answer.append("")

                # Gold answers are plain text from the dataset — no special
                # tokens to strip.  Skipping the tokenize/decode roundtrip
                # avoids a lossy encode (this tokenizer merges "full name"
                # into a single "fullname" token, dropping the space).

                gold_cots = batch.get(getattr(self.data_module, "gen_cot_key", None), [""] * len(prompts))

                all_gold_answers.append(gold_answers)
                all_gold_cots.append(gold_cots)
                all_generated_answers.append(batch_responses)
                all_generated_cot.append(batch_cot)
                all_generated_answer.append(batch_answer)
                all_prompts.append(prompts)
                _remove_hooks(self.model)

            subsets_generations[subset_name] = {
                "prompt": all_prompts,
                "gold_answer": all_gold_answers,
                "gold_cot": all_gold_cots,
                "generated": all_generated_answers,
                "generated_cot": all_generated_cot,
                "generated_answer": all_generated_answer,
            }

        self.tokenizer.padding_side = padding_side
        return subsets_generations

    def inference(self):
        self.results = []
        answers = self._generate()
        self.text_generations = {}
        self.cot_generations = {}
        self.answer_generations = {}

        for subset_name, data in answers.items():
            key = f"{self.data_module.name}_{subset_name}"

            data_gold_answer = [item for sublist in data["gold_answer"] for item in sublist]
            data_gold_cot = [item for sublist in data["gold_cot"] for item in sublist]
            data_generated = [item for sublist in data["generated"] for item in sublist]
            data_cot = [item for sublist in data["generated_cot"] for item in sublist]
            data_answer = [item for sublist in data["generated_answer"] for item in sublist]

            self.text_generations[key] = {"gold": data_gold_answer, "generated": data_generated}
            self.cot_generations[key] = {"gold": data_gold_cot, "generated": data_cot}
            self.answer_generations[key] = {"gold": data_gold_answer, "generated": data_answer}

            # Run answer evaluators (AFE)
            for evaluator in self.evaluator:
                evaluator_outputs = []
                for gold, generated_answer in tqdm(
                    zip(data["gold_answer"], data["generated_answer"]),
                    total=len(data["gold_answer"]),
                    desc=f"Evaluating {evaluator.name} of {self.data_module.name} on {subset_name}",
                ):
                    outputs = evaluator.evaluate(gold, generated_answer)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {f"{self.data_module.name}_{subset_name}_{evaluator.name}": evaluator_outputs}
                )

            # Run CoT evaluators (CFE)
            for evaluator in self.cot_evaluator:
                evaluator_outputs = []
                for gold_cot, generated_cot in tqdm(
                    zip(data["gold_cot"], data["generated_cot"]),
                    total=len(data["gold_cot"]),
                    desc=f"Evaluating {evaluator.name} (CoT) of {self.data_module.name} on {subset_name}",
                ):
                    outputs = evaluator.evaluate(gold_cot, generated_cot)
                    evaluator_outputs.extend(outputs)
                self.results.append(
                    {f"{self.data_module.name}_{subset_name}_cot_{evaluator.name}": evaluator_outputs}
                )
