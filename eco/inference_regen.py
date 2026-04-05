"""Regenerating generation engine for reasoning model unlearning.

Extends ReasoningGenerationEngine with a detect-and-regenerate loop:
after initial generation, checks each CoT for forget-set leakage and
regenerates from the truncation point with prefix corruption and/or
a learned soft token.
"""

import time

import torch
from tqdm import tqdm

from eco.attack.model import AttackedModel
from eco.attack.utils import build_prefix_corruption_mask
from eco.evaluator.utils import split_sentences
from eco.inference import ReasoningGenerationEngine, _remove_hooks
from eco.model.reasoning import ReasoningModel
from eco.utils import fix_bpe, log_print


class RegeneratingReasoningEngine(ReasoningGenerationEngine):
    """ReasoningGenerationEngine with post-generation leak detection and regeneration.

    After initial generation, each CoT is checked for forget-set leakage.
    Leaking samples are regenerated from the truncation point using prefix
    corruption and/or a learned soft token, up to ``regen_max_attempts``.
    """

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
        # Regeneration parameters:
        leak_detector=None,
        regen_corrupt_mode="window",
        regen_window=32,
        regen_window_mode="sentences",
        regen_max_attempts=3,
        soft_token=None,
    ):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            data_module=data_module,
            subset_names=subset_names,
            answer_evaluator=answer_evaluator,
            cot_evaluator=cot_evaluator,
            batch_size=batch_size,
            prompt_prefix=prompt_prefix,
            comparison_length=comparison_length,
            truncate_answers=truncate_answers,
        )
        if regen_corrupt_mode not in ("window", "soft_token", "window+soft_token"):
            raise ValueError(
                f"Unknown regen_corrupt_mode: {regen_corrupt_mode!r}"
            )
        if "soft_token" in regen_corrupt_mode and soft_token is None:
            raise ValueError(
                f"regen_corrupt_mode={regen_corrupt_mode!r} requires "
                f"soft_token to be provided"
            )
        self.leak_detector = leak_detector
        self.regen_corrupt_mode = regen_corrupt_mode
        self.regen_window = regen_window
        self.regen_window_mode = regen_window_mode
        self.regen_max_attempts = regen_max_attempts
        self.soft_token = soft_token

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
                batch_cot, batch_answer = [], []
                for resp in batch_responses:
                    if self.THINK_SUFFIX in resp:
                        cot, answer = resp.split(self.THINK_SUFFIX, 1)
                        batch_cot.append(cot)
                        batch_answer.append(answer)
                    else:
                        batch_cot.append(resp)
                        batch_answer.append("")

                gold_cots = batch.get(
                    getattr(self.data_module, "gen_cot_key", None),
                    [""] * len(prompts),
                )

                # ----------------------------------------------------------
                # Phase 2: Leak detection + regeneration
                # ----------------------------------------------------------
                if self.leak_detector is not None:
                    results = self.leak_detector.detect_batch(batch_cot)

                    for sample_idx, result in enumerate(results):
                        if not result.is_leaking:
                            continue

                        first_leak_index = result.first_leak_index
                        if first_leak_index == 0:
                            # Fallback detected distributed leak but can't
                            # pinpoint a sentence — no clean prefix to keep.
                            # Skip regeneration for this sample.
                            log_print(
                                f"  Sample {sample_idx}: leak at index 0, "
                                f"skipping regeneration (no clean prefix)"
                            )
                            continue
                        clean_prefix = " ".join(
                            result.sentences[:first_leak_index]
                        )

                        prompt = prompts[sample_idx]

                        # Track best output: start with original, update on
                        # each attempt.  The last attempt (strongest corruption)
                        # is kept even if it still leaks — it's better than the
                        # original which is guaranteed to leak.
                        best_cot = batch_cot[sample_idx]
                        best_answer = batch_answer[sample_idx]

                        for attempt in range(self.regen_max_attempts):
                            regen_cot, regen_answer = self._regenerate_sample(
                                prompt, clean_prefix, attempt
                            )
                            best_cot, best_answer = regen_cot, regen_answer

                            # Re-run leak detection on regenerated CoT
                            new_result = self.leak_detector.detect(regen_cot)

                            if not new_result.is_leaking:
                                log_print(
                                    f"  Regeneration succeeded on attempt {attempt + 1}"
                                )
                                break

                            # Update clean prefix if leak moved
                            if (
                                new_result.first_leak_index is not None
                                and new_result.first_leak_index != first_leak_index
                            ):
                                first_leak_index = new_result.first_leak_index
                                clean_prefix = " ".join(
                                    new_result.sentences[:first_leak_index]
                                )
                            # Otherwise: same leak point, escalate window on next attempt
                        else:
                            log_print(
                                f"  Regeneration exhausted {self.regen_max_attempts} attempts"
                            )

                        batch_cot[sample_idx] = best_cot
                        batch_answer[sample_idx] = best_answer
                        # Keep batch_responses consistent with cot/answer
                        if best_answer:
                            batch_responses[sample_idx] = (
                                best_cot + self.THINK_SUFFIX + best_answer
                            )
                        else:
                            batch_responses[sample_idx] = best_cot

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

    # ------------------------------------------------------------------
    # Single-sample regeneration
    # ------------------------------------------------------------------

    def _regenerate_sample(self, prompt, clean_prefix, attempt):
        """Regenerate a single sample from its clean prefix.

        Returns (cot, answer) strings.
        """
        # Build input_ids by concatenating token lists to avoid BPE
        # context-sensitivity at boundaries.  Tokenizing a joined string
        # and then subtracting lengths is unreliable because BPE may
        # merge or split tokens differently at concatenation points.
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=True
        )["input_ids"]
        think_ids = self.tokenizer(
            ReasoningModel.THINK_PREFIX, add_special_tokens=False
        )["input_ids"]
        prefix_ids = self.tokenizer(
            clean_prefix, add_special_tokens=False
        )["input_ids"] if clean_prefix else []

        prompt_len = len(prompt_ids)
        think_len = len(think_ids)
        prefix_token_len = len(prefix_ids)

        all_ids = prompt_ids + think_ids + prefix_ids
        input_ids = torch.tensor([all_ids], device=self.model.device)
        attention_mask = torch.ones_like(input_ids)

        # Build corruption mask and optionally prepare soft token
        mask, input_ids, attention_mask, st_handle = self._build_regen_corruption(
            prompt_len, think_len, prefix_token_len, clean_prefix,
            attempt, input_ids, attention_mask,
        )

        # Unwrap ReasoningModel to get AttackedModel (think prefix is already
        # in the input, so we bypass ReasoningModel.generate())
        inner = self.model._inner
        if not hasattr(inner, "generate_with_mask"):
            raise TypeError(
                f"Regeneration requires a model with generate_with_mask() "
                f"(e.g. AttackedModel), got {type(inner).__name__}"
            )

        try:
            generated = inner.generate_with_mask(
                mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=self.model.generation_config,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            # Clean up soft token hook if one was registered
            if st_handle is not None:
                st_handle.remove()

        # Decode new tokens from the full extended input length
        decode_start = input_ids.shape[1]
        raw = self.tokenizer.decode(
            generated[0][decode_start:], skip_special_tokens=True
        )
        text = fix_bpe(raw)

        # Prepend the clean prefix to the newly generated continuation
        if clean_prefix:
            full_cot_text = clean_prefix + " " + text
        else:
            full_cot_text = text

        # Split at </think>\n\n
        if self.THINK_SUFFIX in full_cot_text:
            cot, answer = full_cot_text.split(self.THINK_SUFFIX, 1)
        else:
            cot = full_cot_text
            answer = ""

        return cot, answer

    def _build_regen_corruption(
        self, prompt_len, think_len, prefix_token_len, clean_prefix,
        attempt, input_ids, attention_mask,
    ):
        """Build the corruption mask and optional soft token hook for regeneration.

        Returns (mask, input_ids, attention_mask, st_handle).
        st_handle is None if no soft token is used.
        """
        st_handle = None
        mode = self.regen_corrupt_mode

        # Compute window size based on mode and attempt escalation
        window_size = self._compute_window_size(
            clean_prefix, prefix_token_len, attempt
        )

        uses_window = mode in ("window", "window+soft_token")
        uses_soft_token = mode in ("soft_token", "window+soft_token")

        if uses_soft_token:
            # Append a pad token for the soft token to replace
            pad_id = self.tokenizer.pad_token_id
            pad_token = torch.tensor(
                [[pad_id]], device=input_ids.device, dtype=input_ids.dtype
            )
            input_ids = torch.cat([input_ids, pad_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(1, 1, device=attention_mask.device, dtype=attention_mask.dtype)],
                dim=1,
            )
            # Soft token position is the last token (the appended pad)
            st_position = input_ids.shape[1] - 1

        if uses_window:
            mask = build_prefix_corruption_mask(
                prompt_len, think_len, prefix_token_len, window_size, batch_size=1
            )
            if uses_soft_token:
                # Extend mask with a 0 for the appended pad token (soft token
                # hook handles that position, not the corruption mask)
                mask = [row + [0] for row in mask]
        else:
            # soft_token only: all-zeros mask (corruption handled by soft token hook)
            total_len = input_ids.shape[1]
            mask = [[0] * total_len]

        if uses_soft_token:
            inner = self.model._inner
            st_handle = self.soft_token.apply_hook(inner.attack_module, st_position)

        return mask, input_ids, attention_mask, st_handle

    def _compute_window_size(self, clean_prefix, prefix_token_len, attempt):
        """Compute the corruption window size, escalating with each attempt."""
        if self.regen_window_mode == "sentences":
            sentences = split_sentences(clean_prefix)
            if not sentences:
                return self.regen_window

            # Sentence count escalates: 1, 2, 4, ... (doubling per attempt)
            n_sentences = min(1 * (2 ** attempt), len(sentences))

            # Compute window by tokenizing the *retained* (non-window) portion
            # and subtracting from total prefix tokens. This avoids BPE context
            # mismatch from tokenizing the window fragment in isolation.
            retained_sentences = sentences[:-n_sentences]
            if not retained_sentences:
                return prefix_token_len  # corrupt entire prefix
            retained_text = " ".join(retained_sentences)
            retained_tokens = self.tokenizer(
                retained_text, add_special_tokens=False
            )["input_ids"]
            return max(
                0, min(prefix_token_len - len(retained_tokens), prefix_token_len)
            )
        else:
            # Token mode: base window doubles each attempt
            return min(self.regen_window * (2 ** attempt), prefix_token_len)
