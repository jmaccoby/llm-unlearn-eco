"""
Examine model responses to R-TOFU dataset queries with ECO corruption applied.

Supports prompt corruption, CoT regeneration, or both.

Usage:
    # Prompt corruption only
    python -m scripts.examine_rtofu_corrupted --split forget10 --num_examples 10 \
        --corrupt_method rand_noise_first_n --dims 500 --strength 50

    # Regeneration only (no prompt corruption)
    python -m scripts.examine_rtofu_corrupted --split forget10 --num_examples 4 \
        --regen_corrupt_mode window --regen_window_mode sentences \
        --regen_corrupt_method rand_noise_first_n --regen_dims 500 --regen_strength 50 \
        --leak_classifier_path leak_classifiers/forget10 \
        --knowledge_bank_dir knowledge_banks/forget10 --show_cot

    # Both prompt corruption and regeneration
    python -m scripts.examine_rtofu_corrupted --split forget10 --num_examples 4 \
        --corrupt_method rand_noise_first_n --dims 500 --strength 50 \
        --regen_corrupt_mode window --regen_window_mode sentences \
        --leak_classifier_path leak_classifiers/forget10 \
        --knowledge_bank_dir knowledge_banks/forget10 --show_cot
"""
import argparse
import csv

from transformers import GenerationConfig

from eco.attack import AttackedModel, PromptClassifier
from eco.dataset.rtofu import RTOFU
from eco.inference import ReasoningGenerationEngine
from eco.model import HFModel, ReasoningModel
from eco.utils import log_print


MODEL_NAME = "LRM-target"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["full", "retain90", "retain50", "retain10", "forget01", "forget05", "forget10"],
    help="R-TOFU split to query",
)
parser.add_argument("--num_examples", type=int, default=10, help="Number of examples to examine (0 = all)")
parser.add_argument("--offset", type=int, default=0, help="Start from this index")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=200)
parser.add_argument("--show_cot", action="store_true", help="Also print the gold chain-of-thought")
parser.add_argument("--output", type=str, default=None, help="Save results to CSV file")
# Prompt corruption (optional)
parser.add_argument("--corrupt_method", type=str, default=None, help="Corruption method name (e.g. rand_noise_first_n)")
parser.add_argument("--dims", type=int, default=None, help="Number of embedding dimensions to corrupt")
parser.add_argument("--strength", type=float, default=None, help="Corruption strength")
parser.add_argument("--classifier_threshold", type=float, default=0.99, help="Prompt classifier confidence threshold")
parser.add_argument("--repetition_penalty", type=float, default=None, help="Repetition penalty for generation")
# Leak detector
parser.add_argument("--leak_classifier_path", type=str, default=None, help="Path to trained leak classifier")
parser.add_argument("--knowledge_bank_dir", type=str, default=None, help="Path to knowledge bank directory")
parser.add_argument("--leak_classifier_threshold", type=float, default=0.5)
parser.add_argument("--leak_cosine_prefilter", type=float, default=0.3)
# Regeneration
parser.add_argument("--regen_corrupt_mode", type=str, default=None,
                    choices=["window", "soft_token", "window+soft_token"],
                    help="Corruption mode for regeneration (None = disabled)")
parser.add_argument("--regen_corrupt_method", type=str, default=None,
                    help="Corruption method for regen (default: same as --corrupt_method)")
parser.add_argument("--regen_dims", type=int, default=None, help="Embedding dims for regen (default: same as --dims)")
parser.add_argument("--regen_strength", type=float, default=None, help="Noise strength for regen (default: same as --strength)")
parser.add_argument("--regen_window", type=int, default=32)
parser.add_argument("--regen_window_mode", type=str, default="sentences", choices=["tokens", "sentences"])
parser.add_argument("--regen_max_attempts", type=int, default=3)
parser.add_argument("--soft_token_path", type=str, default=None, help="Path to trained soft token embedding")
args = parser.parse_args()

# Validate
if args.corrupt_method is None and args.regen_corrupt_mode is None:
    parser.error("At least one of --corrupt_method or --regen_corrupt_mode is required")
if args.regen_corrupt_mode is not None and args.leak_classifier_path is None:
    parser.error("--regen_corrupt_mode requires --leak_classifier_path")
if args.soft_token_path is not None and (
    args.regen_corrupt_mode is None or "soft_token" not in args.regen_corrupt_mode
):
    parser.error("--soft_token_path requires --regen_corrupt_mode to be 'soft_token' or 'window+soft_token'")

# Resolve regen corruption config (fall back to prompt corruption values)
regen_corrupt_method = args.regen_corrupt_method or args.corrupt_method
regen_dims = args.regen_dims if args.regen_dims is not None else args.dims
regen_strength = args.regen_strength if args.regen_strength is not None else args.strength
if args.regen_corrupt_mode is not None and "window" in args.regen_corrupt_mode and regen_corrupt_method is None:
    parser.error("Window regen mode requires either --regen_corrupt_method or --corrupt_method")
if regen_corrupt_method is not None and regen_dims is None:
    parser.error("Regen corruption requires --regen_dims or --dims")

# Load model
log_print(f"Loading model: {MODEL_NAME}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
if args.repetition_penalty is not None:
    generation_config.repetition_penalty = args.repetition_penalty
model = HFModel(
    model_name=MODEL_NAME,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)

# Wrap with AttackedModel (prompt and/or regen corruption)
needs_prompt_corruption = args.corrupt_method is not None
needs_regen_corruption = args.regen_corrupt_mode is not None
if needs_prompt_corruption or needs_regen_corruption:
    prompt_classifier = None
    corrupt_args = None
    if needs_prompt_corruption:
        log_print(f"Loading prompt classifier: rtofu_classifiers/{args.split}")
        prompt_classifier = PromptClassifier(
            model_name="roberta-base",
            model_path=f"rtofu_classifiers/{args.split}",
            batch_size=args.batch_size,
        )
        corrupt_args = {"dims": args.dims}
        if args.strength is not None:
            corrupt_args["strength"] = args.strength

    regen_corrupt_args = None
    if needs_regen_corruption:
        regen_corrupt_args = {"dims": regen_dims}
        if regen_strength is not None:
            regen_corrupt_args["strength"] = regen_strength

    soft_token = None
    if args.soft_token_path is not None:
        from eco.attack.soft_token import SoftToken
        soft_token = SoftToken.load(args.soft_token_path, embed_dim=model.model_config["embedding_dim"])
        log_print(f"Loaded soft token from {args.soft_token_path}")

    model = AttackedModel(
        model=model,
        prompt_classifier=prompt_classifier,
        token_classifier=None,
        corrupt_method=args.corrupt_method,
        corrupt_args=corrupt_args,
        classifier_threshold=args.classifier_threshold,
        regen_corrupt_method=regen_corrupt_method,
        regen_corrupt_args=regen_corrupt_args,
        soft_token=soft_token,
    )

model = ReasoningModel(model)

# Load dataset and select examples
log_print(f"Loading R-TOFU split: {args.split}")
data_module = RTOFU(
    formatting_tokens=model.model_config.get("formatting_tokens"),
    eos_token=model.tokenizer.eos_token,
)
data_module.download()

end = len(data_module.dataset[args.split]) if args.num_examples == 0 else min(
    args.offset + args.num_examples, len(data_module.dataset[args.split])
)
data_module.dataset[args.split] = data_module.dataset[args.split].select(range(args.offset, end))

# Generate via engine
if args.regen_corrupt_mode is not None:
    from eco.attack.leak_detector import CoTLeakDetector
    from eco.inference_regen import RegeneratingReasoningEngine

    kb_dir = args.knowledge_bank_dir or f"knowledge_banks/{args.split}"
    leak_detector = CoTLeakDetector(
        classifier_path=args.leak_classifier_path,
        knowledge_bank_dir=kb_dir,
        classifier_threshold=args.leak_classifier_threshold,
        cosine_prefilter=args.leak_cosine_prefilter,
    )
    engine = RegeneratingReasoningEngine(
        model=model,
        tokenizer=model.tokenizer,
        data_module=data_module,
        subset_names=[args.split],
        answer_evaluator=[],
        batch_size=args.batch_size,
        leak_detector=leak_detector,
        regen_corrupt_mode=args.regen_corrupt_mode,
        regen_window=args.regen_window,
        regen_window_mode=args.regen_window_mode,
        regen_max_attempts=args.regen_max_attempts,
    )
    log_print(f"Regeneration enabled: mode={args.regen_corrupt_mode}, max_attempts={args.regen_max_attempts}")
else:
    engine = ReasoningGenerationEngine(
        model=model,
        tokenizer=model.tokenizer,
        data_module=data_module,
        subset_names=[args.split],
        answer_evaluator=[],
        batch_size=args.batch_size,
    )

corruption_desc = args.corrupt_method or "none"
log_print(f"\nExamining {end - args.offset} examples from split '{args.split}'")
log_print(f"Prompt corruption: {corruption_desc} (dims={args.dims}, strength={args.strength})")
if args.regen_corrupt_mode:
    log_print(f"Regen corruption: {regen_corrupt_method} (dims={regen_dims}, strength={regen_strength})")
log_print("=" * 80)

generations = engine._generate()
data = generations[args.split]

# Flatten batch lists
prompts = [item for batch in data["prompt"] for item in batch]
gold_answers = [item for batch in data["gold_answer"] for item in batch]
gold_cots = [item for batch in data["gold_cot"] for item in batch]
generated_cots = [item for batch in data["generated_cot"] for item in batch]
generated_answers = [item for batch in data["generated_answer"] for item in batch]

# Display results
results = []
for i in range(len(prompts)):
    idx = args.offset + i
    print(f"\n[{idx}] QUESTION:\n{prompts[i]}\n")
    print(f"GOLD ANSWER:\n{gold_answers[i]}\n")
    if args.show_cot:
        if gold_cots[i]:
            print(f"GOLD COT:\n{gold_cots[i]}\n")
        if generated_cots[i]:
            print(f"MODEL THINKING:\n{generated_cots[i].strip()}\n")
    print(f"MODEL RESPONSE:\n{generated_answers[i]}\n")
    print("-" * 80)

    row = {"index": idx, "question": prompts[i], "gold_answer": gold_answers[i], "response": generated_answers[i]}
    if args.show_cot:
        row["gold_cot"] = gold_cots[i]
        row["model_cot"] = generated_cots[i]
    results.append(row)

# Save to CSV if requested
if args.output:
    fieldnames = ["index", "question", "gold_answer", "response"]
    if args.show_cot:
        fieldnames.extend(["gold_cot", "model_cot"])
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_NONNUMERIC,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(results)
    log_print(f"Saved {len(results)} results to {args.output}")
