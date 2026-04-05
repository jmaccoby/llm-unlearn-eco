"""
Evaluate unlearning on R-TOFU using AFE and CFE metrics.

Usage:
    python -m scripts.evaluate_rtofu --split forget10
    python -m scripts.evaluate_rtofu --split forget10 \
        --corrupt_method rand_noise_first_n --dims 500 --strength 50
"""
import argparse
import json
import os

from transformers import GenerationConfig

from eco.attack import AttackedModel, PromptClassifier
from eco.dataset.rtofu import RTOFU
from eco.evaluator import (
    CosineSimilarity,
    EntailmentScore,
    ROUGERecall,
    StepWiseCosineSimilarity,
    StepWiseROUGERecall,
    TokenEntropy,
)
from eco.inference import ReasoningGenerationEngine
from eco.model import HFModel, ReasoningModel
from eco.utils import compute_afe, compute_cfe, log_print, seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["forget01", "forget05", "forget10"],
    help="R-TOFU forget split to evaluate",
)
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--num_examples", type=int, default=0, help="Number of examples per subset to evaluate (0 = all)")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--classifier_threshold", type=float, default=0.99)
parser.add_argument("--corrupt_method", type=str, default=None)
parser.add_argument("--dims", type=int, default=None)
parser.add_argument("--strength", type=float, default=None)
parser.add_argument("--repetition_penalty", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval_retain", action="store_true", help="Also evaluate on the matched retain split")
parser.add_argument("--output_dir", type=str, default="results/rtofu")
# Leak detector arguments
parser.add_argument("--leak_classifier_path", type=str, default=None, help="Path to trained leak classifier (enables leak detection)")
parser.add_argument("--knowledge_bank_dir", type=str, default=None, help="Path to knowledge bank directory")
parser.add_argument("--leak_classifier_threshold", type=float, default=0.5)
parser.add_argument("--leak_cosine_prefilter", type=float, default=0.3)
args = parser.parse_args()

seed_everything(args.seed)

# Load model
log_print(f"Loading model: {args.model_name}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
if args.repetition_penalty is not None:
    generation_config.repetition_penalty = args.repetition_penalty

model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)

# Optionally wrap with corruption
if args.corrupt_method is not None:
    log_print(f"Loading prompt classifier: rtofu_classifiers/{args.split}")
    prompt_classifier = PromptClassifier(
        model_name="roberta-base",
        model_path=f"rtofu_classifiers/{args.split}",
        batch_size=args.batch_size,
    )
    corrupt_args = {"dims": args.dims}
    if args.strength is not None:
        corrupt_args["strength"] = args.strength
    model = AttackedModel(
        model=model,
        prompt_classifier=prompt_classifier,
        token_classifier=None,
        corrupt_method=args.corrupt_method,
        corrupt_args=corrupt_args,
        classifier_threshold=args.classifier_threshold,
    )

model = ReasoningModel(model)

# Load dataset
model_config = model.model_config
rtofu = RTOFU(
    formatting_tokens=model_config.get("formatting_tokens"),
    eos_token=model.tokenizer.eos_token,
)
rtofu.download()

subset_names = [args.split]
if args.eval_retain:
    subset_names.append(RTOFU.match_retain[args.split])

# Optionally limit the number of examples per subset
if args.num_examples > 0:
    for name in subset_names:
        n = min(args.num_examples, len(rtofu.dataset[name]))
        rtofu.dataset[name] = rtofu.dataset[name].select(range(n))

log_print(f"Evaluating on subsets: {subset_names}")

# Answer evaluators (AFE)
answer_evaluators = [
    ROUGERecall(mode="rougeL"),
    CosineSimilarity(),
    EntailmentScore(reverse=False),
    TokenEntropy(tokenizer=model.tokenizer),
]

# CoT evaluators (CFE) — step-wise best-match alignment per sentence
cot_evaluators = [
    StepWiseROUGERecall(mode="rougeL"),
    StepWiseCosineSimilarity(),
]

# Run generation + evaluation
engine = ReasoningGenerationEngine(
    model=model,
    tokenizer=model.tokenizer,
    data_module=rtofu,
    subset_names=subset_names,
    answer_evaluator=answer_evaluators,
    cot_evaluator=cot_evaluators,
    batch_size=args.batch_size,
)
engine.inference()
summary, outputs = engine.summary()

# Compute aggregate scores
all_results = {}
for r in summary:
    all_results.update(r)

forget_prefix = f"rtofu_{args.split}"
all_results["AFE"] = compute_afe(all_results, forget_prefix)
all_results["CFE"] = compute_cfe(all_results, forget_prefix)
log_print(f"AFE: {all_results['AFE']:.4f}")
log_print(f"CFE: {all_results['CFE']:.4f}")

# Optional: run leak detection on generated CoTs
if args.leak_classifier_path is not None:
    from eco.attack.leak_detector import CoTLeakDetector

    kb_dir = args.knowledge_bank_dir or f"knowledge_banks/{args.split}"
    log_print(f"\nRunning leak detection (classifier={args.leak_classifier_path})")
    leak_detector = CoTLeakDetector(
        classifier_path=args.leak_classifier_path,
        knowledge_bank_dir=kb_dir,
        classifier_threshold=args.leak_classifier_threshold,
        cosine_prefilter=args.leak_cosine_prefilter,
    )

    for key, cot_data in engine.cot_generations.items():
        generated_cots = cot_data["generated"]
        results_list = leak_detector.detect_batch(generated_cots)
        n_leaking = sum(1 for r in results_list if r.is_leaking)
        n_total = len(results_list)
        leak_rate = n_leaking / n_total if n_total > 0 else 0.0
        all_results[f"{key}_leak_rate"] = leak_rate
        log_print(f"Leak rate ({key}): {n_leaking}/{n_total} = {leak_rate:.4f}")

# Save results
os.makedirs(args.output_dir, exist_ok=True)
run_name = "_".join(
    filter(None, [
        args.model_name,
        args.split,
        args.corrupt_method or "baseline",
        f"dims={args.dims}" if args.dims else None,
        f"str={args.strength}" if args.strength else None,
    ])
)
output_path = os.path.join(args.output_dir, f"{run_name}.json")
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
log_print(f"\nResults saved to {output_path}")
