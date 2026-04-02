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

import numpy as np
from scipy.stats import hmean
from transformers import GenerationConfig

from eco.attack import AttackedReasoningModel, PromptClassifier
from eco.dataset.rtofu import RTOFU
from eco.evaluator import CosineSimilarity, EntailmentScore, ROUGERecall, TokenEntropy
from eco.inference import ReasoningGenerationEngine
from eco.model import HFModel
from eco.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    default="forget10",
    choices=["forget01", "forget05", "forget10"],
    help="R-TOFU forget split to evaluate",
)
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--classifier_threshold", type=float, default=0.99)
parser.add_argument("--corrupt_method", type=str, default=None)
parser.add_argument("--dims", type=int, default=None)
parser.add_argument("--strength", type=float, default=None)
parser.add_argument("--repetition_penalty", type=float, default=None)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="results/rtofu")
args = parser.parse_args()

seed_everything(args.seed)

# Load model
print(f"Loading model: {args.model_name}")
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
    print(f"Loading prompt classifier: rtofu_classifiers/{args.split}")
    prompt_classifier = PromptClassifier(
        model_name="roberta-base",
        model_path=f"rtofu_classifiers/{args.split}",
        batch_size=args.batch_size,
    )
    corrupt_args = {"dims": args.dims}
    if args.strength is not None:
        corrupt_args["strength"] = args.strength
    model = AttackedReasoningModel(
        model=model,
        prompt_classifier=prompt_classifier,
        token_classifier=None,
        corrupt_method=args.corrupt_method,
        corrupt_args=corrupt_args,
        classifier_threshold=args.classifier_threshold,
    )

# Load dataset
model_config = model.model_config
rtofu = RTOFU(
    formatting_tokens=model_config.get("formatting_tokens"),
    eos_token=model.tokenizer.eos_token,
)
rtofu.download()

retain_split = RTOFU.match_retain[args.split]
subset_names = [args.split, retain_split]
print(f"Evaluating on subsets: {subset_names}")

# AFE evaluators
afe_evaluators = [
    ROUGERecall(mode="rougeL"),
    CosineSimilarity(),
    EntailmentScore(reverse=False),
    TokenEntropy(tokenizer=model.tokenizer),
]

# Run generation + AFE evaluation
afe_engine = ReasoningGenerationEngine(
    model=model,
    tokenizer=model.tokenizer,
    data_module=rtofu,
    subset_names=subset_names,
    evaluator=afe_evaluators,
    batch_size=args.batch_size,
)
afe_engine.inference()
afe_summary, afe_outputs = afe_engine.summary()

# CFE: run evaluators on CoT portions (compare generated CoT vs gold CoT)
cfe_evaluators = [
    ROUGERecall(mode="rougeL"),
    CosineSimilarity(),
    EntailmentScore(reverse=False),
]

# Load gold CoT from dataset for each subset
gold_cots = {}
for subset_name in subset_names:
    dataset = rtofu.dataset[subset_name]
    gold_cots[f"rtofu_{subset_name}"] = [ex.get("cot", "") for ex in dataset]

cfe_results = []
for key, cot_data in afe_engine.cot_generations.items():
    gold_cot = gold_cots.get(key, cot_data["gold"])
    for evaluator in cfe_evaluators:
        scores = evaluator.evaluate(gold_cot, cot_data["generated"])
        result_key = f"{key}_cot_{evaluator.name}"
        cfe_results.append({result_key: scores})
        avg = float(np.mean(scores))
        print({result_key: avg})

# Compute aggregate scores
all_results = {}
for r in afe_summary + [{k: float(np.mean(v)) for k, v in d.items()} for d in cfe_results]:
    all_results.update(r)

# AFE = hmean(1 - forget_rouge, 1 - forget_cosine, 1 - forget_entailment)
forget_prefix = f"rtofu_{args.split}"
afe_metrics = ["rougeL_recall", "cosine_similarity", "entailment_score"]
afe_forget_scores = []
for metric in afe_metrics:
    key = f"{forget_prefix}_{metric}"
    if key in all_results:
        afe_forget_scores.append(1.0 - all_results[key])

if afe_forget_scores and all(s > 0 for s in afe_forget_scores):
    all_results["AFE"] = float(hmean(afe_forget_scores))
    print(f"AFE: {all_results['AFE']:.4f}")

# CFE = hmean(1 - forget_cot_rouge, 1 - forget_cot_cosine, 1 - forget_cot_entailment)
cfe_forget_scores = []
for metric in afe_metrics:
    key = f"{forget_prefix}_cot_{metric}"
    if key in all_results:
        cfe_forget_scores.append(1.0 - all_results[key])

if cfe_forget_scores and all(s > 0 for s in cfe_forget_scores):
    all_results["CFE"] = float(hmean(cfe_forget_scores))
    print(f"CFE: {all_results['CFE']:.4f}")

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
print(f"\nResults saved to {output_path}")
