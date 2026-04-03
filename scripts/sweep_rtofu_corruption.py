"""
Optimize corruption parameters for R-TOFU unlearning using zeroth-order optimization.

For each corruption method and dims setting, uses ZerothOrderOptimizerScalar
to find the strength that maximizes AFE on the forget set.
Structural and parameter-free methods (no strength param) are evaluated directly.

Usage:
    python -m scripts.sweep_rtofu_corruption --split forget10
    python -m scripts.sweep_rtofu_corruption --split forget10 --num_examples 20
    python -m scripts.sweep_rtofu_corruption --split forget10 --methods rand_noise_first_n rand_noise_top_k
"""
import argparse
import json
import os

from scipy.stats import hmean
from transformers import GenerationConfig

from eco.attack import AttackedReasoningModel, PromptClassifier
from eco.dataset.rtofu import RTOFU
from eco.evaluator import CosineSimilarity, EntailmentScore, ROUGERecall, TokenEntropy
from eco.inference import ReasoningGenerationEngine
from eco.model import HFModel
from eco.optimizer import ZerothOrderOptimizerScalar
from eco.utils import seed_everything

# Methods that require dims + strength (optimized via ZOO)
NOISE_METHODS = ["rand_noise_first_n", "rand_noise_top_k", "set_rand_noise_first_n"]
VALUE_METHODS = ["sub_value_top_k", "add_value_least_k", "sub_value_first_n", "add_value_first_n"]

# Methods that require dims only (no strength to optimize)
STRUCTURAL_METHODS = ["zero_out_top_k", "flip_sign_top_k", "zero_out_first_n", "flip_sign_first_n"]

# Parameter-free methods
PARAMFREE_METHODS = ["shuffle", "reverse_order"]

DEFAULT_METHODS = [
    "rand_noise_first_n", "rand_noise_top_k", "set_rand_noise_first_n",
    "sub_value_top_k",
    "zero_out_top_k", "flip_sign_top_k",
    "shuffle", "reverse_order",
]

parser = argparse.ArgumentParser()
parser.add_argument("--split", type=str, default="forget10", choices=["forget01", "forget05", "forget10"])
parser.add_argument("--model_name", type=str, default="LRM-target")
parser.add_argument("--num_examples", type=int, default=0, help="Subsample per subset (0 = all)")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--methods", type=str, nargs="+", default=DEFAULT_METHODS, help="Corruption methods to sweep")
parser.add_argument("--dims_list", type=int, nargs="+", default=[1, 512, 2048, 4096], help="Dims values to try")
parser.add_argument("--structural_dims_list", type=int, nargs="+", default=[256, 512, 1024, 2048, 3072, 4096])
parser.add_argument("--initial_strength", type=float, default=10.0)
parser.add_argument("--min_strength", type=float, default=1.0)
parser.add_argument("--zoo_lr", type=float, default=50.0, help="ZOO learning rate")
parser.add_argument("--zoo_steps", type=int, default=10, help="ZOO optimization steps per config")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output_dir", type=str, default="results/rtofu")
args = parser.parse_args()

seed_everything(args.seed)
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model and dataset (shared across all configs)
# ---------------------------------------------------------------------------

print(f"Loading model: {args.model_name}")
generation_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=args.max_new_tokens,
    use_cache=True,
)
base_model = HFModel(
    model_name=args.model_name,
    config_path="./config/rtofu_model_config",
    generation_config=generation_config,
)

print(f"Loading prompt classifier: rtofu_classifiers/{args.split}")
prompt_classifier = PromptClassifier(
    model_name="roberta-base",
    model_path=f"rtofu_classifiers/{args.split}",
    batch_size=args.batch_size,
)

model_config = base_model.model_config
rtofu = RTOFU(
    formatting_tokens=model_config.get("formatting_tokens"),
    eos_token=base_model.tokenizer.eos_token,
)
rtofu.download()

if args.num_examples > 0:
    n = min(args.num_examples, len(rtofu.dataset[args.split]))
    rtofu.dataset[args.split] = rtofu.dataset[args.split].select(range(n))

answer_evaluators = [
    ROUGERecall(mode="rougeL"),
    CosineSimilarity(),
    EntailmentScore(reverse=False),
    TokenEntropy(tokenizer=base_model.tokenizer),
]
cot_evaluators = [
    ROUGERecall(mode="rougeL"),
    CosineSimilarity(),
    EntailmentScore(reverse=False),
]

AFE_METRICS = ["rougeL_recall", "cosine_similarity", "entailment_score"]

# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_afe_cfe(summary):
    """Extract AFE and CFE from engine summary."""
    all_results = {}
    for r in summary:
        all_results.update(r)

    prefix = f"rtofu_{args.split}"
    afe_scores = []
    for metric in AFE_METRICS:
        key = f"{prefix}_{metric}"
        if key in all_results:
            afe_scores.append(1.0 - all_results[key])

    cfe_scores = []
    for metric in AFE_METRICS:
        key = f"{prefix}_cot_{metric}"
        if key in all_results:
            cfe_scores.append(1.0 - all_results[key])

    afe = float(hmean(afe_scores)) if afe_scores and all(s > 0 for s in afe_scores) else 0.0
    cfe = float(hmean(cfe_scores)) if cfe_scores and all(s > 0 for s in cfe_scores) else 0.0
    return afe, cfe, all_results


def evaluate_config(attacked_model):
    """Run evaluation and return (AFE, CFE, full_results)."""
    engine = ReasoningGenerationEngine(
        model=attacked_model,
        tokenizer=attacked_model.tokenizer,
        data_module=rtofu,
        subset_names=[args.split],
        answer_evaluator=answer_evaluators,
        cot_evaluator=cot_evaluators,
        batch_size=args.batch_size,
    )
    engine.inference()
    summary, _ = engine.summary()
    return compute_afe_cfe(summary)


def make_attacked_model(corrupt_method, corrupt_args):
    return AttackedReasoningModel(
        model=base_model,
        prompt_classifier=prompt_classifier,
        token_classifier=None,
        corrupt_method=corrupt_method,
        corrupt_args=corrupt_args,
        classifier_threshold=0.99,
    )


def result_path(method, dims=None, strength=None):
    parts = [args.model_name, args.split, method]
    if dims is not None:
        parts.append(f"dims={dims}")
    if strength is not None:
        parts.append(f"str={strength}")
    return os.path.join(args.output_dir, f"{'_'.join(parts)}.json")


def save_result(path, afe, cfe, all_results, extra=None):
    data = {**all_results, "AFE": afe, "CFE": cfe}
    if extra:
        data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Score function for ZOO (returns negative AFE — optimizer minimizes)
# ---------------------------------------------------------------------------


def zoo_score(strength, model, dims):
    model.update_corrupt_args({"dims": dims, "strength": strength})
    afe, cfe, _ = evaluate_config(model)
    combined = float(hmean([afe, cfe])) if afe > 0 and cfe > 0 else 0.0
    print(f"    strength={strength:.4f} -> AFE={afe:.4f}, CFE={cfe:.4f}, combined={combined:.4f}")
    # Return negative combined score since ZOO does gradient descent (minimizes)
    return -combined


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

all_sweep_results = []

for method in args.methods:
    print(f"\n{'='*60}")
    print(f"METHOD: {method}")
    print(f"{'='*60}")

    if method in PARAMFREE_METHODS:
        # Single evaluation, no parameters to tune
        path = result_path(method)
        if os.path.exists(path):
            print(f"  SKIP (exists): {path}")
            continue
        attacked = make_attacked_model(method, {})
        afe, cfe, results = evaluate_config(attacked)
        save_result(path, afe, cfe, results)
        all_sweep_results.append({"method": method, "AFE": afe, "CFE": cfe})

    elif method in STRUCTURAL_METHODS:
        # Sweep dims only
        for dims in args.structural_dims_list:
            path = result_path(method, dims=dims)
            if os.path.exists(path):
                print(f"  SKIP (exists): {path}")
                continue
            print(f"\n  dims={dims}")
            attacked = make_attacked_model(method, {"dims": dims})
            afe, cfe, results = evaluate_config(attacked)
            save_result(path, afe, cfe, results)
            all_sweep_results.append({"method": method, "dims": dims, "AFE": afe, "CFE": cfe})

    elif method in NOISE_METHODS + VALUE_METHODS:
        # For each dims value, use ZOO to optimize strength
        for dims in args.dims_list:
            print(f"\n  dims={dims}, optimizing strength via ZOO ({args.zoo_steps} steps)...")
            eps = args.initial_strength * 0.5
            optimizer = ZerothOrderOptimizerScalar(
                lr=args.zoo_lr,
                eps=eps,
                beta=args.initial_strength,
                min_beta=args.min_strength,
            )
            best_combined, best_strength = 0.0, args.initial_strength
            attacked = make_attacked_model(method, {"dims": dims, "strength": args.initial_strength})

            for step in range(args.zoo_steps):
                output = optimizer.step(
                    zoo_score,
                    {"model": attacked, "dims": dims},
                )
                current_combined = -output["f_score"]  # We negated combined in score fn
                print(f"  step {step}: beta={output['beta']:.4f}, combined={current_combined:.4f}")
                if current_combined > best_combined:
                    best_combined = current_combined
                    best_strength = output["beta"]

            # Final evaluation at best strength
            print(f"  Best strength={best_strength:.4f}, evaluating...")
            attacked.update_corrupt_args({"dims": dims, "strength": best_strength})
            afe, cfe, results = evaluate_config(attacked)
            path = result_path(method, dims=dims, strength=round(best_strength, 4))
            save_result(path, afe, cfe, results, extra={"optimized_strength": best_strength})
            all_sweep_results.append({
                "method": method, "dims": dims, "strength": best_strength,
                "AFE": afe, "CFE": cfe,
            })

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print("RESULTS SUMMARY (sorted by AFE)")
print(f"{'='*60}")

all_sweep_results.sort(key=lambda x: x["AFE"], reverse=True)
print(f"{'Method':<30} {'Dims':>6} {'Strength':>10} {'AFE':>8} {'CFE':>8}")
print("-" * 66)
for r in all_sweep_results:
    dims_str = str(r.get("dims", "-"))
    str_str = f"{r['strength']:.2f}" if "strength" in r else "-"
    print(f"{r['method']:<30} {dims_str:>6} {str_str:>10} {r['AFE']:>8.4f} {r['CFE']:>8.4f}")
