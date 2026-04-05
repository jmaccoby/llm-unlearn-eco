"""
Train a Stage 1 sentence-level leak classifier for CoT leak detection.

Follows the same pattern as scripts/train_classifier.py:
  - RoBERTa-base fine-tuned for binary classification
  - Weighted cross-entropy for class imbalance
  - Threshold-based evaluation metrics

Requires training data from generate_leak_labels.py.

Usage:
    python -m scripts.train_leak_classifier --split forget10
    python -m scripts.train_leak_classifier --split forget10 --threshold 0.9
"""
import argparse

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from eco.utils import log_print

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--split",
    type=str,
    required=True,
    choices=["forget01", "forget05", "forget10"],
)
parser.add_argument("--data_dir", type=str, default="leak_detector_data")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--threshold", type=float, default=0.5)
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--output_dir", type=str, default="leak_classifiers")
args = parser.parse_args()


# Load tokenizer
model_name = "roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).cpu().numpy()
    predictions = np.where(probs[:, 1] > args.threshold, 1, 0)
    accuracy = np.sum(predictions == labels) / len(labels)
    errors = np.sum(np.abs(labels - predictions))
    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "errors": errors,
        "acc": accuracy,
        "precision": precision,
        "recall": recall,
    }


# Load data
data_path = f"{args.data_dir}/{args.split}"
log_print(f"Loading training data from {data_path}")
dataset = load_from_disk(data_path)

num_class_0 = dataset["train"]["label"].count(0)
num_class_1 = dataset["train"]["label"].count(1)
log_print(f"Train set: {num_class_0} negative, {num_class_1} positive")

class_weights = (
    torch.tensor([
        (num_class_0 + num_class_1) / max(num_class_0, 1),
        (num_class_0 + num_class_1) / max(num_class_1, 1),
    ])
    .float()
    .to(device)
)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
log_print(f"Class weights: {class_weights}")


def compute_loss_fn(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    return loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2, device_map=device
)
model.config.hidden_dropout_prob = 0.1
model.config.attention_probs_dropout_prob = 0.1
model.config.classifier_dropout = 0.1
log_print(f"Parameters: {model.num_parameters()}")

save_dir = f"{args.output_dir}/{args.split}"

training_args = TrainingArguments(
    output_dir=save_dir,
    learning_rate=args.learning_rate,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=0.0,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=args.num_train_epochs,
    logging_strategy="steps",
    logging_steps=100,
    do_eval=True,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=10,
    load_best_model_at_end=True,
    metric_for_best_model="test_recall",
    greater_is_better=True,
    report_to="none",
)

eval_dataset_keys = ["forget", "retain", "test"]
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset={
        key: tokenized_datasets[key] for key in eval_dataset_keys
    },
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    compute_loss_func=compute_loss_fn,
)
trainer.train()
trainer.save_model(save_dir)
log_print(f"Model saved to {save_dir}")
