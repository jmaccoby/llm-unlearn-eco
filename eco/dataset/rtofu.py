from datasets import Dataset, DatasetDict, load_dataset

from eco.dataset.base import BaseDataset


class RTOFU(BaseDataset):
    dataset_type = "qa"
    path = "sangyon/R-TOFU"
    name = "rtofu"
    subsets = [
        "full",
        "retain90",
        "retain50",
        "retain10",
        "forget01",
        "forget05",
        "forget10",
    ]
    match_retain = {
        "forget01": "retain90",
        "forget05": "retain90",
        "forget10": "retain90",
    }
    keys = ["prompt", "answer", "prompt_formatted", "cot"]
    eval_prompt_key = "prompt_formatted"
    eval_answer_key = "answer"
    gen_prompt_key = "prompt_formatted"
    gen_answer_key = "answer"
    gen_cot_key = "cot"
    eval_dataset_keys = ["retain", "forget", "test"]

    def __init__(self, formatting_tokens=None, eos_token=None, *args, **kwargs):
        super().__init__()
        self.formatting_tokens = formatting_tokens
        self.eos_token = eos_token if eos_token is not None else ""
        for k in ["prompt_prefix", "prompt_suffix", "answer_prefix", "answer_suffix"]:
            (
                setattr(self, k, formatting_tokens[k])
                if formatting_tokens is not None
                else setattr(self, k, "")
            )

    def download(self):
        data_subsets = {
            s: load_dataset(self.path, split=s, keep_in_memory=True)
            for s in self.subsets
        }
        self.dataset = DatasetDict(data_subsets)

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.rename_column("question", "prompt")
        dataset = dataset.map(
            lambda x: {
                "prompt_formatted": f"{self.prompt_prefix}{x['prompt']}{self.prompt_suffix}",
                "answer": self.answer_prefix + x["answer"] + self.eos_token,
            }
        )
        dataset = dataset.map(
            lambda x: {"prompt_formatted": prompt_prefix + x["prompt_formatted"]}
        )
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self, split_name, use_val=False):
        if self.dataset is None:
            self.download()
        assert (
            split_name in self.subsets and split_name in self.match_retain
        ), f"Invalid split name: {split_name}"
        retain_set_name = self.match_retain[split_name]
        forget_set_name = split_name

        retain_dataset = self.dataset[retain_set_name]
        forget_dataset = self.dataset[forget_set_name]
        full_dataset = self.dataset["full"]

        retain_dataset, forget_dataset, full_dataset = map(
            lambda x: x.rename_column("question", "text").remove_columns(
                [c for c in ["answer", "cot"] if c in x.column_names]
            ),
            [retain_dataset, forget_dataset, full_dataset],
        )

        retain_dataset = retain_dataset.map(lambda x: {"label": 0})
        forget_dataset = forget_dataset.map(lambda x: {"label": 1})
        train_dataset = Dataset.from_dict(
            {
                "text": list(retain_dataset["text"]) + list(forget_dataset["text"]),
                "label": list(retain_dataset["label"]) + list(forget_dataset["label"]),
            }
        )
        if use_val:
            split = train_dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset, val_dataset = split["train"], split["test"]

        general_dataset = full_dataset.map(lambda x: {"label": 0})

        dataset = DatasetDict(
            {
                "train": train_dataset,
                "retain": retain_dataset,
                "forget": forget_dataset,
                "test": general_dataset,
            }
        )
        if use_val:
            dataset["valid"] = val_dataset
        return dataset
