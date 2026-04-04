import torch


class ReasoningModel:
    """Wrapper that appends ``<think>\\n`` tokens before delegating to an inner model.

    Works with ``HFModel``, ``AttackedModel``, or any object whose
    ``generate()`` accepts ``input_ids`` and ``attention_mask`` kwargs.
    Attribute access is forwarded to the inner model so that ``.device``,
    ``.tokenizer``, ``.generation_config``, etc. are available directly.
    """

    THINK_PREFIX = "<think>\n"

    def __init__(self, model):
        # Store on the instance dict so __getattr__ is not triggered for it.
        self.__dict__["_inner"] = model
        self.__dict__["_n_think_tokens"] = None

    # ------------------------------------------------------------------
    # Cached property: number of tokens in the think prefix
    # ------------------------------------------------------------------

    @property
    def n_think_tokens(self):
        if self.__dict__["_n_think_tokens"] is None:
            self.__dict__["_n_think_tokens"] = len(
                self._inner.tokenizer(
                    self.THINK_PREFIX, add_special_tokens=False
                )["input_ids"]
            )
        return self.__dict__["_n_think_tokens"]

    # ------------------------------------------------------------------
    # Forward attribute access to the inner model
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # generate(): append think tokens, then delegate
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        # Normalize positional args to kwargs so the modified input_ids
        # is forwarded correctly (we only pass **kwargs to _inner).
        if args:
            kwargs.setdefault("input_ids", args[0])
        input_ids = kwargs.get("input_ids")
        assert input_ids is not None, "input_ids must be provided"

        think_ids = self._inner.tokenizer(
            self.THINK_PREFIX, add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(input_ids.device)

        kwargs["input_ids"] = torch.cat(
            [input_ids, think_ids.expand(input_ids.shape[0], -1)], dim=1
        )

        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, torch.ones_like(think_ids).expand(attention_mask.shape[0], -1)],
                dim=1,
            )

        # Only forward kwargs — positional args were normalized above.
        return self._inner.generate(**kwargs)

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)
