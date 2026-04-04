"""
Tests for transformers 5.x compatibility fixes:

1. Handle-based hook removal in AttackedModel (replaces _forward_hooks assignment)
2. _remove_hooks dispatcher in inference.py
3. compute_loss signature fix in copyright_unlearn_baselines.py
4. compute_loss_func replacement in train_classifier.py
"""

from pathlib import Path

import torch
import torch.nn as nn

from eco.attack.utils import (
    apply_corruption_hook,
    apply_embeddings_extraction_hook,
    remove_hooks,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleEmbedding(nn.Module):
    """Minimal embedding module to attach hooks to."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(100, dim))

    def forward(self, x):
        return x


class FakeInnerModel(nn.Module):
    """Model with a nested embedding, mimicking model.model.embed_tokens."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = SimpleEmbedding()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(self.embed_tokens(x))


# ---------------------------------------------------------------------------
# Issue #3: Handle-based hook removal
# ---------------------------------------------------------------------------


class TestRemoveHooks:
    """Tests for the standalone remove_hooks function in eco/attack/utils.py."""

    def test_remove_hooks_clears_all(self):
        """remove_hooks should clear all forward hooks from all submodules."""
        model = FakeInnerModel()

        # Register hooks on different submodules
        handle1 = model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        handle2 = model.linear.register_forward_hook(lambda m, i, o: o)

        assert len(model.embed_tokens._forward_hooks) == 1
        assert len(model.linear._forward_hooks) == 1

        remove_hooks(model)

        assert len(model.embed_tokens._forward_hooks) == 0
        assert len(model.linear._forward_hooks) == 0

    def test_remove_hooks_preserves_dict_identity(self):
        """remove_hooks should pop entries, not replace the dict object."""
        model = FakeInnerModel()
        original_dict = model.embed_tokens._forward_hooks

        model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        remove_hooks(model)

        # The dict object should be the same (not replaced with a new OrderedDict)
        assert model.embed_tokens._forward_hooks is original_dict

    def test_remove_hooks_noop_when_no_hooks(self):
        """remove_hooks is safe to call when there are no hooks."""
        model = FakeInnerModel()
        remove_hooks(model)  # Should not raise
        assert len(model.embed_tokens._forward_hooks) == 0

    def test_remove_hooks_only_clears_forward_hooks(self):
        """remove_hooks should not affect backward hooks."""
        model = FakeInnerModel()
        bw_handle = model.embed_tokens.register_full_backward_hook(lambda m, gi, go: None)
        fw_handle = model.embed_tokens.register_forward_hook(lambda m, i, o: o)

        remove_hooks(model)

        assert len(model.embed_tokens._forward_hooks) == 0
        assert len(model.embed_tokens._backward_hooks) == 1
        bw_handle.remove()


class TestHandleBasedRemoval:
    """Tests for AttackedModel's handle-tracking approach."""

    def test_apply_corruption_hook_returns_handle(self):
        """apply_corruption_hook should return a valid RemovableHandle."""
        module = SimpleEmbedding()
        pos = [[1, 0, 1]]
        handle = apply_corruption_hook(
            module, "rand_noise_first_n", {"pos": pos, "dims": 1, "strength": 1.0}
        )
        assert len(module._forward_hooks) == 1
        handle.remove()
        assert len(module._forward_hooks) == 0

    def test_apply_embeddings_extraction_hook_returns_handle(self):
        """apply_embeddings_extraction_hook should return a valid RemovableHandle."""
        module = SimpleEmbedding()
        data = []
        handle = apply_embeddings_extraction_hook(module, data)
        assert len(module._forward_hooks) == 1
        handle.remove()
        assert len(module._forward_hooks) == 0

    def test_multiple_handles_independent_removal(self):
        """Multiple hooks can be removed independently via their handles."""
        module = SimpleEmbedding()
        pos = [[1, 0, 1]]
        h1 = apply_corruption_hook(
            module, "rand_noise_first_n", {"pos": pos, "dims": 1, "strength": 1.0}
        )
        h2 = apply_embeddings_extraction_hook(module, [])
        assert len(module._forward_hooks) == 2

        h1.remove()
        assert len(module._forward_hooks) == 1

        h2.remove()
        assert len(module._forward_hooks) == 0

    def test_double_remove_is_safe(self):
        """Removing an already-removed handle should not raise."""
        module = SimpleEmbedding()
        handle = apply_corruption_hook(
            module, "rand_noise_first_n",
            {"pos": [[1, 0]], "dims": 1, "strength": 1.0},
        )
        handle.remove()
        handle.remove()  # Should not raise


class TestAttackedModelHookTracking:
    """Integration tests for AttackedModel._hook_handles lifecycle."""

    def _make_attacked_model(self):
        """Create a minimal AttackedModel with a real embedding module."""
        from eco.attack.model import AttackedModel

        class FakeHFModel:
            def __init__(self):
                self.model_name = "test"
                self.model = FakeInnerModel()
                self.tokenizer = _FakeTokenizer()
                self.model_config = {"attack_module": "embed_tokens"}
                self.device = torch.device("cpu")
                self.generation_config = None

        class _FakePromptClassifier:
            def predict(self, prompts, threshold):
                return [1] * len(prompts)

        hf = FakeHFModel()
        attacked = AttackedModel(
            model=hf,
            prompt_classifier=_FakePromptClassifier(),
            token_classifier=None,
            corrupt_method="rand_noise_first_n",
            corrupt_args={"dims": 1, "strength": 1.0},
        )
        return attacked

    def test_hook_handles_start_empty(self):
        model = self._make_attacked_model()
        assert model._hook_handles == []

    def test_apply_corruption_tracks_handle(self):
        model = self._make_attacked_model()
        model.apply_corruption(["test prompt"])
        assert len(model._hook_handles) == 1
        assert len(model.attack_module._forward_hooks) == 1

    def test_remove_hooks_clears_tracked_handles(self):
        model = self._make_attacked_model()
        model.apply_corruption(["test prompt"])
        assert len(model.attack_module._forward_hooks) == 1

        model.remove_hooks()
        assert len(model._hook_handles) == 0
        assert len(model.attack_module._forward_hooks) == 0

    def test_remove_hooks_then_reapply(self):
        """Simulate the per-batch pattern: remove old hooks, apply new ones."""
        model = self._make_attacked_model()

        # First batch
        model.apply_corruption(["prompt 1"])
        assert len(model._hook_handles) == 1

        # Second batch — remove old, apply new
        model.remove_hooks()
        model.apply_corruption(["prompt 2"])
        assert len(model._hook_handles) == 1
        assert len(model.attack_module._forward_hooks) == 1

    def test_embeddings_extraction_tracked(self):
        model = self._make_attacked_model()
        model.apply_embeddings_extraction()
        assert len(model._hook_handles) == 1

        model.remove_hooks()
        assert len(model._hook_handles) == 0


class _FakeEncoding(dict):
    """Mimics BatchEncoding: supports both dict['key'] and dict.key access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


class _FakeTokenizer:
    """Minimal tokenizer stub for AttackedModel tests."""

    padding_side = "right"

    def __call__(self, text, **kwargs):
        # Return fake token IDs — just split on spaces
        if isinstance(text, str):
            ids = list(range(len(text.split()) + 1))  # +1 for BOS
            return _FakeEncoding({"input_ids": ids})
        return [self(t, **kwargs) for t in text]


# ---------------------------------------------------------------------------
# Issue #3 (continued): _remove_hooks dispatcher in inference.py
# ---------------------------------------------------------------------------


class TestRemoveHooksDispatcher:
    """Tests for the _remove_hooks() function in eco/inference.py."""

    def test_dispatches_to_attacked_model_method(self):
        """_remove_hooks should unwrap ReasoningModel and call AttackedModel.remove_hooks()."""
        from eco.inference import _remove_hooks
        from eco.attack.model import AttackedModel
        from eco.model.reasoning import ReasoningModel

        class FakeHFModel:
            def __init__(self):
                self.model_name = "test"
                self.model = FakeInnerModel()
                self.tokenizer = _FakeTokenizer()
                self.model_config = {"attack_module": "embed_tokens"}
                self.device = torch.device("cpu")
                self.generation_config = None

        hf = FakeHFModel()

        class FakeClassifier:
            def predict(self, prompts, threshold):
                return [1] * len(prompts)

        attacked = AttackedModel(
            model=hf,
            prompt_classifier=FakeClassifier(),
            token_classifier=None,
            corrupt_method="rand_noise_first_n",
            corrupt_args={"dims": 1, "strength": 1.0},
        )

        # Add a hook
        attacked.apply_corruption(["test"])
        assert len(attacked._hook_handles) == 1

        # Wrap in ReasoningModel — _remove_hooks should unwrap and use handle-based method
        reasoning = ReasoningModel(attacked)
        _remove_hooks(reasoning)
        assert len(attacked._hook_handles) == 0
        assert len(attacked.attack_module._forward_hooks) == 0

    def test_dispatches_to_standalone_for_plain_model(self):
        """_remove_hooks should use standalone remove_hooks for non-AttackedModel."""
        from eco.inference import _remove_hooks

        class FakePlainModel:
            def __init__(self):
                self.model = FakeInnerModel()

        plain = FakePlainModel()
        # Manually add a hook to the inner model
        plain.model.embed_tokens.register_forward_hook(lambda m, i, o: o)
        assert len(plain.model.embed_tokens._forward_hooks) == 1

        _remove_hooks(plain)
        assert len(plain.model.embed_tokens._forward_hooks) == 0


# ---------------------------------------------------------------------------
# Issue #1: compute_loss signature
# ---------------------------------------------------------------------------


class TestComputeLossSignature:
    """Verify CustomTrainer.compute_loss accepts num_items_in_batch."""

    def test_copyright_baselines_compute_loss_accepts_kwarg(self):
        """The CustomTrainer in copyright_unlearn_baselines must accept num_items_in_batch."""
        import ast

        with open(_PROJECT_ROOT / "scripts" / "copyright_unlearn_baselines.py") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "CustomTrainer":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "compute_loss":
                        arg_names = [a.arg for a in item.args.args]
                        assert "num_items_in_batch" in arg_names, (
                            "compute_loss must accept num_items_in_batch for transformers 5.x"
                        )
                        return
        raise AssertionError("CustomTrainer.compute_loss not found in file")


# ---------------------------------------------------------------------------
# Issue #4: compute_loss_func replacement
# ---------------------------------------------------------------------------


class TestComputeLossFuncReplacement:
    """Verify train_classifier.py uses compute_loss_func instead of CustomTrainer."""

    def test_no_custom_trainer_class(self):
        """train_classifier.py should not define a CustomTrainer subclass."""
        import ast

        with open(_PROJECT_ROOT / "scripts" / "train_classifier.py") as f:
            tree = ast.parse(f.read())

        class_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        assert "CustomTrainer" not in class_names, (
            "CustomTrainer should be replaced with compute_loss_func"
        )

    def test_compute_loss_fn_defined(self):
        """A compute_loss_fn function should be defined."""
        import ast

        with open(_PROJECT_ROOT / "scripts" / "train_classifier.py") as f:
            tree = ast.parse(f.read())

        func_names = [
            node.name for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]
        assert "compute_loss_fn" in func_names

    def test_no_model_accepts_loss_kwargs_hack(self):
        """The model_accepts_loss_kwargs workaround should be removed."""
        with open(_PROJECT_ROOT / "scripts" / "train_classifier.py") as f:
            content = f.read()
        assert "model_accepts_loss_kwargs" not in content, (
            "model_accepts_loss_kwargs hack should be removed"
        )

    def test_compute_loss_fn_signature(self):
        """compute_loss_fn must accept (outputs, labels, num_items_in_batch=None)."""
        import ast

        with open(_PROJECT_ROOT / "scripts" / "train_classifier.py") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_loss_fn":
                arg_names = [a.arg for a in node.args.args]
                assert "outputs" in arg_names
                assert "labels" in arg_names
                # num_items_in_batch should be a keyword arg with default
                kwarg_names = [a.arg for a in node.args.args]
                assert "num_items_in_batch" in kwarg_names
                return
        raise AssertionError("compute_loss_fn not found")
