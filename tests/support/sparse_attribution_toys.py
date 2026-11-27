"""Toy models for the sparse-attribution checks of the RTE component, on CPU.

Stands in for the parts of interpretune's ``tests/core/circuit_tracer_toy.py``,
``tests/core/test_circuit_tracer_attention_attribution.py`` and ``tests/core/test_core_loop_training.py`` that these
checks used while they lived in interpretune: a tiny Gemma 2 replacement model with random transcoders, a circuit-tracer
module holding only what the head-attribution ops read, and a one-layer module for the core training loop.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import torch
from torch import nn

TOY_PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


class BaseModelBackendStandIn:
    """The model backend a circuit-tracer composition carries beneath its analysis backend, for a fake module that
    exercises only the analysis backend: it declares the surfaces a full model backend provides (``GRADIENTS``,
    ``LATENT_MODELS``) and implements nothing, so a test that reached it would fail rather than pass on a stand-in."""

    def __init__(self) -> None:
        from interpretune.analysis.backends import Capability, LatentModelSupport

        self.capabilities = frozenset({Capability.GRADIENTS, Capability.LATENT_MODELS})
        self.requires = frozenset()
        self.latent_model_support = LatentModelSupport()


def tiny_gemma2_replacement_model(n_layers: int = 3, seed: int = 670):
    """An ``n_layers``-deep, 8-wide Gemma 2 replacement model; builds and runs on CPU in seconds."""
    from transformers import Gemma2Config

    cfg = Gemma2Config(
        architectures=["Gemma2ForCausalLM"],  # circuit-tracer's nnsight mapping keys on the architecture
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=n_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=16,
        query_pre_attn_scalar=4,
        final_logit_softcapping=None,
        torch_dtype="float32",
    )
    return _tiny_replacement_model(cfg, n_layers, seed)


def _tiny_replacement_model(cfg, n_layers: int, seed: int):
    """An 8-wide nnsight replacement model of ``cfg`` with random weights and random transcoders."""
    from circuit_tracer import ReplacementModel
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
    from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    cfg._name_or_path = "openai-community/gpt2"  # the tokenizer from_config loads; warmed for offline CI
    torch.manual_seed(seed)
    transcoders = {
        layer: SingleLayerTranscoder(8, 32, JumpReLU(torch.tensor(0.0), 0.1), layer) for layer in range(n_layers)
    }
    for transcoder in transcoders.values():
        for param in transcoder.parameters():
            torch.nn.init.uniform_(param, a=-1, b=1)
    transcoder_set = TranscoderSet(transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out")
    model = ReplacementModel.from_config(cfg, transcoder_set, backend="nnsight")
    for param in model.parameters():
        torch.nn.init.uniform_(param, a=-1, b=1)
    for transcoder in transcoders.values():
        torch.nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)
    assert isinstance(model, NNSightReplacementModel)
    return model


@contextmanager
def only_token_zero_special(model) -> Iterator[None]:
    """The gpt2 tokenizer's special ids fall outside the toy vocabulary; treat id 0 as the only special token."""
    tokenizer_class = type(model.tokenizer)
    original = tokenizer_class.all_special_ids
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[assignment]
    try:
        yield
    finally:
        tokenizer_class.all_special_ids = original  # type: ignore[assignment]


FROZEN = {"freeze_attention": True, "freeze_norms": True}


def ct_module(replacement_model, **cfg):
    """A circuit-tracer module holding only what the head-attribution ops read, without an interpretune session."""
    from interpretune.adapters.circuit_tracer import BaseCircuitTracerModule, CircuitTracerAttributeMixin
    from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND
    from interpretune.adapters.circuit_tracer.config import CircuitTracerConfig

    class _CTModule(CircuitTracerAttributeMixin, BaseCircuitTracerModule):
        pass

    module = _CTModule.__new__(_CTModule)
    nn.Module.__init__(module)
    object.__setattr__(module, "it_cfg", SimpleNamespace(circuit_tracer_cfg=CircuitTracerConfig(**cfg)))
    object.__setattr__(module, "_replacement_model", replacement_model)
    object.__setattr__(module, "_analysis_backend", DEFAULT_CT_ANALYSIS_BACKEND)
    object.__setattr__(module, "_model_backend", BaseModelBackendStandIn())
    object.__setattr__(module, "attribution_graphs", [])
    return module


class TinyModule(nn.Module):
    """The surface the core loop uses, over a one-layer model."""

    def __init__(self, lr: float = 1.0, epoch_hooks: bool = True, batch_end_hook: bool = True) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Linear(2, 1)
        self.optimizers = [torch.optim.SGD(self.model.parameters(), lr=lr)]
        self.lr_scheduler_configs: list = []
        self.it_cfg = SimpleNamespace(profiler_cfg=SimpleNamespace(which=None))
        self.current_epoch, self.global_step = 0, 0
        self.dtype, self.device = torch.float32, torch.device("cpu")
        self._logged_metrics: dict[str, list[float]] = {}
        self.calls: list[str] = []
        self.val_grad_modes: list[tuple[bool, bool]] = []
        self.train_autocast: list[torch.dtype | None] = []
        self.val_autocast: list[torch.dtype | None] = []
        self.weight_at_batch: list[float] = []
        if epoch_hooks:
            self.on_train_epoch_start = lambda: self.calls.append("epoch_start")
            self.on_train_epoch_end = lambda: self.calls.append("epoch_end")
        if batch_end_hook:
            self.on_train_batch_end = lambda outputs, batch, batch_idx: None

    def log(self, name, value, *args, **kwargs):
        value = value.detach().item() if isinstance(value, torch.Tensor) else value
        self._logged_metrics.setdefault(name, []).append(value)

    def batch_to_device(self, batch):
        return batch

    @staticmethod
    def _autocast_dtype():
        return torch.get_autocast_dtype("cpu") if torch.is_autocast_enabled("cpu") else None

    def training_step(self, batch, batch_idx):
        self.train_autocast.append(self._autocast_dtype())
        self.weight_at_batch.append(self.model.weight.detach().float().flatten()[0].item())
        loss = self.model(batch).pow(2).mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.val_autocast.append(self._autocast_dtype())
        self.val_grad_modes.append((torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        self.log("val_loss", float(self.current_epoch + batch_idx))


def core_datamodule(n_train: int = 2, n_val: int = 2):
    batches = [torch.ones(1, 2) * (i + 1) for i in range(max(n_train, n_val))]
    return SimpleNamespace(train_dataloader=lambda: batches[:n_train], val_dataloader=lambda: batches[:n_val])


def core_train(module, max_epochs=1, **kwargs):
    from interpretune.runners.core import core_train_loop

    return core_train_loop(module, core_datamodule(**kwargs.pop("dm", {})), -1, -1, max_epochs, **kwargs)
