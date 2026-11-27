"""
RTE BoolQ Experiment
===================

Example experiment configuration demonstrating interpretune framework usage.

Note: This example file uses complex mixin composition, datasets API, and other advanced
patterns that would require extensive type annotations. For maintainability in this
example context, type checking is bypassed.
"""
# pyright: reportMissingParameterType=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportArgumentType=false
# pyright: reportReturnType=false
# pyright: reportIndexIssue=false
# pyright: reportCallIssue=false
# pyright: reportOptionalMemberAccess=false

import os
from typing import Any, Tuple, Callable, Generator
from dataclasses import dataclass, field
from pprint import pformat
import logging
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import evaluate
import datasets
from transformers import PreTrainedTokenizerBase
from datasets.arrow_dataset import LazyDict
from transformers.tokenization_utils_base import BatchEncoding

import interpretune as it
from interpretune import (
    ITDataModule,
    MemProfilerHooks,
    AnalysisBatch,
    ChatTemplatePromptConfig,
    PromptConfig,
    ITDataModuleConfig,
    ITConfig,
    GenerativeClassificationConfig,
    BaseGenerationConfig,
    HFGenerationConfig,
    rank_zero_warn,
    sanitize_input_name,
    STEP_OUTPUT,
)
from interpretune.utils.import_utils import package_available
import math
import json
import hashlib
import tempfile
import re
from typing import Sequence, Mapping, Iterator
from contextlib import AbstractContextManager, contextmanager, ExitStack
from copy import deepcopy
from types import SimpleNamespace

# The adapter-specific config classes below exist only when their adapter is installed. This module is the
# entrypoint of the `rte` seed component, which a hub adapter's conformance run hydrates on a bare core
# install; importing every adapter's config at module level made that import need transformer-lens, sae-lens
# and nnsight at once, and the failure named `ITLensConfig` rather than any of them.
_HAS_TL = package_available("transformer_lens")
_HAS_SL = package_available("sae_lens")
_HAS_CT = package_available("circuit_tracer")
_HAS_NS = package_available("nnsight")


log = logging.getLogger(__name__)

TASK_TEXT_FIELD_MAP = {"rte": ("premise", "hypothesis"), "boolq": ("passage", "question")}
TASK_NUM_LABELS = {"boolq": 2, "rte": 2}
DEFAULT_TASK = "rte"
INVALID_TASK_MSG = f" is an invalid task_name. Proceeding with the default task: {DEFAULT_TASK!r}"


@dataclass(kw_only=True)
class RTEBoolqEntailmentMapping:
    entailment_mapping: Tuple = ("Yes", "No")  # RTE style, invert mapping for BoolQ
    entailment_mapping_indices: torch.Tensor | None = None


@dataclass(kw_only=True)
class RTEBoolqGenerativeClassificationConfig(RTEBoolqEntailmentMapping, GenerativeClassificationConfig):
    enabled: bool = False
    lm_generation_cfg: BaseGenerationConfig = field(default_factory=HFGenerationConfig)

    def __repr__(self):
        return f"Generative Classification Config: {os.linesep}{pformat(self.__dict__)}"


@dataclass(kw_only=True)
class RTEBoolqPromptConfig(PromptConfig):
    ctx_question_join: str = "Does the previous passage imply that "
    question_suffix: str = "? Answer with only one word, either Yes or No."
    cust_task_prompt: dict[str, Any] | None = None  # type: ignore[assignment]  # intentional override for demo


@dataclass(kw_only=True)
class RTEBoolqChatTemplatePromptConfig(ChatTemplatePromptConfig, RTEBoolqPromptConfig):
    """Chat-template-first RTE/BoolQ prompt construction (hub design §11.5).

    The default for chat models whose native template is byte-equivalent to the retired manual
    spellings (gemma2/gemma3 — pinned by ``test_prompt_template_equivalence``). Model families whose
    native template violates provenance (llama3's is date-stamped) instead compose a published manual
    definition via ``compose_ref``.
    """


# add our custom model attributes
@dataclass(kw_only=True)
class RTEBoolqConfig(RTEBoolqEntailmentMapping, ITConfig): ...


if _HAS_TL:
    from interpretune import ITLensConfig

    @dataclass(kw_only=True)
    class RTEBoolqTLConfig(RTEBoolqEntailmentMapping, ITLensConfig): ...


if _HAS_SL:
    from interpretune import SAELensConfig

    @dataclass(kw_only=True)
    class RTEBoolqSLConfig(RTEBoolqEntailmentMapping, SAELensConfig): ...


if _HAS_CT:
    from interpretune import CircuitTracerConfig

    @dataclass(kw_only=True)
    class RTEBoolqCTConfig(RTEBoolqEntailmentMapping, ITConfig):
        circuit_tracer_cfg: CircuitTracerConfig | None = None


if _HAS_NS:
    from interpretune.adapters.nnsight.config import ITNNsightConfig, NNsightConfig

    @dataclass(kw_only=True)
    class RTEBoolqNNsightConfig(RTEBoolqEntailmentMapping, ITNNsightConfig):
        nnsight_cfg: NNsightConfig  # re-declare for jsonargparse fail_untyped=True visibility


def chain_task_prompt(premise: str, hypothesis: str, prompt_cfg: PromptConfig) -> str:
    """The task prompt: the stripped premise, a space, the question, the stripped hypothesis, the suffix.

    The sparsity chain's rendering, and every RTE/BoolQ datamodule's: joining the raw fields to ``ctx_question_join``
    ran a premise ending in a full stop into the question.
    """
    return f"{premise.strip()} {prompt_cfg.ctx_question_join.strip()} {hypothesis.strip()}{prompt_cfg.question_suffix}"


def single_bos(text: str, tokenizer: PreTrainedTokenizerBase) -> str:
    """``text`` without a leading BOS, which the tokenizer adds itself: a chat template's text starts with one."""
    bos = tokenizer.bos_token
    return text[len(bos) :] if bos and text.startswith(bos) else text


class RTEBoolqDataModule(ITDataModule):
    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        if itdm_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(itdm_cfg.task_name + INVALID_TASK_MSG)
            itdm_cfg.task_name = DEFAULT_TASK
        itdm_cfg.text_fields = TASK_TEXT_FIELD_MAP[itdm_cfg.task_name]
        super().__init__(itdm_cfg=itdm_cfg)

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None:
        """Load the SuperGLUE dataset."""
        # N.B. prepare_data is called in a single process (rank 0, either per node or globally) so do not use it to
        # assign state (e.g. self.x=y)
        # note for raw pytorch we require a target_model
        # NOTE [HF Datasets Transformation Caching]:
        # HF Datasets' transformation cache fingerprinting algo necessitates construction of these partials as the hash
        # is generated using function args, dataset file, mapping args: https://bit.ly/HF_Datasets_fingerprint_algo)
        # chat-template-first configs delegate to the tokenizer; bind it before template_fn is captured
        if hasattr(self.itdm_cfg.prompt_cfg, "bind_tokenizer"):
            self.itdm_cfg.prompt_cfg.bind_tokenizer(self.tokenizer)
        tokenization_func = partial(
            self.encode_for_rteboolq,
            tokenizer=self.tokenizer,
            text_fields=self.itdm_cfg.text_fields,
            prompt_cfg=self.itdm_cfg.prompt_cfg,
            template_fn=self.itdm_cfg.prompt_cfg.model_chat_template_fn,
            tokenization_pattern=self.itdm_cfg.cust_tokenization_pattern,
            max_seq_length=self.itdm_cfg.max_seq_length,
        )
        dataset = datasets.load_dataset("aps/super_glue", self.itdm_cfg.task_name)
        for split in dataset.keys():
            dataset[split] = dataset[split].map(tokenization_func, **self.itdm_cfg.prepare_data_map_cfg)
            dataset[split] = self._remove_unused_columns(dataset[split], target_model)

        self.save_prepared_dataset(dataset)

    def dataloader_factory(self, split: str, use_train_batch_size: bool = False) -> DataLoader:
        dataloader_kwargs = {
            "dataset": self.dataset[split],
            "collate_fn": self.data_collator,
            **self.itdm_cfg.dataloader_kwargs,
        }
        if use_train_batch_size:  # training reads a new order each epoch, from the seeded RNG, unless a config says not
            dataloader_kwargs.setdefault("shuffle", True)
        else:  # evaluation splits score every example, in order
            dataloader_kwargs.update(shuffle=False, drop_last=False)
        dataloader_kwargs["batch_size"] = (
            self.itdm_cfg.train_batch_size if use_train_batch_size else self.itdm_cfg.eval_batch_size
        )
        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="train", use_train_batch_size=True)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    def test_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="validation")

    @staticmethod
    def fit_sequences(
        sequences: list[str],
        pairs: list[tuple[str, str]],
        sequence_for: Callable[[str, str], str],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> list[str]:
        """Each sequence within ``max_seq_length`` tokens, by trimming the end of its context field.

        Truncating the encoded sequence would cut the chat template's header or the question and answer cue, which
        changes what is asked, so only the context (premise or passage) gives way. A sequence that cannot fit even with
        no context raises.
        """
        lengths = [len(ids) for ids in tokenizer(sequences)["input_ids"]]
        fitted = list(sequences)
        for i, n in enumerate(lengths):
            if n <= max_seq_length:
                continue
            context, question = pairs[i]
            context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            # re-encoding a decoded prefix can merge or split a token at the cut, so re-measure and trim again
            for _ in range(4):
                if n <= max_seq_length:
                    break
                keep = len(context_ids) - (n - max_seq_length)
                if keep <= 0:
                    raise ValueError(
                        f"Example {i} needs {n - len(context_ids)} tokens without its context, more than"
                        f" `max_seq_length` ({max_seq_length})."
                    )
                context_ids = context_ids[:keep]
                fitted[i] = sequence_for(tokenizer.decode(context_ids), question)
                n = len(tokenizer(fitted[i])["input_ids"])
            if n > max_seq_length:
                raise ValueError(f"Example {i} could not be trimmed to `max_seq_length` ({max_seq_length}).")
        return fitted

    # TODO: relax PreTrainedTokenizerBase to the protocol that is actually required
    @staticmethod
    def encode_for_rteboolq(
        example_batch: LazyDict,
        tokenizer: PreTrainedTokenizerBase,
        text_fields: list[str],
        prompt_cfg: PromptConfig,
        template_fn: Callable,
        tokenization_pattern: str | None = None,
        max_seq_length: int | None = None,
    ) -> BatchEncoding:
        def sequence_for(field1: str, field2: str) -> str:
            # TODO: use promptsource instead of this manual approach after tinkering
            if prompt_cfg.cust_task_prompt:
                task_prompt = (
                    prompt_cfg.cust_task_prompt["context"]
                    + "\n"
                    + field1
                    + "\n"
                    + prompt_cfg.cust_task_prompt["question"]
                    + "\n"
                    + field2
                )
            else:
                task_prompt = chain_task_prompt(field1, field2, prompt_cfg)
            return single_bos(template_fn(task_prompt=task_prompt, tokenization_pattern=tokenization_pattern), tokenizer)

        pairs = list(zip(example_batch[text_fields[0]], example_batch[text_fields[1]]))
        example_batch["sequences"] = [sequence_for(field1, field2) for field1, field2 in pairs]
        if max_seq_length is not None:
            example_batch["sequences"] = RTEBoolqDataModule.fit_sequences(
                example_batch["sequences"], pairs, sequence_for, tokenizer, max_seq_length
            )
        features = tokenizer(example_batch["sequences"], padding="longest", padding_side=tokenizer.padding_side)
        features["labels"] = example_batch["label"]  # Rename label to labels, easier to pass to model forward
        features = sanitize_input_name(tokenizer.model_input_names, features)
        return features


class RTEBoolqSteps:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # when using TransformerLens, we need to manually calculate our loss from logit output
        self.loss_fn = CrossEntropyLoss()

    @MemProfilerHooks.memprofilable
    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        # TODO: need to be explicit about the compatibility constraints/contract
        # TODO: note that this example uses generative_step_cfg and lm_head except for the test_step where we demo how
        # to use the GenerativeMixin to run inference with or without a generative_step_cfg enabled as well as with
        # different heads (e.g., seqclassification or LM head in this case)
        answer_logits, labels, _ = self.logits_and_labels(batch, batch_idx)
        loss = self.loss_fn(answer_logits, labels)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    @MemProfilerHooks.memprofilable
    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        answer_logits, labels, orig_labels = self.logits_and_labels(batch, batch_idx)
        val_loss = self.loss_fn(answer_logits, labels)
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
        self.collect_answers(answer_logits, orig_labels)

    @MemProfilerHooks.memprofilable
    def test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        if self.it_cfg.generative_step_cfg.enabled:
            self.generative_classification_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)
        else:
            self.default_test_step(batch, batch_idx, dataloader_idx=dataloader_idx)

    def generative_classification_test_step(
        self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0
    ) -> STEP_OUTPUT | None:
        labels = batch.pop("labels")
        outputs = self.it_generate(batch, **self.it_cfg.generative_step_cfg.lm_generation_cfg.generate_kwargs)
        # We expect a HF ModelOutput with a `.logits` attribute when `output_logits` is used.
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
        else:
            raise ValueError("Expected ModelOutput with `logits` or a logits tensor from generate()")
        self.collect_answers(logits, labels)

    def default_test_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        labels = batch.pop("labels")
        outputs = self(**batch)
        self.collect_answers(outputs.logits, labels)

    def predict_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        labels = batch.pop("labels")
        outputs = self(**batch)
        return self.collect_answers(outputs, labels, mode="return")

    def analysis_step(
        self,
        batch: BatchEncoding,
        batch_idx: int,
        dataloader_idx: int = 0,
        analysis_batch: AnalysisBatch | None = None,
    ) -> Generator[STEP_OUTPUT, None, None]:
        """Run analysis operations on a batch and yield results."""
        # Demo mixing model methods and native IT analysis ops
        label_ids, orig_labels = self.labels_to_ids(batch.pop("labels"))
        analysis_batch = AnalysisBatch(label_ids=label_ids, orig_labels=orig_labels)
        op_kwargs = {"module": self, "batch": batch, "batch_idx": batch_idx}
        analysis_batch = it.model_fwd_w_cache_latent_models(analysis_batch=analysis_batch, **op_kwargs)  # type: ignore[assignment]  # protocol compatibility
        analysis_batch = it.logit_diffs_cache(analysis_batch=analysis_batch, **op_kwargs)  # type: ignore[assignment]  # protocol compatibility
        analysis_batch = it.latent_correct_acts(analysis_batch=analysis_batch, **op_kwargs)  # type: ignore[assignment]  # protocol compatibility

        # note, there is an equivalent existing composite op for the decomposed version above:
        # analysis_batch = it.logit_diffs_latent(**op_kwargs)
        yield from self.analysis_cfg.save_batch(analysis_batch, batch, tokenizer=self.datamodule.tokenizer)


class RTEBoolqModuleMixin:
    def setup(self, *args, **kwargs) -> None:
        super().setup(*args, **kwargs)
        self._init_entailment_mapping()

    def _before_it_cfg_init(self, it_cfg: ITConfig) -> ITConfig:
        if it_cfg.task_name not in TASK_NUM_LABELS.keys():
            rank_zero_warn(it_cfg.task_name + INVALID_TASK_MSG)
            it_cfg.task_name = DEFAULT_TASK
        it_cfg.num_labels = 0 if it_cfg.generative_step_cfg.enabled else TASK_NUM_LABELS[it_cfg.task_name]
        return it_cfg

    def load_metric(self) -> None:
        self.metric = evaluate.load(
            "super_glue", self.it_cfg.task_name, experiment_id=self._it_state._init_hparams["experiment_id"]
        )

    # we override the default labels_to_ids method to demo using our module-specific attributes/logic
    def labels_to_ids(self, labels: torch.Tensor | list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.as_tensor(labels, device=self.it_cfg.entailment_mapping_indices.device)
        return torch.take(self.it_cfg.entailment_mapping_indices, labels), labels

    # We override the default standardize_logits method to demo using custom attributes etc.
    def standardize_logits(self, logits: torch.Tensor) -> torch.Tensor:
        # to support genclassif/non-genclassif configs and LM/SeqClassification heads we adhere to the following logits
        # logical shape invariant: [batch size, positions to consider, answers to consider]
        if isinstance(logits, tuple):
            logits = torch.stack([out for out in logits], dim=1)
        logits = logits.to(device=self.device)
        if logits.ndim == 2:  # if answer logits have already been squeezed
            logits = logits.unsqueeze(1)
        if logits.shape[-1] != self.it_cfg.num_labels:
            logits = torch.index_select(logits, -1, self.it_cfg.entailment_mapping_indices)
            if not self.it_cfg.generative_step_cfg.enabled:
                logits = logits[:, -1:, :]
        return logits

    def _init_entailment_mapping(self) -> None:
        ent_cfg, tokenizer = self.it_cfg, self.datamodule.tokenizer
        token_ids = tokenizer.convert_tokens_to_ids(ent_cfg.entailment_mapping)
        device = self.device if isinstance(self.device, torch.device) else self.output_device
        ent_cfg.entailment_mapping_indices = torch.tensor(token_ids).to(device)


class RTEBoolqModule(RTEBoolqSteps, RTEBoolqModuleMixin, torch.nn.Module): ...


# -- latent tuning of a model too large for one card, sharded with FSDP2 (interpretune#882) --------------------------
#
# The large-model recipe of interpretune#880 as configuration: LoRA on the first blocks, then the decoders of Gemma
# Scope 2 transcoders spliced over two blocks' MLPs, on a Gemma 3 built on the meta device and sharded with FSDP2
# (`runs/rte_latent.gemma3_4b.core.fsdp2.yaml`, `runs/rte_latent.gemma3_27b.core.fsdp2.yaml`). The plan, the offload and
# the mixed precision policy are the run's; only the transcoders need code, because they are attached beside the model.


@dataclass(kw_only=True)
class RTEBoolqLatentCfg:
    """A module configuration mixin carrying the spliced transcoders' settings (``latent_tuning``).

    Keys: ``layers`` (the decoder layers whose MLPs a transcoder replaces) and ``transcoders``, either a published
    Gemma Scope 2 set (``{"repo": ..., "variant": ...}``) or ``{"random": {"d_transcoder": n, "seed": s}}`` for tests.
    """

    latent_tuning: dict[str, Any] = field(default_factory=dict)


class GemmaScopeTranscoder(torch.nn.Module):
    """A Gemma Scope 2 JumpReLU transcoder (``params.safetensors``), from the pre-feedforward-norm output $x$:

    - $a = (x W_{enc} + b_{enc}) \\odot [x W_{enc} + b_{enc} \\gt \\theta]$ are the latents;
    - $\\hat y = a W_{dec} + b_{dec} + x A$ predicts the post-feedforward-norm output, with $A$ the affine skip.

    The weights are held in fp32 whatever the model's dtype, and each call computes in fp32. ``encode`` and ``decode`` go
    through ``forward`` because FSDP2 gathers a unit's weights in its forward hooks, and interpretune's FSDP2 memory
    profiling instruments only those.
    """

    NAMES = ("w_enc", "b_enc", "threshold", "w_dec", "b_dec", "affine_skip_connection")

    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name in self.NAMES:
            setattr(self, name, torch.nn.Parameter(weights[name].float(), requires_grad=False))
        self._skip: torch.Tensor | None = None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self(x=x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self(z=z)

    def forward(self, x: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        if x is not None:
            x = x.to(self.w_enc.dtype)
            self._skip = x @ self.affine_skip_connection
            pre = x @ self.w_enc + self.b_enc
            return pre * (pre > self.threshold)
        skip, self._skip = self._skip, None
        return z @ self.w_dec + self.b_dec + skip


def load_latent_transcoder(layer: int, d_model: int, spec: dict[str, Any]) -> GemmaScopeTranscoder:
    """``layer``'s transcoder, on the CPU: a published one, or a random one of the same shape."""
    if "random" in spec:
        d_tc = int(spec["random"]["d_transcoder"])
        gen = torch.Generator().manual_seed(int(spec["random"].get("seed", 0)) + layer)
        weights = {
            "w_enc": torch.randn(d_model, d_tc, generator=gen) / d_model**0.5,
            "b_enc": torch.zeros(d_tc),
            "threshold": torch.zeros(d_tc),
            "w_dec": torch.randn(d_tc, d_model, generator=gen) / d_tc**0.5,
            "b_dec": torch.zeros(d_model),
            "affine_skip_connection": torch.zeros(d_model, d_model),
        }
        return GemmaScopeTranscoder(weights)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    filename = f"transcoder_all/layer_{layer}_{spec['variant']}/params.safetensors"
    return GemmaScopeTranscoder(load_file(hf_hub_download(spec["repo"], filename)))


class RTEBoolqLatentMixin:
    """Splices the configured transcoders over their layers' MLPs, as latent models ``tc<layer>``.

    Each replaces the output of its block's MLP (the post-feedforward norm's output, as the set was trained) with its
    reconstruction, so the model runs on what the latents capture. A schedule selects them by role (``role:latent``) or
    by name (``latent_models.tc<layer>.<weight>``).
    """

    def register_tunable_modules(self) -> None:
        from interpretune.tuning.latents import LatentSite, attach_latent_model

        super().register_tunable_modules()
        cfg = getattr(self.it_cfg, "latent_tuning", None) or {}
        if not cfg.get("layers"):
            raise ValueError("`latent_tuning` names no `layers` to splice a transcoder at")
        stacks = [n for n, m in self.named_modules() if n.endswith("language_model") and hasattr(m, "layers")]
        if len(stacks) != 1:
            raise ValueError(f"expected one Gemma 3 text model to splice transcoders into, found {stacks}")
        d_model = self.model.config.get_text_config().hidden_size
        for layer in cfg["layers"]:
            block = f"{stacks[0]}.layers.{layer}"
            site = LatentSite(
                f"{block}.mlp", write=f"{block}.post_feedforward_layernorm", read_from="input", op="replace"
            )
            attach_latent_model(self, f"tc{layer}", load_latent_transcoder(layer, d_model, cfg["transcoders"]), site)


class RTEBoolqLatentModule(RTEBoolqLatentMixin, RTEBoolqSteps, RTEBoolqModuleMixin, torch.nn.Module): ...


# -- the sparsity chain (interpretune#698 to #700) as one phased run (interpretune#899) ------------------------------
#
# Phase 0 tunes the final norm and blocks 14-25 of gemma-3-1b-it against a two-way cross-entropy on the Yes/No logits
# (at a temperature) plus an attribution-sparsity penalty on published transcoders' features at the answer position:
# with $y$ the Yes-minus-No gap and $a_i$ a feature's activation, the attribution is $s_i = a_i (W_{dec} g)_i$ where
# $g = \partial y / \partial \mathrm{out}$ is taken through the live model and detached, so the penalty acts on the
# activations only. The penalty is the $\ell_1 / \ell_2$ ratio of $s$ over the regularized layers, divided by its value
# on the untuned model, plus a weight on the share of attribution the transcoders' error terms carry. Phase 1 tunes with
# the simplified graph enforced: features outside the boundary's pruned graphs are held at template-aligned means, which
# shifts the MLP output by $(\mu - a) W_{dec}$ over the dropped features. The phase payloads (``phase_config``) carry
# the loss weights and the enforcement switch; the hooks and gates below are reachable by ``class_path``
# (``rte_boolq.<name>``) from finetuning-scheduler's ``gates`` and ``phase_hooks``.

import random  # noqa: E402

import torch.nn.functional as F  # noqa: E402

CHAIN_SHUFFLE_SEED = 424  # the seeded order of the RTE train rows the chain's analyses share (rte_lens/prompts.py)
CHAIN_ROWS = {"train": ((0, 860), (1300, 2300)), "validation": ((2300, None),), "graph": ((860, 1300),)}


@dataclass(kw_only=True)
class RTEBoolqSparseChainCfg:
    """A module configuration mixin carrying the sparsity chain's static settings (``sparse_chain``).

    Keys: ``reg_layers`` (regularized layers), ``enforce_layers`` (default ``reg_layers``), ``loss_temperature``,
    ``transcoders`` (``{"repo": ..., "subfolder": ...}`` for a circuit-tracer transcoder set, or ``{"random":
    {"d_transcoder": n, "seed": s, "threshold": t}}`` for tests), ``sites`` (``x`` and ``out``: module names under the
    model, with ``{layer}``, whose outputs are each regularized MLP's input and output), ``max_prefix`` and
    ``max_suffix`` (the enforcement buffers' template capacity).
    """

    sparse_chain: dict[str, Any] = field(default_factory=dict)


SPARSE_CHAIN_DEFAULTS = {
    "reg_layers": list(range(17, 26)),
    "enforce_layers": None,
    "loss_temperature": 4.0,
    "transcoders": {"repo": "mwhanna/gemma-scope-2-1b-it", "subfolder": "transcoder_all/width_16k_l0_small_affine"},
    "sites": {
        "x": "model.layers.{layer}.pre_feedforward_layernorm",
        "out": "model.layers.{layer}.post_feedforward_layernorm",
    },
    "max_prefix": 16,
    "max_suffix": 48,
}


class RTEBoolqChainDataModule(RTEBoolqDataModule):
    """RTE with the sparsity chain's row partition and prompt rendering, so graph items and the template stay
    comparable with the chain's results.

    The RTE train rows in the chain's seeded order split into ``train`` (rows 0-859 and 1300-2299), ``validation``
    (rows 2300 on: the dev set driving the schedule) and ``graph`` (rows 860-1299: the graph and reference items, held
    out of training); ``test`` is the 277 RTE validation items. Each prompt is one chat-templated user turn tokenized
    with a single BOS, as the chain encoded it. RTE only.
    """

    def __init__(self, itdm_cfg: ITDataModuleConfig) -> None:
        super().__init__(itdm_cfg=itdm_cfg)
        if self.itdm_cfg.task_name != "rte":
            raise ValueError(f"the chain's partition is defined for RTE, not {self.itdm_cfg.task_name!r}")
        path = Path(self.itdm_cfg.dataset_path)
        if not path.name.endswith("_chain"):  # never share the plain partition's prepared dataset
            self.itdm_cfg.dataset_path = path.with_name(path.name + "_chain")

    @staticmethod
    def chain_splits(dataset: datasets.DatasetDict) -> datasets.DatasetDict:
        order = list(range(len(dataset["train"])))
        random.Random(CHAIN_SHUFFLE_SEED).shuffle(order)
        splits = {
            name: dataset["train"].select([i for lo, hi in spans for i in order[lo:hi]])
            for name, spans in CHAIN_ROWS.items()
        }
        splits["test"] = dataset["validation"]
        return datasets.DatasetDict(splits)

    def prepare_data(self, target_model: torch.nn.Module | None = None) -> None:
        prompt_cfg = self.itdm_cfg.prompt_cfg
        if hasattr(prompt_cfg, "bind_tokenizer"):
            prompt_cfg.bind_tokenizer(self.tokenizer)
        encode = partial(self.encode_for_chain, tokenizer=self.tokenizer, prompt_cfg=prompt_cfg)
        dataset = self.chain_splits(datasets.load_dataset("aps/super_glue", "rte"))
        for split in dataset.keys():
            dataset[split] = dataset[split].map(encode, **self.itdm_cfg.prepare_data_map_cfg)
            dataset[split] = self._remove_unused_columns(dataset[split], target_model)
        self.save_prepared_dataset(dataset)

    @staticmethod
    def encode_for_chain(
        example_batch: LazyDict, tokenizer: PreTrainedTokenizerBase, prompt_cfg: PromptConfig
    ) -> BatchEncoding:
        sequences = []
        for premise, hypothesis in zip(example_batch["premise"], example_batch["hypothesis"]):
            text = prompt_cfg.model_chat_template_fn(task_prompt=chain_task_prompt(premise, hypothesis, prompt_cfg))
            sequences.append(single_bos(text, tokenizer))
        features = tokenizer(sequences, padding="longest", padding_side=tokenizer.padding_side)
        features["labels"] = example_batch["label"]
        return sanitize_input_name(tokenizer.model_input_names, features)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="test")

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="test")

    def graph_dataloader(self) -> DataLoader:
        return self.dataloader_factory(split="graph")


def answer_positions(mask: torch.Tensor) -> torch.Tensor:
    """Each item's last real position, under left or right padding."""
    return mask.shape[1] - 1 - mask.flip(1).long().argmax(1)


def _random_transcoders(layers, d_model, spec, device):
    from circuit_tracer.transcoder.activation_functions import JumpReLU
    from circuit_tracer.transcoder.single_layer_transcoder import SingleLayerTranscoder

    gen, d_tc = torch.Generator().manual_seed(int(spec.get("seed", 0))), int(spec["d_transcoder"])
    out = {}
    for ly in layers:
        thr = torch.full((d_tc,), float(spec.get("threshold", 0.0)))
        tc = SingleLayerTranscoder(d_model, d_tc, JumpReLU(thr), ly, device=device, dtype=torch.float32)
        with torch.no_grad():
            tc.W_enc.copy_(torch.randn(tc.W_enc.shape, generator=gen) / d_model**0.5)
            tc.W_dec.copy_(torch.randn(tc.W_dec.shape, generator=gen) / d_tc**0.5)
            tc.b_enc.zero_()
            tc.b_dec.zero_()
        out[ly] = tc
    return out


def load_chain_transcoders(layers, d_model, spec, device):
    """The regularized layers' transcoders, loaded as circuit-tracer's ReplacementModel loads the published set."""
    if "random" in spec:
        return _random_transcoders(layers, d_model, spec["random"], device)
    import yaml
    from huggingface_hub import hf_hub_download
    from circuit_tracer.transcoder.single_layer_transcoder import load_transcoder

    repo, sub = spec["repo"], spec["subfolder"]
    cfg = yaml.safe_load(open(hf_hub_download(repo, "config.yaml", subfolder=sub)))
    if cfg.get("model_kind") != "transcoder_set" or "transcoders" in cfg:
        raise ValueError(f"unexpected transcoder layout for {repo}/{sub}: {sorted(cfg)}")
    return {
        ly: load_transcoder(
            hf_hub_download(repo, f"layer_{ly}.safetensors", subfolder=sub),
            ly,
            device=device,
            dtype=torch.float32,
            lazy_encoder=False,
            lazy_decoder=False,
        )
        for ly in layers
    }


class ChainNormalizer:
    """The untuned penalty the sparsity term is divided by: run-fitted state, saved with every checkpoint."""

    def __init__(self, value: float = 1.0) -> None:
        self.value = value

    def state_dict(self) -> dict:
        return {"value": self.value}

    def load_state_dict(self, state: dict) -> None:
        self.value = state["value"]


class RTEBoolqSparseMixin:
    """The sparsity chain's losses over an RTE module whose model is a Hugging Face causal LM.

    The transcoders are registered as ``sparse_tcs`` with the ``instrument`` role (never trained, never checkpointed)
    and the enforcement state as ``sparse_enforce_*`` buffers with the ``fitted`` role (checkpointed). Training and
    validation log ``<split>_task`` (the phase-invariant two-way cross-entropy at the loss temperature, the monitor),
    ``<split>_loss`` (the phase's whole objective), ``<split>_pen_raw``, ``<split>_err_share``, ``<split>_ratio``,
    ``<split>_active`` and ``<split>_acc``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sparse_reg_weight, self.sparse_err_weight, self.sparse_enforce = 0.0, 0.0, False
        self.sparse_normalizer = ChainNormalizer()
        self._sparse_saved: dict = {}
        self._sparse_mask: torch.Tensor | None = None

    @property
    def sparse_cfg(self) -> dict:
        cfg = {**SPARSE_CHAIN_DEFAULTS, **(getattr(self.it_cfg, "sparse_chain", None) or {})}
        cfg["enforce_layers"] = list(cfg["enforce_layers"] or cfg["reg_layers"])
        cfg["reg_layers"] = list(cfg["reg_layers"])
        return cfg

    def register_tunable_modules(self) -> None:
        super().register_tunable_modules()
        cfg = self.sparse_cfg
        layers = sorted(set(cfg["reg_layers"]) | set(cfg["enforce_layers"]))
        device = next(self.model.parameters()).device
        d_model = self.model.config.get_text_config().hidden_size
        tcs = load_chain_transcoders(layers, d_model, cfg["transcoders"], device)
        self.sparse_tcs = torch.nn.ModuleDict({str(ly): tc.requires_grad_(False) for ly, tc in tcs.items()})
        d_tc, n_pre, n_suf = tcs[layers[0]].d_transcoder, cfg["max_prefix"], cfg["max_suffix"]
        self.register_buffer("sparse_enforce_template", torch.zeros(2, dtype=torch.long, device=device))
        self.register_buffer("sparse_enforce_pre_ids", torch.zeros(n_pre, dtype=torch.long, device=device))
        self.register_buffer("sparse_enforce_suf_ids", torch.zeros(n_suf, dtype=torch.long, device=device))
        for ly in cfg["enforce_layers"]:
            for region, shape in (("pre", (n_pre, d_tc)), ("suf", (n_suf, d_tc)), ("con", (d_tc,))):
                self.register_buffer(f"sparse_enforce_mean_{region}_{ly}", torch.zeros(shape, device=device))
                self.register_buffer(
                    f"sparse_enforce_keep_{region}_{ly}", torch.ones(shape, dtype=torch.bool, device=device)
                )
        for ly in layers:
            self.model.get_submodule(cfg["sites"]["x"].format(layer=ly)).register_forward_hook(
                self._sparse_keep(ly, "x")
            )
            self.model.get_submodule(cfg["sites"]["out"].format(layer=ly)).register_forward_hook(
                self._sparse_keep(ly, "out")
            )
        # registered after the keep hooks, so the saved ``out`` is the MLP's own output; the enforced output differs
        # from it by a term that depends on the MLP input only, so gradients at ``out`` and error terms are unchanged
        for ly in cfg["enforce_layers"]:
            self.model.get_submodule(cfg["sites"]["out"].format(layer=ly)).register_forward_hook(
                self._sparse_enforce(ly)
            )

    def tunable_roles(self) -> dict[str, str]:
        roles = dict(super().tunable_roles())
        roles.update({n: "instrument" for n, _ in self.named_parameters() if n.startswith("sparse_tcs.")})
        roles.update({n: "fitted" for n, _ in self.named_buffers() if n.startswith("sparse_enforce_")})
        return roles

    def _sparse_keep(self, ly, key):
        def hook(_mod, _inp, out):
            out = out[0] if isinstance(out, tuple) else out
            # with every block below a regularized layer frozen, its MLP output is outside autograd; the penalty still
            # needs d gap / d out there, so track it as a leaf (a no-op whenever it is already tracked)
            if key == "out" and torch.is_grad_enabled() and not out.requires_grad:
                out.requires_grad_(True)
            self._sparse_saved[(ly, key)] = out

        return hook

    @staticmethod
    def _frozen(tc):
        act = tc.activation_function
        if not hasattr(act, "threshold"):
            raise TypeError(f"layer {tc.layer_idx}: expected a JumpReLU transcoder, got {act!r}")
        skip = tc.W_skip.detach() if getattr(tc, "W_skip", None) is not None else None
        return tc.W_enc.detach(), tc.b_enc.detach(), act.threshold.detach(), tc.W_dec.detach(), tc.b_dec.detach(), skip

    def _sparse_enforce(self, ly):
        def hook(_mod, _inp, out):
            if not self.sparse_enforce:
                return None
            mask, x = self._sparse_mask, self._sparse_saved[(ly, "x")]
            if mask is None or tuple(mask.shape) != tuple(x.shape[:2]):
                raise RuntimeError(f"layer {ly}: enforcement needs the current attention mask, got {mask!r}")
            W_enc, b_enc, thr, W_dec, _, _ = self._frozen(self.sparse_tcs[str(ly)])
            with torch.autocast(x.device.type, enabled=False):
                pre = F.linear(x.float(), W_enc, b_enc)
                a = pre * (pre > thr)
                mean, keep = self._sparse_targets(ly, mask)
                # decode is linear in the activations, so holding dropped features at their means shifts the output by
                # (mean - a) @ W_dec on those features; the bias, the skip path and the error term are untouched
                delta = torch.where(keep, 0.0, mean - a) @ W_dec
            return (out.float() + delta).to(out.dtype)

        return hook

    def _sparse_targets(self, ly, mask):
        """Per item and position, the mean each feature is held at and whether it is kept (padding: kept)."""
        n_pre, n_suf = (int(v) for v in self.sparse_enforce_template)
        get = lambda kind, region: getattr(self, f"sparse_enforce_{kind}_{region}_{ly}")  # noqa: E731
        B, T = mask.shape
        d = get("mean", "con").shape[0]
        mean = get("mean", "con").expand(B, T, d).clone()
        keep = get("keep", "con").expand(B, T, d).clone()
        last = answer_positions(mask)
        for b, n in enumerate(mask.sum(1).tolist()):
            end = int(last[b]) + 1
            start = end - n
            mean[b, start : start + n_pre], mean[b, end - n_suf : end] = (
                get("mean", "pre")[:n_pre],
                get("mean", "suf")[:n_suf],
            )
            keep[b, start : start + n_pre], keep[b, end - n_suf : end] = (
                get("keep", "pre")[:n_pre],
                get("keep", "suf")[:n_suf],
            )
            keep[b, :start], keep[b, end:] = True, True
        return mean, keep

    def check_template(self, ids, mask):
        """Refuse an item whose prompt does not share the chat template the enforcement is aligned to."""
        n_pre, n_suf = (int(v) for v in self.sparse_enforce_template)
        pre, suf = self.sparse_enforce_pre_ids[:n_pre], self.sparse_enforce_suf_ids[:n_suf]
        last = answer_positions(mask)
        for b, n in enumerate(mask.sum(1).tolist()):
            end = int(last[b]) + 1
            if not (
                torch.equal(ids[b, end - n : end - n + n_pre], pre) and torch.equal(ids[b, end - n_suf : end], suf)
            ):
                raise ValueError(f"batch item {b} does not share the enforcement template ({n_pre}, {n_suf})")

    def sparse_forward(self, batch):
        """The answer position's Yes/No logits and the answer positions, saving each regularized site."""
        batch = {k: v for k, v in batch.items() if k != "labels"}
        self._sparse_mask = batch["attention_mask"]
        if self.sparse_enforce:
            self.check_template(batch["input_ids"], batch["attention_mask"])
        out = self(**batch)
        logits = out if isinstance(out, torch.Tensor) else out.logits
        z = logits[:, -1].index_select(-1, self.it_cfg.entailment_mapping_indices).float()
        return z, answer_positions(batch["attention_mask"])

    def sparse_features(self, z, last):
        """Per item: the regularized features' activations, their attributions to the gap, and each layer's error-term
        attribution (the MLP output minus the transcoder's reconstruction, dotted with the same gradient)."""
        layers = self.sparse_cfg["reg_layers"]
        rows = torch.arange(z.shape[0], device=z.device)
        outs = [self._sparse_saved[(ly, "out")] for ly in layers]
        grads = torch.autograd.grad((z[:, 0] - z[:, 1]).sum(), outs, retain_graph=True)
        a_all, s_all, e_all = [], [], []
        with torch.autocast(z.device.type, enabled=False):
            for ly, g in zip(layers, grads):
                W_enc, b_enc, thr, W_dec, b_dec, W_skip = self._frozen(self.sparse_tcs[str(ly)])
                x = self._sparse_saved[(ly, "x")][rows, last].float()
                gl = g[rows, last].detach().float()
                pre = F.linear(x, W_enc, b_enc)
                a = pre * (pre > thr)
                recon = a @ W_dec + b_dec + (x @ W_skip.T if W_skip is not None else 0)
                err = self._sparse_saved[(ly, "out")][rows, last].float() - recon
                a_all.append(a)
                s_all.append(a * (gl @ W_dec.T))
                e_all.append((err * gl).sum(-1))
        return torch.cat(a_all, -1), torch.cat(s_all, -1), torch.stack(e_all, -1)

    @staticmethod
    def sparse_ratio(s):
        mag = s.abs()
        return (mag.sum(-1) / mag.norm(dim=-1).clamp_min(1e-8)).mean()

    def sparse_step(self, batch, split):
        labels = torch.as_tensor(batch["labels"], device=self.it_cfg.entailment_mapping_indices.device)
        z, last = self.sparse_forward(batch)
        task = F.cross_entropy(z / self.sparse_cfg["loss_temperature"], labels)
        a, s, s_err = self.sparse_features(z, last)
        raw = self.sparse_ratio(s)
        err_mass = s_err.abs().sum(-1)
        err_share = (err_mass / (s.abs().sum(-1) + err_mass).clamp_min(1e-8)).mean()
        loss = task + self.sparse_reg_weight * raw / self.sparse_normalizer.value + self.sparse_err_weight * err_share
        self._sparse_saved.clear()
        metrics = {
            "task": task,
            "loss": loss,
            "pen_raw": raw,
            "err_share": err_share,
            "ratio": raw.detach(),
            "active": (s.detach().abs() > 0).float().sum(-1).mean(),
            "acc": (z.argmax(-1) == labels).float().mean(),
        }
        for k, v in metrics.items():
            self.log(f"{split}_{k}", v.detach(), prog_bar=k in ("task", "loss") and split == "val", sync_dist=True)
        return loss

    def training_step(self, batch: BatchEncoding, batch_idx: int) -> STEP_OUTPUT:
        return self.sparse_step(batch, "train")

    def validation_step(self, batch: BatchEncoding, batch_idx: int, dataloader_idx: int = 0) -> STEP_OUTPUT | None:
        with torch.enable_grad():  # the attributions need the gap's gradient at validation too
            self.sparse_step(batch, "val")


class RTEBoolqSparseModule(RTEBoolqSparseMixin, RTEBoolqSteps, RTEBoolqModuleMixin, torch.nn.Module): ...


def _batches(module, loader, n):
    device = next(module.model.parameters()).device
    for i, batch in enumerate(loader):
        if i == n:
            return
        yield {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


class ChainApplyPayload:
    """Start hook: apply the phase payload (``reg_weight``, ``err_weight``, ``enforce``) to the module."""

    def __call__(self, event) -> None:
        module, payload = event.tuning_scheduler.model, dict(event.phase_config or {})
        module.sparse_reg_weight = float(payload.get("reg_weight", 0.0))
        module.sparse_err_weight = float(payload.get("err_weight", 0.0))
        module.sparse_enforce = bool(payload.get("enforce", False))


class ChainCalibrate:
    """Phase-0 start hook: fit the penalty's normalizer, the untuned penalty over the first ``n_batches`` dev batches,
    and register it as run state so it is saved with every checkpoint and restored on resume."""

    def __init__(self, n_batches: int = 8) -> None:
        self.n_batches = n_batches

    def __call__(self, event) -> None:
        if event.depth != 0:
            return
        module = event.tuning_scheduler.model
        vals = []
        with torch.enable_grad():
            for batch in _batches(module, module.datamodule.val_dataloader(), self.n_batches):
                z, last = module.sparse_forward(batch)
                vals.append(float(module.sparse_ratio(module.sparse_features(z, last)[1]).detach()))
                module._sparse_saved.clear()
        if not vals or min(vals) <= 0:
            raise RuntimeError(f"the untuned penalty must be positive on every calibration batch, got {vals}")
        module.sparse_normalizer.value = sum(vals) / len(vals)
        event.tuning_scheduler.register_stateful("sparse_normalizer", module.sparse_normalizer)


class ChainFillEnforcement:
    """Start hook: fill the ``fitted`` enforcement buffers from the graphs the boundary's end hook built
    (``module.sparse_graphs``: ``n_pre``, ``n_suf``, ``pre_ids``, ``suf_ids`` and per layer the ``mean_*``/``keep_*``
    tensors for the ``pre``, ``suf`` and ``con`` regions)."""

    def __call__(self, event) -> None:
        module = event.tuning_scheduler.model
        graphs = getattr(module, "sparse_graphs", None)
        if event.depth == 0 or graphs is None:
            return
        n_pre, n_suf = graphs["n_pre"], graphs["n_suf"]
        cap_pre, cap_suf = module.sparse_enforce_pre_ids.shape[0], module.sparse_enforce_suf_ids.shape[0]
        if n_pre > cap_pre or n_suf > cap_suf:
            raise ValueError(f"template ({n_pre}, {n_suf}) exceeds the buffers' capacity ({cap_pre}, {cap_suf})")
        print(f"ChainFillEnforcement phase {event.depth}: template ({n_pre}, {n_suf})", flush=True)
        with torch.no_grad():
            module.sparse_enforce_template.copy_(torch.tensor([n_pre, n_suf]))
            module.sparse_enforce_pre_ids.zero_()[:n_pre] = graphs["pre_ids"]
            module.sparse_enforce_suf_ids.zero_()[:n_suf] = graphs["suf_ids"]
            for ly in module.sparse_cfg["enforce_layers"]:
                for kind in ("mean", "keep"):
                    for region, n in (("pre", n_pre), ("suf", n_suf), ("con", None)):
                        buf = getattr(module, f"sparse_enforce_{kind}_{region}_{ly}")
                        src = graphs[f"{kind}_{region}"][ly]
                        if n is None:
                            buf.copy_(src)
                        else:
                            buf.fill_(kind == "keep")
                            buf[:n] = src


# -- the boundary: graphs of the restored best, the enforcement fitted to them, and the #709 gate ---------------------
# Ported from jspace-experiments rte_finetune/tuned_graphs.py, enforce_mask.py, rte_graphs/baseline_sparsity.py and
# gate_recheck.py (interpretune#899).

CHAIN_GRAPH_START = 860  # graph split row 0 is row 860 of the chain's order


def _graph_items(module, rows):
    """Token ids (padding stripped) of the chain's graph-split ``rows`` and whether each is entailed."""
    split = module.datamodule.dataset["graph"]
    out = {}
    for r in rows:
        item = split[r - CHAIN_GRAPH_START]
        ids, mask = torch.as_tensor(item["input_ids"]), torch.as_tensor(item["attention_mask"]).bool()
        out[r] = (ids[mask], int(item["labels"]) == 0)
    return out


def _template(seqs):
    """Lengths of the prefix and suffix every sequence shares token for token."""
    shortest = min(len(x) for x in seqs)
    n_pre = n_suf = 0
    while n_pre < shortest and all(int(x[n_pre]) == int(seqs[0][n_pre]) for x in seqs):
        n_pre += 1
    while n_suf < shortest - n_pre and all(int(x[-1 - n_suf]) == int(seqs[0][-1 - n_suf]) for x in seqs):
        n_suf += 1
    return n_pre, n_suf


def graph_measures(graph) -> dict:
    """``baseline_sparsity.measure`` for a one-target graph: the error nodes' share of node influence, and
    circuit-tracer's replacement and completeness scores."""
    from circuit_tracer.graph import compute_graph_scores, compute_node_influence

    A = graph.adjacency_matrix.float()
    nf, n_pos, n_layers = len(graph.selected_features), int(graph.input_tokens.shape[-1]), graph.cfg.n_layers
    n_err, n_t = n_layers * n_pos, len(graph.logit_targets)
    lw = torch.zeros(A.shape[0], device=A.device)
    lw[-n_t:] = graph.logit_probabilities.to(A.device)
    infl = compute_node_influence(A, lw)[:-n_t]
    rep, comp = compute_graph_scores(graph)
    return {"errors_share": float(infl[nf : nf + n_err].sum() / infl.sum()), "replacement": rep, "completeness": comp}


class ChainBuildGraphs:
    """End hook: from the restored best weights, the Yes-minus-No attribution graphs of the graph items, their
    measures for the gate (``module.sparse_graph_measures``) and, at ``fit_depths``, the enforcement the next phase
    holds (``module.sparse_graphs``, which :class:`ChainFillEnforcement` copies into the ``fitted`` buffers).

    The keep sets are the features circuit-tracer's ``prune_graph`` keeps on each correctly answered item of
    ``mask_rows``, at the enforced layers, pooled by template region (prefix and suffix by position, content as one
    pool). The means are the published transcoders' template-aligned feature means on the same weights over
    ``ref_rows``. The optimizer's state moves to the CPU while the replacement model is built.
    """

    def __init__(
        self,
        mask_rows: list[int],
        gate_rows: list[int],
        ref_rows: tuple[int, int] = (1260, 1300),
        fit_depths: tuple[int, ...] = (0,),
        transcoder_set: str = "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine",
        zero_positions: int = 1,
        max_feature_nodes: int = 2048,
    ) -> None:
        self.mask_rows, self.gate_rows = list(mask_rows), list(gate_rows)
        self.ref_rows, self.fit_depths = tuple(ref_rows), tuple(fit_depths)
        self.transcoder_set, self.zero_positions, self.max_feature_nodes = (
            transcoder_set,
            zero_positions,
            max_feature_nodes,
        )

    def _replacement_model(self, tmp, device):
        from circuit_tracer import ReplacementModel

        rm = ReplacementModel.from_pretrained(
            tmp, self.transcoder_set, backend="nnsight", dtype=torch.float32, device=device
        )
        rm.zero_positions = slice(0, self.zero_positions)  # the convention of every locally graphed checkpoint
        return rm

    def _attribute(self, attribute, ids, rm, target):
        """One graph, at the chain's batch size for the prompt's length, halving it on running out of memory."""
        batch = max(4, min(64, int(64 * 75 / len(ids))))
        while True:
            try:
                return attribute(
                    ids, rm, attribution_targets=target, max_feature_nodes=self.max_feature_nodes, batch_size=batch
                )
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch <= 2:
                    raise
                batch //= 2

    def __call__(self, event) -> None:
        import tempfile

        from circuit_tracer import attribute
        from circuit_tracer.attribution.targets import CustomTarget
        from circuit_tracer.graph import prune_graph

        module, ts = event.tuning_scheduler.model, event.tuning_scheduler
        device = next(module.model.parameters()).device
        # the replacement model is a second copy of the weights: the training module, its gradients and the
        # optimizer's state wait on the CPU while the graphs are built
        module.zero_grad(set_to_none=True)
        opt_state = _optimizer_state_to(ts.optimizer, "cpu")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                module.model.save_pretrained(tmp)
                module.datamodule.tokenizer.save_pretrained(tmp)
                module.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                rm = self._replacement_model(tmp, device)
                yes, no = (int(i) for i in module.it_cfg.entailment_mapping_indices)
                W_U = rm.unembed_weight.detach().float()
                U = W_U.T if W_U.shape[0] == rm.cfg.d_model else W_U
                target = [CustomTarget("logit_gap", 1.0, U[yes] - U[no])]
                items = _graph_items(module, sorted(set(self.mask_rows) | set(self.gate_rows)))
                measures, graphs = {}, {}
                for r, (ids, entailed) in items.items():
                    ids = ids.to(rm.device)
                    with torch.no_grad():
                        logits, _ = rm.get_activations(ids)
                    gap = float(logits.squeeze(0)[-1, yes] - logits.squeeze(0)[-1, no])
                    graph = self._attribute(attribute, ids, rm, target)
                    measures[r] = {**graph_measures(graph), "correct": (gap > 0) == entailed, "gap": gap}
                    if r in self.mask_rows and measures[r]["correct"] and event.depth in self.fit_depths:
                        nf = len(graph.selected_features)
                        kept = prune_graph(graph).node_mask[:nf].nonzero()[:, 0].cpu()
                        graphs[r] = (ids.cpu(), graph.active_features[graph.selected_features][kept].cpu())
                module.sparse_graph_measures = {"depth": event.depth, "rows": measures}
                n_right = sum(m["correct"] for m in measures.values())
                summary = f"ChainBuildGraphs phase {event.depth}: {len(measures)} items graphed, {n_right} correct"
                if event.depth in self.fit_depths:
                    module.sparse_graphs = enf = self._enforcement(module, rm, graphs)
                    kept = " ".join(
                        f"L{ly} {int(enf['keep_con'][ly].sum())}/{int(enf['keep_suf'][ly][-1].sum())}"
                        for ly in module.sparse_cfg["enforce_layers"]
                    )
                    summary += (
                        f"; enforcement from {len(enf['rows'])} correct items, template ({enf['n_pre']}, "
                        f"{enf['n_suf']}), kept per layer (content / answer position): {kept}"
                    )
                print(summary, flush=True)
                del rm
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            module.to(device)
            _optimizer_state_to(ts.optimizer, opt_state)

    def _enforcement(self, module, rm, graphs):
        layers = module.sparse_cfg["enforce_layers"]
        refs = [ids for ids, _ in _graph_items(module, range(*self.ref_rows)).values()]
        n_pre, n_suf = _template(refs)
        d_tc = module.sparse_tcs[str(layers[0])].d_transcoder
        m_pre = {ly: torch.zeros(n_pre, d_tc) for ly in layers}
        m_suf = {ly: torch.zeros(n_suf, d_tc) for ly in layers}
        m_con, count = {ly: torch.zeros(d_tc) for ly in layers}, 0
        for x in refs:
            with torch.no_grad():
                _, acts = rm.get_activations(x.to(rm.device))
            n = int(x.shape[0])
            for ly in layers:
                a = acts[ly].float().cpu()
                m_pre[ly] += a[:n_pre]
                m_suf[ly] += a[n - n_suf :]
                m_con[ly] += a[n_pre : n - n_suf].sum(0)
            count += n - n_pre - n_suf
        keep = {
            region: {ly: torch.zeros(*shape, d_tc, dtype=torch.bool) for ly in layers}
            for region, shape in (("pre", (n_pre,)), ("suf", (n_suf,)), ("con", ()))
        }
        pre_ids, suf_ids = refs[0][:n_pre], refs[0][len(refs[0]) - n_suf :]
        for r, (ids, feats) in graphs.items():
            n = int(ids.shape[0])
            if not (torch.equal(ids[:n_pre], pre_ids) and torch.equal(ids[n - n_suf :], suf_ids)):
                raise ValueError(f"row {r} does not share the reference template")
            for ly, pos, f in feats.tolist():
                if ly not in keep["con"]:
                    continue
                if pos < n_pre:
                    keep["pre"][ly][pos, f] = True
                elif pos >= n - n_suf:
                    keep["suf"][ly][pos - (n - n_suf), f] = True
                else:
                    keep["con"][ly][f] = True
        if not graphs:
            raise ValueError("no correctly answered graph items to fit the enforcement to")
        return {
            "n_pre": n_pre,
            "n_suf": n_suf,
            "pre_ids": pre_ids,
            "suf_ids": suf_ids,
            "mean_pre": {ly: v / len(refs) for ly, v in m_pre.items()},
            "mean_suf": {ly: v / len(refs) for ly, v in m_suf.items()},
            "mean_con": {ly: v / count for ly, v in m_con.items()},
            "keep_pre": keep["pre"],
            "keep_suf": keep["suf"],
            "keep_con": keep["con"],
            "rows": sorted(graphs),
        }


def _optimizer_state_to(optimizer, where):
    """Move an optimizer's state tensors to ``"cpu"`` and return their devices, or restore them from such a return."""
    if optimizer is None:
        return {}
    if isinstance(where, str):
        devices = {}
        for p, state in optimizer.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    devices[(id(p), k)] = v.device
                    state[k] = v.to(where)
        return devices
    for p, state in optimizer.state.items():
        for k, v in state.items():
            if (id(p), k) in where:
                state[k] = v.to(where[(id(p), k)])
    return {}


class ChainGraphGate:
    """Gate (#709): accept a phase whose graphs keep the median error-node share within ``max_err_points`` points of
    the untuned reference and the median replacement score within ``max_replacement_drop`` of it, over the rows both
    measured. ``reference`` is a JSON file of per-row untuned measures (``{"rows": {row: {"errors_share": ..,
    "replacement": ..}}}``), graphed under the same zero-position convention, and ``sha256`` pins it.

    A phase in ``record_only_depths`` that does not end the run is measured and recorded, with the verdict it would
    have had, but accepted: an intermediate stage, such as the first step of a two-step re-thaw (#936), is judged only
    through the phase that follows it."""

    def __init__(
        self,
        reference: str,
        sha256: str,
        max_err_points: float = 5.0,
        max_replacement_drop: float = 0.10,
        record_only_depths: tuple[int, ...] = (),
    ) -> None:
        self.reference, self.sha256 = reference, sha256
        self.max_err_points, self.max_replacement_drop = max_err_points, max_replacement_drop
        self.record_only_depths = tuple(record_only_depths)

    def _load_reference(self) -> dict:
        import hashlib
        import json

        path = self.reference
        if path.startswith("hub://"):
            from interpretune.hub.components import resolve_component_file

            path = resolve_component_file(path)
        data = Path(path).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.sha256:
            raise ValueError(f"untuned reference {self.reference} has sha256 {digest}, pinned {self.sha256}")
        return {int(k): v for k, v in json.loads(data)["rows"].items()}

    def __call__(self, event):
        import statistics as st

        from finetuning_scheduler.core.transitions import GateResult

        module = event.tuning_scheduler.model
        measured = getattr(module, "sparse_graph_measures", None)
        if measured is None or measured["depth"] != event.depth:
            raise RuntimeError(f"no graph measures for phase {event.depth}: run ChainBuildGraphs as an end hook")
        ref = self._load_reference()
        rows = sorted(set(ref) & set(measured["rows"]))
        med = {
            k: (st.median(measured["rows"][r][k] for r in rows), st.median(ref[r][k] for r in rows))
            for k in ("errors_share", "replacement")
        }
        d_err = 100 * (med["errors_share"][0] - med["errors_share"][1])
        d_rep = med["replacement"][0] - med["replacement"][1]
        within = d_err <= self.max_err_points and d_rep >= -self.max_replacement_drop
        judged = event.final or event.depth not in self.record_only_depths
        record = {"items": len(rows), "d_err_points": d_err, "d_replacement": d_rep, "best_depth": event.best_depth}
        record.update(within_bounds=within, judged=judged)
        reason = None if within else f"error share {d_err:+.1f} points, replacement {d_rep:+.3f} vs untuned"
        verdict = (
            ("accept" if within else "VETO") if judged else f"recorded ({'within' if within else 'outside'} bounds)"
        )
        print(f"ChainGraphGate phase {event.depth}: {verdict} {record}", flush=True)
        return GateResult(accept=within or not judged, reason=reason, record=record)


# --------------------------------------------------------------------------------------------------------------------
# Sparse attribution: sparse_attribution_report
# Aggregate checkpoint attribution measurements without changing their denominator.
#
# Head loadings partition summed source edges, which need not equal the measured decision. This report retains both:
# source and head contributions are aligned with each item's decision and divided by the sum of absolute decisions.
# Layers are separate partitions, so their head shares must not be added together. A supplied bias is measured by the
# collector; an absent bias stays unknown, and the difference from the decision remains explicitly unassigned.
#
# This module aggregates already measured rows. It does not load a checkpoint or certify that its identity describes the
# weights used by the collector. The collector must establish that independently, including the decision check.
# --------------------------------------------------------------------------------------------------------------------


# The partition check's absolute floor, by the precision the edges were computed in. Each partition and edge_total are
# separate sums of the same edges, so they differ by that precision's rounding, and a floor below it refuses a correct
# measurement. In float32 the gap measured flat over the 5,130 partitions of a GPU replay of all 190 RTE validation
# rows (median 1.2e-5, max 2.3e-5); 5e-5 clears it while a 1e-3 mismatch still fails. No other precision has a measured
# floor, so a caller states one explicitly.
PARTITION_ATOL = {"float32": 5e-5}


@dataclass(frozen=True)
class SparseAttributionRow:
    """One prompt's raw signed contributions, before normalization.

    Feature coordinates are ``(layer, position, feature)``. Error edges are layer by position, embedding edges are
    position, and head edges are reported layer by head. ``edge_total`` is the shared total every reported attention
    layer partitions with its bypass. ``bias`` is optional measured evidence, never inferred from a remainder.
    """

    row_id: str
    checkpoint_sha256: str
    basis_id: str
    decision: float
    answer_position: int
    feature_nodes: Sequence[tuple[int, int, int]]
    feature_edges: Sequence[float]
    error_edges: Sequence[Sequence[float]]
    embedding_edges: Sequence[float]
    head_layers: Sequence[int]
    head_edges: Sequence[Sequence[float]]
    bypass_edges: Sequence[float]
    edge_total: float
    bias: float | None = None
    freeze_attention: bool = True
    freeze_norms: bool = True


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}")
    return number


def _index(value: int, size: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{name} must be an integer in [0, {size}), got {value!r}")
    return value


def _report_metadata(
    checkpoint_sha256: str,
    basis_id: str,
    model_id: str,
    model_revision: str,
    target_token_ids: tuple[int, int],
    top_k: int,
    measurement_dtype: str,
    partition_atol: float | None,
    partition_rtol: float,
) -> tuple[float, float]:
    if len(checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA256 digest")
    if not all(isinstance(v, str) and v.strip() for v in (basis_id, model_id, model_revision)):
        raise ValueError("basis_id, model_id and model_revision must be stated")
    if len(target_token_ids) != 2 or any(type(v) is not int or v < 0 for v in target_token_ids):
        raise ValueError("target_token_ids must name two nonnegative token ids")
    if target_token_ids[0] == target_token_ids[1]:
        raise ValueError("target_token_ids must name distinct positive and negative targets")
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(measurement_dtype, str) or not measurement_dtype.strip():
        raise ValueError("measurement_dtype must name the precision the edges were computed in")
    if partition_atol is None:
        if measurement_dtype not in PARTITION_ATOL:
            raise ValueError(
                f"no measured partition_atol for measurement_dtype {measurement_dtype!r} (measured: "
                f"{sorted(PARTITION_ATOL)}); state one explicitly"
            )
        partition_atol = PARTITION_ATOL[measurement_dtype]
    atol, rtol = _finite(partition_atol, "partition_atol"), _finite(partition_rtol, "partition_rtol")
    if atol < 0 or rtol < 0:
        raise ValueError("partition tolerances must be nonnegative")
    return atol, rtol


def _row_identity(
    row: SparseAttributionRow,
    checkpoint_sha256: str,
    basis_id: str,
    bias_known: bool,
    seen: set[str],
) -> None:
    if not row.row_id or row.row_id in seen:
        raise ValueError(f"row_id must be nonempty and unique, got {row.row_id!r}")
    if row.checkpoint_sha256 != checkpoint_sha256 or row.basis_id != basis_id:
        raise ValueError(f"row {row.row_id!r} has a different checkpoint or basis")
    if (row.bias is not None) != bias_known:
        raise ValueError("all rows must either measure bias or leave it unknown")
    if type(row.freeze_attention) is not bool or type(row.freeze_norms) is not bool:
        raise ValueError("each row must record boolean freeze_attention and freeze_norms")


def _row_layout(row: SparseAttributionRow, n_layers: int, layers: tuple[int, ...], n_heads: int) -> None:
    n_pos = len(row.embedding_edges)
    if len(row.error_edges) != n_layers or not n_pos:
        raise ValueError(f"row {row.row_id!r} has different model dimensions")
    if any(len(part) != n_pos for part in row.error_edges):
        raise ValueError(f"row {row.row_id!r} has an incomplete error-node layout")
    if tuple(row.head_layers) != layers or len(row.head_edges) != len(layers):
        raise ValueError(f"row {row.row_id!r} has different head layers")
    if len(row.bypass_edges) != len(layers) or any(len(part) != n_heads for part in row.head_edges):
        raise ValueError(f"row {row.row_id!r} has an incomplete head/bypass layout")
    if len(row.feature_nodes) != len(row.feature_edges) or len(set(row.feature_nodes)) != len(row.feature_nodes):
        raise ValueError(f"row {row.row_id!r} needs one edge per distinct feature coordinate")
    _index(row.answer_position, n_pos, "answer position")
    for layer, position, feature in row.feature_nodes:
        _index(layer, n_layers, "feature layer")
        _index(position, n_pos, "feature position")
        if type(feature) is not int or feature < 0:
            raise ValueError(f"feature id must be a nonnegative integer, got {feature!r}")


def _decision(row: SparseAttributionRow) -> float:
    decision = _finite(row.decision, "decision")
    if decision == 0.0:
        raise ValueError(f"row {row.row_id!r} has zero decision, so its alignment is undefined")
    return decision


def _partition_error(
    row: SparseAttributionRow,
    source_total: float,
    edge_total: float,
    row_heads: list[list[float]],
    row_bypass: list[float],
    atol: float,
    rtol: float,
) -> float:
    totals = [source_total, *(math.fsum((*part, rest)) for part, rest in zip(row_heads, row_bypass))]
    for total in totals:
        if not math.isclose(total, edge_total, abs_tol=atol, rel_tol=rtol):
            raise ValueError(f"row {row.row_id!r}: partition {total} does not match edge_total {edge_total}")
    return max(abs(total - edge_total) for total in totals)


def checkpoint_attribution_report(
    rows: Sequence[SparseAttributionRow],
    *,
    checkpoint_sha256: str,
    basis_id: str,
    model_id: str,
    model_revision: str,
    target_token_ids: tuple[int, int],
    measurement_dtype: str,
    top_k: int = 10,
    partition_atol: float | None = None,
    partition_rtol: float = 1e-5,
) -> dict[str, Any]:
    """Summarize measured rows for one checkpoint and basis, refusing incomplete or incompatible partitions.

    The target is the first token's logit minus the second's. Top answer-position features are ranked by the
    magnitude of their aggregated signed share, so opposing features remain visible. The output is JSON-compatible;
    an unknown bias is ``None``. It reports the head layers it actually received and whether they cover the model.
    ``measurement_dtype`` names the precision the edges were computed in; it selects the partition check's absolute
    floor from ``PARTITION_ATOL`` unless ``partition_atol`` is given, and both are recorded in the report.
    """
    atol, rtol = _report_metadata(
        checkpoint_sha256,
        basis_id,
        model_id,
        model_revision,
        target_token_ids,
        top_k,
        measurement_dtype,
        partition_atol,
        partition_rtol,
    )
    if not rows:
        raise ValueError("a checkpoint report needs at least one measured row")

    reference = rows[0]
    n_layers = len(reference.error_edges)
    n_pos = len(reference.embedding_edges)
    layers = tuple(reference.head_layers)
    n_heads = len(reference.head_edges[0]) if reference.head_edges else 0
    if not n_layers or not n_pos or not layers or not n_heads or len(set(layers)) != len(layers):
        raise ValueError("nonempty source and head dimensions, with distinct head layers, are required")
    for layer in layers:
        _index(layer, n_layers, "head layer")
    bias_known = reference.bias is not None
    seen: set[str] = set()
    denominator = 0.0
    source_sums = dict.fromkeys(("features", "errors", "embeddings", "bias", "unassigned"), 0.0)
    feature_layers = [0.0] * n_layers
    error_layers = [0.0] * n_layers
    heads = [[0.0] * n_heads for _ in layers]
    bypass = [0.0] * len(layers)
    answer_features: dict[tuple[int, int], float] = {}
    max_partition_error = 0.0
    item_records = []

    for row in rows:
        _row_identity(row, checkpoint_sha256, basis_id, bias_known, seen)
        if (row.freeze_attention, row.freeze_norms) != (reference.freeze_attention, reference.freeze_norms):
            raise ValueError("all rows must use the same attribution regime")
        _row_layout(row, n_layers, layers, n_heads)
        decision = _decision(row)
        sign = 1.0 if decision > 0 else -1.0
        edge_total = _finite(row.edge_total, "edge_total")
        features = [_finite(v, "feature edge") for v in row.feature_edges]
        errors = [[_finite(v, "error edge") for v in part] for part in row.error_edges]
        embeddings = [_finite(v, "embedding edge") for v in row.embedding_edges]
        row_heads = [[_finite(v, "head edge") for v in part] for part in row.head_edges]
        row_bypass = [_finite(v, "bypass edge") for v in row.bypass_edges]
        bias = _finite(row.bias, "bias") if row.bias is not None else 0.0
        source_total = math.fsum((*features, *(v for part in errors for v in part), *embeddings))
        max_partition_error = max(
            max_partition_error, _partition_error(row, source_total, edge_total, row_heads, row_bypass, atol, rtol)
        )
        for (layer, position, feature), edge in zip(row.feature_nodes, features):
            feature_layers[layer] += sign * edge
            if position == row.answer_position:
                key = (layer, feature)
                answer_features[key] = answer_features.get(key, 0.0) + sign * edge
        for layer, part in enumerate(errors):
            error_layers[layer] += sign * math.fsum(part)
        for i, part in enumerate(row_heads):
            for head, value in enumerate(part):
                heads[i][head] += sign * value
            bypass[i] += sign * row_bypass[i]
        source_sums["features"] += sign * math.fsum(features)
        source_sums["errors"] += sign * math.fsum(v for part in errors for v in part)
        source_sums["embeddings"] += sign * math.fsum(embeddings)
        source_sums["bias"] += sign * bias
        source_sums["unassigned"] += sign * (decision - source_total - bias)
        denominator += abs(decision)
        item_records.append({"row_id": row.row_id, "decision": decision, "edge_total": edge_total})
        seen.add(row.row_id)

    top = sorted(answer_features, key=lambda key: (-abs(answer_features[key]), key))[:top_k]
    result = {
        "format": "interpretune.sparse-attribution-report/1",
        "checkpoint_sha256": checkpoint_sha256,
        "basis_id": basis_id,
        "model_id": model_id,
        "model_revision": model_revision,
        "target_token_ids": list(target_token_ids),
        "attribution_regime": {
            "freeze_attention": reference.freeze_attention,
            "freeze_norms": reference.freeze_norms,
        },
        "items": len(rows),
        "normalization": "decision_aligned_ratio_of_sums",
        "sum_abs_decision": denominator,
        "source_shares": {k: v / denominator if k != "bias" or bias_known else None for k, v in source_sums.items()},
        "feature_shares_by_layer": [v / denominator for v in feature_layers],
        "error_shares_by_layer": [v / denominator for v in error_layers],
        "head_layers": list(layers),
        "complete_head_layers": set(layers) == set(range(n_layers)),
        "head_shares": [[v / denominator for v in part] for part in heads],
        "bypass_shares": [v / denominator for v in bypass],
        "top_answer_features": [
            {"layer": layer, "feature": feature, "share": answer_features[layer, feature] / denominator}
            for layer, feature in top
        ],
        "max_partition_error": max_partition_error,
        "measurement_dtype": measurement_dtype,
        "partition_atol": atol,
        "partition_rtol": rtol,
        "rows": item_records,
    }
    json.dumps(result, allow_nan=False)
    return result


# --------------------------------------------------------------------------------------------------------------------
# Sparse attribution: sparse_attribution_collect
# Convert public head-loading measurements into checkpoint report rows.
#
# The decision must be measured independently from logits on the selected checkpoint. Head loadings supply the source
# edges, which need not sum to that decision. This conversion neither restores nor verifies checkpoint weights; those
# checks belong to the checkpoint collector before it calls this function.
# --------------------------------------------------------------------------------------------------------------------


def _basis_layer_identity(tc: Any, layer: int) -> list[Any]:
    from circuit_tracer.transcoder import SingleLayerTranscoder
    from circuit_tracer.transcoder.activation_functions import JumpReLU

    if type(tc) is not SingleLayerTranscoder or tc.layer_idx != layer or tc.lazy_encoder or tc.lazy_decoder:
        raise ValueError(f"basis layer {layer} must be a materialized SingleLayerTranscoder in layer order")
    if type(tc.activation_function) is not JumpReLU:
        raise ValueError(
            f"basis layer {layer} requires supported JumpReLU; got {type(tc.activation_function).__name__}"
        )
    bandwidth = float(tc.activation_function.bandwidth)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"basis layer {layer} has invalid JumpReLU bandwidth")
    return [tc.d_model, tc.d_transcoder, bandwidth, tc.W_skip is not None]


def _basis_conventions(basis: Any, zero: Any) -> tuple[list[str], list[int | None]]:
    if not isinstance(zero, slice) or any(
        type(v) is not int and v is not None for v in (zero.start, zero.stop, zero.step)
    ):
        raise ValueError("basis fingerprinting requires an explicit integer zero-position slice")
    hooks = [basis.feature_input_hook, basis.feature_output_hook]
    if any(not isinstance(hook, str) or not hook for hook in hooks):
        raise ValueError("basis fingerprinting requires the feature input and output hooks")
    return hooks, [zero.start, zero.stop, zero.step]


def transcoder_basis_id(replacement_model: Any) -> str:
    """Fingerprint the actual per-layer JumpReLU basis, hooks and zero-position convention used for measurement.

    Names of Hub repositories are not evidence of loaded tensor contents. This hashes every transcoder parameter and
    buffer, including thresholds, and the non-tensor settings that change its feature basis. Lazy, cross-layer and
    unfamiliar activation implementations are refused until their identity can be recorded completely.
    """
    from circuit_tracer.transcoder import TranscoderSet
    from nnsight import Envoy

    basis = replacement_model.transcoders
    if isinstance(basis, Envoy):
        basis = basis._module
    if type(basis) is not TranscoderSet or len(basis) != replacement_model.cfg.n_layers:
        raise ValueError("basis fingerprinting requires one TranscoderSet layer per model layer")
    hooks, zero = _basis_conventions(basis, replacement_model.zero_positions)
    layers = [_basis_layer_identity(tc, layer) for layer, tc in enumerate(basis)]
    identity = {
        "format": "interpretune.transcoder-basis/1",
        "state_sha256": tensor_state_sha256(basis.state_dict()),
        "feature_hooks": hooks,
        "zero_positions": zero,
        "layers": layers,
    }
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def head_loading_row(
    report: Mapping[str, Any],
    *,
    row_id: str,
    checkpoint_sha256: str,
    basis_id: str,
    decision: float,
    answer_position: int,
) -> SparseAttributionRow:
    """Retain the raw source coordinates, head partitions and measured regime of one public op result.

    No bias measurement is exposed by head loadings, so bias remains unknown. The checkpoint report validates layout,
    finite numbers and source/head partition sums when it aggregates the rows.
    """
    regime = json.loads(report["head_loading_regime_json"])
    if not isinstance(regime, dict) or set(regime) != {"freeze_attention", "freeze_norms"}:
        raise ValueError("head-loading regime must record freeze_attention and freeze_norms")
    if any(type(value) is not bool for value in regime.values()):
        raise ValueError("head-loading regime must contain boolean freeze flags")
    return SparseAttributionRow(
        row_id=row_id,
        checkpoint_sha256=checkpoint_sha256,
        basis_id=basis_id,
        decision=decision,
        answer_position=answer_position,
        feature_nodes=[tuple(node) for node in report["head_loading_feature_nodes"].tolist()],
        feature_edges=report["head_loading_feature_edges"].tolist(),
        error_edges=report["head_loading_error_edges"].tolist(),
        embedding_edges=report["head_loading_embedding_edges"].tolist(),
        head_layers=report["head_loading_layers"].tolist(),
        head_edges=report["head_loading_head_edges"].tolist(),
        bypass_edges=report["head_loading_bypass_edges"].tolist(),
        edge_total=float(report["head_loading_total"]),
        freeze_attention=regime["freeze_attention"],
        freeze_norms=regime["freeze_norms"],
    )


# --------------------------------------------------------------------------------------------------------------------
# Sparse attribution: sparse_attribution_checkpoint
# Bind attribution reports to the actual model selected after the tuning scheduler's terminal gates.
#
# A checkpoint policy's best path can still name a rejected candidate after a veto. The callback saves the live selected
# state after FTSCoreCallback ends training, then checks all checkpoint tensors against the view before and after
# measurement. A measurement driver must independently check its reconstructed model's logits and basis; this module
# proves the selected training state and exact validation coverage, not the driver's reconstruction.
# --------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VerifiedAttributionCheckpoint:
    """A checkpoint file whose complete tensor state matches the selected training view."""

    path: Path
    sha256: str
    state_sha256: str
    state_keys: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointMeasurements:
    """A driver's measurements and the complete validation identity set it was asked to measure."""

    rows: Sequence[SparseAttributionRow]
    expected_row_ids: Sequence[str]
    basis_id: str
    model_id: str
    model_revision: str
    target_token_ids: tuple[int, int]
    measurement_dtype: str
    checks: Mapping[str, Any] = field(default_factory=dict)


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash every named tensor's key, dtype, shape and bytes, independent of mapping order or device.

    Dense, materialized tensors are required. Unsupported layouts and non-tensor state are refused rather than omitted,
    because a digest of only part of the model would not identify its selected state.
    """
    import torch

    if not state or any(not isinstance(name, str) or not name for name in state):
        raise ValueError("selected state must contain nonempty tensor names")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.layout != torch.strided
            or tensor.is_quantized
            or tensor.device.type == "meta"
        ):
            raise ValueError(f"selected state tensor {name!r} must be dense and materialized")
        header = json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(memoryview(raw.numpy()))
    return digest.hexdigest()


def verify_attribution_checkpoint(path: str | Path, selected_state: Mapping[str, Any]) -> VerifiedAttributionCheckpoint:
    """Verify a complete, unsharded FTS checkpoint against the actual selected view, without restoring anything.

    Scheduled-only checkpoints cannot prove the unscheduled base and are refused. Callers supply
    ``tuning_scheduler.view.checkpoint_state_dict()`` so fitted tensors are included and artifacts are excluded by
    the same view that wrote the checkpoint. These are trusted local training checkpoints, read with FTS's loader.
    """
    from finetuning_scheduler.core.checkpoint import load_payload

    path = Path(path)
    if not path.is_file():
        raise ValueError(f"attribution checkpoint must be an unsharded file, got {path}")
    with path.open("rb") as stream:
        file_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    payload = load_payload(path)
    if payload.get("version") != 1 or payload.get("sharded") is not False or payload.get("include") != "all":
        raise ValueError("attribution needs a version-1, unsharded FTS checkpoint with include='all'")
    saved = payload.get("model")
    if not isinstance(saved, Mapping) or set(saved) != set(selected_state):
        raise ValueError("checkpoint and selected state have different tensor keys")
    selected_sha = tensor_state_sha256(selected_state)
    if tensor_state_sha256(saved) != selected_sha:
        raise ValueError("checkpoint tensors do not match the selected model state")
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != file_sha:
            raise ValueError("attribution checkpoint changed while it was verified")
    return VerifiedAttributionCheckpoint(path, file_sha, selected_sha, tuple(sorted(saved)))


def _complete_validation_coverage(measurements: CheckpointMeasurements) -> list[str]:
    expected = list(measurements.expected_row_ids)
    actual = [row.row_id for row in measurements.rows]
    if not expected or any(not isinstance(row_id, str) or not row_id for row_id in expected):
        raise ValueError("expected validation row ids must be nonempty strings")
    if len(set(expected)) != len(expected) or len(set(actual)) != len(actual) or set(actual) != set(expected):
        raise ValueError("measured rows must match the complete expected validation identity set exactly")
    return expected


def _report_summary(report: Mapping[str, Any], destination: Path) -> str:
    source = report["source_shares"]
    heads = [
        (layer, head, value)
        for layer, values in zip(report["head_layers"], report["head_shares"])
        for head, value in enumerate(values)
    ]
    largest = max(heads, key=lambda entry: abs(entry[2]), default=None)
    head = f"largest head L{largest[0]} H{largest[1]} {largest[2]:+.4f}" if largest else "no measured heads"
    features = report["top_answer_features"]
    top = (
        f"top answer feature L{features[0]['layer']} F{features[0]['feature']} {features[0]['share']:+.4f}"
        if features
        else "no active answer features"
    )
    bias = "unmeasured" if source["bias"] is None else f"{source['bias']:+.4f}"
    return (
        f"Selected-checkpoint attribution: {report['items']} validation items; "
        f"feature share {source['features']:+.4f}, error share {source['errors']:+.4f}, "
        f"embedding share {source['embeddings']:+.4f}, bias {bias}, unassigned {source['unassigned']:+.4f}; "
        f"{head}; {top}; {destination}"
    )


class SparseAttributionReportCallback:
    """Save and report the live selected checkpoint after FTSCoreCallback's final restore and gates.

    Args:
        collector: A callable, or a ``class_path`` mapping constructing one. It receives the training module and a
            verified checkpoint and returns CheckpointMeasurements. Its responsibilities include reconstructing the
            measurement model, checking its logits against the selected training model, and identifying the actual
            transcoder basis. A repeated checkpoint hash in its rows is not reconstruction evidence.
        output_dir: Parent for a separate checkpoint/report directory per completed run. The selected checkpoint is
            a full FTS checkpoint with optimizer state. Publication follows successful collection and verification.
        top_k: Answer-position features retained in the aggregate report.

    The callback must follow exactly one FTSCoreCallback. Only single-process, unsharded runs with a complete
    checkpoint view are supported. ``module.sparse_attribution_report_path`` names the JSON after it is published.
    """

    def __init__(
        self,
        collector: Callable[[Any, VerifiedAttributionCheckpoint], CheckpointMeasurements] | Mapping[str, Any],
        output_dir: str = "reports/sparse_attribution",
        top_k: int = 10,
    ) -> None:
        if isinstance(collector, Mapping):
            from interpretune.hub.entrypoints import instantiate_hub_aware_class

            collector = instantiate_hub_aware_class(dict(collector))
        if not callable(collector):
            raise ValueError("attribution collector must be callable")
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self._fts: Any = None

    def on_train_start(self, ctx: Any) -> None:
        """Check callback order before training, and refuse distributed collection until it is supported."""
        from interpretune.tuning.core_loop import FTSCoreCallback

        callbacks = list(ctx.callbacks)
        matches = [cb for cb in callbacks if isinstance(cb, FTSCoreCallback)]
        if len(matches) != 1 or self not in callbacks or callbacks.index(matches[0]) >= callbacks.index(self):
            raise ValueError("SparseAttributionReportCallback must follow exactly one FTSCoreCallback")
        if ctx.dist_run is not None:
            raise ValueError("sparse attribution reports currently require a single-process unsharded run")
        self._fts = matches[0]
        ctx.module.sparse_attribution_report_path = None

    def on_train_end(self, ctx: Any) -> None:
        """Save the post-gate subject, measure it, recheck state identity, then atomically publish its report."""
        scheduler = self._fts.tuning_scheduler if self._fts is not None else None
        if scheduler is None or not scheduler.state_dict().get("finished"):
            raise ValueError("attribution reporting requires FTSCoreCallback to finish its terminal gates first")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = Path(tempfile.mkdtemp(prefix="selected-", dir=self.output_dir))
        checkpoint = Path(scheduler.save(str(run_dir / "checkpoint.pt")))
        verified = verify_attribution_checkpoint(checkpoint, scheduler.view.checkpoint_state_dict())
        measurements = self.collector(ctx.module, verified)
        expected = _complete_validation_coverage(measurements)
        report = checkpoint_attribution_report(
            measurements.rows,
            checkpoint_sha256=verified.sha256,
            basis_id=measurements.basis_id,
            model_id=measurements.model_id,
            model_revision=measurements.model_revision,
            target_token_ids=measurements.target_token_ids,
            measurement_dtype=measurements.measurement_dtype,
            top_k=self.top_k,
        )
        if verify_attribution_checkpoint(checkpoint, scheduler.view.checkpoint_state_dict()) != verified:
            raise ValueError("selected attribution checkpoint changed during measurement")
        report["checkpoint_state_sha256"] = verified.state_sha256
        report["checkpoint_tensor_count"] = len(verified.state_keys)
        report["validation_row_ids"] = expected
        report["validation_coverage_verified"] = True
        report["checkpoint_selection"] = "live_after_terminal_gates"
        report["gate_records"] = scheduler.gate_records
        report["measurement_checks"] = dict(measurements.checks)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        destination = run_dir / "attribution.json"
        pending = run_dir / "attribution.json.tmp"
        pending.write_text(payload, encoding="utf-8")
        pending.replace(destination)
        ctx.module.sparse_attribution_report_path = str(destination)
        from interpretune.utils import rank_zero_info

        rank_zero_info(_report_summary(report, destination))


# --------------------------------------------------------------------------------------------------------------------
# Sparse attribution: rte_sparse_attribution
# Measure a sparse RTE run's full validation set against a verified selected checkpoint.
#
# The measurement factory must yield an Interpretune module exposing the public head-attribution technique. It must
# preserve the selected model's behavior, including active fitted enforcement. The collector checks the reconstructed HF
# weights and both answer logits before accepting any attribution row; exporting bare weights is not sufficient.
# --------------------------------------------------------------------------------------------------------------------


def _check_selected_subject(module: Any, checkpoint: VerifiedAttributionCheckpoint) -> None:
    state = module.state_dict()
    if not set(checkpoint.state_keys).issubset(state):
        raise ValueError("RTE collection requires checkpoint keys matching the selected module's physical state")
    selected = {name: state[name] for name in checkpoint.state_keys}
    if tensor_state_sha256(selected) != checkpoint.state_sha256:
        raise ValueError("RTE module tensors do not match the verified selected checkpoint")


def _validation_items(module: Any) -> dict[str, Any]:
    import torch

    items = {}
    for item in module.datamodule.dataset["validation"]:
        missing = {"idx", "input_ids", "attention_mask"} - item.keys()
        if missing:
            raise ValueError(f"validation rows must retain required columns: missing {sorted(missing)}")
        identity = item["idx"]
        if type(identity) not in (int, str) or not str(identity) or str(identity) in items:
            raise ValueError(f"validation idx must be a distinct integer or string, got {identity!r}")
        ids, mask = torch.as_tensor(item["input_ids"]), torch.as_tensor(item["attention_mask"])
        if ids.ndim != 1 or ids.dtype != torch.int64 or mask.shape != ids.shape:
            raise ValueError(f"validation row {identity!r} needs one int64 token sequence and matching attention mask")
        if not ((mask == 0) | (mask == 1)).all() or not mask.any():
            raise ValueError(f"validation row {identity!r} needs a nonempty binary attention mask")
        positions = mask.bool().nonzero().flatten()
        if int(positions[-1] - positions[0]) + 1 != len(positions):
            raise ValueError(f"validation row {identity!r} has internal padding, which cannot be stripped safely")
        items[str(identity)] = ids[mask.bool()].detach().cpu()
    if not items:
        raise ValueError("sparse attribution requires a nonempty validation set")
    return items


@contextmanager
def _reference_inference(module: Any) -> Iterator[None]:
    """Retain every module's mode and the sparse forward's temporary caches, including on a failed measurement."""
    modes = [(child, child.training) for child in module.modules()]
    saved, mask = module._sparse_saved, module._sparse_mask
    module._sparse_saved = {}
    module.eval()
    try:
        yield
    finally:
        module._sparse_saved = saved
        module._sparse_mask = mask
        for child, training in modes:
            child.training = training


def _target_tokens(module: Any) -> tuple[int, int]:
    import torch

    ids = torch.as_tensor(module.it_cfg.entailment_mapping_indices)
    if ids.dtype not in (torch.int32, torch.int64) or ids.shape != (2,):
        raise ValueError("entailment_mapping_indices must contain two integer token ids")
    positive, negative = ids.tolist()
    if min(positive, negative) < 0 or positive == negative:
        raise ValueError("entailment_mapping_indices must contain distinct nonnegative token ids")
    return positive, negative


def _reference_logits(module: Any, items: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = next(module.model.parameters()).device
    reference = {}
    with _reference_inference(module), torch.no_grad():
        for row_id, ids in items.items():
            inputs = ids.to(device)[None]
            pair, _ = module.sparse_forward({"input_ids": inputs, "attention_mask": torch.ones_like(inputs)})
            if pair.shape != (1, 2) or not torch.isfinite(pair).all():
                raise ValueError(f"validation row {row_id!r} did not produce two finite answer logits")
            reference[row_id] = pair[0].detach().float().cpu()
            module._sparse_saved.clear()
    return reference


def _decision_target(model: Any, targets: tuple[int, int]) -> Any:
    from circuit_tracer.attribution.targets import CustomTarget

    config = model._module.config.get_text_config()
    if getattr(config, "final_logit_softcapping", None) is not None:
        raise ValueError("RTE logit-gap attribution does not yet support final_logit_softcapping")
    unembed = model._module.get_output_embeddings().weight.detach()
    if unembed.shape != (model.cfg.d_vocab, model.cfg.d_model) or max(targets) >= unembed.shape[0]:
        raise ValueError("RTE answer tokens require a vocabulary-by-model unembedding weight")
    return CustomTarget("logit_gap", 1.0, unembed[targets[0]] - unembed[targets[1]])


def _measurement_hf_state(model: Any) -> dict[str, Any]:
    """Separate the known registered transcoder child from HF weights; the basis is verified independently."""
    from nnsight import Envoy

    basis = model.transcoders
    if isinstance(basis, Envoy):
        basis = basis._module
    if model._module.get_submodule("transcoders") is not basis:
        raise ValueError("measurement model does not register its verified basis under the expected transcoders child")
    state = model._module.state_dict()
    artifact_keys = {f"transcoders.{name}" for name in basis.state_dict()}
    if not artifact_keys or not artifact_keys.issubset(state):
        raise ValueError("measurement model has an incomplete registered transcoder state")
    return {name: value for name, value in state.items() if name not in artifact_keys}


def _measurement_dtype(state: Mapping[str, Any]) -> str:
    """The one floating precision of the measured HF weights, which the attribution edges are computed in."""
    dtypes = {str(tensor.dtype).removeprefix("torch.") for tensor in state.values() if tensor.is_floating_point()}
    if len(dtypes) != 1:
        raise ValueError(f"measurement HF weights must share one floating precision, got {sorted(dtypes)}")
    return dtypes.pop()


def _model_provenance(module: Any) -> tuple[str, str]:
    config = module.it_cfg
    model_id = getattr(config, "model_name_or_path", None)
    loader = getattr(config, "hf_from_pretrained_cfg", None)
    arguments = getattr(loader, "pretrained_kwargs", None)
    revision = arguments.get("revision") if isinstance(arguments, Mapping) else None
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("RTE attribution requires the selected module's HF model_name_or_path")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("RTE attribution requires an immutable 40-character HF-loader revision pin")
    return model_id, revision


class RteSparseAttributionCollector:
    """Collect the selected sparse RTE model's head/source measurements over its complete validation set.

    Args:
        measurement_factory: A callable or ``class_path`` mapping. Called with the training module and verified
            checkpoint, it returns a context manager yielding a module for ``it.head_loadings``. The yielded
            module's replacement model must preserve the selected HF weights and any fitted enforcement. No
            default factory silently replaces an enforced run with its bare weights.
        logit_atol: Absolute tolerance when comparing both reconstructed answer logits with the selected run.
        logit_rtol: Relative tolerance for that comparison. The default is for float32 CPU/GPU measurement.

    Model identity comes from the selected module's HF load configuration, which must pin ``revision`` to a full
    commit hash. It is not inferred from a moving Hub branch or a separate caller-supplied label. The context manager
    owns construction, device placement and cleanup. This collector never switches enforcement
    off, samples the validation set, changes the target, or treats matching version strings as weight evidence.
    """

    def __init__(
        self,
        measurement_factory: Callable[[Any, VerifiedAttributionCheckpoint], AbstractContextManager[Any]]
        | Mapping[str, Any],
        logit_atol: float = 1e-5,
        logit_rtol: float = 1e-5,
    ) -> None:
        if isinstance(measurement_factory, Mapping):
            from interpretune.hub.entrypoints import instantiate_hub_aware_class

            measurement_factory = instantiate_hub_aware_class(dict(measurement_factory))
        if not callable(measurement_factory):
            raise ValueError("measurement_factory must return a context manager for the selected model")
        if any(not math.isfinite(value) or value < 0 for value in (logit_atol, logit_rtol)):
            raise ValueError("logit tolerances must be finite and nonnegative")
        self.measurement_factory = measurement_factory
        self.logit_atol, self.logit_rtol = logit_atol, logit_rtol

    def __call__(self, module: Any, checkpoint: VerifiedAttributionCheckpoint) -> CheckpointMeasurements:
        """Verify the reconstruction before accepting its attribution rows, then recheck its basis and weights."""
        import torch
        from transformers import BatchEncoding
        import interpretune as it
        import interpretune.analysis
        from interpretune.analysis.ops.base import AnalysisBatch

        _check_selected_subject(module, checkpoint)
        model_id, revision = _model_provenance(module)
        items = _validation_items(module)
        targets = _target_tokens(module)
        enforced = module.sparse_enforce
        if type(enforced) is not bool:
            raise ValueError("the selected module must state its boolean sparse_enforce mode")
        weights_sha = tensor_state_sha256(module.model.state_dict())
        reference = _reference_logits(module, items)
        rows = []
        max_logit_error = 0.0
        with self.measurement_factory(module, checkpoint) as measurement:
            model = measurement.replacement_model
            backend = getattr(model, "backend", None)
            if backend != "nnsight":
                raise ValueError(f"RTE checkpoint collection requires the nnsight backend, got {backend!r}")
            basis = transcoder_basis_id(model)
            measured_state = _measurement_hf_state(model)
            if tensor_state_sha256(measured_state) != weights_sha:
                raise ValueError("measurement HF weights do not match the selected training model")
            measurement_dtype = _measurement_dtype(measured_state)
            target = _decision_target(model, targets)
            for row_id, ids in items.items():
                with torch.no_grad():
                    logits, _ = model.get_activations(ids.to(model.device))
                pair = logits[0, -1, list(targets)].detach().float().cpu()
                if not torch.allclose(pair, reference[row_id], atol=self.logit_atol, rtol=self.logit_rtol):
                    raise ValueError(
                        f"validation row {row_id!r}: measurement answer logits {pair.tolist()} do not match selected "
                        f"logits {reference[row_id].tolist()}; preserve fitted enforcement and model configuration"
                    )
                max_logit_error = max(max_logit_error, float((pair - reference[row_id]).abs().max()))
                raw = it.head_loadings(measurement, AnalysisBatch(), BatchEncoding(), 0, prompt=ids, head_target=target)
                rows.append(
                    head_loading_row(
                        raw,
                        row_id=row_id,
                        checkpoint_sha256=checkpoint.sha256,
                        basis_id=basis,
                        decision=float(reference[row_id][0] - reference[row_id][1]),
                        answer_position=len(ids) - 1,
                    )
                )
            if transcoder_basis_id(model) != basis or tensor_state_sha256(_measurement_hf_state(model)) != weights_sha:
                raise ValueError("measurement basis or HF weights changed during attribution")
        if module.sparse_enforce != enforced:
            raise ValueError("the measurement factory changed the selected sparse enforcement mode")
        if _model_provenance(module) != (model_id, revision):
            raise ValueError("the selected model provenance changed during attribution")
        _check_selected_subject(module, checkpoint)
        return CheckpointMeasurements(
            rows,
            list(items),
            basis,
            model_id,
            revision,
            targets,
            measurement_dtype,
            checks={
                "selected_hf_state_sha256": weights_sha,
                "validation_inputs_sha256": tensor_state_sha256(items),
                "max_answer_logit_abs_error": max_logit_error,
                "logit_atol": self.logit_atol,
                "logit_rtol": self.logit_rtol,
                "sparse_enforcement_enabled": enforced,
            },
        )


# --------------------------------------------------------------------------------------------------------------------
# Sparse attribution: rte_sparse_measurement
# Construct head-attribution measurements without dropping the selected RTE run's fitted enforcement.
# --------------------------------------------------------------------------------------------------------------------


def _source_parameter(module: Any) -> Any:
    import torch
    from circuit_tracer.utils.tl_nnsight_mapping import get_mapping

    source = module.model
    parameter = next(source.parameters())
    if any(p.dtype != torch.float32 or p.device != parameter.device for p in source.parameters()):
        raise ValueError("RTE head measurement currently requires a single-device float32 selected model")
    architectures = source.config.architectures
    if not isinstance(architectures, list) or len(architectures) != 1:
        raise ValueError("RTE head measurement requires one explicit HF model architecture")
    get_mapping(architectures[0])
    return parameter


def _full_basis(module: Any, specification: Any) -> Any:
    from circuit_tracer.transcoder import TranscoderSet
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    parameter = next(module.model.parameters())
    if isinstance(specification, str):
        basis, _ = load_transcoder_from_hub(
            specification, device=parameter.device, dtype=parameter.dtype, lazy_encoder=False, lazy_decoder=False
        )
    else:
        basis = deepcopy(specification)
    if type(basis) is not TranscoderSet:
        raise ValueError("RTE head measurement needs a materialized single-layer TranscoderSet")
    expected = set(range(module.model.config.get_text_config().num_hidden_layers))
    if len(basis.transcoders) != len(expected) or len(basis) != len(expected):
        raise ValueError(f"RTE head measurement needs a basis for every model layer: {sorted(expected)}")
    for layer, transcoder in enumerate(basis):
        _basis_layer_identity(transcoder, layer)
    for layer in module.sparse_cfg["enforce_layers"]:
        if _basis_layer_identity(basis[layer], layer) != _basis_layer_identity(module.sparse_tcs[str(layer)], layer):
            raise ValueError(f"measurement basis differs from the fitted enforcement transcoder at layer {layer}")
        if tensor_state_sha256(basis[layer].state_dict()) != tensor_state_sha256(
            module.sparse_tcs[str(layer)].state_dict()
        ):
            raise ValueError(f"measurement basis differs from the fitted enforcement transcoder at layer {layer}")
    return basis


def _clone_hf_model(module: Any) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    source = module.model
    parameter = _source_parameter(module)
    config = deepcopy(source.config)
    config._attn_implementation = "eager"
    # Construction does not copy hooks: the sparse hooks below are deliberately installed in a known order.
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        clone = AutoModelForCausalLM.from_config(config)
    clone = clone.to(device=parameter.device, dtype=parameter.dtype)
    clone.load_state_dict(source.state_dict(), strict=True)
    return clone.eval()


def _install_enforcement(module: Any, model: Any, stack: ExitStack) -> None:
    import torch

    def prepare(_model: Any, args: tuple, kwargs: dict) -> None:
        ids = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.ndim != 2:
            raise ValueError("RTE head measurement requires batched input_ids, including repeated attribution batches")
        mask = kwargs.get("attention_mask")
        if mask is None:
            mask = torch.ones_like(ids)
        if mask.shape != ids.shape or not bool((mask == 1).all()):
            raise ValueError("RTE head measurement expects padding to be stripped before attribution")
        module._sparse_saved.clear()
        module._sparse_mask = mask
        if module.sparse_enforce:
            module.check_template(ids, mask)

    stack.callback(model.register_forward_pre_hook(prepare, with_kwargs=True).remove)
    cfg = module.sparse_cfg
    for layer in sorted(set(cfg["reg_layers"]) | set(cfg["enforce_layers"])):
        for key in ("x", "out"):
            site = model.get_submodule(cfg["sites"][key].format(layer=layer))
            stack.callback(site.register_forward_hook(module._sparse_keep(layer, key)).remove)
    for layer in cfg["enforce_layers"]:
        site = model.get_submodule(cfg["sites"]["out"].format(layer=layer))
        stack.callback(site.register_forward_hook(module._sparse_enforce(layer)).remove)


class RteSparseMeasurementFactory:
    """Yield an analysis component backed by an independent copy of the selected HF weights.

    Args:
        transcoder_set: A circuit-tracer Hub reference or a materialized ``TranscoderSet`` covering every layer.
            Its enforced layers must match the selected run's instrument transcoders exactly.
        zero_positions: Number of leading positions omitted from transcoder features, recorded in the basis id.

    The run's actual sparse hook factories and fitted buffers supply enforcement. Hooks are installed before NNsight
    wraps the model, so its captured MLP outputs include enforcement. Each repeated attribution forward gets its own
    attention mask and template check. Source caches, modes, weights and parameter gradients are retained.

    The selected float32 model and the measurement copy coexist on one device. Callers must budget for both models,
    optimizer state, full transcoders and attribution batches. GPU memory has not been calibrated by this factory.
    """

    def __init__(self, transcoder_set: Any, zero_positions: int = 1) -> None:
        if type(zero_positions) is not int or zero_positions < 0:
            raise ValueError("zero_positions must be a nonnegative integer")
        self.transcoder_set, self.zero_positions = transcoder_set, zero_positions

    @contextmanager
    def __call__(self, module: Any, checkpoint: VerifiedAttributionCheckpoint) -> Iterator[Any]:
        """Build a separate measurement model and release its enforcement hooks even if attribution raises."""
        from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
        from interpretune.adapters.circuit_tracer.backends import DEFAULT_CT_ANALYSIS_BACKEND

        _check_selected_subject(module, checkpoint)
        _source_parameter(module)
        basis = _full_basis(module, self.transcoder_set)
        clone = _clone_hf_model(module)
        with _reference_inference(module), ExitStack() as hooks:
            _install_enforcement(module, clone, hooks)
            replacement = NNSightReplacementModel(
                clone, tokenizer=module.datamodule.tokenizer, task="text-generation", dispatch=True
            )
            replacement.config = clone.config
            replacement._configure_replacement_model(basis)
            replacement.zero_positions = slice(0, self.zero_positions)
            yield SimpleNamespace(replacement_model=replacement, analysis_backend=DEFAULT_CT_ANALYSIS_BACKEND)
        _check_selected_subject(module, checkpoint)
