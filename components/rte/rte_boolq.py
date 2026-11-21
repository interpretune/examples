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

        save_path = Path(self.itdm_cfg.dataset_path)
        dataset.save_to_disk(save_path)

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
        dataset.save_to_disk(Path(self.itdm_cfg.dataset_path))

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
