"""How a concept direction changes with the states it is built from: the latent-dynamics analysis.

One concept pair (for example "orange" between a Fruit and a Color group) yields a family of directions that differ
only in which states feed the same construction: token embeddings, the answer-position state, the context-token state,
the context-enhanced projection, and the J-lens read at an intermediate layer. This module names each construction as
a *stage* in a table and computes every stage with ``it.concept_direction``. The op is the single implementation of
the direction math, so a stage cannot drift from it the way the earlier analysis's re-implemented builders could.

The table is data: a configuration can reorder, drop or add stages (a further J-lens layer, say) without code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch

import interpretune as it
from interpretune.analysis.ops.bundled.concept.concept_ops import CONCEPT_BASES

from exp.analysis.concept_direction_analysis import compute_concept_direction_geometry

RowSource = Literal["answer", "context", "projected", "final", "store_answer"]
WeightSource = Literal["example", "unit"]
DirectionMode = Literal["mean_difference", "paired_rejection"]

#: The state matrices a store-basis stage can read; each is one row per example, aligned with the group ids.
ROW_SOURCES: tuple[str, ...] = ("answer", "context", "projected", "final", "store_answer")


@dataclass(frozen=True)
class Stage:
    """One construction of the concept direction.

    ``basis`` and ``mode`` are passed to ``it.concept_direction`` unchanged. A ``store`` stage names the state matrix it
    reads (``rows``) and whether examples keep their extraction weights or count equally (``weights``). An ``embed``
    stage reads token groups: the concept pair's groups, or only its two target tokens (``target_tokens_only``). A J-lens
    stage reads the read path at ``jlens_layer``.
    """

    name: str
    basis: str
    mode: DirectionMode
    rows: RowSource | None = None
    weights: WeightSource = "example"
    target_tokens_only: bool = False
    jlens_layer: int | None = None

    def __post_init__(self) -> None:
        if self.basis not in CONCEPT_BASES:
            raise ValueError(f"stage {self.name!r}: basis {self.basis!r} is not one of {list(CONCEPT_BASES)}")
        if self.mode not in ("mean_difference", "paired_rejection"):
            raise ValueError(f"stage {self.name!r}: mode {self.mode!r} is not 'mean_difference' or 'paired_rejection'")
        if self.basis == "store":
            if self.rows not in ROW_SOURCES:
                raise ValueError(f"stage {self.name!r}: a store stage names its rows, one of {list(ROW_SOURCES)}")
        elif self.rows is not None:
            raise ValueError(f"stage {self.name!r}: only a store stage reads state rows, not a {self.basis} stage")
        if self.weights not in ("example", "unit"):
            raise ValueError(f"stage {self.name!r}: weights {self.weights!r} is not 'example' or 'unit'")
        if self.target_tokens_only and self.basis != "embed":
            raise ValueError(f"stage {self.name!r}: target_tokens_only applies to an embed stage only")
        is_jlens = self.basis.startswith("jlens_")
        if is_jlens != (self.jlens_layer is not None):
            raise ValueError(f"stage {self.name!r}: a J-lens stage, and only a J-lens stage, names its jlens_layer")


#: The default table, in construction order: from token identity, through the states the model computes at the answer
#: and context positions, to the context-enhanced store direction the experiment steers with. J-lens stages are added
#: per configured lens layer (:func:`stages_from_config`).
DEFAULT_STAGES: tuple[Stage, ...] = (
    Stage("target_token_embed_difference", "embed", "mean_difference", target_tokens_only=True),
    Stage("embed_mean_difference", "embed", "mean_difference"),
    Stage("answer_state_mean_difference", "store", "mean_difference", rows="answer"),
    Stage("context_state_mean_difference", "store", "mean_difference", rows="context"),
    Stage("projected_context_mean_difference", "store", "mean_difference", rows="projected"),
    Stage("paired_residual_unweighted", "store", "paired_rejection", rows="final", weights="unit"),
    Stage("store_context_paired_rejection", "store", "paired_rejection", rows="final"),
    Stage("embed_paired_rejection", "embed", "paired_rejection"),
    Stage("store_answer_paired_rejection", "store", "paired_rejection", rows="store_answer"),
)


def stages_from_config(latent_dynamics_cfg: Mapping[str, Any]) -> tuple[Stage, ...]:
    """The stage table a ``LATENT_DYNAMICS`` configuration section describes.

    ``stages`` (a list of :class:`Stage` field mappings) replaces the default table; ``jlens_layers`` with
    ``jlens_basis`` (``jlens_folded`` unless set) appends one paired-rejection J-lens stage per layer. Stage names must
    be unique, since every report keys on them.
    """
    raw = latent_dynamics_cfg.get("stages")
    stages = tuple(Stage(**dict(entry)) for entry in raw) if raw is not None else DEFAULT_STAGES
    jlens_basis = str(latent_dynamics_cfg.get("jlens_basis", "jlens_folded"))
    for layer in latent_dynamics_cfg.get("jlens_layers", ()) or ():
        stages += (Stage(f"{jlens_basis}_l{int(layer)}", jlens_basis, "paired_rejection", jlens_layer=int(layer)),)
    names = [stage.name for stage in stages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"LATENT_DYNAMICS stages repeat names {duplicates}; every stage name must be unique")
    return stages


@dataclass
class StageInputs:
    """What the stages read, gathered once per run.

    ``states`` maps each row source to a ``(n_examples, d_model)`` matrix, all aligned with ``group_ids`` (0 for group
    A, 1 for group B), ``weights`` and ``group_names``. Token groups and target tokens serve the embed and J-lens stages.
    """

    states: dict[str, torch.Tensor]
    group_ids: torch.Tensor
    weights: torch.Tensor
    group_names: list[str]
    group_a_tokens: list[str]
    group_b_tokens: list[str]
    target_tokens: tuple[str, str]
    concept_label: str
    jlens_kwargs: dict[str, Any] = field(default_factory=dict)
    # each row's pair: its position within its group across every prompt, which is how the op pairs a full store. With
    # rows dropped (a correct-only store), a paired stage reads only the pairs whose two rows were both kept.
    pair_ids: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n = int(self.group_ids.shape[0])
        misaligned = {name: tuple(m.shape) for name, m in self.states.items() if m.ndim != 2 or m.shape[0] != n}
        if misaligned or int(self.weights.shape[0]) != n:
            raise ValueError(
                f"stage inputs are not aligned with {n} group ids: states {misaligned or 'aligned'}, "
                f"weights {tuple(self.weights.shape)}"
            )
        if self.pair_ids is not None and int(self.pair_ids.shape[0]) != n:
            raise ValueError(f"{int(self.pair_ids.shape[0])} pair ids for {n} store rows")

    def stage_rows(self, stage: Stage) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        """The states, group ids, weights and group names a store stage reads.

        A mean-difference stage reads every row. A paired stage reads the complete pairs only, group A's rows then group
        B's, each in pair order, so the op's positional pairing matches pair to pair; it refuses when no pair is left.
        """
        assert stage.rows is not None
        weights = torch.ones_like(self.weights) if stage.weights == "unit" else self.weights
        states, group_ids, names = self.states[stage.rows], self.group_ids, list(self.group_names)
        if stage.mode != "paired_rejection" or self.pair_ids is None:
            return states, group_ids, weights, names
        pairs_a = {int(p) for p, g in zip(self.pair_ids, group_ids) if int(g) == 0}
        pairs_b = {int(p) for p, g in zip(self.pair_ids, group_ids) if int(g) == 1}
        complete = sorted(pairs_a & pairs_b)
        if not complete:
            raise ValueError(f"stage {stage.name!r} is paired, but no pair kept both of its rows")
        index = [
            next(i for i, (p, g) in enumerate(zip(self.pair_ids, group_ids)) if int(p) == pair and int(g) == group)
            for group in (0, 1)
            for pair in complete
        ]
        rows = torch.tensor(index, dtype=torch.long)
        return states[rows], group_ids[rows], weights[rows], [names[i] for i in index]


def stage_batch(stage: Stage, inputs: StageInputs) -> Any:
    """The ``AnalysisBatch`` that makes ``it.concept_direction`` compute ``stage``."""
    common = {"concept_basis": stage.basis, "concept_direction_mode": stage.mode, "concept_label": inputs.concept_label}
    if stage.basis == "store":
        assert stage.rows is not None  # enforced by Stage
        if stage.rows not in inputs.states:
            raise ValueError(f"stage {stage.name!r} reads {stage.rows!r} states, which this run did not gather")
        states, group_ids, weights, names = inputs.stage_rows(stage)
        return it.AnalysisBatch(
            concept_latent_state_rows=[states],
            concept_group_id_rows=[group_ids],
            concept_example_weight_rows=[weights],
            concept_group_name_rows=[names],
            **common,
        )
    if stage.target_tokens_only:
        group_a, group_b = [inputs.target_tokens[0]], [inputs.target_tokens[1]]
    else:
        group_a, group_b = list(inputs.group_a_tokens), list(inputs.group_b_tokens)
    extra = {**inputs.jlens_kwargs, "jlens_layer": stage.jlens_layer} if stage.jlens_layer is not None else {}
    return it.AnalysisBatch(concept_group_a=group_a, concept_group_b=group_b, **common, **extra)


def compute_stage_directions(module: Any, stages: Sequence[Stage], inputs: StageInputs) -> dict[str, dict[str, Any]]:
    """Every stage's unit direction, computed by ``it.concept_direction``, keyed by stage name.

    Each record carries the direction, its basis and mode, and for a store stage the norm of the unnormalized
    construction (the op returns the unit vector; the raw norm says how strongly the groups separate in that state).
    """
    records: dict[str, dict[str, Any]] = {}
    for stage in stages:
        result = it.concept_direction(module, stage_batch(stage, inputs), None, 0)
        direction = result.concept_direction.detach().cpu().float()
        record: dict[str, Any] = {"stage": stage, "direction": direction, "basis": stage.basis, "mode": stage.mode}
        if stage.basis == "store":
            record["raw_norm"] = _raw_norm(stage, inputs)
        records[stage.name] = record
    return records


def _raw_norm(stage: Stage, inputs: StageInputs) -> float:
    """The norm of a store stage's construction before normalization: how far apart the groups sit in that state."""
    states, group_ids, weights, _ = inputs.stage_rows(stage)
    geometry = compute_concept_direction_geometry(states, group_ids, direction_mode=stage.mode, example_weights=weights)
    return float(torch.linalg.vector_norm(geometry["raw_direction"]))


def cosine_matrix(records: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], torch.Tensor]:
    """Pairwise cosine similarity of the stage directions, in table order."""
    names = list(records)
    stacked = torch.stack([torch.nn.functional.normalize(records[n]["direction"], dim=0) for n in names])
    return names, stacked @ stacked.T


def steering_efficacy(
    cfg: Any, module: Any, tokenizer: Any, records: Mapping[str, Mapping[str, Any]], scale_factor: float
) -> dict[str, dict[str, float]]:
    """How far each stage's direction moves the target-token gap when added to the prompt's residual stream.

    Every direction goes through the experiment's direct-projection intervention (``it.model_fwd_intervention`` at the
    store hook, scaled by ``scale_factor``) on the module whose session the caller holds, so the stages differ only in
    the direction applied. A direction that separates the groups well need not steer: this column says which do.
    """
    from exp.concept_direction import direct_projection_batch_builder
    from exp.pipeline_patterns import direct_projection_on_module

    device = next(module.model.parameters()).device
    efficacy: dict[str, dict[str, float]] = {}
    for name, record in records.items():
        builder = direct_projection_batch_builder(cfg, record["direction"].to(device), override_scale=True)
        applied: list[set[float]] = []

        def _build(*args: Any, _builder: Any = builder, _applied: list = applied) -> Any:
            batch = _builder(*args)
            _applied.append(batch_scale_factors(batch))
            return batch

        result = direct_projection_on_module(
            cfg, module, tokenizer, f"latent_dynamics_{name}", scale_factor=scale_factor, build_analysis_batch=_build
        )
        scales = set().union(*applied) if applied else set()
        # the reported scale is the one the batch carried, never the one asked for
        if scales != {float(scale_factor)}:
            raise ValueError(
                f"steering stage {name!r}: asked for scale {scale_factor}, but the intervention batch carried {sorted(scales)}"
            )
        efficacy[name] = {
            "scale_factor": float(scale_factor),
            **{key: float(result[key]) for key in ("pre_gap", "post_gap", "gap_delta")},
        }
    return efficacy


def batch_scale_factors(batch: Any) -> set[float]:
    """The scale factors an intervention batch carries: each configured intervention's, or the batch-level one."""
    interventions = getattr(batch, "interventions", None)
    if interventions:
        payloads = [p for v in interventions.values() for p in (v if isinstance(v, list) else [v])]
        return {float(p["scale_factor"]) for p in payloads if "scale_factor" in p}
    for key in ("intervention_scale_factor", "direction_scale_factor"):
        value = getattr(batch, key, None)
        if value is not None:
            return {float(value)}
    return set()


def steering_scales(latent_cfg: Mapping[str, Any], default: float) -> list[float]:
    """The scales the steering column applies each direction at: ``steering.scale_factors`` (a sweep), else the single
    ``steering.scale_factor``, else ``default``.

    A stage direction is unit-norm while the state differences it is built from run to tens of units, so one small
    scale can read as "does not steer" when the direction only needs a larger push; a sweep says which. Giving both
    keys is refused: which one the column used would otherwise be a guess.
    """
    return _scales(latent_cfg.get("steering") or {}, default, "LATENT_DYNAMICS.steering")


def _scales(section: Mapping[str, Any], default: float, where: str) -> list[float]:
    if "scale_factors" in section and "scale_factor" in section:
        raise ValueError(f"{where} sets both scale_factor and scale_factors; give one")
    scales = section.get("scale_factors", [section.get("scale_factor", default)])
    if not scales:
        raise ValueError(f"{where}.scale_factors is empty")
    return [float(scale) for scale in scales]


def causal_steering_hook(causal_cfg: Mapping[str, Any], n_layers: int) -> str:
    """The one hook ``LATENT_DYNAMICS.causal_steering.site`` addresses on an ``n_layers``-deep stack.

    The site is a venngram site address (``point:hook_out & layers:10``), so the configuration names an activation
    point and a layer rather than one architecture's hook spelling. The causal column reads the store states at that
    site and steers there, so an address selecting more than one site is refused rather than reduced to its first.
    A ``hook`` key is refused by name: it was the hook-name form of this setting, and reading it as a site would be a
    guess about which spelling it used.
    """
    from interpretune.analysis.venngram.sites import resolve_sites

    if "hook" in causal_cfg:
        raise ValueError(
            "LATENT_DYNAMICS.causal_steering.hook is a hook name; give `site`, a venngram site address such as "
            f"'point:hook_out & layers:10' (got hook={causal_cfg['hook']!r})"
        )
    if "site" not in causal_cfg:
        raise ValueError("LATENT_DYNAMICS.causal_steering needs a site: the address whose states it reads and steers")
    sites = resolve_sites(causal_cfg["site"], n_layers=n_layers).sites
    if len(sites) != 1:
        raise ValueError(
            f"LATENT_DYNAMICS.causal_steering.site {causal_cfg['site']!r} selects {len(sites)} sites {list(sites)}; "
            "it reads and steers one"
        )
    return sites[0]


def steering_site_config(cfg: Any, hook: str) -> Any:
    """``cfg`` with its store extraction and its direct-projection intervention both moved to ``hook``.

    A direction is only steered meaningfully in the basis it was read in, so the causal column reads the store states
    at the site it steers. The configured intervention keeps its payload (operation, basis settings) and changes only
    its hook; more than one configured intervention is refused, since which one to move would be a guess.
    """
    import dataclasses

    interventions = dict(cfg.direct_projection_interventions or {})
    if len(interventions) > 1:
        raise ValueError(
            f"causal steering moves the direct-projection intervention to {hook!r}, but the configuration has "
            f"{len(interventions)} ({sorted(interventions)}); configure one"
        )
    changes: dict[str, Any] = {"store_concept_cache_key": hook}
    if interventions:
        changes["direct_projection_interventions"] = {hook: next(iter(interventions.values()))}
    else:
        changes["direct_projection_intervention_hook_pattern"] = hook
    return dataclasses.replace(cfg, **changes)


#: How far the per-example snapshots' final states may sit from the context store batch's before the two are treated as
#: different extractions. Both come from the same cache through the same arithmetic, so only float noise separates them.
RECONSTRUCTION_ATOL = 1e-4


def assemble_stage_inputs(
    *,
    snapshots: Sequence[Any],
    store_context_states: torch.Tensor,
    store_answer_states: torch.Tensor,
    group_ids: torch.Tensor,
    weights: torch.Tensor,
    group_names: Sequence[str],
    group_a_tokens: Sequence[str],
    group_b_tokens: Sequence[str],
    target_tokens: tuple[str, str],
    concept_label: str,
    jlens_kwargs: Mapping[str, Any] | None = None,
    pair_ids: torch.Tensor | None = None,
) -> tuple[StageInputs, float]:
    """Gather the state matrices every stage reads, and check they describe the extraction the store op made.

    ``snapshots`` are the per-batch :class:`ContextEnhancedExtractionSnapshot` records, in the order the context store
    batch flattened its rows. Their final states must reproduce ``store_context_states``: that is the only evidence the
    snapshot rows line up with ``group_ids`` and ``weights``, which come from the store batch, so a mismatch is refused
    rather than computed over (the earlier analysis carried this check as a "reconstruction" stage). Returns the inputs
    and the largest absolute difference found.
    """
    states = {
        "answer": torch.cat([s.answer_states for s in snapshots], dim=0),
        "context": torch.cat([s.context_states for s in snapshots], dim=0),
        "projected": torch.cat([s.projected_states for s in snapshots], dim=0),
        "final": torch.cat([s.final_latent_states for s in snapshots], dim=0),
        "store_answer": store_answer_states.detach().cpu().float(),
    }
    states = {name: matrix.detach().cpu().float() for name, matrix in states.items()}
    reference = store_context_states.detach().cpu().float()
    if states["final"].shape != reference.shape:
        raise ValueError(
            f"the snapshots hold {tuple(states['final'].shape)} final states and the context store batch "
            f"{tuple(reference.shape)}: they are not the same extraction"
        )
    max_abs_diff = float((states["final"] - reference).abs().max()) if reference.numel() else 0.0
    if max_abs_diff > RECONSTRUCTION_ATOL:
        raise ValueError(
            f"the snapshots' final states differ from the context store batch by up to {max_abs_diff:.3g} "
            f"(tolerance {RECONSTRUCTION_ATOL}), so their rows cannot be trusted to line up with the store's group ids "
            "and weights"
        )
    inputs = StageInputs(
        states=states,
        group_ids=group_ids.detach().cpu().long(),
        weights=weights.detach().cpu().float(),
        group_names=list(group_names),
        group_a_tokens=list(group_a_tokens),
        group_b_tokens=list(group_b_tokens),
        target_tokens=target_tokens,
        concept_label=concept_label,
        jlens_kwargs=dict(jlens_kwargs or {}),
        pair_ids=pair_ids,
    )
    return inputs, max_abs_diff


def paired_diagnostics(states: torch.Tensor, group_ids: torch.Tensor, weights: torch.Tensor) -> list[dict[str, float]]:
    """Per-pair records for a paired-rejection stage: what the rejection removed and what it left.

    The i-th group-A row is paired with the i-th group-B row, as the op pairs them. For each pair: both state norms, the
    norm of A's projection onto B (what was rejected), the norm of the residual (what the direction is built from), the
    A-B cosine, and the pair weight. A residual norm near zero says the pair barely separates in that state.
    """
    a_rows, b_rows = states[group_ids == 0], states[group_ids == 1]
    if a_rows.shape[0] != b_rows.shape[0]:
        raise ValueError(f"paired diagnostics need equal group sizes, got {a_rows.shape[0]} and {b_rows.shape[0]}")
    a_weights, b_weights = weights[group_ids == 0], weights[group_ids == 1]
    records = []
    for index, (a, b) in enumerate(zip(a_rows, b_rows, strict=True)):
        projection = (torch.dot(a, b) / torch.dot(b, b).clamp_min(1e-12)) * b
        records.append(
            {
                "pair_index": index,
                "group_a_norm": float(torch.linalg.vector_norm(a)),
                "group_b_norm": float(torch.linalg.vector_norm(b)),
                "projection_norm": float(torch.linalg.vector_norm(projection)),
                "residual_norm": float(torch.linalg.vector_norm(a - projection)),
                "group_cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                "pair_weight": float((a_weights[index] + b_weights[index]) / 2),
            }
        )
    return records


def kept_by_store(margin: float, *, correct_only: bool) -> bool:
    """Whether the concept extraction keeps a prompt: every prompt, or with ``correct_only`` one the model answers as
    expected (a positive margin). Everything indexed per store row selects prompts by this one rule."""
    return margin > 0 if correct_only else True


def store_pair_ids(group_ids: Sequence[int], margins: Sequence[float], *, correct_only: bool) -> torch.Tensor:
    """Each kept prompt's pair: its position within its group over every prompt, in store order.

    The op pairs a full store's i-th group-A row with its i-th group-B row; numbering the pairs over every prompt, before
    any are dropped, keeps that pairing when the store drops some (``store_concept_correct_only``).
    """
    seen: dict[int, int] = {}
    pair_ids = []
    for group, margin in zip(group_ids, margins, strict=True):
        pair = seen.get(int(group), 0)
        seen[int(group)] = pair + 1
        if kept_by_store(float(margin), correct_only=correct_only):
            pair_ids.append(pair)
    return torch.tensor(pair_ids, dtype=torch.long)


def store_rows(
    margins: torch.Tensor, expected_answers: Sequence[str], *, correct_only: bool
) -> tuple[torch.Tensor, list[str]]:
    """The margins and expected answers of the prompts the store kept, in store order.

    The concept extraction keeps every prompt, or with ``store_concept_correct_only`` only those with a positive margin,
    in prompt order; this applies the same selection, so the per-example records line up with the store rows.
    """
    if len(expected_answers) != int(margins.shape[0]):
        raise ValueError(f"{len(expected_answers)} expected answers for {int(margins.shape[0])} prompt margins")
    keep = torch.tensor([kept_by_store(float(m), correct_only=correct_only) for m in margins], dtype=torch.bool)
    return margins[keep], [answer for answer, kept in zip(expected_answers, keep.tolist(), strict=True) if kept]


def example_records(
    inputs: StageInputs,
    margins: torch.Tensor | None = None,
    expected_answers: Sequence[str] | None = None,
    prompt_details: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-example state summary: which group, its weight, each state's norm, and how far context moved the answer.

    ``answer_context_cosine`` compares the answer-position state with the context-token state;
    ``final_answer_cosine`` says how much of the answer state survives into the store state. ``margins`` are the
    experiment's own per-example scores, the expected answer's logit minus the other target's, so a positive margin
    means the model favours the expected answer (``answered_as_expected``). They are taken as given, never re-signed by
    target order. One margin per store row is required: with ``store_concept_correct_only`` the store keeps a subset of
    the prompts, and indexing all prompts' margins by row would pair examples with the wrong scores.
    ``prompt_details`` (one per store row, from :func:`kept_prompt_details`) name the prompt each example came from and
    the token positions its states were read at.
    """
    n = int(inputs.group_ids.shape[0])
    if prompt_details is not None and len(prompt_details) != n:
        raise ValueError(f"{len(prompt_details)} prompt details for {n} store rows")
    if margins is not None and int(margins.reshape(-1).shape[0]) != n:
        raise ValueError(f"{int(margins.reshape(-1).shape[0])} margins for {n} store rows")
    if expected_answers is not None and len(expected_answers) != n:
        raise ValueError(f"{len(expected_answers)} expected answers for {n} store rows")
    records = []
    states = inputs.states
    for index in range(n):
        record: dict[str, Any] = {
            "example_index": index,
            "group_id": int(inputs.group_ids[index]),
            "group": inputs.group_names[index],
            "weight": float(inputs.weights[index]),
            **{f"{name}_norm": float(torch.linalg.vector_norm(matrix[index])) for name, matrix in states.items()},
        }
        for left, right in (("answer", "context"), ("final", "answer")):
            if left in states and right in states:
                record[f"{left}_{right}_cosine"] = float(
                    torch.nn.functional.cosine_similarity(states[left][index], states[right][index], dim=0)
                )
        if expected_answers is not None:
            record["expected_answer"] = expected_answers[index]
        if margins is not None:
            record["expected_answer_margin"] = float(margins.reshape(-1)[index])
            record["answered_as_expected"] = record["expected_answer_margin"] > 0
        if prompt_details is not None:
            record.update(prompt_details[index])
        records.append(record)
    return records


def kept_prompt_details(
    prompts: Sequence[Mapping[str, Any]], margins: torch.Tensor, *, correct_only: bool
) -> list[dict[str, Any]]:
    """For each prompt the store kept, in store order: which prompt it is and where its states were read.

    ``prompt_index`` counts every prompt, kept or not, so a reader can pair an example with the full probe set.
    ``tokens`` is the cached prompt (the expected answer appended), and ``answer_index``, ``context_index`` and
    ``probe_span`` index it: the answer state is read at the answer token, the context state at ``context_index``.
    """
    if len(prompts) != int(margins.reshape(-1).shape[0]):
        raise ValueError(f"{len(prompts)} prompts for {int(margins.reshape(-1).shape[0])} prompt margins")
    details = []
    for prompt_index, (prompt, margin) in enumerate(zip(prompts, margins.reshape(-1).tolist(), strict=True)):
        if not kept_by_store(float(margin), correct_only=correct_only):
            continue
        artifact = prompt.get("prompt_alignment_artifact") or {}
        start, end = artifact.get("probe_start_index"), artifact.get("probe_end_index")
        details.append(
            {
                "prompt_index": prompt_index,
                "word": str(prompt["probe_surface_text"]),
                "prompt": str(prompt["raw_prompt"]),
                "tokens": list(artifact.get("cache_input_tokens") or []),
                "answer_index": prompt.get("cache_answer_index"),
                "context_index": prompt.get("context_token_index"),
                "context_source": prompt.get("context_token_source"),
                "probe_span": None if start is None or end is None else [int(start), int(end)],
            }
        )
    return details


#: The graph pipeline's result fields a stage keeps (its logits tensors and debug artifacts are left out of the report).
STAGE_GRAPH_FIELDS = (
    "feature_ids",
    "feature_scores",
    "pre_gap",
    "post_gap",
    "gap_delta",
    "graph_artifact",
    # a feature's position indexes this rendered prompt, so a report can name the token it sits on
    "rendered_prompt_token_ids",
)


def stage_graphs(
    cfg: Any,
    module: Any,
    tokenizer: Any,
    records: Mapping[str, Mapping[str, Any]],
    *,
    group_a_tokens: Sequence[str],
    group_b_tokens: Sequence[str],
    scale_factor: float,
    top_n: int,
) -> dict[str, dict[str, Any]]:
    """Each stage direction's attribution graph, top features and their intervention, in the held session.

    Every stage goes through the experiment's concept graph pipeline (``pipeline_on_module`` with the concept graph
    input builder), so the stages differ only in the direction; the group token ids are each group token's last
    token, as the experiment's own direction helpers resolve them.
    """
    from exp.concept_direction import _build_concept_graph_input_builder
    from exp.pipeline_patterns import pipeline_on_module

    group_a_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in group_a_tokens]
    group_b_ids = [tokenizer.encode(token, add_special_tokens=False)[-1] for token in group_b_tokens]
    graphs: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        result = pipeline_on_module(
            cfg,
            module,
            tokenizer,
            f"latent_dynamics_{name}",
            scale_factor=scale_factor,
            top_n=top_n,
            build_graph_analysis_inputs=_build_concept_graph_input_builder(
                cfg, record["direction"], group_a_ids, group_b_ids
            ),
        )
        graphs[name] = {field: result.get(field) for field in STAGE_GRAPH_FIELDS}
    return graphs


def _sweep(cfg: Any, module: Any, tokenizer: Any, records: Mapping[str, Any], scales: Sequence[float]) -> dict:
    """Each stage's steering efficacy at every scale: per stage, one record per scale."""
    per_scale = [steering_efficacy(cfg, module, tokenizer, records, scale) for scale in scales]
    return {name: [efficacy[name] for efficacy in per_scale] for name in records}


def reference_norms(records: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    """The norm a relative scale multiplies, per stage: the stage's own raw norm (the group separation it was built
    from), or for a stage without one (the embed-basis stages) the mean raw norm of the stages that have one."""
    raw = {name: float(rec["raw_norm"]) for name, rec in records.items() if rec.get("raw_norm") is not None}
    if not raw:
        raise ValueError("relative steering scales need at least one stage with a raw norm to scale by")
    mean = sum(raw.values()) / len(raw)
    return {name: raw.get(name, mean) for name in records}


def _relative_sweep(
    cfg: Any, module: Any, tokenizer: Any, records: Mapping[str, Any], fractions: Sequence[float]
) -> dict:
    """Each stage steered at fractions of its own reference norm: per stage, one record per fraction.

    Residual norms grow with depth, so an absolute scale that moves the output at one site is negligible at another;
    a fraction of the separation the direction was read from is comparable across stages and sites.
    """
    norms = reference_norms(records)
    out: dict[str, list[dict[str, float]]] = {}
    for name, record in records.items():
        out[name] = []
        for fraction in fractions:
            (efficacy,) = steering_efficacy(cfg, module, tokenizer, {name: record}, fraction * norms[name]).values()
            out[name].append({"fraction": float(fraction), "reference_norm": norms[name], **efficacy})
    return out


def _stage_records(
    cfg: Any, module: Any, tokenizer: Any, model_backend: Any, stages: Sequence[Stage], latent_cfg: Mapping[str, Any]
) -> tuple[dict[str, Any], StageInputs, float, list[Any], list[Any]]:
    """Both store extractions at ``cfg.store_concept_cache_key``, a snapshot per batch, and every stage's direction.

    Returns the stage records, the inputs they were computed from, the reconstruction check's largest difference, and
    the per-batch logit differences and prompts the per-example records read.
    """
    from exp.concept_direction import (
        construct_concept_pair_analysis_inputs,
        execute_concept_latent_extraction_ops,
        resolve_target_tokens,
    )
    from interpretune.analysis.ops.bundled.concept.concept_ops import flatten_concept_store_rows

    from exp.analysis.concept_direction_analysis import capture_context_enhanced_extraction_snapshot

    device = next(module.model.parameters()).device
    (target_a_id, target_b_id), target_tokens = resolve_target_tokens(cfg, tokenizer)
    cached, answer_idx, context_idx, labels, logit_diffs, _, prompts = construct_concept_pair_analysis_inputs(
        cfg, module, tokenizer, model_backend, device, target_a_id, target_b_id
    )
    extraction_args = (module, cfg, cached, answer_idx, context_idx, labels, logit_diffs, len(prompts))
    plain = execute_concept_latent_extraction_ops(*extraction_args, extraction_mode="answer_position_state")
    context = execute_concept_latent_extraction_ops(*extraction_args, extraction_mode=cfg.store_latent_extraction_mode)
    plain_states, _, _, _ = _flatten(flatten_concept_store_rows, plain[-1])
    context_states, group_ids, weights, group_names = _flatten(flatten_concept_store_rows, context[-1])
    snapshots = [
        capture_context_enhanced_extraction_snapshot(
            it.AnalysisBatch(
                cache=batch,
                answer_indices=answer,
                context_token_indices=context_index,
                concept_cache_key=cfg.store_concept_cache_key,
                orig_labels=label,
                logit_diffs=diff,
                use_answer_state_as_basis=cfg.use_answer_state_as_basis,
            ),
            context_scale=cfg.context_enhanced_scale,
            use_answer_state_as_basis=cfg.use_answer_state_as_basis,
        )
        for batch, answer, context_index, label, diff in zip(
            cached, answer_idx, context_idx, labels, logit_diffs, strict=True
        )
        # one snapshot per store row: the store keeps only some prompts when it keeps correct answers only
        if kept_by_store(float(diff.reshape(-1)[0]), correct_only=cfg.store_concept_correct_only)
    ]
    inputs, reconstruction_diff = assemble_stage_inputs(
        snapshots=snapshots,
        store_context_states=context_states,
        store_answer_states=plain_states,
        group_ids=group_ids,
        weights=weights,
        group_names=group_names,
        group_a_tokens=cfg.concept_pair.group_a_tokens,
        group_b_tokens=cfg.concept_pair.group_b_tokens,
        target_tokens=target_tokens,
        concept_label=cfg.concept_pair.concept_label,
        jlens_kwargs=latent_cfg.get("jlens_kwargs"),
        pair_ids=store_pair_ids(
            [int(label.reshape(-1)[0]) for label in labels],
            [float(diff.reshape(-1)[0]) for diff in logit_diffs],
            correct_only=cfg.store_concept_correct_only,
        ),
    )
    return compute_stage_directions(module, stages, inputs), inputs, reconstruction_diff, logit_diffs, prompts


def run_latent_dynamics_analysis(config_path: str | Path) -> dict[str, Any]:
    """Run the latent-dynamics analysis an experiment configuration describes, and return its report.

    One session: the concept-pair inputs, both store extractions (answer-position and context-enhanced), a snapshot per
    batch, then every stage of the ``LATENT_DYNAMICS`` table through ``it.concept_direction``. The report holds each
    stage's direction record, each direction's steering efficacy at every scale :func:`steering_scales` reads (a list
    per stage, one record per scale; see :func:`steering_efficacy`), the stage cosine
    matrix, per-pair diagnostics for every paired store stage, per-example records and the reconstruction check's
    largest difference. With ``LATENT_DYNAMICS.causal_steering`` (a ``site`` address and its scales), it also reads
    the store states at the hook that site resolves to, recomputes every stage there and steers at the same hook
    (``causal_steering``): the store-site column at ``unembed.hook_in`` is only a readout, since that site follows the
    final norm.
    """
    from exp.concept_direction import build_notebook_harness_config, maybe_zero_softcap, phase_run_name
    from interpretune.harness.nb_harness_utils import _resolve_model_layer_count
    from interpretune.harness.sessions import experiment_session

    cfg, cleanup_work_root, payload = build_notebook_harness_config(config_path)
    latent_cfg = dict(payload.get("LATENT_DYNAMICS") or {})
    stages = stages_from_config(latent_cfg)
    causal_cfg = dict(latent_cfg.get("causal_steering") or {})
    try:
        with experiment_session(cfg.work_root, phase_run_name(cfg, "latent_dynamics"), **cfg.session_kwargs) as (
            _,
            module,
            tokenizer,
        ):
            model_backend = getattr(module, "_model_backend", None)
            if model_backend is None:
                raise RuntimeError("the experiment session's module exposes no _model_backend for the extraction ops")
            with maybe_zero_softcap(module, cfg):
                records, inputs, reconstruction_diff, logit_diffs, prompts = _stage_records(
                    cfg, module, tokenizer, model_backend, stages, latent_cfg
                )
                steering = _sweep(
                    cfg, module, tokenizer, records, steering_scales(latent_cfg, cfg.default_scale_factor)
                )
                graphs = None
                if (latent_cfg.get("stage_graphs") or {}).get("enabled", False):
                    graphs = stage_graphs(
                        cfg,
                        module,
                        tokenizer,
                        records,
                        group_a_tokens=cfg.concept_pair.group_a_tokens,
                        group_b_tokens=cfg.concept_pair.group_b_tokens,
                        scale_factor=float(latent_cfg["stage_graphs"].get("scale_factor", cfg.default_scale_factor)),
                        top_n=int(latent_cfg.get("stage_top_n", cfg.top_n)),
                    )
                causal = None
                if causal_cfg:
                    hook = causal_steering_hook(causal_cfg, _resolve_model_layer_count(module))
                    site_cfg = steering_site_config(cfg, hook)
                    causal_records, _, causal_diff, _, _ = _stage_records(
                        site_cfg, module, tokenizer, model_backend, stages, latent_cfg
                    )
                    causal = {
                        "site": str(causal_cfg["site"]),
                        "hook": hook,
                        "reconstruction_max_abs_diff": causal_diff,
                        "raw_norms": {name: rec.get("raw_norm") for name, rec in causal_records.items()},
                    }
                    if "relative_scales" in causal_cfg:
                        fractions = [float(f) for f in causal_cfg["relative_scales"]]
                        causal["efficacy"] = _relative_sweep(site_cfg, module, tokenizer, causal_records, fractions)
                    else:
                        scales = _scales(causal_cfg, cfg.default_scale_factor, "LATENT_DYNAMICS.causal_steering")
                        causal["efficacy"] = _sweep(site_cfg, module, tokenizer, causal_records, scales)
    finally:
        if cleanup_work_root:
            import shutil

            shutil.rmtree(cfg.work_root, ignore_errors=True)

    names, cosines = cosine_matrix(records)
    diagnostics = {
        stage.name: paired_diagnostics(*inputs.stage_rows(stage)[:3])
        for stage in stages
        if stage.basis == "store" and stage.mode == "paired_rejection" and stage.rows is not None
    }
    projection = dict(latent_cfg.get("projection") or {"method": "umap", "fallback": "refuse"})
    all_margins = torch.cat([d.reshape(-1).float().cpu() for d in logit_diffs])
    correct_only = cfg.store_concept_correct_only
    return {
        "experiment_name": cfg.experiment_name,
        "stages": records,
        # at the store site (`unembed.hook_in` by default, after the final norm) a direction's effect on the gap is
        # linear in its scale: a readout of its alignment with the target unembedding difference, not steering
        "steering": steering,
        "steering_site": cfg.store_concept_cache_key,
        "causal_steering": causal,
        "stage_graphs": graphs,
        "cosine": {"stages": names, "matrix": cosines},
        "pair_diagnostics": diagnostics,
        "examples": example_records(
            inputs,
            *store_rows(all_margins, [str(prompt["expected_answer"]) for prompt in prompts], correct_only=correct_only),
            prompt_details=kept_prompt_details(prompts, all_margins, correct_only=correct_only),
        ),
        "state_separation": state_separation(inputs),
        "projections": {
            "trajectory": trajectory_projection(inputs, projection),
            # the answer states sit far from every other state, which compresses the rest of a shared projection
            "trajectory_without_answer": trajectory_projection(
                inputs, projection, sources=tuple(s for s in TRAJECTORY_SOURCES if s != "answer")
            ),
            "directions": direction_projection(records, projection),
        },
        "reconstruction_max_abs_diff": reconstruction_diff,
    }


#: The per-example states a trajectory visits, in the order the extraction computes them.
TRAJECTORY_SOURCES: tuple[str, ...] = ("answer", "context", "projected", "final")


def project_rows(matrix: torch.Tensor, metadata: Sequence[Mapping[str, Any]], projection: Mapping[str, Any]) -> dict:
    """Project ``matrix`` with ``project_embeddings(**projection)`` and attach each row's coordinates to its metadata.

    ``method`` and ``backend`` are what ran, not what was asked for: with ``fallback: pca`` a UMAP request that could
    not be met comes back labelled ``pca``, and the request is kept in ``metadata``.
    """
    from interpretune.utils.latent_state_projection import project_embeddings

    if matrix.shape[0] != len(metadata):
        raise ValueError(f"{matrix.shape[0]} rows to project but {len(metadata)} metadata records")
    result = project_embeddings(matrix, **dict(projection))
    records = [
        {**meta, **{axis: float(value) for axis, value in zip(("x", "y", "z"), coordinates)}}
        for meta, coordinates in zip(metadata, result.coordinates, strict=True)
    ]
    return {"method": result.method, "backend": result.backend, "metadata": result.metadata, "records": records}


def trajectory_projection(
    inputs: StageInputs, projection: Mapping[str, Any], sources: Sequence[str] = TRAJECTORY_SOURCES
) -> dict:
    """Every example's path through its states (answer, context, projected, final, or ``sources``) in one projection."""
    if unknown := sorted(set(sources) - set(TRAJECTORY_SOURCES)):
        raise ValueError(f"unknown trajectory sources {unknown}; known: {list(TRAJECTORY_SOURCES)}")
    sources = [source for source in TRAJECTORY_SOURCES if source in sources and source in inputs.states]
    n = int(inputs.group_ids.shape[0])
    rows = [inputs.states[source][index] for index in range(n) for source in sources]
    metadata = [
        {"example_index": index, "state": source, "group": inputs.group_names[index]}
        for index in range(n)
        for source in sources
    ]
    return project_rows(torch.stack(rows), metadata, projection)


def state_separation(inputs: StageInputs) -> list[dict[str, Any]]:
    """How far apart the two groups sit in each state, measured in the full residual space rather than a projection.

    Per state: the distance between the two (unweighted) group centroids, the mean distance of an example to its own
    centroid, their ratio (above 1, the groups are further apart than they are wide), the cosine between the centroids
    (near 1, both groups share a large common component), and leave-one-out nearest-centroid accuracy.
    """
    a, b = inputs.group_ids == 0, inputs.group_ids == 1
    if int(a.sum()) < 2 or int(b.sum()) < 2:
        raise ValueError("state separation needs at least two examples in each group")
    rows = []
    for source in [source for source in TRAJECTORY_SOURCES if source in inputs.states]:
        states = inputs.states[source].float()
        mean_a, mean_b = states[a].mean(dim=0), states[b].mean(dim=0)
        between = float(torch.linalg.vector_norm(mean_a - mean_b))
        within = float(
            torch.cat(
                [
                    torch.linalg.vector_norm(states[a] - mean_a, dim=1),
                    torch.linalg.vector_norm(states[b] - mean_b, dim=1),
                ]
            ).mean()
        )
        correct = 0
        for index in range(states.shape[0]):
            keep = torch.ones(states.shape[0], dtype=torch.bool)
            keep[index] = False
            centroids = [states[keep & (inputs.group_ids == group)].mean(dim=0) for group in (0, 1)]
            nearest = min((0, 1), key=lambda group: float(torch.linalg.vector_norm(states[index] - centroids[group])))
            correct += int(nearest == int(inputs.group_ids[index]))
        rows.append(
            {
                "state": source,
                "centroid_distance": between,
                "within_group_spread": within,
                "separation_ratio": between / max(within, 1e-12),
                "centroid_cosine": float(torch.nn.functional.cosine_similarity(mean_a, mean_b, dim=0)),
                "leave_one_out_accuracy": correct / int(states.shape[0]),
            }
        )
    return rows


def direction_projection(records: Mapping[str, Mapping[str, Any]], projection: Mapping[str, Any]) -> dict:
    """The stage directions in one projection, so stages that agree sit together."""
    names = list(records)
    matrix = torch.stack([torch.nn.functional.normalize(records[name]["direction"], dim=0) for name in names])
    metadata = [{"stage": name, "basis": records[name]["basis"], "mode": records[name]["mode"]} for name in names]
    return project_rows(matrix, metadata, projection)


def _flatten(flatten: Any, aggregated_batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """A store batch's rows flattened into (states, group ids, weights, group names)."""
    return flatten(
        aggregated_batch.concept_latent_state_rows,
        aggregated_batch.concept_group_id_rows,
        aggregated_batch.concept_group_name_rows,
        aggregated_batch.concept_example_weight_rows,
    )
