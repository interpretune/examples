"""The latent-dynamics stage table computes each stage through ``it.concept_direction`` and matches the earlier
analysis's own builders, which are kept here as the reference they are checked against."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import interpretune as it
from interpretune.analysis.ops.bundled.concept.concept_ops import concept_direction_impl

from exp.analysis.concept_direction_latent_dynamics import (
    DEFAULT_STAGES,
    RECONSTRUCTION_ATOL,
    Stage,
    StageInputs,
    assemble_stage_inputs,
    compute_stage_directions,
    cosine_matrix,
    example_records,
    kept_prompt_details,
    kept_by_store,
    store_pair_ids,
    store_rows,
    paired_diagnostics,
    stage_batch,
    stages_from_config,
    state_separation,
    trajectory_projection,
)


# The earlier analysis's builders (interpretune 23a0a908, concept_direction_latent_dynamics.py), verbatim in substance.
def _reference_mean_difference(states, group_ids, weights):
    a, b = group_ids == 0, group_ids == 1
    mean_a = torch.sum(states[a] * weights[a].unsqueeze(-1), dim=0) / weights[a].sum().clamp_min(1e-12)
    mean_b = torch.sum(states[b] * weights[b].unsqueeze(-1), dim=0) / weights[b].sum().clamp_min(1e-12)
    raw = mean_a - mean_b
    return raw, raw / torch.linalg.vector_norm(raw).clamp_min(1e-12)


def _reference_paired_rejection(states, group_ids, weights):
    a, b = group_ids == 0, group_ids == 1
    residuals, pair_weights = [], []
    for state_a, state_b, weight_a, weight_b in zip(states[a], states[b], weights[a], weights[b], strict=True):
        projection = (torch.dot(state_a, state_b) / torch.dot(state_b, state_b).clamp_min(1e-12)) * state_b
        residuals.append(state_a - projection)
        pair_weights.append((weight_a + weight_b) / 2)
    r, w = torch.stack(residuals), torch.stack(pair_weights)
    raw = torch.sum(r * w.unsqueeze(-1), dim=0) / w.sum().clamp_min(1e-12)
    return raw, raw / torch.linalg.vector_norm(raw).clamp_min(1e-12)


@pytest.fixture(autouse=True)
def _op_without_a_model(monkeypatch) -> None:
    """Run the op's own implementation; the public wrapper's capability gate needs a loaded model these tests lack."""
    monkeypatch.setattr(
        it,
        "concept_direction",
        lambda module, batch, *_: concept_direction_impl(SimpleNamespace(analysis_cfg=None), batch, None, 0),
    )


@pytest.fixture
def inputs() -> StageInputs:
    torch.manual_seed(7)
    n_pairs, d = 4, 16
    group_ids = torch.tensor([0] * n_pairs + [1] * n_pairs)
    states = {
        source: torch.randn(2 * n_pairs, d) for source in ("answer", "context", "projected", "final", "store_answer")
    }
    # distinct, non-uniform weights, so a stage that dropped or reordered them would not match
    weights = torch.linspace(0.5, 2.0, 2 * n_pairs)
    names = ["Fruit"] * n_pairs + ["Color"] * n_pairs
    return StageInputs(
        states=states,
        group_ids=group_ids,
        weights=weights,
        group_names=names,
        group_a_tokens=["apple", "banana"],
        group_b_tokens=["red", "blue"],
        target_tokens=("fruit", "color"),
        concept_label="Fruit -> Color",
    )


STORE_STAGES = [stage for stage in DEFAULT_STAGES if stage.basis == "store"]


@pytest.mark.parametrize("stage", STORE_STAGES, ids=lambda s: s.name)
def test_store_stage_matches_the_earlier_builder(stage: Stage, inputs: StageInputs) -> None:
    record = compute_stage_directions(None, [stage], inputs)[stage.name]
    states = inputs.states[stage.rows]
    weights = torch.ones_like(inputs.weights) if stage.weights == "unit" else inputs.weights
    reference = _reference_mean_difference if stage.mode == "mean_difference" else _reference_paired_rejection
    raw, unit = reference(states, inputs.group_ids, weights)
    torch.testing.assert_close(record["direction"], unit, rtol=1e-5, atol=1e-6)
    assert record["raw_norm"] == pytest.approx(float(torch.linalg.vector_norm(raw)), rel=1e-5)


def test_unit_weights_change_the_paired_direction(inputs: StageInputs) -> None:
    # positive control for the parity test: the two final-state stages differ only in weights, and must differ
    weighted, unweighted = (
        compute_stage_directions(None, [s], inputs)[s.name]["direction"]
        for s in DEFAULT_STAGES
        if s.name in ("store_context_paired_rejection", "paired_residual_unweighted")
    )
    assert not torch.allclose(weighted, unweighted)


def test_embed_stages_name_their_token_groups(inputs: StageInputs) -> None:
    target = stage_batch(DEFAULT_STAGES[0], inputs)
    groups = stage_batch(DEFAULT_STAGES[1], inputs)
    assert (target.concept_group_a, target.concept_group_b) == (["fruit"], ["color"])
    assert (groups.concept_group_a, groups.concept_group_b) == (["apple", "banana"], ["red", "blue"])
    assert target.concept_basis == "embed"


def test_jlens_stages_come_from_config(inputs: StageInputs) -> None:
    stages = stages_from_config({"jlens_layers": [8, 14]})
    jlens = [s for s in stages if s.basis.startswith("jlens_")]
    assert [(s.name, s.basis, s.jlens_layer) for s in jlens] == [
        ("jlens_folded_l8", "jlens_folded", 8),
        ("jlens_folded_l14", "jlens_folded", 14),
    ]
    batch = stage_batch(jlens[0], inputs)
    assert (batch.concept_basis, batch.jlens_layer) == ("jlens_folded", 8)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"name": "x", "basis": "jlens_paper", "mode": "mean_difference", "jlens_layer": 3}, "is not one of"),
        ({"name": "x", "basis": "store", "mode": "mean_difference"}, "names its rows"),
        ({"name": "x", "basis": "embed", "mode": "mean_difference", "rows": "answer"}, "only a store stage"),
        ({"name": "x", "basis": "jlens_folded", "mode": "mean_difference"}, "names its jlens_layer"),
        ({"name": "x", "basis": "store", "mode": "single_group", "rows": "answer"}, "is not 'mean_difference'"),
    ],
)
def test_a_malformed_stage_is_refused_by_name(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        Stage(**kwargs)


def test_duplicate_stage_names_are_refused() -> None:
    with pytest.raises(ValueError, match="repeat names"):
        stages_from_config({"stages": [{"name": "a", "basis": "embed", "mode": "mean_difference"}] * 2})


def test_misaligned_inputs_are_refused(inputs: StageInputs) -> None:
    with pytest.raises(ValueError, match="not aligned"):
        StageInputs(**{**inputs.__dict__, "weights": inputs.weights[:-1]})


def test_cosine_matrix_is_symmetric_with_a_unit_diagonal(inputs: StageInputs) -> None:
    names, cosines = cosine_matrix(compute_stage_directions(None, STORE_STAGES, inputs))
    assert names == [s.name for s in STORE_STAGES]
    torch.testing.assert_close(cosines, cosines.T)
    torch.testing.assert_close(torch.diagonal(cosines), torch.ones(len(names)))


def _snapshots(inputs: StageInputs, split: int = 3) -> list[SimpleNamespace]:
    """Per-batch snapshot stand-ins holding ``inputs``' rows, split into two batches as a real run produces."""
    s = inputs.states
    return [
        SimpleNamespace(
            answer_states=s["answer"][rows],
            context_states=s["context"][rows],
            projected_states=s["projected"][rows],
            final_latent_states=s["final"][rows],
        )
        for rows in (slice(0, split), slice(split, None))
    ]


def _assemble(inputs: StageInputs, **overrides):
    kwargs = dict(
        snapshots=_snapshots(inputs),
        store_context_states=inputs.states["final"],
        store_answer_states=inputs.states["store_answer"],
        group_ids=inputs.group_ids,
        weights=inputs.weights,
        group_names=inputs.group_names,
        group_a_tokens=inputs.group_a_tokens,
        group_b_tokens=inputs.group_b_tokens,
        target_tokens=inputs.target_tokens,
        concept_label=inputs.concept_label,
    )
    kwargs.update(overrides)
    return assemble_stage_inputs(**kwargs)


def test_assembly_concatenates_the_snapshots_in_order(inputs: StageInputs) -> None:
    assembled, diff = _assemble(inputs)
    assert diff == 0.0
    for source in ("answer", "context", "projected", "final", "store_answer"):
        torch.testing.assert_close(assembled.states[source], inputs.states[source])
    assert assembled.group_names == inputs.group_names


def test_assembly_refuses_snapshots_that_do_not_reproduce_the_store_states(inputs: StageInputs) -> None:
    """The reconstruction check is the only evidence the snapshot rows line up with the store's group ids."""
    shifted = inputs.states["final"].clone()
    shifted[0] += 10 * RECONSTRUCTION_ATOL
    with pytest.raises(ValueError, match="differ from the context store batch"):
        _assemble(inputs, store_context_states=shifted)
    # rows in another order are the misalignment the check exists for
    with pytest.raises(ValueError, match="differ from the context store batch"):
        _assemble(inputs, store_context_states=inputs.states["final"].flip(0))
    with pytest.raises(ValueError, match="not the same extraction"):
        _assemble(inputs, store_context_states=inputs.states["final"][:-1])


def test_assembly_tolerates_float_noise_and_reports_it(inputs: StageInputs) -> None:
    noisy = inputs.states["final"] + RECONSTRUCTION_ATOL / 10
    _, diff = _assemble(inputs, store_context_states=noisy)
    assert 0 < diff <= RECONSTRUCTION_ATOL


def test_paired_diagnostics_split_each_group_a_state_into_projection_and_residual(inputs: StageInputs) -> None:
    states, ids, weights = inputs.states["final"], inputs.group_ids, inputs.weights
    records = paired_diagnostics(states, ids, weights)
    assert [r["pair_index"] for r in records] == list(range(int((ids == 0).sum())))
    for record, a, b in zip(records, states[ids == 0], states[ids == 1], strict=True):
        # the projection and residual are orthogonal parts of A, so their norms satisfy Pythagoras
        assert record["projection_norm"] ** 2 + record["residual_norm"] ** 2 == pytest.approx(
            record["group_a_norm"] ** 2
        )
        assert record["group_b_norm"] == pytest.approx(float(torch.linalg.vector_norm(b)))
        assert record["group_cosine"] == pytest.approx(float(torch.nn.functional.cosine_similarity(a, b, dim=0)))
    # the pair weights are the ones the paired-rejection direction is averaged with
    pair_weights = torch.tensor([r["pair_weight"] for r in records])
    torch.testing.assert_close(pair_weights, (weights[ids == 0] + weights[ids == 1]) / 2)


def test_paired_diagnostics_refuse_unequal_groups(inputs: StageInputs) -> None:
    ids = inputs.group_ids.clone()
    ids[0] = 1
    with pytest.raises(ValueError, match="equal group sizes"):
        paired_diagnostics(inputs.states["final"], ids, inputs.weights)


def test_example_records_name_each_row_and_bound_the_cosines(inputs: StageInputs) -> None:
    margins = torch.arange(int(inputs.group_ids.shape[0]), dtype=torch.float32)
    records = example_records(inputs, margins)
    assert [r["group"] for r in records] == inputs.group_names
    assert [r["group_id"] for r in records] == inputs.group_ids.tolist()
    assert [r["expected_answer_margin"] for r in records] == margins.tolist()
    for record, answer in zip(records, inputs.states["answer"], strict=True):
        assert record["answer_norm"] == pytest.approx(float(torch.linalg.vector_norm(answer)))
        assert -1.0 <= record["answer_context_cosine"] <= 1.0
        assert -1.0 <= record["final_answer_cosine"] <= 1.0


LATENT_DYNAMICS_CONFIG = (
    "concept_direction/configs/gemma3_1b_it_local_latent_dynamics_color_fruit_orange_fs_l10_n5.yaml"
)


def test_the_shipped_config_resolves_to_the_default_table_and_valid_projection_settings() -> None:
    """The config is loaded through the pipeline's own loader, EXTENDS chain included, so a typo fails on CPU.

    Its table is the default one plus a folded J-lens stage at layers 6, 10 and 14, around the block-10 store site.
    """
    import inspect
    from pathlib import Path

    from exp.concept_direction import load_experiment_config
    from interpretune.utils.latent_state_projection import project_embeddings

    # absolute(), not resolve(): in a pulled snapshot this file is a symlink into the blob store
    root = Path(__file__).absolute().parents[1]
    payload = load_experiment_config(root / LATENT_DYNAMICS_CONFIG)
    latent = payload["LATENT_DYNAMICS"]
    stages = stages_from_config(latent)
    assert stages[: len(DEFAULT_STAGES)] == DEFAULT_STAGES
    assert [(s.name, s.basis, s.mode, s.jlens_layer) for s in stages[len(DEFAULT_STAGES) :]] == [
        (f"jlens_folded_l{layer}", "jlens_folded", "paired_rejection", layer) for layer in (6, 10, 14)
    ]
    assert latent["causal_steering"]["hook"] == "blocks.10.hook_out"
    accepted = set(inspect.signature(project_embeddings).parameters) - {"values"}
    unknown = sorted(set(latent["projection"]) - accepted)
    assert not unknown, f"projection settings project_embeddings does not take: {unknown}"
    assert latent["projection"]["fallback"] == "refuse"


PCA = {"method": "pca", "n_components": 2}


def test_the_trajectory_projection_visits_each_examples_states_in_extraction_order(inputs: StageInputs) -> None:
    from exp.analysis.concept_direction_latent_dynamics import TRAJECTORY_SOURCES, trajectory_projection

    result = trajectory_projection(inputs, PCA)
    n = int(inputs.group_ids.shape[0])
    assert result["method"] == "pca"
    assert len(result["records"]) == n * len(TRAJECTORY_SOURCES)
    first = result["records"][: len(TRAJECTORY_SOURCES)]
    assert [r["state"] for r in first] == list(TRAJECTORY_SOURCES)
    assert {r["example_index"] for r in first} == {0}
    assert [r["group"] for r in result["records"][:: len(TRAJECTORY_SOURCES)]] == inputs.group_names
    assert all({"x", "y"} <= set(r) for r in result["records"])


def test_the_direction_projection_has_one_point_per_stage(inputs: StageInputs) -> None:
    from exp.analysis.concept_direction_latent_dynamics import direction_projection

    records = compute_stage_directions(None, STORE_STAGES, inputs)
    result = direction_projection(records, PCA)
    assert [r["stage"] for r in result["records"]] == list(records)
    assert [r["mode"] for r in result["records"]] == [records[name]["mode"] for name in records]


def test_a_umap_request_that_cannot_be_met_is_refused_not_relabelled(inputs: StageInputs) -> None:
    from interpretune.utils.latent_state_projection import UmapUnavailableError

    from exp.analysis.concept_direction_latent_dynamics import direction_projection

    records = compute_stage_directions(None, STORE_STAGES[:2], inputs)  # two points: too few for UMAP
    with pytest.raises(UmapUnavailableError):
        direction_projection(records, {"method": "umap", "fallback": "refuse", "n_neighbors": 6})
    relabelled = direction_projection(records, {"method": "umap", "fallback": "pca", "n_neighbors": 6})
    assert relabelled["method"] == "pca"


def test_project_rows_refuses_metadata_that_does_not_match_the_rows() -> None:
    from exp.analysis.concept_direction_latent_dynamics import project_rows

    with pytest.raises(ValueError, match="metadata records"):
        project_rows(torch.randn(4, 3), [{}] * 3, PCA)


def test_the_margin_is_taken_as_the_experiment_scores_it_never_re_signed_by_target_order(inputs: StageInputs) -> None:
    # The experiment scores each example as the expected answer's logit minus the other target's, so the same margin
    # means the same thing whichever target was expected: a second-target example with margin 2 was answered correctly.
    n = int(inputs.group_ids.shape[0])
    margins = torch.tensor([2.0] * (n - 1) + [-3.0])
    expected = ["Fruit"] * (n // 2) + ["Color"] * (n - n // 2)
    records = example_records(inputs, margins, expected)
    assert [r["expected_answer_margin"] for r in records] == margins.tolist()
    assert [r["answered_as_expected"] for r in records] == [True] * (n - 1) + [False]
    assert [r["expected_answer"] for r in records] == expected
    with pytest.raises(ValueError, match="expected answers for"):
        example_records(inputs, margins, expected[:-1])


def test_example_records_refuse_margins_that_do_not_match_the_store_rows(inputs: StageInputs) -> None:
    n = int(inputs.group_ids.shape[0])
    with pytest.raises(ValueError, match=f"{n + 1} margins for {n} store rows"):
        example_records(inputs, torch.zeros(n + 1))


def _fake_builder(honour_scale: bool = True):
    """A batch builder shaped like the component's: one configured intervention carrying the direction and a scale."""

    def builder(cfg, direction, *, override_scale=False):
        assert override_scale, "the steering column must apply its own scale over a configured one"

        def build(prompt, a, b, scale):
            payload = {"intervention_tensor": direction, "scale_factor": scale if honour_scale else 10.0}
            return SimpleNamespace(interventions={"unembed.hook_in": [payload]})

        return build

    return builder


def _fake_projection_for(module, applied):
    def _fake_projection(cfg, mod, tokenizer, label, *, scale_factor, build_analysis_batch):
        assert mod is module, "steering must reuse the caller's module, not open a session per stage"
        batch = build_analysis_batch("prompt", 1, 2, scale_factor)
        (payload,) = batch.interventions["unembed.hook_in"]
        applied.append((label, payload["intervention_tensor"], payload["scale_factor"]))
        # A gap that depends on the direction and the applied scale, so a mix-up shows in the result.
        gap_delta = float(payload["intervention_tensor"].sum()) * payload["scale_factor"]
        return {"pre_gap": 1.0, "post_gap": 1.0 + gap_delta, "gap_delta": gap_delta}

    return _fake_projection


def test_steering_applies_each_stage_direction_in_the_held_session(monkeypatch, inputs: StageInputs) -> None:
    import exp.concept_direction as concept_direction
    import exp.pipeline_patterns as pipeline_patterns
    from exp.analysis.concept_direction_latent_dynamics import steering_efficacy

    records = compute_stage_directions(None, STORE_STAGES, inputs)
    module = SimpleNamespace(model=torch.nn.Linear(2, 2))
    applied: list[tuple[str, torch.Tensor, float]] = []
    monkeypatch.setattr(concept_direction, "direct_projection_batch_builder", _fake_builder())
    monkeypatch.setattr(pipeline_patterns, "direct_projection_on_module", _fake_projection_for(module, applied))
    efficacy = steering_efficacy(SimpleNamespace(), module, None, records, 4.0)

    assert list(efficacy) == list(records)
    assert [label for label, _, _ in applied] == [f"latent_dynamics_{name}" for name in records]
    for (_, direction, scale), (name, record) in zip(applied, records.items(), strict=True):
        assert scale == 4.0
        torch.testing.assert_close(direction, record["direction"])
        expected = float(record["direction"].sum()) * 4.0
        assert efficacy[name] == {
            "scale_factor": 4.0,
            "pre_gap": 1.0,
            "post_gap": pytest.approx(1.0 + expected),
            "gap_delta": pytest.approx(expected),
        }


def test_steering_refuses_a_batch_that_did_not_carry_the_asked_scale(monkeypatch, inputs: StageInputs) -> None:
    import exp.concept_direction as concept_direction
    import exp.pipeline_patterns as pipeline_patterns
    from exp.analysis.concept_direction_latent_dynamics import steering_efficacy

    records = compute_stage_directions(None, STORE_STAGES, inputs)
    module = SimpleNamespace(model=torch.nn.Linear(2, 2))
    monkeypatch.setattr(concept_direction, "direct_projection_batch_builder", _fake_builder(honour_scale=False))
    monkeypatch.setattr(pipeline_patterns, "direct_projection_on_module", _fake_projection_for(module, []))
    with pytest.raises(ValueError, match=r"asked for scale 25.0, but the intervention batch carried \[10.0\]"):
        steering_efficacy(SimpleNamespace(), module, None, records, 25.0)


def test_a_configured_intervention_scale_wins_unless_the_caller_overrides_it() -> None:
    from exp.concept_direction import direct_projection_batch_builder

    direction = torch.arange(4.0)
    cfg = SimpleNamespace(
        direct_projection_interventions={"unembed.hook_in": {"op": "add", "scale_factor": 10.0}},
        direct_projection_intervention_scale_factor=None,
        direct_projection_intervention_mode=None,
        direct_projection_intervention_use_intervention_tensor_as_basis=None,
    )
    fixed = direct_projection_batch_builder(cfg, direction)("prompt", 3, 5, 50.0)
    assert fixed.interventions["unembed.hook_in"]["scale_factor"] == 10.0
    swept = direct_projection_batch_builder(cfg, direction, override_scale=True)("prompt", 3, 5, 50.0)
    assert swept.interventions["unembed.hook_in"]["scale_factor"] == 50.0


def test_the_direct_projection_batch_injects_the_direction_and_the_callers_scale() -> None:
    from exp.concept_direction import direct_projection_batch_builder

    direction = torch.arange(4.0)
    cfg = SimpleNamespace(
        direct_projection_interventions={"blocks.10.hook_resid_post": [{"op": "add"}, {"op": "add"}]},
        direct_projection_intervention_scale_factor=None,
        direct_projection_intervention_mode="scale",
        direct_projection_intervention_use_intervention_tensor_as_basis=None,
    )
    batch = direct_projection_batch_builder(cfg, direction)("prompt", 3, 5, 2.5)
    payloads = batch.interventions["blocks.10.hook_resid_post"]
    assert [p["scale_factor"] for p in payloads] == [2.5, 2.5]
    # interpretune reads the operation from `op` and refuses the retired `mode` key; a payload's own op wins
    assert [p["op"] for p in payloads] == ["add", "add"] and not any("mode" in p for p in payloads)
    assert all(p["intervention_tensor"] is direction for p in payloads)
    assert batch.logit_target_ids.tolist() == [3]


def test_steering_scales_reads_a_sweep_a_single_scale_or_the_default() -> None:
    from exp.analysis.concept_direction_latent_dynamics import steering_scales

    assert steering_scales({"steering": {"scale_factors": [10, 25]}}, 3.0) == [10.0, 25.0]
    assert steering_scales({"steering": {"scale_factor": 7}}, 3.0) == [7.0]
    assert steering_scales({}, 3.0) == [3.0]
    with pytest.raises(ValueError, match="both scale_factor and scale_factors"):
        steering_scales({"steering": {"scale_factor": 7, "scale_factors": [10]}}, 3.0)
    with pytest.raises(ValueError, match="empty"):
        steering_scales({"steering": {"scale_factors": []}}, 3.0)


def test_the_causal_site_moves_the_extraction_and_the_intervention_together() -> None:
    import dataclasses

    from exp.analysis.concept_direction_latent_dynamics import steering_site_config

    @dataclasses.dataclass
    class _Cfg:
        store_concept_cache_key: str = "unembed.hook_in"
        direct_projection_interventions: dict | None = None
        direct_projection_intervention_hook_pattern: str | None = None

    payload = {"op": "add", "scale_factor": 10.0}
    moved = steering_site_config(
        _Cfg(direct_projection_interventions={"unembed.hook_in": payload}), "blocks.10.hook_out"
    )
    assert moved.store_concept_cache_key == "blocks.10.hook_out"
    assert moved.direct_projection_interventions == {"blocks.10.hook_out": payload}
    bare = steering_site_config(_Cfg(), "blocks.10.hook_out")
    assert bare.direct_projection_intervention_hook_pattern == "blocks.10.hook_out"
    with pytest.raises(ValueError, match="configure one"):
        steering_site_config(_Cfg(direct_projection_interventions={"a": payload, "b": payload}), "blocks.10.hook_out")


def test_reference_norms_use_each_stage_raw_norm_or_the_mean_for_stages_without_one() -> None:
    from exp.analysis.concept_direction_latent_dynamics import reference_norms

    records = {"a": {"raw_norm": 100.0}, "b": {"raw_norm": 300.0}, "embed": {"raw_norm": None}}
    assert reference_norms(records) == {"a": 100.0, "b": 300.0, "embed": 200.0}
    with pytest.raises(ValueError, match="at least one stage with a raw norm"):
        reference_norms({"embed": {}})


def test_stage_graphs_run_the_concept_pipeline_once_per_stage_in_the_held_session(
    monkeypatch, inputs: StageInputs
) -> None:
    import exp.concept_direction as concept_direction
    import exp.pipeline_patterns as pipeline_patterns
    from exp.analysis.concept_direction_latent_dynamics import STAGE_GRAPH_FIELDS, stage_graphs

    records = compute_stage_directions(None, STORE_STAGES, inputs)
    module = SimpleNamespace(model=torch.nn.Linear(2, 2))
    tokenizer = SimpleNamespace(encode=lambda token, add_special_tokens=False: [0, len(token)])
    built: list[tuple[torch.Tensor, list[int], list[int]]] = []
    monkeypatch.setattr(
        concept_direction,
        "_build_concept_graph_input_builder",
        lambda cfg, direction, a, b: built.append((direction, a, b)) or direction,
    )

    def _fake_pipeline(cfg, mod, tok, label, *, scale_factor, top_n, build_graph_analysis_inputs):
        assert mod is module and scale_factor == 20.0 and top_n == 5
        return {"feature_ids": [(1, 2, 3)], "pre_gap": 0.0, "post_gap": 1.0, "gap_delta": 1.0, "pre_logits": "dropped"}

    monkeypatch.setattr(pipeline_patterns, "pipeline_on_module", _fake_pipeline)
    graphs = stage_graphs(
        SimpleNamespace(),
        module,
        tokenizer,
        records,
        group_a_tokens=["ab"],
        group_b_tokens=["abc"],
        scale_factor=20.0,
        top_n=5,
    )
    assert list(graphs) == list(records)
    assert all(set(graph) == set(STAGE_GRAPH_FIELDS) for graph in graphs.values())
    for (direction, a, b), record in zip(built, records.values(), strict=True):
        torch.testing.assert_close(direction, record["direction"])
        assert (a, b) == ([2], [3])


def test_store_rows_select_the_prompts_the_extraction_keeps() -> None:
    margins = torch.tensor([3.0, -1.0, 2.0, -0.5])
    answers = ["Fruit", "Color", "Fruit", "Color"]
    kept_margins, kept_answers = store_rows(margins, answers, correct_only=True)
    assert kept_margins.tolist() == [3.0, 2.0] and kept_answers == ["Fruit", "Fruit"]
    all_margins, all_answers = store_rows(margins, answers, correct_only=False)
    assert all_margins.tolist() == margins.tolist() and all_answers == answers
    with pytest.raises(ValueError, match="3 expected answers for 4 prompt margins"):
        store_rows(margins, answers[:3], correct_only=False)


def test_kept_by_store_is_the_extractions_selection() -> None:
    assert kept_by_store(-1.0, correct_only=False) and kept_by_store(0.5, correct_only=False)
    assert kept_by_store(0.5, correct_only=True) and not kept_by_store(0.0, correct_only=True)
    assert not kept_by_store(-2.0, correct_only=True)


def test_store_pair_ids_number_pairs_over_every_prompt_before_any_are_dropped() -> None:
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    margins = [2.0, 2.0, 2.0, 2.0, -1.0, 3.0, -1.0, 0.5]  # the second group's first and third prompts answered wrongly
    assert store_pair_ids(groups, margins, correct_only=True).tolist() == [0, 1, 2, 3, 1, 3]
    assert store_pair_ids(groups, margins, correct_only=False).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_a_paired_stage_reads_only_the_complete_pairs_and_a_mean_stage_every_row() -> None:
    states = torch.arange(6, dtype=torch.float32).reshape(6, 1).repeat(1, 3)
    inputs = StageInputs(
        states={"final": states},
        group_ids=torch.tensor([0, 0, 0, 0, 1, 1]),
        weights=torch.ones(6),
        group_names=["A", "A", "A", "A", "B", "B"],
        group_a_tokens=["a"],
        group_b_tokens=["b"],
        target_tokens=("a", "b"),
        concept_label="A - B",
        pair_ids=torch.tensor([0, 1, 2, 3, 1, 3]),
    )
    paired = Stage("p", "store", "paired_rejection", rows="final")
    rows, group_ids, _, names = inputs.stage_rows(paired)
    assert rows[:, 0].tolist() == [1.0, 3.0, 4.0, 5.0] and group_ids.tolist() == [0, 0, 1, 1]
    assert names == ["A", "A", "B", "B"]
    mean = Stage("m", "store", "mean_difference", rows="final")
    assert inputs.stage_rows(mean)[0].shape[0] == 6
    lonely = StageInputs(**{**inputs.__dict__, "pair_ids": torch.tensor([0, 1, 2, 3, 4, 5])})
    with pytest.raises(ValueError, match="no pair kept both of its rows"):
        lonely.stage_rows(paired)


def test_a_snapshot_configs_siblings_are_found_beside_the_snapshot_not_the_blobs(tmp_path) -> None:
    # the Hub cache layout: snapshot files are symlinks into a content-addressed blobs/ directory
    from exp.concept_direction import snapshot_path

    blobs, snapshot = tmp_path / "blobs", tmp_path / "snapshots" / "rev" / "configs"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    (blobs / "0f1e").write_text("EXPERIMENT_NAME: x\n")
    (snapshot / "experiment.yaml").symlink_to(blobs / "0f1e")
    path = snapshot_path(snapshot / "experiment.yaml")
    assert path.parent == snapshot and path.stem == "experiment"
    assert (snapshot / "experiment.yaml").resolve().parent == blobs  # what resolving would have used


def _prompts(n: int) -> list[dict]:
    """Prompt records shaped like the experiment's classification examples, each with a distinct word and positions."""
    return [
        {
            "probe_surface_text": f"word{i}",
            "raw_prompt": f"question word{i} : ",
            "cache_answer_index": 10 + i,
            "context_token_index": 5 + i,
            "context_token_source": "probe_end",
            "prompt_alignment_artifact": {
                "cache_input_tokens": [f"t{j}" for j in range(11 + i)],
                "probe_start_index": 4 + i,
                "probe_end_index": 5 + i,
            },
        }
        for i in range(n)
    ]


def test_kept_prompt_details_follow_the_store_selection_and_count_every_prompt() -> None:
    margins = torch.tensor([3.0, -1.0, 2.0, -0.5, 4.0])
    details = kept_prompt_details(_prompts(5), margins, correct_only=True)
    assert [d["prompt_index"] for d in details] == [0, 2, 4]
    assert [d["word"] for d in details] == ["word0", "word2", "word4"]
    assert details[1]["answer_index"] == 12 and details[1]["context_index"] == 7 and details[1]["probe_span"] == [6, 7]
    assert details[1]["tokens"][details[1]["answer_index"]] == "t12"
    # with every prompt kept, every prompt is listed
    assert [d["prompt_index"] for d in kept_prompt_details(_prompts(5), margins, correct_only=False)] == [0, 1, 2, 3, 4]


def test_kept_prompt_details_refuse_a_margin_count_that_does_not_match_the_prompts() -> None:
    with pytest.raises(ValueError, match="5 prompts for 4 prompt margins"):
        kept_prompt_details(_prompts(5), torch.ones(4), correct_only=True)


def test_example_records_carry_each_rows_prompt_details(inputs: StageInputs) -> None:
    n = int(inputs.group_ids.shape[0])
    details = kept_prompt_details(_prompts(n), torch.ones(n), correct_only=True)
    records = example_records(inputs, prompt_details=details)
    assert [r["word"] for r in records] == [f"word{i}" for i in range(n)]
    assert all(r["example_index"] == i and r["prompt_index"] == i for i, r in enumerate(records))
    with pytest.raises(ValueError, match=f"{n - 1} prompt details for {n} store rows"):
        example_records(inputs, prompt_details=details[:-1])


def test_trajectory_projection_keeps_only_the_named_sources(inputs: StageInputs, monkeypatch) -> None:
    import interpretune.utils.latent_state_projection as projection_mod

    seen = {}

    def fake_project(matrix, **_):
        seen["rows"] = int(matrix.shape[0])
        return SimpleNamespace(method="umap", backend="fake", metadata={}, coordinates=torch.zeros(matrix.shape[0], 2))

    monkeypatch.setattr(projection_mod, "project_embeddings", fake_project)
    n = int(inputs.group_ids.shape[0])
    result = trajectory_projection(inputs, {}, sources=("context", "projected", "final"))
    assert seen["rows"] == 3 * n
    assert {r["state"] for r in result["records"]} == {"context", "projected", "final"}
    with pytest.raises(ValueError, match="unknown trajectory sources"):
        trajectory_projection(inputs, {}, sources=("answer", "nowhere"))


def test_state_separation_reads_a_separated_state_as_separated(inputs: StageInputs) -> None:
    torch.manual_seed(3)
    n, d = int(inputs.group_ids.shape[0]), 16
    offset = torch.zeros(d)
    offset[0] = 50.0
    separated = torch.randn(n, d) + offset * (inputs.group_ids == 0).float().unsqueeze(-1)
    mixed = torch.randn(n, d)
    states = {**inputs.states, "answer": separated, "context": mixed}
    rows = {r["state"]: r for r in state_separation(StageInputs(**{**inputs.__dict__, "states": states}))}
    assert rows["answer"]["separation_ratio"] > 5 and rows["answer"]["leave_one_out_accuracy"] == 1.0
    assert rows["context"]["separation_ratio"] < 1.5
    assert -1.0 <= rows["context"]["centroid_cosine"] <= 1.0


def test_state_separation_refuses_a_group_with_fewer_than_two_examples(inputs: StageInputs) -> None:
    lonely = torch.tensor([0] * (int(inputs.group_ids.shape[0]) - 1) + [1])
    with pytest.raises(ValueError, match="at least two examples in each group"):
        state_separation(StageInputs(**{**inputs.__dict__, "group_ids": lonely}))
