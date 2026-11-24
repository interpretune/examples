"""CPU tests for the neuron-communities notebook's helpers, on a small synthetic community and a random tiny Llama."""

from __future__ import annotations

import json
import random

import neuron_communities_helpers as nch
import pytest
import torch

N_LAYERS, WIDTH = 4, 12
RECORD = {
    "name": "synthetic",
    "neurons": [{"layer": 1, "idx": 3}, {"layer": 1, "idx": 7}, {"layer": 2, "idx": 0}, {"layer": 3, "idx": 1}],
    "always_on": [{"layer": 2, "idx": 5}],
    "main_span": {"layers": [1, 2]},
}


@pytest.fixture
def community(tmp_path) -> nch.Community:
    (tmp_path / "community_7.json").write_text(json.dumps(RECORD))
    return nch.load_community("7", tmp_path)  # a record already on disk is read, never fetched


@pytest.fixture(scope="module")
def tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=WIDTH,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(cfg).eval().requires_grad_(False)


def test_a_record_parses_into_members_hubs_and_span(community):
    assert community.name == "synthetic" and community.span == [1, 2]
    assert community.members == [(1, 3), (1, 7), (2, 0), (3, 1)] and community.hubs == [(2, 5)]
    assert len(community.sha256) == 64


def test_the_books_complements_leave_the_community_and_hubs_standing(community):
    from interpretune.analysis.venngram import unit_interventions

    book = nch.community_book(community, WIDTH)
    assert book["community"]["units"] == {1: [3, 7], 2: [0]}  # the member outside the span is not addressed
    payloads = nch.zero_payloads(book, N_LAYERS, WIDTH)
    assert set(payloads) == set(nch.TESTS)
    by_hand = {
        "s": {
            "point": nch.POINT,
            "layers": [1, 2],
            "units": {1: [u for u in range(WIDTH) if u not in (3, 7)], 2: [u for u in range(WIDTH) if u not in (0, 5)]},
        }
    }
    expected = unit_interventions(
        "@s",
        sets=by_hand,
        n_layers=N_LAYERS,
        site_widths={nch.POINT: WIDTH},
        op="replace",
        value=0.0,
        position_scope="last_token",
    )
    assert repr(payloads["sufficiency"]) == repr(expected)
    assert repr(payloads["layer necessity"]) != repr(expected)


def test_the_random_control_never_draws_a_member_or_a_hub(community, monkeypatch):
    books: list[dict] = []
    monkeypatch.setattr(nch, "zero_ablation", lambda address, book, *_: books.append(book))
    for seed in range(50):
        nch.random_payload(community, N_LAYERS, WIDTH, random.Random(seed))
    drawn = {(layer, unit) for book in books for layer, units in book["r"]["units"].items() for unit in units}
    assert {layer for layer, _ in drawn} == {1, 2}
    assert not drawn & (set(community.members) | set(community.hubs))


def test_kl_and_top1_change_read_the_last_position():
    clean = torch.zeros(1, 2, 3)
    edited = clean.clone()
    edited[0, 0] = torch.tensor([9.0, 0.0, 0.0])  # an earlier position must not count
    assert nch.kl_and_top1_change(clean, edited) == ([pytest.approx(0.0, abs=1e-6)], [pytest.approx(0.0, abs=1e-6)])
    edited[0, 1] = torch.tensor([2.0, 0.0, 0.0])
    (kl,), (change,) = nch.kl_and_top1_change(clean, edited)
    p = torch.softmax(edited[0, 1], -1)
    assert kl == pytest.approx(float((torch.tensor([1 / 3] * 3) * (torch.log(torch.tensor(1 / 3)) - p.log())).sum()))
    assert change == pytest.approx(float(abs(1 / 3 - p[0])), abs=1e-6)


def test_the_hand_written_ablation_zeroes_only_the_named_units_at_the_last_position(tiny_llama):
    ids = torch.tensor([[1, 4, 9, 2]])
    clean = tiny_llama(ids).logits.detach()
    edited = nch.hand_written_zero_ablation(tiny_llama, ids, {1: list(range(WIDTH))})
    torch.testing.assert_close(edited[:, :-1], clean[:, :-1])
    assert not torch.allclose(edited[:, -1], clean[:, -1])
    torch.testing.assert_close(nch.hand_written_zero_ablation(tiny_llama, ids, {}), clean)  # hooks are removed after


def test_the_hand_written_gradient_matches_a_finite_difference(tiny_llama):
    # a step where truncation and rounding are both small (RMSNorm computes in float32 whatever the model's dtype)
    ids, layer, unit, eps = torch.tensor([[1, 4, 9, 2]]), 3, 5, 1e-2
    grads = nch.hand_written_gradients(tiny_llama, ids, N_LAYERS)
    assert grads.shape == (1, N_LAYERS, WIDTH)

    def read(step: float) -> float:
        def nudge(_module, inputs):
            x = inputs[0].clone()
            x[:, -1, unit] += step
            return (x, *inputs[1:])

        handle = tiny_llama.model.layers[layer].mlp.down_proj.register_forward_pre_hook(nudge)
        try:
            with torch.no_grad():
                return float(tiny_llama(ids).logits[:, -1].log_softmax(-1).max(-1).values.sum())
        finally:
            handle.remove()

    slope = (read(eps) - read(-eps)) / (2 * eps)
    assert float(grads[0, layer, unit]) == pytest.approx(slope, rel=2e-3)
    others = torch.cat([grads[0, layer, :unit], grads[0, layer, unit + 1 :]])
    assert float((others - slope).abs().min()) > 0.05 * abs(slope)  # no other unit's gradient would have passed


def test_attribution_controls_choose_the_hard_baseline_outside_the_community(community):
    top_sets = [{(1, 3), (1, 7), (1, 0)}, {(1, 3), (2, 0), (1, 0), (2, 9)}, {(2, 5), (1, 0)}]
    assert nch.present_fraction(top_sets, set(community.members)) == pytest.approx(2 / 3)
    controls = nch.attribution_controls(top_sets, community)
    assert controls["present"] == pytest.approx(2 / 3) and controls["mean_members"] == pytest.approx(4 / 3)
    # layer 1's most frequent non-members in the members' counts, and layer 2's; layer 3 has no candidate
    assert controls["hard_mean"] == pytest.approx(4 / 3) and controls["hub_rates"] == [pytest.approx(1 / 3)]
    assert controls["distinct"] == 6
