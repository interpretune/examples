"""The steering demos' shared maths, checked on synthetic tensors so no model or kernel is needed."""

from __future__ import annotations

import pytest
import torch

from interpretune.analysis.optools import UnembedNormInfo
import steering_demo_helpers as helpers


class _Tokenizer:
    """Maps each word to its character codes modulo a tiny vocabulary: multi-token words and empty groups."""

    def __call__(self, word: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) % 8 for c in word]}

    def decode(self, ids: list[int]) -> str:
        return f"t{ids[0]}"


class _Backend:
    def __init__(self, embed: torch.Tensor):
        self.embed = embed

    def get_embedding_weight(self, module):
        return self.embed


def test_embed_pole_pair_means_every_token_of_every_word(monkeypatch):
    embed = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    monkeypatch.setattr(helpers, "require_analysis_backend", lambda module: _Backend(embed))
    poles = helpers.embed_pole_pair(object(), _Tokenizer(), ["ab"], ["c", "d"])
    ids_a = [ord("a") % 8, ord("b") % 8]
    ids_b = [ord("c") % 8, ord("d") % 8]
    torch.testing.assert_close(poles[0], embed[ids_a].mean(dim=0))
    torch.testing.assert_close(poles[1], embed[ids_b].mean(dim=0))


def test_embed_pole_pair_refuses_an_empty_group_by_name(monkeypatch):
    monkeypatch.setattr(helpers, "require_analysis_backend", lambda module: _Backend(torch.zeros(8, 3)))
    with pytest.raises(ValueError, match=r"concept group b \(\[\]\) tokenized to no ids"):
        helpers.embed_pole_pair(object(), _Tokenizer(), ["a"], [])


def test_outside_span_share_uses_the_pseudoinverse_projector_for_non_orthonormal_poles():
    # Non-orthogonal poles spanning the x-y plane: a direction in the plane has nothing outside it even though
    # projecting onto each pole separately would miscount the overlap.
    poles = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    assert helpers.outside_span_share(torch.tensor([0.6, 0.8, 0.0]), poles) == pytest.approx(0.0, abs=1e-6)
    assert helpers.outside_span_share(torch.tensor([0.0, 0.6, 0.8]), poles) == pytest.approx(0.8, abs=1e-6)


def test_central_difference_slopes_recover_a_linear_gap(monkeypatch):
    gradient = torch.tensor([2.0, -3.0])

    def _gap(module, prompt, batch, site, tensor, scale, a, b):
        return float(1.0 + scale * (tensor @ gradient))

    monkeypatch.setattr(helpers, "gap_after_add", _gap)
    directions = torch.eye(2)
    slopes = helpers.central_difference_gap_slopes(None, "p", {}, "site", directions, 0, 1, eps=0.5)
    assert slopes == pytest.approx([2.0, -3.0])


def test_readout_top_tokens_matches_the_jlens_read_readout():
    torch.manual_seed(0)
    d, vocab = 4, 6
    scale = torch.rand(d) + 0.5
    info = UnembedNormInfo(w_u=torch.randn(vocab, d), norm_scale=scale, norm_kind="rmsnorm")
    lens = torch.randn(d, d)
    h = torch.randn(d)
    expected = (((h @ lens.T) * scale) @ info.w_u.T).topk(3)
    top = helpers.jlens_readout_top_tokens(h, lens, info, _Tokenizer(), k=3)
    assert [label for label, _ in top] == [f"t{int(i)}" for i in expected.indices]
    assert [score for _, score in top] == pytest.approx(expected.values.tolist(), rel=1e-5)


def test_pole_swap_prediction_is_exact_for_a_linear_readout_and_signed_by_the_clean_lean():
    """The swap exchanges c = V^+ x; under a linear readout the prediction must equal the realized gap change, and
    a clean state leaning to pole a must move the gap AWAY from a."""
    torch.manual_seed(1)
    d = 6
    poles = torch.randn(2, d)
    u_a, u_b = torch.randn(d), torch.randn(d)
    for lean in (poles[0] * 3.0 + poles[1] * 0.5, poles[0] * 0.5 + poles[1] * 3.0):
        x = lean + 0.1 * torch.randn(d)
        pred = helpers.pole_swap_prediction(x, poles, u_a, u_b)
        c = torch.tensor(pred.coordinates)
        swapped = x + (c.flip(0) - c) @ poles
        realized = float((u_a - u_b) @ (swapped - x))
        assert pred.predicted_gap_delta == pytest.approx(realized, rel=1e-6, abs=1e-9)
        sign_of_pole_alignment = float((poles[0] - poles[1]) @ (u_a - u_b)) > 0
        leans_to_a = pred.coordinates[0] > pred.coordinates[1]
        assert (pred.predicted_gap_delta > 0) == (leans_to_a != sign_of_pole_alignment)


def test_steering_scale_point_reads_both_targets_and_normalizes_over_the_whole_vocabulary():
    pre = torch.tensor([1.0, 3.0, 0.0, 0.0])
    post = torch.tensor([4.0, 2.0, 0.0, 0.0])
    point = helpers.steering_scale_point("arm", 5, pre, post, target_a_id=0, target_b_id=1)
    assert point.pre_logits == (1.0, 3.0) and point.post_logits == (4.0, 2.0)
    assert point.pre_gap == -2.0 and point.post_gap == 2.0
    # Probabilities over all four tokens, not a two-way softmax over the targets.
    expected = torch.softmax(pre, dim=-1)
    assert point.pre_probs == pytest.approx((float(expected[0]), float(expected[1])))
    assert sum(point.pre_probs) < 1.0


# Block depths of the families the steering arms are written for: gemma-2-2b and gemma-3-1b-it (the demos'
# substrates), and Llama-3.2-1B and gpt2, two further mapped families the same addresses must serve unchanged.
_FAMILY_DEPTHS = {"gemma-2-2b": 26, "gemma-3-1b-it": 26, "Llama-3.2-1B": 16, "gpt2": 12}


@pytest.mark.parametrize("n_layers", sorted(set(_FAMILY_DEPTHS.values())))
def test_direct_site_address_is_the_unembed_input_on_every_depth(n_layers):
    assert helpers.steering_site(helpers.DIRECT_SITE_ADDRESS, n_layers) == "unembed.hook_in"


@pytest.mark.parametrize("family", sorted(_FAMILY_DEPTHS))
def test_lens_site_address_names_one_residual_site(family):
    n_layers = _FAMILY_DEPTHS[family]
    assert helpers.steering_site(helpers.lens_site_address(n_layers - 2), n_layers) == (
        f"blocks.{n_layers - 2}.hook_resid_post"
    )
    # The last block by a negative index, which the interp-engine arm uses in place of the unembed's input.
    assert helpers.steering_site(helpers.lens_site_address(-1), n_layers) == f"blocks.{n_layers - 1}.hook_resid_post"


def test_steering_site_refuses_an_address_selecting_several_sites():
    with pytest.raises(ValueError, match=r"selects 4 sites .*an arm edits one"):
        helpers.steering_site("point:hook_resid_post & layers:8:12", 12)


def test_steering_site_refuses_an_unknown_point_by_name():
    from interpretune.utils import MisconfigurationException

    with pytest.raises(MisconfigurationException, match=r"not an activation point"):
        helpers.steering_site("point:hook_resid_sideways", 12)


def test_steering_site_refuses_a_layer_past_the_stack():
    from interpretune.utils import MisconfigurationException

    with pytest.raises(MisconfigurationException, match=r"Layer 24 is outside a stack of 16 blocks"):
        helpers.steering_site(helpers.lens_site_address(24), 16)
