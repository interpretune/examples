"""Offline tests for the jlens_steering_ops collection.

A synthetic lens checkpoint plus a stub HF-shaped model keep everything CPU-only and network-free
(`jlens_lens_path` exists precisely for this). The dispatcher-loading test exercises the collection
through `IT_ANALYSIS_OP_PATHS` -- the path a user actually takes -- rather than importing the impl
module directly, which would bypass schema validation entirely (the lesson interpretune#320 paid
for: eight green direct-call tests on an op that was unreachable through the op surface).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

COLLECTION_DIR = Path(__file__).resolve().parents[1]

D_MODEL, VOCAB = 8, 16
SOURCE_LAYERS = [0, 2, 4]


class _Cfg:
    model_type = "gemma3"
    name_or_path = "google/fake-gemma"


class _LMHead:
    def __init__(self):
        torch.manual_seed(0)
        self.weight = torch.randn(VOCAB, D_MODEL)


class _Norm:
    def __init__(self):
        self.weight = torch.zeros(D_MODEL)  # gemma scale (1 + 0) == identity: keeps expectations exact

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """The behaviour the double stands in for: gemma RMSNorm applies ``x / rms(x) * (1 + weight)``.

        A weight-only double was sufficient while the seam read the scale from a per-family table. The
        module probe evaluates the norm, so the double models the behaviour (the lesson #559 paid for
        in-tree). Deliberately not an ``nn.Module``: the seam's main consumer passes an ``Envoy``.
        """
        rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny).sqrt()
        return x / rms * (1 + self.weight)


class _Inner:
    def __init__(self):
        self.norm = _Norm()


class _Model:
    def __init__(self):
        self.config = _Cfg()
        self.lm_head = _LMHead()
        self.model = _Inner()


class _StubModelBackend:
    """Declares the intervention surface the pair ops stage for, mirroring interpretune's own test doubles.

    The pair op declares ``required_ops: [patch]``, and the dispatcher refuses by name
    on a module whose backend declares no intervention surface. A stub that never executes still has
    to declare what the op it stages requires; LAST_TOKEN is the honest minimal scope claim for one.
    """

    def __init__(self, ops: tuple[str, ...] = ("patch",)):
        from interpretune.analysis.backends import Capability, InterventionSupport, PositionScope

        self.capabilities = frozenset({Capability.INTERVENTION})
        self.intervention_support = InterventionSupport(
            position_scopes=frozenset({PositionScope.LAST_TOKEN}),
            ops=frozenset(ops),
        )


class _Module:
    def __init__(self, ops: tuple[str, ...] = ("patch",)):
        self.model = _Model()
        self._model_backend = _StubModelBackend(ops)


SEQ = 5


class _CapturingBackend(_StubModelBackend):
    """A backend that declares ``add`` and answers ``fwd_w_cache`` with a fixed clean activation.

    The delta op's whole point is a backend that cannot ``patch`` but can capture and add, so the
    double records what it was asked to capture and returns one ``(batch, seq, d_model)`` tensor
    under exactly that name; the test then owns ``h`` and can check the staged delta against the
    in-tree ``patch`` arithmetic on the same activation.
    """

    def __init__(self, activation: torch.Tensor, ops: tuple[str, ...] = ("add",)):
        super().__init__(ops)
        self.activation = activation
        self.captured_filters: list[list[str]] = []

    def fwd_w_cache(self, model, batch, names_filter):
        self.captured_filters.append(list(names_filter))
        return torch.zeros(self.activation.shape[0], self.activation.shape[1], VOCAB), {
            name: self.activation for name in names_filter
        }


class _CapturingModule(_Module):
    def __init__(self, activation: torch.Tensor, ops: tuple[str, ...] = ("add",)):
        super().__init__(ops)
        self._model_backend = _CapturingBackend(activation, ops)


@pytest.fixture(scope="module")
def lens_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("lens") / "fake_lens.pt"
    torch.manual_seed(1)
    torch.save(
        {
            "J": {layer: torch.randn(D_MODEL, D_MODEL) for layer in SOURCE_LAYERS},
            "n_prompts": 3,
            "source_layers": SOURCE_LAYERS,
            "d_model": D_MODEL,
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def dispatched_op(lens_path):
    """Load the collection through IT_ANALYSIS_OP_PATHS and hand back the dispatched op."""
    os.environ["IT_ANALYSIS_OP_PATHS"] = str(COLLECTION_DIR)
    for name in [m for m in sys.modules if m == "interpretune" or m.startswith("interpretune.")]:
        del sys.modules[name]
    import interpretune as it
    import interpretune.analysis  # noqa: F401  (registers top-level op wrappers)

    yield it
    os.environ.pop("IT_ANALYSIS_OP_PATHS", None)


def _batch(it, lens_path, **overrides):
    fields = dict(
        concept_group_a_token_ids=[3],
        concept_group_b_token_ids=[7],
        jlens_lens_path=str(lens_path),
    )
    fields.update(overrides)
    return it.AnalysisBatch(**fields)


class TestThroughTheDispatcher:
    def test_collection_registers_op_and_composite(self, dispatched_op):
        it = dispatched_op
        assert hasattr(it, "jlens_concept_patch_pair")
        assert hasattr(it, "jlens_patch_intervention")
        assert hasattr(it, "jlens_concept_reject_pair")
        assert hasattr(it, "jlens_reject_intervention")
        assert hasattr(it, "jlens_concept_patch_delta")
        assert hasattr(it, "jlens_patch_via_add_intervention")

    def test_reject_pair_is_the_same_pair_staged_under_reject(self, dispatched_op, lens_path):
        """Same maths, different staged verb: the reject op is the patch op's twin for a backend that cannot swap."""
        it = dispatched_op
        patched = it.jlens_concept_patch_pair(_Module(("patch",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        rejected = it.jlens_concept_reject_pair(_Module(("reject",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        torch.testing.assert_close(rejected.intervention_tensor, patched.intervention_tensor)
        assert rejected.intervention_op == "reject"
        assert rejected.intervention_hook_pattern == patched.intervention_hook_pattern == "blocks.2.hook_resid_post"
        assert rejected.jlens_basis == patched.jlens_basis

    def test_each_pair_op_is_refused_early_on_a_backend_lacking_its_mode(self, dispatched_op, lens_path):
        """The declarations are the point: a backend that declares reject but not patch is refused by the patch op
        before staging, and the mirror holds, so a consumer learns compatibility without executing anything."""
        it = dispatched_op
        with pytest.raises(ValueError, match="requires intervention ops \\['patch'\\]"):
            it.jlens_concept_patch_pair(_Module(("reject",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        with pytest.raises(ValueError, match="requires intervention ops \\['reject'\\]"):
            it.jlens_concept_reject_pair(_Module(("patch",)), _batch(it, lens_path, jlens_layer=2), None, 0)

    def test_pair_is_stacked_lens_rows_and_fields_are_staged(self, dispatched_op, lens_path):
        it = dispatched_op
        module = _Module()
        out = it.jlens_concept_patch_pair(module, _batch(it, lens_path, jlens_layer=2), None, 0)
        pair = out.intervention_tensor
        assert tuple(pair.shape) == (2, D_MODEL)
        ckpt = torch.load(lens_path, weights_only=True)
        w_u = module.model.lm_head.weight
        expected_a = w_u[3].float() @ ckpt["J"][2].float()
        expected_b = w_u[7].float() @ ckpt["J"][2].float()
        torch.testing.assert_close(pair[1], expected_a)
        torch.testing.assert_close(pair[0], expected_b)
        assert out.intervention_op == "patch"
        assert out.intervention_hook_pattern == "blocks.2.hook_resid_post"
        assert int(out.jlens_source_layer) == 2

    def test_percentile_picks_from_fitted_layers_only(self, dispatched_op, lens_path):
        it = dispatched_op
        out = it.jlens_concept_patch_pair(_Module(), _batch(it, lens_path, jlens_layer_percentile=1.0), None, 0)
        assert int(out.jlens_source_layer) == SOURCE_LAYERS[-1]

    def test_unfitted_explicit_layer_is_rejected_naming_the_fitted_set(self, dispatched_op, lens_path):
        it = dispatched_op
        with pytest.raises(ValueError, match=r"not a fitted lens layer \(fitted: \[0, 2, 4\]\)"):
            it.jlens_concept_patch_pair(_Module(), _batch(it, lens_path, jlens_layer=3), None, 0)

    def test_wrong_width_lens_names_both_widths(self, dispatched_op, tmp_path_factory):
        it = dispatched_op
        bad = tmp_path_factory.mktemp("bad") / "bad_lens.pt"
        torch.save({"J": {0: torch.randn(4, 4)}, "n_prompts": 1, "source_layers": [0], "d_model": 4}, bad)
        with pytest.raises(ValueError, match="unembed width 8 does not match lens d_model 4"):
            it.jlens_concept_patch_pair(_Module(), _batch(it, bad), None, 0)

    def test_final_norm_scale_is_folded_and_optable_out(self, dispatched_op, lens_path):
        """The gamma finding, pinned: folding is the default and (1 + weight) is the gemma form."""
        it = dispatched_op
        module = _Module()
        module.model.model.norm.weight = torch.full((D_MODEL,), 1.0)  # gemma scale becomes 2.0
        folded = it.jlens_concept_patch_pair(module, _batch(it, lens_path, jlens_layer=0), None, 0)
        unfolded = it.jlens_concept_patch_pair(
            module, _batch(it, lens_path, jlens_layer=0, jlens_apply_final_norm=False), None, 0
        )
        torch.testing.assert_close(folded.intervention_tensor, unfolded.intervention_tensor * 2.0)

    def test_transformerlens_shaped_model_resolves_transposed(self, dispatched_op, lens_path):
        it = dispatched_op

        class _TLModel:
            def __init__(self):
                torch.manual_seed(0)
                self.W_U = torch.randn(D_MODEL, VOCAB)  # TL orientation: (d, vocab)

        class _TLModule:
            def __init__(self):
                self.model = _TLModel()
                self._model_backend = _StubModelBackend()

        out = it.jlens_concept_patch_pair(_TLModule(), _batch(it, lens_path, jlens_layer=0), None, 0)
        ckpt = torch.load(lens_path, weights_only=True)
        expected = _TLModule().model.W_U.transpose(0, 1)[3].float() @ ckpt["J"][0].float()
        torch.testing.assert_close(out.intervention_tensor[1], expected)

    def test_pair_records_the_resolved_basis_name(self, dispatched_op, lens_path):
        """interpretune#540: a result that cannot name its basis is not comparable after the fact.

        The recorded value is the resolved name, not the raw flag: two runs differing only in the
        fold choice must be distinguishable from the stored artifact alone.
        """
        it = dispatched_op
        folded = it.jlens_concept_patch_pair(_Module(), _batch(it, lens_path, jlens_layer=0), None, 0)
        unfolded = it.jlens_concept_patch_pair(
            _Module(), _batch(it, lens_path, jlens_layer=0, jlens_apply_final_norm=False), None, 0
        )
        assert folded.jlens_basis == "jlens_folded"
        assert unfolded.jlens_basis == "jlens_unfolded"
        assert folded.jlens_basis != unfolded.jlens_basis

    def test_a_non_bool_fold_flag_is_refused_rather_than_coerced(self, dispatched_op, lens_path):
        """The fold choice resolves through interpretune's single resolver, so this collection inherits its refusal.

        ``bool("False")`` is True: coercing a string would silently build the basis the caller tried to turn off.
        """
        it = dispatched_op
        with pytest.raises(TypeError, match="jlens_apply_final_norm must be a bool"):
            it.jlens_concept_patch_pair(
                _Module(), _batch(it, lens_path, jlens_layer=0, jlens_apply_final_norm="False"), None, 0
            )


def _clean_activation(seed: int = 3) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, SEQ, D_MODEL)


class TestPatchDeltaOverAnAddOnlyBackend:
    def test_staged_add_lands_exactly_where_patch_lands(self, dispatched_op, lens_path):
        """The delta is the update the ``patch`` op would apply: ``h + delta`` equals the in-tree lens
        coordinate patch of ``h`` on the same pair, at the position the delta was computed for."""
        from interpretune.analysis.backends.interventions import InterventionSpec, _apply_lens_coordinate_patch

        it = dispatched_op
        h = _clean_activation()
        module = _CapturingModule(h)
        out = it.jlens_concept_patch_delta(
            module, _batch(it, lens_path, jlens_layer=2), {"input_ids": torch.ones(1, SEQ)}, 0
        )
        pair = it.jlens_concept_patch_pair(_Module(("patch",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        spec = InterventionSpec(intervention_tensor=pair.intervention_tensor, op="patch", scale_factor=1.0)
        patched = _apply_lens_coordinate_patch(spec, input_value=h[:, -1], target=pair.intervention_tensor)
        torch.testing.assert_close(h[0, -1] + out.intervention_tensor, patched[0], atol=1e-5, rtol=1e-5)
        assert out.intervention_op == "add"
        assert out.intervention_position_scope == "last_token"
        assert out.intervention_hook_pattern == "blocks.2.hook_resid_post"
        assert module._model_backend.captured_filters == [["blocks.2.hook_resid_post"]]
        # The swap is a swap: the reported coordinates are each other's exchange, and the patched activation's
        # own oblique coordinates equal the swapped ones.
        torch.testing.assert_close(out.jlens_swapped_coords, out.jlens_clean_coords.flip(-1))
        v_t = pair.intervention_tensor.T.double()
        torch.testing.assert_close(
            (torch.linalg.pinv(v_t) @ (h[0, -1] + out.intervention_tensor).double()).float(),
            out.jlens_swapped_coords,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_the_pair_is_correlated_so_pinv_and_transpose_disagree(self, dispatched_op, lens_path):
        """A positive control on the equality above: with a non-orthonormal pair, transpose coordinates give a
        different delta, so the test can tell the pseudoinverse from the transpose."""
        it = dispatched_op
        h = _clean_activation()
        out = it.jlens_concept_patch_delta(_CapturingModule(h), _batch(it, lens_path, jlens_layer=2), {}, 0)
        pair = it.jlens_concept_patch_pair(_Module(("patch",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        v_t = pair.intervention_tensor.T.double()
        gram = v_t.T @ v_t
        assert abs(gram[0, 1]) > 1e-3 * (gram[0, 0] * gram[1, 1]).sqrt(), "pair happens to be orthogonal"
        transpose_delta = (v_t @ ((v_t.T @ h[0, -1].double()).flip(-1) - v_t.T @ h[0, -1].double())).float()
        assert not torch.allclose(transpose_delta, out.intervention_tensor)

    def test_scale_multiplies_the_swapped_coordinates_like_patch_alpha(self, dispatched_op, lens_path):
        from interpretune.analysis.backends.interventions import InterventionSpec, _apply_lens_coordinate_patch

        it = dispatched_op
        h = _clean_activation()
        out = it.jlens_concept_patch_delta(
            _CapturingModule(h), _batch(it, lens_path, jlens_layer=2, jlens_patch_scale=2.0), {}, 0
        )
        pair = it.jlens_concept_patch_pair(_Module(("patch",)), _batch(it, lens_path, jlens_layer=2), None, 0)
        spec = InterventionSpec(intervention_tensor=pair.intervention_tensor, op="patch", scale_factor=2.0)
        patched = _apply_lens_coordinate_patch(spec, input_value=h[:, -1], target=pair.intervention_tensor)
        torch.testing.assert_close(h[0, -1] + out.intervention_tensor, patched[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out.jlens_swapped_coords, out.jlens_clean_coords.flip(-1) * 2.0)

    def test_measures_the_position_the_add_lands_on_and_refuses_right_padding(self, dispatched_op, lens_path):
        """A ``last_token`` add edits the final index on every backend, so the delta is measured there: a
        left-padded mask changes nothing, and a right-padded one (final entry a pad) is refused by name."""
        it = dispatched_op
        h = _clean_activation()
        left = it.jlens_concept_patch_delta(
            _CapturingModule(h),
            _batch(it, lens_path, jlens_layer=2),
            {"attention_mask": torch.tensor([[0, 0, 1, 1, 1]])},
            0,
        )
        unmasked = it.jlens_concept_patch_delta(_CapturingModule(h), _batch(it, lens_path, jlens_layer=2), {}, 0)
        torch.testing.assert_close(left.intervention_tensor, unmasked.intervention_tensor)
        with pytest.raises(ValueError, match="ends in a pad"):
            it.jlens_concept_patch_delta(
                _CapturingModule(h),
                _batch(it, lens_path, jlens_layer=2),
                {"attention_mask": torch.tensor([[1, 1, 1, 0, 0]])},
                0,
            )

    def test_refused_early_on_a_backend_without_add(self, dispatched_op, lens_path):
        it = dispatched_op
        with pytest.raises(ValueError, match="requires intervention ops \\['add'\\]"):
            it.jlens_concept_patch_delta(
                _CapturingModule(_clean_activation(), ("patch",)), _batch(it, lens_path, jlens_layer=2), {}, 0
            )

    def test_more_than_one_row_is_refused_by_name(self, dispatched_op, lens_path):
        it = dispatched_op
        torch.manual_seed(4)
        with pytest.raises(ValueError, match="batch has 2 rows"):
            it.jlens_concept_patch_delta(
                _CapturingModule(torch.randn(2, SEQ, D_MODEL)), _batch(it, lens_path, jlens_layer=2), {}, 0
            )

    def test_a_backend_without_capture_is_refused_before_any_maths(self, dispatched_op, lens_path):
        it = dispatched_op
        with pytest.raises(AttributeError, match="fwd_w_cache"):
            it.jlens_concept_patch_delta(_Module(("add",)), _batch(it, lens_path, jlens_layer=2), {}, 0)

    def test_sparse_checkpoint_without_source_layers_or_d_model(self, dispatched_op, tmp_path_factory):
        """Layout variance: a checkpoint carrying only `J` still stages (layers and width inferred)."""
        sparse = tmp_path_factory.mktemp("sparse") / "sparse_lens.pt"
        torch.manual_seed(7)
        torch.save({"J": {0: torch.randn(D_MODEL, D_MODEL), 2: torch.randn(D_MODEL, D_MODEL)}}, sparse)
        it = dispatched_op
        out = it.jlens_concept_patch_pair(_Module(), _batch(it, sparse, jlens_layer=2), None, 0)
        assert int(out.jlens_source_layer) == 2
