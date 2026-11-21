"""J-space steering ops: build lens-coordinate patch pairs from pre-fitted Jacobian lenses.

Implements the write half of the J-space workspace technique (Gurnee et al. 2026,
https://transformer-circuits.pub/2026/workspace/): a concept token's J-lens vector is
``W_U[c] @ J_l`` (the row of ``W_U J_l``), and a concept PAIR stacked as ``(2, d_model)`` drives
interpretune's ``patch`` op, ``h <- h + V(sigma(c) - c)``, which swaps the pair's
lens coordinates while leaving everything orthogonal untouched. By default the final norm's scale is folded
into ``W_U[c]`` first (``jlens_apply_final_norm``), the readout-faithful basis ``jlens_folded``.

Lens artifacts resolve from `neuronpedia/jacobian-lens` (Apache-2.0, fit by Neuronpedia with the
Anthropic jlens companion code); the checkpoint format is
``{"J": {layer: Tensor[d, d]}, "n_prompts": int, "source_layers": [int], "d_model": int}``.

The unembed and final-norm resolution, the lens-direction construction (``jlens_direction_rows``) and the
fold-choice default (``resolve_jlens_fold_choice``) all come from ``interpretune.analysis.optools`` rather than
from a copy here, so this collection consumes a basis rather than defining one. That seam handles the per-family RMSNorm
conventions (which do not follow the family name: gemma3 takes the `(1 + weight)` offset and gemma3n does
not) and the LayerNorm centering projector, both of which this module previously got wrong.

Only sanctioned interpretune surfaces are imported (see the custom ops composition guide).
"""

from __future__ import annotations

from typing import Any

import torch

from interpretune.analysis.optools import (
    jlens_basis_name,
    jlens_direction_rows,
    require_model_backend,
    resolve_jlens_fold_choice,
    resolve_jlens_layer,
    resolve_unembed_and_norm_scale,
)

# ~85% depth: empirically where a scale-1.0 swap flips the fruit-example answer for the gemma
# lenses (later layers favor naming tasks, per the paper), while staying below the final layers.
DEFAULT_LAYER_PERCENTILE = 0.85


def _pick_layer(analysis_batch: Any, source_layers: list[int]) -> int:
    explicit = analysis_batch.get("jlens_layer")
    if explicit is not None:
        layer = int(explicit)
        if layer not in source_layers:
            raise ValueError(f"jlens_layer={layer} is not a fitted lens layer (fitted: {source_layers})")
        return layer
    percentile = analysis_batch.get("jlens_layer_percentile")
    percentile = DEFAULT_LAYER_PERCENTILE if percentile is None else float(percentile)
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"jlens_layer_percentile must be in [0, 1], got {percentile}")
    return source_layers[round(percentile * (len(source_layers) - 1))]


def _group_lens_vector(token_ids: Any, info: Any, j_matrix: torch.Tensor, apply_norm: bool) -> torch.Tensor:
    """The group's mean J-lens direction, through interpretune's single lens construction.

    ``jlens_direction_rows`` is the one construction in the tree, used by the in-tree probe, inventory and
    ``concept_direction``; the mean of its rows equals the direction of the mean folded row, since the
    composition is linear, so averaging after it changes nothing numerically.
    """
    ids = torch.as_tensor(token_ids, dtype=torch.long).reshape(-1)
    if ids.numel() == 0:
        raise ValueError("jlens_concept_patch_pair received an empty concept token group")
    return jlens_direction_rows(info, ids, j_matrix, apply_norm=apply_norm).mean(dim=0)


def _stage_pole_pair(analysis_batch: Any, module: Any, op: str, kwargs: dict | None = None) -> Any:
    """Build the pole pair once and stage it under ``op``: the maths is identical, only the staged verb differs."""
    lens_path = analysis_batch.get("jlens_lens_path")
    if lens_path is not None:
        # Explicit local checkpoint (offline tests, user overrides): load directly.
        ckpt = torch.load(str(lens_path), map_location="cpu", weights_only=True)
        if "J" not in ckpt:
            raise ValueError(f"not a JacobianLens checkpoint (keys: {sorted(ckpt)})")
        source_layers = sorted(int(k) for k in ckpt["J"])
        layer = _pick_layer(analysis_batch, source_layers)
        j_matrix = ckpt["J"][layer].float()
    else:
        # Hub resolution goes through the shared seam, which probes the repository
        # layout (rather than formatting one path) and verifies the match against
        # the sidecar instead of assuming it.
        j_matrix, layer, _artifact = resolve_jlens_layer(module, analysis_batch, {})

    info = resolve_unembed_and_norm_scale(module)
    if info.w_u.shape[1] != j_matrix.shape[0]:
        raise ValueError(
            f"unembed width {info.w_u.shape[1]} does not match lens d_model {j_matrix.shape[0]} -- "
            "wrong lens for this model?"
        )
    apply_norm = resolve_jlens_fold_choice(analysis_batch, kwargs)

    v_a = _group_lens_vector(analysis_batch.get("concept_group_a_token_ids"), info, j_matrix, apply_norm)
    v_b = _group_lens_vector(analysis_batch.get("concept_group_b_token_ids"), info, j_matrix, apply_norm)

    analysis_batch.intervention_tensor = torch.stack([v_b, v_a])
    analysis_batch.intervention_op = op
    analysis_batch.intervention_hook_pattern = f"blocks.{layer}.hook_resid_post"
    analysis_batch.jlens_source_layer = layer
    analysis_batch.jlens_basis = jlens_basis_name(apply_norm)
    return analysis_batch


def jlens_concept_patch_pair_impl(module: Any, analysis_batch: Any, batch: Any, batch_idx: int, **kwargs: Any) -> Any:
    """Stage a lens-coordinate ``patch`` intervention for a concept token pair.

    Writes onto the analysis batch exactly the fields ``model_fwd_intervention`` consumes --
    ``intervention_tensor`` (the ``(2, d_model)`` pole pair), ``intervention_op`` (``"patch"``)
    and ``intervention_hook_pattern`` (the fitted layer's residual) -- so the composite
    ``jlens_patch_intervention`` needs no glue. The pair's order is irrelevant: a swap is symmetric,
    and the scale factor (resolved by the intervention op, default 1.0) scales both coordinates.
    """
    return _stage_pole_pair(analysis_batch, module, "patch", kwargs)


def jlens_concept_reject_pair_impl(module: Any, analysis_batch: Any, batch: Any, batch_idx: int, **kwargs: Any) -> Any:
    """Stage a ``reject`` intervention over the same pole pair: ablate the pair's span instead of swapping.

    Same pair, same basis naming, same hook; the staged op is ``"reject"`` with the pair as the basis
    (``use_intervention_tensor_as_basis`` is the intervention op's default), so the forward removes the
    activation's component in the pair's span, ``h <- h - V V^+ h`` at the default scale of 1.0. This is the
    J-space path a backend that cannot express a coordinate swap can still run exactly: a swap needs per-token
    targets measured during the forward, while rejection is a fixed projector. The patch op keeps its own
    early refusal (``required_ops: [patch]``); this op declares ``reject`` for the same reason.
    """
    return _stage_pole_pair(analysis_batch, module, "reject", kwargs)


def _last_real_position(batch: Any, seq: int) -> int:
    """The position a ``last_token`` intervention edits: the final index, every backend applies it there.

    The delta is measured at the position the add will land on, so the two must agree. Under left padding
    the final index is every row's last real token. Under right padding it is a pad position, and a delta
    measured there would steer nothing real, so that is refused by name rather than measured elsewhere.
    """
    mask = batch.get("attention_mask") if isinstance(batch, dict) else None
    if mask is not None and int(torch.as_tensor(mask).reshape(-1)[-1].item()) == 0:
        raise ValueError(
            "jlens_concept_patch_delta measures the clean activation at the final position because that is where "
            "a last_token intervention is applied, and this batch's attention_mask ends in a pad: left-pad the "
            "prompt so its last real token is the final position."
        )
    return seq - 1


def jlens_concept_patch_delta_impl(module: Any, analysis_batch: Any, batch: Any, batch_idx: int, **kwargs: Any) -> Any:
    """Stage the J-space coordinate swap as a static ``add``: exact ``patch`` on any backend with capture and add.

    Two passes. The pair op's maths builds ``V`` (the ``(2, d_model)`` pole pair, basis named on the
    output); a clean forward through the module's model backend captures the activation ``h`` at the
    pair's hook; ``c = V^+ h`` at the final real position are the activation's oblique coordinates;
    the staged tensor is ``V^T (alpha * sigma(c) - c)``, the delta the ``patch`` op would apply, so the
    forward that follows adds it at scale 1 and lands exactly where ``patch`` lands. ``sigma`` swaps the
    two coordinates; ``alpha`` (``jlens_patch_scale``, default 1.0) scales the swapped coordinates as
    the paper's optional oversteer does, which is why the scale lives here rather than on the add.

    Exists for a backend that cannot express ``patch`` directly (a swap needs the activation's own
    coordinates, measured during the forward, while its steering parameters are static) but does
    declare ``add`` and capture. Exact for one position: the delta is a function of the captured
    activation at that position, so this op stages ``last_token`` scope and refuses a batch of more
    than one row by name rather than averaging deltas that differ per row.
    """
    _stage_pole_pair(analysis_batch, module, "add", kwargs)
    pair = torch.as_tensor(analysis_batch.intervention_tensor).detach()
    hook = str(analysis_batch.intervention_hook_pattern)
    backend = require_model_backend(module)
    _, cache = backend.fwd_w_cache(model=module.model, batch=batch, names_filter=[hook])
    if hook in cache:
        captured = cache[hook]
    elif len(cache) == 1:
        captured = next(iter(cache.values()))
    else:
        raise ValueError(
            f"jlens_concept_patch_delta captured {sorted(str(k) for k in cache)} for {hook!r}: expected exactly the "
            "pair's hook. Refusing to guess which capture is the site."
        )
    h = torch.as_tensor(captured).detach()
    if h.ndim != 3:
        raise ValueError(f"captured activation at {hook!r} has shape {tuple(h.shape)}; expected (batch, seq, d_model)")
    if h.shape[0] != 1:
        raise ValueError(
            f"jlens_concept_patch_delta stages one static delta and the batch has {h.shape[0]} rows whose deltas "
            "differ: run it one prompt at a time, or apply the `patch` op on a backend that declares it."
        )
    last = _last_real_position(batch, int(h.shape[1]))
    h_last = h[0, last].to(dtype=torch.float64, device="cpu")
    v_t = pair.to(dtype=torch.float64, device="cpu").T  # (d_model, 2): the columns V of the paper
    coords = torch.linalg.pinv(v_t) @ h_last  # c = V^+ h, the oblique coordinates (pinv, not transpose)
    raw_alpha = analysis_batch.get("jlens_patch_scale")
    alpha = 1.0 if raw_alpha is None else float(raw_alpha)
    swapped = coords.flip(-1) * alpha
    delta = v_t @ (swapped - coords)
    analysis_batch.intervention_tensor = delta.to(dtype=pair.dtype)
    analysis_batch.intervention_op = "add"
    analysis_batch.intervention_position_scope = "last_token"
    analysis_batch.jlens_clean_coords = coords.to(dtype=torch.float32)
    analysis_batch.jlens_swapped_coords = swapped.to(dtype=torch.float32)
    return analysis_batch
