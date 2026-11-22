---
tags:
- interpretune
- analysis-ops
- jlens
license: apache-2.0
---

# jlens_steering_ops

A genuinely non-bundled [interpretune](https://github.com/speediedan/interpretune) analysis-op
collection: J-space (Jacobian-lens) steering built on pre-fitted lenses from
[`neuronpedia/jacobian-lens`](https://huggingface.co/neuronpedia/jacobian-lens) (fit by Neuronpedia
with the Anthropic [jlens](https://github.com/anthropics/jlens) companion code, Apache-2.0;
technique: Gurnee et al. 2026, [the workspace paper](https://transformer-circuits.pub/2026/workspace/)).

## Ops

| op | what it does |
| --- | --- |
| `jlens_concept_patch_pair` | builds the `(2, d_model)` J-lens pole pair for a concept token pair (`v_c = (W_U[c] * final_norm_scale) @ J_l`) and stages a lens-coordinate `patch` intervention (tensor, op, hook pattern) on the analysis batch |
| `jlens_patch_intervention` | composite: the op above, then interpretune's bundled `model_fwd_intervention`, which applies `h <- h + V(sigma(c) - c)` -- a coordinate swap that leaves everything orthogonal to the pair untouched |
| `jlens_concept_reject_pair` | builds the same pole pair and stages a `reject` intervention over it (the pair as the basis): the J-space path for a backend that cannot express a coordinate swap, since rejection is a fixed projector |
| `jlens_reject_intervention` | composite: the reject op, then `model_fwd_intervention`, which removes the activation's component in the pair's span, `h <- h - V V^+ h`, at scale 1.0 by default |
| `jlens_concept_patch_delta` | builds the same pole pair, captures the clean activation at the pair's hook through the module's model backend, and stages the coordinate swap as a static `add`: the delta `V(alpha*sigma(c) - c)` with `c = V^+ h` at the final real position. Exact `patch` on any backend that declares `add` and capture; for a backend that cannot express `patch` itself. One prompt at a time |
| `jlens_patch_via_add_intervention` | composite: the delta op, then `model_fwd_intervention` adding that delta at scale 1, which lands exactly where the `patch` op lands |

## Usage

```python
import interpretune as it
import interpretune.analysis

it.hub.pull_ops("speediedan/jlens_steering_ops")

out = it.jlens_patch_intervention(
    module,
    it.AnalysisBatch(
        prompts=[prompt],
        concept_group_a_token_ids=[fruit_id],
        concept_group_b_token_ids=[color_id],
        jlens_layer=24,                  # a fitted lens layer; later layers suit naming tasks
        direction_scale_factor=1.0,      # a pure swap -- no oversteering needed
    ),
    batch,
    0,
)
```

Measured in interpretune's concept-steering demos (interpretune `fd4871c3`, this collection at `38fa5c41fc`;
the current revision builds the same pole pair, since the lens composition is linear and averaging before or
after it agrees up to floating-point order): the swap flips the answer (Color -> Fruit) at
scale **1.0**. On gemma-2-2b at layer 24 the gap moves from −1.875 to +2.625, where the naive `add`-op
direct-hook path at scale 20 reaches only +0.125. On gemma-3-1b-it (chat-formatted prompt) at layer 21 it moves
from −7.75 to +2.50, where the `add` path at scale 20 stays at −2.50. Each scale is its own op's multiplier,
not a matched displacement, so this compares what each knob does, not which edit is larger.

`jlens_apply_final_norm` (default on) folds the final norm's elementwise scale into the lens vectors, which is
the direction the paper's own readout formula computes. The default and its validation come from interpretune's
`resolve_jlens_fold_choice`, so a value that is not a bool is refused by name rather than coerced. **Whether
folding helps steering is model-dependent, and measured**: on gemma-3-1b-it it is essential (unfolded vectors
never flip at any layer), while on gemma-2-2b both forms flip and the unfolded form steers ~3x harder at late
layers. Folding aims the swap plane into the coordinates the final norm amplifies, so it helps exactly where the
model stores the task contrast. The cheap first-order check to run on your own model, and the measurements
behind these claims, are in interpretune's
[folding documentation](https://interpretune.org/en/latest/usage/jlens_norm_folding.html).

This revision builds its lens directions through `interpretune.analysis.optools.jlens_direction_rows` and
resolves the fold choice through `resolve_jlens_fold_choice`, so it needs an interpretune that exports both;
with an older interpretune the import fails by name.

Tests live in `tests/` and run offline against a synthetic lens (`jlens_lens_path` override).
