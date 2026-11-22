"""J-space experimental battery (interpretune#425), built on interpretune's own J-lens and intervention path.

Everything that touches the lens or writes to the residual stream is interpretune's production code:
``resolve_jlens_layer`` (refuses a layer the lens was not fitted at), ``jlens_direction_rows`` in the
norm-aware basis, and ``InterventionSpec(mode="patch", position_scope=ALL_POSITIONS)`` executed by the
NNsight model backend's ``fwd_w_intervention``. The battery is therefore evidence about that path, not
about a reimplementation of it.

Prompts are the paper's own Apache-2.0 sets from ``anthropics/jacobian-lens`` (``data/probe-swap.json``,
two-hop items with ``intermediate``/``answer``/``swap_to``/``swap_answer``), vendored under ``data/``.

Measurements, per item and per fitted layer ``L``:

- **baseline**: does a greedy continuation start with ``answer``? Only baseline-correct items can test a
  swap, so every swap rate is reported over that subset with its denominator.
- **readout**: the best rank of ``intermediate`` (and of ``answer``) over all prompt positions in the
  lens readout at ``L``. Readable means rank ``< k``.
- **intermediate swap at L**: patch the ``(intermediate, swap_to)`` J-space pair at ``L`` over every prompt
  position; success if ``swap_answer`` becomes top-1 at the final position (the paper's criterion), or enters
  the top-k on items where it was not already there at baseline (see :func:`summarize`).
- **answer swap at L**: the same with ``(answer, swap_answer)``.
- **band swaps**: the same patches over a contiguous band of layers at once (the paper's protocol).
- **control swap**: the intermediate swapped for an UNRELATED entity (another item's ``swap_to``), scored
  on this item's ``swap_answer``. It is the specificity check the swap rates are read against.

An entity whose leading space-prefixed form is more than one token has no single J-space row. Such items
are refused from the primary denominator by name (``multi_token``) rather than approximated by their
first piece.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@dataclass
class BatteryConfig:
    model_name: str = "google/gemma-3-4b-it"
    jlens_model_id: str = "gemma-3-4b-it"
    items_file: str = "probe-swap.json"
    render_mode: str = "raw"  # raw | chat_prefill
    layers: list[int] | None = None  # None: every fitted layer
    bands: list[list[int]] = field(default_factory=list)  # [[lo, hi], ...] inclusive
    top_k: int = 5
    n_new_tokens: int = 4
    limit: int = 0
    device: str = "cuda"
    flip_band: list[int] | None = None  # [lo, hi]: also measure workspace-flip vs logit-flip for this band swap
    skip_sweep: bool = False  # skip the per-layer and band swaps (run only readout and the flip measurement)


#: Offset used to pick each item's control entity from another item, far enough to leave its category.
CONTROL_OFFSET = 37


def load_items(cfg: BatteryConfig) -> list[dict[str, Any]]:
    """Items, each given a ``control_to``: another item's ``swap_to``, unrelated to this item's swap.

    The control is the specificity check. Swapping the intermediate with an unrelated entity should NOT
    produce this item's ``swap_answer``; if it does at a similar rate, the swap rates measure disruption
    rather than a targeted swap. It is assigned over the full list before ``limit`` so it is stable.
    """
    with open(os.path.join(DATA_DIR, cfg.items_file)) as f:
        items = json.load(f)["items"]
    n = len(items)
    for i, it in enumerate(items):
        for k in range(n):
            cand = items[(i + CONTROL_OFFSET + k) % n]["swap_to"]
            if cand not in (it["swap_to"], it["intermediate"]):
                it["control_to"] = cand
                break
    return items[: cfg.limit] if cfg.limit else items


def render(tokenizer, prompt: str, mode: str) -> str:
    """``raw``: the fact prompt as-is. ``chat_prefill``: a user turn asking for completion, with the fact
    prefilled as the start of the model turn. The template's BOS is left in place and the prompt is then
    tokenized without special tokens, so exactly one BOS is present in both modes."""
    p = prompt.rstrip()
    if mode == "raw":
        return (tokenizer.bos_token or "") + p
    if mode == "chat_prefill":
        t = tokenizer.apply_chat_template([{"role": "user", "content": "Complete the sentence."}], tokenize=False,
                                          add_generation_prompt=True)
        return t + p
    raise ValueError(f"render_mode {mode!r} is not one of 'raw', 'chat_prefill'")


def single_token(tokenizer, text: str) -> int | None:
    """The id of `` text`` when it is exactly one content token, else ``None``.

    Gemma spells a leading-space number as a bare ``▁`` plus digits; the bare piece is not content, so it is
    dropped before counting, which keeps `` 8`` single-token.
    """
    ids = [i for i in tokenizer.encode(" " + text.strip(), add_special_tokens=False) if tokenizer.decode([i]).strip()]
    return ids[0] if len(ids) == 1 else None


class Battery:
    """One loaded model plus the resolved lens; methods run one measurement over one item."""

    def __init__(self, cfg: BatteryConfig):
        from nnsight import LanguageModel

        from interpretune.adapters.nnsight.backends import (
            HookNameResolver,
            NNsightModelBackend,
            get_default_configs_per_pass,
        )
        from interpretune.analysis.optools import resolve_jlens, resolve_unembed_and_norm_scale

        self.cfg = cfg
        kwargs = dict(device_map=cfg.device, dispatch=True, dtype=torch.bfloat16)
        try:
            self.lm = LanguageModel(cfg.model_name, **kwargs)
        except ValueError as err:
            # A multimodal checkpoint (gemma-3-4b-it) is refused by LanguageModel with a message naming
            # VisionLanguageModel, the subclass exposing the same tracing surface; interpretune's NNsight
            # adapter makes the same switch on the same message. Anything else is re-raised.
            if "VisionLanguageModel" not in str(err):
                raise
            from nnsight import VisionLanguageModel

            self.lm = VisionLanguageModel(cfg.model_name, **kwargs)
        self.hf = NNsightModelBackend._get_hf_model(self.lm)
        self.backend = NNsightModelBackend(HookNameResolver(self.hf.config.architectures[0]),
                                           configs_per_pass=get_default_configs_per_pass())
        self.backend.register_model_hooks(self.lm)
        self.tok = self.lm.tokenizer
        self.module = SimpleNamespace(model=self.hf)
        self.info = resolve_unembed_and_norm_scale(self.module)
        # The resolver's norm lookup covers ``model.norm`` layouts; a multimodal Gemma keeps its text norm at
        # ``model.language_model.norm``. An unresolved norm would silently yield UNFOLDED directions, which
        # on gemma-3 fail to steer at all, so a Gemma whose norm did not resolve as RMSNorm is refused.
        if "gemma" in type(self.hf).__name__.lower() and self.info.norm_kind != "rmsnorm":
            raise ValueError(f"{type(self.hf).__name__}: final norm resolved as {self.info.norm_kind!r}, "
                             "not 'rmsnorm'; the norm-aware basis would be wrong")
        self.final_norm = self._find_attr(("model.language_model.norm", "model.norm", "language_model.model.norm"))
        self.artifact = resolve_jlens(self.module, model_id=cfg.jlens_model_id)
        self.layers = list(cfg.layers) if cfg.layers is not None else list(self.artifact.source_layers)
        self._J: dict[int, torch.Tensor] = {}

    # -- lens -------------------------------------------------------------------------------------------------
    def J(self, layer: int) -> torch.Tensor:
        from interpretune.analysis.optools import resolve_jlens_layer

        if layer not in self._J:
            j, resolved, _ = resolve_jlens_layer(self.module, {}, {"jlens_model_id": self.cfg.jlens_model_id,
                                                                  "jlens_layer": layer})
            if resolved != layer:
                raise ValueError(f"asked for lens layer {layer}, resolver returned {resolved}")
            self._J[layer] = j.to(self.cfg.device)
        return self._J[layer]

    def rows(self, token_ids: list[int], layer: int) -> torch.Tensor:
        from interpretune.analysis.optools import jlens_direction_rows

        return jlens_direction_rows(self.info, token_ids, self.J(layer), apply_norm=True).detach().float()

    # -- measurements -----------------------------------------------------------------------------------------
    def encode(self, text: str) -> torch.Tensor:
        return self.tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.cfg.device)

    @torch.no_grad()
    def baseline(self, ids: torch.Tensor, answer: str) -> tuple[bool, str]:
        out = self.hf.generate(ids, max_new_tokens=self.cfg.n_new_tokens, do_sample=False)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text.strip().lower().startswith(answer.strip().lower()), text

    @torch.no_grad()
    def readout_ranks(self, ids: torch.Tensor, token_ids: list[int]) -> dict[int, list[int]]:
        """Best (minimum over prompt positions after BOS) lens rank of each token, per layer; 0 is top-1."""
        decoder_layers = self._decoder_layers()
        acts: dict[int, torch.Tensor] = {}
        hooks = [decoder_layers[l].register_forward_hook(self._grab(acts, l)) for l in self.layers]
        try:
            self.hf(ids)
        finally:
            for h in hooks:
                h.remove()
        out = {}
        tid = torch.tensor(token_ids, device=self.cfg.device)
        for l in self.layers:
            h = acts[l][0, 1:].float() @ self.J(l).T
            logits = self._unembed(h)
            tgt = logits[:, tid]  # [pos, n]
            ranks = (logits.unsqueeze(1) > tgt.unsqueeze(-1)).sum(-1)  # [pos, n]
            out[l] = ranks.min(0).values.tolist()
        return out

    @torch.no_grad()
    def patch(self, ids: torch.Tensor, pair_ids: list[int], layers: list[int]) -> torch.Tensor:
        """Final-position logits after patching the J-space pair at every prompt position of ``layers``."""
        from interpretune.analysis.backends.capabilities import PositionScope
        from interpretune.analysis.backends.interventions import (
            InterventionDict,
            InterventionSpec,
            _validate_intervention_spec,
        )

        specs = {}
        for l in layers:
            pair = self.rows(pair_ids, l)
            hook = f"blocks.{l}.hook_resid_post"
            specs[hook] = (_validate_intervention_spec(
                InterventionSpec(intervention_tensor=pair, mode="patch", scale_factor=1.0,
                                 position_scope=PositionScope.ALL_POSITIONS),
                target_shape=(pair.shape[1],), hook_name=hook),)
        pre, post = self.backend.fwd_w_intervention(model=self.lm, batch={"input_ids": ids},
                                                    interventions=InterventionDict(specs))
        self._last_pre = pre[0, -1].float()
        return post[0, -1].float()

    @torch.no_grad()
    def patch_with_readout(self, ids: torch.Tensor, pair_ids: list[int], layers: list[int],
                           watch_ids: list[int]) -> tuple[torch.Tensor, dict[int, list[int]]]:
        """Patch as :meth:`patch` does, and also read the lens at the final position of every fitted layer.

        Returns the post-intervention final logits and, per fitted layer, the lens rank of each ``watch_ids``
        token in the INTERVENED pass. ``fwd_w_intervention`` runs a clean pass and an intervened pass and plain
        forward hooks fire in both, so the intervened pass is identified rather than assumed: it is the recorded
        pass whose final-layer activation reproduces the returned logits. If none does, this refuses.
        """
        decoder_layers = self._decoder_layers()
        final = len(decoder_layers) - 1
        passes: list[dict[int, torch.Tensor]] = [{}]

        def grab(l):
            def hook(module, inputs, output):
                t = (output if torch.is_tensor(output) else output[0]).detach()
                if l in passes[-1]:
                    passes.append({})
                passes[-1][l] = t[0, -1].float()
            return hook

        watch = sorted(set(self.layers) | {final})
        hooks = [decoder_layers[l].register_forward_hook(grab(l)) for l in watch]
        try:
            post = self.patch(ids, pair_ids, layers)
        finally:
            for h in hooks:
                h.remove()
        w = self.hf.get_output_embeddings().weight
        pre = self._last_pre

        def find(target):
            hit = None
            for rec in passes:
                if final in rec:
                    logits = (self.final_norm(rec[final].to(w.dtype)) @ w.T).float()
                    if torch.allclose(logits, target, rtol=1e-2, atol=1e-1):
                        hit = rec
            return hit

        intervened, clean = find(post), find(pre)
        if intervened is None or clean is None or intervened is clean:
            raise RuntimeError(f"could not tell the clean and intervened passes apart among {len(passes)} recorded "
                               "passes; the readout cannot be attributed")
        tid = torch.tensor(watch_ids, device=post.device)

        def ranks_of(rec):
            out = {}
            for l in self.layers:
                lg = self._unembed((rec[l] @ self.J(l).T).unsqueeze(0))[0]
                out[l] = [int((lg > lg[t]).sum()) for t in tid]
            return out

        return post, ranks_of(intervened), ranks_of(clean)

    # -- helpers ----------------------------------------------------------------------------------------------
    def _find_attr(self, paths):
        for path in paths:
            obj = self.hf
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise ValueError(f"none of {paths} exists on {type(self.hf).__name__}")

    def _decoder_layers(self):
        return self._find_attr(("model.language_model.layers", "model.layers", "language_model.model.layers"))

    @staticmethod
    def _grab(store, l):
        def hook(module, inputs, output):
            store[l] = (output if torch.is_tensor(output) else output[0]).detach()
        return hook

    def _unembed(self, h: torch.Tensor) -> torch.Tensor:
        w = self.hf.get_output_embeddings().weight
        return (self.final_norm(h.to(w.dtype)) @ w.T).float()


def rank_of(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum())


def run_item(b: Battery, item: dict[str, Any]) -> dict[str, Any]:
    cfg = b.cfg
    toks = {k: single_token(b.tok, item[k])
            for k in ("intermediate", "answer", "swap_to", "swap_answer", "control_to")}
    row: dict[str, Any] = {"name": item["name"], "category": item.get("category"), "tokens": toks}
    missing = [k for k, v in toks.items() if v is None and k != "control_to"]
    ids = b.encode(render(b.tok, item["prompt"], cfg.render_mode))
    ok, cont = b.baseline(ids, item["answer"])
    row.update(baseline_correct=ok, continuation=cont)
    if missing:
        row["refused"] = f"multi_token: {missing}"
        return row
    with torch.no_grad():
        base_logits = b.hf(ids).logits[0, -1].float()
    row["baseline_rank"] = {"answer": rank_of(base_logits, toks["answer"]),
                            "swap_answer": rank_of(base_logits, toks["swap_answer"])}
    row["readout_rank"] = b.readout_ranks(ids, [toks["intermediate"], toks["answer"]])
    swaps: dict[str, Any] = {}
    for kind, pair in (("intermediate", [toks["intermediate"], toks["swap_to"]]),
                       ("answer", [toks["answer"], toks["swap_answer"]]),
                       ("control", [toks["intermediate"], toks["control_to"]])):
        if None in pair:  # a multi-token control entity skips the control only, never the item
            continue
        if cfg.skip_sweep:
            break
        per_layer = {}
        for l in b.layers:
            post = b.patch(ids, pair, [l])
            per_layer[l] = {"swap_answer_rank": rank_of(post, toks["swap_answer"]),
                            "answer_rank": rank_of(post, toks["answer"])}
        bands = {}
        for lo, hi in cfg.bands:
            post = b.patch(ids, pair, [l for l in b.layers if lo <= l <= hi])
            bands[f"{lo}-{hi}"] = {"swap_answer_rank": rank_of(post, toks["swap_answer"]),
                                   "answer_rank": rank_of(post, toks["answer"])}
        swaps[kind] = {"per_layer": per_layer, "bands": bands}
    row["swap"] = swaps
    if cfg.flip_band is not None:
        lo, hi = cfg.flip_band
        band = [l for l in b.layers if lo <= l <= hi]
        post, ranks, clean = b.patch_with_readout(ids, [toks["intermediate"], toks["swap_to"]], band,
                                                  [toks["answer"], toks["swap_answer"]])
        # workspace flip: first fitted layer ABOVE the band where the lens ranks swap_answer over answer. The same
        # test on the CLEAN pass is the control: a lens that already prefers swap_answer there without any swap
        # would count as a "flip" and prove nothing.
        above = [l for l in b.layers if l > hi]
        ws = next((l for l in above if ranks[l][1] < ranks[l][0]), None)
        ws_clean = next((l for l in above if clean[l][1] < clean[l][0]), None)
        row["flip"] = {"band": [lo, hi], "logit_flipped": bool(post[toks["swap_answer"]] > post[toks["answer"]]),
                       "workspace_flip_layer": ws, "clean_workspace_flip_layer": ws_clean,
                       "ranks_above_band": {l: ranks[l] for l in above},
                       "clean_ranks_above_band": {l: clean[l] for l in above}}
    return row


def summarize(rows: list[dict[str, Any]], layers: list[int], top_k: int) -> dict[str, Any]:
    """Rates with denominators, over baseline-correct single-token items.

    Two swap criteria, because a top-k criterion alone cannot fail on an item whose ``swap_answer`` is
    already in the baseline top-k (a second-choice answer such as Spanish after Portuguese):

    - ``top1``: ``swap_answer`` becomes the model's argmax (the paper's criterion), over all usable items;
    - ``topk_disc``: ``swap_answer`` enters the top-k, over the DISCRIMINATING items only, those whose baseline
      rank for it is ``>= top_k``. Its denominator is reported separately as ``n_discriminating``.
    """
    usable = [r for r in rows if r["baseline_correct"] and "refused" not in r]
    disc = [r for r in usable if r["baseline_rank"]["swap_answer"] >= top_k]
    flips = [r["flip"] for r in usable if "flip" in r]
    out: dict[str, Any] = {
        "n_items": len(rows),
        "n_baseline_correct": sum(r["baseline_correct"] for r in rows),
        "n_refused_multi_token": sum("refused" in r for r in rows),
        "n_usable": len(usable),
        "n_discriminating": len(disc),
        "top_k": top_k,
    }
    if not usable:
        return out
    if flips:
        flipped = [f for f in flips if f["logit_flipped"]]
        ws_before = [f for f in flipped if f["workspace_flip_layer"] is not None]
        out["flip"] = {
            "band": flips[0]["band"],
            "n": len(flips),
            "logit_flip_rate": len(flipped) / len(flips),
            "n_logit_flipped": len(flipped),
            # of the items whose final logits flipped, how many showed the flip in the workspace at a fitted
            # layer above the band first, and at which layer
            "workspace_flip_first_rate": (len(ws_before) / len(flipped)) if flipped else None,
            "workspace_flip_layers": sorted(f["workspace_flip_layer"] for f in ws_before),
            # workspace flipped but the logits did not: the readout moved and the output held
            "workspace_only_rate": sum(1 for f in flips if not f["logit_flipped"] and f["workspace_flip_layer"]
                                       is not None) / len(flips),
            # the control: how often the CLEAN workspace already "flips" above the band, with no swap at all
            "clean_workspace_flip_rate": sum(1 for f in flips if f.get("clean_workspace_flip_layer") is not None)
            / len(flips),
            "clean_workspace_flip_layers": sorted(f["clean_workspace_flip_layer"] for f in flips
                                                  if f.get("clean_workspace_flip_layer") is not None),
        }

    def rate(pop, pred) -> float | None:
        return sum(1 for r in pop if pred(r)) / len(pop) if pop else None

    out["readable_intermediate"] = {l: rate(usable, lambda r, l=l: r["readout_rank"][l][0] < top_k) for l in layers}
    out["readable_answer"] = {l: rate(usable, lambda r, l=l: r["readout_rank"][l][1] < top_k) for l in layers}
    if not usable[0].get("swap"):
        return out
    bands = list(usable[0]["swap"]["intermediate"]["bands"].keys())
    ctrl = [r for r in usable if "control" in r["swap"]]
    ctrl_disc = [r for r in ctrl if r["baseline_rank"]["swap_answer"] >= top_k]
    out["n_control"], out["n_control_discriminating"] = len(ctrl), len(ctrl_disc)
    for kind in ("intermediate", "answer", "control"):
        pop, dpop = (ctrl, ctrl_disc) if kind == "control" else (usable, disc)
        pl = lambda r, l, kind=kind: r["swap"][kind]["per_layer"][l]["swap_answer_rank"]  # noqa: E731
        bd = lambda r, bk, kind=kind: r["swap"][kind]["bands"][bk]["swap_answer_rank"]  # noqa: E731
        out[f"swap_{kind}_top1"] = {l: rate(pop, lambda r, l=l: pl(r, l) == 0) for l in layers}
        out[f"swap_{kind}_topk_disc"] = {l: rate(dpop, lambda r, l=l: pl(r, l) < top_k) for l in layers}
        out[f"swap_{kind}_band_top1"] = {bk: rate(pop, lambda r, bk=bk: bd(r, bk) == 0) for bk in bands}
        out[f"swap_{kind}_band_topk_disc"] = {bk: rate(dpop, lambda r, bk=bk: bd(r, bk) < top_k) for bk in bands}
    return out


def battery_config_from_experiment(cfg: dict[str, Any]) -> BatteryConfig:
    """Build a :class:`BatteryConfig` from a harness experiment config (the ``EXPERIMENT`` and ``MODEL`` sections).

    Unknown ``EXPERIMENT`` keys are refused by name rather than ignored, so a misspelled knob cannot silently run
    the default.
    """
    exp = dict(cfg.get("EXPERIMENT") or {})
    known = set(BatteryConfig.__dataclass_fields__)
    unknown = sorted(set(exp) - known)
    if unknown:
        raise ValueError(f"EXPERIMENT has keys the battery does not understand: {unknown}; known: {sorted(known)}")
    model = cfg.get("MODEL") or {}
    if "model_name" not in exp and model.get("model_name"):
        exp["model_name"] = model["model_name"]
    return BatteryConfig(**exp)


def run_battery(cfg: BatteryConfig, progress: Any = print) -> dict[str, Any]:
    """Run every item and return ``{"config", "lens", "summary", "rows"}``."""
    b = Battery(cfg)
    rows = []
    for i, item in enumerate(load_items(cfg)):
        rows.append(run_item(b, item))
        if progress is not None:
            r = rows[-1]
            progress(f"[{i + 1}] {r['name']:<30} base={'Y' if r['baseline_correct'] else 'n'} {r.get('refused', '')}")
    return {
        "config": {k: getattr(cfg, k) for k in BatteryConfig.__dataclass_fields__},
        "lens": {"repo_id": b.artifact.repo_id, "path": b.artifact.path,
                 "source_layers": list(b.artifact.source_layers)},
        "summary": summarize(rows, b.layers, cfg.top_k),
        "rows": rows,
    }


def rate_table(summary: dict[str, Any], layers: list[int]) -> str:
    """Markdown per-layer table of the headline rates, with the denominators in the header."""
    cols = [
        ("readable_intermediate", "intermediate readable"),
        ("swap_intermediate_top1", "intermediate swap top-1"),
        ("swap_control_top1", "control swap top-1"),
        ("swap_intermediate_topk_disc", "intermediate swap top-k (disc.)"),
        ("swap_control_topk_disc", "control swap top-k (disc.)"),
        ("readable_answer", "answer readable"),
        ("swap_answer_top1", "answer swap top-1"),
    ]
    head = (f"n_usable={summary['n_usable']}, n_discriminating={summary['n_discriminating']}, "
            f"n_control={summary.get('n_control')}, k={summary['top_k']}\n\n")
    lines = ["| layer | " + " | ".join(c[1] for c in cols) + " |", "| --- |" + " --- |" * len(cols)]
    for l in layers:
        cells = []
        for key, _ in cols:
            v = (summary.get(key) or {}).get(l, (summary.get(key) or {}).get(str(l)))
            cells.append("-" if v is None else f"{v:.2f}")
        lines.append(f"| {l} | " + " | ".join(cells) + " |")
    return head + "\n".join(lines)
