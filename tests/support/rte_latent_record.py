"""A core-loop callback recording what each epoch of the RTE latent tuning runs trained, per rank."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _whole(param: torch.Tensor) -> torch.Tensor:
    return (param.full_tensor() if hasattr(param, "full_tensor") else param).detach().float().cpu()


class LatentEpochRecorder:
    """Writes ``out`` (``{rank}`` filled in) at the end of training; place it after ``FTSCoreCallback``.

    For each epoch: the schedule depth and trainable parameters it started with, its validation loss, and the adapter
    and transcoder weights it changed. Also the FSDP2 units, the checkpointed ones, and each parameter's type and dtype.
    """

    def __init__(self, out: str) -> None:
        self.out = Path(out.format(rank=os.environ.get("RANK", "0")))
        self.epochs: list[dict] = []

    @staticmethod
    def _tuned(module) -> dict[str, torch.Tensor]:
        return {
            n: _whole(p).clone() for n, p in module.named_parameters() if "lora_" in n or n.startswith("latent_models.")
        }

    def on_train_epoch_start(self, ctx) -> None:
        from interpretune.tuning.core_loop import FTSCoreCallback

        fts = next(cb for cb in ctx.callbacks if isinstance(cb, FTSCoreCallback))
        self._start = self._tuned(ctx.module)
        trainable = sorted(n for n, p in ctx.module.named_parameters() if p.requires_grad)
        self.epochs.append({"epoch": ctx.epoch, "depth": fts.tuning_scheduler.curr_depth, "trainable": trainable})

    def on_train_epoch_end(self, ctx, metrics) -> None:
        now = self._tuned(ctx.module)
        self.epochs[-1]["changed"] = sorted(n for n, before in self._start.items() if not torch.equal(before, now[n]))
        self.epochs[-1]["val_loss"] = float(metrics["val_loss"])

    def on_train_end(self, ctx) -> None:
        from torch.distributed._composable.contract import _get_registry
        from torch.distributed.fsdp import FSDPModule

        units = {n: m for n, m in ctx.module.named_modules() if isinstance(m, FSDPModule)}
        for unit in units.values():  # the root unit stays gathered after the last forward
            unit.reshard()
        record = {
            "epochs": self.epochs,
            "units": sorted(units),
            "checkpointed": sorted(n for n, m in units.items() if "checkpoint" in (_get_registry(m) or {})),
            "reduce_dtypes": sorted({str(m._get_fsdp_state()._mp_policy.reduce_dtype) for m in units.values()}),
            "types": sorted({type(p).__name__ for p in ctx.module.parameters()}),
            "dtypes": {n: str(p.dtype) for n, p in ctx.module.named_parameters()},
        }
        self.out.write_text(json.dumps(record), encoding="utf-8")
