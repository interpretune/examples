"""Session setup for the examples tests."""

from __future__ import annotations

import os

import pytest
import torch

# The tests import component entrypoints (speediedan/rte and others) from the components cache, which interpretune
# refuses to execute without an explicit opt-in. These are this repository's own first-party components.
os.environ.setdefault("IT_TRUST_REMOTE_CODE", "1")


# Test selection, standing in for the parts of interpretune's ``tests/conftest.py`` that the RunIf conditions in
# ``tests/support/runif.py`` imply. Interpretune runs its special tiers as separate phases, each selected by one
# environment flag; the flags are checked in interpretune's order, so a flag set here selects what it selects there.
# The tiers no test here carries (standalone, profiling, benchmark) are kept in the order so that setting one of their
# flags selects nothing rather than falling through to the default suite.

_SPECIAL_TIERS = ("standalone", "standalone_cpu", "profiling", "profiling_ci", "optional", "benchmark")


def _marked(item, predicate) -> bool:
    """Whether ``item`` carries a ``RunIf`` mark whose kwargs satisfy ``predicate``, at ANY level (class marks too)."""
    return any(marker.name == "skipif" and predicate(marker.kwargs) for marker in item.iter_markers())


def cuda_phase_selects(kw: dict) -> bool:
    """Whether a ``RunIf`` mark's kwargs put a test in the cuda-marked phase: a GPU mark and no special tier."""
    return not any(kw.get(k) for k in _SPECIAL_TIERS) and bool(kw.get("min_cuda_gpus") or kw.get("bf16_cuda"))


# (flag, value, predicate over the RunIf kwargs), first match wins, in interpretune's order
_PHASE_SELECTORS = (
    ("IT_RUN_STANDALONE_TESTS", ("1",), lambda kw: kw.get("standalone")),
    ("IT_RUN_STANDALONE_CPU_TESTS", ("1",), lambda kw: kw.get("standalone_cpu")),
    ("IT_RUN_CUDA_TESTS", ("1",), cuda_phase_selects),
    ("IT_RUN_PROFILING_TESTS", ("2",), lambda kw: kw.get("profiling")),
    ("IT_RUN_PROFILING_TESTS", ("1",), lambda kw: kw.get("profiling_ci")),
    ("IT_RUN_OPTIONAL_TESTS", ("1",), lambda kw: kw.get("optional")),
    ("IT_RUN_BENCHMARK_TESTS", ("1",), lambda kw: kw.get("benchmark")),
)


def _hub_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(items):
    for flag, values, predicate in _PHASE_SELECTORS:
        if os.getenv(flag, "0") in values:
            items[:] = [item for item in items if _marked(item, predicate)]
            break
    # a test that must reach the live Hub declares it (`hf_live`) and skips when the Hub client is offline
    if _hub_offline():
        for item in items:
            if item.get_closest_marker("hf_live") is not None:
                item.add_marker(pytest.mark.skip(reason="needs the live Hub, and HF_HUB_OFFLINE=1 is set"))


def _declared_gpu_mem_gb(item) -> float:
    """The largest ``RunIf(min_gpu_mem_gb=...)`` declared on ``item`` at any level, or 0.0."""
    return max(
        (float(m.kwargs.get("min_gpu_mem_gb") or 0) for m in item.iter_markers() if m.name == "skipif"), default=0.0
    )


def pytest_runtest_setup(item):
    """Skip (or, with ``IT_GPU_STRICT=1``, fail) a test whose device is smaller than it declares.

    Runs at setup rather than collection: reading device properties initializes CUDA, which every collecting process
    would otherwise pay, and a GPU test about to run initializes it anyway. The tests here use one device.
    """
    need = _declared_gpu_mem_gb(item)
    if not need or not torch.cuda.is_available():
        return
    props = torch.cuda.get_device_properties(0)
    have = props.total_memory / 2**30
    if have < need:
        msg = f"{item.nodeid} declares min_gpu_mem_gb={need:g} but device 0 ({props.name}) has {have:.1f} GiB"
        if os.getenv("IT_GPU_STRICT", "0") == "1":
            pytest.fail(f"{msg}; IT_GPU_STRICT=1, so a misplaced GPU test fails rather than skips", pytrace=False)
        pytest.skip(msg)
