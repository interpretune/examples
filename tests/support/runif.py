"""``RunIf``: a ``pytest.mark.skipif`` factory for the conditions the tutorial tests declare.

Stands in for interpretune's ``tests/runif.py``, carrying only the conditions this repository's tests pass:
``bf16_cuda``, ``min_gpu_mem_gb``, ``optional`` and ``skip_windows``. The skip conditions and the mark kwargs are
interpretune's, so a test skips here exactly where it skips there, and the kwargs left on the mark are what
``tests/conftest.py`` reads to select a phase (``IT_RUN_CUDA_TESTS``, ``IT_RUN_OPTIONAL_TESTS``) and to check declared
GPU memory at setup. A condition this module does not know is refused by name rather than ignored.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch


class RunIf:
    """RunIf wrapper for marking specific cases, a ``pytest.mark.skipif`` decorator factory:

    @RunIf(bf16_cuda=True, min_gpu_mem_gb=10.0)
    def test_on_a_bf16_device():
        ...
    """

    def __new__(
        cls,
        *args,
        min_gpu_mem_gb: float = 0,
        bf16_cuda: bool = False,
        skip_windows: bool = False,
        optional: bool = False,
        **kwargs,
    ):
        """
        Args:
            *args: Any :class:`pytest.mark.skipif` positional arguments.
            min_gpu_mem_gb: The GPU memory, in GiB of device TOTAL, the device this test uses must have. Checked when
                the test is set up, not at collection (which would initialize CUDA in every collecting process): a
                smaller device skips the test, or fails it when ``IT_GPU_STRICT=1``, where a skip would hide a
                misplaced test. Requires ``bf16_cuda``.
            bf16_cuda: Require that the CUDA device supports bf16.
            skip_windows: Skip on the Windows platform.
            optional: Mark the test as for optional/extended testing. It runs only when the
                ``IT_RUN_OPTIONAL_TESTS=1`` environment variable is set.
            **kwargs: Refused by name: every keyword this class accepts is named above.
        """
        if kwargs:
            raise TypeError(
                f"RunIf does not know the condition(s) {sorted(kwargs)}: this repository carries only bf16_cuda, "
                "min_gpu_mem_gb, optional and skip_windows. Add the condition here (from interpretune's "
                "tests/runif.py) rather than passing it through to the mark, where nothing would read it."
            )
        conditions = []
        reasons = []

        if min_gpu_mem_gb:
            if not bf16_cuda:
                raise ValueError(
                    f"RunIf(min_gpu_mem_gb={min_gpu_mem_gb}) declares GPU memory for a test that declares no GPU: "
                    "add bf16_cuda=True, or drop the memory declaration."
                )
            if min_gpu_mem_gb < 0:
                raise ValueError(f"RunIf(min_gpu_mem_gb={min_gpu_mem_gb}) must be a positive number of GiB.")
            # checked in conftest.py::pytest_runtest_setup, when the test is about to use the device anyway
            kwargs["min_gpu_mem_gb"] = float(min_gpu_mem_gb)

        if bf16_cuda:
            try:
                cond = not (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            except (AssertionError, RuntimeError) as e:
                # AssertionError: Torch not compiled with CUDA enabled
                # RuntimeError: Found no NVIDIA driver on your system.
                is_unrelated = "Found no NVIDIA driver" not in str(e) or "Torch not compiled with CUDA" not in str(e)
                if is_unrelated:
                    raise e
                cond = True

            conditions.append(cond)
            reasons.append("CUDA device bf16")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["bf16_cuda"] = True

        if skip_windows:
            conditions.append(sys.platform == "win32")
            reasons.append("unimplemented on Windows")

        if optional:
            env_flag = os.getenv("IT_RUN_OPTIONAL_TESTS", "0")
            conditions.append(env_flag != "1")
            reasons.append("Optional/extended test execution")
            # used in conftest.py::pytest_collection_modifyitems
            kwargs["optional"] = True

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args, condition=any(conditions), reason=f"Requires: [{' + '.join(reasons)}]", **kwargs
        )
