"""Platform facts the tutorial tests branch on.

Stands in for interpretune's ``tests/results.py``, of which the tests here use only ``on_rocm``.
"""

from __future__ import annotations


def on_rocm() -> bool:
    """Whether torch was built for ROCm (HIP), whose ``torch.cuda`` surface these tests run on unchanged."""
    import torch

    return getattr(torch.version, "hip", None) is not None
