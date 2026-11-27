"""Interpretune's static notebook-form cases, run over this repository's published tutorials.

The cases (``interpretune.testing.conformance.NotebookFormConformance``) need no kernel: published copies in sync with
dev, the badge and install cells leading, no ``remove-cell`` survivors, and every import resolvable in the environment
the tutorials run in.
"""

from __future__ import annotations

from pathlib import Path

from interpretune.testing.conformance import NotebookFormConformance


class TestTutorialNotebookForm(NotebookFormConformance):
    """The notebook-form cases, against ``[tool.nbpress]`` in this repository's ``pyproject.toml``."""

    root = Path(__file__).resolve().parents[2]
    # `exp` is the concept-direction component's package: the latent-dynamics tutorial imports it from the snapshot it
    # fetches, after the reader opts in to that component's code, so no environment has it installed
    ignored_imports = ("exp",)
