"""Fixtures for the dashboard tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CATALOG = Path(__file__).resolve().parents[2] / "catalog.yaml"


def _catalog_revision(repo: str) -> str:
    """The revision ``catalog.yaml`` pins for ``repo``; refused when the catalog does not list it."""
    entries = yaml.safe_load(CATALOG.read_text())["components"]
    for entry in entries:
        if entry["repo"] == repo:
            return entry["revision"]
    raise LookupError(f"{repo} is not in {CATALOG.name}, so there is no pinned revision to fetch.")


@pytest.fixture(scope="session", autouse=True)
def rte_component_cached() -> None:
    """Fetch ``speediedan/rte`` at its catalog revision, once per session.

    The dashboard helpers resolve the RTE experiment (``rte_boolq``) from the components cache, and never fall back to
    another copy. A runner starts with that cache empty, so the tests fetch the pinned revision themselves rather than
    depending on a cache warmed elsewhere.
    """
    import interpretune as it

    it.hub.pull("speediedan/rte", revision=_catalog_revision("speediedan/rte"))
