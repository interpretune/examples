"""The RTE experiment's classes, resolved from the cached ``speediedan/rte`` component.

Stands in for interpretune's ``tests/rte_component.py``. Attributes resolve lazily (PEP 562), so importing this module
costs nothing; the first attribute read imports the entrypoint from the components cache. A component that is not
cached fails there, naming the fetch that fixes it, rather than falling back to another copy.
"""

from __future__ import annotations

from types import ModuleType


def rte_entrypoint_module() -> ModuleType:
    """Import the entrypoint of the cached component that owns ``rte_boolq``."""
    from interpretune.hub.entrypoints import find_entrypoint_owner, import_snapshot_entrypoint

    owner = find_entrypoint_owner("rte_boolq")
    if owner is None:
        raise ImportError(
            "The RTE experiment (speediedan/rte) is not in the local components cache. "
            "Fetch it once with it.hub.pull('speediedan/rte')."
        )
    repo_id, entrypoint = owner
    return import_snapshot_entrypoint(
        repo_id, entrypoint, what=f"the experiment entrypoint {entrypoint!r} of {repo_id!r}"
    )


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(rte_entrypoint_module(), name)
