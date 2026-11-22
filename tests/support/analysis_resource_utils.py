"""NNsight test-state reset used around each notebook execution.

Stands in for interpretune's ``tests/analysis_resource_utils.py``, of which the tests here use only
``clear_nnsight_test_state``.
"""

from __future__ import annotations

from typing import Any


def clear_nnsight_test_state(obj: Any) -> None:
    """Best-effort reset of test-visible NNsight tracing state on an object graph.

    This is intended for test teardown after in-process serial execution patterns.
    It clears cached ``Envoy.source`` state and default edit mediators when those
    attributes are present. (nnsight 0.8 keeps trace state per trace scope, so there
    is no global tracing cache left to reset.)
    """

    visited: set[int] = set()

    def _visit(candidate: Any) -> None:
        if candidate is None:
            return
        candidate_id = id(candidate)
        if candidate_id in visited:
            return
        visited.add(candidate_id)

        if hasattr(candidate, "clear_edits"):
            try:
                candidate.clear_edits()
            except Exception:
                pass

        if hasattr(candidate, "_source"):
            try:
                setattr(candidate, "_source", None)
            except Exception:
                pass

        for attr_name in ("module", "replacement_model", "model"):
            nested = getattr(candidate, attr_name, None)
            if nested is not None and nested is not candidate:
                _visit(nested)

        for child in getattr(candidate, "_children", ()) or ():
            _visit(child)

    _visit(obj)
