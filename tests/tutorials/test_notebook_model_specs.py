"""The tutorials bring their own model specs: a table in the notebook, or the file its component ships.

These notebooks need a GPU, so a selection missing from a table, or a call that falls back to the specs bundled with
interpretune, would otherwise surface only in a GPU run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from interpretune.harness.sessions import resolve_model_spec

NOTEBOOKS = Path(__file__).parents[2] / "tutorials" / "publish" / "circuit_tracer_examples"
NOTEBOOK = NOTEBOOKS / "jspace_workspace_demo.ipynb"
LATENT_DYNAMICS = NOTEBOOKS / "concept_direction_latent_dynamics.ipynb"


def _code(notebook: Path = NOTEBOOK) -> list[str]:
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    return ["".join(cell["source"]) for cell in cells if cell["cell_type"] == "code"]


def _assigned(name: str):
    """The literal the notebook assigns to ``name``."""
    for source in _code():
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and [getattr(t, "id", None) for t in node.targets] == [name]:
                return ast.literal_eval(node.value)
    raise AssertionError(f"the notebook assigns no {name}")


def _selections() -> set[tuple[str, str]]:
    """The notebook's default selection and every one its GPU tests inject."""
    tests = ast.parse((Path(__file__).parent / "test_notebooks.py").read_text(encoding="utf-8"))
    found = {(_assigned("MODEL_FAMILY"), _assigned("MODEL_VARIANT"))}
    for node in ast.walk(tests):
        if not isinstance(node, ast.Assign) or getattr(node.targets[0], "id", None) != "JSPACE_WORKSPACE_PARAMS":
            continue
        for mapping in (n for n in ast.walk(node.value) if isinstance(n, ast.Dict)):
            params = ast.literal_eval(mapping)
            found.add((params["MODEL_FAMILY"], params["MODEL_VARIANT"]))
    return found


def test_the_selections_include_an_injected_one():
    assert len(_selections()) > 1  # only the default would mean the tests' parameters were not found


@pytest.mark.parametrize("family, variant", sorted(_selections()))
def test_each_selection_resolves_in_the_notebooks_own_specs(family, variant):
    spec = resolve_model_spec(family, variant, model_specs=_assigned("MODEL_SPECS"))
    assert spec.model_name and "nnsight" in spec.composition


def test_the_session_uses_the_notebooks_own_specs():
    calls = [
        node
        for source in _code()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "experiment_session"
    ]
    assert len(calls) == 1
    assert {kw.arg: getattr(kw.value, "id", None) for kw in calls[0].keywords}["model_specs"] == "MODEL_SPECS"


def test_the_latent_dynamics_tutorial_resolves_from_the_component_it_pulled():
    calls = [
        node
        for source in _code(LATENT_DYNAMICS)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "resolve_model_spec"
    ]
    assert len(calls) == 1
    keywords = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
    assert keywords == {"model_specs": "snapshot_dir / 'model_specs.yaml'"}
