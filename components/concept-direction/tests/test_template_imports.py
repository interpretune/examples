"""The experiment template's imports resolve: it runs on a GPU, so a name it imports that no longer exists is otherwise
found only by someone running it."""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest

# not resolved: in a pulled snapshot this file is a link into the cache's blob store, away from its siblings
TEMPLATE = Path(__file__).parents[1] / "concept_direction" / "concept_direction_template.ipynb"


def _imports() -> list[tuple[str, str]]:
    """Every ``(module, name)`` the template's code cells import with ``from module import name``."""
    cells = json.loads(TEMPLATE.read_text(encoding="utf-8"))["cells"]
    found = []
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.ImportFrom) and node.module and node.module != "__future__":
                found += [(node.module, alias.name) for alias in node.names]
    return sorted(set(found))


def test_the_template_imports_something():
    assert len(_imports()) > 20  # an empty list would make the check below pass on nothing


@pytest.mark.parametrize("module, name", _imports())
def test_a_template_import_resolves(module, name):
    assert hasattr(importlib.import_module(module), name)


def test_the_template_imports_nothing_from_the_retired_examples_package():
    assert not [m for m, _ in _imports() if m.partition(".")[0] == "it_examples"]
