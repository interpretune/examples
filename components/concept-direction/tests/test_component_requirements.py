"""The manifest requires the components this experiment resolves, and no others."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import yaml

# not resolved: in a pulled snapshot this file is a link into the cache's blob store, away from its siblings
COMPONENT = Path(__file__).parents[1]
MANIFEST = yaml.safe_load((COMPONENT / "it_component.yaml").read_text(encoding="utf-8"))
COMPONENT_REFERENCE = re.compile(r"(?P<repo>speediedan/[\w.-]+)#\w+")


def _resolved_components() -> set[str]:
    """Each ``<org>/<repo>`` a string in the payload code names as ``<org>/<repo>#<definition>``."""
    found = set()
    for path in [*(COMPONENT / "exp").rglob("*.py"), *COMPONENT.glob("*.py")]:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                found |= {match["repo"] for match in COMPONENT_REFERENCE.finditer(node.value)}
    return found


def test_the_payload_resolves_a_component():
    assert "speediedan/prompt-configs" in _resolved_components()  # an empty set would make the check below pass


def test_the_manifest_requires_exactly_the_components_it_resolves():
    required = {ref.partition("@")[0] for ref in MANIFEST["requires"]["components"]}
    assert required == _resolved_components()


def test_every_required_component_is_pinned_to_a_commit():
    for ref in MANIFEST["requires"]["components"]:
        assert re.fullmatch(r"[0-9a-f]{40}", ref.partition("@")[2]), ref
