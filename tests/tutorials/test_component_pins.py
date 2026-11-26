"""The tutorials take their helpers from interpretune and pinned Hub components, and each pin is stated once."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

DEV = Path(__file__).resolve().parents[2] / "tutorials" / "dev"

# This one still imports interpretune's in-tree `it_examples`: interp_engine_hub_adapter waits for a host with the
# memory to re-render it.
NOT_YET_REPOINTED = {
    "interp_engine_example/interp_engine_hub_adapter.ipynb",
}


def _code(path: Path) -> str:
    if path.suffix != ".ipynb":
        return path.read_text()
    cells = json.loads(path.read_text())["cells"]
    return "\n".join("".join(cell["source"]) for cell in cells if cell["cell_type"] == "code")


def _pins(path: Path) -> dict[str, str]:
    return dict(re.findall(r'^(\w*REVISION) = "([0-9a-f]{40})"', _code(path), flags=re.MULTILINE))


def test_only_the_listed_tutorials_import_it_examples():
    importing = {
        path.relative_to(DEV).as_posix()
        for path in sorted([*DEV.rglob("*.ipynb"), *DEV.rglob("*.py")])
        if re.search(r"^\s*(from|import) it_examples", _code(path), flags=re.MULTILINE)
    }
    assert importing == NOT_YET_REPOINTED


def test_the_analysis_points_pin_the_notebook_s_revision():
    notebook = _pins(DEV / "attribution_analysis" / "attribution_analysis.ipynb")
    points = _pins(DEV / "attribution_analysis" / "analysis_points.py")
    assert points == {"PROMPT_ANALYSIS_REVISION": notebook["PROMPT_ANALYSIS_REVISION"]}


@pytest.mark.parametrize("path", sorted(DEV.rglob("*.ipynb")), ids=lambda p: p.stem)
def test_a_notebook_that_loads_helpers_pulls_them_first(path):
    code = _code(path)
    for repo in re.findall(r'load_helpers\(\s*"([^"]+)"', code):
        first_load = re.search(rf'load_helpers\(\s*"{re.escape(repo)}"', code).start()
        pulled = re.search(rf'hub\.pull\(\s*"{re.escape(repo)}", revision=', code)
        assert pulled and pulled.start() < first_load, f"{path.stem} loads helpers of {repo} before pulling them pinned"
        assert 'os.environ["IT_TRUST_REMOTE_CODE"] = "1"' in code[:first_load]
