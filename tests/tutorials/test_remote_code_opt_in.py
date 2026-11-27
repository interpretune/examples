"""Every tutorial opts in to remote code before its first call that executes Hub-resident code.

``tests/conftest.py`` sets ``IT_TRUST_REMOTE_CODE`` for the whole session, so a notebook test runs with the opt-in
already in place and cannot see a load that precedes it. A reader's first run, or a render in a fresh process, does
not have it, and stops with ``RemoteCodeNotTrustedError`` at that load. This check reads each notebook's code cells in
order instead: a call into ``interpretune.hub`` that can execute Hub-resident code must follow a statement that sets
the opt-in, and a ``interpretune.hub`` call this module has not classified is refused by name rather than assumed safe.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL_TREES = (REPO_ROOT / "tutorials" / "dev", REPO_ROOT / "tutorials" / "publish")
OPT_IN_VAR = "IT_TRUST_REMOTE_CODE"
AFFIRMATIVE = {"1", "true", "yes", "on"}

# interpretune.hub functions that import or execute code published in a Hub repository (an entrypoint, a helper module,
# an adapter component), each of which refuses without the opt-in
EXECUTES_HUB_CODE = frozenset(
    {
        "load",
        "load_datamodule",
        "load_helpers",
        "load_hub_adapter",
        "load_run_hub_adapters",
        "import_cached_entrypoint",
        "resolve_prompt_config_class",
        "instantiate_prompt_cfg_node",
    }
)
# interpretune.hub names that fetch, describe or resolve without executing Hub-resident code. Op collections are
# here because op discovery degrades (skips the collection with a warning) rather than refusing.
EXECUTES_NOTHING = frozenset(
    {
        "pull",
        "pull_experiment",
        "load_experiment",
        "pull_ops",
        "op_info",
        "prefer_ops",
        "push_analysis_store",
        "pull_analysis_store",
        "describe_analysis_store",
        "subject_from_module",
        "hub_presence",
        "HubAnalysisOpManager",
    }
)


@dataclass(frozen=True)
class Finding:
    """One problem in one notebook, at a code cell and line."""

    cell: int
    line: int
    message: str

    def __str__(self) -> str:
        return f"cell {self.cell}, line {self.line}: {self.message}"


def _cell_sources(notebook: dict) -> list[tuple[int, str]]:
    """``(cell index, source)`` for each code cell, in notebook order."""
    out = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        text = "".join(source) if isinstance(source, list) else source
        out.append((index, text))
    return out


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _environ_key(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == OPT_IN_VAR


def _trusts(value: ast.AST) -> bool:
    return isinstance(value, ast.Constant) and isinstance(value.value, str) and value.value.lower() in AFFIRMATIVE


class _CellReader(ast.NodeVisitor):
    """Walks one cell in source order, updating the shared alias map and trust state, and recording findings."""

    def __init__(self, cell: int, aliases: dict[str, str], state: dict[str, bool | None], findings: list[Finding]):
        self.cell, self.aliases, self.state, self.findings = cell, aliases, state, findings
        self.calls_executing = 0

    def _hub_name(self, func: ast.AST) -> str | None:
        """The ``interpretune.hub`` function a call names, through this notebook's import aliases, or ``None``."""
        dotted = _dotted(func)
        if dotted is None:
            return None
        head, _, rest = dotted.partition(".")
        resolved = self.aliases.get(head)
        if resolved is None:
            return None
        full = f"{resolved}.{rest}" if rest else resolved
        if not full.startswith("interpretune.hub.") and full != "interpretune.hub":
            return None
        return full.rsplit(".", 1)[-1]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "interpretune" or alias.name.startswith("interpretune."):
                if alias.asname:
                    self.aliases[alias.asname] = alias.name
                else:
                    self.aliases["interpretune"] = "interpretune"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if not node.module or node.level:
            return
        for alias in node.names:
            local = alias.asname or alias.name
            if node.module == "interpretune" and alias.name == "hub":
                self.aliases[local] = "interpretune.hub"
            elif node.module == "interpretune.hub" or node.module.startswith("interpretune.hub."):
                self.aliases[local] = f"interpretune.hub.{alias.name}"

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for target in node.targets:
            if (
                isinstance(target, ast.Subscript)
                and _dotted(target.value) == "os.environ"
                and _environ_key(target.slice)
            ):
                self.state["trusted"] = _trusts(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)  # arguments run before the call
        if _dotted(node.func) == "os.environ.setdefault" and node.args and _environ_key(node.args[0]):
            if self.state["trusted"] is None and len(node.args) > 1:
                self.state["trusted"] = _trusts(node.args[1])
            return
        name = self._hub_name(node.func)
        if name is None:
            return
        if name in EXECUTES_HUB_CODE:
            self.calls_executing += 1
            if self.state["trusted"] is not True:
                self.findings.append(
                    Finding(
                        self.cell,
                        node.lineno,
                        f"interpretune.hub.{name} executes Hub-resident code, and no earlier statement sets "
                        f'os.environ["{OPT_IN_VAR}"] = "1": pull the repository first (it executes nothing), then opt '
                        "in, as the hub trust posture page describes",
                    )
                )
        elif name not in EXECUTES_NOTHING:
            self.findings.append(
                Finding(
                    self.cell,
                    node.lineno,
                    f"interpretune.hub.{name} is not classified here: add it to EXECUTES_HUB_CODE if it can import or "
                    "execute code from a Hub repository, else to EXECUTES_NOTHING",
                )
            )


def read_notebook(notebook: dict) -> tuple[list[Finding], int]:
    """The ordering findings for one notebook, and how many calls that execute Hub-resident code it makes."""
    aliases: dict[str, str] = {}
    state: dict[str, bool | None] = {"trusted": None}
    findings: list[Finding] = []
    executing = 0
    for index, text in _cell_sources(notebook):
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("%env") and OPT_IN_VAR in stripped:
                _, _, assignment = stripped.partition(OPT_IN_VAR)
                value = assignment.lstrip("= ").strip().strip("'\"").lower()
                # `%env` sets the variable when the line runs; record it as a plain assignment at this position
                lines.append(f'os.environ["{OPT_IN_VAR}"] = {value!r}')
            elif stripped.startswith(("%", "!", "?")):
                lines.append("")
            else:
                lines.append(line)
        try:
            tree = ast.parse("\n".join(lines))
        except SyntaxError as err:
            findings.append(Finding(index, err.lineno or 0, f"cell does not parse, so its order cannot be read: {err}"))
            continue
        reader = _CellReader(index, aliases, state, findings)
        for statement in tree.body:
            reader.visit(statement)
        executing += reader.calls_executing
    return findings, executing


def _tutorials() -> list[Path]:
    return sorted(p for tree in TUTORIAL_TREES for p in tree.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def test_every_tutorial_opts_in_before_executing_hub_code():
    notebooks = _tutorials()
    assert notebooks, f"no tutorials found under {[str(t) for t in TUTORIAL_TREES]}; this check would pass vacuously"
    problems: dict[str, list[str]] = {}
    executing_notebooks = 0
    for path in notebooks:
        findings, executing = read_notebook(json.loads(path.read_text(encoding="utf-8")))
        executing_notebooks += bool(executing)
        if findings:
            problems[str(path.relative_to(REPO_ROOT))] = [str(f) for f in findings]
    assert not problems, json.dumps(problems, indent=2)
    # the tutorials load components with entrypoints; finding none means the call matching broke, not that none exist
    assert executing_notebooks >= len(notebooks) // 2, (
        f"only {executing_notebooks} of {len(notebooks)} tutorials make a call that executes Hub-resident code"
    )


def _nb(*cells: str) -> dict:
    return {"cells": [{"cell_type": "code", "source": src} for src in cells]}


class TestReaderPositiveControls:
    """The reader fails on each ordering defect it exists to catch."""

    def test_a_load_before_the_opt_in_is_found(self):
        findings, executing = read_notebook(
            _nb(
                "import os\nimport interpretune as it",
                'cfg = it.hub.load("speediedan/prompt-analysis", "k")',
                'os.environ["IT_TRUST_REMOTE_CODE"] = "1"',
            )
        )
        assert executing == 1
        assert [f.cell for f in findings] == [1]

    def test_a_pull_then_opt_in_then_load_passes(self):
        findings, executing = read_notebook(
            _nb(
                "import os\nimport interpretune as it",
                'it.hub.pull("speediedan/rte")\nos.environ["IT_TRUST_REMOTE_CODE"] = "1"',
                'cfg = it.hub.load("speediedan/rte", "k")',
            )
        )
        assert (findings, executing) == ([], 1)

    def test_a_load_on_the_line_before_the_opt_in_in_one_cell_is_found(self):
        findings, _ = read_notebook(
            _nb('import os\nimport interpretune as it\nit.hub.load("r", "k")\nos.environ["IT_TRUST_REMOTE_CODE"] = "1"')
        )
        assert [(f.cell, f.line) for f in findings] == [(0, 3)]

    def test_an_opt_out_after_the_opt_in_is_found(self):
        findings, _ = read_notebook(
            _nb(
                'import os\nimport interpretune as it\nos.environ["IT_TRUST_REMOTE_CODE"] = "1"',
                'os.environ["IT_TRUST_REMOTE_CODE"] = "0"\nit.hub.load_helpers("r", "h")',
            )
        )
        assert len(findings) == 1

    def test_setdefault_and_env_magic_count_as_an_opt_in(self):
        for opt_in in ('os.environ.setdefault("IT_TRUST_REMOTE_CODE", "1")', "%env IT_TRUST_REMOTE_CODE=1"):
            findings, executing = read_notebook(
                _nb("import os\nimport interpretune as it", opt_in, 'it.hub.load("r", "k")')
            )
            assert (findings, executing) == ([], 1), opt_in

    def test_a_call_through_a_from_import_is_found(self):
        findings, executing = read_notebook(
            _nb(
                "from interpretune.hub.promptconfigs import resolve_prompt_config_class as rpc",
                'rpc("speediedan/prompt-analysis:PromptSetConfig")',
            )
        )
        assert executing == 1 and len(findings) == 1

    def test_an_unclassified_hub_call_is_refused_by_name(self):
        findings, _ = read_notebook(_nb("import interpretune as it", 'it.hub.some_new_loader("r")'))
        assert len(findings) == 1 and "some_new_loader is not classified" in findings[0].message

    @pytest.mark.parametrize("name", sorted(EXECUTES_HUB_CODE | EXECUTES_NOTHING))
    def test_every_classified_name_exists_in_interpretune_hub(self, name):
        """A misspelled or retired name classifies nothing, so every entry must be a real ``interpretune.hub`` name."""
        import interpretune.hub as hub
        import interpretune.hub.promptconfigs as promptconfigs

        assert hasattr(hub, name) or hasattr(promptconfigs, name), name
