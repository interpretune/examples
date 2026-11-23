"""Run each first-party component's own tests, one pytest process per component.

A component's ``tests/`` import its payload modules by their top-level names (``exp``, ``jlens_ops``), with the
component directory itself on ``sys.path``. Two components may ship a package of the same name (``exp``), so they
cannot share one interpreter: each runs in its own process, from its own directory.

``--published-layout`` runs each component the way a user, or the catalog smoke, meets it after a pull: staged as the
Hub cache serves a snapshot (every file a symlink into a blob store) with every interpretune and Hub cache empty. A test
that finds its siblings through ``Path(__file__).resolve()``, or reads something only an earlier run left cached, passes
in-tree and fails there, so CI runs this form and the defect surfaces before a publish rather than after.

Usage: ``python scripts/run_component_tests.py [--root DIR] [--published-layout] [NAME ...] [-- PYTEST_ARGS ...]``.
``--root`` is the directory holding ``components/`` (default: this repository); naming components runs only those, and a
named component without a ``tests/`` directory is refused. Exits non-zero naming every component whose tests failed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def component_test_dirs(root: Path) -> dict[str, Path]:
    """Each component under ``root/components`` that ships a ``tests/`` directory, by name."""
    return {p.parent.name: p.parent for p in sorted((root / "components").glob("*/tests")) if p.is_dir()}


def empty_caches(root: Path) -> dict[str, str]:
    """Environment pointing every interpretune and Hub cache under ``root``, which is empty."""
    return {
        "IT_CACHE": str(root / "it"),
        "IT_COMPONENTS_HUB_CACHE": str(root / "it" / "hub" / "components"),
        "IT_ARTIFACTS_HUB_CACHE": str(root / "it" / "hub" / "artifacts"),
        "IT_ANALYSIS_HUB_CACHE": str(root / "it" / "hub" / "analysis"),
        "HF_HUB_CACHE": str(root / "hf"),
        # a component's own code runs only under the explicit opt-in, which a component's tests give on purpose
        "IT_TRUST_REMOTE_CODE": "1",
    }


def stage_published_layout(component_dir: Path, dest: Path) -> Path:
    """Stage ``component_dir`` as the Hub cache serves a snapshot: each file a symlink from the snapshot into blobs."""
    snapshot = dest / "snapshots" / "staged"
    for path in sorted(component_dir.rglob("*")):
        if path.is_dir() or "__pycache__" in path.parts:
            continue
        data = path.read_bytes()
        blob = dest / "blobs" / hashlib.sha256(data).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        link = snapshot / path.relative_to(component_dir)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snapshot


def run_published_layout(component_dir: Path, pytest_args: list[str] | None = None) -> int:
    """Run ``component_dir``'s tests from a staged Hub-shaped snapshot of it, with every cache empty."""
    with tempfile.TemporaryDirectory(prefix=f"published-{component_dir.name}-") as tmp:
        snapshot = stage_published_layout(component_dir, Path(tmp) / "component")
        print(f"\n=== {component_dir.name}: staged as a published snapshot at {snapshot}", flush=True)
        return run_tests(snapshot, pytest_args, extra_env=empty_caches(Path(tmp) / "caches"))


def run_tests(
    component_dir: Path,
    pytest_args: list[str] | None = None,
    tests: str = "tests",
    extra_env: dict[str, str] | None = None,
) -> int:
    """Run ``component_dir/<tests>`` in a fresh interpreter whose import path starts at ``component_dir``.

    Used for an in-tree component and for a pulled Hub snapshot alike, so a catalog smoke that names a component's own
    tests runs them exactly as the components tier does.
    """
    env = {**os.environ, **(extra_env or {})}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(component_dir), env.get("PYTHONPATH", "")) if p)
    # a component's tests import its own entrypoints, which interpretune executes only under the explicit opt-in
    env.setdefault("IT_TRUST_REMOTE_CODE", "1")
    cmd = [sys.executable, "-m", "pytest", tests, "-v", "-p", "no:cacheprovider", *(pytest_args or [])]
    print(f"\n=== {component_dir.name}: {' '.join(cmd[1:])} (in {component_dir})", flush=True)
    return subprocess.run(cmd, cwd=component_dir, env=env, check=False).returncode


def main(argv: list[str]) -> int:
    """Run the selected components' tests; report every failure by name."""
    pytest_args: list[str] = []
    if "--" in argv:
        argv, pytest_args = argv[: argv.index("--")], argv[argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="the directory holding components/")
    parser.add_argument(
        "--published-layout", action="store_true", help="run from a Hub-shaped snapshot with every cache empty"
    )
    parser.add_argument("names", nargs="*", help="components to run (default: every one with a tests/ directory)")
    args = parser.parse_args(argv)
    found = component_test_dirs(args.root)
    if not found:
        print(f"no components/*/tests under {args.root}", file=sys.stderr)
        return 1
    unknown = [n for n in args.names if n not in found]
    if unknown:
        print(
            f"no tests/ directory for {', '.join(unknown)} (components with tests: {', '.join(found)})", file=sys.stderr
        )
        return 2
    selected = {n: found[n] for n in args.names} if args.names else found
    run = run_published_layout if args.published_layout else run_tests
    failed = [name for name, path in selected.items() if run(path, pytest_args) != 0]
    print()
    for name in selected:
        print(f"{'FAILED' if name in failed else 'passed'}: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
