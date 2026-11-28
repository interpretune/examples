"""Collection count guard for the nb_experiments split move (umbrella ruling on design v3 §11.6).

The research tree moved to ``it_examples/experiments/notebook`` and its embedded tests to this root;
the ruling requires the relocated tests' collected count pinned so the move (and any future
restructuring) provably drops zero tests rather than silently retiring them.
"""

from __future__ import annotations

from pathlib import Path


def test_relocated_notebook_test_count_pinned():
    import subprocess
    import sys

    root = Path(__file__).parents[3]
    # the repository's one test root, as its CI invokes it
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:randomly"],
        capture_output=True,
        text=True,
        cwd=root,
    )
    prefix = "tests/tutorials/notebook/"
    collected = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(prefix) and "::" in line and "test_collection_guard" not in line
    ]
    # 5 local-graph-cleanup tests: the concept-direction experiment tests moved out with
    # src/it_examples/experiments/ (the experiment is Hub-published now), leaving only the
    # graph-cleanup tests under this root. This guard file's own additions excluded.
    assert len(collected) == 5, (
        f"relocated notebook test count changed: {len(collected)} != 5 — if deliberate, update this pin; "
        f"if not, a move or conftest change silently dropped tests.\n" + "\n".join(collected[:10])
    )
