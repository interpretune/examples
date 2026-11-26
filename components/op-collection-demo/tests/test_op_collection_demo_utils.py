"""CPU tests for the op collections tutorial's helpers and the two collections it stages."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import op_collection_demo_utils as demo
from op_collection_demo_utils import (
    categorize_operations,
    cleanup_op_collections,
    compare_operation_outputs,
    generate_test_batches,
    setup_hub_op_collection,
    setup_local_op_collection,
)

# no `resolve()`: a pulled snapshot's files are symlinks, and the collections are beside the link
COMPONENT = Path(__file__).parents[1]


def _op(name, source="bundled", composition=None):
    return SimpleNamespace(name=name, source=source, composition=composition, description=name)


def test_operations_are_categorized_by_their_declared_source():
    hub, local = _op("a.b.hub_op", "hub:a.b"), _op("local_op", "local")
    composed = _op("composed", "bundled", composition=["x", "y"])
    ops = {"a.b.hub_op": hub, "hub_op": hub, "local_op": local, "composed": composed, "plain.dotted": _op("plain.dotted")}
    canonical, aliases, hub_ops, local_ops, composed_ops, builtin = categorize_operations(ops)
    assert set(canonical) == {"a.b.hub_op", "local_op", "composed", "plain.dotted"}
    assert aliases["a.b.hub_op"] == ["hub_op"]
    assert set(hub_ops) == {"a.b.hub_op"} and set(local_ops) == {"local_op"} and set(composed_ops) == {"composed"}
    assert builtin == {"composed", "plain.dotted"}  # a dot in the name is not provenance


def test_the_local_collection_is_staged_on_the_op_path_and_cleaned_up(tmp_path, monkeypatch):
    staged = tmp_path / "local_op_collection"
    monkeypatch.setattr(demo, "IT_ANALYSIS_OP_PATHS", [])
    monkeypatch.setenv("IT_ANALYSIS_OP_PATHS", "/elsewhere")
    original, new = setup_local_op_collection(COMPONENT / "local_op_collection", staged)
    assert original == "/elsewhere" and new == f"/elsewhere:{staged}" == os.environ["IT_ANALYSIS_OP_PATHS"]
    assert demo.IT_ANALYSIS_OP_PATHS == [str(staged)]
    assert (staged / "local_op_collection.yaml").is_file() and (staged / "local_op_definitions.py").is_file()
    cleanup_op_collections(tmp_path / "absent", staged, original)
    assert not staged.exists() and os.environ["IT_ANALYSIS_OP_PATHS"] == "/elsewhere"
    assert demo.IT_ANALYSIS_OP_PATHS == []


def test_staging_the_hub_collection_leaves_bytecode_behind_and_refuses_a_missing_source(tmp_path):
    source = tmp_path / "source"
    (source / "__pycache__").mkdir(parents=True)
    (source / "__pycache__" / "x.pyc").write_bytes(b"")
    for path in (COMPONENT / "hub_op_collection").glob("*.*"):
        (source / path.name).write_bytes(path.read_bytes())
    setup_hub_op_collection(source, tmp_path / "staged")
    assert {p.name for p in (tmp_path / "staged").iterdir()} == {
        "hub_op_collection.yaml",
        "hub_op_definitions.py",
        "it_component.yaml",
    }
    with pytest.raises(FileNotFoundError, match="not found"):
        setup_hub_op_collection(tmp_path / "absent", tmp_path / "other")


def test_the_hub_collections_manifest_lists_its_definitions_and_the_implementations_exist():
    """The tutorial publishes this directory, so its manifest must name every definition file."""
    for directory in ("hub_op_collection", "local_op_collection"):
        root = COMPONENT / directory
        definitions = yaml.safe_load((root / f"{directory}.yaml").read_text())
        ops = [d for d in definitions.values() if isinstance(d, dict) and "implementation" in d]
        assert ops
        for op in ops:
            module, _, function = op["implementation"].rpartition(".")
            assert f"def {function}(" in (root / f"{module}.py").read_text()
    manifest = yaml.safe_load((COMPONENT / "hub_op_collection" / "it_component.yaml").read_text())
    assert manifest["ops"]["files"] == ["hub_op_collection.yaml"]


def test_paired_test_batches_hold_equal_separate_labels():
    pairs = list(generate_test_batches(3))
    assert len(pairs) == 3
    for _, separate, pipeline in pairs:
        assert separate.orig_labels is not pipeline.orig_labels
        torch.testing.assert_close(separate.orig_labels, pipeline.orig_labels)


def test_a_mismatched_pair_of_outputs_is_reported():
    assert compare_operation_outputs([{"a": 1}], [{"a": 1}]) is True
    assert compare_operation_outputs([{"a": 1}], [{"a": 2}]) is False
