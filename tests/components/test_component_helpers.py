"""Each first-party component's helper modules load as a notebook loads them: through ``it.hub.load_helpers``.

The components' own tests import a helper by its file name. This is the other half: the manifest declares it, a local
publish carries it, and the loader imports it behind the remote-code opt-in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import interpretune as it
from interpretune.hub.components import local_publish
from interpretune.hub.trust import RemoteCodeNotTrustedError

COMPONENTS = Path(__file__).resolve().parents[2] / "components"


def _manifest(component: Path) -> dict:
    return yaml.safe_load((component / "it_component.yaml").read_text())


def _declared() -> list[tuple[str, str]]:
    return [
        (path.parent.name, name)
        for path in sorted(COMPONENTS.glob("*/it_component.yaml"))
        for name in (_manifest(path.parent).get("helpers") or {}).get("modules", {})
    ]


def _published(component: Path, cache: Path) -> str:
    entrypoint = (_manifest(component).get("module") or {}).get("entrypoint")
    kwargs = {"entrypoint_src": component / entrypoint} if entrypoint else {}
    return local_publish(component, f"local/{component.name}", cache_dir=cache, **kwargs)


def test_the_re_homed_components_declare_helpers():
    assert {name for name, _ in _declared()} >= {
        "bsf-blocks",
        "concept-direction",
        "neuron-communities",
        "neuron-finder",
        "op-collection-demo",
        "prompt-analysis",
        "sasa-blocks",
    }


@pytest.mark.parametrize("component, module", _declared())
def test_a_declared_helper_loads_from_a_published_tree(component, module, tmp_path, monkeypatch):
    revision = _published(COMPONENTS / component, tmp_path)
    monkeypatch.delenv("IT_TRUST_REMOTE_CODE", raising=False)
    with pytest.raises(RemoteCodeNotTrustedError):
        it.hub.load_helpers(f"local/{component}", module, revision=revision, cache_dir=tmp_path)
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    loaded = it.hub.load_helpers(f"local/{component}", module, revision=revision, cache_dir=tmp_path)
    assert loaded.__name__.rpartition(".")[2] == module
    assert str(tmp_path) in str(loaded.__file__)  # the published copy, not this repository's


@pytest.mark.parametrize("component", sorted({name for name, _ in _declared()}))
def test_every_helper_module_in_the_source_is_declared(component):
    """A module a component's tests import but its manifest does not declare would pass here and be missing after a
    pull."""
    manifest = _manifest(COMPONENTS / component)
    entrypoint = (manifest.get("module") or {}).get("entrypoint")
    in_source = {p.name for p in (COMPONENTS / component).glob("*.py")} - {entrypoint}
    assert in_source == set(manifest["helpers"]["modules"].values())


@pytest.mark.parametrize("component", sorted({name for name, _ in _declared()}))
def test_every_test_in_the_source_is_published(component):
    shipped = set(_manifest(COMPONENTS / component).get("extra_files") or [])
    assert {f"tests/{p.name}" for p in (COMPONENTS / component).glob("tests/test_*.py")} <= shipped


def test_a_declared_helper_finds_its_extra_files_beside_itself(tmp_path, monkeypatch):
    """The op collections tutorial reads its demo collections from the pulled snapshot, next to the helper module."""
    revision = _published(COMPONENTS / "op-collection-demo", tmp_path)
    monkeypatch.setenv("IT_TRUST_REMOTE_CODE", "1")
    demo = it.hub.load_helpers(
        "local/op-collection-demo", "op_collection_demo_utils", revision=revision, cache_dir=tmp_path
    )
    beside = Path(demo.__file__).parent
    assert (beside / "hub_op_collection" / "it_component.yaml").is_file()
    assert (beside / "local_op_collection" / "local_op_collection.yaml").is_file()
