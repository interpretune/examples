"""The experiment reads the models it supports from its own ``model_specs.yaml``, not from interpretune's bundled file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from interpretune.harness import sessions

# not resolved: in a pulled snapshot this file is a link into the cache's blob store, away from its siblings
COMPONENT = Path(__file__).parents[1]
MANIFEST = yaml.safe_load((COMPONENT / "it_component.yaml").read_text(encoding="utf-8"))
CONFIGS = COMPONENT / "concept_direction" / "configs"


def _selected_models() -> set[tuple[str, str]]:
    """The ``(family, variant)`` each shipped configuration that sets a MODEL selects."""
    found = set()
    for path in CONFIGS.glob("*.yaml"):
        model = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("MODEL") or {}
        if "family" in model and "variant" in model:
            found.add((model["family"], model["variant"]))
    return found


@pytest.fixture
def no_bundled_specs(monkeypatch):
    """Refuse a read of interpretune's bundled specs, so a lookup that falls back to them fails here."""

    def refuse():
        raise AssertionError("the bundled model specs were read")

    monkeypatch.setattr(sessions, "_hub_model_spec_registry", refuse)


def test_some_configuration_selects_a_model():
    assert _selected_models()  # an empty set would make the check below pass on nothing


def test_every_selected_model_is_in_the_shipped_specs(no_bundled_specs):
    for family, variant in _selected_models():
        spec = sessions.resolve_model_spec(family, variant, model_specs=COMPONENT / "model_specs.yaml")
        assert spec.composition == ("core", "nnsight", "circuit_tracer")


def test_the_session_kwargs_name_the_shipped_specs():
    from exp.concept_direction import MODEL_SPECS_PATH, NotebookHarnessConfig

    assert MODEL_SPECS_PATH == COMPONENT / "model_specs.yaml"
    assert MODEL_SPECS_PATH.is_file()
    # session_kwargs only reads attributes, so a bare instance shows what every session call is given
    cfg = object.__new__(NotebookHarnessConfig)
    fields = NotebookHarnessConfig.__dataclass_fields__
    for name in fields:
        object.__setattr__(cfg, name, None)
    assert cfg.session_kwargs["model_specs"] == MODEL_SPECS_PATH


def test_the_harness_configuration_resolves_from_the_shipped_specs(no_bundled_specs, monkeypatch, tmp_path):
    from exp import concept_direction

    class Stop(Exception):
        """Ends the build at the lookup: what follows it probes the local Neuronpedia stack."""

    seen = {}

    def record(family, variant, **kwargs):
        seen[(family, variant)] = kwargs.get("model_specs")
        raise Stop

    monkeypatch.setattr(concept_direction, "resolve_model_spec", record)
    monkeypatch.setattr(concept_direction, "create_work_root", lambda *args, **kwargs: tmp_path)
    with pytest.raises(Stop):
        concept_direction.build_notebook_harness_config(CONFIGS / "gemma3_1b_it_local_color_fruit_orange.yaml")
    assert seen == {("gemma3", "1b_it"): concept_direction.MODEL_SPECS_PATH}


def test_a_spec_without_an_entry_is_refused_naming_the_shipped_file(no_bundled_specs):
    with pytest.raises(ValueError, match="model_specs.yaml"):
        sessions.resolve_model_spec("nope", "nothing", model_specs=COMPONENT / "model_specs.yaml")


def test_every_experiment_entry_ships_the_specs():
    entries = MANIFEST["experiments"]
    assert entries
    assert [key for key, entry in entries.items() if "model_specs.yaml" not in entry["files"]] == []
