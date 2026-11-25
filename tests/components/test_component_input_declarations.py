"""A component's configurations declare batches their model can be called with.

A configuration that pads on the left yields rows of padding and text. Without the attention mask in the batch, a direct
forward (what a training loop runs) reads the padding as text and returns logits for it, with no error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPONENTS = Path(__file__).parents[2] / "components"
CONFIGS = sorted(COMPONENTS.glob("*/configs/*.yaml"))


def _declaration(path: Path) -> tuple[dict, dict, set[str]] | None:
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    tokenizer = (doc.get("shared_config") or {}).get("tokenizer_kwargs")
    datamodule = (doc.get("registered_cfg") or {}).get("datamodule_cfg") or doc.get("datamodule_cfg")
    if not tokenizer or not datamodule or "model_input_names" not in tokenizer:
        return None
    combinations = (doc.get("reg_info") or {}).get("adapter_combinations") or []
    return tokenizer, datamodule, {adapter for combination in combinations for adapter in combination}


CHECKED = [p for p in CONFIGS if _declaration(p)]


def test_the_declarations_found_are_not_a_subset_of_one_kind():
    """Positive control: the parametrization below covers bridge-keyed and Hugging Face-keyed configurations."""
    keys = {_declaration(p)[0]["model_input_names"][0] for p in CHECKED}  # type: ignore[index]
    assert keys == {"input", "input_ids"}
    assert len(CHECKED) >= 20


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_a_left_padding_configuration_yields_the_mask(path):
    tokenizer, datamodule, _ = _declaration(path)  # type: ignore[misc]
    if tokenizer.get("padding_side") != "left":
        pytest.skip("this configuration does not pad on the left")
    names = tokenizer["model_input_names"]
    assert "attention_mask" in names, f"{path.name} pads on the left but yields no attention mask: {names}"
    assert set(names) <= set(datamodule["signature_columns"]), (names, datamodule["signature_columns"])


@pytest.mark.parametrize("path", CHECKED, ids=lambda p: f"{p.parent.parent.name}/{p.stem}")
def test_an_nnsight_composition_declares_the_hugging_face_key(path):
    tokenizer, _, composition = _declaration(path)  # type: ignore[misc]
    if "nnsight" not in composition:
        pytest.skip("not an nnsight composition")
    assert tokenizer["model_input_names"][0] == "input_ids"
