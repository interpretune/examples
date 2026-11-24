"""The ``speediedan/prompt-analysis`` component's model-only ``[core, nnsight]`` configuration, published locally.

The configuration names a gated 8B model, so the session is built from it with a tiny random Llama in its place: what is
checked is everything else the configuration says, up to an intervention through the model backend the session supplies.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

import interpretune as it
from interpretune.analysis.backends import get_model_backend
from interpretune.analysis.venngram import unit_interventions
from interpretune.hub import components

COMPONENT_DIR = Path(__file__).parent.parent.parent / "components" / "prompt-analysis"
REPO_ID = "speediedan/prompt-analysis"
KEY = "prompt_analysis.llama3.nnsight"
POINT = "mlp.out.hook_in"  # an MLP's down-projection input, whose units are its neurons
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
CONFIG_KEYS = sorted(yaml.safe_load((COMPONENT_DIR / "it_component.yaml").read_text())["module"]["configs"])


@pytest.fixture
def prompt_analysis_cache(tmp_path) -> Path:
    cache = tmp_path / "components"
    components.local_publish(
        COMPONENT_DIR, REPO_ID, entrypoint_src=COMPONENT_DIR / "prompt_analysis.py", cache_dir=cache
    )
    return cache


def test_the_manifest_names_exactly_the_configurations_shipped():
    assert CONFIG_KEYS == sorted(p.stem for p in (COMPONENT_DIR / "configs").glob("*.yaml"))


def test_the_llama_configuration_is_the_model_alone(prompt_analysis_cache):
    dm_cfg, m_cfg, dm_cls, m_cls = it.hub.load(REPO_ID, KEY, cache_dir=prompt_analysis_cache)
    assert (dm_cls.__name__, m_cls.__name__) == ("PromptDataModule", "ReadoutModule")
    assert dm_cfg.prompt_cfg.prompts == []
    assert getattr(m_cfg, "circuit_tracer_cfg", None) is None
    # what the neuron-communities notebook built by hand: the 8B checkpoint, in bf16, on the GPU, loaded at once
    nnsight_cfg = m_cfg.nnsight_cfg
    assert (m_cfg.model_name_or_path, nnsight_cfg.model_name) == ("meta-llama/Llama-3.1-8B",) * 2
    assert (nnsight_cfg.resolved_dtype, nnsight_cfg.device_map, nnsight_cfg.dispatch) == (torch.bfloat16, "cuda", True)
    assert nnsight_cfg.attn_implementation is None  # the transformers default, as the notebook's own load had


def test_the_session_supplies_the_model_and_backend_an_intervention_needs(prompt_analysis_cache):
    dm_cfg, m_cfg, dm_cls, m_cls = it.hub.load(REPO_ID, KEY, cache_dir=prompt_analysis_cache)
    for cfg in (dm_cfg, m_cfg, m_cfg.nnsight_cfg):
        for field in ("model_name_or_path", "model_name", "tokenizer_name"):
            if getattr(cfg, field, None) is not None:
                setattr(cfg, field, TINY_LLAMA)
    m_cfg.nnsight_cfg.device_map = "cpu"
    dm_cfg.tokenizer_id_overrides = {"pad_token_id": 0}  # the configured id is the 8B tokenizer's
    session = it.ITSession(
        it.ITSessionConfig(
            adapter_ctx=(it.Adapter.core, it.Adapter.nnsight),
            datamodule_cfg=dm_cfg,
            module_cfg=m_cfg,
            datamodule_cls=dm_cls,
            module_cls=m_cls,
        )
    )
    it.it_init(**session)
    model, backend = session.module.model, get_model_backend(session.module)
    assert backend is not None

    hf_model = model._module
    layer, units, width = 1, [0, 3], hf_model.config.intermediate_size
    payload = unit_interventions(
        "@edited",
        sets={"edited": {"point": POINT, "layers": [layer], "units": {layer: units}}},
        n_layers=hf_model.config.num_hidden_layers,
        site_widths={POINT: width},
        op="replace",
        value=0.0,
        position_scope="last_token",
    )
    batch = {"input_ids": torch.tensor([[1, 5, 9, 12]])}
    clean, edited = backend.fwd_w_intervention(model, batch, payload)

    def zero_units(_module, inputs):
        hidden = inputs[0].clone()
        hidden[:, -1, units] = 0.0
        return (hidden, *inputs[1:])

    handle = hf_model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(zero_units)
    try:
        by_hand = hf_model(**batch).logits
    finally:
        handle.remove()
    assert torch.allclose(edited[:, -1], by_hand[:, -1], atol=1e-5)
    assert not torch.allclose(clean[:, -1], edited[:, -1], atol=1e-5), "the edit moved nothing, so it was not tested"
