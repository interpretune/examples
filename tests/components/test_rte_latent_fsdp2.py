"""The RTE latent tuning runs (interpretune#882): meta-built, sharded with FSDP2 and tuned in two stages from the CLI.

The shipped 4B run is launched as its header shows (``torchrun --no-python interpretune``) on two gloo ranks, with a tiny
random Gemma 3 and random transcoders in place of the published ones and without the offload (pinned host memory has no
meaning on CPU). The published runs need the two accelerators their headers name.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RTE = REPO_ROOT / "components" / "rte"
KEY = "rte_latent.gemma3.core.fsdp2"
SIZES = {"4b": "google/gemma-3-4b-it", "27b": "google/gemma-3-27b-it"}
TC_LAYERS = [1, 2]


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("size", SIZES)
class TestTheShippedRuns:
    def test_a_run_names_its_configuration_and_the_plan_the_recipe_needs(self, size):
        run = _load(RTE / "runs" / f"rte_latent.gemma3_{size}.core.fsdp2.yaml")
        assert run["session_ref"] == f"speediedan/rte:{KEY}"
        assert _load(RTE / "it_component.yaml")["module"]["configs"][KEY] == f"configs/{KEY}.yaml"
        plan = run["run_cfg"]["fsdp_plan"]
        assert plan["class:Gemma3DecoderLayer"] == {"act_ckpt": "composable"}
        assert "class:GemmaScopeTranscoder" in plan
        assert run["run_cfg"]["fsdp_default_kwargs"] == {
            "cpu_offload_policy": {"pin_memory": True},
            "mp_policy": {"reduce_dtype": "float32"},
        }

    def test_a_run_tunes_the_model_its_name_says_with_that_models_transcoders(self, size):
        """A run's overrides and the configuration it starts from, merged as the CLI merges them."""
        from interpretune.base.components.cli import ITSessionMixin

        merged = ITSessionMixin._deep_merge(
            _load(RTE / "configs" / f"{KEY}.yaml"), _load(RTE / "runs" / f"rte_latent.gemma3_{size}.core.fsdp2.yaml")
        )
        assert merged["shared_config"]["model_name_or_path"] == SIZES[size]
        latent = merged["registered_cfg"]["module_cfg"]["init_args"]["latent_tuning"]
        assert latent["transcoders"]["repo"] == f"google/gemma-scope-2-{size}-it"
        assert latent["transcoders"]["variant"] == "width_16k_l0_small_affine"


def test_the_configuration_adapts_ten_text_layers_and_no_vision_layer():
    cfg = _load(RTE / "configs" / f"{KEY}.yaml")["registered_cfg"]["module_cfg"]["init_args"]
    pattern = cfg["hf_from_pretrained_cfg"]["init_args"]["lora_cfg"]["target_modules"]

    def adapted(name: str) -> bool:
        return re.fullmatch(pattern, name) is not None

    text = "model.language_model.layers.{}.self_attn.q_proj"
    assert [i for i in range(62) if adapted(text.format(i))] == list(range(10))
    assert adapted("model.language_model.layers.9.mlp.down_proj")
    assert not adapted("model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")


@pytest.fixture
def rte_revision(monkeypatch, tmp_path):
    """This tree's ``rte`` component, published into a components cache of this test's own."""
    from interpretune.hub.components import local_publish

    from tests.support.rte_component import rte_entrypoint_src

    cache = tmp_path / "components"
    monkeypatch.setenv("IT_COMPONENTS_HUB_CACHE", str(cache))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return local_publish(RTE, "speediedan/rte", entrypoint_src=rte_entrypoint_src(), cache_dir=cache)


def _tiny_gemma3(path: Path) -> Path:
    import torch
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration, Gemma3TextConfig, SiglipVisionConfig

    text = Gemma3TextConfig(
        vocab_size=262208,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        query_pre_attn_scalar=16,
    )
    vision = SiglipVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=28,
        patch_size=14,
    )
    cfg = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(), mm_tokens_per_image=1)
    torch.manual_seed(0)
    Gemma3ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(path)
    return path


def _run_config(revision: str, work: Path) -> Path:
    """The shipped 4B run with the tiny model, random transcoders, no offload and a recorder."""
    pinned = f"hub://speediedan/rte@{revision}/"
    text = (RTE / "runs" / "rte_latent.gemma3_4b.core.fsdp2.yaml").read_text(encoding="utf-8")
    cfg = yaml.safe_load(text.replace("hub://speediedan/rte/", pinned))
    cfg["session_ref"] = f"{cfg['session_ref']}@{revision}"
    model = str(_tiny_gemma3(work / "tiny"))
    module = cfg["registered_cfg"]["module_cfg"]["init_args"]
    module["latent_tuning"] = {"layers": TC_LAYERS, "transcoders": {"random": {"d_transcoder": 48}}}
    cfg["shared_config"] = {"model_name_or_path": model, "tokenizer_name": "google/gemma-3-1b-it"}
    cfg["registered_cfg"]["datamodule_cfg"] = {"dataset_path": str(work / "dataset")}
    cfg["run_cfg"]["fsdp_default_kwargs"].pop("cpu_offload_policy")
    cfg["run_cfg"].update(limit_train_batches=2, limit_val_batches=2)
    # a larger step than the recipe's, so that every trained tensor moves within two CPU-sized steps
    module["optimizer_init"] = {"class_path": "torch.optim.AdamW", "init_args": {"weight_decay": 0.0, "lr": 1.0e-03}}
    cfg["run_cfg"]["callbacks"].append(
        {
            "class_path": "tests.support.rte_latent_record.LatentEpochRecorder",
            "init_args": {"out": str(work / "record_rank{rank}.json")},
        }
    )
    path = work / "run.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.mark.hf_live
@pytest.mark.skipif(sys.platform != "linux", reason="two gloo ranks under torch.distributed.run, on Linux only")
def test_the_run_tunes_the_adapters_then_the_transcoders_decoders(rte_revision, tmp_path):
    cli = Path(sys.executable).with_name("interpretune")
    if not cli.exists():
        pytest.skip(f"no `interpretune` console script beside {sys.executable}")
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", "2", "--no-python"]
    cmd += [str(cli), "--run_command", "train", "--config", str(_run_config(rte_revision, tmp_path))]
    env = {k: v for k, v in os.environ.items() if k not in ("WORLD_SIZE", "RANK", "LOCAL_RANK")}
    # each rank imports the recorder from `tests`, which a subprocess finds only through PYTHONPATH
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p)
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=1500, check=False)
    assert proc.returncode == 0, f"torchrun failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-6000:]}"
    records = [json.loads((tmp_path / f"record_rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert records[0] == records[1]
    record = records[0]

    layers = [n for n in record["units"] if re.fullmatch(r".*language_model\.layers\.\d+", n)]
    transcoders = [n for n in record["units"] if n.startswith("latent_models.")]
    assert len(layers) == 4 and transcoders == [f"latent_models.tc{layer}" for layer in TC_LAYERS]
    assert record["checkpointed"] == layers
    assert record["types"] == ["DTensor"]
    assert record["reduce_dtypes"] == ["torch.float32"]
    fp32 = {n for n, dtype in record["dtypes"].items() if dtype == "torch.float32"}
    assert fp32 == {n for n in record["dtypes"] if "lora_" in n or n.startswith("latent_models.")}

    adapters = {n for n in record["dtypes"] if "lora_" in n}
    decoders = {f"latent_models.tc{layer}.{w}" for layer in TC_LAYERS for w in ("w_dec", "b_dec")}
    first, second = record["epochs"]
    assert (first["depth"], second["depth"]) == (0, 1)
    assert adapters and set(first["trainable"]) == adapters
    assert set(second["trainable"]) == adapters | decoders
    # What the schedule thawed is what trained. The tiny model's transcoders replace MLPs that carry adapters (the
    # published layers do not overlap), and a replaced MLP's adapters reach no loss, so they alone stay put.
    replaced = {n for n in adapters if re.search(rf"layers\.({'|'.join(map(str, TC_LAYERS))})\.mlp\.", n)}
    assert len(replaced) == 12
    assert set(first["changed"]) == adapters - replaced
    assert set(second["changed"]) == (adapters - replaced) | decoders
