"""This collection, validated by interpretune's op-collection conformance suite from its own working tree.

The suite reads the ops through the dispatcher exactly as a session would: every declared capability, mode and scope
must be a known axis, and against the target each op's requirements are either met or refused by name before
anything runs. The working tree is staged with ``stage_local_collection``, so what is validated is what the next
publish will upload, not the revision currently on the Hub.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch

from interpretune.testing.conformance import ConformanceTarget, OpCollectionConformance, stage_local_collection

pytest_plugins = ["interpretune.testing.conformance.plugin"]

# The conformance target builds its session from interpretune's published `speediedan/rte` example component, whose
# entrypoint is hub-resident code; interpretune executes such code only on an explicit opt-in, which this states.
os.environ.setdefault("IT_TRUST_REMOTE_CODE", "1")

# absolute(), not resolve(): in a pulled Hub snapshot this file is a symlink into the blob store, and resolving it
# would leave the snapshot, whose siblings are what this needs
COLLECTION_DIR = Path(__file__).absolute().parents[1]


def _synthetic_lens(d_model: int = 768, layers: tuple[int, ...] = (0, 6, 11)) -> str:
    """A lens checkpoint at the target model's width (gpt2), so the declared samples run without a Hub download.

    The samples check that each op runs and yields its columns, not what the lens says, so random ``J`` matrices are
    the right fixture; ``jlens_lens_path`` is the op's documented override for exactly this.
    """
    torch.manual_seed(0)
    path = Path(tempfile.mkdtemp(prefix="jlens_conformance_")) / "synthetic_lens.pt"
    torch.save(
        {"J": {ell: torch.randn(d_model, d_model) / d_model**0.5 for ell in layers}, "n_prompts": 1,
         "source_layers": list(layers), "d_model": d_model},
        path,
    )
    return str(path)


_LENS = _synthetic_lens()


@pytest.fixture(scope="module", autouse=True)
def _conformance_seed_cached():
    """Fetch the target's seed component (``speediedan/rte`` at the revision the suite pins) before any case runs.

    The target reads the seed from the components cache only, since loading never downloads, so without this the suite
    passes only where an earlier run left the seed cached, and fails from an empty cache: a CI job's, or the catalog
    smoke's run of these tests from a pulled snapshot. The key-less pull brings the entrypoint and payloads, and the
    configuration pull the seed configuration of the target's datamodule flavour.
    """
    import interpretune as it
    from interpretune.hub.components import pull_component_config
    from interpretune.testing.conformance.inputs import SEED_CONFIGS, SEED_REPO, SEED_REVISION

    it.hub.pull(SEED_REPO, revision=SEED_REVISION)
    flavour = TestJlensSteeringOpsOnBridge.target.datamodule_flavour
    pull_component_config(SEED_REPO, SEED_CONFIGS[flavour], revision=SEED_REVISION)


class TestJlensSteeringOpsOnBridge(OpCollectionConformance):
    """The collection against interpretune's TransformerBridge target (gpt2), which declares every intervention mode."""

    target = ConformanceTarget(
        composition=("core", "sae_lens"),
        forward_family="hf_native",
        datamodule_flavour="bridge",
        load=stage_local_collection(COLLECTION_DIR),
    )
    collection = "jlens_steering_ops"
    run_input_overrides = {
        "jlens_patch_intervention": {"jlens_lens_path": _LENS},
        "jlens_reject_intervention": {"jlens_lens_path": _LENS},
    }
