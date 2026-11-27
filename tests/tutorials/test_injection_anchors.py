"""The attention-attribution tutorial's injection config matches the circuit-tracer this repository installs.

The notebook patches circuit-tracer's source at anchors its config names. When an upstream change moves an anchor, the
patch fails only once the GPU notebook runs its injection setup; this check finds it on the CPU first: every enabled
hook is found exactly once, and each patched file carries its ``HOOK_REGISTRY.execute`` line and still compiles.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path("circuit_tracer_examples") / "attention_attribution_injection_config.yaml"


@pytest.mark.parametrize("tree", ["dev", "publish"])
def test_attention_notebook_hooks_match_the_installed_circuit_tracer(tmp_path, tree):
    import circuit_tracer

    from interpretune.analysis.injection.analysis_hook_patcher import patch_file_with_hooks, validate_file_hooks
    from interpretune.analysis.injection.config_parser import load_config

    config = load_config(REPO_ROOT / "tutorials" / tree / CONFIG)
    package_root = Path(circuit_tracer.__file__).parent
    enabled_hooks = [hook for hook in config.file_hooks.values() if hook.enabled]
    assert enabled_hooks, f"{tree}/{CONFIG} enables no hooks; this check would pass vacuously"
    assert validate_file_hooks(config.file_hooks, package_root) == {"missing": [], "multiple": []}
    for index, hook in enumerate(enabled_hooks):
        patched = patch_file_with_hooks(package_root / hook.file_path, [hook], tmp_path / f"hook_{index}.py")
        source = patched.read_text(encoding="utf-8")
        assert f"HOOK_REGISTRY.execute('{hook.point_id}', locals())" in source, hook.point_id
        compile(source, str(patched), "exec")
