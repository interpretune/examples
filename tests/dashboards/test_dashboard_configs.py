"""Guards over the committed dashboard configs in ``dashboards/``.

Interpretune checked these while the configs lived in its tree; they live here now, so the checks do too.
"""

from __future__ import annotations

from pathlib import Path

import yaml

DASHBOARDS_DIR = Path(__file__).resolve().parents[2] / "dashboards"
TRANSCODER_CAPTURE_HOOK = "blocks.{layer}.ln2.hook_out"


def _configs(pattern: str = "*.yaml") -> list[Path]:
    configs = sorted(DASHBOARDS_DIR.glob(pattern))
    assert configs, f"no dashboard configs match {pattern!r} in {DASHBOARDS_DIR}; this guard would pass vacuously"
    return configs


def test_gemmascope_transcoder_configs_name_the_capture_hook():
    """Every gemma-scope-2 transcoder config captures at the tensor the transcoders were trained on.

    A config that omits ``capture_hook_name`` falls back to the hook the SAE declares, which for these releases is the
    residual BEFORE the block norm. That produces a plausible corpus with no error to notice, so this is a forcing
    function rather than a style check. Resolved through the pipeline's loader rather than grepped, because the configs
    use ``EXTENDS:`` and only some of them set the value directly.
    """
    from interpretune.utils import neuronpedia_dashboard_pipeline as pipeline

    for config_path in _configs("gemmascope-2-*.yaml"):
        payload = pipeline.load_dashboard_pipeline_config_payload(config_path)
        values = pipeline._normalize_pipeline_overrides(
            pipeline._extract_dashboard_pipeline_values(payload, config_path=config_path)
        )
        assert values.get("capture_hook_name") == TRANSCODER_CAPTURE_HOOK, (
            f"{config_path.name} resolves capture_hook_name to {values.get('capture_hook_name')!r}, so it would capture "
            "at a tensor these transcoders were not trained on"
        )


def test_committed_dashboard_configs_name_no_database():
    """A database URL in a committed config is right for one machine and cannot be corrected from the environment."""
    for path in _configs():
        text = path.read_text(encoding="utf-8")
        assert "local_db_url" not in text, path.name
        assert yaml.safe_load(text), f"{path.name} is empty or does not parse"
