"""Execute every catalog entry's ``smoke`` from empty caches, at the revision the entry pins.

An entry's ``smoke`` takes one of two forms:

- a module configuration key (``rte_demo.gpt2.core``): pull the component's manifest and that configuration through
  interpretune at the pinned revision, opt in to its code, load the session, and run one test batch through the module;
- a directory of the published snapshot ending in ``/`` (``tests/``), for a component with no CPU session
  configuration: download the snapshot at the pinned revision and run those tests from it, one process per component,
  as ``scripts/run_component_tests.py`` runs them in-tree.

Each entry runs in its own process with its own empty interpretune and Hub caches, so an entry passes only on what the
Hub serves at its revision, never on what an earlier entry or a developer cache left behind. ``smoke: null`` (a
definitions-only entry) is skipped and reported as such.

Usage: ``python scripts/catalog_smoke.py catalog.yaml [--only owner/name ...]``. Exits non-zero naming every entry whose
smoke failed or could not be loaded.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_component_tests import empty_caches, run_tests  # noqa: E402


def smoke_form(smoke: str) -> str:
    """``"tests"`` for a snapshot directory (ends in ``/``), ``"config"`` for a module configuration key."""
    return "tests" if smoke.endswith("/") else "config"


def run_config_smoke(repo: str, revision: str, key: str, work: Path) -> None:
    """Pull ``key`` from ``repo@revision``, load its session and run one test batch (runs in the entry's process)."""
    import torch

    import interpretune as it
    from interpretune.config.loading import load_session_cfg
    from interpretune.hub.components import pull_component_config

    print(f"interpretune {it.__file__}", flush=True)  # which install ran, not only which version
    # the key-less pull materializes the entrypoint and payloads a configuration's classes come from; a configuration
    # pull alone fetches the manifest and the configuration file, and loading never downloads
    manifest, commit = it.hub.pull(repo, revision=revision)
    if commit != revision:
        raise SystemExit(f"{repo}: asked for revision {revision}, the Hub resolved {commit}")
    configs = (manifest.get("module") or {}).get("configs") or {}
    if key not in configs:
        raise SystemExit(
            f"{repo}@{revision[:12]}: no module configuration {key!r} (manifest has: {', '.join(configs)})"
        )
    canonical, body = pull_component_config(repo, key, revision=commit)
    cfg = load_session_cfg(body, expected_key=canonical)
    cfg.datamodule_cfg.dataset_path = work / "dataset"
    session = it.ITSession(cfg)
    it.it_init(session.module, session.datamodule)
    batch = next(iter(session.datamodule.test_dataloader()))
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    with torch.no_grad():
        out = session.module(**inputs)
    logits = getattr(out, "logits", out)
    print(f"{repo}@{revision[:12]} {key}: one batch of {len(next(iter(inputs.values())))} -> {tuple(logits.shape)}")


def run_tests_smoke(repo: str, revision: str, path: str, env: dict[str, str]) -> int:
    """Download ``repo@revision`` into the entry's empty Hub cache and run its ``path`` tests from the snapshot."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=env["HF_HUB_CACHE"]))
    tests = path.rstrip("/")
    if not (snapshot / tests).is_dir():
        print(
            f"{repo}@{revision[:12]}: smoke names {path!r}, which the published snapshot does not have", file=sys.stderr
        )
        return 1
    return run_tests(snapshot, tests=tests, extra_env=env)


def run_entry(entry: dict) -> int:
    """Run one entry's smoke in a fresh process with fresh caches; its exit status."""
    repo, revision, smoke = entry["repo"], entry["revision"], entry["smoke"]
    with tempfile.TemporaryDirectory(prefix="catalog-smoke-") as tmp:
        env = empty_caches(Path(tmp))
        print(f"\n=== {repo}@{revision[:12]} smoke {smoke!r} ({smoke_form(smoke)})", flush=True)
        if smoke_form(smoke) == "tests":
            return run_tests_smoke(repo, revision, smoke, env)
        cmd = [sys.executable, __file__, "--config-smoke", repo, revision, smoke, tmp]
        return subprocess.run(cmd, env={**os.environ, **env}, check=False).returncode


def main(argv: list[str]) -> int:
    """Run the selected catalog entries' smokes; report each by name."""
    if argv[:1] == ["--config-smoke"]:
        repo, revision, key, work = argv[1:5]
        run_config_smoke(repo, revision, key, Path(work))
        return 0
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--only", nargs="+", default=None, help="run only these entries (owner/name)")
    args = parser.parse_args(argv)
    entries = yaml.safe_load(args.catalog.read_text())["components"]
    if args.only:
        missing = sorted(set(args.only) - {e["repo"] for e in entries})
        if missing:
            print(f"not in {args.catalog}: {', '.join(missing)}", file=sys.stderr)
            return 2
        entries = [e for e in entries if e["repo"] in args.only]
    results = {}
    for entry in entries:
        results[entry["repo"]] = "skipped (definitions only)" if entry["smoke"] is None else run_entry(entry)
    print()
    for repo, rc in results.items():
        print(f"{repo}: {rc if isinstance(rc, str) else 'passed' if rc == 0 else f'FAILED (exit {rc})'}")
    return 1 if any(rc not in (0,) and not isinstance(rc, str) for rc in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
