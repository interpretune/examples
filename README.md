# Interpretune examples

Worked examples for [Interpretune](https://github.com/interpretune/interpretune): gallery tutorials, the sources of the
first-party Hub components they use, and a catalog of tested components. The library itself ships no example content;
it lives here, pinned to an Interpretune release.

Documentation: <https://interpretune.org/projects/examples/> (a subproject of the Interpretune docs).

## Layout

| path | what it holds |
| --- | --- |
| `tutorials/dev/<lane>/<name>.ipynb` | the notebooks as authored (the dev lane) |
| `tutorials/publish/<lane>/<name>.ipynb` | the published notebooks, produced from the dev lane by `nbpress-publish`; never edited by hand |
| `docs/notebook_artifacts/<lane>/<name>.ipynb` | each published notebook executed, with outputs, as the docs render it (`nbpress-render`) |
| `components/<name>/` | the source of a first-party Hub component, mirroring its Hub repository one to one, entrypoint included |
| `catalog.yaml` | the curated list of first-party and community components, each with a pinned revision and a CPU smoke (a configuration run for one batch, or the component's own CPU tests) |
| `dashboards/` | Neuronpedia dashboard generation configurations |
| `tests/` | tests for the tutorials, components and catalog |
| `docs/` | the Read the Docs subproject |

## Conventions

- **Tutorials compose components by pinned reference** (`hub://<repo>@<rev>/...`), so each one is also a worked example
  of composing an experiment.
- **Examples publish the AnalysisStores they compute**, so others can stream, compose and extend the work without
  re-running it. A tutorial names each store it publishes, the op that wrote it and its subject, and shows a streamed
  read of it.
- **A broken tutorial or catalog entry is excluded with an issue**, and removed if it is not fixed by the next
  Interpretune release.

The design is recorded on
[interpretune#849](https://github.com/interpretune/interpretune/issues/849); the move into this repository is
[interpretune#1017](https://github.com/interpretune/interpretune/issues/1017).

## CI

- **Per pull request** (`.github/workflows/ci.yml`): each tier runs only when its paths change. Those tiers are tutorials (which also run when a component changes), components, dashboards and the catalog. The components tier runs each component's own `tests/` (`scripts/run_component_tests.py --published-layout`: one process per component, since two components may ship a package of the same name, each staged as a pulled Hub snapshot with empty caches). A change to shared configuration runs every tier. Tests run on CPU against the interpretune commit `interpretune.ref` pins, installed through interpretune's documented install flow. The tutorials tier also checks that the published notebooks and rendered artifacts are current.
- **Scheduled** (`.github/workflows/scheduled.yml`, daily): every tier and each component's own tests against both the pinned interpretune and interpretune `main`, then every catalog entry's smoke (`scripts/catalog_smoke.py`), each from empty caches at the entry's pinned revision. A failure opens an issue labeled `scheduled-failure`.
- **The pin moves by pull request** (`.github/workflows/bump-interpretune.yml`, weekly): it opens a pull request when interpretune `main` has moved.

Secrets: `HF_MCP_TOKEN_RW` (the Hub token interpretune's CI uses; the workflows pass it to the tests as `HF_TOKEN`, `HF_GATED_PUBLIC_REPO_AUTH_KEY` and `IT_HF_TOKEN`) and `EXAMPLE_BOT_TOKEN` (for the bump pull request, so its CI runs).
