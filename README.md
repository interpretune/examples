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
| `catalog.yaml` | the curated list of first-party and community components, each with a pinned revision and a CPU smoke configuration |
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
