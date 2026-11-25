# First-party component sources

Each directory here is the source of one first-party Hub component, and mirrors that Hub repository one to one:
the manifest (`it_component.yaml`), the entrypoint module and every payload it declares. Two files on the Hub are not
kept here, because publishing generates them: the card (`README.md`) and `.gitattributes`.

A component is published from a reviewed commit of this repository, and the revision it was published to is recorded
in the table below, so a pin in interpretune or a tutorial names a revision whose source can be read here. A component
is also listed in `catalog.yaml` once it has a CPU smoke the scheduled run executes (or, like `prompt-configs`, only
defines what other components compose); the catalog check refuses an entry without one.

| directory | Hub repository | published revision |
| --- | --- | --- |
| `rte/` | [`speediedan/rte`](https://huggingface.co/speediedan/rte) | `834a448741b2d092372d97f481d8aa10d609a615` |
| `prompt-configs/` | [`speediedan/prompt-configs`](https://huggingface.co/speediedan/prompt-configs) | `3d8b9710061283dd14c86a46f0c0649b5e262db0` |
| `concept-direction/` | [`speediedan/concept-direction`](https://huggingface.co/speediedan/concept-direction) | `6c54baa93a99747dbe2e08b052a9ba31d2951cd6` |
| `concept_direction_ops/` | [`speediedan/concept_direction_ops`](https://huggingface.co/speediedan/concept_direction_ops) | `16affe3811eb8c5e601def07bd47b41ef4fcdf47` |
| `jlens_steering_ops/` | [`speediedan/jlens_steering_ops`](https://huggingface.co/speediedan/jlens_steering_ops) | `22b03610f1a2eb4a8afc4957f17cf6a77e46ff87` |
| `jspace-battery/` | [`speediedan/jspace-battery`](https://huggingface.co/speediedan/jspace-battery) | `8cac45ebf7e72f2a7b277f30371261cd336a522d` |
| `prompt-analysis/` | [`speediedan/prompt-analysis`](https://huggingface.co/speediedan/prompt-analysis) | `6dd0dd5260bdd56a2c8cd7fb8a2c377554288034` |
| `neuron-communities/` | [`speediedan/neuron-communities`](https://huggingface.co/speediedan/neuron-communities) | `9176d38f345fcb6805e257ed9fdee377cafc0fd1` |
| `sasa-blocks/` | [`speediedan/sasa-blocks`](https://huggingface.co/speediedan/sasa-blocks) | `6f51af7ff5e59cb44c2e76f6f8efedb0641acdd6` |

Both moved from interpretune's in-tree publish sources (`src/it_examples/examples/rte/` and
`src/it_examples/examples/prompt_configs/`). `rte_boolq.py`, which until now existed only on the Hub, comes from the
published snapshot. Every file is identical to the published revision above, so the move itself publishes nothing.

The rest were gathered from their Hub repositories at each one's `main` of the time; the table records their current
published revisions. `prompt-analysis` was also an in-tree publish source in interpretune (`src/it_examples/examples/prompt_analysis/`),
identical to its published revision. Each repository's earlier history stays on the Hub.

`jlens_steering_ops` keeps its `README.md`: that card is written by hand, not generated. The two op collections
are not built by `local_publish`, so the files here are the published files as they are.
`concept_direction_ops` was first generated from interpretune's bundled `concept` op family
(`scripts/publish_op_collection.py`) and has diverged from it since. Its source is now the directory here, not that
family.

Four components carry the notebook helpers that serve one experiment (kind `helpers`, loaded with
`it.hub.load_helpers`): `concept-direction`, `prompt-analysis`, `neuron-communities` and `sasa-blocks`. The helpers were
copied from interpretune's `it_examples.utils`, and the notebooks import them from there until the components are
published and the notebooks repoint (interpretune/interpretune#1148). All four are published at the revisions above,
which is the first published state of each helper module.

A component with no CPU session configuration may name its own CPU tests as its smoke, run from the published snapshot
(`smoke: tests/`): `jlens_steering_ops`, `concept-direction`, `prompt-analysis`, `neuron-communities` and `sasa-blocks`
are listed that way. `jlens_steering_ops` was republished with a test fix from interpretune/examples#12 (its earlier
tests failed from a pulled snapshot); every other file of it is as first gathered. Not yet in `catalog.yaml`:
`jspace-battery` (no CPU configuration and no CPU tests yet) and `concept_direction_ops` (no tests of its own).

The components tier runs every component's tests the way a pull meets them (`scripts/run_component_tests.py
--published-layout`: each file a symlink into a blob store, every cache empty), so a defect of that kind fails the
pull request that introduces it rather than the scheduled smoke after a publish.
