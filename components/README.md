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
| `rte/` | [`speediedan/rte`](https://huggingface.co/speediedan/rte) | `4e58126454837a6572b13c9d53c1af15e0605478` |
| `prompt-configs/` | [`speediedan/prompt-configs`](https://huggingface.co/speediedan/prompt-configs) | `3d8b9710061283dd14c86a46f0c0649b5e262db0` |
| `concept-direction/` | [`speediedan/concept-direction`](https://huggingface.co/speediedan/concept-direction) | `736e2ca563102fdaaa1c286eefb3ac4bfa6f85a2` |
| `concept_direction_ops/` | [`speediedan/concept_direction_ops`](https://huggingface.co/speediedan/concept_direction_ops) | `16affe3811eb8c5e601def07bd47b41ef4fcdf47` |
| `jlens_steering_ops/` | [`speediedan/jlens_steering_ops`](https://huggingface.co/speediedan/jlens_steering_ops) | `22b03610f1a2eb4a8afc4957f17cf6a77e46ff87` |
| `jspace-battery/` | [`speediedan/jspace-battery`](https://huggingface.co/speediedan/jspace-battery) | `8cac45ebf7e72f2a7b277f30371261cd336a522d` |
| `prompt-analysis/` | [`speediedan/prompt-analysis`](https://huggingface.co/speediedan/prompt-analysis) | `734afa3614851ca3027512089dfc519a30370d5e` |
| `neuron-communities/` | [`speediedan/neuron-communities`](https://huggingface.co/speediedan/neuron-communities) | `9176d38f345fcb6805e257ed9fdee377cafc0fd1` |
| `sasa-blocks/` | [`speediedan/sasa-blocks`](https://huggingface.co/speediedan/sasa-blocks) | `6f51af7ff5e59cb44c2e76f6f8efedb0641acdd6` |
| `neuron-finder/` | [`speediedan/neuron-finder`](https://huggingface.co/speediedan/neuron-finder) | `73b3d860b0ae11b8aa13c2264d6c912eceea8345` |
| `bsf-blocks/` | [`speediedan/bsf-blocks`](https://huggingface.co/speediedan/bsf-blocks) | `0014a7be01cf33d1b2a3db8eb97d1875e9e93145` |
| `op-collection-demo/` | [`speediedan/op-collection-demo`](https://huggingface.co/speediedan/op-collection-demo) | `005039b5fc37fa42f3deb800aa84dd7dadca531e` |

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

Four components carry the notebook helpers that serve one experiment (kind `helpers`): `concept-direction`,
`prompt-analysis`, `neuron-communities` and `sasa-blocks`. The tutorials load them with `it.hub.load_helpers` at the
revision each one pins in its parameters cell, after `it.hub.pull` fetches it and the notebook opts in to running it. A
tutorial's pin moves only with a re-render, so it may trail the table above.

Three more carry the helpers of one tutorial each: `neuron-finder`, `bsf-blocks` and `op-collection-demo` (with the
two collections its tutorial loads). Their tutorials load them the same way.

A component with no CPU session configuration may name its own CPU tests as its smoke, run from the published snapshot
(`smoke: tests/`): `jlens_steering_ops`, `concept-direction`, `prompt-analysis`, `neuron-communities`, `sasa-blocks`,
`neuron-finder`, `bsf-blocks` and `op-collection-demo` are listed that way. `jlens_steering_ops` was republished with a test fix from interpretune/examples#12 (its earlier
tests failed from a pulled snapshot); every other file of it is as first gathered. Not yet in `catalog.yaml`:
`jspace-battery` (no CPU configuration and no CPU tests yet) and `concept_direction_ops` (no tests of its own).

The components tier runs every component's tests the way a pull meets them (`scripts/run_component_tests.py
--published-layout`: each file a symlink into a blob store, every cache empty), so a defect of that kind fails the
pull request that introduces it rather than the scheduled smoke after a publish.
