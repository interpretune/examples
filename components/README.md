# First-party component sources

Each directory here is the source of one first-party Hub component, and mirrors that Hub repository one to one:
the manifest (`it_component.yaml`), the entrypoint module and every payload it declares. Two files on the Hub are not
kept here, because publishing generates them: the card (`README.md`) and `.gitattributes`.

A component is published from a reviewed commit of this repository, and the revision it was published to is recorded
in the table below, so a pin in interpretune or a tutorial names a revision whose source can be read here. A component
is also listed in `catalog.yaml` once it has a CPU smoke configuration the scheduled run executes (or, like
`prompt-configs`, only defines what other components compose); the catalog check refuses an entry without one.

| directory | Hub repository | published revision |
| --- | --- | --- |
| `rte/` | [`speediedan/rte`](https://huggingface.co/speediedan/rte) | `b1a920fc1400837a62bc7a2b155e39e5da206b86` |
| `prompt-configs/` | [`speediedan/prompt-configs`](https://huggingface.co/speediedan/prompt-configs) | `3d8b9710061283dd14c86a46f0c0649b5e262db0` |
| `concept-direction/` | [`speediedan/concept-direction`](https://huggingface.co/speediedan/concept-direction) | `1b173742869f717caf218f2e6afb4deb8f94d0e5` |
| `concept_direction_ops/` | [`speediedan/concept_direction_ops`](https://huggingface.co/speediedan/concept_direction_ops) | `16affe3811eb8c5e601def07bd47b41ef4fcdf47` |
| `jlens_steering_ops/` | [`speediedan/jlens_steering_ops`](https://huggingface.co/speediedan/jlens_steering_ops) | `ebf92e02d14313bc676acde37a17ed1960c1c594` |
| `jspace-battery/` | [`speediedan/jspace-battery`](https://huggingface.co/speediedan/jspace-battery) | `8cac45ebf7e72f2a7b277f30371261cd336a522d` |
| `prompt-analysis/` | [`speediedan/prompt-analysis`](https://huggingface.co/speediedan/prompt-analysis) | `d221f0e9293888836a0101509546968eced813bd` |

Both moved from interpretune's in-tree publish sources (`src/it_examples/examples/rte/` and
`src/it_examples/examples/prompt_configs/`). `rte_boolq.py`, which until now existed only on the Hub, comes from the
published snapshot. Every file is identical to the published revision above, so the move itself publishes nothing.

The rest were gathered from their Hub repositories at the revisions above, which were each one's `main` when
gathered. `prompt-analysis` was also an in-tree publish source in interpretune (`src/it_examples/examples/prompt_analysis/`),
identical to its published revision. Each repository's earlier history stays on the Hub.

`jlens_steering_ops` keeps its `README.md`: that card is written by hand, not generated. The two op collections
are not built by `local_publish`, so the files here are the published files as they are.
`concept_direction_ops` was first generated from interpretune's bundled `concept` op family
(`scripts/publish_op_collection.py`) and has diverged from it since. Its source is now the directory here, not that
family.

Not yet in `catalog.yaml`, because none has a CPU smoke configuration chosen: `concept-direction`, `jspace-battery`
and `prompt-analysis` (no CPU configuration picked yet), and the op collections `concept_direction_ops` (exercised by
the tutorials that load it) and `jlens_steering_ops` (its own `tests/` run its CPU checks). Each gets its entry when a
smoke configuration is chosen.
