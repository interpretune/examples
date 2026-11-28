# Interpretune examples

Runnable tutorials for [interpretune](https://interpretune.org/en/latest/), the published components they load, and
the configurations behind the public Neuronpedia dashboards. Every tutorial page is rendered from a copy executed ahead
of time, so you can read the results without running anything; the notebook you clone is the same one, without outputs.

- {doc}`gallery`: every tutorial, what it needs, and a link to open it in Colab.
- {doc}`user_guide`: install interpretune, pull a component, run it from the CLI, and run a tutorial.

The library itself (its API, adapters and usage guides) is documented at
[interpretune.org](https://interpretune.org/en/latest/). The source of everything here is the
[interpretune/examples](https://github.com/interpretune/examples) repository.

```{toctree}
:maxdepth: 2
:hidden:

gallery
user_guide
```

```{toctree}
:caption: Tutorials
:maxdepth: 1
:hidden:
:glob:

notebooks/*/*
```
