# User guide: install, pull a component, run it

This guide takes you from nothing to a running example in four steps: install interpretune, pull a component from the
Hugging Face Hub, run one of its configurations from the command line, and run a tutorial notebook from this
repository. Every step runs on a CPU; the last section lists the runs that need a GPU.

The examples are validated against the interpretune commit in [`interpretune.ref`](https://github.com/interpretune/examples/blob/main/interpretune.ref). Until
interpretune publishes a release that carries what the tutorials use, you install it from a clone.

## 1. Install interpretune

Interpretune installs with [uv](https://github.com/astral-sh/uv), from a clone of its repository:

```bash
# uv, once
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/interpretune/interpretune.git
cd interpretune
uv venv ~/.venvs/it --python 3.13
source ~/.venvs/it/bin/activate
uv pip install ".[examples,lightning]" --group git-deps
```

Run the install from inside the clone. Interpretune pins TransformerLens and NNsight through uv's
`override-dependencies`, which uv reads from the project it installs and plain `pip` ignores: `pip install .` resolves
an older TransformerLens and fails later in ways that look like bugs. Add `-e` to the last command if you will edit
interpretune itself.

What the extras and the group add:

| piece | what it adds | who needs it |
| --- | --- | --- |
| `examples` | Jupyter, plotting, `evaluate` and the dataset metric dependencies | every tutorial |
| `lightning` | Lightning and finetuning-scheduler (the `tune` extra) | the fine-tuning runs and the tuning tutorials |
| `--group git-deps` | circuit-tracer and SAEDashboard at the commits interpretune pins | every circuit-tracer tutorial and run |

[The gallery](gallery.md) lists what each tutorial needs. Check the install from a directory outside the clone, so
Python cannot pick up the source tree by accident:

```bash
cd ~
python -c "import interpretune; print(interpretune.__file__)"
```

The path printed should be under your virtual environment's `site-packages` (or in the clone, with `-e`).

## 2. Pull a component

A component is a Hugging Face Hub repository with an `it_component.yaml` manifest: configurations, schedules, run
configurations, and the Python entrypoint its configurations name. The catalog ([`catalog.yaml`](https://github.com/interpretune/examples/blob/main/catalog.yaml))
lists the tested ones, each with a pinned revision. This guide uses `speediedan/rte`, the RTE (SuperGLUE textual
entailment) task.

Pulling downloads. Loading never does: everything after a pull reads the local cache.

```python
import interpretune as it

manifest, commit = it.hub.pull("speediedan/rte", revision="fba0afcff43e09a1882aadd767720195880d0755")
print(commit)
print(sorted(manifest["module"]["configs"]))   # the session configurations, e.g. rte_demo.gpt2.core
print(sorted(manifest["datamodules"]))         # standalone datamodule configurations
print(manifest["extra_files"])                 # shipped files: schedules (ft_schedules), CLI run configurations (runs)
```

Without `revision`, the pull resolves the repository's `main` and records that commit as the component's pin, so later
loads read the same snapshot. Pass the catalog's revision when you want the tested one.

A key-less pull, as above, fetches the manifest, the entrypoint and the shipped files, and executes nothing. You can
read the entrypoint module the manifest names (`manifest["module"]["entrypoint"]`) before you run any of it. The
component's source is also in this repository, under [`components/rte/`](https://github.com/interpretune/examples/blob/main/components/rte).

### Opting in to the component's code

A configuration names classes in the component's entrypoint module, so building it imports that module: the
publisher's Python, running in your process. Interpretune refuses that (`RemoteCodeNotTrustedError`) until you opt in.
A pull with a configuration key builds the configuration it fetches, so it needs the opt-in too:

```python
import os

os.environ["IT_TRUST_REMOTE_CODE"] = "1"   # this process only; or export it in your shell
datamodule_cfg, module_cfg, datamodule_cls, module_cls = it.hub.pull("speediedan/rte", "rte_demo.gpt2.core")
```

Once a configuration is pulled, `it.hub.load("speediedan/rte", "rte_demo.gpt2.core")` returns the same four objects
from the cache, without the network, in any later process (still under the opt-in).

Opt in only for publishers you would `pip install` from. Data (manifests, configurations, schedules, AnalysisStores)
never needs the opt-in. Interpretune's [trust posture](https://interpretune.org/en/latest/usage/hub_trust_posture.html)
page lists every path that runs Hub code and what each gate does.

## 3. Run a configuration from the command line

Installing interpretune gives you the `interpretune` command. It resolves components from the cache only, so before a
run, opt in to the component's code and pull both the component and the configuration the run uses:

```bash
export IT_TRUST_REMOTE_CODE=1
python -c 'import interpretune as it; it.hub.pull("speediedan/rte"); it.hub.pull("speediedan/rte", "rte_demo.gpt2.core")'
```

**A session configuration, as is.** `--session_ref <repo>:<config key>[@<revision>]` builds the session from a
component configuration, and `--run_command test` runs its test loop:

```bash
interpretune --run_command test --session_ref speediedan/rte:rte_demo.gpt2.core
```

**A run configuration the component ships.** Components ship complete runs under `runs/`, which the key-less pull
fetches. Each names its session configuration with `session_ref` and adds the runner settings: epochs, callbacks, a
fine-tuning schedule. Pull the configuration a run names, then address the run with `hub://`:

```bash
python -c 'import interpretune as it; it.hub.pull("speediedan/rte", "rte_ft.gpt2.core")'
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.gpt2.core.yaml
```

A run whose configuration is not cached stops before it starts and names the pull that fetches it.

That run tunes GPT-2 on RTE in phases (finetuning-scheduler, on interpretune's core loop). Each run file opens with a
comment saying what it runs, on what hardware, and its command line; read them in
[`components/rte/runs/`](https://github.com/interpretune/examples/blob/main/components/rte/runs).

The GPT-2 configurations run on the CPU (their model loads with `device_map: cpu`). As shipped, the run trains on all of
RTE until the schedule's last phase plateaus, at most ten epochs: about 90 minutes on a desktop CPU. Any setting can be
overridden on the command line, after the `--config`, so for a quick look, limit the epochs and batches:

```bash
interpretune --run_command train --config hub://speediedan/rte/runs/rte_ft.gpt2.core.yaml \
  --run_cfg.max_epochs 3 --run_cfg.limit_train_batches 20 --run_cfg.limit_val_batches 5
```

Runs write their checkpoints and logs under the current directory. The Lightning form of the same run is
`runs/rte_ft.gpt2.lightning.yaml`, given to `--lightning_cli` with Lightning's own subcommand:

```bash
interpretune --lightning_cli fit --config hub://speediedan/rte/runs/rte_ft.gpt2.lightning.yaml
```

On Lightning the same limits are `--trainer.max_epochs`, `--trainer.limit_train_batches` and
`--trainer.limit_val_batches`.

## 4. Run a tutorial notebook

The tutorials live in this repository. Clone it next to your interpretune clone:

```bash
git clone https://github.com/interpretune/examples.git interpretune-examples
cd interpretune-examples
jupyter lab tutorials/publish/quickstart/cpu_quickstart.ipynb
```

Open notebooks from `tutorials/publish/`. The `tutorials/dev/` lane is the authored source that `tutorials/publish/` is
generated from, and its imports name development paths.

The [CPU quickstart](https://interpretune.org/projects/examples/en/latest/notebooks/quickstart/cpu_quickstart.html) is the one to run first: GPT-2 on a CPU, one analysis op over
RTE, the result in an `AnalysisStore`, with no token and no gated weights. To read a tutorial without running it, open
its rendered page from [the gallery](gallery.md): each one is executed ahead of time and shown with its outputs.

What the tutorials ask of you:

- **The opt-in.** A tutorial that runs a component's code sets `IT_TRUST_REMOTE_CODE` in a cell before its first load
  (look for `os.environ["IT_TRUST_REMOTE_CODE"] = "1"`). Read that cell, and the component it pulls, before you run it.
- **Pinned components.** A tutorial's parameters cell names the component revisions it was rendered with. Keep them
  unless you mean to try a newer one.
- **Gated weights.** Some tutorials load models whose licence you accept on the model's Hugging Face page first. Then
  export a token as `HF_GATED_PUBLIC_REPO_AUTH_KEY`, the variable the example configurations read, or log in with
  `hf auth login`. The gallery's table marks which tutorials need it, and which need a GPU.

## Runs that need a GPU

The `rte` component's larger runs use the same commands as step 3 (pull the configuration each one names in its
`session_ref` first), on a GPU with the gated Gemma 3 weights: accept the licence for `google/gemma-3-1b-it`, or
`google/gemma-3-4b-it` for the FSDP2 run, first.

| run | what it does | hardware |
| --- | --- | --- |
| `runs/rte_ft.gemma3.core.yaml` | phased fine-tuning of gemma-3-1b-it, core loop | one GPU |
| `runs/rte_ft.gemma3.lightning.yaml` | the same on Lightning | one GPU |
| `runs/rte_sparse.gemma3.core.yaml` | the sparsity chain: tuning with an attribution penalty, then recovery, each phase graphed and gated against a pinned reference | one 24 GB GPU (the model, a second copy and the transcoders); budget an hour or more |
| `runs/rte_latent.gemma3_4b.core.fsdp2.yaml` | LoRA and spliced-transcoder tuning of gemma-3-4b-it, sharded with FSDP2 | two GPUs, under `torchrun` (the command is in the file) |

```bash
python -c 'import interpretune as it; it.hub.pull("speediedan/rte", "rte_sparse.gemma3.core")'
interpretune --run_command train --config hub://speediedan/rte/runs/rte_sparse.gemma3.core.yaml
```

## When something goes wrong

- **`RemoteCodeNotTrustedError`**: you have not opted in (step 2). The message names the component.
- **A load or a run asks you to pull first** (`ComponentFileNotCachedError`): the component, or the configuration, is
  not in your cache yet. Run the pull the message names; loading never downloads.
- **An import error from TransformerLens, SAELens or SAEDashboard**: the environment was installed with `pip`, or from
  outside the interpretune clone, so uv's version overrides did not apply. Rebuild it as in step 1.
- **A tutorial or catalog entry is broken**: open an issue in this repository. Interpretune's own
  [troubleshooting page](https://interpretune.org/en/latest/usage/when_things_go_wrong.html) covers the library.
