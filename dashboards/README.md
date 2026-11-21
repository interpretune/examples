# Neuronpedia dashboard configurations

Configurations for interpretune's Neuronpedia dashboard pipeline (`interpretune.utils.neuronpedia_dashboard_pipeline`
and its launcher), for gemma-3-1b-it with the GemmaScope 2 transcoders. The pipeline, its options and the run notes
are documented in interpretune's
[dashboard pipeline guide](https://interpretune.readthedocs.io/en/latest/usage/neuronpedia_dashboard_pipeline.html).

| configuration | what it generates |
| --- | --- |
| `gemmascope-2-rte-base.yaml` | the base the RTE configurations extend |
| `gemmascope-2-transcoder-16k-rte-production.yaml` | all layers, 16k transcoders, RTE prompts |
| `gemmascope-2-transcoder-16k-monology-production.yaml` | all layers, 16k transcoders, 4,096 Monology prompts |
| `gemmascope-2-transcoder-16k-monology-24576.yaml` | the same with 24,576 Monology prompts |
| `gemmascope-2-transcoder-262k-rte-gpu0-benchmark-template.yaml`, `-gpu1.yaml`, `-gpu01.yaml` | 262k transcoders, RTE prompts, on one GPU or two |
| `gemmascope-2-transcoder-262k-monology-production.yaml` | 262k transcoders, Monology prompts: a reference baseline, not a starting point |
| `gemmascope-2-transcoder-262k-monology-multigpu-validation.yaml` | a basic multi-GPU and resume check |

## Pretokenizing the RTE prompts

`dashboard_pretokenization_rte.py` renders RTE items with the rte component's chat-template prompt configuration, and
SAEDashboard's pretokenization harness imports it by module name. Run from the repository root with this directory on
the path:

```bash
PYTHONPATH=dashboards python -m sae_dashboard.neuronpedia.prompt_pretokenization \
  --dataset-path aps/super_glue --dataset-name rte --dataset-split train \
  --tokenizer-name google/gemma-3-1b-it --context-size 128 \
  --custom-dataset-module dashboard_pretokenization_rte \
  --windowing-mode max-prompt-pad --no-shuffle \
  --output-dir "${IT_NP_CACHE}/pretokenized/gemma-3-1b-it_rte_boolq_context319_chat_template_full_prompts" --force
```

The prompt configuration comes from `components/rte/configs/rte_demo.gemma3.circuit_tracer.neuronpedia.yaml` by
default (`--experiment-config` overrides it), and the RTE classes come from the cached `speediedan/rte` component
(`it.hub.pull('speediedan/rte')` once).
