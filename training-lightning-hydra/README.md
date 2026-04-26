# Training Example

This directory contains the Lightning-Hydra training harness used by the Qwen
autoresearch loop.

The default training entrypoint expects:

- `configs/config.yaml`

The init stage should create that real default from the `.example` templates.
Template code is reference-only; for real experiments, adapt the data module,
model module, metrics, and config values to match the repository `goal.md` and
available data.

Qwen training runs should use direct Hydra field overrides through the
`train_run` MCP tool, for example:

```bash
++trainer.fast_dev_run=true trainer.accelerator=gpu logger=null
```

Training runs must stay GPU-backed unless a test fixture explicitly overrides
the trainer to CPU for local unit tests.
