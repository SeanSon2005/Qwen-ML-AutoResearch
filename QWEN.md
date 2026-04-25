# Qwen Repository Guide

This repository is an autoresearch workspace for machine-learning experiments.
Qwen can read and modify files across the repository as needed.

## Folders

- `.qwen/`: Qwen runtime configuration, prompt files, and local state. Generated
  state lives under `.qwen/state/`.
- `.qwen/prompts/`: Stage prompts used by `begin_experiment.py` when launching
  bounded Qwen sessions.
- `data/`: Local datasets. The current research data is the BCI Competition IV
  2a EEG `.mat` files.
- `experiments/`: Experiment records created by the result logger. Each
  `EXP-*` directory stores the hypothesis, decision, metric, commit, and linked
  train-run summaries for one experiment.
- `logs/`: Generated Hydra, training, and watchdog logs.
- `src/`: Repo-level utilities that are not part of the Lightning training
  package, such as analysis helpers or one-off automation.
- `tools/result-logger-mcp/`: MCP server for sequential experiment bookkeeping,
  including `experiments_list`, `experiment_create`, and `experiment_finish`.
- `tools/train-watchdog-mcp/`: MCP server for blocking Lightning-Hydra training
  runs through `train_run`, plus resource monitoring and train-run manifests.
- `tools/paper-search-mcp-patched/`: Local patched MCP server for literature
  search tools.
- `training-lightning-hydra/`: Lightning-Hydra training harness. Put training
  configs, data modules, datasets, model modules, metrics, and training tests
  here.
- `training-lightning-hydra/configs/`: Hydra configs for training and
  evaluation.
- `training-lightning-hydra/src/data/`: Lightning data modules and dataset
  loading/preprocessing code.
- `training-lightning-hydra/src/models/`: Lightning modules and neural-network
  components.
- `training-lightning-hydra/src/utils/`: Shared training utilities.
- `training-lightning-hydra/tests/`: Tests for configs, data modules, training,
  evaluation, and related utilities.

## Task Files

- `goal.md`: Research objective, dataset description, constraints, and metric
  guidance.
- `.qwen/prompts/init.md`: Instructions for the first bounded Qwen session that
  builds the initial EEG pipeline and baseline experiment.
- `.qwen/prompts/loop.md`: Instructions for each later bounded Qwen session that
  performs exactly one new experiment.
- `begin_experiment.py`: Local Python orchestrator that repeatedly launches
  stage-bounded Qwen CLI sessions. It is intentionally excluded from Qwen prompt
  context by `.qwenignore`.
- `pyproject.toml`: Root Python project metadata and dependencies.
- `uv.lock`: Root dependency lockfile when present.
