# Qwen Repository Guide

This repository is an autoresearch workspace for machine-learning experiments.
Qwen can read and modify files across the repository as needed.

## Folders

- `data/`: Local datasets for the current research problem.
- `experiments/`: Experiment records created by the result logger. Each
  `EXP-*` directory stores the hypothesis, decision, metric, commit, and linked
  train-run summaries for one experiment.
- `pipeline/`: Repo-level utilities that are not part of the Lightning training
  package, such as analysis helpers, scripts, or automation.
- `training-lightning-hydra/`: Lightning-Hydra training harness. Put training
  configs, data modules, datasets, model modules, metrics, and training tests
  here.
- `training-lightning-hydra/configs/`: Hydra configs for training and
  evaluation. Files ending in `.example` are templates; real defaults should be
  goal-specific files such as `config.yaml`.
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
- `pyproject.toml`: Root Python project metadata and dependencies.
- `uv.lock`: Root dependency lockfile when present.
