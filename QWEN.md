# Qwen Runtime Contract for `training-lightning-hydra`

## Scope and Ownership
- Repository root: `/home/uasai/Documents/Sean/autoresearch`
- Project target: `training-lightning-hydra`
- Writable code/config roots (hard-enforced by hook policy):
  - `training-lightning-hydra/src/data/**`
  - `training-lightning-hydra/src/data/components/**`
  - `training-lightning-hydra/src/models/**`
  - `training-lightning-hydra/src/models/components/**`
  - `training-lightning-hydra/src/utils/**`
  - `training-lightning-hydra/configs/**`
  - `training-lightning-hydra/tests/**`

Do not edit files outside these roots.

## Required Operator Commands
Use only these helper entrypoints for mutation/testing/training:
- `./.qwen/bin/qtest fast`
- `./.qwen/bin/qtest full`
- `./.qwen/bin/qfs rm|mkdir|touch <path>`
- `./.qwen/bin/qtrain start [--timeout-sec N] [--require-full] -- <hydra_overrides...>`
- `./.qwen/bin/qtrain status [--run-id ID|--latest]`
- `./.qwen/bin/qtrain logs [--run-id ID|--latest] [--tail N]`
- `./.qwen/bin/qtrain result [--run-id ID|--latest]`

Direct training calls (`src/train.py`, `python -m src.train`, `uv run python -m src.train`) are blocked unless routed through `qtrain`.

## Lightning-Hydra Usage
- Always run training from watchdog (`qtrain`) so sessions are detached, supervised, and resumable by run ID.
- Hydra overrides go after `--`. Example:
  - `./.qwen/bin/qtrain start --timeout-sec 1800 -- trainer=cpu ++trainer.fast_dev_run=true logger=[] extras.enforce_tags=false extras.print_config=false`
- Use fast smoke-style overrides for rapid validation (`++trainer.fast_dev_run=true`, `trainer=cpu`, `logger=[]`).
- Use full/default configs for major runs and compare run outputs via watchdog logs/results.

## Testing and Regression Discipline
- During iteration: run `./.qwen/bin/qtest fast` after each logical change batch.
- Before major training sessions or final delivery: run `./.qwen/bin/qtest full`.
- If training should be gated by full regression, start with:
  - `./.qwen/bin/qtrain start --require-full -- <hydra_overrides...>`

## Workflow Expectation
1. Implement code/config changes only in allowed roots.
2. Add/update tests in `training-lightning-hydra/tests`.
3. Run `qtest fast` repeatedly until stable.
4. Run `qtest full` (or `qtrain start --require-full ...`) before major training.
5. Start training with `qtrain start ...` and monitor via `status/logs/result`.
