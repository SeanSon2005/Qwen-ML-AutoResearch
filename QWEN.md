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

Allowed git commands for experiment bookkeeping:
- `git status`
- `git diff`
- `git log`
- `git add`
- `git commit`

Direct training calls (`src/train.py`, `python -m src.train`, `uv run python -m src.train`) are blocked unless routed through `qtrain`.

## Goal Source of Truth (`goal.md`)
- The user defines the experiment objective in `/home/uasai/Documents/Sean/autoresearch/goal.md`.
- `goal.md` is the authoritative contract for:
  - what to optimize
  - target metric thresholds
  - hard constraints (for example memory limits)
  - stopping criteria
- At the start of each loop iteration, re-read `goal.md` and verify current progress against it.
- If `goal.md` is changed, treat the new content as authoritative and adapt the plan immediately.

## Experiment Ledger (`results.tsv`)
- Record every completed experiment in `/home/uasai/Documents/Sean/autoresearch/results.tsv`.
- File format must be TSV only (tab-separated, never comma-separated).
- Header row is required and must include at least:
  - `commit`
  - one or more metric columns (`metric_1 ... metric_N` or explicit metric names from `goal.md`)
  - `memory_gb`
  - `status`
  - `description`
- Row rules:
  - `commit`: short git hash, 7 chars
  - metrics: numeric with 6 decimals (for example `1.234567`)
  - crashes: metrics are `0.000000`
  - `memory_gb`: one decimal (peak_vram_mb / 1024), crashes use `0.0`
  - `status`: one of `keep`, `discard`, `crash`
  - `description`: short experiment summary; keep it plain text and avoid tab characters

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

## Mandatory Goal Loop
1. Read `goal.md` and extract objective, metric targets, constraints, and stop condition.
2. Review `results.tsv` and identify the current best `keep` run and open gaps to goal.
3. Propose the next smallest high-impact change.
4. Implement code/config updates only in allowed roots and add/update tests.
5. Run `./.qwen/bin/qtest fast`.
6. Launch experiment with `./.qwen/bin/qtrain start ...`; monitor via `status/logs/result` until terminal state.
7. Parse final metrics and memory usage from run outputs.
8. Decide status:
   - `keep`: moves toward or satisfies goal
   - `discard`: valid run but not an improvement
   - `crash`: failed run (error, timeout, OOM, invalid outputs)
9. Commit the experiment:
   - create one git commit per experiment iteration
   - commit message should summarize key change and status
10. Append one TSV row to `results.tsv` for this experiment using the required format.
11. Check stop condition from `goal.md`:
   - if met, stop and report best commit and metrics
   - if not met, iterate from step 1

## Workflow Expectation
- Continue iterating experiments until the explicit stop condition in `goal.md` is met.
- Do not stop early after one successful run if goal thresholds are not yet satisfied.
- Keep `results.tsv` up to date after every experiment without skipping rows.
