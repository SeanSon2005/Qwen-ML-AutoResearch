# Qwen Runtime Contract

## Scope and Ownership
- Repository root: the directory containing this `QWEN.md` file.
- Use `training-lightning-hydra` only for finetuning, training, and building neural network code/configs.
- Use repo-root `src` for other code Qwen needs to write, such as scripts, analysis utilities, model-use helpers, or one-off automation.
- Writable code/config roots (enforced by `.qwen/hooks/pre_tool_use.py`):
  - `.qwen/agents/**`
  - `src/**`
  - `tools/train-watchdog-mcp/**`
  - `training-lightning-hydra/src/data/**`
  - `training-lightning-hydra/src/data/components/**`
  - `training-lightning-hydra/src/models/**`
  - `training-lightning-hydra/src/models/components/**`
  - `training-lightning-hydra/src/utils/**`
  - `training-lightning-hydra/configs/**`
  - `training-lightning-hydra/tests/**`
- Writable experiment ledger:
  - `results.tsv`

Do not edit files outside these roots and the experiment ledger. Qwen may read any file in the repository.

## Runtime Status
- Training runs must be delegated to the `train-runner` project Subagent.
- The `train-runner` Subagent must call the `train_run` MCP tool exposed by the `train_watchdog` MCP server.
- The `train_run` tool is blocking and returns only after training reaches terminal state.
- Do not call training directly from the parent agent.

## Goal Source of Truth (`goal.md`)
- The user defines the experiment objective in repo-root `goal.md`.
- `goal.md` is the authoritative contract for:
  - what to optimize
  - target metric thresholds
  - hard constraints (for example memory limits)
  - stopping criteria
- At the start of each loop iteration, re-read `goal.md` and verify current progress against it.
- If `goal.md` is changed, treat the new content as authoritative and adapt the plan immediately.

## Experiment Ledger (`results.tsv`)
- Record every completed experiment in repo-root `results.tsv`.
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
- Training is supervised by the `train_run` MCP watchdog tool.
- Hydra overrides should still be passed through unchanged to `src.train`.
- Use fast smoke-style overrides for rapid validation (`++trainer.fast_dev_run=true`, `trainer=cpu`, `logger=[]`).
- Use full/default configs for major runs and compare run outputs in `results.tsv`.

## Testing and Regression Discipline
- During iteration, run a fast test pass after each logical change batch.
- Before major training sessions or final delivery, run the full test suite.
- The v1 `train_run` watchdog does not run tests; run required regression checks before delegating training.

## Mandatory Goal Loop
1. Read `goal.md` and extract objective, metric targets, constraints, and stop condition.
2. Review `results.tsv` and identify the current best `keep` run and open gaps to goal.
3. Propose the next smallest high-impact change.
4. Implement code/config updates only in allowed roots and add/update tests.
5. Run a fast test pass.
6. Delegate training to the `train-runner` Subagent with arbitrary Hydra overrides and the exact information needed from the run.
7. Use the Subagent report from `train_run`; final metrics come from the run's CSV metrics artifact and are not limited to fixed metric names.
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
- Do not stop early after one successful run if goal thresholds are not yet satisfied. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped or your goal thresholds are met. You are autonomous. If you run out of ideas, think harder. Use the `paper-search-mcp` tool for papers (and the web when needed), re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes.
- Keep `results.tsv` up to date after every experiment without skipping rows.
