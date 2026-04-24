# Qwen Runtime Contract

## Scope and Ownership
- Repository root: the directory containing this `QWEN.md` file.
- Use `training-lightning-hydra` only for finetuning, training, and building neural network code/configs.
- Use repo-root `src` for other code Qwen needs to write, such as scripts, analysis utilities, model-use helpers, or one-off automation.
- Writable code/config roots (enforced by `.qwen/hooks/pre_tool_use.py`):
  - `.qwen/agents/**`
  - `experiments/**`
  - `src/**`
  - `tools/result-logger-mcp/**`
  - `tools/train-watchdog-mcp/**`
  - `training-lightning-hydra/src/data/**`
  - `training-lightning-hydra/src/data/components/**`
  - `training-lightning-hydra/src/models/**`
  - `training-lightning-hydra/src/models/components/**`
  - `training-lightning-hydra/src/utils/**`
  - `training-lightning-hydra/configs/**`
  - `training-lightning-hydra/tests/**`

Do not edit files outside these roots. Qwen may read any file in the repository.

## Runtime Status
- Training runs must be delegated to the `train-runner` project Subagent.
- The `train-runner` Subagent must call the `train_run` MCP tool exposed by the `train_watchdog` MCP server.
- The `train_run` tool is blocking and returns only after training reaches terminal state.
- Do not call training directly from the parent agent.
- Experiment bookkeeping must use the `result_logger` MCP tools: `experiment_create`, `experiment_finish`, and `experiments_list`.
- `experiments_list` is the result-summary interface for reviewing what has been tried and what worked or failed.

## Goal Source of Truth (`goal.md`)
- The user defines the experiment objective in repo-root `goal.md`.
- `goal.md` is the authoritative contract for:
  - what to optimize
  - the single optimization metric
  - hard constraints (for example memory limits)
- There is exactly one optimization metric. Use the metric value supplied to `experiment_finish` as the experiment score.
- `goal.md` must not be treated as a stopping contract. There are no stopping criteria for the autonomous loop.
- At the start of each loop iteration, re-read `goal.md` and verify current progress against it.
- If `goal.md` is changed, treat the new content as authoritative and adapt the plan immediately.

## Experiment Records (`experiments/`)
- Experiments are decision-level records. A single experiment may contain multiple sequential training runs.
- The authoritative experiment records live under repo-root `experiments/EXP-*/experiment.json`.
- Use `experiments_list` to review prior attempts, current best results, and failed or discarded ideas.
- Use `experiment_create` before making an experiment change or launching training for that experiment.
- Use `experiment_finish` after committing the experiment. It records `commit`, `status`, the single metric value, and a short description.
- `experiment_finish` infers assigned train runs from the active experiment time window; do not pass train run IDs manually.
- `results.tsv` is legacy context only and is not the active result interface.

## Lightning-Hydra Usage
- Training is supervised by the `train_run` MCP watchdog tool.
- Hydra overrides should still be passed through unchanged to `src.train`.
- Use fast smoke-style overrides for rapid validation (`++trainer.fast_dev_run=true`, `trainer=cpu`, `logger=[]`).
- Use full/default configs for major runs and compare experiment summaries with `experiments_list`.

## Testing and Regression Discipline
- During iteration, run a fast test pass after each logical change batch.
- Before major training sessions or final delivery, run the full test suite.
- The v1 `train_run` watchdog does not run tests; run required regression checks before delegating training.

## Mandatory Goal Loop
1. Read `goal.md` and extract objective, the single optimization metric, and constraints.
2. Use `experiments_list` and identify the current best `keep` experiment and open gaps to improve.
3. Propose the next smallest high-impact experiment.
4. Call `experiment_create` before making the experiment change or launching training.
5. Implement code/config updates only in allowed roots and add/update tests.
6. Run a fast test pass.
7. Delegate one or more sequential training runs to the `train-runner` Subagent with arbitrary Hydra overrides and the exact information needed from each run.
8. Use the Subagent report from `train_run`; final run metrics come from each run's CSV metrics artifact.
9. Decide experiment status:
   - `keep`: moves toward or satisfies goal
   - `discard`: valid run but not an improvement
   - `crash`: failed run (error, timeout, OOM, invalid outputs)
10. Commit the experiment:
   - create one git commit per experiment iteration
   - commit message should summarize key change and status
11. Call `experiment_finish` with the experiment ID, short commit hash, status, single metric value, and description.
12. Iterate immediately from step 1. Never end the loop voluntarily.

## Workflow Expectation
- Continue iterating experiments forever. There are no automatic stopping criteria.
- The loop can only be stopped manually by the user.
- Never prompt the user to continue, approve the next iteration, choose a stopping point, or confirm whether more work should be done.
- Never stop because a metric target was reached, a run succeeded, improvements slowed, ideas seem exhausted, or the user is inactive.
- Assume the user may be away for an unknown amount of time and expects autonomous work to continue indefinitely.
- If progress stalls, keep trying new approaches: use the `paper-search-mcp` tool for papers (and the web when needed), re-read the in-scope files for new angles, combine previous near-misses, try more radical architectural changes, and continue logging every experiment.
- Keep `experiments/EXP-*/experiment.json` up to date through the result logger MCP after every experiment without skipping records.
