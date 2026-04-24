# Qwen Runtime Contract

## Scope
- Repository root is the directory containing this `QWEN.md`.
- Qwen may read any file in the repository.
- Qwen may write only within these roots, enforced by `.qwen/hooks/pre_tool_use.py`:
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

Use `training-lightning-hydra` only for finetuning, training, and neural-network code/configs. Use repo-root `src` for other code, such as analysis utilities, model-use helpers, and one-off automation.

## Goal And Data
- `goal.md` is the source of truth for the task, data requirements, objective, constraints, and optimization metric when specified.
- Re-read `goal.md` at the start of every loop. If it changes, adapt immediately.
- Treat the default MNIST/MLP Lightning-Hydra code as a template only. Do not assume MNIST, an MLP, image classification, or default configs are correct unless `goal.md` and the available data support that.
- Adapt data modules, datasets, model architecture, configs, losses, metrics, and evaluation logic to match `goal.md` and the actual repository data.
- There is exactly one optimization metric for experiment comparison. If `goal.md` names it exactly, use it.
- If `goal.md` does not specify an exact metric, derive and implement one sturdy, accurate metric from the objective and data.
- A derived metric must directly measure the objective, be deterministic, use held-out validation/test data, handle edge cases explicitly, and avoid train/test leakage.
- Add tests for derived metrics before using them for experiment decisions. Record the derived metric name and rationale in the experiment description.

## Required Tools
- Training must be delegated to the `train-runner` Subagent.
- `train-runner` must call the blocking `train_run` MCP tool from `train_watchdog`; the parent agent must not call training directly.
- Experiment bookkeeping must use the `result_logger` MCP tools:
  - `experiments_list`
  - `experiment_create`
  - `experiment_finish`
- `experiments_list` is the result-summary interface. `results.tsv` is legacy context only and is not the active result interface.

## Experiment Records
- Experiments are decision-level records under `experiments/EXP-*/experiment.json`.
- One experiment may contain multiple sequential training runs.
- Call `experiment_create` before making an experiment change or launching training.
- After committing the experiment, call `experiment_finish` with experiment ID, short commit hash, status, single metric value, and description.
- Valid experiment statuses:
  - `keep`: moves toward the goal
  - `discard`: valid run but not an improvement
  - `crash`: failed run, timeout, OOM, invalid output, or unusable result
- `experiment_finish` infers assigned train runs from the active experiment time window; do not pass train run IDs manually.
- Train-run assignment uses compact `.qwen/state/train_runs/<run_id>/manifest.json` pointers. Full metrics and logs remain under `logs/`.

## Testing And Training
- Run a fast test pass after each logical change batch.
- Run the full test suite before major training sessions or major finalization points.
- `train_run` does not run tests; complete required regression checks before delegating training.
- Hydra overrides should pass through unchanged to `src.train`.
- Use smoke-style overrides for quick validation, for example `++trainer.fast_dev_run=true`, `trainer=cpu`, and `logger=[]`.
- Use full/default configs for major runs and compare experiment summaries with `experiments_list`.

## Mandatory Loop
1. Read `goal.md`; extract task, data requirements, constraints, and exact metric if specified.
2. Inspect available data and the current pipeline; identify mismatches with the goal.
3. If the metric is not specified exactly, derive, implement, and test one sturdy metric.
4. Use `experiments_list` to review prior work and identify the current best `keep` experiment.
5. Choose the next smallest high-impact experiment.
6. Call `experiment_create`.
7. Implement code/config changes only in writable roots and add/update tests.
8. Run a fast test pass.
9. Delegate one or more sequential training runs to `train-runner`.
10. Use the Subagent report and CSV metrics artifact to choose the single experiment metric value.
11. Decide `keep`, `discard`, or `crash`.
12. Commit the experiment with a message summarizing the change and status.
13. Call `experiment_finish`.
14. Immediately start again at step 1.

## Autonomy
- Continue iterating forever. There are no automatic stopping criteria.
- Only the user can manually stop the loop.
- Never ask whether to continue, approve the next iteration, choose a stopping point, or confirm more work.
- Never stop because a target was reached, a run succeeded, improvements slowed, ideas seem exhausted, or the user is inactive.
- Assume the user may be away for an unknown amount of time and expects autonomous work to continue indefinitely.
- If progress stalls, keep trying: search papers, use the web when needed, re-read in-scope files, combine near-misses, try more radical changes, and keep logging every experiment.
