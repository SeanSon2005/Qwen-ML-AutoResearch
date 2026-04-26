# Loop Stage Autoresearch Prompt

You are running one bounded loop-stage autoresearch session for this repository.
Complete exactly one new experiment, report the result, and stop. Do not start a
second experiment.

## Objective

Use the current goal, prior experiment history, and existing code to choose and
run exactly one new experiment.

## Required Workflow

1. Read `goal.md`, `QWEN.md`, `.qwen/state/next_agent_note.md` if present, the
   existing root `pipeline/` code, and the existing `training-lightning-hydra/` code
   and configs.
2. Use `experiments_list` to inspect completed and running experiments so far.
3. Choose exactly one smallest useful experiment based on the current goal and
   prior results. The change may involve data, preprocessing, model, metric,
   optimization, validation, comparison, or analysis.
4. Call `experiment_create` before making experiment-specific changes or
   launching training.
5. Implement the experiment and add or update focused tests.
6. Run a fast test pass before training.
7. Run training only through the blocking `train_run` MCP tool. If training
   fails, use the returned evidence to fix and retry under the same experiment
   until there is a usable result or a clear discardable failure.
8. Decide whether the result is `keep` or `discard` using the current goal,
   metric, and prior experiment records.
9. Commit the relevant code, config, and test changes with a message summarizing
   the experiment and decision.
10. Finish the experiment with `experiment_finish`.
11. Overwrite `.qwen/state/next_agent_note.md` with concise guidance for the next
    agent: result interpretation, important artifacts, unresolved issues, and
    the next promising hypothesis.

## Completion Criteria

- Exactly one new experiment is finished as `keep` or `discard`.
- No experiment is left in `running` status.
- Relevant tests pass, or skipped/unrun tests are explicitly justified in your
  final response.
- Your final response is a concise stage report with experiment ID, commit,
  metric, status, key train-run artifact paths, and the next suggested
  hypothesis.

Stop after this stage report.
