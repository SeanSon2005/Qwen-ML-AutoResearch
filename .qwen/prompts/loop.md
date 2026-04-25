# Loop Stage Autoresearch Prompt

You are running one loop-stage experiment for this repository. This is a bounded
one-shot session launched by `begin_experiment.py`; complete exactly one new
experiment, then stop. Do not continue into another experiment. The Python
orchestrator will relaunch Qwen indefinitely.

## Objective

Improve or probe the existing EEG autoresearch pipeline with one new
well-scoped experiment.

## Required Workflow

1. Read `goal.md`, `QWEN.md`, existing experiment records, recent train-run
   manifests, and the current training code/configs.
2. Use `experiments_list` to identify prior work and the best current `keep`
   experiment.
3. Choose exactly one smallest high-impact experiment based on the prior
   results. The experiment may be an architecture, preprocessing, feature,
   metric, training, validation, or comparison change.
4. Call `experiment_create` before making the experiment change or launching
   training.
5. Implement the code/config changes and add or update focused tests.
6. Run a fast test pass before training.
7. Run training only through the blocking `train_run` MCP tool from
   `train_watchdog`.
   - Use GPU-backed Hydra overrides only.
   - If training fails, use the returned evidence to fix and retry under the same
     experiment until there is a usable metric or a clear discardable reason.
8. Decide whether the result is `keep` or `discard` using the single experiment
   metric and prior experiment records.
9. Commit the experiment changes with a message summarizing the experiment and
   status.
10. Finish the experiment with `experiment_finish`.

## Completion Criteria

- Exactly one new experiment is finished as `keep` or `discard`.
- No experiment is left in `running` status.
- Relevant tests pass or any skipped/unrun tests are explicitly justified in
  your final response.
- Your final response is a concise stage report with experiment ID, commit,
  metric, status, key train-run artifact paths, and the next suggested
  hypothesis.

Stop after this stage report.
