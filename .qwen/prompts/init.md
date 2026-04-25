# Init Stage Autoresearch Prompt

You are running the init stage for this repository. This is a bounded one-shot
session launched by `begin_experiment.py`; complete the init stage, then stop.
Do not start a second experiment. The Python orchestrator will relaunch Qwen for
future work.

## Objective

Create the first working EEG autoresearch pipeline and record the first
baseline experiment.

## Required Workflow

1. Read `goal.md` and inspect the repository structure, available data, current
   training harness, MCP tools, and existing tests.
2. Treat the current example Lightning-Hydra code as a template only. Replace or
   adapt it as needed for the BCI Competition IV 2a EEG task in `goal.md`.
3. Derive one deterministic optimization metric if `goal.md` does not name an
   exact metric. Add focused tests for the derived metric before using it for
   experiment decisions.
4. Implement the initial pipeline: dataset loading/preprocessing, channel
   selection that omits C3 and C4 for the target condition, baseline comparison
   support that includes C3 and C4, model, config, metrics, and evaluation.
5. Run a fast test pass after the implementation.
6. Use the `result_logger` MCP tools:
   - Call `experiment_create` for the baseline experiment.
   - Run training only through the blocking `train_run` MCP tool from
     `train_watchdog`.
   - Use GPU-backed Hydra overrides only.
   - If training fails, fix and retry under the same experiment until there is a
     usable metric or a clear discardable reason.
7. Commit the experiment changes with a message summarizing the baseline.
8. Finish the experiment with `experiment_finish`, using `keep` if it produces
   the first usable baseline result, otherwise `discard` with a clear reason.

## Completion Criteria

- At least one `experiments/EXP-*/experiment.json` exists with status `keep` or
  `discard`.
- No experiment is left in `running` status.
- Relevant tests pass or any skipped/unrun tests are explicitly justified in
  your final response.
- Your final response is a concise stage report with experiment ID, commit,
  metric, status, key train-run artifact paths, and any residual risks.

Stop after this stage report.
