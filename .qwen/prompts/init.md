# Init Stage Autoresearch Prompt

You are running the first bounded autoresearch session for this repository.
Complete the init stage, report the result, and stop. Do not start a second
experiment.

## Objective

Read the current project goal, build the first working training pipeline for
that goal, and record the first experiment.

## Required Workflow

1. Read `goal.md`, `QWEN.md`, the repository structure, `.example` templates,
   tests, and MCP tool docs.
2. If `.qwen/state/next_agent_note.md` exists, read it for handoff context.
3. Use `training-lightning-hydra/` strictly for deep-learning training or
   finetuning code: data modules, datasets, transforms, Lightning modules,
   metrics, callbacks, and Hydra configs.
4. Use root `src/` for general Python utilities, scripts, analysis helpers, and
   non-training code.
5. Treat files ending in `.example` as templates only. Create real default files
   that match `goal.md`, including `training-lightning-hydra/configs/config.yaml`.
   Do not run template or stale default training targets.
6. Implement the initial pipeline and add or update focused tests until the code
   path works.
7. Call `experiment_create` for the first experiment.
8. Run training only through the blocking `train_run` MCP tool. If training
   fails, fix and retry under the same experiment until there is a successful
   usable result or a clear discardable failure.
9. Commit the relevant code, config, and test changes with a message summarizing
   the experiment.
10. Finish the experiment with `experiment_finish`, using `keep` for a usable
    first result or `discard` for a clear failed baseline.
11. Overwrite `.qwen/state/next_agent_note.md` with concise guidance for the next
    agent: current best result, important artifacts, known issues, and one or
    more promising next hypotheses.

## Completion Criteria

- At least one `experiments/EXP-*/experiment.json` exists with status `keep` or
  `discard`.
- No experiment is left in `running` status.
- Relevant tests pass, or skipped/unrun tests are explicitly justified in your
  final response.
- Your final response is a concise stage report with experiment ID, commit,
  metric, status, key train-run artifact paths, and residual risks.

Stop after this stage report.
