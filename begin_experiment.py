#!/usr/bin/env python3
"""Launch stage-bounded Qwen autoresearch sessions forever."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


QWEN_BIN = "qwen"
PROMPT_ROOT = Path(".qwen") / "prompts"
EXPERIMENTS_ROOT = Path("experiments")
VALID_FINISHED_STATUSES = {"keep", "discard"}


@dataclass(frozen=True)
class ExperimentSnapshot:
    all_ids: set[str]
    finished_ids: set[str]
    running_ids: set[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def read_prompt(root: Path, stage: str) -> str:
    prompt_path = root / PROMPT_ROOT / f"{stage}.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt file: {prompt_path}") from exc


def load_experiment(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment JSON must be an object: {path}")
    return data


def experiment_snapshot(root: Path) -> ExperimentSnapshot:
    all_ids: set[str] = set()
    finished_ids: set[str] = set()
    running_ids: set[str] = set()

    experiments_root = root / EXPERIMENTS_ROOT
    if not experiments_root.exists():
        return ExperimentSnapshot(all_ids=all_ids, finished_ids=finished_ids, running_ids=running_ids)

    for experiment_path in sorted(experiments_root.glob("EXP-*/experiment.json")):
        try:
            experiment = load_experiment(experiment_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read experiment record {experiment_path}: {exc}") from exc

        experiment_id = str(experiment.get("experiment_id") or experiment_path.parent.name)
        all_ids.add(experiment_id)

        status = experiment.get("status")
        if status == "running":
            running_ids.add(experiment_id)
        elif status in VALID_FINISHED_STATUSES:
            finished_ids.add(experiment_id)

    return ExperimentSnapshot(all_ids=all_ids, finished_ids=finished_ids, running_ids=running_ids)


def choose_stage(snapshot: ExperimentSnapshot) -> str:
    return "init" if not snapshot.all_ids else "loop"


def run_qwen(root: Path, stage: str) -> int:
    prompt = read_prompt(root, stage)
    cmd = [QWEN_BIN, "--approval-mode", "yolo", prompt]
    print(f"\n[orchestrator] starting {stage} stage: {' '.join(cmd[:3])} <prompt>", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(root))
    try:
        return proc.wait()
    except KeyboardInterrupt:
        print("\n[orchestrator] interrupt received; stopping Qwen...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise


def validate_stage_result(stage: str, before: ExperimentSnapshot, after: ExperimentSnapshot) -> None:
    if after.running_ids:
        running = ", ".join(sorted(after.running_ids))
        raise RuntimeError(f"{stage} stage left running experiment(s): {running}")

    if stage == "init":
        if not after.finished_ids:
            raise RuntimeError("init stage did not produce a finished experiment")
        return

    new_ids = after.all_ids - before.all_ids
    if len(new_ids) != 1:
        raise RuntimeError(
            "loop stage must produce exactly one new experiment record; "
            f"found {len(new_ids)} new experiment(s): {sorted(new_ids)}"
        )

    new_experiment_id = next(iter(new_ids))
    if new_experiment_id not in after.finished_ids:
        raise RuntimeError(
            "loop stage produced a new experiment that was not finished as "
            f"`keep` or `discard`: {new_experiment_id}"
        )


def main() -> int:
    root = repo_root()
    print(f"[orchestrator] repo root: {root}", flush=True)

    while True:
        before = experiment_snapshot(root)
        stage = choose_stage(before)

        try:
            returncode = run_qwen(root, stage)
        except KeyboardInterrupt:
            print("[orchestrator] stopped by user", flush=True)
            return 130

        if returncode != 0:
            print(f"[orchestrator] {stage} stage failed with exit code {returncode}", file=sys.stderr)
            return returncode

        try:
            after = experiment_snapshot(root)
            validate_stage_result(stage, before, after)
        except RuntimeError as exc:
            print(f"[orchestrator] {stage} stage validation failed: {exc}", file=sys.stderr)
            return 1

        print(f"[orchestrator] {stage} stage complete", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
