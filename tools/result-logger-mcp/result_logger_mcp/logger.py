"""Sequential experiment records and train-run assignment."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = ROOT / "experiments"
TRAIN_RUNS_ROOT = ROOT / ".qwen" / "state" / "train_runs"
VALID_STATUSES = {"keep", "discard"}
EXPERIMENT_ID_RE = re.compile(r"^EXP-(\d{6})$")
WATCHDOG_KV_RE = re.compile(r"^\[watchdog\]\s+(?P<key>[A-Za-z_]+)=(?P<value>.*)$")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def experiment_dir(experiment_id: str) -> Path:
    return EXPERIMENTS_ROOT / experiment_id


def experiment_path(experiment_id: str) -> Path:
    return experiment_dir(experiment_id) / "experiment.json"


def load_experiments() -> list[dict[str, Any]]:
    if not EXPERIMENTS_ROOT.exists():
        return []
    experiments = []
    for path in sorted(EXPERIMENTS_ROOT.glob("EXP-*/experiment.json")):
        try:
            experiments.append(read_json(path))
        except Exception:
            continue
    return experiments


def next_experiment_id() -> str:
    max_id = 0
    if EXPERIMENTS_ROOT.exists():
        for path in EXPERIMENTS_ROOT.iterdir():
            match = EXPERIMENT_ID_RE.match(path.name)
            if match:
                max_id = max(max_id, int(match.group(1)))
    return f"EXP-{max_id + 1:06d}"


def running_experiments() -> list[dict[str, Any]]:
    return [experiment for experiment in load_experiments() if experiment.get("status") == "running"]


def compact_experiment(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment.get("experiment_id"),
        "status": experiment.get("status"),
        "metric": experiment.get("metric"),
        "commit": experiment.get("commit"),
        "description": experiment.get("description"),
        "created_at": experiment.get("created_at"),
        "finished_at": experiment.get("finished_at"),
        "train_run_ids": experiment.get("train_run_ids", []),
    }


def error(message: str, **extra: Any) -> dict[str, Any]:
    payload = {"ok": False, "error": message}
    payload.update(extra)
    return payload


def experiment_create(
    *,
    hypothesis: str,
    decision_type: str,
    description: str,
) -> dict[str, Any]:
    running = running_experiments()
    if running:
        return error(
            "running_experiment_exists",
            running_experiment=compact_experiment(running[0]),
        )

    experiment_id = next_experiment_id()
    created_at = now_iso()
    experiment = {
        "experiment_id": experiment_id,
        "status": "running",
        "created_at": created_at,
        "updated_at": created_at,
        "finished_at": None,
        "hypothesis": hypothesis,
        "decision_type": decision_type,
        "description": description,
        "commit": None,
        "metric": None,
        "train_run_ids": [],
        "train_runs": [],
        "outcome": None,
    }
    path = experiment_path(experiment_id)
    write_json(path, experiment)
    return {
        "ok": True,
        "experiment_id": experiment_id,
        "experiment_path": str(path),
        "experiment": experiment,
    }


def read_watchdog_log_metadata(log_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if not log_path.exists():
        return metadata
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = WATCHDOG_KV_RE.match(line)
        if match:
            metadata[match.group("key")] = match.group("value")
    return metadata


def load_train_run(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = read_json(manifest_path)
            manifest.setdefault("run_id", run_dir.name)
            manifest.setdefault("manifest_json_path", str(manifest_path))
            return manifest
        except Exception:
            pass

    result_path = run_dir / "result.json"
    if result_path.exists():
        try:
            result = read_json(result_path)
            result.setdefault("run_id", run_dir.name)
            result.setdefault("manifest_json_path", str(manifest_path) if manifest_path.exists() else None)
            result.setdefault("result_json_path", str(result_path))
            return result
        except Exception:
            pass

    log_path = run_dir / "train.log"
    metadata = read_watchdog_log_metadata(log_path)
    if not metadata:
        return None
    return {
        "run_id": metadata.get("run_id", run_dir.name),
        "status": None,
        "started_at": metadata.get("started_at"),
        "finished_at": metadata.get("finished_at"),
        "exit_code": int(metadata["exit_code"]) if str(metadata.get("exit_code", "")).isdigit() else None,
        "log_path": str(log_path),
        "result_json_path": str(result_path) if result_path.exists() else None,
    }


def summarize_train_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "ok": run.get("ok"),
        "exit_code": run.get("exit_code"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "duration_sec": run.get("duration_sec"),
        "hydra_output_dir": run.get("hydra_output_dir"),
        "metrics_csv_path": run.get("metrics_csv_path"),
        "manifest_json_path": run.get("manifest_json_path"),
        "result_json_path": run.get("result_json_path"),
        "log_path": run.get("log_path"),
        "watchdog_log_path": run.get("watchdog_log_path"),
    }


def infer_train_runs(start_at: str, end_at: str) -> list[dict[str, Any]]:
    start_dt = parse_iso(start_at)
    end_dt = parse_iso(end_at)
    if not TRAIN_RUNS_ROOT.exists():
        return []

    assigned = []
    for run_dir in sorted(TRAIN_RUNS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        run = load_train_run(run_dir)
        if run is None:
            continue

        if start_dt is None or end_dt is None:
            continue
        run_started = parse_iso(str(run.get("started_at") or ""))
        if run_started is None:
            continue
        if start_dt <= run_started <= end_dt:
            assigned.append(summarize_train_run(run))
    assigned.sort(key=lambda item: str(item.get("started_at") or ""))
    return assigned


def experiment_finish(
    *,
    experiment_id: str,
    commit: str,
    status: str,
    metric: float | None,
    description: str,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        return error("invalid_status", valid_statuses=sorted(VALID_STATUSES), status=status)
    if status == "keep" and metric is None:
        return error("metric_required", status=status)

    path = experiment_path(experiment_id)
    if not path.exists():
        return error("experiment_not_found", experiment_id=experiment_id)

    experiment = read_json(path)
    if experiment.get("status") != "running":
        return error(
            "experiment_not_running",
            experiment_id=experiment_id,
            current_status=experiment.get("status"),
        )

    finished_at = now_iso()
    train_runs = infer_train_runs(str(experiment.get("created_at") or ""), finished_at)
    experiment["status"] = status
    experiment["updated_at"] = finished_at
    experiment["finished_at"] = finished_at
    experiment["commit"] = commit
    experiment["metric"] = float(metric) if metric is not None else None
    experiment["description"] = description
    experiment["outcome"] = description
    experiment["train_runs"] = train_runs
    experiment["train_run_ids"] = [str(run.get("run_id")) for run in train_runs if run.get("run_id")]
    write_json(path, experiment)

    return {
        "ok": True,
        "experiment_id": experiment_id,
        "experiment_path": str(path),
        "assigned_train_run_ids": experiment["train_run_ids"],
        "experiment": experiment,
    }


def experiments_list(
    *,
    status_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    experiments = load_experiments()
    if status_filter:
        experiments = [exp for exp in experiments if exp.get("status") == status_filter]
    experiments.sort(key=lambda exp: str(exp.get("created_at") or ""), reverse=True)
    if limit is not None and limit >= 0:
        experiments = experiments[:limit]
    return {
        "ok": True,
        "experiments": [compact_experiment(exp) for exp in experiments],
    }
