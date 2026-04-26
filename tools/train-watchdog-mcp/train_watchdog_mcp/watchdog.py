"""Blocking training watchdog and artifact extraction helpers."""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = ROOT / "training-lightning-hydra"
STATE_ROOT = ROOT / ".qwen" / "state" / "train_runs"
EXPERIMENTS_ROOT = ROOT / "experiments"
ACTIVE_TRAINING_LOCK = ROOT / ".qwen" / "state" / "active_training.lock"
METADATA_COLUMNS = {"epoch", "step"}

OUTPUT_DIR_RE = re.compile(r"Output dir:\s*(?P<path>\S+)")
BEST_CKPT_RE = re.compile(r"Best ckpt path:\s*(?P<path>\S+)")
RESTORE_CKPT_RE = re.compile(r"checkpoint path at\s+(?P<path>\S+)")
OOM_PATTERNS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "defaultcpuallocator: can't allocate memory",
    "killed",
)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_run_id() -> str:
    return f"run-{utc_stamp()}-{uuid.uuid4().hex[:8]}"


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def parse_number(value: str | None) -> int | float | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    if parsed.is_integer():
        return int(parsed)
    return parsed


def read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object expected: {path}")
    return data


def load_running_experiments() -> list[dict[str, Any]]:
    if not EXPERIMENTS_ROOT.exists():
        return []

    experiments = []
    for path in sorted(EXPERIMENTS_ROOT.glob("EXP-*/experiment.json")):
        try:
            experiment = read_json(path)
        except Exception:
            continue
        if experiment.get("status") == "running":
            experiments.append(experiment)
    return experiments


def validate_single_running_experiment() -> dict[str, Any] | None:
    running = load_running_experiments()
    if len(running) == 1:
        return None
    if not running:
        return {
            "ok": False,
            "error": "no_running_experiment",
            "message": "Call experiment_create before train_run.",
        }
    return {
        "ok": False,
        "error": "multiple_running_experiments",
        "running_experiment_ids": [
            str(experiment.get("experiment_id"))
            for experiment in running
            if experiment.get("experiment_id")
        ],
        "message": "Finish the running experiment before starting another train_run.",
    }


def write_active_training_lock(
    *,
    run_id: str,
    experiment: dict[str, Any],
    overrides: list[str],
    timeout_sec: int,
    idle_timeout_sec: int,
    monitor_interval_sec: int,
    report_request: str,
) -> None:
    experiment_id = experiment.get("experiment_id")
    write_json(
        ACTIVE_TRAINING_LOCK,
        {
            "experiment_id": experiment_id,
            "idle_timeout_sec": idle_timeout_sec,
            "monitor_interval_sec": monitor_interval_sec,
            "overrides": overrides,
            "report_request": report_request,
            "run_id": run_id,
            "started_at": now_iso(),
            "timeout_sec": timeout_sec,
        },
    )


def clear_active_training_lock(run_id: str) -> None:
    if not ACTIVE_TRAINING_LOCK.exists():
        return
    try:
        lock = read_json(ACTIVE_TRAINING_LOCK)
    except Exception:
        ACTIVE_TRAINING_LOCK.unlink(missing_ok=True)
        return
    if lock.get("run_id") == run_id:
        ACTIVE_TRAINING_LOCK.unlink(missing_ok=True)


def tail_lines(text: str, count: int) -> str:
    if count <= 0:
        return ""
    return "\n".join(text.splitlines()[-count:])


def discover_metrics_csv(hydra_output_dir: Path | None) -> Path | None:
    if hydra_output_dir is None or not hydra_output_dir.exists():
        return None

    direct = hydra_output_dir / "csv" / "metrics.csv"
    if direct.exists():
        return direct

    candidates = sorted(
        (hydra_output_dir / "csv").glob("**/metrics.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def discover_hydra_log(hydra_output_dir: Path | None) -> Path | None:
    if hydra_output_dir is None or not hydra_output_dir.exists():
        return None

    train_log = hydra_output_dir / "train.log"
    if train_log.exists():
        return train_log

    log_files = sorted(
        hydra_output_dir.glob("*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return log_files[0] if log_files else None


def parse_metrics_csv(metrics_path: Path | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {"final": {}, "series": {}}
    if metrics_path is None or not metrics_path.exists():
        return metrics

    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = parse_number(row.get("epoch"))
            step = parse_number(row.get("step"))
            for key, raw_value in row.items():
                if key in METADATA_COLUMNS:
                    continue
                value = parse_float(raw_value)
                if value is None:
                    continue
                point = {"value": value}
                if epoch is not None:
                    point["epoch"] = epoch
                if step is not None:
                    point["step"] = step
                metrics["series"].setdefault(key, []).append(point)
                metrics["final"][key] = value

    return metrics


def extract_first_path(pattern: re.Pattern[str], text: str) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    return matches[-1].group("path")


def extract_tracebacks(text: str) -> list[str]:
    lines = text.splitlines()
    tracebacks: list[str] = []
    idx = 0
    while idx < len(lines):
        if not lines[idx].startswith("Traceback (most recent call last):"):
            idx += 1
            continue

        block = [lines[idx]]
        idx += 1
        while idx < len(lines):
            line = lines[idx]
            if line.startswith("[") and block:
                break
            block.append(line)
            if (
                line
                and not line.startswith((" ", "\t"))
                and ":" in line
                and not line.startswith("Traceback")
            ):
                idx += 1
                break
            idx += 1
        tracebacks.append("\n".join(block).strip())
    return tracebacks


def discover_hydra_output_dir(text: str, started_at: float) -> Path | None:
    parsed = extract_first_path(OUTPUT_DIR_RE, text)
    if parsed:
        path = Path(parsed)
        if path.exists():
            return path

    runs_root = ROOT / "logs" / "train" / "runs"
    if not runs_root.exists():
        return None

    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and path.stat().st_mtime >= started_at - 5
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


@dataclass
class ResourceSummary:
    peak_cpu_percent: float = 0.0
    last_cpu_percent: float = 0.0
    peak_ram_gb: float = 0.0
    last_ram_gb: float = 0.0
    system_ram_total_gb: float = 0.0
    peak_gpu_util_percent: float = 0.0
    last_gpu_util_percent: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    last_gpu_memory_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    peak_gpu_temperature_c: float = 0.0
    last_gpu_temperature_c: float = 0.0
    peak_gpu_power_w: float = 0.0
    last_gpu_power_w: float = 0.0
    samples: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_cpu_percent": round(self.peak_cpu_percent, 3),
            "last_cpu_percent": round(self.last_cpu_percent, 3),
            "peak_ram_gb": round(self.peak_ram_gb, 3),
            "last_ram_gb": round(self.last_ram_gb, 3),
            "system_ram_total_gb": round(self.system_ram_total_gb, 3),
            "peak_gpu_util_percent": round(self.peak_gpu_util_percent, 3),
            "last_gpu_util_percent": round(self.last_gpu_util_percent, 3),
            "peak_gpu_memory_gb": round(self.peak_gpu_memory_gb, 3),
            "last_gpu_memory_gb": round(self.last_gpu_memory_gb, 3),
            "gpu_memory_total_gb": round(self.gpu_memory_total_gb, 3),
            "peak_gpu_temperature_c": round(self.peak_gpu_temperature_c, 3),
            "last_gpu_temperature_c": round(self.last_gpu_temperature_c, 3),
            "peak_gpu_power_w": round(self.peak_gpu_power_w, 3),
            "last_gpu_power_w": round(self.last_gpu_power_w, 3),
            "samples": self.samples,
            "notes": self.notes,
        }


class ResourceMonitor:
    def __init__(self, pid: int, interval_sec: int) -> None:
        self.pid = pid
        self.interval_sec = max(1, interval_sec)
        self.summary = ResourceSummary()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=self.interval_sec + 2)
        with self._lock:
            return self.summary.to_dict()

    def _run(self) -> None:
        try:
            import psutil
        except Exception as exc:  # pragma: no cover - depends on environment
            with self._lock:
                self.summary.notes.append(f"psutil_unavailable: {exc}")
            psutil = None

        process = None
        if psutil is not None:
            try:
                process = psutil.Process(self.pid)
                process.cpu_percent(interval=None)
                self.summary.system_ram_total_gb = psutil.virtual_memory().total / (1024**3)
            except Exception as exc:
                with self._lock:
                    self.summary.notes.append(f"psutil_process_unavailable: {exc}")
                process = None

        while not self._stop.is_set():
            cpu_percent = 0.0
            ram_gb = 0.0
            if process is not None:
                try:
                    processes = [process, *process.children(recursive=True)]
                    for proc in processes:
                        try:
                            cpu_percent += proc.cpu_percent(interval=None)
                            ram_gb += proc.memory_info().rss / (1024**3)
                        except Exception:
                            continue
                except Exception:
                    pass

            gpu = sample_nvidia_smi()
            with self._lock:
                self.summary.samples += 1
                self.summary.last_cpu_percent = cpu_percent
                self.summary.peak_cpu_percent = max(self.summary.peak_cpu_percent, cpu_percent)
                self.summary.last_ram_gb = ram_gb
                self.summary.peak_ram_gb = max(self.summary.peak_ram_gb, ram_gb)
                if gpu:
                    self.summary.last_gpu_util_percent = gpu["util_percent"]
                    self.summary.peak_gpu_util_percent = max(
                        self.summary.peak_gpu_util_percent, gpu["util_percent"]
                    )
                    self.summary.last_gpu_memory_gb = gpu["memory_gb"]
                    self.summary.peak_gpu_memory_gb = max(
                        self.summary.peak_gpu_memory_gb, gpu["memory_gb"]
                    )
                    self.summary.gpu_memory_total_gb = max(
                        self.summary.gpu_memory_total_gb, gpu["memory_total_gb"]
                    )
                    self.summary.last_gpu_temperature_c = gpu["temperature_c"]
                    self.summary.peak_gpu_temperature_c = max(
                        self.summary.peak_gpu_temperature_c, gpu["temperature_c"]
                    )
                    self.summary.last_gpu_power_w = gpu["power_w"]
                    self.summary.peak_gpu_power_w = max(
                        self.summary.peak_gpu_power_w, gpu["power_w"]
                    )

            self._stop.wait(self.interval_sec)


def sample_nvidia_smi() -> dict[str, float] | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None

    if result.returncode != 0 or not result.stdout.strip():
        return None

    util = 0.0
    memory_mb = 0.0
    total_mb = 0.0
    temp = 0.0
    power = 0.0
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            util = max(util, float(parts[0]))
            memory_mb += float(parts[1])
            total_mb += float(parts[2])
            temp = max(temp, float(parts[3]))
            power += float(parts[4])
        except ValueError:
            continue

    return {
        "util_percent": util,
        "memory_gb": memory_mb / 1024,
        "memory_total_gb": total_mb / 1024,
        "temperature_c": temp,
        "power_w": power,
    }


def terminate_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def is_high_memory(resource_summary: dict[str, Any]) -> bool:
    gpu_total = float(resource_summary.get("gpu_memory_total_gb") or 0.0)
    gpu_peak = float(resource_summary.get("peak_gpu_memory_gb") or 0.0)
    ram_total = float(resource_summary.get("system_ram_total_gb") or 0.0)
    ram_peak = float(resource_summary.get("peak_ram_gb") or 0.0)
    return (gpu_total > 0 and gpu_peak / gpu_total >= 0.9) or (
        ram_total > 0 and ram_peak / ram_total >= 0.9
    )


def classify_status(
    *,
    returncode: int | None,
    text: str,
    timed_out: bool,
    idle_timed_out: bool,
    resource_summary: dict[str, Any],
) -> str:
    lowered = text.lower()
    if timed_out:
        return "timeout"
    if idle_timed_out:
        return "idle_timeout"
    if any(pattern in lowered for pattern in OOM_PATTERNS):
        return "oom"
    if returncode == 137:
        return "oom"
    if returncode == -signal.SIGKILL and is_high_memory(resource_summary):
        return "oom"
    if returncode == 0:
        return "success"
    if returncode is None:
        return "crashed"
    return "failed"


def build_failure_evidence(status: str, text: str, returncode: int | None) -> list[str]:
    evidence = []
    if status != "success":
        evidence.append(f"status={status}")
        evidence.append(f"exit_code={returncode}")

    tracebacks = extract_tracebacks(text)
    evidence.extend(tracebacks)

    lowered = text.lower()
    matched_lines = []
    for line in text.splitlines():
        lower = line.lower()
        if any(pattern in lower for pattern in OOM_PATTERNS):
            matched_lines.append(line)
    if matched_lines:
        evidence.append("\n".join(matched_lines[-20:]))

    if status != "success" and not tracebacks and not matched_lines:
        nonblank_lines = [line for line in text.splitlines() if line.strip()]
        if nonblank_lines:
            evidence.append("error_excerpt:\n" + "\n".join(nonblank_lines[-40:]))

    return [item for item in evidence if item]


def build_manifest(result: dict[str, Any]) -> dict[str, Any]:
    manifest_keys = [
        "run_id",
        "status",
        "ok",
        "started_at",
        "finished_at",
        "duration_sec",
        "exit_code",
        "run_dir",
        "hydra_output_dir",
        "metrics_csv_path",
        "log_path",
        "watchdog_log_path",
        "best_checkpoint_path",
    ]
    manifest = {key: result.get(key) for key in manifest_keys}
    if result.get("status") != "success":
        manifest["failure_evidence"] = result.get("failure_evidence", [])
    return manifest


def train_run(
    *,
    overrides: list[str],
    timeout_sec: int = 7200,
    idle_timeout_sec: int = 900,
    monitor_interval_sec: int = 5,
    report_request: str = "metrics and best checkpoint path",
    log_tail_lines: int = 200,
) -> dict[str, Any]:
    validation_error = validate_single_running_experiment()
    if validation_error is not None:
        return validation_error
    running_experiment = load_running_experiments()[0]

    run_id = make_run_id()
    run_dir = STATE_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    watchdog_log = run_dir / "train.log"
    command = ["uv", "run", "python", "-m", "src.train", *overrides]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("COLUMNS", "240")

    started_at_monotonic = time.monotonic()
    started_at_wall = time.time()
    started_at_iso = now_iso()
    last_output_monotonic = started_at_monotonic
    timed_out = False
    idle_timed_out = False

    try:
        write_active_training_lock(
            run_id=run_id,
            experiment=running_experiment,
            overrides=overrides,
            timeout_sec=timeout_sec,
            idle_timeout_sec=idle_timeout_sec,
            monitor_interval_sec=monitor_interval_sec,
            report_request=report_request,
        )
        with watchdog_log.open("w", encoding="utf-8") as log_file:
            log_file.write(f"[watchdog] run_id={run_id}\n")
            log_file.write(f"[watchdog] started_at={started_at_iso}\n")
            log_file.write(f"[watchdog] cwd={TRAINING_ROOT}\n")
            log_file.write(f"[watchdog] command={' '.join(command)}\n")
            log_file.write(f"[watchdog] report_request={report_request}\n")
            log_file.flush()

            proc = subprocess.Popen(
                command,
                cwd=str(TRAINING_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            monitor = ResourceMonitor(proc.pid, monitor_interval_sec)
            monitor.start()

            def reader() -> None:
                nonlocal last_output_monotonic
                assert proc.stdout is not None
                for line in proc.stdout:
                    last_output_monotonic = time.monotonic()
                    log_file.write(line)
                    log_file.flush()

            reader_thread = threading.Thread(target=reader, daemon=True)
            reader_thread.start()

            while proc.poll() is None:
                elapsed = time.monotonic() - started_at_monotonic
                idle_elapsed = time.monotonic() - last_output_monotonic
                if timeout_sec >= 0 and elapsed > timeout_sec:
                    timed_out = True
                    log_file.write(f"\n[watchdog] timeout_sec exceeded: {timeout_sec}\n")
                    log_file.flush()
                    terminate_process_group(proc.pid)
                    break
                if idle_timeout_sec >= 0 and idle_elapsed > idle_timeout_sec:
                    idle_timed_out = True
                    log_file.write(f"\n[watchdog] idle_timeout_sec exceeded: {idle_timeout_sec}\n")
                    log_file.flush()
                    terminate_process_group(proc.pid)
                    break
                time.sleep(0.5)

            if timed_out or idle_timed_out:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    kill_process_group(proc.pid)

            returncode = proc.wait()
            reader_thread.join(timeout=5)
            resource_summary = monitor.stop()
            finished_at_iso = now_iso()
            log_file.write(f"\n[watchdog] finished_at={finished_at_iso}\n")
            log_file.write(f"[watchdog] exit_code={returncode}\n")
            log_file.flush()
    finally:
        clear_active_training_lock(run_id)

    watchdog_text = read_text(watchdog_log)
    hydra_output_dir = discover_hydra_output_dir(watchdog_text, started_at_wall)
    hydra_log = discover_hydra_log(hydra_output_dir)
    hydra_text = read_text(hydra_log)
    combined_text = "\n".join(part for part in (watchdog_text, hydra_text) if part)

    metrics_csv = discover_metrics_csv(hydra_output_dir)
    metrics = parse_metrics_csv(metrics_csv)
    best_checkpoint_path = extract_first_path(BEST_CKPT_RE, combined_text) or extract_first_path(
        RESTORE_CKPT_RE, combined_text
    )
    status = classify_status(
        returncode=returncode,
        text=combined_text,
        timed_out=timed_out,
        idle_timed_out=idle_timed_out,
        resource_summary=resource_summary,
    )
    failure_evidence = build_failure_evidence(status, combined_text, returncode)

    duration_sec = time.monotonic() - started_at_monotonic
    result = {
        "ok": status == "success",
        "status": status,
        "run_id": run_id,
        "duration_sec": round(duration_sec, 3),
        "exit_code": returncode,
        "command": command,
        "overrides": overrides,
        "report_request": report_request,
        "run_dir": str(run_dir),
        "log_path": str(hydra_log if hydra_log and hydra_log.exists() else watchdog_log),
        "watchdog_log_path": str(watchdog_log),
        "hydra_output_dir": str(hydra_output_dir) if hydra_output_dir else None,
        "metrics_csv_path": str(metrics_csv) if metrics_csv else None,
        "metrics": metrics,
        "best_checkpoint_path": best_checkpoint_path,
        "resource_summary": resource_summary,
        "failure_evidence": failure_evidence,
        "log_excerpt": {
            "tail": tail_lines(combined_text, log_tail_lines),
            "tracebacks": extract_tracebacks(combined_text),
        },
        "started_at": started_at_iso,
        "finished_at": finished_at_iso,
    }
    manifest_path = run_dir / "manifest.json"
    result["manifest_json_path"] = str(manifest_path)
    write_json(manifest_path, build_manifest(result))
    return result
