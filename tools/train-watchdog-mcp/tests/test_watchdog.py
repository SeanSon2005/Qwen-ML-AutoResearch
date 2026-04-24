import io
import json
from pathlib import Path

from train_watchdog_mcp.watchdog import (
    build_failure_evidence,
    build_manifest,
    classify_status,
    discover_hydra_log,
    discover_metrics_csv,
    extract_tracebacks,
    parse_metrics_csv,
    train_run,
    write_json,
)


def patch_train_run_roots(monkeypatch, tmp_path: Path) -> None:
    import train_watchdog_mcp.watchdog as watchdog

    monkeypatch.setattr(watchdog, "STATE_ROOT", tmp_path / "train_runs")
    monkeypatch.setattr(watchdog, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    monkeypatch.setattr(watchdog, "TRAINING_ROOT", tmp_path / "training-lightning-hydra")
    monkeypatch.setattr(watchdog, "ACTIVE_TRAINING_LOCK", tmp_path / "active_training.lock")


def write_experiment(tmp_path: Path, experiment_id: str, status: str = "running") -> None:
    experiment_path = tmp_path / "experiments" / experiment_id / "experiment.json"
    experiment_path.parent.mkdir(parents=True)
    experiment_path.write_text(
        json.dumps({"experiment_id": experiment_id, "status": status}),
        encoding="utf-8",
    )


class FakeProcess:
    def __init__(self, returncode: int = 0) -> None:
        self.pid = 12345
        self.returncode = returncode
        self.stdout = io.StringIO("training output\n")

    def poll(self) -> int:
        return self.returncode

    def wait(self, timeout: int | None = None) -> int:
        return self.returncode


class FakeMonitor:
    def __init__(self, pid: int, interval_sec: int) -> None:
        self.pid = pid
        self.interval_sec = interval_sec

    def start(self) -> None:
        pass

    def stop(self) -> dict[str, float]:
        return {"peak_gpu_memory_gb": 0.0, "gpu_memory_total_gb": 0.0}


def patch_fake_training(monkeypatch, returncode: int) -> None:
    import train_watchdog_mcp.watchdog as watchdog

    def fake_popen(*args, **kwargs):
        return FakeProcess(returncode=returncode)

    monkeypatch.setattr(watchdog.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(watchdog, "ResourceMonitor", FakeMonitor)


def test_train_run_requires_experiment_id_without_starting_subprocess(
    monkeypatch, tmp_path: Path
) -> None:
    patch_train_run_roots(monkeypatch, tmp_path)

    def fail_popen(*args, **kwargs):
        raise AssertionError("training subprocess should not start")

    import train_watchdog_mcp.watchdog as watchdog

    monkeypatch.setattr(watchdog.subprocess, "Popen", fail_popen)

    result = train_run(experiment_id=None, overrides=[])

    assert result == {"ok": False, "error": "missing_experiment_id"}


def test_train_run_rejects_invalid_or_nonexistent_experiment(monkeypatch, tmp_path: Path) -> None:
    patch_train_run_roots(monkeypatch, tmp_path)

    invalid = train_run(experiment_id="bad-id", overrides=[])
    missing = train_run(experiment_id="EXP-999999", overrides=[])

    assert invalid["error"] == "invalid_experiment_id"
    assert missing["error"] == "experiment_not_found"


def test_train_run_rejects_non_running_experiment(monkeypatch, tmp_path: Path) -> None:
    patch_train_run_roots(monkeypatch, tmp_path)
    write_experiment(tmp_path, "EXP-000001", status="discard")

    result = train_run(experiment_id="EXP-000001", overrides=[])

    assert result["ok"] is False
    assert result["error"] == "experiment_not_running"
    assert result["current_status"] == "discard"


def test_train_run_writes_experiment_id_and_clears_lock_on_success(
    monkeypatch, tmp_path: Path
) -> None:
    patch_train_run_roots(monkeypatch, tmp_path)
    patch_fake_training(monkeypatch, returncode=0)
    write_experiment(tmp_path, "EXP-000001")

    result = train_run(experiment_id="EXP-000001", overrides=["trainer=gpu"])

    assert result["ok"] is True
    assert result["experiment_id"] == "EXP-000001"
    assert not (tmp_path / "active_training.lock").exists()
    manifest = json.loads(Path(result["manifest_json_path"]).read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == "EXP-000001"


def test_train_run_clears_lock_on_failure(monkeypatch, tmp_path: Path) -> None:
    patch_train_run_roots(monkeypatch, tmp_path)
    patch_fake_training(monkeypatch, returncode=1)
    write_experiment(tmp_path, "EXP-000001")

    result = train_run(experiment_id="EXP-000001", overrides=[])

    assert result["ok"] is False
    assert result["status"] == "failed"
    assert not (tmp_path / "active_training.lock").exists()


def test_parse_sparse_metrics_csv_with_arbitrary_names(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    metrics_path.write_text(
        "epoch,step,dice/score,custom loss,weird_metric\n"
        "0,1,0.1,,5\n"
        "0,2,,0.9,\n"
        "1,3,0.4,0.7,6\n",
        encoding="utf-8",
    )

    parsed = parse_metrics_csv(metrics_path)

    assert parsed["final"] == {
        "dice/score": 0.4,
        "custom loss": 0.7,
        "weird_metric": 6.0,
    }
    assert parsed["series"]["dice/score"] == [
        {"value": 0.1, "epoch": 0, "step": 1},
        {"value": 0.4, "epoch": 1, "step": 3},
    ]


def test_discover_metrics_csv_supports_direct_and_versioned_paths(tmp_path: Path) -> None:
    direct = tmp_path / "csv" / "metrics.csv"
    direct.parent.mkdir()
    direct.write_text("epoch,step,m\n", encoding="utf-8")
    assert discover_metrics_csv(tmp_path) == direct

    direct.unlink()
    versioned = tmp_path / "csv" / "version_0" / "metrics.csv"
    versioned.parent.mkdir()
    versioned.write_text("epoch,step,m\n", encoding="utf-8")
    assert discover_metrics_csv(tmp_path) == versioned


def test_discover_hydra_log_supports_task_specific_log_names(tmp_path: Path) -> None:
    debug_log = tmp_path / "debug.log"
    debug_log.write_text("debug", encoding="utf-8")

    assert discover_hydra_log(tmp_path) == debug_log

    train_log = tmp_path / "train.log"
    train_log.write_text("train", encoding="utf-8")

    assert discover_hydra_log(tmp_path) == train_log


def test_extract_traceback_block() -> None:
    text = """python src/train.py
Traceback (most recent call last):
  File "/repo/training-lightning-hydra/src/train.py", line 52, in <module>
    from src.utils import (
ModuleNotFoundError: No module named 'src'
[2026-04-24][next][INFO] - message
"""

    tracebacks = extract_tracebacks(text)

    assert len(tracebacks) == 1
    assert "ModuleNotFoundError: No module named 'src'" in tracebacks[0]
    assert "[2026-04-24]" not in tracebacks[0]


def test_status_classification() -> None:
    resources = {"peak_gpu_memory_gb": 0.0, "gpu_memory_total_gb": 0.0}

    assert classify_status(
        returncode=0,
        text="done",
        timed_out=False,
        idle_timed_out=False,
        resource_summary=resources,
    ) == "success"
    assert classify_status(
        returncode=1,
        text="CUDA out of memory",
        timed_out=False,
        idle_timed_out=False,
        resource_summary=resources,
    ) == "oom"
    assert classify_status(
        returncode=1,
        text="error",
        timed_out=True,
        idle_timed_out=False,
        resource_summary=resources,
    ) == "timeout"
    assert classify_status(
        returncode=1,
        text="error",
        timed_out=False,
        idle_timed_out=True,
        resource_summary=resources,
    ) == "idle_timeout"
    assert classify_status(
        returncode=1,
        text="error",
        timed_out=False,
        idle_timed_out=False,
        resource_summary=resources,
    ) == "failed"


def test_failure_evidence_includes_non_traceback_error_excerpt() -> None:
    evidence = build_failure_evidence(
        "failed",
        "Could not override 'trainer.limit_train_batches'.\nKey is not in struct\n",
        1,
    )

    assert "status=failed" in evidence
    assert "exit_code=1" in evidence
    assert any("Could not override" in item for item in evidence)


def test_write_json_writes_structured_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "run-1" / "manifest.json"

    write_json(manifest_path, {"ok": True, "status": "success"})

    assert manifest_path.read_text(encoding="utf-8").strip().startswith("{")
    assert '"status": "success"' in manifest_path.read_text(encoding="utf-8")


def test_build_manifest_omits_heavy_result_fields() -> None:
    manifest = build_manifest(
        {
            "run_id": "run-1",
            "status": "success",
            "ok": True,
            "metrics": {"series": {"m": [{"value": 1.0}]}},
            "log_excerpt": {"tail": "long log"},
            "resource_summary": {"peak_ram_gb": 1.0},
            "metrics_csv_path": "metrics.csv",
        }
    )

    assert manifest["run_id"] == "run-1"
    assert manifest["metrics_csv_path"] == "metrics.csv"
    assert "metrics" not in manifest
    assert "log_excerpt" not in manifest
    assert "resource_summary" not in manifest
