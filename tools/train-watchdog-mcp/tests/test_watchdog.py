from pathlib import Path

from train_watchdog_mcp.watchdog import (
    build_failure_evidence,
    build_manifest,
    classify_status,
    discover_hydra_log,
    discover_metrics_csv,
    extract_tracebacks,
    parse_metrics_csv,
    write_json,
)


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
