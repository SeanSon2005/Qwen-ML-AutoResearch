import json
from pathlib import Path

import result_logger_mcp.logger as logger


def patch_roots(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(logger, "EXPERIMENTS_ROOT", tmp_path / "experiments")
    monkeypatch.setattr(logger, "TRAIN_RUNS_ROOT", tmp_path / "train_runs")


def test_experiment_create_allocates_sequential_ids(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)

    first = logger.experiment_create(
        hypothesis="try cnn",
        decision_type="architecture",
        description="first",
    )
    assert first["ok"] is True
    assert first["experiment_id"] == "EXP-000001"

    finish = logger.experiment_finish(
        experiment_id="EXP-000001",
        commit="abcdef1",
        status="keep",
        metric=0.5,
        description="done",
    )
    assert finish["ok"] is True

    second = logger.experiment_create(
        hypothesis="try wider cnn",
        decision_type="architecture",
        description="second",
    )
    assert second["experiment_id"] == "EXP-000002"


def test_experiment_create_fails_when_one_is_running(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)

    created = logger.experiment_create(
        hypothesis="try cnn",
        decision_type="architecture",
        description="first",
    )
    blocked = logger.experiment_create(
        hypothesis="try cnn again",
        decision_type="architecture",
        description="second",
    )

    assert created["ok"] is True
    assert blocked["ok"] is False
    assert blocked["error"] == "experiment_already_running"


def test_experiment_finish_infers_train_runs_in_window(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)
    monkeypatch.setattr(
        logger,
        "now_iso",
        iter(
            [
                "2026-04-24T00:00:00+00:00",
                "2026-04-24T00:10:00+00:00",
            ]
        ).__next__,
    )
    before_dir = logger.TRAIN_RUNS_ROOT / "run-before"
    inside_dir = logger.TRAIN_RUNS_ROOT / "run-inside"
    after_dir = logger.TRAIN_RUNS_ROOT / "run-after"
    before_dir.mkdir(parents=True)
    inside_dir.mkdir(parents=True)
    after_dir.mkdir(parents=True)
    (before_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-before", "started_at": "2026-04-23T23:59:00+00:00"}),
        encoding="utf-8",
    )
    (inside_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-inside",
                "status": "success",
                "started_at": "2026-04-24T00:05:00+00:00",
                "manifest_json_path": str(inside_dir / "manifest.json"),
            }
        ),
        encoding="utf-8",
    )
    (after_dir / "manifest.json").write_text(
        json.dumps({"run_id": "run-after", "started_at": "2026-04-24T00:11:00+00:00"}),
        encoding="utf-8",
    )

    created = logger.experiment_create(
        hypothesis="try cnn",
        decision_type="architecture",
        description="first",
    )
    finished = logger.experiment_finish(
        experiment_id=created["experiment_id"],
        commit="abcdef1",
        status="keep",
        metric=0.7,
        description="done",
    )

    assert finished["ok"] is True
    assert finished["assigned_train_run_ids"] == ["run-inside"]
    assert finished["experiment"]["train_runs"][0]["status"] == "success"
    assert finished["experiment"]["train_runs"][0]["manifest_json_path"].endswith(
        "manifest.json"
    )


def test_load_train_run_supports_legacy_result_json(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)
    run_dir = logger.TRAIN_RUNS_ROOT / "run-legacy"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(
        json.dumps(
            {
                "run_id": "run-legacy",
                "status": "success",
                "started_at": "2026-04-24T00:05:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    run = logger.load_train_run(run_dir)

    assert run is not None
    assert run["run_id"] == "run-legacy"
    assert run["result_json_path"].endswith("result.json")


def test_experiment_finish_rejects_invalid_status(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)
    created = logger.experiment_create(
        hypothesis="try cnn",
        decision_type="architecture",
        description="first",
    )

    result = logger.experiment_finish(
        experiment_id=created["experiment_id"],
        commit="abcdef1",
        status="success",
        metric=0.7,
        description="done",
    )

    assert result["ok"] is False
    assert result["error"] == "invalid_status"


def test_experiments_list_returns_newest_first(monkeypatch, tmp_path: Path) -> None:
    patch_roots(monkeypatch, tmp_path)
    stamps = iter(
        [
            "2026-04-24T00:00:00+00:00",
            "2026-04-24T00:01:00+00:00",
            "2026-04-24T00:02:00+00:00",
            "2026-04-24T00:03:00+00:00",
        ]
    )
    monkeypatch.setattr(logger, "now_iso", stamps.__next__)

    first = logger.experiment_create(
        hypothesis="a",
        decision_type="architecture",
        description="first",
    )
    logger.experiment_finish(
        experiment_id=first["experiment_id"],
        commit="abcdef1",
        status="discard",
        metric=0.1,
        description="done",
    )
    second = logger.experiment_create(
        hypothesis="b",
        decision_type="architecture",
        description="second",
    )
    logger.experiment_finish(
        experiment_id=second["experiment_id"],
        commit="abcdef2",
        status="keep",
        metric=0.2,
        description="done",
    )

    listed = logger.experiments_list()

    assert [item["experiment_id"] for item in listed["experiments"]] == [
        "EXP-000002",
        "EXP-000001",
    ]
