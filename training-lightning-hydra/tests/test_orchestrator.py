import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from begin_experiment import (  # noqa: E402
    ACTIVE_TRAINING_LOCK,
    ALLOWED_MCP_SERVER_NAMES,
    Dashboard,
    ExperimentSnapshot,
    StreamRenderer,
    choose_stage,
    load_active_training_state,
    qwen_mcp_allowlist_args,
)


def empty_snapshot() -> ExperimentSnapshot:
    return ExperimentSnapshot(all_ids=set(), finished_ids=set(), running_ids=set(), records={})


def test_stage_selection() -> None:
    assert choose_stage(empty_snapshot()) == "init"
    assert (
        choose_stage(
            ExperimentSnapshot(
                all_ids={"EXP-000001"},
                finished_ids={"EXP-000001"},
                running_ids=set(),
                records={"EXP-000001": {"status": "keep"}},
            )
        )
        == "loop"
    )


def test_qwen_mcp_allowlist_args_include_only_repo_servers() -> None:
    args = qwen_mcp_allowlist_args()

    assert args.count("--allowed-mcp-server-names") == len(ALLOWED_MCP_SERVER_NAMES)
    assert set(args[1::2]) == {"train_watchdog", "result_logger", "paper_search"}
    assert "paper-search-mcp" not in args


def test_dashboard_training_state_handles_absent_and_active_lock(tmp_path: Path) -> None:
    assert load_active_training_state(tmp_path) == {"status": "idle"}

    lock_path = tmp_path / ACTIVE_TRAINING_LOCK
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "experiment_id": "EXP-000001",
                "started_at": "2026-04-26T00:00:00+00:00",
                "overrides": ["trainer.accelerator=gpu"],
                "timeout_sec": 7,
            }
        ),
        encoding="utf-8",
    )

    state = load_active_training_state(tmp_path)

    assert state["status"] == "running"
    assert state["run_id"] == "run-1"
    assert state["experiment_id"] == "EXP-000001"
    assert state["elapsed"] is not None


def test_stream_renderer_formats_key_qwen_events(tmp_path: Path) -> None:
    dashboard = Dashboard(tmp_path, "init", empty_snapshot())
    renderer = StreamRenderer(dashboard)

    renderer.render(
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
            "model": "qwen-test",
            "tools": ["tool-a"],
        }
    )
    renderer.render(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "thinking_delta", "thinking": "plan"},
            },
        }
    )
    renderer.render(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hello"},
            },
        }
    )
    renderer.render(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "tool-1", "name": "run_shell_command"},
            },
        }
    )
    renderer.render(
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "echo hi"}'},
            },
        }
    )
    renderer.render({"type": "stream_event", "event": {"type": "content_block_stop", "index": 0}})
    renderer.render(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "ok", "is_error": False}
                ]
            },
        }
    )
    renderer.render(
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "bad", "is_error": True}
                ]
            },
        }
    )
    renderer.render_line("node warning from stderr\n")
    renderer.render({"type": "result", "subtype": "success", "duration_ms": 123})

    text = "\n".join(line.plain for line in dashboard.log)

    assert "session session-1" in text
    assert "thinking..." in text
    assert "assistant: hello" in text
    assert "tool: run_shell_command" in text
    assert "input: echo hi" in text
    assert "output: run_shell_command ok | ok" in text
    assert "output: run_shell_command error | bad" in text
    assert "node warning from stderr" in text
    assert "result: success (123 ms)" in text
