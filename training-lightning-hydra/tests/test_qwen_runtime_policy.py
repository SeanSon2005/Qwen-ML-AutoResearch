import json
import subprocess
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HOOK = ROOT / ".qwen" / "hooks" / "pre_tool_use.py"
QTRAIN = ROOT / ".qwen" / "bin" / "qtrain"
QTEST = ROOT / ".qwen" / "bin" / "qtest"
QFS = ROOT / ".qwen" / "bin" / "qfs"


def run_cmd(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc


def run_hook(payload: dict) -> dict:
    proc = subprocess.run(
        [str(HOOK)],
        cwd=str(ROOT),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), "Hook produced empty output"
    return json.loads(proc.stdout)


def hook_decision(payload: dict) -> tuple[str, str]:
    response = run_hook(payload)
    hs = response["hookSpecificOutput"]
    return hs["permissionDecision"], hs["permissionDecisionReason"]


def qtrain_json(*args: str, check_code: int | None = None) -> tuple[int, dict]:
    proc = run_cmd([str(QTRAIN), *args], check=False)
    assert proc.stdout.strip(), f"qtrain returned empty stdout: {' '.join(args)}"
    payload = json.loads(proc.stdout)
    if check_code is not None:
        assert proc.returncode == check_code, proc.stdout + proc.stderr
    return proc.returncode, payload


def wait_for_terminal(run_id: str, timeout_sec: int = 180) -> dict:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        code, payload = qtrain_json("status", "--run-id", run_id)
        assert code == 0, payload
        status = payload.get("status")
        if status in {"success", "failed", "timeout", "crashed"}:
            return payload
        time.sleep(1.0)
    raise AssertionError(f"Run did not reach terminal state within {timeout_sec}s: {run_id}")


def test_hook_allows_edit_in_writable_root() -> None:
    path = ROOT / "training-lightning-hydra" / "src" / "data" / "components" / "dataset.py"
    decision, reason = hook_decision(
        {
            "tool_name": "edit",
            "tool_input": {"file_path": str(path)},
            "tool_use_id": "t1",
            "permission_mode": "yolo",
        }
    )
    assert decision == "allow"
    assert "approved writable roots" in reason


def test_hook_denies_edit_outside_writable_roots() -> None:
    decision, reason = hook_decision(
        {
            "tool_name": "edit",
            "tool_input": {"file_path": str(ROOT / "QWEN.md")},
            "tool_use_id": "t2",
            "permission_mode": "yolo",
        }
    )
    assert decision == "deny"
    assert "restricted to approved roots" in reason


def test_hook_denies_direct_training_shell_command() -> None:
    decision, reason = hook_decision(
        {
            "tool_name": "run_shell_command",
            "tool_input": {"command": "uv run python -m src.train trainer=cpu"},
            "tool_use_id": "t3",
            "permission_mode": "yolo",
        }
    )
    assert decision == "deny"
    assert "Direct training commands are blocked" in reason


def test_hook_allows_watchdog_and_readonly_git_shell_commands() -> None:
    allowed_payloads = [
        {
            "tool_name": "run_shell_command",
            "tool_input": {"command": "./.qwen/bin/qtrain status --latest"},
            "tool_use_id": "t4",
            "permission_mode": "yolo",
        },
        {
            "tool_name": "run_shell_command",
            "tool_input": {"command": "git status --short"},
            "tool_use_id": "t5",
            "permission_mode": "yolo",
        },
    ]

    for payload in allowed_payloads:
        decision, _ = hook_decision(payload)
        assert decision == "allow"


def test_qfs_enforces_writable_roots() -> None:
    allowed_rel = Path("training-lightning-hydra/tests/.qwen_policy_tmp/allowed_file.txt")
    allowed_abs = ROOT / allowed_rel
    if allowed_abs.exists():
        allowed_abs.unlink()

    touch_ok = run_cmd([str(QFS), "touch", str(allowed_rel)], check=True)
    assert allowed_abs.exists(), touch_ok.stdout

    rm_ok = run_cmd([str(QFS), "rm", str(allowed_rel)], check=True)
    assert not allowed_abs.exists(), rm_ok.stdout

    denied = run_cmd([str(QFS), "touch", "QWEN.md"], check=False)
    assert denied.returncode != 0
    assert denied.stdout.strip()
    payload = json.loads(denied.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "path_outside_writable_roots"


@pytest.mark.slow
def test_qtrain_lifecycle_success() -> None:
    code, start_payload = qtrain_json(
        "start",
        "--timeout-sec",
        "120",
        "--",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
        "logger=[]",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        check_code=0,
    )
    assert code == 0
    run_id = start_payload["run_id"]

    status_payload = wait_for_terminal(run_id, timeout_sec=180)
    assert status_payload["status"] == "success"

    _, result_payload = qtrain_json("result", "--run-id", run_id, check_code=0)
    assert result_payload["status"] == "success"
    assert result_payload["summary"]
    assert result_payload["log_path"]


@pytest.mark.slow
def test_qtrain_lifecycle_failed() -> None:
    _, start_payload = qtrain_json(
        "start",
        "--timeout-sec",
        "60",
        "--",
        "trainer=does_not_exist",
        check_code=0,
    )
    run_id = start_payload["run_id"]

    status_payload = wait_for_terminal(run_id, timeout_sec=120)
    assert status_payload["status"] == "failed"
    assert status_payload["summary"]


@pytest.mark.slow
def test_qtrain_lifecycle_timeout() -> None:
    _, start_payload = qtrain_json(
        "start",
        "--timeout-sec",
        "0",
        "--",
        "trainer=cpu",
        "++trainer.fast_dev_run=true",
        "logger=[]",
        "extras.enforce_tags=false",
        "extras.print_config=false",
        check_code=0,
    )
    run_id = start_payload["run_id"]

    status_payload = wait_for_terminal(run_id, timeout_sec=60)
    assert status_payload["status"] == "timeout"
    assert status_payload["summary"]


def test_qtrain_result_missing_run_id_returns_structured_error() -> None:
    missing_run = f"missing-{uuid.uuid4().hex[:8]}"
    proc = run_cmd([str(QTRAIN), "result", "--run-id", missing_run], check=False)
    assert proc.returncode != 0
    assert proc.stdout.strip()
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert payload["error"] == "run_id_not_found"
    assert payload["run_id"] == missing_run


def test_qtest_help_output_is_nonempty() -> None:
    proc = run_cmd([str(QTEST), "--help"], check=False)
    assert proc.returncode == 0
    assert proc.stdout.strip()
