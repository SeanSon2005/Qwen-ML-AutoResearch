#!/usr/bin/env python3
"""Shared policy/runtime helpers for project-scoped Qwen guardrails."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training-lightning-hydra"
STATE_ROOT = ROOT / ".qwen" / "state" / "train_runs"

WRITABLE_ROOTS = [
    TRAINING_ROOT / "src" / "data",
    TRAINING_ROOT / "src" / "data" / "components",
    TRAINING_ROOT / "src" / "models",
    TRAINING_ROOT / "src" / "models" / "components",
    TRAINING_ROOT / "src" / "utils",
    TRAINING_ROOT / "configs",
    TRAINING_ROOT / "tests",
]

TERMINAL_STATES = {"success", "failed", "timeout", "crashed"}

ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
DIRECT_TRAIN_RE = re.compile(r"(^|\s)-m\s+src\.train(\s|$)")

HELPER_COMMAND_ALIASES = {
    str(ROOT / ".qwen" / "bin" / "qtrain"),
    str(ROOT / ".qwen" / "bin" / "qtest"),
    str(ROOT / ".qwen" / "bin" / "qfs"),
    "./.qwen/bin/qtrain",
    "./.qwen/bin/qtest",
    "./.qwen/bin/qfs",
    ".qwen/bin/qtrain",
    ".qwen/bin/qtest",
    ".qwen/bin/qfs",
}

GIT_ALLOWED_SUBCOMMANDS = {
    "status",
    "diff",
    "show",
    "log",
    "branch",
    "rev-parse",
    "ls-files",
    "add",
    "commit",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    tmp.replace(path)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_repo_path(path_str: str, *, base: Path | None = None) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    anchor = base or ROOT
    return (anchor / raw).resolve()


def writable_roots_strings() -> list[str]:
    return [str(p) for p in WRITABLE_ROOTS]


def is_writable_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(is_within(resolved, root.resolve()) for root in WRITABLE_ROOTS)


def parse_shell_tokens(command: str) -> list[str] | None:
    try:
        return shlex.split(command)
    except ValueError:
        return None


def strip_env_prefix(tokens: list[str]) -> list[str]:
    idx = 0
    while idx < len(tokens) and ENV_ASSIGNMENT_RE.match(tokens[idx]):
        idx += 1
    return tokens[idx:]


def _is_git_allowed(tokens: list[str]) -> bool:
    if not tokens or tokens[0] != "git":
        return False
    if len(tokens) < 2:
        return False
    sub = tokens[1]
    if sub in GIT_ALLOWED_SUBCOMMANDS:
        return True
    # Special-case: git remote -v is safe and useful.
    if sub == "remote" and len(tokens) >= 3 and tokens[2] == "-v":
        return True
    return False


def _is_helper_command(tokens: list[str]) -> bool:
    if not tokens:
        return False
    cmd = tokens[0]
    if cmd in HELPER_COMMAND_ALIASES:
        return True
    # Allow canonical resolved path for helper command references.
    resolved = str(resolve_repo_path(cmd)) if not Path(cmd).is_absolute() else str(Path(cmd).resolve())
    return resolved in HELPER_COMMAND_ALIASES


def shell_policy_decision(command: str) -> tuple[bool, str]:
    command = command.strip()
    if not command:
        return False, "Empty shell command is not allowed by project policy."

    lowered = command.lower()
    if "src/train.py" in lowered or DIRECT_TRAIN_RE.search(command):
        return False, (
            "Direct training commands are blocked. Use watchdog entrypoint: "
            "./.qwen/bin/qtrain start -- <hydra_overrides...>"
        )

    tokens = parse_shell_tokens(command)
    if tokens is None:
        return False, "Unparseable shell command is blocked by project policy."

    # Reject chaining to prevent bypassing policy with mixed commands.
    if any(tok in {"&&", "||", ";", "|"} for tok in tokens):
        return False, "Command chaining is blocked. Run a single approved command per invocation."

    tokens = strip_env_prefix(tokens)
    if not tokens:
        return False, "Shell command only contains environment assignments."

    if _is_helper_command(tokens):
        return True, "Approved helper command."

    if _is_git_allowed(tokens):
        return True, "Approved git command."

    return False, (
        "Shell command is blocked. Allowed shell usage is limited to helper commands "
        "(./.qwen/bin/qtrain, ./.qwen/bin/qtest, ./.qwen/bin/qfs) and approved git commands."
    )


def extract_tool_path(tool_input: Any) -> str | None:
    if not isinstance(tool_input, dict):
        return None

    for key in (
        "file_path",
        "path",
        "filepath",
        "target_file",
        "target_path",
        "directory_path",
        "dir_path",
    ):
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def make_pretool_output(decision: str, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }


def fail_closed_pretool(reason: str) -> int:
    print(json.dumps(make_pretool_output("deny", reason)))
    return 0


def print_error_json(message: str, *, code: int = 1, extra: dict[str, Any] | None = None) -> int:
    payload: dict[str, Any] = {"ok": False, "error": message}
    if extra:
        payload.update(extra)
    print(pretty_json(payload))
    return code


def run_id_new() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rand = os.urandom(3).hex()
    return f"run-{stamp}-{rand}"


def run_dir(run_id: str) -> Path:
    return STATE_ROOT / run_id


def ensure_state_defaults(state: dict[str, Any]) -> dict[str, Any]:
    state.setdefault("updated_at", now_utc_iso())
    state.setdefault("summary", "Run initialized.")
    return state


def load_state(run_id: str) -> dict[str, Any]:
    state_file = run_dir(run_id) / "state.json"
    state = read_json(state_file)
    return ensure_state_defaults(state)


def save_state(run_id: str, state: dict[str, Any]) -> None:
    state = ensure_state_defaults(state)
    state["updated_at"] = now_utc_iso()
    write_json(run_dir(run_id) / "state.json", state)


def latest_run_id() -> str | None:
    if not STATE_ROOT.exists():
        return None
    run_dirs = [p for p in STATE_ROOT.iterdir() if p.is_dir() and (p / "state.json").exists()]
    if not run_dirs:
        return None
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0].name


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass
class CmdResult:
    returncode: int | None
    output: str
    timed_out: bool


def terminate_process_group(proc_pid: int) -> None:
    try:
        os.killpg(proc_pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def run_command_with_timeout(
    command: list[str],
    *,
    cwd: Path,
    timeout_sec: int | None,
) -> CmdResult:
    import subprocess

    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout_sec)
        return CmdResult(returncode=proc.returncode, output=output or "", timed_out=False)
    except subprocess.TimeoutExpired as exc:
        partial = exc.output or ""
        terminate_process_group(proc.pid)
        try:
            output, _ = proc.communicate(timeout=5)
        except Exception:
            output = ""
        return CmdResult(returncode=None, output=(partial or "") + (output or ""), timed_out=True)


def traceback_summary(prefix: str) -> str:
    return f"{prefix}: {traceback.format_exc().strip()}"


def canonicalize_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip().lower()
    mapping = {
        "writefile": "write_file",
        "write_file": "write_file",
        "edit": "edit",
        "editfile": "edit",
        "run_shell_command": "run_shell_command",
        "shell": "run_shell_command",
        "shelltool": "run_shell_command",
        "bash": "run_shell_command",
    }
    return mapping.get(normalized, normalized)


def parse_json_stdin() -> dict[str, Any] | None:
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def list_stripped(lines: Iterable[str], *, tail: int | None = None) -> str:
    seq = list(lines)
    if tail is not None and tail >= 0:
        seq = seq[-tail:]
    return "".join(seq)
