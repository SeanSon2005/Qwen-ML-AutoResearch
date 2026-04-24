#!/usr/bin/env python3
"""Qwen PreToolUse hook enforcing project write-root policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = ROOT / "training-lightning-hydra"
ACTIVE_TRAINING_LOCK = ROOT / ".qwen" / "state" / "active_training.lock"

WRITABLE_ROOTS = [
    ROOT / ".qwen" / "agents",
    ROOT / "experiments",
    ROOT / "src",
    ROOT / "tools" / "result-logger-mcp",
    ROOT / "tools" / "train-watchdog-mcp",
    TRAINING_ROOT / "src" / "data",
    TRAINING_ROOT / "src" / "data" / "components",
    TRAINING_ROOT / "src" / "models",
    TRAINING_ROOT / "src" / "models" / "components",
    TRAINING_ROOT / "src" / "utils",
    TRAINING_ROOT / "configs",
    TRAINING_ROOT / "tests",
]

WRITABLE_FILES: list[Path] = []

TRAINING_LOCKED_ROOTS = [
    TRAINING_ROOT / "src",
    TRAINING_ROOT / "configs",
    TRAINING_ROOT / "tests",
]


def parse_json_stdin() -> dict | None:
    raw = sys.stdin.read()
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def canonicalize_tool_name(tool_name: str) -> str:
    normalized = tool_name.strip().lower()
    mapping = {
        "writefile": "write_file",
        "write_file": "write_file",
        "edit": "edit",
        "editfile": "edit",
    }
    return mapping.get(normalized, normalized)


def extract_tool_path(tool_input: object) -> str | None:
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


def resolve_repo_path(path_str: str) -> Path:
    raw = Path(path_str).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (ROOT / raw).resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def is_writable_path(path: Path) -> bool:
    resolved = path.resolve()
    if any(resolved == file_path.resolve() for file_path in WRITABLE_FILES):
        return True
    return any(is_within(resolved, root.resolve()) for root in WRITABLE_ROOTS)


def is_training_locked_path(path: Path) -> bool:
    resolved = path.resolve()
    return any(is_within(resolved, root.resolve()) for root in TRAINING_LOCKED_ROOTS)


def active_training_lock_exists() -> bool:
    return ACTIVE_TRAINING_LOCK.exists()


def writable_roots_strings() -> list[str]:
    return [str(path) for path in WRITABLE_ROOTS + WRITABLE_FILES]


def make_pretool_output(decision: str, reason: str) -> dict:
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


def _allow(reason: str) -> int:
    print(json.dumps(make_pretool_output("allow", reason)))
    return 0


def main() -> int:
    payload = parse_json_stdin()
    if payload is None:
        return fail_closed_pretool("Hook received invalid or empty JSON input.")

    tool_name_raw = str(payload.get("tool_name", "")).strip()
    tool_name = canonicalize_tool_name(tool_name_raw)
    tool_input = payload.get("tool_input", {})

    if tool_name in {"edit", "write_file"}:
        path_str = extract_tool_path(tool_input)
        if not path_str:
            return fail_closed_pretool(
                "Edit/Write request did not include a file path. Operation denied by policy."
            )

        target = resolve_repo_path(path_str)
        if active_training_lock_exists() and is_training_locked_path(target):
            return fail_closed_pretool(
                "Training is currently active. Edits to training source, configs, "
                f"and tests are blocked until the active train_run finishes. Blocked path: {target}."
            )

        if not is_writable_path(target):
            allowed = "\n".join(f"- {root}" for root in writable_roots_strings())
            return fail_closed_pretool(
                "Write operations are restricted to approved roots. "
                f"Blocked path: {target}.\nAllowed roots:\n{allowed}"
            )

        return _allow("Write path is within approved writable roots.")

    # This hook only enforces write permissions. Read-only tools and shell
    # commands are allowed here; command safety belongs outside this hook.
    return _allow("Tool allowed by default policy.")


if __name__ == "__main__":
    raise SystemExit(main())
