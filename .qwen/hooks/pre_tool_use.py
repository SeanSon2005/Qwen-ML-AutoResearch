#!/usr/bin/env python3
"""Qwen PreToolUse hook enforcing project write-root policy."""

from __future__ import annotations

import json
import re
import shlex
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

SHELL_TOOL_NAMES = {
    "bash",
    "shell",
    "terminal",
    "run_shell_command",
    "exec_command",
}

SHELL_WRITE_COMMANDS = {
    "cp",
    "install",
    "ln",
    "mkdir",
    "mv",
    "rm",
    "rmdir",
    "tee",
    "touch",
    "truncate",
}

SHELL_EDIT_COMMANDS = {
    "ed",
    "ex",
    "nano",
    "perl",
    "python",
    "python3",
    "ruby",
    "sed",
    "vi",
    "vim",
}

INTERACTIVE_EDIT_COMMANDS = {"ed", "ex", "nano", "vi", "vim"}

SHELL_CONTROL_OPERATORS = {
    "&&",
    "||",
    ";",
    "|",
    "|&",
    "&",
    "(",
    ")",
}

REDIRECT_OPERATORS = {">", ">>", ">|", "<>", "&>", "&>>"}

PYTHON_WRITE_PATTERNS = (
    re.compile(r"""open\(\s*['"](?P<path>[^'"]+)['"]\s*,\s*['"][wax+]"""),
    re.compile(r"""Path\(\s*['"](?P<path>[^'"]+)['"]\s*\)\.write_(?:text|bytes)\("""),
)

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


def extract_shell_command(tool_input: object) -> str | None:
    if isinstance(tool_input, str) and tool_input.strip():
        return tool_input
    if not isinstance(tool_input, dict):
        return None

    for key in ("command", "cmd", "script", "shell_command"):
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


def check_write_target(path_str: str) -> str | None:
    target = resolve_repo_path(path_str)
    if active_training_lock_exists() and is_training_locked_path(target):
        return (
            "Training is currently active. Edits to training source, configs, "
            f"and tests are blocked until the active train_run finishes. Blocked path: {target}."
        )

    if not is_writable_path(target):
        allowed = "\n".join(f"- {root}" for root in writable_roots_strings())
        return (
            "Write operations are restricted to approved roots. "
            f"Blocked path: {target}.\nAllowed roots:\n{allowed}"
        )
    return None


def split_shell_tokens(command: str) -> list[str] | None:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        return list(lexer)
    except ValueError:
        return None


def looks_like_option(token: str) -> bool:
    return token.startswith("-") and token not in {"-", "--"}


def command_basename(token: str) -> str:
    assignment = token.split("=", 1)[-1]
    return Path(assignment).name


def command_segments(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in SHELL_CONTROL_OPERATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def command_name(segment: list[str]) -> str | None:
    for token in segment:
        if "=" in token and not token.startswith(("/", ".", "~")):
            continue
        return command_basename(token)
    return None


def shell_write_targets_from_segment(segment: list[str]) -> list[str]:
    targets: list[str] = []
    command = command_name(segment)
    if command is None:
        return targets

    for index, token in enumerate(segment[:-1]):
        if token in REDIRECT_OPERATORS:
            targets.append(segment[index + 1])
        elif token.startswith((">", ">>")) and len(token) > 1:
            targets.append(token.lstrip(">"))

    args = [token for token in segment[1:] if token not in REDIRECT_OPERATORS]
    positional = [token for token in args if not looks_like_option(token)]

    if command in {"cp", "install", "ln", "mv"} and len(positional) >= 2:
        targets.append(positional[-1])
    elif command in {"mkdir", "rm", "rmdir", "touch", "truncate"}:
        targets.extend(positional)
    elif command == "tee":
        targets.extend(positional)
    elif command in INTERACTIVE_EDIT_COMMANDS:
        targets.extend(positional)
    elif command in SHELL_EDIT_COMMANDS and any(arg in {"-i", "--in-place"} or arg.startswith("-i") for arg in args):
        if command == "sed" and len(positional) > 1:
            targets.extend(positional[1:])
        else:
            targets.extend(positional)

    return targets


def validate_shell_command(command: str) -> str | None:
    for pattern in PYTHON_WRITE_PATTERNS:
        for match in pattern.finditer(command):
            reason = check_write_target(match.group("path"))
            if reason:
                return reason

    tokens = split_shell_tokens(command)
    if tokens is None:
        return "Unable to parse shell command safely. Use the edit/write tool for file changes."

    for segment in command_segments(tokens):
        for target in shell_write_targets_from_segment(segment):
            reason = check_write_target(target)
            if reason:
                return reason
    return None


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

        reason = check_write_target(path_str)
        if reason:
            return fail_closed_pretool(reason)

        return _allow("Write path is within approved writable roots.")

    if tool_name in SHELL_TOOL_NAMES:
        command = extract_shell_command(tool_input)
        if command:
            reason = validate_shell_command(command)
            if reason:
                return fail_closed_pretool(reason)
        return _allow("Shell command allowed by write-root policy.")

    # This hook only enforces write permissions. Read-only tools are allowed here.
    return _allow("Tool allowed by default policy.")


if __name__ == "__main__":
    raise SystemExit(main())
