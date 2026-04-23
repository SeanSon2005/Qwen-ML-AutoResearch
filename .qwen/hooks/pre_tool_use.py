#!/usr/bin/env python3
"""Qwen PreToolUse hook enforcing project hard-lock policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BIN_DIR = SCRIPT_DIR.parent / "bin"
if str(BIN_DIR) not in sys.path:
    sys.path.insert(0, str(BIN_DIR))

from qpolicy_lib import (  # noqa: E402
    canonicalize_tool_name,
    extract_tool_path,
    fail_closed_pretool,
    is_writable_path,
    make_pretool_output,
    parse_json_stdin,
    resolve_repo_path,
    shell_policy_decision,
    writable_roots_strings,
)


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
        if not is_writable_path(target):
            allowed = "\n".join(f"- {root}" for root in writable_roots_strings())
            return fail_closed_pretool(
                "Write operations are restricted to approved roots. "
                f"Blocked path: {target}.\nAllowed roots:\n{allowed}"
            )

        return _allow("Write path is within approved writable roots.")

    if tool_name == "run_shell_command":
        command = ""
        if isinstance(tool_input, dict):
            command = str(tool_input.get("command", "")).strip()

        allowed, reason = shell_policy_decision(command)
        if allowed:
            return _allow(reason)

        return fail_closed_pretool(reason)

    # Other tools are not write-capable under this policy and are allowed.
    return _allow("Tool allowed by default policy.")


if __name__ == "__main__":
    raise SystemExit(main())
