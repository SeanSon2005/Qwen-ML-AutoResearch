#!/usr/bin/env python3
"""Launch stage-bounded Qwen autoresearch sessions forever."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


QWEN_BIN = "qwen"
PROMPT_ROOT = Path(".qwen") / "prompts"
EXPERIMENTS_ROOT = Path("experiments")
VALID_FINISHED_STATUSES = {"keep", "discard"}


@dataclass(frozen=True)
class ExperimentSnapshot:
    all_ids: set[str]
    finished_ids: set[str]
    running_ids: set[str]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def read_prompt(root: Path, stage: str) -> str:
    prompt_path = root / PROMPT_ROOT / f"{stage}.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt file: {prompt_path}") from exc


def load_experiment(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment JSON must be an object: {path}")
    return data


def experiment_snapshot(root: Path) -> ExperimentSnapshot:
    all_ids: set[str] = set()
    finished_ids: set[str] = set()
    running_ids: set[str] = set()

    experiments_root = root / EXPERIMENTS_ROOT
    if not experiments_root.exists():
        return ExperimentSnapshot(all_ids=all_ids, finished_ids=finished_ids, running_ids=running_ids)

    for experiment_path in sorted(experiments_root.glob("EXP-*/experiment.json")):
        try:
            experiment = load_experiment(experiment_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read experiment record {experiment_path}: {exc}") from exc

        experiment_id = str(experiment.get("experiment_id") or experiment_path.parent.name)
        all_ids.add(experiment_id)

        status = experiment.get("status")
        if status == "running":
            running_ids.add(experiment_id)
        elif status in VALID_FINISHED_STATUSES:
            finished_ids.add(experiment_id)

    return ExperimentSnapshot(all_ids=all_ids, finished_ids=finished_ids, running_ids=running_ids)


def choose_stage(snapshot: ExperimentSnapshot) -> str:
    return "init" if not snapshot.all_ids else "loop"


class StreamRenderer:
    def __init__(self) -> None:
        self._line_open = False
        self._thinking_open = False
        self._active_tools: dict[int, dict[str, Any]] = {}
        self._tool_names_by_id: dict[str, str] = {}

    def _ensure_newline(self) -> None:
        if self._line_open:
            print(flush=True)
            self._line_open = False

    def _close_thinking(self) -> None:
        if self._thinking_open:
            print(" done", flush=True)
            self._thinking_open = False

    def render(self, payload: dict[str, Any]) -> None:
        payload_type = payload.get("type")

        if payload_type == "system" and payload.get("subtype") == "init":
            self._ensure_newline()
            print(
                "[qwen] session "
                f"{payload.get('session_id')} | model {payload.get('model')} | "
                f"tools {len(payload.get('tools') or [])}",
                flush=True,
            )
            return

        if payload_type == "stream_event":
            self._render_stream_event(payload.get("event") or {})
            return

        if payload_type == "user":
            self._render_user_message(payload.get("message") or {})
            return

        if payload_type == "result":
            self._close_thinking()
            self._ensure_newline()
            subtype = payload.get("subtype")
            duration_ms = payload.get("duration_ms")
            print(f"[qwen] result: {subtype} ({duration_ms} ms)", flush=True)
            return

        if payload_type in {"tool_use", "tool_result"}:
            self._close_thinking()
            self._ensure_newline()
            print(f"[qwen] {payload_type}: {payload.get('name') or payload.get('tool_name')}", flush=True)

    def _render_stream_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")

        if event_type == "content_block_start":
            self._render_content_block_start(event)
            return

        if event_type == "content_block_delta":
            delta = event.get("delta") or {}
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                self._close_thinking()
                text = str(delta.get("text") or "")
                if text:
                    print(text, end="", flush=True)
                    self._line_open = True
            elif delta_type == "thinking_delta" and not self._thinking_open:
                self._ensure_newline()
                print("[qwen] thinking...", end="", flush=True)
                self._thinking_open = True
            elif delta_type == "input_json_delta":
                index = int(event.get("index") or 0)
                tool = self._active_tools.setdefault(index, {"input_parts": []})
                tool.setdefault("input_parts", []).append(str(delta.get("partial_json") or ""))
            return

        if event_type == "content_block_stop":
            index = int(event.get("index") or 0)
            self._render_tool_input(index)
            self._close_thinking()
            return

        if event_type == "message_stop":
            self._close_thinking()
            self._ensure_newline()

    def finish(self) -> None:
        self._close_thinking()
        self._ensure_newline()

    def _render_content_block_start(self, event: dict[str, Any]) -> None:
        block = event.get("content_block") or {}
        if block.get("type") != "tool_use":
            return

        self._close_thinking()
        self._ensure_newline()

        index = int(event.get("index") or 0)
        tool_id = str(block.get("id") or "")
        name = str(block.get("name") or "tool")
        self._active_tools[index] = {
            "id": tool_id,
            "name": name,
            "input": block.get("input") if isinstance(block.get("input"), dict) else {},
            "input_parts": [],
        }
        if tool_id:
            self._tool_names_by_id[tool_id] = name
        print(f"[qwen] tool: {name}", flush=True)

    def _render_tool_input(self, index: int) -> None:
        tool = self._active_tools.pop(index, None)
        if not tool:
            return

        input_data = tool.get("input") if isinstance(tool.get("input"), dict) else {}
        partial = "".join(tool.get("input_parts") or [])
        if partial:
            try:
                parsed = json.loads(partial)
                if isinstance(parsed, dict):
                    input_data = {**input_data, **parsed}
            except json.JSONDecodeError:
                input_data = {**input_data, "input": partial}

        summary = self._summarize_tool_input(str(tool.get("name") or "tool"), input_data)
        if summary:
            print(f"[qwen] tool input: {summary}", flush=True)

    def _render_user_message(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return

        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            self._close_thinking()
            self._ensure_newline()
            tool_id = str(item.get("tool_use_id") or "")
            name = self._tool_names_by_id.get(tool_id, "tool")
            status = "error" if item.get("is_error") else "ok"
            result = self._one_line(str(item.get("content") or ""), limit=220)
            print(f"[qwen] tool result: {name} {status} | {result}", flush=True)

    def _summarize_tool_input(self, name: str, input_data: dict[str, Any]) -> str:
        if not input_data:
            return ""

        if name == "run_shell_command":
            command = input_data.get("command") or input_data.get("cmd")
            description = input_data.get("description")
            if command and description:
                return f"{self._one_line(str(command), 180)} ({self._one_line(str(description), 120)})"
            if command:
                return self._one_line(str(command), 220)

        for key in ("file_path", "path", "pattern", "query", "glob", "command"):
            if key in input_data:
                return f"{key}={self._one_line(str(input_data[key]), 220)}"

        compact = json.dumps(input_data, sort_keys=True)
        return self._one_line(compact, 260)

    @staticmethod
    def _one_line(value: str, limit: int) -> str:
        compact = " ".join(value.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."


def render_qwen_stream(proc: subprocess.Popen[str]) -> int:
    renderer = StreamRenderer()
    assert proc.stdout is not None

    try:
        for line in proc.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                renderer.finish()
                print(line, end="", flush=True)
                continue
            renderer.render(payload)
    finally:
        renderer.finish()

    return proc.wait()


def run_qwen(root: Path, stage: str) -> int:
    prompt = read_prompt(root, stage)
    cmd = [
        QWEN_BIN,
        "--approval-mode",
        "yolo",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        prompt,
    ]
    print(
        "\n[orchestrator] starting "
        f"{stage} stage: {QWEN_BIN} --approval-mode yolo --output-format stream-json <prompt>",
        flush=True,
    )
    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
        bufsize=1,
    )
    try:
        return render_qwen_stream(proc)
    except KeyboardInterrupt:
        print("\n[orchestrator] interrupt received; stopping Qwen...", flush=True)
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        elapsed = time.monotonic() - started
        print(f"[orchestrator] {stage} qwen process ended after {elapsed:.1f}s", flush=True)


def validate_stage_result(stage: str, before: ExperimentSnapshot, after: ExperimentSnapshot) -> None:
    if after.running_ids:
        running = ", ".join(sorted(after.running_ids))
        raise RuntimeError(f"{stage} stage left running experiment(s): {running}")

    if stage == "init":
        if not after.finished_ids:
            raise RuntimeError("init stage did not produce a finished experiment")
        return

    new_ids = after.all_ids - before.all_ids
    if len(new_ids) != 1:
        raise RuntimeError(
            "loop stage must produce exactly one new experiment record; "
            f"found {len(new_ids)} new experiment(s): {sorted(new_ids)}"
        )

    new_experiment_id = next(iter(new_ids))
    if new_experiment_id not in after.finished_ids:
        raise RuntimeError(
            "loop stage produced a new experiment that was not finished as "
            f"`keep` or `discard`: {new_experiment_id}"
        )


def main() -> int:
    root = repo_root()
    print(f"[orchestrator] repo root: {root}", flush=True)

    while True:
        before = experiment_snapshot(root)
        stage = choose_stage(before)

        try:
            returncode = run_qwen(root, stage)
        except KeyboardInterrupt:
            print("[orchestrator] stopped by user", flush=True)
            return 130

        if returncode != 0:
            print(f"[orchestrator] {stage} stage failed with exit code {returncode}", file=sys.stderr)
            return returncode

        try:
            after = experiment_snapshot(root)
            validate_stage_result(stage, before, after)
        except RuntimeError as exc:
            print(f"[orchestrator] {stage} stage validation failed: {exc}", file=sys.stderr)
            return 1

        print(f"[orchestrator] {stage} stage complete", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
