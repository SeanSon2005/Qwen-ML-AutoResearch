#!/usr/bin/env python3
"""Launch stage-bounded Qwen autoresearch sessions forever."""

from __future__ import annotations

import json
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


QWEN_BIN = "qwen"
ALLOWED_MCP_SERVER_NAMES = ("train_watchdog", "result_logger", "paper_search")
PROMPT_ROOT = Path(".qwen") / "prompts"
EXPERIMENTS_ROOT = Path("experiments")
ACTIVE_TRAINING_LOCK = Path(".qwen") / "state" / "active_training.lock"
NEXT_AGENT_NOTE = Path(".qwen") / "state" / "next_agent_note.md"
VALID_FINISHED_STATUSES = {"keep", "discard"}
MAX_LOG_LINES = 140

console = Console()


@dataclass(frozen=True)
class ExperimentSnapshot:
    all_ids: set[str]
    finished_ids: set[str]
    running_ids: set[str]
    records: dict[str, dict[str, Any]]


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def read_prompt(root: Path, stage: str) -> str:
    prompt_path = root / PROMPT_ROOT / f"{stage}.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Missing prompt file: {prompt_path}") from exc


def load_experiment(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Experiment JSON must be an object: {path}")
    return data


def experiment_snapshot(root: Path) -> ExperimentSnapshot:
    all_ids: set[str] = set()
    finished_ids: set[str] = set()
    running_ids: set[str] = set()
    records: dict[str, dict[str, Any]] = {}

    experiments_root = root / EXPERIMENTS_ROOT
    if not experiments_root.exists():
        return ExperimentSnapshot(all_ids, finished_ids, running_ids, records)

    for experiment_path in sorted(experiments_root.glob("EXP-*/experiment.json")):
        try:
            experiment = load_experiment(experiment_path)
        except Exception as exc:
            raise RuntimeError(f"Unable to read experiment record {experiment_path}: {exc}") from exc

        experiment_id = str(experiment.get("experiment_id") or experiment_path.parent.name)
        all_ids.add(experiment_id)
        records[experiment_id] = experiment

        status = experiment.get("status")
        if status == "running":
            running_ids.add(experiment_id)
        elif status in VALID_FINISHED_STATUSES:
            finished_ids.add(experiment_id)

    return ExperimentSnapshot(all_ids, finished_ids, running_ids, records)


def choose_stage(snapshot: ExperimentSnapshot) -> str:
    return "init" if not snapshot.all_ids else "loop"


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def parse_iso_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def elapsed_since(value: Any) -> float | None:
    parsed = parse_iso_datetime(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - parsed.astimezone(timezone.utc)).total_seconds()


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def next_experiment_label(snapshot: ExperimentSnapshot) -> str:
    if snapshot.running_ids:
        return ", ".join(sorted(snapshot.running_ids))
    return f"EXP-{len(snapshot.all_ids) + 1:06d}"


def current_experiment_elapsed(snapshot: ExperimentSnapshot) -> str:
    elapsed_values = [
        elapsed_since(snapshot.records[experiment_id].get("started_at"))
        or elapsed_since(snapshot.records[experiment_id].get("created_at"))
        for experiment_id in snapshot.running_ids
        if experiment_id in snapshot.records
    ]
    elapsed_values = [value for value in elapsed_values if value is not None]
    if not elapsed_values:
        return "-"
    return format_duration(max(elapsed_values))


def load_active_training_state(root: Path) -> dict[str, Any]:
    lock_path = root / ACTIVE_TRAINING_LOCK
    if not lock_path.exists():
        return {"status": "idle"}
    try:
        with lock_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return {"status": "unreadable", "error": str(exc)}
    if not isinstance(data, dict):
        return {"status": "unreadable", "error": "lock was not a JSON object"}
    data["status"] = "running"
    data["elapsed"] = elapsed_since(data.get("started_at"))
    return data


def read_next_agent_note(root: Path, limit: int = 160) -> str:
    note_path = root / NEXT_AGENT_NOTE
    if not note_path.exists():
        return "-"
    try:
        text = " ".join(note_path.read_text(encoding="utf-8").split())
    except Exception as exc:
        return f"unreadable: {exc}"
    if not text:
        return "-"
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Dashboard:
    def __init__(self, root: Path, stage: str, before: ExperimentSnapshot) -> None:
        self.root = root
        self.stage = stage
        self.before = before
        self.started = time.monotonic()
        self.commit = git_commit(root)
        self.log: list[Text] = []
        self._assistant_line: Text | None = None

    def append(self, message: str, style: str = "") -> Text:
        line = Text()
        line.append(time.strftime("%H:%M:%S"), style="dim")
        line.append(" ")
        line.append(message, style=style)
        self.log.append(line)
        if len(self.log) > MAX_LOG_LINES:
            self.log = self.log[-MAX_LOG_LINES:]
        self._assistant_line = None
        return line

    def assistant_append(self, text: str) -> None:
        if not text:
            return
        if self._assistant_line is None:
            self._assistant_line = self.append("assistant: ")
        self._assistant_line.append(text)

    def render(self) -> RenderableType:
        return Group(self._header(), self._log_panel())

    def _header(self) -> Panel:
        snapshot = experiment_snapshot(self.root)
        active_train = load_active_training_state(self.root)

        train_status = str(active_train.get("status") or "idle")
        run_id = str(active_train.get("run_id") or "-")
        train_elapsed = format_duration(active_train.get("elapsed"))
        experiment_id = active_train.get("experiment_id") or active_train.get("running_experiment_id")
        if experiment_id:
            train_status = f"{train_status} ({experiment_id})"

        table = Table.grid(expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_column(ratio=1)
        table.add_row(
            self._field("stage", self.stage),
            self._field("experiment", next_experiment_label(snapshot)),
            self._field("qwen elapsed", format_duration(time.monotonic() - self.started)),
        )
        table.add_row(
            self._field("commit", self.commit),
            self._field("finished/running", f"{len(snapshot.finished_ids)}/{len(snapshot.running_ids)}"),
            self._field("experiment elapsed", current_experiment_elapsed(snapshot)),
        )
        table.add_row(
            self._field("train watchdog", train_status),
            self._field("train run", run_id),
            self._field("train elapsed", train_elapsed),
        )
        table.add_row(
            self._field("next note", read_next_agent_note(self.root)),
            "",
            "",
        )
        return Panel(table, title="Qwen Autoresearch Orchestrator", border_style="bright_blue")

    def _log_panel(self) -> Panel:
        body: RenderableType
        if self.log:
            body = Group(*self.log[-MAX_LOG_LINES:])
        else:
            body = Text("waiting for qwen output...", style="dim")
        return Panel(body, title="Qwen Stream", border_style="dim")

    @staticmethod
    def _field(name: str, value: str) -> Text:
        text = Text()
        text.append(f"{name}: ", style="bold")
        text.append(str(value))
        return text


class StreamRenderer:
    def __init__(self, dashboard: Dashboard) -> None:
        self.dashboard = dashboard
        self._thinking_open = False
        self._active_tools: dict[int, dict[str, Any]] = {}
        self._tool_names_by_id: dict[str, str] = {}

    def render_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            self.finish()
            self.dashboard.append(stripped, style="yellow")
            return
        self.render(payload)

    def render(self, payload: dict[str, Any]) -> None:
        payload_type = payload.get("type")

        if payload_type == "system" and payload.get("subtype") == "init":
            self.finish()
            self.dashboard.append(
                "session "
                f"{payload.get('session_id')} | model {payload.get('model')} | "
                f"tools {len(payload.get('tools') or [])}",
                style="bold magenta",
            )
            return

        if payload_type == "stream_event":
            self._render_stream_event(payload.get("event") or {})
            return

        if payload_type == "user":
            self._render_user_message(payload.get("message") or {})
            return

        if payload_type == "result":
            self.finish()
            subtype = payload.get("subtype")
            duration_ms = payload.get("duration_ms")
            self.dashboard.append(f"result: {subtype} ({duration_ms} ms)", style="bold yellow")
            return

        if payload_type in {"tool_use", "tool_result"}:
            self.finish()
            self.dashboard.append(
                f"{payload_type}: {payload.get('name') or payload.get('tool_name')}",
                style="cyan",
            )

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
                self.dashboard.assistant_append(str(delta.get("text") or ""))
            elif delta_type == "thinking_delta" and not self._thinking_open:
                self.dashboard.append("thinking...", style="dim italic")
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
            self.finish()

    def finish(self) -> None:
        self._close_thinking()
        self.dashboard._assistant_line = None

    def _close_thinking(self) -> None:
        if self._thinking_open:
            self.dashboard.append("thinking... done", style="dim green")
            self._thinking_open = False

    def _render_content_block_start(self, event: dict[str, Any]) -> None:
        block = event.get("content_block") or {}
        if block.get("type") != "tool_use":
            return

        self.finish()
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
        self.dashboard.append(f"tool: {name}", style="cyan")

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
            self.dashboard.append(f"input: {summary}", style="dim cyan")

    def _render_user_message(self, message: dict[str, Any]) -> None:
        content = message.get("content")
        if not isinstance(content, list):
            return

        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            self.finish()
            tool_id = str(item.get("tool_use_id") or "")
            name = self._tool_names_by_id.get(tool_id, "tool")
            status = "error" if item.get("is_error") else "ok"
            result = self._one_line(str(item.get("content") or ""), limit=220)
            style = "red" if item.get("is_error") else "green"
            self.dashboard.append(f"output: {name} {status} | {result}", style=style)

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


def _read_process_lines(proc: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
    assert proc.stdout is not None
    for line in proc.stdout:
        output.put(line)
    output.put(None)


def render_qwen_stream(proc: subprocess.Popen[str], dashboard: Dashboard) -> int:
    output: queue.Queue[str | None] = queue.Queue()
    reader = threading.Thread(target=_read_process_lines, args=(proc, output), daemon=True)
    reader.start()
    renderer = StreamRenderer(dashboard)

    with Live(dashboard.render(), console=console, refresh_per_second=4, transient=False) as live:
        while True:
            try:
                line = output.get(timeout=0.25)
            except queue.Empty:
                live.update(dashboard.render())
                continue
            if line is None:
                break
            renderer.render_line(line)
            live.update(dashboard.render())

        renderer.finish()
        returncode = proc.wait()
        style = "bold green" if returncode == 0 else "bold red"
        dashboard.append(
            f"qwen exited with code {returncode} after {format_duration(time.monotonic() - dashboard.started)}",
            style=style,
        )
        live.update(dashboard.render())

    return returncode


def run_qwen(root: Path, stage: str, before: ExperimentSnapshot) -> int:
    prompt = read_prompt(root, stage)
    cmd = [
        QWEN_BIN,
        "--approval-mode",
        "yolo",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        *qwen_mcp_allowlist_args(),
        prompt,
    ]
    dashboard = Dashboard(root, stage, before)
    dashboard.append(
        f"starting {stage} stage: {QWEN_BIN} --approval-mode yolo --output-format stream-json <prompt>",
        style="bold blue",
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        return render_qwen_stream(proc, dashboard)
    except KeyboardInterrupt:
        dashboard.append("interrupt received; stopping Qwen...", style="bold yellow")
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


def qwen_mcp_allowlist_args() -> list[str]:
    args: list[str] = []
    for server_name in ALLOWED_MCP_SERVER_NAMES:
        args.extend(["--allowed-mcp-server-names", server_name])
    return args


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
    console.print(f"[bold blue][orchestrator][/bold blue] repo root: {root}")

    while True:
        before = experiment_snapshot(root)
        stage = choose_stage(before)

        try:
            returncode = run_qwen(root, stage, before)
        except KeyboardInterrupt:
            console.print("[bold yellow][orchestrator][/bold yellow] stopped by user")
            return 130

        if returncode != 0:
            console.print(
                f"[bold red][orchestrator][/bold red] {stage} stage failed with exit code {returncode}",
                file=sys.stderr,
            )
            return returncode

        try:
            after = experiment_snapshot(root)
            validate_stage_result(stage, before, after)
        except RuntimeError as exc:
            console.print(
                f"[bold red][orchestrator][/bold red] {stage} stage validation failed: {exc}",
                file=sys.stderr,
            )
            return 1

        console.print(f"[bold green][orchestrator][/bold green] {stage} stage complete")


if __name__ == "__main__":
    raise SystemExit(main())
