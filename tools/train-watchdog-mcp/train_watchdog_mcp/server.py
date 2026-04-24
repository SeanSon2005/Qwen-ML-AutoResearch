"""MCP server exposing the blocking training watchdog tool."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .watchdog import train_run as run_training

mcp = FastMCP("train_watchdog")


@mcp.tool()
async def train_run(
    overrides: list[str] | None = None,
    timeout_sec: int = 7200,
    idle_timeout_sec: int = 900,
    monitor_interval_sec: int = 5,
    report_request: str = "metrics and best checkpoint path",
    log_tail_lines: int = 200,
) -> dict[str, Any]:
    """Run Lightning-Hydra training under a blocking watchdog.

    Args:
        overrides: Arbitrary Hydra override strings passed through to src.train.
        timeout_sec: Maximum wall-clock runtime before termination.
        idle_timeout_sec: Maximum seconds without train process output before termination.
        monitor_interval_sec: Resource sampling interval.
        report_request: Caller request preserved for the Train Subagent report.
        log_tail_lines: Number of log lines to include in the bounded tail excerpt.

    Returns:
        Structured run evidence. This tool returns only after terminal state.
    """

    return run_training(
        overrides=overrides or [],
        timeout_sec=timeout_sec,
        idle_timeout_sec=idle_timeout_sec,
        monitor_interval_sec=monitor_interval_sec,
        report_request=report_request,
        log_tail_lines=log_tail_lines,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
