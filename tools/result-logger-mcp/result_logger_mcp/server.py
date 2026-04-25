"""MCP server exposing sequential experiment logging tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from .logger import (
    experiment_create as create_experiment,
    experiment_finish as finish_experiment,
    experiments_list as list_experiments,
)

mcp = FastMCP("result_logger")


@mcp.tool()
async def experiment_create(
    hypothesis: str,
    decision_type: str,
    description: str,
) -> dict[str, Any]:
    """Create the next experiment record.

    A new experiment is rejected while another experiment is running.
    """

    return create_experiment(
        hypothesis=hypothesis,
        decision_type=decision_type,
        description=description,
    )


@mcp.tool()
async def experiment_finish(
    experiment_id: str,
    commit: str,
    status: str,
    metric: float | None = None,
    description: str = "",
) -> dict[str, Any]:
    """Finish an experiment and assign train runs by experiment time window."""

    return finish_experiment(
        experiment_id=experiment_id,
        commit=commit,
        status=status,
        metric=metric,
        description=description,
    )


@mcp.tool()
async def experiments_list(
    status_filter: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """List compact experiment summaries newest first."""

    return list_experiments(status_filter=status_filter, limit=limit)


if __name__ == "__main__":
    mcp.run(transport="stdio")
