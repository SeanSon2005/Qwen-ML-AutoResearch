"""Stable test entrypoint for the result logger MCP package."""

from __future__ import annotations

import os
import sys


def main() -> int:
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    import pytest

    return int(pytest.main(["-q", "-p", "no:cacheprovider", *sys.argv[1:]]))


if __name__ == "__main__":
    raise SystemExit(main())
