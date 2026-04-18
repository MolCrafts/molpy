"""Top-level CLI entry: builds argparse tree and dispatches subcommands.

Registered subcommands:
  * ``molpy mcp``         -- Model Context Protocol server.
  * ``molpy moltemplate`` -- Native execution of moltemplate .lt scripts.
"""

from __future__ import annotations

import argparse
import sys

from . import mcp, moltemplate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molpy", description="MolPy CLI")
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    sub = parser.add_subparsers(dest="command")
    mcp.register(sub)
    moltemplate.register(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        from molpy.version import version

        print(f"molpy {version}")
        return 0

    cmd = args.command
    handler = getattr(args, "func", None)
    if cmd is None or handler is None:
        parser.print_help()
        return 0
    try:
        rc = handler(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(f"molpy: error: {exc}", file=sys.stderr)
        return 1
    return int(rc) if rc is not None else 0
