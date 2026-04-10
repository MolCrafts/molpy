"""Unified CLI for MolPy.

Usage::

    molpy mcp                    # stdio (default)
    molpy mcp --transport http   # streamable-http on 127.0.0.1:8787
    molpy mcp --port 9000        # custom port
"""

from __future__ import annotations

import argparse
import sys


def _cmd_mcp(args: argparse.Namespace) -> None:
    try:
        from molpy_mcp import create_server
    except ImportError:
        print(
            "MCP dependencies not installed. Run:  pip install molpy[mcp]",
            file=sys.stderr,
        )
        sys.exit(1)

    server = create_server()
    kwargs: dict = {"transport": args.transport}
    if args.transport != "stdio":
        kwargs["host"] = args.host
        kwargs["port"] = args.port
    server.run(**kwargs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="molpy", description="MolPy CLI")
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    sub = parser.add_subparsers(dest="command")

    # --- molpy mcp --------------------------------------------------------
    mcp_parser = sub.add_parser("mcp", help="Start the MCP server.")
    mcp_parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio).",
    )
    mcp_parser.add_argument(
        "--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)."
    )
    mcp_parser.add_argument(
        "--port", "-p", type=int, default=8787, help="Port (default: 8787)."
    )

    args = parser.parse_args(argv)

    if args.version:
        from molpy.version import version

        print(f"molpy {version}")
        return

    if args.command == "mcp":
        _cmd_mcp(args)
    else:
        parser.print_help()
