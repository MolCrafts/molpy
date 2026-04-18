"""``molpy mcp`` subcommand — launch the Model Context Protocol server."""

from __future__ import annotations

import argparse
import sys


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("mcp", help="Start the MCP server.")
    p.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "streamable-http", "sse"],
        default="stdio",
        help="Transport protocol (default: stdio).",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1).",
    )
    p.add_argument("--port", "-p", type=int, default=8787, help="Port (default: 8787).")
    p.set_defaults(func=_cmd_mcp)


def _cmd_mcp(args: argparse.Namespace) -> int:
    try:
        from molpy_mcp import create_server
    except ImportError:
        print(
            "MCP dependencies not installed. Run:  pip install molpy[mcp]",
            file=sys.stderr,
        )
        return 1
    server = create_server()
    kwargs: dict = {"transport": args.transport}
    if args.transport != "stdio":
        kwargs["host"] = args.host
        kwargs["port"] = args.port
    server.run(**kwargs)
    return 0
