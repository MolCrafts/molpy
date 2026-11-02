"""The ``molpy`` command. ``convert`` copies frames between file formats."""

from __future__ import annotations

import argparse
import sys

from molpy.io.formats import convert, known_formats


def main(argv: list[str] | None = None) -> int:
    """Run the command in ``argv``. Returns a process status.

    Args:
        argv: Arguments after the program name. ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when the command finishes, ``1`` when conversion fails.
        Argument errors exit through ``argparse``.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        convert(
            args.source,
            args.destination,
            in_format=args.in_format,
            out_format=args.out_format,
            frame=args.frame,
        )
    except (OSError, ValueError) as exc:
        print(f"molpy convert: {exc}", file=sys.stderr)
        return 1
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="molpy")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser(
        "convert",
        help="copy frames from one file format to another",
    )
    command.add_argument("--in", dest="source", required=True, metavar="PATH")
    command.add_argument("--out", dest="destination", required=True, metavar="PATH")
    command.add_argument(
        "--in-format",
        default=None,
        metavar="NAME",
        help=f"input format; default is the file suffix ({known_formats()})",
    )
    command.add_argument(
        "--out-format",
        default=None,
        metavar="NAME",
        help="output format; default is the file suffix",
    )
    command.add_argument(
        "--frame",
        type=int,
        default=None,
        help="write this frame only, counting from 0",
    )
    return parser


__all__ = ["main"]
