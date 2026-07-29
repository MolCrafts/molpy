#!/usr/bin/env python3
"""Pre-push gate: molcrafts-molrs==X.Y.Z in pyproject.toml must exist on PyPI.

Release order (see .claude/notes/release.md): ship molrs (master + tag +
publish) before molpy may push a pin that depends on it. Editable monorepo
builds do not count.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PIN_RE = re.compile(r"molcrafts-molrs==([0-9][^\"']*)")
PYPI = "https://pypi.org/pypi/molcrafts-molrs/{ver}/json"


def main() -> int:
    root = Path.cwd()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        print("BLOCK: pyproject.toml not found (run from repo root)", file=sys.stderr)
        return 1

    match = PIN_RE.search(pyproject.read_text(encoding="utf-8"))
    if not match:
        print(
            "BLOCK: no exact molcrafts-molrs== pin in pyproject.toml",
            file=sys.stderr,
        )
        return 1

    ver = match.group(1)
    url = PYPI.format(ver=ver)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            json.load(resp)
    except urllib.error.HTTPError as exc:
        print(
            f"BLOCK PUSH: molcrafts-molrs=={ver} is not on PyPI "
            f"(HTTP {exc.code}).",
            file=sys.stderr,
        )
        print(
            "Release molrs first (master + tag vX.Y.Z + publish), then pin "
            "molpy. See .claude/notes/release.md",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # noqa: BLE001 — surface network/parse failures
        print(
            f"BLOCK PUSH: could not verify molcrafts-molrs=={ver} on PyPI "
            f"({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return 1

    print(f"ok: molcrafts-molrs=={ver} is on PyPI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
