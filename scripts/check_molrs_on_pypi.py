#!/usr/bin/env python3
"""Release gate: the molcrafts-molrs minor line molpy declares is on PyPI.

release.yml runs it before testing a tag against molrs from PyPI. Release
order (docs/developer/release-process.md): ship molrs (master + tag +
publish) before molpy. On dev molpy builds molrs from its dev
([tool.uv.sources]), so this does not apply there.

Accepts the minor-line form only:
  molcrafts-molrs>=X.Y.0,<X.(Y+1)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# PEP 440 lower/upper minor range.
RANGE_RE = re.compile(r"molcrafts-molrs>=([0-9][^\"',\s]*),\s*<([0-9][^\"',\s]*)")
PYPI_PROJECT = "https://pypi.org/pypi/molcrafts-molrs/json"


def _minor_tuple(ver: str) -> tuple[int, int]:
    core = ver.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 2:
        raise ValueError(f"expected at least major.minor, got {ver!r}")
    return int(parts[0]), int(parts[1])


def _is_next_minor(lo: tuple[int, int], hi: tuple[int, int]) -> bool:
    """True if hi is the next minor after lo (e.g. 0.10 → 0.11, or 0.99 → 1.0)."""
    return hi == (lo[0], lo[1] + 1) or hi == (lo[0] + 1, 0)


def _parse_pin(text: str) -> tuple[int, int]:
    """Return the (major, minor) line of the molcrafts-molrs range pin."""
    m = RANGE_RE.search(text)
    if not m:
        raise ValueError("no molcrafts-molrs minor-line pin found")
    lo, hi = m.group(1), m.group(2)
    lo_mm = _minor_tuple(lo)
    hi_mm = _minor_tuple(hi)
    if not _is_next_minor(lo_mm, hi_mm):
        raise ValueError(f"expected upper bound next minor after {lo}, got <{hi}")
    return lo_mm


def _fetch_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def main() -> int:
    root = Path.cwd()
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        print("BLOCK: pyproject.toml not found (run from repo root)", file=sys.stderr)
        return 1

    text = pyproject.read_text(encoding="utf-8")
    try:
        major, minr = _parse_pin(text)
    except ValueError as exc:
        print(
            f"BLOCK: {exc}; need molcrafts-molrs>=X.Y.0,<X.(Y+1) in pyproject.toml",
            file=sys.stderr,
        )
        return 1

    try:
        data = _fetch_json(PYPI_PROJECT)
        releases = data.get("releases") or {}
        matches = []
        for ver in releases:
            try:
                if _minor_tuple(ver) == (major, minr) and releases[ver]:
                    matches.append(ver)
            except ValueError:
                continue
        if not matches:
            print(
                f"BLOCK RELEASE: no published molcrafts-molrs {major}.{minr}.* "
                f"on PyPI.",
                file=sys.stderr,
            )
            print(
                "Release molrs first (master + tag vX.Y.Z + publish). See "
                "docs/developer/release-process.md",
                file=sys.stderr,
            )
            return 1
        matches.sort(key=lambda v: [int(p) for p in v.split(".") if p.isdigit()])
        print(
            f"ok: molcrafts-molrs {major}.{minr}.* on PyPI "
            f"({len(matches)} release(s); latest {matches[-1]})"
        )
        return 0
    except urllib.error.HTTPError as exc:
        print(
            f"BLOCK RELEASE: could not verify molcrafts-molrs on PyPI "
            f"(HTTP {exc.code}).",
            file=sys.stderr,
        )
        print(
            "Release molrs first (master + tag vX.Y.Z + publish). See "
            "docs/developer/release-process.md",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:  # surface network/parse failures
        print(
            f"BLOCK RELEASE: could not verify molcrafts-molrs on PyPI "
            f"({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
