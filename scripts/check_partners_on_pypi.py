#!/usr/bin/env python3
"""Release gate: every partner range molpy declares has a release on PyPI.

release.yml runs it before testing a tag with ``uv run --no-sources``, which
drops ``[tool.uv.sources]`` and so takes each partner from PyPI. The partners
are the dependencies that table builds from sibling checkouts on dev
(molcrafts-molrs, molcrafts-mollog, molcrafts-molcfg; see
.github/partners.env). Release order (docs/developer/release-process.md):
ship a partner (master + tag + publish) before the molpy that needs it.

Each partner is declared as ``name>=LOW,<HIGH`` and passes when PyPI has a
non-yanked final release in that range. molcrafts-molrs must also be on
molpy's shared minor line: ``>=X.Y.0,<X.(Y+1)``.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

#: The partner whose major.minor molpy shares (docs/developer/release-process.md).
MINOR_LINE = "molcrafts-molrs"

RANGE_RE = re.compile(
    r"^([A-Za-z0-9_.-]+)\s*>=\s*([0-9][0-9.]*)\s*,\s*<\s*([0-9][0-9.]*)$"
)
FINAL_RE = re.compile(r"^[0-9]+(\.[0-9]+)*$")


def _key(version: str) -> tuple[int, ...]:
    parts = [int(p) for p in version.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _ranges(pyproject: dict) -> dict[str, tuple[str, str]]:
    """Partner name -> (low, high) from [project] dependencies."""
    partners = set(pyproject.get("tool", {}).get("uv", {}).get("sources", {}))
    found: dict[str, tuple[str, str]] = {}
    for spec in pyproject["project"]["dependencies"]:
        name = re.split(r"[\s<>=!~\[;]", spec, maxsplit=1)[0]
        if name not in partners:
            continue
        match = RANGE_RE.match(spec.strip())
        if not match:
            raise ValueError(f"{spec!r}: a partner is declared as {name}>=LOW,<HIGH")
        found[name] = (match.group(2), match.group(3))
    missing = partners - set(found)
    if missing:
        raise ValueError(
            f"[tool.uv.sources] names {sorted(missing)}, not in dependencies"
        )
    return found


def _check_minor_line(low: str, high: str) -> None:
    major, minor = (_key(low) + (0,))[:2]
    if _key(high) not in {(major, minor + 1), (major + 1,)}:
        raise ValueError(
            f"{MINOR_LINE}>={low},<{high} is not one minor line (<{major}.{minor + 1})"
        )


def _published(name: str) -> list[str]:
    url = f"https://pypi.org/pypi/{name}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        releases = json.load(resp).get("releases") or {}
    return [
        version
        for version, files in releases.items()
        if FINAL_RE.match(version) and any(not f.get("yanked") for f in files)
    ]


def main() -> int:
    pyproject = Path("pyproject.toml")
    if not pyproject.is_file():
        print(
            "BLOCK: pyproject.toml not found (run from the repository root)",
            file=sys.stderr,
        )
        return 1
    try:
        ranges = _ranges(tomllib.loads(pyproject.read_text(encoding="utf-8")))
        if MINOR_LINE in ranges:
            _check_minor_line(*ranges[MINOR_LINE])
    except ValueError as exc:
        print(f"BLOCK: {exc}", file=sys.stderr)
        return 1

    failed = False
    for name, (low, high) in sorted(ranges.items()):
        try:
            versions = _published(name)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            print(
                f"BLOCK RELEASE: cannot read {name} from PyPI ({exc})", file=sys.stderr
            )
            failed = True
            continue
        inside = sorted(
            (v for v in versions if _key(low) <= _key(v) < _key(high)), key=_key
        )
        if not inside:
            print(
                f"BLOCK RELEASE: no {name} release in >={low},<{high} on PyPI. Release "
                f"{name} first (docs/developer/release-process.md).",
                file=sys.stderr,
            )
            failed = True
            continue
        print(f"ok: {name}>={low},<{high} on PyPI (latest {inside[-1]})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
