#!/usr/bin/env python3
"""Run every ```python block of the manual, top to bottom per page.

The manual is executable (AGENTS.md, "Documentation"): each page's python
blocks run in order in one namespace, in a fresh interpreter whose working
directory is an empty temporary directory, against the installed molpy, with
the example scripts' directories (``molpy/examples``) on ``sys.path``. A
block whose first line starts with ``# docs: skip`` is not run; the marker
carries its reason. A page fails on the first block that raises, and the run
fails when any page does.

Usage::

    python -I scripts/run_doc_blocks.py                 # every page under docs/
    python -I scripts/run_doc_blocks.py docs/tutorials  # pages under a path
    python -I scripts/run_doc_blocks.py -j 8 --timeout 900

``scripts/check.sh docs`` runs it with molpy's environment and the tested
wheel.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
# The example scripts' directories: a page imports their helper modules
# (``from eo_kit import library  # examples/topology/``) as a reader running
# the examples does.
EXAMPLES = ROOT / "molpy" / "examples"
SKIP = "# docs: skip"
FENCE = re.compile(r"^```python\s*$")
CLOSE = re.compile(r"^```\s*$")

# Runs inside the page's interpreter: exec each block in one namespace, the
# block's page and line in every traceback.
DRIVER = """
import json, sys, traceback
page, paths, blocks = json.load(sys.stdin)
sys.path[:0] = paths
namespace = {"__name__": "__main__"}
for line, code in blocks:
    try:
        exec(compile(code, f"{page}:{line}", "exec"), namespace)
    except BaseException:
        print(f"{page}:{line}: the block raised", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
"""


def blocks(page: Path) -> list[tuple[int, str]]:
    """``(first code line, code)`` of each python block the page runs."""
    out: list[tuple[int, str]] = []
    lines = page.read_text(encoding="utf-8").splitlines()
    k = 0
    while k < len(lines):
        if not FENCE.match(lines[k]):
            k += 1
            continue
        start = k + 1
        end = start
        while end < len(lines) and not CLOSE.match(lines[end]):
            end += 1
        if end == len(lines):
            raise ValueError(f"{page}:{k + 1}: a python block is never closed")
        code = lines[start:end]
        if not (code and code[0].lstrip().startswith(SKIP)):
            # Pad so the compiled block's line numbers are the page's.
            out.append((start + 1, "\n" * start + "\n".join(code) + "\n"))
        k = end + 1
    return out


def example_paths() -> list[str]:
    """Every directory under ``molpy/examples`` that holds a module."""
    return sorted(
        {str(p.parent) for p in EXAMPLES.rglob("*.py") if "__pycache__" not in p.parts}
    )


def pages(targets: list[str]) -> list[Path]:
    """Every Markdown page under each target (a page or a directory)."""
    found: list[Path] = []
    for target in targets or [str(DOCS)]:
        path = Path(target).resolve()
        if path.is_dir():
            found.extend(sorted(path.rglob("*.md")))
        elif path.suffix == ".md" and path.is_file():
            found.append(path)
        else:
            raise SystemExit(f"{target}: no Markdown page or directory")
    return found


def run(page: Path, timeout: float) -> tuple[Path, int, str]:
    """Run one page; ``(page, blocks run, failure text or "")``."""
    code = blocks(page)
    if not code:
        return page, 0, ""
    name = str(page.relative_to(ROOT))
    with tempfile.TemporaryDirectory(prefix="docs-") as work:
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-c", DRIVER],
                input=json.dumps([name, example_paths(), code]),
                capture_output=True,
                text=True,
                cwd=work,
                timeout=timeout,
                env={**os.environ, "MPLBACKEND": "Agg"},
            )
        except subprocess.TimeoutExpired:
            return page, len(code), f"{name}: timed out after {timeout:g} s"
    if done.returncode != 0:
        return page, len(code), done.stderr.strip() or f"{name}: exit {done.returncode}"
    return page, len(code), ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "targets", nargs="*", help="pages or directories (default: docs/)"
    )
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds per page")
    args = parser.parse_args()
    todo = pages(args.targets)
    failures: list[str] = []
    total = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for page, n, failure in pool.map(lambda p: run(p, args.timeout), todo):
            total += n
            if failure:
                failures.append(failure)
                print(f"FAIL {page.relative_to(ROOT)}", flush=True)
    print(f"{total} blocks on {len(todo)} pages; {len(failures)} pages failed")
    for failure in failures:
        print("\n" + failure, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
