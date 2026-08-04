"""Every ``python`` block in ``docs/`` must execute — except declared skips.

The gate is positional, not sampled: it walks *every* markdown file under
``docs/``, executes each page's blocks in document order in one namespace —
which is how a reader consumes them — and fails on any page that raises. A
page a reader can copy top to bottom is the contract.

Blocks that shell out to third-party binaries, need a pre-written data file
from an offline workflow, or are narrative-only must declare::

    # docs: skip — <reason>

as the first non-empty line. The suite never re-runs AmberTools, LAMMPS,
Packmol, or similar; those surfaces are unit-tested via mocks and script
literals under ``tests/test_wrapper`` / ``tests/test_engine``.
"""

from __future__ import annotations

import io
import os
import re
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"

_SKIP_RE = re.compile(r"^#\s*docs:\s*skip\b", re.IGNORECASE)


def _python_blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(encoding="utf-8"), re.S)


def _is_docs_skip(block: str) -> bool:
    """True when the block's first non-empty line is ``# docs: skip …``."""
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return bool(_SKIP_RE.match(stripped))
    return False


def _doc_pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md") if _python_blocks(p))


#: Doc snippets that import a local helper (``eo_kit``) are written to be run
#: from the example directory that defines it, so put it on the path.
HELPER_PATHS = [
    str(REPO_ROOT / "examples"),
    str(REPO_ROOT / "examples" / "topology"),
]


#: Name the page's namespace runs under. It is registered in ``sys.modules``
#: because a class defined in a doc block names this as its ``__module__``, and
#: machinery like ``@dataclass`` resolves annotations by looking the module up
#: there — an unregistered name makes ordinary code fail for a reason no reader
#: would ever hit.
_DOC_MODULE = "__doc_check__"


def _run_page(path: Path) -> list[str]:
    """Execute a page's blocks in order; return one message per real failure."""
    module = ModuleType(_DOC_MODULE)
    namespace = module.__dict__
    failures: list[str] = []
    added = [p for p in HELPER_PATHS if p not in sys.path]
    sys.path[:0] = added
    sys.modules[_DOC_MODULE] = module
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            for index, block in enumerate(_python_blocks(path)):
                if _is_docs_skip(block):
                    continue
                sink = io.StringIO()
                try:
                    with redirect_stdout(sink), redirect_stderr(sink):
                        exec(compile(block, f"{path.name}[{index}]", "exec"), namespace)
                except BaseException as exc:  # noqa: BLE001 - this is the assertion
                    failures.append(
                        f"block {index}: {type(exc).__name__}: {str(exc)[:160]}\n"
                        f"{traceback.format_exc(limit=1)}"
                    )
        finally:
            os.chdir(cwd)
            for entry in added:
                sys.path.remove(entry)
            del sys.modules[_DOC_MODULE]
    return failures


class TestEveryDocBlockExecutes:
    def test_the_scan_actually_finds_the_docs(self):
        # Without this, every parametrised case below vanishes and the suite
        # reports green over nothing — which is exactly how this rotted.
        pages = _doc_pages()
        assert len(pages) > 60, f"only found {len(pages)} doc pages with python"
        assert sum(len(_python_blocks(p)) for p in pages) > 250

    def test_docs_skip_is_recognized(self):
        assert _is_docs_skip("# docs: skip — needs AmberTools\nprint(1)\n")
        assert _is_docs_skip("\n  # docs: skip\nx = 1\n")
        assert not _is_docs_skip("print(1)\n")
        assert not _is_docs_skip("# not a skip directive\n")

    @pytest.mark.parametrize(
        "page", [p.relative_to(DOCS).as_posix() for p in _doc_pages()]
    )
    def test_page_blocks_execute(self, page: str):
        failures = _run_page(DOCS / page)
        assert not failures, f"{page} has stale example code:\n" + "\n".join(failures)
