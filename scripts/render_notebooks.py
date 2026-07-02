#!/usr/bin/env python3
"""Render the user-guide notebooks to self-contained Markdown for the docs site.

The documentation is built with Zensical, which does not run the ``mkdocs-jupyter``
plugin. The notebooks under ``docs/user-guide/`` are therefore pre-rendered to
Markdown by this script and the resulting ``.md`` pages are committed alongside
the ``.ipynb`` sources (the notebooks remain the editable source of truth).

Each notebook is executed so its output cells are captured. The notebooks listed
in :data:`NEEDS_LAMMPS` run a LAMMPS simulation and are executed only when the
``lmp_serial`` executable is available, otherwise they fall back to render-only
(code and prose, no output). Any image outputs are inlined as base64 data URIs so
each ``.md`` is self-contained, which keeps the repository's ``docs/`` policy of
tracking only ``.md`` / ``.ipynb`` files intact (no stray PNGs).

Usage::

    python scripts/render_notebooks.py            # render every notebook
    python scripts/render_notebooks.py 01 05      # render notebooks matching a prefix

Run this whenever a user-guide notebook changes, then rebuild with ``zensical build``.
"""

from __future__ import annotations

import base64
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

USER_GUIDE = Path(__file__).resolve().parent.parent / "docs" / "user-guide"

# Notebooks that run a LAMMPS simulation (via molpy.engine.LAMMPSEngine). They are
# executed only when the `lmp_serial` executable is available; otherwise they fall
# back to render-only (code and prose, no output) so the docs can still be
# regenerated on a machine without LAMMPS installed.
NEEDS_LAMMPS = {"04_crosslinking", "05_polydisperse_systems"}
LAMMPS_EXECUTABLE = "lmp_serial"

# Per-cell execution timeout in seconds (some cells run packmol/LAMMPS).
CELL_TIMEOUT = 900

# Matches Markdown image references that point at an extracted ``*_files`` asset.
_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]*_files/[^)]+)\)")

# Matches relative Markdown links to a sibling notebook, e.g. ``](02_polymer.ipynb)``
# or ``](../user-guide/06_typifier.ipynb#anchor)``. Absolute http(s) links (the
# "Open in Colab" badge, which must keep pointing at the .ipynb on GitHub) are
# excluded via the negative lookahead.
_NB_LINK_RE = re.compile(r"\]\((?!https?://)(?P<path>[^)]*?)\.ipynb(?P<frag>[)#])")


def _inline_images(markdown: str, base: Path) -> str:
    """Replace ``![alt](stem_files/img.png)`` references with base64 data URIs."""

    def repl(match: re.Match[str]) -> str:
        image = base / match.group("path")
        data = base64.b64encode(image.read_bytes()).decode("ascii")
        ext = image.suffix.lstrip(".") or "png"
        return f"![{match.group('alt')}](data:image/{ext};base64,{data})"

    return _IMG_RE.sub(repl, markdown)


def _rewrite_notebook_links(markdown: str) -> str:
    """Repoint relative ``.ipynb`` cross-links at the rendered ``.md`` pages."""
    return _NB_LINK_RE.sub(r"](\g<path>.md\g<frag>", markdown)


def _fence_stream_outputs(markdown: str) -> str:
    """Wrap nbconvert's *indented* cell-output blocks in fenced code blocks.

    ``nbconvert --to markdown`` emits stream / text outputs as 4-space *indented*
    code blocks. Zensical's Markdown parser only recognizes *fenced* code blocks,
    so an indented output line such as ``labels=['EO2', ...]`` is parsed as prose
    and its ``[...]`` is (wrongly) treated as an unresolved reference link. Re-emit
    each output block — the indented block that directly follows a fenced code
    cell — as a fenced ``text`` block, so brackets in a repr are never parsed.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    in_fence = False
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("```") or line.startswith("~~~"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            if in_fence:
                continue
            # Just closed a fence: absorb a following indented output block.
            j = i
            while j < n and lines[j].strip() == "":
                j += 1
            if j < n and lines[j].startswith("    ") and lines[j].strip():
                out.extend(lines[i:j])  # keep the blank separator(s)
                i = j
                block: list[str] = []
                while i < n:
                    if lines[i].startswith("    "):
                        block.append(lines[i][4:])
                        i += 1
                    elif lines[i] == "":
                        # Absorb blank separators only when more indented output
                        # follows — a single cell can emit several chunks (logging
                        # to stderr + a stdout print), split by blank lines.
                        k = i
                        while k < n and lines[k] == "":
                            k += 1
                        if k < n and lines[k].startswith("    "):
                            block.extend(lines[i:k])
                            i = k
                        else:
                            break
                    else:
                        break
                out.append("```text")
                out.extend(block)
                out.append("```")
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def render(notebook: Path) -> None:
    """Render a single notebook to ``docs/user-guide/<stem>.md``."""
    stem = notebook.stem
    execute = True
    if stem in NEEDS_LAMMPS and shutil.which(LAMMPS_EXECUTABLE) is None:
        execute = False
        print(
            f"warning: {LAMMPS_EXECUTABLE} not found; rendering {stem} without execution"
        )
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "markdown",
            "--output",
            stem,
            "--output-dir",
            tmp,
        ]
        if execute:
            cmd += ["--execute", f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}"]
        cmd.append(str(notebook))
        # Run with the notebook's directory as CWD so relative inputs (oplsaa.xml)
        # and example output dirs resolve exactly as they did under mkdocs-jupyter.
        subprocess.run(cmd, check=True, cwd=USER_GUIDE)
        rendered = (Path(tmp) / f"{stem}.md").read_text(encoding="utf-8")
        rendered = _inline_images(rendered, Path(tmp))
        rendered = _rewrite_notebook_links(rendered)
        rendered = _fence_stream_outputs(rendered)
    (USER_GUIDE / f"{stem}.md").write_text(rendered, encoding="utf-8")
    print(f"rendered {stem}.md (execute={execute})")


def main(argv: list[str]) -> int:
    notebooks = sorted(USER_GUIDE.glob("*.ipynb"))
    if argv:
        notebooks = [nb for nb in notebooks if any(nb.name.startswith(p) for p in argv)]
    if not notebooks:
        print("no matching notebooks found", file=sys.stderr)
        return 1
    for notebook in notebooks:
        render(notebook)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
