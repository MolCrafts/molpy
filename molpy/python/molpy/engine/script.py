"""Script: an input file's text, its name and, optionally, its path.

A :class:`Script` is an immutable value. :meth:`Engine.run
<molpy.engine.Engine.run>` writes it into the run directory; :meth:`Script.save`
writes it anywhere. Neither changes it.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Script:
    """An input script: a name, its text and the file it came from.

    Attributes:
        name: Logical name; an engine writes an unpathed script to
            ``<name><extension>``.
        text: The full text, written as is.
        path: The file it was read from, or ``None``; an engine writes it
            under this file name.
    """

    name: str
    text: str
    path: Path | None = None

    @classmethod
    def from_text(
        cls, name: str, text: str, *, path: str | Path | None = None
    ) -> Script:
        """A script from a string literal.

        The block is dedented and ends in exactly one newline, so an indented
        triple-quoted string in code becomes a flush-left file.

        Args:
            name: Logical name of the script.
            text: The script text.
            path: The file name an engine writes it under.
        """
        body = textwrap.dedent(text).rstrip("\n")
        return cls(
            name=name,
            text=body + "\n" if body else "",
            path=Path(path) if path is not None else None,
        )

    @classmethod
    def from_path(cls, path: str | Path) -> Script:
        """The script in the file at *path*, named after its stem.

        The text is the file's, unchanged.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        return cls(name=path.stem, text=path.read_text(encoding="utf-8"), path=path)

    def save(self, path: str | Path | None = None) -> Path:
        """Write the text to *path* (default: :attr:`path`); return where.

        Parent directories are created. The script itself is unchanged.

        Raises:
            ValueError: If no *path* is given and the script has none.
        """
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError(f"script {self.name!r} has no path; pass one to save()")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.text, encoding="utf-8")
        return target

    def preview(self, max_lines: int = 20) -> str:
        """The first *max_lines* lines, numbered, and how many more there are."""
        lines = self.text.splitlines()
        if not lines:
            return "(empty script)"
        width = len(str(len(lines)))
        shown = [
            f"{number:>{width}} | {line}"
            for number, line in enumerate(lines[:max_lines], start=1)
        ]
        remaining = len(lines) - max_lines
        if remaining > 0:
            shown.append(f"... ({remaining} more line{'s' if remaining != 1 else ''})")
        return "\n".join(shown)
