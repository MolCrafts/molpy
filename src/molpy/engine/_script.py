"""
Script - Editable script management with filesystem and URL support.

This module provides a Script class for managing script content that can be
stored locally or loaded from URLs. It supports editing, formatting, and
filesystem operations without any execution logic.
"""

import textwrap
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ScriptLanguage = Literal["bash", "python", "slurm", "yaml", "other"]


@dataclass
class Script:
    """
    Represents an editable script with filesystem and URL support.

    This class manages script content, metadata, and filesystem operations.
    It does NOT provide execution logic - only content management.

    Attributes:
        name: Logical name of the script
        language: Script language type
        description: Optional human-readable description
        _lines: Internal storage for multi-line content
        path: Local file path if stored on disk
        url: URL the script was loaded from (if any)
        tags: Optional lightweight tag system
    """

    name: str
    language: ScriptLanguage = "bash"
    description: str | None = None
    _lines: list[str] = field(default_factory=list, repr=False)
    path: Path | None = None
    url: str | None = None
    tags: set[str] = field(default_factory=set)

    @classmethod
    def from_text(
        cls,
        name: str,
        text: str,
        *,
        language: ScriptLanguage = "bash",
        description: str | None = None,
        path: str | Path | None = None,
        url: str | None = None,
    ) -> "Script":
        """
        Create a Script from text content.

        Args:
            name: Logical name of the script
            text: Multi-line text content
            language: Script language type
            description: Optional description
            path: Optional local file path
            url: Optional URL source

        Returns:
            Script instance with normalized content
        """
        # Normalize indentation and split into lines
        normalized = textwrap.dedent(text)
        # Remove trailing newlines but preserve internal ones
        normalized = normalized.rstrip("\n")
        lines = normalized.splitlines() if normalized else []

        path_obj = Path(path) if path is not None else None

        return cls(
            name=name,
            language=language,
            description=description,
            _lines=lines,
            path=path_obj,
            url=url,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        language: ScriptLanguage | None = None,
        description: str | None = None,
    ) -> "Script":
        """
        Create a Script from a local file path.

        Args:
            path: Path to the script file
            language: Optional language override. If None, guessed from extension
            description: Optional description

        Returns:
            Script instance loaded from file

        Raises:
            FileNotFoundError: If the file does not exist
            IOError: If the file cannot be read
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"Script file not found: {path_obj}")

        # Read file content
        try:
            content = path_obj.read_text(encoding="utf-8")
        except Exception as e:
            raise OSError(f"Failed to read script file {path_obj}: {e}") from e

        # Derive name from stem
        name = path_obj.stem

        # Guess language if not provided
        if language is None:
            language = cls._guess_language(path_obj)

        return cls.from_text(
            name=name,
            text=content,
            language=language,
            description=description,
            path=path_obj,
        )

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        name: str | None = None,
        language: ScriptLanguage = "other",
        description: str | None = None,
    ) -> "Script":
        """
        Create a Script from a URL.

        Args:
            url: URL to fetch the script from
            name: Optional name. If None, derived from URL
            language: Script language type
            description: Optional description

        Returns:
            Script instance loaded from URL

        Raises:
            urllib.error.URLError: If the URL cannot be fetched
        """
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode("utf-8")
        except Exception as e:
            raise urllib.error.URLError(
                f"Failed to fetch script from {url}: {e}"
            ) from e

        # Derive name from URL if not provided
        if name is None:
            # Extract last path segment
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path_segment = parsed.path.rstrip("/").split("/")[-1]
            if path_segment:
                # Remove extension if present
                name = Path(path_segment).stem or "script"
            else:
                name = "script"

        return cls.from_text(
            name=name,
            text=content,
            language=language,
            description=description,
            url=url,
        )

    @staticmethod
    def _guess_language(path: Path) -> ScriptLanguage:
        """
        Guess script language from file extension.

        Args:
            path: File path

        Returns:
            Guessed language type
        """
        suffix = path.suffix.lower()

        if suffix == ".sh":
            return "bash"
        elif suffix == ".py":
            return "python"
        elif suffix in (".slurm", ".sbatch"):
            return "slurm"
        elif suffix in (".yml", ".yaml"):
            return "yaml"
        else:
            return "other"

    @property
    def lines(self) -> list[str]:
        """
        Get a copy of all script lines.

        Returns:
            Copy of internal lines list
        """
        return self._lines.copy()

    @property
    def text(self) -> str:
        """
        Get the full script as a single string.

        Returns:
            Script content with lines joined by newlines, with exactly one trailing newline
        """
        return "\n".join(self._lines) + "\n" if self._lines else ""

    def clear(self) -> None:
        """Remove all lines from the script."""
        self._lines.clear()

    def append(self, line: str = "") -> None:
        """
        Append a single line to the end of the script.

        Args:
            line: Line content to append
        """
        self._lines.append(line)

    def extend(self, lines: Iterable[str]) -> None:
        """
        Append multiple lines in order to the end of the script.

        Args:
            lines: Iterable of lines to append
        """
        self._lines.extend(lines)

    def append_block(self, block: str) -> None:
        """
        Append a multi-line block to the script.

        The block is dedented, trailing newlines are stripped,
        and then split into lines.

        Args:
            block: Multi-line string block to append
        """
        normalized = textwrap.dedent(block)
        normalized = normalized.rstrip("\n")
        if normalized:
            self._lines.extend(normalized.splitlines())

    def insert(self, index: int, line: str) -> None:
        """
        Insert a single line at the given index.

        Args:
            index: 0-based index where to insert
            line: Line content to insert

        Raises:
            IndexError: If index is out of range
        """
        self._lines.insert(index, line)

    def replace(self, index: int, line: str) -> None:
        """
        Replace the line at the given index.

        Args:
            index: 0-based index of line to replace
            line: New line content

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._lines):
            raise IndexError(f"Line index {index} out of range [0, {len(self._lines)})")
        self._lines[index] = line

    def delete(self, index: int) -> None:
        """
        Delete the line at the given index.

        Args:
            index: 0-based index of line to delete

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self._lines):
            raise IndexError(f"Line index {index} out of range [0, {len(self._lines)})")
        del self._lines[index]

    def format(self, **kwargs: Any) -> "Script":
        """
        Apply string formatting to all lines and return a new Script.

        Uses Python's str.format(**kwargs) on each line.

        Args:
            **kwargs: Format arguments

        Returns:
            New Script instance with formatted lines
        """
        formatted_lines = [line.format(**kwargs) for line in self._lines]

        return Script(
            name=self.name,
            language=self.language,
            description=self.description,
            _lines=formatted_lines,
            path=self.path,
            url=self.url,
            tags=self.tags.copy(),
        )

    def format_with_mapping(self, mapping: Mapping[str, Any]) -> "Script":
        """
        Apply string formatting to all lines using a mapping and return a new Script.

        Uses Python's str.format_map(mapping) on each line.

        Args:
            mapping: Format mapping

        Returns:
            New Script instance with formatted lines
        """
        formatted_lines = [line.format_map(mapping) for line in self._lines]

        return Script(
            name=self.name,
            language=self.language,
            description=self.description,
            _lines=formatted_lines,
            path=self.path,
            url=self.url,
            tags=self.tags.copy(),
        )

    def preview(
        self,
        max_lines: int = 20,
        *,
        with_line_numbers: bool = True,
    ) -> str:
        """
        Generate a preview of the script.

        Args:
            max_lines: Maximum number of lines to show
            with_line_numbers: Whether to include line numbers

        Returns:
            Preview string
        """
        if not self._lines:
            return "(empty script)"

        lines_to_show = self._lines[:max_lines]
        has_more = len(self._lines) > max_lines

        if with_line_numbers:
            # Calculate width for line numbers
            max_num = len(self._lines) if has_more else len(lines_to_show)
            width = len(str(max_num))

            preview_lines = [
                f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines_to_show)
            ]
        else:
            preview_lines = lines_to_show

        result = "\n".join(preview_lines)

        if has_more:
            remaining = len(self._lines) - max_lines
            result += f"\n... ({remaining} more line{'s' if remaining != 1 else ''})"

        return result

    def save(self, path: str | Path | None = None) -> Path:
        """
        Save the script to a file.

        Args:
            path: Optional path to save to. If None, uses self.path

        Returns:
            Path where the script was saved

        Raises:
            ValueError: If no path is provided and self.path is None
            IOError: If the file cannot be written
        """
        if path is None:
            if self.path is None:
                raise ValueError(
                    "No path provided and script has no associated path. "
                    "Provide a path argument or set script.path first."
                )
            save_path = self.path
        else:
            save_path = Path(path)

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            save_path.write_text(self.text, encoding="utf-8")
        except Exception as e:
            raise OSError(f"Failed to write script to {save_path}: {e}") from e

        # Update internal path
        self.path = save_path

        return save_path

    def reload(self) -> None:
        """
        Reload the script content from its associated path.

        Raises:
            ValueError: If script has no associated path
            FileNotFoundError: If the file does not exist
            IOError: If the file cannot be read
        """
        if self.path is None:
            raise ValueError("Cannot reload: script has no associated path")

        if not self.path.exists():
            raise FileNotFoundError(f"Script file not found: {self.path}")

        try:
            content = self.path.read_text(encoding="utf-8")
        except Exception as e:
            raise OSError(f"Failed to read script file {self.path}: {e}") from e

        # Update content
        normalized = content.rstrip("\n")
        self._lines = normalized.splitlines() if normalized else []

    def delete_file(self) -> None:
        """
        Delete the script file from the filesystem.

        Raises:
            ValueError: If script has no associated path
            FileNotFoundError: If the file does not exist
            OSError: If the file cannot be deleted
        """
        if self.path is None:
            raise ValueError("Cannot delete: script has no associated path")

        if not self.path.exists():
            raise FileNotFoundError(f"Script file not found: {self.path}")

        try:
            self.path.unlink()
        except Exception as e:
            raise OSError(f"Failed to delete script file {self.path}: {e}") from e

        # Clear the path reference
        self.path = None

    def move(self, new_path: str | Path) -> Path:
        """
        Move the script file to a new location.

        Args:
            new_path: New file path

        Returns:
            New path where the script was moved

        Raises:
            ValueError: If script has no associated path
            FileNotFoundError: If the original file does not exist
            OSError: If the file cannot be moved
        """
        if self.path is None:
            raise ValueError("Cannot move: script has no associated path")

        if not self.path.exists():
            raise FileNotFoundError(f"Script file not found: {self.path}")

        new_path_obj = Path(new_path)

        # Ensure parent directory exists
        new_path_obj.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.path.rename(new_path_obj)
        except Exception as e:
            raise OSError(
                f"Failed to move script from {self.path} to {new_path_obj}: {e}"
            ) from e

        # Update internal path
        self.path = new_path_obj

        return new_path_obj

    def rename(self, new_name: str) -> Path:
        """
        Rename the script file (keeping the same directory).

        Args:
            new_name: New file name (with or without extension)

        Returns:
            New path where the script was renamed

        Raises:
            ValueError: If script has no associated path
            FileNotFoundError: If the original file does not exist
            OSError: If the file cannot be renamed
        """
        if self.path is None:
            raise ValueError("Cannot rename: script has no associated path")

        # Preserve extension if new_name doesn't have one
        new_path = self.path.parent / new_name
        if new_path.suffix == "" and self.path.suffix:
            new_path = new_path.with_suffix(self.path.suffix)

        return self.move(new_path)
