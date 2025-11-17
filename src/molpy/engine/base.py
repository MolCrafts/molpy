"""
Engine base classes for molecular simulation engines.

This module provides abstract base classes for running external computational
chemistry programs like LAMMPS, CP2K, etc. It integrates with the core Script
class for input file management.
"""

import shutil
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    pass

from molpy import Script


class Engine(ABC):
    """
    Abstract base class for computational chemistry engines.

    Provides a common interface for running external programs like LAMMPS, CP2K, etc.
    Each engine handles setup, execution, and output processing for its specific program.

    The Engine class integrates with the core Script class for input file management.
    Scripts can be created from text, loaded from files, or loaded from URLs.

    Attributes:
        executable: Path or command to the executable
        work_dir: Working directory for calculations
        scripts: List of Script objects for input files
        input_script: Primary input script (first script or script with 'input' tag)
        output_file: Primary output file from the calculation

    Example:
        >>> from molpy.core.script import Script
        >>> from molpy.engine import LAMMPSEngine
        >>>
        >>> # Create input script
        >>> script = Script.from_text(
        ...     name="input",
        ...     text="units real\\natom_style full\\n",
        ...     language="other"
        ... )
        >>>
        >>> # Create engine and prepare
        >>> engine = LAMMPSEngine(executable="lmp")
        >>> engine.prepare(work_dir="./calc", scripts=[script])
        >>>
        >>> # Run calculation
        >>> result = engine.run()
        >>> print(result.returncode)
        0
    """

    def __init__(self, executable: str, *, check_executable: bool = True):
        """
        Initialize the engine.

        Args:
            executable: Path or command to the executable
            check_executable: Whether to check if executable exists in PATH
                             (default: True)

        Raises:
            FileNotFoundError: If check_executable is True and executable not found
        """
        self.executable = executable
        if check_executable:
            self.check_executable()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the engine."""
        pass

    def check_executable(self) -> None:
        """
        Check if the executable is available in the system PATH.

        Raises:
            FileNotFoundError: If the executable is not found
        """
        if not shutil.which(self.executable):
            raise FileNotFoundError(
                f"Executable '{self.executable}' not found in PATH. "
                f"Please install the engine or set the correct executable path."
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(executable='{self.executable}')>"

    def prepare(
        self,
        work_dir: str | Path,
        scripts: Script | Sequence[Script],
        *,
        auto_save: bool = True,
    ) -> "Self":
        """
        Prepare the engine for execution by setting up the working directory and scripts.

        Args:
            work_dir: Path to the working directory
            scripts: Single Script object or sequence of Script objects
            auto_save: Whether to automatically save scripts to the working directory
                      (default: True)

        Returns:
            Self: The engine instance for method chaining

        Example:
            >>> script = Script.from_text("input", "units real\\natom_style full\\n")
            >>> engine.prepare("./calc", script)
            >>> # Or with multiple scripts
            >>> engine.prepare("./calc", [script1, script2])
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Normalize scripts to a list
        if isinstance(scripts, Script):
            self.scripts = [scripts]
        else:
            self.scripts = list(scripts)

        if not self.scripts:
            raise ValueError("At least one script is required")

        # Save scripts to the working directory if auto_save is True
        if auto_save:
            for script in self.scripts:
                # Determine filename from script path or name
                if script.path is not None:
                    # Use the filename from the script's path
                    script_path = self.work_dir / script.path.name
                else:
                    # Generate filename from script name
                    # Try to guess extension from language or use default
                    ext = self._get_default_extension()
                    script_path = self.work_dir / f"{script.name}{ext}"

                # Save script to working directory
                script.save(script_path)

        # Set input_script (first script or script with 'input' tag)
        self.input_script = self._find_input_script()

        return self

    def _find_input_script(self) -> Script | None:
        """
        Find the primary input script.

        Returns:
            Script object with 'input' tag, or first script if no tag found
        """
        # Look for script with 'input' tag
        for script in self.scripts:
            if "input" in script.tags:
                return script

        # Return first script if no tag found
        return self.scripts[0] if self.scripts else None

    @abstractmethod
    def _get_default_extension(self) -> str:
        """
        Get the default file extension for input files.

        Returns:
            Default file extension (e.g., '.inp', '.lmp')
        """
        pass

    def get_script(
        self, name: str | None = None, tag: str | None = None
    ) -> Script | None:
        """
        Get a script by name or tag.

        Args:
            name: Name of the script (logical name or filename)
            tag: Tag to search for

        Returns:
            Script object or None if not found

        Example:
            >>> script = engine.get_script(name="input")
            >>> script = engine.get_script(tag="input")
        """
        if not hasattr(self, "scripts"):
            return None

        for script in self.scripts:
            if name is not None:
                # Match by logical name or filename
                if script.name == name:
                    return script
                if script.path is not None and script.path.name == name:
                    return script

            if tag is not None:
                # Match by tag
                if tag in script.tags:
                    return script

        return None

    @abstractmethod
    def run(self, **kwargs: Any) -> subprocess.CompletedProcess:
        """
        Execute the engine calculation.

        Args:
            **kwargs: Additional arguments for the specific engine
                     (e.g., input_file, output_file, etc.)

        Returns:
            CompletedProcess object with execution results

        Raises:
            RuntimeError: If engine is not prepared (prepare() not called)
        """
        if not hasattr(self, "work_dir"):
            raise RuntimeError("Engine not prepared. Call prepare() first.")
        pass

    def clean(self, keep_scripts: bool = True) -> None:
        """
        Clean up calculation files.

        Args:
            keep_scripts: Whether to keep input scripts (default: True)
        """
        if not hasattr(self, "work_dir") or not self.work_dir.exists():
            return

        if not keep_scripts and hasattr(self, "scripts"):
            for script in self.scripts:
                if script.path is not None:
                    script_path = self.work_dir / script.path.name
                    if script_path.exists():
                        script_path.unlink()

    def list_output_files(self) -> list[Path]:
        """
        List all output files in the working directory.

        Returns:
            List of output file paths
        """
        if not hasattr(self, "work_dir") or not self.work_dir.exists():
            return []

        # Get all script paths to exclude them
        script_paths = set()
        if hasattr(self, "scripts"):
            for script in self.scripts:
                if script.path is not None:
                    script_paths.add(self.work_dir / script.path.name)

        # Return all files except scripts
        return [
            f for f in self.work_dir.iterdir() if f.is_file() and f not in script_paths
        ]

    def get_output_file(self, name: str | None = None) -> Path | None:
        """
        Get the output file path.

        Args:
            name: Name of the output file. If None, returns default output file.

        Returns:
            Path to the output file or None if not found
        """
        if not hasattr(self, "work_dir") or not self.work_dir.exists():
            return None

        if name is not None:
            output_path = self.work_dir / name
            if output_path.exists():
                return output_path
            return None

        # Try to find default output file
        output_files = self.list_output_files()
        if output_files:
            # Return the first output file (could be improved with heuristics)
            return output_files[0]

        return None
