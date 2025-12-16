"""Base Wrapper class for external package wrappers.

This module provides the minimal base class for wrapping external binaries and CLIs.
Wrappers are peer-level to Adapters:
- Adapter: Keeps MolPy ↔ external data structures in sync
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)

Wrappers must NOT contain high-level domain logic such as "build ligand system",
"solvate box", etc. That belongs in Compute nodes.

Wrappers are just well-typed, consistent shells around external tools.
"""

from __future__ import annotations

import os
import subprocess
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Wrapper(ABC):
    """Minimal base class for external package wrappers.

    This class does NOT encode domain-specific semantics.
    It only encapsulates how to invoke the underlying binary.

    **Responsibilities:**
    - Manage executable path/name
    - Handle working directory
    - Manage environment variables
    - Provide generic run() method for subprocess invocation

    **Limitations:**
    - Does NOT build command-line arguments (Compute nodes do this)
    - Does NOT construct scripts (Compute nodes do this)
    - Does NOT interpret results (Compute nodes do this)

    **Usage Pattern:**
        >>> wrapper = MyWrapper(name="mytool", exe="mytool", workdir=Path("work"))
        >>> # Compute node builds args
        >>> args = ["-i", "input.txt", "-o", "output.txt"]
        >>> proc = wrapper.run(args=args)
        >>> # Compute node interprets proc.returncode, proc.stdout, etc.

    Attributes:
        name: Human-readable name of the wrapper/tool
        exe: Binary name or path (e.g., "antechamber", "tleap", "/usr/bin/antechamber")
        workdir: Default working directory for tool execution
        env: Additional environment variables to set (merged with current env)
    """

    name: str
    exe: str  # binary name or path, e.g. "antechamber", "tleap"
    workdir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    def run(
        self,
        args: list[str] | None = None,
        *,
        input_text: str | None = None,
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Run the external command with given arguments.

        This is the ONLY place where subprocess.run is invoked in wrappers.
        High-level logic (what args, what input_text) is handled by Compute nodes.

        Args:
            args: Command-line arguments to pass to the executable.
                 If None, only the executable is run.
            input_text: Text to send to stdin (optional).
            cwd: Working directory for execution. If None, uses self.workdir.
            capture_output: Whether to capture stdout/stderr (default: True).
            check: Whether to raise CalledProcessError on non-zero exit (default: False).

        Returns:
            CompletedProcess object with execution results.

        Raises:
            subprocess.CalledProcessError: If check=True and process returns non-zero.
            FileNotFoundError: If executable is not found.

        Example:
            >>> wrapper = Wrapper(name="test", exe="echo")
            >>> proc = wrapper.run(args=["hello", "world"])
            >>> print(proc.stdout)
            hello world
        """
        final_args = [self.exe]
        if args:
            final_args.extend(args)

        real_cwd = cwd or self.workdir
        if real_cwd is not None:
            real_cwd.mkdir(parents=True, exist_ok=True)

        proc = subprocess.run(
            final_args,
            cwd=str(real_cwd) if real_cwd is not None else None,
            input=input_text,
            capture_output=capture_output,
            text=True,
            env=self._merged_env(),
            check=check,
        )
        return proc

    def _merged_env(self) -> dict[str, str]:
        """Internal helper to merge self.env with the current process env.

        Returns:
            Merged environment dictionary.
        """
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    def __repr__(self) -> str:
        """String representation of wrapper."""
        workdir_str = str(self.workdir) if self.workdir else "None"
        return (
            f"<{self.__class__.__name__}(name='{self.name}', "
            f"exe='{self.exe}', workdir={workdir_str})>"
        )
