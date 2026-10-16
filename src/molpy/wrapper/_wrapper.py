"""Base Wrapper class for external package wrappers.

Wrappers are minimal shells around external binaries and CLIs.
They are peer-level to Adapters:
- Adapter: Converts MolPy ↔ external data structures in memory
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)

Wrappers MUST NOT contain high-level domain logic.

A wrapper's executable, environment, environment variables and time limit
are its ``wrapper.<tool>`` settings in molpy's configuration
(:mod:`molpy.config`), resolved by the shared :class:`~molpy.wrapper.Tool`
core; each run is logged to ``molpy.wrapper.<tool>`` (:mod:`mollog`).
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

from molpy.wrapper._process import ToolError, tail_of
from molpy.wrapper._tool import Tool


class Wrapper(Tool):
    """Minimal base class for external tool wrappers.

    A subclass names its configuration table in :attr:`tool` (``"tleap"``
    reads ``[wrapper.tleap]``, falling back to ``[wrapper]``). Settings,
    environment, executable, logger and the failure contract come from
    :class:`~molpy.wrapper.Tool`.

    Args:
        workdir: Where the tool runs; ``None`` is the caller's directory.
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).

    Raises:
        ValueError: The configured environment is incomplete or its manager
            unsupported, or no executable is configured.
    """

    group: ClassVar[str] = "wrapper"

    def run(
        self,
        args: Sequence[str] = (),
        *,
        input_text: str | None = None,
        capture_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the wrapper's command in :attr:`workdir`.

        Args:
            args: Command-line arguments (without the executable).
            input_text: Text to send to stdin.
            capture_output: Whether to capture stdout/stderr.
            check: Raise :class:`~molpy.wrapper.ToolError` on a non-zero exit.

        Returns:
            The completed process result.

        Raises:
            ToolError: The tool could not start, timed out, or (with *check*)
                exited non-zero.
        """
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        return self.run_command(
            self.command(args),
            cwd=self.workdir,
            capture_output=capture_output,
            check=check,
            input_text=input_text,
        )


def run_step(
    tool: Tool,
    output: Path,
    call: Callable[[], subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    """Run one tool step and require the file it must write.

    Args:
        tool: The tool ``call`` runs; checked for its executable first.
        output: The file the step must leave behind.
        call: Runs the step (typically a bound wrapper method in a lambda).

    Returns:
        The completed process.

    Raises:
        ToolError: The executable is missing, the step fails, or it exits 0
            but leaves ``output`` unwritten. The error carries the tool's
            stderr, or its stdout when stderr is empty (tleap and prepgen
            report there).
    """
    tool.check_executable()
    result = call()
    if not output.is_file():
        tool.logger.error("step output missing", output=str(output))
        raise ToolError(
            tool.key,
            f"{tool.executable} left {output.name} unwritten",
            returncode=result.returncode,
            tail=tail_of(result.stderr, result.stdout),
        )
    return result
