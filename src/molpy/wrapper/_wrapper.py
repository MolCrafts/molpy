"""Base Wrapper class for external package wrappers.

Wrappers are minimal shells around external binaries and CLIs.
They are peer-level to Adapters:
- Adapter: Keeps MolPy ↔ external data structures in sync
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)

Wrappers MUST NOT contain high-level domain logic.

A wrapper's executable, environment, environment variables and time limit
are its ``wrapper.<tool>`` settings in molpy's configuration
(:mod:`molpy.config`); each run is logged to ``molpy.wrapper.<tool>``
(:mod:`mollog`).
"""

from __future__ import annotations

import subprocess
from abc import ABC
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import mollog

from molpy.config import load_config, tool_settings

from ._environment import EnvironmentSpec
from ._process import run_process

if TYPE_CHECKING:
    from molcfg import Config


class Wrapper(ABC):
    """Minimal base class for external tool wrappers.

    A subclass names its configuration table in :attr:`tool` (``"tleap"``
    reads ``[wrapper.tleap]``, falling back to ``[wrapper]``).

    Args:
        workdir: Where the tool runs; ``None`` is the caller's directory.
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).

    Attributes:
        settings: The resolved :class:`~molpy.config.ToolSettings`.
        exe: The executable (``settings.executable``).
        environment: The :class:`EnvironmentSpec` the tool runs in.
        env_vars: Variables set for the subprocess.
        timeout: Seconds before the subprocess is killed (``None``: no limit).
        logger: The tool's logger, ``molpy.wrapper.<tool>``.

    Raises:
        KeyError: :attr:`tool` is not a configured wrapper.
        ValueError: The configured environment is incomplete or its manager
            unsupported.
    """

    tool: ClassVar[str]

    def __init__(
        self,
        workdir: str | Path | None = None,
        *,
        config: Config | None = None,
    ) -> None:
        self.workdir = Path(workdir) if workdir is not None else None
        self.settings = tool_settings(
            config if config is not None else load_config(), f"wrapper.{self.tool}"
        )
        if self.settings.executable is None:
            raise ValueError(f"wrapper.{self.tool}.executable is not configured")
        self.exe: str = self.settings.executable
        self.environment = EnvironmentSpec.resolve(
            self.settings.env,
            self.settings.env_manager,
            conda_executable=self.settings.conda_executable,
        )
        self.env_vars = dict(self.settings.env_vars)
        self.timeout = self.settings.timeout
        self.logger = mollog.get_logger(f"molpy.wrapper.{self.tool}")

    def resolve_executable(self) -> str | None:
        """Resolve the configured executable to an absolute path if possible.

        Returns:
            The resolved executable path, or None if it cannot be found.
        """
        return self.environment.resolve_executable(self.exe)

    def is_available(self) -> bool:
        """Return True if the executable can be resolved on this machine."""
        return self.resolve_executable() is not None

    def check(self) -> str:
        """Validate the wrapper configuration.

        Returns:
            The resolved executable path.

        Raises:
            FileNotFoundError: if the executable is not found.
        """
        resolved = self.resolve_executable()
        if resolved is None:
            raise FileNotFoundError(
                f"Executable '{self.exe}' for {type(self).__name__} is not available. "
                "Install the tool and put it on PATH, or set "
                f"wrapper.{self.tool}.executable / env / env_manager in molpy's "
                "configuration."
            )
        return resolved

    def run(
        self,
        args: list[str] | None = None,
        *,
        input_text: str | None = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute the wrapper's command in the configured workdir.

        Args:
            args: Command-line arguments (without the executable name).
            input_text: Text to send to stdin.
            capture_output: Whether to capture stdout/stderr.
            check: Whether to raise if returncode != 0.

        Returns:
            The completed process result.
        """
        command = [*self.environment.command_prefix(), self.exe, *(args or [])]
        if self.workdir is not None:
            self.workdir.mkdir(parents=True, exist_ok=True)
        return run_process(
            self.logger,
            command,
            cwd=self.workdir,
            env=self.environment.merge_environ(extra=self.env_vars),
            capture_output=capture_output,
            check=check,
            timeout=self.timeout,
            input_text=input_text,
        )

    def __repr__(self) -> str:
        workdir_str = str(self.workdir) if self.workdir else "None"
        return (
            f"<{self.__class__.__name__}(exe='{self.exe}', "
            f"workdir={workdir_str}, environment={self.environment!r})>"
        )


def run_step(
    tool: Wrapper,
    output: Path,
    call: Callable[[], subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    """Run one tool step and require the file it must write.

    Args:
        tool: The wrapper ``call`` runs; checked for its executable first.
        output: The file the step must leave behind.
        call: Runs the step (typically a bound wrapper method in a lambda).

    Returns:
        The completed process.

    Raises:
        RuntimeError: The executable is missing, exits non-zero, or leaves
            ``output`` unwritten. The message carries the tool's stderr, or
            its stdout when stderr is empty (tleap and prepgen report there).
    """
    if not tool.is_available():
        raise RuntimeError(
            f"{tool.exe} is not available: not on PATH or in {tool.environment!r}"
        )
    result = call()
    if result.returncode != 0 or not output.is_file():
        if result.returncode == 0:  # a failed exit is already logged by run()
            tool.logger.error("step output missing", output=str(output))
        raise RuntimeError(
            f"{tool.exe} failed (exit {result.returncode}, {output.name} "
            f"{'written' if output.is_file() else 'missing'}):\n"
            f"{result.stderr or result.stdout}"
        )
    return result
