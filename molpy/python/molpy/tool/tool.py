"""The one tool-resolution core every external program molpy runs is built on.

A :class:`Tool` resolves, in order: its settings (:func:`molpy.config.tool_settings`
of its ``"<group>.<tool>"`` table) → the :class:`EnvironmentSpec` it runs in →
its executable → the environment variables → the time limit → its logger
(``molpy.<group>.<tool>``). It starts every subprocess through
:func:`~molpy.tool.process.run_process`, so every failure is one
:class:`ToolError`.

:class:`molpy.engine.Engine` (group ``engine``) and the AmberTools programs'
:class:`~molpy.tool.ambertools.programs.AmberProgram` (group ``ambertools``)
subclass it; a third-party engine subclasses :class:`~molpy.engine.Engine`
and names its table in ``tool``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import mollog

from molpy.config import load_config, tool_settings
from molpy.tool.environment import EnvironmentSpec
from molpy.tool.process import ToolError, run_process

if TYPE_CHECKING:
    from molcfg import Config


class Tool:
    """An external program resolved from molpy's configuration.

    Subclasses set :attr:`group` (the base class does: ``"engine"`` or
    ``"ambertools"``) and :attr:`tool`, the name of the configuration table
    (``[<group>.<tool>]``, falling back to ``[<group>]``).

    Args:
        workdir: Where the tool runs; ``None`` lets the subclass decide.
        config: molpy's configuration (:func:`molpy.config.load_config`);
            ``None`` loads it (package defaults, user and project files).

    Attributes:
        settings: The resolved :class:`~molpy.config.ToolSettings`.
        environment: The :class:`EnvironmentSpec` the tool runs in.
        logger: The tool's logger, ``molpy.<group>.<tool>``.
        executable: The program to run (``settings.executable``, else
            :meth:`_default_executable`).
        workdir: Where the tool runs, or ``None``.
        launcher: MPI / scheduler prefix before the executable (engines;
            empty for the AmberTools programs).
        env_vars: Environment variables set for the subprocess.
        timeout: Seconds before a subprocess is killed (``None``: no limit).

    Raises:
        KeyError: :attr:`group` is not a configured group.
        ValueError: The configured environment is incomplete or its manager
            unsupported, or no executable is configured and the tool has no
            default.
    """

    group: ClassVar[str]
    tool: ClassVar[str]

    def __init__(
        self,
        workdir: str | Path | None = None,
        *,
        config: Config | None = None,
    ) -> None:
        self.settings = tool_settings(
            config if config is not None else load_config(), self.key
        )
        self.environment = EnvironmentSpec.resolve(
            self.settings.env,
            self.settings.env_manager,
            conda_executable=self.settings.conda_executable,
        )
        self.logger = mollog.get_logger(f"molpy.{self.key}")
        self.executable: str = (
            self.settings.executable
            if self.settings.executable is not None
            else self._default_executable()
        )
        self.workdir = Path(workdir) if workdir is not None else None
        self.launcher: list[str] = list(self.settings.launcher)
        self.env_vars: dict[str, str] = dict(self.settings.env_vars)
        self.timeout = self.settings.timeout

    @property
    def key(self) -> str:
        """The settings table, ``"<group>.<tool>"`` (``"engine.lammps"``)."""
        return f"{self.group}.{self.tool}"

    def _default_executable(self) -> str:
        """The executable when the configuration names none.

        Override it to give a third-party tool a default, or to probe
        several names (:class:`~molpy.engine.LammpsEngine`).

        Raises:
            ValueError: This tool has no default; configure
                ``<group>.<tool>.executable``.
        """
        raise ValueError(f"{self.key}.executable is not configured")

    # -- availability ---------------------------------------------------------

    def resolve_executable(self) -> str | None:
        """The executable's absolute path in the environment, or ``None``.

        Looked up once per instance (:meth:`EnvironmentSpec.resolve_executable`).
        """
        return self.environment.resolve_executable(self.executable, logger=self.logger)

    def is_available(self) -> bool:
        """True when the executable can be found in the environment."""
        return self.resolve_executable() is not None

    def check_executable(self) -> str:
        """The executable's resolved path.

        Raises:
            ToolError: The executable is not found; the message names the
                settings to fix.
        """
        resolved = self.resolve_executable()
        if resolved is None:
            raise ToolError(
                self.key,
                f"executable {self.executable!r} not found in {self.environment!r}; "
                f"install it, or set {self.key}.executable / env / env_manager "
                "in molpy's configuration",
            )
        return resolved

    # -- running --------------------------------------------------------------

    def command(self, args: Sequence[str] = (), *, launch: bool = True) -> list[str]:
        """The full argv: ``[env prefix] [launcher] executable args``.

        The env prefix is :meth:`EnvironmentSpec.command_prefix` (conda's
        ``conda run --no-capture-output -n <env>``; empty for venv and the
        system, which act through the environment variables).

        Args:
            args: What follows the executable.
            launch: Put :attr:`launcher` before the executable (``False`` for
                a serial step such as GROMACS's ``grompp``).
        """
        return [
            *self.environment.command_prefix(no_capture_output=True),
            *(self.launcher if launch else ()),
            self.executable,
            *args,
        ]

    def run_command(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None,
        capture_output: bool,
        check: bool = True,
        input_text: str | None = None,
        log: Path | None = None,
        **fields: Any,
    ) -> subprocess.CompletedProcess[str]:
        """Start *command* in *cwd* with the tool's environment and time limit.

        The run is logged to :attr:`logger` (``process started`` /
        ``finished`` / ``failed`` / ``timed out``), each record carrying the
        command, the directory and *fields* (e.g. ``step="mdrun"``).

        Args:
            command: The full argv (usually :meth:`command`).
            cwd: Working directory; ``None`` is the caller's.
            capture_output: Capture stdout / stderr.
            check: Raise :class:`ToolError` on a non-zero exit.
            input_text: Text sent to stdin.
            log: The program's log file, the error's tail when nothing was
                captured.
            **fields: Extra structured fields on every record.

        Raises:
            ToolError: The program could not start, timed out, or (with
                *check*) exited non-zero.
        """
        return run_process(
            self.logger,
            command,
            tool=self.key,
            cwd=cwd,
            env=self._environ(),
            capture_output=capture_output,
            check=check,
            timeout=self.timeout,
            input_text=input_text,
            log=log,
            **fields,
        )

    def _environ(self) -> dict[str, str] | None:
        """The subprocess environment; ``None`` inherits the caller's.

        ``None`` when the tool runs in the system environment with no
        :attr:`env_vars`, so the subprocess inherits the parent's without a
        copy; otherwise :meth:`EnvironmentSpec.merge_environ` (venv ``PATH``
        / ``VIRTUAL_ENV``, then :attr:`env_vars`).
        """
        if self.environment.is_system and not self.env_vars:
            return None
        return self.environment.merge_environ(extra=self.env_vars)

    def __repr__(self) -> str:
        parts = [f"executable={self.executable!r}"]
        if self.workdir is not None:
            parts.append(f"workdir={str(self.workdir)!r}")
        if self.launcher:
            parts.append(f"launcher={self.launcher!r}")
        if not self.environment.is_system:
            parts.append(f"environment={self.environment!r}")
        return f"<{type(self).__name__}({', '.join(parts)})>"
