"""Base Wrapper class for external package wrappers.

Wrappers are minimal shells around external binaries and CLIs.
They are peer-level to Adapters:
- Adapter: Keeps MolPy ↔ external data structures in sync
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)

Wrappers MUST NOT contain high-level domain logic.

Environment isolation is owned by :class:`~molpy.wrapper.env.EnvSpec`
(``env`` + ``env_manager``).  See that module for supported managers.
"""

from __future__ import annotations

import subprocess
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path

from .env import EnvSpec


@dataclass
class Wrapper(ABC):
    """Minimal base class for external tool wrappers."""

    name: str
    exe: str
    workdir: Path | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    env: str | Path | None = None
    env_manager: str | None = None

    def process_env(self) -> EnvSpec:
        """Return the validated :class:`EnvSpec` for this wrapper."""
        return EnvSpec.resolve(self.env, self.env_manager)

    def resolve_executable(self) -> str | None:
        """Resolve the configured executable to an absolute path if possible.

        Returns:
            The resolved executable path, or None if it cannot be found.
        """
        return self.process_env().resolve_executable(self.exe)

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
                "Install the tool and ensure it is on PATH, or configure env/env_manager, "
                "or set wrapper.exe to an absolute path."
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
        spec = self.process_env()
        final_args = [*spec.command_prefix(), self.exe]
        if args:
            final_args.extend(args)

        real_cwd = self.workdir
        if real_cwd is not None:
            real_cwd.mkdir(parents=True, exist_ok=True)

        return subprocess.run(
            final_args,
            cwd=str(real_cwd) if real_cwd is not None else None,
            input=input_text,
            capture_output=capture_output,
            text=True,
            env=spec.merge_environ(extra=self.env_vars),
            check=check,
        )

    def __repr__(self) -> str:
        workdir_str = str(self.workdir) if self.workdir else "None"
        env_bits = ""
        if self.env is not None or self.env_manager is not None:
            env_bits = f", env={self.env!r}, env_manager={self.env_manager!r}"
        return (
            f"<{self.__class__.__name__}(name='{self.name}', "
            f"exe='{self.exe}', workdir={workdir_str}{env_bits})>"
        )
