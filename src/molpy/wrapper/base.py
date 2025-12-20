"""Base Wrapper class for external package wrappers.

Wrappers are minimal shells around external binaries and CLIs.
They are peer-level to Adapters:
- Adapter: Keeps MolPy ↔ external data structures in sync
- Wrapper: Encapsulates external package invocation (binaries, CLIs, scripts)

Wrappers MUST NOT contain high-level domain logic.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Wrapper(ABC):
    """Minimal base class for external tool wrappers."""

    name: str
    exe: str
    workdir: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    conda_env: str | None = None
    conda_prefix: Path | None = None

    def _conda_exe(self) -> str | None:
        """Return the conda executable to use, if available."""

        # CONDA_EXE is set by conda when environments are activated.
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            return conda_exe
        return shutil.which("conda")

    def _conda_run_prefix(self) -> list[str] | None:
        """Return the prefix for running commands inside a conda env, or None."""

        if self.conda_env is None and self.conda_prefix is None:
            return None

        conda_exe = self._conda_exe()
        if conda_exe is None:
            return None

        if self.conda_env is not None and self.conda_prefix is not None:
            raise ValueError(
                "Wrapper conda configuration is ambiguous: set only one of conda_env or conda_prefix."
            )

        if self.conda_env is not None:
            return [conda_exe, "run", "-n", self.conda_env]

        return [conda_exe, "run", "-p", str(self.conda_prefix)]

    def _resolve_executable_via_conda(self) -> str | None:
        """Resolve the executable inside the configured conda env (best-effort)."""

        prefix = self._conda_run_prefix()
        if prefix is None:
            return None

        # If exe is already an absolute existing file, use it directly.
        exe_path = Path(self.exe)
        if exe_path.is_file():
            return str(exe_path)

        try:
            proc = subprocess.run(
                [*prefix, "which", self.exe],
                capture_output=True,
                text=True,
                check=False,
                env=self._merged_env(),
            )
        except OSError:
            return None

        if proc.returncode != 0:
            return None
        resolved = (proc.stdout or "").strip().splitlines()[:1]
        if not resolved:
            return None
        return resolved[0]

    def resolve_executable(self) -> str | None:
        """Resolve the configured executable to an absolute path if possible.

        Returns:
            The resolved executable path, or None if it cannot be found.
        """

        exe_path = Path(self.exe)
        if exe_path.is_file():
            return str(exe_path)

        resolved = self._resolve_executable_via_conda()
        if resolved is not None:
            return resolved

        return shutil.which(self.exe)

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
                "Install the tool and ensure it is on PATH, or configure conda_env/conda_prefix, "
                "or set wrapper.exe to an absolute path."
            )
        return resolved

    def run(
        self,
        args: list[str] | None = None,
        *,
        input_text: str | None = None,
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        conda_prefix = self._conda_run_prefix()
        final_args = [self.exe]
        if args:
            final_args.extend(args)

        if conda_prefix is not None:
            final_args = [*conda_prefix, *final_args]

        real_cwd = cwd or self.workdir
        if real_cwd is not None:
            real_cwd.mkdir(parents=True, exist_ok=True)

        return subprocess.run(
            final_args,
            cwd=str(real_cwd) if real_cwd is not None else None,
            input=input_text,
            capture_output=capture_output,
            text=True,
            env=self._merged_env(),
            check=check,
        )

    def _merged_env(self) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update(self.env)
        return merged

    def __repr__(self) -> str:
        workdir_str = str(self.workdir) if self.workdir else "None"
        conda_bits = ""
        if self.conda_env is not None:
            conda_bits = f", conda_env='{self.conda_env}'"
        elif self.conda_prefix is not None:
            conda_bits = f", conda_prefix='{self.conda_prefix}'"
        return f"<{self.__class__.__name__}(name='{self.name}', exe='{self.exe}', workdir={workdir_str}{conda_bits})>"
