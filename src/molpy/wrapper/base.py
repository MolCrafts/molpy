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
    env_vars: dict[str, str] = field(default_factory=dict)
    env: str | Path | None = None
    env_manager: str | None = None

    def _normalized_env_manager(self) -> str | None:
        if self.env is None and self.env_manager is None:
            return None

        if self.env is None or self.env_manager is None:
            raise ValueError(
                "Wrapper environment configuration is incomplete: set both env and env_manager, or set neither."
            )

        manager = self.env_manager.strip().lower()
        if manager in {"pip", "venv", "virtualenv"}:
            return "venv"
        if manager == "conda":
            return "conda"

        raise ValueError(
            f"Unsupported env_manager '{self.env_manager}'. Supported values: 'conda', 'pip' (venv), 'venv'."
        )

    def _looks_like_path(self, value: str) -> bool:
        # Best-effort heuristic:
        # - explicit path separators (POSIX/Windows)
        # - explicit relative/absolute prefixes
        return (
            (os.sep in value)
            or ("/" in value)
            or ("\\" in value)
            or value.startswith((".", "~", "/"))
        )

    def _env_bin_dir(self, prefix: Path) -> Path:
        # Windows virtualenvs use 'Scripts', POSIX uses 'bin'.
        return prefix / ("Scripts" if os.name == "nt" else "bin")

    def _conda_exe(self) -> str | None:
        """Return the conda executable to use, if available."""

        # CONDA_EXE is set by conda when environments are activated.
        conda_exe = os.environ.get("CONDA_EXE")
        if conda_exe:
            return conda_exe

        return shutil.which("conda")

    def _conda_run_prefix(self) -> list[str] | None:
        """Return the prefix for running commands inside a conda env, or None."""

        if self._normalized_env_manager() != "conda":
            return None

        conda_exe = self._conda_exe()
        if conda_exe is None:
            # User explicitly requested conda isolation. Keep behavior consistent
            # across platforms/CI by still constructing a `conda run` command.
            # If conda truly isn't installed, subprocess will error rather than
            # silently ignoring env/env_manager.
            conda_exe = "conda"

        env_value = self.env
        if env_value is None:
            raise ValueError("env must be set when env_manager='conda'.")

        if isinstance(env_value, Path):
            return [conda_exe, "run", "-p", str(env_value)]

        env_str = str(env_value)
        if self._looks_like_path(env_str):
            return [conda_exe, "run", "-p", env_str]

        return [conda_exe, "run", "-n", env_str]

    def _venv_prefix(self) -> Path | None:
        if self._normalized_env_manager() != "venv":
            return None
        if self.env is None:
            raise ValueError("env must be set when env_manager is 'pip'/'venv'.")
        return self.env if isinstance(self.env, Path) else Path(str(self.env))

    def _resolve_executable_via_env(self) -> str | None:
        """Resolve the executable inside the configured env (best-effort)."""

        manager = self._normalized_env_manager()
        if manager is None:
            return None

        # If exe is already an absolute existing file, use it directly.
        exe_path = Path(self.exe)
        if exe_path.is_file():
            return str(exe_path)

        if manager == "venv":
            prefix = self._venv_prefix()
            if prefix is None:
                return None
            bin_dir = self._env_bin_dir(prefix)

            candidates = [bin_dir / self.exe]
            if os.name == "nt" and not str(self.exe).lower().endswith(".exe"):
                candidates.append(bin_dir / f"{self.exe}.exe")

            for candidate in candidates:
                if candidate.is_file():
                    return str(candidate)

            # Fall back to which() using the venv PATH.
            merged = self._merged_env()
            return shutil.which(self.exe, path=merged.get("PATH"))

        # conda
        prefix = self._conda_run_prefix()
        if prefix is None:
            return None

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

        resolved = self._resolve_executable_via_env()
        if resolved is not None:
            return resolved

        # For venv/pip we already injected PATH in _merged_env(); use it here too.
        merged = self._merged_env()
        return shutil.which(self.exe, path=merged.get("PATH"))

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
        cwd: Path | None = None,
        capture_output: bool = True,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        prefix = self._conda_run_prefix()

        final_args = [self.exe]
        if args:
            final_args.extend(args)

        if prefix is not None:
            final_args = [*prefix, *final_args]

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

        manager = self._normalized_env_manager()
        if manager == "venv":
            prefix = self._venv_prefix()
            if prefix is not None:
                bin_dir = self._env_bin_dir(prefix)
                existing = merged.get("PATH", "")
                merged["PATH"] = str(bin_dir) + (os.pathsep + existing if existing else "")
                merged["VIRTUAL_ENV"] = str(prefix)

        merged.update(self.env_vars)
        return merged

    def __repr__(self) -> str:
        workdir_str = str(self.workdir) if self.workdir else "None"
        env_bits = ""
        if self.env is not None or self.env_manager is not None:
            env_bits = f", env={self.env!r}, env_manager={self.env_manager!r}"
        return f"<{self.__class__.__name__}(name='{self.name}', exe='{self.exe}', workdir={workdir_str}{env_bits})>"
