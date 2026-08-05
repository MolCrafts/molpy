"""Utility functions for Amber-based polymer building."""

from __future__ import annotations

from pathlib import Path

from molpy.wrapper import AntechamberWrapper, PrepgenWrapper, TLeapWrapper


def _resolve_wrapper_env(
    env: str | Path | None,
    env_manager: str | None,
) -> tuple[str | Path | None, str | None]:
    """Default to the system environment; activate only when *env* is set.

    When the user provides an environment name/path but no manager, assume
    ``conda``.  ``env_manager`` without ``env`` is incomplete and rejected.
    """
    if env is None and env_manager is None:
        return None, None
    if env is None:
        raise ValueError(
            "env_manager requires env; omit both to use the system environment."
        )
    if env_manager is None:
        return env, "conda"
    return env, env_manager


def configure_amber_wrappers(
    workdir: Path,
    env: str | Path | None = None,
    *,
    env_manager: str | None = None,
) -> tuple[AntechamberWrapper, PrepgenWrapper, TLeapWrapper]:
    """Configure Amber tool wrappers with consistent settings.

    By default wrappers use the **system** environment (``PATH``).  Pass
    ``env`` only when AmberTools lives in a named conda env (or prefix);
    ``env_manager`` defaults to ``"conda"`` when ``env`` is set.

    Args:
        workdir: Working directory for intermediate files.
        env: Optional conda env name or prefix path.  ``None`` (default)
            means do not activate any environment.
        env_manager: Environment manager (``"conda"``, ``"venv"``, …).
            Defaults to ``"conda"`` when ``env`` is provided.

    Returns:
        Tuple of (AntechamberWrapper, PrepgenWrapper, TLeapWrapper).

    Example:
        >>> from pathlib import Path
        >>> workdir = Path("/tmp/amber_work")
        >>> # system PATH (AmberTools already active / installed globally)
        >>> antechamber, prepgen, tleap = configure_amber_wrappers(workdir)
        >>> # optional named conda env
        >>> antechamber, prepgen, tleap = configure_amber_wrappers(
        ...     workdir, env="AmberTools25"
        ... )
    """
    resolved_env, resolved_manager = _resolve_wrapper_env(env, env_manager)

    antechamber = AntechamberWrapper(
        name="antechamber",
        exe="antechamber",
        workdir=workdir,
        env=resolved_env,
        env_manager=resolved_manager,
    )

    prepgen = PrepgenWrapper(
        name="prepgen",
        exe="prepgen",
        workdir=workdir,
        env=resolved_env,
        env_manager=resolved_manager,
    )

    tleap = TLeapWrapper(
        name="tleap",
        exe="tleap",
        workdir=workdir,
        env=resolved_env,
        env_manager=resolved_manager,
    )

    return antechamber, prepgen, tleap


def check_amber_tools_available(
    env: str | Path | None = None,
    *,
    env_manager: str | None = None,
) -> bool:
    """Check if AmberTools binaries are available.

    Defaults to the system environment.  Pass ``env`` to check inside a
    named conda environment (or other manager via ``env_manager``).

    Args:
        env: Optional conda env name or prefix path.
        env_manager: Environment manager; defaults to ``"conda"`` when
            ``env`` is set.

    Returns:
        True if all required tools are available, False otherwise.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        antechamber, prepgen, tleap = configure_amber_wrappers(
            Path(tmpdir), env, env_manager=env_manager
        )
        return all(
            [
                antechamber.is_available(),
                prepgen.is_available(),
                tleap.is_available(),
            ]
        )
