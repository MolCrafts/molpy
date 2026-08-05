"""Utility functions for Amber-based polymer building."""

from __future__ import annotations

from pathlib import Path

from molpy.wrapper import AntechamberWrapper, EnvSpec, PrepgenWrapper, TLeapWrapper


def configure_amber_wrappers(
    workdir: Path,
    env: str | Path | None = None,
    *,
    env_manager: str | None = None,
) -> tuple[AntechamberWrapper, PrepgenWrapper, TLeapWrapper]:
    """Configure Amber tool wrappers with consistent settings.

    Environment isolation uses the shared :class:`~molpy.wrapper.EnvSpec`
    contract: omit both ``env`` and ``env_manager`` for the system
    ``PATH``, or set both explicitly (no auto-detection).

    Args:
        workdir: Working directory for intermediate files.
        env: Conda env name / prefix, or venv prefix.  ``None`` with
            ``env_manager=None`` means the system environment.
        env_manager: ``"conda"``, ``"venv"``, ``"pip"``, or
            ``"virtualenv"``.  Must be set together with ``env``.

    Returns:
        Tuple of (AntechamberWrapper, PrepgenWrapper, TLeapWrapper).

    Example:
        >>> from pathlib import Path
        >>> workdir = Path("/tmp/amber_work")
        >>> antechamber, prepgen, tleap = configure_amber_wrappers(workdir)
        >>> antechamber, prepgen, tleap = configure_amber_wrappers(
        ...     workdir, env="AmberTools25", env_manager="conda"
        ... )
    """
    spec = EnvSpec.resolve(env, env_manager)

    antechamber = AntechamberWrapper(
        name="antechamber",
        exe="antechamber",
        workdir=workdir,
        env=spec.env,
        env_manager=spec.env_manager,
    )

    prepgen = PrepgenWrapper(
        name="prepgen",
        exe="prepgen",
        workdir=workdir,
        env=spec.env,
        env_manager=spec.env_manager,
    )

    tleap = TLeapWrapper(
        name="tleap",
        exe="tleap",
        workdir=workdir,
        env=spec.env,
        env_manager=spec.env_manager,
    )

    return antechamber, prepgen, tleap


def check_amber_tools_available(
    env: str | Path | None = None,
    *,
    env_manager: str | None = None,
) -> bool:
    """Check if AmberTools binaries are available.

    Defaults to the system environment.  Pass both ``env`` and
    ``env_manager`` to check inside an isolated environment.

    Args:
        env: Conda env name / prefix, or venv prefix.
        env_manager: Must pair with ``env`` (see :class:`~molpy.wrapper.EnvSpec`).

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
