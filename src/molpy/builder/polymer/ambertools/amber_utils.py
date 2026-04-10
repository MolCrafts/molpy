"""Utility functions for Amber-based polymer building."""

from __future__ import annotations

from pathlib import Path

from molpy.wrapper import AntechamberWrapper, PrepgenWrapper, TLeapWrapper


def configure_amber_wrappers(
    workdir: Path,
    conda_env: str = "AmberTools25",
) -> tuple[AntechamberWrapper, PrepgenWrapper, TLeapWrapper]:
    """Configure Amber tool wrappers with consistent settings.

    Args:
        workdir: Working directory for intermediate files.
        conda_env: Conda environment name containing AmberTools.

    Returns:
        Tuple of (AntechamberWrapper, PrepgenWrapper, TLeapWrapper)
        configured for the specified conda environment.

    Example:
        >>> from pathlib import Path
        >>> workdir = Path("/tmp/amber_work")
        >>> antechamber, prepgen, tleap = configure_amber_wrappers(workdir)
        >>> antechamber.is_available()
        True
    """
    antechamber = AntechamberWrapper(
        name="antechamber",
        exe="antechamber",
        workdir=workdir,
        env=conda_env,
        env_manager="conda",
    )

    prepgen = PrepgenWrapper(
        name="prepgen",
        exe="prepgen",
        workdir=workdir,
        env=conda_env,
        env_manager="conda",
    )

    tleap = TLeapWrapper(
        name="tleap",
        exe="tleap",
        workdir=workdir,
        env=conda_env,
        env_manager="conda",
    )

    return antechamber, prepgen, tleap


def check_amber_tools_available(conda_env: str = "AmberTools25") -> bool:
    """Check if AmberTools are available in the conda environment.

    Args:
        conda_env: Conda environment name to check.

    Returns:
        True if all required tools are available, False otherwise.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        antechamber, prepgen, tleap = configure_amber_wrappers(Path(tmpdir), conda_env)
        return all(
            [
                antechamber.is_available(),
                prepgen.is_available(),
                tleap.is_available(),
            ]
        )
