"""
Version information for MolPy.

This module provides simple version information for MolPy.

MolPy and its Rust backend ``molcrafts-molrs`` are released together and share a
single version number. :func:`check_molrs_version` verifies that the installed
molrs matches this version; it is called once on ``import molpy`` so a stale
editable build or an out-of-date pin surfaces immediately.
"""

version = "0.9.1"
release_date = "2026-07-22"


def check_molrs_version() -> str:
    """Require the installed ``molcrafts-molrs`` to match MolPy exactly.

    MolPy and molrs converge on one version number and are released as a pair, so
    a mismatch almost always means a stale editable molrs build or an out-of-date
    dependency pin.

    Returns:
        The exact installed molrs version.

    Raises:
        ImportError: If package metadata is missing or the version differs.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        molrs_version = _pkg_version("molcrafts-molrs")
    except PackageNotFoundError as exc:
        raise ImportError(
            "molpy requires the exact runtime dependency "
            f"molcrafts-molrs=={version}, but its package metadata is missing"
        ) from exc

    if molrs_version == version:
        return molrs_version

    message = (
        f"Version mismatch: molpy {version} but molcrafts-molrs {molrs_version}. "
        "MolPy and molrs are released together and should share a version. "
        f"Install a matching molrs (`pip install molcrafts-molrs=={version}`) or "
        "rebuild the editable molrs (`maturin develop` in molrs-python)."
    )
    raise ImportError(message)


def __str__() -> str:
    """String representation of version."""
    return version


def __repr__() -> str:
    """Detailed string representation of version."""
    return f"MolPy version {version} (released {release_date})"


# Export version attributes
__all__ = [
    "version",
    "release_date",
    "check_molrs_version",
]


# Exact dependency validation is part of import. Missing metadata and every
# non-current version fail here; no warning or permissive fallback exists.
check_molrs_version()
