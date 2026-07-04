"""
Version information for MolPy.

This module provides simple version information for MolPy.

MolPy and its Rust backend ``molcrafts-molrs`` are released together and share a
single version number. :func:`check_molrs_version` verifies that the installed
molrs matches this version; it is called once on ``import molpy`` so a stale
editable build or an out-of-date pin surfaces immediately.
"""

version = "0.6.0"
release_date = "2026-07-03"


def check_molrs_version(*, strict: bool = False) -> str | None:
    """Check that the installed ``molcrafts-molrs`` matches MolPy's version.

    MolPy and molrs converge on one version number and are released as a pair, so
    a mismatch almost always means a stale editable molrs build or an out-of-date
    dependency pin.

    Args:
        strict: When ``True``, raise :class:`ImportError` on a mismatch instead of
            emitting a warning.

    Returns:
        The installed molrs version string when it does **not** match (after
        warning), or ``None`` when the versions agree or molrs is not installed.
    """
    try:
        from importlib.metadata import PackageNotFoundError
        from importlib.metadata import version as _pkg_version
    except ImportError:  # pragma: no cover - importlib.metadata ships with 3.12
        return None

    try:
        molrs_version = _pkg_version("molcrafts-molrs")
    except PackageNotFoundError:
        # molrs is a hard runtime dependency, but do not hard-fail the version
        # check if metadata is unavailable (e.g. an unusual install layout).
        return None

    if molrs_version == version:
        return None

    message = (
        f"Version mismatch: molpy {version} but molcrafts-molrs {molrs_version}. "
        "MolPy and molrs are released together and should share a version. "
        f"Install a matching molrs (`pip install molcrafts-molrs=={version}`) or "
        "rebuild the editable molrs (`maturin develop` in molrs-python)."
    )
    if strict:
        raise ImportError(message)

    import warnings

    warnings.warn(message, stacklevel=2)
    return molrs_version


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


# Run the compatibility check once, when this module is imported — and therefore
# on ``import molpy``, which imports version first. Guarded so a version check
# can never prevent import.
try:
    check_molrs_version()
except Exception:  # pragma: no cover - defensive: never break import
    pass
