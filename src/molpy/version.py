"""
Version information for MolPy.

This module provides simple version information for MolPy.

:func:`check_molrs_version` verifies that the installed ``molcrafts-molrs``
shares MolPy's **minor** version (``major.minor``). Patch-level drift is
allowed so a newer molrs patch can pair with molpy without a hard fail.
It is called once on ``import molpy`` so a stale major/minor pin surfaces
immediately.
"""

version = "0.12.0"
release_date = "2026-08-04"


def _minor_tuple(ver: str) -> tuple[int, int]:
    """Return ``(major, minor)`` from a version string like ``0.10.1``."""
    # Strip local/pre-release suffixes (e.g. 0.10.1a1, 0.10.1+local).
    core = ver.split("+", 1)[0].split("-", 1)[0]
    parts = core.split(".")
    if len(parts) < 2:
        raise ValueError(f"expected at least major.minor, got {ver!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"non-numeric version {ver!r}") from exc


def check_molrs_version() -> str:
    """Require installed ``molcrafts-molrs`` to match MolPy's major.minor.

    Patch versions may differ (e.g. molpy ``0.10.0`` with molrs ``0.10.1``).
    A different major or minor fails hard.

    Returns:
        The exact installed molrs version string.

    Raises:
        ImportError: If package metadata is missing or major.minor differs.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _pkg_version

    try:
        molrs_version = _pkg_version("molcrafts-molrs")
    except PackageNotFoundError as exc:
        raise ImportError(
            "molpy requires the runtime dependency molcrafts-molrs, "
            "but its package metadata is missing"
        ) from exc

    try:
        molpy_mm = _minor_tuple(version)
        molrs_mm = _minor_tuple(molrs_version)
    except ValueError as exc:
        raise ImportError(
            f"Cannot parse versions for compatibility check: "
            f"molpy={version!r}, molcrafts-molrs={molrs_version!r} ({exc})"
        ) from exc

    if molpy_mm == molrs_mm:
        return molrs_version

    major, minor = molpy_mm
    raise ImportError(
        f"Minor-version mismatch: molpy {version} (expects {major}.{minor}.*) "
        f"but molcrafts-molrs {molrs_version}. "
        f"Install a matching minor of molrs "
        f"(`pip install 'molcrafts-molrs>={major}.{minor}.0,<{major}.{minor + 1}'`) "
        "or rebuild the editable molrs (`maturin develop` in molrs-python)."
    )


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


# Minor-version dependency validation is part of import. Missing metadata and
# every major.minor mismatch fail here; patch drift is allowed.
check_molrs_version()
