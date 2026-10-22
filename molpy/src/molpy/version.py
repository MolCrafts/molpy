"""
Version information for MolPy.

This module provides simple version information for MolPy.

:func:`check_molrs_version` verifies that the installed ``molcrafts-molrs``
shares MolPy's **minor** version (``major.minor``). Patch-level drift is
allowed so a newer molrs patch can pair with molpy without a hard fail.
It is called once on ``import molpy`` so a stale major/minor pin surfaces
immediately.
"""

version = "0.19.0"
release_date = "2026-10-21"


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


def _molrs_requirement() -> str:
    """The ``molcrafts-molrs`` range this molpy needs: its own major.minor line.

    ``pyproject.toml`` pins exactly this range (a test holds them together).

    Returns:
        A PEP 440 specifier such as ``>=0.19.0,<0.20``.
    """
    major, minor = _minor_tuple(version)
    return f">={major}.{minor}.0,<{major}.{minor + 1}"


def check_molrs_version() -> str:
    """Require installed ``molcrafts-molrs`` to match MolPy's **major.minor** only.

    Compatibility rule (SemVer-style for the 0.x line)::

        molpy X.Y.*  ↔  molrs X.Y.*

    * **Patch is ignored** — e.g. molpy ``0.12.1`` works with molrs ``0.12.0``
      or ``0.12.3``.
    * **Major or minor mismatch fails hard** — e.g. molpy ``0.12.1`` rejects
      molrs ``0.11.9`` or ``0.13.0``.

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
        f"Patch versions are not part of the check — install any "
        f"molrs {major}.{minor}.* "
        f"(`pip install 'molcrafts-molrs{_molrs_requirement()}'`) "
        "or rebuild the editable molrs (`maturin develop` in molrs-python)."
    )


# Export version attributes
__all__ = [
    "version",
    "release_date",
    "check_molrs_version",
]


# Minor-version dependency validation is part of import. Missing metadata and
# every major.minor mismatch fail here; patch drift is allowed.
check_molrs_version()
