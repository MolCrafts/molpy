"""OpenMM / OPLS XML force-field I/O (molrs-backed).

Read: :func:`molrs.ff.read_forcefield_xml` / :func:`molrs.ff.read_opls_xml`.
Write: :func:`molrs.ff.write_forcefield_xml`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import molrs.ff as _mff
from molpy.core.forcefield import ForceField


import math

ANGLE_UNITS = ("radian", "degree")
_TWO_PI = 2.0 * math.pi
_ANGLE_RANGES = {
    "equilibrium": (0.0, 2.0 * math.pi),
    "phase": (-2.0 * math.pi, 2.0 * math.pi),
}


class AngleUnitWarning(UserWarning):
    """An angle value looks inconsistent with its declared ``angle_unit``."""


def _check_angle_unit(angle_unit: str) -> str:
    if angle_unit not in ANGLE_UNITS:
        raise ValueError(
            f"angle_unit must be one of {ANGLE_UNITS}, got {angle_unit!r}."
        )
    return angle_unit


def _angle_to_internal(value: float, angle_unit: str) -> float:
    """Convert an input angle in *angle_unit* to the internal unit (radians)."""
    return value if angle_unit == "radian" else math.radians(value)


def _angle_from_internal(value_rad: float, angle_unit: str) -> float:
    """Convert an internal-radians angle to *angle_unit* for serialisation."""
    return value_rad if angle_unit == "radian" else math.degrees(value_rad)


def _normalize_angle(raw: float, angle_unit: str, *, kind: str, label: str) -> float:
    """Convert *raw* (in *angle_unit*) to internal radians, warning on anomalies."""
    if angle_unit == "radian" and abs(raw) > _TWO_PI + 1e-6:
        warnings.warn(
            f"{label}={raw:g} exceeds 2π but angle_unit='radian'; the value looks "
            f"like degrees — pass angle_unit='degree' if the file is in degrees.",
            AngleUnitWarning,
            stacklevel=3,
        )
    rad = _angle_to_internal(raw, angle_unit)
    lo, hi = _ANGLE_RANGES[kind]
    if not (lo - 1e-6 <= rad <= hi + 1e-6):
        warnings.warn(
            f"{label}={rad:g} rad is far outside the {kind} sanity bound "
            f"[{lo:g}, {hi:g}] rad (angle_unit='{angle_unit}'); likely an angle_unit "
            f"mismatch.",
            AngleUnitWarning,
            stacklevel=3,
        )
    return rad


def _tag_layer(ff: ForceField, layer: int) -> None:
    """Stamp overlay layer on atom types when non-zero."""
    if layer == 0:
        return
    try:
        for cat_name in ff.style_names():
            category, sname = cat_name.split(":", 1)
            if category != "atom":
                continue
            for tname, _params in ff.types(category, sname):
                ff.set_type_param(category, sname, tname, "layer", float(layer))
    except Exception:
        pass


def _resolve_forcefield_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.exists():
        return path
    raise FileNotFoundError(f"Force field file not found: {path}")


def read_xml_forcefield(
    filepath: str | Path,
    forcefield: ForceField | None = None,
    layer: int = 0,
) -> ForceField:
    """Read an OpenMM/OPLS XML force field (molrs)."""
    path = _resolve_forcefield_path(filepath)
    loaded = _mff.read_forcefield_xml(str(path))
    _tag_layer(loaded, layer)
    if forcefield is None:
        return loaded
    forcefield.merge(loaded)
    return forcefield


def read_oplsaa_forcefield(
    filepath: str | Path,
    forcefield: ForceField | None = None,
    layer: int = 0,
) -> ForceField:
    """Read OPLS-AA / OpenMM XML with molrs OPLS unit conversion."""
    path = _resolve_forcefield_path(filepath)
    loaded = _mff.read_opls_xml(str(path))
    _tag_layer(loaded, layer)
    if forcefield is None:
        return loaded
    forcefield.merge(loaded)
    return forcefield


class XMLForceFieldReader:
    """Deprecated shell: prefer :func:`read_xml_forcefield` (molrs)."""

    def __init__(self, filepath: str | Path, *, angle_unit: str = "radian") -> None:
        self._file = Path(filepath)
        self._angle_unit = _check_angle_unit(angle_unit)

    def read(
        self,
        forcefield: ForceField | None = None,
        layer: int = 0,
    ) -> ForceField:
        return read_xml_forcefield(self._file, forcefield=forcefield, layer=layer)


class OPLSAAForceFieldReader(XMLForceFieldReader):
    """Deprecated shell: prefer :func:`read_oplsaa_forcefield` (molrs)."""

    def read(
        self,
        forcefield: ForceField | None = None,
        layer: int = 0,
    ) -> ForceField:
        return read_oplsaa_forcefield(self._file, forcefield=forcefield, layer=layer)


class XMLForceFieldWriter:
    """Write a ForceField to OpenMM-style XML (via molrs)."""

    def __init__(
        self, filepath: str | Path, precision: int = 6, *, angle_unit: str = "radian"
    ) -> None:
        self._file = Path(filepath)
        self._prec = precision
        self._angle_unit = _check_angle_unit(angle_unit)

    def write(self, forcefield: ForceField) -> None:
        """Serialize *forcefield* to XML."""
        try:
            _mff.write_forcefield_xml(str(self._file), forcefield, precision=self._prec)
        except Exception as exc:
            raise ValueError(f"Failed to write XML force field: {exc}") from exc


def write_xml_forcefield(filepath: str | Path, forcefield: ForceField) -> None:
    """Convenience function to write a force field to XML."""
    XMLForceFieldWriter(filepath).write(forcefield)
