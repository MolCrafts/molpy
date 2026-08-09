"""LAMMPS data file I/O (structure via molrs, coeffs via molrs.ff)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from molrs import Frame, MetaValue
from molpy._frame_meta import update_frame_meta
from molpy.core.fields import CHARGE, MOL_ID, FieldFormatter
from molpy.core.forcefield import ForceField

from .base import DataReader, DataWriter


class LammpsFieldFormatter(FieldFormatter):
    """LAMMPS-specific field name translation.

    Maps LAMMPS atom_style column names to canonical field names::

        "q"   → "charge"
        "mol" → "mol_id"
    """

    _field_formatters = {
        "q": CHARGE,
        "mol": MOL_ID,
    }


def _is_int_type_token(value: object) -> bool:
    """True when ``value`` is an integer or a pure digit string (optional sign)."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return True
    s = str(value).strip()
    if not s:
        return False
    if s[0] in "+-":
        s = s[1:]
    return s.isdigit()


def _sorted_type_names(names: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    """Order type names for dense LAMMPS ids.

    Pure-integer labels sort **numerically** (``2`` before ``10``). Mixed or
    non-numeric labels keep lexicographic order. String sort of digit labels is
    a classic Type Labels bug: id 2 maps to ``\"10\"`` and Pair Coeffs scramble.
    """
    items = [str(n) for n in names]
    if items and all(_is_int_type_token(n) for n in items):
        return sorted(items, key=lambda n: int(n))
    return sorted(items)


@dataclass(frozen=True, slots=True)
class LammpsDataResult:
    """Explicit products of parsing one LAMMPS data file."""

    frame: Frame
    forcefield: ForceField
    counts: dict[str, int]
    type_labels: dict[str, list[str]]


class LammpsDataReader(DataReader[LammpsDataResult]):
    """Reader for LAMMPS data files."""

    def __init__(self, path: str | Path, atom_style: str = "full") -> None:
        super().__init__(Path(path))
        self.atom_style = atom_style

    def read(self, frame: Frame | None = None) -> LammpsDataResult:
        """Read a LAMMPS data file into frame + forcefield products.

        Structure, Type Labels, header counts, and ``* Coeffs`` text are
        produced by :func:`molrs.io.read_lammps_data` (single pass). Coeffs
        become a :class:`~molpy.ForceField` via
        :func:`molrs.ff.read_lammps_data_coeffs`. This class only adapts the
        molpy surface (``type`` column, atom_style column drop, result bundle).
        """
        del frame  # molrs always returns a new Frame
        if not self._path.exists():
            raise FileNotFoundError(f"LAMMPS data file not found: {self._path}")

        import molrs.ff as mff
        import molrs.io
        from molpy._frame_meta import get_frame_meta

        try:
            frame = molrs.io.read_lammps_data(
                str(self._path), atom_style=self.atom_style
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            msg = str(exc).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(str(exc)) from exc
            raise ValueError(f"Failed to read LAMMPS data file: {exc}") from exc

        missing_axes = self._missing_box_axes(frame)
        if missing_axes:
            raise ValueError(f"missing box bounds for axis {missing_axes}")

        type_labels = self._type_labels_from_meta(frame)
        self._adapt_frame(frame, type_labels)

        coeffs_text = get_frame_meta(frame, "lammps_coeffs_text", None)
        if coeffs_text:
            try:
                forcefield = mff.read_lammps_data_coeffs(
                    str(coeffs_text),
                    units="real",
                    atom_labels=type_labels.get("atom"),
                    bond_labels=type_labels.get("bond"),
                    angle_labels=type_labels.get("angle"),
                    dihedral_labels=type_labels.get("dihedral"),
                    improper_labels=type_labels.get("improper"),
                )
            except Exception as e:
                msg = str(e)
                # Preserve historical error shape for bad *Coeffs lines.
                if (
                    "not a number" in msg
                    or "not a float" in msg
                    or "unexpected token" in msg
                    or "pair_coeff" in msg
                ):
                    raise ValueError(f"malformed PairCoeffs: {e}") from e
                raise ValueError(
                    f"Failed to parse LAMMPS force-field coeffs: {e}"
                ) from e
        else:
            forcefield = mff.ForceField("LAMMPS")

        counts = self._counts_from_meta(frame)
        for key, block in (
            ("atoms", "atoms"),
            ("bonds", "bonds"),
            ("angles", "angles"),
            ("dihedrals", "dihedrals"),
            ("impropers", "impropers"),
        ):
            if block in frame:
                counts.setdefault(key, int(frame[block].nrows))

        update_frame_meta(
            frame,
            {
                "format": MetaValue("string", "lammps_data"),
                "atom_style": MetaValue("string", self.atom_style),
                "source_file": MetaValue("string", str(self._path)),
            },
        )

        return LammpsDataResult(
            frame=frame,
            forcefield=forcefield,
            counts=counts,
            type_labels={
                f"{key}_types": [labels[i] for i in sorted(labels)]
                for key, labels in type_labels.items()
            },
        )

    def _type_labels_from_meta(self, frame: Frame) -> dict[str, dict[int, str]]:
        """Parse molrs ``*_type_labels`` meta (``id:label,...``) into maps."""
        from molpy._frame_meta import get_frame_meta

        out: dict[str, dict[int, str]] = {}
        for kind, meta_key in (
            ("atom", "atom_type_labels"),
            ("bond", "bond_type_labels"),
            ("angle", "angle_type_labels"),
            ("dihedral", "dihedral_type_labels"),
            ("improper", "improper_type_labels"),
        ):
            packed = get_frame_meta(frame, meta_key, None)
            if not packed:
                continue
            id_to_label: dict[int, str] = {}
            for item in str(packed).split(","):
                item = item.strip()
                if not item or ":" not in item:
                    continue
                sid, lab = item.split(":", 1)
                try:
                    id_to_label[int(sid)] = lab
                except ValueError:
                    continue
            if id_to_label:
                out[kind] = id_to_label
        return out

    def _counts_from_meta(self, frame: Frame) -> dict[str, int]:
        from molpy._frame_meta import get_frame_meta

        counts: dict[str, int] = {}
        packed = get_frame_meta(frame, "lammps_counts", None)
        if not packed:
            return counts
        for part in str(packed).split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            try:
                counts[k.strip()] = int(v)
            except ValueError:
                continue
        return counts

    def _missing_box_axes(self, frame: Frame) -> list[str]:
        """Axes absent from the data header (not merely zero-length)."""
        from molpy._frame_meta import get_frame_meta

        packed = get_frame_meta(frame, "lammps_box_axes", None)
        if packed is None:
            # Older molrs without the flag: treat missing box as all axes.
            return [] if frame.box is not None else ["x", "y", "z"]
        flags: dict[str, bool] = {}
        for part in str(packed).split(","):
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            flags[k.strip()] = v.strip() in ("1", "true", "True")
        return [ax for ax in ("x", "y", "z") if not flags.get(ax, False)]

    def _adapt_frame(
        self,
        frame: Frame,
        type_labels: dict[str, dict[int, str]],
    ) -> None:
        """Post-process molrs Frame for molpy surface compatibility.

        - Expose ``type`` (string labels when a Type Labels section exists,
          else numeric ``type_id``).
        - Drop columns the requested ``atom_style`` does not carry (molrs
          auto-detects style from the file and may keep extra fields).
        """
        style = self.atom_style.lower().split("/")[0]
        drop_on_atoms: set[str] = set()
        if style == "atomic":
            drop_on_atoms.update({"mol_id", "charge"})
        elif style == "charge":
            drop_on_atoms.add("mol_id")

        label_keys = {
            "atoms": "atom",
            "bonds": "bond",
            "angles": "angle",
            "dihedrals": "dihedral",
            "impropers": "improper",
        }
        for block_name, label_key in label_keys.items():
            if block_name not in frame:
                continue
            block = frame[block_name]
            labels = type_labels.get(label_key, {})
            if "type_id" in block:
                type_ids = np.asarray(block["type_id"])
                # Frame schema declares ``type`` as string (label-aware). Always
                # materialise a unicode column — numeric ids become digit strings
                # when no Type Labels section is present.
                if labels:
                    mapped = [labels.get(int(t), str(int(t))) for t in type_ids]
                else:
                    mapped = [str(int(t)) for t in type_ids]
                block["type"] = np.asarray(mapped, dtype=str)
            if block_name == "atoms" and drop_on_atoms:
                for col in list(drop_on_atoms):
                    if col in block:
                        try:
                            del block[col]
                        except Exception:
                            try:
                                block.remove(col)  # type: ignore[attr-defined]
                            except Exception:
                                pass

    _formatter = LammpsFieldFormatter()


class LammpsDataWriter(DataWriter):
    """Structure-only LAMMPS data writer (thin molrs façade).

    Structure emission is :func:`molrs.io.write_lammps_data`. molrs resolves
    atom ``id`` (1..N when absent), ``type`` / ``type_id`` numbering, Masses
    (element-preferred), and ``* Type Labels`` from the Frame. This class only:

    1. Optionally seeds meta with constructor ``type_labels`` (unused-type
       inventory) so molrs can merge them with labels present on the Frame.
    2. Prepends a Drude ``fix drude`` comment when shells are detected.

    Force-field ``* Coeffs`` are **not** written here — call
    :func:`write_lammps_data_coeffs` as a separate step.

    **Frame requirements:**
    - Atoms must carry ``type`` and/or ``type_id`` (and connectivity blocks
      that are present must too). Prefer
      :meth:`~molrs.ff.forcefield.ForceField.map_type` first.
    - Connectivity endpoints are 0-based row indices (as from
      ``Atomistic.to_frame()``); molrs maps them to atom IDs.
    """

    #: Constructor key → frame meta key for optional unused-type inventory.
    _TYPE_LABEL_META = (
        ("atom_types", "atom_type_labels"),
        ("bond_types", "bond_type_labels"),
        ("angle_types", "angle_type_labels"),
        ("dihedral_types", "dihedral_type_labels"),
        ("improper_types", "improper_type_labels"),
    )

    def __init__(
        self,
        path: str | Path,
        atom_style: str = "full",
        *,
        type_labels: dict[str, list[str]] | None = None,
    ) -> None:
        """Structure-only writer.

        Args:
            path: Output data file path.
            atom_style: Accepted for API parity (layout from columns).
            type_labels: Optional **extra** unused-type inventory only. Type
                ids always come from the Frame (``type`` then ``type_id``).
                Prefer :meth:`~molrs.ff.forcefield.ForceField.map_type` on the
                Frame before write. Do **not** pass a ForceField here — use
                :func:`write_lammps_data_coeffs` as a separate step.
        """
        super().__init__(Path(path))
        self.atom_style = atom_style
        self.type_labels = {
            key: list(labels) for key, labels in (type_labels or {}).items()
        }

    _formatter = LammpsFieldFormatter()

    def write(self, frame: Frame) -> None:
        """Write Frame structure to a LAMMPS data file (via molrs).

        Frame must already carry ``type`` and/or ``type_id`` on atoms (and on
        any connectivity blocks that are present). Force-field ``* Coeffs`` are
        **not** written here — call :func:`write_lammps_data_coeffs` after.
        """
        import molrs.io

        if "atoms" not in frame or frame["atoms"].nrows == 0:
            raise ValueError("Frame has no atoms to write")

        work = frame
        if self.type_labels:
            work = frame.copy()
            self._seed_type_label_meta(work)

        try:
            molrs.io.write_lammps_data(str(self._path), work)
        except OSError as exc:
            # molrs maps InvalidData to OSError; surface Frame-validation as
            # ValueError so callers keep a Python-native error type.
            msg = str(exc)
            if "neither 'type' nor 'type_id'" in msg or "no atoms to write" in msg:
                raise ValueError(msg) from exc
            raise

        drude_flags = self._drude_flag_string(frame)
        if drude_flags:
            self._prepend_drude_comment(drude_flags)

    def _seed_type_label_meta(self, frame: Frame) -> None:
        """Pack constructor ``type_labels`` into frame meta for molrs merge.

        molrs unions these labels with string ``type`` columns on the Frame
        and assigns dense 1-based ids after numeric-aware sort. Empty labels
        raise — the inventory must be usable as Type Labels text.
        """
        from molpy._frame_meta import update_frame_meta
        from molrs import MetaValue

        meta_update: dict[str, MetaValue] = {}
        for type_key, meta_key in self._TYPE_LABEL_META:
            labels = self.type_labels.get(type_key)
            if not labels:
                continue
            for label in labels:
                if not label or not str(label).strip():
                    raise ValueError(f"Found empty explicit type label for {type_key}")
            ordered = _sorted_type_names(labels)
            packed = ",".join(f"{i}:{lab}" for i, lab in enumerate(ordered, 1))
            meta_update[meta_key] = MetaValue("string", packed)
        if meta_update:
            update_frame_meta(frame, meta_update)

    def _prepend_drude_comment(self, flags: str) -> None:
        header = (
            f"# CL&Pol Drude — paste into input script:\n"
            f"#   fix DRUDE all drude {flags}\n"
        )
        path = Path(self._path)
        path.write_text(header + path.read_text())

    def _ordered_atom_type_names(self, frame: Frame) -> list[str]:
        """Sorted atom type names matching molrs Type Labels order."""
        names: set[str] = set()
        if "atoms" in frame and frame["atoms"].nrows > 0 and "type" in frame["atoms"]:
            names.update(str(t) for t in np.asarray(frame["atoms"]["type"]).flat)
        names.update(str(t) for t in self.type_labels.get("atom_types", []))
        return _sorted_type_names(names)

    def _drude_flag_string(self, frame: Frame) -> str | None:
        """Build the ``fix drude`` C/D/N flag string, or None if not Drude.

        Emits one flag per atom type in the file's (sorted) type-ID order — the
        ordering the LAMMPS DRUDE package's ``fix drude`` consumes: ``D`` for a
        Drude shell type (element ``D``), ``C`` for a polarizable core (an atom
        joined to a shell by a ``drude`` spring bond), ``N`` otherwise.
        """
        if "atoms" not in frame:
            return None
        atoms = frame["atoms"]
        if "element" not in atoms or "type" not in atoms:
            return None
        elements = np.asarray(atoms["element"]).astype(str)
        if not np.any(elements == "D"):
            return None  # no Drude particles → ordinary system

        types = np.asarray(atoms["type"]).astype(str)
        shell_types = set(types[elements == "D"].tolist())

        core_types: set[str] = set()
        bonds = frame["bonds"] if "bonds" in frame else None
        if bonds is not None and bonds.nrows > 0 and "style" in bonds:
            b_style = np.asarray(bonds["style"]).astype(str)
            b_i = np.asarray(bonds["atomi"]).astype(int)
            b_j = np.asarray(bonds["atomj"]).astype(int)
            for k in np.flatnonzero(b_style == "drude"):
                i, j = int(b_i[k]), int(b_j[k])
                core = i if elements[i] != "D" else j
                core_types.add(str(types[core]))

        ordered = self._ordered_atom_type_names(frame)
        if not ordered:
            ordered = _sorted_type_names(set(types.tolist()))
        flags = [
            "D" if t in shell_types else "C" if t in core_types else "N"
            for t in ordered
        ]
        return " ".join(flags)


def _type_ids_from_frame(frame: Frame) -> dict[str, int]:
    """Build ForceField type-name → id map from Frame ``type`` / ``type_id``."""
    out: dict[str, int] = {}
    for block_name in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
        if block_name not in frame:
            continue
        block = frame[block_name]
        if block.nrows == 0:
            continue
        if "type" in block and "type_id" in block:
            types = np.asarray(block["type"]).astype(str)
            tids = np.asarray(block["type_id"]).astype(int)
            for name, tid in zip(types, tids, strict=False):
                out[str(name)] = int(tid)
        elif "type" in block:
            types = [str(t) for t in np.asarray(block["type"])]
            if types and all(_is_int_type_token(t) for t in types):
                for t in types:
                    out[t] = int(t)
            else:
                ordered = _sorted_type_names(set(types))
                for i, name in enumerate(ordered, 1):
                    out[name] = i
        elif "type_id" in block:
            # Numeric-only body: id strings as names for integer FF types.
            for tid in np.unique(np.asarray(block["type_id"]).astype(int)):
                out[str(int(tid))] = int(tid)
    return out


def write_lammps_data_coeffs(
    path: str | Path,
    frame: Frame,
    forcefield: ForceField,
    *,
    units: str = "real",
    precision: int = 6,
) -> None:
    """Insert ``* Coeffs`` into an existing LAMMPS data file.

    Type ids are taken from the Frame (``type`` / ``type_id``), not from a
    caller-supplied inventory. Coefficient numbers (form map + units) are
    produced entirely by molrs.

    Args:
        path: Path to a data file already written by :class:`LammpsDataWriter`.
        frame: Frame whose type columns define the id space (call
            ``forcefield.map_type(frame)`` first when needed).
        forcefield: Force field in molrs store units.
        units: LAMMPS ``units`` style for numeric conversion (``real`` /
            ``metal`` / ``lj``).
        precision: Decimal places for floating coefficients.
    """
    import molrs.ff as mff

    type_ids = _type_ids_from_frame(frame)
    text = mff.write_lammps_data_coeffs(
        forcefield,
        precision=precision,
        units=units,
        type_ids=type_ids or None,
    )
    if not text.strip():
        return
    path = Path(path)
    body = path.read_text()
    insert = text if text.endswith("\n") else text + "\n"
    for marker in ("\nAtoms #", "\nAtoms\n"):
        if marker in body:
            path.write_text(body.replace(marker, "\n" + insert + marker, 1))
            return
    path.write_text(body.rstrip() + "\n" + insert)
