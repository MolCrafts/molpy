"""LAMMPS data files: the molpy read (``LammpsDataResult``) and write doors."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import molrs.ff
import molrs.io
from molrs import Frame
from molrs.ff import ForceField


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


def _type_labels_from_meta(frame: Frame) -> dict[str, dict[int, str]]:
    """Parse the native ``*_type_labels`` meta (``id:label,...``) into maps."""
    out: dict[str, dict[int, str]] = {}
    for kind, meta_key in (
        ("atom", "atom_type_labels"),
        ("bond", "bond_type_labels"),
        ("angle", "angle_type_labels"),
        ("dihedral", "dihedral_type_labels"),
        ("improper", "improper_type_labels"),
    ):
        packed = frame.meta.get(meta_key)
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


@dataclass(frozen=True, slots=True)
class LammpsDataResult:
    """Explicit products of parsing one LAMMPS data file.

    Frame-like lookup (``result["atoms"]``, ``"atoms" in result``,
    ``result.box``) delegates to :attr:`frame` so callers can treat the
    result as the structure without dropping ``.frame`` / ``.forcefield``.

    The structure never depends on the ``* Coeffs`` sections: they are kept
    as text in ``frame.meta["lammps_coeffs_text"]`` and become a
    :class:`~molpy.ForceField` only when :attr:`forcefield` is first read.
    """

    frame: Frame
    counts: dict[str, int]
    type_labels: dict[str, list[str]]
    _forcefield: ForceField | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @property
    def forcefield(self) -> ForceField:
        """Force field parsed from the file's ``* Coeffs`` sections.

        Parsed on first access from ``frame.meta["lammps_coeffs_text"]`` and
        the file's Type Labels, then kept; later edits to ``frame.meta`` do
        not re-parse. Coefficients are read in the LAMMPS ``units`` style
        named by ``frame.meta["lammps_units"]`` (the ``units = <style>``
        suffix of a ``write_data`` header); without that key the documented
        default is ``"real"``. A file without ``* Coeffs`` yields an empty
        ForceField.

        Raises:
            ValueError: The ``* Coeffs`` text cannot become a ForceField
                (``"malformed PairCoeffs: ..."`` for unparseable numbers,
                ``"Failed to parse LAMMPS force-field coeffs: ..."``
                otherwise). Nothing is cached, so the next access re-raises.
        """
        cached = object.__getattribute__(self, "_forcefield")
        if cached is not None:
            return cached

        frame = object.__getattribute__(self, "frame")
        coeffs_text = frame.meta.get("lammps_coeffs_text")
        if not coeffs_text:
            forcefield = ForceField("LAMMPS")
        else:
            labels = _type_labels_from_meta(frame)
            try:
                forcefield = molrs.ff.read_lammps_data_coeffs(
                    str(coeffs_text),
                    units=str(frame.meta.get("lammps_units", "real")),
                    atom_labels=labels.get("atom"),
                    bond_labels=labels.get("bond"),
                    angle_labels=labels.get("angle"),
                    dihedral_labels=labels.get("dihedral"),
                    improper_labels=labels.get("improper"),
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
        object.__setattr__(self, "_forcefield", forcefield)
        return forcefield

    def __getitem__(self, key: str):
        """Return ``self.frame[key]``."""
        return object.__getattribute__(self, "frame")[key]

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` is a block on :attr:`frame`."""
        return key in object.__getattribute__(self, "frame")

    def __getattr__(self, name: str):
        """Delegate unknown attributes (e.g. ``box``) to :attr:`frame`."""
        return getattr(object.__getattribute__(self, "frame"), name)


def read_lammps_data(file: str | Path, atom_style: str = "full") -> LammpsDataResult:
    """Read a LAMMPS data file into its explicit parse products.

    Structure, Type Labels, header counts and ``* Coeffs`` text come from the
    native ``read_lammps_data`` in one pass. The coeffs text is not parsed
    here: :attr:`LammpsDataResult.forcefield` parses it on first access, so a
    ``* Coeffs`` section the native coeffs reader cannot handle never fails
    the structure read.

    On top of the native frame, every topology block gets a string ``type``
    column (the Type Labels, else the numeric ids as digit strings), and the
    atom columns ``atom_style`` does not carry are dropped (``atomic`` drops
    ``mol_id`` and ``charge``, ``charge`` drops ``mol_id``).

    Args:
        file: Path to the data file.
        atom_style: LAMMPS atom style; the native reader detects the columns
            itself, this names the ones to keep.

    Returns:
        ``LammpsDataResult`` with ``frame``, ``forcefield``, ``counts`` and
        ``type_labels``.

    Raises:
        ValueError: If the header lacks the bounds of a box axis.
    """
    path = Path(file)
    frame = molrs.io.read_lammps_data(path)

    missing_axes = _missing_box_axes(frame)
    if missing_axes:
        raise ValueError(f"missing box bounds for axis {missing_axes}")

    type_labels = _type_labels_from_meta(frame)
    _adapt_frame(frame, type_labels, atom_style)

    counts = _counts_from_meta(frame)
    for block in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
        if block in frame:
            counts.setdefault(block, int(frame[block].nrows))

    frame.meta.update(
        {"format": "lammps_data", "atom_style": atom_style, "source_file": str(path)}
    )
    return LammpsDataResult(
        frame=frame,
        counts=counts,
        type_labels={
            f"{key}_types": [labels[i] for i in sorted(labels)]
            for key, labels in type_labels.items()
        },
    )


def _counts_from_meta(frame: Frame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in str(frame.meta.get("lammps_counts") or "").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        try:
            counts[k.strip()] = int(v)
        except ValueError:
            continue
    return counts


def _missing_box_axes(frame: Frame) -> list[str]:
    """Axes absent from the data header (not merely zero-length)."""
    flags: dict[str, bool] = {}
    for part in str(frame.meta.get("lammps_box_axes") or "").split(","):
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        flags[k.strip()] = v.strip() in ("1", "true", "True")
    return [ax for ax in ("x", "y", "z") if not flags.get(ax, False)]


def _adapt_frame(
    frame: Frame, type_labels: dict[str, dict[int, str]], atom_style: str
) -> None:
    """Add the string ``type`` columns and drop what ``atom_style`` lacks."""
    style = atom_style.lower().split("/")[0]
    drop_on_atoms: set[str] = set()
    if style == "atomic":
        drop_on_atoms.update({"mol_id", "charge"})
    elif style == "charge":
        drop_on_atoms.add("mol_id")

    for block_name, label_key in (
        ("atoms", "atom"),
        ("bonds", "bond"),
        ("angles", "angle"),
        ("dihedrals", "dihedral"),
        ("impropers", "improper"),
    ):
        if block_name not in frame:
            continue
        block = frame[block_name]
        labels = type_labels.get(label_key, {})
        if "type_id" in block:
            block["type"] = np.asarray(
                [labels.get(int(t), str(int(t))) for t in np.asarray(block["type_id"])],
                dtype=str,
            )
        if block_name == "atoms":
            for col in drop_on_atoms:
                if col in block:
                    del block[col]


#: ``type_labels`` key → frame meta key of the native Type Labels merge.
_TYPE_LABEL_META = (
    ("atom_types", "atom_type_labels"),
    ("bond_types", "bond_type_labels"),
    ("angle_types", "angle_type_labels"),
    ("dihedral_types", "dihedral_type_labels"),
    ("improper_types", "improper_type_labels"),
)


def write_lammps_data(
    file: str | Path,
    frame: Frame,
    *,
    type_labels: dict[str, list[str]] | None = None,
) -> None:
    """Write a frame's structure to a LAMMPS data file.

    Atoms, topology, Masses and ``* Type Labels`` are the native
    ``write_lammps_data``'s; the atom style follows from the columns present.
    Force-field ``* Coeffs`` are not written:
    ``mp.io.write_lammps_data_coeffs(ff, frame)`` returns that section text for
    the same frame's labels. On top of the native writer this adds the
    ``type_labels`` inventory and, for a Drude system (atoms with element
    ``D``), a header comment carrying the ``fix drude`` C/D/N flags in type-id
    order.

    Args:
        file: Output path.
        frame: Frame whose atoms (and connectivity blocks) carry ``type``
            and/or ``type_id``; connectivity endpoints are 0-based rows.
        type_labels: Optional **extra** type labels per category
            (``{"atom_types": [...], ...}``), declared even when no row uses
            them; ids are dense, 1-based, after a numeric-aware sort.

    Raises:
        ValueError: If the frame has no atoms, a type label is empty, or rows
            carry neither ``type`` nor ``type_id``.
    """
    if "atoms" not in frame or frame["atoms"].nrows == 0:
        raise ValueError("Frame has no atoms to write")

    extra = {key: list(labels) for key, labels in (type_labels or {}).items()}
    work = frame
    if extra:
        work = frame.copy()
        for type_key, meta_key in _TYPE_LABEL_META:
            labels = extra.get(type_key)
            if not labels:
                continue
            if any(not str(label).strip() for label in labels):
                raise ValueError(f"Found empty explicit type label for {type_key}")
            ordered = _sorted_type_names(labels)
            work.meta[meta_key] = ",".join(
                f"{i}:{lab}" for i, lab in enumerate(ordered, 1)
            )

    try:
        molrs.io.write_lammps_data(file, work)
    except OSError as exc:
        # The native core reports frame validation as an I/O error.
        msg = str(exc)
        if "neither 'type' nor 'type_id'" in msg:
            raise ValueError(msg) from exc
        raise

    flags = _drude_flags(frame, extra.get("atom_types", []))
    if flags:
        path = Path(file)
        path.write_text(
            "# CL&Pol Drude — paste into input script:\n"
            f"#   fix DRUDE all drude {flags}\n" + path.read_text()
        )


def _drude_flags(frame: Frame, extra_atom_types: list[str]) -> str | None:
    """The ``fix drude`` C/D/N flag string, or None if the frame has no shell.

    One flag per atom type in the file's (sorted) type-id order — the order the
    LAMMPS DRUDE package's ``fix drude`` consumes: ``D`` for a Drude shell type
    (element ``D``), ``C`` for a polarizable core (an atom joined to a shell by
    a ``drude`` spring bond), ``N`` otherwise.
    """
    atoms = frame["atoms"]
    if "element" not in atoms or "type" not in atoms:
        return None
    elements = np.asarray(atoms["element"]).astype(str)
    if not np.any(elements == "D"):
        return None

    types = np.asarray(atoms["type"]).astype(str)
    shell_types = set(types[elements == "D"].tolist())

    core_types: set[str] = set()
    if "bonds" in frame and frame["bonds"].nrows > 0 and "style" in frame["bonds"]:
        bonds = frame["bonds"]
        b_style = np.asarray(bonds["style"]).astype(str)
        b_i = np.asarray(bonds["atomi"]).astype(int)
        b_j = np.asarray(bonds["atomj"]).astype(int)
        for k in np.flatnonzero(b_style == "drude"):
            i, j = int(b_i[k]), int(b_j[k])
            core_types.add(str(types[i if elements[i] != "D" else j]))

    ordered = _sorted_type_names(set(types.tolist()) | set(extra_atom_types))
    return " ".join(
        "D" if t in shell_types else "C" if t in core_types else "N" for t in ordered
    )
