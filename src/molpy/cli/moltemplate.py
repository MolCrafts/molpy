"""``molpy moltemplate`` — execute moltemplate .lt scripts natively.

Sub-sub-commands:
  * ``run``      -- parse .lt and emit a full engine input set.
  * ``parse``    -- parse .lt and dump the IR (debug).
  * ``info``     -- print counts (atom types, molecules, ...).
  * ``convert``  -- FF-only conversion to MolPy XML.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "moltemplate",
        help="Execute moltemplate .lt scripts natively.",
    )
    mtsub = p.add_subparsers(dest="mt_cmd", required=True)

    # run
    r = mtsub.add_parser(
        "run",
        help="Parse .lt and emit complete input set for one or more engines.",
    )
    r.add_argument("script", type=Path, help="Path to the .lt script.")
    r.add_argument(
        "--emit",
        "-e",
        action="append",
        default=None,
        help=(
            "Target engine (repeatable). Choices: lammps, openmm, gromacs, "
            "xml, all. Default: lammps."
        ),
    )
    r.add_argument(
        "--out-dir",
        "-o",
        type=Path,
        default=Path("."),
        help="Output directory (default: cwd).",
    )
    r.add_argument(
        "--prefix",
        default="system",
        help="Filename prefix (default: 'system').",
    )
    r.set_defaults(func=_cmd_run)

    # parse
    pr = mtsub.add_parser("parse", help="Parse a .lt file and dump the IR.")
    pr.add_argument("script", type=Path)
    pr.add_argument(
        "--json", type=Path, default=None, help="Write IR as JSON to this path."
    )
    pr.set_defaults(func=_cmd_parse)

    # info
    info = mtsub.add_parser("info", help="Summarise a .lt file.")
    info.add_argument("script", type=Path)
    info.set_defaults(func=_cmd_info)

    # convert
    c = mtsub.add_parser(
        "convert",
        help=(
            "Convert a moltemplate file. Output format is inferred from the "
            "destination extension: ``.xml`` emits a MolPy canonical force "
            "field; ``.py`` emits a self-contained MolPy Python script that "
            "rebuilds the system using the MolPy API."
        ),
    )
    c.add_argument("src", type=Path, help="Input .lt file.")
    c.add_argument("dst", type=Path, help="Output path (.xml or .py).")
    c.set_defaults(func=_cmd_convert)

    # ltemplify — inverse of ``run`` / ``convert``: write ``.lt`` from a
    # LAMMPS data file (or any MolPy-loadable atomistic) plus its FF.
    l = mtsub.add_parser(
        "ltemplify",
        help=(
            "Serialise a MolPy system back to a moltemplate ``.lt`` template. "
            "Input is either a LAMMPS data file (with --ff pointing at an "
            "in.settings-style coefficients file) or a pre-parsed ``.lt`` "
            "file whose contents we re-emit as a single flat class."
        ),
    )
    l.add_argument("src", type=Path, help="Input system file (.lt or .data).")
    l.add_argument("dst", type=Path, help="Output .lt path.")
    l.add_argument(
        "--ff",
        type=Path,
        default=None,
        help=(
            "Path to a LAMMPS in.settings-style coefficients file. Required "
            "when --src is a LAMMPS data file; ignored otherwise."
        ),
    )
    l.add_argument(
        "--class-name",
        default="System",
        help="Name of the emitted moltemplate class (default: 'System').",
    )
    l.set_defaults(func=_cmd_ltemplify)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    from molpy.io.emit import EMITTERS, emit, emit_all
    from molpy.io.forcefield.moltemplate import read_moltemplate_system

    if not args.script.exists():
        print(f"molpy: error: {args.script} not found", file=sys.stderr)
        return 1

    atomistic, ff = read_moltemplate_system(args.script)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    engines = args.emit or ["lammps"]
    if "all" in engines:
        results = emit_all(atomistic, ff, out_dir, prefix=args.prefix)
        for engine, paths in results.items():
            _print_emit_result(engine, paths)
        return 0

    unknown = [e for e in engines if e not in EMITTERS]
    if unknown:
        print(
            f"molpy: error: unknown engine(s): {unknown}. "
            f"Registered: {sorted(EMITTERS)}",
            file=sys.stderr,
        )
        return 2

    for engine in engines:
        paths = emit(engine, atomistic, ff, out_dir, prefix=args.prefix)
        _print_emit_result(engine, paths)
    return 0


def _cmd_parse(args: argparse.Namespace) -> int:
    from molpy.parser.moltemplate import parse_file

    doc = parse_file(args.script)
    if args.json:
        args.json.write_text(json.dumps(_ir_to_jsonable(doc), indent=2))
        print(f"IR written to {args.json}")
        return 0
    # Default: print a summary
    kinds: dict[str, int] = {}
    for s in doc.statements:
        kinds[type(s).__name__] = kinds.get(type(s).__name__, 0) + 1
    print(f"{args.script}:")
    for k, n in sorted(kinds.items()):
        print(f"  {k}: {n}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    from molpy.core.forcefield import AtomStyle, AtomType
    from molpy.io.forcefield.moltemplate import read_moltemplate_system

    atomistic, ff = read_moltemplate_system(args.script)
    astyle = ff.get_style_by_name("full", AtomStyle)
    n_atomtypes = len(astyle.types.bucket(AtomType)) if astyle else 0
    print(f"{args.script}:")
    print(f"  atom types: {n_atomtypes}")
    print(f"  atoms:      {len(list(atomistic.atoms))}")
    print(f"  bonds:      {len(list(atomistic.bonds))}")
    print(f"  angles:     {len(list(atomistic.angles))}")
    print(f"  dihedrals:  {len(list(atomistic.dihedrals))}")
    print(f"  ff styles:  {len(list(ff.styles.bucket(object)))}")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    if not args.src.exists():
        print(f"molpy: error: {args.src} not found", file=sys.stderr)
        return 1
    suffix = args.dst.suffix.lower()
    if suffix == ".py":
        from molpy.parser.moltemplate import emit_python, parse_file

        doc = parse_file(args.src)
        out = emit_python(doc, args.dst, base_dir=args.src.parent)
        print(f"{args.src} -> {out}")
        return 0
    if suffix in (".xml", ".ffxml"):
        from molpy.io.forcefield.moltemplate import read_moltemplate
        from molpy.io.forcefield.xml import XMLForceFieldWriter

        ff = read_moltemplate(args.src)
        args.dst.parent.mkdir(parents=True, exist_ok=True)
        XMLForceFieldWriter(args.dst).write(ff)
        print(f"{args.src} -> {args.dst}")
        return 0
    print(
        f"molpy: error: unsupported output extension {suffix!r}; use .py or .xml",
        file=sys.stderr,
    )
    return 2


def _cmd_ltemplify(args: argparse.Namespace) -> int:
    if not args.src.exists():
        print(f"molpy: error: {args.src} not found", file=sys.stderr)
        return 1
    src_suffix = args.src.suffix.lower()
    if src_suffix == ".lt":
        from molpy.io.forcefield.moltemplate import read_moltemplate_system

        atomistic, ff = read_moltemplate_system(args.src)
    elif src_suffix in (".data", ".lmp"):
        atomistic, ff = _load_lammps_system(args.src, args.ff)
    else:
        print(
            f"molpy: error: unsupported input extension {src_suffix!r}; "
            "use .lt or .data",
            file=sys.stderr,
        )
        return 2

    from molpy.parser.moltemplate import write_moltemplate

    out = write_moltemplate(atomistic, ff, args.dst, class_name=args.class_name)
    print(f"{args.src} -> {out}")
    return 0


def _load_lammps_system(data_path: Path, settings_path: Path | None):
    """Load an Atomistic + ForceField from a LAMMPS data + settings pair."""
    from molpy.core.forcefield import ForceField
    from molpy.io.data.lammps import LammpsDataReader

    frame = LammpsDataReader(data_path).read()
    atomistic = (
        frame.to_atomistic()
        if hasattr(frame, "to_atomistic")
        # Fallback: rebuild Atomistic from the Frame manually.
        else (_atomistic_from_frame(frame))
    )
    ff = ForceField(name=data_path.stem, units="real")
    ff.metadata = {}  # type: ignore[attr-defined]
    if settings_path is not None and settings_path.exists():
        from molpy.io.forcefield.lammps import LAMMPSForceFieldReader

        LAMMPSForceFieldReader(settings_path).read(ff)
    return atomistic, ff


def _atomistic_from_frame(frame):
    """Construct an ``Atomistic`` from a LAMMPS ``Frame`` (fallback helper)."""
    from molpy.core.atomistic import Atomistic

    system = Atomistic()
    atoms = frame["atoms"]
    keys = list(atoms.keys()) if hasattr(atoms, "keys") else []
    n = atoms.nrows if hasattr(atoms, "nrows") else len(atoms[keys[0]])
    atom_objs = []
    for i in range(n):
        kwargs = {k: atoms[k][i] for k in keys}
        xyz = [
            float(kwargs.pop("x", 0.0)),
            float(kwargs.pop("y", 0.0)),
            float(kwargs.pop("z", 0.0)),
        ]
        atom_objs.append(system.def_atom(xyz=xyz, **kwargs))
    for name, adder in (
        ("bonds", lambda a, b, **kw: system.def_bond(a, b, **kw)),
        ("angles", None),
        ("dihedrals", None),
        ("impropers", None),
    ):
        if name not in frame:
            continue
        block = frame[name]
        nrows = block.nrows
        block_keys = list(block.keys()) if hasattr(block, "keys") else []
        for i in range(nrows):
            row = {k: block[k][i] for k in block_keys}
            ids = [int(row.pop(f"atom{c}")) for c in "ijkl" if f"atom{c}" in row]
            endpoints = [atom_objs[idx] for idx in ids if 0 <= idx < len(atom_objs)]
            if name == "bonds" and len(endpoints) == 2:
                system.def_bond(*endpoints, **row)
            elif name == "angles" and len(endpoints) == 3:
                system.def_angle(*endpoints, **row)
            elif name == "dihedrals" and len(endpoints) == 4:
                system.def_dihedral(*endpoints, **row)
            elif name == "impropers" and len(endpoints) == 4:
                system.def_improper(*endpoints, **row)
    return system


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_emit_result(engine: str, paths: list[Path]) -> None:
    print(f"[{engine}]")
    for p in paths:
        print(f"  {p}")


def _ir_to_jsonable(obj):
    if is_dataclass(obj):
        d = {"_kind": type(obj).__name__}
        d.update(asdict(obj))
        # Recurse into nested dataclasses in lists
        for k, v in list(d.items()):
            d[k] = _ir_to_jsonable(v)
        return d
    if isinstance(obj, list):
        return [_ir_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _ir_to_jsonable(v) for k, v in obj.items()}
    return obj
