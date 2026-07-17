"""The LAMMPS data writer emits the ``fix drude`` C/D/N flag string.

For a Drude-polarizable frame (shells carry element ``D`` + a type, springs are
``style="drude"`` bonds), the data writer — which owns the atom-type → ID
ordering — writes the ready-to-paste ``fix drude`` flags as a header comment.
"""

import warnings

import molpy as mp
import pytest
from molpy import Atom, Atomistic, Bond
from molpy.builder.virtualsite import DrudeBuilder
from molpy.io.data.lammps import LammpsDataWriter
from molpy.typifier import ClpTypifier


def _ntf2_polarized():
    el = ["C", "F", "F", "F", "S", "N", "O", "O", "S", "O", "O", "C", "F", "F", "F"]
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (4, 5),
        (4, 6),
        (4, 7),
        (5, 8),
        (8, 9),
        (8, 10),
        (8, 11),
        (11, 12),
        (11, 13),
        (11, 14),
    ]
    asm = Atomistic()
    # Spread atoms along x so cores (and the shells co-located on them) carry
    # coordinates the LAMMPS data writer can emit.
    atoms = [asm.def_atom(element=e, x=1.5 * i, y=0.0, z=0.0) for i, e in enumerate(el)]
    for i, j in edges:
        asm.def_bond(atoms[i], atoms[j])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        typed = ClpTypifier().typify(asm.get_topo(gen_angle=True, gen_dihe=True))
        pol = DrudeBuilder().apply(typed)
    for i, atom in enumerate(pol.atoms, start=1):
        atom["id"] = i
        atom["mol_id"] = 1
    return pol


def test_data_writer_emits_fix_drude_flags(tmp_path):
    frame = _ntf2_polarized().to_frame()
    path = tmp_path / "ntf2.data"
    LammpsDataWriter(path).write(frame)
    text = path.read_text()

    assert "fix DRUDE all drude" in text
    flag_line = next(line for line in text.splitlines() if "all drude" in line)
    flags = flag_line.split("all drude", 1)[1].split()

    # One flag per atom type, in sorted (type-ID) order.
    import numpy as np

    atom_types = sorted(set(np.asarray(frame["atoms"]["type"]).astype(str).tolist()))
    assert len(flags) == len(atom_types)
    mapping = dict(zip(atom_types, flags))
    assert mapping["NBT"] == "C"  # polarizable core
    assert mapping["DNBT"] == "D"  # its Drude shell
    assert set(flags) <= {"C", "D", "N"}


def test_data_writer_no_drude_comment_for_plain_system(tmp_path):
    """A non-polarizable frame (no element ``D``) gets no fix-drude comment."""
    asm = Atomistic()
    asm.def_atoms(
        [
            dict(
                id=1,
                mol_id=1,
                element="C",
                type="CT",
                charge=0.0,
                x=0.0,
                y=0.0,
                z=0.0,
                mass=12.0,
            ),
            dict(
                id=2,
                mol_id=1,
                element="H",
                type="HC",
                charge=0.0,
                x=1.0,
                y=0.0,
                z=0.0,
                mass=1.0,
            ),
        ]
    )
    path = tmp_path / "plain.data"
    LammpsDataWriter(path).write(asm.to_frame())
    assert "fix DRUDE" not in path.read_text()
