"""The LAMMPS data writer emits the ``fix drude`` C/D/N flag string.

For a Drude-polarizable frame (shells carry element ``D`` + a type, springs are
``style="drude"`` bonds), molrs's data writer — which owns the atom-type → ID
ordering — writes the ``fix drude`` flags as a header comment.
"""

from pathlib import Path

import molpy as mp
from molpy.core import Atomistic
from molpy.builder import DrudeBuilder


def _ntf2_polarized(test_data_dir: Path) -> Atomistic:
    """CL&P-typed [NTf2]- (atoms named by element, spread along x), Drude-polarized."""
    frame = mp.io.read_mol2(test_data_dir / "mol2" / "ntf2_clp_typed.mol2")
    frame["atoms"]["element"] = frame["atoms"]["name"]
    pol = DrudeBuilder().apply(Atomistic.from_frame(frame))
    for i, atom in enumerate(pol.atoms, start=1):
        atom["id"] = i
        atom["mol_id"] = 1
    return pol


def test_data_writer_emits_fix_drude_flags(tmp_path, TEST_DATA_DIR):
    frame = _ntf2_polarized(TEST_DATA_DIR).to_frame()
    path = tmp_path / "ntf2.data"
    mp.io.write_lammps_data(path, frame)
    text = path.read_text(encoding="utf-8")

    flag_line = next(line for line in text.splitlines() if "fix drude flags" in line)
    flags = flag_line.split(":", 1)[1].split()

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
    asm.def_atom(
        id=1,
        mol_id=1,
        element="C",
        type="CT",
        charge=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        mass=12.0,
    )
    asm.def_atom(
        id=2,
        mol_id=1,
        element="H",
        type="HC",
        charge=0.0,
        x=1.0,
        y=0.0,
        z=0.0,
        mass=1.0,
    )
    path = tmp_path / "plain.data"
    mp.io.write_lammps_data(path, asm.to_frame())
    assert "fix drude" not in path.read_text(encoding="utf-8")
