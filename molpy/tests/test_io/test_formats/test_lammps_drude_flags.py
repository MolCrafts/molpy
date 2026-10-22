"""The LAMMPS data writer emits the ``fix drude`` C/D/N flag string.

For a Drude-polarizable frame (shells are ``vsite == "drude"`` atoms with a type,
each tied to its core by a molrs ``drudes`` row), molrs's data writer — which
owns the atom-type → ID ordering — writes the ``fix drude`` flags as a header
comment.
"""

from pathlib import Path


import molpy as mp


def _ntf2_polarized(test_data_dir: Path) -> mp.Atomistic:
    """CL&P-typed [NTf2]- (atoms named by element, spread along x), Drude-polarized."""
    frame = mp.io.read_mol2(test_data_dir / "mol2" / "ntf2_clp_typed.mol2")
    frame["atoms"]["element"] = frame["atoms"]["name"]
    mp.Element.fill_atomic_numbers(frame)
    pol = mp.builder.DrudeBuilder().apply(mp.Atomistic.from_frame(frame))
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


def test_data_writer_writes_drude_springs_as_bonds(tmp_path, TEST_DATA_DIR):
    pol = _ntf2_polarized(TEST_DATA_DIR)
    frame = pol.to_frame()
    n_springs = len(frame["drudes"]["atomi"])
    path = tmp_path / "ntf2.data"
    mp.io.write_lammps_data(path, frame)
    header = path.read_text(encoding="utf-8").splitlines()
    n_bonds = int(next(line for line in header if line.endswith(" bonds")).split()[0])
    assert n_bonds == len(list(pol.bonds)) + n_springs


def test_data_writer_no_drude_comment_for_plain_system(tmp_path):
    """A non-polarizable frame (no element ``D``) gets no fix-drude comment."""
    asm = mp.Atomistic()
    asm.add_atom(
        id=1,
        mol_id=1,
        atomic_number=6,
        type="CT",
        charge=0.0,
        x=0.0,
        y=0.0,
        z=0.0,
        mass=12.0,
    )
    asm.add_atom(
        id=2,
        mol_id=1,
        atomic_number=1,
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
