import pytest
import numpy as np
import pyarrow as pa
from molpy.io.pyarrow import to_arrow
from molpy.core.frame import Frame


def make_simple_frame():
    atoms = {
        "xyz": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        "name": ["C", "O", "N"],
        "element": ["C", "O", "N"],
        "charge": [0, -1, 1],
    }
    bonds = {
        "i": [0, 1],
        "j": [1, 2],
        "order": [1, 2],
    }
    return Frame({"atoms": atoms, "bonds": bonds})


def test_to_arrow_atoms_and_bonds():
    frame = make_simple_frame()
    result = to_arrow(frame, atom_fields=["name", "element", "charge"], bond_fields=["i", "j", "order"])
    assert "atoms" in result and result["atoms"] is not None
    assert "bonds" in result and result["bonds"] is not None
    # Atoms
    atoms_buf = result["atoms"]
    atoms_table = pa.ipc.open_stream(atoms_buf).read_all()
    # Should have 9 rows (3 atoms * 3 spatial)
    assert atoms_table.num_rows == 9
    assert set(["xyz", "name", "element", "charge", "spatial", "index"]).issubset(atoms_table.column_names)
    # Each atom should have 3 rows (x/y/z)
    for idx in range(3):
        rows = atoms_table.filter(pa.array([i//3==idx for i in range(9)]))
        assert set(rows["name"].to_pylist()) == {frame["atoms"].to_dataframe().reset_index()["name"][idx*3]}
    # Bonds
    bonds_buf = result["bonds"]
    bonds_table = pa.ipc.open_stream(bonds_buf).read_all()
    assert bonds_table.num_rows == 2
    assert set(["i", "j", "order", "index"]).issubset(bonds_table.column_names)


def test_to_arrow_atoms_only():
    frame = make_simple_frame()
    del frame["bonds"]
    result = to_arrow(frame)
    assert result["atoms"] is not None
    assert result["bonds"] is None
    atoms_table = pa.ipc.open_stream(result["atoms"]).read_all()
    assert atoms_table.num_rows == 9
    assert "xyz" in atoms_table.column_names
    assert "name" in atoms_table.column_names
    assert "spatial" in atoms_table.column_names
    assert "index" in atoms_table.column_names
