"""Every reader must hand back canonical column names.

`FieldFormatter.canonicalize` is the documented boundary: a reader speaks its
format's vocabulary internally and molpy's on the way out. Nothing enforced it,
so a reader could simply not have a formatter — and one did not. The cost is
silent: downstream code reads the canonical key, finds nothing, and takes its
"field absent" branch. No exception is raised and no test fails, because the
column *is* there, under another name.

So the gate executes readers rather than reading their source. A grep finds
spellings; only running the reader finds the column it actually wrote.
"""

from __future__ import annotations

import molrs
import pytest

import molpy as mp

#: The canonical vocabulary, straight from molrs — not a copy maintained here.
CANONICAL = frozenset(column.key for column in molrs.schema.columns)


def _read_prmtop(data_dir):
    frame, _ff = mp.io.read_amber(
        data_dir / "prmtop" / "LiTFSI.prmtop", data_dir / "inpcrd" / "LiTFSI.inpcrd"
    )
    return frame


def _read_pdb(data_dir):
    return mp.io.read_pdb(data_dir / "pdb" / "1bcu.pdb")


def _read_gro(data_dir):
    return mp.io.read_gro(data_dir / "gro" / "cod_4020641.gro")


def _read_mol2(data_dir):
    return mp.io.read_mol2(data_dir / "mol2" / "ethane.mol2")


def _read_xyz(data_dir):
    return mp.io.read_xyz(data_dir / "xyz" / "extended.xyz")


#: ``(id, reader, allowed non-canonical columns)``. An entry in the third field
#: is a debt, not a licence: it says molrs' vocabulary does not yet name
#: something a real file of this format carries. Keep it empty where possible.
READERS = [
    ("amber-prmtop", _read_prmtop, frozenset()),
    # A PDB chain is a real concept that molrs' column vocabulary does not name
    # yet. Until it does, the honest thing is to say so here.
    ("pdb", _read_pdb, frozenset({"chain_id"})),
    ("gro", _read_gro, frozenset()),
    # `CS` is a user-defined per-atom property declared by this extended-XYZ
    # fixture's Properties= line. Extended XYZ carries arbitrary named columns by
    # design — that is data the file chose, not a spelling the reader invented —
    # so it is named here rather than waved through by a blanket exemption.
    ("xyz", _read_xyz, frozenset({"CS"})),
    # SYBYL status bits are file bookkeeping, not chemistry; nothing in molpy
    # reads them and molrs will never name them.
    ("mol2", _read_mol2, frozenset({"status_bit", "status_bits"})),
]


@pytest.mark.parametrize(
    "reader, allowed", [pytest.param(r, a, id=i) for i, r, a in READERS]
)
def test_reader_emits_canonical_column_names(TEST_DATA_DIR, reader, allowed):
    frame = reader(TEST_DATA_DIR)
    offenders = sorted(
        f"{block_name}.{key}"
        for block_name in frame.keys()
        for key in frame[block_name].keys()
        if key not in CANONICAL and key not in allowed
    )
    assert not offenders, (
        f"non-canonical column(s) {offenders}; translate them in this format's "
        "FieldFormatter (canonicalize at the reader's exit) rather than letting "
        "the format's own spelling reach molpy"
    )


def test_the_gate_covers_a_format_that_once_failed_it():
    """A gate nobody can fail is not a gate.

    ``read_amber`` is the reader this test was written for — it wrote ``residue``
    where molpy names ``res_id`` — so it must stay in the parametrisation.
    """
    assert "amber-prmtop" in {entry[0] for entry in READERS}


def test_canonical_vocabulary_is_read_from_molrs_not_restated():
    assert "res_id" in CANONICAL
    assert "residue" not in CANONICAL
