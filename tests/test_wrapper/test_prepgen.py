"""write_prepgen_control_file: the three residue variants and their guards."""

import pytest

from molpy.wrapper import write_prepgen_control_file


def test_chain_variant_lists_both_ends_types_and_omissions(tmp_path):
    path = tmp_path / "mol.chain"
    write_prepgen_control_file(
        path,
        variant="chain",
        head_name="C1",
        tail_name="O5",
        head_type="c3",
        tail_type="os",
        omit_names=["H1", "H2"],
        charge=-1,
    )
    assert path.read_text().splitlines() == [
        "HEAD_NAME C1",
        "TAIL_NAME O5",
        "PRE_HEAD_TYPE c3",
        "POST_TAIL_TYPE os",
        "OMIT_NAME H1",
        "OMIT_NAME H2",
        "CHARGE -1",
    ]


def test_head_variant_needs_only_a_tail(tmp_path):
    path = tmp_path / "mol.head"
    write_prepgen_control_file(path, variant="head", tail_name="O5", tail_type="os")
    assert path.read_text().splitlines() == [
        "TAIL_NAME O5",
        "POST_TAIL_TYPE os",
        "CHARGE 0",
    ]


def test_tail_variant_needs_only_a_head(tmp_path):
    path = tmp_path / "mol.tail"
    write_prepgen_control_file(path, variant="tail", head_name="C1")
    assert path.read_text().splitlines() == ["HEAD_NAME C1", "CHARGE 0"]


@pytest.mark.parametrize(
    "variant, kwargs",
    [("chain", {"head_name": "C1"}), ("head", {}), ("tail", {"tail_name": "O5"})],
)
def test_missing_connection_atom_raises(tmp_path, variant, kwargs):
    with pytest.raises(ValueError, match="requires"):
        write_prepgen_control_file(tmp_path / "x", variant=variant, **kwargs)
