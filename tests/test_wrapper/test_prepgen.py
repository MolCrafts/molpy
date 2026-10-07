"""prepgen_control_text: the three residue variants and their guards."""

import pytest

from molpy.wrapper._prepgen import prepgen_control_text


def test_chain_variant_lists_both_ends_types_and_omissions():
    text = prepgen_control_text(
        variant="chain",
        head_name="C1",
        tail_name="O5",
        head_type="c3",
        tail_type="os",
        omit_names=["H1", "H2"],
        charge=-1,
    )
    assert text.splitlines() == [
        "HEAD_NAME C1",
        "TAIL_NAME O5",
        "PRE_HEAD_TYPE c3",
        "POST_TAIL_TYPE os",
        "OMIT_NAME H1",
        "OMIT_NAME H2",
        "CHARGE -1",
    ]
    assert text.endswith("\n")


def test_head_variant_needs_only_a_tail():
    text = prepgen_control_text(variant="head", tail_name="O5", tail_type="os")
    assert text.splitlines() == [
        "TAIL_NAME O5",
        "POST_TAIL_TYPE os",
        "CHARGE 0",
    ]


def test_tail_variant_needs_only_a_head():
    text = prepgen_control_text(variant="tail", head_name="C1")
    assert text.splitlines() == ["HEAD_NAME C1", "CHARGE 0"]


@pytest.mark.parametrize(
    "variant, kwargs",
    [("chain", {"head_name": "C1"}), ("head", {}), ("tail", {"tail_name": "O5"})],
)
def test_missing_connection_atom_raises(variant, kwargs):
    with pytest.raises(ValueError, match="requires"):
        prepgen_control_text(variant=variant, **kwargs)
