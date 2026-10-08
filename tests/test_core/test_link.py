import pytest

from molpy.core import Atomistic


def test_relation_endpoints_are_live_and_immutable() -> None:
    graph = Atomistic()
    a = graph.def_atom(id=1)
    b = graph.def_atom(id=2)
    link = graph.def_bond(a, b, bond_type=1, bond_number=1)

    assert link.endpoints == (a, b)
    assert link["bond_type"] == 1
    assert link["bond_number"] == 1


def test_relation_has_no_detached_form() -> None:
    from molpy.core import Bond

    with pytest.raises(TypeError):
        Bond(object(), object())  # type: ignore[call-arg]
