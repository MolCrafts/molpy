import pytest
import molpy as mp


def test_relation_endpoints_are_live_and_immutable() -> None:
    graph = mp.Atomistic()
    a = graph.add_atom(id=1, atomic_number=6)
    b = graph.add_atom(id=2, atomic_number=6)
    link = graph.add_bond(a, b)
    graph.set_bond_class(link, mp.core.BondOrder.Single, mp.core.BondNumber.Single)

    assert link.endpoints == (a, b)
    assert link["bond_type"] == 1
    assert link["bond_number"] == 1


def test_relation_has_no_detached_form() -> None:

    with pytest.raises(TypeError):
        mp.Bond(object(), object())  # type: ignore[call-arg]
