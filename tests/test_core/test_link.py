import pytest

from molpy import Atomistic


def test_relation_endpoints_are_live_and_immutable() -> None:
    graph = Atomistic()
    a = graph.def_atom(id=1)
    b = graph.def_atom(id=2)
    link = graph.def_bond(a, b, order=1)

    assert link.endpoints == (a, b)
    assert link["order"] == 1
    assert not hasattr(link, "replace_endpoint")


def test_relation_has_no_detached_form() -> None:
    from molpy import Bond

    with pytest.raises(TypeError):
        Bond(object(), object())  # type: ignore[call-arg]
