import importlib.util

import pytest

import molrs
import molpy
import molpy.core
from molpy import Element


def test_element_has_one_public_owner():
    """Element is a molrs type, identity-re-exported on the molpy facade.

    Users import from molpy (never molrs). There is no molpy.core.element
    module, no ElementData, and no second class.
    """
    assert Element is molrs.Element
    assert molpy.Element is molrs.Element
    assert not hasattr(molrs, "ElementData")
    assert not hasattr(molpy, "ElementData")
    assert not hasattr(molpy.core, "ElementData")
    assert not hasattr(molpy.core, "Element")
    assert importlib.util.find_spec("molpy.core.element") is None


@pytest.mark.parametrize(
    ("identifier", "number", "name", "symbol"),
    [
        ("hydrogen", 1, "Hydrogen", "H"),
        ("c", 6, "Carbon", "C"),
        ("OXYGEN", 8, "Oxygen", "O"),
        (118, 118, "Oganesson", "Og"),
    ],
)
def test_native_element_lookup(identifier, number, name, symbol):
    element = Element(identifier)
    assert element.number == number
    assert element.name == name
    assert element.symbol == symbol
    assert repr(element) == f"<Element {symbol}>"


def test_all_elements_round_trip_and_expose_rust_properties():
    for number in range(1, 119):
        element = Element(number)
        assert Element(element.symbol) == element
        assert Element(element.name.upper()) == element
        assert element.mass > 0
        assert element.vdw > 0
        assert element.covalent > 0


def test_element_convenience_methods():
    assert Element.get_symbols([1, "carbon", "o", 7]) == ["H", "C", "O", "N"]
    assert Element.get_atomic_number("fe") == 26


@pytest.mark.parametrize("identifier", [0, -1, 119, 999, "X", "nonexistent"])
def test_invalid_element_fails_fast(identifier):
    with pytest.raises(KeyError, match="Element not found"):
        Element(identifier)
