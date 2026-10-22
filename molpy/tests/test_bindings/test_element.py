import molpy as mp
import numpy as np
import pytest

Element = mp.core.Element


@pytest.mark.parametrize(
    ("symbol", "number", "name", "canonical"),
    [
        ("h", 1, "Hydrogen", "H"),
        ("C", 6, "Carbon", "C"),
        ("fe", 26, "Iron", "Fe"),
        ("OG", 118, "Oganesson", "Og"),
    ],
)
def test_element_by_symbol(symbol, number, name, canonical):
    element = Element.by_symbol(symbol)
    assert element.z == number
    assert element.name == name
    assert element.symbol == canonical
    assert repr(element) == f"<Element {canonical}>"


def test_all_real_elements_round_trip_through_rust_table():
    for number in range(1, 119):
        element = Element.by_number(number)
        assert element.z == number
        assert Element.by_symbol(element.symbol) == element
        assert element.mass > 0.0
        assert element.vdw > 0.0
        assert element.covalent > 0.0


def test_element_is_not_constructed_directly():
    with pytest.raises(TypeError):
        Element("C")  # type: ignore[call-arg]


@pytest.mark.parametrize("symbol", ["X", "*", "", "carbon", "not-an-element"])
def test_unknown_symbol_fails_fast(symbol):
    with pytest.raises(ValueError, match="not an element symbol"):
        Element.by_symbol(symbol)


@pytest.mark.parametrize("number", [-1, 0, 119, 999])
def test_unknown_number_fails_fast(number):
    with pytest.raises(ValueError, match="not an atomic number"):
        Element.by_number(number)


def test_by_number_refuses_a_bool():
    with pytest.raises(TypeError, match="not a bool"):
        Element.by_number(True)


def test_fill_atomic_numbers_is_the_explicit_symbol_to_number_step():
    frame = mp.core.Frame({"atoms": {"element": ["O", "h", "*"]}})
    Element.fill_atomic_numbers(frame)
    assert frame["atoms"]["atomic_number"].tolist() == [8, 1, 0]
    assert frame["atoms"]["element"].tolist() == ["O", "h", "*"]


def test_fill_atomic_numbers_names_the_bad_row():
    frame = mp.core.Frame({"atoms": {"element": ["C", "Xx"]}})
    with pytest.raises(ValueError, match="row 1"):
        Element.fill_atomic_numbers(frame)


def test_fill_atomic_numbers_needs_an_element_column():
    frame = mp.core.Frame({"atoms": {"x": [0.0]}})
    with pytest.raises(ValueError, match="element"):
        Element.fill_atomic_numbers(frame)


def test_fill_symbols_is_the_explicit_number_to_symbol_step():
    frame = mp.core.Frame(
        {"atoms": {"atomic_number": np.array([6, 1, 0], dtype=np.uint64)}}
    )
    Element.fill_symbols(frame)
    assert frame["atoms"]["element"].tolist() == ["C", "H", "*"]


def test_fill_symbols_needs_an_atomic_number_column():
    frame = mp.core.Frame({"atoms": {"element": ["C"]}})
    with pytest.raises(ValueError, match="atomic_number"):
        Element.fill_symbols(frame)
