"""Copy behavior for live graph refs."""

from molpy.core.atomistic import Atomistic


def _molecule() -> tuple[Atomistic, object, object]:
    struct = Atomistic()
    carbon = struct.def_atom(element="C", port="port_1")
    hydrogen = struct.def_atom(element="H")
    oxygen = struct.def_atom(element="O")
    struct.def_bond(carbon, hydrogen)
    struct.def_bond(carbon, oxygen)
    return struct, carbon, hydrogen


def test_copy_preserves_nodes_relations_and_fields() -> None:
    struct, carbon, _ = _molecule()
    clone = struct.copy()

    assert len(clone.atoms) == 3
    assert len(clone.bonds) == 2
    assert sorted(atom["element"] for atom in clone.atoms) == ["C", "H", "O"]
    copied_carbon = next(atom for atom in clone.atoms if atom.get("port") == "port_1")
    assert copied_carbon is not carbon


def test_copy_relations_reference_copied_nodes() -> None:
    struct, carbon, hydrogen = _molecule()
    clone = struct.copy()
    copied = set(clone.atoms)

    assert all(
        endpoint in copied for bond in clone.bonds for endpoint in bond.endpoints
    )
    assert all(
        endpoint is not carbon and endpoint is not hydrogen
        for bond in clone.bonds
        for endpoint in bond.endpoints
    )


def test_multiple_copies_are_independent() -> None:
    struct, _, _ = _molecule()
    first = struct.copy()
    second = struct.copy()
    first.def_atom(element="N")
    second.def_atom(element="F")

    assert len(struct.atoms) == 3
    assert len(first.atoms) == 4
    assert len(second.atoms) == 4
