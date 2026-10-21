"""``molrs.builder.Assembler`` / ``SitePlacer`` / ``AxisOrienter`` FFI seam:
library values of both graph types, a ``CoarseGrain`` site graph in, the public
world out as the library's graph type, the id columns and error
mapping. Port assignment, placement
and the link rule are proven by the Rust suite.

Hand counts: two bonded ``M`` sites are two copies of the 4-atom
``H0-C0-C1-H1`` monomer (ports ``>`` on C0/H0, ``<`` on C1/H1), 8 atoms and 4
ports; the one link removes one hydrogen per joined port, leaving 6 atoms and
2 ports; the unbonded Li site adds 1 atom, so the world has 7 atoms and 2
ports.
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest
from molrs.builder import Assembler, AxisOrienter, GrowthPlacer, SitePlacer


def _monomer() -> molrs.core.Atomistic:
    fragment = molrs.core.Atomistic()
    c0 = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0, mass=12.011)
    c1 = fragment.add_atom(atomic_number=6, x=1.54, y=0.0, z=0.0, mass=12.011)
    h0 = fragment.add_atom(atomic_number=1, x=-0.5, y=0.9, z=0.0, mass=1.008)
    h1 = fragment.add_atom(atomic_number=1, x=2.04, y=0.9, z=0.0, mass=1.008)
    fragment.add_bond(c0, c1)
    fragment.add_bond(c0, h0)
    fragment.add_bond(c1, h1)
    fragment.add_port(c0, h0, ">")
    fragment.add_port(c1, h1, "<")
    return fragment


def _lithium() -> molrs.core.Atomistic:
    mol = molrs.core.Atomistic()
    mol.add_atom(atomic_number=3, x=0.0, y=0.0, z=0.0, mass=6.94)
    return mol


def _sites(names: list[str], *, axes: bool = True) -> molrs.core.CoarseGrain:
    """Sites along +x, 4 Å apart; the first two bonded; axes +z unless off."""
    n = len(names)
    atoms = {
        "type": np.array(names),
        "x": 4.0 * np.arange(n, dtype=np.float64),
        "y": np.zeros(n),
        "z": np.zeros(n),
    }
    if axes:
        atoms |= {"axis_x": np.zeros(n), "axis_y": np.zeros(n), "axis_z": np.ones(n)}
    frame = molrs.core.Frame(
        {"atoms": atoms, "bonds": {"atomi": np.array([0]), "atomj": np.array([1])}}
    )
    return molrs.core.CoarseGrain.from_frame(frame)


def _assembler(library: dict) -> Assembler:
    return Assembler(library, SitePlacer(), AxisOrienter())


def test_assemble_links_bonded_sites_and_places_atomistic_singles() -> None:
    world = _assembler({"M": _monomer(), "Li": _lithium()}).assemble(
        _sites(["M", "M", "Li"])
    )

    assert type(world) is molrs.core.Atomistic
    assert world.n_atoms == 7
    assert world.n_ports == 2
    atoms = world.to_frame()["atoms"]
    mol_id = np.asarray(atoms["mol_id"])
    atomic_number = np.asarray(atoms["atomic_number"])
    assert set(mol_id.tolist()) == {1, 2}
    assert mol_id[atomic_number == 3].tolist() == [2]


def test_an_unknown_name_is_a_value_error_naming_the_site() -> None:
    with pytest.raises(ValueError, match=r"site 2 names 'Li'"):
        _assembler({"M": _monomer()}).assemble(_sites(["M", "M", "Li"]))


def test_a_chain_site_without_an_axis_is_a_value_error_naming_the_site() -> None:
    with pytest.raises(ValueError, match=r"site 0 \('M'\) cannot be oriented"):
        _assembler({"M": _monomer()}).assemble(_sites(["M", "M"], axes=False))


def test_a_placer_that_is_not_a_site_placer_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Assembler({"M": _monomer()}, object(), AxisOrienter())


def test_an_orienter_that_is_not_an_axis_orienter_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Assembler({"M": _monomer()}, SitePlacer(), object())


def test_placers_and_orienter_take_no_arguments() -> None:
    with pytest.raises(TypeError):
        SitePlacer(1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        GrowthPlacer(1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        AxisOrienter(1.0)  # type: ignore[call-arg]


def test_growth_placer_builds_a_topology_without_positions() -> None:
    sites = molrs.io.cgsmiles.read_cgsmiles("{[#M]|3}").to_coarsegrain()
    world = Assembler({"M": _monomer()}, GrowthPlacer()).assemble(sites)

    # Three copies of 4 atoms, two links remove 4 hydrogens.
    assert world.n_atoms == 8
    assert world.n_ports == 2


def test_an_orienter_on_a_site_graph_without_positions_is_a_value_error() -> None:
    sites = molrs.io.cgsmiles.read_cgsmiles("{[#M]|2}").to_coarsegrain()
    with pytest.raises(ValueError, match="needs site positions"):
        Assembler({"M": _monomer()}, GrowthPlacer(), AxisOrienter()).assemble(sites)


def _bare_monomer() -> molrs.core.MolGraph:
    """The monomer as a bare MolGraph: the same nodes, bonds and ports."""
    graph = molrs.core.MolGraph()
    nodes = [
        graph.add_node(atomic_number=6, x=0.0, y=0.0, z=0.0, mass=12.011),
        graph.add_node(atomic_number=6, x=1.54, y=0.0, z=0.0, mass=12.011),
        graph.add_node(atomic_number=1, x=-0.5, y=0.9, z=0.0, mass=1.008),
        graph.add_node(atomic_number=1, x=2.04, y=0.9, z=0.0, mass=1.008),
    ]
    graph.register_kind("bonds", 2)
    for i, j in ((0, 1), (0, 2), (1, 3)):
        graph.add_relation("bonds", [nodes[i], nodes[j]], bond_type=1, bond_number=1)
    graph.register_kind("ports", 2)
    graph.add_relation(
        "ports", [nodes[0], nodes[2]], port_kind=">", port_label="", port_order=1
    )
    graph.add_relation(
        "ports", [nodes[1], nodes[3]], port_kind="<", port_label="", port_order=1
    )
    return graph


def test_the_world_is_of_the_librarys_graph_type() -> None:
    sites = molrs.io.cgsmiles.read_cgsmiles("{[#M]|2}").to_coarsegrain()
    world = Assembler({"M": _bare_monomer()}, GrowthPlacer()).assemble(sites)

    assert type(world) is molrs.core.MolGraph
    assert world.n_nodes == 6


def test_a_library_of_mixed_graph_types_is_a_type_error() -> None:
    bare = molrs.core.MolGraph()
    with pytest.raises(TypeError, match="one graph type"):
        Assembler({"M": _monomer(), "X": bare}, GrowthPlacer())
    with pytest.raises(TypeError, match="graphs"):
        Assembler({"M": {}}, GrowthPlacer())


def test_an_empty_library_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="at least one template"):
        Assembler({}, GrowthPlacer())
