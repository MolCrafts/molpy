"""``mp.builder.Assembler`` / ``mp.builder.SitePlacer`` / ``mp.builder.AxisOrienter`` FFI seam:
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

import molpy as mp
import numpy as np
import pytest


def _monomer() -> mp.core.Atomistic:
    fragment = mp.core.Atomistic()
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


def _lithium() -> mp.core.Atomistic:
    mol = mp.core.Atomistic()
    mol.add_atom(atomic_number=3, x=0.0, y=0.0, z=0.0, mass=6.94)
    return mol


def _sites(names: list[str], *, axes: bool = True) -> mp.core.CoarseGrain:
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
    frame = mp.core.Frame(
        {"atoms": atoms, "bonds": {"atomi": np.array([0]), "atomj": np.array([1])}}
    )
    return mp.core.CoarseGrain.from_frame(frame)


def _assembler(library: dict) -> mp.builder.Assembler:
    return mp.builder.Assembler(
        library, mp.builder.SitePlacer(), mp.builder.AxisOrienter()
    )


def test_assemble_links_bonded_sites_and_places_atomistic_singles() -> None:
    world = _assembler({"M": _monomer(), "Li": _lithium()}).assemble(
        _sites(["M", "M", "Li"])
    )

    assert type(world) is mp.core.Atomistic
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
        mp.builder.Assembler({"M": _monomer()}, object(), mp.builder.AxisOrienter())


def test_an_orienter_that_is_not_an_axis_orienter_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        mp.builder.Assembler({"M": _monomer()}, mp.builder.SitePlacer(), object())


def test_placers_and_orienter_take_no_arguments() -> None:
    with pytest.raises(TypeError):
        mp.builder.SitePlacer(1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        mp.builder.GrowthPlacer(1.0)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        mp.builder.AxisOrienter(1.0)  # type: ignore[call-arg]


def test_growth_placer_builds_a_topology_without_positions() -> None:
    sites = mp.io.cgsmiles.read_cgsmiles("{[#M]|3}").to_coarsegrain()
    world = mp.builder.Assembler({"M": _monomer()}, mp.builder.GrowthPlacer()).assemble(
        sites
    )

    # Three copies of 4 atoms, two links remove 4 hydrogens.
    assert world.n_atoms == 8
    assert world.n_ports == 2


def test_an_orienter_on_a_site_graph_without_positions_is_a_value_error() -> None:
    sites = mp.io.cgsmiles.read_cgsmiles("{[#M]|2}").to_coarsegrain()
    with pytest.raises(ValueError, match="needs site positions"):
        mp.builder.Assembler(
            {"M": _monomer()}, mp.builder.GrowthPlacer(), mp.builder.AxisOrienter()
        ).assemble(sites)


def _bare_monomer() -> mp.core.MolGraph:
    """The monomer as a bare MolGraph: the same nodes, bonds and ports."""
    graph = mp.core.MolGraph()
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
    sites = mp.io.cgsmiles.read_cgsmiles("{[#M]|2}").to_coarsegrain()
    world = mp.builder.Assembler(
        {"M": _bare_monomer()}, mp.builder.GrowthPlacer()
    ).assemble(sites)

    assert type(world) is mp.core.MolGraph
    assert world.n_nodes == 6


def test_a_library_of_mixed_graph_types_is_a_type_error() -> None:
    bare = mp.core.MolGraph()
    with pytest.raises(TypeError, match="one graph type"):
        mp.builder.Assembler({"M": _monomer(), "X": bare}, mp.builder.GrowthPlacer())
    with pytest.raises(TypeError, match="graphs"):
        mp.builder.Assembler({"M": {}}, mp.builder.GrowthPlacer())


def test_an_empty_library_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="at least one template"):
        mp.builder.Assembler({}, mp.builder.GrowthPlacer())


def test_a_grown_chain_joins_units_at_the_covalent_bond_length() -> None:
    """The inter-unit C–C bond is two carbon covalent radii (1.52 Å) long, not
    the C–H length of the hydrogen it replaced."""
    sites = mp.io.cgsmiles.read_cgsmiles("{[#M]|3}").to_coarsegrain()
    frame = (
        mp.builder.Assembler({"M": _monomer()}, mp.builder.GrowthPlacer())
        .assemble(sites)
        .to_frame()
    )
    atoms, bonds = frame["atoms"], frame["bonds"]
    xyz = np.stack(
        [np.asarray(atoms["x"]), np.asarray(atoms["y"]), np.asarray(atoms["z"])], axis=1
    )
    z = np.asarray(atoms["atomic_number"])
    lengths = sorted(
        round(float(np.linalg.norm(xyz[i] - xyz[j])), 3)
        for i, j in zip(np.asarray(bonds["atomi"]), np.asarray(bonds["atomj"]))
        if z[i] == 6 and z[j] == 6
    )
    assert lengths == [1.52, 1.52, 1.54, 1.54, 1.54]


def _ladder_unit() -> mp.core.Atomistic:
    """Four carbons in a square, two ``<`` ports on the left and two ``>`` on
    the right (labels a, b pair the rails), hydrogens as leaving groups."""
    unit = mp.core.Atomistic()
    c = [
        unit.add_atom(atomic_number=6, x=x, y=y, z=0.0, mass=12.011)
        for x, y in [(0, 0), (0, 1.4), (1.4, 0), (1.4, 1.4)]
    ]
    h = [
        unit.add_atom(atomic_number=1, x=x, y=y, z=0.0, mass=1.008)
        for x, y in [(-1.09, 0), (-1.09, 1.4), (2.49, 0), (2.49, 1.4)]
    ]
    for i, j in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        unit.add_bond(c[i], c[j])
    for ci, hi in zip(c, h):
        unit.add_bond(ci, hi)
    unit.add_port(c[0], h[0], "<", "a")
    unit.add_port(c[1], h[1], "<", "b")
    unit.add_port(c[2], h[2], ">", "a")
    unit.add_port(c[3], h[3], ">", "b")
    return unit


def test_a_ladder_closes_both_bonds_between_neighbours() -> None:
    """Two site bonds between neighbours (a ladder) are both closed at the
    covalent length, with no atoms piled on each other."""
    frame = mp.core.Frame(
        {
            "atoms": {"type": np.array(["U", "U", "U"])},
            "bonds": {"atomi": np.array([0, 0, 1, 1]), "atomj": np.array([1, 1, 2, 2])},
        }
    )
    world = (
        mp.builder.Assembler({"U": _ladder_unit()}, mp.builder.GrowthPlacer())
        .assemble(mp.core.CoarseGrain.from_frame(frame))
        .to_frame()
    )
    atoms, bonds = world["atoms"], world["bonds"]
    xyz = np.stack(
        [np.asarray(atoms["x"]), np.asarray(atoms["y"]), np.asarray(atoms["z"])], axis=1
    )
    frag = np.asarray(atoms["frag_id"])
    pairs = list(zip(np.asarray(bonds["atomi"]), np.asarray(bonds["atomj"])))
    joins = [
        float(np.linalg.norm(xyz[i] - xyz[j])) for i, j in pairs if frag[i] != frag[j]
    ]
    assert len(joins) == 4
    assert np.allclose(joins, 1.52, atol=1e-3), joins
    bonded = {tuple(sorted((int(i), int(j)))) for i, j in pairs}
    distances = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    crowded = [
        (i, j)
        for i in range(len(xyz))
        for j in range(i + 1, len(xyz))
        if (i, j) not in bonded and distances[i, j] < 1.0
    ]
    assert crowded == []


@pytest.mark.parametrize("src", [-1, True, 1.5])
def test_port_ordinals_must_be_unsigned_integers(src) -> None:
    sites = mp.CoarseGrain()
    a = sites.add_bead(bead_type="M")
    b = sites.add_bead(bead_type="M")
    with pytest.raises(ValueError, match="port"):
        sites.add_bond(a, b, src_port=src, dst_port=0)
    assert sites.n_bonds == 0


@pytest.mark.parametrize("props", [{"src_port": 0}, {"dst_port": 0}])
def test_a_lone_port_ordinal_is_refused_by_the_assembler(props) -> None:
    sites = mp.CoarseGrain()
    a = sites.add_bead(bead_type="M")
    b = sites.add_bead(bead_type="M")
    sites.add_bond(a, b, **props)
    with pytest.raises(ValueError, match="supplied together"):
        mp.builder.Assembler({"M": _monomer()}, mp.builder.GrowthPlacer()).assemble(
            sites
        )


def test_explicit_port_keywords_reach_assembler_and_roundtrip() -> None:
    sites = mp.CoarseGrain()
    a = sites.add_bead(bead_type="M")
    b = sites.add_bead(bead_type="M")
    sites.add_bond(a, b, src_port=1, dst_port=0)
    restored = mp.CoarseGrain.from_frame(sites.to_frame())
    assert restored.bonds["src_port"].tolist() == [1]
    assert restored.bonds["dst_port"].tolist() == [0]
    world = mp.builder.Assembler({"M": _monomer()}, mp.builder.GrowthPlacer()).assemble(
        restored
    )
    assert isinstance(world, mp.Atomistic)
    assert world.n_atoms == 6
    # Passing two left ports must refuse rather than choose another pair.
    bad = mp.CoarseGrain()
    x = bad.add_bead(bead_type="M")
    y = bad.add_bead(bead_type="M")
    bad.add_bond(x, y, src_port=1, dst_port=1)
    with pytest.raises(ValueError, match=r"bond 0.*site 0.*site 1.*incompatible"):
        mp.builder.Assembler({"M": _monomer()}, mp.builder.GrowthPlacer()).assemble(bad)


def test_conformer_preserves_ordinal_port_identity() -> None:
    template = mp.io.smiles.read_smiles_fragment("[$]COCC[$]").to_template()
    for ordinal, port in enumerate(template.ports):
        port.anchor["port_site"] = ordinal
    posed = mp.conformer.Conformer(seed=42).generate(template)[0]
    assert [port.anchor["port_site"] for port in posed.ports] == [0, 1]
    assert [port.anchor["port_site"] for port in template.ports] == [0, 1]
    assert posed.n_ports == template.n_ports == 2


def _typed_monomer() -> mp.core.Atomistic:
    """The monomer with its angles and pairs generated, angles stamped, and
    a charge on every atom."""
    fragment = _monomer()
    fragment.generate_topology(gen_dihedral=False)
    for angle in fragment.angles:
        angle["type"] = "stamped"
    for atom in fragment.atoms:
        atom["charge"] = 0.25 if atom["atomic_number"] == 1 else -0.25
    return fragment


def test_assemble_with_report_crosses_the_junctions() -> None:
    world, report = _assembler({"M": _typed_monomer()}).assemble_with_report(
        _sites(["M", "M"])
    )

    assert type(world) is mp.core.Atomistic
    assert isinstance(report, mp.builder.JunctionReport)
    (junction,) = report.junctions
    assert {a.handle for a in junction.anchors} == {
        a.handle for a in junction.bond.endpoints
    }
    assert junction.folds == (0.25, 0.25)
    # C0-C1-C1' and C1-C1'-C0' are new, and bare; the monomers' angles keep
    # their stamp.
    assert report.n_added_angles == 2
    bare = [angle for angle in world.angles if "type" not in angle]
    assert len(bare) == 2
    assert len(report.reclassified_pairs) == 0


def test_assemble_with_report_needs_atomistic_templates() -> None:
    sites = mp.core.CoarseGrain()
    template = mp.core.CoarseGrain()
    template.add_bead(bead_type="B", x=0.0, y=0.0, z=0.0)
    sites.add_bead(bead_type="B", x=0.0, y=0.0, z=0.0)
    with pytest.raises(TypeError, match="Atomistic"):
        mp.builder.Assembler(
            {"B": template}, mp.builder.SitePlacer()
        ).assemble_with_report(sites)


def test_update_junctions_takes_what_link_returned() -> None:
    """A hand-made ring: the 4-carbon chain's ends linked. The former 1-4 pair
    becomes 1-2 and loses its override; the ring's angles through the new
    bond are added."""
    chain = mp.core.Atomistic()
    carbons = [chain.add_atom(atomic_number=6) for _ in range(4)]
    for a, b in zip(carbons, carbons[1:]):
        chain.add_bond(a, b)
    ha = chain.add_atom(atomic_number=1)
    hd = chain.add_atom(atomic_number=1)
    chain.add_bond(carbons[0], ha)
    chain.add_bond(carbons[3], hd)
    left = chain.add_port(carbons[0], ha, "<")
    right = chain.add_port(carbons[3], hd, ">")
    chain.generate_topology()
    for pair in chain.terms("pairs"):
        pair["epsilon"] = 0.5

    junction = chain.link(right, left)
    report = mp.builder.update_junctions(chain, [junction])

    assert isinstance(report, mp.builder.JunctionReport)
    (pair,) = report.reclassified_pairs
    assert pair["n_bonds"] == 1 and "epsilon" not in pair
    assert report.n_added_angles == 2
    again = mp.builder.update_junctions(chain, [junction])
    assert (again.n_added_angles, len(again.reclassified_pairs)) == (0, 0)


def test_update_junctions_refuses_a_junction_of_another_graph() -> None:
    first = _monomer()
    first.merge(_monomer())
    right = next(p for p in first.ports if p["port_kind"] == ">")
    left = [p for p in first.ports if p["port_kind"] == "<"][-1]
    junction = first.link(right, left)
    with pytest.raises(ValueError, match="another graph"):
        mp.builder.update_junctions(_monomer(), [junction])


def test_a_patch_retypes_the_linked_copy() -> None:
    """Row 0 (C0) is the ``>`` anchor: in a two-site chain only the first
    copy's C0 is linked, so only it takes the patched type."""
    template = _monomer()
    for atom in template.atoms:
        atom["type"] = "c" if atom["atomic_number"] == 6 else "h"
    patch = mp.builder.Patch({("M", 0): {0: "c-linked"}})
    world, report = _assembler_with({"M": template}, patch).assemble_with_report(
        _sites(["M", "M"])
    )

    linked = [atom for atom in world.atoms if atom["type"] == "c-linked"]
    assert len(linked) == 1
    ((atom, before, after),) = report.retyped
    assert (atom.handle, before, after) == (linked[0].handle, "c", "c-linked")


def test_a_patch_that_cannot_apply_is_a_value_error() -> None:
    chain = mp.core.Atomistic()
    first, second = _monomer(), _monomer()
    chain.merge(first)
    chain.merge(second)
    right = next(p for p in chain.ports if p["port_kind"] == ">")
    left = [p for p in chain.ports if p["port_kind"] == "<"][-1]
    junction = chain.link(right, left)
    # A hand-made graph without provenance cannot be patched.
    with pytest.raises(ValueError, match="frag_id"):
        mp.builder.update_junctions(
            chain, [junction], retype=mp.builder.Patch({("M", 0): {0: "x"}})
        )


def test_retype_needs_atomistic_templates() -> None:
    template = mp.core.CoarseGrain()
    template.add_bead(bead_type="B", x=0.0, y=0.0, z=0.0)
    with pytest.raises(TypeError, match="Atomistic"):
        mp.builder.Assembler(
            {"B": template}, mp.builder.SitePlacer(), retype=mp.builder.Patch({})
        )


def _assembler_with(library: dict, retype: mp.builder.Patch) -> mp.builder.Assembler:
    return mp.builder.Assembler(
        library, mp.builder.SitePlacer(), mp.builder.AxisOrienter(), retype=retype
    )
