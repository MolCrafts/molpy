"""Python surface for ports: named attachment points any graph may carry.

These are FFI-seam tests: they prove that ports reach Python on a plain
``mp.core.Atomistic`` through its one port writer, ``add_port``, that the
typed writers reach the core ones (a Python-built bond carries its class and a
Python-built port is validated), that graph-out paths keep the public class
and its ports, and that the port vocabulary crosses as the notation glyph.

They re-derive no chemistry and no geometry: every claim about ports,
embedding and ``frag_id`` propagation is owned by the Rust unit tests in
``crates/molrs/src/core/port.rs``, ``crates/molrs/src/io/cgsmiles/`` and
``crates/molrs/src/conformer/``, and is reused here only to show Python sees the same
answer. Fixtures are built in process; no third-party scientific software runs.
"""

from __future__ import annotations

import copy
import math

import molpy as mp
import pytest

# An OH-capped PEO trimer: the fragment table names exactly ``OH`` and ``PEO``.
F2 = "{[#OH][#PEO]|3[#OH]}.{#OH=[$]O,#PEO=[$]COC[$]}"


def _capped_fragment() -> tuple[
    mp.core.Atomistic, mp.core.Atom, mp.core.Atom, mp.core.Atom
]:
    """``C–O–H`` with one unnamed ``$`` port on the capping hydrogen.

    The smallest graph that can carry a legal port: ``add_port`` requires the
    handle to be a hydrogen *bonded to* its anchor. Returns
    ``(fragment, carbon, oxygen, hydrogen)``.
    """
    fragment = mp.core.Atomistic()
    carbon = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    oxygen = fragment.add_atom(atomic_number=8, x=1.43, y=0.0, z=0.0)
    hydrogen = fragment.add_atom(atomic_number=1, x=2.39, y=0.0, z=0.0)
    fragment.add_bond(carbon, oxygen)
    fragment.add_bond(oxygen, hydrogen)
    fragment.add_port(oxygen, hydrogen, "$")
    return fragment, carbon, oxygen, hydrogen


# ---------------------------------------------------------------------------
# Ports: the kind appears on the first port
# ---------------------------------------------------------------------------


def test_a_graph_without_ports_has_none() -> None:
    mol = mp.core.Atomistic()
    assert mol.n_ports == 0
    assert list(mol.ports) == []

    fragment, _carbon, _oxygen, _hydrogen = _capped_fragment()

    assert fragment.n_ports == 1
    # Ports are no term block: they have typed doors.
    assert fragment.term_blocks() == []


def test_port_on_leaving_group_and_del_port() -> None:
    fragment, carbon, oxygen, hydrogen = _capped_fragment()
    port = fragment.port_on(oxygen, hydrogen)
    assert port is fragment.ports[0]
    assert fragment.port_on(carbon, oxygen) is None
    assert fragment.leaving_group(port) == [hydrogen]
    fragment.del_port(port)
    assert fragment.n_ports == 0
    assert fragment.n_bonds == 2  # the bond stays
    fragment.add_port(oxygen, hydrogen, "$")
    fragment.del_ports(fragment.ports)
    assert fragment.n_ports == 0


def test_fragment_ports_are_interned_live_views() -> None:
    fragment, _carbon, oxygen, hydrogen = _capped_fragment()

    assert fragment.n_atoms == 3
    assert fragment.n_ports == 1
    assert fragment.ports[0] is fragment.ports[0]

    port = fragment.ports[0]
    assert port.anchor is oxygen
    assert port.handle_atom is hydrogen
    assert port["port_kind"] == "$"
    assert port["port_label"] == ""
    assert port["port_order"] == 1


def test_add_bond_stamps_bond_class() -> None:
    fragment = mp.core.Atomistic()
    carbon = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    oxygen = fragment.add_atom(atomic_number=8, x=1.43, y=0.0, z=0.0)
    fragment.add_bond(carbon, oxygen)

    # The native writer stamps both facts; the generic relation path does not.
    assert fragment.bonds[0]["bond_type"] == 1
    assert fragment.bonds[0]["bond_number"] == 1


def test_frag_id_round_trips() -> None:
    fragment, carbon, oxygen, hydrogen = _capped_fragment()
    carbon["frag_id"] = 7
    oxygen["frag_id"] = 7

    assert carbon["frag_id"] == 7
    assert "frag_id" not in hydrogen

    # One pass: the unlabelled degree-1 hydrogen inherits from its only bonded
    # neighbour; the labelled atoms are left alone.
    assert fragment.inherit_frag_ids() == 1
    assert hydrogen["frag_id"] == 7


def test_graph_out_paths_keep_the_public_fragment_type() -> None:
    fragment, carbon, oxygen, hydrogen = _capped_fragment()
    # ``to_frame`` emits the ``frag_id`` column only when every atom carries
    # one, so all three are labelled — with distinct ids, which also pins the
    # per-atom order across the round trip.
    for atom, frag_id in ((carbon, 1), (oxygen, 2), (hydrogen, 3)):
        atom["frag_id"] = frag_id

    for result in (
        copy.copy(fragment),
        mp.core.Atomistic.from_frame(fragment.to_frame()),
    ):
        assert type(result) is mp.core.Atomistic
        assert result.n_atoms == 3
        assert result.n_ports == 1
        assert result.ports[0]["port_kind"] == "$"
        assert result.atoms["frag_id"].tolist() == [1, 2, 3]


# ---------------------------------------------------------------------------
# Geometry: the leaf's own atoms move, not the empty base graph
# ---------------------------------------------------------------------------


def test_translate_moves_fragment_atoms() -> None:
    fragment, carbon, _oxygen, _hydrogen = _capped_fragment()

    fragment.translate((1.0, 0.0, 0.0))

    assert carbon["x"] == pytest.approx(1.0, abs=1e-12)
    assert carbon["y"] == pytest.approx(0.0, abs=1e-12)
    assert carbon["z"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Embedding: leaf in, same leaf out
# ---------------------------------------------------------------------------


def test_conformer_keeps_the_ports_of_an_atomistic() -> None:
    # Three heavy atoms plus the capping hydrogen the port needs.
    fragment = mp.core.Atomistic()
    head = fragment.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0)
    ether = fragment.add_atom(atomic_number=8, x=1.43, y=0.0, z=0.0)
    tail = fragment.add_atom(atomic_number=6, x=2.86, y=0.0, z=0.0)
    hydrogen = fragment.add_atom(atomic_number=1, x=-1.09, y=0.0, z=0.0)
    fragment.add_bond(head, ether)
    fragment.add_bond(ether, tail)
    fragment.add_bond(head, hydrogen)
    fragment.add_port(head, hydrogen, "$")

    embedded, report = mp.conformer.Conformer(speed="fast", seed=42).generate(fragment)

    assert type(embedded) is mp.core.Atomistic
    assert isinstance(report, mp.conformer.ConformerReport)
    assert embedded.n_ports == 1
    for atom in embedded.atoms:
        # Coordinates are Å; the seam claim is that they exist and are finite.
        assert all(math.isfinite(atom[key]) for key in ("x", "y", "z"))


def test_conformer_rejects_a_non_graph() -> None:
    with pytest.raises(TypeError) as excinfo:
        mp.conformer.Conformer(speed="fast", seed=42).generate(object())

    message = str(excinfo.value)
    assert "Atomistic" in message


# ---------------------------------------------------------------------------
# The port vocabulary is the glyph, and the writer validates it
# ---------------------------------------------------------------------------


def test_add_port_rejects_an_unknown_kind() -> None:
    fragment, _carbon, oxygen, hydrogen = _capped_fragment()

    # Only possible if ``add_port`` reaches core ``add_port``: the generic
    # relation path would write ``"Z"`` into ``port_kind`` without complaint.
    with pytest.raises(ValueError):
        fragment.add_port(oxygen, hydrogen, "Z")
    assert fragment.n_ports == 1


def test_add_port_refuses_an_atom_of_another_graph() -> None:
    fragment, _carbon, oxygen, _hydrogen = _capped_fragment()
    _other, _c, _o, foreign_hydrogen = _capped_fragment()

    with pytest.raises(ValueError, match="another graph"):
        fragment.add_port(oxygen, foreign_hydrogen, "$")
    assert fragment.n_ports == 1


# ---------------------------------------------------------------------------
# Public-API example (this repo's stand-in for a regressions/ script)
# ---------------------------------------------------------------------------


def test_templates_returns_named_ported_templates() -> None:
    """Read the two fragment bodies of an OH-capped PEO trimer.

    Hand-derived from the notation: ``#PEO=[$]COC[$]`` writes two bonding
    descriptors and ``#OH=[$]O`` writes one, so the fragments they build carry
    two and one port respectively. No third-party tool produced these numbers.
    """
    fragments = mp.io.cgsmiles.read_cgsmiles(F2).templates()

    assert isinstance(fragments, dict)
    assert set(fragments) == {"OH", "PEO"}
    for fragment in fragments.values():
        assert type(fragment) is mp.core.Atomistic
    assert fragments["PEO"].n_ports == 2
    assert fragments["OH"].n_ports == 1


def test_from_fragment_to_template_builds_one_unit() -> None:
    """``[<]OCC[>]``: three heavy atoms plus one hydrogen handle per
    descriptor, one ``<`` and one ``>`` port — the same unit a one-entry table
    builds."""
    unit = mp.io.smiles.read_smiles_fragment("[<]OCC[>]").to_template()
    assert type(unit) is mp.core.Atomistic
    assert (unit.n_atoms, unit.n_bonds, unit.n_ports) == (5, 4, 2)
    from_table = mp.io.cgsmiles.read_cgsmiles("{[#EO]}.{#EO=[<]OCC[>]}").templates()[
        "EO"
    ]
    assert (from_table.n_atoms, from_table.n_ports) == (5, 2)


def test_plain_smiles_ir_refuses_a_descriptor() -> None:
    with pytest.raises(ValueError):
        mp.io.smiles.read_smiles("[<]OCC[>]")


# ---------------------------------------------------------------------------
# merge / link
#
# Seam only: the handle maps, the leaving-group removal and the refusals are
# proven by ``crates/molrs/src/core/port.rs`` and ``link.rs``. These tests
# check the Python shapes (the merged atoms, the new bond's view) and that
# every refusal is a ``ValueError`` that leaves the graph as it was.
# ---------------------------------------------------------------------------


def _add_ported_monomer(
    fragment: mp.core.Atomistic, y: float = 0.0
) -> dict[str, mp.core.Port]:
    """Write ``H0–C0–C1–H1`` with ports ``(C0, H0, ">")`` and ``(C1, H1, "<")``
    into ``fragment`` through the native writers; return the ports keyed by
    glyph.

    Positions in Å and masses in g/mol are hand-set: C0 (0, y, 0), C1 (1.54,
    y, 0), H0 (-1, y, 0), H1 (2.54, y, 0); C 12.011, H 1.008.
    """
    c0 = fragment.add_atom(atomic_number=6, x=0.0, y=y, z=0.0, mass=12.011)
    c1 = fragment.add_atom(atomic_number=6, x=1.54, y=y, z=0.0, mass=12.011)
    h0 = fragment.add_atom(atomic_number=1, x=-1.0, y=y, z=0.0, mass=1.008)
    h1 = fragment.add_atom(atomic_number=1, x=2.54, y=y, z=0.0, mass=1.008)
    fragment.add_bond(c0, c1)
    fragment.add_bond(c0, h0)
    fragment.add_bond(c1, h1)
    head = fragment.add_port(c0, h0, ">")
    tail = fragment.add_port(c1, h1, "<")
    return {">": head, "<": tail}


def _ported_monomer() -> mp.core.Atomistic:
    """One ported monomer (see :func:`_add_ported_monomer`) in its own
    fragment."""
    fragment = mp.core.Atomistic()
    _add_ported_monomer(fragment)
    return fragment


def _two_monomers() -> tuple[
    mp.core.Atomistic, dict[str, mp.core.Port], dict[str, mp.core.Port]
]:
    """One fragment holding two separate ported monomers, written directly
    (no ``merge``), and each monomer's ports keyed by glyph.

    The second copy sits 10 Å along +y so the two are disjoint in space too.
    """
    fragment = mp.core.Atomistic()
    first = _add_ported_monomer(fragment, y=0.0)
    second = _add_ported_monomer(fragment, y=10.0)
    return fragment, first, second


def test_merge_carries_the_ports_across_and_empties_other() -> None:
    world = _ported_monomer()
    other = _ported_monomer()

    merged = world.merge(other)

    assert len(merged) == 4
    assert set(merged) <= set(world.atoms)
    assert other.n_atoms == 0
    assert other.n_ports == 0
    assert world.n_atoms == 8
    assert world.n_ports == 4


def test_merge_type_conflict_is_a_value_error() -> None:
    world = mp.core.Atomistic()
    world.add_atom(atomic_number=6, tag=1.0)
    other = mp.core.Atomistic()
    other.add_atom(atomic_number=6, tag="one")

    with pytest.raises(ValueError):
        world.merge(other)


def test_link_returns_the_bond_and_consumes_both_ports() -> None:
    world, first, second = _two_monomers()
    assert world.n_atoms == 8
    assert world.n_ports == 4

    bond = world.link(first[">"], second["<"])

    assert type(bond) is mp.core.Bond
    assert bond in world.bonds
    assert {bond.itom["atomic_number"], bond.jtom["atomic_number"]} == {6}
    # One leaving hydrogen per port.
    assert world.n_atoms == 6
    assert world.n_ports == 2


def test_link_refusal_is_a_value_error_naming_int_handles() -> None:
    world, first, second = _two_monomers()

    # Two ">" ports are not complements, so the pair is refused.
    with pytest.raises(ValueError) as excinfo:
        world.link(first[">"], second[">"])

    message = str(excinfo.value)
    assert "RelationId(" not in message
    assert "NodeId(" not in message
    assert str(first[">"].handle) in message
    assert str(second[">"].handle) in message
    assert world.n_atoms == 8
    assert world.n_ports == 4


def test_link_of_a_stale_port_is_a_value_error_naming_int_handles() -> None:
    world, first, second = _two_monomers()
    world.link(first[">"], second["<"])
    stale = first[">"]  # consumed by the link above
    n_atoms, n_ports = world.n_atoms, world.n_ports

    with pytest.raises(ValueError) as excinfo:
        world.link(stale, second[">"])

    message = str(excinfo.value)
    assert str(stale.handle) in message
    assert "RelationId(" not in message
    assert "NodeId(" not in message
    assert world.n_atoms == n_atoms
    assert world.n_ports == n_ports


def test_link_refuses_a_port_of_another_graph() -> None:
    world, first, _ = _two_monomers()
    _, foreign, _ = _two_monomers()
    with pytest.raises(ValueError, match="another graph"):
        world.link(first[">"], foreign["<"])
    assert world.n_ports == 4
