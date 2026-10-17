"""``molrs.builder.Coarsener`` FFI seam: sources accepted, the graph out and
error mapping. The centre-of-mass mapping itself is proven by the Rust suite.
"""

from __future__ import annotations

import itertools

import molrs
import pytest
from molrs.builder import Coarsener


def _bead_chain() -> tuple[molrs.core.CoarseGrain, list[molrs.core.Bead]]:
    """Four unit-mass beads ``b0-b1-b2-b3`` on the x axis, 1 Å apart."""
    cg = molrs.core.CoarseGrain()
    beads = [
        cg.add_bead(bead_type="S", x=float(i), y=0.0, z=0.0, mass=1.0) for i in range(4)
    ]
    for a, b in itertools.pairwise(beads):
        cg.add_bond(a, b)
    return cg, beads


def test_coarsen_returns_a_coarsegrain_with_one_site_per_group() -> None:
    cg, (b0, b1, b2, b3) = _bead_chain()

    sites = Coarsener(cg).coarsen([[b0, b1], [b2, b3]], ["A", "B"])

    assert type(sites) is molrs.core.CoarseGrain
    assert sites.n_beads == 2
    assert sites.n_bonds == 1
    assert sites.bead_types(sites.beads) == ["A", "B"]
    # Each site's members are its group, as views of the source.
    assert sites.bead_members(sites.beads[0]) == (b0, b1)
    assert sites.beads_of_atom(b3) == [sites.beads[1]]


def test_an_atomistic_source_is_accepted() -> None:
    mol = molrs.core.Atomistic()
    c = mol.add_atom(atomic_number=6, x=0.0, y=0.0, z=0.0, mass=12.011)
    h = mol.add_atom(atomic_number=1, x=1.09, y=0.0, z=0.0, mass=1.008)
    mol.add_bond(c, h)

    sites = Coarsener(mol).coarsen([[c, h]], ["CH"])

    assert type(sites) is molrs.core.CoarseGrain
    assert sites.n_beads == 1
    assert sites.bead_members(sites.beads[0]) == (c, h)


def test_a_group_node_of_another_graph_is_a_value_error() -> None:
    cg, _ = _bead_chain()
    _, (foreign, *_) = _bead_chain()
    with pytest.raises(ValueError, match="another graph"):
        Coarsener(cg).coarsen([[foreign]], ["A"])


def test_a_frame_source_is_a_type_error() -> None:
    with pytest.raises(TypeError):
        Coarsener(molrs.core.Frame())


def test_an_overlap_is_a_value_error_naming_the_int_handle() -> None:
    cg, (b0, b1, b2, _) = _bead_chain()

    with pytest.raises(ValueError) as caught:
        Coarsener(cg).coarsen([[b0, b1], [b1, b2]], ["A", "B"])

    message = str(caught.value)
    assert str(b1.handle) in message
    assert "NodeId(" not in message
