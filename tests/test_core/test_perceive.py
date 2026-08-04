"""molpy's :class:`~molpy.core.perceive.Perceive` — the perception builder.

The subclass exists for exactly one reason: perception must hand back a molpy
graph, so no caller writes ``Atomistic.adopt(...)`` around the result. These
tests pin that, and pin it for **every** method rather than for a sampled one —
a method added to :class:`molrs.perceive.Perceive` and not overridden here would silently
start returning bare molrs graphs again, and a spot-check would not see it.
"""

from __future__ import annotations

import molrs
import pytest

import molpy as mp
from molpy.core.atomistic import Atomistic
from molpy.core.perceive import Perceive


def _perception_methods() -> list[str]:
    """Every ``find_*`` molrs exposes — read off molrs, never hand-listed."""
    return sorted(
        name
        for name in dir(molrs.perceive.Perceive)
        if name.startswith("find_") and callable(getattr(molrs.perceive.Perceive, name))
    )


class TestPerceive:
    def test_molrs_exposes_the_perception_methods_this_suite_scans(self):
        # If this ever empties, every other test in the class passes vacuously.
        methods = _perception_methods()
        assert len(methods) >= 7, methods
        assert "find_hydrogens" in methods and "find_aromaticity" in methods

    def test_is_a_molrs_perceive(self):
        assert issubclass(Perceive, molrs.perceive.Perceive)
        assert mp.Perceive is Perceive

    @pytest.mark.parametrize("method", _perception_methods())
    def test_every_perception_method_is_overridden(self, method):
        assert method in vars(Perceive), (
            f"{method} is inherited unchanged, so it returns a bare molrs graph"
        )

    @pytest.mark.parametrize("method", _perception_methods())
    def test_every_perception_method_returns_a_molpy_graph(self, method):
        mol = mp.io.read_smiles("c1ccccc1CO")
        out = getattr(Perceive(), method)(mol)
        assert type(out) is Atomistic, f"{method} returned {type(out).__name__}"

    @pytest.mark.parametrize("method", _perception_methods())
    def test_every_perception_method_leaves_its_argument_alone(self, method):
        mol = mp.io.read_smiles("c1ccccc1CO")
        before = len(list(mol.atoms))
        getattr(Perceive(), method)(mol)
        assert len(list(mol.atoms)) == before

    def test_find_hydrogens_fills_open_valences(self):
        bare = mp.io.read_smiles("CCO")
        filled = Perceive().find_hydrogens(bare)
        assert len(list(bare.atoms)) == 3
        assert len(list(filled.atoms)) == 9  # ethanol, C2H6O

    def test_find_aromaticity_does_not_need_the_hydrogens_first(self):
        """Adding hydrogens is optional and changes no answer.

        A heavy-atom benzene *is* benzene: its hydrogens are implicit, not
        absent, and perception reads them off each carbon's valence. This test
        used to assert the opposite — that the bare skeleton perceives as
        non-aromatic — which pinned a limitation as if it were chemistry and
        made `find_aromaticity` silently depend on a structural edit.
        """

        def aromatic_count(graph):
            atoms = graph.to_frame()["atoms"]
            return sum(int(v) for v in atoms["is_aromatic"])

        bare = mp.io.read_smiles("c1ccccc1")
        filled = Perceive().find_hydrogens(bare)

        assert aromatic_count(Perceive().find_aromaticity(bare)) == 6
        assert aromatic_count(Perceive().find_aromaticity(filled)) == 6

    def test_find_aromaticity_leaves_no_fractional_bond_order(self):
        """Aromatic is a bond *type*; the localized number is always an integer."""
        std = Perceive().find_aromaticity(mp.io.read_smiles("c1ccccc1"))
        bonds = std.to_frame()["bonds"]

        assert list(bonds["bond_type"]) == [4] * 6
        assert sorted(bonds["bond_number"]) == [1, 1, 1, 2, 2, 2]
        assert "order" not in bonds.keys()
        assert "kekule_order" not in bonds.keys()


class TestAtomisticFromSmiles:
    def test_parses_a_single_molecule(self):
        assert len(list(mp.io.read_smiles("CCO").atoms)) == 3

    def test_returns_a_molpy_atomistic(self):
        assert type(mp.io.read_smiles("CCO")) is Atomistic

    def test_does_not_add_implicit_hydrogens(self):
        # A SMILES states connectivity; filling valences is a separate step.
        assert len(list(mp.io.read_smiles("C").atoms)) == 1

    def test_rejects_a_multi_component_smiles(self):
        with pytest.raises(ValueError, match="one component"):
            mp.io.read_smiles("[Li+].[F-]")

    def test_rejects_invalid_smiles(self):
        with pytest.raises(ValueError):
            mp.io.read_smiles("C(((")

    def test_components_are_the_documented_route_for_a_mixture(self):
        parts = [Atomistic.adopt(m) for m in mp.SmilesIR("[Li+].[F-]").components()]
        assert [len(list(p.atoms)) for p in parts] == [1, 1]
        assert all(type(p) is Atomistic for p in parts)


class TestRingInfo:
    def test_reports_rings_without_touching_the_graph(self):
        mol = mp.io.read_smiles("c1ccc2ccccc2c1")  # naphthalene
        info = mp.RingInfo(mol)
        assert info.num_rings() == 2
        assert info.ring_sizes() == [6, 6]

    def test_fuses_sharing_rings_into_one_system(self):
        info = mp.RingInfo(mp.io.read_smiles("c1ccc2ccccc2c1"))
        assert [len(system) for system in info.ring_systems()] == [10]

    def test_is_a_query_not_the_perceive_decorator(self):
        """The two ``find_rings`` spellings answer different questions."""
        mol = mp.io.read_smiles("c1ccccc1")
        assert isinstance(mp.RingInfo(mol).rings(), list)
        assert type(Perceive().find_rings(mol)) is Atomistic
