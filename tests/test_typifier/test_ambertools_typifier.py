"""AmberToolsTypifier: one ``match``, types only, no charges.

Charges are conserved by folding a cap's charge onto its site atom on the
template, never recomputed from a sliced fragment — so this typifier reads GAFF
atom types back and deliberately drops whatever charge antechamber wrote.

These exercise the pure-Python surface with a stub ``result``, so they need no
AmberTools executables.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import pytest

import molpy as mp
from molpy.typifier.affected_region import AffectedRegion
from molpy.typifier.ambertools import AmberToolsTypifier
from molpy.typifier.base import Typifier
from molpy.typifier.region import RegionTypes


class _EmptyForceField:
    """A force field that parameterises nothing — antechamber's GAFF is stubbed."""

    def get_types(self, _cls: type) -> list:
        return []

    def merge(self, _other: object) -> None: ...


def _linear_carbon_region(interior_reach: int = 1) -> AffectedRegion:
    """Ball around the middle carbon of a short C5 chain."""
    s = mp.Atomistic()
    atoms = [s.def_atom(element="C", x=1.5 * i, y=0.0, z=0.0) for i in range(5)]
    for a, b in zip(atoms, atoms[1:], strict=False):
        s.def_bond(a, b)
    return AffectedRegion._from(
        s, [atoms[2]], extract_radius=interior_reach + 1, interior_reach=interior_reach
    )


def _amber_returning(graph, *, charge: float | None = None) -> Mock:
    """A wrapper whose ``parameterize`` types every atom of ``graph`` as ``c3``."""
    frame = graph.to_frame()
    n = frame["atoms"].nrows
    frame["atoms"]["type"] = ["c3"] * n
    if charge is not None:
        frame["atoms"]["charge"] = [charge] * n
    amber = Mock()
    amber.parameterize.return_value = types.SimpleNamespace(
        frame=frame, ff=_EmptyForceField()
    )
    return amber


class TestContract:
    def test_it_is_a_typifier_that_only_implements_match(self):
        assert issubclass(AmberToolsTypifier, Typifier)
        assert "typify" not in AmberToolsTypifier.__dict__
        assert "match" in AmberToolsTypifier.__dict__

    def test_reach_is_not_its_business_any_more(self):
        """The radius belongs to whoever cuts the region, not to the typifier."""
        with pytest.raises(TypeError):
            AmberToolsTypifier(Mock(), reach=2)  # type: ignore[call-arg]

        assert AmberToolsTypifier(Mock()) is not None


class TestGasCharges:
    def test_antechamber_is_driven_with_gas_charges(self):
        """No ``sqm`` solve: atom types and bonded params are charge-independent."""
        molecule = _linear_carbon_region().complete_valence()
        amber = _amber_returning(molecule, charge=0.0)

        AmberToolsTypifier(amber).typify(molecule)

        _, kwargs = amber.parameterize.call_args
        assert kwargs["charge_method"] == "gas"

    def test_it_never_writes_a_charge_back(self):
        molecule = _linear_carbon_region().complete_valence()
        amber = _amber_returning(molecule, charge=0.5)

        typed = AmberToolsTypifier(amber).typify(molecule)

        assert all(atom["type"] == "c3" for atom in typed.atoms)
        assert all(atom.get("charge") is None for atom in typed.atoms)


class TestSnapshot:
    def test_the_region_snapshot_records_types_not_charges(self):
        region = _linear_carbon_region()
        amber = _amber_returning(region.complete_valence(), charge=0.5)

        snap = RegionTypes.of(region, AmberToolsTypifier(amber))

        assert snap.atoms, "interior atom(s) should be captured"
        for _pos, info in snap.atoms:
            assert info.type == "c3"
            assert info.params == ()  # the charge on the frame was ignored

    def test_the_region_completes_itself_before_antechamber_sees_it(self):
        """antechamber must never be handed a raw slice; the region caps it once."""
        region = _linear_carbon_region()
        capped = region.complete_valence()
        amber = _amber_returning(capped)

        RegionTypes.of(region, AmberToolsTypifier(amber))

        (given,), _ = amber.parameterize.call_args
        assert len(list(given.atoms)) == len(list(capped.atoms))
        assert len(list(given.atoms)) > len(list(region.atoms))
