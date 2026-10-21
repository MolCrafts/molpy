"""``molrs.core.BondOrder`` / ``BondNumber``: the stored bond codes by name."""

from __future__ import annotations

import molrs
from molrs.core import BondNumber, BondOrder


def test_codes_are_the_stored_ones() -> None:
    assert [
        int(o)
        for o in (
            BondOrder.Unknown,
            BondOrder.Single,
            BondOrder.Double,
            BondOrder.Triple,
            BondOrder.Aromatic,
        )
    ] == [0, 1, 2, 3, 4]
    assert BondOrder.Aromatic.code == 4
    assert BondNumber.Quadruple.code == 4


def test_from_code_reads_unknown_codes_as_unknown() -> None:
    assert BondOrder.from_code(4) == BondOrder.Aromatic
    assert BondOrder.from_code(9) == BondOrder.Unknown
    assert BondNumber.from_code(2) == BondNumber.Double


def test_aromatic_implies_no_number() -> None:
    assert BondOrder.Aromatic.is_aromatic()
    assert BondOrder.Aromatic.implied_number() is None
    assert BondOrder.Double.implied_number() == BondNumber.Double


def test_a_number_implies_its_class_and_counts_its_valence() -> None:
    assert BondNumber.Single.implied_type() == BondOrder.Single
    assert BondNumber.Quadruple.implied_type() == BondOrder.Double
    assert BondNumber.Unknown.implied_type() == BondOrder.Unknown
    assert BondNumber.Triple.count() == 3
    assert BondNumber.Unknown.count() == 0


def test_set_bond_class_takes_the_classes() -> None:
    mol = molrs.io.smiles.read_smiles("CC").to_atomistic()
    bond = next(iter(mol.bonds))
    mol.set_bond_class(bond, BondOrder.Double, BondNumber.Double)
    assert BondOrder.from_code(int(bond["bond_type"])) == BondOrder.Double
    assert BondNumber.from_code(int(bond["bond_number"])) == BondNumber.Double
