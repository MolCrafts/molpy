"""Tests for incremental typification in Reacter.run().

Regression test for the bug where pair typification crashes on newly
created atoms that have not yet been atom-typed:

    ValueError: Atom must have 'type' attribute before pair typification

The root cause was that atom_typifier.typify() returns a NEW Atomistic
copy with types assigned, but _incremental_typify discarded the return
value, so the original assembly atoms never got their 'type' attribute.
"""

from typing import Any


from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    Reacter,
    find_port,
    form_single_bond,
    select_hydrogens,
    select_self,
)
from molpy.typifier.base import TypifierBase


class StubAtomTypifier(TypifierBase):
    """Atom typifier that assigns 'type' = symbol to every atom.

    Returns a NEW copy (matching the real atom typifier contract).
    """

    def __init__(self) -> None:
        # Bypass ForceField requirement for testing.
        self.ff = None  # type: ignore[assignment]
        self.strict = False

    def typify(self, elem: Atomistic) -> Atomistic:
        new_struct = elem.copy()
        for atom in new_struct.atoms:
            symbol = atom.get("symbol", "X")
            atom.data["type"] = f"type_{symbol}"
        return new_struct


class StubPairTypifier(TypifierBase):
    """Pair typifier that requires 'type' to be present.

    Mimics the real PairTypifier's strict check.
    """

    def __init__(self) -> None:
        self.ff = None  # type: ignore[assignment]
        self.strict = True

    def typify(self, elem: Any) -> Any:
        atom = elem
        atom_type = atom.get("type", None)
        if atom_type is None:
            raise ValueError(
                f"Atom must have 'type' attribute before pair typification: {atom}"
            )
        atom.data["pair_params"] = f"pair_{atom_type}"
        return atom


class StubCompositeTypifier(TypifierBase):
    """Composite typifier with atom_typifier and pair_typifier sub-typifiers."""

    def __init__(self) -> None:
        self.ff = None  # type: ignore[assignment]
        self.strict = False
        self.atom_typifier = StubAtomTypifier()
        self.pair_typifier = StubPairTypifier()
        # No bond/angle/dihedral typifiers for this minimal test.

    def typify(self, elem: Any) -> Any:
        raise NotImplementedError("Use sub-typifiers via Reacter._incremental_typify")


def _make_ch_struct(port_label: str) -> Atomistic:
    """Create a minimal C-H structure with a port marker on C."""
    struct = Atomistic()
    c = Atom(symbol="C")
    h = Atom(symbol="H")
    struct.add_entity(c, h)
    struct.add_link(Bond(c, h))
    c["port"] = port_label
    return struct


class TestIncrementalTypify:
    """Tests for Reacter._incremental_typify integration."""

    def test_incremental_typify_assigns_type_before_pair(self):
        """Atom typing must run BEFORE pair typing on modified atoms.

        This is the core regression test: without the fix, pair_typifier
        would crash with ValueError because atoms lack 'type'.
        """
        struct_L = _make_ch_struct("1")
        struct_R = _make_ch_struct("2")

        reacter = Reacter(
            name="C-C_coupling",
            anchor_selector_left=select_self,
            anchor_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )

        port_atom_L = find_port(struct_L, "1")
        port_atom_R = find_port(struct_R, "2")

        typifier = StubCompositeTypifier()

        # This used to raise:
        #   ValueError: Atom must have 'type' attribute before pair typification
        result = reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            typifier=typifier,
        )

        # All atoms in the product should have 'type' set
        product = result.product
        for atom in product.atoms:
            assert atom.get("type") is not None, (
                f"Atom {atom} should have 'type' after incremental typification"
            )

    def test_incremental_typify_pair_params_assigned(self):
        """Pair parameters should be assigned to modified atoms after typing."""
        struct_L = _make_ch_struct("1")
        struct_R = _make_ch_struct("2")

        reacter = Reacter(
            name="C-C_coupling",
            anchor_selector_left=select_self,
            anchor_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )

        port_atom_L = find_port(struct_L, "1")
        port_atom_R = find_port(struct_R, "2")

        typifier = StubCompositeTypifier()

        result = reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            typifier=typifier,
        )

        # Modified atoms (the two site atoms) should have pair params
        _ = result.product  # access to verify it exists
        site_L = result.anchor_L
        site_R = result.anchor_R

        assert site_L is not None
        assert site_R is not None
        assert site_L.get("pair_params") is not None, (
            "Left site atom should have pair_params after incremental typification"
        )
        assert site_R.get("pair_params") is not None, (
            "Right site atom should have pair_params after incremental typification"
        )

    def test_incremental_typify_does_not_crash_without_type(self):
        """Even if atom typifier fails to assign type, pair typifier should not crash.

        The fix adds a guard: pair typifier only runs on atoms that have 'type'.
        """
        struct_L = _make_ch_struct("1")
        struct_R = _make_ch_struct("2")

        reacter = Reacter(
            name="C-C_coupling",
            anchor_selector_left=select_self,
            anchor_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )

        port_atom_L = find_port(struct_L, "1")
        port_atom_R = find_port(struct_R, "2")

        # Create a typifier where atom_typifier deliberately does NOT
        # assign types (simulating a pattern match failure)
        class NoOpAtomTypifier(TypifierBase):
            def __init__(self) -> None:
                self.ff = None  # type: ignore[assignment]
                self.strict = False

            def typify(self, elem: Atomistic) -> Atomistic:
                return elem.copy()  # Returns copy without setting 'type'

        typifier = StubCompositeTypifier()
        typifier.atom_typifier = NoOpAtomTypifier()

        # Should NOT raise ValueError even though atoms lack 'type'
        result = reacter.run(
            struct_L,
            struct_R,
            port_atom_L=port_atom_L,
            port_atom_R=port_atom_R,
            typifier=typifier,
        )

        assert result.product is not None
