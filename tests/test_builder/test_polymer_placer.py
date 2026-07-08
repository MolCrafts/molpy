"""Unit tests for placer module: port utilities and Placer components."""

import numpy as np
import pytest

from molpy.builder.polymer.connectors import port_role, ports_compatible
from molpy.builder.polymer.core import get_ports_on_node
from molpy.builder.polymer.placer import (
    CovalentSeparator,
    LinearOrienter,
    Placer,
    VdWSeparator,
)
from molpy.core.atomistic import Atomistic


# ---- Port Role / Compatibility Tests ----


class TestPortRole:
    def test_left(self):
        assert port_role("<") == "left"
        assert port_role("<1") == "left"

    def test_right(self):
        assert port_role(">") == "right"
        assert port_role(">2") == "right"

    def test_terminal(self):
        assert port_role("$") == "terminal"
        assert port_role("$1") == "terminal"


class TestPortsCompatible:
    def test_right_left(self):
        assert ports_compatible(">", "<")
        assert ports_compatible("<", ">")

    def test_same_name(self):
        assert ports_compatible("$", "$")
        assert ports_compatible("$1", "$1")

    def test_incompatible(self):
        assert not ports_compatible(">", ">")
        assert not ports_compatible("<", "<")
        assert not ports_compatible("$1", "$2")


# ---- Port Utility Tests ----


class TestPortsOnNode:
    def test_get_ports_on_node(self):
        struct = Atomistic()
        struct.def_atom(symbol="C", port=">", monomer_node_id=0)
        struct.def_atom(symbol="C", port="<", monomer_node_id=1)

        node0_ports = get_ports_on_node(struct, 0)
        assert ">" in node0_ports
        assert "<" not in node0_ports

        node1_ports = get_ports_on_node(struct, 1)
        assert "<" in node1_ports
        assert ">" not in node1_ports

    def test_get_ports_on_node_empty(self):
        struct = Atomistic()
        struct.def_atom(symbol="C", monomer_node_id=0)
        assert get_ports_on_node(struct, 0) == {}


# ---- Separator Tests ----


class TestVdWSeparator:
    def test_basic_separation(self):
        sep = VdWSeparator(buffer=0.0)
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="C", x=1.0, y=0.0, z=0.0)

        dist = sep.get_separation(struct, struct, a, b)
        assert dist > 0

    def test_buffer(self):
        sep0 = VdWSeparator(buffer=0.0)
        sep1 = VdWSeparator(buffer=1.0)
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="C", x=1.0, y=0.0, z=0.0)

        d0 = sep0.get_separation(struct, struct, a, b)
        d1 = sep1.get_separation(struct, struct, a, b)
        assert d1 == pytest.approx(d0 + 1.0)


class TestCovalentSeparator:
    # Bond length = sum of Element covalent radii: C=0.76, O=0.66 Å.
    def test_cc_bond_length(self):
        sep = CovalentSeparator(buffer=0.0)
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="C", x=1.0, y=0.0, z=0.0)

        dist = sep.get_separation(struct, struct, a, b)
        assert dist == pytest.approx(1.52, abs=0.01)  # 0.76 + 0.76

    def test_co_bond_length(self):
        sep = CovalentSeparator(buffer=0.0)
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="O", x=1.0, y=0.0, z=0.0)

        dist = sep.get_separation(struct, struct, a, b)
        assert dist == pytest.approx(1.42, abs=0.01)  # 0.76 + 0.66

    def test_buffer(self):
        sep = CovalentSeparator(buffer=-0.1)
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="C", x=1.0, y=0.0, z=0.0)

        dist = sep.get_separation(struct, struct, a, b)
        assert dist == pytest.approx(1.52 - 0.1, abs=0.01)

    def test_unknown_elements_fall_back_to_carbon(self):
        sep = CovalentSeparator(buffer=0.0)
        struct = Atomistic()
        a = struct.def_atom(symbol="Xx", x=0.0, y=0.0, z=0.0)
        b = struct.def_atom(symbol="Zz", x=1.0, y=0.0, z=0.0)

        dist = sep.get_separation(struct, struct, a, b)
        assert dist == pytest.approx(1.52, abs=0.01)  # carbon fallback: 0.76 + 0.76


# ---- Orienter Tests ----


class TestLinearOrienter:
    def _make_struct_with_port(self, x, port_name):
        struct = Atomistic()
        a = struct.def_atom(symbol="C", x=x, y=0.0, z=0.0, port=port_name)
        b = struct.def_atom(symbol="C", x=x - 1.0, y=0.0, z=0.0)
        struct.def_bond(a, b, order=1)
        return struct, a

    def test_orientation_returns_correct_shapes(self):
        orienter = LinearOrienter()
        left, lp = self._make_struct_with_port(0.0, ">")
        right, rp = self._make_struct_with_port(5.0, "<")

        translation, rotation = orienter.get_orientation(left, right, lp, rp, 1.5)
        assert translation.shape == (3,)
        assert rotation.shape == (3, 3)

    def test_rotation_is_orthogonal(self):
        orienter = LinearOrienter()
        left, lp = self._make_struct_with_port(0.0, ">")
        right, rp = self._make_struct_with_port(5.0, "<")

        _, rotation = orienter.get_orientation(left, right, lp, rp, 1.5)
        identity = rotation @ rotation.T
        np.testing.assert_allclose(identity, np.eye(3), atol=1e-10)


# ---- Placer Integration Tests ----


class TestPlacer:
    def test_place_monomer_modifies_right(self):
        left = Atomistic()
        la = left.def_atom(symbol="C", x=0.0, y=0.0, z=0.0, port=">")
        left.def_atom(symbol="C", x=-1.0, y=0.0, z=0.0)
        left.def_bond(list(left.atoms)[0], list(left.atoms)[1], order=1)

        right = Atomistic()
        ra = right.def_atom(symbol="C", x=10.0, y=0.0, z=0.0, port="<")
        right.def_atom(symbol="C", x=11.0, y=0.0, z=0.0)
        right.def_bond(list(right.atoms)[0], list(right.atoms)[1], order=1)

        placer = Placer(
            separator=CovalentSeparator(buffer=0.0),
            orienter=LinearOrienter(),
        )

        orig_x = ra["x"]
        placer.place_monomer(left, right, la, ra)
        assert ra["x"] != orig_x


# ---- _apply_transform Equivalence Tests (builder-reacter-05-perf) ----


class TestApplyTransformEquivalence:
    """_apply_transform must equal the per-atom reference loop.

    Locks the vectorization target ``(coords - pivot) @ R.T + pivot + t``
    as numerically identical to the current per-atom implementation.
    """

    _COORDS = [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (1.5, 1.2, 0.0),
        (0.3, 0.4, 2.2),
        (-1.1, 0.7, 0.5),
    ]

    # 90 degrees about z
    _ROTATION = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    _TRANSLATION = np.array([0.5, -1.0, 2.0])

    def _make_struct(self) -> Atomistic:
        struct = Atomistic()
        for x, y, z in self._COORDS:
            struct.def_atom(symbol="C", x=x, y=y, z=z)
        return struct

    @staticmethod
    def _make_placer() -> Placer:
        return Placer(separator=CovalentSeparator(), orienter=LinearOrienter())

    @classmethod
    def _reference_positions(cls, pivot: np.ndarray) -> np.ndarray:
        """Per-atom reference: rotate about pivot, then translate."""
        expected = []
        for xyz in cls._COORDS:
            pos = np.asarray(xyz, dtype=float)
            expected.append(cls._ROTATION @ (pos - pivot) + pivot + cls._TRANSLATION)
        return np.array(expected)

    @staticmethod
    def _actual_positions(struct: Atomistic) -> np.ndarray:
        return np.array([[a["x"], a["y"], a["z"]] for a in struct.atoms])

    def test_transform_matches_reference_with_explicit_pivot(self) -> None:
        struct = self._make_struct()
        pivot = np.array([1.0, 2.0, 3.0])
        expected = self._reference_positions(pivot)

        self._make_placer()._apply_transform(
            struct, self._TRANSLATION, self._ROTATION, pivot=pivot
        )

        np.testing.assert_allclose(self._actual_positions(struct), expected, atol=1e-10)

    def test_transform_matches_reference_with_default_centroid_pivot(self) -> None:
        struct = self._make_struct()
        centroid = np.mean(np.array(self._COORDS, dtype=float), axis=0)
        expected = self._reference_positions(centroid)

        self._make_placer()._apply_transform(struct, self._TRANSLATION, self._ROTATION)

        np.testing.assert_allclose(self._actual_positions(struct), expected, atol=1e-10)
