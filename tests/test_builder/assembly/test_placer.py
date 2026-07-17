"""Unit tests for :mod:`molpy.builder.assembly._placer`."""

import inspect

import numpy as np
import pytest

from molpy.builder.assembly import GraphAssembler, Placer, ResiduePlacer
from molpy.core import fields
from molrs import Element


class TestPlacer:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            Placer()

    def test_is_injected_into_the_assembler_constructor(self):
        assert "placer" in inspect.signature(GraphAssembler.__init__).parameters


class TestResiduePlacer:
    def test_spreads_overlapping_residue_templates(self, builder_factory):
        stacked = builder_factory().build("{[#EO]|5}")
        spread = builder_factory(placer=ResiduePlacer()).build("{[#EO]|5}")

        assert self._minimum_separation(stacked) < 1e-9
        assert self._minimum_separation(spread) > 0.5

    def test_initial_bond_length_is_the_sum_of_covalent_radii(self, builder_factory):
        chain = builder_factory(placer=ResiduePlacer()).build("{[#EO]|4}")
        lengths = [
            float(
                np.linalg.norm(
                    np.array([bond.itom["x"], bond.itom["y"], bond.itom["z"]])
                    - np.array([bond.jtom["x"], bond.jtom["y"], bond.jtom["z"]])
                )
            )
            for bond in chain.bonds
        ]
        expected = Element("C").covalent + Element("O").covalent
        assert max(lengths) == pytest.approx(expected, abs=1e-6)

    def test_alignment_is_always_a_proper_rotation(self):
        rng = np.random.default_rng(0)
        cases = [rng.normal(size=(2, 3)) for _ in range(30)]
        cases.extend(
            [
                np.array([[1.0, 0, 0], [-1.0, 0, 0]]),
                np.array([[0, 0, 1.0], [0, 0, -1.0]]),
                np.array([[1.0, 0, 0], [1.0, 0, 0]]),
            ]
        )
        for source, target in cases:
            source = source / np.linalg.norm(source)
            target = target / np.linalg.norm(target)
            rotation = ResiduePlacer._align(source, target)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
            assert rotation @ source == pytest.approx(target, abs=1e-9)

    def test_unknown_element_is_not_guessed(self, eo_factory):
        atom = eo_factory().atoms[0]
        del atom[fields.ELEMENT]
        with pytest.raises(KeyError, match="covalent radius"):
            ResiduePlacer._radius(atom)

    @staticmethod
    def _minimum_separation(graph) -> float:
        positions = np.array([[a["x"], a["y"], a["z"]] for a in graph.atoms])
        distances = (
            np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
            + np.eye(len(positions)) * 1e9
        )
        return float(distances.min())
