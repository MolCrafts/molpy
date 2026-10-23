"""``mp.ff.compile.ExplicitTerms``: any style's kernel over explicit instances.

One builder for every style the force-field IR prices — the built-ins, a
style registered from Python by expression or by a numpy kernel, a style of a
custom category, an unregistered style given its expression. Each term is its
atoms and its own parameter row, as stored (angle values in degrees). The
tests prove each prices its LAMMPS energy, ``Potentials.push`` moves it, and
a collection assembled by hand prices a molecule exactly as the compiled
force field does.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import molpy as mp
import numpy as np
import pytest
from molpy.ff.potential import Potentials
from molpy.ff import ir, style_registry
from molpy.ff.compile import ExplicitTerms

# A non-planar four-atom chain: every angle and the dihedral are generic.
XYZ = np.array(
    [[1.2, -0.4, 0.3], [0.0, 0.0, 0.0], [-0.2, 1.5, 0.1], [0.9, 2.1, -0.8]],
    dtype=np.float64,
)
FLAT = XYZ.reshape(-1)


@pytest.fixture
def registered() -> Iterator[list[tuple[str, str]]]:
    """Styles a test registers, unregistered afterwards."""
    names: list[tuple[str, str]] = []
    yield names
    for category, name in names:
        if any(s.name == name for s in style_registry.styles(category)):
            style_registry.unregister_style(category, name)


def _dist(a: int, b: int) -> float:
    return float(np.linalg.norm(XYZ[a] - XYZ[b]))


def _angle(a: int, b: int, c: int) -> float:
    u, v = XYZ[a] - XYZ[b], XYZ[c] - XYZ[b]
    return math.acos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def _dihedral(a: int, b: int, c: int, d: int) -> float:
    b1, b2, b3 = XYZ[b] - XYZ[a], XYZ[c] - XYZ[b], XYZ[d] - XYZ[c]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return math.atan2(m1 @ n2, n1 @ n2)


def _energy(pots: Potentials) -> float:
    return pots.evaluate(FLAT)[0]


def _fd_forces(make, h: float = 1e-6) -> np.ndarray:
    """Central-difference ``-dE/dx``."""
    pots = make()
    out = np.zeros_like(FLAT)
    for i in range(FLAT.size):
        plus, minus = FLAT.copy(), FLAT.copy()
        plus[i] += h
        minus[i] -= h
        out[i] = -(pots.evaluate(plus)[0] - pots.evaluate(minus)[0]) / (2 * h)
    return out.reshape(-1, 3)


class TestBuiltins:
    def test_bond_harmonic_prices_one_row_per_term(self) -> None:
        def make() -> Potentials:
            return ExplicitTerms(
                "bond", "harmonic", [[0, 1], [1, 2]], k=[300.0, 200.0], r0=1.4
            ).compile()

        want = 300.0 * (_dist(0, 1) - 1.4) ** 2 + 200.0 * (_dist(1, 2) - 1.4) ** 2
        e, f = make().evaluate(FLAT)
        assert math.isclose(e, want, rel_tol=1e-12)
        np.testing.assert_allclose(f, _fd_forces(make), atol=1e-5)

    def test_angle_harmonic_takes_theta0_in_degrees(self) -> None:
        pots = ExplicitTerms(
            "angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5
        ).compile()
        want = 50.0 * (_angle(0, 1, 2) - math.radians(109.5)) ** 2
        assert math.isclose(_energy(pots), want, rel_tol=1e-12)

    def test_dihedral_periodic_sums_its_indexed_series(self) -> None:
        k, n, phase = (1.3, 0.4), (1.0, 2.0), (0.0, 180.0)

        def make() -> Potentials:
            return ExplicitTerms(
                "dihedral",
                "periodic",
                [[0, 1, 2, 3]],
                k1=k[0],
                periodicity1=n[0],
                phase1=phase[0],
                k2=k[1],
                periodicity2=n[1],
                phase2=phase[1],
            ).compile()

        phi = _dihedral(0, 1, 2, 3)
        want = sum(
            k[t] * (1 + math.cos(n[t] * phi - math.radians(phase[t]))) for t in range(2)
        )
        e, f = make().evaluate(FLAT)
        assert math.isclose(e, want, rel_tol=1e-12)
        np.testing.assert_allclose(f, _fd_forces(make), atol=1e-5)

    def test_improper_cvff_prices_the_dihedral_of_the_listed_order(self) -> None:
        pots = ExplicitTerms(
            "improper", "cvff", [[1, 0, 2, 3]], k=2.0, sign=-1.0, periodicity=2.0
        ).compile()
        phi = _dihedral(1, 0, 2, 3)
        assert math.isclose(_energy(pots), 2.0 * (1 - math.cos(2 * phi)), rel_tol=1e-12)

    def test_improper_periodic_takes_its_phase_in_degrees(self) -> None:
        pots = ExplicitTerms(
            "improper", "periodic", [[0, 2, 1, 3]], k=1.1, periodicity=2.0, phase=180.0
        ).compile()
        phi = _dihedral(0, 2, 1, 3)
        assert math.isclose(
            _energy(pots), 1.1 * (1 + math.cos(2 * phi - math.pi)), rel_tol=1e-12
        )

    def test_a_pair_term_is_priced_with_its_own_row(self) -> None:
        pots = ExplicitTerms(
            "pair", "lj/cut", [[0, 3], [1, 3]], epsilon=[0.2, 0.1], sigma=3.1
        ).compile()
        want = sum(
            4 * eps * ((3.1 / _dist(i, 3)) ** 12 - (3.1 / _dist(i, 3)) ** 6)
            for i, eps in ((0, 0.2), (1, 0.1))
        )
        assert math.isclose(_energy(pots), want, rel_tol=1e-12)

    def test_coul_cut_reads_per_atom_charges(self) -> None:
        q = [0.3, 0.0, 0.0, -0.4]
        pots = ExplicitTerms(
            "pair", "coul/cut", [[0, 3]], charges=q, coulomb=332.06371, dielectric=1.0
        ).compile()
        want = 332.06371 * 0.3 * -0.4 / _dist(0, 3)
        assert math.isclose(_energy(pots), want, rel_tol=1e-12)


class TestCustomStyles:
    def test_a_style_registered_by_expression(self, registered) -> None:
        class Quartic(style_registry.StyleDeclaration):
            category = "bond"
            name = "quartic/kernel-test"
            params = {"k": "E/L^4", "r0": "L"}
            expression = "k*(r-r0)^4"

        registered.append(("bond", Quartic.name))
        pots = ExplicitTerms("bond", Quartic.name, [[0, 1]], k=2.0, r0=1.0).compile()
        assert math.isclose(
            _energy(pots), 2.0 * (_dist(0, 1) - 1.0) ** 4, rel_tol=1e-12
        )

    def test_a_style_priced_by_a_numpy_kernel(self, registered) -> None:
        def cubic(r, k, r0):
            d = r - r0
            return k * d**3, 3 * k * d**2

        style_registry.register_style(
            "bond", "cubic/kernel-test", params={"k": "E/L^3", "r0": "L"}, kernel=cubic
        )
        registered.append(("bond", "cubic/kernel-test"))
        pots = ExplicitTerms(
            "bond", "cubic/kernel-test", [[0, 1]], k=3.0, r0=0.5
        ).compile()
        assert math.isclose(
            _energy(pots), 3.0 * (_dist(0, 1) - 0.5) ** 3, rel_tol=1e-12
        )

    def test_a_style_of_a_custom_category(self, registered) -> None:
        style_registry.register_category("urey_bradley", 3)
        style_registry.register_style(
            "urey_bradley",
            "kernel-test",
            params={"k_ub": "E/L^2", "r_ub": "L"},
            expression="k_ub*(distance(p1,p3)-r_ub)^2",
        )
        registered.append(("urey_bradley", "kernel-test"))
        pots = ExplicitTerms(
            "urey_bradley", "kernel-test", [[0, 1, 2]], k_ub=20.0, r_ub=2.4
        ).compile()
        assert math.isclose(
            _energy(pots), 20.0 * (_dist(0, 2) - 2.4) ** 2, rel_tol=1e-12
        )

    def test_an_unregistered_style_by_its_expression(self) -> None:
        pots = ExplicitTerms(
            "bond", "unregistered/kernel-test", [[0, 1]], expression="k*r^2", k=0.5
        ).compile()
        assert math.isclose(_energy(pots), 0.5 * _dist(0, 1) ** 2, rel_tol=1e-12)


class TestRefusals:
    def test_unknown_category(self) -> None:
        with pytest.raises(ir.UnknownCategoryError):
            ExplicitTerms("nope", "x", [[0, 1]]).compile()

    def test_a_term_of_the_wrong_arity(self) -> None:
        with pytest.raises(ir.ArityError):
            ExplicitTerms("bond", "harmonic", [[0, 1, 2]], k=1.0, r0=1.0).compile()

    def test_an_undeclared_parameter(self) -> None:
        with pytest.raises(TypeError, match="no parameter `kb`"):
            ExplicitTerms("bond", "harmonic", [[0, 1]], kb=1.0, r0=1.0).compile()

    def test_a_missing_parameter(self) -> None:
        with pytest.raises(ir.MissingParamError) as err:
            ExplicitTerms("bond", "harmonic", [[0, 1]], k=1.0).compile()
        assert (err.value.style, err.value.type, err.value.param) == (
            "harmonic",
            "0",
            "r0",
        )

    @pytest.mark.parametrize(
        ("category", "style", "atoms", "params", "param"),
        [
            ("angle", "charmm", [[0, 1, 2]], {"k": 1.0, "theta0": 100.0}, "k_ub"),
            ("dihedral", "charmm", [[0, 1, 2, 3]], {"k": 1.0}, "periodicity"),
            ("dihedral", "periodic", [[0, 1, 2, 3]], {"k1": 1.0}, "periodicity1"),
            ("improper", "cvff", [[0, 1, 2, 3]], {"k": 1.0}, "sign"),
            ("pair", "coul/cut", [[0, 1]], {}, "coulomb"),
            ("pair", "lj/charmm", [[0, 1]], {"cutoff": 10.0}, "inner"),
        ],
    )
    def test_every_built_in_refuses_a_missing_parameter_typed(
        self,
        category: str,
        style: str,
        atoms: list[list[int]],
        params: dict,
        param: str,
    ) -> None:
        charges = [0.5, -0.5] if category == "pair" else None
        if style == "lj/charmm":
            params = {**params, "epsilon": 0.1, "sigma": 3.0}
        with pytest.raises(ir.MissingParamError) as err:
            ExplicitTerms(category, style, atoms, charges=charges, **params).compile()
        assert (err.value.style, err.value.param) == (style, param)

    def test_a_value_outside_its_choices(self) -> None:
        with pytest.raises(ir.BadValueError) as err:
            ExplicitTerms(
                "pair", "lj/cut", [[0, 1]], mixing="lorentz", epsilon=0.1, sigma=3.0
            ).compile()
        assert (err.value.style, err.value.param) == ("lj/cut", "mixing")

    def test_an_unregistered_style_without_expression(self) -> None:
        with pytest.raises(ir.NoKernelError):
            ExplicitTerms("bond", "nothing/kernel-test", [[0, 1]], k=1.0).compile()

    def test_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="2 terms"):
            ExplicitTerms(
                "bond", "harmonic", [[0, 1], [1, 2]], k=[1.0, 2.0, 3.0], r0=1.0
            ).compile()


class TestPotentialsAssembly:
    def test_push_moves_the_kernel(self) -> None:
        bond = ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile()
        pots = Potentials()
        pots.push(bond)
        assert len(pots) == 1
        with pytest.raises(ValueError, match="moved"):
            pots.push(bond)

    def test_a_hand_assembled_collection_prices_what_the_compiled_field_prices(
        self,
    ) -> None:
        ff = mp.ff.forcefield.ForceField("chain", units="real")
        atoms = ff.def_style("atom", "full")
        c = atoms.def_type("C", mass=12.011)
        ff.def_style("bond", "harmonic").def_type("CC", c, c, k=300.0, r0=1.4)
        ff.def_style("angle", "harmonic").def_type("CCC", c, c, c, k=50.0, theta0=109.5)
        torsion = dict(
            k1=1.3, periodicity1=1.0, phase1=0.0, k2=0.4, periodicity2=2.0, phase2=180.0
        )
        ff.def_style("dihedral", "periodic").def_type("CCCC", c, c, c, c, **torsion)
        frame = mp.core.Frame()
        block = mp.core.Block()
        for d, key in enumerate("xyz"):
            block.insert(key, XYZ[:, d].copy())
        block.insert("type", ["C"] * 4)
        frame["atoms"] = block
        bonds, angles, dihedrals = (
            [[0, 1], [1, 2], [2, 3]],
            [[0, 1, 2], [1, 2, 3]],
            [[0, 1, 2, 3]],
        )
        for name, rows, label in (
            ("bonds", bonds, "CC"),
            ("angles", angles, "CCC"),
            ("dihedrals", dihedrals, "CCCC"),
        ):
            topo = mp.core.Block()
            for column, atom in zip(("atomi", "atomj", "atomk", "atoml"), zip(*rows)):
                topo.insert(column, np.array(atom, dtype=np.uint32))
            topo.insert("type", [label] * len(rows))
            frame[name] = topo
        compiled = mp.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(FLAT)

        pots = Potentials()
        pots.push(ExplicitTerms("bond", "harmonic", bonds, k=300.0, r0=1.4).compile())
        pots.push(
            ExplicitTerms("angle", "harmonic", angles, k=50.0, theta0=109.5).compile()
        )
        pots.push(ExplicitTerms("dihedral", "periodic", dihedrals, **torsion).compile())
        by_hand = pots.evaluate(FLAT)
        assert math.isclose(by_hand[0], compiled[0], rel_tol=1e-12)
        np.testing.assert_allclose(by_hand[1], compiled[1], rtol=1e-12, atol=1e-12)


class TestDefaults:
    """A default the spec declares prices an absent parameter exactly as
    stating it does: no kernel states one of its own."""

    def test_coul_cut_dielectric_is_declared_not_defaulted(self) -> None:
        coords = np.array([0.0, 0.0, 0.0, 2.5, 0.0, 0.0])
        with pytest.raises(ValueError, match="dielectric"):
            ExplicitTerms(
                "pair", "coul/cut", [[0, 1]], charges=[0.5, -0.4], coulomb=332.06371
            ).compile()
        stated = ExplicitTerms(
            "pair",
            "coul/cut",
            [[0, 1]],
            charges=[0.5, -0.4],
            coulomb=332.06371,
            dielectric=2.0,
        ).compile()
        e = stated.evaluate(coords)[0]
        assert math.isclose(e, 332.06371 / 2.0 * 0.5 * -0.4 / 2.5, rel_tol=1e-12)

    def test_a_phase_defaults_to_0(self) -> None:
        bare = ExplicitTerms(
            "dihedral", "charmm", [[0, 1, 2, 3]], k=1.3, periodicity=3.0
        ).compile()
        stated = ExplicitTerms(
            "dihedral",
            "charmm",
            [[0, 1, 2, 3]],
            k=1.3,
            periodicity=3.0,
            phase=0.0,
            w=0.0,
        ).compile()
        assert _energy(bare) == _energy(stated)
        assert _energy(bare) != 0.0

    def test_lj_cut_exponents_and_shift_default_to_12_6_no(self) -> None:
        params = {"epsilon": 0.2, "sigma": 1.1}
        bare = ExplicitTerms("pair", "lj/cut", [[0, 1]], **params).compile()
        stated = ExplicitTerms(
            "pair", "lj/cut", [[0, 1]], n=12.0, m=6.0, shift=0.0, **params
        ).compile()
        assert _energy(bare) == _energy(stated)


class TestEvaluate:
    """``evaluate`` is the one door of ``Potential::evaluate``: energy and
    forces together, from a flat coordinate vector or a Frame."""

    def test_energy_and_forces_come_together(self) -> None:
        pots = ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile()
        e, f = pots.evaluate(FLAT)
        assert f.shape == (4, 3)
        assert math.isclose(e, 300.0 * (_dist(0, 1) - 1.4) ** 2, rel_tol=1e-12)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)

    def test_a_coordinate_vector_must_be_flat_3n(self) -> None:
        pots = ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile()
        with pytest.raises(ValueError, match="length 3N"):
            pots.evaluate(FLAT[:-1])

    def test_a_duck_typed_member_is_evaluated_through_its_evaluate(self) -> None:
        class Spring:
            def evaluate(self, pos):
                return 0.5 * float((pos * pos).sum()), -pos

        assert isinstance(Spring(), mp.ff.potential.Potential)
        pots = Potentials()
        pots.push(Spring())
        pots.push(
            ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4).compile()
        )
        e, f = pots.evaluate(FLAT)
        bond_e, bond_f = (
            ExplicitTerms("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4)
            .compile()
            .evaluate(FLAT)
        )
        assert math.isclose(e, bond_e + 0.5 * float((XYZ * XYZ).sum()), rel_tol=1e-12)
        np.testing.assert_allclose(f, bond_f - XYZ, rtol=1e-12, atol=1e-12)

    def test_a_python_member_exception_is_re_raised(self) -> None:
        class Broken:
            def evaluate(self, pos):
                raise RuntimeError("boom")

        pots = Potentials()
        pots.push(Broken())
        with pytest.raises(RuntimeError, match="boom"):
            pots.evaluate(FLAT)

    def test_push_refuses_an_object_without_evaluate(self) -> None:
        with pytest.raises(TypeError, match="evaluate"):
            Potentials().push(object())


class TestPairLjCut:
    """``pair_energy_force`` for one pair, ``evaluate_pairs`` over a table."""

    @staticmethod
    def _lj(r: float) -> tuple[float, float]:
        """Shifted 12-6 LJ (eps 1, sigma 1, cutoff 2.5): energy and dE/dr."""
        e = lambda x: 4 * (x**-12 - x**-6)  # noqa: E731
        de = -4 * (12 * r**-13 - 6 * r**-7)
        return e(r) - e(2.5), de

    def test_pair_energy_force_is_the_shifted_kernel(self) -> None:
        lj = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5)
        e, f = lj.pair_energy_force(1.21, [1.1, 0.0, 0.0])
        want_e, de = self._lj(1.1)
        assert math.isclose(e, want_e, rel_tol=1e-12)
        assert math.isclose(abs(f[0]), abs(de), rel_tol=1e-12)
        assert lj.pair_energy_force(9.0, [3.0, 0.0, 0.0]) is None

    def test_evaluate_pairs_sums_the_table(self) -> None:
        pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])
        box = mp.core.Box(mp.core.Cell.lengths([20.0] * 3))
        table = mp.core.NeighborQuery(box, pos, 2.5).query_self()
        lj = mp.ff.potential.PairLjCut(1.0, 1.0, 2.5)
        e, f = lj.evaluate_pairs(pos, table)
        r12 = float(np.linalg.norm(pos[1] - pos[2]))
        want = sum(self._lj(r)[0] for r in (1.1, 1.3, r12))
        assert math.isclose(e, want, rel_tol=1e-12)
        assert f.shape == (3, 3)
        np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-10)
        # A weight per pair; zero skips the pair.
        zero, _ = lj.evaluate_pairs(pos, table, np.zeros(table.n_pairs))
        assert zero == 0.0
        with pytest.raises(ValueError, match="weights for"):
            lj.evaluate_pairs(pos, table, np.ones(table.n_pairs + 1))

    def test_evaluate_pairs_refuses_a_table_past_pos(self) -> None:
        pos = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        box = mp.core.Box(mp.core.Cell.lengths([20.0] * 3))
        table = mp.core.NeighborQuery(box, pos, 2.5).query_self()
        with pytest.raises(ValueError, match="past the 1 rows"):
            mp.ff.potential.PairLjCut(1.0, 1.0, 2.5).evaluate_pairs(pos[:1], table)
