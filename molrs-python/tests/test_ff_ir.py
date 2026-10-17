"""``molrs.ff.ir``: the force-field IR as a protocol, from Python.

A style registered here — by expression, by a numpy kernel, or as a
``StyleDeclaration`` class — prices at compile exactly like a built-in, with nothing
in molrs rebuilt; what does not conform is refused by the ``IrError``
subclass named after the Rust variant, naming the item.

The registry is process-wide, so every test registers under its own names and
takes them out again (``registered``).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import molrs
import numpy as np
import pytest
from molrs.ff import ir, style_registry

# LAMMPS `bond_style fene` (Kremer-Grest): k, R0, epsilon, sigma.
K, R0, EPS, SIG = 30.0, 1.5, 1.0, 1.0
FENE = (
    "-0.5*k*r0^2*log(1-(r/r0)^2)"
    "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)"
)
FENE_PARAMS = [
    ir.ParamSpec("k", "E/L^2"),
    ir.ParamSpec("r0", "L"),
    ir.ParamSpec("epsilon", "E"),
    ir.ParamSpec("sigma", "L"),
]
# A bead chain: bonds of 0.97, 1.06, 1.18 and 1.31 (both sides of the WCA
# cutoff 2^(1/6) = 1.1225; (r/R0)^2 < 0.9 everywhere).
CHAIN = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.97, 0.0, 0.0],
        [0.97, 1.06, 0.0],
        [0.97, 1.06, 1.18],
        [0.97 + 1.31 * 0.6, 1.06 + 1.31 * 0.8, 1.18],
    ]
)


@pytest.fixture
def registered() -> Iterator[list[tuple[str, str]]]:
    """Styles a test registers, unregistered afterwards."""
    names: list[tuple[str, str]] = []
    yield names
    for category, name in names:
        try:
            style_registry.unregister_style(category, name)
        except ir.IrError:
            pass


def fene_energy(r: np.ndarray) -> np.ndarray:
    x = (r / R0) ** 2
    wca = np.where(
        r < 2 ** (1 / 6) * SIG,
        4 * EPS * ((SIG / r) ** 12 - (SIG / r) ** 6) + EPS,
        0.0,
    )
    return -0.5 * K * R0**2 * np.log(1 - x) + wca


def fene_derivative(r: np.ndarray) -> np.ndarray:
    dwca = np.where(
        r < 2 ** (1 / 6) * SIG,
        4 * EPS * (-12 * SIG**12 / r**13 + 6 * SIG**6 / r**7),
        0.0,
    )
    return K * r / (1 - (r / R0) ** 2) + dwca


def fene_kernel(r, k, r0, epsilon, sigma):
    """FENE in numpy: one call per evaluation, every term at once."""
    x = (r / r0) ** 2
    rc = 2 ** (1 / 6) * sigma
    inner = r < rc
    s6 = (sigma / r) ** 6
    e = -0.5 * k * r0**2 * np.log(1 - x) + np.where(
        inner, 4 * epsilon * (s6 * s6 - s6) + epsilon, 0.0
    )
    de = k * r / (1 - x) + np.where(
        inner, 4 * epsilon * (-12 * s6 * s6 + 6 * s6) / r, 0.0
    )
    return e, de


def chain_bonds() -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(len(CHAIN) - 1)
    return i, i + 1


def bond_frame(xyz: np.ndarray = CHAIN) -> molrs.core.Frame:
    atoms = molrs.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, xyz[:, d].copy())
    atoms.insert("type", ["B"] * len(xyz))
    i, j = chain_bonds()
    bonds = molrs.core.Block()
    bonds.insert("atomi", i.astype(np.uint32))
    bonds.insert("atomj", j.astype(np.uint32))
    bonds.insert("type", ["B-B"] * len(i))
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    return frame


def bond_ff(style: str, **params: float) -> molrs.ff.forcefield.ForceField:
    ff = molrs.ff.forcefield.ForceField("beads", units="lj")
    b = ff.def_style("atom", "full").def_type("B", mass=1.0)
    ff.def_style("bond", style).def_type("B-B", b, b, **params)
    return ff


def fene_ff(style: str) -> molrs.ff.forcefield.ForceField:
    return bond_ff(style, k=K, r0=R0, epsilon=EPS, sigma=SIG)


def energy_forces(ff: molrs.ff.forcefield.ForceField, frame: molrs.core.Frame):
    pots = molrs.ff.compile.PotentialCompiler(ff).compile(frame)
    return pots.evaluate(frame)[0], pots.evaluate(frame)[1]


def analytic_chain() -> tuple[float, np.ndarray]:
    i, j = chain_bonds()
    d = CHAIN[j] - CHAIN[i]
    r = np.linalg.norm(d, axis=1)
    f = np.zeros_like(CHAIN)
    pull = (fene_derivative(r) / r)[:, None] * d
    np.add.at(f, i, pull)
    np.add.at(f, j, -pull)
    return float(fene_energy(r).sum()), f


def register_fene(registered, name: str = "fene/expr", **kw) -> None:
    style_registry.register_style(
        "bond", name, params=FENE_PARAMS, expression=FENE, **kw
    )
    registered.append(("bond", name))


# ---------------------------------------------------------------------------
# A new style by expression, by numpy, as a class
# ---------------------------------------------------------------------------


def test_fene_by_expression_is_the_analytic_formula(registered) -> None:
    register_fene(registered)
    e, f = energy_forces(fene_ff("fene/expr"), bond_frame())
    e_ref, f_ref = analytic_chain()
    assert math.isclose(e, e_ref, rel_tol=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=0, atol=1e-10 * np.abs(f_ref).max())


def test_a_numpy_kernel_prices_as_the_expression(registered) -> None:
    register_fene(registered)
    calls = []

    def counted(r, **p):
        calls.append(len(r))
        return fene_kernel(r, **p)

    style_registry.register_style("bond", "fene/np", params=FENE_PARAMS, kernel=counted)
    registered.append(("bond", "fene/np"))
    frame = bond_frame()
    e_x, f_x = energy_forces(fene_ff("fene/expr"), frame)
    calls.clear()
    pots = molrs.ff.compile.PotentialCompiler(fene_ff("fene/np")).compile(frame)
    calls.clear()
    e_np, f_np = pots.evaluate(frame)
    # One Python call per style per evaluation, every term in it.
    assert calls == [4]
    assert math.isclose(e_np, e_x, rel_tol=1e-12)
    np.testing.assert_allclose(f_np, f_x, rtol=0, atol=1e-10 * np.abs(f_x).max())


def test_a_kernel_beside_its_expression_must_agree(registered) -> None:
    sample = {"q": (0.8, 1.4), "k": K, "r0": R0, "epsilon": EPS, "sigma": SIG}
    style_registry.register_style(
        "bond",
        "fene/both",
        params=FENE_PARAMS,
        expression=FENE,
        kernel=fene_kernel,
        samples=[sample],
    )
    registered.append(("bond", "fene/both"))

    def off(r, k, r0, epsilon, sigma):
        e, de = fene_kernel(r, k, r0, epsilon, sigma)
        return e * (1 + 1e-6), de * (1 + 1e-6)

    with pytest.raises(ir.DisagreeError) as err:
        style_registry.register_style(
            "bond",
            "fene/off",
            params=FENE_PARAMS,
            expression=FENE,
            kernel=off,
            samples=[sample],
        )
    assert err.value.style == "fene/off"
    assert "fene/off" not in {s.name for s in style_registry.styles("bond")}


def test_a_wrong_derivative_is_refused_at_registration_and_at_compile(
    registered,
) -> None:
    def wrong(r, k, r0):
        return k * (r - r0) ** 2, k * (r - r0)  # dE/dr is 2k(r - r0)

    with pytest.raises(ir.DerivativeError) as err:
        style_registry.register_style(
            "bond",
            "half",
            params={"k": "E/L^2", "r0": "L"},
            kernel=wrong,
            samples=[{"q": (0.8, 1.4), "k": 300.0, "r0": 1.0}],
        )
    assert err.value.style == "half"
    # Without samples the check runs at the style's first compile.
    style_registry.register_style(
        "bond", "half", params={"k": "E/L^2", "r0": "L"}, kernel=wrong
    )
    registered.append(("bond", "half"))
    with pytest.raises(ir.DerivativeError, match="half"):
        energy_forces(bond_ff("half", k=300.0, r0=1.0), bond_frame())


def test_a_style_spec_class_registers_on_definition(registered) -> None:
    class Fene(style_registry.StyleDeclaration):
        category = "bond"
        name = "fene/class"
        params = FENE_PARAMS
        expression = FENE
        samples = ({"q": (0.8, 1.4), "k": K, "r0": R0, "epsilon": EPS, "sigma": SIG},)

        def kernel(self, r, **p):
            return fene_kernel(r, **p)

    registered.append(("bond", "fene/class"))
    (info,) = [s for s in style_registry.styles("bond") if s.name == "fene/class"]
    assert info.kernel == "scalar"
    assert info.expression == FENE
    e, _ = energy_forces(fene_ff("fene/class"), bond_frame())
    assert math.isclose(e, analytic_chain()[0], rel_tol=1e-12)
    r = np.array([0.9, 1.2])
    e, de = Fene.evaluate(r, k=K, r0=R0, epsilon=EPS, sigma=SIG)
    np.testing.assert_allclose(e, fene_energy(r), rtol=1e-12)
    np.testing.assert_allclose(de, fene_derivative(r), rtol=1e-12)


def test_a_style_spec_without_its_name_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="name"):

        class Nameless(style_registry.StyleDeclaration):
            category = "bond"
            expression = "r"


def test_a_style_spec_base_can_opt_out_of_registering(registered) -> None:
    class Harmonicish(style_registry.StyleDeclaration, register=False):
        category = "bond"
        params = (ir.ParamSpec("k", "E/L^2"), ir.ParamSpec("r0", "L"))

    class Quad(Harmonicish):
        name = "quad"
        expression = "k*(r-r0)^2"

    registered.append(("bond", "quad"))
    assert [p.name for p in style_registry.styles("bond") if p.name == "quad"] == [
        "quad"
    ]


# ---------------------------------------------------------------------------
# A Python kernel that raises, or returns the wrong shape
# ---------------------------------------------------------------------------


def test_a_kernel_that_raises_is_reraised_from_compile_and_calc(registered) -> None:
    boom = {"on": True}

    def flaky(r, k, r0):
        if boom["on"]:
            raise ZeroDivisionError("flaky kernel")
        return k * (r - r0) ** 2, 2 * k * (r - r0)

    style_registry.register_style(
        "bond", "flaky", params={"k": "E/L^2", "r0": "L"}, kernel=flaky
    )
    registered.append(("bond", "flaky"))
    ff, frame = bond_ff("flaky", k=300.0, r0=1.0), bond_frame()
    with pytest.raises(ir.KernelShapeError, match="flaky") as err:
        molrs.ff.compile.PotentialCompiler(ff).compile(frame)
    assert isinstance(err.value.__cause__, ZeroDivisionError)
    assert err.value.style == "flaky"

    boom["on"] = False
    pots = molrs.ff.compile.PotentialCompiler(ff).compile(frame)
    e = pots.evaluate(frame)[0]
    boom["on"] = True
    with pytest.raises(ir.KernelShapeError) as err:
        pots.evaluate(frame)[0]
    assert isinstance(err.value.__cause__, ZeroDivisionError)
    # Re-raised once: the next good evaluation is clean.
    boom["on"] = False
    assert pots.evaluate(frame)[0] == e
    boom["on"] = True
    with pytest.raises(ir.KernelShapeError):
        style_registry.evaluate("bond", "flaky", [1.0], k=1.0, r0=1.0)


def test_a_kernel_of_the_wrong_shape_is_named(registered) -> None:
    def short(r, k, r0):
        return k * (r - r0) ** 2, np.zeros(len(r) + 1)

    style_registry.register_style(
        "bond", "short", params={"k": "E/L^2", "r0": "L"}, kernel=short
    )
    registered.append(("bond", "short"))
    with pytest.raises(ir.KernelShapeError, match="de_dq") as err:
        style_registry.evaluate("bond", "short", [1.0, 1.1], k=1.0, r0=1.0)
    assert "(2,)" in str(err.value) and "(3,)" in str(err.value)

    def single(r, k, r0):
        return k * (r - r0) ** 2

    style_registry.register_style(
        "bond", "single", params={"k": "E/L^2", "r0": "L"}, kernel=single
    )
    registered.append(("bond", "single"))
    with pytest.raises(ir.KernelShapeError, match=r"tuple \(e, de_dq\)"):
        energy_forces(bond_ff("single", k=1.0, r0=1.0), bond_frame())

    def ints(r, k, r0):
        n = len(r)
        return np.zeros(n, dtype=np.int64), np.zeros(n)

    with pytest.raises(ir.KernelShapeError, match="float64"):
        style_registry.register_style(
            "bond",
            "ints",
            params={"k": "E/L^2", "r0": "L"},
            kernel=ints,
            samples=[{"q": (0.9, 1.1), "k": 1.0, "r0": 1.0}],
        )
    assert "ints" not in {s.name for s in style_registry.styles("bond")}


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_every_refusal_is_a_value_error_named_after_its_variant_plus_error() -> None:
    assert issubclass(ir.IrError, ValueError)
    for name in [
        "UnknownCategory",
        "BadName",
        "Arity",
        "BlockName",
        "ReservedParam",
        "DuplicateParam",
        "Dimension",
        "Parse",
        "UnboundVariable",
        "UnknownFunction",
        "FunctionArity",
        "Point",
        "CoordinateMismatch",
        "Derivative",
        "Disagree",
        "Asymmetric",
        "Sealed",
        "Conflict",
        "NoKernel",
        "NoMixing",
        "MissingParam",
        "BadValue",
        "KernelShape",
        "NoEngineForm",
        "FormConflict",
        "NoForm",
        "Malformed",
    ]:
        cls = getattr(ir, name + "Error")
        assert issubclass(cls, ir.IrError) and cls.__name__ == name + "Error"


def test_a_built_in_is_sealed() -> None:
    for attempt in (
        lambda: style_registry.register_style(
            "bond", "harmonic", params={"k": "E/L^2"}, expression="k*r^2"
        ),
        lambda: style_registry.register_style(
            "bond", "harmonic", params={"k": "E/L^2"}, expression="k*r^2", replace=True
        ),
        lambda: style_registry.unregister_style("bond", "harmonic"),
    ):
        with pytest.raises(ir.SealedError) as err:
            attempt()
        assert (err.value.category, err.value.style) == ("bond", "harmonic")
    # Restating a built-in category exactly is a no-op; anything else is sealed.
    style_registry.register_category("bond", 2, coordinate="distance")
    with pytest.raises(ir.SealedError):
        style_registry.register_category(
            "bond", 2, coordinate="distance", order="ordered"
        )


def test_expressions_reading_what_is_not_there_are_refused() -> None:
    with pytest.raises(ir.UnboundVariableError) as err:
        style_registry.register_style(
            "bond", "bent", params={"k": "E"}, expression="k*theta^2"
        )
    assert (err.value.style, err.value.name) == ("bent", "theta")
    with pytest.raises(ir.UnknownFunctionError) as err:
        style_registry.register_style(
            "bond", "odd", params={"k": "E"}, expression="k*cosh(r)"
        )
    assert err.value.name == "cosh"
    with pytest.raises(ir.FunctionArityError) as err:
        style_registry.register_style(
            "bond", "odd", params={"k": "E"}, expression="k*min(r)"
        )
    assert (err.value.name, err.value.given, err.value.expected) == ("min", 1, 2)
    with pytest.raises(ir.ParseError):
        style_registry.register_style(
            "bond", "odd", params={"k": "E"}, expression="k*(r-"
        )
    with pytest.raises(ir.PointError) as err:
        style_registry.register_style(
            "bond", "odd", params={"k": "E"}, expression="k*distance(p1,p3)"
        )
    assert err.value.point == "p3"
    assert "odd" not in {s.name for s in style_registry.styles("bond")}


def test_parameter_declarations_are_checked() -> None:
    with pytest.raises(ir.DimensionError) as err:
        ir.ParamSpec("k", "E/L^^2")
    assert (err.value.param, err.value.dim) == ("k", "E/L^^2")
    with pytest.raises(ir.DimensionError):
        ir.ParamSpec("theta0", "A^2")  # a positive angle power is an angle value: A
    with pytest.raises(ir.DimensionError):
        style_registry.register_style(
            "bond", "odd", params={"k": "E/Z"}, expression="k*r"
        )
    with pytest.raises(ir.ReservedParamError) as err:
        style_registry.register_style("bond", "odd", params={"r": "L"}, expression="r")
    assert err.value.param == "r"
    with pytest.raises(ir.DuplicateParamError):
        style_registry.register_style(
            "bond",
            "odd",
            params=[ir.ParamSpec("k"), ir.ParamSpec("k")],
            expression="k*r",
        )
    with pytest.raises(ir.BadNameError):
        style_registry.register_style("bond", "odd", params={"2k": "E"}, expression="r")
    with pytest.raises(ir.UnknownCategoryError) as err:
        style_registry.register_style(
            "nosuch", "odd", params={"k": "E"}, expression="k"
        )
    assert err.value.category == "nosuch"
    with pytest.raises(ir.NoKernelError):
        style_registry.register_style("bond", "odd", params={"k": "E"})
    p = ir.ParamSpec("epsilon", "E", mix=("lj_epsilon", "sigma"), default=0.5)
    assert (p.name, p.dim, p.kind, p.mix, p.default) == (
        "epsilon",
        "E",
        "scalar",
        ("lj_epsilon", "sigma"),
        0.5,
    )
    assert repr(ir.ParamSpec("k", "E/L^2")) == "ParamSpec('k', 'E/L^2')"
    assert ir.ParamSpec("mode", kind="text", choices=["a", "b"]).choices == ["a", "b"]
    with pytest.raises(ValueError):
        ir.ParamSpec("mode", kind="text", choices=["a"], default="b")


def test_replace_overrides_a_custom_style_only(registered) -> None:
    register_fene(registered, "fene/r")
    register_fene(registered, "fene/r")  # identical: a no-op
    fene = ("bond", "fene/r")
    with pytest.raises(ir.ConflictError) as err:
        style_registry.register_style(*fene, params=FENE_PARAMS, expression=FENE + "+0")
    assert (err.value.category, err.value.style) == fene
    style_registry.register_style(
        *fene, params=FENE_PARAMS, expression=FENE + "+1", replace=True
    )
    (info,) = [s for s in style_registry.styles("bond") if s.name == "fene/r"]
    assert info.expression == FENE + "+1"
    # A refused replacement leaves the registered one in place.
    with pytest.raises(ir.UnknownFunctionError):
        style_registry.register_style(
            *fene, params=FENE_PARAMS, expression="foo(r)", replace=True
        )
    (info,) = [s for s in style_registry.styles("bond") if s.name == "fene/r"]
    assert info.expression == FENE + "+1"
    # The same callable registered again is the same kernel: a no-op.
    style_registry.register_style(
        "bond", "fene/k", params=FENE_PARAMS, kernel=fene_kernel
    )
    registered.append(("bond", "fene/k"))
    style_registry.register_style(
        "bond", "fene/k", params=FENE_PARAMS, kernel=fene_kernel
    )
    with pytest.raises(ir.ConflictError):
        style_registry.register_style(
            "bond", "fene/k", params=FENE_PARAMS, kernel=lambda r, **p: (r, r)
        )
    style_registry.unregister_style(*fene)
    with pytest.raises(ir.NoKernelError):
        style_registry.unregister_style(*fene)


def test_compile_refusals_raise_their_variant(registered) -> None:
    frame = bond_frame()
    with pytest.raises(ir.NoKernelError) as err:
        energy_forces(bond_ff("nosuch/style", k=1.0), frame)
    assert err.value.style == "nosuch/style"
    assert "molrs.ff.style_registry.register_style" in str(err.value)
    register_fene(registered)
    ff = bond_ff("fene/expr", k=K, r0=R0, epsilon=EPS)  # no sigma
    with pytest.raises(ir.MissingParamError) as err:
        energy_forces(ff, frame)
    assert (err.value.style, err.value.param) == ("fene/expr", "sigma")


# ---------------------------------------------------------------------------
# evaluate, introspection
# ---------------------------------------------------------------------------


def test_evaluate_prices_a_built_in_by_its_expression() -> None:
    r = np.array([1.0, 1.5])
    e, de = style_registry.evaluate("bond", "harmonic", r, k=300.0, r0=1.2)
    np.testing.assert_allclose(e, 300.0 * (r - 1.2) ** 2, rtol=1e-14)
    np.testing.assert_allclose(de, 600.0 * (r - 1.2), rtol=1e-14)
    theta = np.array([1.8, 2.0])
    e, de = style_registry.evaluate("angle", "harmonic", theta, k=50.0, theta0=109.5)
    t0 = math.radians(109.5)
    np.testing.assert_allclose(e, 50.0 * (theta - t0) ** 2, rtol=1e-12)
    with pytest.raises(ir.MissingParamError) as err:
        style_registry.evaluate("bond", "harmonic", r, k=300.0)
    assert err.value.param == "r0"
    with pytest.raises(TypeError, match="r00"):
        style_registry.evaluate("bond", "harmonic", r, k=300.0, r00=1.2)
    with pytest.raises(ValueError, match="constructor"):
        style_registry.evaluate("bond", "mmff_bond", r, kb=1.0, r0=1.0)


def test_the_registry_lists_categories_and_styles(registered) -> None:
    register_fene(registered)
    cats = {c.name: c for c in style_registry.categories()}
    assert (cats["bond"].arity, cats["bond"].block, cats["bond"].coordinate) == (
        2,
        "bonds",
        "distance",
    )
    assert (
        cats["bond"].builtin
        and cats["pair"].pair
        and cats["improper"].order == "ordered"
    )
    bonds = {s.name: s for s in style_registry.styles("bond")}
    assert bonds["harmonic"].builtin and bonds["harmonic"].kernel == "constructor"
    fene = bonds["fene/expr"]
    assert (
        not fene.builtin and fene.kernel == "expression" and fene.source == "type_rows"
    )
    assert [(p.name, p.dim) for p in fene.params] == [
        ("k", "E/L^2"),
        ("r0", "L"),
        ("epsilon", "E"),
        ("sigma", "L"),
    ]
    assert {s.category for s in style_registry.styles()} >= {
        "bond",
        "angle",
        "pair",
        "cmap",
    }
    assert style_registry.styles("dihedral")[0].category == "dihedral"
    assert {s.name for s in style_registry.styles("dihedral")} >= {"rb", "periodic"}


# ---------------------------------------------------------------------------
# Pair styles: per-parameter mixing, x1/x2 self rows
# ---------------------------------------------------------------------------

LJ_SMOOTH = (
    "4*epsilon*((sigma/r)^12-(sigma/r)^6)-4*epsilon*((sigma/cutoff)^12-(sigma/cutoff)^6)"
    "+(r-cutoff)*4*epsilon*(12*(sigma/cutoff)^12-6*(sigma/cutoff)^6)/cutoff"
)
RC = 2.5
PAIR_XYZ = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.3, 1.2, 0.4]])
PAIR_TYPES = ["A", "B", "A"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def pair_frame() -> molrs.core.Frame:
    atoms = molrs.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, PAIR_XYZ[:, d].copy())
    atoms.insert("type", PAIR_TYPES)
    pairs = molrs.core.Block()
    pairs.insert("atomi", np.array([i for i, _ in PAIRS], dtype=np.uint64))
    pairs.insert("atomj", np.array([j for _, j in PAIRS], dtype=np.uint64))
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["pairs"] = pairs
    return frame


def pair_ff(
    style: str, rows: dict[str, dict[str, float]], **style_params
) -> molrs.ff.forcefield.ForceField:
    ff = molrs.ff.forcefield.ForceField("pairs", units="lj")
    atoms = ff.def_style("atom", "full")
    types = {t: atoms.def_type(t, mass=1.0) for t in rows}
    pair = ff.def_style("pair", style, {"cutoff": RC, **style_params})
    for t, params in rows.items():
        pair.def_type(t, types[t], **params)
    return ff


def test_a_pair_style_by_expression_mixes_per_parameter(registered) -> None:
    style_registry.register_style(
        "pair",
        "lj/smooth/linear/x",
        params=[
            ir.ParamSpec("epsilon", "E", mix=("lj_epsilon", "sigma")),
            ir.ParamSpec("sigma", "L", mix=("lj_sigma", "epsilon")),
        ],
        style_params=[ir.ParamSpec("cutoff", "L")],
        expression=LJ_SMOOTH,
    )
    registered.append(("pair", "lj/smooth/linear/x"))
    rows = {"A": {"epsilon": 0.8, "sigma": 1.0}, "B": {"epsilon": 1.3, "sigma": 1.2}}
    frame = pair_frame()
    e, f = energy_forces(pair_ff("lj/smooth/linear/x", rows), frame)

    def lj(r, eps, sig):
        return 4 * eps * ((sig / r) ** 12 - (sig / r) ** 6)

    def dlj(r, eps, sig):
        return 4 * eps * (-12 * sig**12 / r**13 + 6 * sig**6 / r**7)

    e_ref, f_ref = 0.0, np.zeros_like(PAIR_XYZ)
    for i, j in PAIRS:
        a, b = rows[PAIR_TYPES[i]], rows[PAIR_TYPES[j]]
        eps = math.sqrt(a["epsilon"] * b["epsilon"])  # arithmetic (Lorentz-Berthelot)
        sig = 0.5 * (a["sigma"] + b["sigma"])
        d = PAIR_XYZ[j] - PAIR_XYZ[i]
        r = float(np.linalg.norm(d))
        e_ref += lj(r, eps, sig) - lj(RC, eps, sig) - (r - RC) * dlj(RC, eps, sig)
        de = dlj(r, eps, sig) - dlj(RC, eps, sig)
        f_ref[i] += de * d / r
        f_ref[j] -= de * d / r
    assert math.isclose(e, e_ref, rel_tol=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=0, atol=1e-10 * np.abs(f_ref).max())


def test_a_pair_expression_reads_the_self_rows(registered) -> None:
    soft = "(1+cos(3.141592653589793*r/cutoff))"
    params = [ir.ParamSpec("a", "E", mix="geometric")]
    cutoff = [ir.ParamSpec("cutoff", "L")]
    style_registry.register_style(
        "pair", "soft/bare", params=params, style_params=cutoff, expression="a*" + soft
    )
    registered.append(("pair", "soft/bare"))
    style_registry.register_style(
        "pair",
        "soft/x12",
        params=params,
        style_params=cutoff,
        expression="sqrt(a1*a2)*" + soft,
    )
    registered.append(("pair", "soft/x12"))
    rows = {"A": {"a": 2.0}, "B": {"a": 5.0}}
    frame = pair_frame()
    e_bare, f_bare = energy_forces(pair_ff("soft/bare", rows), frame)
    e_x12, f_x12 = energy_forces(pair_ff("soft/x12", rows), frame)
    e_ref = sum(
        math.sqrt(rows[PAIR_TYPES[i]]["a"] * rows[PAIR_TYPES[j]]["a"])
        * (1 + math.cos(math.pi * np.linalg.norm(PAIR_XYZ[j] - PAIR_XYZ[i]) / RC))
        for i, j in PAIRS
    )
    assert math.isclose(e_bare, e_ref, rel_tol=1e-12)
    assert math.isclose(e_x12, e_ref, rel_tol=1e-12)
    np.testing.assert_allclose(f_x12, f_bare, rtol=0, atol=1e-12 * np.abs(f_bare).max())
    # evaluate: the self rows default to the pair value, or are given.
    e, _ = style_registry.evaluate("pair", "soft/x12", [1.0], a=3.0, cutoff=RC)
    assert math.isclose(e[0], 3.0 * (1 + math.cos(math.pi / RC)), rel_tol=1e-12)
    e, _ = style_registry.evaluate(
        "pair", "soft/x12", [1.0], a=0.0, a1=2.0, a2=8.0, cutoff=RC
    )
    assert math.isclose(e[0], 4.0 * (1 + math.cos(math.pi / RC)), rel_tol=1e-12)
    # A pair energy must not change when its atoms are exchanged.
    with pytest.raises(ir.AsymmetricError):
        style_registry.register_style(
            "pair",
            "soft/lopsided",
            params=params,
            style_params=cutoff,
            expression="a1*" + soft,
            samples=[{"q": (0.5, 2.0), "a": 1.0, "cutoff": RC}],
        )


# ---------------------------------------------------------------------------
# A new category: registered, typed in a ForceField, priced
# ---------------------------------------------------------------------------

UB = (33.0, 2.2)  # k_ub, r_ub


def ub_frame(xyz: np.ndarray, block: str) -> molrs.core.Frame:
    atoms = molrs.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, xyz[:, d].copy())
    atoms.insert("type", ["A"] * len(xyz))
    rows = molrs.core.Block()
    for key, atom in (("atomi", 0), ("atomj", 1), ("atomk", 2)):
        rows.insert(key, np.array([atom], dtype=np.uint32))
    rows.insert("type", ["t"])
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame[block] = rows
    return frame


def ub_reference(xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """LAMMPS ``angle_style charmm`` with K = 0: the 1-3 spring alone."""
    ff = molrs.ff.forcefield.ForceField("charmm")
    a = ff.def_style("atom", "full").def_type("A", mass=1.0)
    ff.def_style("angle", "charmm").def_type(
        "t", a, a, a, k=0.0, theta0=109.5, k_ub=UB[0], r_ub=UB[1]
    )
    frame = ub_frame(xyz, "angles")
    return molrs.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)


def test_a_new_category_registers_and_evaluates(registered) -> None:
    style_registry.register_category("urey_bradley", 3)
    style_registry.register_category("urey_bradley", 3)  # identical: a no-op
    cat = {c.name: c for c in style_registry.categories()}["urey_bradley"]
    assert (cat.arity, cat.block, cat.coordinate, cat.order, cat.builtin) == (
        3,
        "urey_bradleys",
        "compound",
        "reversible",
        False,
    )
    with pytest.raises(ir.ConflictError):
        style_registry.register_category("urey_bradley", 4)
    style_registry.register_style(
        "urey_bradley",
        "harmonic",
        params={"k_ub": "E/L^2", "r_ub": "L"},
        expression="k_ub*(distance(p1,p3)-r_ub)^2",
    )
    registered.append(("urey_bradley", "harmonic"))
    x = np.array(
        [
            [[1.53, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.30, 1.05, 0.10]],
            [[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.0, 1.4, 0.3]],
        ]
    )
    d = x[:, 0] - x[:, 2]
    r13 = np.linalg.norm(d, axis=1)
    e, grad = style_registry.evaluate(
        "urey_bradley", "harmonic", x=x, k_ub=UB[0], r_ub=UB[1]
    )
    np.testing.assert_allclose(e, UB[0] * (r13 - UB[1]) ** 2, rtol=1e-12)
    g1 = (2 * UB[0] * (r13 - UB[1]) / r13)[:, None] * d
    np.testing.assert_allclose(grad[:, 0], g1, rtol=1e-12)
    np.testing.assert_allclose(grad[:, 2], -g1, rtol=1e-12)
    np.testing.assert_allclose(grad[:, 1], 0.0, atol=1e-15)

    # The same energy by a numpy compound kernel, agreeing on samples.
    def ub(x, k_ub, r_ub):
        d = x[:, 0] - x[:, 2]
        r = np.linalg.norm(d, axis=1)
        g = np.zeros_like(x)
        g[:, 0] = (2 * k_ub * (r - r_ub) / r)[:, None] * d
        g[:, 2] = -g[:, 0]
        return k_ub * (r - r_ub) ** 2, g

    style_registry.register_style(
        "urey_bradley",
        "harmonic/np",
        params={"k_ub": "E/L^2", "r_ub": "L"},
        expression="k_ub*(distance(p1,p3)-r_ub)^2",
        kernel=ub,
        samples=[{"q": (1.0, 1.6), "k_ub": UB[0], "r_ub": UB[1]}],
    )
    registered.append(("urey_bradley", "harmonic/np"))
    e_np, grad_np = style_registry.evaluate(
        "urey_bradley", "harmonic/np", x=x, k_ub=UB[0], r_ub=UB[1]
    )
    np.testing.assert_allclose(e_np, e, rtol=1e-12)
    np.testing.assert_allclose(grad_np, grad, rtol=1e-12, atol=1e-14)
    # Typed in a ForceField and compiled: LAMMPS `angle charmm` with K = 0.
    reference = ub_reference(x[0])
    for style in ("harmonic", "harmonic/np"):
        ff = molrs.ff.forcefield.ForceField("ub")
        a = ff.def_style("atom", "full").def_type("A", mass=1.0)
        ff.def_style("urey_bradley", style).def_type(
            "t", a, a, a, k_ub=UB[0], r_ub=UB[1]
        )
        frame = ub_frame(x[0], "urey_bradleys")
        e, f = molrs.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)
        assert math.isclose(e, reference[0], rel_tol=1e-12)
        np.testing.assert_allclose(
            f, reference[1], rtol=0, atol=1e-12 * np.abs(reference[1]).max()
        )
    # A type with the wrong number of endpoints is the IR's Arity.
    with pytest.raises(ir.ArityError, match="got 2") as err:
        ff.get_style("urey_bradley", "harmonic/np").def_type(
            "x", a, a, k_ub=1.0, r_ub=1.0
        )
    assert (err.value.category, err.value.arity) == ("urey_bradley", 2)
    with pytest.raises(TypeError, match="pass x"):
        style_registry.evaluate("urey_bradley", "harmonic", [1.0], k_ub=1.0, r_ub=1.0)
    with pytest.raises(ir.PointError):
        style_registry.register_style(
            "urey_bradley", "far", params={"k": "E"}, expression="k*distance(p1,p4)"
        )


def test_a_category_that_does_not_conform_is_refused() -> None:
    with pytest.raises(ir.ArityError) as err:
        style_registry.register_category("six_body", 6)
    assert (err.value.category, err.value.arity) == ("six_body", 6)
    with pytest.raises(ir.ArityError):
        style_registry.register_category("lonely", 1)
    with pytest.raises(ir.BadNameError):
        style_registry.register_category("Bad-Name", 2)
    with pytest.raises(ir.CoordinateMismatchError):
        style_registry.register_category("bent_pair", 2, coordinate="angle")
    with pytest.raises(ValueError, match="coordinate"):
        style_registry.register_category("odd", 2, coordinate="torsion")
    style_registry.register_category(
        "restraint2", 2, coordinate="distance", order="ordered"
    )
    assert {c.name: c.coordinate for c in style_registry.categories()}[
        "restraint2"
    ] == "distance"
