"""The force-field IR is a protocol, from Python (``ff-ir-02-protocol``,
P-Python): what conforms to its form extends it — a style by expression or
by a numpy callable, a new category, an array parameter — with nothing in
molrs rebuilt; it writes as the LAMMPS style it is, persists to a process that
registered nothing, and what does not conform is refused by the
``mp.ff.ir.IrError`` subclass naming the item.

| Row | Test | Criterion |
|---|---|---|
| fene by expression = analytic formula | ``test_fene_by_expression_is_the_analytic_formula`` | E rel ≤ 1e-12, F ≤ 1e-12 of the force scale |
| numpy callable = expression | ``test_a_numpy_kernel_is_the_expression`` | E rel ≤ 1e-12, F rel ≤ 1e-10 |
| fene written as LAMMPS ``bond_style fene`` | ``test_fene_writes_as_lammps_bond_style_fene`` | the deck's style and coefficients |
| ``urey_bradley`` from Python | ``test_a_python_category_registers_as_urey_bradley`` | arity, block, not built in |
| pair ``lj/smooth/linear`` by expression and numpy, its cutoff straddling the pairs, written as LAMMPS ``lj/smooth/linear``; ``compile_typed`` (MD's first force call) = ``compile`` | ``test_a_python_pair_style_writes_as_lj_smooth_linear_and_prices_alike_at_both_doors`` | the deck's style; doors rel ≤ 1e-12 |
| ``.mrec`` round trip: expression byte for byte, a fresh subprocess prices it | ``test_a_record_prices_the_same_bits_in_a_fresh_process`` | bit for bit |
| callable-only style in a fresh process | ``test_a_callable_only_style_is_no_kernel_in_a_fresh_process`` | ``NoKernel`` (a ``ValueError``) naming the style and ``mp.ff.style_registry.register_style`` |
| refusals | ``test_what_does_not_conform_is_refused_by_name`` | each its ``IrError`` subclass, naming the item |
| array param ``dihedral table/linear`` (``table: f64[N]``), numpy kernel | ``test_an_array_param_style_is_hand_linear_interpolation_and_round_trips`` | rel ≤ 1e-12, bits round-trip |
| class2's bond-angle term as a new category (optional) | ``test_class2_bond_angle_is_one_energy_three_ways`` | expression = numpy = hand −π/60, rel ≤ 1e-12; F = −∇E by central differences |

Relative error is ``|got − want| / max(|want|, s)``, ``s`` the largest
|force component| of the configuration (an energy's own magnitude for the
energy).

Neighbouring suites prove the same protocol from other sides:
``test_ff_ir.py`` (registration, kernels, refusals), ``test_ff_ir_engine.py``
(engine forms) and ``test_ff_ir_persist.py`` (records).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import molpy as mp
import numpy as np
import pytest
from molpy.ff import ir, style_registry


def _lammps_include(
    ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame, **options: object
) -> str:
    """The LAMMPS force-field include ``ff`` writes for ``frame``."""
    buf = bytearray()
    mp.io.lammps.LammpsForcefieldWriter(frame, **options).write(buf, ff)  # type: ignore[arg-type]
    return buf.decode()


# LAMMPS `bond_style fene` (Kremer-Grest): K R0 epsilon sigma.
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
# LAMMPS `pair_style lj/smooth/linear`: φ(r) − φ(rc) − (r − rc) φ′(rc), φ the
# 12-6 Lennard-Jones, so energy and force vanish at the cutoff.
SMOOTH = (
    "4*epsilon*((sigma/r)^12-(sigma/r)^6)-4*epsilon*((sigma/cutoff)^12-(sigma/cutoff)^6)"
    "+(r-cutoff)*4*epsilon*(12*(sigma/cutoff)^12-6*(sigma/cutoff)^6)/cutoff"
)
SMOOTH_PARAMS = [
    ir.ParamSpec("epsilon", "E", mix=("lj_epsilon", "sigma")),
    ir.ParamSpec("sigma", "L", mix=("lj_sigma", "epsilon")),
]
SMOOTH_STYLE = [
    ir.ParamSpec("cutoff", "L"),
    ir.ParamSpec(
        "mixing",
        kind="text",
        choices=["arithmetic", "geometric", "sixthpower"],
        default="arithmetic",
    ),
]
UB = "k_ub*(distance(p1,p3)-r_ub)^2"
BOND_ANGLE = (
    "(n1*(distance(p1,p2)-r1)+n2*(distance(p2,p3)-r2))"
    "*(angle(p1,p2,p3)-theta0*(pi/180)); pi=3.141592653589793"
)
BA_PARAMS = {"n1": "E/L/A", "n2": "E/L/A", "r1": "L", "r2": "L", "theta0": "A"}

# A bead chain on the 0.01 grid: bonds 0.97, 1.06, 1.18, 1.31 — both sides
# of the WCA cutoff 2^(1/6) σ = 1.1225, (r/R0)² < 0.8 everywhere.
BEADS = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.97, 0.0, 0.0],
        [0.97, 1.06, 0.0],
        [0.97, 1.06, 1.18],
        [1.76, 2.11, 1.18],
    ]
)
# Six unbonded atoms of two types on the 0.01 grid (the Rust proof's), and the
# cutoff that straddles their pairs: ten inside, five beyond, the nearest
# 0.19 Å from it, in each configuration.
SMOOTH_XYZ = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.71, 0.12, 0.0],
        [0.05, 3.93, 0.21],
        [3.62, 3.84, 0.43],
        [1.83, 1.91, 3.52],
        [6.53, 0.24, 0.11],
    ]
)
SMOOTH_TYPES = ["A", "A", "B", "B", "B", "A"]
SMOOTH_ROWS = {"A": (0.2, 3.1), "B": (0.15, 3.6)}
SMOOTH_RC = 5.0
# Four atoms A-B-B-A: the 1-3 terms (0, 1, 2) and (1, 2, 3).
CHAIN = np.array(
    [[0.0, 0.0, 0.0], [1.42, 0.31, 0.0], [2.05, 1.62, 0.12], [3.47, 1.81, 0.4]]
)
UB_ROWS = [((0, 1, 2), "A-B-B", 22.5, 2.45), ((1, 2, 3), "B-B-A", 18.0, 2.62)]
TABLE = np.array([1.25 + math.sin(0.7 * i) / 3.0 - 0.01 * i * i for i in range(12)])
NO_KERNEL = (
    "no kernel for {} `{}`: register it (molpy.ff.style_registry.register_style) "
    "or give it an expression"
)


def moved(x: np.ndarray, d: float) -> np.ndarray:
    """``x`` with atom ``i`` moved by ``d`` times ``(i mod 3 − 1, (i + 1) mod 2,
    2 (i mod 2) − 1)``: another configuration on the grid, every distance and
    angle changed."""
    i = np.arange(len(x))
    step = np.stack([i % 3 - 1.0, (i + 1) % 2 * 1.0, 2.0 * (i % 2) - 1.0], axis=1)
    return np.round(x + d * step, 2)


# ---------------------------------------------------------------------------
# The extensions: numpy kernels, and the module's registrations
# ---------------------------------------------------------------------------


def fene_kernel(r, k, r0, epsilon, sigma):
    """FENE in numpy: one call per evaluation, every term at once."""
    x = (r / r0) ** 2
    s6 = (sigma / r) ** 6
    inner = r < 2 ** (1 / 6) * sigma
    e = -0.5 * k * r0**2 * np.log(1 - x) + np.where(
        inner, 4 * epsilon * (s6 * s6 - s6) + epsilon, 0.0
    )
    de = k * r / (1 - x) + np.where(
        inner, 4 * epsilon * (6 * s6 - 12 * s6 * s6) / r, 0.0
    )
    return e, de


def smooth_kernel(r, epsilon, sigma, cutoff, **_):
    """``lj/smooth/linear`` in numpy (the other inputs — charges, the
    special-bonds weights, ``mixing`` — arrive as keywords and are unused)."""

    def lj(x):
        s6 = (sigma / x) ** 6
        return 4 * epsilon * (s6 * s6 - s6), 24 * epsilon * (s6 - 2 * s6 * s6) / x

    (u, du), (uc, duc) = lj(r), lj(cutoff)
    return u - uc - (r - cutoff) * duc, du - duc


def bond_angle_kernel(x, n1, n2, r1, r2, theta0):
    """class2's bond-angle term over ``x[n, 3, 3]``: energy and ∂E/∂x."""
    u, v = x[:, 0] - x[:, 1], x[:, 2] - x[:, 1]
    a, b = np.linalg.norm(u, axis=1), np.linalg.norm(v, axis=1)
    c = np.clip(np.einsum("ij,ij->i", u, v) / (a * b), -1.0, 1.0)
    stretch = n1 * (a - r1) + n2 * (b - r2)
    bend = np.arccos(c) - np.radians(theta0)  # angle parameters are stored in degrees
    s = np.sqrt(1 - c * c)
    ab = (a * b)[:, None]
    dtheta1 = -(v / ab - (c / a**2)[:, None] * u) / s[:, None]
    dtheta3 = -(u / ab - (c / b**2)[:, None] * v) / s[:, None]
    g1 = (n1 * bend / a)[:, None] * u + stretch[:, None] * dtheta1
    g3 = (n2 * bend / b)[:, None] * v + stretch[:, None] * dtheta3
    return stretch * bend, np.stack([g1, -g1 - g3, g3], axis=1)


def table_at(phi: np.ndarray, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hand linear interpolation of a periodic table on ``φ_i = −π + 2π i/N``,
    one row's table per term (``table[n, N]``)."""
    n = table.shape[1]
    h = 2 * math.pi / n
    s = (phi + math.pi) / h
    i = np.floor(s).astype(int) % n
    t = s - np.floor(s)
    rows = np.arange(len(phi))
    lo, hi = table[rows, i], table[rows, (i + 1) % n]
    return lo + t * (hi - lo), (hi - lo) / h


def table_kernel(phi, table):
    """The array-parameter style's numpy kernel: ``table`` is ``[n, N]``."""
    assert table.shape == (len(phi), len(TABLE))
    return table_at(phi, table)


@pytest.fixture(scope="module", autouse=True)
def extensions() -> Iterator[None]:
    """Everything this module adds to the IR, taken out again at the end
    (the categories stay: a registered category is not removed)."""
    style_registry.register_style(
        "bond",
        "fene/proof",
        params=FENE_PARAMS,
        expression=FENE,
        lammps="positional:fene",
    )
    style_registry.register_style(
        "bond", "fene/proof-np", params=FENE_PARAMS, kernel=fene_kernel
    )
    style_registry.register_style(
        "pair",
        "lj/smooth/linear/proof",
        params=SMOOTH_PARAMS,
        style_params=SMOOTH_STYLE,
        special="lj",
        expression=SMOOTH,
        lammps="positional:lj/smooth/linear",
    )
    style_registry.register_style(
        "pair",
        "lj/smooth/linear/proof-np",
        params=SMOOTH_PARAMS,
        style_params=SMOOTH_STYLE,
        special="lj",
        kernel=smooth_kernel,
    )
    style_registry.register_category("urey_bradley", 3)
    style_registry.register_style(
        "urey_bradley", "proof", params={"k_ub": "E/L^2", "r_ub": "L"}, expression=UB
    )
    style_registry.register_category("bond_angle", 3)
    style_registry.register_style(
        "bond_angle", "class2", params=BA_PARAMS, expression=BOND_ANGLE
    )
    style_registry.register_style(
        "bond_angle",
        "class2/np",
        params=BA_PARAMS,
        expression=BOND_ANGLE,
        kernel=bond_angle_kernel,
        samples=[
            {
                "q": (1.2, 1.7),
                "n1": 10.0,
                "n2": 8.0,
                "r1": 1.5,
                "r2": 1.45,
                "theta0": 105.0,
            }
        ],
    )
    style_registry.register_style(
        "dihedral",
        "table/linear",
        params=[ir.ParamSpec("table", "E", kind="array", rank=1)],
        kernel=table_kernel,
    )
    yield
    for category, name in [
        ("bond", "fene/proof"),
        ("bond", "fene/proof-np"),
        ("pair", "lj/smooth/linear/proof"),
        ("pair", "lj/smooth/linear/proof-np"),
        ("urey_bradley", "proof"),
        ("bond_angle", "class2"),
        ("bond_angle", "class2/np"),
        ("dihedral", "table/linear"),
    ]:
        style_registry.unregister_style(category, name)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

MASSES = {"A": 12.011, "B": 14.007}
ENDPOINTS = ("atomi", "atomj", "atomk", "atoml")


def frame(
    xyz: np.ndarray,
    types: list[str],
    block: str | None = None,
    rows: list[tuple[int, ...]] = (),
    row_types: list[str] = (),
) -> mp.core.Frame:
    """Atoms of ``types`` at ``xyz`` (masses, no charge, in a box past every
    atom) and the terms ``rows`` in ``block``."""
    n = len(xyz)
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, np.ascontiguousarray(xyz[:, d]))
    atoms.insert("type", list(types))
    atoms.insert("mass", np.array([MASSES[t] for t in types]))
    atoms.insert("charge", np.zeros(n))
    atoms.insert("mol_id", np.ones(n, dtype=np.uint32))
    out = mp.core.Frame()
    out["atoms"] = atoms
    if block is not None:
        terms = mp.core.Block()
        for i, key in enumerate(ENDPOINTS[: len(rows[0])]):
            terms.insert(key, np.array([r[i] for r in rows], dtype=np.uint32))
        terms.insert("type", list(row_types))
        out[block] = terms
    out.box = mp.core.Box(
        mp.core.Cell.lengths([100.0] * 3),
        origin=np.full(3, -50.0),
        pbc=np.zeros(3, dtype=bool),
    )
    return out


def bead_frame(xyz: np.ndarray = BEADS) -> mp.core.Frame:
    rows = [(i, i + 1) for i in range(len(xyz) - 1)]
    return frame(xyz, ["B"] * len(xyz), "bonds", rows, ["B-B"] * len(rows))


def field(name: str, units: str) -> tuple[mp.ff.forcefield.ForceField, dict]:
    ff = mp.ff.forcefield.ForceField(name, units=units)
    atoms = ff.def_style("atom", "full")
    return ff, {t: atoms.def_type(t, mass=m, charge=0.0) for t, m in MASSES.items()}


def fene_ff(style: str, **row: float) -> mp.ff.forcefield.ForceField:
    ff, t = field("beads", "lj")
    params = {"k": K, "r0": R0, "epsilon": EPS, "sigma": SIG} | row
    params = {key: v for key, v in params.items() if v is not None}
    ff.def_style("bond", style).def_type("B-B", t["B"], t["B"], **params)
    return ff


def smooth_ff(style: str) -> mp.ff.forcefield.ForceField:
    ff, t = field("smooth", "real")
    ff.set_special_bonds([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    pair = ff.def_style("pair", style, {"cutoff": SMOOTH_RC})
    for name, (eps, sigma) in SMOOTH_ROWS.items():
        pair.def_type(name, t[name], epsilon=eps, sigma=sigma)
    return ff


def smooth_frame(xyz: np.ndarray, ff: mp.ff.forcefield.ForceField) -> mp.core.Frame:
    """The six atoms at ``xyz`` and their ``pairs`` list under ``ff``'s
    weights: every pair, none bonded."""
    f = frame(xyz, SMOOTH_TYPES)
    f["pairs"] = mp.ff.potential.intramolecular_pairs(f, ff)
    return f


def price(
    ff: mp.ff.forcefield.ForceField, f: mp.core.Frame
) -> tuple[float, np.ndarray]:
    e, forces = mp.ff.compile.PotentialCompiler(ff).compile(f).evaluate(f)
    return float(e), np.asarray(forces)


def rel(got: np.ndarray, want: np.ndarray, scale: float) -> float:
    """``max |got − want| / max(|want|, scale)``."""
    got, want = np.ravel(got), np.ravel(want)
    return float(np.max(np.abs(got - want) / np.maximum(np.abs(want), scale)))


def fene_analytic(xyz: np.ndarray) -> tuple[float, np.ndarray]:
    """The chain's FENE energy and forces, by hand."""
    d = xyz[1:] - xyz[:-1]
    r = np.linalg.norm(d, axis=1)
    x = (r / R0) ** 2
    s6 = (SIG / r) ** 6
    wca = r < 2 ** (1 / 6) * SIG
    e = -0.5 * K * R0**2 * np.log(1 - x) + np.where(
        wca, 4 * EPS * (s6 * s6 - s6) + EPS, 0
    )
    de = K * r / (1 - x) + np.where(wca, 4 * EPS * (6 * s6 - 12 * s6 * s6) / r, 0)
    f = np.zeros_like(xyz)
    pull = (de / r)[:, None] * d
    f[:-1] += pull
    f[1:] -= pull
    return float(e.sum()), f


# ---------------------------------------------------------------------------
# A new style, by expression and by numpy
# ---------------------------------------------------------------------------


def test_fene_by_expression_is_the_analytic_formula() -> None:
    worst = 0.0
    for xyz in (BEADS, moved(BEADS, 0.02)):
        e, f = price(fene_ff("fene/proof"), bead_frame(xyz))
        e_ref, f_ref = fene_analytic(xyz)
        assert math.isclose(e, e_ref, rel_tol=1e-12)
        worst = max(
            worst, abs(e - e_ref) / abs(e_ref), rel(f, f_ref, np.abs(f_ref).max())
        )
    assert worst <= 1e-12
    print(f"MEASURED fene expression vs analytic: {worst:.1e}")


def test_a_numpy_kernel_is_the_expression() -> None:
    worst_e = worst_f = 0.0
    for xyz in (BEADS, moved(BEADS, 0.02)):
        e_x, f_x = price(fene_ff("fene/proof"), bead_frame(xyz))
        e_np, f_np = price(fene_ff("fene/proof-np"), bead_frame(xyz))
        worst_e = max(worst_e, abs(e_np - e_x) / abs(e_x))
        worst_f = max(worst_f, rel(f_np, f_x, np.abs(f_x).max()))
    assert worst_e <= 1e-12 and worst_f <= 1e-10
    print(f"MEASURED fene numpy vs expression: E {worst_e:.1e}, F {worst_f:.1e}")


def test_fene_writes_as_lammps_bond_style_fene() -> None:
    deck = _lammps_include(fene_ff("fene/proof"), bead_frame(), units="lj")
    assert "bond_style fene\n" in deck
    (coeff,) = [l for l in deck.splitlines() if l.startswith("bond_coeff B-B ")]
    assert [float(v) for v in coeff.split()[2:]] == [30.0, 1.5, 1.0, 1.0]


def typed_energy_forces(
    ff: mp.ff.forcefield.ForceField, f: mp.core.Frame, xyz: np.ndarray
) -> tuple[float, np.ndarray]:
    """The neighbour-driven door: ``compile_typed`` mounted on ``MD`` over a
    neighbour list past every pair, and the setup force call."""
    from molpy.md import MD, VelocityVerlet

    work = f.copy()
    atoms = work["atoms"]
    for axis, key in enumerate("xyz"):
        atoms[key] = np.ascontiguousarray(xyz[:, axis])
    for key in ("vx", "vy", "vz"):
        atoms[key] = np.zeros(len(xyz))
    atoms["mass"] = np.ones(len(xyz))
    work.box = mp.core.Box(
        mp.core.Cell.lengths([100.0] * 3),
        origin=np.full(3, -50.0),
        pbc=np.ones(3, dtype=bool),
    )
    sim = MD()
    sim.set_integrator(VelocityVerlet(1.0))
    sim.set_neighbor(mp.core.NeighborList(30.0), skin=1.0)
    sim.set_evaluator(mp.ff.compile.PotentialCompiler(ff).compile_typed(f))
    sim.run(work, 0)
    forces = np.column_stack(
        [np.asarray(work["atoms"][key]) for key in ("fx", "fy", "fz")]
    )
    return float(work.meta["energy"]), forces


def test_a_python_pair_style_writes_as_lj_smooth_linear_and_prices_alike_at_both_doors() -> (
    None
):
    deck = _lammps_include(
        smooth_ff("lj/smooth/linear/proof"),
        frame(SMOOTH_XYZ, SMOOTH_TYPES),
        units="real",
    )
    (style,) = [l.split() for l in deck.splitlines() if l.startswith("pair_style ")]
    assert style[1] == "lj/smooth/linear" and float(style[2]) == SMOOTH_RC, deck
    assert "pair_modify mix arithmetic" in deck, deck
    worst_doors = 0.0
    for xyz in (SMOOTH_XYZ, moved(SMOOTH_XYZ, 0.13)):
        d = xyz[:, None] - xyz[None]
        r = np.linalg.norm(d, axis=-1)[np.triu_indices(len(xyz), 1)]
        assert (r < SMOOTH_RC).any() and (r >= SMOOTH_RC).any(), r
        for style in ("lj/smooth/linear/proof", "lj/smooth/linear/proof-np"):
            ff = smooth_ff(style)
            f = smooth_frame(xyz, ff)
            e, forces = price(ff, f)
            e_t, f_t = typed_energy_forces(ff, f, xyz)
            scale = float(np.abs(forces).max())
            worst_doors = max(
                worst_doors, abs(e_t - e) / abs(e), rel(f_t, forces, scale)
            )
    assert worst_doors <= 1e-12
    print(f"MEASURED smooth compile_typed vs compile: {worst_doors:.1e}")


def test_a_python_category_registers_as_urey_bradley() -> None:
    cat = {c.name: c for c in style_registry.categories()}["urey_bradley"]
    assert (cat.arity, cat.block, cat.builtin) == (3, "urey_bradleys", False)


# ---------------------------------------------------------------------------
# Persistence, in a process that registered nothing
# ---------------------------------------------------------------------------

FRESH = textwrap.dedent(
    """
    import json, pathlib, sys
    import molpy as mp
    import numpy as np

    path, frames = pathlib.Path(sys.argv[1]), json.loads(sys.argv[2])
    registered = [(s.category, s.name) for s in mp.ff.style_registry.styles() if not s.builtin]
    ff = mp.io.mrec.MrecFile(path).forcefield().to_forcefield()
    out = {"registered": registered, "styles": [[s.category, s.name] for s in ff.get_styles(mp.ff.forcefield.Style)]}
    for name, spec in frames.items():
        f = mp.core.Frame()
        atoms = mp.core.Block()
        xyz = np.array(spec["xyz"])
        for d, key in enumerate("xyz"):
            atoms.insert(key, np.ascontiguousarray(xyz[:, d]))
        atoms.insert("type", spec["types"])
        f["atoms"] = atoms
        terms = mp.core.Block()
        for i, key in enumerate(("atomi", "atomj", "atomk", "atoml")[: len(spec["rows"][0])]):
            terms.insert(key, np.array([r[i] for r in spec["rows"]], dtype=np.uint32))
        terms.insert("type", spec["row_types"])
        f[spec["block"]] = terms
        try:
            e, forces = mp.ff.compile.PotentialCompiler(ff).compile(f).evaluate(f)
            out[name] = {"e": float(e).hex(), "f": [float(v).hex() for v in np.ravel(forces)]}
        except ValueError as err:
            out[name] = {"error": type(err).__name__, "message": str(err),
                         "value_error": isinstance(err, ValueError)}
    print(json.dumps(out))
    """
)


def everything() -> tuple[mp.ff.forcefield.ForceField, dict[str, dict]]:
    """A force field holding every kind of extension — an expression style
    whose instance states no expression (the registry's is written), a
    category from Python, a callable-only array style — and one frame per
    style (a style prices only where its block has rows)."""
    ff, t = field("everything", "real")
    ff.def_style("bond", "fene/proof").def_type(
        "B-B", t["B"], t["B"], k=K, r0=R0, epsilon=EPS, sigma=SIG
    )
    ub = ff.def_style("urey_bradley", "proof")
    for _, name, k_ub, r_ub in UB_ROWS:
        ub.def_type(name, *(t[e] for e in name.split("-")), k_ub=k_ub, r_ub=r_ub)
    table = ff.def_style("dihedral", "table/linear")
    table.def_type("A-B-B-A", t["A"], t["B"], t["B"], t["A"], table=TABLE)
    frames = {
        "fene": {
            "xyz": BEADS.tolist(),
            "types": ["B"] * 5,
            "block": "bonds",
            "rows": [[i, i + 1] for i in range(4)],
            "row_types": ["B-B"] * 4,
        },
        "urey_bradley": {
            "xyz": CHAIN.tolist(),
            "types": ["A", "B", "B", "A"],
            "block": "urey_bradleys",
            "rows": [list(r[0]) for r in UB_ROWS],
            "row_types": [r[1] for r in UB_ROWS],
        },
        "table": {
            "xyz": CHAIN.tolist(),
            "types": ["A", "B", "B", "A"],
            "block": "dihedrals",
            "rows": [[0, 1, 2, 3]],
            "row_types": ["A-B-B-A"],
        },
    }
    return ff, frames


def spec_frame(spec: dict) -> mp.core.Frame:
    return frame(
        np.array(spec["xyz"]),
        spec["types"],
        spec["block"],
        [tuple(r) for r in spec["rows"]],
        spec["row_types"],
    )


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict, dict]:
    """The record written here, this process's prices, and what a fresh
    process that registered nothing makes of the record."""
    ff, frames = everything()
    path = tmp_path_factory.mktemp("proof") / "everything.mrec"
    mp.io.write_mrec(path, forcefield=mp.io.mrec.ForceFieldSection.from_forcefield(ff))
    here = {}
    for name, spec in frames.items():
        e, f = price(ff, spec_frame(spec))
        here[name] = {"e": e.hex(), "f": [float(v).hex() for v in f.ravel()]}
    done = subprocess.run(
        [sys.executable, "-I", "-c", FRESH, str(path), json.dumps(frames)],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr
    return (
        mp.io.mrec.MrecFile(path).forcefield().document,
        here,
        json.loads(done.stdout),
    )


def test_a_record_prices_the_same_bits_in_a_fresh_process(fresh) -> None:
    document, here, there = fresh
    assert there["registered"] == []
    expressions = {
        (s["category"], s["style"]): s.get("expression") for s in document["styles"]
    }
    # Byte for byte: the registry's, written since the instance states none.
    assert expressions[("bond", "fene/proof")] == FENE
    assert expressions[("urey_bradley", "proof")] == UB
    assert expressions[("dihedral", "table/linear")] is None
    for name in ("fene", "urey_bradley"):
        assert there[name] == here[name], name
    print("MEASURED fresh process: fene, urey_bradley bit for bit")


def test_a_callable_only_style_is_no_kernel_in_a_fresh_process(fresh) -> None:
    _, here, there = fresh
    assert ["dihedral", "table/linear"] in there["styles"]
    assert "e" in here["table"]
    assert there["table"] == {
        "error": "NoKernelError",
        "message": NO_KERNEL.format("dihedral", "table/linear"),
        "value_error": True,
    }


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _short(r, k, r0):
    return k * (r - r0) ** 2, np.zeros(len(r) + 1)


def _raising(r, k, r0):
    raise ZeroDivisionError("a kernel that raises")


def _compile_with(kernel: Callable, name: str) -> None:
    style_registry.register_style(
        "bond", name, params={"k": "E/L^2", "r0": "L"}, kernel=kernel
    )
    try:
        price(fene_ff("fene/proof"), bead_frame())  # unaffected
        ff, t = field("x", "lj")
        ff.def_style("bond", name).def_type("B-B", t["B"], t["B"], k=1.0, r0=1.0)
        price(ff, bead_frame())
    finally:
        style_registry.unregister_style("bond", name)


def _wrong_arity() -> None:
    ff, t = field("x", "real")
    ff.def_style("urey_bradley", "proof").def_type(
        "A-B", t["A"], t["B"], k_ub=1.0, r_ub=1.0
    )


def _gromacs(tmp: Path) -> None:
    # GROMACS's own units: the refusal is the style's, not the units'.
    ff, t = field("beads", "real")
    ff.def_style("bond", "fene/proof").def_type(
        "B-B", t["B"], t["B"], k=K, r0=R0, epsilon=EPS, sigma=SIG
    )
    mp.io.gromacs.GromacsTopForcefieldWriter().write(tmp / "x.top", ff)


REFUSALS = [
    (
        "unknown function",
        lambda tmp: style_registry.register_style(
            "bond", "x/proof", params={"k": "E"}, expression="k*sinh(r)"
        ),
        ir.UnknownFunctionError,
        {"name": "sinh"},
    ),
    (
        "unbound variable",
        lambda tmp: style_registry.register_style(
            "bond", "x/proof", params={"k": "E"}, expression="k*theta"
        ),
        ir.UnboundVariableError,
        {"style": "x/proof", "name": "theta"},
    ),
    (
        "sealed bond harmonic",
        lambda tmp: style_registry.register_style(
            "bond",
            "harmonic",
            params={"k": "E/L^2", "r0": "L"},
            expression="2*k*(r-r0)^2",
        ),
        ir.SealedError,
        {"category": "bond", "style": "harmonic"},
    ),
    (
        "wrong def_type arity",
        lambda tmp: _wrong_arity(),
        ir.ArityError,
        {"category": "urey_bradley", "arity": 2},
    ),
    (
        "kernel of the wrong shape",
        lambda tmp: _compile_with(_short, "short/proof"),
        ir.KernelShapeError,
        {"style": "short/proof"},
    ),
    (
        "kernel raising",
        lambda tmp: _compile_with(_raising, "raising/proof"),
        ir.KernelShapeError,
        {"style": "raising/proof"},
    ),
    (
        "write_top_forcefield",
        _gromacs,
        ir.NoEngineFormError,
        {"engine": "GROMACS", "category": "bond", "style": "fene/proof"},
    ),
    (
        "missing param at compile",
        lambda tmp: price(fene_ff("fene/proof", sigma=None), bead_frame()),
        ir.MissingParamError,
        {"style": "fene/proof", "type": "B-B", "param": "sigma"},
    ),
]


@pytest.mark.parametrize(
    ("what", "act", "variant", "names"), REFUSALS, ids=[r[0] for r in REFUSALS]
)
def test_what_does_not_conform_is_refused_by_name(
    what, act, variant, names, tmp_path
) -> None:
    with pytest.raises(variant) as err:
        act(tmp_path)
    assert isinstance(err.value, ir.IrError) and isinstance(err.value, ValueError)
    assert {k: getattr(err.value, k) for k in names} == names
    if what == "kernel raising":
        assert isinstance(err.value.__cause__, ZeroDivisionError)
    assert "x/proof" not in {s.name for s in style_registry.styles("bond")}


# ---------------------------------------------------------------------------
# An array parameter, priced by a numpy kernel
# ---------------------------------------------------------------------------


def test_an_array_param_style_is_hand_linear_interpolation_and_round_trips(
    tmp_path,
) -> None:
    (info,) = [s for s in style_registry.styles("dihedral") if s.name == "table/linear"]
    assert [(p.name, p.kind, p.rank) for p in info.params] == [("table", "array", 1)]
    ff, frames = everything()
    worst = 0.0
    for xyz in (CHAIN, moved(CHAIN, 0.09)):
        spec = dict(frames["table"], xyz=xyz.tolist())
        e, f = price(ff, spec_frame(spec))
        b1, b2, b3 = xyz[1] - xyz[0], xyz[2] - xyz[1], xyz[3] - xyz[2]
        n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
        phi = math.atan2(
            np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)), np.dot(n1, n2)
        )
        e_ref, de_ref = table_at(np.array([phi]), TABLE[None, :])
        worst = max(worst, abs(e - e_ref[0]) / abs(e_ref[0]))
        # The kernel's dE/dφ, through the chain rule: the force is −∇E.
        h = 1e-6
        fd = []
        for k in range(12):
            up, down = xyz.copy().ravel(), xyz.copy().ravel()
            up[k] += h
            down[k] -= h
            fd.append(
                -(
                    price(ff, spec_frame(dict(spec, xyz=up.reshape(4, 3).tolist())))[0]
                    - price(
                        ff, spec_frame(dict(spec, xyz=down.reshape(4, 3).tolist()))
                    )[0]
                )
                / (2 * h)
            )
        assert rel(f, np.array(fd), np.abs(f).max()) <= 1e-6
    assert worst <= 1e-12
    print(f"MEASURED table/linear vs hand interpolation: {worst:.1e}")
    # Round trip: the column f64[T, N] and the energy, bit for bit.
    path = tmp_path / "table.mrec"
    mp.io.write_mrec(path, forcefield=ff)
    section = mp.io.mrec.MrecFile(path).forcefield()
    column = section.table("dihedral", "table/linear")["table"]
    assert column.dtype == np.float64 and column.shape == (1, len(TABLE))
    assert column.tobytes() == TABLE.tobytes()
    back = section.to_forcefield()
    f0 = spec_frame(frames["table"])
    assert price(back, f0)[0].hex() == price(ff, f0)[0].hex()


# ---------------------------------------------------------------------------
# class2's bond-angle term, three ways
# ---------------------------------------------------------------------------

BA = {"n1": 10.0, "n2": 8.0, "r1": 1.5, "r2": 1.45, "theta0": 105.0}


def test_class2_bond_angle_is_one_energy_three_ways() -> None:
    # The hand value: the middle atom at the origin, arms 1.6 and 1.4 at
    # 100°: E = (10·0.1 − 8·0.05)·(−5°) = −π/60.
    t = math.radians(100)
    hand = np.array(
        [
            [
                [1.6, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.4 * math.cos(t), 1.4 * math.sin(t), 0.0],
            ]
        ]
    )
    want = -math.pi / 60
    worst = 0.0
    for style in ("class2", "class2/np"):
        (e,), _ = style_registry.evaluate("bond_angle", style, x=hand, **BA)
        worst = max(worst, abs(e - want) / abs(want))
    e_np, g_np = bond_angle_kernel(hand, **BA)
    worst = max(worst, abs(e_np[0] - want) / abs(want))
    # Expression = numpy over the chain's terms, energy and gradient.
    x = np.stack([CHAIN[[0, 1, 2]], CHAIN[[1, 2, 3]], moved(CHAIN, 0.09)[[0, 1, 2]]])
    e_x, g_x = style_registry.evaluate("bond_angle", "class2", x=x, **BA)
    e_n, g_n = style_registry.evaluate("bond_angle", "class2/np", x=x, **BA)
    scale = max(np.abs(e_x).max(), np.abs(g_x).max())
    worst = max(worst, rel(e_n, e_x, scale), rel(g_n, g_x, scale))
    assert worst <= 1e-12
    # Compiled, the force is minus the central difference of the energy.
    ff, tp = field("ba", "real")
    ff.def_style("bond_angle", "class2/np").def_type(
        "A-B-B", tp["A"], tp["B"], tp["B"], **BA
    )
    f0 = frame(CHAIN[:3], ["A", "B", "B"], "bond_angles", [(0, 1, 2)], ["A-B-B"])
    _, forces = price(ff, f0)
    h, fd = 1e-6, []
    for k in range(9):
        up, down = CHAIN[:3].copy().ravel(), CHAIN[:3].copy().ravel()
        up[k] += h
        down[k] -= h
        e_up = price(
            ff,
            frame(
                up.reshape(3, 3), ["A", "B", "B"], "bond_angles", [(0, 1, 2)], ["A-B-B"]
            ),
        )[0]
        e_dn = price(
            ff,
            frame(
                down.reshape(3, 3),
                ["A", "B", "B"],
                "bond_angles",
                [(0, 1, 2)],
                ["A-B-B"],
            ),
        )[0]
        fd.append(-(e_up - e_dn) / (2 * h))
    fd_err = rel(forces, np.array(fd), np.abs(forces).max())
    assert fd_err <= 1e-7
    print(
        f"MEASURED bond_angle expression = numpy = hand: {worst:.1e}; F vs FD {fd_err:.1e}"
    )
