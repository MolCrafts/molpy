"""A user's own force-field style, from molpy, with nothing rebuilt.

The proof of the force-field IR as a protocol at the molpy layer
(``ff-ir-02-protocol``, P-molpy): the snippet of
``docs/developer/extending-forcefield.md`` is run as written. It declares
LAMMPS ``bond_style fene`` by its expression (``class Fene(StyleDeclaration)``),
types a bead chain with a ``BeadSpring`` typifier (lj units) whose ``assign``
returns ``TypeAssignment(nodes, links={Bond: rows}, styles=[...])``, compiles the
typed frame and writes it with its force field to ``.mrec``. Then:

* the energy and forces are the analytic FENE sum (rel 1e-12 / 1e-10);
* the record holds the style and its expression, byte for byte;
* a fresh process that registered nothing prices the record bit for bit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import molrs
import numpy as np
import pytest

import molpy as mp

DOC = Path(__file__).parents[2] / "docs" / "developer" / "extending-forcefield.md"
K, R0, EPS, SIG = 30.0, 1.5, 1.0, 1.0
# Bonds of 0.97, 1.06, 1.18 and 1.31: both sides of the WCA cutoff
# 2^(1/6) sigma = 1.1225, and (r/R0)^2 < 0.9 everywhere.
CHAIN = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.97, 0.0, 0.0],
        [0.97, 1.06, 0.0],
        [0.97, 1.06, 1.18],
        [0.97 + 1.31 * 0.6, 1.06 + 1.31 * 0.8, 1.18],
    ]
)


def _snippet() -> str:
    """The doc's first ``python`` block under "A new style in 30 lines"."""
    text = DOC.read_text()
    start = text.index("```python\n", text.index("## A new style in 30 lines"))
    start += len("```python\n")
    return text[start : text.index("```", start)]


def _chain() -> mp.Atomistic:
    mol = mp.Atomistic()
    beads = [mol.def_atom(name="B", x=x, y=y, z=z) for x, y, z in CHAIN]
    for a, b in zip(beads, beads[1:]):
        mol.def_bond(a, b)
    return mol


def _fene(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """LAMMPS ``bond_style fene`` energy and dE/dr, by hand."""
    x = (r / R0) ** 2
    e = -0.5 * K * R0**2 * np.log(1 - x)
    de = K * r / (1 - x)
    wca = r < 2 ** (1 / 6) * SIG
    s6 = (SIG / r) ** 6
    e = e + np.where(wca, 4 * EPS * (s6 * s6 - s6) + EPS, 0.0)
    de = de + np.where(wca, 4 * EPS * (-12 * s6 * s6 + 6 * s6) / r, 0.0)
    return e, de


@pytest.fixture(scope="module")
def snippet(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The doc snippet's namespace, run in a scratch directory; its style is
    unregistered afterwards."""
    workdir = tmp_path_factory.mktemp("user_style")
    namespace: dict[str, Any] = {"chain": _chain()}
    here = Path.cwd()
    os.chdir(workdir)
    try:
        exec(compile(_snippet(), str(DOC), "exec"), namespace)
    finally:
        os.chdir(here)
    yield namespace, workdir
    namespace["Fene"].unregister()


def test_the_doc_snippet_is_under_30_lines() -> None:
    assert len(_snippet().splitlines()) < 30


def test_the_style_is_registered_from_molpy(snippet) -> None:
    ns, _ = snippet
    assert mp.ff.ir.StyleDeclaration is molrs.ff.ir.StyleDeclaration
    (info,) = [s for s in mp.ff.ir.styles("bond") if s.name == "fene"]
    assert info.expression == ns["Fene"].expression
    assert not info.builtin
    assert [p.name for p in info.params] == ["k", "r0", "epsilon", "sigma"]


def test_the_chain_is_typed_with_the_style(snippet) -> None:
    ns, _ = snippet
    frame, ff = ns["frame"], ns["ff"]
    assert frame["bonds"]["type"].tolist() == ["B-B"] * 4
    assert ff.units == "lj"
    assert [(s.category, s.name) for s in ff.styles] == [
        ("atom", "full"),
        ("bond", "fene"),
    ]


def test_energy_and_forces_are_the_analytic_fene_sum(snippet) -> None:
    ns, _ = snippet
    d = CHAIN[1:] - CHAIN[:-1]
    r = np.linalg.norm(d, axis=1)
    e, de = _fene(r)
    forces = np.zeros_like(CHAIN)
    pull = (de / r)[:, None] * d  # -dE/dx of the first atom of each bond
    forces[:-1] += pull
    forces[1:] -= pull
    assert ns["energy"] == pytest.approx(e.sum(), rel=1e-12)
    scale = np.abs(forces).max()
    np.testing.assert_allclose(ns["forces"], forces, rtol=0, atol=1e-10 * scale)


def test_the_record_keeps_the_style_and_its_expression(snippet) -> None:
    ns, workdir = snippet
    section = mp.io.read_mrec_forcefield(workdir / "chain.mrec")
    (entry,) = [s for s in section.document["styles"] if s["style"] == "fene"]
    assert entry["category"] == "bond"
    assert entry["expression"] == ns["Fene"].expression
    ff = section.to_forcefield()
    (style,) = ff.get_styles("bond")
    assert style.name == "fene"
    assert [t.name for t in style.get_types()] == ["B-B"]


FRESH = textwrap.dedent(
    """
    import json, sys
    import molpy as mp

    path = sys.argv[1]
    registered = [s.name for s in mp.ff.ir.styles("bond") if s.name == "fene"]
    frame = mp.io.read_mrec_frame(path)
    ff = mp.io.read_mrec_forcefield(path).to_forcefield()
    e, f = mp.ff.potential.PotentialCompiler(ff).compile(frame).calc_energy_forces(frame)
    print(json.dumps({"registered": registered, "e": float(e).hex(),
                      "f": [float(x).hex() for x in f.ravel()]}))
    """
)


def test_a_fresh_process_that_registered_nothing_prices_it_bit_for_bit(snippet) -> None:
    ns, workdir = snippet
    done = subprocess.run(
        [sys.executable, "-c", FRESH, str(workdir / "chain.mrec")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    there = json.loads(done.stdout)
    assert there["registered"] == []
    assert there["e"] == float(ns["energy"]).hex()
    assert there["f"] == [float(x).hex() for x in np.ravel(ns["forces"])]
