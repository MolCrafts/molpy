"""Custom force-field IR styles persist in ``.mrec`` (``ff-ir-02-protocol`` §7).

A force field holding styles and categories molrs does not ship is written
to a record; a **fresh Python process** that registered nothing reads it
back. There:

* an expression style (``bond fene``) and a custom category priced by its
  expression (``urey_bradley``, ``k_ub*(distance(p1,p3)-r_ub)^2``) price the
  same energy and forces, bit for bit, and keep the expression byte for byte;
* a style with no expression (native-only where it was made) is read whole,
  and compiling it is refused by name — "no kernel for … register it
  (molpy.ff.style_registry.register_style) or give it an expression";
* an array parameter (``dihedral table/linear``, ``table: f64[N]``) comes back
  bit for bit, through ``mp.io.mrec`` as an ``f64[T, N]`` column;
* a category nothing ever registers is kept;
* a custom style registered here with ``mp.ff.style_registry.register_style`` and an
  expression, its instances carrying none, is written with the registry's
  expression (D16), so the fresh process prices it all the same.

The custom category and style are registered through ``mp.ff.ir``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import molpy as mp
import numpy as np
import pytest
from ff_ir_persist_child import exact, price

FENE = (
    "-0.5*k*r0^2*log(1-(r/r0)^2) + "
    "step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)"
)
UB = "k_ub*(distance(p1,p3)-r_ub)^2"
CHILD = Path(__file__).with_name("ff_ir_persist_child.py")
NO_KERNEL = (
    "no kernel for {} `{}`: register it (molpy.ff.style_registry.register_style) "
    "or give it an expression"
)

# This process's registry gains the category, and a bond style priced by
# the registry's expression alone (its instances state none).
mp.ff.style_registry.register_category("urey_bradley", 3)
mp.ff.style_registry.register_style(
    "bond",
    "fene/registered",
    params={"k": "E/L^2", "r0": "L", "epsilon": "E", "sigma": "L"},
    expression=FENE,
)

TABLE = np.array([1.25 + np.sin(0.7 * i) / 3.0 - 0.01 * i * i for i in range(12)])
UB_ROWS = ([[0, 1, 2], [1, 2, 3]], ["t", "u"])


def _atom(ff: mp.ff.forcefield.ForceField) -> mp.ff.forcefield.AtomType:
    return ff.def_style("atom", "full").def_type("A", mass=12.0)


def _fene() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("fene")
    a = _atom(ff)
    ff.def_style("bond", "fene", {"expression": FENE}).def_type(
        "t", a, a, k=30.0, r0=2.25, epsilon=1.1, sigma=1.4
    )
    return ff


def _fene_registered() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("fene")
    a = _atom(ff)
    ff.def_style("bond", "fene/registered").def_type(
        "t", a, a, k=30.0, r0=2.25, epsilon=1.1, sigma=1.4
    )
    return ff


def _ub(style: str, expression: str | None) -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("ub")
    a = _atom(ff)
    params = {} if expression is None else {"expression": expression}
    s = ff.def_style("urey_bradley", style, params)
    for name, k_ub, r_ub in (("t", 20.0, 2.45), ("u", 11.0, 2.2)):
        s.def_type(name, a, a, a, k_ub=k_ub, r_ub=r_ub)
    return ff


def _table() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("table")
    a = _atom(ff)
    style = ff.def_style("dihedral", "table/linear")
    style.def_type("t", a, a, a, a, table=TABLE)
    style.def_type("u", a, a, a, a, table=TABLE[::-1].copy())
    return ff


def _bespoke() -> mp.io.mrec.ForceFieldSection:
    """A record whose ``bespoke`` category nothing registers in any process:
    the native-only ``urey_bradley`` force field with its category renamed."""
    section = mp.io.mrec.ForceFieldSection.from_forcefield(_ub("spring", None))
    document, tables = section.document, section.tables
    (entry,) = [s for s in document["styles"] if s["category"] == "urey_bradley"]
    entry["category"] = "bespoke"
    block = mp.io.mrec.ForceFieldSection.block_name
    tables[block("bespoke", "spring")] = tables.pop(block("urey_bradley", "spring"))
    return mp.io.mrec.ForceFieldSection(document, tables)


@pytest.fixture(scope="module")
def fresh(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, dict]:
    """Every record written here, and what a fresh process makes of it."""
    tmp = tmp_path_factory.mktemp("persist")
    records = {
        "fene": (
            mp.io.mrec.ForceFieldSection.from_forcefield(_fene()),
            "bonds",
            [[0, 1]],
            ["t"],
        ),
        "fene_registered": (
            mp.io.mrec.ForceFieldSection.from_forcefield(_fene_registered()),
            "bonds",
            [[0, 1]],
            ["t"],
        ),
        "ub_expr": (
            mp.io.mrec.ForceFieldSection.from_forcefield(_ub("spring", UB)),
            "urey_bradleys",
            *UB_ROWS,
        ),
        "ub_native": (
            mp.io.mrec.ForceFieldSection.from_forcefield(_ub("harmonic", None)),
            "urey_bradleys",
            *UB_ROWS,
        ),
        "table": (
            mp.io.mrec.ForceFieldSection.from_forcefield(_table()),
            "dihedrals",
            [[0, 1, 2, 3]],
            ["t"],
        ),
        "bespoke": (_bespoke(), "bespokes", *UB_ROWS),
    }
    here_out, cases = {}, {}
    for name, (section, block, rows, types) in records.items():
        path = tmp / f"{name}.mrec"
        mp.io.write_mrec(path, forcefield=section)
        ff = section.to_forcefield()
        here_out[name] = {
            "price": price(ff, block, rows, types),
            "section": exact(section),
        }
        cases[name] = [str(path), block, rows, types]
    done = subprocess.run(
        [sys.executable, str(CHILD), json.dumps(cases)],
        capture_output=True,
        text=True,
        check=False,
        encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr
    return here_out, json.loads(done.stdout)


def test_the_fresh_process_registered_nothing() -> None:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import molpy as mp\n"
                "try:\n"
                "    mp.ff.forcefield.ForceField('t').def_style('urey_bradley', 'x')\n"
                "except ValueError:\n"
                "    print('unknown')"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    assert done.stdout.strip() == "unknown"


@pytest.mark.parametrize("name", ["fene", "ub_expr", "fene_registered"])
def test_an_expression_style_prices_the_same_bits_in_a_fresh_process(
    fresh: tuple[dict, dict], name: str
) -> None:
    here_out, there = fresh
    assert "error" not in here_out[name]["price"], here_out[name]["price"]
    assert float.fromhex(here_out[name]["price"]["e"]) > 0.0
    assert there[name]["price"] == here_out[name]["price"]


def test_the_expression_is_kept_byte_for_byte(fresh: tuple[dict, dict]) -> None:
    _, there = fresh
    styles = {
        s["style"]: s.get("expression")
        for name in ("fene", "ub_expr")
        for s in there[name]["section"]["document"]["styles"]
    }
    assert styles["fene"] == FENE
    assert styles["spring"] == UB


def test_a_registered_style_is_written_with_the_registry_expression(
    fresh: tuple[dict, dict],
) -> None:
    """D16: the instance states no expression; the record carries the
    registry's, so a process that registered nothing prices it the same."""
    style = _fene_registered().get_style("bond", "fene/registered")
    assert "expression" not in style.params
    here_out, there = fresh
    (entry,) = [
        s
        for s in here_out["fene_registered"]["section"]["document"]["styles"]
        if s["style"] == "fene/registered"
    ]
    assert entry["expression"] == FENE
    assert there["fene_registered"]["price"] == here_out["fene_registered"]["price"]
    # One force law: the same bits as the instance-expression `fene`.
    assert here_out["fene_registered"]["price"] == here_out["fene"]["price"]


@pytest.mark.parametrize(
    ("name", "category", "style"),
    [
        ("ub_native", "urey_bradley", "harmonic"),
        ("table", "dihedral", "table/linear"),
        ("bespoke", "bespoke", "spring"),
    ],
)
def test_a_style_without_an_expression_is_read_whole_and_refused_by_name(
    fresh: tuple[dict, dict], name: str, category: str, style: str
) -> None:
    here_out, there = fresh
    assert [category, style] in there[name]["styles"]
    # Read whole: the section a fresh process writes back is the one written.
    assert there[name]["section"] == here_out[name]["section"]
    assert there[name]["price"] == {"error": NO_KERNEL.format(category, style)}


def test_an_array_param_round_trips_bit_for_bit_as_an_f64_t_n_column(
    fresh: tuple[dict, dict], tmp_path: Path
) -> None:
    here_out, there = fresh
    block = mp.io.mrec.ForceFieldSection.block_name("dihedral", "table/linear")
    shape, values = there["table"]["section"]["tables"][block]["table"]
    assert shape == [2, 12]
    want = np.stack([TABLE, TABLE[::-1]])
    assert values == [float(x).hex() for x in want.ravel()]
    assert there["table"]["section"] == here_out["table"]["section"]

    # Through mp.io.mrec in this process too: the section, and the types.
    path = tmp_path / "table.mrec"
    mp.io.write_mrec(path, forcefield=_table())
    section = mp.io.mrec.MrecFile(path).forcefield()
    assert section is not None
    column = section.table("dihedral", "table/linear")["table"]
    assert column.dtype == np.float64 and column.shape == (2, 12)
    assert column.tobytes() == want.tobytes()
    back = section.to_forcefield()
    t = back.get_style("dihedral", "table/linear").get_type_by_name("t")
    assert np.asarray(t["table"]).tobytes() == TABLE.tobytes()
