# Extending the Force Field

A new interaction style, or a whole new category of terms, is a **registration
from Python**. Nothing is rebuilt: no Rust, no wheel, no writer arm to add.

The force-field IR (molrs's, which adopts the LAMMPS standard) is a
*protocol*. A **category** says how many atoms a term has and which `Frame`
block its terms live in; a **style** says its ordered parameters, each with a
dimension, and its energy, as an expression or as a Python kernel. Anything of
that form registers into the registry every `PotentialCompiler` reads, and from
then on it is typed, priced, saved and read back exactly like a built-in.
molpy keeps no parallel IR: `mp.potential` re-exports molrs's (`mp.potential.StyleSpec
is molrs.ff.ir.StyleSpec`).

## A new style in 30 lines

A Kremer-Grest bead-spring chain: LAMMPS `bond_style fene`, declared by its
expression, and a typifier that types every bead `B` and every bond with it.

```python
import molpy as mp
from molpy.potential import Param, StyleSpec
from molpy.typifier import Match, Typifier

class Fene(StyleSpec):  # LAMMPS bond_style fene, by its expression
    category, name = "bond", "fene"
    params = [Param("k", "E/L^2"), Param("r0", "L"), Param("epsilon", "E"), Param("sigma", "L")]
    expression = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
                  "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")

class BeadSpring(Typifier):  # every bead B, every bond FENE; lj units
    def library(self):
        return mp.ForceField("bead-spring", units="lj")
    def match(self, graph):
        bead = {"type": ("full", "B", (), {"mass": 1.0})}
        spring = {"type": ("fene", "B-B", ("B", "B"),
                           {"k": 30.0, "r0": 1.5, "epsilon": 1.0, "sigma": 1.0})}
        bonds = graph.links.exact_bucket(mp.Bond)
        return Match([bead] * len(graph.atoms), links={mp.Bond: [spring] * len(bonds)},
                     styles=[("atom", "full", {}), ("bond", "fene", {})])

typifier = BeadSpring()
frame = typifier.typify(chain).to_frame()  # chain: an mp.Atomistic of bonded beads
ff = typifier.forcefield()
energy, forces = mp.PotentialCompiler(ff).compile(frame).calc_energy_forces(frame)
mp.io.write_mrec("chain.mrec", frame, forcefield=ff)  # the expression travels along
```

Defining the class registers the style; `Fene.unregister()` takes it out
again. The record carries the expression, so a process that registered
nothing reads `chain.mrec` back with
`mp.ForceField.from_section(mp.io.read_mrec_forcefield("chain.mrec"))` and
prices it identically. molpy's test suite runs this snippet as written
(`tests/test_potential/test_user_style.py`), and proves the energy and forces
equal the analytic FENE sum and that a fresh process prices the record bit for
bit.

## The pieces

| Name (`mp.potential.…`) | What it does |
|---|---|
| `StyleSpec` | Subclass it: `category`, `name`, `params`, and an `expression` and/or a `kernel` method. The subclass statement registers it. |
| `Param(name, dim, *, kind, default, mix, indexed, …)` | One parameter: its name and dimension (`"E/L^2"`; `E` `L` `A` `Q` `M` for energy, length, angle, charge, mass). `params` also takes a `{name: dim}` dict. |
| `register_style(category, name, *, params, expression=None, kernel=None, …)` | The same as a function. |
| `register_category(name, arity, *, coordinate="compound", order="reversible")` | A new category of 2–5 atoms; its terms live in the block `f"{name}s"`. |
| `styles(category=None)`, `categories()`, `evaluate(...)`, `unregister(...)` | Introspection, a style's energy on a batch of coordinates, removal of a custom style. |
| `kernel(category, style, atoms, **params)` | Any style's kernel over explicit instances (atom indices, one parameter row per term), no typifier needed. |
| `IrError` | Every refusal is a subclass of it (a `ValueError`) named after what was refused, on `molrs.ff.ir` (`Sealed`, `NoKernel`, `UnboundVariable`, …). |

**Parameters arrive as stored.** An angle value (dimension `A`) is in degrees,
as the force field stores it, and the expression converts it:
`k*(theta-theta0*0.017453292519943295)^2`. Coordinates (`theta`, `phi`) are
radians.

**Variables** by category: `r` (bond, pair), `theta` (angle), `phi`
(dihedral), `phi` and `chi = abs(phi)` (improper); any category with points
can use `distance(p1,p3)`, `angle(p1,p2,p3)`, `dihedral(p1,p2,p3,p4)`. Pair
expressions also see `q1`, `q2` and the self rows `x1`, `x2` of a parameter
`x`. The grammar is Lepton's: `+ - * / ^`, `exp log sqrt sin cos tan asin acos
atan abs min max step delta select`, definitions after `;`.

**A Python kernel** instead of (or beside) an expression: a method
`kernel(self, q, **params) -> (e, de_dq)` vectorised over the terms, or for a
compound category `kernel(self, x, **params) -> (e, grad)` with `x` of shape
`(n, arity, 3)`. With both, they must agree; registration checks the
derivative against a central difference.

## A new category

A Urey-Bradley 1-3 spring as its own category, typed through a custom relation
kind of the graph:

```python
mp.potential.register_category("urey_bradley", 3)


class UreyBradley(mp.potential.StyleSpec):
    category, name = "urey_bradley", "harmonic"
    params = {"k_ub": "E/L^2", "r_ub": "L"}
    expression = "k_ub*(distance(p1,p3)-r_ub)^2"
```

A typifier's `Match` then keys its rows by the kind name
(`links={"urey_bradleys": rows}`, after `graph.register_kind("urey_bradleys", 3)`),
and the terms land in the frame's `urey_bradleys` block.

## Engines

Exporting to an engine is molrs's job, and molpy adds no formatter of its
own: molpy's LAMMPS emitter (`mp.io.emit`) and `LAMMPSEngine` take every
`*_style` line and coefficient from molrs's LAMMPS writer
(`mp.io.write_lammps_forcefield`), so they write whatever it can write —
`hybrid` styles and `angle charmm` with its Urey-Bradley term included — and a
style an engine cannot hold is refused by molrs, by name, never written
half-formed. The `.mrec` record is the format that always holds a registered
style: its expression travels with it.

## Checklist

- [ ] The style declares its parameters in LAMMPS `*_coeff` order, each with
      its dimension
- [ ] An expression, a kernel, or both (then they agree)
- [ ] A test: energy and forces against a hand formula, and a `.mrec`
      round trip if the style is to be saved
