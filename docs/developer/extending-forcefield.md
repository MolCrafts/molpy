# Extending the Force Field

A new interaction style, or a whole new category of terms, is a **registration
from Python**. Nothing is rebuilt: no Rust, no wheel, no writer arm to add.

The force-field IR (the Rust core's, which adopts the LAMMPS standard) is a
*protocol*. A **category** says how many atoms a term has and which `Frame`
block its terms live in; a **style** says its ordered parameters, each with a
dimension, and its energy, as an expression or as a Python kernel. Anything of
that form registers into the registry every `PotentialCompiler` reads, and from
then on it is typed, priced, saved and read back exactly like a built-in.
There is no parallel IR in Python: `mp.ff` is the core's force-field
subsystem, one submodule per Rust owner (`mp.ff.ir.StyleSpec` is a
registered style's record).
The IR's vocabulary (`ParamSpec`, `StyleSpec`, `CategorySpec`, the refusals)
is `mp.ff.ir`; the registry, `GlobalRegistry`, is `mp.ff.style_registry`; the
compiler is
`mp.ff.PotentialCompiler`; the kernels are `mp.ff.potential`; and the force-field
container is `mp.ff.forcefield`.

## A new style in 30 lines

A Kremer-Grest bead-spring chain: LAMMPS `bond_style fene`, declared by its
expression, and a typifier that types every bead `B` and every bond with it.

```python
from pathlib import Path
import molpy as mp
from molpy.ff.forcefield import ForceField
from molpy.ff import PotentialCompiler
from molpy.ff.ir import ParamSpec
from molpy.ff.style_registry import GlobalRegistry
from molpy.ff.typifier import TypeAssignment, Typifier

registry = GlobalRegistry()
registry.register_style(  # LAMMPS bond_style fene, by its expression
    "bond", "fene",
    params=[ParamSpec("k", "E/L^2"), ParamSpec("r0", "L"), ParamSpec("epsilon", "E"), ParamSpec("sigma", "L")],
    expression=("-0.5*k*r0^2*log(1-(r/r0)^2)"
                "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)"),
    lammps="positional",  # written as bond_style fene, coefficients in this order
)

class BeadSpring(Typifier):  # every bead B, every bond FENE; lj units
    def source_forcefield(self):
        return ForceField("bead-spring", units="lj")
    def assign(self, graph):
        bead = {"type": ("full", "B", (), {"mass": 1.0})}
        spring = {"type": ("fene", "B-B", ("B", "B"),
                           {"k": 30.0, "r0": 1.5, "epsilon": 1.0, "sigma": 1.0})}
        bonds = graph.bonds
        return TypeAssignment([bead] * len(graph.atoms), links={mp.Bond: [spring] * len(bonds)},
                     styles=[("atom", "full", {}), ("bond", "fene", {})])

chain = mp.Atomistic()  # ten bonded beads (atomic number 0: no element)
beads = [chain.add_atom(atomic_number=0, x=0.97 * i, y=0.0, z=0.0) for i in range(10)]
for a, b in zip(beads, beads[1:]):
    chain.add_bond(a, b)

typifier = BeadSpring()
frame = typifier.typify(chain).to_frame()
ff = typifier.forcefield()
energy, forces = PotentialCompiler(ff).compile(frame).evaluate(frame)
mp.io.write_mrec(Path("chain.mrec"), frame=frame, forcefield=ff)
```

`register_style` registers the style; `registry.unregister_style("bond", "fene")`
takes it out again. The record carries the expression, so a process that registered
nothing reads `chain.mrec` back with
`mp.io.mrec.MrecFile(Path("chain.mrec")).forcefield().to_forcefield()` and
prices it identically.

## The pieces

| Name (`GlobalRegistry().…` from `mp.ff.style_registry`; `ParamSpec` and `IrError` on `mp.ff.ir`, `ExplicitTerms` on `mp.ff`) | What it does |
|---|---|
| `register_style(category, name, *, params, expression=None, kernel=None, …)` | Register a style: its ordered parameters, and an expression and/or a Python kernel. |
| `ParamSpec(name, dim, *, kind, default, mix, indexed, …)` | One parameter: its name and dimension (`"E/L^2"`; `E` `L` `A` `Q` `M` for energy, length, angle, charge, mass). `params` also takes a `{name: dim}` dict. |
| `register_category(name, arity, *, coordinate="compound", order="reversible")` | A new category of 2–5 atoms; its terms live in the block `f"{name}s"`. |
| `styles(category=None)`, `categories()`, `evaluate(...)`, `unregister_style(category, name)` | Introspection (`styles` returns `StyleSpec` records, `categories` `CategorySpec` records), a style's energy on a batch of coordinates, removal of a custom style. |
| `ExplicitTerms(category, style, atoms, **params).compile()` | Any style's kernel over explicit instances (atom indices, one parameter row per term), no typifier needed. |
| `IrError` | Every refusal is a subclass of it (a `ValueError`) named after what was refused (`SealedError`, `NoKernelError`, `UnboundVariableError`, …). |

**Parameters arrive as stored.** An angle value (dimension `A`) is in degrees,
as the force field stores it, and the expression converts it with
`(pi/180)`, defining `pi` last (Lepton has no named constants):
`k*(theta-theta0*(pi/180))^2; pi=3.141592653589793`. Coordinates (`theta`,
`phi`) are radians.

**Variables** by category: `r` (bond, pair), `theta` (angle), `phi`
(dihedral), `phi` and `chi = abs(phi)` (improper); any category with points
can use `distance(p1,p3)`, `angle(p1,p2,p3)`, `dihedral(p1,p2,p3,p4)`. Pair
expressions also see `q1`, `q2` and the self rows `x1`, `x2` of a parameter
`x`. The grammar is Lepton's: `+ - * / ^`, `exp log sqrt sin cos tan asin acos
atan abs min max step delta select`, definitions after `;`.

**A Python kernel** instead of (or beside) an expression: a function
`kernel(q, **params) -> (e, de_dq)` vectorised over the terms, or for a
compound category `kernel(x, **params) -> (e, grad)` with `x` of shape
`(n, arity, 3)`. With both, they must agree; registration checks the
derivative against a central difference.

## A new category

A Urey-Bradley 1-3 spring as its own category, typed through a custom relation
kind of the graph:

```python
registry = mp.ff.style_registry.GlobalRegistry()
registry.register_category("urey_bradley", 3)
registry.register_style(
    "urey_bradley",
    "harmonic",
    params={"k_ub": "E/L^2", "r_ub": "L"},
    expression="k_ub*(distance(p1,p3)-r_ub)^2",
)
```

A typifier's `TypeAssignment` then keys its rows by the kind name
(`links={"urey_bradleys": rows}`, after `graph.declare_term("urey_bradleys", 3)`),
and the terms land in the frame's `urey_bradleys` block.

## Engines

Exporting to an engine is the Rust core's job, and Python adds no formatter of its
own: `LammpsEngine.generate_inputs` (and the `minimize` / `md` runs built on
it) takes every `*_style` line and coefficient from the core's LAMMPS writer
(`mp.io.lammps.LammpsForcefieldWriter`), so it writes whatever that can write —
`hybrid` styles and `angle charmm` with its Urey-Bradley term included — and a
style an engine cannot hold is refused by the core, by name, never written
half-formed. The `.mrec` record is the format that always holds a registered
style: its expression travels with it.

## Checklist

- [ ] The style declares its parameters in LAMMPS `*_coeff` order, each with
      its dimension
- [ ] An expression, a kernel, or both (then they agree)
- [ ] A test: energy and forces against a hand formula, and a `.mrec`
      round trip if the style is to be saved
