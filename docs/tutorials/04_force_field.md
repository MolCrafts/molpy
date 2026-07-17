# Force Field

A force field in MolPy is data you can query and validate — long before it becomes numbers in an engine file.

## Why separate structure from parameters?

In classical molecular dynamics, the physics is entirely determined by the force field — the set of equations and numerical parameters that define how atoms interact. Getting these parameters wrong is the most common source of silent errors in simulation workflows. A misassigned atom type or a missing dihedral term does not crash the program; it produces wrong results that look plausible.

Most simulation tools bundle structure and parameters together in a single file (LAMMPS data + coefficients, GROMACS topology). This makes it hard to inspect or compare parameter assignments before running an expensive simulation. MolPy keeps them separate so you can validate the model before committing to a computation.


## Parameters first, execution later

Many libraries collapse force-field definition and execution into a single layer. You set a parameter and immediately get a numerical object, with no intermediate state you can inspect or validate. MolPy separates the two steps deliberately.

**A force field should remain inspectable as data before it is compiled into executable numerical objects.**

This matters because parameterization is where silent mistakes become expensive. If an atom type is wrong, a key is missing, or a bond parameter is inconsistent, you want to find out while the model is still a transparent data structure — not after it has been baked into arrays or engine-specific files.


## The three layers: Style, Type, Potential

MolPy organizes force field data in three nested layers:

```text
ForceField
├── AtomStyle "full"
│   ├── AtomType "CT"  (mass=12.011, charge=-0.18)
│   └── AtomType "HC"  (mass=1.008, charge=0.06)
├── BondStyle "harmonic"
│   ├── BondType "CT-HC"  (k=340.0, r0=1.09)
│   └── BondType "CT-CT"  (k=268.0, r0=1.529)
├── AngleStyle "harmonic"
│   └── AngleType "HC-CT-HC"  (k=33.0, theta0=107.8)
├── DihedralStyle "opls"
│   └── DihedralType "HC-CT-CT-HC"  (K1=0.0, K2=0.0, K3=0.3, K4=0.0)
└── PairStyle "lj126/cut"
    ├── PairType "CT"  (epsilon=0.066, sigma=3.50)
    └── PairType "HC"  (epsilon=0.030, sigma=2.50)
```

A `Style` defines an interaction family — harmonic bonds, OPLS dihedrals, Lennard-Jones pairs — and its parameter contract. A `Type` is one concrete parameter record inside that family. The `Potentials` evaluator is the numerical realization, produced from the complete model and run against a typed `Frame`. The kernels themselves live in the molrs Rust extension.

The progression is always: define styles → fill in types → evaluate as potentials.


## Building a minimal force field

Start by creating a `ForceField` and defining atom types. Atom types form the foundation — every bonded or nonbonded interaction references them.

```python
import molrs
import molpy as mp

ff = mp.AtomisticForcefield(name="tutorial", units="real")

# "full" corresponds to LAMMPS atom_style full (charge + molecule ID per atom)
atom_style = ff.def_atomstyle("full")
ct = atom_style.def_type("CT", mass=12.011, charge=-0.18, element="C")
hc = atom_style.def_type("HC", mass=1.008,  charge=0.06,  element="H")
oh = atom_style.def_type("OH", mass=15.999, charge=-0.68, element="O")
```

Bond, angle, dihedral, and pair styles follow the same pattern: create the style, then add types with explicit parameter names.

```python
bond_style = ff.def_bondstyle("harmonic")
bond_style.def_type(ct, hc, k=340.0, r0=1.09)
bond_style.def_type(ct, ct, k=268.0, r0=1.529)
bond_style.def_type(ct, oh, k=320.0, r0=1.41)

angle_style = ff.def_anglestyle("harmonic")
angle_style.def_type(hc, ct, hc, k=33.0, theta0=107.8)

dihedral_style = ff.def_dihedralstyle("opls")
dihedral_style.def_type(hc, ct, ct, hc, K1=0.0, K2=0.0, K3=0.3, K4=0.0)

# "lj126/cut" = 12-6 Lennard-Jones with cutoff (LAMMPS: lj/cut)
pair_style = ff.def_pairstyle("lj126/cut")
pair_style.def_type(ct, epsilon=0.066, sigma=3.50)
pair_style.def_type(hc, epsilon=0.030, sigma=2.50)
pair_style.def_type(oh, epsilon=0.170, sigma=3.12)
```

At this point the force field is a complete data structure. No numerical kernel has been created yet. Everything is still readable and editable.


## Inspecting the model

Before any export, inspect the force field as data. A file can be syntactically valid and still contain wrong parameters.

Individual types expose their parameters through dictionary access.

```python
print(f"CT mass={ct['mass']}, charge={ct['charge']}")
print(f"CT element={ct.get('element')}")

bt = bond_style.get_type_by_name("CT-OH")
print(f"CT-OH: k={bt['k']}, r0={bt['r0']}")
```

A full listing of all styles and types gives a global snapshot of the model state.

```python
from molpy.core.forcefield import Style, Type

for style in ff.get_styles(Style):
    types = style.get_types(Type)
    print(f"style={style.name!r}  [{len(types)} types]")
    for t in types:
        params = {k: v for k, v in t.params.kwargs.items()}
        print(f"  {t.name}: {params}")
```

Name-based lookup targets a specific style or type directly.

```python
bs = ff.get_style("bond", "harmonic")
ct_ct = bs.get_type_by_name("CT-CT")
print(f"CT-CT k={ct_ct['k']}")
```


## Evaluating as Potentials

Evaluation is the first strict integrity test of the model. `ff.to_potentials()`
returns a *deferred* `Potentials` — it carries no frame yet (`len() == 0`, not
iterable). To compute numbers you pass a typed `Frame`: an `atoms` block with
coordinates plus a bonded block (`bonds`, `angles`, …) carrying a `type` column.
The numerical kernels run in the molrs Rust extension.

```python
import numpy as np

# A minimal frame: two atoms 1.2 Å apart joined by one CT-HC bond.
frame = molrs.Frame()
atoms = molrs.Block()
atoms.insert("x", np.array([0.0, 1.2]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms

bonds = molrs.Block()
bonds.insert("atomi", np.array([0], dtype=np.uint32))
bonds.insert("atomj", np.array([1], dtype=np.uint32))
bonds.insert("type", np.array(["CT-HC"], dtype=str))
frame["bonds"] = bonds

pots = ff.to_potentials()
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
print(f"energy = {energy}")
print(f"forces =\n{forces}")
```

If a referenced type is missing or a required parameter is absent, evaluation
raises here rather than producing a plausible-but-wrong number.


## Exporting to simulation engines

Once the model is internally consistent, serialization becomes an interface problem rather than a modeling problem. The same force field can be rendered into different engine formats without redefining the physics.

### LAMMPS

```python
import io
from molpy.io.forcefield import LAMMPSForceFieldWriter

buf = io.StringIO()
writer = LAMMPSForceFieldWriter(buf, precision=4)
writer.write(ff)
print(buf.getvalue())
```

### GROMACS

```python
from molpy.io.forcefield.top import GromacsForceFieldWriter

GromacsForceFieldWriter("system.itp", precision=4).write(ff)
```

### XML

```python
from molpy.io.forcefield import XMLForceFieldWriter

XMLForceFieldWriter("system.xml", precision=6).write(ff)
```


## When to move beyond built-in styles

Real projects eventually need interaction forms not covered by built-in styles — Morse bonds, Buckingham pairs, custom torsion profiles. The numerical kernel for a new form is added in the molrs Rust extension; on the Python side you expose a thin named `Style` and register parameter formatters for each export backend.

See [Extending Force Field](../developer/extending-forcefield.md) for the full extension recipe.


## The force field is not inside the molecule

One more distinction worth making explicit: structure and parameterization are related but separate. A molecule can exist before it is typed. A typed system can exist before the force field is exported. MolPy preserves those boundaries because it makes model validation and format conversion much easier to reason about.

See also: [Atomistic and Topology](01_atomistic_and_topology.md), [Block and Frame](02_block_and_frame.md).
