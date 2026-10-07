# Force Field

A wrong atom type will not always crash the run — it will produce a plausible
wrong trajectory. How do you inspect the parameters *before* they become engine
arrays?

**In MolPy a force field is data you can query and validate**, layered as styles,
types, and potentials, long before anything is compiled for LAMMPS or OpenMM.

What it is **not**: the molecular graph (that is `Atomistic`), and not the
numeric forces on a frame (that comes after typing and evaluation).

## Why separate structure from parameters?

Classical MD is entirely defined by the force field: functional forms and the
numbers that go with them. Most tools bury those numbers inside a single
topology or data file, so you cannot easily compare assignments or catch a
missing dihedral until after an expensive run.

MolPy keeps structure and parameters apart on purpose. If a type is wrong or a
key is missing, you want that while the model is still transparent data — not
after it is baked into engine-specific arrays.

## The three layers: Style, ForceFieldType, Potential

Force-field data nests in three layers:

```text
ForceField
├── AtomStyle "full"
│   ├── AtomType "CT"  (mass=12.011, charge=-0.18)
│   └── AtomType "HC"  (mass=1.008, charge=0.06)
├── BondStyle "harmonic"
│   ├── BondType "CT-HC"  (k=340.0, r0=1.09)
│   └── BondType "CT-CT"  (k=268.0, r0=1.529)
├── AngleStyle "harmonic"
│   └── AngleType "HC-CT-HC"  (k=33.0, theta0=107.8°)
├── DihedralStyle "opls"
│   └── DihedralType "HC-CT-CT-HC"  (k1=0.0, k2=0.0, k3=0.3, k4=0.0)
└── PairStyle "lj/cut"
    ├── PairType "CT"  (epsilon=0.066, sigma=3.50)
    └── PairType "HC"  (epsilon=0.030, sigma=2.50)
```

A `Style` defines an interaction family — harmonic bonds, OPLS dihedrals, Lennard-Jones pairs — and its parameter contract. A `ForceFieldType` (`AtomType`, `BondType`, …) is one concrete parameter record inside that family. The `Potentials` evaluator is the numerical realization, produced from the complete model and run against a typed `Frame`. The kernels themselves live in the native Rust core.

The progression is always: define styles → fill in types → evaluate as potentials.


## Building a minimal force field

Start by creating a `ForceField` and defining atom types. Atom types form the foundation — every bonded or nonbonded interaction references them.

`ff.def_style(category, name)` defines a style and returns its handle; the
handle's `def_type` defines one type and returns that type's handle.
Parameters are keywords: numbers go to the numeric parameters, strings
(`element`, …) to the string ones.

```python
import molpy as mp

ff = mp.ff.forcefield.ForceField(name="tutorial", units="real")

# "full" corresponds to LAMMPS atom_style full (charge + molecule ID per atom)
atom_style = ff.def_style("atom", "full")
ct = atom_style.def_type("CT", mass=12.011, charge=-0.18, element="C")
hc = atom_style.def_type("HC", mass=1.008, charge=0.06, element="H")
oh = atom_style.def_type("OH", mass=15.999, charge=-0.68, element="O")
```

Bond, angle, dihedral, and pair styles follow the same pattern, with one
addition: a type between atoms is given its **endpoints** — the atom-type
handles it connects — right after its name. The name is just a name: building
it from the endpoints (`"CT-HC"`) is a convention, and it is the label a typed
`Frame` uses, but molpy never reads endpoints out of it. Parameters are as the
force-field IR stores them, and the IR adopts the LAMMPS standard: each style's
energy, factors and parameter units are its LAMMPS style's, so a harmonic bond
is `E = k(r − r₀)²` (no ½, `k` is LAMMPS's `K`) and every angle-valued
parameter is in degrees.

```python
bond_style = ff.def_style("bond", "harmonic")
bond_style.def_type("CT-HC", ct, hc, k=340.0, r0=1.09)
bond_style.def_type("CT-CT", ct, ct, k=268.0, r0=1.529)
bond_style.def_type("CT-OH", ct, oh, k=320.0, r0=1.41)

angle_style = ff.def_style("angle", "harmonic")
angle_style.def_type("HC-CT-HC", hc, ct, hc, k=33.0, theta0=107.8)

dihedral_style = ff.def_style("dihedral", "opls")
dihedral_style.def_type("HC-CT-CT-HC", hc, ct, ct, hc, k1=0.0, k2=0.0, k3=0.3, k4=0.0)

# 12-6 Lennard-Jones (LAMMPS: lj/cut); one self pair per atom type
pair_style = ff.def_style("pair", "lj/cut")
pair_style.def_type("CT", ct, epsilon=0.066, sigma=3.50)
pair_style.def_type("HC", hc, epsilon=0.030, sigma=2.50)
pair_style.def_type("OH", oh, epsilon=0.170, sigma=3.12)

# The atom types carry charges: their Coulomb term is a pair style too.
ff.def_style("pair", "coul/cut")
```

`pair_style.def_type(name, itom)` with no second atom type is the self pair of
`itom`; `pair_style.def_type(name, itom, jtom)` is an explicit cross pair.

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
from molpy.ff.forcefield import ForceFieldType, Style

for style in ff.get_styles(Style):
    types = style.get_types(ForceFieldType)
    print(f"style={style.name!r}  [{len(types)} types]")
    for t in types:
        params = dict(t.params)
        print(f"  {t.name}: {params}")
```

Name-based lookup targets a specific style or type directly.

```python
bs = ff.get_style("bond", "harmonic")
ct_ct = bs.get_type_by_name("CT-CT")
print(f"CT-CT k={ct_ct['k']}")
```


## Evaluating as Potentials

Evaluation is the first strict integrity test of the model.
`mp.ff.potential.PotentialCompiler(ff)` compiles the force field against a typed `Frame`:
an `atoms` block with coordinates and a `type` column, plus bonded blocks
(`bonds`, `angles`, …) whose `type` column names force-field types. The
numerical kernels run in the native Rust core.

```python
# A minimal frame: two atoms 1.2 Å apart joined by one CT-HC bond.
frame = mp.Frame(
    blocks={
        "atoms": {"x": [0.0, 1.2], "y": [0.0, 0.0], "z": [0.0, 0.0], "type": ["CT", "HC"]},
        "bonds": {"atomi": [0], "atomj": [1], "type": ["CT-HC"]},
    }
)

pots = mp.ff.potential.PotentialCompiler(ff).compile(frame)
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
print(f"energy = {energy}")
print(f"forces =\n{forces}")
```

If a referenced type is missing or a required parameter is absent, compilation
raises here rather than producing a plausible-but-wrong number.


## Exporting to simulation engines

Once the model is internally consistent, serialization becomes an interface problem rather than a modeling problem. The same force field can be rendered into different engine formats without redefining the physics.

### GROMACS

```python
mp.io.write_gromacs_top_forcefield("system.itp", ff, precision=4)
```

### XML

```python
mp.io.write_openmm_xml_forcefield("system.xml", ff, precision=6)
```

### LAMMPS

A LAMMPS include holds the coefficients a system uses, so the writer takes the
typed frame as well: every type label of the frame is looked up in the force
field by its exact name, and types no label uses are not written. Its
`pair_style` line needs a cutoff, and a cutoff is a run setting, not a force
field parameter — so you declare it on the pair style yourself; molpy never
invents one. (GROMACS keeps its cutoffs in the `.mdp`, so the GROMACS writer
does not write a style-level cutoff.)

```python
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0
print(mp.io.write_lammps_forcefield_str(ff, frame, precision=4))
```


## When to move beyond built-in styles

Real projects eventually need interaction forms not covered by built-in styles — a FENE spring, a custom torsion profile, a cross term of three atoms. The force-field IR is a protocol: declare the new style in Python (`class Fene(mp.ff.ir.StyleDeclaration)`, with its ordered parameters and its energy as an expression or a Python kernel), and it is typed, compiled and saved like a built-in, with nothing rebuilt.

See [Extending Force Field](../developer/extending-forcefield.md) for the full extension recipe.


## The force field is not inside the molecule

One more distinction worth making explicit: structure and parameterization are related but separate. A molecule can exist before it is typed. A typed system can exist before the force field is exported. MolPy preserves those boundaries because it makes model validation and format conversion much easier to reason about.

See also: [Atomistic and Topology](01_atomistic_and_topology.md), [Block and Frame](02_block_and_frame.md).
