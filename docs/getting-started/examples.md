# Example Gallery

Short, runnable workflows that each take a molecular description to a
simulation-ready object in a handful of lines. The examples span the capability
spectrum — a single small molecule, a packed solvent box, virtual-site models,
and polymer systems (the stress test for MolPy's editing machinery). Every
example links to the in-depth guide that explains the steps behind it.

For a fully narrated, step-by-step walkthrough — including the full LAMMPS
export — start with the [Quickstart](quickstart.md).

## Small molecule — parse, type, export

Parse a SMILES string, add hydrogens and coordinates, and assign OPLS-AA types.

```python
import molpy as mp

mol = mp.io.read_smiles("CCO") # ethanol from SMILES (heavy atoms)
mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
 mol
) # add hydrogens + 3D coordinates
ff = mp.io.read_xml_forcefield("oplsaa.xml") # bundled OPLS-AA
typed = mp.typifier.OPLSAATypifier().typify(mol) # assign force-field types

frame = typed.to_frame() # simulation-ready columnar arrays
# mp.io.write_lammps_system("output/", frame, ff) writes system.data + system.ff
# (set frame.box and a per-atom mol_id first — see the Quickstart).
```

See also: [Parsing Chemistry](../user-guide/01_parsing_chemistry.md) ·
[Force Field Typification](../user-guide/06_typifier.md).

## Solvent box — pack 500 waters

Build one molecule, then fill a cube with clash-free copies through
**[molpack](https://docs.molcrafts.org/molpack/)**
(`pip install molcrafts-molpack`).

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
import molpy as mp
from molpack import InsideBoxRestraint, Molpack, Target

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

target = (
 Target(water.to_frame(), count=500)
.with_name("water")
.with_restraint(InsideBoxRestraint([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]))
)
packed = Molpack().with_seed(42).pack([target], max_loops=200)
# → one packed Frame (1500 atoms)
```

See also: [Packing Systems](../user-guide/09_packing.md).

## Virtual sites — TIP4P water

Augment a water molecule with an off-atom M-site on the HOH bisector. The
builder copies the input, places the site, and redistributes charge.

```python
import molpy as mp
from molpy.builder.virtualsite import Tip4pBuilder

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.000, y=0.000, z=0.000, charge=-0.834)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000, charge=0.417)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000, charge=0.417)
water.def_bond(o, h1)
water.def_bond(o, h2)

# The M-site carries the oxygen's charge, so the input must already have one.
water4p = Tip4pBuilder(d_om=0.1546).apply(
 water
) # d_om: O–M distance in nm; input unchanged
```

See also: [Polarizable & Virtual-Site Models](../user-guide/10_polarizable.md).

## Polymer topologies — one monomer, eleven architectures

Guides and scripts share names under parallel trees:

| Docs | Examples |
|------|----------|
| [`user-guide/topology/`](../user-guide/topology/index.md) | `examples/topology/` |
| `01_linear.md` … `11_prepolymer_agent.md` | `01_linear.py` … `11_prepolymer_agent.py` |

```bash
cd examples
python topology/01_linear.py
```

Minimal linear chain (`build_linear` ≡ `build(linear_topology(["EO"] * 10))`):

```python
# run from examples/topology/ or put that dir on PYTHONPATH
from eo_kit import eo_builder

chain = eo_builder().build_linear("EO", 10)
```

See also: [Polymer Topologies](../user-guide/topology/index.md) ·
[Assembly](../user-guide/02_assembly.md).

## Carbon nanotubes — topology from chirality

Build open or axially periodic zigzag, armchair, and chiral tubes without a
public planning object:

```python
from molpy.builder import CarbonTubeBuilder

zigzag = CarbonTubeBuilder(8, 0, length=30.0).build()
armchair = CarbonTubeBuilder(6, 6, cells=4, periodic=True).build()
chiral = CarbonTubeBuilder(6, 3, cells=2).build(finalize="topology")
```

See also: [Nanostructures](../user-guide/04_nanostructures.md).

## Polydisperse melt — Schulz-Zimm distribution

Sample a reproducible chain population from a molecular-weight distribution.

```python
import numpy as np
from molpy.builder.polymer import (
 PolydisperseChainGenerator,
 SchulzZimmPolydisperse,
 SystemPlanner,
 WeightedSequenceGenerator,
)

# Mn = 1500 Da, Mw = 3000 Da, total mass ≈ 500 kDa
planner = SystemPlanner(
 PolydisperseChainGenerator(
 WeightedSequenceGenerator({"EO": 1.0}),
 {"EO": 44.05},
 distribution=SchulzZimmPolydisperse(1500, 3000),
),
 target_total_mass=5e5,
)
plan = planner.plan_system(np.random.default_rng(42))
print(f"Planned {len(plan.chains)} chains") # reproducible chain population

# Each planned chain is a residue sequence; hand it to a builder to get a graph.
lengths = [len(c.monomers) for c in plan.chains[:3]]
```

See also: [Polydisperse Systems](../user-guide/05_polydisperse_systems.md) ·
[Packing Systems](../user-guide/09_packing.md).

## AmberTools pipeline — GAFF2 parameters

Run a monomer through antechamber, parmchk2, and tleap to produce an AMBER
topology with GAFF2 parameters and partial charges.

!!! note "Requires AmberTools"
    This workflow shells out to `antechamber`, `parmchk2`, and `tleap`. Install
    AmberTools and activate its environment first.

```python
# docs: skip — AmberPolymerBuilder shells out; builder unit-tested with mocks
import molpy as mp
from molpy.builder import AmberPolymerBuilder
from molpy.builder.assembly import SiteMap
from molpy.conformer import Conformer

eo, _ = Conformer(add_hydrogens=True, seed=42).generate(
 mp.io.read_smiles("OCCO")
)
SiteMap(eo).label_elements("O", "a", "b")

builder = AmberPolymerBuilder(
 library={"EO": eo},
 reaction=mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]"),
 force_field="gaff2",
 charge_method="bcc", # runs antechamber + parmchk2 + prepgen + tleap
)
result = builder.build("{[#EO]|20}")
# result.frame, result.forcefield, and the Amber intermediates under work_dir
```

See also: [AmberTools Integration](../user-guide/13_ambertools_integration.md).
