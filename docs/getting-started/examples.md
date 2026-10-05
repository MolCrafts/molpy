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
mol, _ = mp.Conformer(add_hydrogens=True, seed=42).generate(
 mol
) # add hydrogens + 3D coordinates
typifier = mp.typifier.OPLSAATypifier() # embedded OPLS-AA table
typed = typifier.typify(mol) # assign force-field types
ff = typifier.forcefield() # the parameters it assigned

frame = typed.to_frame() # simulation-ready columnar arrays
# mp.io.write_lammps_data(...) + mp.io.write_lammps_forcefield(path, ff, frame)
# write system.data + system.ff (set frame.box and a per-atom mol_id first —
# see the Quickstart).
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
from molpack import GenCanPack, Target

water = mp.Atomistic(name="water")
o = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

target = (
 Target(water.to_frame(), count=500)
.with_name("water")
.with_restraint(mp.Cuboid([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]))
)
packed = GenCanPack().with_seed(42).run([target], max_loops=200).frame
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
) # d_om: O–M distance in Å (TIP4P/2005); input unchanged
```

See also: [Polarizable & Virtual-Site Models](../user-guide/10_polarizable.md).

## Polymer topologies — one monomer, six architectures

Guides and scripts share names under parallel trees. Every unit is a CGsmiles
fragment whose bonding descriptors are its ports, every topology is a CGsmiles
string, and `mp.Assembler` with `mp.GrowthPlacer` grows it into an
`mp.Atomistic`.

| Docs | Examples |
|------|----------|
| [`user-guide/topology/`](../user-guide/topology/index.md) | `examples/topology/` |
| `01_linear.md` … `06_telechelic.md` | `01_linear.py` … `06_telechelic.py` |

```bash
cd examples
python topology/01_linear.py
```

Minimal linear chain, ten EO units:

```python
# run from examples/topology/ or put that dir on PYTHONPATH
import molpy as mp
from eo_kit import library

sites = mp.CGSmilesIR("{[#EO]|10}").to_coarsegrain()
chain = mp.Assembler(library(), mp.GrowthPlacer()).assemble(sites, mp.Atomistic)
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
import molpy as mp
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

# Each planned chain is a unit sequence; written as CGsmiles it is a site graph.
from eo_kit import library  # examples/topology/

first = plan.chains[0]
sites = mp.CGSmilesIR("{" + "".join(f"[#{m}]" for m in first.monomers) + "}").to_coarsegrain()
chain = mp.Assembler(library(), mp.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

See also: [Polydisperse Systems](../user-guide/05_polydisperse_systems.md) ·
[Packing Systems](../user-guide/09_packing.md).

## AmberTools pipeline — GAFF2 parameters

Type each PEO monomer once with antechamber, assemble a capped chain from the
typed monomers, then finish it with tleap to get GAFF2 parameters and partial
charges. tleap never re-assigns types or charges, so each monomer is a complete
molecule whose leaving groups mimic its chain neighbours: the EO unit is
CH3–O–CH2–CH2–O–CH3 (its O is typed ether `os`), not the H-capped `[<]OCC[>]`
(ethanol, whose O antechamber types hydroxyl `oh`).

!!! note "Requires AmberTools"
    This workflow shells out to `antechamber`, `parmchk2`, and `tleap`. Install
    AmberTools and activate its environment first.

```python
# docs: skip — needs AmberTools
import molpy as mp

conformer = mp.Conformer(seed=42)
# name: (complete molecule, ports as (anchor, leaving handle, kind) SMILES indices)
recipes = {
    "CAPA": ("COC", [(0, 1, ">")]),  # keeps CH3; O–CH3 leaves
    "EO": ("COCCOC", [(1, 0, "<"), (3, 4, ">")]),  # keeps O–CH2–CH2; CH3 and O–CH3 leave
    "CAPB": ("COC", [(1, 2, "<")]),  # keeps O–CH3; CH3 leaves
}
units = {}
for name, (smiles, ports) in recipes.items():
    unit = conformer.generate(mp.io.read_smiles(smiles))[0]  # hydrogens appended after the SMILES atoms
    atoms = list(unit.atoms)
    for anchor, leaving, kind in ports:
        unit.def_port(atoms[anchor], atoms[leaving], kind)
    units[name] = unit

ante = mp.typifier.AntechamberTypifier(atom_type="gaff2", charge_method="bcc")
lib = {name: ante.typify(unit) for name, unit in units.items()}  # antechamber + parmchk2 + tleap, once per monomer

sites = mp.CGSmilesIR("{[#CAPA][#EO]|10[#CAPB]}").to_coarsegrain()
chain = mp.Assembler(lib, mp.GrowthPlacer()).assemble(sites, mp.Atomistic)  # types and charges travel with the templates

leap = mp.typifier.TLeapTypifier(leaprc="gaff2", forcefield=ante.forcefield())
peo = leap.typify(chain)  # tleap only: junction terms from leaprc.gaff2; types and charges unchanged
ff = leap.forcefield()  # units real, AMBER 1-4 scaling declared
```

See also: [AmberTools Integration](../user-guide/13_ambertools_integration.md) ·
[Assembly](../user-guide/02_assembly.md).
