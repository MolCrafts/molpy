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

mol   = mp.parser.parse_molecule("CCO")                  # ethanol from SMILES (heavy atoms)
mol   = mp.adapter.RDKitAdapter(mol).generate_3d(add_hydrogens=True)  # add hydrogens + 3D coordinates
ff    = mp.io.read_xml_forcefield("oplsaa.xml")          # bundled OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)         # assign force-field types

frame = typed.to_frame()   # simulation-ready columnar arrays
# mp.io.write_lammps_system("output/", frame, ff) writes system.data + system.ff
# (set frame.box and a per-atom mol_id first — see the Quickstart).
```

See also: [Parsing Chemistry](../user-guide/01_parsing_chemistry.md) ·
[Force Field Typification](../user-guide/06_typifier.md).

## Solvent box — pack 500 waters

Build one molecule, then fill a periodic cube with clash-free copies through the
Packmol backend.

!!! note "Requires the `packmol` executable"
    Packing shells out to Packmol. Install it and make sure `packmol` is on your
    `PATH`.

```python
import molpy as mp
from molpy.pack import Packmol, InsideBoxConstraint

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

p = Packmol(workdir="pack_out")
p.def_target(
    water.to_frame(),
    number=500,
    constraint=InsideBoxConstraint(length=30.0),  # a 30 Å cube
)
packed = p(max_steps=1000, seed=42)       # → one packed Frame (1500 atoms)
```

See also: [Packing Systems](../user-guide/09_packing.md).

## Virtual sites — TIP4P water

Augment a water molecule with an off-atom M-site on the HOH bisector. The
builder copies the input, places the site, and redistributes charge.

```python
import molpy as mp
from molpy.builder.virtualsite import Tip4pBuilder

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

water4p = Tip4pBuilder(d_om=0.1546).apply(water)   # d_om: O–M distance in nm; input unchanged
```

See also: [Polarizable & Virtual-Site Models](../user-guide/10_polarizable.md).

## Polymer chain — G-BigSMILES to a typed frame

Build a polymer chain with 3D coordinates straight from a G-BigSMILES string,
then type it.

```python
import molpy as mp
from molpy.builder import polymer

peo   = polymer("{[<]CCOCC[>]}|10|")             # PEO, degree of polymerization = 10
ff    = mp.io.read_xml_forcefield("oplsaa.xml")
typed = mp.typifier.OplsTypifier(ff).typify(peo)

frame = typed.to_frame()   # write with mp.io.write_lammps_system(dir, frame, ff)
```

See also: [Topology-Driven Assembly](../user-guide/03_polymer_topology.md).

## Polydisperse melt — Schulz-Zimm distribution

Sample a reproducible chain population from a molecular-weight distribution.

```python
import molpy as mp
from molpy.builder import polymer_system

# Mn = 1500 Da, Mw = 3000 Da, total mass ≈ 500 kDa
chains = polymer_system(
    "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
    random_seed=42,
)
print(f"Built {len(chains)} chains")   # reproducible chain population
frames = [c.to_frame() for c in chains]
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
import molpy as mp
from molpy.builder import polymer, prepare_monomer

eo = prepare_monomer("{[<]CCOCC[>]}")  # BigSMILES → 3D + ports

result = polymer(
    "{[#EO]|20}",
    library={"EO": eo},
    backend="amber",   # runs antechamber + parmchk2 + tleap
)
# result.prmtop_path, result.inpcrd_path, result.pdb_path
```

See also: [AmberTools Integration](../user-guide/13_ambertools_integration.md).
