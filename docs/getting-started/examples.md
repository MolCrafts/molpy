# Example Gallery

Short, copy-paste-ready workflows that each take a chemistry notation string to
a simulation-ready output in a handful of lines. Every example links to the
in-depth guide that explains the steps behind it.

For a fully narrated, step-by-step walkthrough, start with the
[Quickstart](quickstart.md).

## Small molecule — parse, type, export

Parse a SMILES string, assign OPLS-AA types, and write LAMMPS input files.

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")          # ethanol from SMILES
ff    = mp.io.read_xml_forcefield("oplsaa.xml")  # bundled OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)

mp.io.write_lammps_system("output/", typed.to_frame(), ff)
# → output/system.data  output/system.in
```

See also: [Parsing Chemistry](../user-guide/01_parsing_chemistry.ipynb) ·
[Force Field Typification](../user-guide/06_typifier.ipynb).

## Polymer chain — G-BigSMILES to LAMMPS

Build a polymer chain with 3D coordinates straight from a G-BigSMILES string.

```python
import molpy as mp
from molpy.builder import polymer

# PEO chain with degree of polymerization = 10, built with 3D coordinates
peo   = polymer("{[<]CCOCC[>]}|10|")
ff    = mp.io.read_xml_forcefield("oplsaa.xml")
typed = mp.typifier.OplsTypifier(ff).typify(peo)
mp.io.write_lammps_system("output/", typed.to_frame(), ff)
```

See also: [Topology-Driven Assembly](../user-guide/03_polymer_topology.ipynb).

## Polydisperse melt — Schulz-Zimm distribution

Sample a reproducible chain population from a molecular-weight distribution and
pack it into a periodic box.

```python
import molpy as mp
from molpy.builder import polymer_system

# Mn = 1500 Da, Mw = 3000 Da, total mass ≈ 500 kDa
chains = polymer_system(
    "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
    random_seed=42,
)
print(f"Built {len(chains)} chains")   # reproducible chain population

ff     = mp.io.read_xml_forcefield("oplsaa.xml")
frames = [c.to_frame() for c in chains]
packed = mp.pack.pack(frames, box=[80, 80, 80])
mp.io.write_lammps_system("peo_bulk/", packed, ff)
```

See also: [Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb).

## AmberTools pipeline — GAFF2 parameters

Run a monomer through antechamber, parmchk2, and tleap to produce an AMBER
topology with GAFF2 parameters and partial charges.

```python
import molpy as mp
from molpy.builder import polymer, PrepareMonomer

eo = PrepareMonomer().run("{[<]CCOCC[>]}")  # BigSMILES → 3D + ports

result = polymer(
    "{[#EO]|20}",
    library={"EO": eo},
    backend="amber",   # runs antechamber + parmchk2 + tleap
)
# result.prmtop_path, result.inpcrd_path, result.pdb_path
```

See also: [AmberTools Integration](../user-guide/07_ambertools_integration.md).
