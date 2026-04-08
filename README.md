# MolPy — Composable molecular modeling in Python

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://molcrafts.github.io/molpy)
[![CI](https://github.com/MolCrafts/molpy/workflows/CI/badge.svg)](https://github.com/MolCrafts/molpy/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: ty](https://img.shields.io/badge/type%20checked-ty-blue.svg)](https://github.com/astral-sh/ty)

> **MolPy is under active development.** The API evolves rapidly and may change between minor versions.

MolPy is a strongly typed Python toolkit for building, parameterizing, and exporting molecular systems — from a single small molecule to a polydisperse polymer melt with thousands of chains.

## What it can do

**Small molecule — parse, type, export:**

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")          # ethanol from SMILES
ff    = mp.io.read_xml_forcefield("oplsaa.xml")  # bundled OPLS-AA
typed = mp.typifier.OplsAtomisticTypifier(ff).typify(mol)

mp.io.write_lammps_system("output/", typed.to_frame(), ff)
# → output/system.data  output/system.in
```

**Polymer chain — G-BigSMILES to LAMMPS in one call:**

```python
import molpy as mp

# PEO chain with degree of polymerization = 10, built with 3D coordinates
peo   = mp.tool.polymer("{[<]CCOCC[>]}|10|")
ff    = mp.io.read_xml_forcefield("oplsaa.xml")
typed = mp.typifier.OplsAtomisticTypifier(ff).typify(peo)
mp.io.write_lammps_system("output/", typed.to_frame(), ff)
```

**Polydisperse melt — Schulz-Zimm distribution, fully atomistic:**

```python
import molpy as mp

# Mn = 1500 Da, Mw = 3000 Da, total mass ≈ 500 kDa
chains = mp.tool.polymer_system(
    "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
    random_seed=42,
)
print(f"Built {len(chains)} chains")   # reproducible chain population

frames = [c.to_frame() for c in chains]
packed = mp.pack.pack(frames, box=[80, 80, 80])
mp.io.write_lammps_system("peo_bulk/", packed, ff)
```

**AmberTools pipeline — GAFF2 parameters, partial charges, AMBER topology:**

```python
import molpy as mp

eo = mp.tool.PrepareMonomer().run("{[<]CCOCC[>]}")  # BigSMILES → 3D + ports

result = mp.tool.polymer(
    "{[#EO]|20}",
    library={"EO": eo},
    backend="amber",   # runs antechamber + parmchk2 + tleap
)
# result.prmtop_path, result.inpcrd_path, result.pdb_path
```

---

## Features

| Area | What MolPy provides |
|------|---------------------|
| **Parsers** | SMILES, BigSMILES, CGSmiles, G-BigSMILES natively |
| **Polymer builders** | Linear, branched, cyclic topologies from CGSmiles strings |
| **Polydispersity** | Schulz-Zimm, Poisson, Flory-Schulz, Uniform distributions |
| **Reaction framework** | Anchor + leaving-group reactions; LAMMPS `fix bond/react` templates |
| **Force field typing** | OPLS-AA, GAFF/GAFF2 via SMARTS/SMIRKS pattern matching |
| **AmberTools integration** | Antechamber, parmchk2, prepgen, tleap — all from Python |
| **Packing** | Packmol wrapper with density targets and typed constraints |
| **I/O** | LAMMPS DATA, PDB, XYZ, AMBER prmtop/inpcrd, GRO, HDF5 |
| **MD engines** | LAMMPS and CP2K input generation |
| **Adapters** | RDKit (3D embedding, SMILES export), OpenBabel |
| **Data model** | `Atomistic` graph · `Frame`/`Block` arrays · `ForceField` dict |

---

## Architecture

```
SMILES / BigSMILES / CGSmiles / G-BigSMILES
           │
           ▼ parser
      Atomistic                  ← editable graph: Atom, Bond, Angle, Dihedral
      (topology graph)
           │
    ┌──────┴──────┐
    ▼             ▼
reacter        builder           ← reactions, polymer assembly
    └──────┬──────┘
           │
           ▼ typifier
    Typed Atomistic              ← SMARTS pattern matching assigns ForceField types
           │
           ▼ .to_frame()
         Frame                   ← columnar NumPy arrays; Box; metadata
           │
    ┌──────┴──────┐
    ▼             ▼
   pack          io / engine     ← Packmol packing; LAMMPS / CP2K / AMBER export
```

Each boundary is an explicit function call. No hidden state, no monkey-patching.

---

## Installation

```bash
pip install molcrafts-molpy
```

Core dependencies: NumPy, igraph, Lark. Optional: RDKit (3D geometry), AmberTools (GAFF charges).

For development:

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not external"
```

---

## Documentation

Full docs with interactive notebooks: **[https://molcrafts.github.io/molpy](https://molcrafts.github.io/molpy)**

- [Getting Started](https://molcrafts.github.io/molpy/getting-started/) — install and first example
- [Guides](https://molcrafts.github.io/molpy/user-guide/) — task-oriented notebooks
- [Concepts](https://molcrafts.github.io/molpy/tutorials/) — data model deep dives
- [API Reference](https://molcrafts.github.io/molpy/api/) — full API

---

## MolCrafts Ecosystem

| Project | Role |
|---------|------|
| **MolPy** | Python toolkit — this repo |
| **MolRS** | Rust backend: typed array structures and fast compute kernels (native + WASM) |
| **MolVis** | WebGL molecular visualization and interactive editing |

---

## Contributing

```bash
pip install -e ".[dev]"
pre-commit install
pytest tests/ -m "not external" -q
```

Issues and pull requests are welcome. See [Contributing](https://molcrafts.github.io/molpy/developer/contributing/).

---

## License

BSD-3-Clause — see [LICENSE](LICENSE).
