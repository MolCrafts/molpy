# MolPy — A programmable toolkit for molecular simulation workflows

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://molpy.molcrafts.org/)
[![CI](https://github.com/MolCrafts/molpy/workflows/CI/badge.svg)](https://github.com/MolCrafts/molpy/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type checked: ty](https://img.shields.io/badge/type%20checked-ty-blue.svg)](https://github.com/astral-sh/ty)

> **MolPy is under active development.** Public APIs may change between minor releases.

MolPy is a Python toolkit for building, editing, typing, and exporting
molecular systems. It keeps topology, force fields, numerical frames, and
engine I/O explicit and composable in Python.

Polymer construction and reactive topology editing are core strengths, but
MolPy is built for broader molecular simulation workflows, from system
preparation and topology transformation to force-field assignment and export.

## Representative Workflows

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

## Core Capabilities

| Area | Capability |
|------|------------|
| **Parsing** | Native support for SMILES, BigSMILES, CGSmiles, and G-BigSMILES |
| **Polymer construction** | Linear, branched, and cyclic assembly from notation-driven specifications |
| **Polydispersity** | Schulz-Zimm, Poisson, Flory-Schulz, and uniform chain-length distributions |
| **Reactive topology editing** | Anchor and leaving-group selectors, bond-forming reactions, and LAMMPS `fix bond/react` template generation |
| **Force-field assignment** | OPLS-AA and GAFF/GAFF2 typing through SMARTS/SMIRKS matching |
| **External parameterization** | AmberTools interfaces for antechamber, parmchk2, prepgen, and tleap |
| **Packing and export** | Packmol-based packing and export to LAMMPS, PDB, XYZ, AMBER, GRO, and HDF5 |
| **Simulation interfaces** | Input generation for LAMMPS and CP2K |
| **Interoperability** | RDKit and OpenBabel adapters for conversion and structure preparation |
| **Explicit data model** | Distinct `Atomistic`, `Frame`/`Block`, and `ForceField` representations |
| **Agent interface** | Optional MCP server exposing source symbols and documentation to large language model agents |

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

Each boundary is an explicit function call. State transitions remain visible throughout the workflow; no hidden coupling or monkey-patching is required.

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

Full documentation, including executable notebooks: **[https://molcrafts.github.io/molpy](https://molcrafts.github.io/molpy)**

- [Getting Started](https://molcrafts.github.io/molpy/getting-started/) — install and first example
- [Guides](https://molcrafts.github.io/molpy/user-guide/) — task-oriented notebooks
- [Concepts](https://molcrafts.github.io/molpy/tutorials/) — data model deep dives
- [API Reference](https://molcrafts.github.io/molpy/api/) — full API

---

## MolCrafts Ecosystem

| Project | Role |
|---------|------|
| **MolPy** | Python toolkit — this repo |
| **MolVis** | WebGL molecular visualization and interactive editing |
| **MolRS** | Rust backend: typed array structures and fast compute kernels (native + WASM) |

---

## Contributing

Issues and pull requests are welcome. See [Contributing](https://molcrafts.github.io/molpy/developer/contributing/).

---

## License

BSD-3-Clause — see [LICENSE](LICENSE).
