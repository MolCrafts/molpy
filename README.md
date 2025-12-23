# 🚀 MolPy — A fast, clean, and composable toolkit for molecular modeling

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](./LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://molcrafts.github.io/molpy)
[![CI](https://github.com/MolCrafts/molpy/workflows/CI/badge.svg)](https://github.com/MolCrafts/molpy/actions)

> **⚠️ MolPy is under active development.**
> The API is evolving rapidly and may change between minor versions.

**MolPy** is a modern, strongly typed, and extensible toolkit for molecular modeling.
It provides a clean data model, flexible builders, explicit force-field typing, and lightweight analysis — a foundation designed for reproducible workflows and AI-assisted development.

## 🚀 Installation

```bash
pip install molcrafts-molpy
```

For development:

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
pip install -e ".[dev]"
```

---

## ✨ Features

- **Unified Data Model** — `Frame` / `Block` / `Box` for molecular structures
- **Explicit Topology** — `Atom`, `Bond`, `Angle`, `Dihedral` with typed metadata
- **Force-Field Typing** — rule-based, engine-agnostic typifiers (OPLS-AA, custom SMARTS/SMIRKS)
- **Polymer Builders** — automatic polymer assembly from BigSMILES/GBigSMILES/CGSmiles
  - Linear, branched, and cyclic topologies
  - Polydisperse systems with Schulz-Zimm, Poisson, Uniform, Flory-Schulz distributions
  - Reaction-based crosslinking and network formation
- **Chemistry Parsers** — SMILES, BigSMILES, GBigSMILES, and CGSmiles support
- **Adapter System** — seamless integration with RDKit, LAMMPS, OpenMM, Packmol
- **Reaction Framework** — explicit chemical reactions with leaving groups
- **Lightweight Analysis** — composable compute operators
- **Robust I/O** — PDB, XYZ, LAMMPS DATA, JSON-based formats
- **AI-Friendly APIs** — predictable, strongly typed, minimal magic

---

## 📚 Documentation

Full documentation: **[https://molcrafts.github.io/molpy](https://molcrafts.github.io/molpy)**

---

## 🌌 MolCrafts Ecosystem

### **MolVis** — Interactive Molecular Visualization

WebGL-based visualization and editing.

### **MolRS** — High-Performance Rust Backend

Typed array structures, compute kernels, and fast builders (native + WASM).

---

## 🤝 Contributing

```bash
pip install -e ".[dev]"
pre-commit install
pytest -q
```

We welcome issues and pull requests.

---

## 📄 License

BSD-3-Clause — see [LICENSE](LICENSE).

---

**Built with ❤️ by MolCrafts.**
