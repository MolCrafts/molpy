---
template: home.html
hide:
  - navigation
  - toc
---

# MolPy

<p class="lead" markdown>A composable, strongly typed toolkit for computational molecular modeling — from single-molecule parameterization to polydisperse polymer system construction.</p>

<div class="badges" markdown>
  [![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
  [![License: BSD](https://img.shields.io/badge/license-BSD-green.svg)](https://github.com/MolCrafts/molpy/blob/master/LICENSE)
  [![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://molcrafts.github.io/molpy)
  [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
  [![Type checked: ty](https://img.shields.io/badge/type%20checked-ty-blue.svg)](https://github.com/astral-sh/ty)
</div>

<div class="button-group" markdown>
  [Get Started](getting-started/index.md){ .md-button .md-button--primary }
  [Guides](user-guide/index.md){ .md-button }
  [API Reference](api/index.md){ .md-button }
</div>

---

## Representative Workflows

=== "Small Molecule"

    Parameterize a small organic molecule from a SMILES string using the bundled OPLS-AA force field and export complete LAMMPS input files.

    ```python
    import molpy as mp

    mol   = mp.parser.parse_molecule("CCO")          # ethanol from SMILES
    ff    = mp.io.read_xml_forcefield("oplsaa.xml")  # bundled OPLS-AA
    typed = mp.typifier.OplsAtomisticTypifier(ff).typify(mol)

    mp.io.write_lammps_system("output/", typed.to_frame(), ff)
    # → output/system.data  output/system.in
    ```

=== "Polymer Chain"

    Specify a poly(ethylene oxide) chain via G-BigSMILES notation. MolPy generates three-dimensional coordinates and exports a simulation-ready topology.

    ```python
    import molpy as mp

    # PEO chain, degree of polymerization = 10
    peo = mp.tool.polymer("{[<]CCOCC[>]}|10|")

    ff    = mp.io.read_xml_forcefield("oplsaa.xml")
    typed = mp.typifier.OplsAtomisticTypifier(ff).typify(peo)
    mp.io.write_lammps_system("output/", typed.to_frame(), ff)
    ```

=== "Polydisperse System"

    Sample a Schulz–Zimm molecular-weight distribution, construct each chain atomistically, and pack the ensemble into a periodic simulation box.

    ```python
    import molpy as mp

    # Mn = 1500 Da, Mw = 3000 Da, target total mass ≈ 500 kDa
    chains = mp.tool.polymer_system(
        "{[<]CCOCC[>]}|schulz_zimm(1500,3000)||5e5|",
        random_seed=42,
    )
    print(f"Built {len(chains)} chains")

    frames = [c.to_frame() for c in chains]
    packed = mp.pack.pack(frames, box=[80, 80, 80])
    mp.io.write_lammps_system("peo_bulk/", packed, ff)
    ```

=== "AmberTools Pipeline"

    Prepare a monomer with partial charges via antechamber, assemble a chain with GAFF2 parameters via tleap, and retrieve AMBER topology files programmatically.

    ```python
    import molpy as mp

    # BigSMILES → three-dimensional structure with port annotation
    eo = mp.tool.PrepareMonomer().run("{[<]CCOCC[>]}")

    # Assemble DP = 20 chain via AmberTools
    result = mp.tool.polymer(
        "{[#EO]|20}",
        library={"EO": eo},
        backend="amber",
    )
    # result.prmtop_path  result.inpcrd_path  result.pdb_path
    ```

---

## Design Principles

<div class="feature-list" markdown>

- :material-graph: **[Explicit representational hierarchy](tutorials/01_atomistic_and_topology.md)** — Molecular graphs (`Atomistic`), numerical snapshots (`Frame`), and force field parameters (`ForceField`) occupy distinct layers with explicit conversion boundaries.

- :material-text-search: **[Native support for polymer chemistry notations](user-guide/01_parsing_chemistry.ipynb)** — SMILES, BigSMILES, CGSmiles, and G-BigSMILES are parsed directly. A monomer, an architecture, or a polydisperse ensemble can each be expressed as a single string.

- :material-chart-bell-curve: **[Statistical molecular-weight distributions](user-guide/05_polydisperse_systems.ipynb)** — Schulz–Zimm, Poisson, Flory–Schulz, and uniform distributions are implemented natively. Target number- and weight-average molecular weights are specified directly; reproducible chain populations are generated from a fixed random seed.

- :material-database-search: **[Force fields as queryable data structures](tutorials/04_force_field.md)** — A `ForceField` object is an inspectable typed dictionary. Parameter completeness and type consistency are verifiable at the Python level before any file export occurs.

- :material-vector-link: **[Programmatic reaction framework](user-guide/04_crosslinking.ipynb)** — Chemical reactions are expressed through composable anchor selectors and leaving-group selectors. Pre- and post-reaction topology templates for LAMMPS `fix bond/react` are generated automatically.

- :material-puzzle: **[Modular, independently composable packages](api/index.md)** — The parser, builder, typifier, packer, and I/O subsystems share no hidden coupling. Each may be used independently or assembled into composite pipelines through explicit function calls.

</div>

---

## External Integrations

<div class="feature-list" markdown>

- :material-atom: **[AmberTools](user-guide/07_ambertools_integration.ipynb)** — Antechamber (partial charge assignment), parmchk2 (missing parameter estimation), and tleap (topology assembly) are invoked programmatically with structured Python interfaces.

- :material-flask: **[RDKit](api/adapter.md)** — `RDKitAdapter` provides bidirectional conversion between `Atomistic` and RDKit `Mol` objects, enabling three-dimensional embedding, conformer generation, and SMILES export.

- :material-cube-outline: **[Packmol](api/pack.md)** — Molecule packing into periodic simulation boxes is managed through a typed constraint interface wrapping the Packmol executable.

- :material-lightning-bolt: **[LAMMPS · CP2K](api/engine.md)** — Complete input decks are generated from MolPy data objects. The engine abstraction layer decouples system description from simulation-code-specific syntax.

</div>

---

## Documentation Structure

<div class="feature-list" markdown>

- :material-rocket-launch: **[Getting Started](getting-started/index.md)** — Installation, environment verification, and a five-minute end-to-end example establishing the `Atomistic → Frame → export` pipeline.

- :material-book-open-variant: **[Concepts](tutorials/index.md)** — Systematic exposition of the core data model: `Atomistic`, `Block`, `Frame`, `Box`, `Trajectory`, `ForceField`, and their inter-relationships.

- :material-hammer-wrench: **[Guides](user-guide/index.md)** — Task-oriented executable notebooks covering chemistry parsing, polymer construction, force field typification, and simulation file generation.

- :material-code-braces: **[Developer Guide](developer/index.md)** — Conventions, extension patterns, and internal architecture for contributors and library developers.

</div>
