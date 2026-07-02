---
title: MolPy
description: A programmable toolkit for molecular simulation workflows.
hide:
  - navigation
  - toc
hero:
  kicker: MolPy Manual
  title: MolPy
  description: Build, edit, type, pack, analyze, and export molecular systems in Python with explicit data structures that stay inspectable at every workflow boundary.
  actions:
    - label: Get started
      href: getting-started/
      style: primary
    - label: Guides
      href: user-guide/
    - label: API reference
      href: api/
  metrics:
    - label: Model
      value: Atomistic · Frame · ForceField
    - label: Workflows
      value: Parse · build · type · export
    - label: Runtime
      value: Python 3.12+
---

<h1 class="molcrafts-sr-only">MolPy</h1>

<div class="molcrafts-manual-home molpy-home" markdown>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">At a glance</span>

## A molecular description in, a runnable system out

MolPy keeps chemistry, coordinates, and force-field parameters in separate, inspectable layers. Parsing, coordinate generation, and typing are each explicit steps:

</div>

```python
import molpy as mp

mol   = mp.parser.parse_molecule("CCO")                  # ethanol from SMILES (heavy atoms)
mol   = mp.adapter.generate_3d(mol, add_hydrogens=True)  # add hydrogens + 3D coordinates
ff    = mp.io.read_xml_forcefield("oplsaa.xml")          # bundled OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(mol)         # assign force-field types

frame = typed.to_frame()   # columnar arrays, ready for analysis or export
```

From that typed `Frame`, one call writes a LAMMPS deck (`mp.io.write_lammps_system`), and the same machinery scales to packed solvent boxes, virtual-site models, and crosslinked polymer networks. The [Quickstart](getting-started/quickstart/) walks through a full solvated system; the [Example Gallery](getting-started/examples/) collects short end-to-end workflows.

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Find your page</span>

## The manual in six sections

</div>

<nav class="molcrafts-manual-index molpy-entry-index" aria-label="Manual sections">
  <a href="getting-started/">
    <span>01</span>
    <strong>Getting Started</strong>
    <em>Install MolPy, run the quickstart, and browse copy-paste example workflows.</em>
  </a>
  <a href="tutorials/">
    <span>02</span>
    <strong>Concepts</strong>
    <em>The data model: Atomistic, Frame, Box, ForceField, and the explicit boundaries between them.</em>
  </a>
  <a href="user-guide/">
    <span>03</span>
    <strong>Guides</strong>
    <em>Task-oriented workflows: parsing, construction, typification, packing, export, and tool integrations.</em>
  </a>
  <a href="compute/">
    <span>04</span>
    <strong>Compute</strong>
    <em>Trajectory analysis: distributions, transport, spectra, order parameters, and analysis workflows.</em>
  </a>
  <a href="api/">
    <span>05</span>
    <strong>API Reference</strong>
    <em>Complete reference for every public module, from core data structures to engine adapters.</em>
  </a>
  <a href="developer/">
    <span>06</span>
    <strong>Developer Guide</strong>
    <em>Contributing workflow, architecture overview, and the extension points for new capabilities.</em>
  </a>
</nav>

</section>

</div>
