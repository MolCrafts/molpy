---
title: MolPy
description: A Python toolkit for building, editing, and parameterizing complex molecular systems.
hide:
  - navigation
  - toc
hero:
  kicker: MolPy Manual
  title: MolPy
  description: A programmable Python toolkit for molecular simulation workflows — from chemistry text to a runnable system and back to analysis.
  install:
    label: Install
    command: pip install molcrafts-molpy
  badges:
    - img: https://img.shields.io/pypi/v/molcrafts-molpy?color=0284c7&label=PyPI
      href: https://pypi.org/project/molcrafts-molpy/
      alt: PyPI version
    - img: https://img.shields.io/pypi/pyversions/molcrafts-molpy?color=0f766e
      href: https://pypi.org/project/molcrafts-molpy/
      alt: Python versions
    - img: https://img.shields.io/github/stars/MolCrafts/molpy?style=flat&color=c8841d
      href: https://github.com/MolCrafts/molpy
      alt: GitHub stars
  actions:
    - label: Get started
      href: tutorials/
      style: primary
    - label: Browse examples
      href: getting-started/examples/
    - label: API reference
      href: api/
---

<h1 class="molcrafts-sr-only">MolPy</h1>

<div class="molcrafts-manual-home molpy-home" markdown>

<section class="molcrafts-manual-section molcrafts-manual-section--stack molpy-system-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">The pipeline</span>

## From a molecule description to a runnable system

Chemistry, coordinates, and force-field parameters stay in separate layers.
You can stop after any stage, inspect what you have, and continue — without
shuttling everything through disk.

</div>

<div class="molpy-system-panel">
<div class="molpy-system-panel__header">
<span>One representation · six stages</span>
<strong>The same Atomistic graph becomes a typed, packed Frame — export it or analyze it in place</strong>
</div>
<div class="molpy-system-flow">
<div>
<span>01 · Parse / build</span>
<a href="user-guide/01_parsing_chemistry/"><strong>SMILES, BigSMILES, or a file → an editable graph</strong></a>
</div>
<div>
<span>02 · Edit</span>
<a href="user-guide/02_assembly/"><strong>React, crosslink, and assemble on the graph</strong></a>
</div>
<div>
<span>03 · Typify</span>
<a href="user-guide/06_typifier/"><strong>Assign OPLS-AA / GAFF types and parameters</strong></a>
</div>
<div>
<span>04 · Pack</span>
<a href="user-guide/09_packing/"><strong>Fill a periodic box, clash-free</strong></a>
</div>
<div>
<span>05 · Export</span>
<a href="user-guide/11_io/"><strong>LAMMPS, GROMACS, PDB, Zarr, and more</strong></a>
</div>
<div>
<span>06 · Analyze</span>
<a href="compute/"><strong>RDF, MSD, order parameters, spectra — on the same Frame</strong></a>
</div>
</div>
</div>

The [Quickstart](getting-started/quickstart/) walks through one full system
end to end. The [Example Gallery](getting-started/examples/) collects shorter
copy-paste recipes.

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">In practice</span>

## Each stage is a few lines of Python

The cards below follow the same six stages. Polymers show up often because
crosslinking and polydispersity stress the editing machinery — not because
MolPy is limited to polymers.

</div>

<div class="molcrafts-workflow-list" markdown>

<article markdown>

<div class="molcrafts-workflow-list__meta">01 · Parse / build</div>

### [Describe chemistry as text](user-guide/01_parsing_chemistry/)

One line of SMILES or BigSMILES becomes an editable structure — a single
molecule or a polymer chain.

```python
import molpy as mp
from molpy.conformer import Conformer

mol = mp.io.read_smiles("CCO")  # one molecule from SMILES
mol, report = Conformer(seed=42).generate(mol)  # hydrogens + 3D coordinates
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">02 · Edit</div>

### [Rewire the topology](user-guide/02_assembly/)

Merge structures, form and break bonds, drop leaving groups, then re-derive
angles and dihedrals across the new junction.

```python
dimer = mol.copy().merge(mol.copy())  # combine two copies
dimer.get_topo(gen_angle=True, gen_dihe=True)  # derive angles/dihedrals in place
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">03 · Typify</div>

### [Assign force-field types](user-guide/06_typifier/)

SMARTS matching maps every atom, bond, angle, and dihedral to parameters you
can inspect before anything is exported.

```python
ff = mp.io.read_xml_forcefield("oplsaa.xml")  # bundled OPLS-AA
typed = mp.typifier.OPLSAATypifier().typify(mol)
system = typed.to_frame()  # the numeric Frame
system.box = mp.Box.cubic(30.0)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">04 · Pack</div>

### [Fill a periodic box](user-guide/09_packing/)

Clash-free placement at a target density via
[molpack](https://molcrafts.github.io/molpack/) — Packmol-grade packing in
Rust, no external binary (`pip install molcrafts-molpack`).

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
from molpack import InsideBoxRestraint, Molpack, Target

target = (
    Target(system, count=500)
    .with_restraint(InsideBoxRestraint([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]))
)
system = Molpack().with_seed(42).pack([target], max_loops=200)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">05 · Export</div>

### [Write files your engine runs](user-guide/11_io/)

One call per file: LAMMPS data plus force-field coefficients. GROMACS, PDB,
and Zarr (`MolStore`) writers share the same pattern.

```python
import numpy as np

atoms = system["atoms"]
atoms["mol_id"] = np.ones(atoms.nrows, dtype=np.uint32)  # full atom style needs mol_id
mp.io.write_lammps_data("system.data", system, atom_style="full")
mp.io.write_lammps_forcefield("system.ff", ff)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">06 · Analyze</div>

### [Turn trajectories into observables](compute/)

Feed the same Frame into the Rust-backed compute layer — neighbor search and
$g(r)$ in two calls, with many more analyses behind them.

```python
from molpy.compute import NeighborList, RDF

system.box = mp.Box.cubic(30.0)
neighbors = NeighborList(cutoff=8.0)(system)
result = RDF(n_bins=50, r_max=8.0)([system], [neighbors])  # g(r) over the box
```

</article>

</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">By design</span>

## Built to be composed, not locked in

A library first: one shared data model, a Rust core, and explicit seams. Take
one piece, leave the rest, or extend any layer without forking the package.

</div>

<dl class="molcrafts-feature-matrix molcrafts-feature-matrix--cards">
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/></svg></span>
<dt><a href="tutorials/02_block_and_frame/">One data structure across the ecosystem</a></dt>
<dd>molpack, molvis, and molmcp speak the same molrs-backed <code>Frame</code> / <code>Block</code>. No converters between libraries.</dd>
</div>
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></span>
<dt><a href="developer/molrs-backend/">A Rust kernel underneath</a></dt>
<dd>Storage and compute live in molrs. Python sees zero-copy NumPy views and the same objects identity-re-exported on the molpy facade.</dd>
</div>
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg></span>
<dt><a href="user-guide/15_mcp/">Built for LLM agents</a></dt>
<dd>The molmcp suite exposes symbols and docs over MCP so an agent can call the real API instead of guessing from training data.</dd>
</div>
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg></span>
<dt><a href="developer/architecture-overview/">Use one piece or all of them</a></dt>
<dd>Parser, builder, typifier, packer, I/O, and compute talk only through explicit data. Import the layer you need and ignore the rest.</dd>
</div>
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.01a2.5 2.5 0 1 0 3.259-3.259c-.482-.196-.933-.558-1.01-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 12 1.998c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z"/></svg></span>
<dt><a href="developer/extending-compute/">Registries, not hardcoded lists</a></dt>
<dd>Register a compute operator, I/O format, force-field style, or typifier from outside the core without patching the package itself.</dd>
</div>
<div>
<span class="molcrafts-feature-matrix__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/></svg></span>
<dt><a href="developer/coding-style/">Typed end to end</a></dt>
<dd>Public APIs carry full type hints, checked in CI with Astral’s <code>ty</code>. Your editor sees real signatures, not <code>Any</code>.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--body-col" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Ecosystem</span>

## The structure you build is the structure everything else reads

MolPy, molpack, molvis, and molmcp share the same abstract data structure. You
do not write glue adapters between them. [Figure 1](#fig-aspirin) is
[molvis](https://github.com/MolCrafts/molvis) drawing that structure in the
browser — drag to rotate (needs the published molvis-stage Web Component).

</div>

<figure id="fig-aspirin" class="molcrafts-figure">
  <div class="molcrafts-figure__body">
    <!-- XYZ must start on the same line as <template> (no leading blank line). -->
    <molvis-viewer format="xyz" representation="ball-and-stick" controls="view" height="280px">
      <template data-molvis-source>21
name=aspirin Connct="[0,4,0,11,1,10,1,20,2,10,3,11,4,5,4,6,5,7,5,10,6,8,6,13,7,9,7,14,8,9,8,15,9,16,11,12,12,17,12,18,12,19]"
O   1.2333   0.5540   0.7792
O  -0.6952  -2.7148  -0.7502
O   0.7958  -2.1843   0.8685
O   1.7813   0.8105  -1.4821
C  -0.0857   0.6088   0.4403
C  -0.7927  -0.5515   0.1244
C  -0.7288   1.8464   0.4133
C  -2.1426  -0.4741  -0.2184
C  -2.0787   1.9238   0.0706
C  -2.7855   0.7636  -0.2453
C  -0.1409  -1.8536   0.1477
C   2.1094   0.6715  -0.3113
C   3.5305   0.5996   0.1635
H  -0.1851   2.7545   0.6593
H  -2.7247  -1.3605  -0.4564
H  -2.5797   2.8872   0.0506
H  -3.8374   0.8238  -0.5090
H   3.7290   1.4184   0.8593
H   4.2045   0.6969  -0.6924
H   3.7105  -0.3659   0.6426
H  -0.2555  -3.5916  -0.7337</template>
    </molvis-viewer>
  </div>
  <figcaption>
    <span class="molcrafts-figure__label">Figure 1.</span>
    Aspirin (PubChem) in molvis, ball-and-stick, from inline XYZ.
  </figcaption>
</figure>

<dl class="molcrafts-tile-grid">
<div>
<dt><a href="https://molcrafts.github.io/molpack/">molpack</a></dt>
<dd>Clash-free packing as a CLI, a Rust crate, and a Python package — same engine everywhere.</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molmcp">molmcp</a></dt>
<dd>MCP server for LLM agents: code discovery plus live ecosystem providers.</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molrs">molrs</a></dt>
<dd>The shared Rust molecular kernel — Frame, Block, and compute, with Python and other bindings.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--flip" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Integrations</span>

## Optional tools, explicit boundaries

External packages plug in through adapters and wrappers. Nothing is required
beyond the default install; every seam is visible in the API.

</div>

<dl class="molcrafts-link-list">
<div>
<dt><a href="api/adapter/">RDKit</a></dt>
<dd>Bidirectional <code>Atomistic</code> ↔ <code>Mol</code> for embedding, conformers, and SMILES export.</dd>
</div>
<div>
<dt><a href="user-guide/13_ambertools_integration/">AmberTools</a></dt>
<dd>antechamber, parmchk2, and tleap driven from Python for GAFF charges and topologies.</dd>
</div>
<div>
<dt><a href="user-guide/09_packing/">molpack</a></dt>
<dd>Clash-free packing into periodic boxes through a typed restraint interface.</dd>
</div>
<div>
<dt><a href="user-guide/12_engine/">LAMMPS · CP2K · OpenMM</a></dt>
<dd>Ready-to-run input decks generated from MolPy data objects.</dd>
</div>
<div>
<dt><a href="developer/molrs-backend/">molrs · MCP</a></dt>
<dd>Rust column store and compute underneath; MCP exposes symbols and docs to agents.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Find your page</span>

## How the manual is split

**Tutorials** teach: install, a first system, then the data model chapter by
chapter. **Guides** do the work: end-to-end recipes once you know the model.
Reach for compute, API, or developer pages when you already know the task.

</div>

<div class="molcrafts-doc-map molcrafts-doc-map--cards">
<section>
<h3><a href="tutorials/">Tutorials</a></h3>
<p>Install, quickstart, then the data model one chapter at a time.</p>
</section>
<section>
<h3><a href="user-guide/">Guides</a></h3>
<p>Task recipes — parse, build, typify, pack, export — that assume the tutorials.</p>
</section>
<section>
<h3><a href="compute/">Compute</a></h3>
<p>Trajectory analysis: distributions, transport, order, spectra, workflows.</p>
</section>
<section>
<h3><a href="api/">API Reference</a></h3>
<p>Every public module, from core data structures to engine adapters.</p>
</section>
<section>
<h3><a href="developer/">Developer Guide</a></h3>
<p>Contributing, architecture, and how to extend compute, I/O, and typifiers.</p>
</section>
</div>

</section>

</div>
