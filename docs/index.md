---
title: MolPy
description: A Python toolkit for building, editing, and parameterizing complex molecular systems.
hide:
  - navigation
  - toc
hero:
  kicker: MolPy Manual
  title: MolPy
  description: Programmable Python toolkit for molecular simulation workflows.
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

<section class="molcrafts-manual-section molpy-system-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">The pipeline</span>

## A molecular description in, a runnable system out

One molecule, every stage, no detours through disk. Chemistry, coordinates,
and parameters live in separate layers — pause at any boundary, inspect what
you have, keep going.

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

The [Quickstart](getting-started/quickstart/) narrates a full system end to
end; the [Example Gallery](getting-started/examples/) collects short,
copy-paste workflows.

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">In practice</span>

## Every stage is a few lines of Python

The six cards below mirror the pipeline, stage for stage. Polymers appear as
a *demonstration domain* — coupling, crosslinking, and polydispersity stress
every part of the editing machinery — not as the limit of what MolPy models.

</div>

<div class="molcrafts-workflow-list molpy-workflow-list" markdown>

<article markdown>

<div class="molcrafts-workflow-list__meta">01 · Parse / build</div>

### [Describe chemistry as text](user-guide/01_parsing_chemistry/)

One line of SMILES or BigSMILES becomes an editable structure — a single
molecule or a whole polymer chain.

```python
from molpy.builder import polymer

mol = mp.parser.parse_molecule("CCO")   # one molecule from SMILES
peo = polymer("{[<]CCOCC[>]}|10|")      # or a whole chain, DP = 10
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">02 · Edit</div>

### [Rewire the topology, atom by atom](user-guide/02_assembly/)

Merge structures, form and break bonds, drop leaving groups — then re-derive
angles and dihedrals across the new junction.

```python
chain = mon_a.merge(mon_b)                # combine two monomers
chain.def_bond(anchor_C, port_O)          # form the new C–O bond
chain.del_atom(o_leave, h1, h2)           # remove the leaving water
chain = chain.get_topo(gen_angle=True, gen_dihe=True)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">03 · Typify</div>

### [Assign types while it's still data](user-guide/06_typifier/)

SMARTS matching maps every atom, bond, angle, and dihedral to force-field
parameters — inspectable and checkable before anything is exported.

```python
ff    = mp.io.read_xml_forcefield("oplsaa.xml")      # bundled OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(chain)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">04 · Pack</div>

### [Fill a periodic box with Packmol](user-guide/09_packing/)

Clash-free placement at target density, driving the battle-tested Packmol
executable from Python. Prefer pure Rust? Our own
[molpack](https://molcrafts.github.io/molpack/) packer is in beta — try it.

```python
from molpy.pack import Packmol, Target, InsideBoxConstraint

target = Target(typed.to_frame(), 500, InsideBoxConstraint(length=30.0))
packed = Packmol()([target], seed=42)     # one clash-free Frame
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">05 · Export</div>

### [Write files your engine actually runs](user-guide/11_io/)

One call per file: LAMMPS data plus force-field coefficients. GROMACS, PDB,
and Zarr (`MolStore`) writers share the same pattern.

```python
packed.simbox = mp.Box.cubic(30.0)
mp.io.write_lammps_data("system.data", packed, atom_style="full")
mp.io.write_lammps_forcefield("system.ff", ff)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">06 · Analyze</div>

### [Turn trajectories into observables](compute/)

Feed the same Frame straight into the Rust-backed compute kernel — neighbor
search and g(r) in two calls, thirty more analyses behind them.

```python
from molpy.compute import NeighborList, RDF

neighbors = NeighborList(cutoff=8.0)(packed)
result    = RDF(n_bins=50, r_max=8.0)(packed, neighbors)   # g(r) over the box
```

</article>

</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">By design</span>

## Engineered to build on

A library first, not a black box: one shared data model, a Rust core, and
explicit seams throughout — take a single piece, swap another out, or extend
any of it without forking.

</div>

<dl class="molcrafts-feature-matrix molpy-feature-matrix molpy-feature-cards">
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/></svg></span>
<dt><a href="tutorials/02_block_and_frame/">One data structure, whole ecosystem</a></dt>
<dd>Every MolCrafts tool speaks the same molrs-backed abstract data structure — molpack, molvis, and molmcp read it directly. No converters, no glue code between libraries.</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></span>
<dt><a href="developer/molrs-backend/">A Rust kernel under everything</a></dt>
<dd><code>Frame</code>, <code>Block</code>, and every compute operator are backed by molrs — a Rust column store with zero-copy NumPy views and O(N) linked-cell neighbor search.</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg></span>
<dt><a href="user-guide/15_mcp/">Built for LLM agents</a></dt>
<dd>The molmcp suite serves MolPy's symbols, docs, and live structures over MCP — an agent inspects your <code>Frame</code> and calls the real API, grounded in the source rather than guessed.</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg></span>
<dt><a href="developer/architecture-overview/">Use one piece or all of them</a></dt>
<dd>Parser, builder, typifier, packer, I/O, and compute talk only through explicit data — no hidden shared state. Import the single layer you need and ignore the rest.</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.01a2.5 2.5 0 1 0 3.259-3.259c-.482-.196-.933-.558-1.01-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 12 1.998c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z"/></svg></span>
<dt><a href="developer/extending-compute/">Nothing is hardcoded</a></dt>
<dd>Register a new compute operator, I/O format, force-field style, or typifier from outside the core — the internal catalogs are open registries, not baked-in lists.</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/></svg></span>
<dt><a href="developer/coding-style/">Typed end to end</a></dt>
<dd>Public APIs carry full type hints, checked in CI with Astral's <code>ty</code>. Your editor autocompletes real signatures instead of falling back to <code>Any</code>.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molpy-ecosystem-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Ecosystem</span>

## One carefully designed abstract data structure — no glue code

The carefully designed abstract data structure you build here is the same
object every MolCrafts tool reads. One shared data model, so nothing between
them needs an adapter.

</div>

<div class="molpy-stack-feature">
<div class="molpy-stack-feature__viewer">
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/dist/model-viewer.min.js"></script>
<script src="assets/vendor/mv-init.js"></script>
<model-viewer
  class="molpy-aspirin"
  data-src="assets/models/aspirin.glb"
  alt="Aspirin molecule rendered as a ball-and-stick model"
  camera-controls
  disable-pan
  auto-rotate
  auto-rotate-delay="0"
  rotation-per-second="26deg"
  interaction-prompt="none"
  environment-image="neutral"
  shadow-intensity="0.28"
  shadow-softness="1"
  exposure="1.08"
  camera-orbit="0deg 76deg auto"
  loading="eager"></model-viewer>
</div>
<div class="molpy-stack-feature__body">
<h3>molvis — interactive 3D molecules</h3>
<p>Render the core abstract data structure in the browser: a standalone
JavaScript library, an embeddable viewer for Jupyter notebooks, and an
editor — one GPU renderer everywhere. Drag the aspirin to rotate it.</p>
<p><a class="molpy-stack-link" href="https://github.com/MolCrafts/molvis">Explore molvis →</a></p>
</div>
</div>

<dl class="molpy-integration-grid molpy-stack-grid">
<div>
<dt><a href="https://molcrafts.github.io/molpack/">molpack</a></dt>
<dd>Molecular packing engine — the same clash-free packer as a CLI, a Rust crate, and a Python package.</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molmcp">molmcp</a></dt>
<dd>MCP server for LLM agents — graph-based code discovery plus live ecosystem providers.</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molrs">molrs</a></dt>
<dd>The shared Rust molecular kernel — the core abstract data structure, with Python, WASM, and C bindings.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molpy-section--flip" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Integrations</span>

## Plays with the tools you already run

External tools connect through explicit adapters and wrappers — every
integration optional, every boundary visible.

</div>

<dl class="molpy-integration-grid molpy-integration-list">
<div>
<dt><a href="api/adapter/">RDKit</a></dt>
<dd>Bidirectional <code>Atomistic</code> ↔ <code>Mol</code> conversion for embedding, conformers, and SMILES export.</dd>
</div>
<div>
<dt><a href="user-guide/13_ambertools_integration/">AmberTools</a></dt>
<dd>antechamber, parmchk2, and tleap driven programmatically for GAFF charges and topologies.</dd>
</div>
<div>
<dt><a href="user-guide/09_packing/">Packmol</a></dt>
<dd>Clash-free packing into periodic boxes through a typed constraint interface.</dd>
</div>
<div>
<dt><a href="user-guide/12_engine/">LAMMPS · CP2K · OpenMM</a></dt>
<dd>Complete, ready-to-run input decks generated from MolPy data objects.</dd>
</div>
<div>
<dt><a href="developer/molrs-backend/">molrs · MCP</a></dt>
<dd>A Rust column store and compute kernel underneath; the MCP suite exposes symbols and docs to LLM agents.</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Find your page</span>

## The manual

The manual splits by intent. **Tutorials** teach: get running, then the data
model, chapter by chapter — read them once. **Guides** do: end-to-end task
recipes — reach for one whenever you have a task in hand.

</div>

<div class="molcrafts-doc-map molpy-doc-map">
<section>
<h3><a href="tutorials/">Tutorials</a></h3>
<p><strong>Learn.</strong> Install, run your first system in fifteen minutes, then the data model chapter by chapter.</p>
</section>
<section>
<h3><a href="user-guide/">Guides</a></h3>
<p><strong>Do the work.</strong> End-to-end recipes — parse, build, typify, pack, export — that assume the tutorials.</p>
</section>
<section>
<h3><a href="compute/">Compute</a></h3>
<p>Trajectory analysis: distributions, transport, order parameters, spectra, and analysis workflows.</p>
</section>
<section>
<h3><a href="api/">API Reference</a></h3>
<p>Every public module, from core data structures to engine adapters.</p>
</section>
<section>
<h3><a href="developer/">Developer Guide</a></h3>
<p>Contributing workflow, architecture overview, and the extension points for new capabilities.</p>
</section>
</div>

</section>

</div>
