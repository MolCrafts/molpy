---
title: MolPy
description: A Python toolkit for building, editing, and parameterizing complex molecular systems.
hide:
  - navigation
  - toc
hero:
  kicker: MolPy Manual
  title: MolPy
  description: Build, edit, and parameterize complex molecular systems in Python.
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

<div class="molcrafts-manual-home" markdown>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Parameterize</span>

## Typed and relaxed

Coordinates, force-field types, and a minimized geometry in a few calls.

[Typifier guide →](user-guide/06_typifier/)

</div>

```python
import molpy as mp

# SMILES → 3D
mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()
mol, _ = mp.conformer.Conformer(seed=42).generate(mol)

# OPLS-AA typing
typifier = mp.ff.typifier.OplsAaTypifier()
frame = typifier.typify(mol).to_frame()
ff = typifier.forcefield()

# minimize
potentials = mp.ff.compile.PotentialCompiler(ff).compile(frame)
frame, report = mp.optimize.Lbfgs(potentials, fmax=0.05).minimize(frame)
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Assemble</span>

## Any architecture

Define a unit once; a CGsmiles string sets the shape — chain, ring, star.

[Assembly guide →](user-guide/02_assembly/)

</div>

```python
# a unit with two ports
eo = mp.conformer.Conformer(seed=42).generate(
    mp.io.smiles.SmilesIr.from_fragment("[<]OCC[>]").to_template()
)[0]
assembler = mp.builder.Assembler({"EO": eo}, mp.builder.GrowthPlacer())

def build(cgsmiles):
    sites = mp.io.cgsmiles.CgSmilesIr(cgsmiles).to_coarsegrain()
    return assembler.assemble(sites, mp.Atomistic)

# shape from CGsmiles
chain = build("{[#EO]|10}")
ring = build("{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}")
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Materials</span>

## Beyond molecules

Nanotubes and graphene sheets, bonded across the periodic boundary.

[Nanostructures guide →](user-guide/04_nanostructures/)

</div>

```python
from molpy.builder import CarbonTubeBuilder, GrapheneBuilder

# (10,10) nanotube
tube = CarbonTubeBuilder(10, 10, length=50.0, periodic=True)
cnt = tube.build()
cnt.box = tube.cell(vacuum=12.0)

# graphene sheet
sheet = GrapheneBuilder(8, 8, periodic_xy=True).build()
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Simulate</span>

## Engine-ready inputs

LAMMPS, GROMACS, CP2K, and OpenMM inputs from one system and force field.

[Engines guide →](user-guide/12_engine/)

</div>

```python
frame.box = mp.Box.cube(30.0)
# molecule IDs
frame["atoms"]["mol_id"] = (
    mp.Topology.from_frame(frame).connected_components() + 1
)
# cutoffs
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0

lammps = mp.engine.LammpsEngine(check_executable=False)
lammps.generate_inputs(frame, ff, "./lammps")

gromacs = mp.engine.GromacsEngine(check_executable=False)
gromacs.generate_inputs(frame, ff, "./gromacs")
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--compact" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Analyze</span>

## Analysis in place

RDF, diffusion, order parameters, and spectra, in compiled code.

[Compute section →](compute/)

</div>

```python
from molpy.compute import Rdf

# 8 Å neighbor list
nl = mp.core.NeighborList(8.0)
nl.build(frame.coords, frame.box)

# g(r)
rdf = Rdf(n_bins=50, r_max=8.0, r_min=0.0).compute(pairs=[nl.neighbors()])
```

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--body-col" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Ecosystem</span>

## One shared structure

[molpack](https://docs.molcrafts.org/molpack/),
[molvis](https://github.com/MolCrafts/molvis), and
[molmcp](https://github.com/MolCrafts/molmcp) read it directly. Drag to rotate.

</div>

<figure id="fig-aspirin" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body">
    <!-- XYZ must start on the same line as <template> (no leading blank line). -->
    <molvis-viewer format="xyz" representation="ball-and-stick" controls="view" height="260px">
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

**Figure 1.** Aspirin in molvis.
</figure>

</section>

<section class="molcrafts-manual-section molcrafts-manual-section--body-col" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">Where to start</span>

## Start here

</div>

<dl class="molcrafts-link-list">
<div>
<dt><a href="getting-started/quickstart/">Quickstart</a></dt>
<dd>SMILES to LAMMPS input, end to end.</dd>
</div>
<div>
<dt><a href="tutorials/">Tutorials</a></dt>
<dd>The data model, chapter by chapter.</dd>
</div>
<div>
<dt><a href="user-guide/">Guides</a></dt>
<dd>One recipe per task.</dd>
</div>
<div>
<dt><a href="compute/">Compute</a></dt>
<dd>Trajectory analysis and its theory.</dd>
</div>
<div>
<dt><a href="api/">API Reference</a></dt>
<dd>Every public class and function.</dd>
</div>
</dl>

</section>

</div>
