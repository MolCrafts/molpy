---
title: MolPy
description: MolPy is the Python toolkit that ties molecular modeling together.
hide:
  - navigation
  - toc
hero:
  kicker: MolPy Manual
  title: MolPy
  description: Molecular modeling in Python.
  actions:
    - label: Get started
      href: getting-started/quickstart/
      style: primary
    - label: Examples
      href: getting-started/examples/
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
---

<h1 class="molcrafts-sr-only">MolPy</h1>

<div class="molcrafts-manual-home molpy-home" markdown>

<section class="molpy-band molpy-intro" markdown>

MolPy is the Python toolkit that ties molecular modeling together: structures,
force fields and trajectories in one model, passed cleanly between builders,
engines and analyses.

</section>

<section class="molpy-band molpy-modules" markdown>

<div class="molpy-module" markdown>

<div class="molpy-module__copy" markdown>

### [Build](user-guide/02_assembly.md)

<code class="molpy-module__name">mp.builder</code>

Molecules from SMILES, polymers from CGsmiles, crystals and nanotubes.

</div>

```python
import molpy as mp

drug = mp.io.smiles.read_smiles("CC(=O)Nc1ccc(O)cc1").to_atomistic()
drug, _ = mp.conformer.Conformer(seed=42).generate(drug)  # paracetamol, 3D

tube = mp.builder.CarbonTubeBuilder(10, 10, length=50.0, periodic=True)
```

</div>

<div class="molpy-module" markdown>

<div class="molpy-module__copy" markdown>

### [Force fields](user-guide/06_typifier.md)

<code class="molpy-module__name">mp.ff</code>

OPLS-AA and GAFF atom types, their parameters, energies and forces.

</div>

```python
typifier = mp.ff.typifier.OplsAaTypifier()
frame = typifier.typify(drug).to_frame()  # OPLS-AA atom types
ff = typifier.forcefield()  # and their parameters
```

</div>

<div class="molpy-module" markdown>

<div class="molpy-module__copy" markdown>

### [Read and write](user-guide/11_io.md)

<code class="molpy-module__name">mp.io</code>

Structures and trajectories in the formats your other tools speak.

</div>

```python
from pathlib import Path

mp.io.write_gro(Path("paracetamol.gro"), frame)
again = mp.io.read_gro(Path("paracetamol.gro"))
```

</div>

<div class="molpy-module" markdown>

<div class="molpy-module__copy" markdown>

### [Engines](user-guide/12_engine.md)

<code class="molpy-module__name">mp.engine</code>

A typed system handed to GROMACS, LAMMPS or OpenMM as ready inputs.

</div>

```python
frame.box = mp.Box.cube(40.0)  # Å

gromacs = mp.engine.GromacsEngine(check_executable=False)
files = gromacs.generate_inputs(frame, ff, "paracetamol_md")
```

</div>

<div class="molpy-module" markdown>

<div class="molpy-module__copy" markdown>

### [Analysis](compute/index.md)

<code class="molpy-module__name">mp.compute</code>

Structure, dynamics and spectra from the trajectories that come back.

</div>

```python
pairs = mp.core.NeighborList(6.0)
pairs.build(tube.build().coords, tube.cell(vacuum=12.0))

rdf = mp.compute.Rdf(n_bins=120, r_max=6.0, r_min=0.0).accumulator()
rdf.push(pairs.neighbors())
g = rdf.finish().rdf  # g(r) across the nanotube wall
```

</div>

</section>

<section class="molpy-band molpy-gallery" markdown>

## From crystals to polymers

<div class="molpy-gallery__grid">
<figure class="molpy-tile">
<div class="molpy-tile__stage"><molvis-viewer src="series/showcase/rock_salt.pdb" format="pdb" representation="ball-and-stick" controls="context-menu" background="#14291d" width="100%" height="100%"></molvis-viewer></div>
<figcaption><a href="user-guide/04_nanostructures/#bulk-crystals-from-a-cif">Rock salt</a></figcaption>
</figure>
<figure class="molpy-tile">
<div class="molpy-tile__stage"><molvis-viewer src="series/showcase/aspirin.pdb" format="pdb" representation="ball-and-stick" controls="context-menu" background="#14291d" width="100%" height="100%"></molvis-viewer></div>
<figcaption><a href="user-guide/07_conformers/">Aspirin</a></figcaption>
</figure>
<figure class="molpy-tile">
<div class="molpy-tile__stage"><molvis-viewer src="series/showcase/star.pdb" format="pdb" representation="ball-and-stick" controls="context-menu" background="#14291d" width="100%" height="100%"></molvis-viewer></div>
<figcaption><a href="user-guide/topology/04_star/">Star polymer</a></figcaption>
</figure>
<figure class="molpy-tile">
<div class="molpy-tile__stage"><molvis-viewer src="series/showcase/nanotube.pdb" format="pdb" representation="ball-and-stick" controls="context-menu" background="#14291d" width="100%" height="100%"></molvis-viewer></div>
<figcaption><a href="user-guide/04_nanostructures/">Carbon nanotube</a></figcaption>
</figure>
</div>

</section>

<section class="molpy-band molpy-resources" markdown>

## Learning resources

<div class="molpy-resources__grid">
<div class="molpy-resource"><h3><a href="tutorials/">Tutorials</a></h3><ul><li><a href="getting-started/installation/">Installation</a></li><li><a href="getting-started/quickstart/">Quickstart</a></li><li><a href="tutorials/02_block_and_frame/">Block and Frame</a></li><li><a href="tutorials/04_force_field/">Force Field</a></li><li><a href="tutorials/05_trajectory/">Trajectory</a></li></ul></div>
<div class="molpy-resource"><h3><a href="user-guide/">Guides</a></h3><ul><li><a href="user-guide/02_assembly/">Polymer assembly</a></li><li><a href="user-guide/06_typifier/">Force-field typing</a></li><li><a href="user-guide/09_packing/">Packing a box</a></li><li><a href="user-guide/12_engine/">Simulation engines</a></li><li><a href="user-guide/11_io/">File I/O</a></li></ul></div>
<div class="molpy-resource"><h3><a href="compute/">Compute</a></h3><ul><li><a href="compute/rdf/">Radial distribution</a></li><li><a href="compute/msd/">Diffusion (MSD)</a></li><li><a href="compute/hbond/">Hydrogen bonds</a></li><li><a href="compute/order/">Order parameters</a></li><li><a href="compute/spectra/">Spectra</a></li></ul></div>
<div class="molpy-resource"><h3><a href="api/">Reference</a></h3><ul><li><a href="api/">API Reference</a></li><li><a href="getting-started/examples/">Example gallery</a></li><li><a href="tutorials/glossary/">Glossary</a></li><li><a href="getting-started/faq/">FAQ</a></li></ul></div>
</div>

</section>

</div>
