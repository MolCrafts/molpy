# molcrafts-molrs

[![PyPI](https://img.shields.io/pypi/v/molcrafts-molrs.svg)](https://pypi.org/project/molcrafts-molrs/)

Python bindings for the [molrs](https://github.com/MolCrafts/molrs) molecular modeling toolkit.

Install with `pip install molcrafts-molrs` and `import molrs`. Full docs:
<https://docs.molcrafts.org/molrs/>.

## Install

```bash
pip install molcrafts-molrs
```

Requires Python 3.12+.

## Quick start

```python
import molrs

# SMILES → atomistic graph (class API under molrs.io)
mol = molrs.io.smiles.SmilesIr.parse("CCO").to_atomistic()

# 3D coordinates
from molrs.conformer import Conformer

mol, report = Conformer().generate(mol)

# Force field: typify → pairs → potentials
from molrs.ff.typifier import Mmff94Typifier
from molrs.ff.potential import intramolecular_pairs
from molrs.ff.compile import PotentialCompiler

typifier = Mmff94Typifier()
typed = typifier.typify(mol)
frame = typed.to_frame()
ff = typifier.forcefield()  # a copy of exactly the types typify assigned
frame["pairs"] = intramolecular_pairs(frame, ff)
pots = PotentialCompiler(ff).compile(frame)
energy, forces = pots.evaluate(frame)
assert forces.shape == (frame["atoms"].n_rows, 3)
```

## Package layout

The top level is the subsystems, plus the core data classes a user handles
directly (`molrs.Frame is molrs.core.Frame`: `Frame`,
`Block`, `Trajectory`, `Box`, `MolGraph`, `Atomistic`, `CoarseGrain`, the
entity classes, `Element`, `Topology`). Everything else has one path, named
after its Rust owner.

| Import | Owns |
|--------|------|
| `molrs.core` | `Frame`, `Block`, `Trajectory`, frame metadata; `Box` and its `Cell`, neighbour search, regions, `TriMesh`, `Trace`; `MolGraph`, `Atomistic`, `CoarseGrain` and their live views, `Element`, `Topology`; `Unit`, `Quantity`, `UnitPreset`, `UnitRegistry` |
| `molrs.core.keys` / `.schema` / `.constants` | the column vocabulary, its specifications, and every physical and engine constant |
| `molrs.io` | Every file reader and writer: one door per format and direction, `read_<fmt>(source)` / `write_<fmt>(sink, frame)` (a source is a `pathlib.Path`, `bytes` or `molrs.io.Text`; a sink a `pathlib.Path` or a `bytearray`); each format's classes in its own submodule — the multi-frame `<Fmt>Reader` / `<Fmt>Writer` of `io.pdb`, `io.xyz`, `io.gro`, `io.mol2`, `io.cif`, `io.sdf`, `io.dcd`, `io.trr`, `io.xtc`, `io.lammps`, and the force-field reader / writer classes of `io.lammps`, `io.gromacs`, `io.amber`, `io.openmm_xml`; `io.smiles`, `io.cgsmiles`, `io.mrec` |
| `molrs.io.mrec` | `*.mrec` store pieces: `MrecFile` (per-section reads), streaming `SequenceSchema`, `MrecWriter`, `MrecReader`, `ForceFieldSection`, `pack_mrec_zip`, `validation` (a whole record is written by `molrs.io.write_mrec`) |
| `molrs.ff.*` | `forcefield`, `potential`, `typifier`, `charge`, `ir`, `params`, `clpol_scaling` |
| `molrs.optimize` | `Lbfgs`, `OptimizationReport` |
| `molrs.md` | Integrators and the `MD` driver |
| `molrs.compute` | One class per `molrs::compute` type, under its core name (flat): RDF, MSD, correlations, fits, spectra, order, clusters, … |
| `molrs.conformer` | 3D generation |
| `molrs.perceive` | Rings, aromaticity, SMARTS, reactions |
| `molrs.builder` | Structure builders, site-graph assembly, `Coarsener` |

An analysis is configured when it is built and given its data when it
computes: `Steinhardt(l=[6])` then `.compute(...)`. The constructor takes
configuration only, every optional setting is `None` unless given (all
defaults live in the core; `Msd.spec()` lists them), and `compute` takes the
data as keyword arguments named after the core input's fields. `Msd`, `Vacf`
and `Rdf` are streams, as in the core: `acc = Msd(max_lag).accumulator()`,
`acc.push(positions, dt=...)` per frame, then `acc.finish()`. A result is a read-only object that the next compute accepts as is —
`LinearFit(0.2, 0.8).compute(correlation=msd_result, channel=0)`.

Kernels take `dt` in the time unit of your trajectory, and time-valued
results come back in that unit. MSD needs **unwrapped** coordinates
(`Unwrap` produces them). VACF is the unbiased \(C(\tau)\) used for
Green–Kubo D and VDOS.

## Development

```bash
maturin develop --release
pytest -q
```
