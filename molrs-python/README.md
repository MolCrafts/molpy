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
mol = molrs.io.smiles.SmilesIr("CCO").to_atomistic()

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
energy, forces = pots.calc_energy_forces(frame)
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
| `molrs.core` | `Frame`, `Block`, `Trajectory`, frame metadata; `Box`, neighbour search, regions, `TriMesh`, `Trace`; `MolGraph`, `Atomistic`, `CoarseGrain` and their live views, `Element`, `Topology`; `Unit`, `Quantity`, `UnitPreset`, `UnitRegistry` |
| `molrs.core.keys` / `.schema` / `.constants` | the column vocabulary, its specifications, and every physical and engine constant |
| `molrs.io` | Every file reader and writer (structure, trajectory, force-field files, `*.mrec`, SMILES) as `read_<fmt>[_<what>]` / `write_<fmt>[_<what>]` (`_str` / `_bytes` in memory); each format's classes in its own submodule: `io.pdb`, `io.xyz`, `io.gro`, `io.dcd`, `io.trr`, `io.xtc`, `io.lammps`, `io.smiles`, `io.cgsmiles`, `io.mrec` |
| `molrs.io.mrec` | `*.mrec` store pieces: streaming `SequenceSchema`, `MrecWriter`, `MrecReader`, `ForceFieldSection`, `section_names`, `pack_mrec_zip`, `validation` (whole records: `molrs.io.read_mrec_frame` / `write_mrec_frame` and partners) |
| `molrs.ff.*` | `forcefield`, `potential`, `typifier`, `charge`, `ir`, `params`, `clpol_scaling` |
| `molrs.optimize` | `Lbfgs`, `OptimizationReport` |
| `molrs.md` | Integrators and the `MD` driver |
| `molrs.compute` | RDF, MSD, transport, dielectric, … (flat) |
| `molrs.conformer` | 3D generation |
| `molrs.perceive` | Rings, aromaticity, SMARTS, reactions |
| `molrs.builder` | Structure builders, site-graph assembly, `Coarsener` |

Analysis kernels take `dt` in the time unit of your trajectory, and
time-valued results come back in that unit. MSD needs **unwrapped**
coordinates. VACF is the unbiased \(C(\tau)\) used for Green–Kubo D and VDOS.

## Development

```bash
maturin develop --release
pytest -q
```
