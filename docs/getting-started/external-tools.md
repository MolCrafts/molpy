# Optional external tools

`pip install molcrafts-molpy` is enough for the default path: parse, build,
embed, typify, pack, export, and analyze on the default MolPy stack (plus
[molpack](https://docs.molcrafts.org/molpack/) for packing). No system
scientific binaries are required.

Anything that shells out to another package or executable is **optional**.
This page is the only place those integrations are documented as prerequisites.

## Default path

| Task | Use |
|------|-----|
| Parse SMILES / SMARTS | `mp.io.smiles.SmilesIr`, `mp.perceive.SmartsPattern` |
| 3D coordinates | `mp.conformer.Conformer` |
| Polymer assembly | `mp.builder.Assembler` on a CGsmiles site graph (native); see [Polymer Topologies](../user-guide/topology/index.md) |
| Pack a box | [molpack](https://docs.molcrafts.org/molpack/) (`molcrafts-molpack`, installed separately) |
| OPLS-AA / MMFF94 typing | `molpy.ff.typifier` |
| Trajectory analysis | `molpy.compute` (kernels) |
| Files (PDB, LAMMPS data, XML FF, …) | `molpy.io` |

Workflow guides and the [Quickstart](quickstart.md) assume only this path.

## AmberTools (GAFF parameters)

GAFF / GAFF2 types and bonded terms need no AmberTools:
`mp.ff.typifier.AtdTypifier` then `GaffTypifier`
([Force Field Typification](../user-guide/06_typifier.md)); against
AmberTools 26 they assign antechamber's types and the parameter values
tleap's prmtop carries for the same set. AM1-BCC charges need one program,
sqm: `mp.ff.charge.Am1BccModel` runs it (through antechamber `-c mul`) and
corrects its AM1 charges with the native `BccModel`, which gives antechamber
`-c bcc`'s charges for the same sqm run. That is the default route, for a
molecule on its own and for one mixed with an AMBER polymer chain: a native
force field merges with the chain's, which is read back from tleap's prmtop.

The whole AmberTools pipeline (antechamber → parmchk2 → tleap, read back
from the prmtop) is kept for what only it does. A GAFF polymer is one
oligomer (head, chain and tail monomers already bonded) typed once by
antechamber, cut into residues by prepgen and joined by tleap `sequence`.
`TleapTypifier` loads parameters of your own on top of the leaprc, and
`AntechamberTypifier` runs an antechamber charge method other than AM1-BCC /
ABCG2. A term `GaffTypifier` estimated exports to every engine: the writers
skip its provenance. Every entry point takes
molrs's `parameter_set` (`"gaff"` or `"gaff2"`), which has no default, so a
chain and the molecules it is mixed with always name the same set:

| Surface | Role |
|---------|------|
| `molpy.ff.charge.Am1BccModel` | AM1-BCC (or ABCG2) charges: sqm's AM1 charges through antechamber `-c mul`, corrected by molrs `BccModel`; net charge from the atoms' formal charges. Types and bonded terms stay native |
| `molpy.ff.typifier.AntechamberTypifier` | antechamber (types + charges, any `-c` method) → parmchk2 → tleap for one complete molecule; net charge from the atoms' formal charges. For a charge method the native route lacks, and for exporting an estimated GAFF term while molrs's writers refuse it |
| `molpy.ff.typifier.TleapTypifier` | tleap only, for a finished graph whose atoms already carry AMBER types and charges, with parameters of your own (`forcefield=`) loaded as a frcmod on top of the leaprc; a graph with ports is refused |
| `molpy.builder.AmberPolymerBuilder` | antechamber + parmchk2 on an oligomer, prepgen per residue (`AmberCut`), tleap `sequence` over a linear site graph; `AmberPieces` writes the oligomer and cuts from three SMILES |

Install AmberTools in its own conda env (example):

```bash
conda create -n AmberTools25 -c conda-forge ambertools=25
conda activate AmberTools25
which antechamber tleap prepgen parmchk2
```

Name the env once, in molpy's configuration — `molpy.toml` in your project
directory (or `~/.molcrafts/molpy/config/config.toml` for every project):

```toml
[ambertools]
env = "AmberTools25"
env_manager = "conda"
```

Every AmberTools program (antechamber, parmchk2, prepgen, tleap) then
runs through `conda run -n AmberTools25`; see
[Configure and Log a Run](../user-guide/14_configure_and_log.md) for the
layers and the per-program tables.

```python
# docs: skip — needs AmberTools (sqm); the model is unit-tested with antechamber faked
import molpy as mp
from molpy.ff.charge import Am1BccModel
from molpy.ff.typifier import AtdTypifier, GaffTypifier

mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.smiles.read_smiles("CCO").to_atomistic()
)  # sqm needs 3D coordinates
labelled = AtdTypifier(parameter_set="gaff2").typify(mol)  # GAFF2 types, native
charges = Am1BccModel(parameter_set="bcc", work_dir="amber_work").assign(labelled)
for atom, charge in zip(labelled.atoms, charges):
    atom["charge"] = float(charge)  # AM1-BCC: sqm, then the native correction
gaff = GaffTypifier(parameter_set="gaff2")
typed = gaff.typify(labelled)  # bonded terms from gaff2.dat, native
ff = gaff.forcefield()  # the parameters of the types just assigned
```

`ff` merges with an `AmberPolymerBuilder` chain's force field
(`built.forcefield.merge(ff)`). `mp.ff.typifier.AntechamberTypifier(parameter_set="gaff2")`
types the same molecule through antechamber, parmchk2 and tleap instead.

End-to-end recipes that use this path:

- [AmberTools electrolyte workflow](../user-guide/13_ambertools_integration.md)

Unit tests never shell out to antechamber/tleap — the programs and the
shared pipeline are mocked under `tests/test_tool/test_ambertools`, the
typifiers and `Am1BccModel` under `tests/test_ff`. Offline recipes in the user guide mark those blocks with
`# docs: skip` so the doc gate does not re-run them.

## MD engines (input decks and optional run)

`molpy.engine` **writes** the inputs for LAMMPS, GROMACS and OpenMM from a
typed frame and its force field (`generate_inputs`), with no binary
installed. Launching a binary is optional; CP2K only runs an input you write:

| Engine | Generate (`generate_inputs`) | Run |
|--------|------------------------------|-----|
| `LammpsEngine` | input script, data file, force-field settings, init | `lmp` / `lmp_serial` / `lmp_mpi` on `PATH` |
| `GromacsEngine` | `em.mdp`, `.gro`, `.top`, `nvt.mdp` | `gmx` on `PATH` |
| `OpenmmEngine` | `simulate.py`, PDB, force-field XML | Python with `openmm` importable |
| `Cp2kEngine` | — (write the CP2K input yourself, as a `Script`) | `cp2k.psmp` on `PATH` |

Each binary, its environment and launcher are `[engine.<name>]` settings in
molpy's configuration ([Configure and Log a Run](../user-guide/14_configure_and_log.md)).

```python
from molpy.engine import LammpsEngine

engine = LammpsEngine(check_executable=False) # generate / write only
# engine.run(script, workdir="run") # needs a LAMMPS binary
```

See [Simulation Engines](../user-guide/12_engine.md).

## Pip extras (not system tools)

These are Python package groups, not scientific executables:

| Extra | Command | Role |
|-------|---------|------|
| `dev` | `pip install molcrafts-molpy[dev]` | pytest, ruff, ty, tox |
| `doc` | `pip install molcrafts-molpy[doc]` | zensical + theme for docs builds |

## See also

- [Installation](installation.md)
