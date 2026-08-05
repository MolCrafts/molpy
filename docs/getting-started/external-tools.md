# Optional external tools

`pip install molcrafts-molpy` is enough for the default path: parse, build,
embed, typify, pack, export, and analyze on the **molrs** stack (plus
[molpack](https://docs.molcrafts.org/molpack/) for packing). No system
scientific binaries are required.

Anything that shells out to another package or executable is **optional**.
This page is the only place those integrations are documented as prerequisites.

## Default path (molrs)

| Task | Use |
|------|-----|
| Parse SMILES / SMARTS | `molpy.parser` (`SmilesIR`, `SmartsPattern`) |

| 3D coordinates | `molpy.conformer.Conformer` (molrs) |
| Graph assembly / polymers | `molpy.builder` (native) |
| Pack a box | `molpy.pack` → molpack |
| OPLS-AA / CL&P / MMFF typing | `molpy.typifier` |
| Trajectory analysis | `molpy.compute` (molrs kernels) |
| Files (PDB, LAMMPS data, XML FF, …) | `molpy.io` |

Workflow guides and the [Quickstart](quickstart.md) assume only this path.

## AmberTools (GAFF polymer builds)

Kept for GAFF charges, residue templates, and Amber-backed polymer construction:

| Surface | Role |
|---------|------|
| `molpy.builder.AmberTools` | antechamber → parmchk2 → tleap for one molecule |
| `molpy.builder.AmberPolymerBuilder` | CGSmiles polymers via prepgen + tleap |
| `molpy.typifier.AmberToolsTypifier` | Apply GAFF types from an AmberTools run |
| `molpy.wrapper` (`AntechamberWrapper`, `Parmchk2Wrapper`, `PrepgenWrapper`, `TLeapWrapper`) | Thin subprocess wrappers |

Install AmberTools in its own conda env (example):

```bash
conda create -n AmberTools25 -c conda-forge ambertools=25
conda activate AmberTools25
which antechamber tleap prepgen parmchk2
```

Pass the env into the facade when you call it:

```python
# docs: skip — AmberTools.parameterize shells out; wrappers unit-tested with mocks
import molpy as mp
from molpy.builder import AmberTools

mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.read_smiles("CCO")
)  # antechamber needs 3D coordinates
amber = AmberTools(work_dir="amber_work", env="AmberTools25", env_manager="conda")
result = amber.parameterize(mol, name="ligand", net_charge=0)
# result.frame, result.forcefield, and the Amber intermediates under work_dir
```

End-to-end recipes that use this path:

- [AmberTools electrolyte workflow](../user-guide/13_ambertools_integration.md)
- [Building a Crosslinked Gel](../user-guide/16_crosslinked_gel.md) (GAFF chain segment)

Unit tests never shell out to antechamber/tleap — wrappers are mocked under
`tests/test_wrapper`. Offline recipes in the user guide mark those blocks with
`# docs: skip` so the doc gate does not re-run them.

## MD engines (input decks and optional run)

`molpy.engine` always **writes** input for LAMMPS, CP2K, and OpenMM. Launching
a binary is optional:

| Engine | Generate | Run |
|--------|----------|-----|
| `LAMMPSEngine` | control script + data/ff you already wrote | `lmp` / `lmp_serial` on `PATH` |
| `CP2KEngine` | CP2K input | `cp2k` on `PATH` |
| `OpenMMEngine` | PDB + XML + `simulate.py` | Python with `openmm` importable for `run` / `serialize_system` |

```python
from molpy.engine import LAMMPSEngine

engine = LAMMPSEngine(check_executable=False)  # generate / write only
# engine.run(script, workdir="run")          # needs a LAMMPS binary
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
- [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md) — how wrappers differ from adapters
