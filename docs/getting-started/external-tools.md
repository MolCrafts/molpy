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
| Parse SMILES / SMARTS | `mp.SmilesIR`, `mp.SmartsPattern` |
| 3D coordinates | `mp.Conformer` |
| Polymer assembly | `mp.Assembler` on a CGsmiles site graph (native); see [Polymer Topologies](../user-guide/topology/index.md) |
| Pack a box | [molpack](https://docs.molcrafts.org/molpack/) (`molcrafts-molpack`, installed separately) |
| OPLS-AA / MMFF94 typing | `molpy.typifier` |
| Trajectory analysis | `molpy.compute` (kernels) |
| Files (PDB, LAMMPS data, XML FF, …) | `molpy.io` |

Workflow guides and the [Quickstart](quickstart.md) assume only this path.

## AmberTools (GAFF parameters)

Kept for GAFF types and charges. Each monomer is typed once as a complete
molecule; a polymer is assembled by MolPy from the typed monomers and finished
by tleap. tleap never changes types or charges, so choose each monomer so its
leaving groups mimic the chain neighbour:

| Surface | Role |
|---------|------|
| `molpy.typifier.AntechamberTypifier` | antechamber (types + charges) → parmchk2 → tleap for one complete molecule; net charge from the atoms' formal charges |
| `molpy.typifier.TLeapTypifier` | tleap only, for a graph whose atoms already carry AMBER types and charges (an assembled chain); junction terms from the leaprc |
| `molpy.wrapper` (`AntechamberWrapper`, `Parmchk2Wrapper`, `PrepgenWrapper`, `TLeapWrapper`, `SanderWrapper`) | Thin subprocess wrappers |

Install AmberTools in its own conda env (example):

```bash
conda create -n AmberTools25 -c conda-forge ambertools=25
conda activate AmberTools25
which antechamber tleap prepgen parmchk2
```

Pass the env into the typifier when you construct it:

```python
# docs: skip — needs AmberTools; typifiers unit-tested with the executables faked
import molpy as mp

mol, _ = mp.Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.read_smiles("CCO")
)  # antechamber needs 3D coordinates
ante = mp.typifier.AntechamberTypifier(
    atom_type="gaff2", charge_method="bcc",
    work_dir="amber_work", env="AmberTools25", env_manager="conda",
)
typed = ante.typify(mol)  # GAFF2 types, BCC charges, bonded terms
ff = ante.forcefield()  # the parameters of the types just assigned
```

End-to-end recipes that use this path:

- [AmberTools electrolyte workflow](../user-guide/13_ambertools_integration.md)

Unit tests never shell out to antechamber/tleap — wrappers are mocked under
`tests/test_wrapper` and the typifiers under `tests/test_typifier`. Offline recipes in the user guide mark those blocks with
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

engine = LAMMPSEngine(check_executable=False) # generate / write only
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
- [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md) — how wrappers differ from adapters
