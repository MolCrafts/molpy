# Engine

MD / simulation engine abstractions for LAMMPS, GROMACS, OpenMM and CP2K.
Each engine has one input writer, `generate_inputs`, and `run`.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `LammpsEngine` | `generate_inputs(frame, ff, out)` → data, settings, init, input script; `minimize` / `md` relax a frame through the same deck | LAMMPS decks and relaxations |
| `GromacsEngine` | `generate_inputs(frame, ff, out)` → `.gro`, the whole `.top` (`write_gromacs_system`), `em.mdp`, `nvt.mdp`; `run` grompp's and mdrun's an `.mdp` | GROMACS input sets |
| `OpenmmEngine` | `generate_inputs(frame, ff, config, out)` → PDB, force-field XML, Python script | Running OpenMM simulations |
| `OpenmmSimulationConfig` | OpenMM run configuration | Configuring an OpenMM run |
| `Cp2kEngine` | Runs a CP2K input | Running CP2K simulations |
| `Script` / `ScriptLanguage` | An editable script with a path; what `run` takes | Writing or loading an input script |

## Related

- [Guide: Simulation Engines](../user-guide/12_engine.md)
- [Guide: File I/O](../user-guide/11_io.md) (the data and force-field files the script reads)

---

## Full API

### Base

::: molpy.engine.Engine

### CP2K

::: molpy.engine.Cp2kEngine

### LAMMPS

::: molpy.engine.LammpsEngine

### GROMACS

::: molpy.engine.GromacsEngine

### Script

::: molpy.engine.Script

::: molpy.engine.ScriptLanguage

### OpenMM

::: molpy.engine.OpenmmEngine

::: molpy.engine.OpenmmSimulationConfig
