# Engine

MD / simulation engine abstractions for LAMMPS, GROMACS, OpenMM and CP2K.
Every engine has `run`. LAMMPS, GROMACS and OpenMM also have one input
writer each, with one signature: `generate_inputs(frame, forcefield,
output_dir, *, ...)` returns the written paths by name, `"input"` always the
runnable input `run` takes. CP2K runs a CP2K input you write; it generates
no inputs.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `LammpsEngine` | `generate_inputs(frame, ff, out)` → `input` (the input script), `data`, `settings`, `init`; `minimize` / `md` relax a frame through the same deck | LAMMPS decks and relaxations |
| `GromacsEngine` | `generate_inputs(frame, ff, out)` → `input` (`em.mdp`), `gro`, the whole `top` (`write_top_system`), `nvt` (`nvt.mdp`); `run` grompp's and mdrun's an `.mdp` | GROMACS input sets |
| `OpenmmEngine` | `generate_inputs(frame, ff, out, simulation=…)` → `input` (the Python script), `pdb`, `forcefield` (XML with residue templates) | Running OpenMM simulations |
| `OpenmmSimulationConfig` | OpenMM run configuration, validated on construction | Configuring an OpenMM run |
| `Cp2kEngine` | Runs a CP2K input you write (no `generate_inputs`) | Running CP2K simulations |
| `Script` | An immutable (name, text, path) value; what `run` takes | Writing or loading an input script |

## Key behavior

- An engine's executable, environment (`env` + `env_manager`), `launcher`,
  `env_vars` and `timeout` are its `[engine.<name>]` settings in molpy's
  configuration, falling back to `[engine]` ([`molpy.config`](config.md));
  the constructor takes `config`, `workdir` and `check_executable`; they are
  resolved by the shared [`mp.tool.Tool`](tool.md) core
- `run(scripts, *, workdir, capture_output, check)` writes the scripts (the
  first is the input) and runs the engine; it keeps no state and does not
  change the scripts. `check=True` is the default. GROMACS's `run` also takes
  `structure=`, the coordinates `grompp -c` reads: `<prefix>.gro` by default,
  the previous stage's output (`"em.gro"`) for a later stage
- Every failure is an [`mp.tool.ToolError`](tool.md) carrying `tool`
  (`"engine.lammps"`), `returncode` and `tail` — the program's stderr or, when
  nothing was captured, its log (`log.lammps`, `cp2k.out`, GROMACS's
  `<stem>.log`)
- Every subprocess is logged to `molpy.engine.<name>` (`lammps`, `gromacs`,
  `openmm`, `cp2k`) as mollog records carrying the `command`, `cwd`,
  `returncode` and `elapsed_s` (GROMACS adds `step`: `grompp` / `mdrun`);
  `generate_inputs` logs `inputs written` with the files
- The decks write the force field's own nonbonded settings (each pair
  style's `cutoff`; `coul/long/pme` long-range, `coul/cut` cut off); an
  explicit argument overrides them
- A frame without a periodic box is a non-periodic system in every engine:
  LAMMPS shrink-wraps it, OpenMM uses `NoCutoff`; a method that needs a
  periodic cell (Ewald / PME / PPPM, `CutoffPeriodic`, NPT) is refused for it,
  and GROMACS, periodic only, refuses it

## Related

- [Guide: Simulation Engines](../user-guide/12_engine.md)
- [Guide: Configure and Log a Run](../user-guide/14_configure_and_log.md)
- [API: Config](config.md)
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

### OpenMM

::: molpy.engine.OpenmmEngine

::: molpy.engine.OpenmmSimulationConfig
