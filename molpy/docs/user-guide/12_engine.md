# The Engine Module Bridges Python Data and MD Programs

An `Engine` turns your typed system into ready-to-run input for LAMMPS, GROMACS, OpenMM, or CP2K — and either hands the files to your scheduler or presses start itself.

---

## The problem is always the same

You have built a molecular system, typed its atoms, and exported the coordinate and force field files. Now you face the last mile: LAMMPS expects a control script with `units`, `atom_style`, and `run` commands; CP2K requires a structured `&GLOBAL` / `&FORCE_EVAL` input; OpenMM needs a PDB, an XML force field file, and a Python driver script. Each program has its own invocation syntax and its own conventions for where to look for files. The `engine` module handles this translation without mixing it into your modeling code.

**An Engine is MolPy's adapter between Python data objects and a specific MD program — it knows how to write engine-readable input files and how to invoke the executable.**

What the engine module does not do is equally important. It does not build molecules, assign atom types, or analyze trajectories, and it formats no file itself: the structure and force-field files come from molrs's writers (`mp.io`). Each engine has **one** input writer, `generate_inputs`, which calls those writers and adds the control files that tell the MD program what physics to simulate and where to find them.

---

## Generating files costs nothing; running does

Think of an Engine like a laboratory instrument controller that operates in two distinct modes. In "print the protocol" mode the controller writes out the full experimental procedure — every setting, every step — without pressing start. In "run the instrument" mode it executes that same procedure. The underlying protocol is identical in both cases. The only difference is whether the button is pressed.

This separation is intentional. It lets you inspect and hand-edit the generated files before committing to a run, or copy them to an HPC cluster and submit them to a job scheduler without ever touching the engine's `run()` method. Generate-only is not a degraded mode; it is the primary workflow on any system where the MD binary is not installed locally.

`LammpsEngine.generate_inputs(frame, ff, out)` writes a complete LAMMPS deck, `GromacsEngine.generate_inputs(frame, ff, out)` a GROMACS input set, and `OpenmmEngine.generate_inputs(frame, ff, out, simulation=config)` the PDB, XML force field and Python driver; none of them needs the program installed. All three share one signature and return the written paths by name, `"input"` always being the runnable input `run()` takes. For a hand-written control script (and for CP2K, which generates no inputs), you build a `Script` object yourself and either save it to disk or pass it to `run()`.

---

## Act 1 — Generating input files without running

### LAMMPS: the whole deck from a typed frame

`generate_inputs` writes four files: the data file, the force-field settings
(molrs's include: every `*_style` line the frame's types use, their
coefficients, `special_bonds`, `pair_modify`), an init with `units`,
`atom_style`, `boundary` and `neighbor`, and the input script that includes
them and runs a starter minimise + NVT body (`body=` replaces it). The init
states the data file's own `atom_style` and `units`: `atom_style=None` (the
default) is the style the frame's atom columns hold (`full` with `mol_id` and
`charge`, `atomic` with neither; `mp.io.lammps.infer_lammps_atom_style`), and
`units=` (`real`, `metal` or `lj`; any other is refused) is what molrs
converts both the data file and the coefficients to. A frame
with a periodic box gets `boundary p p p`; one without (an
`Atomistic.to_frame()`) gets a shrink-wrapped, non-periodic deck.

```python
import molpy as mp

water = mp.Frame(
    blocks={
        "atoms": {
            "type": ["OW", "HW", "HW"],
            "element": ["O", "H", "H"],
            "charge": [-0.834, 0.417, 0.417],
            "mol_id": [1, 1, 1],
            "x": [0.0, 0.9572, -0.24],
            "y": [0.0, 0.0, 0.927],
            "z": [0.0, 0.0, 0.0],
        },
        "bonds": {"atomi": [0, 0], "atomj": [1, 2], "type": ["OW-HW"] * 2},
        "angles": {"atomi": [1], "atomj": [0], "atomk": [2], "type": ["HW-OW-HW"]},
    }
)
water.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))

ff = mp.ff.forcefield.ForceField("tip3p", units="real")
atoms = ff.def_style("atom", "full")
ow = atoms.def_type("OW", mass=15.999, charge=-0.834, element="O")
hw = atoms.def_type("HW", mass=1.008, charge=0.417, element="H")
ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)
ff.def_style("angle", "harmonic").def_type("HW-OW-HW", hw, ow, hw, k=55.0, theta0=104.52)
pairs = ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
pairs.def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
pairs.def_type("HW", hw, epsilon=0.0, sigma=0.0)
ff.def_style("pair", "coul/cut", {"cutoff": 10.0})  # the charges' pair term

deck = mp.engine.LammpsEngine(check_executable=False).generate_inputs(
    water, ff, "./lammps_run"
)
print(sorted(deck))  # ['data', 'init', 'input', 'settings']
```

### GROMACS: structure, topology and run parameters

```python
gmx = mp.engine.GromacsEngine(check_executable=False)
files = gmx.generate_inputs(water, ff, "./gromacs_run", temperature=300.0)
print(sorted(files))  # ['gro', 'input', 'nvt', 'top']
```

The `.top` is the whole topology `mp.io.write_top_system`
writes: the force field's directives, one `[ moleculetype ]` per molecule
(by the atoms' `mol_id`) with each row's parameters, `[ system ]` and
`[ molecules ]` — what `grompp -p` reads. GROMACS excludes every pair within
three bonds, so the frame carries its angles (and dihedrals:
`Atomistic.generate_topology`). `files["input"]` is `em.mdp`;
`run(Script.from_path(files["input"]))` grompp's
the `.mdp` against the `.gro` and `.top`, then mdrun's it (under the engine's
`launcher`). Each later stage starts from the structure the stage before it
wrote — `mdrun -deffnm em` writes `em.gro` — which `run` is told by
`structure=`; without it, a stage starts from the initial `.gro`:

```python
# docs: skip — runs grompp and mdrun
from molpy.engine import Script

gmx.run(Script.from_path(files["input"]), workdir="./gromacs_run")  # EM
gmx.run(
    Script.from_path(files["nvt"]), workdir="./gromacs_run", structure="em.gro"
)  # NVT from the minimised structure
```

### A frame without a box is a non-periodic system

Every engine follows one periodicity policy. A frame with a periodic box is a
periodic system; a frame without one (an `Atomistic.to_frame()` carries
none) is a non-periodic, vacuum system. LAMMPS shrink-wraps it
(`boundary s s s`, and `minimize` / `md` relax it the same way), and OpenMM
computes it without periodic images (`NoCutoff` by default,
`CutoffNonPeriodic` on request). Whatever needs a periodic cell is refused for
it, by name, before any file is written: a long-range Coulomb style
(LAMMPS's PPPM), OpenMM's `PME`, `Ewald` and `CutoffPeriodic`, and NPT. GROMACS
refuses a box-free frame altogether: its Verlet cutoff scheme simulates
periodic systems only, so a molecule in vacuum is given a box larger than
itself plus twice the cutoff.

### Nonbonded settings are the force field's

No engine replaces the force field's nonbonded treatment with one of its own.
The force field states it on its pair styles: each style's `cutoff`, and its
Coulomb style — `coul/long/pme` is long-range (Ewald / PME), `coul/cut` and
`coul/charmm` are cut off. Each deck writes exactly that, and an explicit
argument overrides it:

| Engine | From the force field | Override | Where it states nothing |
|--------|----------------------|----------|-------------------------|
| LAMMPS (`generate_inputs`, `minimize`, `md`) | its own `pair_style` line and cutoffs; `kspace_style pppm 1.0e-4` for a long-range Coulomb style | `pair_style=` | — |
| GROMACS (`.mdp`) | `coulombtype` (`PME` / `Cut-off`), `rcoulomb`, `rvdw` (nm) | `coulombtype=`, `rcoulomb=`, `rvdw=` | `PME`, 1.2 nm |
| OpenMM (`createSystem`) | `nonbondedMethod` (`PME` / `CutoffPeriodic`), `nonbondedCutoff` | `OpenmmSimulationConfig(nonbonded_method=, nonbonded_cutoff_nm=)` | `PME`, 1.0 nm |

A 12 Å `lj/cut` + `coul/cut` force field therefore runs at 12 Å with cut-off
Coulomb in all three. OpenMM's cutoff methods add a reaction field (its
`NonbondedForce` has no plain truncated Coulomb), and molrs's GROMACS and
OpenMM writers do not yet hold `coul/long/pme`.

### LAMMPS: writing a control script by hand

A `Script` is an immutable value: a name, the text of an input file and, optionally, the path it came from. `Script.from_text` creates one from a string literal (dedented, ending in one newline); `script.save(path)` writes it and `script.preview()` shows its first lines. Neither changes the script.

The examples below share this setup:

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(200, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box(mp.core.Cell.lengths([20.0] * 3))
```

```python
import molpy as mp
from molpy.engine import LammpsEngine
from molpy.engine import Script

lammps_input = """\
units           real
atom_style      full
read_data       system.data
include         system.ff   # styles, mixing, special_bonds, coefficients

timestep        1.0
fix             integrate all nve
thermo          1000
run             500000
"""

script = Script.from_text("input", lammps_input)
print(script.preview())  # inspect before saving

script.save("./submit/input.lmp")
# -> ./submit/input.lmp written
```

The include written by `mp.io.write_lammps_forcefield` already declares every style, so the script does not repeat them; a different `pair_style` issued after it would discard its pair coefficients (for long-range electrostatics, write the include with `skip_pair_style=True` and declare the pair style in the script before `include`). Because the script states `units` before `read_data`, write the include with `skip_units=True`: LAMMPS refuses a `units` line once the box exists. The saved control script, together with the `system.data` and `system.ff` pair produced by `mp.io.write_lammps_data` and `mp.io.write_lammps_forcefield` (there is no bundled `.in` — the control script above *is* the input deck, written separately by `Script.save`), is a complete LAMMPS job. Drop all three into a Slurm submission script and the cluster needs nothing from MolPy.

`Script.from_path` is the mirror image — it reads an existing file as is, to save elsewhere or pass to `run()`.

```python
script = Script.from_path("./submit/input.lmp")
```

### OpenMM: letting the engine build the files

OpenMM's workflow is more tightly integrated because the three required files are interdependent: the Python driver script embeds the filenames of the PDB and XML force field, and `createSystem` matches every residue of the PDB against a `<Residue>` template of the XML by its atom names. Rather than assembling these by hand, `OpenmmEngine.generate_inputs()` accepts MolPy's own data objects and writes all three files consistently: the PDB with `mp.io.write_pdb`, the XML with `mp.io.write_openmm_system` (the force field plus the PDB's residue templates).

The simulation is a dataclass — `OpenmmSimulationConfig`, passed as the keyword-only `simulation=` — whose fields document their units explicitly. It is validated when it is built (an unknown `ensemble`, `nonbonded_method` or `constraints`, such as `"nvt"` or `"EWALD"`, raises `ValueError`, from `from_json` too) and round-trips through JSON, which makes it easy to store alongside the generated files for reproducibility. `NVE` runs a `VerletIntegrator` from velocities drawn at `temperature`; `minimize` writes the minimised coordinates to `output_pdb`.

```python
from pathlib import Path

from molpy.engine import OpenmmEngine, OpenmmSimulationConfig

Path("./omm_run").mkdir(parents=True, exist_ok=True)

simulation = OpenmmSimulationConfig(
    ensemble="NVT",  # one water: NPT (pressure=, in bar) is for a condensed box
    temperature=300.0,  # K
    timestep_fs=2.0,  # fs
    n_steps=500_000,
    platform="CUDA",
)
simulation.to_json("./omm_run/config.json")

engine = OpenmmEngine(check_executable=False)
paths = engine.generate_inputs(water, ff, "./omm_run", simulation=simulation)
# paths -> {"input": Path("./omm_run/simulate.py"),
#           "pdb": Path("./omm_run/system.pdb"),
#           "forcefield": Path("./omm_run/forcefield.xml")}
```

`check_executable=False` tells the engine not to verify that its Python interpreter (`engine.openmm.executable`, `python` by default) is on PATH at construction time. This is the right choice whenever you are only generating files — the Python interpreter that will eventually run the simulation may be on a different machine entirely.

The returned `paths` dictionary maps string keys to `Path` objects. You can pass `paths["input"]` directly to `engine.run()` later, or hand all three files to a cluster job that has OpenMM installed.

---

## Act 2 — Running the engine directly from Python

### Local execution with LAMMPS

When the MD binary is available locally, `engine.run()` writes the script(s) to a working directory and launches the subprocess; the first script is the input. Every call names what it runs: the engine keeps no state between calls, and the script is not changed. The return value is a standard `subprocess.CompletedProcess`, so you can inspect the exit code, stdout, and stderr without any MolPy-specific handling.

```python
# docs: skip — engine.run launches an MD binary; engines unit-tested with mocks / script literals
engine = LammpsEngine()  # the first of lmp, lmp_serial, lmp_mpi found

result = engine.run(
    script,
    workdir="./calc",
    capture_output=True,
    check=True,
)
print(result.returncode)  # 0 on success
if result.stderr:
    print(result.stderr[:500])
```

`check=True` (the default) makes `run()` raise `molpy.tool.ToolError` on a non-zero exit code. The error carries the tool (`error.tool == "engine.lammps"`), the exit code (`error.returncode`) and the tail of the program's stderr or, when nothing was captured, its log file (`error.tail` — `log.lammps` for LAMMPS, `cp2k.out` for CP2K). A missing executable, a program that cannot start and a timeout raise the same error. Set `check=False` during automated parameter scans where you want to continue after a failed run and inspect the log file yourself.

### Executables, launchers and environments are configuration

Which binary runs, under which MPI or scheduler launcher, in which conda or
venv environment, with which variables and time limit, is not an argument of
the engine: it is the engine's table in molpy's configuration, read when the
engine is constructed. A `molpy.toml` in the project directory:

```toml
[engine.lammps]
executable = "lmp"
launcher = ["mpirun", "-np", "16"]   # or ["srun", "--ntasks=16"] under Slurm
timeout = 7200                       # seconds
```

```python
# docs: skip — engine.run launches an MD binary; engines unit-tested with mocks / script literals
engine = LammpsEngine()  # reads [engine.lammps] from ./molpy.toml
result = engine.run(script, workdir="./calc")
```

The command that runs is `mpirun -np 16 lmp -in input.lmp -log log.lammps -screen none`. The `-screen none` flag is added automatically to prevent LAMMPS from writing per-timestep data to stdout, which avoids pipe-buffer deadlocks when `capture_output=True`.

For one run that differs from the file — a short test without the launcher,
say — pass the exception to `load_config`; it is the last layer and wins:

```python
# docs: skip — engine.run launches an MD binary; engines unit-tested with mocks / script literals
from molpy.config import load_config

config = load_config({"engine": {"lammps": {"launcher": [], "timeout": 300}}})
result = LammpsEngine(config=config).run(script, workdir="./smoke")
```

[Configure and Log a Run](14_configure_and_log.md) has the four layers
(package defaults, `~/.molcrafts/molpy/config/config.toml`, `molpy.toml`,
the run), the keys every engine and AmberTools program reads, and the log records each
run leaves.

### Conda environment activation

Some HPC workflows install LAMMPS or OpenMM inside a Conda environment that is not active in the submission environment. Setting `env` and `env_manager` in the engine's table wraps the subprocess call with `conda run`:

```toml
[engine.lammps]
env = "lammps-env"
env_manager = "conda"
```

```python
# docs: skip — engine.run launches an MD binary; engines unit-tested with mocks / script literals
result = LammpsEngine().run(script, workdir="./calc")
# runs: conda run --no-capture-output -n lammps-env lmp -in input.lmp ...
```

`env` and `env_manager` are set together or not at all — the engine raises `ValueError` if only one is given. Put them in `[engine]` instead and every engine runs in that environment.

### Running OpenMM after generating inputs

Once `generate_inputs()` has produced the files, calling `run()` with the script path launches the generated Python driver under the configured interpreter.

```python
# docs: skip — needs conda env `openmm-env` with OpenMM; engines unit-tested with mocks
from molpy.config import load_config

# `config` is molpy's configuration (which interpreter runs the script);
# `simulation` is what the script simulates.
config = load_config({"engine": {"openmm": {"env": "openmm-env", "env_manager": "conda"}}})
engine = OpenmmEngine(config=config)
paths = engine.generate_inputs(water, ff, "./omm_run", simulation=simulation)
assert paths["input"].exists()

# The launch itself is a real simulation, so it is the one step this page does
# not perform for you:
#
#     result = engine.run(paths["input"], workdir="./omm_run", capture_output=True)
#
# which becomes: conda run --no-capture-output -n openmm-env python simulate.py
```

The driver script is self-contained — it imports OpenMM, reads the PDB and XML files from the same directory, and runs. You can edit `simulate.py` by hand between `generate_inputs` and `run` without touching Python.

---

## Choose the right engine for the right problem

LAMMPS is the right choice for classical MD with complex bonded force fields, reactive systems using `fix bond/react`, or any workflow that requires LAMMPS-specific fix commands. CP2K is for QM/MM, DFT energy evaluations, and ab initio molecular dynamics where quantum effects are essential. OpenMM is for GPU-accelerated classical and alchemical MD, and for cases where you want the simulation logic expressed in Python — `generate_inputs` produces an editable, human-readable script rather than a binary job file.

Use generate-only when you are submitting to a cluster scheduler, when you want to inspect or edit the files before running, or when the engine binary is not installed on the machine running MolPy. Use `run()` for local prototyping, automated parameter sweeps, and CI validation runs where you need the return code and log output inline.

---

## Adding a new engine takes a table and two methods

Every engine subclasses `Engine`, names its configuration table in `tool`, and implements two methods: `_get_default_extension()`, the file extension of an unpathed input script, and `_execute(run_dir, input_file, *, capture_output, check)`, which builds the command with `self.command(...)` and starts it with `self.run_command(...)`. The shared tool core (`molpy.tool.Tool`) resolves the settings, environment, executable, launcher, environment variables, time limit and logger; `Engine.run` writes the scripts. An engine that writes its own input set adds one `generate_inputs`, calling molrs's writers for every file it does not template itself. Its settings are the `[engine.<tool>]` table of any configuration layer — molpy accepts a table for any engine name — and its default executable is `_default_executable()` (the built-in engines state theirs in `molpy.config.DEFAULTS`).

```python
from pathlib import Path

from molpy.engine import Engine


class NamdEngine(Engine):
    tool = "namd"  # reads [engine.namd], logs to molpy.engine.namd

    def _default_executable(self) -> str:
        return "namd3"  # unless [engine.namd] executable says otherwise

    def _get_default_extension(self) -> str:
        return ".namd"

    def _execute(self, run_dir: Path, input_file: Path, *, capture_output, check):
        return self.run_command(
            self.command([input_file.name]),
            cwd=run_dir,
            capture_output=capture_output,
            check=check,
            log=run_dir / "namd.log",  # a failure carries its tail
        )
```

`command` prepends the conda prefix and the launcher; `run_command` applies the environment variables and the time limit, logs the run, and raises `ToolError` on failure.

---

## See also

- [I/O Subsystem](11_io.md) — writing LAMMPS data files, force field coefficient files, PDB and GRO files; the engine assumes these files exist before it runs.
- [Configure and Log a Run](14_configure_and_log.md) — the configuration every engine reads and the log records every run leaves.
- [PEO–LiTFSI Electrolyte via AmberTools](13_ambertools_integration.md) — an end-to-end workflow that writes AMBER input files and invokes external tools, illustrating the same generate-then-run pattern applied to a different toolchain.
- API Reference: `molpy.engine`, `mp.engine.Script`.
