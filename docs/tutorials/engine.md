# The Engine Module Bridges Python Data and MD Programs

This page describes how MolPy's `engine` module translates Python-level molecular data into engine-readable input files, and how those files can either be handed to an external scheduler or executed directly from Python.

---

## The problem is always the same

You have built a molecular system, typed its atoms, and exported the coordinate and force field files. Now you face the last mile: LAMMPS expects a control script with `units`, `atom_style`, and `run` commands; CP2K requires a structured `&GLOBAL` / `&FORCE_EVAL` input; OpenMM needs a PDB, an XML force field file, and a Python driver script. Each program has its own invocation syntax and its own conventions for where to look for files. The `engine` module handles this translation without mixing it into your modeling code.

**An Engine is MolPy's adapter between Python data objects and a specific MD program — it knows how to write engine-readable input files and how to invoke the executable.**

What the engine module does not do is equally important. It does not build molecules, assign atom types, or analyze trajectories. It does not write LAMMPS data files or force field coefficient files — that is `mp.io`'s job. The engine is responsible for exactly one thing: the *control script* that tells the MD program what physics to simulate and where to find the coordinate data that `mp.io` already wrote.

---

## Generating files costs nothing; running does

Think of an Engine like a laboratory instrument controller that operates in two distinct modes. In "print the protocol" mode the controller writes out the full experimental procedure — every setting, every step — without pressing start. In "run the instrument" mode it executes that same procedure. The underlying protocol is identical in both cases. The only difference is whether the button is pressed.

This separation is intentional. It lets you inspect and hand-edit the generated files before committing to a run, or copy them to an HPC cluster and submit them to a job scheduler without ever touching the engine's `run()` method. Generate-only is not a degraded mode; it is the primary workflow on any system where the MD binary is not installed locally.

For LAMMPS and CP2K the two steps are fully decoupled: you build a `Script` object yourself and either save it to disk or pass it to `run()`. For OpenMM the engine can generate all three required files — PDB, XML force field, and Python simulation script — from MolPy's `Frame` and `ForceField` objects via `generate_inputs()`, and that method does not require OpenMM to be installed.

---

## Act 1 — Generating input files without running

### LAMMPS: writing a control script to disk

The `Script` class holds the text of an input file and knows how to write it to disk. `Script.from_text` creates one from a string. Before saving you can call `script.preview()` to inspect the content — useful when the script is assembled programmatically from many fragments.

```python
import molpy as mp
from molpy.engine import LAMMPSEngine
from molpy.core.script import Script

lammps_input = """\
units           real
atom_style      full
read_data       system.data
include         system.ff

pair_style      lj/cut/coul/long 12.0
kspace_style    pppm 1.0e-4

thermo          1000
run             500000
"""

script = Script.from_text("input", lammps_input, language="other")
print(script.preview())          # inspect before saving

script.save("./submit/input.lmp")
# -> ./submit/input.lmp written
```

The saved file, together with `system.data` and `system.ff` produced by `mp.io.write_lammps_system`, is a complete LAMMPS job. Drop all three into a Slurm submission script and the cluster needs nothing from MolPy.

`Script.from_path` is the mirror image — load an existing file, modify it programmatically, and save it back or pass it to `run()`.

```python
script = Script.from_path("./submit/input.lmp")
```

### OpenMM: letting the engine build the files

OpenMM's workflow is more tightly integrated because the three required files are interdependent: the Python driver script embeds the filenames of the PDB and XML force field. Rather than assembling these by hand, `OpenMMEngine.generate_inputs()` accepts MolPy's own data objects and writes all three files consistently.

The configuration is a Pydantic model — `OpenMMSimulationConfig` — whose fields document their units explicitly. It round-trips through JSON, which makes it easy to store alongside the generated files for reproducibility.

```python
from molpy.engine import OpenMMEngine, OpenMMSimulationConfig

config = OpenMMSimulationConfig(
    ensemble="NPT",
    temperature=300.0,       # K
    pressure=1.0,            # bar
    timestep_fs=2.0,         # fs
    n_steps=500_000,
    platform="CUDA",
)
config.to_json("./omm_run/config.json")

engine = OpenMMEngine(check_executable=False)
paths = engine.generate_inputs(frame, ff, config, "./omm_run")
# paths -> {"pdb": Path("./omm_run/system.pdb"),
#            "forcefield": Path("./omm_run/forcefield.xml"),
#            "script": Path("./omm_run/simulate.py")}
```

`check_executable=False` tells the engine not to verify that `python` is on PATH at construction time. This is the right choice whenever you are only generating files — the Python interpreter that will eventually run the simulation may be on a different machine entirely.

The returned `paths` dictionary maps string keys to `Path` objects. You can pass `paths["script"]` directly to `engine.run()` later, or hand all three files to a cluster job that has OpenMM installed.

---

## Act 2 — Running the engine directly from Python

### Local execution with LAMMPS

When the MD binary is available locally, `engine.run()` writes the script to a working directory and launches the subprocess. The return value is a standard `subprocess.CompletedProcess`, so you can inspect the exit code, stdout, and stderr without any MolPy-specific handling.

```python
engine = LAMMPSEngine("lmp")

result = engine.run(
    script,
    workdir="./calc",
    capture_output=True,
    check=True,
)
print(result.returncode)          # 0 on success
if result.stderr:
    print(result.stderr[:500])
```

`check=True` causes `run()` to raise `subprocess.CalledProcessError` on a non-zero exit code — the same semantics as `subprocess.run`. Set `check=False` during automated parameter scans where you want to continue after a failed run and inspect the log file yourself.

### MPI and job-scheduler launchers

MPI parallelism is configured at engine construction, not at runtime. Passing `launcher` prepends the MPI command before the LAMMPS executable in the subprocess call.

```python
# OpenMPI
engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])

# Slurm srun (common on HPC clusters)
engine = LAMMPSEngine("lmp", launcher=["srun", "--ntasks=16"])

result = engine.run(script, workdir="./calc")
```

The command that runs is `mpirun -np 16 lmp -in input.lmp -log log.lammps -screen none`. The `-screen none` flag is added automatically to prevent LAMMPS from writing per-timestep data to stdout, which avoids pipe-buffer deadlocks when `capture_output=True`.

### Conda environment activation

Some HPC workflows install LAMMPS or OpenMM inside a Conda environment that is not active in the submission environment. Providing both `env` and `env_manager` wraps the subprocess call with `conda run`:

```python
engine = LAMMPSEngine(
    "lmp",
    env="lammps-env",
    env_manager="conda",
)
result = engine.run(script, workdir="./calc")
# runs: conda run --no-capture-output -n lammps-env lmp -in input.lmp ...
```

`env` and `env_manager` must be provided together or omitted together — the engine raises `ValueError` if only one is given.

### Running OpenMM after generating inputs

Once `generate_inputs()` has produced the files, calling `run()` with the script path launches the generated Python driver under the configured interpreter.

```python
engine = OpenMMEngine("python", env="openmm-env", env_manager="conda")
paths = engine.generate_inputs(frame, ff, config, "./omm_run")

result = engine.run(paths["script"], workdir="./omm_run", capture_output=True)
```

The driver script is self-contained — it imports OpenMM, reads the PDB and XML files from the same directory, and runs. You can edit `simulate.py` by hand between `generate_inputs` and `run` without touching Python.

---

## Choose the right engine for the right problem

LAMMPS is the right choice for classical MD with complex bonded force fields, reactive systems using `fix bond/react`, or any workflow that requires LAMMPS-specific fix commands. CP2K is for QM/MM, DFT energy evaluations, and ab initio molecular dynamics where quantum effects are essential. OpenMM is for GPU-accelerated classical and alchemical MD, and for cases where you want the simulation logic expressed in Python — `generate_inputs` produces an editable, human-readable script rather than a binary job file.

Use generate-only when you are submitting to a cluster scheduler, when you want to inspect or edit the files before running, or when the engine binary is not installed on the machine running MolPy. Use `run()` for local prototyping, automated parameter sweeps, and CI validation runs where you need the return code and log output inline.

---

## Adding a new engine takes three methods

Every engine subclasses `Engine` and implements three things: a `name` property that returns a human-readable identifier, `_get_default_extension()` that returns the file extension for the primary input file, and `_execute()` that builds the subprocess command and calls `subprocess.run`. The base class handles working-directory management, script normalization, launcher prefixing, and Conda environment wrapping.

```python
from molpy.engine.base import Engine
import subprocess
from pathlib import Path

class GromacsEngine(Engine):
    @property
    def name(self) -> str:
        return "GROMACS"

    def _get_default_extension(self) -> str:
        return ".mdp"

    def _execute(self, run_dir: Path, capture_output=False,
                 check=True, timeout=None, **kwargs):
        cmd = self._build_full_command(
            ["grompp", "-f", self.input_script.path.name,
             "-o", "topol.tpr"]
        )
        return subprocess.run(cmd, cwd=run_dir,
                              capture_output=capture_output,
                              text=True, check=check, timeout=timeout)
```

`_build_full_command` prepends the launcher and Conda wrapper automatically.

---

## See also

- [I/O Subsystem](io.md) — writing LAMMPS data files, force field coefficient files, PDB and GRO files; the engine assumes these files exist before it runs.
- [PEO–LiTFSI Electrolyte via AmberTools](../user-guide/07_ambertools_integration.md) — an end-to-end workflow that writes AMBER input files and invokes external tools, illustrating the same generate-then-run pattern applied to a different toolchain.
- API Reference: `molpy.engine`, `molpy.core.script.Script`.
