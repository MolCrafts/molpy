# Configure and Log a Run

One file says which programs a run uses, where they live and how they are
launched; one call says where the run's log goes. The code that builds and
simulates the system names neither.

---

## The settings an engine or wrapper reads live in one place

Every engine (`LammpsEngine`, `GromacsEngine`, `OpenmmEngine`, `Cp2kEngine`)
and every AmberTools wrapper (antechamber, tleap, parmchk2, prepgen, sander)
runs an external program. What it needs to know about that program is not
part of the science:

| Setting | Meaning | Engines | Wrappers |
|---------|---------|:-------:|:--------:|
| `executable` | the program, a name looked up in the environment or a path | ✓ | ✓ |
| `env`, `env_manager` | the conda env (name or prefix) or venv prefix it lives in; `"conda"` or `"venv"` | ✓ | ✓ |
| `launcher` | the MPI / scheduler prefix, e.g. `["mpirun", "-np", "16"]` | ✓ | |
| `env_vars` | variables set for the subprocess, e.g. `OMP_NUM_THREADS` | ✓ | ✓ |
| `timeout` | seconds before the subprocess is killed | ✓ | ✓ |

These are settings in molpy's configuration (`molpy.config`, built on
[molcfg](https://docs.molcrafts.org/molcfg/)), not constructor arguments. A
tool reads its own table — `[engine.lammps]`, `[wrapper.tleap]` — and falls
back to its group's, `[engine]` or `[wrapper]`, for any key its table leaves
unset. `[conda] executable` names the `conda` that `conda run` invokes.

## Four layers, later winning

| Layer | Where | Typical content |
|-------|-------|-----------------|
| `defaults` | molpy itself (`molpy.config.DEFAULTS`) | `gmx`, `python`, `cp2k.psmp`, the AmberTools names; LAMMPS takes the first of `lmp`, `lmp_serial`, `lmp_mpi` it finds |
| `user` | `~/.molcrafts/molpy/config/config.toml` (`MOLCRAFTS_HOME` moves `~/.molcrafts`) | what this machine has: the conda, the AmberTools env |
| `project` | `molpy.toml` in the project directory (the working directory by default) | what this study runs with: binaries, launchers, limits |
| `run` | the dict passed to `load_config` | one run's exception |

Each key is taken from the last layer that sets it; tables such as
`env_vars` merge across layers, lists such as `launcher` are replaced. A
missing file is skipped. An unknown key or a value of the wrong type is
refused when the configuration loads, with the key and the files read in the
message — a typo never silently falls back to a default.

## One file drives the run

The project file for this guide: LAMMPS from an MPI build under `srun`,
GROMACS and every AmberTools program from conda environments.

```python
from pathlib import Path

project = Path("14_output")
project.mkdir(exist_ok=True)
(project / "molpy.toml").write_text(
    """\
[engine]
timeout = 7200                    # every engine: two hours at most

[engine.env_vars]
OMP_NUM_THREADS = "1"

[engine.lammps]
executable = "lmp_mpi"
launcher = ["srun", "--ntasks=16"]

[engine.gromacs]
env = "gromacs-2024"
env_manager = "conda"

[wrapper]                         # every AmberTools program
env = "AmberTools25"
env_manager = "conda"
""",
    encoding="utf-8",
)
```

`load_config` merges the layers; `tool_settings` resolves one tool and says
where each value came from, as `(layer, key)`:

```python
from molpy.config import load_config, tool_settings

config = load_config(project_dir=project)  # project_dir defaults to the working directory

lammps = tool_settings(config, "engine.lammps")
print(lammps.executable, lammps.launcher)  # lmp_mpi ('srun', '--ntasks=16')
print(lammps.timeout, lammps.env_vars)  # 7200.0 {'OMP_NUM_THREADS': '1'}
print(lammps.sources["launcher"])  # ('project', 'engine.lammps.launcher')
print(lammps.sources["timeout"])  # ('project', 'engine.timeout')

tleap = tool_settings(config, "wrapper.tleap")
print(tleap.env, tleap.env_manager)  # AmberTools25 conda
print(tleap.sources["executable"])  # ('defaults', 'wrapper.tleap.executable')
```

The engines and wrappers take that configuration as it is. Nothing else in
the script mentions a binary, an environment or a launcher:

```python
from molpy.engine import GromacsEngine, LammpsEngine

lmp = LammpsEngine(config=config, check_executable=False)
print(lmp._build_full_command(["-in", "system.in"]))
# ['srun', '--ntasks=16', 'lmp_mpi', '-in', 'system.in']

gmx = GromacsEngine(config=config, check_executable=False)
print(gmx.environment)  # EnvironmentSpec(env='gromacs-2024', env_manager='conda')
```

`check_executable=False` only skips the lookup at construction, for a page
that runs where LAMMPS is not installed; leave it on in a real run, so a
missing binary is reported before any file is written.

### One run's exception is the `run` layer

A short test run on a login node: same project, no launcher, a tighter time
limit. The override is a dict shaped like the file and wins over every file:

```python
smoke = load_config(
    {"engine": {"lammps": {"launcher": [], "timeout": 300}}},
    project_dir=project,
)
settings = tool_settings(smoke, "engine.lammps")
print(settings.launcher, settings.timeout)  # () 300.0
print(settings.sources["timeout"])  # ('run', 'engine.lammps.timeout')
print(smoke.meta("engine.lammps.executable"))
# {'history': ('defaults', 'project'), 'source': 'project'}
```

`config.meta(key)` is molcfg's record of the layers that set a key, in
order (`history`), and of the last one (`source`).

## Every run is logged as structured records

The engines and wrappers log through [mollog](https://docs.molcrafts.org/mollog/),
one logger per subsystem:

| Logger | Emits |
|--------|-------|
| `molpy.engine.lammps`, `.gromacs`, `.openmm`, `.cp2k` | every subprocess; `inputs written` from `generate_inputs`; `relaxation finished` from LAMMPS `minimize` / `md` |
| `molpy.wrapper.antechamber`, `.tleap`, `.parmchk2`, `.prepgen`, `.sander` | every subprocess; `step output missing` when a step exits 0 without its file |
| `molpy.config` | `config loaded` (the layers read), `tool settings resolved` (each setting's source), at DEBUG |

A subprocess is `process started` (DEBUG), then `process finished` (INFO) or
`process failed` (ERROR), or `process timed out` / `process not started`
(ERROR). Each record carries fields, not just text: `command` (the argv
list), `cwd`, `returncode`, `elapsed_s`, `timeout_s`, and for GROMACS `step`
(`grompp` or `mdrun`).

molpy never configures logging itself. The application does, once — here,
INFO to the terminal and DEBUG as JSON lines to a file:

```python
import mollog

mollog.configure(
    level="INFO",
    filename=project / "run.log",
    file_level="DEBUG",
    file_formatter=mollog.JSONFormatter(),
)
mollog.get_logger("molpy.config").set_level("INFO")  # no config chatter
```

Writing a deck is logged like a run. The frame and force field are the
[Simulation Engines](12_engine.md) guide's water:

```python
import json

import molpy as mp

water = mp.Frame(
    blocks={
        "atoms": {
            "type": ["OW", "HW", "HW"],
            "charge": [-0.834, 0.417, 0.417],
            "mol_id": [1, 1, 1],
            "x": [0.0, 0.9572, -0.24],
            "y": [0.0, 0.0, 0.927],
            "z": [0.0, 0.0, 0.0],
        },
        "bonds": {"atomi": [0, 0], "atomj": [1, 2], "type": ["OW-HW"] * 2},
    }
)
water.box = mp.Box.cube(20.0)
ff = mp.ff.forcefield.ForceField("tip3p", units="real")
atoms = ff.def_style("atom", "full")
ow = atoms.def_type("OW", mass=15.999, charge=-0.834, element="O")
hw = atoms.def_type("HW", mass=1.008, charge=0.417, element="H")
ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=450.0, r0=0.9572)
pairs = ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
pairs.def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
pairs.def_type("HW", hw, epsilon=0.0, sigma=0.0)

lmp.generate_inputs(water, ff, project / "lammps")

record = json.loads((project / "run.log").read_text(encoding="utf-8").splitlines()[-1])
print(record["logger_name"], record["message"])  # molpy.engine.lammps inputs written
print(record["files"]["input"])  # system.in
```

The run itself goes through the same configuration: `srun --ntasks=16
lmp_mpi`, at most two hours, `OMP_NUM_THREADS=1`. Its records land in the
same file:

```python
# docs: skip — launches LAMMPS under srun
relaxed = lmp.minimize(water, ff, workdir=project / "minimize")
```

```json
{"timestamp": "…", "level": "INFO", "logger_name": "molpy.engine.lammps", "message": "process finished", "command": ["srun", "--ntasks=16", "lmp_mpi", "-in", "system.in", "-log", "log.lammps", "-screen", "none"], "cwd": "14_output/minimize", "returncode": 0, "elapsed_s": 41.7}
```

Levels follow the logger tree: `mollog.get_logger("molpy.wrapper").set_level("WARNING")`
keeps the AmberTools programs to failures while the engines still report
each run; `mollog.get_logger("molpy").set_level("ERROR")` silences molpy
except for failures.

---

## See also

- [Simulation Engines](12_engine.md) — what each engine writes and runs
- [PEO–LiTFSI with AmberTools](13_ambertools_integration.md) — the AmberTools wrappers driven from a `[wrapper]` table
- API: [`molpy.config`](../api/config.md), [`molpy.engine`](../api/engine.md), [`molpy.wrapper`](../api/wrapper.md)
