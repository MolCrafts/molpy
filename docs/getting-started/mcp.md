# MCP: Letting LLM Agents Read MolPy's Source

## What MCP solves

An LLM agent working with MolPy faces the same problem as any user relying on memory: API knowledge goes stale. A guessed function name, parameter list, or return type may look plausible and still be wrong.

The MolCrafts MCP suite fixes that by exposing the installed packages as structured tools. Instead of guessing, the agent can ask what modules exist, which symbols a module exports, what a callable accepts, and how a function is implemented. The answers come from the local installation, not from the model's training data.

In practice, the workflow is simple:

1. Discover modules and symbols.
2. Read docstrings, signatures, and source when needed.
3. Write normal Python code against the real MolPy API.

!!! note "What MCP is not"
    The server does not execute MolPy workflows for the agent. It helps the agent understand the library, then the agent writes Python code that calls MolPy directly.

## Where MCP support lives

MCP support for MolPy is provided by [`molcrafts-mcp-suite`](https://github.com/MolCrafts/molcrafts-mcp-suite), the MCP integration layer for the MolCrafts ecosystem. The suite is built on `molmcp`, the shared MCP foundation that ships the Provider protocol, source-introspection tools, and safety middleware used across MolCrafts packages.

A single `molcrafts-mcp` command starts a coordinated server that exposes MolPy (and any other installed MolCrafts packages) through the same MCP session. MCP code does **not** live in MolPy itself; this keeps the chemistry library focused on its own data model and lets a single MCP launcher stay consistent across the ecosystem.

## The seven tools

The suite exposes seven introspection tools. Together they let an agent navigate the package tree, inspect the public API, and read implementation details only when necessary.

| Tool | Use it for |
| --- | --- |
| `list_modules` | Discover importable modules under a prefix such as `molpy` or `molpy.builder`. |
| `list_symbols` | Inspect the public API of a module with one-line summaries. |
| `get_docstring` | Read structured usage guidance from the source docstring. |
| `get_signature` | Check parameter names, types, defaults, and return signatures. |
| `get_source` | Inspect implementation details when docstrings are not enough. |
| `search_source` | Search the codebase by substring, like a lightweight `grep`. |
| `read_file` | Read a specific source file by line range. |

Used together, these tools let an agent explore MolPy the same way a human developer would: browse, narrow, verify, then write code.

## Install and register the server

Install the suite. It pulls in MolPy as a dependency, so a separate MolPy install is not required.

```bash
pip install molcrafts-mcp-suite
```

### Claude Code

Register the server once. Use project scope unless you want it available in every repo:

```bash
# Project-level (recommended). Writes .mcp.json in the repo root.
claude mcp add molcrafts --scope project -- molcrafts-mcp

# User-level. Writes to ~/.claude/settings.json.
claude mcp add molcrafts -- molcrafts-mcp
```

Start a new Claude Code session, then run `/mcp`. You should see `molcrafts` and the seven introspection tools.

!!! tip "Virtual environments"
    If `molcrafts-mcp` is not on your PATH, register the executable explicitly:

    ```bash
    claude mcp add molcrafts -- /path/to/venv/bin/molcrafts-mcp
    ```

    Or let `uv` launch it from a project directory:

    ```bash
    claude mcp add molcrafts -- uv run --directory /path/to/project molcrafts-mcp
    ```

### Claude Desktop

Claude Desktop reads its own config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add a `mcpServers` entry:

```json
{
  "mcpServers": {
    "molcrafts": {
      "command": "molcrafts-mcp"
    }
  }
}
```

Restart Claude Desktop after saving. The MolCrafts tools should appear in the tool picker.

### Inspecting and customizing the launch

The suite exposes a couple of options that are occasionally useful. Inspect the resolved configuration without starting a server:

```bash
molcrafts-mcp --print-config
```

Restrict introspection to MolPy alone, or expose additional roots:

```bash
molcrafts-mcp --no-default-import-roots --import-root molpy
molcrafts-mcp --import-root molpy --import-root molexp --import-root your_package
```

Serve over HTTP instead of stdio, for clients that cannot launch subprocesses:

```bash
molcrafts-mcp --transport streamable-http --host 127.0.0.1 --port 8787
```

## How MCP works

The server runs as a separate process. The client launches `molcrafts-mcp`, asks which tools are available, and then sends tool calls over JSON-RPC.

```text
┌─────────────┐   stdin/stdout    ┌──────────────────┐
│  LLM Client │ ◄──────────────► │  molcrafts-mcp   │
│  (Claude)   │   JSON-RPC msgs   │  (MCP Server)    │
└─────────────┘                   └──────────────────┘
```

A typical exchange looks like this:

1. The client starts the server and requests its capabilities.
2. The server advertises the seven introspection tools.
3. The agent calls those tools to inspect MolPy before writing code.

`stdio` is the default transport and works in most local setups. `streamable-http` and `sse` expose the same protocol over HTTP for clients that cannot launch subprocesses.

For example, an agent asked to build a polymer workflow may inspect MolPy like this:

```text
1. list_modules("molpy.builder")
2. list_symbols("molpy.builder.polymer")
3. get_signature("molpy.builder.polymer.ambertools.AmberPolymerBuilder.build")
4. get_docstring("molpy.pack.Molpack")
5. search_source("write_lammps_forcefield")
```

That pattern is the point of MCP: inspect first, code second.

## Writing effective prompts

MCP gives the agent access to MolPy's API, but it does not replace a clear task description. The best prompts specify the system and constraints, then let the agent discover the implementation.

### Describe the system, not the API

Tell the agent what you want to build. Do not tell it which function names to call.

| Too low-level | Better |
| --- | --- |
| `Use polymer() to build a PEG chain` | `Build a PEG chain with 15 repeat units` |
| `Call Molpack to pack molecules` | `Pack 15 chains into a 20 nm cubic box` |
| `Use the Box class` | `Create a periodic simulation box for the system` |

If your prompt names specific MolPy functions, it is usually too low-level.

### Include the physical parameters

Molecular workflows are defined by numbers. If you omit them, the agent has to guess.

At minimum, specify:

- molecule type
- chain length or composition
- number of molecules
- box size or density
- output format, if it matters

For example:

```text
Generate a 20 x 20 x 20 nm box containing 15 PEG chains, each with
15 repeat units. Export to LAMMPS DATA format.
```

### Keep one prompt to one workflow

Do not ask the agent to build a system, run MD, and analyze results in one step.

Instead, break the work into stages:

1. Build and pack the system.
2. Prepare the simulation inputs.
3. Run the analysis.

This matches how real modeling workflows are debugged and validated.

### State constraints, then let the agent explore

Call out constraints that change the result, such as:

- `Use the Amber backend with GAFF2 parameters`
- `Use OPLS-AA typing`

After that, let the agent inspect the API through MCP. Avoid:

- forcing a specific function name
- pasting remembered code snippets
- over-specifying the implementation

If the agent still fails after exploring, that is usually useful feedback about the API or documentation.

### Quick checklist

Before sending a prompt, check:

- Does it describe the molecular system rather than API calls?
- Are the important numbers present?
- Is this one workflow rather than several?
- Did I state the constraints that materially affect the result?

## Worked example: TIP3P water box

This first example is intentionally small. It uses only built-in MolPy data, so
it is a good sanity check that MCP is connected and the agent is reading the
local API instead of guessing from memory.

### The prompt

```text
Use MolPy to build a small periodic TIP3P water box for LAMMPS. Create
64 water molecules in a 4 x 4 x 4 grid with 0.32 nm spacing. Use MolPy's
built-in tip3p.xml force field, assign atom, bond, and angle types, and
write the result to quickstart-output/water_box_tip3p.data and
quickstart-output/water_box_tip3p.ff.
```

### Agent exploration

Claude first looks for force-field loading, typification, topology generation,
and export.

**Step 1 — find the main entry points**

```
list_symbols("molpy.io")
```

```
read_xml_forcefield      Convenience function to read an XML force field file
write_lammps_data        Write a Frame object to a LAMMPS data file
write_lammps_forcefield  Write a ForceField object to a LAMMPS force field file
```

```
list_symbols("molpy.typifier")
```

```
OplsAtomisticTypifier    OPLS-AA atomistic typifier orchestrator
```

**Step 2 — confirm the built-in TIP3P file path**

```
get_signature("molpy.io.read_xml_forcefield")
```

```
(filepath: str | Path, forcefield: AtomisticForcefield | None = None)
    -> AtomisticForcefield
```

Claude reads the docstring and sees that passing `"tip3p.xml"` loads MolPy's
built-in TIP3P force field from the package data directory.

**Step 3 — verify the non-obvious topology step**

```
get_docstring("molpy.core.atomistic.Atomistic.get_topo")
```

The docstring makes one subtle point clear: when `gen_angle=True`,
`get_topo()` returns a new `Atomistic` instead of mutating the existing one.
Claude therefore writes `system = system.get_topo(...)` rather than calling it
in place.

**Step 4 — check the box and typing APIs**

```
get_signature("molpy.core.Box.orth")
```

```
(lengths: ArrayLike, pbc: ArrayLike = ..., origin: ArrayLike = ...,
 central: bool = False) -> Box
```

```
get_signature("molpy.typifier.OplsAtomisticTypifier.__init__")
```

```
(forcefield: ForceField, skip_atom_typing: bool = False,
 skip_pair_typing: bool = False, skip_bond_typing: bool = False,
 skip_angle_typing: bool = False, skip_dihedral_typing: bool = False,
 strict_typing: bool = True)
```

With that information Claude writes the script below.

### The generated script

```python
from pathlib import Path

import numpy as np
import molpy as mp
from molpy.io import read_xml_forcefield, write_lammps_data, write_lammps_forcefield
from molpy.typifier import OplsAtomisticTypifier

theta = 1.82421813418
r_oh = 0.09572  # nm

water = mp.Atomistic(name="water_tip3p")
o = water.def_atom(element="O", name="O", x=0.0, y=0.0, z=0.0, charge=-0.834)
h1 = water.def_atom(element="H", name="H1", x=r_oh, y=0.0, z=0.0, charge=0.417)
h2 = water.def_atom(
    element="H",
    name="H2",
    x=r_oh * float(np.cos(theta)),
    y=r_oh * float(np.sin(theta)),
    z=0.0,
    charge=0.417,
)
water.def_bond(o, h1, order=1)
water.def_bond(o, h2, order=1)

system = mp.Atomistic(name="water_box_tip3p")
mol_id = 1
for iz in range(4):
    for iy in range(4):
        for ix in range(4):
            mol = water.copy()
            mol.move(delta=[ix * 0.32, iy * 0.32, iz * 0.32])
            for atom in mol.atoms:
                atom["mol_id"] = mol_id
            system.merge(mol)
            mol_id += 1

system = system.get_topo(gen_angle=True, gen_dihe=False)

ff = read_xml_forcefield("tip3p.xml")
typifier = OplsAtomisticTypifier(ff, skip_dihedral_typing=True, strict_typing=True)
system = typifier.typify(system)

frame = system.to_frame()
frame.box = mp.Box.orth([1.28, 1.28, 1.28])
frame["atoms"]["id"] = np.arange(1, frame["atoms"].nrows + 1, dtype=int)
frame["atoms"]["mol_id"] = np.asarray(frame["atoms"]["mol_id"], dtype=int)
frame["atoms"]["charge"] = np.asarray(frame["atoms"]["charge"], dtype=float)

out_dir = Path("quickstart-output")
out_dir.mkdir(parents=True, exist_ok=True)
write_lammps_data(out_dir / "water_box_tip3p.data", frame, atom_style="full")
write_lammps_forcefield(out_dir / "water_box_tip3p.ff", ff)
```

### Output

```
atoms 192
bonds 128
angles 64
files ['water_box_tip3p.data', 'water_box_tip3p.ff']
```

This example stays completely local: no AmberTools, no Packmol, and no
literature lookup. It is usually the fastest way to confirm that the MCP client
can inspect MolPy, synthesize a correct script, and export a real simulation
input.

## Worked example: polydisperse PEO/LiTFSI electrolyte

The next example is much larger. The MCP server is still doing the same job,
but the agent also has to inspect AmberTools-facing builders, packing APIs, and
force-field merge behavior before it can write the full workflow.

### The prompt

```text
Use MolPy to generate an atomistic PEO/LiTFSI polymer electrolyte system with
the following strict constraints. Build polydisperse PEO chains using a
Schulz–Zimm distribution with a target number-average degree of polymerization
(DP_n = 20) and polydispersity index (PDI = 1.20). Construct exactly 40 PEO
chains. Use AmberTools to generate the force-field parameters and connectivity
for the PEO monomer and polymer chains, using GAFF with chemically correct
linkage and end-group handling. Add LiTFSI salt at a fixed composition of
EO:Li = 20:1, and compute the exact number of LiTFSI molecules from the total
number of EO repeat units in the sampled polymer ensemble. Look up literature
for Li+ nonbond parameters. Pack with Packmol at a very low initial density of
0.10 g/cm³. The workflow should be fully end-to-end: define the PEO repeat unit
and LiTFSI, sample chain lengths from the Schulz–Zimm distribution, build all
PEO chains, assign parameters with AmberTools, add LiTFSI, pack the full system,
and export coordinates and force-field files for downstream molecular dynamics.
```

### Agent exploration

Claude starts by calling `list_modules` to orient itself, then drills into the modules it needs.

**Step 1 — discover the package structure**

```
list_modules("molpy")
```

Returns 148 modules across 18 top-level packages (excerpt):

```
molpy.builder   Crystal and polymer builders (AmberTools integration, stochastic generation)
molpy.io        I/O for AMBER, LAMMPS, PDB, GRO, MOL2, XYZ ...
molpy.pack      Packing (constraints, targets, Packmol integration)
molpy.parser    Parsers for SMILES, BigSMILES, CGSmiles, GBigSMILES
molpy.wrapper   External tool wrappers (antechamber, parmchk2, prepgen, tleap)
```

**Step 2 — find the distribution and polymer-builder classes**

```
list_symbols("molpy.builder.polymer")
```

```
SchulzZimmPolydisperse    Schulz-Zimm molecular weight distribution for polydisperse polymer chains
UniformPolydisperse       Uniform distribution over degree of polymerization
PoissonPolydisperse       Poisson distribution for degree of polymerization
FlorySchulzPolydisperse   Flory-Schulz (geometric) distribution
PolydisperseChainGenerator  Middle layer: samples DP/mass, generates monomer sequences
SystemPlanner             Top layer: accumulates chains until a target total mass is reached
AmberPolymerBuilder       Polymer builder backed by the AmberTools pipeline
PolymerBuilder            CGSmiles-based polymer builder with pluggable typifier
```

**Step 3 — read the Schulz–Zimm signature and docstring**

```
get_signature("molpy.builder.polymer.distributions.SchulzZimmPolydisperse.__init__")
```
```
(Mn: float, Mw: float, random_seed: int | None = None)
```

```
get_docstring("molpy.builder.polymer.distributions.SchulzZimmPolydisperse")
```
```
Schulz-Zimm molecular weight distribution for polydisperse polymer chains.
Implements MassDistribution — sampling is done directly in molecular-weight space.

The PDF is:
    f(M) = z^(z+1)/Γ(z+1) · M^(z−1)/Mn^z · exp(−zM/Mn)
where z = Mn/(Mw − Mn).  Equivalent to Gamma(shape=z, scale=Mw−Mn).

Args:
    Mn: Number-average molecular weight (g/mol).
    Mw: Weight-average molecular weight (g/mol), must satisfy Mw > Mn.

Methods:
    sample_mass(rng) → float     draw one mass sample
    mass_pdf(mass_array) → ndarray
```

Claude notes: z = 1/(PDI − 1) = 5.0 for PDI = 1.20.

**Step 4 — understand AmberPolymerBuilder**

```
get_signature("molpy.builder.polymer.ambertools.AmberPolymerBuilder.__init__")
```
```
(library: dict[str, Atomistic],
 force_field: str = "gaff2",
 charge_method: str = "bcc",
 work_dir: Path = Path("amber_work"),
 env: str = "AmberTools25",
 env_manager: str = "conda")
```

```
get_docstring("molpy.builder.polymer.ambertools.AmberPolymerBuilder.build")
```
```
Build a polymer from a CGSmiles string.

Args:
    cgsmiles: CGSmiles notation, e.g. "{[#MeH][#EO]|10[#MeT]}"
              |N means N repeat units of the preceding monomer.

Returns:
    AmberBuildResult with .frame (Frame) and .forcefield (ForceField).

Pipeline (automatic):
    antechamber  → GAFF atom types + BCC charges (mol2 + ac files)
    parmchk2     → missing torsion/vdW parameters (frcmod)
    prepgen      → HEAD / CHAIN / TAIL residue variants (prepi)
    tleap        → build polymer and generate prmtop / inpcrd
```

**Step 5 — check the Li⁺ parameter source**

Claude does not guess at Li⁺ LJ parameters — it searches the codebase for the canonical reference:

```
search_source("Aqvist")
```
```
tests/test_e2e_peo_litfsi.py:147:    """Write Åqvist (1990) Li+ frcmod and build prmtop via tleap."""
tests/test_e2e_peo_litfsi.py:149:    # σ = 2 * Rmin/2 / 2^(1/6); Rmin/2 = 1.137 Å → σ = 2.026 Å
tests/test_e2e_peo_litfsi.py:150:    R_MIN_HALF = 1.137   # Å
tests/test_e2e_peo_litfsi.py:151:    EPS_LI     = 0.0183  # kcal/mol
```

This confirms the parameters: **Åqvist (1990), J. Phys. Chem. 94, 8021**.
Rmin/2 = 1.137 Å, ε = 0.0183 kcal/mol. Claude writes these directly into a frcmod file.

**Step 6 — find the packing and export interfaces**

```
list_symbols("molpy.pack")
```
```
Molpack                  High-level Packmol packing interface
InsideBoxConstraint      Place molecules inside a rectangular box
OutsideBoxConstraint     Keep molecules outside a box
InsideSphereConstraint   Sphere constraint
MinDistanceConstraint    Minimum pairwise distance
Target                   One packing target (frame + count + constraint)
```

```
get_signature("molpy.pack.Molpack.optimize")
```
```
(max_steps: int = 20000, seed: int = 12345) → Frame
```

```
search_source("write_lammps_forcefield")
```
```
src/molpy/io/__init__.py:  from molpy.io.writers import write_lammps_forcefield
src/molpy/io/writers.py:   def write_lammps_forcefield(path, forcefield, skip_pair_style=False)
```

```
get_signature("molpy.io.write_lammps_forcefield")
```
```
(path: Path | str,
 forcefield: ForceField,
 precision: int = 6,
 skip_pair_style: bool = False) → None
```

Claude notes: `skip_pair_style=True` is needed so the LAMMPS input script can control `kspace_style` independently.

**Step 7 — confirm ForceField.merge**

```
get_signature("molpy.core.forcefield.ForceField.merge")
```
```
(other: ForceField) → ForceField
```

```
get_docstring("molpy.core.forcefield.ForceField.merge")
```
```
Merge two ForceField objects.  Returns a new ForceField containing all styles
and parameters from both.  Raises if incompatible styles are found.
```

With this information Claude has everything it needs to assemble the script.

!!! note "The full generated script"
    The full end-to-end script is at `docs/user-guide/08_peo_litfsi_electrolyte.py`. It runs antechamber/parmchk2/tleap for TFSI⁻, builds a Li⁺ frcmod from Åqvist parameters, samples 40 chain lengths from a Schulz–Zimm distribution, calls `AmberPolymerBuilder` per unique DP, merges the three force fields, packs with Packmol at 0.10 g/cm³, and exports a LAMMPS data file and `system.ff` to `peo_litfsi_output/lammps/`. Running it requires AmberTools and Packmol in a conda environment named `AmberTools25`.

## See Also

- [molcrafts-mcp-suite](https://github.com/MolCrafts/molcrafts-mcp-suite) — install, command-line options, and provider auto-discovery
- [Tool Layer](../tutorials/tools.md) — what the Tool workflows do and when to use them
- [Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb) — end-to-end workflow from distribution to LAMMPS export
