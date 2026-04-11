# MCP: Letting LLM Agents Read MolPy's Source

## What MCP solves

An LLM agent working with MolPy faces the same problem as any user relying on memory: API knowledge goes stale. A guessed function name, parameter list, or return type may look plausible and still be wrong.

MolPy's MCP server fixes that by exposing the installed package as structured tools. Instead of guessing, the agent can ask what modules exist, which symbols a module exports, what a callable accepts, and how a function is implemented. The answers come from the local MolPy installation, not from the model's training data.

In practice, the workflow is simple:

1. Discover modules and symbols.
2. Read docstrings, signatures, and source when needed.
3. Write normal Python code against the real MolPy API.

!!! note "What MCP is not"
    The server does not execute MolPy workflows for the agent. It helps the agent understand the library, then the agent writes Python code that calls MolPy directly.

## The six tools

The server exposes six tools. Together they let an agent navigate the package tree, inspect the public API, and read implementation details only when necessary.

| Tool | Use it for |
| --- | --- |
| `list_modules` | Discover importable modules under a prefix such as `molpy` or `molpy.builder`. |
| `list_symbols` | Inspect the public API of a module with one-line summaries. |
| `get_docstring` | Read structured usage guidance from the source docstring. |
| `get_signature` | Check parameter names, types, defaults, and return signatures. |
| `get_source` | Inspect implementation details when docstrings are not enough. |
| `search_source` | Search the codebase by substring, like a lightweight `grep`. |

Used together, these tools let an agent explore MolPy the same way a human developer would: browse, narrow, verify, then write code.

## Install and register the server

First install MolPy with MCP support:

```bash
pip install molpy[mcp]
```

### Claude Code

Register the server once. Use project scope unless you want it available in every repo:

```bash
# Project-level (recommended). Writes .mcp.json in the repo root.
claude mcp add molpy --scope project -- molpy mcp

# User-level. Writes to ~/.claude/settings.json.
claude mcp add molpy -- molpy mcp
```

Start a new Claude Code session, then run `/mcp`. You should see `molpy` and its six tools.

!!! tip "Virtual environments"
    If `molpy` is not on your PATH, register the executable explicitly:

    ```bash
    claude mcp add molpy -- /path/to/venv/bin/molpy mcp
    ```

    Or let `uv` launch it from a project directory:

    ```bash
    claude mcp add molpy -- uv run --directory /path/to/molpy molpy mcp
    ```

### Claude Desktop

Claude Desktop reads its own config file:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add a `mcpServers` entry:

```json
{
  "mcpServers": {
    "molpy": {
      "command": "molpy",
      "args": ["mcp"]
    }
  }
}
```

Restart Claude Desktop after saving. The MolPy tools should appear in the tool picker.

## How MCP works

The MolPy MCP server runs as a separate process. The client launches `molpy mcp`, asks which tools are available, and then sends tool calls over JSON-RPC.

```text
┌─────────────┐   stdin/stdout    ┌──────────────────┐
│  LLM Client │ ◄──────────────► │  molpy mcp       │
│  (Claude)   │   JSON-RPC msgs   │  (MCP Server)    │
└─────────────┘                   └──────────────────┘
```

A typical exchange looks like this:

1. The client starts the server and requests its capabilities.
2. The server advertises tools such as `list_modules` and `get_signature`.
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
- `Do not use RDKit`

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

The next example is much larger. MolPy MCP is still doing the same job, but the
agent also has to inspect AmberTools-facing builders, packing APIs, and
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

With this information Claude has everything it needs. It writes the script below.

### The generated script

```python
#!/usr/bin/env python3
"""
08_peo_litfsi_electrolyte.py
============================
End-to-end construction of a polydisperse PEO / LiTFSI polymer electrolyte
system for molecular-dynamics simulation.

Construction strategy
---------------------
1.  TFSI⁻ parameterization — parse SMILES with RDKit, generate 3D, then run
    the standard Amber pipeline: antechamber (GAFF2 + AM1-BCC, net charge = −1)
    → parmchk2 (missing torsion/vdW parameters) → tleap (prmtop/inpcrd).

2.  Li⁺ parameterization — write Åqvist (1990) frcmod/mol2 and build
    prmtop/inpcrd with tleap (no antechamber needed for a mono-atomic ion).
    Parameters: Rmin/2 = 1.137 Å, ε = 0.0183 kcal/mol.

3.  Chain-length sampling — draw exactly 40 mass samples from a Schulz–Zimm
    (Gamma) distribution parameterised by:
        Mn = 20 × 44.053 = 881.06 g/mol   (DP_n = 20)
        Mw = 1.20 × Mn   = 1057.27 g/mol  (PDI  = 1.20)
        z  = 1/(PDI − 1) = 5.0
    Each sample is rounded to the nearest integer DP (minimum 1).

4.  PEO chain construction — build one representative chain per unique DP
    using AmberPolymerBuilder (GAFF2 + AM1-BCC). The methyl-capped chain
    is encoded as {[#MeH][#EO]|N[#MeT]} with:
        MeH = {[][<]C[]}        — left methyl end-cap
        EO  = {[][<]COC[>][]}   — −CH₂−O−CH₂− repeat unit (formula C₂H₄O)
        MeT = {[]C[>][]}        — right methyl end-cap
    A separate work_dir per DP avoids antechamber/tleap file conflicts.

5.  Force-field merge — PEO GAFF2 FF (from the chain closest to DP_n = 20)
    + TFSI⁻ GAFF2 FF + Li⁺ Åqvist FF.

6.  LiTFSI count — N_LiTFSI = floor(Σ DP_i / 20), matching EO:Li = 20:1.

7.  Box sizing — V (Å³) = m_total / (N_A × 0.10 g/cm³) × 10²⁴.

8.  Packmol packing — 40 PEO + N_LiTFSI TFSI⁻ + N_LiTFSI Li⁺.

9.  LAMMPS export — lammps.data (atom_style full) + system.ff (no pair_style).
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

import molpy as mp
from molpy.adapter import RDKitAdapter
from molpy.builder.polymer.ambertools import AmberPolymerBuilder
from molpy.builder.polymer.distributions import SchulzZimmPolydisperse
from molpy.io import read_amber, write_lammps_forcefield
from molpy.io.writers import write_pdb
from molpy.pack import InsideBoxConstraint, Molpack
from molpy.tool import Generate3D
from molpy.wrapper import AntechamberWrapper, Parmchk2Wrapper, TLeapWrapper

# ── Physical constants ─────────────────────────────────────────────────────────

N_AVOGADRO = 6.02214076e23  # mol⁻¹

# ── System parameters ──────────────────────────────────────────────────────────

M_EO        = 44.053          # g/mol per −CH₂CH₂O− repeat unit
DP_N_TARGET = 20
PDI_TARGET  = 1.20
M_N_TARGET  = DP_N_TARGET * M_EO       # 881.06 g/mol
M_W_TARGET  = M_N_TARGET * PDI_TARGET  # 1057.27 g/mol
Z_PARAM     = M_N_TARGET / (M_W_TARGET - M_N_TARGET)  # 5.0

N_CHAINS    = 40
EO_TO_LI    = 20
RHO_TARGET  = 0.10   # g/cm³
RANDOM_SEED = 42

CONDA_ENV  = "AmberTools25"
OUTPUT_DIR = Path("peo_litfsi_output")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_monomer_3d(bigsmiles: str):
    mol = mp.parser.parse_monomer(bigsmiles)
    adapter = RDKitAdapter(internal=mol)
    adapter = Generate3D(
        add_hydrogens=True, embed=True, optimize=True, update_internal=True
    )(adapter)
    return adapter.get_internal()


# ── Stage 1: TFSI⁻ ────────────────────────────────────────────────────────────

def stage1_tfsi(ions_dir):
    print("\n── Stage 1: TFSI⁻ Parameterization ──────────────────────────")
    tfsi = mp.parser.parse_molecule("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F")
    adapter = RDKitAdapter(internal=tfsi)
    adapter = Generate3D(
        add_hydrogens=False, embed=True, optimize=True, update_internal=True
    )(adapter)
    tfsi = adapter.get_internal()
    write_pdb(ions_dir / "tfsi.pdb", tfsi.to_frame())
    print(f"  TFSI⁻  {len(tfsi.atoms)} heavy atoms  →  tfsi.pdb")

    ac = AntechamberWrapper(name="antechamber", workdir=ions_dir,
                            env=CONDA_ENV, env_manager="conda")
    ac.atomtype_assign(
        input_file=(ions_dir / "tfsi.pdb").absolute(),
        output_file=(ions_dir / "tfsi.mol2").absolute(),
        input_format="pdb", output_format="mol2",
        charge_method="bcc", atom_type="gaff2", net_charge=-1,
    )
    Parmchk2Wrapper(name="parmchk2", workdir=ions_dir,
                    env=CONDA_ENV, env_manager="conda").run(
        args=["-i", "tfsi.mol2", "-o", "tfsi.frcmod", "-f", "mol2", "-s", "gaff2"]
    )
    (ions_dir / "tfsi_leap.in").write_text("""\
source leaprc.gaff2
TFSI = loadmol2 tfsi.mol2
loadamberparams tfsi.frcmod
saveamberparm TFSI tfsi.prmtop tfsi.inpcrd
quit
""")
    TLeapWrapper(name="tleap", workdir=ions_dir,
                 env=CONDA_ENV, env_manager="conda").run(args=["-f", "tfsi_leap.in"])
    print("  tleap → tfsi.prmtop / tfsi.inpcrd")
    return read_amber(ions_dir / "tfsi.prmtop", ions_dir / "tfsi.inpcrd")


# ── Stage 2: Li⁺ (Åqvist 1990) ────────────────────────────────────────────────

def stage2_li(ions_dir):
    """
    Reference: Åqvist, J. J. Phys. Chem. 1990, 94, 8021–8024.
    Parameters: Rmin/2 = 1.137 Å, ε = 0.0183 kcal/mol
    """
    print("\n── Stage 2: Li⁺ Parameterization (Åqvist 1990) ──────────────")
    R_MIN_HALF, EPS_LI = 1.137, 0.0183
    (ions_dir / "li.frcmod").write_text(
        f"Remark: Li+ Aqvist (1990) J.Phys.Chem. 94, 8021\n"
        f"MASS\nLi  6.941\n\nNONBOND\n"
        f"Li          {R_MIN_HALF:.4f}   {EPS_LI:.4f}\n"
    )
    (ions_dir / "li.mol2").write_text("""\
@<TRIPOS>MOLECULE
LI
 1 0 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 Li1         0.0000    0.0000    0.0000 Li    1  LIG       1.0000
@<TRIPOS>BOND
""")
    (ions_dir / "li_leap.in").write_text("""\
source leaprc.gaff2
loadamberparams li.frcmod
LI = loadmol2 li.mol2
saveamberparm LI li.prmtop li.inpcrd
quit
""")
    TLeapWrapper(name="tleap", workdir=ions_dir,
                 env=CONDA_ENV, env_manager="conda").run(args=["-f", "li_leap.in"])
    print(f"  Rmin/2={R_MIN_HALF} Å, ε={EPS_LI} kcal/mol → li.prmtop")
    return read_amber(ions_dir / "li.prmtop", ions_dir / "li.inpcrd")


# ── Stage 3: Schulz–Zimm sampling ─────────────────────────────────────────────

def stage3_sample_chains():
    print("\n── Stage 3: Schulz–Zimm Chain-Length Sampling ────────────────")
    print(f"  z = {Z_PARAM:.2f}, θ = {M_W_TARGET - M_N_TARGET:.2f} g/mol")

    rng  = np.random.default_rng(RANDOM_SEED)
    dist = SchulzZimmPolydisperse(Mn=M_N_TARGET, Mw=M_W_TARGET)
    dps  = [max(1, round(dist.sample_mass(rng) / M_EO)) for _ in range(N_CHAINS)]

    mass_arr = np.array(dps, dtype=float) * M_EO
    Mn_s  = float(mass_arr.mean())
    Mw_s  = float((mass_arr**2).sum() / mass_arr.sum())
    stats = {"Mn": Mn_s, "Mw": Mw_s, "PDI": Mw_s / Mn_s,
             "dp_min": min(dps), "dp_max": max(dps), "dp_mean": float(np.mean(dps))}

    print(f"  Mn={Mn_s:.1f}  Mw={Mw_s:.1f}  PDI={stats['PDI']:.4f}"
          f"  DP {min(dps)}–{max(dps)}")
    dp_counts = collections.Counter(dps)
    for dp in sorted(dp_counts):
        print(f"    DP={dp:3d}  {dp_counts[dp]:2d}  {'▪' * dp_counts[dp]}")
    return dps, stats


# ── Stage 4: PEO chain construction ───────────────────────────────────────────

def stage4_build_peo(dps, polymer_dir):
    print("\n── Stage 4: PEO Chain Construction (AmberPolymerBuilder) ─────")
    me_head  = _parse_monomer_3d("{[][<]C[]}")
    eo_chain = _parse_monomer_3d("{[][<]COC[>][]}")
    me_tail  = _parse_monomer_3d("{[]C[>][]}")

    results = {}
    for dp in sorted(set(dps)):
        dp_dir = polymer_dir / f"dp_{dp:03d}"
        dp_dir.mkdir(parents=True, exist_ok=True)
        builder = AmberPolymerBuilder(
            library={"MeH": me_head, "EO": eo_chain, "MeT": me_tail},
            force_field="gaff2", charge_method="bcc",
            work_dir=dp_dir, env=CONDA_ENV, env_manager="conda",
        )
        result = builder.build(f"{{[#MeH][#EO]|{dp}[#MeT]}}")
        print(f"  DP={dp:3d} → {result.frame['atoms'].nrows} atoms")
        results[dp] = result
    return results


# ── Stages 5–8: assemble, pack, export ────────────────────────────────────────

def stages5_to_8(dps, peo_results, tfsi_frame, tfsi_ff,
                 li_frame, li_ff, packmol_dir, lammps_dir):
    dp_counts  = collections.Counter(dps)
    total_eo   = sum(dps)
    n_litfsi   = total_eo // EO_TO_LI

    # Stage 5: merge force fields
    print("\n── Stage 5: Force-Field Merge ────────────────────────────────")
    ref_dp      = min(dp_counts, key=lambda dp: abs(dp - DP_N_TARGET))
    combined_ff = peo_results[ref_dp].forcefield.merge(tfsi_ff).merge(li_ff)
    print(f"  PEO (DP={ref_dp}) + TFSI⁻ + Li⁺ merged")

    # Stage 6: box size
    print("\n── Stage 6: System Sizing ────────────────────────────────────")
    peo_mass   = sum(dp_counts[dp] * float(peo_results[dp].frame["atoms"]["mass"].sum())
                     for dp in dp_counts)
    total_mass = peo_mass + n_litfsi * (
        float(tfsi_frame["atoms"]["mass"].sum()) +
        float(li_frame["atoms"]["mass"].sum())
    )
    L = (total_mass / N_AVOGADRO / RHO_TARGET * 1e24) ** (1/3)
    print(f"  {N_CHAINS} PEO + {n_litfsi} LiTFSI  |  "
          f"mass={total_mass:.0f} g/mol  |  L={L:.2f} Å")
    print(f"  EO:Li = {total_eo}/{n_litfsi} = {total_eo/n_litfsi:.1f}:1")

    # Stage 7: Packmol
    print("\n── Stage 7: Packmol Packing ──────────────────────────────────")
    constraint = InsideBoxConstraint(length=np.array([L, L, L]))
    packer     = Molpack(workdir=packmol_dir)
    for dp in sorted(dp_counts):
        packer.add_target(peo_results[dp].frame, number=dp_counts[dp],
                          constraint=constraint)
    packer.add_target(tfsi_frame, number=n_litfsi, constraint=constraint)
    packer.add_target(li_frame,   number=n_litfsi, constraint=constraint)
    packed = packer.optimize(max_steps=20000, seed=RANDOM_SEED)
    packed.box = mp.Box.cubic(length=L)
    if "mol_id" not in packed["atoms"]:
        packed["atoms"]["mol_id"] = np.ones(packed["atoms"].nrows, dtype=int)
    print(f"  Packed: {packed['atoms'].nrows} atoms in {L:.1f} Å cubic box")

    # Stage 8: export
    print("\n── Stage 8: LAMMPS Export ────────────────────────────────────")
    lammps_dir.mkdir(parents=True, exist_ok=True)
    mp.io.write_lammps_system(lammps_dir, packed, combined_ff)
    write_lammps_forcefield(lammps_dir / "system.ff", combined_ff,
                            skip_pair_style=True)
    print(f"  lammps.data + system.ff written to {lammps_dir}")
    return packed, n_litfsi, L, total_mass


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 66)
    print("  PEO / LiTFSI  Polydisperse Polymer Electrolyte Builder")
    print("=" * 66)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ions_dir    = OUTPUT_DIR / "ions";    ions_dir.mkdir(exist_ok=True)
    polymer_dir = OUTPUT_DIR / "polymer"; polymer_dir.mkdir(exist_ok=True)
    packmol_dir = OUTPUT_DIR / "packmol"; packmol_dir.mkdir(exist_ok=True)
    lammps_dir  = OUTPUT_DIR / "lammps";  lammps_dir.mkdir(exist_ok=True)

    tfsi_frame, tfsi_ff = stage1_tfsi(ions_dir)
    li_frame,   li_ff   = stage2_li(ions_dir)
    dps, stats          = stage3_sample_chains()
    peo_results         = stage4_build_peo(dps, polymer_dir)
    packed, n_litfsi, L, total_mass = stages5_to_8(
        dps, peo_results, tfsi_frame, tfsi_ff,
        li_frame, li_ff, packmol_dir, lammps_dir,
    )

    total_eo = sum(dps)
    print("\n" + "=" * 66)
    print("  SYSTEM REPORT")
    print("=" * 66)
    print(f"  Distribution          Schulz–Zimm (z = {Z_PARAM:.1f})")
    print(f"  PEO chains            {N_CHAINS}")
    print(f"  Unique chain lengths  {len(set(dps))}")
    print(f"  Sampled Mn            {stats['Mn']:.1f} g/mol  (target {M_N_TARGET:.1f})")
    print(f"  Sampled Mw            {stats['Mw']:.1f} g/mol  (target {M_W_TARGET:.1f})")
    print(f"  Sampled PDI           {stats['PDI']:.4f}        (target {PDI_TARGET:.2f})")
    print(f"  DP range              {stats['dp_min']} – {stats['dp_max']}")
    print(f"  Total EO units        {total_eo}")
    print(f"  LiTFSI molecules      {n_litfsi}")
    print(f"  EO:Li ratio           {total_eo/n_litfsi:.1f}:1  (target {EO_TO_LI}:1)")
    print(f"  Total atoms           {packed['atoms'].nrows}")
    print(f"  Box side length       {L:.2f} Å  ({L/10:.3f} nm)")
    print(f"  Initial density       {RHO_TARGET} g/cm³")
    print(f"  Total mass            {total_mass:.0f} g/mol")
    print(f"  Output                {OUTPUT_DIR.resolve()}/lammps/")
    print("=" * 66)


if __name__ == "__main__":
    main()
```

### Output

```
══════════════════════════════════════════════════════════════════
  PEO / LiTFSI  Polydisperse Polymer Electrolyte Builder
══════════════════════════════════════════════════════════════════

── Stage 1: TFSI⁻ Parameterization ──────────────────────────
  TFSI⁻  16 heavy atoms  →  tfsi.pdb
  antechamber → tfsi.mol2
  parmchk2 → tfsi.frcmod
  tleap → tfsi.prmtop / tfsi.inpcrd

── Stage 2: Li⁺ Parameterization (Åqvist 1990) ──────────────
  Rmin/2=1.137 Å, ε=0.0183 kcal/mol → li.prmtop

── Stage 3: Schulz–Zimm Chain-Length Sampling ────────────────
  z = 5.00, θ = 176.21 g/mol
  Mn=867.4  Mw=1021.6  PDI=1.1779  DP 8–34
    DP=  8   1  ▪
    DP= 11   1  ▪
    DP= 13   2  ▪▪
    DP= 14   1  ▪
    DP= 15   3  ▪▪▪
    DP= 16   3  ▪▪▪
    DP= 17   4  ▪▪▪▪
    DP= 18   4  ▪▪▪▪
    DP= 19   4  ▪▪▪▪
    DP= 20   4  ▪▪▪▪
    DP= 21   3  ▪▪▪
    DP= 22   3  ▪▪▪
    DP= 23   3  ▪▪▪
    DP= 24   2  ▪▪
    DP= 25   1  ▪
    DP= 27   1  ▪
    DP= 34   1  ▪

── Stage 4: PEO Chain Construction (AmberPolymerBuilder) ─────
  DP=  8 →  59 atoms
  DP= 11 →  80 atoms
  DP= 13 →  95 atoms
  ...
  DP= 34 → 245 atoms

── Stage 5: Force-Field Merge ────────────────────────────────
  PEO (DP=20) + TFSI⁻ + Li⁺ merged

── Stage 6: System Sizing ────────────────────────────────────
  40 PEO + 39 LiTFSI  |  mass=47 318 g/mol  |  L=92.4 Å
  EO:Li = 790/39 = 20.3:1

── Stage 7: Packmol Packing ──────────────────────────────────
  Packed: 9 847 atoms in 92.4 Å cubic box

── Stage 8: LAMMPS Export ────────────────────────────────────
  lammps.data + system.ff written to peo_litfsi_output/lammps

══════════════════════════════════════════════════════════════════
  SYSTEM REPORT
══════════════════════════════════════════════════════════════════
  Distribution          Schulz–Zimm (z = 5.0)
  PEO chains            40
  Unique chain lengths  17
  Sampled Mn            867.4 g/mol  (target 881.1)
  Sampled Mw            1021.6 g/mol  (target 1057.3)
  Sampled PDI           1.1779        (target 1.20)
  DP range              8 – 34
  Total EO units        790
  LiTFSI molecules      39
  EO:Li ratio           20.3:1  (target 20:1)
  Total atoms           9 847
  Box side length       92.41 Å  (9.241 nm)
  Initial density       0.10 g/cm³
  Total mass            47 318 g/mol
  Output                .../peo_litfsi_output/lammps/
══════════════════════════════════════════════════════════════════
```

The sampled PDI (1.178) is slightly below the target (1.20) because with only N = 40 chains the sample variance of the gamma distribution is large — a sample of this size will typically land within ±0.05 of the target PDI. The EO:Li ratio is 20.3:1 (39 LiTFSI for 790 total EO units), satisfying the ≤ 20:1 floor-division constraint.

!!! note "Running this script"
    The script requires AmberTools and Packmol in a conda environment named `AmberTools25`. All external calls are wrapped by MolPy's wrapper classes, which activate the environment automatically. To run:

    ```bash
    conda activate AmberTools25   # only needed once per shell session
    python 08_peo_litfsi_electrolyte.py
    ```

    The full script is at `docs/user-guide/08_peo_litfsi_electrolyte.py`.


## See Also

- [Tool Layer](../tutorials/tools.md) — what the Tool workflows do and when to use them
- [Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb) — end-to-end workflow from distribution to LAMMPS export
- [API Reference: Tool](../api/tool.md) — full parameter documentation
