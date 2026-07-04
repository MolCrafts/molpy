# MCP: Letting LLM Agents Read MolPy's Source

## What MCP solves

An LLM agent working with MolPy faces the same problem as any user relying on memory: API knowledge goes stale. A guessed function name, parameter list, or return type may look plausible and still be wrong — and the cost is a long debugging round trip on code that never had a chance to run.

The MolCrafts MCP suite fixes that by exposing MolPy's installed source as a **structured, graph-indexed view of the code**. Instead of guessing, the agent asks the indexed code graph: which symbol implements this capability, what is its signature, where is it called, what tests exercise it, what example uses it. Every answer is grounded in the MolPy source tree that is currently importable in the active Python environment — not in the model's training data.

In practice, the workflow is simple:

1. Resolve a capability (or browse the package outline).
2. Drill into the relevant symbol — signature, docstring, source, callers, tests.
3. Write normal Python code against the real MolPy API.

!!! note "What MCP is not"
    The server does not execute MolPy workflows for the agent. It is a *discovery layer*: it helps the agent understand the library, and the agent writes Python code that calls MolPy directly.

## Why a code graph

MolPy is not a flat list of functions. A real workflow — say, building a polymer, typifying it, and writing LAMMPS files — traverses *relationships*: `AmberPolymerBuilder` *contains* a `build` method that *calls* `tleap`, which *produces* a `Frame` whose `box` attribute is read by `write_lammps_data`. A grep on `"build"` cannot tell you any of that. A flat symbol table cannot either.

The MCP server indexes MolPy into a **typed code graph** so the agent can ask graph-shaped questions:

- "Which class implements a radial distribution function?" → search nodes by capability.
- "What calls `Atomistic.get_topo`?" → walk `calls` edges in reverse.
- "What tests exercise `ForceField.merge`?" → follow `tests` edges to pytest nodes.
- "What examples show how to use `Box.orth`?" → follow `exemplifies` edges to extracted docstring examples.
- "What breaks if I change `read_xml_forcefield`?" → multi-hop impact walk.

This is the central capability the MCP server provides. The six tools described later are graph queries with different shapes; the rest of this section explains what is in the graph.

### The pipeline that builds it

```
MolPy source ─► SourceResolver ─► Snapshot ─► Extractor ─► Resolver ─► GraphStore
  pkg:molpy                       (immutable,  (phase 1)   (phase 2)   (SQLite
  or local                         content-                            graph.db)
  checkout                         hashed)
```

1. **Source resolution.** The MolPy source — either the installed `pkg:molpy` or a local checkout — resolves to an immutable `Snapshot`. The snapshot is keyed on a **content hash** of the files, never a branch name; a cached graph is therefore always tied to exact source.
2. **Extraction (phase 1).** Each Python file is parsed with the stdlib `ast` module. The analyzer emits *nodes* — modules, classes, functions, methods, properties, fields, constants — with signatures, docstrings, decorators, and file/line spans, plus *unresolved references* for every call, base class, and import it sees.
3. **Resolution (phase 2).** A resolver links those references to nodes already in the graph: relative imports are resolved, pytest tests are linked to the symbols they exercise, and *docstring code blocks are lifted into first-class `example` nodes*. References that are genuinely dynamic (e.g. `getattr(obj, name)()`) stay marked `unresolved` so they do not silently corrupt the graph.
4. **Storage.** The graph is written to one SQLite `graph.db` per snapshot under `~/.cache/molmcp/discovery/`, with a derived FTS5 index for symbol search.

### Nodes and edges

Every analyzer emits the same schema, so a query that works on one source works on any.

**Node kinds** carried in the graph:

| Kind | What it represents |
| --- | --- |
| `package`, `module`, `file` | The package tree as importable units. |
| `class`, `function`, `method`, `property`, `field`, `constant` | The MolPy public/private API surface. Each carries `qualname`, `signature`, `docstring`, file/line span. |
| `example` | A code block lifted from a docstring — *real* usage that ships with the source. |
| `test` | A pytest test function, with the test file/line it lives at. |
| `capability` | A domain capability (e.g. "compute RDF") attached to one or more symbols via an overlay. |

**Edges** carry the relationships you want to query:

| Edge | Used to answer |
| --- | --- |
| `contains` | "What symbols live in `molpy.builder.polymer`?" |
| `calls` | "What does `AmberPolymerBuilder.build` call?" — and reverse: "Who calls `tleap`?" |
| `extends` | "What subclasses `Struct`?" |
| `imports` | "Which modules import from `molpy.core.atomistic`?" |
| `exemplifies` | "Show me a usage example of `Box.orth`." |
| `tests` | "What pytest tests cover `ForceField.merge`?" |
| `references` | "Where is `EPS_LI` mentioned?" |
| `provides_capability` | "Which symbol implements *radial distribution function*?" |

Each edge carries a `provenance` (`ast` / `heuristic` / `resolved`) so the agent — and you — can tell a confident structural fact from an inference.

### Snapshots, freshness, incremental re-indexing

A few practical properties fall out of the design:

- **Local sources are always fresh.** They are re-resolved on every query; only the per-file extractor caches.
- **Incremental re-indexing.** A content-addressed `ExtractCache` lets unchanged files skip the analyzer, so editing one MolPy file does not re-parse the package.
- **Every tool response carries `snapshot`**, including a `freshness` flag, so the agent always knows which revision of MolPy it is looking at.
- **The graph is plain SQLite.** Open `~/.cache/molmcp/discovery/snapshots/<slug>/graph.db` in any SQLite browser to inspect the `nodes`, `edges`, and `files` tables directly.

## The six graph-query tools

The MCP server exposes six composable, read-only tools (all `readOnlyHint=True`, so MCP clients can auto-approve them safely). They differ in the *shape* of the graph query they run.

| Tool | Graph query | Use it for |
| --- | --- | --- |
| `molmcp_outline` | Walk `contains` edges from the package root. | Orient in MolPy — "where do I look?" |
| `molmcp_find_capability` | Rank symbols by capability + FTS + structural signals. | Primary tool — describe a task, get ranked symbol matches with signature, summary, examples, tests, and callers. |
| `molmcp_search_symbols` | FTS5 over names, qualnames, and summaries, optional `kind` filter. | Quick lookup when you already know the name. |
| `molmcp_describe_symbol` | Read one node, optionally with full source. | Final-step detail: signature, cleaned docstring, file/line span, source. |
| `molmcp_relations` | Walk one edge type from a symbol (`callers`, `callees`, `implementers`, `subclasses`, `implementations`, `references`, `examples`, `tests`, `impact` 1–4 hops). | The questions a flat index can't answer. |
| `molmcp_refresh` | Force a new snapshot of a source. | Rare; local sources auto-refresh. |

!!! tip "The qualname rule"
    Resolve every qualname (`molpy.compute.rdf.RDF`, …) from a prior tool result — `molmcp_outline`, `molmcp_search_symbols`, or `molmcp_find_capability` — never guess. A wrong qualname returns a structured `{"error": …}` rather than a hallucinated payload, by design.

### Typical traversal patterns

The tools compose. A few patterns the agent uses repeatedly on MolPy:

**"What's the right class for this task?"**

```text
molmcp_find_capability("compute a radial distribution function")
  → matches: [molpy.compute.rdf.RDF, …]
molmcp_describe_symbol("molpy.compute.rdf.RDF")
  → signature + docstring
```

**"How is this actually used?"**

```text
molmcp_relations("molpy.core.Box.orth", relation="examples")
  → docstring examples + tests that exercise it
```

**"What does my change break?"**

```text
molmcp_relations("molpy.io.read_xml_forcefield", relation="callers", depth=2)
  → every site, two hops out, that depends on the function
```

**"Show me the structure of a subpackage."**

```text
molmcp_outline(path="molpy/builder/polymer")
  → modules + classes + functions, scoped to one subtree
```

These are not search hits — they are graph walks. That is the point of indexing MolPy as a graph rather than a list.

## Install and register the server

Install `molmcp` from PyPI. Pin to the 0.2 line:

```bash
pip install "molcrafts-molmcp>=0.2,<0.3"
```

Requires Python ≥ 3.12. The PyPI distribution is `molcrafts-molmcp`; the import name and CLI entry are both `molmcp`. The discovery engine adds no required runtime dependency — it uses the standard library.

Start the server in stdio mode (what MCP clients expect):

```bash
python -m molmcp --source pkg:molpy
```

### Claude Code

Project-level registration writes `.mcp.json` at the repository root:

```bash
claude mcp add molpy --scope project -- python -m molmcp --source pkg:molpy
```

Omit `--scope project` to register at user scope.

!!! tip "Virtual environments"
    If `python` resolves to the wrong interpreter (or `molmcp` lives in a project-local venv), register the binary explicitly:

    ```bash
    claude mcp add molpy -- /path/to/venv/bin/python -m molmcp --source pkg:molpy
    ```

    Or let `uv` pick the right environment from a project directory:

    ```bash
    claude mcp add molpy -- uv run --directory /path/to/molpy python -m molmcp --source pkg:molpy
    ```

Start a new Claude Code session, then run `/mcp`. You should see the server with tool names prefixed `mcp__molpy__molmcp_…`.

### Claude Desktop

Edit Claude Desktop's `mcpServers` block:

```json
{
  "mcpServers": {
    "molpy": {
      "command": "python",
      "args": ["-m", "molmcp", "--source", "pkg:molpy"]
    }
  }
}
```

Config file location:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Restart Claude Desktop after saving. MolPy's discovery tools should appear in the tool picker.

### Indexing a local checkout

Point `--source` at the working tree of a MolPy checkout to expose your in-progress edits — the graph is rebuilt incrementally on every query, so changes you make in your editor are visible to the agent within one tool call:

```bash
python -m molmcp --source /path/to/molpy
```

### Verifying the server works

`molmcp` ships a built-in self-check that indexes MolPy and prints counts, FTS status, and a sample query — exiting non-zero on failure, so it doubles as a CI/setup check:

```bash
molmcp discovery verify pkg:molpy
```

Inspect or rebuild the graph from the CLI without going through an MCP client:

```bash
molmcp discovery index pkg:molpy        # build the graph
molmcp discovery outline pkg:molpy      # high-level map
molmcp discovery query pkg:molpy "radial distribution function"
```

## Writing effective prompts

MCP gives the agent access to MolPy's API, but it does not replace a clear task description. The best prompts specify the system and constraints, then let the agent discover the implementation.

### Describe the system, not the API

Tell the agent what you want to build. Do not tell it which function names to call.

| Too low-level | Better |
| --- | --- |
| `Use polymer() to build a PEG chain` | `Build a PEG chain with 15 repeat units` |
| `Call Packmol to pack molecules` | `Pack 15 chains into a 20 nm cubic box` |
| `Use the Box class` | `Create a periodic simulation box for the system` |

If your prompt names specific MolPy functions, it is usually too low-level. The point of `molmcp_find_capability` is that the agent maps the *task* onto the right symbol — feeding it the symbol up front bypasses the strongest part of the pipeline.

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

Claude first asks the graph for the capabilities the task needs: force-field
loading, typification, topology generation, and LAMMPS export.

**Step 1 — resolve capabilities by task**

```
molmcp_find_capability("read an XML force field and write LAMMPS data + ff")
```

```
matches:
  • molpy.io.read_xml_forcefield        (function)
  • molpy.io.write_lammps_data          (function)
  • molpy.io.write_lammps_forcefield    (function)
  • molpy.typifier.OplsTypifier (class)
```

**Step 2 — confirm the built-in TIP3P file path**

```
molmcp_describe_symbol("molpy.io.read_xml_forcefield")
```

```
signature: (filepath: str | Path, forcefield: AtomisticForcefield | None = None)
           -> AtomisticForcefield
docstring: Read an XML force field file. Relative filenames resolve against the
           package data directory, so passing "tip3p.xml" loads the built-in
           TIP3P force field.
```

**Step 3 — verify the non-obvious topology step**

```
molmcp_describe_symbol("molpy.core.atomistic.Atomistic.get_topo")
```

The docstring makes one subtle point clear: when `gen_angle=True`,
`get_topo()` returns a *new* `Atomistic` instead of mutating the existing one.
Claude therefore writes `system = system.get_topo(...)` rather than calling it
in place.

**Step 4 — check the box and typing APIs**

```
molmcp_describe_symbol("molpy.core.Box.orth")
```

```
(lengths: ArrayLike, pbc: ArrayLike = ..., origin: ArrayLike = ...,
 central: bool = False) -> Box
```

```
molmcp_describe_symbol("molpy.typifier.OplsTypifier.__init__")
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
from molpy.typifier import OplsTypifier

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
typifier = OplsTypifier(ff, skip_dihedral_typing=True, strict_typing=True)
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
but the agent walks deeper into the graph — finding builders, packing classes,
and the tests that pin down Li⁺ parameters — before it can write the full
workflow.

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

Claude orients in the package, then drills into the symbols it needs.

**Step 1 — orient in the package**

```
molmcp_outline()
```

Returns MolPy's top-level packages and modules (excerpt):

```
molpy.builder   Crystal and polymer builders (AmberTools integration, stochastic generation)
molpy.io        I/O for AMBER, LAMMPS, PDB, GRO, MOL2, XYZ ...
molpy.pack      Packing (constraints, targets, Packmol integration)
molpy.parser    Parsers for SMILES, BigSMILES, CGSmiles, GBigSMILES
molpy.wrapper   External tool wrappers (antechamber, parmchk2, prepgen, tleap)
```

**Step 2 — find the distribution and polymer-builder classes**

```
molmcp_outline(path="molpy/builder/polymer")
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
molmcp_describe_symbol(
    "molpy.builder.polymer.distributions.SchulzZimmPolydisperse",
    include_source=False,
)
```

```
signature: (Mn: float, Mw: float, random_seed: int | None = None)
docstring:
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

**Step 4 — locate `AmberPolymerBuilder` by capability, then drill in**

```
molmcp_find_capability("build a polymer with AmberTools / GAFF parameters")
```

picks out `molpy.builder.polymer.ambertools.AmberPolymerBuilder`. Claude then
reads its constructor and `build` method:

```
molmcp_describe_symbol("molpy.builder.polymer.ambertools.AmberPolymerBuilder")
```

```
signature: (library: dict[str, Atomistic],
            force_field: str = "gaff2",
            charge_method: str = "bcc",
            work_dir: Path = Path("amber_work"),
            env: str = "AmberTools25",
            env_manager: str = "conda")
```

```
molmcp_describe_symbol("molpy.builder.polymer.ambertools.AmberPolymerBuilder.build")
```

```
docstring:
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

**Step 5 — pin Li⁺ parameters via the `tests` edge**

Claude does not guess at Li⁺ LJ parameters. Instead of `grep`, it walks the
graph: any reference to "Aqvist" is a node, and the `test` nodes around it tell
Claude exactly which parameter values MolPy considers canonical:

```
molmcp_search_symbols("Aqvist")
```

```
test_e2e_peo_litfsi.test_li_frcmod   (test, tests/test_e2e_peo_litfsi.py:147)
    "Write Åqvist (1990) Li+ frcmod and build prmtop via tleap."
```

```
molmcp_describe_symbol(
    "test_e2e_peo_litfsi.test_li_frcmod", include_source=True
)
```

reveals the canonical reference: **Åqvist (1990), J. Phys. Chem. 94, 8021** —
Rmin/2 = 1.137 Å, ε = 0.0183 kcal/mol. Claude writes these directly into a
frcmod file.

**Step 6 — find the packing and export interfaces**

```
molmcp_outline(path="molpy/pack")
```

```
Packmol                  High-level Packmol packing interface
InsideBoxConstraint      Place molecules inside a rectangular box
OutsideBoxConstraint     Keep molecules outside a box
InsideSphereConstraint   Sphere constraint
MinDistanceConstraint    Minimum pairwise distance
Target                   One packing target (frame + count + constraint)
```

```
molmcp_describe_symbol("molpy.pack.Packmol.pack")
```

```
(max_steps: int = 20000, seed: int = 12345) → Frame
```

```
molmcp_describe_symbol("molpy.io.write_lammps_forcefield")
```

```
(path: Path | str,
 forcefield: ForceField,
 precision: int = 6,
 skip_pair_style: bool = False) → None
```

Claude notes: `skip_pair_style=True` is needed so the LAMMPS input script can control `kspace_style` independently.

**Step 7 — confirm `ForceField.merge` with the `examples` edge**

```
molmcp_relations(
    "molpy.core.forcefield.ForceField.merge", relation="examples"
)
```

returns the docstring example plus tests that exercise the method on real force
fields, confirming the contract:

```
docstring:
  Merge two ForceField objects.  Returns a new ForceField containing all
  styles and parameters from both.  Raises if incompatible styles are found.
```

**Step 8 — sanity-check impact before committing**

Before writing the script, Claude does one final graph query — a multi-hop
`impact` walk from the export functions it plans to call, to make sure it
understands the dependencies it is about to wire together:

```
molmcp_relations(
    "molpy.io.write_lammps_data", relation="impact", depth=2
)
```

confirms `write_lammps_data` ultimately reads `frame.box` and per-atom
`charge`/`mol_id` columns — which is why the generated script explicitly fills
those columns before exporting.

With this information Claude has everything it needs to assemble the script.

!!! note "The full generated script"
    The full end-to-end script is at `docs/user-guide/08_peo_litfsi_electrolyte.py`. It runs antechamber/parmchk2/tleap for TFSI⁻, builds a Li⁺ frcmod from Åqvist parameters, samples 40 chain lengths from a Schulz–Zimm distribution, calls `AmberPolymerBuilder` per unique DP, merges the three force fields, packs with Packmol at 0.10 g/cm³, and exports a LAMMPS data file and `system.ff` to `peo_litfsi_output/lammps/`. Running it requires AmberTools and Packmol in a conda environment named `AmberTools25`.

## See Also

- [Polydisperse Systems](../user-guide/05_polydisperse_systems.md) — end-to-end workflow from distribution to LAMMPS export
- [Discovery engine reference](https://github.com/MolCrafts/molmcp) — the full code-graph schema, snapshot/cache mechanics, and CLI for the curious
