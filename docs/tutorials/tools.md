# Tools: Packaged Workflows for Common Tasks

## Modularity costs keystrokes

MolPy splits molecular modeling into focused modules — parser, adapter, builder, reacter, typifier — each owning one responsibility. This makes the library predictable and testable, but it also means that accomplishing a single task often requires importing from four or five packages, calling them in the right order, and handling edge cases like a missing RDKit installation or a failed 3D embedding. The boilerplate is identical every time. Only the inputs change.

**A Tool is a packaged workflow that wires multiple MolPy modules together into a single callable for a common task.**

Tools are not low-level building blocks. They sit *above* the class layer (`PolymerBuilder`, `Connector`, `Reacter`) and make opinionated choices on your behalf — which connector to use, how to detect leaving groups, what fallback to apply when RDKit is absent. When those choices match your workflow, a Tool saves you from repeating the same five-step setup. When they do not, drop to the class layer and wire things yourself. Tools also differ from `Compute` operations (MSD, displacement correlations), which analyze trajectory data rather than orchestrate build workflows.

## Think of a Tool as a predefined workflow

The key design idea is that a Tool separates *what stays fixed* from *what varies*. Configuration — which force field to use, whether to optimize geometry, what random seed — is set once at construction and cannot change afterward. Runtime data — which SMILES string, which monomer library — flows through `run()` on each call.

This matters because it mirrors how experimental workflows work: the method stays fixed across a study, and only the samples change. A frozen `PrepareMonomer()` instance *is* the protocol. You can reuse it, serialize it, log it, or hand it to another function, and it will behave identically every time.

In practice, you will interact with this split naturally. When you write `PrepareMonomer(optimize=True)`, the `optimize=True` is configuration — it defines the protocol. When you call `prep.run("{[<]CCO[>]}")`, the SMILES string is runtime data — it is the sample you feed through that protocol. If you later need the same workflow without geometry optimization, create a second instance with `optimize=False` rather than mutating the first one.

## From five imports to one call

To see the difference, consider preparing a monomer. Without Tools, you need the parser to convert BigSMILES into an `Atomistic`, the RDKit adapter to bridge representations, the `Generate3D` operation to embed coordinates, a sync step back to internal format, and a topology call for angles and dihedrals. With `PrepareMonomer`, all of that collapses into a single call that also handles the fallback when RDKit is not installed:

```python
from molpy.tool import PrepareMonomer

prep = PrepareMonomer()
eo = prep.run("{[<]CCO[>]}")
```

The returned `eo` is a fully prepared `Atomistic` struct — port atoms tagged with `port="<"` and `port=">"`, 3D coordinates embedded (if RDKit is available), and angles and dihedrals computed from the bond graph. This struct is ready to feed into a polymer builder or export to a file format.

The same principle scales up. `polymer()` auto-detects notation (G-BigSMILES, CGSmiles, or CGSmiles with inline fragments) and dispatches to the right internal path — parser, monomer preparation, connector setup, chain assembly — so you do not need to know which classes are involved:

```python
from molpy.tool import polymer

# G-BigSMILES — monomer structure + degree of polymerization in one string
chain = polymer("{[<]CCO[>]}|10|")

# CGSmiles with inline fragment definitions
chain = polymer("{[#EO]|10}.{#EO=[<]CCO[>]}")

# CGSmiles with an external monomer library
chain = polymer("{[#EO]|10}", library={"EO": eo})
```

All three notations produce the same result: a 10-unit PEO chain. The difference is where the monomer definition lives — inside the string, as an inline fragment, or in a pre-prepared library. `polymer()` figures out which path to take based on the string syntax.

For polydisperse systems, `polymer_system()` adds distribution sampling and batch building on top of the same machinery. A single G-BigSMILES string encodes monomer structure, distribution type, distribution parameters, and target total mass:

```python
from molpy.tool import polymer_system

chains = polymer_system(
    "{[<]CCO[>]}|schulz_zimm(1500,3000)||50000|",
    random_seed=42,
)
```

This returns a `list[Atomistic]` — one struct per chain, with chain lengths sampled from a Schulz-Zimm distribution (Mn=1500, Mw=3000) until the total system mass reaches approximately 50,000 g/mol. The `random_seed` makes the sampling reproducible.

When you need to inspect intermediate results — checking the system plan before committing to a build, or building chains one at a time with different settings — use the step-level Tools directly instead of the all-in-one `polymer_system()`. These expose the same workflow in finer granularity:

```python
from molpy.tool import PrepareMonomer, BuildPolymer, PlanSystem
```

- `PrepareMonomer` — parse BigSMILES, generate 3D coordinates, compute topology
- `PlanSystem` — sample chain lengths from a distribution, return chain specifications without creating atoms
- `BuildPolymer` — assemble a single chain from a CGSmiles string and a monomer library

## When to drop to the class layer

Tools cover the common path. Drop to the class layer when you need to change the rules — a custom leaving group selector, non-standard connector rules between monomer pairs, or a novel placement strategy. For example, if your polymerization mechanism removes a specific functional group rather than a hydrogen, you need to define a custom `Reacter` with your own site and leaving selectors, which the Tool layer does not expose.

In practice, most research workflows stay at the Tool level. The class layer is for library developers and unusual chemistry.

## Defining your own Tool

If you find yourself repeating the same multi-module setup across scripts — parse, adapt, parameterize, export — that sequence is a candidate for a Tool. Inherit from `Tool`, declare configuration as frozen dataclass fields, and implement `run()`:

```python
from dataclasses import dataclass
from molpy.tool import Tool

@dataclass(frozen=True)
class ParameterizeMolecule(Tool):
    force_field: str = "gaff2"
    charge_method: str = "bcc"

    def run(self, smiles: str):
        # parse -> adapt -> parameterize -> return
        ...
```

Because the dataclass is frozen, the configuration cannot drift between calls. The workflow is reproducible by construction. Once defined, your custom Tool works exactly like the built-in ones — create an instance with your configuration, then call `run()` with different inputs.

## See Also

- [API Reference: Tool](../api/tool.md) -- parameter details for all built-in Tools
- [Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb) -- end-to-end workflow from distribution design to LAMMPS export
