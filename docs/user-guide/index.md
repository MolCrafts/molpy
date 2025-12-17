# User Guide

Welcome to the MolPy User Guide! This section is organized around **tasks and workflows**. Each guide provides comprehensive examples to help you accomplish specific goals.

## Guides by Topic

### 🧪 [Parsing Chemistry](01_parsing_chemistry.ipynb)
**Goal:** Define molecules using syntax instead of manual construction.
-   **BigSMILES**: Define monomers with reactive ports.
-   **CGSmiles**: Define coarse-grained topology (blocks, branches).
-   **GBigSMILES**: Define systems with distributions.

**Core APIs and Data Structures**  
Use the `molpy.parser.smiles` entry points (`parse_bigsmiles`, `parse_gbigsmiles`, `parse_cgsmiles`) to convert notation into `BigSmilesMoleculeIR`, `GBigSmilesSystemIR`, or `CGSmilesIR`. These IR objects expose explicit `BondingDescriptorIR`, `StochasticObjectIR`, and terminal descriptors so downstream builders know which ports (`[<]`, `[>]`, `[$]`) exist and how repeat units are grouped.

**Design Notes**  
`BigSmilesParserImpl` enforces the BigSMILES v1.1 rule that every stochastic object has leading and trailing terminal descriptors, while the generative path (`allow_generative=True`) is the only mode that accepts weight annotations, dot-size hints, or system-level `DistributionIR`. CGSmiles parsing keeps the topology graph intact (nodes, bonds, repeat operators), so nothing is lost before `PolymerBuilder` consumes it.

**Canonical Example**  
The parser returns structured IR; validation happens during parsing, so later steps work with clean data:

```python
from molpy.parser.smiles import parse_bigsmiles

molecule_ir = parse_bigsmiles("{[<]CC[>]}[$]CO[$]")
assert molecule_ir.stochastic_objects[0].terminals[0].role == "left"
```

### 🏗️ [Building Polymers](02_polymer_stepwise.ipynb)
**Goal:** Assemble complex polymer architectures.
-   **Linear Polymers**: Standard chain growth.
-   **Copolymers**: Block and random sequences.
-   **Complex Architectures**: Graft, star, and cross-linked polymers.

**Core APIs and Data Structures**  
`PolymerBuilder` consumes CGSmiles IR graphs node-by-node, cloning every monomer template in the `library: Mapping[str, Atomistic]` and tagging atoms with their originating node id so depth-first assembly can reconnect branches and rings deterministically. Connectors (`AutoConnector`, `TableConnector`, `ReacterConnector`) translate BigSMILES port roles into explicit `(left_port, right_port)` choices, while `PortInfo` (exposed by `get_all_port_info`) tracks whether a port has already been consumed. Optional `Placer` and `TypifierBase` hooks let you pre-place monomers and retype only the atoms touched by a connection.

**Design Notes**  
`PolymerBuilder._validate_ir` rejects sequences whose labels are missing from the monomer library, so CGSmiles references cannot silently fall back to incorrect fragments. DFS traversal retains each node's adjacency so branches are connected exactly once, and the builder records every `ConnectionMetadata` entry—useful when you later need to re-type or audit the assembly history. When you switch to `ReacterConnector`, you must provide an explicit `port_map`; there is intentionally no implicit fallback because the reacter layer performs chemistry (bond changes, leaving groups) that must stay deterministic.

**Canonical Example**  
Create the builder with the smallest viable inputs: a monomer library, a connector, and (optionally) a typifier.

```python
from molpy.core.atomistic import Atomistic
from molpy.builder.polymer import PolymerBuilder
from molpy.builder.polymer.connectors import ReacterConnector
from molpy.builder.polymer.port_utils import set_port_metadata
from molpy.reacter import Reacter
from molpy.reacter.selectors import select_identity, select_none
from molpy.reacter.transformers import form_single_bond

def monomer_with_port(port_name: str, role: str) -> Atomistic:
    struct = Atomistic()
    atom = struct.def_atom(symbol="C")
    atom["port"] = port_name
    set_port_metadata(atom, port_name, role=role)
    return struct

library = {
    "EO2": monomer_with_port("right", role="right"),
    "PS": monomer_with_port("left", role="left"),
}

reaction = Reacter(
    name="stub-bond",
    anchor_selector_left=select_identity,
    anchor_selector_right=select_identity,
    leaving_selector_left=select_none,
    leaving_selector_right=select_none,
    bond_former=form_single_bond,
)

connector = ReacterConnector(
    default=reaction,
    port_map={("EO2", "PS"): ("right", "left")},
)

builder = PolymerBuilder(library=library, connector=connector)
result = builder.build("{[#EO2][#PS]}")
assembled = result.polymer
```

### 🔗 [Polymer SMILES](03_polymer_smiles.ipynb)
**Goal:** Program chemical reactivity.
-   **Manual Reactions**: Connecting specific molecules.
-   **Custom Mechanisms**: Defining new reaction rules with Selectors.
-   **Templates**: Generating reaction templates for MD engines (LAMMPS).

**Core APIs and Data Structures**  
`molpy.reacter.Reacter` composes anchor selectors, leaving selectors, and `BondFormer` callables so a single object captures one reaction mechanism. Selector utilities (e.g., `select_identity`, `select_c_neighbor`, `select_one_hydrogen`) operate on `Atomistic` graphs, mapping the SMILES-marked port atom to the real anchor atom and enumerating atoms to delete. `ReactionResult` keeps reactant metadata (ports vs. anchors), all topology changes, and whether the run requires retypification.

**Design Notes**  
Reacter deliberately refuses to guess ports—every call passes explicit `port_atom_L`/`port_atom_R`. That separation lets higher-level objects such as `MonomerLinker` or `ReacterConnector` manage context (sequence labels, port names) while the reaction engine focuses on bond creation, leaving group removal, and incremental topology re-typing through `TopologyDetector`. Because selectors are just functions, you can mix and match them to express mechanisms ranging from dehydration to radical coupling without modifying the core reactor.

**Canonical Example**  
Define the minimal selectors plus a bond former, then execute a reaction on two port-annotated fragments.

```python
from molpy.core.atomistic import Atomistic
from molpy.reacter import Reacter
from molpy.reacter.selectors import select_identity, select_none
from molpy.reacter.transformers import form_single_bond

def single_port_struct(port_name: str):
    struct = Atomistic()
    atom = struct.def_atom(symbol="C")
    atom["port"] = port_name
    return struct, atom

left_struct, left_port = single_port_struct("L")
right_struct, right_port = single_port_struct("R")

cc_coupling = Reacter(
    name="C-C",
    anchor_selector_left=select_identity,
    anchor_selector_right=select_identity,
    leaving_selector_left=select_none,
    leaving_selector_right=select_none,
    bond_former=form_single_bond,
)
result = cc_coupling.run(left_struct, right_struct, port_atom_L=left_port, port_atom_R=right_port)
```

### 🔗 [Polymer Crosslinking](04_polymer_crosslinking.ipynb)
**Goal:** Create cross-linked polymer networks.
-   **Reaction Templates**: Defining crosslinking reactions.
-   **Network Formation**: Building 3D polymer networks.
-   **LAMMPS Integration**: Generating reaction templates for MD.

**Core APIs and Data Structures**  
`TemplateReacter` wraps a base `Reacter` and layers on `react_id` tracking plus `TemplateResult` (pre/post subgraphs, edge atoms, atom id maps) so the same reaction specification can both build a network and emit LAMMPS `fix bond/react` templates. Every run returns `(ReactionResult, TemplateResult)`, giving you access to `removed_atoms`, `new_bonds`, and the exact subgraph extracted with the configured `radius`.

**Design Notes**  
Before execution, `TemplateReacter` assigns deterministic `react_id` values to all atoms—this guarantees pre/post mapping consistency even when the reaction prunes atoms or reorders bonds. Subgraph extraction uses the merged reactants (before chemistry) and the final product separately, so template edges never mix states. Because `TemplateReacter` delegates chemistry to the underlying `Reacter`, you can switch between different bond formers or selectors without rewriting the template logic; only the radius or naming changes.

**Canonical Example**  
Wrap an existing reaction and request both the reacted product and the corresponding template fragments.

```python
from molpy.core.atomistic import Atomistic
from molpy.reacter.template import TemplateReacter
from molpy.reacter.selectors import select_identity, select_none
from molpy.reacter.transformers import form_single_bond

def stub_struct(port_name: str):
    struct = Atomistic()
    atom = struct.def_atom(symbol="C")
    atom["port"] = port_name
    return struct, atom

left, port_atom_L = stub_struct("L")
right, port_atom_R = stub_struct("R")

template_reacter = TemplateReacter(
    name="amine_epoxy",
    anchor_selector_left=select_identity,
    anchor_selector_right=select_identity,
    leaving_selector_left=select_none,
    leaving_selector_right=select_none,
    bond_former=form_single_bond,
    radius=2,
)
reaction, template = template_reacter.run_with_template(left, right, port_atom_L, port_atom_R)
```

### 📊 [Polydisperse Systems](05_polymer_polydisperse.ipynb)
**Goal:** Model realistic material distributions.
-   **Distributions**: Schulz-Zimm, Poisson, Flory-Schulz.
-   **Ensembles**: Generating representative populations.
-   **Analysis**: Calculating molecular weight moments ($M_n$, $M_w$).

**Core APIs and Data Structures**  
Sequence construction is layered: a `SequenceGenerator` controls monomer ordering (e.g., `WeightedSequenceGenerator`), `PolydisperseChainGenerator` samples degree of polymerization or chain mass using `DPDistribution`/`MassDistribution` implementations (Poisson, Flory-Schulz, Schulz-Zimm), and `SystemPlanner` keeps requesting chains until `target_total_mass` is satisfied within `max_rel_error`. When a gBigSMILES `DistributionIR` is present, `create_polydisperse_from_ir` translates it into the appropriate distribution object so parser annotations flow directly into sampling.

**Design Notes**  
`PolydisperseChainGenerator` refuses to mix incompatible capabilities (calling `sample_dp` on a mass-only distribution raises `TypeError`), ensuring you match your sampling space to the distribution definition. The planner’s `_try_trim_chain` method uses expected composition and end-group mass to compute how much of the final chain to retain, so trimming never creates a chain heavier than the remaining budget. Mass accounting happens per-chain, letting you compute $M_n$ and $M_w$ directly from the `SystemPlan`.

**Canonical Example**  
Provide the three layers explicitly so every design assumption stays visible:

```python
from molpy.builder.polymer.system import (
    WeightedSequenceGenerator,
    PolydisperseChainGenerator,
    SystemPlanner,
    PoissonPolydisperse,
)
import random

seq = WeightedSequenceGenerator(monomer_weights={"EO2": 3, "PS": 1})
dist = PoissonPolydisperse(lambda_param=20.0)
chain_gen = PolydisperseChainGenerator(seq, monomer_mass={"EO2": 44.0, "PS": 104.0}, distribution=dist)
planner = SystemPlanner(chain_gen, target_total_mass=5e5)
plan = planner.plan_system(rng=random.Random(7))
```

### ⚙️ [Simulation Preparation](06_simulation_preparation.ipynb)
**Goal:** Prepare a system for molecular dynamics.
-   **Packing**: Creating dense simulation boxes (solvation).
-   **Typifier**: Assigning force field parameters (OPLS-AA, GAFF).
-   **Optimization**: Minimizing energy to remove overlaps.
-   **Export**: Writing ready-to-run LAMMPS data files.

**Core APIs and Data Structures**  
Packing uses `Molpack`, which orchestrates `Target` objects (frame, copy count, spatial `Constraint`) and dispatches to the configured Packmol backend via `get_packer`. Constraints compose with logical `&`/`|` to encode complex regions, and per-target min-distance penalties prevent overlaps before MD. Typing relies on `TypifierBase` implementations such as `OplsAtomisticTypifier`, `OplsBondTypifier`, and the `LayeredTypingEngine`, which runs SMARTS patterns level-by-level while respecting dependency ordering and circular constraints detected by `DependencyAnalyzer`. Geometry relaxation typically goes through `optimize.LBFGS`, whose stepper enforces `maxstep` and stores curvature pairs (`s_history`, `y_history`) so convergence is predictable. Export paths are centralized in `molpy.io.writers`, letting you hand off the final `Frame` or `ForceField` to LAMMPS/GROMACS/HDF5 writers without duplicating formatting logic.

**Design Notes**  
`Molpack.optimize` always creates its working directory and passes explicit seeds to Packmol, so results are reproducible when you set `seed`. Typifiers expect every atom to carry at least a `type` or `class_` tag; missing values raise immediately (`atomtype_matches` checks both), preventing partially typed structures from slipping into the force field writer. The `LBFGS` optimizer resets its history per-structure id, guaranteeing that successive optimization runs never leak curvature information from earlier systems.

**Canonical Example**  
Minimal pipeline from packing to export:

```python
from pathlib import Path
from molpy.core.frame import Frame
from molpy.pack import Molpack, InsideBoxConstraint

solvent_frame = Frame(blocks={"atoms": {"id": [1], "type": [1], "x": [0.0], "y": [0.0], "z": [0.0]}})
packer = Molpack(workdir=Path("packing-demo"))
constraint = InsideBoxConstraint(length=[20.0, 20.0, 20.0])
target = packer.add_target(solvent_frame, number=25, constraint=constraint)
# When Packmol is available, call: packed = packer.optimize(max_steps=500, seed=11)
```



---

## Need API Basics?

If you are looking for detailed explanations of core classes (like `Atomistic`, `Frame`, `Topology`), check out the **[Tutorials](../tutorials/index.md)** section.
