# Building a Crosslinked PEO Gel

Grow a GAFF-typed PEO chain, replicate it into a melt, crosslink the melt into a single percolated network, equilibrate it in LAMMPS, and read the network's connectivity back — a complete offline crosslinking workflow driven from MolPy.

!!! warning "External dependencies"
    This guide requires **AmberTools** (via conda) for parameterization and **LAMMPS** (`lmp_mpi`) for equilibration. The connectivity analysis at the end needs neither — it runs on the written data file alone.

??? note "Setting up AmberTools"
    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    which antechamber tleap sander   # each should print a path
    ```

    MolPy's wrappers activate the environment automatically; the `env="AmberTools25"` argument below tells them which one. Replace the name throughout if you use a different one.

## Workflow overview

Offline crosslinking builds the whole network *before* the simulation starts, so the exported system is a finished topology rather than a set of reaction templates. The path is: parameterize one ethylene-oxide monomer and stitch it into a chain; mark where crosslinks may form while the chain is still a single molecule; replicate the chain onto a jittered grid so the copies interpenetrate; let a deterministic crosslinker pair nearby marked carbons and re-type each new junction; merge the junction force-field terms and write a LAMMPS system; equilibrate; and finally reload the relaxed structure to measure the network it produced.

Two ingredients carry the chemistry — `AmberTools` for parameterization and
`GraphAssembler` + `ExhaustiveSelector` for the graph edit (sites via
`fields.SITE` / `SiteMap`, melt via `Replicas`) — and nothing in between is
hand-rolled. For pure-topology teaching without Amber, see the
[Polymer Topologies](topology/index.md) section
(`examples/topology/07_gel_exhaustive.py` and siblings).

## A monomer parameterized once becomes a GAFF-typed chain

The ethylene-oxide repeat is written as dimethyl ether (`COC`) and cut on its C–C bonds, so the ether oxygen stays *internal* (antechamber types it `os`) and every residue junction is a `c3–c3` bond. Standard `fields.SITE` labels and a `Reaction` tell every backend where and how to stitch.

```python
import molpy as mp
from molpy.builder.ambertools import AmberTools
from molpy.builder.assembly import SiteMap
from molpy.conformer import Conformer
from molpy.core.atomistic import Atomistic
from molpy.parser import parse_molecule

GAFF = dict(work_dir="amber", force_field="gaff2", env="AmberTools25", env_manager="conda")


def eo_monomer() -> Atomistic:
    m = Conformer(speed="fast", seed=1).generate(parse_molecule("COC"))[0]
    SiteMap(m).label_elements("C", "a", "b")
    return m


BACKBONE = mp.Reaction("[C;%a:1][H].[C;%b:2][H]>>[C:1][C:2]")
```

`AmberTools.build_polymer` parameterizes the monomer *once* (antechamber assigns GAFF types and AM1-BCC charges, prepgen builds a residue) and tleap's `sequence` stitches the requested number of copies into one chain. The result is a `Frame` plus a `ForceField`; `Atomistic.from_frame` turns the frame into a graph we can edit, and a short sander minimization relaxes the single chain before it is replicated.

```python
def build_peo_chain(amber: AmberTools, *, dp: int = 20) -> tuple[Atomistic, object]:
    res = amber.build_polymer(
        f"{{[#EO]|{dp}}}", library={"EO": eo_monomer()}, reaction=BACKBONE
    )
    strand = Atomistic.from_frame(res.frame)
    relaxed = amber.minimize(res, max_iter=400)          # sander, native minimizer
    for atom, p in zip(strand.atoms, relaxed["atoms"][["x", "y", "z"]]):
        atom["x"], atom["y"], atom["z"] = map(float, p)
    strand.move(list(-strand.xyz.mean(0)), entity_type=mp.Atom)
    return strand, res.ff
```

A DP-20 chain comes out as ~142 atoms carrying only `c3`, `h1`, and `os` — a clean, coiled, LAMMPS-runnable PEO strand.

## Crosslink sites are marked before replication

We know *a priori* where crosslinks may form — on the backbone carbons — so we tag a uniform subset once, on the single chain, before it is copied. Use **`fields.SITE`** via `SiteMap` (do not rewrite GAFF atom types): every `spacing`-th carbon gets SITE `"x"`, its leaving hydrogen SITE `"h"`, with charge folded so deleted atoms carry zero.

```python
from molpy.builder.assembly import SiteMap
from molpy.core import fields

def mark_crosslink_sites(strand, *, spacing=3, site="x", leaving="h"):
    carbons = [
        a for a in strand.atoms
        if a.get(fields.ELEMENT) == "C"
        and any(n.get(fields.ELEMENT) == "H" for n in strand.get_neighbors(a))
    ]
    SiteMap(strand).every_nth(carbons, spacing, site, leaving=leaving, fold_charge=True)
    return strand
```

The marker is matched by a molrs `%LABEL` predicate in the reaction SMARTS (`%x` / `%h`), not by geometry — unmarked carbons are never considered.

## The chain is replicated onto a jittered grid

`Replicas.grid` rigidly rotates and translates copies onto a lattice so neighbouring chains interpenetrate. Each copy gets a `mol_id`; the selector forbids same-component pairs while chains remain separate.

```python
from molpy.builder.assembly import Replicas

box = Replicas(strand).grid(3, spacing=9.5, jitter=1.0, seed=7)
# 3³ = 27 chains
```

## The assembler forms C–C bonds and re-types each junction

`GraphAssembler` + `ExhaustiveSelector` copies the packed graph, matches the two reactant patterns, pairs occurrences within `cutoff`, and applies the reaction to each chosen pair:

```python
XLINK = "[C;%x:1][H;%h].[C;%x:2][H;%h]>>[C:1][C:2]"
```

Passing `typifier=` parameterizes junctions as they form. `AmberToolsTypifier` types each *affected region* (small ball around the new bond) and writes atom **types** back, accumulating junction bonded terms. Charge is conserved by the pre-fold above (not recomputed on the fragment). `reach=2` matches GAFF’s one-to-two-bond environment; identical junctions hit the retype cache once.

```python
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler
from molpy.typifier import AmberToolsTypifier

amber = AmberTools(charge_method="bcc", **GAFF)
strand, net_ff = build_peo_chain(amber)
mark_crosslink_sites(strand, spacing=3)
box = Replicas(strand).grid(3, spacing=9.5, jitter=1.0, seed=7)

typifier = AmberToolsTypifier(amber)
gel = GraphAssembler(
    mp.Reaction(XLINK), typifier=typifier, reach=2,
).assemble(
    box,
    ExhaustiveSelector(cutoff=6.5, exclude_same_molecule=True, exclude_same_match=True),
)
# thaw unreacted leaving H (q0), then clear temporary SITE labels — types were never rewritten
```

## The force field is assembled and exported to LAMMPS

The junction terms the typifier collected are merged into the chain force field. `get_topo` perceives the network's angles and dihedrals, and `ForceFieldParams` labels every bond, angle, and dihedral against the merged force field. `ForceFieldParams` is not a typifier — it decides no atom type, it spends one — and it is the tail every force-field typifier ends with; here the atom types are already on the graph, so it is used on its own. A final equal shift over all charges removes the tiny residue left by the folds, and `write_lammps_system` emits a data + force-field pair — restricting every coefficient to the types the structure actually uses.

```python
from molpy.io import write_lammps_system
from molpy.typifier import ForceFieldParams

net_ff.merge(typifier.forcefield)                    # + junction bonded terms
shift = sum(a.get("charge", 0.0) for a in gel.atoms) / gel.n_atoms
for a in gel.atoms:
    a["charge"] = a.get("charge", 0.0) - shift       # exact neutrality

gel = ForceFieldParams(net_ff, strict=False).assign(
    gel.get_topo(gen_angle=True, gen_dihe=True)
)
frame = gel.to_frame()
frame["atoms"]["id"] = np.arange(1, frame["atoms"].nrows + 1)
xyz = frame["atoms"][["x", "y", "z"]]
frame.box = mp.Box.cubic(float((xyz.max(0) - xyz.min(0)).max()) + 8.0, origin=xyz.min(0) - 4.0)
write_lammps_system("gel", frame, net_ff)            # gel/system.data + gel/system.ff
```

## The melt is equilibrated in LAMMPS

The packed melt starts with overlapping atoms, so equilibration begins with a capped-displacement push-off to separate them, followed by an energy minimization and a short NVT run. `LAMMPSEngine` writes the input, runs `lmp_mpi`, and returns the relaxed frame; the final structure is written to `nvt/relaxed.data`.

```python
from molpy.engine import LAMMPSEngine

engine = LAMMPSEngine("lmp_mpi", launcher=["mpirun", "-np", "8"])
frame = engine.md(frame, net_ff, ensemble="nve/limit", steps=4000, limit=0.1,
                  timestep=1.0, temperature=300.0, workdir="push")
frame = engine.minimize(frame, net_ff, max_iter=500, workdir="relax")
frame = engine.md(frame, net_ff, ensemble="nvt", steps=6000, temperature=300.0,
                  timestep=1.0, workdir="nvt")
```

## The network's connectivity is read back from the data file

Everything above builds the gel; the last step measures it. `read_lammps_data` returns explicit format products; this example selects `.frame`, then `Atomistic.from_frame` rebuilds bonds, angles, and dihedrals as a graph. No AmberTools or LAMMPS is needed here.

```python
frame = mp.io.read_lammps_data("nvt/relaxed.data", atom_style="full").frame
gel = mp.Atomistic.from_frame(frame)
```

**How many subgraphs?** A gel is, by definition, one giant crosslinked molecule, so a well-formed network should be a single connected component. `topo_distances` does a native breadth-first traversal from an atom and returns every atom it can reach; repeating from each unvisited atom counts the components.

```python
seen, sizes = set(), []
for atom in gel.atoms:
    if atom.handle in seen:
        continue
    reached = [h for h, _ in gel.topo_distances(atom.handle)]
    seen.update(reached)
    sizes.append(len(reached))
# -> [3574]  : one component, every chain fused into the network
```

**How many crosslinks?** In linear PEO every backbone CH₂ has exactly one carbon neighbour; a crosslinked carbon gained a second C–C bond, so it has two. Counting those interior junctions gives the crosslinked carbons directly. The total number of crosslink *bonds* also follows from a purely topological invariant — the cyclomatic number `Z = bonds − atoms + components`, the count of independent loops the crosslinks closed — plus the bridges needed to fuse the separate chains (`chains − components`).

```python
def carbon_neighbours(atom):
    return sum(1 for nb in gel.get_neighbors(atom) if nb.get("type") == "c3")

junctions = [a for a in gel.atoms if a.get("type") == "c3" and carbon_neighbours(a) >= 2]
n_bonds = sum(1 for _ in gel.bonds)
loops = n_bonds - gel.n_atoms + len(sizes)          # cyclomatic number
n_xlink = loops + (27 - len(sizes))                 # loops + chain-fusing bridges
```

**What is the crosslink density?** Divide the crosslink count by the box volume for a number density, and by the chain count for a per-chain figure.

```python
V = float(np.prod(frame.box.lengths))
density = n_xlink / V                                # A^-3
density_molar = density / 6.022e23 * 1e24            # mol/cm^3
```

For the 27-chain, DP-20 melt built above, the analysis reports:

| quantity | value |
| --- | --- |
| atoms / bonds | 3574 / 3677 |
| connected subgraphs | **1** (fully percolated) |
| crosslinks | **130** (104 loops + 26 bridges) |
| interior junction carbons | 236 |
| crosslink density | **4.6 × 10⁻⁴ mol/cm³** (2.8 × 10⁻⁴ Å⁻³) |
| crosslinks per chain | 4.8 |
| mean atoms per strand | 27.5 |

A single connected component confirms the melt gelled rather than staying a collection of chains, and the crosslink count recovered from the topology matches the number of reactions the crosslinker reported — the network on disk is the network you built.
