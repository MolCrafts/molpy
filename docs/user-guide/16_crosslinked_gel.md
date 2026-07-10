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

Two ingredients carry the chemistry — `AmberTools` for parameterization and `GraphAssembler + ExhaustiveSelector` for the graph edit — and nothing in between is hand-rolled.

## A monomer parameterized once becomes a GAFF-typed chain

The ethylene-oxide repeat is written as dimethyl ether (`COC`) and cut on its C–C bonds, so the ether oxygen stays *internal* (antechamber types it `os`) and every residue junction is a `c3–c3` bond. Ports on the two carbons tell the polymer builder where to stitch.

```python
import molpy as mp
from molpy.builder.ambertools import AmberTools
from molpy.conformer import Conformer
from molpy.core.atomistic import Atomistic
from molpy.parser import parse_molecule

GAFF = dict(work_dir="amber", force_field="gaff2", env="AmberTools25", env_manager="conda")


def eo_monomer() -> Atomistic:
    m = Conformer(speed="fast", seed=1).generate(parse_molecule("COC"))[0]
    c1, c2 = (a for a in m.atoms if a.get("element") == "C")
    c1["port"], c2["port"] = "<", ">"   # stitch here
    return m
```

`AmberTools.build_polymer` parameterizes the monomer *once* (antechamber assigns GAFF types and AM1-BCC charges, prepgen builds a residue) and tleap's `sequence` stitches the requested number of copies into one chain. The result is a `Frame` plus a `ForceField`; `Atomistic.from_frame` turns the frame into a graph we can edit, and a short sander minimization relaxes the single chain before it is replicated.

```python
def build_peo_chain(amber: AmberTools, *, dp: int = 20) -> tuple[Atomistic, object]:
    res = amber.build_polymer(f"{{[#EO]|{dp}}}", library={"EO": eo_monomer()})
    strand = Atomistic.from_frame(res.frame)
    relaxed = amber.minimize(res, max_iter=400)          # sander, native minimizer
    for atom, p in zip(strand.atoms, relaxed["atoms"][["x", "y", "z"]]):
        atom["x"], atom["y"], atom["z"] = map(float, p)
    strand.move(list(-strand.xyz.mean(0)), entity_type=mp.Atom)
    return strand, res.ff
```

A DP-20 chain comes out as ~142 atoms carrying only `c3`, `h1`, and `os` — a clean, coiled, LAMMPS-runnable PEO strand.

## Crosslink sites are marked before replication

We know *a priori* where crosslinks may form — on the backbone carbons — so we tag a uniform subset once, on the single chain, before it is copied. Retyping every `spacing`-th carbon to a marker `cx` (a crosslink-site type with the same GAFF parameters as `c3`) means the reaction can later target only those carbons, and every grid copy inherits the marks for free.

```python
def mark_crosslink_sites(strand, *, spacing=3, marker="cx"):
    carbons = [a for a in strand.atoms if a.get("element") == "C"]
    for carbon in carbons[::spacing]:
        carbon["type"] = marker
    return strand
```

The marker is matched by a molrs `%LABEL` predicate in the reaction SMARTS, not by geometry — so there is exactly one kind of matcher, and unmarked carbons are never even considered.

## The chain is replicated onto a jittered grid

Each copy is rigidly rotated and translated onto a lattice with a spacing smaller than the coil's extent, so neighbouring chains interpenetrate and present crosslinkable carbons to one another. Every copy gets its own `mol_id`, which the crosslinker uses to forbid a chain from bonding to itself.

```python
import numpy as np

def grid_pack(strand, grid=3, pack=9.5, jitter=1.0, seed=7):
    rng = np.random.default_rng(seed)
    box = mp.Atomistic()
    for mid, (i, j, k) in enumerate(
        (i, j, k) for i in range(grid) for j in range(grid) for k in range(grid)
    ):
        copy = strand.copy()
        axis = rng.normal(size=3)
        copy.rotate(list(axis / np.linalg.norm(axis)), float(rng.uniform(0, 2 * np.pi)))
        copy.move(list(np.array([i, j, k], float) * pack + rng.uniform(-jitter, jitter, 3)),
                  entity_type=mp.Atom)
        for atom in copy.atoms:
            atom["mol_id"] = mid
        box.merge(copy)
    return box
```

A `3³` grid gives 27 chains — about 3800 atoms once packed.

## The crosslinker forms C–C bonds and re-types each junction

`GraphAssembler + ExhaustiveSelector` copies the packed graph, matches the two reactant patterns, pairs occurrences within `cutoff`, and applies the reaction to each chosen pair. The reaction abstracts one hydrogen from each of two marked carbons and forms a C–C bond between them:

```python
XLINK = "[C;%cx:1][H].[C;%cx:2][H]>>[C:1][C:2]"
```

`[C;%cx]` matches a carbon only if the per-atom label map marks it `cx`, so the search is confined to the sites we tagged. The unmapped hydrogens are the leaving groups.

Passing a `typifier=` hook makes the crosslinker parameterize the junctions it creates. `AmberToolsTypifier` types each formed junction's *affected region* — the small ball of atoms around the new bond — and writes the recomputed atom **types** back onto the network, while accumulating the junction's bonded terms (the `c3–c3–c3` angle and junction dihedrals a linear chain never has). Three choices keep this correct and cheap:

- **Charge is conserved, not recomputed.** The reaction deletes the leaving hydrogens; the crosslinker folds each deleted charge onto the carbon it detached from, so the network stays neutral locally. Re-deriving charges from a capped fragment would be non-local and biased (a sliced ether oxygen looks like a hydroxyl to antechamber), so the typifier reads back types only and leaves charge alone.
- **Regions are typed with a small radius.** GAFF atom types depend on a one-to-two-bond environment, so `AmberToolsTypifier(amber, context_radius=2)` (the default) types the junction identically to a wide ball — while keeping the region from reaching neighbouring junctions, which would fragment the retype cache into many spurious variants. Raise `context_radius` only for a bulky or fused junction.
- **Identical junctions are typed once.** A per-`apply` cache keyed by the region's isomorphism-invariant hash means the dozens of chemically identical PEO junctions are parameterized a single time.

```python
from molpy.builder.crosslink import GraphAssembler + ExhaustiveSelector
from molpy.typifier.ambertools import AmberToolsTypifier

amber = AmberTools(charge_method="bcc", **GAFF)
strand, net_ff = build_peo_chain(amber)
mark_crosslink_sites(strand, spacing=3)
box = grid_pack(strand)

typifier = AmberToolsTypifier(amber)                 # context_radius=2 by default
xl = GraphAssembler + ExhaustiveSelector(
    XLINK, cutoff=6.5, typifier=typifier,
    exclude_same_molecule=True, exclude_same_match=True,
)
gel = xl.apply(box)                                  # 27 chains -> one network
```

After crosslinking, the marker has done its job, so the unreacted `cx` carbons are restored to `c3`:

```python
def restore_crosslink_sites(gel, *, marker="cx", to="c3"):
    for atom in gel.atoms:
        if atom.get("type") == marker:
            atom["type"] = to
    return gel

restore_crosslink_sites(gel)
```

## The force field is assembled and exported to LAMMPS

The junction terms the typifier collected are merged into the chain force field. `get_topo` perceives the network's angles and dihedrals, and `assign_bonded_types` labels every bond, angle, and dihedral by its atom-type tuple against the merged force field. A final equal shift over all charges removes the tiny residue left by the folds, and `write_lammps_system` emits a data + force-field pair — restricting every coefficient to the types the structure actually uses.

```python
from molpy.io import write_lammps_system

net_ff.merge(typifier.forcefield)                    # + junction bonded terms
shift = sum(a.get("charge", 0.0) for a in gel.atoms) / gel.n_atoms
for a in gel.atoms:
    a["charge"] = a.get("charge", 0.0) - shift       # exact neutrality

gel = gel.get_topo(gen_angle=True, gen_dihe=True).assign_bonded_types(net_ff)
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

Everything above builds the gel; the last step measures it. `read_lammps_data` returns a `Frame`, and `Atomistic.from_frame` rebuilds the topology — bonds, angles, and dihedrals — as a graph. No AmberTools or LAMMPS is needed here.

```python
frame = mp.io.read_lammps_data("nvt/relaxed.data", atom_style="full")
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
