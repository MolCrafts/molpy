# Polydisperse Systems

From a target molecular-weight distribution to a packed, LAMMPS-ready box: sample the chains, build each one, re-typify the junctions, pack, export.

!!! note "Prerequisites"
    This guide requires RDKit, `molcrafts-molpack`, and the `oplsaa.xml` force field. Familiarity with [Assembly](02_assembly.md) is assumed.

## From distribution to simulation box

Most polymer samples are polydisperse rather than monodisperse. This workflow starts from a target molecular-weight distribution and proceeds through explicit sampling, chain construction, and packing to produce a simulation-ready system.

## Typification happens per monomer, not per chain

Each monomer is parsed, expanded to 3D with hydrogens, and assigned force field types individually. Two of its backbone carbons are then annotated as reaction sites. The builder re-types each junction as the bond forms, so there is no need to re-typify the entire chain after assembly.

```
import molpy as mp
from molpy.core import fields
from molpy import Element
from molpy.builder.assembly import SiteMap
from molpy.conformer import Conformer
from molpy.typifier import OPLSAATypifier

typifier = OPLSAATypifier(strict=False)

# The repeat unit as plain SMILES. Its first two atoms are the backbone carbons
# that will react; everything after them is the pendant group.
MONOMERS = {
 "Sty": "CC(c1ccccc1)",
 "MA": "CC(C(=O)OC)",
}

def build_typed_monomer(smiles, typifier):
 monomer = mp.io.read_smiles(smiles)
 # Name the two reactive carbons directly. The assembler binds the reaction
 # SMARTS below to these `site` annotations, so this is the only place the
 # chemistry of "where does a chain grow" is stated.
 SiteMap(monomer).label_atoms(list(monomer.atoms)[:2], "head", "tail")
 monomer, _ = Conformer(add_hydrogens=True, seed=42).generate(monomer)
 monomer.get_topo(gen_angle=True, gen_dihe=True) # in place
 return typifier.typify(monomer)

library = {label: build_typed_monomer(smi, typifier) for label, smi in MONOMERS.items()}

monomer_mass = {}
for label, mon in library.items():
 mass = sum(Element(a.get("element")).mass for a in mon.atoms)
 sites = [a.get(fields.SITE) for a in mon.atoms if a.get(fields.SITE)]
 monomer_mass[label] = mass
 print(f"{label}: atoms={len(mon.atoms)}, mass={mass:.1f}, sites={sites}")
```

## Sampling draws chain lengths from a statistical distribution

The sampling layer has three components that compose cleanly. `WeightedSequenceGenerator` controls the monomer mole ratio (80:20 here). `PolydisperseChainGenerator` draws a degree of polymerization or mass from the chosen distribution for each chain. `SystemPlanner` accumulates chains until a target total mass is reached, stopping when the accumulated mass is within `max_rel_error` of the target. Four distributions are demonstrated below so that their shape differences become visible in the next section.

```
import numpy as np
from molpy.builder.polymer import (
 SchulzZimmPolydisperse,
 UniformPolydisperse,
 PoissonPolydisperse,
 FlorySchulzPolydisperse,
 WeightedSequenceGenerator,
 PolydisperseChainGenerator,
 SystemPlanner,
)

distributions = {
 "Schulz-Zimm": SchulzZimmPolydisperse(Mn=1400, Mw=1500),
 "Uniform": UniformPolydisperse(min_dp=8, max_dp=22),
 "Poisson": PoissonPolydisperse(lambda_param=14),
 "Flory-Schulz": FlorySchulzPolydisperse(a=0.08),
}

seq_gen = WeightedSequenceGenerator(monomer_weights={"Sty": 8.0, "MA": 2.0})
target_total_mass = 5e5

results = {}
for name, dist in distributions.items():
 chain_gen = PolydisperseChainGenerator(
 seq_generator=seq_gen,
 monomer_mass=monomer_mass,
 end_group_mass=0.0,
 distribution=dist,
)
 planner = SystemPlanner(
 chain_generator=chain_gen,
 target_total_mass=target_total_mass,
 max_rel_error=0.02,
)
 plan = planner.plan_system(np.random.default_rng(42))
 results[name] = plan.chains

for name, chains in results.items():
 mw = np.array([c.mass for c in chains])
 Mn = float(np.mean(mw))
 Mw = float(np.sum(mw**2) / np.sum(mw))
 print(f"{name:15s}: {len(chains):4d} chains, Mn={Mn:.0f}, PDI={Mw / Mn:.3f}")
```

## Visualising the sampled ensembles reveals the distribution shape

The four panels below overlay sampled histograms against their theoretical curves. Schulz-Zimm is plotted as a continuous probability density; the other three are plotted as probability mass functions over degree of polymerization. Vertical dashed lines mark Mn and Mw for each ensemble.

```
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ── colour palette ──
CLR_HIST = "#6baed6" # steel blue – sampled histogram
CLR_EDGE = "#3182bd" # darker blue – histogram edge
CLR_THEO = "#e6550d" # orange – theoretical curve
CLR_MN = "#31a354" # green – Mn line
CLR_MW = "#de2d26" # red – Mw line
CLR_BOX = "#f7f7f7" # near-white – annotation box

def annotate_stats(ax, Mn, Mw, PDI, n_chains):
 txt = "\n".join(
 [
 rf"$M_n = {Mn:.0f}$ g/mol",
 rf"$M_w = {Mw:.0f}$ g/mol",
 rf"PDI $= {PDI:.3f}$",
 rf"$N = {n_chains}$",
 ]
)
 ax.text(
 0.97,
 0.97,
 txt,
 transform=ax.transAxes,
 ha="right",
 va="top",
 fontsize=6.5,
 linespacing=1.4,
 family="monospace",
 bbox=dict(
 boxstyle="round,pad=0.35",
 facecolor=CLR_BOX,
 edgecolor="0.75",
 alpha=0.95,
 linewidth=0.6,
),
)

fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), constrained_layout=True)

for idx, (ax, (name, chains)) in enumerate(zip(axes.flatten(), results.items())):
 dist_obj = distributions[name]
 mw = np.array([c.mass for c in chains])
 dps = np.array([c.dp for c in chains])
 Mn = float(np.mean(mw))
 Mw = float(np.sum(mw**2) / np.sum(mw))
 PDI = Mw / Mn

 if isinstance(dist_obj, SchulzZimmPolydisperse):
 # ── continuous: Freedman–Diaconis bins ──
 iqr = np.subtract(*np.percentile(mw, [75, 25]))
 bw = max(2.0 * iqr / len(mw) ** (1 / 3), 20)
 bins = np.arange(mw.min() - bw, mw.max() + 2 * bw, bw)

 ax.hist(
 mw,
 bins=bins,
 density=True,
 color=CLR_HIST,
 edgecolor=CLR_EDGE,
 linewidth=0.5,
 alpha=0.55,
)
 M_grid = np.linspace(max(0, mw.min() * 0.3), mw.max() * 1.3, 500)
 ax.plot(
 M_grid,
 dist_obj.mass_pdf(M_grid),
 color=CLR_THEO,
 linewidth=1.6,
 label="Theory",
)
 ax.axvline(Mn, color=CLR_MN, ls="--", lw=1, label=r"$M_n$")
 ax.axvline(Mw, color=CLR_MW, ls="--", lw=1, label=r"$M_w$")
 ax.set_xlabel(r"Molecular weight $M$ (g mol$^{-1}$)", fontsize=8)
 ax.set_ylabel("Probability density", fontsize=8)
 else:
 # ── discrete: unit-width bars centred on integers ──
 dp_min, dp_max = int(dps.min()), int(dps.max())
 counts = np.bincount(dps)[dp_min:]
 freq = counts / (counts.sum() or 1)
 x_bar = np.arange(dp_min, dp_min + len(counts))

 ax.bar(
 x_bar,
 freq,
 width=1.0,
 align="center",
 color=CLR_HIST,
 edgecolor=CLR_EDGE,
 linewidth=0.5,
 alpha=0.55,
)

 # Theory curve extends to 99th-percentile of the sample
 if isinstance(dist_obj, UniformPolydisperse):
 support = np.arange(dp_min, dp_max + 1)
 else:
 hi = max(dp_max, int(np.percentile(dps, 99) * 1.15))
 support = np.arange(max(1, dp_min), hi + 1)
 pmf = dist_obj.dp_pmf(support)
 ax.plot(
 support,
 pmf,
 "-o",
 color=CLR_THEO,
 markersize=2.2,
 linewidth=1.3,
 markeredgewidth=0,
 label="Theory",
)

 avg_mass = float(np.mean(mw / dps))
 ax.axvline(Mn / avg_mass, color=CLR_MN, ls="--", lw=1, label=r"$M_n$")
 ax.axvline(Mw / avg_mass, color=CLR_MW, ls="--", lw=1, label=r"$M_w$")

 # Clip x-axis so long tails don't crush the peak
 x_hi = int(np.percentile(dps, 99.5)) + 2
 ax.set_xlim(max(0, dp_min - 1), x_hi)
 ax.xaxis.set_major_locator(MaxNLocator(integer=True))
 ax.set_xlabel(r"Degree of polymerization $n$", fontsize=8)
 ax.set_ylabel("Probability", fontsize=8)

 ax.set_title(name, fontsize=9.5, fontweight="semibold", pad=6)
 ax.tick_params(labelsize=7)
 ax.spines[["top", "right"]].set_visible(False)
 annotate_stats(ax, Mn, Mw, PDI, len(chains))
 if idx == 0:
 ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

plt.savefig("05_polydisperse_distributions.png", dpi=200, bbox_inches="tight")
plt.show()
```

## Radical addition couples monomers without leaving groups

The reaction here is **radical addition**: each connection removes one hydrogen from each backbone carbon and forms a new C–C bond. This differs from the dehydration condensation used in earlier guides — there are no hydroxyl leaving groups, only hydrogen removal from both sides.

That difference lives entirely in the reaction SMARTS. `PolymerBuilder` is the same class either way: it stamps out one residue per monomer of the sequence and applies the reaction between adjacent residues. Because both monomer labels of the copolymer are drawn from one library, a single builder handles every sequence the planner sampled — and reusing it across chains is what lets them share one retype cache.

```
from molpy.builder.assembly import (
 MonomerLibrary,
 PolymerBuilder,
 ResiduePlacer,
 linear_topology,
)

# One H leaves each backbone carbon; the two carbons become a C-C bond. Atoms on
# the left that do not reappear on the right are the leaving groups.
ADDITION = "[C;%tail:1][H].[C;%head:2][H]>>[C:1][C:2]"

builder = PolymerBuilder(
 MonomerLibrary(library),
 mp.Reaction(ADDITION),
 typifier=typifier, # junctions are re-typed locally, as they form
 reach=2, # neighbourhood radius, in bonds, that decides one atom's type
 placer=ResiduePlacer(),
)

sz_chains = results["Schulz-Zimm"]
atomistic_chains = []
n_chains = 10 # truncated for this tutorial; use len(sz_chains) for a production run
for i, chain in enumerate(sz_chains[:n_chains]):
 atomistic_chains.append(builder.build(linear_topology(chain.monomers)))
 if (i + 1) % 5 == 0:
 print(f" built {i + 1}/{n_chains} chains...")

total_atoms = sum(len(c.atoms) for c in atomistic_chains)
print(f"built {len(atomistic_chains)} chains, total atoms: {total_atoms}")
```

## Packing and exporting follow the same pattern as earlier guides

The box size follows from total molecular weight and target density. Each chain is added to the packer as an individual target with count 1, and the packed frame is written as a LAMMPS data file together with the force field.

```python
# docs: skip — continues polydisperse guide (needs atomistic_chains from above)
from molpack import InsideBoxRestraint, Molpack, Target

total_mw = sum(
 sum(Element(a.get("element")).mass for a in c.atoms) for c in atomistic_chains
)
target_density = 0.05 # g/cm^3 (use ~1.0 for production)
volume = (total_mw / 6.022e23) / target_density * 1e24
box_length = volume ** (1 / 3)

box = InsideBoxRestraint([0.0, 0.0, 0.0], [box_length] * 3)
targets = [
 Target(chain.to_frame(), count=1).with_restraint(box)
 for chain in atomistic_chains
]
packed = Molpack().with_seed(42).pack(targets, max_loops=200)
packed.box = mp.Box.cubic(length=box_length)

mp.io.write_lammps_system("05_output/lammps", packed, ff)
print(f"packed: {packed['atoms'].nrows} atoms, box: {box_length:.1f} A")
```

## The engine assembles a runnable input script from the exported data

Writing the data file is only half the story. To actually run the simulation, LAMMPS needs an input script that says how to read that file, which force field styles to activate, and what protocol to follow. MolPy models this through `LAMMPSEngine`, which pairs a `Script` object with subprocess management.

**A `Script` is an editable, ordered list of lines** that can be built programmatically and saved to disk without executing anything. This separation matters: you can inspect, modify, and version-control the script before committing to a run. When you are ready, `engine.run()` writes the script to the working directory and launches `lmp -in input.lmp -log log.lammps -screen none`.

The code below builds a minimal OPLS-AA equilibration protocol for the packed system. The force field styles must match those written by `write_lammps_system`—`harmonic` bonds and angles, `opls` dihedrals, `lj/cut/coul/long` non-bonds—because LAMMPS validates style consistency when it reads the data file.

```
from molpy.core.script import Script
from molpy.engine import LAMMPSEngine

# Build the LAMMPS input script line-by-line.
# Script.from_text() dedents and normalises the block.
lmp_script = Script.from_text(
 name="input",
 language="other",
 text="""
 # Polydisperse PS/PMA system — generated by MolPy
 units real
 atom_style full

 read_data lammps.data
 include lammps.ff

 pair_style lj/cut/coul/long 12.0
 pair_modify mix arithmetic tail yes
 kspace_style pppm 1e-4

 bond_style harmonic
 angle_style harmonic
 dihedral_style opls
 improper_style cvff

 # Energy minimisation before dynamics
 minimize 1.0e-4 1.0e-6 10000 100000

 timestep 1.0
 thermo 1000
 thermo_style custom step temp press etotal

 # NVT equilibration at 300 K
 fix nvt all nvt temp 300.0 300.0 100.0
 run 100000
 """,
)

# Save the script alongside the data files without launching LAMMPS.
# check_executable=False lets the call succeed in notebooks where lmp
# may not be on PATH.
engine = LAMMPSEngine("lmp", check_executable=False)
script_path = lmp_script.save("05_output/lammps/input.lmp")
print("Input script written to:", script_path)
print(lmp_script.preview(max_lines=12))

# To run the simulation, replace the two lines above with:
# result = engine.run(lmp_script, workdir="05_output/lammps")
# print("Exit code:", result.returncode)
```

## The specification lives in code, not in a notation

Earlier versions of this guide ended with a single G-BigSMILES string encoding
the repeat units, their weights, the chain-length distribution and the target
system mass all at once. MolPy no longer parses BigSMILES, CGSmiles or
G-BigSMILES — those grammars were removed along with the Lark stack.

Nothing is lost, because every part of that string already appears above as
ordinary Python: the repeat units are the `MONOMERS` SMILES, the weights are
the composition passed to the builder, the distribution is the `numpy` sampling
step, and the target mass is the number of chains you choose to build. A string
that encodes all four is shorter to type and harder to inspect, debug or
parameterise; the code above is the specification.

## Troubleshooting

| Step | Check |
|------|-------|
| Monomer mass wrong | Verify monomer has explicit hydrogens before mass calculation |
| SystemPlanner total mass off | Check `max_rel_error` setting |
| Chain topology missing | Call `get_topo(gen_angle=True, gen_dihe=True)` before building |
| `MonomerLibrary` rejects a monomer | Every template needs at least one atom carrying `fields.SITE` |
| Reaction matches nothing | The `%name` in the SMARTS must equal the `fields.SITE` value you set |
| Packing fails | Lower target density or increase `max_steps` |

See also: [Assembly](02_assembly.md), [Force Field Typification](06_typifier.md).
