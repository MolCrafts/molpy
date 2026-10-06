# Polydisperse Systems

From a target molecular-weight distribution to a packed, LAMMPS-ready box: sample the chains, assemble each one, type it, pack, export.

!!! note "Prerequisites"
    This guide requires `molcrafts-molpack` for the packing step. Familiarity with [Assembly](02_assembly.md) is assumed.

## From distribution to simulation box

Most polymer samples are polydisperse rather than monodisperse. This workflow starts from a target molecular-weight distribution and proceeds through explicit sampling, chain assembly, and packing to produce a simulation-ready system.

## Each monomer is a CGsmiles fragment with two ports

A styrene / methyl-acrylate copolymer grows by radical addition: each junction is a new C–C bond between the backbone carbons of neighbouring units, and one hydrogen leaves each side. That is exactly what a port expresses. Each unit is written as a CGsmiles fragment whose bonding descriptors sit on the two backbone carbons (`[<]` on one, `[>]` on the other), so `<` of one unit joins `>` of the next, head to tail. `mp.Conformer` embeds each fragment in 3D with hydrogens; the hydrogen on each port is its leaving atom.

Two methyl caps, `HEAD` and `TAIL`, close the chain ends. Because every handle leaves when its port bonds, the mass a unit adds to a chain is its template mass minus its handles — the numbers the planner needs below.

```
import molpy as mp
from molpy import Element

# One CGsmiles fragment per unit; its bonding descriptors are its ports.
UNITS = {
    "Sty": "[<]CC(c1ccccc1)[>]",  # -CH2-CH(Ph)-
    "MA": "[<]CC(C(=O)OC)[>]",  # -CH2-CH(COOCH3)-
    "HEAD": "C[>]",  # CH3- : starts a chain
    "TAIL": "[<]C",  # -CH3 : ends a chain
}

conformer = mp.Conformer(seed=42)
library = {
    name: conformer.generate(
        mp.io.SmilesIR.from_fragment(body).to_template()
    )[0]
    for name, body in UNITS.items()
}


def mass(atoms):
    return sum(Element(a.get("element")).mass for a in atoms)


# The mass a unit adds to a chain: its template minus the handles that leave.
unit_mass = {
    name: mass(unit.atoms) - mass(port.handle_atom for port in unit.ports)
    for name, unit in library.items()
}
monomer_mass = {name: unit_mass[name] for name in ("Sty", "MA")}
end_group_mass = unit_mass["HEAD"] + unit_mass["TAIL"]

for name, unit in library.items():
    print(
        f"{name:4s}: atoms={unit.n_atoms:2d}, ports={unit.n_ports}, "
        f"adds {unit_mass[name]:.2f} g/mol"
    )
```

## Sampling draws chain lengths from a statistical distribution

The sampling layer has three components that compose cleanly. `WeightedSequenceGenerator` controls the monomer mole ratio (80:20 here). `PolydisperseChainGenerator` draws a degree of polymerization or mass from the chosen distribution for each chain. `SystemPlanner` accumulates chains until a target total mass is reached, stopping when the accumulated mass is within `max_rel_error` of the target. Four distributions are demonstrated below so that their shape differences become visible in the next section.

The monomer and end-group masses come from the unit library above, so a planned chain's mass is the mass of the chain assembled from it.

```
import numpy as np
from molpy.builder import (
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
        end_group_mass=end_group_mass,
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
CLR_HIST = "#6baed6"  # steel blue – sampled histogram
CLR_EDGE = "#3182bd"  # darker blue – histogram edge
CLR_THEO = "#e6550d"  # orange – theoretical curve
CLR_MN = "#31a354"  # green – Mn line
CLR_MW = "#de2d26"  # red – Mw line
CLR_BOX = "#f7f7f7"  # near-white – annotation box


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

## Each planned chain becomes a site graph

A planned `Chain` carries its monomer sequence. Written as a CGsmiles string between the two caps — `{[#HEAD][#Sty][#Sty][#MA]...[#TAIL]}` — it is the chain's topology: `to_coarsegrain()` turns it into a site graph with one site per unit and one bond per junction.

`mp.builder.Assembler` places one copy of `library[bead_type]` per site and joins one port of each neighbour per bond, removing the two leaving hydrogens. `GrowthPlacer` needs no site positions: it grows the chain from its first unit, setting each copy's anchor where its parent's leaving hydrogen was. One assembler serves every sequence the planner sampled, because both monomers and both caps come from one library. Every port is consumed, so the chains have no open ports left.

```
assembler = mp.builder.Assembler(library, mp.builder.GrowthPlacer())

sz_chains = results["Schulz-Zimm"]
n_chains = 5  # a few chains for this guide; use len(sz_chains) for a production run
atomistic_chains = []
for chain in sz_chains[:n_chains]:
    notation = "{[#HEAD]" + "".join(f"[#{m}]" for m in chain.monomers) + "[#TAIL]}"
    sites = mp.io.CGSmilesIR(notation).to_coarsegrain()
    atomistic_chains.append(assembler.assemble(sites, mp.Atomistic))

for chain, built in zip(sz_chains, atomistic_chains):
    print(
        f"dp={chain.dp:3d}  atoms={built.n_atoms:4d}  open ports={built.n_ports}  "
        f"planned {chain.mass:.1f} g/mol, built {mass(built.atoms):.1f} g/mol"
    )
```

## The assembled chains are typed as whole molecules

Assembly assigns no force-field types. Each finished chain is an ordinary `mp.Atomistic`, so it is typed like any other molecule; the typifier's `forcefield()` then holds the parameters it assigned, ready for export.

```
from molpy.ff.typifier import OPLSAATypifier

typifier = OPLSAATypifier(strict=True)
typed_chains = [typifier.typify(chain) for chain in atomistic_chains]
ff = typifier.forcefield()
print(f"typed {len(typed_chains)} chains, {sum(c.n_atoms for c in typed_chains)} atoms")
```

## Packing and exporting follow the same pattern as earlier guides

The box size follows from total molecular weight and target density. Each chain is added to the packer as an individual target with count 1; `mp.builder.PackingTemplate` hands the packer the chain's frame together with its `hydrogens` indices, so the hydrogens are named once and given a small packing radius. The packed frame is written as a LAMMPS data file together with the force field. The growth placer leaves bond lengths and close contacts to a later relaxation, which is why the LAMMPS script below minimises before any dynamics.

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
from molpack import GenCanPack, Target

total_mw = sum(mass(c.atoms) for c in typed_chains)
target_density = 0.05  # g/cm^3 (use ~1.0 for production)
volume = (total_mw / 6.022e23) / target_density * 1e24
box_length = volume ** (1 / 3)

box = mp.Cuboid([0.0, 0.0, 0.0], [box_length] * 3)
targets = []
for chain in typed_chains:
    template = mp.builder.PackingTemplate(chain)  # frame + perceived atom roles
    targets.append(
        Target(template.frame, count=1)
        .with_restraint(box)
        .with_hydrogens(template.hydrogens)
        .with_atom_radius(template.hydrogens, 0.2)  # H relaxes away in early MD
    )
packed = GenCanPack().with_seed(42).run(targets, max_loops=200).frame  # carries mol_id
packed.box = mp.Box.cube(length=box_length)

mp.io.write_lammps_data("05_output/system.data", packed)
mp.ff.forcefield.write_lammps_forcefield("05_output/system.ff", ff, packed)
print(f"packed: {packed['atoms'].nrows} atoms, box: {box_length:.1f} A")
```

## The engine assembles a runnable input script from the exported data

Writing the data file is only half the story. To actually run the simulation, LAMMPS needs an input script that says how to read that file, which force field styles to activate, and what protocol to follow. MolPy models this through `LAMMPSEngine`, which pairs a `Script` object with subprocess management.

**A `Script` is an editable, ordered list of lines** that can be built programmatically and saved to disk without executing anything. This separation matters: you can inspect, modify, and version-control the script before committing to a run. When you are ready, `engine.run()` writes the script to the working directory and launches `lmp -in input.lmp -log log.lammps -screen none`.

The code below builds a minimal equilibration protocol for the packed system. It reads the `system.data` and `system.ff` written above. The include already declares every style, the mixing rule (geometric, as OPLS-AA defines it) and `special_bonds`, so the script does not repeat them: a `pair_style` of another kind issued after the include would discard the pair coefficients it just set. For long-range electrostatics, write the include with `write_lammps_forcefield(..., skip_pair_style=True)` and declare `pair_style`, `pair_modify` and `special_bonds` in the script before `include`, as the [AmberTools guide](13_ambertools_integration.md) does.

```
from molpy.engine import Script
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

 read_data system.data
 # every style, the mixing rule, special_bonds and the coefficients
 include system.ff

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

# Save the script next to the data files without launching LAMMPS.
# check_executable=False lets the call succeed in notebooks where lmp
# may not be on PATH.
engine = LAMMPSEngine("lmp", check_executable=False)
script_path = lmp_script.save("05_output/input.lmp")
print("Input script written to:", script_path)
print(lmp_script.preview(max_lines=12))

# To run the simulation, replace the two lines above with:
# result = engine.run(lmp_script, workdir="05_output")
# print("Exit code:", result.returncode)
```

## The notation describes one chain; the ensemble is code

MolPy reads CGsmiles (`mp.io.CGSmilesIR`), and every chain above passed through it: the units are CGsmiles fragments and each sampled sequence is a CGsmiles string. What a CGsmiles string does not hold is the ensemble — the monomer weights, the chain-length distribution and the target system mass. Those stay ordinary Python objects (`WeightedSequenceGenerator`, a distribution, `SystemPlanner`), which are easier to inspect, debug and parameterise than a single string. BigSMILES and G-BigSMILES are not parsed.

## Troubleshooting

| Step | Check |
|------|-------|
| Planned and built masses differ | Derive `monomer_mass` and `end_group_mass` from the library: template mass minus port handles |
| SystemPlanner total mass off | Check `max_rel_error` setting |
| `assemble` raises "has no port for every bond" | Each monomer needs a `[<]` and a `[>]` descriptor; each cap needs the one its neighbour accepts |
| Close contacts after assembly | Expected: `GrowthPlacer` does not relax the chain; minimise before dynamics |
| Packing fails | Lower target density or increase `max_loops` |

See also: [Assembly](02_assembly.md), [Polymer Topologies](topology/index.md), [Force Field Typification](06_typifier.md).
