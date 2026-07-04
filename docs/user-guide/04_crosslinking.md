[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/04_crosslinking.ipynb)

# Crosslinked Networks

Everything LAMMPS `fix bond/react` needs — local reaction templates, a packed initial configuration, consistently numbered topology and force-field files — generated in one MolPy workflow.

!!! note "Prerequisites"
    This guide requires RDKit, Packmol, and the `oplsaa.xml` force field. Familiarity with [Stepwise Polymer Construction](02_polymer_stepwise.md) is assumed.

## Reactive MD requires four outputs

LAMMPS `fix bond/react` drives bond formation during MD. It needs reaction templates that capture the local topology before and after each bond formation event, a packed configuration at realistic density, force field coefficients that cover every type in both the initial configuration and the post-reaction templates, and an input script that sequences these ingredients through minimisation, equilibration, and reactive MD. MolPy's `BondReactReacter` generates the template files, `Packmol` handles packing, and `write_lammps_bond_react_system` exports the data, force field, and templates with unified type numbering in one call.

## Two monomers share one reaction template

We use two monomers: **EO2** (linear, two `$` ports) and **EO3** (branched, three `$` ports). Although the system contains two different species, it only needs **one** reaction template.

The reason is that `fix bond/react` matches templates by *local* topology, not by whole-molecule identity. The template only captures a few bond-hops around the reaction site (controlled by the `radius` parameter). Both EO2 and EO3 share the same arm chemistry — every reactive port sits at the end of a `...COCCO[$]` segment. When the BFS stops at `radius=4`, it has not yet reached the EO3 branching carbon, so the extracted subgraph is structurally identical regardless of whether the port belongs to an EO2 or an EO3 molecule.

```text
EO2:   HO–OCCOCCOCCO–OH           ← 2 identical arms
            [$]          [$]

EO3:        COCCO–OH               ← 3 identical arms
           /     [$]
     C–COCCO–OH
           \     [$]
            COCCO–OH
                  [$]

Template radius captures only the port neighbourhood:
     ...COCCO–OH   (same for every arm)
         ^^^^
         radius=4 stops here
```

If you had monomers with chemically *different* port environments (e.g. an amine reacting with an epoxide), you would need separate templates — one per distinct reaction type.


```python
from pathlib import Path
import numpy as np
import molpy as mp
from molpy.core.atomistic import Atom, Atomistic
from molpy.core.element import Element
from molpy.typifier import OplsTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=False)
```

```text
2026-07-01 03:07:38,585 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```


```text
2026-07-01 03:07:38,589 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-07-01 03:07:38,589 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-07-01 03:07:38,596 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-07-01 03:07:38,597 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,599 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,601 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-07-01 03:07:38,603 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-07-01 03:07:38,605 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-07-01 03:07:38,605 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)
```


Each monomer is parsed from BigSMILES, expanded to 3D with hydrogens, and typified with OPLS-AA. Atom IDs must be assigned before typification because the force field writer expects them.


```python
eo2 = mp.parser.parse_monomer("{[][$]OCCOCCOCCO[$][]}")
eo2 = mp.adapter.generate_3d(eo2, add_hydrogens=True, optimize=True)
eo2 = eo2.get_topo(gen_angle=True, gen_dihe=True)
for idx, atom in enumerate(eo2.atoms, start=1):
    atom["id"] = idx
eo2 = typifier.typify(eo2)
```


```python
eo3 = mp.parser.parse_monomer("{[]C(COCCO[$])(COCCO[$])COCCO[$][]}")
eo3 = mp.adapter.generate_3d(eo3, add_hydrogens=True, optimize=True)
eo3 = eo3.get_topo(gen_angle=True, gen_dihe=True)
for idx, atom in enumerate(eo3.atoms, start=1):
    atom["id"] = idx
eo3 = typifier.typify(eo3)
```

```text
/Users/roykid/work/molcrafts/molpy/src/molpy/adapter/rdkit.py:719: UserWarning: UFF optimization returned code 1. Code 1 typically means convergence not reached within 200 iterations. The structure may still be improved.
  warnings.warn(msg, UserWarning)
```


Verify that the port markers survived 3D generation:


```python
for label, mon in [("EO2", eo2), ("EO3", eo3)]:
    ports = [a.get("port") for a in mon.atoms if a.get("port")]
    print(f"{label}: atoms={len(mon.atoms)}, ports={ports}")
```

```text
EO2: atoms=24, ports=['$', '$']
EO3: atoms=38, ports=['$', '$', '$']
```


## Template generation captures the local reaction environment

`BondReactReacter` extends `Reacter` with subgraph extraction. It runs a representative reaction between two monomers, then extracts the local environment around each reaction site (controlled by `radius`) to produce pre-reaction and post-reaction molecule templates plus an atom equivalence map.

The `radius` parameter controls how many bond-hops the BFS extends from the anchor atoms. A value of 4 gives enough buffer so that all type-changed atoms are at least two bonds from the template edge — a requirement of LAMMPS `fix bond/react`.

The reaction follows **dehydration condensation**, the same chemistry as in [Stepwise Polymer Construction](02_polymer_stepwise.md). On the left side, the anchor is the carbon neighbor of the port atom, and the leaving group is the hydroxyl (OH). On the right side, the anchor is the port atom itself, and the leaving group is one hydrogen.


```python
from molpy.reacter import (
    BondReactReacter,
    find_neighbors,
    find_port,
    form_single_bond,
    select_hydrogens,
    select_hydroxyl_group,
    select_neighbor,
    select_self,
)
```

The typifier must be passed to `run()`. Without it, the newly formed C–O bond has no force field type and gets silently dropped from the post template — LAMMPS would then remove leaving-group atoms but never create the crosslink.


```python
reacter = BondReactReacter(
    name="rxn1",
    anchor_selector_left=select_neighbor("C"),
    anchor_selector_right=select_self,
    leaving_selector_left=select_hydroxyl_group,
    leaving_selector_right=select_hydrogens(1),
    bond_former=form_single_bond,
    radius=4,
)

left = eo2.copy()
right = eo2.copy()
result = reacter.run(
    left=left,
    right=right,
    port_atom_L=find_port(left, "$"),
    port_atom_R=find_port(right, "$"),
    compute_topology=True,
    typifier=typifier,
)
template = result.template
```

```text
2026-07-01 03:07:47,385 - molpy.reacter.bond_react - WARNING - Charge not conserved in bond/react template: sum(q_post) - sum(q_pre) = 0.278 e exceeds tolerance 1e-06 e.
```


The pre and post templates have the same number of atoms. Atoms in the leaving group are marked for deletion in the `.map` file — LAMMPS removes them after applying the post-template topology.


```python
print(f"pre:  {len(template.pre.atoms)} atoms")
print(f"post: {len(template.post.atoms)} atoms")
```

```text
pre:  23 atoms
post: 23 atoms
```


## Packing at realistic density

Compute the box size from total molecular weight and target density, then let Packmol find non-overlapping placements. 27 bifunctional + 9 trifunctional monomers give up to 81 reactive ports.


```python
from molpy.pack import InsideBoxConstraint, Packmol

N_EO2, N_EO3 = 27, 9
TARGET_DENSITY = 1.1  # g/cm³ (amorphous PEO ≈ 1.1–1.2)

total_mass_g = (
    N_EO2 * sum(Element(a.get("element")).mass for a in eo2.atoms)
    + N_EO3 * sum(Element(a.get("element")).mass for a in eo3.atoms)
) / 6.022e23
box_length = ((total_mass_g / TARGET_DENSITY) * 1e24) ** (1 / 3)
```


```python
packer = Packmol(workdir=Path("04_output/packmol"))
constraint = InsideBoxConstraint(
    length=np.array([box_length] * 3),
    origin=np.zeros(3),
)
packer.def_target(eo2.to_frame(), number=N_EO2, constraint=constraint)
packer.def_target(eo3.to_frame(), number=N_EO3, constraint=constraint)

packed = packer(max_steps=20000, seed=42)
packed.box = mp.Box.cubic(length=box_length)
print(f"packed: {packed['atoms'].nrows} atoms in {box_length:.1f} Å box")
```

```text
packed: 990 atoms in 21.1 Å box
```


## One call exports data, force field, and templates together

`write_lammps_bond_react_system` collects all atom, bond, angle, and dihedral types across the packed frame *and* every template, builds a unified type map, and writes everything to one directory. This guarantees that the numeric type IDs in `04.data`, `04.ff`, `rxn1_pre.mol`, and `rxn1_post.mol` are mutually consistent.

### Template validity guarantees

`BondReactReacter` validates every generated template against the REACTER
protocol (Gissinger et al., *Polymer* **128** (2017) 211–217,
DOI: 10.1016/j.polymer.2017.06.038; Gissinger et al., *Macromolecules* **53**
(2020) 9953–9961, DOI: 10.1021/acs.macromol.0c02012) and the LAMMPS
[`fix bond/react`](https://docs.lammps.org/fix_bond_react.html) contract:

- **Equivalences** are a bijection between pre and post template atoms.
- **InitiatorIDs** contains exactly 2 atoms (the bond-forming anchors), written
  in deterministic order (left anchor first). An anchor sitting on the template
  boundary raises a `ValueError` suggesting a larger `radius`.
- **Edge atoms** must be identical pre vs post in `type` and `charge` —
  a mismatch (typically caused by the typifier re-typing atoms near the
  reaction site) raises a `ValueError`; increase `radius` so the retyped
  shell lies inside the template.
- **Impropers propagate** into the post template, so sp2 centers (amides,
  vinyls, aromatic rings) keep their planarity terms after the reaction.
- **Total charge** (elementary charge units) is checked for conservation
  within `CHARGE_CONSERVATION_TOL = 1e-6` e; violations log a warning.

Known limitations: the optional `Constraints` and `ChiralIDs` map sections
are not generated.



```python
from pathlib import Path

output_dir = Path("04_output")
output_dir.mkdir(exist_ok=True)

mp.io.write_lammps_bond_react_system(
    output_dir,
    packed,
    ff,
    templates={"rxn1": template},
)
```

The directory now contains the packed system configuration (`04.data`), force field coefficients covering every type that appears in the initial state or either template (`04.ff`), the pre-reaction and post-reaction molecule templates (`rxn1_pre.mol` and `rxn1_post.mol`), and the atom equivalence, edge, and delete ID map (`rxn1.map`).

## The LAMMPS input script runs a five-stage protocol

The simulation progresses through five stages in sequence. Stage 1 removes steric clashes from the packed configuration with conjugate-gradient energy minimisation. Stage 2 heats the system to 300 K under NVT for 5 ps so that kinetic energy is distributed before the box is allowed to relax. Stage 3 switches to NPT at 1 atm for another 5 ps to bring the density to its equilibrium value. Stage 4 activates `fix bond/react`: the `stabilization yes` keyword places freshly reacted atoms into a separate thermostat group for a brief settling period, and `molecule inter` restricts reactions to atoms on different molecules, preventing intramolecular ring closure. Stage 5 cleans up, recalculates molecule IDs from the new bond topology, and writes the final snapshot. The `run` lengths in the script below are deliberately shortened to a few steps so this page builds in seconds — it produces a representative log skeleton, not an equilibrated network; use the per-stage durations above for a production run.



```python
lammps_script = """\
# ====================================================================
# PEO crosslinked network – OPLS-AA / fix bond/react
# ====================================================================
units           real
atom_style      full
boundary        p p p

read_data       04_output.data
include         04_output.ff

# -- Long-range electrostatics --
kspace_style    pppm 1.0e-4

# -- OPLS-AA 1-4 scaling --
special_bonds   lj 0.0 0.0 0.5 coul 0.0 0.0 0.5

neighbor        2.0 bin
neigh_modify    every 1 delay 0 check yes

# ====================================================================
# Stage 1: Energy minimisation (remove packing overlaps)
# ====================================================================
thermo          100
thermo_style    custom step temp pe ke etotal press vol density
min_style       cg
minimize        1.0e-4 1.0e-6 2000 20000

reset_timestep  0

# ====================================================================
# NOTE: run lengths are abbreviated (a few steps each) so this doc page
# renders quickly; use 5 / 5 / 10+ ps for a real crosslinking run.
# Stage 2: NVT equilibration (300 K)
# ====================================================================
variable step equal "step"
variable T    equal "temp"
variable rho  equal "density"
variable rxn1 equal "f_rxns[1]"

fix out all print 100 "${step} ${T} ${rho}" &
    file thermo_rxn.dat screen no &
    title "# step temp density"
velocity        all create 300.0 12345 dist gaussian

fix             nvt_eq all nvt temp 300.0 300.0 100.0
timestep        1.0
run             200
unfix           nvt_eq

# ====================================================================
# Stage 3: NPT equilibration (300 K, 1 atm)
# ====================================================================
fix             equil all npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0
run             200
run             200
run             100
unfix           equil

# ====================================================================
# Stage 4: Reactive MD with fix bond/react (300 K NPT)
# ====================================================================
molecule        rxn1_pre rxn1_pre.mol
molecule        rxn1_post rxn1_post.mol

# bond/react: attempt every step, 5 Å cutoff, intermolecular only
fix             rxns all bond/react stabilization yes npt_grp 0.03 &
                react rxn1 all 1 0.0 5.0 rxn1_pre rxn1_post rxn1.map prob 0.01 1234

fix             npt_grp_react npt_grp_REACT npt temp 300.0 300.0 100.0 iso 1.0 1.0 1000.0

# -- Atom trajectory dump (OVITO-compatible) --
dump            traj all custom 200 traj.lammpstrj &
                id mol type x y z vx vy vz
dump_modify     traj sort id

# -- Bond topology dump (OVITO: load as topology) --
compute         bnd all property/local batom1 batom2 btype
dump            bonds all local 200 bonds.dump c_bnd[1] c_bnd[2] c_bnd[3]
dump_modify     bonds colname 1 batom1 colname 2 batom2 colname 3 btype

# -- Thermo includes reaction count from fix bond/react --
thermo          200
thermo_style    custom step temp pe ke etotal press density f_rxns[1]
run             1000

# ====================================================================
# Stage 5: Write final state
# ====================================================================
write_data      final.data
"""
```

The `reset_mol_ids all` command at the end recalculates molecule IDs from the current bond topology. If crosslinking succeeded, the original 36 molecules collapse into a small number of connected components.

`LAMMPSEngine` wraps the script into a `Script` object and runs `lmp_serial` in the output directory where all data and template files already sit.


```python
from molpy.engine import LAMMPSEngine

script = mp.Script.from_text("run", lammps_script, language="other")
script.tags.add("input")

engine = LAMMPSEngine(executable="lmp_serial", check_executable=False)
proc = engine.run(
    script,
    workdir=output_dir,
    capture_output=True,
    check=False,
    timeout=600,
)

for line in proc.stdout.splitlines()[-20:]:
    print(line)
assert proc.returncode == 0, f"LAMMPS failed:\n{proc.stderr or proc.stdout[-1000:]}"
```

!!! tip "Visualising in OVITO"
    Load `traj.lammpstrj` as the main file, then load `bonds.dump` as a topology overlay. The bond dump uses `batom1`/`batom2`/`btype` columns that OVITO recognises directly. For more detail please see the [OVITO manual](https://www.ovito.org/manual/reference/file_formats/input/lammps_dump_local.html).

## Verifying the output

Read back the final snapshot and check that crosslinks actually formed. The `reset_mol_ids` command at the end of the script reassigns molecule IDs based on bond connectivity — if the count dropped from 36, networks have formed.


```python
final = mp.io.read_lammps_data(output_dir / "final.data", atom_style="full")
n_atoms_final = final["atoms"].nrows
n_mols_final = len(set(final["atoms"]["mol_id"]))
print(f"final: {n_atoms_final} atoms, {n_mols_final} molecules (started with 36)")
```

```text
final: 891 atoms, 3 molecules (started with 36)
```


## Troubleshooting

**Template generation fails**

Print the selected site and leaving-group atoms to see what the selectors found:


```python
site = find_port(left, "$")
carbon = [a for a in find_neighbors(left, site) if a.get("element") == "C"][0]
print(f"site: {carbon.get('element')} name={carbon.get('name')}")
for nb in find_neighbors(left, carbon):
    print(f"  neighbor: {nb.get('element')} name={nb.get('name')}")
```

```text
site: C name=None
  neighbor: O name=None
  neighbor: C name=None
  neighbor: H name=None
  neighbor: H name=None
```


**Packing fails**

Reduce the target density or monomer count. Packmol needs enough room to place molecules without overlap.

**LAMMPS: "Atom type affected by reaction is too close to template edge"**

Increase the `radius` parameter in `BondReactReacter`. A larger radius captures more of the local environment, pushing the template boundary further from type-changed atoms.

**LAMMPS: reactions fire but molecule count stays constant**

The post template is missing the new crosslinking bond. This happens when the typifier is not passed to `reacter.run()` — the new bond has no force field type and gets silently dropped during export. Always pass `typifier=typifier`.

**LAMMPS rejects templates**

Check that the `.map` file contains valid equivalence IDs:


```python
print((output_dir / "rxn1.map").read_text()[:500])
```

```text
# auto-generated map file for fix bond/react

23 equivalences
2 edgeIDs
3 deleteIDs

InitiatorIDs

2
14

EdgeIDs

6
18

DeleteIDs

1
7
19

Equivalences

1   1
2   2
3   3
4   4
5   5
6   6
7   7
8   8
9   9
10   10
11   11
12   12
13   13
14   14
15   15
16   16
17   17
18   18
19   19
20   20
21   21
22   22
23   23
```



See also: [Topology-Driven Assembly](03_polymer_topology.md), [Polydisperse Systems](05_polydisperse_systems.md).
