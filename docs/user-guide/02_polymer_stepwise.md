[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/02_polymer_stepwise.ipynb)

# Stepwise Polymer Construction

This guide examines polymer construction at the reaction level. It shows which atoms are selected, which bond is formed, which atoms are removed, and why topology and force-field types must be regenerated after each coupling step. The workflow is first carried out explicitly with `Atomistic` operations — `merge`, `def_bond`, `del_atom`, `get_topo`, `typify` — and then repeated through `Reacter` to show the same logic in packaged form.

!!! note "Prerequisites"
    This guide requires RDKit (for `generate_3d`) and the `oplsaa.xml` force field file.

The monomer is **ethylene oxide (EO)**, represented in BigSMILES as `{[][<]OCCOCCOCCO[>][]}`. Each EO unit carries two reactive port markers: `<` on the left oxygen and `>` on the right oxygen. They are not separate atoms — they annotate existing atoms to signal where coupling occurs.

---

## Part 1 — Coupling two monomers by hand

### Prepare and inspect both monomers

Each monomer needs 3D coordinates with explicit hydrogens, a complete topology, and OPLS-AA types before any coupling can happen.


```python
import molpy as mp
import numpy as np
from pathlib import Path
from molpy.reacter import find_neighbors, find_port
from molpy.typifier import OplsTypifier

output_dir = Path("02_output")
output_dir.mkdir(exist_ok=True)

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=False)

MONOMER_BIGSMILES = "{[][<]OCCOCCOCCO[>][]}"


def make_eo_monomer():
    m = mp.parser.parse_monomer(MONOMER_BIGSMILES)
    m = mp.adapter.generate_3d(m, add_hydrogens=True, optimize=True)
    m = m.get_topo(gen_angle=True, gen_dihe=True)
    m = typifier.typify(m)
    return m


mon_a = make_eo_monomer()
mon_b = make_eo_monomer()
mon_b.move([10.0, 0.0, 0.0])  # displace so coordinates don't overlap after merging

print(f"monomer: {len(mon_a.atoms)} atoms, {len(mon_a.bonds)} bonds")
print(f"         {len(mon_a.angles)} angles, {len(mon_a.dihedrals)} dihedrals")
print(f"untyped: {sum(1 for a in mon_a.atoms if a.get('type') is None)} atoms")
```

    2026-06-30 21:08:06,136 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml


    2026-06-30 21:08:06,140 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


    2026-06-30 21:08:06,140 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


    2026-06-30 21:08:06,146 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


    2026-06-30 21:08:06,147 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


    2026-06-30 21:08:06,149 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


    2026-06-30 21:08:06,151 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


    2026-06-30 21:08:06,153 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


    2026-06-30 21:08:06,155 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


    2026-06-30 21:08:06,155 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)


    monomer: 24 atoms, 23 bonds
             40 angles, 45 dihedrals
    untyped: 0 atoms


Export the monomer as the first checkpoint:


```python
def save_step(name: str, chain, ff, output_dir: Path) -> None:
    frame = chain.to_frame()
    atoms = frame["atoms"]
    atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)
    coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
    lo = coords.min(axis=0) - 5.0
    hi = coords.max(axis=0) + 5.0
    frame.box = mp.Box(matrix=hi - lo, origin=lo)
    mp.io.write_lammps_system(output_dir / name, frame, ff)
    print(f"  saved: {output_dir / name / (name + '.data')}")


save_step("peo1", mon_a, ff, output_dir)
```

      saved: 02_output/peo1/peo1.data


### Identify the four actors in the coupling reaction

Dehydration condensation removes one water molecule per bond formed. Before touching any topology, locate the four atoms involved:

1. **Anchor carbon** — the carbon on `mon_a`'s right end that will form the new C–O bond
2. **Port oxygen of mon_a** (`>`) — this is the hydroxyl oxygen; it leaves as part of –OH
3. **Hydroxyl hydrogen** — the H attached to that port oxygen; leaves with it
4. **Leaving hydrogen of mon_b** — one H on `mon_b`'s left port oxygen (`<`); leaves as the second half of water


```python
# Right port of mon_a and left port of mon_b
port_r = find_port(mon_a, ">")
port_l = find_port(mon_b, "<")

# The anchor carbon is the C neighbour of the right port oxygen
anchor_C = [a for a in find_neighbors(mon_a, port_r) if a.get("element") == "C"][0]

# The leaving –OH: the port oxygen and its single hydrogen
leaving_OH_oxygen = port_r
leaving_OH_hydrogen = find_neighbors(mon_a, leaving_OH_oxygen, element="H")[0]

# The leaving H: one hydrogen on mon_b's left port
leaving_H_b = find_neighbors(mon_b, port_l, element="H")[0]

print(f"new bond will form:  {anchor_C.get('element')} — {port_l.get('element')}")
print(
    f"leaving from mon_a:  {leaving_OH_oxygen.get('element')} + {leaving_OH_hydrogen.get('element')}  (= OH)"
)
print(f"leaving from mon_b:  {leaving_H_b.get('element')}  (= H)")
print(f"net removal:         H₂O")
```

    new bond will form:  C — O
    leaving from mon_a:  O + H  (= OH)
    leaving from mon_b:  H  (= H)
    net removal:         H₂O


### Merge, form the bond, remove leaving atoms

Three Atomistic operations execute the coupling:


```python
# 1. Merge both monomers into a single Atomistic (atoms and bonds from both are combined)
dimer = mon_a.merge(mon_b)
print(
    f"after merge: {len(dimer.atoms)} atoms  (= {len(mon_a.atoms)} + {len(mon_b.atoms)})"
)

# 2. Form the new C–O bond between the anchor carbon and mon_b's port oxygen
new_bond = dimer.def_bond(anchor_C, port_l)
print(
    f"new bond: {new_bond.endpoints[0].get('element')}—{new_bond.endpoints[1].get('element')}"
)

# 3. Remove the three leaving atoms; their incident bonds are dropped automatically
dimer.del_atom(leaving_OH_oxygen, leaving_OH_hydrogen, leaving_H_b)
print(
    f"after removal: {len(dimer.atoms)} atoms  (= {len(dimer.atoms)} = {len(mon_a.atoms) + len(mon_b.atoms)} − 3)"
)
```

    after merge: 48 atoms  (= 48 + 24)
    new bond: C—O
    after removal: 45 atoms  (= 45 = 69 − 3)


`del_atom` drops all bonds, angles, and dihedrals that reference the removed atoms. The new C–O bond exists, but the topology around it is incomplete.

### Regenerate topology and re-type the junction

The new bond creates three-body and four-body paths that did not exist in either monomer. `get_topo` enumerates them; `typify` assigns force field types to everything it finds.


```python
# Check what is unresolved at the junction before fixing
print(
    f"untyped bonds before get_topo:      {sum(1 for b in dimer.bonds if b.get('type') is None)}"
)
print(
    f"untyped angles before get_topo:     {sum(1 for a in dimer.angles if a.get('type') is None)}"
)
print(
    f"untyped dihedrals before get_topo:  {sum(1 for d in dimer.dihedrals if d.get('type') is None)}"
)

angles_before = len(dimer.angles)
dihe_before = len(dimer.dihedrals)
dimer = dimer.get_topo(gen_angle=True, gen_dihe=True)

print(
    f"\nangles:    {angles_before} → {len(dimer.angles)}  (+{len(dimer.angles) - angles_before} cross-junction)"
)
print(
    f"dihedrals: {dihe_before} → {len(dimer.dihedrals)}  (+{len(dimer.dihedrals) - dihe_before} cross-junction)"
)

dimer = typifier.typify(dimer)

print(f"\nuntyped bonds:      {sum(1 for b in dimer.bonds if b.get('type') is None)}")
print(f"untyped angles:     {sum(1 for a in dimer.angles if a.get('type') is None)}")
print(f"untyped dihedrals:  {sum(1 for d in dimer.dihedrals if d.get('type') is None)}")
```

    untyped bonds before get_topo:      1
    untyped angles before get_topo:     0
    untyped dihedrals before get_topo:  0

    angles:    75 → 79  (+4 cross-junction)
    dihedrals: 81 → 90  (+9 cross-junction)



    untyped bonds:      0
    untyped angles:     0
    untyped dihedrals:  0


Every interaction is now typed. Export the dimer:


```python
save_step("peo2", dimer, ff, output_dir)
```

      saved: 02_output/peo2/peo2.data


### The same five operations build a chain of any length

The pattern — `merge` → `def_bond` → `del_atom` → `get_topo` → `typify` — can be repeated identically for every subsequent unit. A loop drives it from the dimer to a pentamer, writing a snapshot after each step.


```python
chain = make_eo_monomer()
save_step("peo1", chain, ff, output_dir)

for i in range(1, 5):
    unit = make_eo_monomer()
    unit.move([10.0 * i, 0.0, 0.0])

    # Identify reactive atoms on the current chain end and the incoming unit
    p_r = find_port(chain, ">")
    p_l = find_port(unit, "<")
    anc = [a for a in find_neighbors(chain, p_r) if a.get("element") == "C"][0]
    l_O = p_r
    l_H1 = find_neighbors(chain, l_O, element="H")[0]
    l_H2 = find_neighbors(unit, p_l, element="H")[0]

    # Couple
    chain = chain.merge(unit)
    chain.def_bond(anc, p_l)
    chain.del_atom(l_O, l_H1, l_H2)
    chain = chain.get_topo(gen_angle=True, gen_dihe=True)
    chain = typifier.typify(chain)

    name = f"peo{i + 1}"
    save_step(name, chain, ff, output_dir)
    print(f"{name}: {len(chain.atoms)} atoms, {len(chain.bonds)} bonds")
```

      saved: 02_output/peo1/peo1.data


      saved: 02_output/peo2/peo2.data
    peo2: 45 atoms, 44 bonds


      saved: 02_output/peo3/peo3.data
    peo3: 66 atoms, 65 bonds


      saved: 02_output/peo4/peo4.data
    peo4: 87 atoms, 86 bonds


      saved: 02_output/peo5/peo5.data
    peo5: 108 atoms, 107 bonds


After this loop `02_output/` contains five data files — `peo1.data` through `peo5.data` — each a valid LAMMPS input at a different chain length.

---

## Part 2 — The same coupling, automated by Reacter

Part 1 made every decision explicit. The five-step sequence it used — identify actors, merge, def_bond, del_atom, get_topo, typify — is exactly what `Reacter` encodes as a reusable rule. You write the selection logic once; `rxn.run` handles the rest.

### Define the reaction rule

A `Reacter` needs four selector functions: one to find the anchor atom on each side, and one to find the leaving atoms on each side.


```python
from molpy.core.atomistic import Atom, Atomistic
from molpy.reacter import (
    Reacter,
    form_single_bond,
    select_hydrogens,
    select_neighbor,
    select_self,
)


def select_hydroxyl_group(struct: Atomistic, reaction_site: Atom) -> list[Atom]:
    """Return [O, H] — the hydroxyl leaving group on the anchor carbon."""
    for neighbor in find_neighbors(struct, reaction_site):
        if neighbor.get("element") != "O":
            continue
        h_neighbors = find_neighbors(struct, neighbor, element="H")
        if h_neighbors:
            return [neighbor, h_neighbors[0]]
    raise ValueError("No hydroxyl group found near reaction site")


rxn = Reacter(
    name="dehydration",
    anchor_selector_left=select_neighbor("C"),  # same as: find the C next to port >
    anchor_selector_right=select_self,  # same as: the port < O itself
    leaving_selector_left=select_hydroxyl_group,  # same as: l_O + l_H1
    leaving_selector_right=select_hydrogens(1),  # same as: l_H2
    bond_former=form_single_bond,  # same as: def_bond(anc, p_l)
)
```

Every selector maps directly to an operation that Part 1 performed by hand.

### Build the same pentamer with one call per step


```python
chain = make_eo_monomer()
save_step("peo1_rxn", chain, ff, output_dir)

for i in range(1, 5):
    unit = make_eo_monomer()
    unit.move([10.0 * i, 0.0, 0.0])

    result = rxn.run(
        left=chain,
        right=unit,
        port_atom_L=find_port(chain, ">"),
        port_atom_R=find_port(unit, "<"),
        compute_topology=True,
    )
    chain = result.product
    chain = chain.get_topo(gen_angle=True, gen_dihe=True)
    chain = typifier.typify(chain)

    name = f"peo{i + 1}_rxn"
    save_step(name, chain, ff, output_dir)
    print(f"{name}: {len(chain.atoms)} atoms")
```

      saved: 02_output/peo1_rxn/peo1_rxn.data


      saved: 02_output/peo2_rxn/peo2_rxn.data
    peo2_rxn: 45 atoms


      saved: 02_output/peo3_rxn/peo3_rxn.data
    peo3_rxn: 66 atoms


      saved: 02_output/peo4_rxn/peo4_rxn.data
    peo4_rxn: 87 atoms


      saved: 02_output/peo5_rxn/peo5_rxn.data
    peo5_rxn: 108 atoms


`rxn.run` performs the merge, `def_bond`, and `del_atom` internally — the same three operations from Part 1, in the same order. The caller is still responsible for `get_topo` and `typify` after each step, because those depend on force field context that the reaction rule itself does not carry.

## Troubleshooting

**Port not found**

Check that port markers are present on the expected atoms:


```python
for a in mon_a.atoms:
    if a.get("port"):
        print(f"  element={a.get('element')}  port={a.get('port')}")
```

      element=O  port=<
      element=O  port=<
      element=O  port=>


**"No hydroxyl group found"** (Part 2 only)

Print the neighbours of the anchor carbon to see what the selector is searching through:


```python
p_r = find_port(mon_a, ">")
anc = [a for a in find_neighbors(mon_a, p_r) if a.get("element") == "C"][0]
for nb in find_neighbors(mon_a, anc):
    print(f"  element={nb.get('element')}")
```

      element=C
      element=O
      element=H
      element=H


**Untyped interactions after coupling (both parts)**

`get_topo` must come before `typify`, and both must come after the merge + bond formation + removal:

```python
chain = chain.merge(unit)
chain.def_bond(anc, p_l)
chain.del_atom(l_O, l_H1, l_H2)
chain.get_topo(gen_angle=True, gen_dihe=True)   # first
chain = typifier.typify(chain)                  # second
```

**Export fails: `mol_id` missing**

LAMMPS `full` atom style requires `mol_id`. The `save_step` helper sets it, but if you export manually:

```python
frame["atoms"]["mol_id"] = np.ones(frame["atoms"].nrows, dtype=int)
```

See also: [Topology-Driven Assembly](03_polymer_topology.md), [Crosslinked Networks](04_crosslinking.md).
