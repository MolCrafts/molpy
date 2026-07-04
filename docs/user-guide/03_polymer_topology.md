[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/03_polymer_topology.ipynb)

# Topology-Driven Assembly with CGSmiles

Change the CGSmiles string, keep everything else: one builder configuration yields linear chains, rings, and branched stars.

!!! note "Prerequisites"
    This guide requires RDKit (for `generate_3d`), the `oplsaa.xml` force field, and familiarity with [Stepwise Polymer Construction](02_polymer_stepwise.md).

## One builder, multiple architectures

In the stepwise guide, the reaction kernel was always the same — only the loop structure changed. CGSmiles pushes that idea further: the same builder configuration produces different architectures depending solely on the topology expression.

```text
linear:  {[#EO2]|4[#PS]}
ring:    {[#EO2]1[#PS][#EO2][#PS][#EO2]1}
branch:  {[#PS][#EO3]([#PS])[#PS]}
```

The builder does not change between these three products. Only the string changes.

## Parse the topology string before you build anything

A CGSmiles expression encodes both the monomer labels and the connectivity pattern between them as a graph. Validating that graph before any chemistry runs costs almost nothing, and it catches label typos and structural mistakes — missing ring-closure digits, wrong branching parentheses — before they surface as cryptic errors deep inside the builder.


```python
from molpy.parser import parse_cgsmiles

expressions = {
    "linear": "{[#EO2]|4[#PS]}",
    "ring": "{[#EO2]1[#PS][#EO2][#PS][#EO2]1}",
    "branch": "{[#PS][#EO3]([#PS])[#PS]}",
}

for name, expr in expressions.items():
    ir = parse_cgsmiles(expr)
    labels = [node.label for node in ir.base_graph.nodes]
    print(f"{name}: nodes={len(ir.base_graph.nodes)}, labels={labels}")
```

```text
linear: nodes=5, labels=['EO2', 'EO2', 'EO2', 'EO2', 'PS']
ring: nodes=5, labels=['EO2', 'PS', 'EO2', 'PS', 'EO2']
branch: nodes=4, labels=['PS', 'EO3', 'PS', 'PS']
```


With the graph validated, the next step is giving each node a physical structure.

## Every label in the topology string needs a molecular template

The parser produces a graph whose nodes carry labels like `EO2`, `EO3`, and `PS`. The builder resolves each label by looking it up in a library of typed `Atomistic` objects. If a label is present in the CGSmiles string but absent from the library, the build fails immediately. All templates use `$` as the reactive port marker.


```python
import molpy as mp
from molpy.typifier import OplsTypifier

ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=True)

BIGSMILES = {
    "EO2": "{[][$]OCCO[$][]}",
    "EO3": "{[][$]OCC(CO[$])(CO[$])[]}",
    "PS": "{[][$]OCC(c1ccccc1)CO[$][]}",
}


def build_monomer(bigsmiles, typifier):
    monomer = mp.parser.parse_monomer(bigsmiles)
    monomer = mp.adapter.generate_3d(monomer, add_hydrogens=True, optimize=False)
    monomer = monomer.get_topo(gen_angle=True, gen_dihe=True)
    monomer = typifier.typify(monomer)
    return monomer


library = {label: build_monomer(bs, typifier) for label, bs in BIGSMILES.items()}

for label, mon in library.items():
    ports = [a.get("port") for a in mon.atoms if a.get("port")]
    print(f"{label}: atoms={len(mon.atoms)}, ports={ports}")
```

```text
2026-06-30 21:08:23,619 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```


```text
2026-06-30 21:08:23,623 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-06-30 21:08:23,623 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-06-30 21:08:23,630 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-06-30 21:08:23,631 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,634 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,635 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-06-30 21:08:23,638 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-06-30 21:08:23,640 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-06-30 21:08:23,640 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)


EO2: atoms=10, ports=['$', '$']
EO3: atoms=17, ports=['$', '$', '$']
PS: atoms=29, ports=['$', '$']
```


With a complete library, the only remaining definition is the chemistry that connects one template to another.

## The reaction chemistry is the same regardless of topology

The reaction kernel here is identical to the dehydration condensation defined in the [Stepwise Polymer Construction](02_polymer_stepwise.md) guide. The `select_hydroxyl_group` function finds the -OH leaving group on the left monomer's reaction site; `select_one_hydrogen` picks one H from the right monomer's port atom. Together they remove -OH and -H, form a new C–O bond, and release water. None of that logic depends on whether the final product is a linear chain, a ring, or a branch — the topology is entirely the responsibility of the CGSmiles string.

??? note "Reaction setup (identical to the stepwise guide)"
    The `select_hydroxyl_group` function finds the -OH leaving group on the left monomer's reaction site. The `select_one_hydrogen` function picks one H from the right monomer's port atom. Together they implement dehydration condensation: -OH + -H are removed, forming a new C-O bond and releasing water.


```python
from molpy.core.atomistic import Atom, Atomistic
from molpy.builder.polymer import (
    Connector,
    CovalentSeparator,
    LinearOrienter,
    Placer,
    PolymerBuilder,
)
from molpy.reacter import (
    Reacter,
    find_neighbors,
    form_single_bond,
    select_neighbor,
    select_self,
)


def select_hydroxyl_group(struct: Atomistic, site: Atom) -> list[Atom]:
    for nb in find_neighbors(struct, site):
        if nb.get("element") != "O":
            continue
        hs = [a for a in find_neighbors(struct, nb, element="H")]
        if hs:
            return [nb, hs[0]]
    raise ValueError("No hydroxyl group found")


def select_one_hydrogen(struct: Atomistic, site: Atom) -> list[Atom]:
    hs = [a for a in find_neighbors(struct, site, element="H")]
    if not hs:
        raise ValueError("No hydrogen found")
    return [hs[0]]


rxn = Reacter(
    name="dehydration",
    anchor_selector_left=select_neighbor("C"),
    anchor_selector_right=select_self,
    leaving_selector_left=select_hydroxyl_group,
    leaving_selector_right=select_one_hydrogen,
    bond_former=form_single_bond,
)

rules = {(l, r): ("$", "$") for l in library for r in library}
connector = Connector(port_map=rules, reacter=rxn)
placer = Placer(
    separator=CovalentSeparator(buffer=-0.1),
    orienter=LinearOrienter(),
)

builder = PolymerBuilder(
    library=library,
    connector=connector,
    placer=placer,
    typifier=typifier,
)
```

The builder is now fully configured. The three expressions from the opening section are the only inputs that differ between the three products.

## The CGSmiles string alone determines the architecture

Passing the three expressions to the same builder produces three structurally distinct polymers. The linear expression encodes a chain; the ring expression encodes a cycle by repeating the ring-closure digit; the branch expression encodes a trifunctional junction by using parentheses. The builder resolves each graph edge as one call to the reaction kernel and one call to the placer — it has no separate mode for rings or branches.


```python
for name, expr in expressions.items():
    result = builder.build(expr)
    polymer = result.polymer
    print(f"{name}: atoms={len(polymer.atoms)}, steps={result.total_steps}")
```

```text
linear: atoms=57, steps=4
```


```text
ring: atoms=73, steps=5


branch: atoms=95, steps=3
```


The same builder, the same reaction, the same library — only the CGSmiles string changes. This is the key advantage of topology-driven assembly: new architectures do not require new code. Once a product exists in memory, writing it to disk follows a single pattern regardless of topology.

## Exporting each product to LAMMPS follows the same pattern

Each product can be exported using the same pattern.


```python
import numpy as np
from pathlib import Path

output_dir = Path("03_output")
output_dir.mkdir(exist_ok=True)

for name, expr in expressions.items():
    result = builder.build(expr)
    typed_polymer = typifier.typify(result.polymer)
    frame = typed_polymer.to_frame()
    atoms = frame["atoms"]
    if "mol_id" not in atoms:
        atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)

    coords = np.column_stack([atoms["x"], atoms["y"], atoms["z"]])
    lo = coords.min(axis=0) - 5.0
    hi = coords.max(axis=0) + 5.0
    frame.box = mp.Box(matrix=hi - lo, origin=lo)

    mp.io.write_lammps_system(output_dir / name, frame, ff)
```

## Troubleshooting

Debug in this order:

1. Parse CGSmiles and verify node/bond counts first
2. Confirm each label exists in the library
3. Confirm connector rules exist for each reacting label pair
4. Print selected site/leaving atoms if reaction fails
5. Tune `CovalentSeparator(buffer=...)` if geometry overlaps

See also: [Stepwise Polymer Construction](02_polymer_stepwise.md), [Crosslinked Networks](04_crosslinking.md).
