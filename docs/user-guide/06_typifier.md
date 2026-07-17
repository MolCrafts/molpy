[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/06_typifier.ipynb)

# Force Field Typification

Typification is the bridge between chemistry and parameters: SMARTS patterns from the force field decide which type every atom, bond, angle, and dihedral gets.

!!! note "Prerequisites"
    This guide requires RDKit for 3D coordinate generation. Typification itself does not require RDKit.

## The problem typification solves

A molecular structure has atoms and bonds, but a simulation needs *types* — identifiers that map each atom, bond, angle, and dihedral to specific force field parameters. The same carbon atom might be `opls_135` (aliphatic CH₃) or `opls_145` (aromatic ring carbon) depending on its chemical environment. Getting this assignment wrong silently produces wrong physics.

**A Typifier examines the chemical environment of each atom via SMARTS pattern matching and assigns the corresponding force field type.**

MolPy's `OplsTypifier` handles the full assignment in one call: atom types first, then pair parameters, then bond/angle/dihedral types derived from the atom type assignments.

## What typification looks like end to end

The workflow is always the same: build the structure, load a force field, create a typifier, call `typify`.


```python
import molpy as mp
from molpy.typifier import OplsTypifier

# 1. Build the structure
mol = mp.parser.parse_molecule("CCO")
mol = mp.adapter.RDKitAdapter(mol).generate_3d(add_hydrogens=True, optimize=True)
mol = mol.get_topo(gen_angle=True, gen_dihe=True)

print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")
print(f"angles: {len(mol.angles)}, dihedrals: {len(mol.dihedrals)}")
```

```text
atoms: 9, bonds: 8
angles: 13, dihedrals: 12
```


Loading the force field is a separate step from building the structure because the force field is an independent object — it can be shared across multiple molecules, swapped for a different variant, or inspected before any typification takes place. Once the force field is in hand, the typifier is constructed with it and `typify` is called on the molecule.


```python
# 2. Load force field and typify
# "oplsaa.xml" is bundled with MolPy — no separate download needed
ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsTypifier(ff, strict_typing=True)

typed_mol = typifier.typify(mol)
```

```text
2026-06-30 21:11:37,176 - molpy.io.forcefield.xml - INFO - Using built-in force field: /Users/roykid/work/molcrafts/molpy/src/molpy/data/forcefield/oplsaa.xml
```


```text
2026-06-30 21:11:37,181 - molpy.io.forcefield.xml - INFO - Parsing force field: OPLS-AA v0.1.0


2026-06-30 21:11:37,181 - molpy.io.forcefield.xml - INFO - Combining rule: geometric


2026-06-30 21:11:37,187 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types


2026-06-30 21:11:37,188 - molpy.io.forcefield.xml - INFO - Parsed 307 bond types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,190 - molpy.io.forcefield.xml - INFO - Parsed 964 angle types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,192 - molpy.io.forcefield._rb_opls - WARNING - RB coefficients do not lie on the ideal 4-term OPLS manifold (C0+C1+C2+C3+C4 = 10.041600, expected ≈ 0). Conversion will preserve forces and relative energies exactly, but will introduce a constant energy offset of ΔE = 10.041600 kJ/mol. This does not affect MD simulations.


2026-06-30 21:11:37,195 - molpy.io.forcefield.xml - INFO - Parsed 1089 dihedral types (OPLS-AA with unit conversion)


2026-06-30 21:11:37,197 - molpy.io.forcefield.xml - INFO - Parsed 825 nonbonded parameters (OPLS-AA with unit conversion)


2026-06-30 21:11:37,197 - molpy.io.forcefield.xml - INFO - Parsed 825 atom types (by type)
```


`typify` modifies the structure in-place and returns it — atoms in the returned object carry a `type` key and associated parameters.


```python
# 3. Inspect results
for atom in typed_mol.atoms:
    element = atom.get("element", "?")
    atype = atom.get("type", "untyped")
    charge = atom.get("charge") or 0.0
    print(f"  {element:2s} -> {atype:15s}  q={charge:+.4f}")
```

```text
  C  -> opls_135         q=-0.1800
  C  -> opls_157         q=+0.1450
  O  -> opls_154         q=-0.6830
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_140         q=+0.0600
  H  -> opls_155         q=+0.4180
```


## How atom typing works

The typifier reads SMARTS patterns from the force field XML. Each pattern defines one atom type — for example, `[CX4;H3]` matches an sp3 carbon with three hydrogens (a methyl carbon). The typifier walks through all atoms, matches each one against the pattern library, and assigns the best-matching type.

When multiple patterns match, priority and override rules in the force field resolve the conflict. This layered matching handles complex cases like aromatic vs. aliphatic nitrogen without manual intervention.

When no pattern matches, MolPy does not perform implicit parameter estimation for unmatched environments. Instead, unmatched cases are explicitly reported, allowing users to inspect and extend the rule set as needed. This design separates parameter assignment from parameter development, ensuring that force field definitions remain transparent and reproducible.

With atom types in place, the typifier has everything it needs to derive bonded types mechanically — a process described in the next section.

## How bonded typing works

Once atom types are assigned, bonded interactions follow mechanically. A bond between atom types `CT` and `OH` maps to bond type `CT-OH`. The same logic extends to angles (three-type sequences) and dihedrals (four-type sequences). Wildcard types (`*`) in the force field act as fallbacks when no specific match exists. MolPy does not define a specific electrostatics model. Partial charges are either taken from predefined force fields (e.g., OPLS-style parameters) or assigned externally using established workflows. MolPy focuses on storing and propagating charge information once defined, rather than performing charge derivation.

## Strict vs. non-strict mode

In strict mode (`strict_typing=True`), any untyped atom raises an error immediately. This is the right default during development — it catches missing force field parameters before they become silent errors in production.

In non-strict mode (`strict_typing=False`), untyped atoms are silently skipped. Use this when you know some atoms will not match — for example, when using a general-purpose force field on a molecule with exotic functional groups.

## Every atom, bond, angle, and dihedral carries its assigned type

After typification, you can iterate over bonds, angles, and dihedrals to see their assigned types.


```python
# Bond types
for bond in typed_mol.bonds[:3]:
    i_sym = bond.itom.get("element")
    j_sym = bond.jtom.get("element")
    btype = bond.get("type", "untyped")
    print(f"  {i_sym}-{j_sym} -> {btype}")

# Angle types
for angle in typed_mol.angles[:3]:
    names = [a.get("type", "?") for a in angle.endpoints]
    atype = angle.get("type", "untyped")
    print(f"  {'-'.join(names)} -> {atype}")
```

```text
  C-C -> CT-CT
  C-O -> CT-OH
  C-H -> CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
  opls_157-opls_135-opls_140 -> CT-CT-HC
```


## A typed structure is ready for simulation export

A typed structure is ready for simulation export. Convert to a `Frame`, attach a box, and write to LAMMPS or GROMACS format.


```python
import numpy as np
from pathlib import Path

frame = typed_mol.to_frame()
frame.simbox = mp.Box.cubic(30.0)

# mol_id is not set by typifier — add it for LAMMPS full atom style
atoms = frame["atoms"]
if "mol_id" not in atoms:
    atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)

outdir = Path("06_output")
outdir.mkdir(exist_ok=True)

mp.io.write_lammps_system(outdir / "ethanol", frame, ff)

print(f"exported to {outdir}")
```

```text
exported to 06_output
```


The `write_lammps_system` convenience function automatically filters the force field to only include types present in the frame, and translates canonical field names (`charge`, `mol_id`) to LAMMPS-specific names (`q`, `mol`) via the formatter system.

## Incremental re-typification at polymer junctions

When a `molpy.Reaction` forms a new bond between two monomers, the atoms at the junction change their chemical environment. The old atom types, bond types, angle types, and dihedral types at the junction become invalid.

Rather than re-typifying the entire chain after each coupling step, MolPy re-types only the neighbourhood the edit disturbed.

How wide that neighbourhood is is not a knob on the typifier. `GraphAssembler` is told the `reach` its typifier needs — the number of bonds it must see around an atom before it can name that atom's type — and that single number fixes both radii of the operation. `AffectedRegion.around` extracts a ball of `2 x reach` bonds around each new bond, and only the inner `reach` shell is written back: those are the atoms whose environment actually changed. The outer shell exists solely to give them a correct environment to be typed against. Atoms beyond the inner shell were already right and are left alone.

The region completes its own cut valences before any typifier sees it, and that is not a convenience. Because the extracted ball is exactly `interior_reach + reach` wide, an interior atom's receptive field reaches precisely to the boundary atoms — and a raw cut leaves those with unfilled valences, which a SMARTS matcher reads as radicals. Measured on p-xylene at `reach = 2`, 12 of its 19 raw slices cannot be typed at all.

Identical junctions hash to the same key and are typed once, so the number of typing passes tracks the number of *distinct* chemical environments in the system rather than the number of bonds formed. Building a 1000-mer costs about as many typing passes as building a 10-mer.

To enable this, pass the typifier to the builder at construction:

```text
from molpy.typifier import AmberToolsTypifier

builder = PolymerBuilder(
    MonomerLibrary({"EO": eo}),
    mp.Reaction(ETHER),
    typifier=AmberToolsTypifier(amber),
    reach=2,                     # GAFF: a 1-2 bond environment names an atom type
    placer=ResiduePlacer(),
)
chain = builder.build("{[#EO]|20}")
```

Omit the typifier and assembly assigns no types at all; typify the finished chain instead. That gives the same answer, at a cost proportional to the whole chain rather than to its junctions.

## When standard force fields are not enough

Standard OPLS-AA covers common organic functional groups. Specialized molecules — ionic liquids (TFSI), metal complexes, reactive intermediates — often need custom force field parameters. In those cases:

1. Use a specialized force field XML that includes the required SMARTS patterns and types
2. Or drop to the [Force Field](../tutorials/04_force_field.md) layer and define types manually

The typifier itself is agnostic to the force field content. It only needs SMARTS patterns and type definitions in the XML. If those are present, it will match them.

See also: [Force Field](../tutorials/04_force_field.md), [Stepwise Polymer Construction](02_assembly.md).
