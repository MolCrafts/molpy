---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
---

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/06_typifier.ipynb)

# Force Field Typification

After reading this page you will be able to assign atom types, bond types, angle types, and dihedral types to a molecular structure using SMARTS-based pattern matching.

!!! note "Prerequisites"
    This guide requires RDKit for 3D coordinate generation. Typification itself does not require RDKit.

## The problem typification solves

A molecular structure has atoms and bonds, but a simulation needs *types* — identifiers that map each atom, bond, angle, and dihedral to specific force field parameters. The same carbon atom might be `opls_135` (aliphatic CH₃) or `opls_145` (aromatic ring carbon) depending on its chemical environment. Getting this assignment wrong silently produces wrong physics.

**A Typifier examines the chemical environment of each atom via SMARTS pattern matching and assigns the corresponding force field type.**

MolPy's `OplsAtomisticTypifier` handles the full assignment in one call: atom types first, then pair parameters, then bond/angle/dihedral types derived from the atom type assignments.


## What typification looks like end to end

The workflow is always the same: build the structure, load a force field, create a typifier, call `typify`.

```python
import molpy as mp
from molpy.typifier import OplsAtomisticTypifier

# 1. Build the structure
mol = mp.parser.parse_molecule("CCO")
mol = mp.tool.generate_3d(mol, add_hydrogens=True, optimize=True)
mol.get_topo(gen_angle=True, gen_dihe=True)

print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")
print(f"angles: {len(mol.angles)}, dihedrals: {len(mol.dihedrals)}")
```

Loading the force field is a separate step from building the structure because the force field is an independent object — it can be shared across multiple molecules, swapped for a different variant, or inspected before any typification takes place. Once the force field is in hand, the typifier is constructed with it and `typify` is called on the molecule.

```python
# 2. Load force field and typify
# "oplsaa.xml" is bundled with MolPy — no separate download needed
ff = mp.io.read_xml_forcefield("oplsaa.xml")
typifier = OplsAtomisticTypifier(ff, strict_typing=True)

typed_mol = typifier.typify(mol)
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


## A typed structure is ready for simulation export

A typed structure is ready for simulation export. Convert to a `Frame`, attach a box, and write to LAMMPS or GROMACS format.

```python
import numpy as np
from pathlib import Path

frame = typed_mol.to_frame()
frame.box = mp.Box.cubic(30.0)

# mol_id is not set by typifier — add it for LAMMPS full atom style
atoms = frame["atoms"]
if "mol_id" not in atoms:
    atoms["mol_id"] = np.ones(atoms.nrows, dtype=int)

outdir = Path("06_output")
outdir.mkdir(exist_ok=True)

mp.io.write_lammps_system(outdir / "ethanol", frame, ff)

print(f"exported to {outdir}")
```

The `write_lammps_system` convenience function automatically filters the force field to only include types present in the frame, and translates canonical field names (`charge`, `mol_id`) to LAMMPS-specific names (`q`, `mol`) via the formatter system.


## Incremental re-typification at polymer junctions

When a `Reacter` forms a new bond between two monomers, the atoms at the junction change their chemical environment. The old atom types, bond types, angle types, and dihedral types at the junction become invalid.

Rather than re-typifying the entire chain after each coupling step, MolPy performs **incremental re-typification** — it re-computes force field parameters only for the affected atoms and their neighbors.

The `Reacter` records exactly which atoms were modified in `ReactionResult.modified_atoms`. When a typifier is passed to `Reacter.run()`, the internal `_incremental_typify()` method runs a six-step pipeline on just those atoms:

1. Clear `type` on modified atoms
2. Re-run atom typing (SMARTS matching) on the full structure
3. Re-assign pair parameters (LJ sigma/epsilon) for modified atoms
4. Re-type new bonds and bonds touching modified atoms
5. Re-type new angles and angles touching modified atoms
6. Re-type new dihedrals and dihedrals touching modified atoms

For a 20-mer, each coupling step only re-types ~4 atoms and their neighbors — much faster than full-chain re-typification.

To enable this in `PolymerBuilder`, pass the typifier at construction:

```text
builder = PolymerBuilder(
    library={"EO": eo_typed},       # pre-typed monomer
    connector=connector,
    placer=placer,
    typifier=typifier,              # enables incremental re-typification
)
```

If you prefer to typify the entire chain at the end instead, simply omit the typifier from the builder and call `typifier.typify(result.polymer)` after building.


## When standard force fields are not enough

Standard OPLS-AA covers common organic functional groups. Specialized molecules — ionic liquids (TFSI), metal complexes, reactive intermediates — often need custom force field parameters. In those cases:

1. Use a specialized force field XML that includes the required SMARTS patterns and types
2. Or drop to the [Force Field](../tutorials/04_force_field.md) layer and define types manually

The typifier itself is agnostic to the force field content. It only needs SMARTS patterns and type definitions in the XML. If those are present, it will match them.

See also: [Force Field](../tutorials/04_force_field.md), [Stepwise Polymer Construction](02_polymer_stepwise.md).
