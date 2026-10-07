[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/06_typifier.ipynb)

# Force Field Typification

Typification is the bridge between chemistry and parameters: SMARTS patterns from the force field decide which type every atom, bond, angle, and dihedral gets.

!!! note "Prerequisites"
    Nothing beyond MolPy itself: the 3D coordinates come from the native `mp.Conformer`, and the OPLS-AA rules and parameters ship with the native core.

## The problem typification solves

A molecular structure has atoms and bonds, but a simulation needs *types* — identifiers that map each atom, bond, angle, and dihedral to specific force field parameters. The same carbon atom might be `opls_135` (aliphatic CH₃) or `opls_145` (aromatic ring carbon) depending on its chemical environment. Getting this assignment wrong silently produces wrong physics.

**A Typifier examines the chemical environment of each atom via SMARTS pattern matching and assigns the corresponding force field type.**

MolPy's `OPLSAATypifier` handles the full assignment in one call: atom types first, then pair parameters, then bond/angle/dihedral types derived from the atom type assignments.

## What typification looks like end to end

The workflow is always the same: build the structure, load a force field, create a typifier, call `typify`.


```python
import molpy as mp
from molpy.ff.typifier import OPLSAATypifier

# 1. Build the structure
mol = mp.io.SmilesIR("CCO").to_atomistic()
mol, _ = mp.Conformer(add_hydrogens=True, seed=42).generate(mol)
mol.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)  # in place

print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")
print(f"angles: {len(mol.angles)}, dihedrals: {len(mol.dihedrals)}")
```

```text
atoms: 9, bonds: 8
angles: 13, dihedrals: 12
```


The OPLS-AA parameters live in the native core, so there is no force-field file to load: `OPLSAATypifier` carries the whole OPLS-AA library (`typifier.library()`). `typify` assigns the types, and `typifier.forcefield()` then returns exactly the parameters of the types it assigned — the force field of this system, ready to export or compile.


```python
# 2. Typify; the typifier owns the OPLS-AA library
typifier = OPLSAATypifier(strict=True)

typed_mol = typifier.typify(mol)
ff = typifier.forcefield()  # the parameters of the assigned types
```

`typify` returns a typed **copy** and leaves `mol` untouched — atoms in the returned object carry a `type` key and their charge, and its bonds, angles and dihedrals carry their types.


```python
# 3. Inspect results
for atom in typed_mol.atoms:
 element = atom.get("element", "?")
 atype = atom.get("type", "untyped")
 charge = atom.get("charge") or 0.0
 print(f" {element:2s} -> {atype:15s} q={charge:+.4f}")
```

```text
 C  -> opls_135        q=-0.1800
 C  -> opls_157        q=+0.1450
 O  -> opls_154        q=-0.6830
 H  -> opls_140        q=+0.0600
 H  -> opls_140        q=+0.0600
 H  -> opls_140        q=+0.0600
 H  -> opls_140        q=+0.0600
 H  -> opls_140        q=+0.0600
 H  -> opls_155        q=+0.4180
```


## How atom typing works

The typifier matches the SMARTS patterns of its force-field library (the embedded OPLS-AA table, or the XML file passed as `OPLSAATypifier(source=...)`). Each pattern defines one atom type — for example, `[CX4;H3]` matches an sp3 carbon with three hydrogens (a methyl carbon). The typifier walks through all atoms, matches each one against the pattern library, and assigns the best-matching type.

When multiple patterns match, priority and override rules in the force field resolve the conflict. This layered matching handles complex cases like aromatic vs. aliphatic nitrogen without manual intervention.

When no pattern matches, MolPy does not perform implicit parameter estimation for unmatched environments. Instead, unmatched cases are explicitly reported, allowing users to inspect and extend the rule set as needed. This design separates parameter assignment from parameter development, ensuring that force field definitions remain transparent and reproducible.

With atom types in place, the typifier has everything it needs to derive bonded types mechanically — a process described in the next section.

## How bonded typing works

Once atom types are assigned, bonded interactions follow mechanically. A bond between atom types `CT` and `OH` maps to bond type `CT-OH`. The same logic extends to angles (three-type sequences) and dihedrals (four-type sequences). Wildcard types (`*`) in the force field act as fallbacks when no specific match exists. Partial charges come with the atom types of the force field (OPLS-AA here), from a native charge model (`mp.ff.charge.GasteigerModel`, `mp.ff.charge.BccModel`, `mp.ff.charge.MullikenModel`), or from AmberTools (see [AmberTools Integration](13_ambertools_integration.md)).

## Strict vs. non-strict mode

An atom no rule matches always raises `ValueError` — there is no partial atom typing. `strict` governs the bonded terms: with `strict=True` (the default) a bond, angle or dihedral whose atom types have no parameters in the force field raises an error. This is the right default during development — it catches missing force field parameters before they become silent errors in production.

With `strict=False` such bonded terms are left unparameterised (no `type`) instead. Use this when you know some terms will not match and you will supply their parameters yourself.

## Every atom, bond, angle, and dihedral carries its assigned type

After typification, you can iterate over bonds, angles, and dihedrals to see their assigned types.


```python
# Bond types
for bond in typed_mol.bonds[:3]:
 i_sym = bond.itom.get("element")
 j_sym = bond.jtom.get("element")
 btype = bond.get("type", "untyped")
 print(f" {i_sym}-{j_sym} -> {btype}")

# Angle types
for angle in typed_mol.angles[:3]:
 names = [a.get("type", "?") for a in angle.endpoints]
 atype = angle.get("type", "untyped")
 print(f" {'-'.join(names)} -> {atype}")
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
from pathlib import Path

frame = typed_mol.to_frame()
frame.box = mp.Box.cube(30.0)

# LAMMPS molecular styles need a molecule ID per atom: one per bonded component
frame["atoms"]["mol_id"] = mp.Topology.from_frame(frame).connected_components() + 1

outdir = Path("06_output")
outdir.mkdir(exist_ok=True)

# the pair cutoff is a run setting: you declare it, molpy never invents one
ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
ff.get_style("pair", "coul/cut")["cutoff"] = 10.0

mp.io.write_lammps_data(outdir / "ethanol.data", frame)
mp.ff.forcefield.write_lammps_forcefield(outdir / "ethanol.ff", ff, frame)

print(f"exported to {outdir}")
```

```text
exported to 06_output
```


The structure and the coefficients are two files and two calls. `write_lammps_forcefield` looks up every type label the frame uses and writes only those coefficients. OPLS-AA defines no cutoff — a cutoff belongs to the run, not to the force field — so it is declared on the two pair styles before writing. The data writer refuses a bonded frame without `mol_id`: which atoms form a molecule is your decision, and the bond graph's connected components are the usual answer.

## Typing an assembled polymer

`mp.builder.Assembler` joins units along a site graph (see [Assembly](02_assembly.md)) and assigns no types. The chain it returns is an ordinary `mp.Atomistic`, so it is typed like any other structure: one `typify` call on the finished molecule. Here the chain is a methyl-capped poly(ethylene oxide) hexamer whose caps close both ends, so no port is left open.


```python
units = {"CAPA": "C[>]", "EO": "[<]OCC[>]", "CAPB": "[<]OC"}
conformer = mp.Conformer(seed=42)
library = {
    name: conformer.generate(
        mp.io.SmilesIR.from_fragment(body).to_template()
    )[0]
    for name, body in units.items()
}

sites = mp.io.CGSmilesIR("{[#CAPA][#EO]|6[#CAPB]}").to_coarsegrain()
chain = mp.builder.Assembler(library, mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)

typed_chain = OPLSAATypifier(strict=True).typify(chain)
print(f"atoms: {typed_chain.n_atoms}, open ports: {typed_chain.n_ports}")
print("atom types:", sorted({atom.get("type") for atom in typed_chain.atoms}))
```

```text
atoms: 51, open ports: 0
atom types: ['opls_180', 'opls_181', 'opls_182', 'opls_185']
```


## When standard force fields are not enough

Standard OPLS-AA covers common organic functional groups. Specialized molecules — ionic liquids (TFSI), metal complexes, reactive intermediates — often need custom force field parameters. In those cases:

1. Use a specialized OPLS-style force field XML that includes the required SMARTS patterns and types (`OPLSAATypifier(source="custom.xml")`)
2. Or drop to the [Force Field](../tutorials/04_force_field.md) layer and define types manually

The typifier itself is agnostic to the force field content. It only needs SMARTS patterns and type definitions in the XML. If those are present, it will match them. For GAFF / GAFF2 there is the native `mp.ff.typifier.AtdTypifier` (atom types only) and the AmberTools route.

See also: [Force Field](../tutorials/04_force_field.md), [Assembly](02_assembly.md).
