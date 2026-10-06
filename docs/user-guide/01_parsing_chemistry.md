[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/01_parsing_chemistry.ipynb)

# Parsing Chemistry

From a one-line string to an editable structure. MolPy reads three chemical
notations — **SMILES** for one concrete molecule, **SMARTS** for a structural
query, **CGsmiles** for units and how they join — all parsed by the chemistry
engine, exposed as types rather than helper functions.

## Two notations, two purposes

Chemical notation is a compression scheme, and each format answers a different
question. SMILES asks *"what is this exact molecule?"* and encodes atoms, bonds
and stereochemistry. SMARTS asks *"what structural pattern should I match?"* and
encodes logical constraints rather than physical atoms — it never builds a
structure.

There is no parser *function* to look up: you name the type you want.
`mp.io.read_smiles` gives you a graph, `SmilesIR` gives you the parsed
intermediate representation, `SmartsPattern` gives you a compiled query,
`CGSmilesIR` gives you a parsed CGsmiles string.

> **Polymer notations.** CGsmiles is read by `mp.io.CGSmilesIR`
> ([below](#cgsmiles-describes-units-and-how-they-join)); it is how units and
> polymer topologies are written for [Assembly](02_assembly.md). BigSMILES and
> G-BigSMILES are not parsed.

## SMILES describes one specific molecule

`mp.io.read_smiles` is the right choice whenever you have a single, fully
specified molecule. It parses the string and returns an `Atomistic` containing
atoms and bonds.

```python
import molpy as mp

mol = mp.io.read_smiles("CC(=O)OCC") # ethyl acetate
print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")

elements = [atom.get("element") for atom in mol.atoms]
print(elements)
```

```text
atoms: 6, bonds: 5
['C', 'C', 'O', 'O', 'C', 'C']
```

**Hydrogens are not added.** A SMILES string states connectivity; filling
open valences is a separate perception step, so `read_smiles` gives you exactly
the heavy-atom skeleton the string names. Ask for the hydrogens when you want
them:

```python
skeleton = mp.io.read_smiles("CCO")
filled = mp.Perceive().find_hydrogens(skeleton)

print(f"skeleton: {len(skeleton.atoms)} atoms") # C, C, O
print(f"filled: {len(filled.atoms)} atoms") # + 6 H
print("the input is untouched:", len(skeleton.atoms))
```

```text
skeleton: 3 atoms
filled: 9 atoms
the input is untouched: 3
```

A `.`-separated SMILES names a *set* of molecules, not a molecule — ion
pairs and solvent mixtures use this. `read_smiles` refuses it rather than
silently returning a disconnected graph; `SmilesIR.components()` takes it
apart.

```python
try:
 mp.io.read_smiles("[Li+].[F-]")
except ValueError as exc:
 print("refused:", exc)

ions = mp.io.SmilesIR("[Li+].[F-]").components()
print(f"components: {len(ions)} -> {[len(i.atoms) for i in ions]}")
```

```text
refused: read_smiles needs one component, '[Li+].[F-]' has 2. Use mp.io.SmilesIR(smiles).components(), or pass one component at a time.
components: 2 -> [1, 1]
```

### Aromaticity comes from the notation, and perception can revise it

Aromatic atoms are lowercase in SMILES, and the parser records that as
`is_aromatic` on each atom. Ring-closure digits must match: the first
occurrence opens the ring, the second closes it.

```python
benzene = mp.io.read_smiles("c1ccccc1")
print([atom.get("is_aromatic") for atom in benzene.atoms])
```

```text
[1, 1, 1, 1, 1, 1]
```

`Perceive().find_aromaticity()` **re-derives** the flag from the ring and its
bonds rather than trusting the notation, so a Kekulé structure written with
explicit double bonds comes out aromatic too:

```python
kekule = mp.io.read_smiles("C1=CC=CC=C1")
print("as written:  ", [atom.get("is_aromatic") for atom in kekule.atoms])

perceived = mp.Perceive().find_aromaticity(kekule)
print("re-perceived:", [atom.get("is_aromatic") for atom in perceived.atoms])
```

```text
as written:   [None, None, None, None, None, None]
re-perceived: [1, 1, 1, 1, 1, 1]
```

## SMARTS: pattern matching, not structure building

SMARTS shares SMILES syntax on the surface, but its semantics are entirely
different. Where SMILES encodes one concrete molecule, SMARTS encodes a query:
`[C;X4][O;H1]` means "an sp3 carbon bonded to a hydroxyl oxygen" and matches
*any* molecule containing that environment. A `SmartsPattern` has no atoms to
read — it has matches to find.

```python
query = mp.SmartsPattern("[C;X4][O;H1]")
print(f"query atoms: {query.num_query_atoms}, max bond depth: {query.max_bond_depth}")

ethanol = mp.Perceive().find_hydrogens(mp.io.read_smiles("CCO"))
print("matches ethanol:", query.has_match(ethanol))
for match in query.find_matches(ethanol):
 print(" matched atom handles:", match.atoms)
```

```text
query atoms: 2, max bond depth: 1
matches ethanol: True
 matched atom handles: [4294967298, 4294967299]
```

Note the pattern is matched against the **hydrogen-filled** structure:
`X4` counts connections and `H1` counts hydrogens, so both are answered wrong on
a bare skeleton. Perceive the hydrogens first, query after.

SMARTS is the language of force-field typification: patterns map atom
environments to force-field types. See *Typifier* in this guide.

## Splitting parse from convert

`mp.io.read_smiles` parses and converts in one call, which suits most
workflows. `SmilesIR` is the step in between, for when you want to know what the
string said before committing to a graph — how many molecules it names, and
whether to take them together or separately.

```python
ir = mp.io.SmilesIR("CCO.O")
print(f"components: {ir.n_components}")

together = ir.to_atomistic()
print(f"to_atomistic(): one graph of {len(together.atoms)} atoms")

separate = ir.components()
print(
 f"components(): {len(separate)} graphs of {[len(m.atoms) for m in separate]} atoms"
)
```

```text
components: 2
to_atomistic(): one graph of 4 atoms
components(): 2 graphs of [3, 1] atoms
```

## CGsmiles describes units and how they join

CGsmiles writes a molecule at more than one resolution. The first block is a
graph of named beads — here three `EO` units in a row — and the block after it
says what each name is made of. A bonding descriptor such as `[<]` or `[>]`
marks where a fragment may bond: `<` joins `>`.

`CGSmilesIR` parses the string once and reads it three ways. `templates()`
returns each fragment as an `mp.Atomistic` template that keeps its descriptors
as **ports**: an anchor atom plus a hydrogen handle that leaves when the port
bonds. `to_coarsegrain()` returns the bead graph, one site per unit, which is
the topology [Assembly](02_assembly.md) grows. `to_atomistic()` expands the
whole string into one heavy-atom graph, turning every paired descriptor into a
bond.

```python
ir = mp.io.CGSmilesIR("{[#EO]|3}.{#EO=[<]OCC[>]}")

eo = ir.templates()["EO"]  # the unit: a ported template
print(f"EO template: {eo.n_atoms} atoms, {eo.n_ports} ports")
for port in eo.ports:
    anchor, handle = port.anchor.get("element"), port.handle_atom.get("element")
    print(f"  port {port.get('port_kind')}: anchor {anchor}, leaving {handle}")

sites = ir.to_coarsegrain()  # the topology: one site per unit
bead_types = [bead.get("bead_type") for bead in sites.beads]
print(f"site graph: {bead_types}, {len(sites.cgbonds)} bonds")

chain = ir.to_atomistic()  # the whole molecule, descriptors consumed as bonds
print(f"to_atomistic(): {chain.n_atoms} heavy atoms, {len(chain.bonds)} bonds")
```

```text
EO template: 5 atoms, 2 ports
  port <: anchor O, leaving H
  port >: anchor C, leaving H
site graph: ['EO', 'EO', 'EO'], 2 bonds
to_atomistic(): 9 heavy atoms, 8 bonds
```

## Choosing the right entry point

| You have | You want | Use |
| --- | --- | --- |
| A SMILES string, one molecule | An editable graph | `mp.io.read_smiles(s)` |
| A SMILES string, several molecules | One graph each | `mp.io.SmilesIR(s).components()` |
| A SMILES string | To inspect before converting | `mp.io.SmilesIR(s)` |
| A structural rule | To find where it matches | `mp.SmartsPattern(p)` |
| Missing hydrogens / aromaticity | A perceived structure | `mp.Perceive().find_*(mol)` |
| A CGsmiles string with fragments | Ported unit templates | `mp.io.CGSmilesIR(s).templates()` |
| One fragment body such as `[<]OCC[>]` | One ported unit template | `mp.io.SmilesIR.from_fragment(body).to_template()` |
| A CGsmiles string | A site graph (topology) | `mp.io.CGSmilesIR(s).to_coarsegrain()` |
| Unit templates and a site graph | A polymer | `mp.builder.Assembler(library, mp.builder.GrowthPlacer())` |

Reach for `SmartsPattern` only for matching rules that feed the typifier, never
for structure creation. And when the molecule is a polymer, write its units and
its topology in CGsmiles and join them with `mp.builder.Assembler` — see
[Assembly](02_assembly.md) and [Polymer Topologies](topology/index.md).
