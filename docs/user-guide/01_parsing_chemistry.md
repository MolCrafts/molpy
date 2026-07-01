[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/user-guide/01_parsing_chemistry.ipynb)

# Parsing Chemistry

This guide explains how MolPy interprets SMILES, SMARTS, BigSMILES, and CGSmiles, and when each notation is appropriate in a molecular or polymer modeling workflow.

## Four notations, four purposes

Chemical notation is a compression scheme: each format answers a specific question and encodes exactly the information that question requires. SMILES asks "what is this exact molecule?" and encodes atoms, bonds, and stereochemistry. SMARTS asks "what structural pattern should I match?" and encodes logical constraints rather than physical atoms. BigSMILES asks "what is this repeat unit, and where does it attach?" and encodes polymer connectivity intent. CGSmiles asks "how are building blocks arranged?" and encodes topology without any chemistry at all.

MolPy's parser module, available as `mp.parser`, provides one function for each question. Convenience functions return `Atomistic` objects directly; IR-level functions return intermediate representations for advanced inspection and are covered at the end of this chapter.

## SMILES describes one specific molecule

`parse_molecule` is the right choice whenever you have a single, fully specified molecule and want to work with it immediately. It parses the SMILES string and returns an `Atomistic` object containing atoms and bonds. Hydrogens are implicit in SMILES and can be added later during coordinate generation.


```python
import molpy as mp

mol = mp.parser.parse_molecule("CC(=O)OCC")  # ethyl acetate
print(f"atoms: {len(mol.atoms)}, bonds: {len(mol.bonds)}")

elements = [atom.get("element") for atom in mol.atoms]
print(elements)  # ['C', 'C', 'O', 'O', 'C', 'C']
```

    atoms: 6, bonds: 5
    ['C', 'C', 'O', 'O', 'C', 'C']


When the input contains dot-separated components — a common SMILES convention for ion pairs and solvent mixtures — use `parse_mixture` instead, which always returns a list.


```python
ions = mp.parser.parse_mixture("[Li+].[F-]")
print(len(ions))  # 2

# parse_mixture also works for single molecules
mols = mp.parser.parse_mixture("CCO")
print(len(mols))  # 1
```

    2
    1


Aromatic atoms are lowercase in SMILES. The ring closure digits must match: the first occurrence opens the ring and the second closes it. Parsing benzene this way preserves aromaticity flags on each atom, which downstream typifiers rely on.


```python
benzene = mp.parser.parse_molecule("c1ccccc1")
for atom in benzene.atoms:
    print(f"{atom.get('element')}, aromatic={atom.get('aromatic')}")
```

    C, aromatic=True
    C, aromatic=True
    C, aromatic=True
    C, aromatic=True
    C, aromatic=True
    C, aromatic=True


Stereochemical information defined in SMILES/SMARTS (e.g., tetrahedral and double-bond stereochemistry) is parsed and preserved during topology construction when explicitly specified.

SMILES is the right starting point for small molecules with known connectivity. When you need to describe a recurring structural motif rather than one specific structure, SMARTS is the appropriate notation.

## SMARTS: pattern matching, not structure building

SMARTS shares SMILES syntax on the surface, but its semantics are entirely different. Where SMILES encodes one concrete molecule, SMARTS encodes a query: a set of constraints that may match many different molecules. Each atom specification can carry logical operators, property tests, and wildcard matches.

The parser returns a `SmartsIR` object, not an `Atomistic`. This distinction is intentional and matters: a SMARTS expression is a matching rule used by the typifier, not a physical structure you can simulate.


```python
query = mp.parser.parse_smarts("[C;X4][O;H1]")

print(f"query atoms: {len(query.atoms)}")
print(f"query bonds: {len(query.bonds)}")

for i, atom in enumerate(query.atoms):
    print(f"  atom {i}: {atom.expression}")
```

    query atoms: 2
    query bonds: 1
      atom 0: AtomExpressionIR(op='weak_and', children=[AtomPrimitiveIR(type='symbol', value='C'), AtomPrimitiveIR(type='neighbor_count', value=4)])
      atom 1: AtomExpressionIR(op='weak_and', children=[AtomPrimitiveIR(type='symbol', value='O'), AtomPrimitiveIR(type='hydrogen_count', value=1)])


SMARTS is the language of force-field typification: SMARTS patterns map atom environments to force-field types. Once you have typed atoms and assigned parameters, you may want to model how those atoms repeat into a polymer chain — which is what BigSMILES is for.

## BigSMILES: monomers with connection intent

Standard SMILES has no concept of a repeating unit or a connection point. BigSMILES introduces port markers (`<`, `>`, `$`) directly into the string to make polymer connectivity explicit. Each marker designates an atom as a terminal that can bond to another repeat unit.

`parse_monomer` produces an `Atomistic` with port metadata on the relevant atoms, ready for a polymer builder to chain together.


```python
monomer = mp.parser.parse_monomer("{[][<]CC(c1ccccc1)[>][]}")

print(f"atoms: {len(monomer.atoms)}")
ports = [a for a in monomer.atoms if a.get("port")]
print(f"ports: {len(ports)}")
for p in ports:
    print(f"  port '{p.get('port')}' on {p.get('element')}")
```

    atoms: 8
    ports: 2
      port '<' on C
      port '>' on C


For copolymer systems that specify multiple distinct monomers in one string, `parse_polymer` preserves the segment-level organization so that each monomer type remains independently accessible.


```python
spec = mp.parser.parse_polymer("{[<]CC[>],[<]CC(C)[>]}")

print(f"topology:  {spec.topology}")
print(f"monomers:  {len(spec.all_monomers())}")
```

    topology:  random_copolymer
    monomers:  2


BigSMILES captures the chemistry of each block: its atoms, its bonds, its connection ports. When you need to step back further and describe how those blocks are arranged at the architectural level — without specifying their internal chemistry at all — CGSmiles takes over.

## CGSmiles describes how blocks connect, not what they are

CGSmiles operates at a higher level of abstraction than any notation seen so far. Nodes in a CGSmiles string are labeled building blocks; edges between them are connections. The string says nothing about the atoms inside each block. Fragment definitions, supplied after a period, bind each label to its chemistry.


```python
from molpy.parser import parse_cgsmiles

# Linear chain: 5 copies of PEO with a fragment definition
cg = parse_cgsmiles("{[#PEO]|5}.{#PEO=[$]COC[$]}")

print(f"nodes: {len(cg.base_graph.nodes)}")
print(f"bonds: {len(cg.base_graph.bonds)}")
print(f"fragments: {len(cg.fragments)}")
```

    nodes: 5
    bonds: 4
    fragments: 1


The syntax is compact but precise. `[#LABEL]` references a named block; `|n` repeats it; parentheses introduce a branch; matched digits close rings, exactly as in SMILES. CGSmiles is the input to `PolymerBuilder`: BigSMILES defines what each block is, while CGSmiles defines how those blocks are arranged.

## Splitting parse and convert exposes the intermediate representation

The convenience functions (`parse_molecule`, `parse_monomer`) perform parsing and conversion in a single call, which is appropriate for most workflows. For diagnostics — inspecting aromaticity perception, verifying port assignment, or debugging unexpected atom counts — you can separate these into two explicit steps.


```python
from molpy.parser import parse_smiles, smilesir_to_atomistic

# parse_smiles returns SmilesGraphIR (single molecule) or list[SmilesGraphIR]
# (dot-separated mixture). For guaranteed single-molecule input, index [0] is
# not needed — a bare SMILES string always returns a single IR object.
ir = parse_smiles("CCO")
print(f"IR atoms: {len(ir.atoms)}")

for atom_ir in ir.atoms:
    print(f"  element={atom_ir.element}, aromatic={atom_ir.aromatic}")

mol = smilesir_to_atomistic(ir)
print(f"Atomistic atoms: {len(mol.atoms)}")
```

    IR atoms: 3
      element=C, aromatic=False
      element=C, aromatic=False
      element=O, aromatic=False
    Atomistic atoms: 3


The same pattern is available for BigSMILES, where inspecting the IR before conversion reveals how port markers were resolved and which atoms were assigned connection roles.


```python
from molpy.parser import parse_bigsmiles, bigsmilesir_to_polymerspec

ir = parse_bigsmiles("{[<]CC[>],[<]CC(C)[>]}")
spec = bigsmilesir_to_polymerspec(ir)
print(f"topology: {spec.topology}")
```

    topology: random_copolymer


## Choosing the right parser

The four notations cover a spectrum from fully specified chemistry to pure topology. Use `parse_molecule` or `parse_mixture` whenever you have a concrete small molecule in SMILES. Move to `parse_monomer` or `parse_polymer` the moment connectivity ports become relevant — that is, whenever your molecule is a polymer repeat unit. Reserve `parse_smarts` for matching rules that feed into the typifier, never for structure creation. Use `parse_cgsmiles` when you need to express architectural arrangements of named blocks, leaving the internal chemistry to the fragment definitions.

For 3D coordinates after parsing, use `mp.adapter.generate_3d(mol)` (requires RDKit).

See also: [Stepwise Polymer Construction](02_polymer_stepwise.md), [Atomistic and Topology](../tutorials/01_atomistic_and_topology.md).
