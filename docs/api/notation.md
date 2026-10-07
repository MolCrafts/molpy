# Notation

Chemical string notation, parsed by the native core. Three notations are supported —
SMILES, CGsmiles and SMARTS — and each is reached through a **type**; the reader
functions are `mp.io.read_smiles_str` and `mp.io.read_cgsmiles_str`, a whole molecule straight to a graph. Every notation type is a native type in the molpy module that mirrors
its molrs owner — SMILES in `mp.io.smiles` (`mp.io.smiles.SmilesIr`), CGsmiles in
`mp.io.cgsmiles` (`mp.io.cgsmiles.CgSmilesIr`),
SMARTS in `mp.perceive` (`mp.perceive.SmartsPattern`).

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.read_smiles_str(s)` | SMILES | `Atomistic` | One molecule (a `.`-separated set is refused) |
| `mp.io.smiles.SmilesIr(s).to_atomistic()` | SMILES | `Atomistic` | Any SMILES (several components: one disconnected graph) |
| `mp.io.smiles.SmilesIr(s)` | SMILES | `SmilesIr` | Inspect before converting |
| `mp.io.smiles.SmilesIr(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.io.smiles.SmilesIr(s).components()` | dot-separated SMILES | `list[Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.perceive.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

## Canonical example

```python
import molpy as mp

mol = mp.io.read_smiles_str("CCO") # Atomistic (heavy atoms only)
mol = mp.perceive.add_hydrogens(mol) #... with hydrogens

ions = mp.io.smiles.SmilesIr("[Li+].[F-]").components() # [Atomistic, Atomistic]

query = mp.perceive.SmartsPattern("[C;X4][O;H1]") # compiled query
query.find_matches(mol) # -> list[SmartsMatch]
```

A `.`-separated string names a *set* of molecules: `mp.io.read_smiles_str` refuses
it, `SmilesIr(s).to_atomistic()` returns
them as one disconnected graph, `components()` one graph each, and
`n_components` says how many there are.

## Polymer notations

CGsmiles is parsed by `mp.io.cgsmiles.CgSmilesIr`: `templates()` gives each fragment
as an `Atomistic` whose bonding descriptors are ports (one fragment body alone:
`mp.io.smiles.SmilesIr.from_fragment(body).to_template()`), and `to_coarsegrain()`
gives the site graph that [`mp.builder.Assembler`](builder.md) builds.
BigSMILES and G-BigSMILES are not parsed.

## Related

- `mp.perceive.add_hydrogens`, `assign_aromaticity`, `assign_rings`, `assign_stereo` — hydrogens, aromaticity, rings, stereo (perceive *before* you
 match: `X4` and `H1` count what is actually in the graph)
- `mp.perceive.RingSet` — ring / ring-system queries
- `mp.perceive.Reaction` — a reaction SMARTS applied to a graph in place: forms and
 breaks bonds, deletes the unmapped leaving atoms
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

::: molpy.io.read_smiles_str

::: molpy.io.smiles.SmilesIr

::: molpy.io.cgsmiles.CgSmilesIr

::: molpy.perceive.SmartsPattern

::: molpy.perceive.SmartsMatch

::: molpy.io.smiles.SmilesError
