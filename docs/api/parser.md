# Parser

Chemical string notation, parsed by the native core. Two notations are supported —
SMILES and SMARTS — and both are reached through a **type**; the one reader
function is `mp.io.read_smiles`, a whole molecule straight to a graph. Every parser type is a native type in the molpy module that mirrors
its molrs owner — SMILES and CGsmiles in `mp.io.smiles` (`mp.io.smiles.SmilesIR`),
SMARTS in `mp.perceive` (`mp.perceive.SmartsPattern`); there is no
`molpy.parser` module. One molecule straight to a graph is `mp.io.read_smiles(s)`.

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.read_smiles(s)` | SMILES | `Atomistic` | One molecule (a `.`-separated set is refused) |
| `mp.io.smiles.SmilesIR(s).to_atomistic()` | SMILES | `Atomistic` | Any SMILES (several components: one disconnected graph) |
| `mp.io.smiles.SmilesIR(s)` | SMILES | `SmilesIR` | Inspect before converting |
| `mp.io.smiles.SmilesIR(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.io.smiles.SmilesIR(s).components()` | dot-separated SMILES | `list[Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.perceive.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

There is no `parse_smiles` / `parse_smarts` / `parse_molecule` /
`parse_mixture`: each was a wrapper whose body was a constructor call. Name
the type instead, or read one molecule with `mp.io.read_smiles`.

## Canonical example

```python
import molpy as mp

mol = mp.io.read_smiles("CCO") # Atomistic (heavy atoms only)
mol = mp.perceive.Perceive().find_hydrogens(mol) #... with hydrogens

ions = mp.io.smiles.SmilesIR("[Li+].[F-]").components() # [Atomistic, Atomistic]

query = mp.perceive.SmartsPattern("[C;X4][O;H1]") # compiled query
query.find_matches(mol) # -> list[SmartsMatch]
```

A `.`-separated string names a *set* of molecules: `mp.io.read_smiles` refuses
it, `SmilesIR(s).to_atomistic()` returns
them as one disconnected graph, `components()` one graph each, and
`n_components` says how many there are.

## Polymer notations

CGsmiles is parsed by `mp.io.smiles.CGSmilesIR`: `templates()` gives each fragment
as an `Atomistic` whose bonding descriptors are ports (one fragment body alone:
`mp.io.smiles.SmilesIR.from_fragment(body).to_template()`), and `to_coarsegrain()`
gives the site graph that [`mp.builder.Assembler`](builder.md) builds.
BigSMILES and G-BigSMILES are not parsed.

## Related

- `mp.perceive.Perceive` — hydrogens, aromaticity, rings, stereo (perceive *before* you
 match: `X4` and `H1` count what is actually in the graph)
- `mp.perceive.RingInfo` — ring / ring-system queries
- `mp.perceive.Reaction` — a reaction SMARTS applied to a graph in place: forms and
 breaks bonds, deletes the unmapped leaving atoms
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

::: molpy.io.read_smiles

::: molpy.io.smiles.SmilesIR

::: molpy.io.smiles.CGSmilesIR

::: molpy.perceive.SmartsPattern

::: molpy.perceive.SmartsMatch

::: molpy.io.smiles.SmilesError
