# Parser

Chemical string notation, parsed by the native core. Two notations are supported —
SMILES and SMARTS — and both are reached through a **type**, not a helper
function. Every parser type is a native type re-exported at the package root
(`mp.io.SmilesIR`); there is no `molpy.parser` module.

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.SmilesIR(s).to_atomistic()` | SMILES | `Atomistic` | One specific molecule (several components: one disconnected graph) |
| `mp.io.SmilesIR(s)` | SMILES | `SmilesIR` | Inspect before converting |
| `mp.io.SmilesIR(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.io.SmilesIR(s).components()` | dot-separated SMILES | `list[Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

There is no `read_smiles` / `parse_smiles` / `parse_smarts` /
`parse_molecule` / `parse_mixture`: each was a wrapper whose body was a
constructor call. Name the type instead.

## Canonical example

```python
import molpy as mp

mol = mp.io.SmilesIR("CCO").to_atomistic() # Atomistic (heavy atoms only)
mol = mp.Perceive().find_hydrogens(mol) #... with hydrogens

ions = mp.io.SmilesIR("[Li+].[F-]").components() # [Atomistic, Atomistic]

query = mp.SmartsPattern("[C;X4][O;H1]") # compiled query
query.find_matches(mol) # -> list[SmartsMatch]
```

A `.`-separated string names a *set* of molecules: `to_atomistic()` returns
them as one disconnected graph, `components()` one graph each, and
`n_components` says how many there are.

## Polymer notations

CGsmiles is parsed by `mp.io.CGSmilesIR`: `templates()` gives each fragment
as an `Atomistic` whose bonding descriptors are ports (one fragment body alone:
`mp.io.SmilesIR.from_fragment(body).to_template()`), and `to_coarsegrain()`
gives the site graph that [`mp.builder.Assembler`](builder.md) builds.
BigSMILES and G-BigSMILES are not parsed.

## Related

- `mp.Perceive` — hydrogens, aromaticity, rings, stereo (perceive *before* you
 match: `X4` and `H1` count what is actually in the graph)
- `mp.RingInfo` — ring / ring-system queries
- `mp.Reaction` — a reaction SMARTS applied to a graph in place: forms and
 breaks bonds, deletes the unmapped leaving atoms
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

::: molpy.io.SmilesIR

::: molpy.io.CGSmilesIR

::: molpy.SmartsPattern

::: molpy.SmartsMatch

::: molpy.io.SmilesError
