# Parser

Chemical string notation, parsed by the native core. Two notations are supported —
SMILES and SMARTS — and both are reached through a **type**, not a helper
function. Every parser type is a native type re-exported at the package root
(`mp.SmilesIR`); there is no `molpy.parser` module.

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.read_smiles(s)` | SMILES, one component | `Atomistic` | One specific molecule |
| `mp.SmilesIR(s)` | SMILES | `SmilesIR` | Inspect before converting |
| `mp.SmilesIR(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.SmilesIR(s).to_atomistic()` | SMILES | `Atomistic` | Every component as one graph |
| `mp.SmilesIR(s).components()` | dot-separated SMILES | `list[Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

There is no `parse_smiles` / `parse_smarts` / `parse_molecule` /
`parse_mixture`: each was a wrapper whose body was a constructor call. Name the
type instead.

## Canonical example

```python
import molpy as mp

mol = mp.io.read_smiles("CCO") # Atomistic (heavy atoms only)
mol = mp.Perceive().find_hydrogens(mol) #... with hydrogens

ions = mp.SmilesIR("[Li+].[F-]").components() # [Atomistic, Atomistic]

query = mp.SmartsPattern("[C;X4][O;H1]") # compiled query
query.find_matches(mol) # -> list[SmartsMatch]
```

`read_smiles` raises on a `.`-separated string: that names a *set* of molecules,
not a molecule. Use `components()`.

## Polymer notations

CGsmiles is parsed by `mp.CGSmilesIR`: `templates()` gives each fragment
as an `Atomistic` whose bonding descriptors are ports (one fragment body alone:
`mp.SmilesIR.from_fragment(body).to_template()`), and `to_coarsegrain()`
gives the site graph that [`mp.Assembler`](builder.md) builds.
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

::: molpy.SmilesIR

::: molpy.CGSmilesIR

::: molpy.SmartsPattern

::: molpy.SmartsMatch

::: molpy.SmilesError
