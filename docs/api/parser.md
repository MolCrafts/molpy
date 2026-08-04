# Parser

Chemical string notation, parsed by `molrs`. Two notations are supported —
SMILES and SMARTS — and both are reached through a **type**, not a helper
function.

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.read_smiles(s)` | SMILES, one component | `Atomistic` | One specific molecule |
| `mp.SmilesIR(s)` | SMILES | `SmilesIR` | Inspect before converting |
| `mp.SmilesIR(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.SmilesIR(s).to_atomistic()` | SMILES | `molrs.Atomistic` | Every component as one graph |
| `mp.SmilesIR(s).components()` | dot-separated SMILES | `list[molrs.Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

There is no `parse_smiles` / `parse_smarts` / `parse_molecule` /
`parse_mixture`: each was a wrapper whose body was a constructor call. Name the
type instead.

## Canonical example

```python
import molpy as mp

mol = mp.io.read_smiles("CCO")  # Atomistic (heavy atoms only)
mol = mp.Perceive().find_hydrogens(mol)  # ... with hydrogens

ions = [
    mp.Atomistic.adopt(m)  # [Atomistic, Atomistic]
    for m in mp.SmilesIR("[Li+].[F-]").components()
]

query = mp.SmartsPattern("[C;X4][O;H1]")  # compiled query
query.find_matches(mol)  # -> list[SmartsMatch]
```

`read_smiles` raises on a `.`-separated string: that names a *set* of molecules,
not a molecule. Use `components()`.

## Polymer notations

BigSMILES, CGSmiles and G-BigSMILES are **no longer parsed**. Polymer
architecture is built explicitly with
[`molpy.builder.assembly`](builder.md) — `MonomerLibrary`, `PolymerBuilder`,
`GraphAssembler` — where the architecture is code rather than a string to
decode.

## Related

- `mp.Perceive` — hydrogens, aromaticity, rings, stereo (perceive *before* you
  match: `X4` and `H1` count what is actually in the graph)
- `mp.RingInfo` — ring / ring-system queries
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

::: molpy.parser
