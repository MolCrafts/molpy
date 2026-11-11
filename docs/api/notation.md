# Notation

Chemical string notation, parsed by the native core. One SMILES family is supported —
SMILES, SMARTS, PSMILES, CGsmiles and (generative) BigSMILES share one atom grammar and one
bonding-descriptor vocabulary — and each is reached through a **type**; the reader entry
points are `read_smiles(text)`, `read_smiles_fragment(body)`, `read_psmiles(unit)`,
`read_cgsmiles(text)` and `read_bigsmiles(text)`. Every notation type lives in the module of
its Rust owner — SMILES and PSMILES in `mp.io.smiles` (`mp.io.smiles.SmilesIr`), CGsmiles in
`mp.io.cgsmiles` (`mp.io.cgsmiles.CgSmilesIr`), BigSMILES in `mp.io.bigsmiles`
(`mp.io.bigsmiles.BigSmilesIr`), SMARTS in `mp.perceive` (`mp.perceive.SmartsPattern`).

## Quick reference

| Expression | Input | Output | Use when |
|------------|-------|--------|----------|
| `mp.io.smiles.read_smiles(s).to_atomistic()` | SMILES | `Atomistic` | One graph; `.`-separated components remain disconnected |
| `mp.io.smiles.read_smiles(s)` | SMILES | `SmilesIr` | Inspect before converting |
| `mp.io.smiles.read_smiles(s).n_components` | SMILES | `int` | How many molecules the string names |
| `mp.io.smiles.read_smiles(s).components()` | dot-separated SMILES | `list[Atomistic]` | One graph per component (`[Li+].[F-]`) |
| `mp.perceive.SmartsPattern(p)` | SMARTS | `SmartsPattern` | Pattern matching / typification |

## Canonical example

```python
import molpy as mp

mol = mp.io.smiles.read_smiles("CCO").to_atomistic() # Atomistic (heavy atoms only)
mol = mp.perceive.add_hydrogens(mol) #... with hydrogens

ions = mp.io.smiles.read_smiles("[Li+].[F-]").components() # [Atomistic, Atomistic]

query = mp.perceive.SmartsPattern("[C;X4][O;H1]") # compiled query
query.find_matches(mol) # -> list[SmartsMatch]
```

A `.`-separated string names a set of molecules: `to_atomistic()` returns
them as one disconnected graph, `components()` returns one graph each, and
`n_components` says how many there are.

## Polymer notations

CGsmiles is parsed by `mp.io.cgsmiles.CgSmilesIr`: `templates()` gives each fragment
as an `Atomistic` whose bonding descriptors are ports (one fragment body alone:
`mp.io.smiles.read_smiles_fragment(body).to_template()`), and `to_coarsegrain()`
gives the site graph that [`mp.builder.Assembler`](builder.md) builds.

PSMILES writes one repeat unit whose two `*` are its joining sites:
`mp.io.smiles.read_psmiles("[*]CC[*]")` reads each `*` as the `$` descriptor, so it is
the fragment `read_smiles_fragment("[$]CC[$]")` gives.

BigSMILES writes a polymer as SMILES with stochastic objects `{[] …; … []}`; its generative
extension adds port weights (`[<|0.5|]`), size distributions (`|schulz_zimm(1500, 1400)|`) and
system amounts (`.|5e5|`). `mp.io.bigsmiles.read_bigsmiles(text)` reads both; `to_system()`
gives the [`mp.builder.System`](builder.md) whose samples the assembler builds, sizes counted
in heavy-atom mass. An object written without end groups (`{[][$]CC[$][]}`) ends in hydrogen.
Nested stochastic objects, objects inside branches, fragment declarations
and ladder or non-covalent descriptors are refused.

## Related

- `mp.perceive.add_hydrogens`, `assign_aromaticity`, `assign_rings`, `assign_stereo` — hydrogens, aromaticity, rings, stereo (perceive *before* you
 match: `X4` and `H1` count what is actually in the graph)
- `mp.perceive.RingSet` — ring / ring-system queries
- `mp.perceive.Reaction` — a reaction SMARTS applied to a graph in place: forms and
 breaks bonds, deletes the unmapped leaving atoms
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

::: molpy.io.smiles.SmilesIr

::: molpy.io.cgsmiles.CgSmilesIr

::: molpy.io.bigsmiles.BigSmilesIr

::: molpy.perceive.SmartsPattern

::: molpy.perceive.SmartsMatch

::: molpy.io.smiles.SmilesError
