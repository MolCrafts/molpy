# Parser

Grammar-based parsing for chemical string notations. Convenience functions at `mp.parser.*`.

## Quick reference

| Function | Input | Output | Use when |
|----------|-------|--------|----------|
| `parse_molecule(s)` | SMILES | `Atomistic` | One specific molecule |
| `parse_mixture(s)` | dot-separated SMILES | `list[Atomistic]` | Multi-component (`[Li+].[F-]`) |
| `parse_monomer(s)` | BigSMILES | `Atomistic` (with ports) | Repeat unit with `<`/`>`/`$` markers |
| `parse_polymer(s)` | BigSMILES | `PolymerSpec` | Multi-monomer specification |
| `parse_smarts(s)` | SMARTS | `SmartsIR` | Pattern matching / typification |
| `parse_smiles(s)` | SMILES | `SmilesGraphIR` | IR-level inspection |
| `parse_bigsmiles(s)` | BigSMILES | `BigSmilesMoleculeIR` | IR-level BigSMILES inspection |
| `parse_cgsmiles(s)` | CGSmiles | `CGSmilesIR` | Topology architecture graphs |
| `parse_gbigsmiles(s)` | GBigSMILES | `GBigSmilesSystemIR` | System specs with distributions |

## Canonical example

```python
import molpy as mp

mol = mp.parser.parse_molecule("CCO")           # Atomistic
ions = mp.parser.parse_mixture("[Li+].[F-]")    # [Atomistic, Atomistic]
monomer = mp.parser.parse_monomer("{[][<]CCO[>][]}") # Atomistic with ports
spec = mp.parser.parse_polymer("{[<]CC[>],[<]CC(C)[>]}") # PolymerSpec
```

## Related

- `smilesir_to_atomistic` — SMILES IR → Atomistic
- `bigsmilesir_to_monomer` — BigSMILES IR → Atomistic
- `bigsmilesir_to_polymerspec` — BigSMILES IR → PolymerSpec
- [Guide: Parsing Chemistry](../user-guide/01_parsing_chemistry.md)

---

## Full API

### Convenience layer

::: molpy.parser

### SMARTS

::: molpy.parser.smarts

### SMILES / BigSMILES / CGSmiles

::: molpy.parser.smiles
