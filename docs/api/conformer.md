# Conformer

3D conformer generation from a molecular graph. `Conformer` takes an
`Atomistic` (typically from a SMILES / BigSMILES parse, which carries no
coordinates) and returns a structure with embedded 3D positions, using the
molrs backend. Available via `import molpy as mp` (`mp.conformer.Conformer`).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Conformer` | Generate 3D coordinates for a graph | Turning a parsed graph into a 3D structure |
| `ConformerReport` | Per-run summary of the generation pipeline | Inspecting which stages ran / succeeded |
| `ConformerStageReport` | Single-stage record within a report | Debugging a specific embedding stage |

## Related

- [Builder](builder.md) — `PolymerBuilder` consumes 3D monomers; embed them
  with `Conformer` (this module) or `RDKitAdapter.generate_3d`.

---

## Full API

::: molpy.conformer
