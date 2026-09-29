# Conformer

3D conformer generation from a molecular graph. `Conformer` takes an
`Atomistic` (typically from a SMILES or CGsmiles parse, which carries no
coordinates) and returns a structure with embedded 3D positions, using the
 backend. Available via `import molpy as mp` (`mp.conformer.Conformer`).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Conformer` | Generate 3D coordinates for a graph | Turning a parsed graph into a 3D structure |
| `ConformerReport` | Per-run summary of the generation pipeline | Inspecting which stages ran / succeeded |
| `ConformerStageReport` | Single-stage record within a report | Debugging a specific embedding stage |

## Related

- [Builder](builder.md) — `Assembler` consumes 3D units; embed a CGsmiles
 fragment with `Conformer` (this module) and its ports are kept. It is the only embedder MolPy ships.

---

## Full API

::: molpy.Conformer

::: molpy.ConformerReport

::: molpy.ConformerStageReport
