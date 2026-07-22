# molpy ↔ molrs spec pairing

Updated: 2026-07-22 (close-out batch).

## Version

| | version |
|--|--|
| molpy | 0.9.0 |
| molcrafts-molrs | ==0.9.0 |

## Live specs

| molrs | molpy | note |
|-------|-------|------|
| `chem-perceive-15-final-acceptance` (in-progress) | — | overall chain gate; gaff defect fixed separately |
| `net-streaming` (approved, net deferred) | — | serde/stream shipped 0.7+; WebSocket deferred |

## Closed this batch (all ACs verified → deleted)

| Spec | Repo |
|------|------|
| graph-assembler-01-reach | molpy |
| graph-assembler-02-kernel | molpy |
| graph-assembler-03-purge | molpy |
| graph-assembler-04-scope-from-molrs | molpy |
| molrs-core-cutover | molpy |
| pattern-syntax-01 | molrs |
| gaff-electrostatics | molrs |
| opls-typifier-downsink, mmff-native-01/02 | molpy (earlier) |

## Delivered code

- molrs: `SmartsPattern.max_bond_depth` / `ring_primitives` / `Atomistic.max_ring_system_size`
- molpy: `TypeScope` + `SmartsTypifier` (`typifier/scope.py`, `typifier/smarts.py`)
- molrs: GAFF `pair/coul/cut` + sander energy oracle tests
- molpy: OPLS assembly oracle, MMFF improper region oracle, in-repo scaling check

## Tests

- molpy: `pytest tests/ -m "not external"` → **1796 passed**
- molrs: `gaff_energy` + `pattern_syntax` green
