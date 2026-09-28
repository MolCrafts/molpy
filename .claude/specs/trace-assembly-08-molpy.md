---
title: "trace-assembly-08: molpy re-exports, and the backmap script end to end on the real data"
slug: trace-assembly-08-molpy
status: code-complete
created: 2026-09-27
repo: molpy
depends_on: [trace-assembly-07-python (molrs), molrs-015-align-01-import, molrs-015-align-02-core, molrs-015-align-03-backmap-reexport]
---

# trace-assembly-08: molpy re-exports, and the backmap script end to end on the real data

## Summary

The binding script `/home/jicli594/work/backmap_pe_pma/backmap.py` imports only `molpy`. This link makes its last missing names reachable, and it is accepted by running the script on the real data. The names:

- at the molpy root, by identity: `mp.Coarsener`, `mp.Trace`, `mp.Assembler`, `mp.TracePlacer` (`mp.Perceive` already is `molrs.perceive.Perceive`);
- in `mp.typifier`: `ElementTypifier`.

`mp.typifier` does not import today (molpy notes.md 2026-09-27), so its package `__init__` is narrowed to the native typifiers, and the unimported molpy modules are recorded as bounded debt.

Acceptance is the script run with the active venv's `python`. It exits 0 on the real data and writes a LAMMPS data file:
- 6,170,200 atoms in a bond-capable `Atoms # molecular` layout;
- LAMMPS `read_data` accepts the file when `lmp` is on PATH;
- `mp.io.read_lammps_data` reads it back with matching counts;
- the box origin equals the CG origin · f, and every atom lies inside the box;
- whole-script wall time and RSS stay under the recorded bars.

## Design

**Constitution.** molpy `.claude/notes/law.md`:

- **Law 13, native facade.** Identity re-exports only.
- **Law 3, earn complexity.** Nothing beyond the script's names.
- **Law 10, no silent debt.** The typifier narrowing and the `write_lammps_data` wrapper are recorded.
- **Law 17, unit-only gate.** No new tests; the script run is a one-off acceptance.
- **Law 16 / § VII.** The "molrs 0.15 co-development pin" exception now names branch `ci/narrow-gitignore` (molpy commit e51d1fa).

**Preconditions,** checked by task 1 and stopping on failure:

- **P0** — molpy `.claude/notes/law.md` § VII's co-development exception Scope names the branch `ci/narrow-gitignore` and covers this link, and `git rev-parse --abbrev-ref HEAD` in molpy prints `ci/narrow-gitignore`.
- **P1** — molrs links 01–05 and 07 have landed (06 dropped) on the local editable molrs (feat/cgsmiles).
- **P2** — the active venv holds a **release** build of that molrs (`maturin develop --release --manifest-path molrs-python/Cargo.toml` with the venv active), and a one-liner touching `molrs.perceive.Coarsener`, `molrs.Trace`, `molrs.builder.Assembler`, `molrs.builder.TracePlacer` and `molrs.ff.typifier.ElementTypifier` exits 0.
- **P3** — molrs-015-align 01–03 are **committed** in molpy (the coordinator commits them before this link). `git status --porcelain -- src/` is empty at start, and that commit's SHA is recorded as the diff base for ac-008.
- **P4** — script line 25 reads `mp.Box(frame.box.h * f, pbc=frame.box.pbc, origin=frame.box.origin * f)` (fixed by the coordinator; architect 🔴). The script is otherwise not edited.

**Root re-exports (`src/molpy/__init__.py`).**
- `Trace` joins the `from molrs import (...)` identity band.
- `Coarsener` joins the `from molrs.perceive import (...)` block.
- A new `from molrs.builder import Assembler, TracePlacer` line is added.
- All four go into `__all__`.
- `molpy.builder`'s own broken imports of retired names (`builder/assembly/__init__.py:16`) stay with the deferred builder link; the root does not import `molpy.builder`.

**`mp.typifier` (`src/molpy/typifier/__init__.py`).**
- It becomes identity re-exports of `ElementTypifier`, `OPLSAATypifier` and `MMFF94Typifier as MMFFTypifier` (the existing public name) from `molrs.ff.typifier`, with a docstring.
- The molpy modules `base`, `clp`, `smarts`, `ambertools`, `scope`, `forcefield`, `region`, `affected_region`, `cache` and `_matching` are no longer imported by the package; importing one directly fails as today.
- The names dropped from `__all__` were unreachable, because the package failed at import.
- A Law-10 entry records this. Scope: `src/molpy/typifier/`. Removal condition: the deferred typifier link deletes or ports them, per the operator ruling "molpy deletes its typifiers, re-exports molrs". Owner: operator.

**Recorded, not changed.** `mp.io.write_lammps_data` (`src/molpy/io/writers.py:23-48`) is a module-level function under `mp`, conflicting with ruling (a). It forwards to the "thin native façade" `LammpsDataWriter` (law 13) and silently drops `atom_style` (`del atom_style`, law 14). The script depends on it, so it stays. A Law-10 entry records it, routed `/mol:refactor` (Owner: operator; Removal condition: the refactor lands the OOP writer surface).

**The run and its bars.**
- `python /home/jicli594/work/backmap_pe_pma/backmap.py`, with the active venv and not `uv run`, under `/usr/bin/time -v`.
- **Bars:** exit 0; whole-script wall time **< 15 min**; max RSS **< 64 GB**. The bars cover step 6's full copies (`to_atomistic`, `Typing::typify`'s clone plus per-element stamps, `to_frame`), which have no molrs-side bar.
- It prints `wrote …: 6170200 atoms`, since N = 100·(33·100 − 2·99) + 10000 + 13·450000.

**One-off checks** (scratchpad, not committed):
- *Read-back.* `mp.io.read_lammps_data(OUT, atom_style="molecular")` has 6,170,200 atoms and a bond count equal to the header's `bonds` line. There are 460,100 distinct molecule IDs. The Atoms header reads `# molecular`. The atom type labels are exactly the elements present, and every bond label is a canonical element pair.
- *Box.* `xlo`/`ylo`/`zlo` equal `frame.box.origin * f` (to 1e-9 Å, where f = 4.2 is the lj_sigma→Å factor). The extents equal the CG box lengths · f. Every written atom satisfies `lo ≤ x < hi` on each axis.
- *LAMMPS.* If `lmp` is on PATH, a one-line input (`units real`, `atom_style molecular`, `read_data OUT`) exits 0. Otherwise the check records that `lmp` is absent and relies on the header style being bond-capable (`molecular`), as ac-005 requires.

**No workaround on failure.** Any failure of an existing surface stops the run, is reported with path:line, and is routed to molrs `/mol:fix` / `/mol:spec` (law 10).

### Reuse decision

- `reuse molrs.perceive.Coarsener`, `molrs.Trace`, `molrs.builder.Assembler`, `molrs.builder.TracePlacer` and `molrs.ff.typifier.ElementTypifier` as identity re-exports.
- `reuse molrs.perceive.Perceive`, already the molpy identity.
- `pattern` the identity bands and the one-off acceptance scripts of molrs-015-align-03.
- `new` — none.

## Files to create or modify

- `src/molpy/__init__.py`
- `src/molpy/typifier/__init__.py`
- `.claude/notes/notes.md`

## Tasks

- [x] Verify preconditions P0–P4 with a one-off scratchpad script (branch `ci/narrow-gitignore` and § VII scope, landed molrs links, release `maturin develop` into the active venv, the committed molrs-015-align 01–03 with its SHA recorded, script line 25); stop and report on any failure
- [x] Re-export `Trace`, `Coarsener`, `Assembler` and `TracePlacer` by identity at the root in `src/molpy/__init__.py` (bands plus `__all__`)
- [x] Narrow `src/molpy/typifier/__init__.py` to identity re-exports of `ElementTypifier`, `OPLSAATypifier` and `MMFFTypifier`, with the docstring; verify `python -c "import molpy.typifier"` exits 0
- [x] Record Law-10 entries in `.claude/notes/notes.md` for the typifier narrowing and for `mp.io.write_lammps_data` (free function, forwarding façade, dropped `atom_style`), routed `/mol:refactor`
- [x] Verify the identities of `mp.Coarsener`, `mp.Trace`, `mp.Assembler`, `mp.TracePlacer`, `mp.Perceive` and `mp.typifier.ElementTypifier` with a one-off scratchpad script
- [x] Run `python /home/jicli594/work/backmap_pe_pma/backmap.py` with the active venv under `/usr/bin/time -v`; require exit 0, `6170200 atoms`, wall time < 15 min and max RSS < 64 GB
- [x] Check the written file with a one-off scratchpad script: `mp.io.read_lammps_data(OUT, atom_style="molecular")` counts, 460,100 molecule IDs, type-label sets, box origin = CG origin · f, every atom inside the box, and `lmp` `read_data` if `lmp` is on PATH
- [x] Record the wall time, max RSS, host, molrs commit, counts, header style, box check and `lmp` result in the 2026-09-27 entry of `.claude/notes/notes.md`

## Testing strategy

- **No new molpy tests** (law 17; re-exports get no molpy test). The typifier `__init__` change adds no molpy-owned behaviour.
- **One-off acceptance scripts** cover identities, the read-back, the box and LAMMPS, as in molrs-015-align-03. The end-to-end run is acceptance, not a suite test (law 11). LAMMPS runs only in this one-off, never in a gate.
- **The full molpy suite and `tox -e lint` stay deferred** to the molrs-015-align close link (builder/typifier tests fail at collection).

## Out of scope

- **The deferred molrs-015-align links:** typifier deletion, `molpy.builder` retired imports, moltemplate and the LAMMPS FF writers.
- **Refactoring `mp.io.write_lammps_data`.** Recorded, routed `/mol:refactor`.
- **A force field and `* Coeffs`.** Conformation only.
- **Relaxation.**
- **Tagging or releasing molpy,** and bumping the pin (law 16).
- **Editing the script** beyond the coordinator's line-25 fix (P4).
