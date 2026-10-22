# AGENTS.md — molpy

The single instruction file for every coding agent in this repository (Claude
Code reads it natively; Codex and others read it by name). Evolving decisions
live in `.claude/NOTES.md`; stable ones are promoted here. Specs live in
`.claude/specs/`, indexed by `.claude/specs/INDEX.md`.

## What this is

molpy is one project shaped like numpy: a Rust core and one binding per
language, all of equal standing. The repository root is the project (README,
manual, gates); the Python binding is one of the bindings.

| Path | What | Published as |
|---|---|---|
| `crates/molrs/` | the Rust core, crate `molcrafts-molrs` (lib `molrs`) | crates.io `molcrafts-molrs` |
| `crates/molrs-ffi/` | the ABI layer: handles, capsules, the only ABI contract | — |
| `molpy/` | Python binding: pyo3 crate (`molpy/src/`) building `molpy._core`, the flat package `molpy/molpy/`, tests `molpy/tests/` | PyPI `molcrafts-molpy` |
| `molwasm/` | WebAssembly binding (wasm-bindgen) | npm `@molcrafts/molwasm` |
| `molc/` | C API, header `molc.h`, symbols `molc_*` / `Molc*` / `MOLC_*` | — |
| `molcxx/` | CXX bridge for Atomiverse C++ (`FrameView`, zero-copy) | — |
| `docs/` + `zensical.toml` | the molpy manual (Zensical), `docs/bindings/` per binding | — |
| `scripts/check.sh` | the one definition of every gate | — |

There is no Python `molrs` package and no `molrs-python`. Python code imports
`molpy`; the native module is `molpy._core` (the only module whose name starts
with `_`). The version is `[workspace.package].version` in the root
Cargo.toml, and every standalone binder manifest carries the same number;
maturin reads `molpy/Cargo.toml`, and `molpy.version` re-exports
`molpy._core.__version__`.

**Releases are frozen** until 04-downstream lands (see
`.claude/specs/molpy-unification.md`): push no `v*` tag, and push nothing
without the operator confirming the target remote and branch.

**ABI.** The minor line is the ABI contract. Capsule names
(`molrs.FrameRef/<major.minor>`, …) are defined only in
`crates/molrs-ffi/src/abi.rs`; a layout change inside a minor line fails the
`ffi` gate.

## Design preferences

### Iron law — no silent debt

A failing test, broken invariant, anti-pattern or clear bug in the surface
you touch or depend on is prioritized or hard-stopped, never ignored:

1. Do not leave it as "pre-existing", skip-mark it, weaken an assert, or land
   a feature on known rot.
2. Fix it now if it is local; otherwise stop and report `path:line`.
3. Name it in the summary (found / fixed / blocking).

### Shape

- **OOP by default.** Domain concepts are types with methods
  (`NeighborList.build`, `ForceField.to_potentials`). Module-level functions
  only for true free operations or a format's one-frame door (`read_pdb`).
- **Primitive, single-responsibility public APIs.** Callers compose:
  construct → configure → one concern → read result. No `run_everything`,
  `compute_all` or pipeline façades; no `make_*` / `create_*` factories
  aliasing a constructor; no god structs or ambient context bags.
- **Inline until the second real use.** No forwarding wrappers, no help
  functions on the `mp.*` façade.
- **Core sinks to molrs.** Force fields, topology, typifiers, IO and compute
  kernels live in `crates/molrs`; a binding binds them and never
  re-implements them. Python keeps only what composes native calls at a call
  site (examples, docs) — not a molpy function wrapping them.
- **No backward-compatibility apparatus before 1.0**: no aliases, legacy
  spellings, deprecation shims or migration notes. A rename changes every
  caller in the same change.
- **No `try`/`except` (Rust: no swallowed `Result`) except at an external
  boundary** — file, subprocess, network, a third-party library — and there
  the error is re-raised or converted to a named domain error. No
  `import x as _x` aliases; import the real name.
- **No special cases to satisfy a test.** A failing test means the logic is
  wrong or the test is wrong; fix whichever it is.

### Naming

- Module names never start with `_` (except `molpy._core`) and are never a
  container word (`data`, `base`, `types`, `common`, `utils`, `helpers`,
  `env`).
- Acronyms are cased as words: `LammpsEngine`, `OpenmmXmlReader`, `Rdf`,
  `Lbfgs`. Counts are `n_*`.
- File IO is one door per format and direction on `mp.io`
  (`read_pdb(source)` / `write_pdb(sink, frame)`); multi-frame and
  force-field formats are classes in the format submodule
  (`mp.io.lammps.LammpsForcefieldWriter`). No extension-dispatched
  `read_frame`. Sources are `os.PathLike`, `bytes` or `mp.io.Text`; a bare
  `str` is refused. Sinks are `os.PathLike` or `bytearray`.
- Coordinates are columns `x`/`y`/`z`; never a packed `xyz` column or key.
  `type` is a string label, `type_id` an unsigned index.
- What is public is what `__all__` lists; user-facing code and tests write
  `import molpy as mp` and `mp.<sub>.<Name>`.
- External tools sit under `mp.tool` (e.g. `molpy.tool.ambertools`), never at
  the package top level. GAFF typing goes through AmberTools only.
- Engine export of a style comes from its registration (`lammps = …` on the
  style declaration), read by the core writers.

## Testing

Tests check correctness only: what code computes, writes, raises or refuses.
Not tests: style, naming or module-location scans, API-shape or absence
checks, source greps, doc snippets run as tests, end-to-end pipelines,
numbers captured from another program, "molpy returns what molrs returns".

- **No third-party scientific software at test time** (RDKit, AmberTools,
  OpenMM, LAMMPS, GROMACS, Packmol, freud) and **no test skipped** for a
  missing program — such a test runs nowhere CI looks, so it is not written.
  Goldens are hand-written numbers or committed static data regenerated
  offline.
- Rust: unit tests next to the code (`#[cfg(test)]` in `crates/molrs/src/**`);
  there is no integration-binary tree. Doctests are a separate gate.
- Bindings test their seam (construct, call, round-trip types, error
  mapping). Python: `molpy/tests/` (`--import-mode=importlib`, warnings are
  errors, `tests/test_bindings/` for the former native-binding suite).
  Python IO fixtures are written in-process by molpy writers; small committed
  files sit under `molpy/tests/tests-data/` (`TEST_DATA_DIR` fixture).
- A gate must be proven to bite: break the code on purpose, watch it fail,
  restore.
- `tests-data/` (root) and `molpy/tests/tests-data/` stay gitignored: the
  36 small fixtures already tracked there stay tracked, a new one is added
  with `git add -f`, and a downloaded corpus is never committed.

## Build and gates

Every gate is `scripts/check.sh <gate…>`; prek hooks and CI call the same
script. `scripts/check.sh verify` is the pre-push gate (every check once).
Never bypass hooks (`--no-verify`); fix the failure.

Gates: `fmt ruff partners clippy doc test ffi cxx capi python types wasm mrec
docs`. `python` builds one wheel, installs it into `molpy/.venv` and runs
pytest; `types`, `mrec` and `docs` reuse that wheel.

```bash
cargo mrs-test                         # core unit tests (alias, one feature set)
cargo mrs-test -- ff::potential        # one module, no rebuild
cargo mrs-doctest                      # rustdoc examples
scripts/check.sh python                # wheel + molpy pytest
scripts/check.sh docs                  # strict manual build (zensical)
uv --directory molpy run --no-sync pytest tests/test_bindings/test_md.py   # one file, after `python`
```

Always use the `cargo mrs-*` aliases (`.cargo/config.toml`): a differently
spelled feature list is a different fingerprint and recompiles the whole
core (and `full` alone leaves out `zarr`/`filesystem`). Every binder (`crates/molrs-ffi`, `molpy`, `molwasm`, `molc`, `molcxx`)
is its own workspace — address it with `--manifest-path <dir>/Cargo.toml`,
never `-p`. All roots share `<repo>/target`. Linking is static by default;
the shared-dylib form is a command-line opt-in documented in
`docs/bindings/interop.md`.

Run Python from `molpy/` with `uv run --no-sync` (the project venv is
`molpy/.venv`), and `pytest` directly, never `python -m pytest` — the flat
layout would put the source tree ahead of the installed wheel. A site-packages
check uses `python -I`.

## Core data model and conventions

- `F = f64`, `I = i32`, `U = u32`, no precision switch.
- `Block`: string keys → typed columns with one `nrows`; component values
  are numpy arrays/scalars, and the core raises on anything else. `Frame`:
  string keys → `Block`s plus optional box and `meta`. Lengths in a `Frame`
  are always Å (GROMACS nm is converted at the reader); no binding converts
  again.
- `MolGraph`/`Atomistic`: slotmap arenas with kind-tagged relations;
  `Topology` is native adjacency (no petgraph in the core).
- `perceive` (rings, aromaticity, stereo, SMARTS) is always compiled, one
  layer above `core`, below `ff`/`io`/`conformer`.
- Angles are radians internally; each reader converts degrees at its
  boundary. Potentials take flat `[x0,y0,z0,…]` coordinates.
- Every force-field parameter table is committed Rust under
  `crates/molrs/src/ff/params/`; nothing parses parameter text at runtime.
- A force field declares its electrostatic constants (`coulomb`,
  `dielectric`, `coulomb14scale`); omitting one is an error, never a default.
- Compute types are configured by their constructor; per-frame streams
  (`Msd`, `Vacf`, `Rdf`) use `.accumulator()` → `push(...)` → `finish()`.
- `Cell<f64>` is not `Sync`; use `AtomicU64` with `to_bits`/`from_bits`.

## Documentation

The manual is executable: every ```python block runs, top to bottom per page,
against the current API. A block that cannot run in the docs environment
starts with `# docs: skip — <reason>`. Tutorials and guides are textbook
prose (concept → motivation → mechanics), not bullet lists; no visionary
claims. Public Python API has type hints and Google-style docstrings; Rust
uses rustdoc with runnable examples. Figures come from real runs
(`molpy/scripts/docs_series`, writing `docs/series/`), never hand-typed
points.

## Workflow

Project skills in `.claude/skills/` (prefix `mp-`): `mp-spec`, `mp-impl`,
`mp-fix`, `mp-refactor`, `mp-debug`, `mp-review` (+ single-axis `mp-arch`,
`mp-science`, `mp-perf`, `mp-bindings`, `mp-test`), `mp-docs`, `mp-note`.
Agents in `.claude/agents/`: `mp-architect`, `mp-science`, `mp-perf`,
`mp-bindings`, `mp-tester`, `mp-documenter`.

Git: PRs target `dev`; feature branches stay local and are deleted after
merge; the fork keeps only `master`/`dev`/`nightly`.
