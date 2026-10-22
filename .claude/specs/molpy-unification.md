---
title: molpy unification — one project, one Python package, molrs as its native core
slug: molpy-unification
status: approved
created: 2026-10-22
chain: molpy-unification (01-user-paths → 02-repo-merge → 03-restructure → 04-downstream → 05-repo-rename)
repos: molpy (01), molrs (02, 03, 05), molpack / molvis / molab / Atomiverse / molnex / molhub / monorepo (04)
---

# molpy unification

## Summary

molrs and molpy already ship as one product in two repositories. Tags v0.15
through v0.19 match one to one. molpy pins `molcrafts-molrs>=0.X.0,<0.X+1`, and
56 of its 89 source files import molrs. Every Rust change costs a wheel
rebuild before molpy's tests see it. Every release costs two PRs, two tags and
a pin bump.

The end state is one project named **molpy**, in one repository whose
language bindings stand side by side (D10):

```
molpy/                     the project repository (today's molrs repo, renamed)
├── README.md  CONTRIBUTING.md  LICENSE …   project level (molpy's)
├── docs/  zensical.toml                     the project manual (molpy's)
├── scripts/check.sh                         every gate
├── Cargo.toml                               the workspace
├── crates/
│   ├── molrs/         Rust core (crates.io molcrafts-molrs)
│   └── molrs-ffi/     the ABI layer
├── molpy/             Python binding: pyproject.toml, Cargo.toml + src/ (pyo3),
│                      molpy/ (the package, flat), tests/, examples/
├── molwasm/           WebAssembly binding
├── molc/              C binding
└── molcxx/            C++ binding
```

| Today | End state |
|---|---|
| repo `MolCrafts/molrs` + repo `MolCrafts/molpy` | repo `MolCrafts/molpy` (the molrs repo, renamed) |
| Python dist `molcrafts-molrs` (`import molrs`, native `molrs._native`) | gone, no shim |
| Python dist `molcrafts-molpy` (pure Python) | `molcrafts-molpy`, a maturin abi3 wheel; native module `molpy._core` |
| `molrs-python/` | folded into `molpy/` (its pyo3 crate is `molpy/Cargo.toml` + `molpy/src/`) |
| `molrs/`, `molrs-ffi/` | `crates/molrs/`, `crates/molrs-ffi/` (crate names unchanged) |
| `molrs-wasm/` (npm `@molcrafts/molrs`) | `molwasm/` |
| `molrs-capi/` (`molcrafts-molrs-capi`) | `molc/` |
| `molrs-cxxapi/` (`molcrafts-molrs-cxxapi`) | `molcxx/` |
| `molrs-ext-example/` | deleted |
| molrs README, `docs/`, `molrs-python/docs` | deleted; useful pages folded into the molpy manual |

## Decisions (operator, 2026-10-22)

- **D1. Name.** The project and the Python package are called molpy. The
  Rust core keeps the name molrs.
- **D2. No `molrs` Python package.** It is deleted in 03, with no
  re-exporting shim. Downstream migrates in the same minor.
- **D3. Native module `molpy._core`**, following numpy's convention. It is the
  only module whose name starts with `_`.
- **D4. Module names never start with `_`.** Privacy is expressed by
  `__all__`, not by a name prefix. Snake-case names stay (`clpol_scaling`,
  `openmm_xml`).
- **D5. User-facing API is tested the way the manual writes it.** That means
  `import molpy as mp` and `mp.<subsystem>.<Name>`. Internal helpers may be
  imported directly from their implementation module. Tests are not split into
  two kinds.
- **D6. The repo merge is mechanical.** Files are copied in one commit. Git
  history and release history are not carried over.
- **D7. No adapter, no wrapper (ruled 2026-10-22, during 01).** `molpy.adapter`
  (RDKit) and `molpy.wrapper` are deleted, along with the `rdkit` extra. The
  external-program core is the public subsystem `mp.tool` (`Tool`, `ToolError`,
  `EnvironmentSpec`); `Engine` subclasses `Tool`. The AmberTools pipeline is
  `molpy.tool.ambertools`, an implementation package. It is never a top-level
  module. Its configuration group is `[ambertools]`.
- **D9. Tests check correctness only (ruled 2026-10-22).** No test of code
  style, naming, module location, API shape or absence, source scans, doc
  snippets, or end-to-end pipelines. The 01 gates that did this were deleted:
  the public-path rules, the user-path AST gate, the API snapshot and the
  untested ledger. What remains is `tests/test_init.py`'s check that every
  `__all__` entry resolves.
- **D8. No test skips for a missing program.** Such a test runs nowhere, since
  CI installs none of AmberTools, LAMMPS, GROMACS or OpenMM, and these tests
  hid real rot (`mp.io.read_prmtop_system`). External programs are tested
  with a mocked `subprocess`.

- **D10. One project, equal bindings (ruled 2026-10-22).** The repository
  is the molpy *project*, not the Python package's repository. Every
  language binding is a self-contained `mol<language>/` directory at the
  root (`molpy/`, `molwasm/`, `molc/`, `molcxx/`), each with its own build
  manifest, sources and tests. The Rust core and the ABI layer, which are not
  bindings, live in `crates/`. The project manual, README and contributor
  files are molpy's, at the root.
- **D11. Flat Python package.** The package is `molpy/molpy/`, next to the
  binding's `Cargo.toml` and `src/`; maturin picks it up as the mixed
  project's Python source. Every `src/` in the repository is Rust. Tests run
  against an installed molpy (`maturin develop` puts `_core` into the
  package; CI installs the tested wheel and runs `pytest`, never
  `python -m pytest` from a directory that shadows it).
- **D12. molrs-ext-example is deleted.**
- **D13. molrs's docs are folded.** The molrs-python docs site and the
  molrs README are deleted. `attribution`, `interop` and `mic-matrix` move to
  the manual's `docs/developer/`; `releasing` folds into `release-process`.
- **D14. This round changes this repository only.** Downstream repositories
  migrate in 04, after the operator reviews the structure. Releases stay
  frozen until then. The result is force-pushed after review.
## Chain

No release happens from 02 until 04 is done.

| Link | Scope | Depends |
|---|---|---|
| 01-user-paths | §1, in the molpy repo. | — |
| 02-repo-merge | §2. | 01 |
| 03-restructure | §3: the D10 layout, the Python fold, the binder renames. Breaking. | 02 |
| 04-downstream | §4: every downstream repository moves to `molpy`; the minor release. | 03 |
| 05-repo-rename | §5. | 04 |

## 1. 01-user-paths — the user-facing test contract (molpy repo)

**Goal.** A broken export fails a test directly. That covers a name dropped
from `__all__`, a module that stops re-exporting, and a misspelled path in a
test that never runs.

**Scope.**

1. **Drop leading underscores from module names (D4).** For example,
   `engine/_lammps.py` becomes `engine/lammps.py`, `builder/_polymer/` becomes
   `builder/polymer/` and `molpy/_ambertools/` becomes `molpy/ambertools/`.
   The renamed modules stay out of their parent's `__all__`. Update every
   in-tree reference, including `docs/api/*.md` mkdocstrings paths.
2. **Delete `molpy.wrapper` and `molpy.adapter`; add `mp.tool` (D7).** Delete
   every test skipped for a missing program (D8).
3. **The public surface is defined by `__all__`.** The walk starts at
   `molpy.__all__` and enters only modules listed in a public module's
   `__all__`. It lives in `tests/conftest.py` and is shared by every gate.
   The existing rules of `test_public_paths.py` are kept, re-based on
   `__all__`:
   - one public path per symbol;
   - the root holds modules, promoted classes and version metadata only;
   - molrs mirrors are identical objects;
   - the io/ff structure;
   - acronym casing;
   - no container-word module names.
4. **New gates.**
   - No module name starts with `_` (except `_core` after 03).
   - Every public module defines `__all__`, and every entry resolves.
   - **Public API snapshot** `tests/public_api.txt`: any addition or removal
     fails until the snapshot is regenerated
     (`python scripts/public_api.py --write`).
   - **User-path gate** `tests/test_user_paths.py`, an AST check over
     `tests/**/*.py`:
     - R1: `import molpy` only as `import molpy as mp`.
     - R2: `from molpy.<m> import n` only for an object that is not public.
     - R3: every `mp.…` attribute chain resolves statically to its canonical
       path, whether or not the test is skipped.
     - R4: no `import molrs`. The mirror-identity tests are the only exception;
       03 deletes it.
   - **Coverage ledger** `tests/untested_public_api.txt`: public names no test
     reaches through `mp.`. It may only shrink.
   - Every gate has reverse tests: synthetic snippets it must flag.
5. **Rewrite the test suite** to satisfy the gates. Delete the empty test
   directories.

**Status (2026-10-22): implemented on molpy's local branch `user-paths`, not
yet committed.** Item 4's gates were built, then deleted under D9. The suite
has 864 passing tests and 0 skipped.

**Acceptance.**

- `uv run pytest` is green in molpy, with no skipped tests.
- Each of these must fail a gate:
  - deleting `LammpsEngine` from `mp.engine.__all__`;
  - `from molpy.core import Frame` in a test;
  - a misspelled `mp.…` path in a skipped test;
  - a new `_`-prefixed module.
- `rg 'molpy(\.[a-z0-9_]+)*\._[a-z]' src tests docs examples` is empty.
- The docs build with no new warnings.

## 2. 02-repo-merge — copy molpy into the molrs repo

**Goal.** One repository, so that one change can span the Rust core and the
Python API and be tested together.

**Rulings (2026-10-22).**

- molpy lives at `molpy/`, next to `molrs-python/`. It is the whole molpy
  project: `pyproject.toml`, `src/`, `tests/`, `docs/`, `examples/` and
  `scripts/`. 03 reshapes the layout.
- The source is molpy `6a2c480` (branch `user-paths`), copied as is. The
  molpy repository is frozen from then on.
- mollog and molcfg come from PyPI ranges. molpy's `../mollog` and `../molcfg`
  path sources and their partner pins go away; molrs's `partners.env` is
  unchanged.
- Releases are frozen until 03 ships. molpy's release and nightly workflows
  are deleted. molrs's release workflows stay, but no tag is pushed.

**Scope.**

1. **Copy.** Extract `git archive 6a2c480` into `molpy/`, without the
   repository infrastructure the molrs root owns:
   - `.github/` and `.pre-commit-config.yaml`;
   - `scripts/check.py`, `scripts/partners.py`,
     `scripts/check_partners_on_pypi.py` and `scripts/hook-run.sh`.

   `.github/copilot-instructions.md` becomes `molpy/AGENTS.md`, which is local
   only: molrs's `.gitignore` ignores every `AGENTS.md`, and the text stays in
   molpy's history. The issue templates go too: GitHub reads them only at the
   repository root. Files molpy tracked despite its own `.gitignore` are added
   with `git add -f`, which keeps exactly molpy's tracked set:
   - the 36 fixtures under `molpy/tests/tests-data/` (the directory stays
     ignored);
   - 4 files under `docs/assets/`.

   The `uv.lock` ignore rule is deleted: `uv sync --locked` needs the lock.
2. **Dependencies.** In `molpy/pyproject.toml`, `[tool.uv.sources]` keeps only
   `molcrafts-molrs = { path = "../molrs-python" }`, and `molpy/uv.lock` is
   regenerated.
3. **Gates in `scripts/check.sh`.** The `rust-python` CI profile runs
   `check.sh` gates, so molcrafts-ci needs no change.
   - `molpy`: sync molpy's locked environment, install `MOLRS_TESTED_WHEEL`
     (or build one), then run `ruff format --check`, `ty check src/molpy` and
     `pytest`.
   - `molpy_docs`: zensical `--strict` for `molpy/` on the same wheel.
   - Both join `VERIFY` after `python`. The root `ruff` gate already covers
     `molpy/`, through `molpy/pyproject.toml`'s `[tool.ruff]`.
4. **CI (`test.yml`).** The `python` job runs `check.sh python molpy` in one
   process, so molpy is tested on the wheel that job built, on every OS leg.
   The `docs` job runs `check.sh docs molpy_docs`. `test / result` is
   unchanged.
5. **Docs deploy.** The operator re-points the Cloudflare Pages project for
   docs.molcrafts.org/molpy to this repository, with root `molpy/`.
6. **Freeze the old repository.** The operator archives MolCrafts/molpy after
   its README is pointed at `molrs/molpy/`.

**Status (2026-10-22):** on molrs's local branch `molpy-repo-merge`.
`scripts/check.sh molpy molpy_docs` passes locally: format and ty are clean,
864 tests pass in 239 s on the dev wheel, and the docs build `--strict`.

**Acceptance.**

- `scripts/check.sh python molpy molpy_docs` passes locally: molpy runs on the
  wheel built from the same commit.
- `git ls-files molpy/tests/tests-data` lists the 36 fixtures.
- molpy's suite passes with only the tracked fixtures present.

**Risk.** CI time grows by molpy's suite on each `python` leg: about 4 min,
measured locally on the dev-profile wheel the `python` gate builds.
molpy loses its Python 3.14 leg until 03; it is pure Python on an abi3 core.

## 3. 03-restructure — the D10 layout, one Python package, the binder renames

**Goal.** The repository has its final shape: equal bindings around a
Rust core, one Python package with the native module `molpy._core`, and no
`molrs` Python package.

**Scope, in local commits (reviewed before the force-push).**

1. **Rust layout.**
   - `molrs/` and `molrs-ffi/` move under `crates/`.
   - `molrs-wasm/` becomes `molwasm/`, `molrs-capi/` becomes `molc/`, and
     `molrs-cxxapi/` becomes `molcxx/`: directories, crate and library names,
     headers and CMake targets.
   - `molrs-ext-example/` is deleted.
   - The workspace, the binders' path dependencies, `.cargo/config.toml`
     aliases and the shared target directory follow.
2. **The Python fold.**
   - `molrs-python/`'s pyo3 crate becomes `molpy/Cargo.toml` + `molpy/src/`,
     built as `molpy._core`. Every `#[pyclass(module = "molrs.…")]` and
     registered submodule name becomes `molpy.…`.
   - The ABI capsule names (`molrs.FrameRef/<major.minor>`, …) are molrs-ffi's
     and stay.
   - Each `molrs-python/python/molrs/<m>` merges into `molpy/molpy/<m>`. The
     molrs module is the base, and molpy's additions join it (`Am1BccModel`
     in `ff.charge`, the metric readers in `io.*`, the selectors in `core`, …).
     molpy's identity re-export shells disappear.
   - `_native.pyi` becomes `_core.pyi`; the `_protocol` and
     `_style_declaration` modules lose their underscore (D4).
   - `molpy/src/molpy/` becomes `molpy/molpy/` (D11).
   - `molpy/pyproject.toml` builds with maturin.
   - `molrs-python/tests` join `molpy/tests`, and `import molrs` becomes
     `import molpy as mp`.
   - `molpy.version.check_molrs_version` is deleted; the version comes from
     `[workspace.package].version`.
3. **Project level.**
   - molpy's `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`,
     `SECURITY.md`, `LICENSE`, `docs/` and `zensical.toml` move to the root.
   - molrs's README, root `docs/` and `molrs-python/docs` go, per D13.
4. **Gates and CI.**
   - `scripts/check.sh`: `python` builds and tests the molpy wheel; `types`
     checks `molpy/molpy/_core.pyi`; `docs` builds the one manual; the
     separate `molpy` and `molpy_docs` gates fold into those.
   - `test.yml`, `release.yml` and `nightly.yml` use the new paths;
     `release.yml` publishes `molcrafts-molpy` and no longer publishes
     `molcrafts-molrs` to PyPI.
   - The binder directories in the prek hooks, `dependabot.yml` and
     `.gitignore` follow.

**Status (2026-10-22): implemented locally on molrs `dev` (unpushed), for
operator review before the force-push.**
- molrs-python's `molpy/tests/test_bindings` join molpy's suite: 2473 tests
  pass, 0 skipped.
- Per D9 and D8, the binding suite's identity, absence, `__all__`-location,
  public-path and stub-parity tests and its OpenMM / LAMMPS-binary tests were
  deleted.
- try/except remains only at external boundaries (subprocesses, config files,
  file sniffing, JSON lines). The Pyodide-only names (`Publisher`, mrec) are
  gated on `sys.platform != "emscripten"`, not on an import failure (operator
  ruling).
- Known red until 04: the `mrec` gate, whose molrec adapter
  (`molrec/tests/molrs_adapter.py`) imports `molrs`.

**Acceptance.**

- `scripts/check.sh verify` passes (except `mrec`, until 04).
- `import molrs` fails in a fresh environment that has only the built
  `molcrafts-molpy` wheel.
- `rg -n 'molrs-(python|wasm|capi|cxxapi)|molrs_capi|molrs_cxxapi|molrs\._native'`
  finds only the changelog.
- `git ls-files | rg '(^|/)_[a-z][^/]*\.py$'` lists nothing but `_core`
  stubs.

**Risk.** This is one large breaking change. It is reviewed locally as a
series of commits before the force-push, and downstream stays on the last
release (0.19) until 04.

## 4. 04-downstream — every consumer moves to molpy; the minor release

- `import molrs` becomes `import molpy as mp` in molpack (`python/`), molvis
  (`python/`), molab, Atomiverse, molnex and molhub, and in the monorepo's
  `repos.toml` and `*-ci-align` mirrors.
- The molvis npm dependency `@molcrafts/molrs` becomes the molwasm package.
- molpack's capsule round trip runs against `molpy._core`. The capsule host
  changed, so this is a minor (ABI) bump.
- One `v*` tag publishes the crate, the npm package and `molcrafts-molpy`.
- Re-point the publishing pipelines: crates.io, npm and PyPI Trusted
  Publishing, and the prek link-static / verify-shared-dylib pair.

## 5. 05-repo-rename

- Archive or delete MolCrafts/molpy, then rename MolCrafts/molrs to
  MolCrafts/molpy.
- Re-point the PyPI / crates.io / npm Trusted Publishing repository
  bindings, the Cloudflare Pages project (root `.`), the fork remotes and the
  monorepo `repos.toml`.

## Settled questions (operator, 2026-10-22)

- The npm package is `@molcrafts/molwasm`.
- The C binding is `molc`. Its header is `molc.h`, and its whole C surface
  is renamed: functions `molc_*`, types `Molc*`, constants `MOLC_*`. There
  was no consumer outside vendored molrs copies, which follow in 04.
- Every LICENSE copy is the project's: "Copyright (c) 2025, MolCrafts".
