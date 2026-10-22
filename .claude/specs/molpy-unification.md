---
title: molpy unification — one project, one Python package, molrs as its native core
slug: molpy-unification
status: approved
created: 2026-10-22
chain: molpy-unification (01-user-paths → 02-repo-merge → 03-api-fold → 04-binder-rename → 05-repo-rename)
repos: molpy (01), molrs (02–05), molpack / molvis / molab / Atomiverse / molnex / molhub / monorepo (03, 04 consumers)
---

# molpy unification

## Summary

molrs and molpy already ship as one product in two repositories. Tags v0.15
through v0.19 match one to one. molpy pins `molcrafts-molrs>=0.X.0,<0.X+1`, and
56 of its 89 source files import molrs. Every Rust change costs a wheel
rebuild before molpy's tests see it. Every release costs two PRs, two tags and
a pin bump.

The end state mirrors numpy: one project named **molpy**, one Python package
and one native core.

| Today | End state |
|---|---|
| repo `MolCrafts/molrs` + repo `MolCrafts/molpy` | repo `MolCrafts/molpy` (the molrs repo, renamed) |
| Python dist `molcrafts-molrs` (`import molrs`, native `molrs._native`) | gone, no shim |
| Python dist `molcrafts-molpy` (pure Python) | `molcrafts-molpy`, a maturin abi3 wheel; native module `molpy._core` |
| `molrs-python/` | folded into the molpy package; directory deleted |
| `molrs-wasm/` (npm `@molcrafts/molrs`) | `molwasm/` |
| `molrs-capi/` (`molcrafts-molrs-capi`) | `molc/` |
| `molrs-cxxapi/` (`molcrafts-molrs-cxxapi`) | `molcxx/` |
| `molrs-ffi/` | unchanged |
| Rust core crate `molcrafts-molrs` (`molrs`) | unchanged |

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

## Chain

No release happens inside 02. 03 and its downstream migrations ship together
as one minor release.

| Link | Scope | Depends |
|---|---|---|
| 01-user-paths | §1, in the molpy repo. | — |
| 02-repo-merge | §2. | 01 |
| 03-api-fold | §3. Breaking change; minor bump; downstream in the same minor. | 02 |
| 04-binder-rename | §4. | 03 |
| 05-repo-rename | §5. | 03 |

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

## 3. 03-api-fold — one Python package, native `molpy._core`

**Goal.** molrs's Python API becomes molpy's. Each mirror module is the place
where its names are defined, not a re-export.

**Scope.**

- The pyo3 crate builds `molpy._core` (maturin `module-name = "molpy._core"`,
  `python-source` pointing at the molpy package). `molcrafts-molpy` becomes an
  abi3 maturin wheel. Its version comes from `[workspace.package].version`.
- Fold `molrs-python/python/molrs/*` into molpy's subsystems of the same name.
  Rename `_protocol.py` and `_style_declaration.py` under D4.
- Delete `molrs-python/`, `molpy.version.check_molrs_version`, the
  `molcrafts-molrs` Python dist, and every molpy-vs-molrs identity assertion
  (`test_public_paths.py`, `test_init.py`, R4's exception list).
- The ABI capsules keep their names (`molrs.FrameRef/<major.minor>`,
  `molrs.ForceFieldRef/<major.minor>`), which are defined by molrs-ffi. Their
  host module becomes `molpy._core`, and that is an ABI change: bump the minor.
- **Downstream, in the same minor (D2):**
  - molpack `python/`;
  - molvis `python/`;
  - molab;
  - Atomiverse (`python/` and its vendored `molrs/`);
  - molnex and molhub (pin bump);
  - the monorepo `repos.toml` and the `*-ci-align` mirrors.

**Acceptance.**

- `import molrs` fails in a fresh environment with only `molcrafts-molpy`
  installed.
- `rg -n 'import molrs|from molrs' --type py` is empty across all MolCrafts
  repos.
- molpack's capsule round-trip tests pass against `molpy._core`.

**Risk.** This is the one breaking release. Every downstream PR must be ready
before the tag. Rollback means re-tagging the previous minor; no shim is
offered.

## 4. 04-binder-rename

**Scope.**

- `molrs-wasm` → `molwasm`: directory, crate, npm package (today
  `@molcrafts/molrs`), and molvis's `@molcrafts/molrs` dependency and
  jsdelivr URLs.
- `molrs-capi` → `molc`: directory, crate, library and header names.
- `molrs-cxxapi` → `molcxx`: directory, crate, library and header names.
- `molrs-ffi` and the core crate are unchanged.
- Re-point npm and crates.io Trusted Publishing. Update the prek hooks
  (link-static / verify-shared-dylib) and every `--manifest-path`.

**Acceptance.**

- `rg 'molrs-(wasm|capi|cxxapi)'` is empty, apart from changelogs.
- `prek run --all-files --hook-stage pre-push` is green.

## 5. 05-repo-rename

**Scope.**

- Archive or delete the old `MolCrafts/molpy` repository, then rename
  `MolCrafts/molrs` to `MolCrafts/molpy`.
- Update the PyPI and crates.io Trusted Publishing repository bindings, the
  docs domain, the fork remotes (Roy-Kid) and the monorepo `repos.toml`.

**Acceptance.** A `v*` tag on the renamed repository publishes the crate, the
wheel and the npm package.

## Open questions

- The name of the pyo3 crate after `molrs-python/` is gone.
- The npm package name for molwasm (`@molcrafts/molwasm`?).
- The C header prefix for molc and molcxx.
