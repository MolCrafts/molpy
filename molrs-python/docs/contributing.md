# Contributing to Documentation

The documentation system has one central invariant: public API prose starts in
Rust `///` comments. Zensical pages may explain workflows and concepts, but
reference pages should inject generated API docs or link to generated API docs
instead of copying signatures by hand.

For Python, keep `molrs-python/python/molrs/_native.pyi` synchronized with the
PyO3 module exports. `molrs-python/tests/test_stub_parity.py` is the freshness
guard and runs in `scripts/check.sh python`: it fails when a compiled export is missing from
the stub, and when a parameter name differs between the stub and the compiled
signature.

For WASM, build declarations with the same `wasm-pack` flags used by npm
publishing. The generated `pkg/` directory is ignored and must not be committed.

Local documentation loop. The Zensical config lives at
`molrs-python/zensical.toml`, so build and serve from that directory (its
`[doc]` extra pulls in Zensical plus the mkdocstrings Python handler):

```bash
cd molrs-python
pip install -e ".[doc]"                 # zensical + mkdocstrings[python]
maturin develop                         # build the molrs module so the API reference can introspect it
zensical build                          # reads ./zensical.toml, writes ./site
zensical serve -a localhost:8000        # live preview
```

## Hooks

Every gate is a `scripts/check.sh <gate>` call, and `.pre-commit-config.yaml`
runs the same calls as git hooks. Every check runs exactly once: static checks
run only in the hooks (CI does not repeat them), and each test suite runs once
per OS. Install both hook types once with
`prek install` (or `pre-commit install`). **Never `git commit --no-verify` or
`git push --no-verify`, and never merge a red pull request**: a hook that
fails is a CI job that would have failed.

| Stage | Hooks |
| --- | --- |
| pre-commit | the CI profile's commit stage: `workflows` (the CI scheme), `fmt`, `ruff`, `os-cfg`, `partners`. Nothing compiles. |
| pre-push | `scripts/check.sh verify`: the commit stage once on `--all-files`, then `clippy doc test ffi cxx ext capi python wasm mrec docs`, each once, without file filters. |
| CI (`test.yml`) | the tests only, on native Linux/macOS/Windows runners. A local host only proves its own OS. |

- `clippy` — `-D warnings` on the core (`cargo mrs-clippy`, the one `full`
  feature set: no per-feature matrix) and each binder; `molrs-wasm` is linted
  for wasm32 only.
- `python` — one dev-profile wheel (no `--release`, no LTO), installed and
  tested once; `mrec` and `docs` install that same wheel.
- `wasm` — `wasm-pack test --node`, which compiles the crate once and runs the
  suite. The release npm build is `release.yml`'s.

- `ruff` — `ruff.toml`'s one rule, PLW1514: text-mode file I/O (`open`,
  `read_text`, `write_text`) names its encoding. Unnamed, Python decodes with
  the locale, cp1252 on Windows, so the read fails only on CI's Windows leg.
  Ruff sees only receivers it can type (`Path(...)`, annotated names); spell
  `encoding="utf-8"` on every text read and write regardless.
- `partners` — every partner in `.github/partners.env` resolves (see
  [Partners](#partners)), no path dependency points where CI has no checkout,
  and no workflow spells a partner commit of its own.
- `mrec` — molrec's conformance suite through `molrs.io.mrec`, exactly as
  `test.yml`'s `test / mrec` runs it: molrec fetched at its resolved commit into a temp
  dir (never your sibling's working tree), the wheel already tested in this push (or `maturin develop` when run on
  its own), in a fresh Python 3.12 venv, `scripts/ci-conformance.py`. Any case that does not pass
  fails it.
- `docs` — the site using the locked `dev,doc` tools and the tested wheel,
  then `zensical build --clean`, with `--strict`, so an mkdocstrings
  reference to a symbol that does not exist fails the push.
- `os-cfg` — `std::os::unix` / `std::os::windows` only directly under a
  `#[cfg(...)]` attribute (line above or same line). A use inside an item that
  is gated further up is fine: end that line with `// os-gated: <why>`.
- Every cargo, maturin and wasm-pack call is `--locked`, every uv call runs on
  Python 3.12 against the committed lock, and the compiler is the one
  `rust-toolchain.toml` pins — the same as CI.

**Dispatch on the MolCrafts cluster.** The checkouts' shared `core.hooksPath`
runs the hooks in place on the login node and sets `MOLCRAFTS_HOOK_RUNNER`.
`scripts/check.sh` keeps `fmt`, `ruff` and `partners` in place and hands every other
gate to that runner, which runs it on a compute node (it reuses the
`$USER-hooks` allocation, or requests one and fails after 20 minutes without a
node — it never passes a gate it did not run). So a commit never waits for
Slurm, and every push runs the complete gate catalogue. Anywhere else
the variable is unset and every gate runs locally, exactly as CI runs it.

For the complete local gates, install CMake and a C++20 compiler, Node 24 or newer (the latest release works) and
wasm-pack 0.15.0, in addition to uv and rustup. On Windows,
use Git Bash for shell hooks and a C++ compiler matching the Rust target
(Visual Studio Build Tools for MSVC). The C API test links the import library
and copies its DLL beside the test executable. Native CI verifies that path;
WSL only verifies Linux behavior.

## CI

CI runs only what a laptop cannot: the cross-OS tests (`test.yml`) and the
release artifacts (`release.yml`). Formatting, linters, clippy, rustdoc, the
workflow scheme and the partner pins were proven by the hooks before the push
and are not run again. Every job is a `scripts/check.sh` call. Every push of
any branch runs `test`, on a fork as on MolCrafts. A pull request into `dev` or `master` runs them again unless it is
a pull request inside a fork (that was already run by its push). Those
decisions (tier, fork or upstream, the duplicate pull request) are made in
one place: every workflow's first job, `<file> / context`, runs
the pinned `MolCrafts/molcrafts-ci/actions/ci-context`, and every other job reads
its outputs.

| workflow | feature-branch push to MolCrafts | everything else: any push to a fork, `dev`/`master` on MolCrafts, pull requests, tags, dispatches | upstream only |
| --- | --- | --- | --- |
| `test.yml` | fast: `test / rust` (`test`), `test / python (ubuntu-latest)`, `test / docs` | full: `test / rust` (+ `ffi cxx ext`), `test / python` on Linux, macOS and Windows, `test / docs`, `test / capi` on Linux and Windows, `test / wasm`, `test / mrec` | Cloudflare Pages deploys the site from MolCrafts |
| `nightly.yml` | — | — | nightly: tests, coverage and conformance snapshots to molcrafts-ci; a `nightly` branch push: wheels to `molcrafts-molrs-nightly` |
| `release.yml` | — | dispatch: dry run (builds, uploads nothing) | `v*` tag: a guard that the commit's `test.yml` is green (no lint or test run of its own), then crates.io, npm, PyPI, GitHub Release (`docs/releasing.md`) |

So a fork branch gets the full tier on its push: push to your fork, wait for
green, then open the pull request into MolCrafts `dev`. Branches pushed to
MolCrafts itself (Dependabot's) get the fast tier, and their pull requests the
full one. The `require-green-ci` (`dev`) and `protect-master` rulesets require
`test / context` and the full tier's jobs. Shared setup is
pinned `MolCrafts/molcrafts-ci/actions/<name>` (`setup-rust`, `setup-python`);
only `setup-wasm` is molrs's own, in `.github/actions/`.

`test / python (ubuntu-latest)` uploads the wheel it built and tested;
`test / docs` and `test / mrec` download that file (`MOLRS_TESTED_WHEEL`) and
compile nothing. `test / rust` and `test / python` run their gates as
`scripts/check.sh --report .ci-out <gate>`, which also writes what the tests ran
(`cargo-test.log`; `junit.xml` and the Python layer's `coverage.json`).
the pinned `MolCrafts/molcrafts-ci/actions/report` turns them into a table of passed,
failed and skipped counts and line and branch coverage, in the run and pull
request summary. Test thresholds remain in their gates. Report generation
failures emit warnings; they do not change the underlying test result.

## Partners

molrs is judged against molrec's conformance suite (`mrec`).
`.github/partners.env` pins molrec to a full commit and pins the shared
resolver to `CI_REF`. `scripts/partners.py` runs that shared resolver through
uv. Local hooks, branch CI, tags and nightly fetch the same declared commit;
neither branch names nor sibling working trees override it.

For a coordinated change, push the partner commit, update its repository/SHA
in the manifest, then run `scripts/check.sh verify`. A release keeps those
same pins. Updating a dependency is a reviewed source change.

Pre-push runs every local gate once. Clippy only lints; binder gates only
test. CI runs the tests again only on the operating systems the pushing host
is not; the crates.io tarball is verified once, by `cargo publish` in
`release.yml`.
