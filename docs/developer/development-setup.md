# Development Setup

After following this page you will have a working local environment with editable install, pre-commit hooks, and passing tests.

## Prerequisites

You need Python 3.12 or newer, Git, [uv](https://docs.astral.sh/uv/), and the
Rust toolchain ([`rustup`](https://rustup.rs/)): molpy builds
its [Rust core](rust-core.md) from source. The repository pins
the toolchain channel and components in `rust-toolchain.toml`, so no manual
component setup is required. Install CMake/CTest, Node 24 or newer, wasm-pack 0.15.0 and Binaryen 133
for the native binding/package gates; doctor checks prerequisites.


## Quick setup

The repository holds the whole molpy project: the Rust core (`crates/molrs`,
`crates/molrs-ffi`) and one directory per language binding — `molpy/`
(Python), `molwasm/`, `molc/`, `molcxx/`. The Python binding is a maturin
project: `molpy/Cargo.toml` + `molpy/src/` build the native module
`molpy._core`, and `molpy/python/molpy/` is the package.

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
uv tool install "molcrafts-ci @ git+https://github.com/MolCrafts/molcrafts-ci@$(sed -n 's/^CI_REF=//p' .github/partners.env)"
molci doctor --install-hooks
uv --directory molpy sync --extra dev   # builds molpy in place (editable)
cd molpy && uv run --no-sync pytest     # the Python suite
```

Run `pytest`, not `python -m pytest`. The suite imports the installed wheel.
The package source is `molpy/python/molpy/`.


## Gates

`.pre-commit-config.yaml` is the single hook entry point. Commit runs the
cheap stage; push requires all applicable native gates. MolCrafts CI's pinned
`rust-python` registry owns ordering, tool requirements and successful-result
caching. Unknown source paths invalidate every cached success. Source changes,
lockfiles, platform/tool versions and dependency artifacts enter cache keys.
A missing tool or unavailable runner fails with a reproduction command.

Use `molci check` for commit checks, `verify` for native validation
with valid successes reused, and `verify --full` to execute the entire suite.
`run GATE` and `explain GATE` use the same registry. `molci verify --full`
is the full-suite convenience command. The Python gate builds one wheel;
types, Molrec, executable documentation and package inspection reuse its exact
bytes. `molpy/uv.lock` pins their shared environments, including the conformance group.

The installed push transport preserves every Git ref before pre-commit reads
stdin. Pushed refs must resolve to the verified HEAD and source must be clean;
use separate clean worktrees for distinct source snapshots. Reinstall through
`doctor --install-hooks` after running another hook install command. Hooks can
be bypassed, so external PRs receive independent native verification. The trusted
personal development path receives every platform lane; master receives the full
independent native suite. A normal fork push starts no redundant GitHub compilation.

Use `molci partners pin-ci SHA` when updating shared infrastructure; it updates
hooks, Actions and the CI package source together, then refreshes the lock.

## Developing across operating systems

Use the pinned Rust toolchain and Python 3.12 for the local gates. On Windows,
run the shell hooks in Git Bash with Python and uv on PATH; a WSL run proves
Linux behavior, so native Windows CI must still pass. The CI matrix separately
checks all three native OSes, one abi3 wheel per OS built on Python 3.12.

Write text with `encoding="utf-8"`, use `pathlib` for Python paths, and preserve
LF endings through `.gitattributes`. Avoid filenames that differ only by case
and assumptions about `/tmp`, executable suffixes or a venv's `bin` directory.
Use `.venv/Scripts` and `.exe` on Windows. A container can align Linux tools,
but cannot validate Windows or macOS native bindings.

## Documentation preview

The doc site is built with [Zensical](https://zensical.org) (Material for MkDocs'
successor), configured by `zensical.toml` at the repository root. Install the doc
extras into molpy's environment and start a local preview server from the
repository root. The server runs from `molpy/`, so `DOCS_DIR` tells the theme
where a figure's `$file` data lives.

```bash
DOCS_DIR="$PWD/docs" uv --directory molpy run --locked --extra doc zensical serve -f ../zensical.toml
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.


## External tools

LAMMPS and AmberTools are optional programs for *using* MolPy's engines and
`mp.tool.ambertools`; packing uses molpack (`pip install molcrafts-molpack`).
None of them is part of the test suite: engines and the AmberTools programs
are tested with mocks and script literals. Doc blocks that would shell out
declare `# docs: skip — <reason>`.


## Common commands

```bash
molci check                 # formatting and lint, Rust and Python
molci run python
molci run types             # molpy wheel + suite, stub check
molci run docs                     # this manual, --strict
molci verify --full                   # the pre-push gate: every check once
cargo mrs-test                            # Rust core unit tests
cd molpy && uv run --no-sync ruff format python tests   # auto-format Python
```


## Troubleshooting

If imports fail after pulling new code, rebuild molpy in place: `uv --directory molpy sync --extra dev --reinstall-package molcrafts-molpy` (the Rust side changed). The manual builds with `molci run docs`. If formatting checks fail, run `uv --directory molpy run --no-sync ruff format python tests` before pushing.
