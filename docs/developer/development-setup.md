# Development Setup

After following this page you will have a working local environment with editable install, pre-commit hooks, and passing tests.

## Prerequisites

You need Python 3.12 or newer, Git, [uv](https://docs.astral.sh/uv/), and the
Rust toolchain ([`rustup`](https://rustup.rs/)): on `dev`, molpy builds
[molrs](molrs-backend.md), its Rust compute core, from source. molrs pins the
toolchain channel and components in its `rust-toolchain.toml`, so no manual
component setup is required. Everything else is installed by the commands
below.


## Quick setup

Clone molpy next to its partners — molrs, mollog and molcfg — sync with dev
dependencies (this compiles molrs's Python extension), and run the test
suite.

```bash
git clone https://github.com/MolCrafts/molrs.git
git clone https://github.com/MolCrafts/mollog.git
git clone https://github.com/MolCrafts/molcfg.git
git clone https://github.com/MolCrafts/molpy.git
cd molpy
uv sync --locked --extra dev
pre-commit install --hook-type pre-commit --hook-type pre-push
uv run --locked --extra dev python -m pytest tests/ -n auto
```

If all tests pass, the environment is ready.


## Partners come from their tips

On `dev`, molpy is built against its partners' tips, not against releases:
`pyproject.toml`'s `[tool.uv.sources]` builds `molcrafts-molrs` from the
sibling `../molrs/molrs-python` (molrs's `dev`), and `molcrafts-mollog` /
`molcrafts-molcfg` — the logging and configuration libraries the engines and
wrappers use — from `../mollog` / `../molcfg` (their `master`; they have no
`dev`). The wheel itself still declares only ranges: molrs's
**major.minor** (`>=X.Y.0,<X.(Y+1)`), mollog's and molcfg's major
(`>=A.B.C,<A+1`); pip ignores the table, and a release is tested against all
three from PyPI (see [Release Process](release-process.md)). Import-time
`check_molrs_version` accepts patch drift inside molrs's minor.

uv rebuilds molrs when the sibling's commit moves; after uncommitted changes
to the molrs Rust source, rebuild by hand:

```bash
uv sync --extra dev --reinstall-package molcrafts-molrs
```

`uv.lock` records each partner's package metadata (its version and
dependencies), not a commit. When a partner's tip changes that metadata,
`uv lock --check` fails;
relock in a commit of its own with the recipe in `.github/partners.env`. See
the [molrs build-from-source guide](https://docs.molcrafts.org/molrs/getting-started/installation/)
for the native-crate and WASM build targets.


## Partners

CI and the pre-push hooks never build against your siblings' working trees.
`.github/partners.env` names each partner's branch (`MOLRS_REF=dev`,
`MOLLOG_REF=master`, `MOLCFG_REF=master`: partners are tracked, not pinned),
and `scripts/partners.py` resolves each -- for CI (`partners.py fetch`, into
`../molrs`, `../mollog`, `../molcfg`) and for the hooks (`partners.py run`, a
copy of this tree next to the resolved partners) alike -- to the first of:

1. the partner's branch named like the one being built (CI: the pushed
   branch or a pull request's head branch; locally: the checked-out branch),
   looked up first on the fork the build comes from (`<owner>/<partner>`,
   where `<owner>` owns the pull request's head repository or the repository
   CI runs in; in a git hook, the remote being pushed to), then on
   MolCrafts/<partner>;
2. outside CI only, that branch in your sibling clone `../<partner>`, when it
   has one and neither remote does yet;
3. the partner's `<NAME>_REF` on MolCrafts (`dev` for molrs, `master` for
   mollog and molcfg).

A molpy change that needs a partner change lands as two same-named branches,
never by skipping a gate: create the same branch (say `converge/x`) in both
checkouts; push both to your forks, never to MolCrafts (molpy's gates take
molrs's branch from your fork, or from your sibling before it is pushed); each
push runs the full CI tier on your fork, and molpy's run resolves molrs's
`converge/x` there; only once both forks
are green, open the pull requests into MolCrafts `dev`, land molrs's, then
molpy's (never a red one), and delete the branches.


## Documentation preview

The doc site is built with [Zensical](https://zensical.org) (Material for MkDocs'
successor), configured by `zensical.toml` at the repo root. Install the doc
extras and start a local preview server from the repo root.

```bash
uv sync --extra doc
uv run zensical serve
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.

User-guide notebooks are pre-rendered to Markdown (Zensical does not run notebooks
at build time). After editing one, regenerate its page with
`python scripts/render_notebooks.py`.


## External tools

LAMMPS and AmberTools are optional for *using* MolPy offline recipes; packing uses molpack (`pip install molcrafts-molpack`).
they are **not** part of the unit suite. Wrappers and engines are tested with
mocks and script literals. Doc blocks that would shell out declare
`# docs: skip`.


## Common commands

```bash
ruff format --check src tests             # check formatting
ruff format src tests                     # auto-format
ruff check src                            # lint source tree
uv run --locked --extra dev python -m pytest tests/ -n auto   # the CI test command
pre-commit run --all-files                # all pre-commit hooks (see CONTRIBUTING.md "Hooks")
zensical build                            # build static doc site into site/
```


## Troubleshooting

If imports fail after pulling new code, run `uv sync --extra dev` (add `--reinstall-package molcrafts-molrs` when molrs changed). Docs site build is `uv sync --extra doc` + `uv run zensical build` (theme + mkdocstrings only — no notebook/matplotlib stack). If formatting checks fail in CI, run `ruff format src tests` locally before pushing.
