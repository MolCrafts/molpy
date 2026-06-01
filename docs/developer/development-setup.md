# Development Setup

After following this page you will have a working local environment with editable install, pre-commit hooks, and passing tests.

## Prerequisites

You need Python 3.12+, Git, and pip. Everything else is installed by the setup script below.


## Quick setup

Clone the repository, create a virtualenv, install in editable mode with dev dependencies, and run the test suite to confirm everything works.

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v -m "not external"
```

If all tests pass, the environment is ready.


## Building molrs from source

The quick setup above resolves [molrs](molrs-backend.md) — molpy's required
Rust compute core — from the published `molcrafts-molrs` wheel on PyPI. That is
the right path for most molpy development.

If you are changing the Rust core *and* molpy together, build molrs editable
from a local checkout instead. molrs ships its Python bindings as a
[maturin](https://www.maturin.rs/) project, so this step needs the Rust
toolchain — install it via [`rustup`](https://rustup.rs/); molrs pins the
toolchain channel and components in its `rust-toolchain.toml`, so no manual
component setup is required inside the checkout.

```bash
# in a sibling checkout next to molpy
git clone https://github.com/MolCrafts/molrs.git
cd molrs
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release   # installs `molrs` editable into the venv

# back in molpy, the editable install now resolves the local molrs
cd ../molpy
pip install -e ".[dev]"
python -c "import molrs, molpy as mp; print(mp.Box.cubic(10.0), isinstance(mp.Box.cubic(10.0), molrs.Box))"
```

Re-run `maturin develop` after any change to the molrs Rust source to recompile
the extension. See the
[molrs build-from-source guide](https://molcrafts.github.io/molrs/getting-started/installation/)
for the native-crate and WASM build targets.


## Documentation preview

The doc site uses MkDocs with Material theme. Install the doc extras and start a local preview server.

```bash
pip install -e ".[doc]"
mkdocs serve
```

The site is at `http://localhost:8000`. Changes to `.md` files are reflected immediately.


## External tools

Some tests and workflows require external executables that are not Python packages: LAMMPS, Packmol, and AmberTools. These are not needed for core development. Tests that depend on them are marked `@pytest.mark.external` and excluded from the default test run via `-m "not external"`.

If you have one of these tools installed and want to run its tests:

```bash
pytest tests/ -v                          # all tests including external
pytest tests/ -v -k "lammps"              # only LAMMPS-related tests
```


## Common commands

```bash
ruff format --check src tests             # check formatting
ruff format src tests                     # auto-format
ruff check src                            # lint source tree
pytest tests/ -v -m "not external"        # local test suite
pytest --cov=src/molpy tests/ -v          # with coverage
pre-commit run --all-files                # all pre-commit hooks
mkdocs build                              # build static doc site
```


## Troubleshooting

If imports fail after pulling new code, reinstall the editable package: `pip install -e ".[dev]"`. If notebook cells fail during doc build, install doc dependencies: `pip install -e ".[doc]"`. If formatting checks fail in CI, run `ruff format src tests` locally before pushing.
