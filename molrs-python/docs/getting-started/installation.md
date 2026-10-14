# Installation

molrs is published as separate packages for Rust, Python, and npm. The packages
share the same core model, but each one follows the conventions of its host
ecosystem.

| Environment | Package | Command |
| --- | --- | --- |
| Rust | `molcrafts-molrs` | `cargo add molcrafts-molrs --features full` |
| Python | `molcrafts-molrs` | `python -m pip install molcrafts-molrs` |
| JavaScript / TypeScript | `@molcrafts/molrs` | `npm install @molcrafts/molrs` |

The Python import name is `molrs`. The npm package uses the scoped name
`@molcrafts/molrs`, while the generated TypeScript module exports classes such
as `Frame`, `Block`, `Box`, and analysis helpers directly.

## Python nightly builds

Bleeding-edge **Python** wheels are published to a separate PyPI project,
`molcrafts-molrs-nightly`, on every push to the `nightly` branch. This is
Python-only — the Rust crates (crates.io) and the npm package ship exclusively
from `v*` release tags and have no nightly channel.

Each build is versioned `X.Y.Z.devN` (a PEP 440 dev release), so opt in with
`--pre`:

```bash
pip install --pre molcrafts-molrs-nightly
```

The nightly wheel imports as `molrs`, exactly like the stable one, so the two
**cannot be installed at the same time**. Use a dedicated virtual environment
for nightly testing.

## Verify the Environment

=== "Python"

    ```bash
    python -m pip install molcrafts-molrs
    python - <<'PY'
    import molrs

    ir = molrs.io.smiles.SmilesIr("O")
    print("components:", ir.n_components)
    print("atoms:", ir.to_atomistic().n_atoms)
    PY
    ```

    The import name is intentionally shorter than the package name. If
    `import molrs` fails after installation, check that the interpreter running
    the script is the same interpreter used by `python -m pip`.

=== "Rust"

    ```bash
    cargo new molrs-smoke
    cd molrs-smoke
    cargo add molcrafts-molrs --features full
    ```

    In `src/main.rs`:

    ```rust
    use molrs::io::read_smiles_str;

    fn main() -> Result<(), Box<dyn std::error::Error>> {
        let mol = read_smiles_str("O")?;
        println!("atoms: {}", mol.n_atoms());
        Ok(())
    }
    ```

=== "TypeScript"

    ```bash
    npm install @molcrafts/molrs
    ```

    In a bundler that loads WebAssembly modules (Vite, webpack, …):

    ```ts
    import { SmilesIr } from "@molcrafts/molrs";

    console.log(SmilesIr.parse("O").nComponents);
    ```

## Source Builds

### Prerequisites

Source builds need the Rust toolchain, installed via
[`rustup`](https://rustup.rs/). The repository pins one exact rustc / rustfmt /
clippy release plus the `wasm32-unknown-unknown` target in
`rust-toolchain.toml`, so `rustup` provisions them automatically on the first
build inside the checkout — no manual `rustup component add` needed.

### Native crates

Clone the repository and build through the committed `cargo mrs-*` aliases
(`.cargo/config.toml`). They pin one feature set, so every command reuses the
same build:

```bash
git clone https://github.com/MolCrafts/molrs.git
cd molrs
cargo mrs-build          # compile the library
cargo mrs-test           # unit tests
cargo mrs-doctest        # rustdoc examples
scripts/check.sh all     # every CI gate
```

The binding crates (`molrs-python`, `molrs-wasm`, `molrs-capi`,
`molrs-cxxapi`, `molrs-ffi`) are standalone workspaces; build each with
`cargo build --manifest-path <crate>/Cargo.toml`.

### Python extension

For local Python development, install the extension module in editable form
with [maturin](https://www.maturin.rs/). This compiles the `molrs-python` PyO3
crate and installs it into the active virtualenv as `molrs`:

```bash
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release
python -c "import molrs; print(molrs.io.smiles.SmilesIr('O').n_components)"
```

### WASM / npm

For the npm package, build the same bundler target used by release publishing:

```bash
cd molrs-wasm
wasm-pack build --release --target bundler --scope molcrafts --out-name molrs
```

### Documentation

For documentation work, build the local site from `molrs-python` after the
Python extension is installed (Zensical config is `molrs-python/zensical.toml`):

```bash
cd molrs-python
pip install ".[doc]"    # builds the extension and installs zensical
zensical build --clean  # writes ./site
```

This is the command sequence the hosted site is built with.

## Version Boundaries

Rust, Python, and npm packages are released separately but generated from the
same repository. If examples behave differently across languages, first check
the package versions. The documentation site follows the repository `master`
branch, while crates.io, PyPI, npm, and docs.rs describe released artifacts.

**Consumers (e.g. molpy)** pin the shared **major.minor** line
(`molcrafts-molrs>=0.18.0,<0.19`), not an exact patch. Patch may drift.

## Browser (Pyodide)

PyPI publishes an Emscripten wheel. In a Pyodide session:

```python
import micropip
await micropip.install("molcrafts-molrs")
import molrs
```

This is the **Python** extension for the browser. For the JS API used by
MolVis, install `@molcrafts/molrs` from npm instead.
