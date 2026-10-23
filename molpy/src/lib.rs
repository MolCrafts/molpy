//! Python bindings for the molrs molecular simulation library.
//!
//! This crate builds the private native module `molpy._core`; the Python
//! package `molrs` (`python/molrs`) names every symbol after its Rust owner.
//! The package root also re-exports the core data classes a user handles
//! directly (`molrs.Frame is molpy.core.Frame`); nothing else has a second path:
//!
//! | binding                 | Rust owner          | Python module        |
//! |-------------------------|---------------------|----------------------|
//! | [`core`]                | `molrs::core`       | `molpy.core`         |
//! | [`op`]                  | `molrs::op`         | `molpy.op`           |
//! | [`perceive`]            | `molrs::perceive`   | `molpy.perceive`     |
//! | [`io`]                  | `molrs::io`         | `molpy.io`           |
//! | [`ff`]                  | `molrs::ff`         | `molpy.ff.*`         |
//! | [`optimize`]            | `molrs::optimize`   | `molpy.optimize`     |
//! | [`md`]                  | `molrs::md`         | `molpy.md`           |
//! | [`conformer`]           | `molrs::conformer`  | `molpy.conformer`    |
//! | [`builder`]             | `molrs::builder`    | `molpy.builder`      |
//! | [`compute`]             | `molrs::compute`    | `molpy.compute`      |
//! | [`stream`]              | `molrs::stream`     | `molpy.stream`       |
//!
//! Every subsystem binding registers its own classes and functions through
//! its `register`. Most land flat on `_core`; a namespace with vocabulary of
//! its own is a `_core` submodule ([`add_submodule`]: `op`, `md`, `ff.ir` as
//! `ir`, `io.mrec` as `mrec`, and core's `keys` / `schema` / `constants`).
//!
//! Every class and function names its public path as its `__module__`, so
//! `repr`, pickle and the docs name it the way users import it: a class by
//! `#[pyclass(module = "molpy.…")]`, a flat function by [`add_function`], and
//! everything on a submodule by that submodule's name. Cross-cutting plumbing
//! has one home each: exceptions and Rust-error mapping in [`error`], the
//! pickle protocol in [`pickle`], path arguments in [`path`], array arguments
//! in [`array`].
//!
//! # Float Precision
//!
//! Every floating-point array crosses as `f64` (numpy `float64`): molrs fixes
//! `F = f64`, and there is no precision feature.

use pyo3::prelude::*;

mod array;
mod error;
mod pickle;

mod builder;
mod compute;
mod conformer;
mod core;
mod ff;
mod io;
mod lattice;
mod md;
mod op;
mod optimize;
mod perceive;
mod stream;
mod symmetry;
mod virtual_sites;

/// The FFI ABI handshake token of this build.
///
/// Returns ``(abi_line, version, frameref_capsule_name, forcefield_capsule_name,
/// regionref_capsule_name)``
/// — e.g. ``("0.19", "0.19.0", "molrs.FrameRef/0.19+<fingerprint>", …)``, the
/// fingerprint naming the compiler and handle layout of this build.
///
/// A downstream extension that exchanges ``molrs_ffi`` handle capsules with
/// this wheel (e.g. molpack) calls this once at import and compares
/// ``abi_line`` against the line of the molrs it statically embeds; a mismatch
/// is raised as a clear ``ImportError`` instead of surfacing later as a
/// capsule-name ``ValueError`` (or, before capsule names were versioned,
/// undefined behavior). The capsule names also carry the build fingerprint,
/// so two builds exchange handles only when compiler and layout agree too
/// (`molrs-ffi`'s layout snapshot test is the gate on the layout).
#[pyfunction]
fn _ffi_abi_token() -> (&'static str, &'static str, String, String, String) {
    (
        molrs_ffi::abi::abi_line(),
        env!("CARGO_PKG_VERSION"),
        molrs_ffi::abi::frameref_capsule_name()
            .to_string_lossy()
            .into_owned(),
        molrs_ffi::abi::forcefield_capsule_name()
            .to_string_lossy()
            .into_owned(),
        molrs_ffi::abi::regionref_capsule_name()
            .to_string_lossy()
            .into_owned(),
    )
}

/// Add the `#[pyfunction]` `f` to `m` with `module`, the Python module it is
/// re-exported from, as its `__module__` — what `#[pyclass(module = …)]` is
/// for a class.
pub(crate) fn add_function(
    m: &Bound<'_, PyModule>,
    module: &str,
    f: Bound<'_, pyo3::types::PyCFunction>,
) -> PyResult<()> {
    f.setattr("__module__", module)?;
    m.add_function(f)
}

/// Add a native submodule reached as `parent.<attr>` and named `path`, its
/// public Python module, so each function `register` adds to it reports
/// `__module__ == path`.
pub(crate) fn add_submodule(
    parent: &Bound<'_, PyModule>,
    attr: &str,
    path: &str,
    register: impl FnOnce(&Bound<'_, PyModule>) -> PyResult<()>,
) -> PyResult<()> {
    let module = PyModule::new(parent.py(), path)?;
    register(&module)?;
    parent.add(attr, module)
}

/// The native module, `molpy._core`: every subsystem's bindings.
#[pymodule]
#[pyo3(name = "_core")]
fn molpy_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(_ffi_abi_token, m)?)?;
    m.add_function(wrap_pyfunction!(pickle::_restore_pickled_state, m)?)?;

    core::register(m)?;
    perceive::register(m)?;
    io::register(m)?;
    ff::register(m)?;
    optimize::register(m)?;
    conformer::register(m)?;
    builder::register(m)?;
    compute::register(m)?;
    stream::register(m)?;

    add_submodule(m, "op", "molpy.op", op::register)?;
    add_submodule(m, "md", "molpy.md", md::register)?;
    Ok(())
}
