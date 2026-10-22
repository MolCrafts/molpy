//! Python bindings for `molrs::ff`, one file (or directory) per Rust
//! submodule, each the binding of its owner and exposed at the Python module
//! of the same name:
//!
//! | binding            | Rust owner           | Python module          |
//! |--------------------|----------------------|------------------------|
//! | [`forcefield`]     | `ff::forcefield`     | `molpy.ff.forcefield`  |
//! | [`potential`]      | `ff::potential`      | `molpy.ff.potential`   |
//! | [`typifier`]       | `ff::typifier`       | `molpy.ff.typifier`    |
//! | [`charge`]         | `ff::charge`         | `molpy.ff.charge`      |
//! | [`ir`]             | `ff::ir`             | `molpy.ff.ir`          |
//! | [`style_registry`] | `ff::style_registry` | `molpy.ff.style_registry` |
//! | [`compile`]        | `ff::compile`        | `molpy.ff.compile`     |
//! | [`params`]         | `ff::params`         | `molpy.ff.params`      |
//! | [`clpol_scaling`]  | `ff::clpol_scaling`  | `molpy.ff.clpol_scaling` |
//!
//! No file format is here: force-field files, like every other file, are
//! `molpy.io`'s (`crate::io`).

pub mod charge;
pub mod clpol_scaling;
pub mod compile;
pub mod forcefield;
pub mod ir;
pub mod params;
pub mod potential;
pub mod style_registry;
pub mod typifier;

use pyo3::prelude::*;

/// Register every `molrs::ff` binding: the flat classes and functions on the
/// native module, `molpy.ff.ir`'s vocabulary as its `ir` submodule and
/// `molpy.ff.style_registry`'s registry as its `style_registry` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    forcefield::register(m)?;
    potential::register(m)?;
    typifier::register(m)?;
    charge::register(m)?;
    params::register(m)?;
    clpol_scaling::register(m)?;
    compile::register(m)?;
    crate::add_submodule(m, "ir", "molpy.ff.ir", ir::register)?;
    crate::add_submodule(
        m,
        "style_registry",
        "molpy.ff.style_registry",
        style_registry::register,
    )
}
