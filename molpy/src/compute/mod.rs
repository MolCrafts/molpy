//! Python bindings for `molrs::compute` (`molpy.compute`).
//!
//! Every entry of `ComputeSpec::all()` is one class here, under its core
//! name, and every class has the same two-step shape the core has:
//!
//! - `__init__` takes **configuration only** — the constructor arguments and
//!   `with_*` settings of the core type, by their spec names. An optional one
//!   is `None` in the signature, and the core setter is called only when it is
//!   given, so every default lives in the core.
//! - `compute(...)` takes **the data**, as keyword arguments named after the
//!   fields of the core `Input` struct. A compute whose core input is a slice
//!   of per-frame records (`Rdf`, `DistributionFunction`,
//!   `CombinedDistribution`, `SpatialDistribution`) takes one list per field.
//!
//! Each core output is a frozen class with read-only getters ([`py_result!`]),
//! and an output another compute consumes is passed back as that object
//! (`TimeCorrelation` into the fits, `SteinhardtResult` into `Pca`, …).
//!
//! Shared plumbing has one home each: the error mapping in [`error`], the
//! adapters from Python arguments to core inputs in [`args`], the output
//! conversion in [`output`], the descriptor classes in [`spec`] and the
//! configuration value types (`Axis`, `RegularGrid`, `Histogram`) in
//! [`values`].
//!
//! [`py_result!`]: output::py_result

mod args;
mod error;
mod output;
mod spec;
mod values;

mod cluster;
mod decomposition;
mod density;
mod dielectric;
mod diffraction;
mod distribution;
mod dynamics;
mod environment;
mod fitting;
mod hbond;
mod kinetic;
mod kmeans;
mod order;
mod pmft;
mod rdf;
mod shape;
mod spectroscopy;
mod time_series;
mod voronoi;

use pyo3::prelude::*;

/// Register every compute class on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    spec::register(m)?;
    values::register(m)?;
    time_series::register(m)?;
    fitting::register(m)?;
    kinetic::register(m)?;
    dielectric::register(m)?;
    spectroscopy::register(m)?;
    rdf::register(m)?;
    density::register(m)?;
    distribution::register(m)?;
    diffraction::register(m)?;
    dynamics::register(m)?;
    hbond::register(m)?;
    order::register(m)?;
    environment::register(m)?;
    pmft::register(m)?;
    cluster::register(m)?;
    shape::register(m)?;
    decomposition::register(m)?;
    kmeans::register(m)?;
    voronoi::register(m)?;
    Ok(())
}
