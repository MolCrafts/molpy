//! Python bindings for `molrs::compute` (`molrs.compute`): one binding file
//! per Rust compute domain, every class at the one path `molrs.compute.<Name>`
//! — the Rust facade is flat, so the Python module is too.
//!
//! This file holds what every domain binding shares: how a `compute(...)`
//! call takes its frames and neighbour tables (one or a list of them).

mod analysis_contract;
mod cluster;
mod correlation;
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
mod msd;
mod order;
mod pmft;
mod rdf;
mod shape;
mod spectroscopy;
mod transport;
mod voronoi;

use pyo3::prelude::*;

use crate::core::frame::PyFrame;

/// No frames: what a compute over arrays alone passes where its trait takes
/// a frame slice.
const EMPTY_FRAMES: &[&molrs::core::Frame] = &[];

pub(crate) fn was_batched(frames: &Bound<'_, PyAny>) -> bool {
    frames.extract::<PyRef<'_, PyFrame>>().is_err()
}

// Sibling analysis submodules (mirrors molrs core `compute/`).
/// Collect owned core [`Frame`]s from a single `Frame` or a list of them.
/// Used by every batch-`compute` binding to accept both shapes.
///
/// [`Frame`]: molrs::core::Frame
pub(crate) fn collect_frames(frames: &Bound<'_, PyAny>) -> PyResult<Vec<molrs::core::Frame>> {
    use crate::core::frame::PyFrame;
    if let Ok(single) = frames.extract::<PyRef<'_, PyFrame>>() {
        return Ok(vec![single.clone_core_frame()?]);
    }
    let list: Vec<PyRef<'_, PyFrame>> = frames.extract()?;
    list.iter().map(|f| f.clone_core_frame()).collect()
}

/// Collect owned [`Neighbors`] tables from a single wrapper or a list of them.
///
/// The analyses take one materialized table per frame, so this is where the
/// binder accepts either shape. The engine that produced a table
/// (`NeighborList`) is deliberately not accepted: it would have to guess a
/// column policy, and a guess that drops `disp` would fail silently.
///
/// [`Neighbors`]: molrs::core::Neighbors
pub(crate) fn collect_neighbors(arg: &Bound<'_, PyAny>) -> PyResult<Vec<molrs::core::Neighbors>> {
    use crate::core::neighborlist::PyNeighbors;
    if let Ok(single) = arg.extract::<PyRef<'_, PyNeighbors>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyNeighbors>> = arg.extract()?;
    Ok(list.iter().map(|n| n.inner.clone()).collect())
}

/// Register every compute domain on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    analysis_contract::register(m)?;
    cluster::register(m)?;
    correlation::register(m)?;
    kmeans::register(m)?;
    decomposition::register(m)?;
    density::register(m)?;
    dielectric::register(m)?;
    diffraction::register(m)?;
    distribution::register(m)?;
    dynamics::register(m)?;
    environment::register(m)?;
    fitting::register(m)?;
    hbond::register(m)?;
    kinetic::register(m)?;
    msd::register(m)?;
    order::register(m)?;
    pmft::register(m)?;
    rdf::register(m)?;
    shape::register(m)?;
    spectroscopy::register(m)?;
    transport::register(m)?;
    voronoi::register(m)?;
    Ok(())
}
