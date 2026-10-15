//! Binning primitives: [`Axis`] (one validated binned coordinate) and
//! [`Histogram`] (weighted counts over `N` axes). Every histogram, radial
//! distribution and angular map in `compute` bins through them.

mod axis;
mod histogram;

pub use axis::{Axis, UpperEdge};
pub use histogram::Histogram;
