//! Trajectory analysis for the C++ engine — the CXX face of `molrs::compute`.
//!
//! Each bridged analysis is the molrs type of the same name, wrapped only so
//! cxx can own it: its constructor takes configuration and `accumulator()`
//! hands out the molrs streaming accumulator, which takes one frame per
//! `push` and reads the result at `finish`. molrs' MSD / VACF / RDF are
//! accumulator-only, and so is the bridge: a stored trajectory is pushed frame
//! by frame. Every fallible call returns the molrs [`ComputeError`], which cxx
//! raises as a C++ `rust::Error`; nothing answers an error with an empty
//! curve, a NaN or `false`.
//!
//! The one thing done here is the buffer convention: a frame crosses the
//! bridge as a flat buffer blocked by component (`x[0..n) y[0..n) z[0..n)`),
//! and a box as a 9-value row-major cell matrix H. [`blocked_frame`] and
//! [`simbox_of`] are the only conversions; no analysis math lives here.

use molrs::compute::{self as mc, Accumulate, Compute, ComputeError, Stream};
use molrs::core::{NeighborList, Points, SimBox};
use ndarray::Array2;

use crate::bridge;
use crate::frame::simbox_from_h;

/// Components per item of a position frame (blocked `x|y|z`).
const SPATIAL: usize = 3;
/// Components per item of a velocity frame fed to the VACF: every degree of
/// freedom is its own item.
const SCALAR: usize = 1;
/// Values in a cell matrix H.
const CELL: usize = 9;

/// The `n × D` frame of a flat buffer blocked by component — the one
/// marshaling path for every frame that crosses the bridge.
///
/// # Errors
/// [`ComputeError::BadShape`] when the buffer does not split into `D` equal
/// component blocks.
fn blocked_frame(buffer: &[f64], components: usize) -> Result<Array2<f64>, ComputeError> {
    if !buffer.len().is_multiple_of(components) {
        return Err(ComputeError::BadShape {
            expected: format!("{components} blocked components per item"),
            got: format!("{} values", buffer.len()),
        });
    }
    let n = buffer.len() / components;
    Ok(Array2::from_shape_fn((n, components), |(i, d)| {
        buffer[d * n + i]
    }))
}

/// A periodic box from one 9-value row-major cell matrix H.
///
/// # Errors
/// [`ComputeError::DimensionMismatch`] unless `h` holds exactly 9 values;
/// [`ComputeError::BadShape`] for a singular H.
fn simbox_of(h: &[f64]) -> Result<SimBox, ComputeError> {
    simbox_with(h, &[true; 3])
}

/// The box of cell matrix `h` (9 row-major values), periodic on the axes
/// `pbc` (3 flags) names.
///
/// # Errors
/// As [`simbox_of`], and [`ComputeError::DimensionMismatch`] unless `pbc`
/// holds exactly 3 flags.
fn simbox_with(h: &[f64], pbc: &[bool]) -> Result<SimBox, ComputeError> {
    let pbc: [bool; 3] = pbc
        .try_into()
        .map_err(|_| ComputeError::DimensionMismatch {
            expected: 3,
            got: pbc.len(),
            what: "periodic flags (x, y, z)",
        })?;
    let periodic = simbox_cell(h)?;
    SimBox::new(molrs::core::Cell::Matrix(periodic.matrix()), [0.0; 3], pbc).map_err(|e| {
        ComputeError::BadShape {
            expected: "a non-singular row-major 3×3 cell matrix H".into(),
            got: format!("{e:?}"),
        }
    })
}

/// The periodic box of `h`, validated.
fn simbox_cell(h: &[f64]) -> Result<SimBox, ComputeError> {
    if h.len() != CELL {
        return Err(ComputeError::DimensionMismatch {
            expected: CELL,
            got: h.len(),
            what: "cell matrix H (9 row-major values)",
        });
    }
    simbox_from_h(h).map_err(|got| ComputeError::BadShape {
        expected: "a non-singular row-major 3×3 cell matrix H".into(),
        got,
    })
}

// ── Time correlations ───────────────────────────────────────────────────────

/// A C++ `u64` count as a `usize`.
fn to_usize(value: u64, field: &'static str) -> Result<usize, ComputeError> {
    usize::try_from(value).map_err(|_| ComputeError::OutOfRange {
        field,
        value: value.to_string(),
    })
}

/// A correlation curve — molrs `compute::TimeCorrelation`, the output of
/// [`Msd`], [`Vacf`] and their [`CorrelatorAccumulator`].
pub struct TimeCorrelation {
    inner: mc::TimeCorrelation,
}

impl TimeCorrelation {
    fn boxed(inner: mc::TimeCorrelation) -> Box<Self> {
        Box::new(Self { inner })
    }

    /// The lag of each row, in the unit of `dt`.
    pub fn lag_times(&self) -> Vec<f64> {
        self.inner.lag_times.to_vec()
    }

    /// Number of lags, `max_lag + 1`.
    pub fn n_lags(&self) -> u64 {
        self.inner.n_lags() as u64
    }

    /// Number of channels (1 for an item-averaged curve).
    pub fn n_channels(&self) -> u64 {
        self.inner.n_channels() as u64
    }

    /// The curve of one channel, indexed by lag.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a channel the curve does not have.
    pub fn channel(&self, channel: u64) -> Result<Vec<f64>, ComputeError> {
        let channel = usize::try_from(channel).map_err(|_| ComputeError::OutOfRange {
            field: "channel",
            value: channel.to_string(),
        })?;
        Ok(self.inner.channel(channel)?.to_vec())
    }
}

/// The streaming accumulator of a correlator surface — molrs
/// `compute::CorrelatorAccumulator`, from [`Msd::accumulator`] or
/// [`Vacf::accumulator`]. It remembers how many blocked components one item
/// of a pushed frame has (3 for positions, 1 for velocity DOFs).
pub struct CorrelatorAccumulator {
    inner: mc::CorrelatorAccumulator,
    components: usize,
}

impl CorrelatorAccumulator {
    /// Fold one frame (blocked by component) with the series' sample interval
    /// `dt` and, for a displacement kernel, its box `h` (9 values; empty skips
    /// the wrapped-trajectory check).
    ///
    /// # Errors
    /// The molrs `CorrelatorAccumulator::push` error, or a buffer that does
    /// not fit the layout; a rejected frame leaves the state unchanged.
    pub fn push(&mut self, frame: &[f64], dt: f64, h: &[f64]) -> Result<(), ComputeError> {
        let a = blocked_frame(frame, self.components)?;
        // cxx has no `Option` of a slice: an empty H is "no box".
        let simbox = if h.is_empty() {
            None
        } else {
            Some(simbox_of(h)?)
        };
        self.inner.push(mc::CorrelatorFrame {
            a: a.view(),
            b: None,
            dt,
            simbox: simbox.as_ref(),
        })
    }

    /// Number of frames folded so far.
    pub fn n_frames(&self) -> u64 {
        self.inner.n_frames() as u64
    }

    /// The correlation over the frames so far (does not consume the state).
    ///
    /// # Errors
    /// The molrs `CorrelatorAccumulator::finish` error (e.g. no frames).
    pub fn finish(&self) -> Result<Box<TimeCorrelation>, ComputeError> {
        self.inner.finish().map(TimeCorrelation::boxed)
    }
}

/// Mean squared displacement — molrs `compute::Msd`.
pub struct Msd {
    inner: mc::Msd,
}

/// An MSD over lags `0 … max_lag` frames (molrs `Msd::new`).
pub(crate) fn msd_new(max_lag: u64) -> Box<Msd> {
    Box::new(Msd {
        inner: mc::Msd::new(max_lag as usize),
    })
}

impl Msd {
    /// Average every origin (`"all"`) or measure from frame 0 (`"first"`) —
    /// molrs `Msd::with_origins`, parsed by the molrs `Origins` names.
    ///
    /// # Errors
    /// [`ComputeError::UnknownName`] for any other name.
    pub fn set_origins(&mut self, origins: &str) -> Result<(), ComputeError> {
        self.inner = self.inner.with_origins(origins.parse()?);
        Ok(())
    }

    /// Take every `stride`-th frame as a time origin — molrs
    /// `Msd::with_stride`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0` (a stride with the
    /// `"first"` origin is refused by the accumulator).
    pub fn set_stride(&mut self, stride: u64) -> Result<(), ComputeError> {
        self.inner = self.inner.with_stride(to_usize(stride, "stride")?)?;
        Ok(())
    }

    /// The streaming form; push one blocked `x|y|z` position frame at a time.
    pub fn accumulator(&self) -> Box<CorrelatorAccumulator> {
        Box::new(CorrelatorAccumulator {
            inner: self.inner.accumulator(),
            components: SPATIAL,
        })
    }
}

/// Velocity autocorrelation, averaged over degrees of freedom — molrs
/// `compute::Vacf`.
pub struct Vacf {
    inner: mc::Vacf,
}

/// A VACF over lags `0 … max_lag` frames (molrs `Vacf::new`).
pub(crate) fn vacf_new(max_lag: u64) -> Box<Vacf> {
    Box::new(Vacf {
        inner: mc::Vacf::new(max_lag as usize),
    })
}

impl Vacf {
    /// Scale the lagged sums — molrs `Vacf::with_normalization`, parsed by
    /// the molrs `Normalization` names.
    ///
    /// # Errors
    /// [`ComputeError::UnknownName`] for any other name.
    pub fn set_normalization(&mut self, normalization: &str) -> Result<(), ComputeError> {
        self.inner = self.inner.with_normalization(normalization.parse()?);
        Ok(())
    }

    /// Take every `stride`-th frame as a time origin — molrs
    /// `Vacf::with_stride`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `stride == 0`.
    pub fn set_stride(&mut self, stride: u64) -> Result<(), ComputeError> {
        self.inner = self.inner.with_stride(to_usize(stride, "stride")?)?;
        Ok(())
    }

    /// Subtract or keep each component's time mean — molrs
    /// `Vacf::with_mean`, parsed by the molrs `MeanPolicy` names.
    ///
    /// # Errors
    /// [`ComputeError::UnknownName`] for any other name.
    pub fn set_mean(&mut self, mean: &str) -> Result<(), ComputeError> {
        self.inner = self.inner.with_mean(mean.parse()?);
        Ok(())
    }

    /// The streaming form; push one velocity frame (`n_dof` values) at a time.
    pub fn accumulator(&self) -> Box<CorrelatorAccumulator> {
        Box::new(CorrelatorAccumulator {
            inner: self.inner.accumulator(),
            components: SCALAR,
        })
    }
}

/// Ordinary-least-squares line through one channel of a [`TimeCorrelation`]
/// — molrs `compute::LinearFit`. The Einstein `D` of an MSD curve is
/// `slope / (2 · dims)`.
pub struct LinearFit {
    inner: mc::LinearFit,
}

/// A fit over the window `(start, end)`, fractions of the last lag (molrs
/// `LinearFit::new`).
///
/// # Errors
/// [`ComputeError::OutOfRange`] unless `0 ≤ start < end ≤ 1`.
pub(crate) fn linear_fit_new(start: f64, end: f64) -> Result<Box<LinearFit>, ComputeError> {
    Ok(Box::new(LinearFit {
        inner: mc::LinearFit::new(start, end)?,
    }))
}

impl LinearFit {
    /// Fit `channel` of `correlation` against its lag times.
    ///
    /// # Errors
    /// The molrs `LinearFit::compute` error.
    pub fn compute(
        &self,
        correlation: &TimeCorrelation,
        channel: u64,
    ) -> Result<bridge::ffi::LinearFitResult, ComputeError> {
        let fit = self.inner.compute(mc::LinearFitInput {
            correlation: &correlation.inner,
            channel: channel as usize,
        })?;
        Ok(bridge::ffi::LinearFitResult {
            slope: fit.slope,
            intercept: fit.intercept,
            r2: fit.r2,
            fit_start: fit.fit_start as u64,
            fit_end: fit.fit_end as u64,
        })
    }
}

// ── Radial distribution ─────────────────────────────────────────────────────

/// The `x`, `y`, `z` columns of a blocked `x|y|z` buffer.
///
/// # Errors
/// [`ComputeError::BadShape`] unless the buffer holds 3 values per point.
fn blocked_columns(positions: &[f64]) -> Result<[&[f64]; 3], ComputeError> {
    if !positions.len().is_multiple_of(SPATIAL) {
        return Err(ComputeError::BadShape {
            expected: "blocked x|y|z positions, 3 values per atom".into(),
            got: format!("{} values", positions.len()),
        });
    }
    let n = positions.len() / SPATIAL;
    let (x, rest) = positions.split_at(n);
    let (y, z) = rest.split_at(n);
    Ok([x, y, z])
}

fn rdf_result(result: mc::RdfResult) -> bridge::ffi::RdfResult {
    bridge::ffi::RdfResult {
        bin_edges: result.bin_edges.to_vec(),
        bin_centers: result.bin_centers.to_vec(),
        rdf: result.rdf.to_vec(),
        n_r: result.n_r.to_vec(),
        n_points: result.n_points as u64,
        n_query_points: result.n_query_points as u64,
        self_query: result.self_query,
        volume: result.volume,
        r_min: result.r_min,
        n_frames: result.n_frames as u64,
        dimensionality: result.dimensionality,
    }
}

/// Radial distribution function g(r) — molrs `compute::Rdf`. Each frame's
/// pairs come from a cell-list search with cutoff `r_max` in the frame's box,
/// periodic on the axes the caller names: a self-query of one point set, or
/// the cross query of a second set against a reference set.
pub struct Rdf {
    inner: mc::Rdf,
}

/// Bin pair distances in `[r_min, r_max)` Å into `n_bins` bins (molrs
/// `Rdf::new`; note `r_max` before `r_min`).
///
/// # Errors
/// The molrs `Rdf::new` error: `n_bins == 0`, `r_min < 0` or `r_max ≤ r_min`.
pub(crate) fn rdf_new(n_bins: u64, r_max: f64, r_min: f64) -> Result<Box<Rdf>, ComputeError> {
    Ok(Box::new(Rdf {
        inner: mc::Rdf::new(n_bins as usize, r_max, r_min)?,
    }))
}

impl Rdf {
    /// The streaming form; push one position frame and its H at a time.
    pub fn accumulator(&self) -> Box<RdfAccumulator> {
        Box::new(RdfAccumulator {
            inner: self.inner.accumulator(),
            cutoff: self.inner.r_max(),
        })
    }
}

/// The streaming g(r) accumulator — molrs `compute::RdfAccumulator`, from
/// [`Rdf::accumulator`] (O(n_bins) state); the one form of the g(r).
pub struct RdfAccumulator {
    inner: mc::RdfAccumulator,
    cutoff: f64,
}

impl RdfAccumulator {
    /// Fold one blocked `x|y|z` position frame, self-queried in the box `h`
    /// (9 row-major values) periodic on the axes `pbc` (3 flags) names.
    ///
    /// # Errors
    /// The molrs `RdfAccumulator::push` error, or a buffer that does not fit;
    /// a rejected frame leaves the state unchanged.
    pub fn push(&mut self, positions: &[f64], h: &[f64], pbc: &[bool]) -> Result<(), ComputeError> {
        let simbox = simbox_with(h, pbc)?;
        let [x, y, z] = blocked_columns(positions)?;
        let mut list = NeighborList::new(self.cutoff).map_err(|e| ComputeError::OutOfRange {
            field: "r_max",
            value: e.cutoff.to_string(),
        })?;
        list.build(Points::Columns([x, y, z]), &simbox);
        self.inner.push(&list)
    }

    /// Fold one frame of the A–B g(r): the blocked `x|y|z` `query` points
    /// against the `reference` points, in the box `h` periodic on `pbc`.
    ///
    /// # Errors
    /// As [`push`](Self::push).
    pub fn push_cross(
        &mut self,
        reference: &[f64],
        query: &[f64],
        h: &[f64],
        pbc: &[bool],
    ) -> Result<(), ComputeError> {
        let simbox = simbox_with(h, pbc)?;
        let reference = blocked_columns(reference)?;
        let query = blocked_columns(query)?;
        let index =
            molrs::core::NeighborQuery::new(&simbox, Points::Columns(reference), self.cutoff)
                .map_err(|e| ComputeError::OutOfRange {
                    field: "r_max",
                    value: e.cutoff.to_string(),
                })?;
        self.inner.push(&index.query(Points::Columns(query)))
    }

    /// Number of frames folded so far.
    pub fn n_frames(&self) -> u64 {
        self.inner.n_frames() as u64
    }

    /// The normalized g(r) over the frames so far (does not consume the state).
    ///
    /// # Errors
    /// The molrs `RdfAccumulator::finish` error (e.g. no frames).
    pub fn finish(&self) -> Result<bridge::ffi::RdfResult, ComputeError> {
        self.inner.finish().map(rdf_result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cube of side `l` as a row-major H.
    fn cube(l: f64) -> [f64; 9] {
        [l, 0.0, 0.0, 0.0, l, 0.0, 0.0, 0.0, l]
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tol, "{actual:?} vs {expected:?}");
        }
    }

    /// Frame `t` of `n` atoms moving ballistically along x (atom `i` at speed
    /// `0.1 · (i + 1)` per frame), blocked `x|y|z`.
    fn ballistic(t: usize, n: usize) -> Vec<f64> {
        let mut out: Vec<f64> = (0..n)
            .map(|i| i as f64 + 0.1 * (i + 1) as f64 * t as f64)
            .collect();
        out.extend((0..n).map(|i| 0.5 * i as f64));
        out.extend(std::iter::repeat_n(1.0, n));
        out
    }

    /// A 4 × 4 × 4 simple cubic lattice of spacing 2.5 Å, blocked `x|y|z` —
    /// it fills a 10 Å periodic cube exactly.
    fn lattice() -> Vec<f64> {
        let mut xyz = [Vec::new(), Vec::new(), Vec::new()];
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    for (axis, index) in [i, j, k].into_iter().enumerate() {
                        xyz[axis].push(2.5 * index as f64);
                    }
                }
            }
        }
        xyz.concat()
    }

    /// Ballistic atoms: MSD(lag) = 0.01 · lag² · mean((i + 1)²), which is
    /// 0.075 · lag² for four atoms, at lag times `lag · dt`.
    #[test]
    fn msd_accumulator_measures_a_ballistic_curve() {
        let mut acc = msd_new(3).accumulator();
        for t in 0..6 {
            acc.push(&ballistic(t, 4), 0.5, &cube(100.0)).unwrap();
        }
        assert_eq!(acc.n_frames(), 6);
        let curve = acc.finish().unwrap();
        assert_eq!(curve.n_lags(), 4);
        assert_eq!(curve.n_channels(), 1);
        assert_close(&curve.lag_times(), &[0.0, 0.5, 1.0, 1.5], 1e-15);
        assert_close(&curve.channel(0).unwrap(), &[0.0, 0.075, 0.3, 0.675], 1e-12);
        assert!(matches!(
            curve.channel(1),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// One atom at unit speed: MSD(t) = t² on t = 0..=9, whose least-squares
    /// line has slope 9 and intercept −12.
    #[test]
    fn msd_curve_feeds_linear_fit() {
        let mut acc = msd_new(9).accumulator();
        for t in 0..10 {
            acc.push(&[t as f64, 0.0, 0.0], 1.0, &[]).unwrap();
        }
        let curve = acc.finish().unwrap();
        let fit = linear_fit_new(0.0, 1.0)
            .unwrap()
            .compute(&curve, 0)
            .unwrap();
        assert!((fit.slope - 9.0).abs() < 1e-9, "{}", fit.slope);
        assert!((fit.intercept + 12.0).abs() < 1e-9, "{}", fit.intercept);
        assert_eq!((fit.fit_start, fit.fit_end), (0, 9));
    }

    #[test]
    fn msd_refuses_a_ragged_frame_a_wrapped_step_and_a_bad_configuration() {
        let mut acc = msd_new(2).accumulator();
        assert!(acc.finish().is_err(), "no frames, no curve");
        assert!(matches!(
            acc.push(&[0.0; 10], 1.0, &[]),
            Err(ComputeError::BadShape { .. })
        ));
        assert!(matches!(
            acc.push(&[0.0; 3], 1.0, &[1.0; 4]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        acc.push(&[0.0; 3], 1.0, &cube(4.0)).unwrap();
        assert!(matches!(
            acc.push(&[3.9, 0.0, 0.0], 1.0, &cube(4.0)),
            Err(ComputeError::Wrapped { .. })
        ));
        assert_eq!(acc.n_frames(), 1, "a rejected frame leaves the state");

        let mut msd = msd_new(2);
        assert!(matches!(
            msd.set_origins("middle"),
            Err(ComputeError::UnknownName { .. })
        ));
        msd.set_origins("first").unwrap();
        assert!(matches!(
            msd.set_stride(0),
            Err(ComputeError::OutOfRange { .. })
        ));
        // A stride with the single first origin is the accumulator's refusal.
        msd.set_stride(2).unwrap();
        assert!(matches!(
            msd.accumulator().push(&[0.0; 3], 1.0, &[]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// An alternating velocity `(−1)^t` has zero mean, so the unbiased VACF is
    /// `(−1)^lag` exactly — with every origin or every other one.
    #[test]
    fn vacf_of_an_alternating_velocity() {
        for stride in [1, 2] {
            let mut vacf = vacf_new(3);
            vacf.set_stride(stride).unwrap();
            let mut acc = vacf.accumulator();
            for t in 0..8 {
                let v = if t % 2 == 0 { 1.0 } else { -1.0 };
                acc.push(&[v], 2.0, &[]).unwrap();
            }
            let curve = acc.finish().unwrap();
            assert_close(&curve.lag_times(), &[0.0, 2.0, 4.0, 6.0], 1e-15);
            assert_close(&curve.channel(0).unwrap(), &[1.0, -1.0, 1.0, -1.0], 1e-12);
        }
    }

    /// A constant velocity `c` correlates to `c²` at every lag when its mean
    /// is kept, and to 0 when it is subtracted (the default).
    #[test]
    fn vacf_mean_policy_keeps_or_subtracts_the_mean() {
        let curve_of = |mean: Option<&str>| {
            let mut vacf = vacf_new(2);
            if let Some(mean) = mean {
                vacf.set_mean(mean).unwrap();
            }
            let mut acc = vacf.accumulator();
            for _ in 0..5 {
                acc.push(&[1.5, 1.5], 1.0, &[]).unwrap();
            }
            acc.finish().unwrap().channel(0).unwrap()
        };
        assert_close(&curve_of(Some("keep")), &[2.25; 3], 1e-12);
        assert_close(&curve_of(Some("subtract")), &[0.0; 3], 1e-12);
        assert_close(&curve_of(None), &[0.0; 3], 1e-12);
    }

    #[test]
    fn vacf_refuses_unknown_names_and_a_width_change() {
        let mut vacf = vacf_new(2);
        assert!(matches!(
            vacf.set_normalization("weird"),
            Err(ComputeError::UnknownName { .. })
        ));
        vacf.set_normalization("biased").unwrap();
        assert!(matches!(
            vacf.set_mean("weird"),
            Err(ComputeError::UnknownName { .. })
        ));
        assert!(matches!(
            vacf.set_stride(0),
            Err(ComputeError::OutOfRange { .. })
        ));
        let mut acc = vacf.accumulator();
        assert!(acc.finish().is_err());
        acc.push(&[1.0, 2.0, 3.0], 1.0, &[]).unwrap();
        assert!(acc.push(&[1.0, 2.0], 1.0, &[]).is_err());
        assert_eq!(acc.n_frames(), 1);
    }

    /// A simple cubic lattice: every pair within 4 Å is a first neighbour at
    /// 2.5 Å (6 per atom) or a second at 2.5·√2 ≈ 3.54 Å (12 per atom), so the
    /// counts fill exactly two bins, the second holding twice the first.
    #[test]
    fn rdf_of_a_simple_cubic_lattice() {
        let rdf = rdf_new(20, 4.0, 0.0).unwrap();
        let mut acc = rdf.accumulator();
        for _ in 0..2 {
            acc.push(&lattice(), &cube(10.0), &[true; 3]).unwrap();
        }
        assert_eq!(acc.n_frames(), 2);
        let result = acc.finish().unwrap();
        assert_eq!(result.n_frames, 2);
        assert_eq!(result.rdf.len(), 20);
        assert_eq!(result.bin_edges.len(), 21);
        // molrs sums the cell volume over frames (RdfResult::volume).
        assert!((result.volume - 2000.0).abs() < 1e-9, "{}", result.volume);
        let first = result.n_r[12];
        assert!(first > 0.0);
        assert_eq!(result.n_r[17], 2.0 * first);
        for (bin, &count) in result.n_r.iter().enumerate() {
            if bin != 12 && bin != 17 {
                assert_eq!(count, 0.0, "bin {bin}");
                assert_eq!(result.rdf[bin], 0.0, "bin {bin}");
            }
        }
    }

    #[test]
    fn rdf_errors_are_raised_not_swallowed() {
        assert!(matches!(
            rdf_new(10, 1.0, 2.0),
            Err(ComputeError::OutOfRange { .. })
        ));
        let rdf = rdf_new(10, 4.0, 0.0).unwrap();
        let mut acc = rdf.accumulator();
        assert!(acc.finish().is_err());
        assert!(matches!(
            acc.push(&lattice(), &[1.0; 4], &[true; 3]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            acc.push(&lattice(), &[0.0; 9], &[true; 3]),
            Err(ComputeError::BadShape { .. })
        ));
        assert!(matches!(
            acc.push(&[0.0; 4], &cube(10.0), &[true; 3]),
            Err(ComputeError::BadShape { .. })
        ));
        assert!(matches!(
            acc.push(&lattice(), &cube(10.0), &[true; 2]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert_eq!(acc.n_frames(), 0);
    }

    /// The caller's periodicity decides which pairs exist: two atoms 1.5 Å
    /// apart through the x face are a pair in a periodic box and not in a
    /// free one; the cross query counts the A–B pairs only.
    #[test]
    fn rdf_honours_the_periodicity_and_queries_across_sets() {
        // Blocked x|y|z of atoms at x = 0.5 and x = 9.0.
        let two = [0.5, 9.0, 5.0, 5.0, 5.0, 5.0];
        let pairs = |pbc: [bool; 3]| {
            let rdf = rdf_new(8, 4.0, 0.0).unwrap();
            let mut acc = rdf.accumulator();
            acc.push(&two, &cube(10.0), &pbc).unwrap();
            acc.finish().unwrap().rdf.iter().sum::<f64>()
        };
        assert!(pairs([true; 3]) > 0.0);
        assert_eq!(pairs([false; 3]), 0.0);

        let rdf = rdf_new(8, 4.0, 0.0).unwrap();
        let mut acc = rdf.accumulator();
        acc.push_cross(&[0.5, 5.0, 5.0], &[9.0, 5.0, 5.0], &cube(10.0), &[true; 3])
            .unwrap();
        let result = acc.finish().unwrap();
        assert!(!result.self_query);
        assert!(result.rdf.iter().sum::<f64>() > 0.0);
    }

    #[test]
    fn linear_fit_rejects_a_bad_window() {
        assert!(matches!(
            linear_fit_new(0.8, 0.2),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
