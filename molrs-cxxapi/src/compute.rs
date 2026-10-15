//! Trajectory analysis for the C++ engine — the CXX face of `molrs::compute`.
//!
//! Each bridged analysis is the molrs type of the same name, wrapped only so
//! cxx can own it: its constructor takes configuration, `compute` takes a whole
//! trajectory and `accumulator()` hands out the molrs streaming accumulator,
//! which takes one frame per `push`. Every fallible call returns the molrs
//! [`ComputeError`], which cxx raises as a C++ `rust::Error`; nothing answers
//! an error with an empty curve, a NaN or `false`.
//!
//! The one thing done here is the buffer convention: a trajectory crosses the
//! bridge as a flat row-major `[n_frames, D · n]` buffer whose frames are
//! blocked by component (`x[0..n) y[0..n) z[0..n)`), and a box as a 9-value
//! row-major cell matrix H. [`blocked_series`] and [`simboxes`] are the only
//! conversions; no analysis math lives here.

use molrs::compute::{self as mc, Accumulate, Compute, ComputeError, Stream};
use molrs::core::{NeighborList, PairSource, SimBox};
use ndarray::{Array3, ArrayView2, Axis};

use crate::bridge;
use crate::frame::simbox_from_h;

/// Components per item of a position frame (blocked `x|y|z`).
const SPATIAL: usize = 3;
/// Components per item of a velocity frame fed to the VACF: every degree of
/// freedom is its own item.
const SCALAR: usize = 1;
/// Values in a cell matrix H.
const CELL: usize = 9;

/// A `T × n × D` series from a flat row-major `[n_frames, D · n]` buffer whose
/// frames are blocked by component — the one marshaling path for every
/// trajectory and every single frame (`n_frames = 1`) that crosses the bridge.
///
/// # Errors
/// [`ComputeError::EmptyInput`] for zero frames; [`ComputeError::BadShape`]
/// when the buffer does not split into `n_frames` rows of `D · n` values.
fn blocked_series(
    buffer: &[f64],
    n_frames: u64,
    components: usize,
) -> Result<Array3<f64>, ComputeError> {
    let n_frames = n_frames as usize;
    if n_frames == 0 {
        return Err(ComputeError::EmptyInput);
    }
    let row = buffer.len() / n_frames;
    if buffer.len() != n_frames * row || !row.is_multiple_of(components) {
        return Err(ComputeError::BadShape {
            expected: format!("{n_frames} frames of {components} blocked components per item"),
            got: format!("{} values", buffer.len()),
        });
    }
    let n = row / components;
    Ok(Array3::from_shape_fn(
        (n_frames, n, components),
        |(t, i, d)| buffer[t * row + d * n + i],
    ))
}

/// One periodic box per frame from `n_frames` consecutive 9-value row-major
/// cell matrices H.
///
/// # Errors
/// [`ComputeError::DimensionMismatch`] unless `h` holds exactly `9 · n_frames`
/// values; [`ComputeError::BadShape`] for a singular H.
fn simboxes(h: &[f64], n_frames: usize) -> Result<Vec<SimBox>, ComputeError> {
    if h.len() != CELL * n_frames {
        return Err(ComputeError::DimensionMismatch {
            expected: CELL * n_frames,
            got: h.len(),
            what: "cell matrices H (9 row-major values per frame)",
        });
    }
    h.as_chunks::<CELL>()
        .0
        .iter()
        .map(|cell| {
            simbox_from_h(cell).map_err(|got| ComputeError::BadShape {
                expected: "a non-singular row-major 3×3 cell matrix H".into(),
                got,
            })
        })
        .collect()
}

/// The boxes of an optional box buffer: cxx has no `Option` of a slice, so
/// the bridge spells "no boxes" as an empty buffer.
fn optional_simboxes(h: &[f64], n_frames: usize) -> Result<Option<Vec<SimBox>>, ComputeError> {
    if h.is_empty() {
        Ok(None)
    } else {
        simboxes(h, n_frames).map(Some)
    }
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
        let series = blocked_series(frame, 1, self.components)?;
        let simbox = optional_simboxes(h, 1)?;
        self.inner.push(mc::CorrelatorFrame {
            a: series.index_axis(Axis(0), 0),
            b: None,
            dt,
            simbox: simbox.as_ref().map(|boxes| &boxes[0]),
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
    /// `"first"` origin is refused at `compute`).
    pub fn set_stride(&mut self, stride: u64) -> Result<(), ComputeError> {
        self.inner = self.inner.with_stride(to_usize(stride, "stride")?)?;
        Ok(())
    }

    /// The MSD of a `[n_frames, 3 · n_atoms]` unwrapped position buffer at
    /// frame spacing `dt`; `boxes` holds one H per frame to reject a wrapped
    /// trajectory, or is empty to skip that check.
    ///
    /// # Errors
    /// The molrs `Msd::compute` error, or a buffer that does not fit.
    pub fn compute(
        &self,
        positions: &[f64],
        n_frames: u64,
        dt: f64,
        boxes: &[f64],
    ) -> Result<Box<TimeCorrelation>, ComputeError> {
        let series = blocked_series(positions, n_frames, SPATIAL)?;
        let boxes = optional_simboxes(boxes, n_frames as usize)?;
        self.inner
            .compute(mc::MsdInput {
                positions: series.view(),
                dt,
                boxes: boxes.as_deref(),
            })
            .map(TimeCorrelation::boxed)
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

    /// The VACF of a `[n_frames, n_dof]` velocity buffer at frame spacing `dt`.
    ///
    /// # Errors
    /// The molrs `Vacf::compute` error, or a buffer that does not fit.
    pub fn compute(
        &self,
        velocities: &[f64],
        n_frames: u64,
        dt: f64,
    ) -> Result<Box<TimeCorrelation>, ComputeError> {
        let series = blocked_series(velocities, n_frames, SCALAR)?;
        self.inner
            .compute(mc::VacfInput {
                velocities: series.index_axis(Axis(2), 0),
                dt,
            })
            .map(TimeCorrelation::boxed)
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

/// The self-query pair source of one frame: a cell list over `points` in
/// `simbox`, which streams its pairs to the RDF without materializing them.
fn self_pairs(points: ArrayView2<'_, f64>, simbox: &SimBox, cutoff: f64) -> NeighborList {
    let mut list = NeighborList::new(cutoff);
    list.build(points, simbox);
    list
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
/// pairs come from a periodic cell-list self-query with cutoff `r_max`.
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
    /// g(r) over a `[n_frames, 3 · n_atoms]` position buffer with one cell
    /// matrix H per frame in `boxes` (`[n_frames, 9]`, so NPT is supported).
    ///
    /// # Errors
    /// The molrs `Rdf::compute` error, or a buffer that does not fit.
    pub fn compute(
        &self,
        positions: &[f64],
        boxes: &[f64],
        n_frames: u64,
    ) -> Result<bridge::ffi::RdfResult, ComputeError> {
        let series = blocked_series(positions, n_frames, SPATIAL)?;
        let boxes = simboxes(boxes, n_frames as usize)?;
        let lists: Vec<NeighborList> = series
            .outer_iter()
            .zip(&boxes)
            .map(|(frame, simbox)| self_pairs(frame, simbox, self.inner.r_max()))
            .collect();
        let sources: Vec<&(dyn PairSource + Sync)> = lists
            .iter()
            .map(|list| list as &(dyn PairSource + Sync))
            .collect();
        self.inner.compute(&sources).map(rdf_result)
    }

    /// The streaming form; push one position frame and its H at a time.
    pub fn accumulator(&self) -> Box<RdfAccumulator> {
        Box::new(RdfAccumulator {
            inner: self.inner.accumulator(),
            cutoff: self.inner.r_max(),
        })
    }
}

/// The streaming g(r) accumulator — molrs `compute::RdfAccumulator`, from
/// [`Rdf::accumulator`] (O(n_bins) state).
pub struct RdfAccumulator {
    inner: mc::RdfAccumulator,
    cutoff: f64,
}

impl RdfAccumulator {
    /// Fold one blocked `x|y|z` position frame searched in the box `h`
    /// (9 row-major values).
    ///
    /// # Errors
    /// The molrs `RdfAccumulator::push` error, or a buffer that does not fit;
    /// a rejected frame leaves the state unchanged.
    pub fn push(&mut self, positions: &[f64], h: &[f64]) -> Result<(), ComputeError> {
        let series = blocked_series(positions, 1, SPATIAL)?;
        let simbox = simboxes(h, 1)?;
        let list = self_pairs(series.index_axis(Axis(0), 0), &simbox[0], self.cutoff);
        self.inner.push(&list)
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

    /// `n_frames` frames of `n` atoms moving ballistically along x, blocked.
    fn ballistic(n_frames: usize, n: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for t in 0..n_frames {
            let x = (0..n).map(|i| i as f64 + 0.1 * (i + 1) as f64 * t as f64);
            out.extend(x);
            out.extend((0..n).map(|i| 0.5 * i as f64));
            out.extend(std::iter::repeat_n(1.0, n));
        }
        out
    }

    /// A deterministic 3-D lattice with jitter, blocked `x|y|z`.
    fn lattice(seed: usize) -> Vec<f64> {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        let mut zs = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    let jitter = 0.05 * (((i * 16 + j * 4 + k + seed) * 7919 % 13) as f64);
                    xs.push(2.5 * i as f64 + jitter);
                    ys.push(2.5 * j as f64 + 0.5 * jitter);
                    zs.push(2.5 * k as f64);
                }
            }
        }
        [xs, ys, zs].concat()
    }

    #[test]
    fn msd_stream_equals_batch() {
        let (t, n) = (6, 4);
        let positions = ballistic(t, n);
        let msd = msd_new(3);
        let batch = msd.compute(&positions, t as u64, 0.5, &[]).unwrap();
        let mut acc = msd.accumulator();
        for frame in positions.chunks(3 * n) {
            acc.push(frame, 0.5, &cube(100.0)).unwrap();
        }
        let streamed = acc.finish().unwrap();
        assert_eq!(acc.n_frames(), t as u64);
        assert_eq!(batch.lag_times(), streamed.lag_times());
        for (a, b) in batch
            .channel(0)
            .unwrap()
            .iter()
            .zip(streamed.channel(0).unwrap())
        {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
    }

    #[test]
    fn msd_curve_feeds_linear_fit() {
        // One atom moving at unit speed: MSD(t) = t², so a fit over the
        // windowed curve is increasing; a single channel exists.
        let positions: Vec<f64> = (0..10).flat_map(|t| [t as f64, 0.0, 0.0]).collect();
        let curve = msd_new(9).compute(&positions, 10, 1.0, &[]).unwrap();
        assert_eq!(curve.n_channels(), 1);
        let fit = linear_fit_new(0.0, 1.0)
            .unwrap()
            .compute(&curve, 0)
            .unwrap();
        assert!(fit.slope > 0.0);
        assert_eq!(fit.fit_end, 9);
    }

    #[test]
    fn msd_rejects_a_ragged_buffer_and_an_unknown_origin() {
        let msd = msd_new(2);
        assert!(matches!(
            msd.compute(&[0.0; 10], 3, 1.0, &[]),
            Err(ComputeError::BadShape { .. })
        ));
        assert!(matches!(
            msd.compute(&[], 0, 1.0, &[]),
            Err(ComputeError::EmptyInput)
        ));
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
        // A stride with the single first origin is the compute's refusal.
        msd.set_stride(2).unwrap();
        assert!(matches!(
            msd.compute(&[0.0; 9], 3, 1.0, &[]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn vacf_takes_a_normalization_and_a_stride() {
        let (t, n_dof) = (9, 2);
        let velocities: Vec<f64> = (0..t * n_dof).map(|k| (k as f64 * 0.7).sin()).collect();
        let mut vacf = vacf_new(3);
        assert!(matches!(
            vacf.set_normalization("weird"),
            Err(ComputeError::UnknownName { .. })
        ));
        vacf.set_normalization("biased").unwrap();
        vacf.set_stride(2).unwrap();
        assert!(matches!(
            vacf.set_mean("weird"),
            Err(ComputeError::UnknownName { .. })
        ));
        vacf.set_mean("keep").unwrap();
        assert_eq!(vacf.inner.correlator().mean(), mc::MeanPolicy::Keep);
        let batch = vacf.compute(&velocities, t as u64, 1.0).unwrap();
        let mut acc = vacf.accumulator();
        for frame in velocities.chunks(n_dof) {
            acc.push(frame, 1.0, &[]).unwrap();
        }
        let streamed = acc.finish().unwrap();
        for (a, b) in batch
            .channel(0)
            .unwrap()
            .iter()
            .zip(streamed.channel(0).unwrap())
        {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn correlator_accumulator_errors_instead_of_returning_empty() {
        let mut acc = vacf_new(2).accumulator();
        assert!(acc.finish().is_err());
        acc.push(&[1.0, 2.0, 3.0], 1.0, &[]).unwrap();
        assert!(acc.push(&[1.0, 2.0], 1.0, &[]).is_err());
        assert_eq!(acc.n_frames(), 1);
    }

    #[test]
    fn vacf_stream_equals_batch() {
        let (t, n_dof) = (8, 6);
        let velocities: Vec<f64> = (0..t * n_dof)
            .map(|k| ((k * 37 % 11) as f64 - 5.0) * 0.1)
            .collect();
        let vacf = vacf_new(4);
        let batch = vacf.compute(&velocities, t as u64, 2.0).unwrap();
        let mut acc = vacf.accumulator();
        for frame in velocities.chunks(n_dof) {
            acc.push(frame, 2.0, &[]).unwrap();
        }
        let streamed = acc.finish().unwrap();
        for (a, b) in batch
            .channel(0)
            .unwrap()
            .iter()
            .zip(streamed.channel(0).unwrap())
        {
            assert!((a - b).abs() < 1e-9, "{a} vs {b}");
        }
        assert!(matches!(
            batch.channel(1),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn rdf_stream_equals_batch() {
        let frames = [lattice(0), lattice(3)];
        let positions = frames.concat();
        let boxes = [cube(10.0), cube(10.0)].concat();
        let rdf = rdf_new(20, 4.0, 0.0).unwrap();
        let batch = rdf.compute(&positions, &boxes, 2).unwrap();
        let mut acc = rdf.accumulator();
        for frame in &frames {
            acc.push(frame, &cube(10.0)).unwrap();
        }
        let streamed = acc.finish().unwrap();
        assert_eq!(batch.rdf, streamed.rdf);
        assert_eq!(batch.n_frames, 2);
        assert_eq!(batch.rdf.len(), 20);
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
            acc.push(&lattice(0), &[1.0; 4]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            acc.push(&lattice(0), &[0.0; 9]),
            Err(ComputeError::BadShape { .. })
        ));
        assert_eq!(acc.n_frames(), 0);
    }

    #[test]
    fn linear_fit_rejects_a_bad_window() {
        assert!(matches!(
            linear_fit_new(0.8, 0.2),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
