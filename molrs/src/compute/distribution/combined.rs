//! Combined Distribution Functions (CDF): joint 2-D / 3-D observable histograms.

use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;

use super::observable::{AtomGroups, Observable};
use super::{AngleObservable, DihedralObservable, DistanceObservable, DistributionResult};
use crate::core::UnitPreset;

/// One axis of a [`CombinedDistribution`]: bin count + range + optional
/// solid-angle weighting (mirrors link-01's sin θ ADF correction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisSpec {
    /// Number of bins along this axis (≥ 1).
    pub bins: usize,
    /// Lower edge of bin 0.
    pub min: F,
    /// Upper edge of the last bin (must exceed `min`).
    pub max: F,
    /// Mark this axis angular so its [`marginal`](CombinedDistributionResult::marginal)
    /// carries the sin θ-corrected density.
    pub sin_weight: bool,
}

impl AxisSpec {
    /// A linear axis with `bins` bins over `[min, max]`.
    pub fn new(bins: usize, min: F, max: F) -> Result<Self, ComputeError> {
        if bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "AxisSpec::bins",
                value: "0".to_string(),
            });
        }
        if max <= min || !min.is_finite() || !max.is_finite() {
            return Err(ComputeError::OutOfRange {
                field: "AxisSpec::range",
                value: format!("min={min}, max={max}"),
            });
        }
        Ok(Self {
            bins,
            min,
            max,
            sin_weight: false,
        })
    }

    /// Set the sin θ solid-angle weighting flag (for an angular axis).
    pub fn with_sin_weight(mut self, on: bool) -> Self {
        self.sin_weight = on;
        self
    }

    fn bin_width(&self) -> F {
        (self.max - self.min) / self.bins as F
    }

    fn fac(&self) -> F {
        self.bins as F / (self.max - self.min)
    }

    /// Per-axis **cloud-in-cell** contribution (reference implementation `C2DF`/`C3DF::AddToBin`,
    /// the multidimensional extension of link-01's [`Histogram1d::add`]
    /// (super::Histogram1d)): returns the two straddling bin indices and their
    /// weights `[(ip, 1−frac), (ip+1, frac)]`, or `None` if `d` is outside
    /// `[min, max]` (reference implementation `m_fSkipEntries`). The N-D deposit is the tensor
    /// product of these per-axis contributions, so summing the joint histogram
    /// over the other axes reproduces the link-01 1-D CIC histogram (marginal
    /// consistency). For a single-bin axis both contributions point at bin 0
    /// with weights `[1, 0]` (no out-of-range index).
    fn cic(&self, d: F) -> Option<([usize; 2], [F; 2])> {
        if d < self.min || d > self.max {
            return None;
        }
        // Reuse the shared 1-D cloud-in-cell primitive so the CDF axis binning
        // and the link-01 `Histogram1d` never drift (marginal consistency).
        Some(super::histogram1d::cic_split(
            d,
            self.min,
            self.fac(),
            self.bins,
        ))
    }

    fn edges(&self) -> Array1<F> {
        let w = self.bin_width();
        Array1::from_iter((0..=self.bins).map(|i| self.min + i as F * w))
    }

    fn centers(&self) -> Array1<F> {
        let w = self.bin_width();
        Array1::from_iter((0..self.bins).map(|i| self.min + (i as F + 0.5) * w))
    }
}

/// Object-safe observable holder.
///
/// [`Observable::sample`] is generic over the [`FrameAccess`] type, so the
/// trait is **not** dyn-compatible and `Box<dyn Observable>` is impossible.
/// This enum is the object-safe stand-in for the spec's "boxed observables":
/// it carries any of the link-01 concretes and dispatches statically.
#[derive(Debug, Clone)]
pub enum InternalCoordinate {
    Distance(DistanceObservable),
    Angle(AngleObservable),
    Dihedral(DihedralObservable),
}

impl InternalCoordinate {
    /// Parse an observable-kind string into the variant and its arity:
    /// `"distance"` → 2, `"angle"` → 3, `"dihedral"` → 4.
    pub fn from_kind(kind: &str) -> Result<(Self, usize), ComputeError> {
        let obs = match kind {
            "distance" => Self::Distance(DistanceObservable),
            "angle" => Self::Angle(AngleObservable),
            "dihedral" => Self::Dihedral(DihedralObservable),
            other => {
                return Err(ComputeError::OutOfRange {
                    field: "InternalCoordinate::kind",
                    value: other.to_string(),
                });
            }
        };
        let arity = obs.arity();
        Ok((obs, arity))
    }

    fn arity(&self) -> usize {
        match self {
            Self::Distance(o) => o.arity(),
            Self::Angle(o) => o.arity(),
            Self::Dihedral(o) => o.arity(),
        }
    }

    fn is_angular(&self) -> bool {
        match self {
            Self::Distance(o) => o.is_angular(),
            Self::Angle(o) => o.is_angular(),
            Self::Dihedral(o) => o.is_angular(),
        }
    }

    /// Sample into a reused buffer (see [`Observable::sample_into`]).
    fn sample_into<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &AtomGroups,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        match self {
            Self::Distance(o) => o.sample_into(frame, groups, out),
            Self::Angle(o) => o.sample_into(frame, groups, out),
            Self::Dihedral(o) => o.sample_into(frame, groups, out),
        }
    }
}

impl From<DistanceObservable> for InternalCoordinate {
    fn from(o: DistanceObservable) -> Self {
        Self::Distance(o)
    }
}
impl From<AngleObservable> for InternalCoordinate {
    fn from(o: AngleObservable) -> Self {
        Self::Angle(o)
    }
}
impl From<DihedralObservable> for InternalCoordinate {
    fn from(o: DihedralObservable) -> Self {
        Self::Dihedral(o)
    }
}

/// A joint distribution of 2 or 3 [`Observable`]s over aligned selections.
///
/// Each observable `k` is sampled on its own [`AtomGroups`] (passed as
/// `Args = &[AtomGroups]`, one per axis). The per-tuple samples are zipped into
/// N-tuples and accumulated into a flat row-major N-D histogram. All
/// observables must emit the same number of samples per frame (equal
/// `AtomGroups` lengths), validated as a typed [`ComputeError`] — never a
/// silent zip-truncation.
///
/// A [`CombinedDistribution`] jointly histograms two or three [`Observable`]s
/// evaluated on aligned per-frame selections, producing a correlated density
/// (RDF×ADF, distance×dihedral, angle×angle, …) whose marginals recover the
/// link-01 1-D [`DistributionResult`](super::DistributionResult)s.
///
/// Ported from the reference implementation (`src/2df.cpp`, `src/2df.h`, `src/3df.h`):
/// - The flat **row-major** bin layout `m_pBin[iy*m_iRes[0]+ix]` of `C2DF`
///   (axis 0 fastest-varying) — generalized here to N axes with
///   `flat = Σ_a idx[a]·stride[a]`, `stride[0]=1`.
/// - The skip-out-of-range / running-entry bookkeeping of
///   `C2DF::AddToBin(double x, double y)` (`m_fSkipEntries` / `m_fBinEntries`).
/// - The **bilinear / trilinear cloud-in-cell** deposition of
///   `C2DF`/`C3DF::AddToBin`: each sample is spread over the `2^N` straddling
///   bins as the tensor product of the per-axis cloud-in-cell weights (the
///   N-D extension of link-01's [`Histogram1d::add`](super::Histogram1d::add)).
///   Because it is the tensor product of the same per-axis CIC scheme, summing
///   the joint histogram over the other axes reproduces the link-01 1-D CIC
///   distribution *exactly* — the defining CDF marginal-consistency contract
///   (ac-001) holds bit for bit, not through a nearest-bin approximation.
///
/// # References
/// - Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**, 51, 2007–2023 (reference implementation).
/// - Brehm et al., *J. Chem. Phys.* **2020**, 152, 164105.
#[derive(Debug, Clone)]
pub struct CombinedDistribution {
    observables: Vec<InternalCoordinate>,
    axes: Vec<AxisSpec>,
}

impl CombinedDistribution {
    /// Build from N observables and N axis specs (N ∈ {2, 3}).
    pub fn new(
        observables: Vec<InternalCoordinate>,
        axes: Vec<AxisSpec>,
    ) -> Result<Self, ComputeError> {
        let n = observables.len();
        if !(2..=3).contains(&n) {
            return Err(ComputeError::OutOfRange {
                field: "CombinedDistribution::ndim",
                value: format!("{n} (only 2-D and 3-D supported)"),
            });
        }
        if axes.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: axes.len(),
                what: "axis count vs observable count",
            });
        }
        Ok(Self { observables, axes })
    }

    /// Number of axes (2 or 3).
    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    /// Row-major strides, axis 0 fastest (reference implementation `m_pBin[iy*nx+ix]` layout).
    fn strides(&self) -> Vec<usize> {
        let mut s = vec![1usize; self.axes.len()];
        for a in 1..self.axes.len() {
            s[a] = s[a - 1] * self.axes[a - 1].bins;
        }
        s
    }

    fn total_bins(&self) -> usize {
        self.axes.iter().map(|a| a.bins).product()
    }

    /// Deposit one frame's N-tuples into `counts` (flat row-major), advancing
    /// `binned` and `n_raw_samples`. `cols` is a reused per-axis sample buffer
    /// (one `Vec<F>` per observable) so no allocation happens per frame. This is
    /// the per-frame body shared by the serial and rayon accumulation paths.
    #[allow(clippy::too_many_arguments)]
    fn accumulate_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        groups: &[AtomGroups],
        strides: &[usize],
        n_tuples: usize,
        cols: &mut Vec<Vec<F>>,
        counts: &mut Array1<F>,
        binned: &mut F,
        n_raw_samples: &mut usize,
    ) -> Result<(), ComputeError> {
        let n = self.observables.len();
        if cols.len() < n {
            cols.resize_with(n, Vec::new);
        }
        // Sample every observable on its own groups into the reused buffers.
        for (a, (obs, g)) in self.observables.iter().zip(groups.iter()).enumerate() {
            obs.sample_into(frame, g, &mut cols[a])?;
        }
        // Defensive: an observable must honour its AtomGroups length.
        for c in cols.iter().take(n) {
            if c.len() != n_tuples {
                return Err(ComputeError::DimensionMismatch {
                    expected: n_tuples,
                    got: c.len(),
                    what: "observable sample count",
                });
            }
        }
        *n_raw_samples += n_tuples;
        // `t` indexes the parallel per-axis sample columns `cols[a][t]`, not
        // a single collection — a genuine range loop, not a needless one.
        #[allow(clippy::needless_range_loop)]
        for t in 0..n_tuples {
            // Per-axis cloud-in-cell contributions. Skip the whole N-tuple
            // if any axis is out of range (reference implementation C2DF: m_fSkipEntries when
            // any coordinate is outside its range).
            let mut contribs: [([usize; 2], [F; 2]); 3] = [([0, 0], [0.0, 0.0]); 3];
            let mut in_range = true;
            for a in 0..n {
                match self.axes[a].cic(cols[a][t]) {
                    Some(c) => contribs[a] = c,
                    None => {
                        in_range = false;
                        break;
                    }
                }
            }
            if !in_range {
                continue;
            }
            // Tensor-product deposit over the 2^n straddling bins; the
            // per-axis weights sum to 1, so each tuple deposits total
            // weight 1 (mass-conserving) — reference implementation bilinear/trilinear CIC.
            for corner in 0..(1usize << n) {
                let mut flat = 0usize;
                let mut w = 1.0;
                for a in 0..n {
                    let pick = (corner >> a) & 1;
                    flat += contribs[a].0[pick] * strides[a];
                    w *= contribs[a].1[pick];
                }
                if w != 0.0 {
                    counts[flat] += w;
                }
            }
            *binned += 1.0;
        }
        Ok(())
    }

    /// Accumulate the joint histogram over all frames → `(counts, binned,
    /// n_raw_samples)`.
    ///
    /// Frames are independent, so with `rayon` they are folded into per-thread
    /// count buffers and merged; each thread reuses its per-axis sample buffers.
    /// Falls back to a serial fold below [`PAR_THRESHOLD`] frames (and always
    /// when `rayon` is off). An empty selection (`n_tuples == 0`) contributes
    /// nothing.
    fn accumulate<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        groups: &[AtomGroups],
        strides: &[usize],
        n_tuples: usize,
    ) -> Result<(Array1<F>, F, usize), ComputeError> {
        let total_bins = self.total_bins();
        if n_tuples == 0 {
            return Ok((Array1::<F>::zeros(total_bins), 0.0, 0));
        }

        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 4;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .try_fold(
                    || (Array1::<F>::zeros(total_bins), 0.0 as F, 0usize, Vec::new()),
                    |(mut counts, mut binned, mut nrs, mut cols), frame| {
                        self.accumulate_frame(
                            *frame,
                            groups,
                            strides,
                            n_tuples,
                            &mut cols,
                            &mut counts,
                            &mut binned,
                            &mut nrs,
                        )?;
                        Ok((counts, binned, nrs, cols))
                    },
                )
                .map(|r| r.map(|(c, b, nrs, _)| (c, b, nrs)))
                .try_reduce(
                    || (Array1::<F>::zeros(total_bins), 0.0 as F, 0usize),
                    |(mut ca, ba, nra), (cb, bb, nrb)| {
                        ca += &cb;
                        Ok((ca, ba + bb, nra + nrb))
                    },
                );
        }

        let mut counts = Array1::<F>::zeros(total_bins);
        let mut binned: F = 0.0;
        let mut n_raw_samples: usize = 0;
        let mut cols: Vec<Vec<F>> = Vec::new();
        for frame in frames {
            self.accumulate_frame(
                *frame,
                groups,
                strides,
                n_tuples,
                &mut cols,
                &mut counts,
                &mut binned,
                &mut n_raw_samples,
            )?;
        }
        Ok((counts, binned, n_raw_samples))
    }
}

impl LegacyCompute for CombinedDistribution {
    /// One [`AtomGroups`] per observable/axis, aligned by tuple index.
    type Args<'a> = &'a [AtomGroups];
    type Output = CombinedDistributionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        groups: &'a [AtomGroups],
    ) -> Result<CombinedDistributionResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let n = self.observables.len();
        if groups.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: groups.len(),
                what: "AtomGroups count vs observable count",
            });
        }
        for (k, (obs, g)) in self.observables.iter().zip(groups.iter()).enumerate() {
            if g.arity() != obs.arity() {
                return Err(ComputeError::BadShape {
                    expected: format!("axis {k}: arity {}", obs.arity()),
                    got: format!("arity {}", g.arity()),
                });
            }
        }
        // All observables must emit the same number of samples per frame:
        // equal AtomGroups lengths. Reject mismatch (ac-004) — no zip-truncation.
        let n_tuples = groups[0].len();
        for g in groups.iter() {
            if g.len() != n_tuples {
                return Err(ComputeError::DimensionMismatch {
                    expected: n_tuples,
                    got: g.len(),
                    what: "per-axis sample count (all axes must match axis 0)",
                });
            }
        }

        let strides = self.strides();
        let (counts, binned, n_raw_samples) =
            self.accumulate(frames, groups, &strides, n_tuples)?;

        let bin_widths: Vec<F> = self.axes.iter().map(|a| a.bin_width()).collect();
        let cell_volume: F = bin_widths.iter().product();

        let mut result = CombinedDistributionResult {
            axes: self.axes.clone(),
            strides,
            bin_widths,
            cell_volume,
            edges: self.axes.iter().map(|a| a.edges()).collect(),
            centers: self.axes.iter().map(|a| a.centers()).collect(),
            angular: self.observables.iter().map(|o| o.is_angular()).collect(),
            counts,
            density: Array1::zeros(self.total_bins()),
            binned,
            n_raw_samples,
            n_frames: frames.len(),
            finalized: false,
        };
        result.finalize();
        Ok(result)
    }
}

/// Result of a [`CombinedDistribution`]: a flat row-major N-D histogram with a
/// normalized joint density, per-axis edges/centers, and `marginal` /
/// `free_energy` helpers.
#[derive(Debug, Clone)]
pub struct CombinedDistributionResult {
    axes: Vec<AxisSpec>,
    strides: Vec<usize>,
    bin_widths: Vec<F>,
    cell_volume: F,
    /// Per-axis bin edges (`bins + 1` each).
    pub edges: Vec<Array1<F>>,
    /// Per-axis bin centers (`bins` each).
    pub centers: Vec<Array1<F>>,
    angular: Vec<bool>,
    /// Flat row-major counts (axis 0 fastest), summed across frames.
    pub counts: Array1<F>,
    /// Flat row-major normalized joint density (∫…∫ p = 1). Set by [`finalize`](ComputeResult::finalize).
    pub density: Array1<F>,
    /// In-range binned N-tuples (sum of `counts`).
    pub binned: F,
    /// Total N-tuples emitted (including out-of-range).
    pub n_raw_samples: usize,
    /// Number of frames.
    pub n_frames: usize,
    finalized: bool,
}

impl CombinedDistributionResult {
    /// Number of axes.
    pub fn ndim(&self) -> usize {
        self.axes.len()
    }

    /// Flat row-major index for a multi-axis bin coordinate.
    pub fn flat_index(&self, idx: &[usize]) -> usize {
        idx.iter()
            .zip(self.strides.iter())
            .map(|(&i, &s)| i * s)
            .sum()
    }

    /// Product of all axis bin widths — the N-D cell "volume" used to turn the
    /// density into a per-cell probability mass (`Σ density·cell = 1`).
    pub fn bin_width_product(&self) -> F {
        self.cell_volume
    }

    /// The 1-D marginal [`DistributionResult`] along `axis` (summing the joint
    /// counts over all other axes). Equals the link-01 distribution of that
    /// axis's observable within rounding (the CDF marginal-consistency
    /// contract). The sin θ-corrected density is attached when the axis was
    /// flagged `sin_weight`.
    pub fn marginal(&self, axis: usize) -> DistributionResult {
        let bins = self.axes[axis].bins;
        let mut counts = Array1::<F>::zeros(bins);
        // Walk every flat bin, project onto `axis`.
        for (flat, &c) in self.counts.iter().enumerate() {
            if c == 0.0 {
                continue;
            }
            let ia = (flat / self.strides[axis]) % bins;
            counts[ia] += c;
        }
        let bin_width = self.bin_widths[axis];
        let n_binned: F = counts.sum();
        let mut result = DistributionResult {
            bin_centers: self.centers[axis].clone(),
            bin_edges: self.edges[axis].clone(),
            counts,
            density: Array1::zeros(bins),
            density_sin_corrected: None,
            bin_width,
            n_binned,
            n_raw_samples: self.n_raw_samples,
            n_frames: self.n_frames,
            angular: self.angular[axis] || self.axes[axis].sin_weight,
            finalized: false,
        };
        result.finalize();
        result
    }

    /// Free-energy surface `G = −k_B T · ln p` (kcal/mol), with `p` the
    /// normalized joint density. Populated bins are finite; **empty bins are
    /// floored** to the maximum populated-bin free energy (the highest finite
    /// barrier) rather than `+∞`, so the surface is finite everywhere. Returns
    /// the flat row-major array aligned with [`density`](Self::density).
    pub fn free_energy(&self, temperature: F) -> Array1<F> {
        let kt = UnitPreset::real().boltzmann() * temperature;
        let mut g = Array1::<F>::zeros(self.density.len());
        let mut g_max = F::NEG_INFINITY;
        for (i, &p) in self.density.iter().enumerate() {
            if p > 0.0 {
                let gi = -kt * p.ln();
                g[i] = gi;
                if gi > g_max {
                    g_max = gi;
                }
            }
        }
        let floor = if g_max.is_finite() { g_max } else { 0.0 };
        for (i, &p) in self.density.iter().enumerate() {
            if p <= 0.0 {
                g[i] = floor;
            }
        }
        g
    }
}

impl ComputeResult for CombinedDistributionResult {
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        if self.binned > 0.0 {
            let denom = self.binned * self.cell_volume;
            self.density = self.counts.mapv(|c| c / denom);
        } else {
            self.density = Array1::zeros(self.counts.len());
        }
        self.finalized = true;
    }
}
