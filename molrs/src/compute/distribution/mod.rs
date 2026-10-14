//! User-defined geometric distribution functions (ADF / DDF / distance DF).

mod angle;
mod combined;
mod dihedral;
mod distance;
mod histogram1d;
mod observable;

pub use angle::AngleObservable;
pub use combined::{
    AxisSpec, CombinedDistribution, CombinedDistributionResult, InternalCoordinate,
};
pub use dihedral::DihedralObservable;
pub use distance::DistanceObservable;
pub use histogram1d::{Histogram1d, renormalize_density};
pub use observable::{AtomGroups, Observable};

use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;

/// A 1-D distribution function over the samples of an [`Observable`].
///
/// Stateless parameter bag (the observable plus histogram bounds); the
/// histogram is accumulated inside [`compute`](LegacyCompute::compute). For an
/// angular observable the sin θ-corrected density is computed alongside the raw
/// one.
///
/// A [`DistributionFunction`] histograms the scalar stream produced by an
/// [`Observable`] (distance, angle, dihedral) over a trajectory, reusing the
/// shared [`Histogram1d`] binning and the `SimBox` minimum-image convention so
/// the distance DF agrees with [`Rdf`](crate::compute::Rdf).
///
/// Ported from the reference implementation (`src/tddf.cpp`, `src/geodens.cpp`, `src/df.cpp`); see
/// each submodule for the specific function each routine derives from. The
/// angular distribution additionally exposes a sin θ solid-angle correction —
/// both the raw and corrected densities are returned, because conflating them
/// is the most common ADF mistake.
///
/// # References
/// - Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**, 51, 2007–2023 (reference implementation).
/// - Brehm, Thomas, Gehrke, Kirchner, *J. Chem. Phys.* **2020**, 152, 164105.
#[derive(Debug, Clone)]
pub struct DistributionFunction<O: Observable> {
    observable: O,
    n_bins: usize,
    min: F,
    max: F,
}

impl<O: Observable> DistributionFunction<O> {
    /// Create a distribution function binning the observable's samples into
    /// `n_bins` bins over `[min, max]`.
    pub fn new(observable: O, n_bins: usize, min: F, max: F) -> Result<Self, ComputeError> {
        if n_bins == 0 {
            return Err(ComputeError::OutOfRange {
                field: "DistributionFunction::n_bins",
                value: "0".to_string(),
            });
        }
        if max <= min || !min.is_finite() || !max.is_finite() {
            return Err(ComputeError::OutOfRange {
                field: "DistributionFunction::range",
                value: format!("min={min}, max={max}"),
            });
        }
        Ok(Self {
            observable,
            n_bins,
            min,
            max,
        })
    }

    /// Create over the observable's [`natural_range`](Observable::natural_range)
    /// (e.g. `[0, π]` for an angle). Errors if the observable has none
    /// (distance — supply explicit bounds with [`new`](Self::new)).
    pub fn over_natural_range(observable: O, n_bins: usize) -> Result<Self, ComputeError> {
        let (min, max) = observable.natural_range().ok_or(ComputeError::OutOfRange {
            field: "DistributionFunction::natural_range",
            value: "observable has no natural range".to_string(),
        })?;
        Self::new(observable, n_bins, min, max)
    }

    /// Histogram the observable's samples over all frames (non-empty selection).
    ///
    /// Frames are independent, so above `PAR_THRESHOLD` frames they are folded
    /// into per-thread partial histograms with `rayon` and merged (each thread
    /// reuses its own sample buffer); below the threshold, and whenever `rayon`
    /// is off, a serial fold reuses one buffer across every frame. The `O: Sync`
    /// bound on the [`Compute`] impl is what unlocks the parallel fold — the
    /// built-in observables are all `Sync`. Merging sums the per-thread bin
    /// counts, so the total matches the serial accumulation up to floating-point
    /// summation order (the same tolerance the cloud-in-cell deposition already
    /// carries), consistent with
    /// [`CombinedDistribution`](combined::CombinedDistribution)'s frame fold.
    fn accumulate<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        groups: &AtomGroups,
    ) -> Result<(Histogram1d, usize), ComputeError>
    where
        O: Sync,
    {
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 4;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .try_fold(
                    || {
                        (
                            Histogram1d::new(self.n_bins, self.min, self.max),
                            0usize,
                            Vec::<F>::new(),
                        )
                    },
                    |(mut hist, mut nrs, mut scratch), frame| {
                        self.observable.sample_into(*frame, groups, &mut scratch)?;
                        nrs += scratch.len();
                        for &s in &scratch {
                            hist.add(s);
                        }
                        Ok((hist, nrs, scratch))
                    },
                )
                .map(|r| r.map(|(hist, nrs, _)| (hist, nrs)))
                .try_reduce(
                    || (Histogram1d::new(self.n_bins, self.min, self.max), 0usize),
                    |(mut ha, nra), (hb, nrb)| {
                        ha.merge(&hb);
                        Ok((ha, nra + nrb))
                    },
                );
        }

        let mut hist = Histogram1d::new(self.n_bins, self.min, self.max);
        let mut n_raw_samples: usize = 0;
        let mut scratch: Vec<F> = Vec::new();
        for frame in frames {
            self.observable.sample_into(*frame, groups, &mut scratch)?;
            n_raw_samples += scratch.len();
            for &s in &scratch {
                hist.add(s);
            }
        }
        Ok((hist, n_raw_samples))
    }
}

impl<O: Observable + Sync> LegacyCompute for DistributionFunction<O> {
    type Args<'a> = &'a AtomGroups;
    type Output = DistributionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        groups: &'a AtomGroups,
    ) -> Result<DistributionResult, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if groups.arity() != self.observable.arity() {
            return Err(ComputeError::BadShape {
                expected: format!("arity {}", self.observable.arity()),
                got: format!("arity {}", groups.arity()),
            });
        }

        // Empty selection contributes no samples (ac-006: no panic).
        let (hist, n_raw_samples) = if groups.is_empty() {
            (Histogram1d::new(self.n_bins, self.min, self.max), 0usize)
        } else {
            self.accumulate(frames, groups)?
        };

        let mut result = DistributionResult {
            bin_centers: hist.centers(),
            bin_edges: hist.edges(),
            counts: hist.counts().clone(),
            density: Array1::zeros(self.n_bins),
            density_sin_corrected: None,
            bin_width: hist.bin_width(),
            n_binned: hist.binned(),
            n_raw_samples,
            n_frames: frames.len(),
            angular: self.observable.is_angular(),
            finalized: false,
        };
        result.finalize();
        Ok(result)
    }
}

/// Result of a [`DistributionFunction`] over one or more frames.
///
/// `density` is the probability density (∫ p dx = 1). For an angular
/// observable, `density_sin_corrected` is `density / sin θ` renormalized to
/// unit integral (the solid-angle-corrected ADF).
#[derive(Debug, Clone)]
pub struct DistributionResult {
    /// Bin centers (n_bins), in the observable's units (Å or radians).
    pub bin_centers: Array1<F>,
    /// Bin edges (n_bins + 1).
    pub bin_edges: Array1<F>,
    /// Raw per-bin counts summed across frames.
    pub counts: Array1<F>,
    /// Normalized probability density (∫ p dx = 1). Populated by [`finalize`](ComputeResult::finalize).
    pub density: Array1<F>,
    /// sin θ-corrected density for angular observables, else `None`.
    pub density_sin_corrected: Option<Array1<F>>,
    /// Bin width.
    pub bin_width: F,
    /// In-range binned entries (sum of `counts`).
    pub n_binned: F,
    /// Total samples emitted by the observable (including out-of-range).
    pub n_raw_samples: usize,
    /// Number of frames.
    pub n_frames: usize,
    /// Whether the observable is angular (drives the sin θ correction).
    pub angular: bool,
    finalized: bool,
}

impl ComputeResult for DistributionResult {
    fn finalize(&mut self) {
        if self.finalized {
            return;
        }
        // Probability density: counts / (binned * bin_width) → ∫ p dx = 1.
        if self.n_binned > 0.0 {
            let denom = self.n_binned * self.bin_width;
            self.density = self.counts.mapv(|c| c / denom);
        } else {
            self.density = Array1::zeros(self.counts.len());
        }

        // sin θ solid-angle correction (ADF): divide by sin(center), renormalize.
        if self.angular {
            let mut w = Array1::<F>::zeros(self.counts.len());
            for i in 0..self.counts.len() {
                let s = self.bin_centers[i].sin();
                w[i] = if s.abs() > 1e-12 {
                    self.density[i] / s
                } else {
                    0.0
                };
            }
            self.density_sin_corrected = Some(renormalize_density(&w, self.bin_width));
        }
        self.finalized = true;
    }
}
