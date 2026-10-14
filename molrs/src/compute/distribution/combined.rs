//! Combined Distribution Functions (CDF): joint 2-D / 3-D observable histograms.

use molrs::core::{Mic, SimBox};
use molrs::op::F;
use ndarray::{Array1, ArrayD, ArrayView2};

use super::observable::Observable;
use super::{AngleObservable, DihedralObservable, DistanceObservable, DistributionResult};
use crate::compute::named_enum::named_enum;
use crate::compute::{Accumulate, Axis, Compute, ComputeError, Histogram, Stream};
use crate::core::UnitPreset;

named_enum! {
    /// An internal coordinate a distribution can bin: the name-selectable
    /// [`Observable`] of a [`CombinedDistribution`] axis.
    pub enum InternalCoordinate {
        /// [`DistanceObservable`]: a pair's minimum-image distance (arity 2).
        Distance => "distance",
        /// [`AngleObservable`]: the vertex angle of a triple (arity 3).
        Angle => "angle",
        /// [`DihedralObservable`]: the signed torsion of a quadruple (arity 4).
        Dihedral => "dihedral",
    }
}

impl Observable for InternalCoordinate {
    fn arity(&self) -> usize {
        match self {
            Self::Distance => DistanceObservable.arity(),
            Self::Angle => AngleObservable.arity(),
            Self::Dihedral => DihedralObservable.arity(),
        }
    }

    fn is_angular(&self) -> bool {
        match self {
            Self::Distance => DistanceObservable.is_angular(),
            Self::Angle => AngleObservable.is_angular(),
            Self::Dihedral => DihedralObservable.is_angular(),
        }
    }

    fn natural_range(&self) -> Option<(F, F)> {
        match self {
            Self::Distance => DistanceObservable.natural_range(),
            Self::Angle => AngleObservable.natural_range(),
            Self::Dihedral => DihedralObservable.natural_range(),
        }
    }

    fn sample_into(
        &self,
        positions: ArrayView2<'_, F>,
        mic: &Mic,
        tuples: ArrayView2<'_, usize>,
        out: &mut Vec<F>,
    ) -> Result<(), ComputeError> {
        match self {
            Self::Distance => DistanceObservable.sample_into(positions, mic, tuples, out),
            Self::Angle => AngleObservable.sample_into(positions, mic, tuples, out),
            Self::Dihedral => DihedralObservable.sample_into(positions, mic, tuples, out),
        }
    }
}

/// A joint distribution of `N` (2 or 3) [`InternalCoordinate`]s over aligned
/// tuple arrays.
///
/// Configuration: one observable and one [`Axis`] per dimension, plus which
/// axes are angular for their marginal's sin θ correction. Data, per frame
/// ([`CombinedDistributionFrame`]): positions, an optional box, and one
/// atom-index tuple array per axis. Row `t` of every array is one joint
/// sample, so all arrays must have the same number of rows — a mismatch is a
/// typed [`ComputeError`], never a silent zip-truncation.
///
/// Ported from the reference implementation (`src/2df.cpp`, `src/2df.h`,
/// `src/3df.h`):
/// - the skip-out-of-range / running-entry bookkeeping of
///   `C2DF::AddToBin(double x, double y)` (a tuple outside any axis is
///   skipped whole);
/// - the **bilinear / trilinear cloud-in-cell** deposition of
///   `C2DF`/`C3DF::AddToBin`: each sample spreads over the `2^N` straddling
///   bins as the tensor product of the per-axis linear weights
///   ([`Histogram`]'s linear deposit), so summing the joint histogram over the
///   other axes reproduces the 1-D [`DistributionFunction`](super::DistributionFunction)
///   of that axis (the CDF marginal-consistency contract).
///
/// A [`Stream`]: `compute` is its accumulator folded over the frames.
///
/// # References
/// - Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**, 51, 2007–2023 (reference implementation).
/// - Brehm et al., *J. Chem. Phys.* **2020**, 152, 164105.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CombinedDistribution<const N: usize> {
    observables: [InternalCoordinate; N],
    axes: [Axis; N],
    sin_weight: [bool; N],
}

/// One frame of a [`CombinedDistribution`].
#[derive(Debug, Clone, Copy)]
pub struct CombinedDistributionFrame<'a, const N: usize> {
    /// Positions, `n_atoms × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box for minimum-image displacements; `None` for open boundaries.
    pub simbox: Option<&'a SimBox>,
    /// One `n_tuples × arity` atom-index array per axis, row-aligned.
    pub tuples: [ArrayView2<'a, usize>; N],
}

impl<const N: usize> CombinedDistribution<N> {
    /// Joint histogram of `observables` binned on `axes` (N ∈ {2, 3}).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `N` outside `{2, 3}`.
    pub fn new(
        observables: [InternalCoordinate; N],
        axes: [Axis; N],
    ) -> Result<Self, ComputeError> {
        if !(2..=3).contains(&N) {
            return Err(ComputeError::OutOfRange {
                field: "CombinedDistribution::ndim",
                value: format!("{N} (only 2-D and 3-D supported)"),
            });
        }
        Ok(Self {
            observables,
            axes,
            sin_weight: [false; N],
        })
    }

    /// Mark `axis` angular, so its [`marginal`](CombinedDistributionResult::marginal)
    /// carries the sin θ-corrected density.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `axis ≥ N`.
    pub fn with_sin_weight(mut self, axis: usize) -> Result<Self, ComputeError> {
        if axis >= N {
            return Err(ComputeError::OutOfRange {
                field: "CombinedDistribution::sin_weight axis",
                value: format!("{axis} (the distribution has {N} axes)"),
            });
        }
        self.sin_weight[axis] = true;
        Ok(self)
    }

    /// The binned axes.
    pub fn axes(&self) -> &[Axis; N] {
        &self.axes
    }

    /// The observables, one per axis.
    pub fn observables(&self) -> &[InternalCoordinate; N] {
        &self.observables
    }
}

impl<const N: usize> Compute for CombinedDistribution<N> {
    type Input<'a> = &'a [CombinedDistributionFrame<'a, N>];
    type Output = CombinedDistributionResult<N>;

    fn compute(&self, frames: Self::Input<'_>) -> Result<Self::Output, ComputeError> {
        self.accumulator().fold(frames.iter().copied())
    }
}

impl<const N: usize> Stream for CombinedDistribution<N> {
    type Accumulator = CombinedDistributionAccumulator<N>;

    fn accumulator(&self) -> Self::Accumulator {
        CombinedDistributionAccumulator {
            config: *self,
            histogram: Histogram::new(self.axes),
            n_raw_samples: 0,
            n_frames: 0,
            samples: std::array::from_fn(|_| Vec::new()),
        }
    }
}

/// The frame-by-frame form of a [`CombinedDistribution`].
#[derive(Debug, Clone)]
pub struct CombinedDistributionAccumulator<const N: usize> {
    config: CombinedDistribution<N>,
    histogram: Histogram<N>,
    n_raw_samples: usize,
    n_frames: usize,
    /// Per-axis sample buffers, reused across frames.
    samples: [Vec<F>; N],
}

impl<const N: usize> Accumulate for CombinedDistributionAccumulator<N> {
    type Item<'a> = CombinedDistributionFrame<'a, N>;
    type Output = CombinedDistributionResult<N>;

    /// Deposit one frame's joint samples.
    ///
    /// # Errors
    /// Tuple-array errors of the observables; [`ComputeError::DimensionMismatch`]
    /// for tuple arrays of different lengths; [`ComputeError::NonFinite`] for
    /// a non-finite sample. A rejected frame deposits nothing.
    fn push(&mut self, frame: CombinedDistributionFrame<'_, N>) -> Result<(), ComputeError> {
        let mic = frame.simbox.map_or(Mic::Free, SimBox::mic);
        let n_tuples = frame.tuples[0].nrows();
        for (a, tuples) in frame.tuples.iter().enumerate() {
            if tuples.nrows() != n_tuples {
                return Err(ComputeError::DimensionMismatch {
                    expected: n_tuples,
                    got: tuples.nrows(),
                    what: "per-axis tuple count (all axes must match axis 0)",
                });
            }
            self.config.observables[a].sample_into(
                frame.positions,
                &mic,
                *tuples,
                &mut self.samples[a],
            )?;
            if let Some(index) = self.samples[a].iter().position(|v| !v.is_finite()) {
                return Err(ComputeError::NonFinite {
                    where_: "observable sample",
                    index,
                });
            }
        }
        for t in 0..n_tuples {
            let x: [F; N] = std::array::from_fn(|a| self.samples[a][t]);
            self.histogram.add_linear(x, 1.0)?;
        }
        self.n_raw_samples += n_tuples;
        self.n_frames += 1;
        Ok(())
    }

    /// # Errors
    /// [`ComputeError::EmptyInput`] before any frame, or when no sample fell
    /// inside the axes (an empty histogram has no density).
    fn finish(&self) -> Result<CombinedDistributionResult<N>, ComputeError> {
        if self.n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let sin_corrected: [bool; N] = std::array::from_fn(|a| {
            self.config.observables[a].is_angular() || self.config.sin_weight[a]
        });
        Ok(CombinedDistributionResult {
            density: self.histogram.density()?,
            histogram: self.histogram.clone(),
            n_raw_samples: self.n_raw_samples,
            n_frames: self.n_frames,
            sin_corrected,
        })
    }
}

/// Result of a [`CombinedDistribution`]: the joint histogram, its normalized
/// density, and the [`marginal`](Self::marginal) /
/// [`free_energy`](Self::free_energy) views.
#[derive(Debug, Clone, PartialEq)]
pub struct CombinedDistributionResult<const N: usize> {
    /// Joint counts, indexed by bin tuple, summed across frames.
    pub histogram: Histogram<N>,
    /// Joint probability density (`Σ density · bin_volume = 1`), indexed like
    /// the counts.
    pub density: ArrayD<F>,
    /// Total joint samples emitted (including out-of-range ones).
    pub n_raw_samples: usize,
    /// Number of frames.
    pub n_frames: usize,
    /// Per axis: whether its marginal carries the sin θ-corrected density.
    pub sin_corrected: [bool; N],
}

impl<const N: usize> CombinedDistributionResult<N> {
    /// The 1-D marginal along `axis` (the joint counts summed over the other
    /// axes): the 1-D distribution of that axis's observable within rounding
    /// (the CDF marginal-consistency contract).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `axis ≥ N`; as
    /// [`DistributionResult`]'s normalization otherwise.
    pub fn marginal(&self, axis: usize) -> Result<DistributionResult, ComputeError> {
        if axis >= N {
            return Err(ComputeError::OutOfRange {
                field: "CombinedDistributionResult::marginal axis",
                value: format!("{axis} (the distribution has {N} axes)"),
            });
        }
        let along = self.histogram.axes()[axis];
        let mut counts = Array1::<F>::zeros(along.n_bins());
        for (bin, &c) in self.histogram.counts().indexed_iter() {
            counts[bin[axis]] += c;
        }
        let histogram = Histogram::from_counts([along], counts.into_dyn());
        DistributionResult::new(
            histogram,
            self.n_raw_samples,
            self.n_frames,
            self.sin_corrected[axis],
        )
    }

    /// Free-energy surface `G = −k_B T · ln p` (kcal/mol), with `p` the joint
    /// density and `temperature` in K. Populated bins are finite; **empty bins
    /// are floored** to the maximum populated-bin free energy (the highest
    /// finite barrier) rather than `+∞`, so the surface is finite everywhere.
    pub fn free_energy(&self, temperature: F) -> ArrayD<F> {
        let kt = UnitPreset::real().boltzmann() * temperature;
        let mut g = ArrayD::<F>::zeros(self.density.raw_dim());
        let mut g_max = F::NEG_INFINITY;
        for (gi, &p) in g.iter_mut().zip(self.density.iter()) {
            if p > 0.0 {
                *gi = -kt * p.ln();
                g_max = g_max.max(*gi);
            }
        }
        let floor = if g_max.is_finite() { g_max } else { 0.0 };
        for (gi, &p) in g.iter_mut().zip(self.density.iter()) {
            if p <= 0.0 {
                *gi = floor;
            }
        }
        g
    }
}

crate::compute::spec::compute_spec! {
    @[const N: usize] CombinedDistribution<N> as "CombinedDistribution", Frames;
    params |s| {
        observables: ChoiceList(InternalCoordinate::NAMES);
        axes: AxisList;
        sin_weight: IntList
            = crate::compute::ParamDefault::UNSET
            => s.sin_weight.contains(&true).then(|| {
                crate::compute::ParamValue::IntList(
                    (0..N).filter(|&axis| s.sin_weight[axis]).map(|axis| axis as u64).collect(),
                )
            });
    }
    inputs {
        positions: Positions;
        simbox: SimBox, optional;
        tuples: TuplesList;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{DistributionFrame, DistributionFunction};
    use ndarray::{Array2, array};

    /// A bent, twisted chain: distances, angles and torsions all vary.
    fn chain(n: usize) -> Array2<F> {
        Array2::from_shape_fn((n, 3), |(i, c)| {
            let t = i as F;
            match c {
                0 => t * 1.1,
                1 => (t * 0.9).sin(),
                _ => (t * 0.6).cos() * 0.8,
            }
        })
    }

    #[test]
    fn coordinate_names_round_trip() {
        for &c in InternalCoordinate::ALL {
            assert_eq!(c.as_str().parse::<InternalCoordinate>().unwrap(), c);
        }
        assert_eq!(InternalCoordinate::Angle.arity(), 3);
        assert!("torsion".parse::<InternalCoordinate>().is_err());
    }

    #[test]
    fn ndim_and_sin_axis_are_validated() {
        let ax = Axis::new(0.0, 1.0, 4).unwrap();
        assert!(matches!(
            CombinedDistribution::new([InternalCoordinate::Distance], [ax]),
            Err(ComputeError::OutOfRange { .. })
        ));
        let cdf = CombinedDistribution::new([InternalCoordinate::Distance; 2], [ax; 2]).unwrap();
        assert!(matches!(
            cdf.with_sin_weight(2),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// The marginal of a distance × angle CDF is the 1-D distance
    /// distribution of the same pairs.
    #[test]
    fn marginal_matches_the_one_dimensional_distribution() {
        let positions = chain(12);
        let pairs = Array2::from_shape_fn((10, 2), |(t, k)| t + k);
        let triples = Array2::from_shape_fn((10, 3), |(t, k)| t + k);
        let r_axis = Axis::new(0.5, 2.5, 16).unwrap();
        let theta_axis = Axis::new(0.0, std::f64::consts::PI, 9).unwrap();

        let cdf = CombinedDistribution::new(
            [InternalCoordinate::Distance, InternalCoordinate::Angle],
            [r_axis, theta_axis],
        )
        .unwrap();
        let joint = cdf
            .compute(&[CombinedDistributionFrame {
                positions: positions.view(),
                simbox: None,
                tuples: [pairs.view(), triples.view()],
            }])
            .unwrap();
        assert_eq!(joint.n_raw_samples, 10);
        assert!((joint.histogram.binned() - 10.0).abs() < 1e-12);

        let one_d = DistributionFunction::new(DistanceObservable, r_axis)
            .compute(&[DistributionFrame {
                positions: positions.view(),
                simbox: None,
                tuples: pairs.view(),
            }])
            .unwrap();
        let marginal = joint.marginal(0).unwrap();
        for (a, b) in marginal.density.iter().zip(one_d.density.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
        // The angle marginal is sin θ-corrected because the axis is angular.
        assert!(joint.marginal(1).unwrap().density_sin_corrected.is_some());
        assert!(joint.marginal(0).unwrap().density_sin_corrected.is_none());
        assert!(joint.free_energy(300.0).iter().all(|g| g.is_finite()));
    }

    #[test]
    fn misaligned_tuple_arrays_are_dimension_mismatch() {
        let positions = chain(6);
        let pairs = array![[0usize, 1], [1, 2]];
        let triples = array![[0usize, 1, 2]];
        let ax = Axis::new(0.0, 4.0, 4).unwrap();
        let cdf = CombinedDistribution::new(
            [InternalCoordinate::Distance, InternalCoordinate::Angle],
            [ax, Axis::new(0.0, 3.2, 4).unwrap()],
        )
        .unwrap();
        assert!(matches!(
            cdf.compute(&[CombinedDistributionFrame {
                positions: positions.view(),
                simbox: None,
                tuples: [pairs.view(), triples.view()],
            }]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        // An empty histogram is an error, not a zero density.
        let far = Axis::new(100.0, 200.0, 4).unwrap();
        let cdf = CombinedDistribution::new([InternalCoordinate::Distance; 2], [far; 2]).unwrap();
        assert!(matches!(
            cdf.compute(&[CombinedDistributionFrame {
                positions: positions.view(),
                simbox: None,
                tuples: [pairs.view(), pairs.view()],
            }]),
            Err(ComputeError::EmptyInput)
        ));
    }
}
