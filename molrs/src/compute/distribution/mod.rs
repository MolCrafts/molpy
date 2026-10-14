//! User-defined geometric distribution functions (ADF / DDF / distance DF).

mod angle;
mod combined;
mod dihedral;
mod distance;
mod observable;

pub use angle::AngleObservable;
pub use combined::{
    CombinedDistribution, CombinedDistributionAccumulator, CombinedDistributionFrame,
    CombinedDistributionResult, InternalCoordinate,
};
pub use dihedral::DihedralObservable;
pub use distance::DistanceObservable;
pub use observable::Observable;

use molrs::core::{Mic, SimBox};
use molrs::op::F;
use ndarray::{Array1, ArrayView2};

use crate::compute::{Accumulate, Axis, Compute, ComputeError, Histogram, Stream};

/// A 1-D distribution function over the samples of an [`Observable`].
///
/// Configuration: the observable and its [`Axis`]. Data, per frame
/// ([`DistributionFrame`]): the positions, an optional box (minimum image)
/// and the `n_tuples × arity` atom-index tuples to evaluate. Samples are
/// deposited by cloud-in-cell into a [`Histogram`], so the distance DF agrees
/// with [`Rdf`](crate::compute::Rdf) on the minimum-image convention. For an
/// angular observable the sin θ-corrected density is returned alongside the
/// raw one, because conflating them is the most common ADF mistake.
///
/// A [`Stream`]: `compute` is its accumulator folded over the frames.
///
/// Ported from the reference implementation (`src/tddf.cpp`,
/// `src/geodens.cpp`, `src/df.cpp`).
///
/// # References
/// - Brehm & Kirchner, *J. Chem. Inf. Model.* **2011**, 51, 2007–2023 (reference implementation).
/// - Brehm, Thomas, Gehrke, Kirchner, *J. Chem. Phys.* **2020**, 152, 164105.
#[derive(Debug, Clone, PartialEq)]
pub struct DistributionFunction<O: Observable> {
    observable: O,
    axis: Axis,
}

/// One frame of a [`DistributionFunction`].
#[derive(Debug, Clone, Copy)]
pub struct DistributionFrame<'a> {
    /// Positions, `n_atoms × 3`.
    pub positions: ArrayView2<'a, F>,
    /// The box for minimum-image displacements; `None` for open boundaries.
    pub simbox: Option<&'a SimBox>,
    /// Atom-index tuples, `n_tuples × arity` of the observable.
    pub tuples: ArrayView2<'a, usize>,
}

impl<O: Observable> DistributionFunction<O> {
    /// Bin the observable's samples on `axis`.
    pub fn new(observable: O, axis: Axis) -> Self {
        Self { observable, axis }
    }

    /// Bin over the observable's [`natural_range`](Observable::natural_range)
    /// (e.g. `[0, π]` for an angle) in `n_bins` bins.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] if the observable has no natural range
    /// (distance — give an explicit [`Axis`]) or `n_bins == 0`.
    pub fn over_natural_range(observable: O, n_bins: usize) -> Result<Self, ComputeError> {
        let (min, max) = observable.natural_range().ok_or(ComputeError::OutOfRange {
            field: "DistributionFunction::natural_range",
            value: "observable has no natural range".to_string(),
        })?;
        Ok(Self::new(observable, Axis::new(min, max, n_bins)?))
    }

    /// The observable.
    pub fn observable(&self) -> &O {
        &self.observable
    }

    /// The binned axis.
    pub fn axis(&self) -> Axis {
        self.axis
    }
}

impl<O: Observable + Clone> Compute for DistributionFunction<O> {
    type Input<'a> = &'a [DistributionFrame<'a>];
    type Output = DistributionResult;

    fn compute(&self, frames: Self::Input<'_>) -> Result<DistributionResult, ComputeError> {
        self.accumulator().fold(frames.iter().copied())
    }
}

impl<O: Observable + Clone> Stream for DistributionFunction<O> {
    type Accumulator = DistributionAccumulator<O>;

    fn accumulator(&self) -> DistributionAccumulator<O> {
        DistributionAccumulator {
            observable: self.observable.clone(),
            histogram: Histogram::new([self.axis]),
            n_raw_samples: 0,
            n_frames: 0,
            samples: Vec::new(),
        }
    }
}

/// The frame-by-frame form of a [`DistributionFunction`].
#[derive(Debug, Clone)]
pub struct DistributionAccumulator<O: Observable> {
    observable: O,
    histogram: Histogram<1>,
    n_raw_samples: usize,
    n_frames: usize,
    /// Sample buffer, reused across frames.
    samples: Vec<F>,
}

impl<O: Observable> Accumulate for DistributionAccumulator<O> {
    type Item<'a> = DistributionFrame<'a>;
    type Output = DistributionResult;

    /// Deposit one frame's samples.
    ///
    /// # Errors
    /// The observable's tuple and geometry errors;
    /// [`ComputeError::NonFinite`] for a non-finite sample. A rejected frame
    /// deposits nothing.
    fn push(&mut self, frame: DistributionFrame<'_>) -> Result<(), ComputeError> {
        let mic = frame.simbox.map_or(Mic::Free, SimBox::mic);
        self.observable
            .sample_into(frame.positions, &mic, frame.tuples, &mut self.samples)?;
        if let Some(index) = self.samples.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "observable sample",
                index,
            });
        }
        for &s in &self.samples {
            self.histogram.add_linear([s], 1.0)?;
        }
        self.n_raw_samples += self.samples.len();
        self.n_frames += 1;
        Ok(())
    }

    /// # Errors
    /// [`ComputeError::EmptyInput`] before any frame, or when no sample fell
    /// inside the axis (an empty histogram has no density).
    fn finish(&self) -> Result<DistributionResult, ComputeError> {
        if self.n_frames == 0 {
            return Err(ComputeError::EmptyInput);
        }
        DistributionResult::new(
            self.histogram.clone(),
            self.n_raw_samples,
            self.n_frames,
            self.observable.is_angular(),
        )
    }
}

/// Result of a [`DistributionFunction`] over one or more frames.
///
/// `density` is the probability density (∫ p dx = 1). For an angular
/// observable, `density_sin_corrected` is `density / sin θ` renormalized to
/// unit integral (the solid-angle-corrected ADF).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct DistributionResult {
    /// Per-bin counts summed across frames; its axis gives the edges and
    /// centres (in the observable's units, Å or radians).
    pub histogram: Histogram<1>,
    /// Normalized probability density (∫ p dx = 1).
    pub density: Array1<F>,
    /// sin θ-corrected density for angular observables, else `None`.
    pub density_sin_corrected: Option<Array1<F>>,
    /// Total samples emitted by the observable (including out-of-range).
    pub n_raw_samples: usize,
    /// Number of frames.
    pub n_frames: usize,
}

impl DistributionResult {
    /// Normalize `histogram`; with `sin_corrected`, also divide by `sin θ`
    /// at each bin centre and renormalize to unit integral.
    ///
    /// # Errors
    /// [`ComputeError::EmptyInput`] for an empty histogram;
    /// [`ComputeError::OutOfRange`] when the sin θ-weighted density has no
    /// positive integral (an angular axis outside `(0, π)`).
    pub(crate) fn new(
        histogram: Histogram<1>,
        n_raw_samples: usize,
        n_frames: usize,
        sin_corrected: bool,
    ) -> Result<Self, ComputeError> {
        let density: Array1<F> = histogram.density()?.iter().copied().collect();
        let axis = histogram.axes()[0];
        let density_sin_corrected = if sin_corrected {
            let weights: Array1<F> = density
                .iter()
                .zip(axis.centers().iter())
                .map(|(&p, &center)| {
                    let s = center.sin();
                    if s.abs() > 1e-12 { p / s } else { 0.0 }
                })
                .collect();
            let total = weights.sum() * axis.width();
            if !(total.is_finite() && total > 0.0) {
                return Err(ComputeError::OutOfRange {
                    field: "sin θ-corrected density (integral)",
                    value: total.to_string(),
                });
            }
            Some(weights.mapv(|w| w / total))
        } else {
            None
        };
        Ok(Self {
            histogram,
            density,
            density_sin_corrected,
            n_raw_samples,
            n_frames,
        })
    }
}

crate::compute::spec::compute_spec! {
    DistributionFunction<InternalCoordinate> as "DistributionFunction", Frames;
    params |s| {
        observable: Choice(InternalCoordinate::NAMES);
        axis: Axis = crate::compute::ParamDefault::RequiredIn("new");
        n_bins: Int = crate::compute::ParamDefault::RequiredIn("over_natural_range");
    }
    inputs {
        positions: Positions;
        simbox: SimBox, optional;
        tuples: Tuples;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use std::f64::consts::PI;

    fn frame<'a>(positions: &'a Array2<F>, tuples: &'a Array2<usize>) -> DistributionFrame<'a> {
        DistributionFrame {
            positions: positions.view(),
            simbox: None,
            tuples: tuples.view(),
        }
    }

    /// Cloud-in-cell deposition of the distance DF: a distance at a bin
    /// centre stays whole, one between centres splits linearly.
    #[test]
    fn distance_df_deposits_by_cloud_in_cell() {
        let positions = array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.75, 0.0, 0.0]];
        let pairs = array![[0usize, 1], [0, 2]];
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 4).unwrap());
        let r = df.compute(&[frame(&positions, &pairs)]).unwrap();
        let counts: Vec<F> = r.histogram.counts().iter().copied().collect();
        assert!((counts[0] - 1.0).abs() < 1e-12);
        assert!((counts[1] - 0.75).abs() < 1e-12);
        assert!((counts[2] - 0.25).abs() < 1e-12);
        assert_eq!(r.n_raw_samples, 2);
        let integral: F = r.density.sum() * 1.0;
        assert!((integral - 1.0).abs() < 1e-12);
    }

    /// Minimum image applies when a box is given.
    #[test]
    fn distance_uses_the_minimum_image() {
        let positions = array![[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]];
        let pairs = array![[0usize, 1]];
        let simbox = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 4).unwrap());
        let r = df
            .compute(&[DistributionFrame {
                positions: positions.view(),
                simbox: Some(&simbox),
                tuples: pairs.view(),
            }])
            .unwrap();
        // |Δx| = 1 under the minimum image: halfway between centres 0.5, 1.5.
        let counts: Vec<F> = r.histogram.counts().iter().copied().collect();
        assert!((counts[0] - 0.5).abs() < 1e-12);
        assert!((counts[1] - 0.5).abs() < 1e-12);
    }

    /// The ADF over its natural range: a straight angle (θ = π, the closed
    /// upper end) is binned; the sin θ-corrected density integrates to one.
    #[test]
    fn angle_df_over_natural_range_with_sin_correction() {
        let positions = array![
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0]
        ];
        let triples = array![[0usize, 1, 2], [0, 1, 3]];
        let df = DistributionFunction::over_natural_range(AngleObservable, 6).unwrap();
        let r = df.compute(&[frame(&positions, &triples)]).unwrap();
        assert!((r.histogram.binned() - 2.0).abs() < 1e-12);
        let corrected = r.density_sin_corrected.unwrap();
        let width = PI / 6.0;
        assert!((corrected.sum() * width - 1.0).abs() < 1e-12);
        assert!(DistributionFunction::over_natural_range(DistanceObservable, 6).is_err());
    }

    /// Frames accumulate; streaming equals batch.
    #[test]
    fn frames_accumulate_and_stream_equals_batch() {
        let a = array![[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]];
        let b = array![[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]];
        let pairs = array![[0usize, 1]];
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 8).unwrap());
        let frames = [frame(&a, &pairs), frame(&b, &pairs)];
        let batch = df.compute(&frames).unwrap();
        let mut acc = df.accumulator();
        for f in frames {
            acc.push(f).unwrap();
        }
        assert_eq!(acc.finish().unwrap(), batch);
        assert_eq!(batch.n_frames, 2);
        assert!((batch.histogram.binned() - 2.0).abs() < 1e-12);
    }

    /// An empty selection, or samples all outside the axis, is an error —
    /// an empty histogram has no density.
    #[test]
    fn empty_histogram_is_an_error() {
        let positions = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let none = Array2::<usize>::zeros((0, 2));
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 4).unwrap());
        assert!(matches!(
            df.compute(&[frame(&positions, &none)]),
            Err(ComputeError::EmptyInput)
        ));
        let far = DistributionFunction::new(DistanceObservable, Axis::new(5.0, 9.0, 4).unwrap());
        let pairs = array![[0usize, 1]];
        assert!(matches!(
            far.compute(&[frame(&positions, &pairs)]),
            Err(ComputeError::EmptyInput)
        ));
        assert!(matches!(df.compute(&[]), Err(ComputeError::EmptyInput)));
    }

    /// An atom index past the positions used to panic; it is an error.
    #[test]
    fn out_of_range_atom_index_is_an_error() {
        let positions = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let pairs = array![[0usize, 5]];
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 4).unwrap());
        assert!(matches!(
            df.compute(&[frame(&positions, &pairs)]),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// A non-finite position gives a non-finite sample: an error, not bin 0.
    #[test]
    fn nan_sample_is_non_finite() {
        let positions = array![[0.0, 0.0, 0.0], [F::NAN, 0.0, 0.0]];
        let pairs = array![[0usize, 1]];
        let df = DistributionFunction::new(DistanceObservable, Axis::new(0.0, 4.0, 4).unwrap());
        assert!(matches!(
            df.compute(&[frame(&positions, &pairs)]),
            Err(ComputeError::NonFinite { .. })
        ));
    }
}
