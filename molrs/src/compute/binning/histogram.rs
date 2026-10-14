//! [`Histogram`] — weighted counts over `N` binned axes.

use molrs::op::F;
use ndarray::{ArrayD, IxDyn};

use super::Axis;
use crate::compute::ComputeError;

/// Weighted counts over `N` [`Axis`] — the output every binning analysis
/// builds its result from.
///
/// `counts` has shape `(n_bins₀, …, n_bins_{N−1})` and is indexed by the bin
/// tuple. [`binned`](Self::binned) is the total weight that landed inside the
/// axes; samples outside them are not counted.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize),
    serde(bound(serialize = "[Axis; N]: serde::Serialize"))
)]
pub struct Histogram<const N: usize> {
    axes: [Axis; N],
    counts: ArrayD<F>,
    binned: F,
}

impl<const N: usize> Histogram<N> {
    /// An empty histogram over `axes`.
    pub(crate) fn new(axes: [Axis; N]) -> Self {
        let shape: Vec<usize> = axes.iter().map(Axis::n_bins).collect();
        Self {
            axes,
            counts: ArrayD::zeros(IxDyn(&shape)),
            binned: 0.0,
        }
    }

    /// A histogram holding `counts` (shape must match `axes`), every one of
    /// them inside the axes.
    pub(crate) fn from_counts(axes: [Axis; N], counts: ArrayD<F>) -> Self {
        let binned = counts.sum();
        Self {
            axes,
            counts,
            binned,
        }
    }

    /// The binned axes.
    pub fn axes(&self) -> &[Axis; N] {
        &self.axes
    }

    /// The counts, indexed by bin tuple.
    pub fn counts(&self) -> &ArrayD<F> {
        &self.counts
    }

    /// Total weight inside the axes (the sum of [`counts`](Self::counts)).
    pub fn binned(&self) -> F {
        self.binned
    }

    /// The volume of one bin: the product of the axis widths.
    pub fn bin_volume(&self) -> F {
        self.axes.iter().map(Axis::width).product()
    }

    /// The probability density `counts / (binned · bin_volume)`, whose
    /// integral over the axes is 1.
    ///
    /// # Errors
    /// [`ComputeError::EmptyInput`] when nothing was binned: an empty
    /// histogram has no density.
    pub fn density(&self) -> Result<ArrayD<F>, ComputeError> {
        if self.binned <= 0.0 {
            return Err(ComputeError::EmptyInput);
        }
        let denom = self.binned * self.bin_volume();
        Ok(self.counts.mapv(|c| c / denom))
    }

    /// Count one sample in the half-open bin holding it ([`Axis::index`]).
    /// Returns whether it was inside every axis.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a non-finite coordinate; nothing is
    /// counted.
    pub(crate) fn add(&mut self, x: [F; N]) -> Result<bool, ComputeError> {
        let mut bin = [0usize; N];
        for (a, axis) in self.axes.iter().enumerate() {
            match axis.index(x[a])? {
                Some(i) => bin[a] = i,
                None => return Ok(false),
            }
        }
        self.counts[IxDyn(&bin)] += 1.0;
        self.binned += 1.0;
        Ok(true)
    }

    /// Deposit `weight` by cloud-in-cell: the tensor product of each axis's
    /// [`linear`](Axis::linear) split over the `2^N` bins whose centres
    /// straddle the sample. Returns whether it was inside every axis (whose
    /// support is closed, `[min, max]`).
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a non-finite coordinate; nothing is
    /// deposited.
    pub(crate) fn add_linear(&mut self, x: [F; N], weight: F) -> Result<bool, ComputeError> {
        let mut splits = [([0usize; 2], [0.0; 2]); N];
        for (a, axis) in self.axes.iter().enumerate() {
            match axis.linear(x[a])? {
                Some(split) => splits[a] = split,
                None => return Ok(false),
            }
        }
        let mut bin = [0usize; N];
        for corner in 0..(1usize << N) {
            let mut w = weight;
            for (a, (bins, weights)) in splits.iter().enumerate() {
                let pick = (corner >> a) & 1;
                bin[a] = bins[pick];
                w *= weights[pick];
            }
            if w != 0.0 {
                self.counts[IxDyn(&bin)] += w;
            }
        }
        self.binned += weight;
        Ok(true)
    }

    /// Add another histogram over the same axes (same shape) into this one.
    pub(crate) fn merge(&mut self, other: &Self) {
        self.counts += &other.counts;
        self.binned += other.binned;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis(min: F, max: F, n: usize) -> Axis {
        Axis::new(min, max, n).unwrap()
    }

    #[test]
    fn add_counts_half_open_bins() {
        let mut h = Histogram::new([axis(0.0, 4.0, 4)]);
        assert!(h.add([0.0]).unwrap());
        assert!(h.add([1.0]).unwrap());
        assert!(h.add([3.99]).unwrap());
        assert!(!h.add([4.0]).unwrap());
        assert!(!h.add([-0.1]).unwrap());
        assert_eq!(h.counts().as_slice().unwrap(), &[1.0, 1.0, 0.0, 1.0]);
        assert_eq!(h.binned(), 3.0);
    }

    /// A NaN sample used to land in bin 0; it is an error and counts nothing.
    #[test]
    fn nan_sample_is_non_finite_and_counts_nothing() {
        let mut h = Histogram::new([axis(0.0, 1.0, 2), axis(0.0, 1.0, 2)]);
        assert!(matches!(
            h.add([0.5, F::NAN]),
            Err(ComputeError::NonFinite { .. })
        ));
        assert!(matches!(
            h.add_linear([F::NAN, 0.5], 1.0),
            Err(ComputeError::NonFinite { .. })
        ));
        assert_eq!(h.binned(), 0.0);
        assert_eq!(h.counts().sum(), 0.0);
    }

    #[test]
    fn empty_density_is_an_error() {
        let h = Histogram::new([axis(0.0, 1.0, 3)]);
        assert!(matches!(h.density(), Err(ComputeError::EmptyInput)));
    }

    #[test]
    fn density_integrates_to_one() {
        let mut h = Histogram::new([axis(0.0, 10.0, 10)]);
        for i in 0..100 {
            h.add_linear([(i % 10) as F + 0.5], 1.0).unwrap();
        }
        let d = h.density().unwrap();
        let integral: F = d.iter().map(|&p| p * h.bin_volume()).sum();
        assert!((integral - 1.0).abs() < 1e-12, "integral = {integral}");
    }

    /// The 2-D linear deposit is the tensor product of the 1-D splits, and
    /// its marginal is the 1-D deposit.
    #[test]
    fn linear_deposit_is_a_tensor_product() {
        let mut h2 = Histogram::new([axis(0.0, 4.0, 4), axis(0.0, 2.0, 2)]);
        h2.add_linear([1.75, 1.0], 1.0).unwrap();
        let c = h2.counts();
        assert!((c[[1, 0]] - 0.375).abs() < 1e-15);
        assert!((c[[1, 1]] - 0.375).abs() < 1e-15);
        assert!((c[[2, 0]] - 0.125).abs() < 1e-15);
        assert!((c[[2, 1]] - 0.125).abs() < 1e-15);
        assert_eq!(h2.binned(), 1.0);

        let mut h1 = Histogram::new([axis(0.0, 4.0, 4)]);
        h1.add_linear([1.75], 1.0).unwrap();
        let marginal = c.sum_axis(ndarray::Axis(1));
        assert_eq!(
            marginal.iter().copied().collect::<Vec<_>>(),
            h1.counts().iter().copied().collect::<Vec<_>>()
        );
    }

    #[test]
    fn merge_sums_counts_and_weight() {
        let mut a = Histogram::new([axis(0.0, 2.0, 2)]);
        let mut b = a.clone();
        a.add([0.5]).unwrap();
        b.add([1.5]).unwrap();
        b.add([1.5]).unwrap();
        a.merge(&b);
        assert_eq!(a.counts().as_slice().unwrap(), &[1.0, 2.0]);
        assert_eq!(a.binned(), 3.0);
        let from = Histogram::from_counts(*a.axes(), a.counts().clone());
        assert_eq!(from, a);
    }
}
