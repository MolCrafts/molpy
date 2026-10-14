//! The analysis contract: one trait, one verb.
//!
//! An analysis is a type whose constructor takes **configuration only** — bin
//! axes, cutoffs, `max_lag`, kernels, windows, tolerances, seeds, the unit
//! system — and whose [`compute`](Compute::compute) takes **the data being
//! analysed**: arrays, a [`SimBox`](molrs::core::SimBox), a
//! [`Neighbors`](molrs::core::Neighbors) table, an upstream result. A configured
//! analysis is a reusable instrument; data flows through `compute` and nowhere
//! else. Nothing here reads a `Frame`: taking columns out of a frame is the
//! caller's step, through the `Frame`/`Block` API.
//!
//! # `Input` is named arguments
//!
//! [`Compute::Input`] is, per analysis, a plain struct of borrowed data written
//! as a struct literal at the call site — Rust's spelling of named arguments.
//! It has no constructor: there is no data object to build before calling
//! `compute`. Bindings expose its fields as the keyword arguments of `compute`.
//!
//! ```
//! use molrs::compute::{Compute, ComputeError};
//!
//! /// Configuration: the factor. Data: the values, given to `compute`.
//! struct Scale { factor: f64 }
//! struct ScaleInput<'a> { values: &'a [f64] }
//!
//! impl Compute for Scale {
//!     type Input<'a> = ScaleInput<'a>;
//!     type Output = Vec<f64>;
//!     fn compute(&self, input: ScaleInput<'_>) -> Result<Vec<f64>, ComputeError> {
//!         Ok(input.values.iter().map(|v| v * self.factor).collect())
//!     }
//! }
//!
//! let doubled = Scale { factor: 2.0 }.compute(ScaleInput { values: &[1.0, 2.5] }).unwrap();
//! assert_eq!(doubled, vec![2.0, 5.0]);
//! ```
//!
//! A fit is a `Compute` whose input is an upstream output; a check is a
//! `Compute` whose output implements [`Verdict`]. One verb is what lets any
//! output feed the next step without glue.

use std::collections::BTreeMap;

use molrs::op::F;
use ndarray::{ArrayBase, ArrayView1, Axis, Data, Ix1, Ix2};

use crate::compute::ComputeError;

/// One analysis = one type = one verb.
///
/// `&self` is the configuration and never changes: identical `(self, input)`
/// gives an identical output. An output is final when it is returned — there
/// is no finalization step.
pub trait Compute {
    /// The data this analysis consumes, as a struct of named, borrowed fields.
    type Input<'a>;

    /// The result. Shareable across threads and downstream consumers.
    type Output: Clone + Send + Sync + 'static;

    /// Analyse `input` under this configuration.
    fn compute(&self, input: Self::Input<'_>) -> Result<Self::Output, ComputeError>;
}

/// The conclusion of a check, alongside the diagnostics that produced it.
///
/// Every check output answers "did it pass" without the caller knowing which
/// check ran; the diagnostics stay on the concrete type.
pub trait Verdict {
    /// Whether the data satisfied the criterion.
    fn passed(&self) -> bool;
}

/// The bounded-memory, item-by-item form of a [`Stream`] analysis.
///
/// An accumulator is obtained from a configured analysis through
/// [`Stream::accumulator`]; it receives data one item (typically one frame) at
/// a time through [`push`](Self::push).
pub trait Accumulate {
    /// One item of data, e.g. one frame's neighbor table or velocities.
    type Item<'a>;

    /// What [`finish`](Self::finish) reports.
    type Output;

    /// Fold one item into the running state.
    fn push(&mut self, item: Self::Item<'_>) -> Result<(), ComputeError>;

    /// The result over every item pushed so far. Does not consume the
    /// accumulator, so a long run can be inspected and continued.
    fn finish(&self) -> Result<Self::Output, ComputeError>;

    /// Push every item in order, then [`finish`](Self::finish). The first
    /// failing push ends the fold with its error.
    fn fold<'a, I>(mut self, items: I) -> Result<Self::Output, ComputeError>
    where
        Self: Sized,
        I: IntoIterator<Item = Self::Item<'a>>,
    {
        for item in items {
            self.push(item)?;
        }
        self.finish()
    }
}

/// An analysis that also streams.
///
/// The algorithm lives once, in the accumulator: a `Stream` type's
/// [`compute`](Compute::compute) body is exactly
/// `self.accumulator().fold(items)`, so batch and streaming answers cannot
/// drift apart. The one exception is an estimator with a faster batch
/// algorithm than its fold — the [`Correlator`](crate::compute::Correlator)'s
/// FFT — whose `compute` stays the batch algorithm; its unit tests pin the
/// two equal at every output point.
pub trait Stream: Compute {
    /// The accumulator this configuration produces.
    type Accumulator: Accumulate<Output = Self::Output>;

    /// A fresh accumulator configured like `self`.
    fn accumulator(&self) -> Self::Accumulator;
}

/// The same analysis applied to many independent inputs — one per snapshot —
/// in parallel when the `rayon` feature is on.
///
/// Results come back in input order. If several inputs fail, the error of the
/// earliest one is returned, so the outcome never depends on scheduling.
#[derive(Debug, Clone)]
pub struct Batch<C>(pub C);

impl<C> Compute for Batch<C>
where
    C: Compute + Sync,
    for<'a> C::Input<'a>: Send,
{
    type Input<'a> = Vec<C::Input<'a>>;
    type Output = Vec<C::Output>;

    fn compute(&self, inputs: Self::Input<'_>) -> Result<Self::Output, ComputeError> {
        #[cfg(feature = "rayon")]
        let outcomes: Vec<Result<C::Output, ComputeError>> = {
            use rayon::prelude::*;
            inputs
                .into_par_iter()
                .map(|input| self.0.compute(input))
                .collect()
        };
        #[cfg(not(feature = "rayon"))]
        let outcomes: Vec<Result<C::Output, ComputeError>> = inputs
            .into_iter()
            .map(|input| self.0.compute(input))
            .collect();
        outcomes.into_iter().collect()
    }
}

/// Which group each item belongs to.
///
/// Implemented by plain label arrays — a `mol_id` column read from a frame —
/// and by every result that partitions items (cluster, k-means, environment
/// match, hydrogen-bond components), so an analysis that reduces over groups
/// accepts either without a wrapper. Keys need not be contiguous; an analysis
/// orders groups by ascending key.
pub trait Partition {
    /// Number of items the partition covers.
    fn n_items(&self) -> usize;

    /// The group key of `item`, or `None` if the item belongs to no group.
    /// `item` must be below [`n_items`](Self::n_items).
    fn group_of(&self, item: usize) -> Option<u64>;

    /// The number of items in each group, keyed — and so iterated — by
    /// ascending group key, the order every group reduction reports its
    /// groups in. Items in no group are not counted.
    fn sizes(&self) -> BTreeMap<u64, usize> {
        let mut sizes = BTreeMap::new();
        for item in 0..self.n_items() {
            if let Some(key) = self.group_of(item) {
                *sizes.entry(key).or_insert(0) += 1;
            }
        }
        sizes
    }
}

/// A label array is a partition: every item belongs to the group its label
/// names.
impl<T: Copy + Into<u64>> Partition for &[T] {
    fn n_items(&self) -> usize {
        self.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        Some(self[item].into())
    }
}

impl<T: Copy + Into<u64>> Partition for Vec<T> {
    fn n_items(&self) -> usize {
        self.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        Some(self[item].into())
    }
}

impl<S> Partition for ArrayBase<S, Ix1>
where
    S: Data,
    S::Elem: Copy + Into<u64>,
{
    fn n_items(&self) -> usize {
        self.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        Some(self[item].into())
    }
}

/// A rows × columns table of features, one row per sample.
///
/// Implemented by any `ndarray` matrix and by every descriptor result, so a
/// decomposition or clustering takes either without a wrapper.
pub trait FeatureMatrix {
    /// Number of samples.
    fn n_rows(&self) -> usize;

    /// Number of features per sample.
    fn n_cols(&self) -> usize;

    /// The features of sample `row`.
    fn row(&self, row: usize) -> ArrayView1<'_, F>;
}

impl<S: Data<Elem = F>> FeatureMatrix for ArrayBase<S, Ix2> {
    fn n_rows(&self) -> usize {
        self.nrows()
    }

    fn n_cols(&self) -> usize {
        self.ncols()
    }

    fn row(&self, row: usize) -> ArrayView1<'_, F> {
        self.index_axis(Axis(0), row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};

    /// Configuration: a factor. Data: values.
    struct Scale {
        factor: F,
    }

    #[derive(Clone, Copy)]
    struct ScaleInput<'a> {
        values: &'a [F],
    }

    impl Compute for Scale {
        type Input<'a> = ScaleInput<'a>;
        type Output = Vec<F>;

        fn compute(&self, input: ScaleInput<'_>) -> Result<Vec<F>, ComputeError> {
            if input.values.is_empty() {
                return Err(ComputeError::EmptyInput);
            }
            if let Some(index) = input.values.iter().position(|v| !v.is_finite()) {
                return Err(ComputeError::NonFinite {
                    where_: "values",
                    index,
                });
            }
            Ok(input.values.iter().map(|v| v * self.factor).collect())
        }
    }

    /// A streaming mean over frames of values.
    struct Mean;

    struct MeanAccumulator {
        sum: F,
        count: usize,
    }

    impl Accumulate for MeanAccumulator {
        type Item<'a> = &'a [F];
        type Output = F;

        fn push(&mut self, frame: &[F]) -> Result<(), ComputeError> {
            if let Some(index) = frame.iter().position(|v| !v.is_finite()) {
                return Err(ComputeError::NonFinite {
                    where_: "frame",
                    index,
                });
            }
            self.sum += frame.iter().sum::<F>();
            self.count += frame.len();
            Ok(())
        }

        fn finish(&self) -> Result<F, ComputeError> {
            if self.count == 0 {
                return Err(ComputeError::EmptyInput);
            }
            Ok(self.sum / self.count as F)
        }
    }

    impl Compute for Mean {
        type Input<'a> = &'a [&'a [F]];
        type Output = F;

        fn compute(&self, frames: &[&[F]]) -> Result<F, ComputeError> {
            self.accumulator().fold(frames.iter().copied())
        }
    }

    impl Stream for Mean {
        type Accumulator = MeanAccumulator;

        fn accumulator(&self) -> MeanAccumulator {
            MeanAccumulator { sum: 0.0, count: 0 }
        }
    }

    #[test]
    fn compute_is_a_pure_function_of_configuration_and_data() {
        let scale = Scale { factor: 3.0 };
        let input = ScaleInput {
            values: &[1.0, -2.0],
        };

        assert_eq!(scale.compute(input).unwrap(), vec![3.0, -6.0]);
        assert_eq!(scale.compute(input).unwrap(), scale.compute(input).unwrap());
    }

    #[test]
    fn fold_equals_push_then_finish() {
        let frames: [&[F]; 2] = [&[1.0, 2.0], &[6.0]];

        let mut manual = Mean.accumulator();
        for frame in frames {
            manual.push(frame).unwrap();
        }

        assert_eq!(Mean.accumulator().fold(frames).unwrap(), 3.0);
        assert_eq!(manual.finish().unwrap(), 3.0);
        assert_eq!(Mean.compute(&frames).unwrap(), 3.0);
    }

    #[test]
    fn finish_peeks_without_consuming() {
        let mut acc = Mean.accumulator();
        acc.push(&[2.0]).unwrap();
        assert_eq!(acc.finish().unwrap(), 2.0);

        acc.push(&[4.0]).unwrap();
        assert_eq!(acc.finish().unwrap(), 3.0);
    }

    #[test]
    fn fold_stops_at_the_first_failing_push() {
        let frames: [&[F]; 3] = [&[1.0], &[F::NAN], &[F::INFINITY]];

        let err = Mean.accumulator().fold(frames).unwrap_err();

        assert!(matches!(
            err,
            ComputeError::NonFinite {
                where_: "frame",
                index: 0
            }
        ));
    }

    #[test]
    fn fold_over_nothing_is_empty_input() {
        let none: [&[F]; 0] = [];
        assert!(matches!(
            Mean.accumulator().fold(none),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn batch_keeps_input_order() {
        let batch = Batch(Scale { factor: 2.0 });
        let inputs = vec![
            ScaleInput { values: &[1.0] },
            ScaleInput {
                values: &[2.0, 3.0],
            },
            ScaleInput { values: &[4.0] },
        ];

        let out = batch.compute(inputs).unwrap();

        assert_eq!(out, vec![vec![2.0], vec![4.0, 6.0], vec![8.0]]);
    }

    /// Inputs 1 and 3 both fail; the reported error is input 1's, whatever
    /// order the threads finished in.
    #[test]
    fn batch_reports_the_earliest_failure() {
        let batch = Batch(Scale { factor: 2.0 });
        let inputs = vec![
            ScaleInput { values: &[1.0] },
            ScaleInput { values: &[] },
            ScaleInput { values: &[3.0] },
            ScaleInput {
                values: &[0.0, F::NAN],
            },
        ];

        assert!(matches!(
            batch.compute(inputs),
            Err(ComputeError::EmptyInput)
        ));
    }

    #[test]
    fn label_arrays_are_partitions() {
        let mol_id: &[u64] = &[7, 3, 7, 9];
        let p: &dyn Partition = &mol_id;
        assert_eq!(p.n_items(), 4);
        assert_eq!(
            (0..4).map(|i| p.group_of(i)).collect::<Vec<_>>(),
            vec![Some(7), Some(3), Some(7), Some(9)]
        );

        let labels: Array1<u32> = array![0, 1, 1];
        let p: &dyn Partition = &labels;
        assert_eq!(p.n_items(), 3);
        assert_eq!(p.group_of(2), Some(1));

        let owned: Vec<u32> = vec![5, 5];
        let p: &dyn Partition = &owned;
        assert_eq!(p.group_of(1), Some(5));
    }

    #[test]
    fn sizes_are_ordered_by_ascending_key() {
        let mol_id: &[u64] = &[7, 3, 7, 9];
        let sizes: Vec<(u64, usize)> = (&mol_id as &dyn Partition).sizes().into_iter().collect();
        assert_eq!(sizes, vec![(3, 1), (7, 2), (9, 1)]);
    }

    #[test]
    fn matrices_are_feature_matrices() {
        let m = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let f: &dyn FeatureMatrix = &m;
        assert_eq!((f.n_rows(), f.n_cols()), (2, 3));
        assert_eq!(f.row(1).to_vec(), vec![4.0, 5.0, 6.0]);

        // A transposed view is not contiguous by row; rows still read right.
        let t = m.t();
        let f: &dyn FeatureMatrix = &t;
        assert_eq!((f.n_rows(), f.n_cols()), (3, 2));
        assert_eq!(f.row(2).to_vec(), vec![3.0, 6.0]);
    }
}
