//! Generic distance-binned correlation `⟨A_i · B_j⟩(r)`.

use molrs::core::{PairSource, QueryMode};
use molrs::op::F;
use ndarray::{Array1, ArrayView1};

use crate::compute::{Axis, Compute, ComputeError, require_dist_sq};

/// Distance-binned correlation of two per-point fields.
///
/// Mirrors `freud.density.CorrelationFunction`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/density/CorrelationFunction.cc)).
///
/// Configuration: the radial bins (`[r_min, r_max)` Å, half-open). Data: one
/// snapshot's pair source and the real scalar fields `A` (on the query
/// points) and `B` (on the reference points). The result is the histogram
/// `⟨A_i · B_j⟩(r)` averaged over all pairs in each shell `[r, r + dr)`;
/// empty bins report 0. A half-shell self-query contributes each pair from
/// both ends (`A_i B_j` and `A_j B_i`).
///
/// freud also supports complex-valued fields with the convention
/// `A_i · conj(B_j)`; the real version is implemented here. For many
/// snapshots, wrap in [`Batch`](crate::compute::Batch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrelationFunction {
    axis: Axis,
}

/// The data a [`CorrelationFunction`] analyses.
#[derive(Clone, Copy)]
pub struct CorrelationFunctionInput<'a> {
    /// The snapshot's pairs; must carry the `dist_sq` column.
    pub pairs: &'a (dyn PairSource + Sync),
    /// `A` on the query points (`i` side), one value per query point.
    pub values_a: ArrayView1<'a, F>,
    /// `B` on the reference points (`j` side), one value per reference point.
    pub values_b: ArrayView1<'a, F>,
}

impl CorrelationFunction {
    /// `n_bins` radial bins over `[r_min, r_max)` (Å).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for `r_min < 0` (or NaN), `n_bins == 0`
    /// or `r_max ≤ r_min`.
    pub fn new(n_bins: usize, r_max: F, r_min: F) -> Result<Self, ComputeError> {
        if r_min.is_nan() || r_min < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "CorrelationFunction::r_min",
                value: r_min.to_string(),
            });
        }
        Ok(Self {
            axis: Axis::new(r_min, r_max, n_bins)?,
        })
    }

    /// The radial bins.
    pub fn axis(&self) -> Axis {
        self.axis
    }
}

impl Compute for CorrelationFunction {
    type Input<'a> = CorrelationFunctionInput<'a>;
    type Output = CorrelationFunctionResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] when `values_a` / `values_b` do
    /// not have one value per query / reference point;
    /// [`ComputeError::BadShape`] for a source without `dist_sq`;
    /// [`ComputeError::NonFinite`] for a non-finite distance.
    fn compute(
        &self,
        input: CorrelationFunctionInput<'_>,
    ) -> Result<CorrelationFunctionResult, ComputeError> {
        let pairs = input.pairs;
        let (n_query, n_points, symmetric) = match pairs.mode() {
            QueryMode::SelfQuery { n_points } => (n_points, n_points, true),
            QueryMode::CrossQuery {
                n_query_points,
                n_points,
            } => (n_query_points, n_points, false),
        };
        if input.values_a.len() != n_query {
            return Err(ComputeError::DimensionMismatch {
                expected: n_query,
                got: input.values_a.len(),
                what: "CorrelationFunction values_a (one per query point)",
            });
        }
        if input.values_b.len() != n_points {
            return Err(ComputeError::DimensionMismatch {
                expected: n_points,
                got: input.values_b.len(),
                what: "CorrelationFunction values_b (one per reference point)",
            });
        }
        require_dist_sq(pairs)?;

        let n_bins = self.axis.n_bins();
        let mut sum = Array1::<F>::zeros(n_bins);
        let mut counts = Array1::<u64>::zeros(n_bins);
        let (a, b) = (input.values_a, input.values_b);
        let mut failure: Option<ComputeError> = None;
        let mut k = 0usize;
        pairs.visit_pairs(&mut |i, j, dist_sq, _| {
            let index = k;
            k += 1;
            if failure.is_some() {
                return;
            }
            let Some(d2) = dist_sq else {
                return;
            };
            if d2 == 0.0 {
                return;
            }
            let bin = match self.axis.index(d2.sqrt()) {
                Ok(Some(bin)) => bin,
                Ok(None) => return,
                Err(_) => {
                    failure = Some(ComputeError::NonFinite {
                        where_: "pair dist_sq",
                        index,
                    });
                    return;
                }
            };
            let (i, j) = (i as usize, j as usize);
            // A self-query indexes one set from both ends.
            let (i_limit, j_limit) = if symmetric {
                (n_points, n_points)
            } else {
                (n_query, n_points)
            };
            if i >= i_limit || j >= j_limit {
                failure = Some(ComputeError::OutOfRange {
                    field: "pair index",
                    value: format!("({i}, {j}) of pair {index} past the source's point counts"),
                });
                return;
            }
            sum[bin] += a[i] * b[j];
            counts[bin] += 1;
            if symmetric {
                // A self-query holds each unordered pair once; the reversed
                // pair (j → i) is implicit.
                sum[bin] += a[j] * b[i];
                counts[bin] += 1;
            }
        });
        if let Some(err) = failure {
            return Err(err);
        }

        let correlation = Array1::from_iter((0..n_bins).map(|bin| {
            if counts[bin] > 0 {
                sum[bin] / counts[bin] as F
            } else {
                0.0
            }
        }));
        Ok(CorrelationFunctionResult {
            bin_edges: self.axis.edges(),
            bin_centers: self.axis.centers(),
            bin_counts: counts,
            correlation,
        })
    }
}

/// One snapshot's correlation function.
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct CorrelationFunctionResult {
    /// Bin edges (length `n_bins + 1`).
    pub bin_edges: Array1<F>,
    /// Bin centres (length `n_bins`).
    pub bin_centers: Array1<F>,
    /// Pair count per bin (length `n_bins`).
    pub bin_counts: Array1<u64>,
    /// `⟨A · B⟩(r)` per bin (length `n_bins`); zero for empty bins.
    pub correlation: Array1<F>,
}

crate::compute::spec::compute_spec! {
    CorrelationFunction as "CorrelationFunction";
    params |s| {
        n_bins: Int;
        r_max: Float;
        r_min: Float;
    }
    inputs {
        pairs: PairSource;
        values_a: Scalars;
        values_b: Scalars;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use crate::compute::fixtures::self_neighbors;
    use crate::core::Cell;
    use molrs::core::{NeighborColumns, Neighbors, SimBox};
    use ndarray::{Array2, array};

    fn table(positions: &[[F; 3]], box_len: F, cutoff: F) -> Neighbors {
        let points = Array2::from_shape_fn((positions.len(), 3), |(i, c)| positions[i][c]);
        let simbox = SimBox::new(
            Cell::Lengths([box_len; 3]),
            [0.0 as F, 0.0 as F, 0.0 as F],
            [false; 3],
        )
        .unwrap();
        self_neighbors(points.view(), &simbox, cutoff)
    }

    fn correlate(
        cf: &CorrelationFunction,
        nl: &Neighbors,
        vals: &Array1<F>,
    ) -> Result<CorrelationFunctionResult, ComputeError> {
        cf.compute(CorrelationFunctionInput {
            pairs: nl,
            values_a: vals.view(),
            values_b: vals.view(),
        })
    }

    #[test]
    fn constant_values_yield_constant_correlation() {
        // A = B = 1.0 for all particles → ⟨A·B⟩(r) = 1 in every populated bin.
        let positions = [
            [1.0_f64, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let nl = table(&positions, 20.0, 5.0);
        let vals = Array1::from_elem(positions.len(), 1.0_f64);
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let r = correlate(&cf, &nl, &vals).unwrap();
        for b in 0..10 {
            if r.bin_counts[b] > 0 {
                assert!(
                    (r.correlation[b] - 1.0).abs() < 1e-12,
                    "bin {b}: got {}",
                    r.correlation[b]
                );
            }
        }
    }

    #[test]
    fn alternating_values_average_to_minus_one() {
        // A = B = [+1, -1, +1, -1]. Each NN pair is opposite-sign so
        // A_i · B_j = -1; bin 0 (the only populated one for unit spacing)
        // should average to -1.
        let positions = [
            [0.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ];
        let nl = table(&positions, 20.0, 1.2);
        let vals = array![1.0_f64, -1.0, 1.0, -1.0];
        let cf = CorrelationFunction::new(5, 1.5, 0.0).unwrap();
        let r = correlate(&cf, &nl, &vals).unwrap();
        // The first non-empty bin contains the unit-spacing NN pairs.
        let first = r.bin_counts.iter().position(|&c| c > 0).unwrap();
        assert!((r.correlation[first] + 1.0).abs() < 1e-12);
    }

    #[test]
    fn empty_bin_returns_zero() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0, 1.5);
        let vals = array![1.0_f64, 1.0];
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let r = correlate(&cf, &nl, &vals).unwrap();
        // Bins beyond 1.5 should all be empty.
        for b in 4..10 {
            assert_eq!(r.bin_counts[b], 0);
            assert_eq!(r.correlation[b], 0.0);
        }
    }

    #[test]
    fn invalid_params_error() {
        assert!(CorrelationFunction::new(0, 1.0, 0.0).is_err());
        assert!(CorrelationFunction::new(10, 1.0, 1.0).is_err());
        assert!(CorrelationFunction::new(10, 0.5, 1.0).is_err());
    }

    /// A field with one value per point is required — a short field used to
    /// fail only when a pair happened to index past it.
    #[test]
    fn field_length_must_match_the_points() {
        let nl = table(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            20.0,
            1.5,
        );
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        assert!(matches!(
            correlate(&cf, &nl, &array![1.0, 1.0]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        let lean = nl.repack(NeighborColumns::INDICES_ONLY);
        assert!(matches!(
            correlate(&cf, &lean, &array![1.0, 1.0, 1.0]),
            Err(ComputeError::BadShape { .. })
        ));
    }

    /// Many snapshots go through `Batch`, in order.
    #[test]
    fn batch_matches_single_snapshots() {
        let positions = [
            [1.0_f64, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
        ];
        let nl = table(&positions, 20.0, 5.0);
        let vals = Array1::from_elem(positions.len(), 1.0_f64);
        let cf = CorrelationFunction::new(10, 5.0, 0.0).unwrap();
        let solo = correlate(&cf, &nl, &vals).unwrap();
        let input = CorrelationFunctionInput {
            pairs: &nl,
            values_a: vals.view(),
            values_b: vals.view(),
        };
        let many = Batch(cf).compute(vec![input, input]).unwrap();
        assert_eq!(many, vec![solo.clone(), solo]);
    }
}
