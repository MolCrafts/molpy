//! Principal component analysis of a feature matrix.

use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView1};

use crate::compute::{Compute, ComputeError, FeatureMatrix};
use crate::op::linalg::eigh_sym;

/// Principal component analysis: the projection of every sample onto the
/// `n_components` leading principal axes of the standardised features.
///
/// Configuration: the number of components. Data: any [`FeatureMatrix`] — an
/// `Array2`, or a descriptor result such as
/// [`SteinhardtResult`](crate::compute::SteinhardtResult) or
/// [`LocalDescriptorsResult`](crate::compute::LocalDescriptorsResult) — one
/// row per sample. The result is itself a [`FeatureMatrix`], so it feeds
/// [`Kmeans`](crate::compute::Kmeans) directly.
///
/// # Algorithm
///
/// 1. Standardise each column: `z = (x − mean) / std` (sample standard
///    deviation). A zero-variance column carries no direction and is an
///    error.
/// 2. Covariance `C = Zᵀ Z / (n_rows − 1)` — the correlation matrix.
/// 3. Full symmetric eigendecomposition of `C` (the crate's one cyclic-Jacobi
///    solver); the `n_components` largest eigenpairs are the axes and their
///    explained variances. The eigenvalues are reported as computed, never
///    clamped. Each axis's sign is fixed so its largest-magnitude loading is
///    positive, so the projection does not depend on the solver's sign
///    choice.
/// 4. Project every standardised row onto the axes.
#[derive(Debug, Clone, Copy)]
pub struct Pca {
    n_components: usize,
}

/// The data a [`Pca`] decomposes.
#[derive(Clone, Copy)]
pub struct PcaInput<'a> {
    /// The samples, one row each.
    pub features: &'a (dyn FeatureMatrix + Sync),
}

/// A PCA projection.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaResult {
    /// The scores, `n_rows × n_components`.
    pub coords: Array2<F>,
    /// Explained variance of each component, descending (the covariance
    /// eigenvalues; they sum to `n_cols` over all components).
    pub variance: Array1<F>,
}

impl FeatureMatrix for PcaResult {
    fn n_rows(&self) -> usize {
        self.coords.nrows()
    }

    fn n_cols(&self) -> usize {
        self.coords.ncols()
    }

    fn row(&self, row: usize) -> ArrayView1<'_, F> {
        self.coords.row(row)
    }
}

/// Below this sample standard deviation a column is treated as constant.
const STD_FLOOR: F = 1e-12;

impl Pca {
    /// Keep the `n_components` leading components.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0.
    pub fn new(n_components: usize) -> Result<Self, ComputeError> {
        if n_components == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Pca::n_components",
                value: "0".into(),
            });
        }
        Ok(Self { n_components })
    }

    /// The number of components kept.
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// `(x − mean) / std` per column, the sample standard deviation.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a column whose deviation is below
    /// [`STD_FLOOR`].
    fn standardized(mut x: Array2<F>) -> Result<Array2<F>, ComputeError> {
        let n_minus_1 = (x.nrows() - 1) as F;
        for (j, mut column) in x.columns_mut().into_iter().enumerate() {
            let mean = column.sum() / column.len() as F;
            column.mapv_inplace(|v| v - mean);
            let std = (column.dot(&column) / n_minus_1).sqrt();
            if std < STD_FLOOR {
                return Err(ComputeError::OutOfRange {
                    field: "Pca::column_std",
                    value: format!("column {j} stddev {std:e} below floor {STD_FLOOR:e}"),
                });
            }
            column.mapv_inplace(|v| v / std);
        }
        Ok(x)
    }
}

impl Compute for Pca {
    type Input<'a> = PcaInput<'a>;
    type Output = PcaResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] for no rows; [`ComputeError::OutOfRange`]
    /// for fewer than 3 rows, fewer columns than components, or a
    /// zero-variance column; [`ComputeError::NonFinite`] for a non-finite
    /// feature.
    fn compute(&self, input: PcaInput<'_>) -> Result<PcaResult, ComputeError> {
        let features = input.features;
        let (n_rows, n_cols) = (features.n_rows(), features.n_cols());
        if n_rows == 0 {
            return Err(ComputeError::EmptyInput);
        }
        if n_rows < 3 {
            return Err(ComputeError::OutOfRange {
                field: "Pca::n_rows",
                value: format!("{n_rows} (at least 3 samples)"),
            });
        }
        if n_cols < self.n_components {
            return Err(ComputeError::OutOfRange {
                field: "Pca::n_cols",
                value: format!("{n_cols} features for {} components", self.n_components),
            });
        }
        let x = Array2::from_shape_fn((n_rows, n_cols), |(i, j)| features.row(i)[j]);
        if let Some(index) = x.iter().position(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "Pca features",
                index,
            });
        }

        let z = Self::standardized(x)?;
        let n_minus_1 = (n_rows - 1) as F;
        let mut cov = vec![0.0; n_cols * n_cols];
        for a in 0..n_cols {
            for b in a..n_cols {
                let c = z.column(a).dot(&z.column(b)) / n_minus_1;
                cov[a * n_cols + b] = c;
                cov[b * n_cols + a] = c;
            }
        }

        let (values, vectors) = eigh_sym(&cov, n_cols);
        let k = self.n_components;
        let mut axes = Array2::from_shape_fn((n_cols, k), |(r, c)| vectors[r * n_cols + c]);
        for mut axis in axes.columns_mut() {
            let largest = axis
                .iter()
                .copied()
                .fold(0.0, |m: F, v| if v.abs() > m.abs() { v } else { m });
            if largest < 0.0 {
                axis.mapv_inplace(|v| -v);
            }
        }

        Ok(PcaResult {
            coords: z.dot(&axes),
            variance: Array1::from_iter(values.into_iter().take(k)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::standard_normal;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn three_blobs_rows(n_per_cluster: usize, seed: u64) -> Array2<F> {
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let sigma: F = 0.5;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::with_capacity(6 * n_per_cluster);
        for (cx, cy) in centers.iter().copied() {
            for _ in 0..n_per_cluster {
                rows.push(cx + sigma * standard_normal(&mut rng));
                rows.push(cy + sigma * standard_normal(&mut rng));
            }
        }
        Array2::from_shape_vec((3 * n_per_cluster, 2), rows).unwrap()
    }

    fn pca(n_components: usize, rows: &Array2<F>) -> Result<PcaResult, ComputeError> {
        Pca::new(n_components)?.compute(PcaInput { features: rows })
    }

    #[test]
    fn fit_transform_on_three_blobs() {
        let rows = three_blobs_rows(20, 42);
        let result = pca(2, &rows).unwrap();
        assert_eq!(result.coords.dim(), (rows.nrows(), 2));
        assert!(result.variance[0] > 0.0);
        assert!(result.variance[1] > 0.0);
        assert!(result.variance[0] >= result.variance[1]);
    }

    #[test]
    fn err_on_too_few_rows() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let err = pca(2, &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_too_few_cols() {
        let rows = Array2::<F>::ones((5, 1));
        let err = pca(2, &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_nan_input() {
        let mut rows = Array2::<F>::zeros((5, 4));
        rows[[1, 2]] = F::NAN;
        let err = pca(2, &rows).unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { .. }));
    }

    #[test]
    fn err_on_zero_variance_column() {
        // column 1 constant across 5 rows
        let rows = Array2::from_shape_fn((5, 2), |(i, j)| if j == 0 { i as F } else { 1.0 });
        let err = pca(2, &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn variance_sum_tracks_trace() {
        let rows = three_blobs_rows(40, 7);
        let result = pca(2, &rows).unwrap();
        let sum = result.variance[0] + result.variance[1];
        assert!(
            (sum - 2.0).abs() < 1e-6,
            "variance sum {sum} should be ~2.0 (trace of standardized cov)"
        );
    }

    #[test]
    fn zero_components_is_out_of_range() {
        assert!(matches!(
            Pca::new(0),
            Err(ComputeError::OutOfRange {
                field: "Pca::n_components",
                ..
            })
        ));
    }

    #[test]
    fn no_rows_is_empty_input() {
        assert!(matches!(
            pca(1, &Array2::<F>::zeros((0, 2))),
            Err(ComputeError::EmptyInput)
        ));
    }

    /// Two perfectly correlated columns: the correlation matrix is
    /// `[[1, 1], [1, 1]]`, eigenvalues 2 and 0, the first axis `(1, 1)/√2`.
    /// The second eigenvalue is reported as the solver returns it (spec §5:
    /// no `max(0.0)` clamp) — zero to round-off, of either sign.
    #[test]
    fn a_rank_one_matrix_has_one_axis() {
        let rows = array![[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]];
        let result = pca(2, &rows).unwrap();
        assert!((result.variance[0] - 2.0).abs() < 1e-12);
        assert!(result.variance[1].abs() < 1e-12);
        // Scores on axis 1 are (z₀ + z₁)/√2 = √2 z₀, z₀ ∝ (−1.5, −0.5, 0.5, 1.5).
        let std = (5.0_f64 / 3.0).sqrt();
        for (i, &x0) in [-1.5, -0.5, 0.5, 1.5].iter().enumerate() {
            let expected = std::f64::consts::SQRT_2 * x0 / std;
            assert!((result.coords[[i, 0]] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn the_projection_is_a_feature_matrix() {
        let rows = three_blobs_rows(5, 1);
        let result = pca(1, &rows).unwrap();
        let f: &dyn FeatureMatrix = &result;
        assert_eq!((f.n_rows(), f.n_cols()), (15, 1));
        assert_eq!(f.row(3)[0], result.coords[[3, 0]]);
    }
}
