//! 2-component Principal Component Analysis (PCA).

use std::marker::PhantomData;

use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::{ComputeResult, DescriptorRow};

/// Result of a 2-component PCA projection.
#[derive(Debug, Clone, Default)]
pub struct PcaResult {
    /// Projected coordinates, row-major of shape `[n_rows, 2]`.
    pub coords: Vec<F>,
    /// Explained variance per component, `variance[0] >= variance[1] >= 0`.
    pub variance: [F; 2],
}

impl ComputeResult for PcaResult {}

/// Stateless PCA calculator with two components, generic over the descriptor
/// row type.
///
/// Construct with `Pca::<T>::new()` where `T: DescriptorRow`. Each
/// [`compute`](LegacyCompute::compute) call expects an `&Vec<T>` of length ≥ 3.
///
/// [`Pca`] is a [`LegacyCompute`](crate::compute::LegacyCompute) that consumes any upstream `Vec<T>` where
/// `T: DescriptorRow` — each descriptor row is one observation. PCA
/// standardizes the columns, computes covariance, and projects onto the top
/// two eigenvectors via power iteration with deflation. No external
/// linear-algebra crate is required; matrices are small (tens of columns).
///
/// # Algorithm
///
/// 1. Collect rows into an `n_rows × n_cols` matrix (row-major).
/// 2. Standardize each column: `(x - mean) / std`. Zero-variance columns are
///    rejected.
/// 3. Covariance `C = (Z^T Z) / (n_rows - 1)`.
/// 4. Top two eigenvectors via power iteration with deflation.
/// 5. Project every row onto the two eigenvectors → `[n_rows, 2]` scores.
///
/// Same input → identical output (deterministic initial vector, deflation).
#[derive(Debug)]
pub struct Pca<T: DescriptorRow + Clone + Send + Sync + 'static> {
    _marker: PhantomData<fn() -> T>,
}

impl<T: DescriptorRow + Clone + Send + Sync + 'static> Pca<T> {
    /// No parameters — fixed 2-component projection.
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
}

impl<T: DescriptorRow + Clone + Send + Sync + 'static> Clone for Pca<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: DescriptorRow + Clone + Send + Sync + 'static> Copy for Pca<T> {}

impl<T: DescriptorRow + Clone + Send + Sync + 'static> Default for Pca<T> {
    fn default() -> Self {
        Self::new()
    }
}

const POWER_ITER_TOL: F = 1e-12;
const POWER_ITER_MAX: usize = 200;
const STD_FLOOR: F = 1e-12;

impl<T: DescriptorRow + Clone + Send + Sync + 'static> LegacyCompute for Pca<T> {
    type Args<'a> = &'a Vec<T>;
    type Output = PcaResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        rows: &'a Vec<T>,
    ) -> Result<PcaResult, ComputeError> {
        if rows.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let first = rows[0].as_row();
        let n_cols = first.len();
        let n_rows = rows.len();
        let mut matrix = Vec::with_capacity(n_rows * n_cols);
        for (i, row) in rows.iter().enumerate() {
            let r = row.as_row();
            if r.len() != n_cols {
                return Err(ComputeError::BadShape {
                    expected: format!("row length {n_cols}"),
                    got: format!("row {i} length {}", r.len()),
                });
            }
            matrix.extend_from_slice(r);
        }
        fit_transform(&matrix, n_rows, n_cols)
    }
}

/// Standardize columns, compute the covariance matrix, and project rows onto
/// the top two eigenvectors.
pub(crate) fn fit_transform(
    matrix: &[F],
    n_rows: usize,
    n_cols: usize,
) -> Result<PcaResult, ComputeError> {
    if n_rows < 3 {
        return Err(ComputeError::OutOfRange {
            field: "PCA::n_rows",
            value: n_rows.to_string(),
        });
    }
    if n_cols < 2 {
        return Err(ComputeError::OutOfRange {
            field: "PCA::n_cols",
            value: n_cols.to_string(),
        });
    }
    if matrix.len() != n_rows * n_cols {
        return Err(ComputeError::DimensionMismatch {
            expected: n_rows * n_cols,
            got: matrix.len(),
            what: "PCA matrix length",
        });
    }
    for (i, &v) in matrix.iter().enumerate() {
        if !v.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "PCA matrix",
                index: i,
            });
        }
    }

    let mut mean = vec![0.0 as F; n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            mean[j] += matrix[i * n_cols + j];
        }
    }
    let inv_n = 1.0 / n_rows as F;
    for m in mean.iter_mut() {
        *m *= inv_n;
    }

    let mut var = vec![0.0 as F; n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            let d = matrix[i * n_cols + j] - mean[j];
            var[j] += d * d;
        }
    }
    let n_minus_1 = (n_rows - 1) as F;
    for v in var.iter_mut() {
        *v /= n_minus_1;
    }
    let mut std = vec![0.0 as F; n_cols];
    for (j, &v) in var.iter().enumerate() {
        let s = v.sqrt();
        if s < STD_FLOOR {
            return Err(ComputeError::OutOfRange {
                field: "PCA::column_std",
                value: format!("column {j} stddev {s:e} below floor {STD_FLOOR:e}"),
            });
        }
        std[j] = s;
    }

    let mut z = vec![0.0 as F; n_rows * n_cols];
    for i in 0..n_rows {
        for j in 0..n_cols {
            z[i * n_cols + j] = (matrix[i * n_cols + j] - mean[j]) / std[j];
        }
    }

    let mut cov = vec![0.0 as F; n_cols * n_cols];
    for a in 0..n_cols {
        for b in a..n_cols {
            let mut s = 0.0 as F;
            for i in 0..n_rows {
                s += z[i * n_cols + a] * z[i * n_cols + b];
            }
            let c = s / n_minus_1;
            cov[a * n_cols + b] = c;
            cov[b * n_cols + a] = c;
        }
    }

    let v1 = power_iteration(&cov, n_cols);
    let lam1 = rayleigh_quotient(&cov, &v1, n_cols);
    let mut cov2 = cov.clone();
    for a in 0..n_cols {
        for b in 0..n_cols {
            cov2[a * n_cols + b] -= lam1 * v1[a] * v1[b];
        }
    }
    let v2 = power_iteration(&cov2, n_cols);
    let lam2 = rayleigh_quotient(&cov, &v2, n_cols);

    let mut coords = vec![0.0 as F; n_rows * 2];
    for i in 0..n_rows {
        let mut pc1 = 0.0 as F;
        let mut pc2 = 0.0 as F;
        for j in 0..n_cols {
            let zij = z[i * n_cols + j];
            pc1 += zij * v1[j];
            pc2 += zij * v2[j];
        }
        coords[2 * i] = pc1;
        coords[2 * i + 1] = pc2;
    }

    let variance = [lam1.max(0.0), lam2.max(0.0)];
    Ok(PcaResult { coords, variance })
}

fn power_iteration(mat: &[F], n: usize) -> Vec<F> {
    let mut v = vec![1.0 / (n as F).sqrt(); n];
    for _ in 0..POWER_ITER_MAX {
        let mut next = mat_vec(mat, &v, n);
        let norm = vec_norm(&next);
        if norm <= 0.0 {
            return v;
        }
        for x in next.iter_mut() {
            *x /= norm;
        }
        let dot: F = v.iter().zip(next.iter()).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            for x in next.iter_mut() {
                *x = -*x;
            }
        }
        let diff: F = v
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<F>()
            .sqrt();
        v = next;
        if diff < POWER_ITER_TOL {
            break;
        }
    }
    v
}

fn rayleigh_quotient(mat: &[F], v: &[F], n: usize) -> F {
    let mv = mat_vec(mat, v, n);
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

fn mat_vec(mat: &[F], v: &[F], n: usize) -> Vec<F> {
    let mut out = vec![0.0 as F; n];
    for a in 0..n {
        let mut s = 0.0 as F;
        for b in 0..n {
            s += mat[a * n + b] * v[b];
        }
        out[a] = s;
    }
    out
}

fn vec_norm(v: &[F]) -> F {
    v.iter().map(|&x| x * x).sum::<F>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::standard_normal;
    use molrs::core::Frame;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Descriptor row wrapping a Vec<F>.
    #[derive(Clone)]
    struct Row(Vec<F>);
    impl DescriptorRow for Row {
        fn as_row(&self) -> &[F] {
            &self.0
        }
    }
    impl ComputeResult for Row {}

    fn three_blobs_rows(n_per_cluster: usize, seed: u64) -> Vec<Row> {
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let sigma: F = 0.5;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::with_capacity(3 * n_per_cluster);
        for (cx, cy) in centers.iter().copied() {
            for _ in 0..n_per_cluster {
                rows.push(Row(vec![
                    cx + sigma * standard_normal(&mut rng),
                    cy + sigma * standard_normal(&mut rng),
                ]));
            }
        }
        rows
    }

    #[test]
    fn fit_transform_on_three_blobs() {
        let rows = three_blobs_rows(20, 42);
        let frame = Frame::new();
        let result = Pca::<Row>::new().compute(&[&frame], &rows).unwrap();
        assert_eq!(result.coords.len(), 2 * rows.len());
        assert!(result.variance[0] > 0.0);
        assert!(result.variance[1] > 0.0);
        assert!(result.variance[0] >= result.variance[1]);
    }

    #[test]
    fn err_on_too_few_rows() {
        let rows = vec![Row(vec![1.0, 2.0]), Row(vec![3.0, 4.0])];
        let frame = Frame::new();
        let err = Pca::<Row>::new().compute(&[&frame], &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_too_few_cols() {
        let rows = vec![Row(vec![1.0]); 5];
        let frame = Frame::new();
        let err = Pca::<Row>::new().compute(&[&frame], &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_nan_input() {
        let mut rows = vec![Row(vec![0.0; 4]); 5];
        rows[1].0[2] = F::NAN;
        let frame = Frame::new();
        let err = Pca::<Row>::new().compute(&[&frame], &rows).unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { .. }));
    }

    #[test]
    fn err_on_zero_variance_column() {
        // column 1 constant across 5 rows
        let rows: Vec<Row> = (0..5).map(|i| Row(vec![i as F, 1.0])).collect();
        let frame = Frame::new();
        let err = Pca::<Row>::new().compute(&[&frame], &rows).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn variance_sum_tracks_trace() {
        let rows = three_blobs_rows(40, 7);
        let frame = Frame::new();
        let result = Pca::<Row>::new().compute(&[&frame], &rows).unwrap();
        let sum = result.variance[0] + result.variance[1];
        assert!(
            (sum - 2.0).abs() < 1e-6,
            "variance sum {sum} should be ~2.0 (trace of standardized cov)"
        );
    }

    #[test]
    fn ragged_rows_error() {
        let rows = vec![Row(vec![1.0, 2.0]), Row(vec![3.0])];
        let frame = Frame::new();
        let err = Pca::<Row>::new().compute(&[&frame], &rows).unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }
}
