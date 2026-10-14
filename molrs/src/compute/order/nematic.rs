//! Nematic order parameter and director from particle orientations.

use crate::op::eigh_sym_3x3;

use molrs::op::F;
use ndarray::ArrayView2;

use crate::compute::{Compute, ComputeError, require_directions};

/// Nematic order parameter calculator.
///
/// Mirrors `freud.order.Nematic`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/Nematic.cc)).
///
/// Given a set of unit vectors `{û_i}` (one per particle), build the
/// traceless symmetric `Q` tensor
///
/// ```text
///   Q_ab = (1/N) Σ_i (3/2) û_{i,a} û_{i,b} − (1/2) δ_ab
/// ```
///
/// The largest eigenvalue `S` of `Q` is the **nematic order parameter** and
/// the corresponding eigenvector is the **director** of the system.
/// `S → 0` corresponds to an isotropic ensemble; `S → 1` is perfect
/// alignment.
///
/// No configuration. Data: one snapshot's directors, `N × 3`, each
/// normalised here — different applications choose different conventions
/// (atom orientations, molecular long axes, `head − tail` bond axes), so the
/// caller builds them. A zero director has no direction and is an error.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nematic;

/// The data a [`Nematic`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct NematicInput<'a> {
    /// One snapshot's directors, `N × 3` (any non-zero length).
    pub directors: ArrayView2<'a, F>,
}

impl Nematic {
    /// No parameters — the directors are `compute` data.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for Nematic {
    type Input<'a> = NematicInput<'a>;
    type Output = NematicResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] for no directors;
    /// [`ComputeError::DimensionMismatch`] when they are not `N × 3`;
    /// [`ComputeError::NonFinite`] / [`ComputeError::OutOfRange`] for a
    /// non-finite or zero director.
    fn compute(&self, input: NematicInput<'_>) -> Result<NematicResult, ComputeError> {
        let units = require_directions(input.directors, "directors")?;
        if units.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // Build Q = (3/2) <û û> − (1/2) I.
        let mut q = [[0.0_f64; 3]; 3];
        for u in &units {
            for (a, ua) in u.iter().enumerate() {
                for (b, ub) in u.iter().enumerate() {
                    q[a][b] += ua * ub;
                }
            }
        }
        let inv_n = 1.0 / units.len() as F;
        for (a, row) in q.iter_mut().enumerate() {
            for (b, v) in row.iter_mut().enumerate() {
                *v = 1.5 * *v * inv_n - if a == b { 0.5 } else { 0.0 };
            }
        }

        // Eigenvalues descending; the director is the column of the largest.
        let (vals, vecs) = eigh_sym_3x3(&q);
        let dir = [vecs[0][0], vecs[1][0], vecs[2][0]];

        Ok(NematicResult {
            order: vals[0],
            eigenvalues: [vals[0], vals[1], vals[2]],
            director: dir,
            q_tensor: q,
        })
    }
}

/// One snapshot's nematic order parameter.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct NematicResult {
    /// Largest eigenvalue of `Q` (scalar order parameter).
    pub order: F,
    /// All three eigenvalues, descending.
    pub eigenvalues: [F; 3],
    /// Director (eigenvector for the largest eigenvalue), unit vector.
    pub director: [F; 3],
    /// Full traceless symmetric `Q` tensor, row-major 3×3.
    pub q_tensor: [[F; 3]; 3],
}

crate::compute::spec::compute_spec! {
    Nematic as "Nematic";
    params |s| {}
    inputs {
        directors: Vectors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    const TOL: F = 1e-10;

    fn nematic(dirs: &[[F; 3]]) -> Result<NematicResult, ComputeError> {
        let directors = Array2::from_shape_fn((dirs.len(), 3), |(i, c)| dirs[i][c]);
        Nematic::new().compute(NematicInput {
            directors: directors.view(),
        })
    }

    #[test]
    fn perfectly_aligned_gives_unity() {
        // All directors along +z → S = 1, director = (0, 0, 1)
        let dirs = vec![[0.0, 0.0, 1.0]; 100];
        let res = nematic(&dirs).unwrap();
        assert!((res.order - 1.0).abs() < TOL);
        // Director should be along z (or -z; signed eigenvectors are
        // determined only up to a sign).
        assert!(res.director[2].abs() > 1.0 - TOL);
    }

    #[test]
    fn isotropic_three_axis_set_gives_zero() {
        // Equal directors along ±x, ±y, ±z → Q = 0, all eigenvalues 0.
        let dirs = vec![
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let res = nematic(&dirs).unwrap();
        for &e in &res.eigenvalues {
            assert!(e.abs() < TOL, "eigenvalue {e} should be ≈ 0");
        }
    }

    #[test]
    fn antiparallel_directors_equal_parallel() {
        // {+z, +z, -z} should give the same Q as {+z, +z, +z} because
        // û and -û produce identical Q contributions (uu = (-u)(-u)).
        let a = nematic(&[[0.0, 0.0, 1.0]; 3]).unwrap();
        let b = nematic(&[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]).unwrap();
        assert!((a.order - b.order).abs() < TOL);
    }

    #[test]
    fn q_tensor_is_traceless() {
        let dirs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let res = nematic(&dirs).unwrap();
        let tr = res.q_tensor[0][0] + res.q_tensor[1][1] + res.q_tensor[2][2];
        assert!(tr.abs() < TOL, "trace of Q should be 0, got {tr}");
    }

    #[test]
    fn empty_directors_is_error() {
        let err = nematic(&[]).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    /// A zero director used to be skipped silently; it has no direction.
    #[test]
    fn zero_director_is_out_of_range() {
        let err = nematic(&[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn directors_must_have_three_components() {
        let directors = Array2::<F>::zeros((2, 4));
        let err = Nematic::new()
            .compute(NematicInput {
                directors: directors.view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }
}
