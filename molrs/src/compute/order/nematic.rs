//! Nematic order parameter and director from particle orientations.

// 3×3 tensor traceless adjustment reads cleanest with index loops.
#![allow(clippy::needless_range_loop)]

use crate::compute::ComputeResult;
use crate::op::eigh_sym_3x3;

use molrs::core::FrameAccess;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

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
/// This analyzer is stateless and takes the directors as an `Args` slice
/// rather than reading them from a `Frame` block — different applications
/// choose different conventions (atom orientations vs molecular long axes
/// vs bond directions), so the caller passes them in directly.
#[derive(Debug, Clone, Default)]
pub struct Nematic;

impl Nematic {
    /// No parameters — the director comes from the orientations in `Args`.
    pub fn new() -> Self {
        Self
    }

    fn one_frame(&self, directors: &[[F; 3]]) -> Result<NematicResult, ComputeError> {
        if directors.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // Build Q = (3/2) <û û> − (1/2) I, with directors normalised.
        let mut q = [[0.0_f64; 3]; 3];
        let mut n_used = 0_usize;
        for u in directors {
            let r = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
            if r == 0.0 {
                continue;
            }
            let inv = 1.0 / r;
            let ux = u[0] * inv;
            let uy = u[1] * inv;
            let uz = u[2] * inv;
            for (a, ua) in [ux, uy, uz].iter().enumerate() {
                for (b, ub) in [ux, uy, uz].iter().enumerate() {
                    q[a][b] += ua * ub;
                }
            }
            n_used += 1;
        }
        if n_used == 0 {
            return Err(ComputeError::EmptyInput);
        }
        let inv_n = 1.0 / n_used as F;
        for a in 0..3 {
            for b in 0..3 {
                q[a][b] = 1.5 * q[a][b] * inv_n - if a == b { 0.5 } else { 0.0 };
            }
        }

        // Eigenvalues descending; the director is the column of the largest.
        let (vals, vecs) = eigh_sym_3x3(&q);
        let dir = [vecs[0][0], vecs[1][0], vecs[2][0]];

        Ok(NematicResult {
            order: vals[0],
            eigenvalues: [vals[0], vals[1], vals[2]],
            director: [dir[0], dir[1], dir[2]],
            q_tensor: q,
        })
    }
}

impl LegacyCompute for Nematic {
    type Args<'a> = &'a [[F; 3]];
    type Output = Vec<NematicResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        directors: &'a [[F; 3]],
    ) -> Result<Vec<NematicResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // Same director set is broadcast across all frames — typical usage
        // is single-frame, but matching the Compute trait shape.
        let mut out = Vec::with_capacity(frames.len());
        for _ in frames {
            out.push(self.one_frame(directors)?);
        }
        Ok(out)
    }
}

/// Per-frame nematic order parameter.
#[derive(Debug, Clone, Default)]
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

impl ComputeResult for NematicResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;

    const TOL: F = 1e-10;

    fn frame() -> Frame {
        Frame::new()
    }

    #[test]
    fn perfectly_aligned_gives_unity() {
        // All directors along +z → S = 1, director = (0, 0, 1)
        let dirs = vec![[0.0, 0.0, 1.0]; 100];
        let res = Nematic::new().compute(&[&frame()], &dirs).unwrap();
        assert!((res[0].order - 1.0).abs() < TOL);
        // Director should be along z (or -z; signed eigenvectors are
        // determined only up to a sign).
        assert!(res[0].director[2].abs() > 1.0 - TOL);
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
        let res = Nematic::new().compute(&[&frame()], &dirs).unwrap();
        for &e in &res[0].eigenvalues {
            assert!(e.abs() < TOL, "eigenvalue {e} should be ≈ 0");
        }
    }

    #[test]
    fn antiparallel_directors_equal_parallel() {
        // {+z, +z, -z} should give the same Q as {+z, +z, +z} because
        // û and -û produce identical Q contributions (uu = (-u)(-u)).
        let a = Nematic::new()
            .compute(&[&frame()], &[[0.0, 0.0, 1.0]; 3])
            .unwrap();
        let b = Nematic::new()
            .compute(
                &[&frame()],
                &[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]],
            )
            .unwrap();
        assert!((a[0].order - b[0].order).abs() < TOL);
    }

    #[test]
    fn q_tensor_is_traceless() {
        let dirs = vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let res = Nematic::new().compute(&[&frame()], &dirs).unwrap();
        let tr = res[0].q_tensor[0][0] + res[0].q_tensor[1][1] + res[0].q_tensor[2][2];
        assert!(tr.abs() < TOL, "trace of Q should be 0, got {tr}");
    }

    #[test]
    fn empty_directors_is_error() {
        let err = Nematic::new().compute(&[&frame()], &[]).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
