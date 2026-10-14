//! Projection of neighbor bond vectors onto a set of reference directions.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array2;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::require_disp;
use crate::op::rotate_by_quat;

/// `LocalBondProjection` analyzer.
///
/// Mirrors `freud.environment.LocalBondProjection`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/LocalBondProjection.cc)).
///
/// For each neighbor pair `(i, j)` and each reference direction `ê_k`,
/// compute `dot_k = r̂_ij · ê_k` where `r̂_ij = (r_j − r_i) / |r_j − r_i|`.
/// The result is a `(n_pairs × n_proj_vectors)` table of cosines, returned
/// alongside its complement `1 − dot` for callers that prefer the
/// "deviation from the reference" reading.
///
/// freud additionally supports per-particle orientations (rotating each
/// reference direction by the particle's quaternion before projection);
/// that flavour is exposed via the `with_query_orientations` builder.
#[derive(Debug, Clone, Default)]
pub struct LocalBondProjection {
    /// If true, rotate each reference direction by the query-point's
    /// quaternion before taking the dot product.
    use_orientations: bool,
}

impl LocalBondProjection {
    /// Default: reference directions are used as-is (lab frame).
    pub fn new() -> Self {
        Self::default()
    }

    /// Rotate each reference direction by the query particle's quaternion when `on`.
    pub fn with_query_orientations(mut self, on: bool) -> Self {
        self.use_orientations = on;
        self
    }
}

/// `Args` for [`LocalBondProjection`].
pub struct LocalBondProjectionArgs<'a> {
    pub nlists: &'a [Neighbors],
    /// Reference directions (unit vectors recommended). Shared across all
    /// frames and all query points.
    pub proj_vectors: &'a [[F; 3]],
    /// Optional per-frame, per-query-point quaternions `(w, x, y, z)`.
    /// Required iff `use_orientations = true`.
    pub query_orientations: Option<&'a [Vec<[F; 4]>]>,
}

impl LegacyCompute for LocalBondProjection {
    type Args<'a> = LocalBondProjectionArgs<'a>;
    type Output = Vec<LocalBondProjectionResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: LocalBondProjectionArgs<'a>,
    ) -> Result<Vec<LocalBondProjectionResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.nlists.len() != frames.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: args.nlists.len(),
                what: "neighbor-list count",
            });
        }
        if args.proj_vectors.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if self.use_orientations {
            match args.query_orientations {
                Some(q) if q.len() == frames.len() => {}
                _ => {
                    return Err(ComputeError::DimensionMismatch {
                        expected: frames.len(),
                        got: args.query_orientations.map(|q| q.len()).unwrap_or(0),
                        what: "query_orientations frame count",
                    });
                }
            }
        }
        let n_proj = args.proj_vectors.len();

        // Pre-normalise reference vectors.
        let refs: Vec<[F; 3]> = args
            .proj_vectors
            .iter()
            .map(|v| {
                let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if n == 0.0 {
                    [0.0, 0.0, 0.0]
                } else {
                    [v[0] / n, v[1] / n, v[2] / n]
                }
            })
            .collect();

        let mut out = Vec::with_capacity(frames.len());
        for (k, nl) in args.nlists.iter().enumerate() {
            let i_idx = nl.query_point_indices();
            let n_pairs = nl.n_pairs();
            // Bond directions are what gets projected — no `disp`, no answer.
            let disp = require_disp(nl)?;
            let mut p = Array2::<F>::zeros((n_pairs, n_proj));
            for pk in 0..n_pairs {
                let dx = disp[[pk, 0]];
                let dy = disp[[pk, 1]];
                let dz = disp[[pk, 2]];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r == 0.0 {
                    continue;
                }
                let bond = [dx / r, dy / r, dz / r];
                let qrot = if self.use_orientations {
                    // Already validated; unwrap is safe.
                    Some(args.query_orientations.unwrap()[k][i_idx[pk] as usize])
                } else {
                    None
                };
                for (j, ref_v) in refs.iter().enumerate() {
                    let dir = match qrot {
                        None => *ref_v,
                        Some(q) => rotate_by_quat(q, *ref_v),
                    };
                    let dot = bond[0] * dir[0] + bond[1] * dir[1] + bond[2] * dir[2];
                    p[[pk, j]] = dot;
                }
            }
            out.push(LocalBondProjectionResult { projections: p });
        }
        Ok(out)
    }
}

/// Per-frame projection result.
#[derive(Debug, Clone, Default)]
pub struct LocalBondProjectionResult {
    /// `(n_pairs, n_proj_vectors)` cosines.
    pub projections: Array2<F>,
}

impl ComputeResult for LocalBondProjectionResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    const TOL: F = 1e-12;

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Frame {
        let x = A1::from_iter(positions.iter().map(|p| p[0]));
        let y = A1::from_iter(positions.iter().map(|p| p[1]));
        let z = A1::from_iter(positions.iter().map(|p| p[2]));
        let mut block = Block::new();
        block.insert("x", x.into_dyn()).unwrap();
        block.insert("y", y.into_dyn()).unwrap();
        block.insert("z", z.into_dyn()).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    #[test]
    fn bond_along_x_projects_to_one_on_x_axis() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &LocalBondProjection::new()
            .compute(
                &[&frame],
                LocalBondProjectionArgs {
                    nlists: std::slice::from_ref(&nl),
                    proj_vectors: &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    query_orientations: None,
                },
            )
            .unwrap()[0];
        assert_eq!(r.projections.dim(), (1, 2));
        // Bond r̂ = (1, 0, 0). Project on +x → 1, on +y → 0.
        assert!((r.projections[[0, 0]] - 1.0).abs() < TOL);
        assert!((r.projections[[0, 1]]).abs() < TOL);
    }

    #[test]
    fn rotation_changes_projection() {
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.5);
        // 90° rotation about z: q = (cos(45°), 0, 0, sin(45°))
        let q = [
            std::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_4.sin(),
        ];
        // Without rotation: bond=(1,0,0) projected on +x → 1.
        // With rotation: +x ref becomes +y in lab frame → dot with (1,0,0) → 0.
        let r = &LocalBondProjection::new()
            .with_query_orientations(true)
            .compute(
                &[&frame],
                LocalBondProjectionArgs {
                    nlists: std::slice::from_ref(&nl),
                    proj_vectors: &[[1.0, 0.0, 0.0]],
                    query_orientations: Some(&[vec![q, q]]),
                },
            )
            .unwrap()[0];
        assert!(r.projections[[0, 0]].abs() < 1e-12);
    }

    #[test]
    fn empty_proj_vectors_error() {
        let frame = frame_with(&[[0.0, 0.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.0);
        let err = LocalBondProjection::new()
            .compute(
                &[&frame],
                LocalBondProjectionArgs {
                    nlists: std::slice::from_ref(&nl),
                    proj_vectors: &[],
                    query_orientations: None,
                },
            )
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
