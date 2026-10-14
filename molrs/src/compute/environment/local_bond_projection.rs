//! Projection of neighbor bond vectors onto a set of reference directions.

use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::{Array2, ArrayView2};

use crate::compute::{
    Compute, ComputeError, require_directions, require_disp, require_quaternions,
};
use crate::op::rotate_by_quat;

/// `LocalBondProjection` analyzer.
///
/// Mirrors `freud.environment.LocalBondProjection`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/LocalBondProjection.cc)).
///
/// No configuration. Data: one snapshot's neighbor table with the `disp`
/// column, the reference directions, and optionally the query particles'
/// orientations. For each neighbor pair `(i, j)` and each reference direction
/// `ê_k`, compute `dot_k = r̂_ij · ê_k` where `r̂_ij = (r_j − r_i) / |r_j −
/// r_i|`. The result is a `(n_pairs × n_proj_vectors)` table of cosines. With
/// orientations, each reference direction is first rotated by the query
/// particle's quaternion — the reference directions are then body-frame axes.
/// A zero-length bond has no direction and gets an all-zero row.
#[derive(Debug, Clone, Copy, Default)]
pub struct LocalBondProjection;

/// The data a [`LocalBondProjection`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct LocalBondProjectionInput<'a> {
    /// One snapshot's neighbor table with the `disp` column.
    pub neighbors: &'a Neighbors,
    /// Reference directions, `n_proj × 3`, each normalised here.
    pub proj_vectors: ArrayView2<'a, F>,
    /// Query-particle orientations, quaternions `(w, x, y, z)`,
    /// `n_query_points × 4`; `None` projects in the lab frame.
    pub orientations: Option<ArrayView2<'a, F>>,
}

impl LocalBondProjection {
    /// No parameters — every datum is `compute` input.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for LocalBondProjection {
    type Input<'a> = LocalBondProjectionInput<'a>;
    type Output = LocalBondProjectionResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] for no reference direction;
    /// [`ComputeError::BadShape`] for a table without `disp`;
    /// [`ComputeError::DimensionMismatch`] for a reference or orientation
    /// array of the wrong shape, or orientations not one per query point;
    /// [`ComputeError::OutOfRange`] for a zero reference direction or a zero
    /// quaternion (which used to read as a zero direction / no rotation).
    fn compute(
        &self,
        input: LocalBondProjectionInput<'_>,
    ) -> Result<LocalBondProjectionResult, ComputeError> {
        let refs = require_directions(input.proj_vectors, "proj_vectors")?;
        if refs.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let nlist = input.neighbors;
        let orientations = match input.orientations {
            Some(q) => {
                let q = require_quaternions(q, "query-point quaternions")?;
                if q.len() != nlist.n_query_points() {
                    return Err(ComputeError::DimensionMismatch {
                        expected: nlist.n_query_points(),
                        got: q.len(),
                        what: "orientations (one per query point)",
                    });
                }
                Some(q)
            }
            None => None,
        };
        // Bond directions are what gets projected — no `disp`, no answer.
        let disp = require_disp(nlist)?;
        let i_idx = nlist.query_point_indices();
        let mut p = Array2::<F>::zeros((nlist.n_pairs(), refs.len()));
        for (pk, (row, mut out)) in disp.outer_iter().zip(p.outer_iter_mut()).enumerate() {
            let (dx, dy, dz) = (row[0], row[1], row[2]);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r == 0.0 {
                continue;
            }
            let bond = [dx / r, dy / r, dz / r];
            let rotation = orientations.as_ref().map(|q| q[i_idx[pk] as usize]);
            for (slot, ref_v) in out.iter_mut().zip(&refs) {
                let dir = match rotation {
                    None => *ref_v,
                    Some(q) => rotate_by_quat(q, *ref_v),
                };
                *slot = bond[0] * dir[0] + bond[1] * dir[1] + bond[2] * dir[2];
            }
        }
        Ok(LocalBondProjectionResult { projections: p })
    }
}

/// One snapshot's projections.
#[derive(Debug, Clone, Default)]
pub struct LocalBondProjectionResult {
    /// `(n_pairs, n_proj_vectors)` cosines, pairs in table order.
    pub projections: Array2<F>,
}

crate::compute::spec::compute_spec! {
    LocalBondProjection as "LocalBondProjection";
    params |s| {}
    inputs {
        neighbors: Neighbors;
        proj_vectors: Vectors;
        orientations: Quaternions, optional;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::SimBox;
    use ndarray::array;

    const TOL: F = 1e-12;

    /// A self-query table over `positions` in an open cube of side `box_len`.
    fn table(positions: &[[F; 3]], box_len: F, cutoff: F) -> Neighbors {
        let simbox =
            SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap();
        neighbors_of(positions, &simbox, cutoff)
    }

    fn project(
        nl: &Neighbors,
        proj: ArrayView2<'_, F>,
        orientations: Option<ArrayView2<'_, F>>,
    ) -> Result<LocalBondProjectionResult, ComputeError> {
        LocalBondProjection::new().compute(LocalBondProjectionInput {
            neighbors: nl,
            proj_vectors: proj,
            orientations,
        })
    }

    #[test]
    fn bond_along_x_projects_to_one_on_x_axis() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.5);
        let proj = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let r = &project(&nl, proj.view(), None).unwrap();
        assert_eq!(r.projections.dim(), (1, 2));
        // Bond r̂ = (1, 0, 0). Project on +x → 1, on +y → 0.
        assert!((r.projections[[0, 0]] - 1.0).abs() < TOL);
        assert!((r.projections[[0, 1]]).abs() < TOL);
    }

    #[test]
    fn rotation_changes_projection() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.5);
        // 90° rotation about z: q = (cos(45°), 0, 0, sin(45°))
        let q = [
            std::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_4.sin(),
        ];
        // Without rotation: bond=(1,0,0) projected on +x → 1.
        // With rotation: +x ref becomes +y in lab frame → dot with (1,0,0) → 0.
        let orientations = array![[q[0], q[1], q[2], q[3]], [q[0], q[1], q[2], q[3]]];
        let proj = array![[1.0, 0.0, 0.0]];
        let r = &project(&nl, proj.view(), Some(orientations.view())).unwrap();
        assert!(r.projections[[0, 0]].abs() < 1e-12);
    }

    #[test]
    fn empty_proj_vectors_error() {
        let nl = table(&[[0.0, 0.0, 0.0]], 10.0, 1.0);
        let proj = Array2::<F>::zeros((0, 3));
        let err = project(&nl, proj.view(), None).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    /// A zero reference direction used to project to 0 silently; a zero
    /// quaternion is no rotation. Both are errors now.
    #[test]
    fn zero_reference_or_quaternion_is_out_of_range() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.5);
        let zero_ref = array![[0.0, 0.0, 0.0]];
        assert!(matches!(
            project(&nl, zero_ref.view(), None),
            Err(ComputeError::OutOfRange { .. })
        ));
        let proj = array![[1.0, 0.0, 0.0]];
        let zero_q = array![[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]];
        assert!(matches!(
            project(&nl, proj.view(), Some(zero_q.view())),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn orientations_must_cover_every_query_point() {
        let nl = table(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 10.0, 1.5);
        let proj = array![[1.0, 0.0, 0.0]];
        let one = array![[1.0, 0.0, 0.0, 0.0]];
        assert!(matches!(
            project(&nl, proj.view(), Some(one.view())),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
