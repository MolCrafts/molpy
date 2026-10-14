//! Pairwise angular separation between unit quaternions.

use crate::compute::ComputeResult;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::Array2;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::op::Quat;
use crate::op::{quat_dot, quat_norm};

/// Angular distance between two unit quaternions, in radians.
///
/// `θ = 2 · arccos(|q₁ · q₂|)` when `equivalent_orientations = true`
/// (the default), giving values in `[0, π/2]`. When `false`, the absolute
/// value is dropped and the result lies in `[0, π]`.
///
/// Mirrors `freud.environment.AngularSeparationGlobal` and
/// `AngularSeparationNeighbor`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/AngularSeparation.cc)).
///
/// For two unit quaternions `q₁` and `q₂` the (rotational) angular
/// distance is
///
/// ```text
///   θ = 2 · arccos( |q₁ · q₂| )
/// ```
///
/// where the absolute value accounts for the double cover (`q` and `−q`
/// represent the same rotation). The result is in radians, `0 ≤ θ ≤ π/2`
/// (between rotations), or `0 ≤ θ ≤ π` if the user does *not* want the
/// double-cover identification (`equivalent_orientations = false` in
/// freud).
///
/// Two flavours are provided:
///
/// - [`AngularSeparationGlobal`]: dense `(N_query × N_global)` table of
///   angular distances between every query orientation and every reference
///   orientation.
/// - [`AngularSeparationNeighbor`]: sparse, one angular distance per
///   neighbor pair, driven by a `Neighbors`.
pub fn angular_distance(q1: Quat, q2: Quat, equivalent_orientations: bool) -> F {
    let n1 = quat_norm(q1);
    let n2 = quat_norm(q2);
    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }
    let mut d = quat_dot(q1, q2) / (n1 * n2);
    if equivalent_orientations {
        d = d.abs();
    }
    2.0 * d.clamp(-1.0, 1.0).acos()
}

// ---------------------------------------------------------------------------
// Global: dense (N_query × N_global) table
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AngularSeparationGlobal {
    equivalent_orientations: bool,
}

impl Default for AngularSeparationGlobal {
    fn default() -> Self {
        Self::new()
    }
}

impl AngularSeparationGlobal {
    /// Defaults: symmetry-equivalent orientations enabled.
    pub fn new() -> Self {
        Self {
            equivalent_orientations: true,
        }
    }

    /// Also minimize over symmetry-equivalent orientations when `on` (default true).
    pub fn with_equivalent_orientations(mut self, on: bool) -> Self {
        self.equivalent_orientations = on;
        self
    }
}

/// Args for `AngularSeparationGlobal`: (per-particle query quaternions,
/// global reference quaternions). The query set typically comes from the
/// frame's atom orientations; the reference set is a small fixed list of
/// canonical orientations (e.g. crystallographic point-group symmetries).
pub struct AngularSeparationGlobalArgs<'a> {
    pub query: &'a [Quat],
    pub global: &'a [Quat],
}

impl LegacyCompute for AngularSeparationGlobal {
    type Args<'a> = AngularSeparationGlobalArgs<'a>;
    type Output = Vec<AngularSeparationGlobalResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: AngularSeparationGlobalArgs<'a>,
    ) -> Result<Vec<AngularSeparationGlobalResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.query.is_empty() || args.global.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let n_query = args.query.len();
        let n_global = args.global.len();
        let mut out = Vec::with_capacity(frames.len());
        for _ in frames {
            // O(n_query × n_global) by definition: the observable *is* the
            // full matrix of pairwise angular distances, every entry of which
            // is returned.
            let mut a = Array2::<F>::zeros((n_query, n_global));
            for i in 0..n_query {
                for j in 0..n_global {
                    a[[i, j]] = angular_distance(
                        args.query[i],
                        args.global[j],
                        self.equivalent_orientations,
                    );
                }
            }
            out.push(AngularSeparationGlobalResult { angles: a });
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Neighbor: sparse, one angle per pair
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AngularSeparationNeighbor {
    equivalent_orientations: bool,
}

impl Default for AngularSeparationNeighbor {
    fn default() -> Self {
        Self::new()
    }
}

impl AngularSeparationNeighbor {
    /// Defaults: symmetry-equivalent orientations enabled.
    pub fn new() -> Self {
        Self {
            equivalent_orientations: true,
        }
    }

    /// Also minimize over symmetry-equivalent orientations when `on` (default true).
    pub fn with_equivalent_orientations(mut self, on: bool) -> Self {
        self.equivalent_orientations = on;
        self
    }
}

pub struct AngularSeparationNeighborArgs<'a> {
    pub nlists: &'a [Neighbors],
    /// Per-frame query orientations (indexed by `query_point_indices`).
    pub query_orientations: &'a [Vec<Quat>],
    /// Per-frame point orientations (indexed by `point_indices`).
    pub point_orientations: &'a [Vec<Quat>],
}

impl LegacyCompute for AngularSeparationNeighbor {
    type Args<'a> = AngularSeparationNeighborArgs<'a>;
    type Output = Vec<AngularSeparationNeighborResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: AngularSeparationNeighborArgs<'a>,
    ) -> Result<Vec<AngularSeparationNeighborResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        let nf = frames.len();
        if args.nlists.len() != nf
            || args.query_orientations.len() != nf
            || args.point_orientations.len() != nf
        {
            return Err(ComputeError::DimensionMismatch {
                expected: nf,
                got: args
                    .nlists
                    .len()
                    .min(args.query_orientations.len())
                    .min(args.point_orientations.len()),
                what: "AngularSeparationNeighbor frame-aligned inputs",
            });
        }
        let mut out = Vec::with_capacity(nf);
        for k in 0..nf {
            let nl = &args.nlists[k];
            let q = &args.query_orientations[k];
            let p = &args.point_orientations[k];
            let i_idx = nl.query_point_indices();
            let j_idx = nl.point_indices();
            let mut angles = Vec::with_capacity(nl.n_pairs());
            for pk in 0..nl.n_pairs() {
                let i = i_idx[pk] as usize;
                let j = j_idx[pk] as usize;
                if i >= q.len() || j >= p.len() {
                    return Err(ComputeError::DimensionMismatch {
                        expected: i.max(j) + 1,
                        got: q.len().min(p.len()),
                        what: "AngularSeparationNeighbor orientations length",
                    });
                }
                angles.push(angular_distance(q[i], p[j], self.equivalent_orientations));
            }
            out.push(AngularSeparationNeighborResult { angles });
        }
        Ok(out)
    }
}

#[derive(Debug, Clone, Default)]
pub struct AngularSeparationGlobalResult {
    /// `(n_query, n_global)` table of angular distances (radians).
    pub angles: Array2<F>,
}

impl ComputeResult for AngularSeparationGlobalResult {}

#[derive(Debug, Clone, Default)]
pub struct AngularSeparationNeighborResult {
    /// One angle per neighbor pair, in radians; index matches the
    /// underlying [`Neighbors`] pair index.
    pub angles: Vec<F>,
}

impl ComputeResult for AngularSeparationNeighborResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Frame;

    const TOL: F = 1e-10;

    fn frame() -> Frame {
        Frame::new()
    }

    #[test]
    fn identity_quaternion_zero_distance() {
        let q = [1.0_f64, 0.0, 0.0, 0.0];
        assert!(angular_distance(q, q, true).abs() < TOL);
        assert!(angular_distance(q, q, false).abs() < TOL);
    }

    #[test]
    fn antipodal_quaternions_zero_with_equivalence() {
        let q = [1.0_f64, 0.0, 0.0, 0.0];
        let neg_q = [-1.0_f64, 0.0, 0.0, 0.0];
        // With equivalent_orientations = true, q and -q describe the same
        // rotation → angular distance 0.
        assert!(angular_distance(q, neg_q, true).abs() < TOL);
        // Without equivalence the bare formula gives 2·acos(-1) = 2π.
        let theta = angular_distance(q, neg_q, false);
        assert!((theta - 2.0 * std::f64::consts::PI).abs() < TOL);
    }

    #[test]
    fn ninety_degree_rotation_about_z() {
        // Quaternion for 90° rotation about z: (cos(45°), 0, 0, sin(45°))
        let q = [
            std::f64::consts::FRAC_PI_4.cos(),
            0.0,
            0.0,
            std::f64::consts::FRAC_PI_4.sin(),
        ];
        let identity = [1.0_f64, 0.0, 0.0, 0.0];
        let theta = angular_distance(q, identity, true);
        // Expected rotation angle: 90° = π/2.
        assert!((theta - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn global_shape_and_values() {
        let q = vec![[1.0_f64, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let g = vec![[1.0_f64, 0.0, 0.0, 0.0]];
        let r = &AngularSeparationGlobal::new()
            .compute(
                &[&frame()],
                AngularSeparationGlobalArgs {
                    query: &q,
                    global: &g,
                },
            )
            .unwrap()[0];
        assert_eq!(r.angles.dim(), (2, 1));
        assert!(r.angles[[0, 0]].abs() < TOL);
        // (0,1,0,0) is a 180° rotation about x; cf. equivalent_orientations
        // = true → |dot| = 0 → angle = π.
        assert!((r.angles[[1, 0]] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn empty_inputs_error() {
        let err = AngularSeparationGlobal::new()
            .compute(
                &[&frame()],
                AngularSeparationGlobalArgs {
                    query: &[],
                    global: &[[1.0, 0.0, 0.0, 0.0]],
                },
            )
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }
}
