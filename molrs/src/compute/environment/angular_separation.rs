//! Pairwise angular separation between orientation quaternions.
//!
//! The angular distance between two rotations `q₁`, `q₂` (unit quaternions)
//! is
//!
//! ```text
//!   θ = 2 · arccos( |q₁ · q₂| )
//! ```
//!
//! where the absolute value accounts for the double cover (`q` and `−q`
//! represent the same rotation), giving `0 ≤ θ ≤ π` in radians. Without the
//! double-cover identification (`equivalent_orientations = false`, freud's
//! name) the absolute value is dropped and `θ ∈ [0, 2π]`.
//!
//! Mirrors `freud.environment.AngularSeparationGlobal` and
//! `AngularSeparationNeighbor`
//! ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/AngularSeparation.cc)):
//!
//! - [`AngularSeparationGlobal`]: dense `(N_query × N_global)` table between
//!   every query orientation and every reference orientation.
//! - [`AngularSeparationNeighbor`]: sparse, one angular distance per neighbor
//!   pair.
//!
//! Quaternions are `(w, x, y, z)` rows of an `N × 4` array, normalised here;
//! a zero quaternion is no rotation and is an error (it used to give 0).

use molrs::core::Neighbors;
use molrs::op::{F, Quat, quat_dot};
use ndarray::{Array2, ArrayView2};

use crate::compute::{Compute, ComputeError, require_quaternions};

/// The angular distance between unit quaternions, radians.
fn angular_distance(q1: Quat, q2: Quat, equivalent_orientations: bool) -> F {
    let mut d = quat_dot(q1, q2);
    if equivalent_orientations {
        d = d.abs();
    }
    2.0 * d.clamp(-1.0, 1.0).acos()
}

// ---------------------------------------------------------------------------
// Global: dense (N_query × N_global) table
// ---------------------------------------------------------------------------

/// Dense angular separation between query and reference orientations.
///
/// Configuration: whether `q` and `−q` are the same orientation. Data: the
/// query quaternions (typically one per particle) and a small fixed list of
/// reference quaternions (e.g. point-group symmetries).
#[derive(Debug, Clone, Copy)]
pub struct AngularSeparationGlobal {
    equivalent_orientations: bool,
}

/// The data an [`AngularSeparationGlobal`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct AngularSeparationGlobalInput<'a> {
    /// Query quaternions, `N_query × 4`.
    pub query: ArrayView2<'a, F>,
    /// Reference quaternions, `N_global × 4`.
    pub global: ArrayView2<'a, F>,
}

impl Default for AngularSeparationGlobal {
    fn default() -> Self {
        Self::new()
    }
}

impl AngularSeparationGlobal {
    /// Defaults: `q` and `−q` are the same orientation.
    pub fn new() -> Self {
        Self {
            equivalent_orientations: true,
        }
    }

    /// Identify `q` with `−q` when `on` (default true).
    pub fn with_equivalent_orientations(mut self, on: bool) -> Self {
        self.equivalent_orientations = on;
        self
    }
}

impl Compute for AngularSeparationGlobal {
    type Input<'a> = AngularSeparationGlobalInput<'a>;
    type Output = AngularSeparationGlobalResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] when either set is empty;
    /// [`ComputeError::DimensionMismatch`] for an array that is not `N × 4`;
    /// [`ComputeError::NonFinite`] / [`ComputeError::OutOfRange`] for a
    /// non-finite or zero quaternion.
    fn compute(
        &self,
        input: AngularSeparationGlobalInput<'_>,
    ) -> Result<AngularSeparationGlobalResult, ComputeError> {
        let query = require_quaternions(input.query, "query")?;
        let global = require_quaternions(input.global, "global")?;
        if query.is_empty() || global.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        // O(n_query × n_global) by definition: the observable *is* the full
        // matrix of pairwise angular distances.
        let angles = Array2::from_shape_fn((query.len(), global.len()), |(i, j)| {
            angular_distance(query[i], global[j], self.equivalent_orientations)
        });
        Ok(AngularSeparationGlobalResult { angles })
    }
}

// ---------------------------------------------------------------------------
// Neighbor: sparse, one angle per pair
// ---------------------------------------------------------------------------

/// Angular separation between the orientations of every neighbor pair.
///
/// Configuration: whether `q` and `−q` are the same orientation. Data: one
/// snapshot's neighbor table (any columns), the query points' and the
/// reference points' quaternions.
#[derive(Debug, Clone, Copy)]
pub struct AngularSeparationNeighbor {
    equivalent_orientations: bool,
}

/// The data an [`AngularSeparationNeighbor`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct AngularSeparationNeighborInput<'a> {
    /// One snapshot's neighbor table.
    pub neighbors: &'a Neighbors,
    /// Query-point quaternions (indexed by `i`), `n_query_points × 4`.
    pub query_orientations: ArrayView2<'a, F>,
    /// Reference-point quaternions (indexed by `j`), `n_points × 4`.
    pub point_orientations: ArrayView2<'a, F>,
}

impl Default for AngularSeparationNeighbor {
    fn default() -> Self {
        Self::new()
    }
}

impl AngularSeparationNeighbor {
    /// Defaults: `q` and `−q` are the same orientation.
    pub fn new() -> Self {
        Self {
            equivalent_orientations: true,
        }
    }

    /// Identify `q` with `−q` when `on` (default true).
    pub fn with_equivalent_orientations(mut self, on: bool) -> Self {
        self.equivalent_orientations = on;
        self
    }
}

impl Compute for AngularSeparationNeighbor {
    type Input<'a> = AngularSeparationNeighborInput<'a>;
    type Output = AngularSeparationNeighborResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] for an orientation array that is
    /// not `N × 4` or not one per point of its set;
    /// [`ComputeError::NonFinite`] / [`ComputeError::OutOfRange`] for a
    /// non-finite or zero quaternion.
    fn compute(
        &self,
        input: AngularSeparationNeighborInput<'_>,
    ) -> Result<AngularSeparationNeighborResult, ComputeError> {
        let nl = input.neighbors;
        let q = require_quaternions(input.query_orientations, "query_orientations")?;
        let p = require_quaternions(input.point_orientations, "point_orientations")?;
        if q.len() != nl.n_query_points() {
            return Err(ComputeError::DimensionMismatch {
                expected: nl.n_query_points(),
                got: q.len(),
                what: "query_orientations (one per query point)",
            });
        }
        if p.len() != nl.n_points() {
            return Err(ComputeError::DimensionMismatch {
                expected: nl.n_points(),
                got: p.len(),
                what: "point_orientations (one per reference point)",
            });
        }
        let mut angles = Vec::with_capacity(nl.n_pairs());
        for (k, (&i, &j)) in nl
            .query_point_indices()
            .iter()
            .zip(nl.point_indices())
            .enumerate()
        {
            let (i, j) = (i as usize, j as usize);
            if i >= q.len() || j >= p.len() {
                return Err(ComputeError::OutOfRange {
                    field: "pair index",
                    value: format!("({i}, {j}) of pair {k} past the table's point counts"),
                });
            }
            angles.push(angular_distance(q[i], p[j], self.equivalent_orientations));
        }
        Ok(AngularSeparationNeighborResult { angles })
    }
}

/// The dense angular-separation table.
#[derive(Debug, Clone, Default)]
pub struct AngularSeparationGlobalResult {
    /// `(n_query, n_global)` table of angular distances (radians).
    pub angles: Array2<F>,
}

/// The per-pair angular separations.
#[derive(Debug, Clone, Default)]
pub struct AngularSeparationNeighborResult {
    /// One angle per neighbor pair, in radians; index matches the
    /// underlying [`Neighbors`] pair index.
    pub angles: Vec<F>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{NeighborColumns, NeighborPair, QueryMode};
    use ndarray::array;

    const TOL: F = 1e-10;

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
        let q = array![[1.0_f64, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let g = array![[1.0_f64, 0.0, 0.0, 0.0]];
        let r = &AngularSeparationGlobal::new()
            .compute(AngularSeparationGlobalInput {
                query: q.view(),
                global: g.view(),
            })
            .unwrap();
        assert_eq!(r.angles.dim(), (2, 1));
        assert!(r.angles[[0, 0]].abs() < TOL);
        // (0,1,0,0) is a 180° rotation about x; cf. equivalent_orientations
        // = true → |dot| = 0 → angle = π.
        assert!((r.angles[[1, 0]] - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn empty_inputs_error() {
        let none = Array2::<F>::zeros((0, 4));
        let g = array![[1.0, 0.0, 0.0, 0.0]];
        let err = AngularSeparationGlobal::new()
            .compute(AngularSeparationGlobalInput {
                query: none.view(),
                global: g.view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    /// A zero quaternion used to give distance 0 to everything; it is no
    /// rotation, so it is an error now.
    #[test]
    fn zero_quaternion_is_out_of_range() {
        let q = array![[0.0, 0.0, 0.0, 0.0]];
        let g = array![[1.0, 0.0, 0.0, 0.0]];
        let err = AngularSeparationGlobal::new()
            .compute(AngularSeparationGlobalInput {
                query: q.view(),
                global: g.view(),
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    /// One angle per pair: `(0, 1)` between the identity and a 90° turn
    /// about z.
    #[test]
    fn neighbor_angles_follow_the_table() {
        let nl = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let h = std::f64::consts::FRAC_PI_4;
        let q = array![[1.0, 0.0, 0.0, 0.0], [h.cos(), 0.0, 0.0, h.sin()]];
        let r = AngularSeparationNeighbor::new()
            .compute(AngularSeparationNeighborInput {
                neighbors: &nl,
                query_orientations: q.view(),
                point_orientations: q.view(),
            })
            .unwrap();
        assert_eq!(r.angles.len(), 1);
        assert!((r.angles[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-12);

        let short = array![[1.0, 0.0, 0.0, 0.0]];
        assert!(matches!(
            AngularSeparationNeighbor::new().compute(AngularSeparationNeighborInput {
                neighbors: &nl,
                query_orientations: short.view(),
                point_orientations: q.view(),
            }),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }
}
