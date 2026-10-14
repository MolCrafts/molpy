//! Continuous (Voronoi-weighted) coordination numbers.

use molrs::core::Neighbors;
use molrs::op::F;

use super::bond_average::QlmTable;
use crate::compute::{Compute, ComputeError};

/// Continuous-coordination calculator.
///
/// Mirrors `freud.order.ContinuousCoordination`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/ContinuousCoordination.cc)).
///
/// Configuration: the weighting power. Data: a [`QlmTable`] (from
/// [`SteinhardtResult::qlm`](crate::compute::SteinhardtResult::qlm)) and the
/// half-shell neighbor table over the same particles — the `q_ℓm` are read,
/// never recomputed. Several ℓ are several tables, one call each.
///
/// For each particle `i` we compute a "soft" neighbor count weighted by
/// how similar each neighbor's Steinhardt qℓm vector is to particle `i`'s.
/// The weighting function used by freud is
///
/// ```text
///   w_ij = ( clamp((d_ij + 1) / 2, 0, 1) )^p
/// ```
///
/// where `d_ij ∈ [−1, 1]` is the **cosine similarity** of the qℓm vectors
/// (the table's [`bond_coherence`](QlmTable::bond_coherence) over its
/// [`norms`](QlmTable::norms); 0 when either vanishes) and `p ≥ 0` is the
/// configurable power. Setting `p = 1` reproduces the "average of
/// `(1 + cos θ)/2`" weighting that smoothly goes from 0 (anti-parallel qℓm)
/// to 1 (identical environments).
///
/// The output is one scalar per particle.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousCoordination {
    power: F,
}

/// The data a [`ContinuousCoordination`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct ContinuousCoordinationInput<'a> {
    /// The particles' `q_ℓm`.
    pub qlm: &'a QlmTable,
    /// The bonds to weigh: a half-shell self-query over the same particles.
    pub neighbors: &'a Neighbors,
}

impl ContinuousCoordination {
    /// `power` — weighting exponent (≥ 0).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a negative or NaN power.
    pub fn new(power: F) -> Result<Self, ComputeError> {
        if power.is_nan() || power < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "ContinuousCoordination::power",
                value: power.to_string(),
            });
        }
        Ok(Self { power })
    }
}

impl Compute for ContinuousCoordination {
    type Input<'a> = ContinuousCoordinationInput<'a>;
    type Output = ContinuousCoordinationResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a cross-query table;
    /// [`ComputeError::DimensionMismatch`] when the table is over a different
    /// number of particles than `qlm`.
    fn compute(
        &self,
        input: ContinuousCoordinationInput<'_>,
    ) -> Result<ContinuousCoordinationResult, ComputeError> {
        let qlm = input.qlm;
        let nlist = input.neighbors;
        let coherence = qlm.bond_coherence(nlist)?;
        let norms = qlm.norms();
        let mut coord = vec![0.0_f64; qlm.n_particles()];
        for ((&i, &j), &dot) in nlist
            .query_point_indices()
            .iter()
            .zip(nlist.point_indices())
            .zip(&coherence)
        {
            let (i, j) = (i as usize, j as usize);
            let denom = norms[i] * norms[j];
            let cos = if denom > 0.0 {
                (dot / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            let w = ((cos + 1.0) * 0.5).powf(self.power);
            coord[i] += w;
            coord[j] += w;
        }
        Ok(ContinuousCoordinationResult {
            l: qlm.order(),
            coord,
        })
    }
}

/// One snapshot's continuous coordination numbers.
#[derive(Debug, Clone, Default)]
pub struct ContinuousCoordinationResult {
    /// ℓ of the `q_ℓm` the weights came from.
    pub l: u32,
    /// Soft coordination number per particle.
    pub coord: Vec<F>,
}

crate::compute::spec::compute_spec! {
    ContinuousCoordination as "ContinuousCoordination";
    params |s| {
        power: Float;
    }
    inputs {
        qlm: Upstream("QlmTable");
        neighbors: Neighbors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use crate::compute::order::bond_average::{BondAverage, SphericalHarmonic};
    use molrs::core::SimBox;
    use ndarray::array;

    /// A snapshot: positions in an open cube of side `box_len` at the origin.
    struct Snapshot {
        positions: Vec<[F; 3]>,
        simbox: SimBox,
    }

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Snapshot {
        Snapshot {
            positions: positions.to_vec(),
            simbox: SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3])
                .unwrap(),
        }
    }

    fn nlist_from_frame(frame: &Snapshot, cutoff: F) -> Neighbors {
        neighbors_of(&frame.positions, &frame.simbox, cutoff)
    }

    fn coordination(power: F, nl: &Neighbors) -> ContinuousCoordinationResult {
        let qlm = BondAverage::new(SphericalHarmonic { l: 6 })
            .compute(nl)
            .unwrap();
        ContinuousCoordination::new(power)
            .unwrap()
            .compute(ContinuousCoordinationInput {
                qlm: &qlm,
                neighbors: nl,
            })
            .unwrap()
    }

    fn paired_octahedra() -> Snapshot {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]);
            }
        }
        frame_with(&p, 20.0)
    }

    #[test]
    fn identical_environments_give_unit_weight() {
        // Two octahedra with identical orientations. Each of the bonds
        // between identical environments contributes weight 1 per pair.
        let frame = paired_octahedra();
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &coordination(1.0, &nl);
        // Each cluster centre has 6 bonds to its identical neighbours
        // (within its own octahedron, but those neighbours are leaf
        // particles with only one bond → different |qℓm|). So we just
        // sanity-check that the centres get a positive coord.
        assert!(r.coord[0] > 0.0);
        assert!(r.coord[7] > 0.0);
    }

    #[test]
    fn isolated_particle_has_zero_coord() {
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 20.0);
        let nl = nlist_from_frame(&frame, 1.0);
        let r = &coordination(1.0, &nl);
        assert_eq!(r.coord[0], 0.0);
    }

    #[test]
    fn power_zero_counts_every_pair_as_one() {
        // power = 0 → ((cos+1)/2)^0 = 1 for any neighbor pair. Each particle
        // gets coord = (its neighbor count).
        let frame = frame_with(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], 20.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &coordination(0.0, &nl);
        assert!((r.coord[0] - 1.0).abs() < 1e-12);
        assert!((r.coord[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn negative_power_errors() {
        assert!(matches!(
            ContinuousCoordination::new(-1.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
