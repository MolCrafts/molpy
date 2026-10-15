//! Hexatic order parameter `ψ_k` for 2-D systems.

use molrs::core::Complex;
use molrs::core::Neighbors;

use super::bond_average::{BondAverage, PlanarHarmonic};
use crate::compute::{Compute, ComputeError};

/// Hexatic order parameter calculator.
///
/// Configuration: the symmetry order `k`. Data: one snapshot's neighbor
/// table, carrying the `disp` column (Å); a table without it is
/// [`ComputeError::BadShape`], never a silent zero. Many snapshots go through
/// [`Batch`](crate::compute::Batch).
///
/// The table must also be a half-shell
/// [`SelfQuery`](molrs::core::QueryMode::SelfQuery): every row is
/// visited once and credited to *both* of its particles, which double-counts on
/// a [`CrossQuery`](molrs::core::QueryMode::CrossQuery) table, so
/// that table is [`ComputeError::BadShape`] too.
///
/// Mirrors `freud.order.Hexatic`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/HexaticTranslational.cc)).
///
/// For each particle `i` and a chosen integer rotational symmetry `k`
/// (typically 6 for hexagonal lattices, 4 for square, 3 for triangular),
///
/// ```text
///   ψ_k(i) = (1/N_i) Σ_{j ∈ neigh(i)} e^{i k θ_{ij}}
/// ```
///
/// where `N_i` is the number of neighbors of `i` and `θ_{ij}` is the in-plane
/// angle in radians (`atan2(dy, dx)`) of the bond `r_j − r_i`. `ψ_k` is a
/// dimensionless complex number with `|ψ_k| ≤ 1`: the magnitude is 1 when the
/// bonds sit at perfect `2π/k` spacing and 0 when their `k`-fold phases cancel
/// (four bonds at 90° give `|ψ_6| = 0`), and the argument is the local lattice
/// orientation. Isolated particles get `|ψ_k| = 0`. It is the planar
/// bond-harmonic average — the same engine as Steinhardt's `q_ℓm`.
///
/// The z-component of the bond vector is ignored — callers must arrange
/// that the configuration is genuinely planar (typically `Lz = 1`,
/// `pbc.z = false`).
///
/// A self-query table is half-shell — it holds each bond once, as `(i, j)` with
/// `i < j` — so the average visits a pair once and updates *both*
/// particles: from `j`'s side the same bond points the other way, `θ + π`, and
/// `e^{i k (θ + π)} = (−1)^k e^{i k θ}`.
///
/// # Required neighbor columns
///
/// Only the bond *direction* enters `ψ_k`, so the neighbor table must carry the
/// minimum-image displacement column `disp` (Å) — materialize it with
/// [`NeighborColumns::DISP`](molrs::core::NeighborColumns::DISP)
/// or [`FULL`](molrs::core::NeighborColumns::FULL). A `DIST_SQ`
/// or `INDICES_ONLY` table stores no directions and reads back `None` rather
/// than zeros, so `compute` answers [`ComputeError::BadShape`] naming the
/// missing column instead of indexing an empty view.
#[derive(Debug, Clone, Copy)]
pub struct Hexatic {
    k: u32,
}

/// The data a [`Hexatic`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct HexaticInput<'a> {
    /// One snapshot's half-shell self-query table with the `disp` column.
    pub neighbors: &'a Neighbors,
}

impl Hexatic {
    /// Calculator for `k`-fold symmetry (6 = hexatic, 4 = square/tetratic,
    /// 3 = triangular). Dimensionless — `k` counts lattice directions.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] when `k == 0`: `ψ_0` would be the mean of
    /// `e^{i·0} = 1` over the bonds, which measures nothing.
    pub fn new(k: u32) -> Result<Self, ComputeError> {
        if k == 0 {
            return Err(ComputeError::OutOfRange {
                field: "Hexatic::k",
                value: "0".into(),
            });
        }
        Ok(Self { k })
    }

    /// The configured `k`-fold symmetry order.
    pub fn k(&self) -> u32 {
        self.k
    }
}

impl Compute for Hexatic {
    type Input<'a> = HexaticInput<'a>;
    type Output = HexaticResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp` or a cross-query
    /// table.
    fn compute(&self, input: HexaticInput<'_>) -> Result<HexaticResult, ComputeError> {
        let table = BondAverage::new(PlanarHarmonic { k: self.k }).compute(input.neighbors)?;
        Ok(HexaticResult {
            k: self.k,
            psi: table.values().column(0).to_vec(),
        })
    }
}

/// One snapshot's hexatic order parameter.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct HexaticResult {
    /// Rotational symmetry order `k` the parameter was computed for.
    pub k: u32,
    /// Per-particle complex `ψ_k(i)`, one entry per particle in table order.
    /// Dimensionless, with `|ψ_k| ≤ 1`; its argument is the local lattice
    /// orientation (radians, modulo `2π/k`).
    pub psi: Vec<Complex>,
}

crate::compute::spec::compute_spec! {
    Hexatic as "Hexatic";
    params |s| {
        k: Int;
    }
    inputs {
        neighbors: Neighbors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::SimBox;
    use molrs::op::F;
    use ndarray::array;

    /// A planar snapshot: positions in a `box_len × box_len × 1` open box.
    struct Snapshot {
        positions: Vec<[F; 3]>,
        simbox: SimBox,
    }

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Snapshot {
        Snapshot {
            positions: positions.to_vec(),
            simbox: SimBox::ortho(
                array![box_len, box_len, 1.0_f64],
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        }
    }

    fn nlist_from_frame(frame: &Snapshot, cutoff: F) -> Neighbors {
        neighbors_of(&frame.positions, &frame.simbox, cutoff)
    }

    fn psi(k: u32, nl: &Neighbors) -> Vec<Complex> {
        Hexatic::new(k)
            .unwrap()
            .compute(HexaticInput { neighbors: nl })
            .unwrap()
            .psi
    }

    /// Six neighbors on a regular hexagon around a centre particle.
    fn hex_environment(box_len: F) -> Snapshot {
        let c = box_len * 0.5;
        let mut positions = vec![[c, c, 0.0]];
        for k in 0..6 {
            let theta = 2.0_f64 * std::f64::consts::PI * k as F / 6.0;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        frame_with(&positions, box_len)
    }

    /// Four neighbors on a square around a centre particle.
    fn square_environment(box_len: F) -> Snapshot {
        let c = box_len * 0.5;
        let positions = [
            [c, c, 0.0],
            [c + 1.0, c, 0.0],
            [c - 1.0, c, 0.0],
            [c, c + 1.0, 0.0],
            [c, c - 1.0, 0.0],
        ];
        frame_with(&positions, box_len)
    }

    #[test]
    fn psi_6_on_perfect_hexagon_is_unity() {
        let frame = hex_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let psi_center = psi(6, &nl)[0];
        assert!(
            (psi_center.norm() - 1.0).abs() < 1e-10,
            "|ψ_6(centre)| = {} (expected ≈ 1)",
            psi_center.norm(),
        );
    }

    #[test]
    fn psi_4_on_perfect_square_is_unity() {
        let frame = square_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let psi_center = psi(4, &nl)[0];
        assert!(
            (psi_center.norm() - 1.0).abs() < 1e-10,
            "|ψ_4(square)| = {} (expected ≈ 1)",
            psi_center.norm(),
        );
    }

    #[test]
    fn psi_6_on_square_is_zero() {
        // For 4 bonds at θ = 0, π/2, π, 3π/2: Σ e^{i 6 θ} = e^0 + e^{i 3π}
        // + e^{i 6π} + e^{i 9π} = 1 − 1 + 1 − 1 = 0.
        let frame = square_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let psi_center = psi(6, &nl)[0];
        assert!(
            psi_center.norm() < 1e-10,
            "ψ_6(square) should be 0, got {}",
            psi_center.norm(),
        );
    }

    #[test]
    fn isolated_particle_psi_is_zero() {
        let frame = frame_with(&[[5.0, 5.0, 0.0]], 10.0);
        let nl = nlist_from_frame(&frame, 1.0);
        assert_eq!(psi(6, &nl)[0], Complex::ZERO);
    }

    #[test]
    fn k_zero_is_error() {
        assert!(Hexatic::new(0).is_err());
    }

    /// ac-002 (spec neighborlist-03-compute): `ψ_k` is built entirely from bond
    /// *directions*, so an **indices-only** table — one that never stored a
    /// physical column at all — must be refused with `BadShape` instead of
    /// indexing an empty `disp` view (out-of-bounds panic / WASM `unreachable`).
    ///
    /// The pairs are the regular hexagon's own bonds, hard-coded rather than
    /// searched: centre 0 bonded to its six ring neighbours 1..=6 at unit
    /// distance and angles `2πk/6`. All satisfy `i < j`, so
    /// `SelfQuery { n_points: 7 }` is a legal label.
    #[test]
    fn hexatic_indices_only_neighbors_is_bad_shape() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        let pairs: Vec<NeighborPair> = (0..6u32)
            .map(|k| {
                let theta = 2.0 * std::f64::consts::PI * k as F / 6.0;
                NeighborPair {
                    i: 0,
                    j: k + 1,
                    dist_sq: 1.0,
                    disp: [theta.cos(), theta.sin(), 0.0],
                }
            })
            .collect();
        let nl = Neighbors::from_pairs(
            pairs,
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: 7 },
        );
        assert_eq!(
            nl.n_pairs(),
            6,
            "the guard must be tested on a non-empty table"
        );
        assert!(nl.disp().is_none());
        assert!(nl.dist_sq().is_none());

        let err = Hexatic::new(6)
            .unwrap()
            .compute(HexaticInput { neighbors: &nl })
            .unwrap_err();
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "indices-only Neighbors must be BadShape, not a panic; got {err:?}"
        );
    }

    /// ac-002 (spec neighborlist-03-compute): the accumulator visits each row
    /// once and credits the bond to *both* endpoints, using the parity of
    /// `e^{i k (θ + π)}` for the `j` side. That is only exact on a half-shell
    /// self-query; a **cross-query** table carries the reversed row as well, so
    /// every bond would be counted twice. The answer is then wrong rather than
    /// missing, which is why the mode is refused outright instead of computed.
    ///
    /// The table is otherwise impeccable — `FULL` storage, so `disp` and
    /// `dist_sq` both read back `Some` — to pin that it is the *mode* being
    /// refused and not a missing column.
    ///
    /// The pairs are the regular hexagon's own bonds, hard-coded rather than
    /// searched: centre 0 bonded to its six ring neighbours 1..=6 at unit
    /// distance and angles `2πk/6`. Both orderings of every bond are present,
    /// as a cross-query over two copies of the same point set reports them;
    /// `disp` stays `r_j − r_i`, so the reversed row carries `−disp`.
    #[test]
    fn hexatic_cross_query_table_is_bad_shape() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        let mut pairs: Vec<NeighborPair> = Vec::with_capacity(12);
        for k in 0..6u32 {
            let theta = 2.0 * std::f64::consts::PI * k as F / 6.0;
            let disp = [theta.cos(), theta.sin(), 0.0];
            let j = k + 1;
            pairs.push(NeighborPair {
                i: 0,
                j,
                dist_sq: 1.0,
                disp,
            });
            pairs.push(NeighborPair {
                i: j,
                j: 0,
                dist_sq: 1.0,
                disp: [-disp[0], -disp[1], -disp[2]],
            });
        }
        let nl = Neighbors::from_pairs(
            pairs,
            NeighborColumns::FULL,
            QueryMode::CrossQuery {
                n_query_points: 7,
                n_points: 7,
            },
        );
        assert_eq!(nl.n_pairs(), 12, "the guard must see a non-empty table");
        assert!(
            nl.disp().is_some() && nl.dist_sq().is_some(),
            "no column is missing here — the query mode must be what is refused"
        );

        // Deliberately not `expect_err`: the Ok payload is a ψ_k vector per
        // frame, and dumping it buries the one thing the failure says.
        let Err(err) = Hexatic::new(6)
            .unwrap()
            .compute(HexaticInput { neighbors: &nl })
        else {
            panic!("Hexatic::compute must refuse a cross-query table outright, but returned Ok");
        };
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "cross-query Neighbors must be BadShape; got {err:?}"
        );
    }

    #[test]
    fn rotation_invariant_in_magnitude() {
        // Rotate the hexagon by an arbitrary angle and check |ψ_6| unchanged.
        let frame = hex_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let mag = psi(6, &nl)[0].norm();

        let c = 10.0_f64;
        let shift = 0.3_f64;
        let mut positions = vec![[c, c, 0.0]];
        for kk in 0..6 {
            let theta = 2.0 * std::f64::consts::PI * kk as F / 6.0 + shift;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        let frame2 = frame_with(&positions, 20.0);
        let nl2 = nlist_from_frame(&frame2, 1.2);
        let b = psi(6, &nl2);
        assert!(
            (b[0].norm() - mag).abs() < 1e-10,
            "|ψ_6| should be rotation-invariant: {} vs {}",
            mag,
            b[0].norm(),
        );
    }

    /// [`Batch`] (rayon when enabled) must return exactly what one call
    /// does, in input order: two identical snapshots → two identical results,
    /// each equal to the single-snapshot result.
    #[test]
    fn parallel_matches_serial() {
        let frame = hex_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let hx = Hexatic::new(6).unwrap();
        let solo = hx.compute(HexaticInput { neighbors: &nl }).unwrap();
        let par = Batch(hx)
            .compute(vec![HexaticInput { neighbors: &nl }; 2])
            .unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].psi, solo.psi);
        assert_eq!(par[1].psi, solo.psi);
    }
}
