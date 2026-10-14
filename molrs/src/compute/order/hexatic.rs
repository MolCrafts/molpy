//! Hexatic order parameter `ψ_k` for 2-D systems.

use crate::compute::ComputeResult;
use molrs::core::Complex;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::compute::{require_disp, require_self_query};

/// Hexatic order parameter calculator.
///
/// Stateless parameter container: just the symmetry order `k`.
/// [`compute`](LegacyCompute::compute) takes `&Vec<Neighbors>` — one neighbor table
/// per frame, index-aligned with `frames`, each carrying the `disp` column (Å);
/// a table without it is [`ComputeError::BadShape`], never a silent zero.
///
/// Each table must also be a half-shell
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
/// orientation. Isolated particles get `|ψ_k| = 0`.
///
/// The z-component of the bond vector is ignored — callers must arrange
/// that the configuration is genuinely planar (typically `Lz = 1`,
/// `pbc.z = false`).
///
/// A self-query table is half-shell — it holds each bond once, as `(i, j)` with
/// `i < j` — so the accumulator visits a pair once and updates *both*
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
/// than zeros, so [`Hexatic::compute`](LegacyCompute::compute) answers
/// [`ComputeError::BadShape`] naming the missing column instead of indexing an
/// empty view.
#[derive(Debug, Clone, Copy)]
pub struct Hexatic {
    k: u32,
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

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<HexaticResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let mut psi = vec![Complex::ZERO; n];
        let mut count = vec![0_u32; n];

        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        // The bond directions are the whole computation: a table without the
        // `disp` column cannot supply them, and zeros would be silent nonsense.
        let disp = require_disp(nlist)?;
        // The loop below reads each row once and updates both `i` and `j`, using
        // the parity of `e^{i k (θ + π)}` for the `j` side. That credits `j` as
        // if it indexed the same point set as `i`, which only a self-query
        // guarantees: on a cross table `j` indexes the *reference* set, so the
        // bond is attributed to whichever frame particle happens to share that
        // index — or panics outright once `j >= n`. And when the two sets are
        // the same coordinates, every bond arrives in both orderings, so each
        // particle's `count` is `2·N_i`. Refuse the table instead.
        require_self_query(nlist)?;

        // For the j-side of each self-query bond, the bond direction is
        // reversed: θ + π. `exp(i k (θ + π)) = (-1)^k exp(i k θ)`.
        let parity = if self.k & 1 == 0 { 1.0 } else { -1.0 };

        for kp in 0..n_pairs {
            let i = i_idx[kp] as usize;
            let j = j_idx[kp] as usize;
            let dx = disp[[kp, 0]];
            let dy = disp[[kp, 1]];
            if dx == 0.0 && dy == 0.0 {
                continue;
            }
            let theta = dy.atan2(dx);
            let phase = Complex::from_polar(1.0, self.k as F * theta);
            psi[i] += phase;
            psi[j] += phase.scale(parity);
            count[i] += 1;
            count[j] += 1;
        }
        for i in 0..n {
            if count[i] > 0 {
                let inv = 1.0 / count[i] as F;
                psi[i] = psi[i].scale(inv);
            }
        }
        Ok(HexaticResult { k: self.k, psi })
    }
}

impl LegacyCompute for Hexatic {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<HexaticResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<HexaticResult>, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if frames.len() != nlists.len() {
            return Err(ComputeError::DimensionMismatch {
                expected: frames.len(),
                got: nlists.len(),
                what: "neighbor-list count",
            });
        }
        #[cfg(feature = "rayon")]
        const PAR_THRESHOLD: usize = 2;

        #[cfg(feature = "rayon")]
        if frames.len() >= PAR_THRESHOLD {
            use rayon::prelude::*;
            return frames
                .par_iter()
                .zip(nlists.par_iter())
                .map(|(f, nl)| self.one_frame(*f, nl))
                .collect();
        }

        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame hexatic order parameter result.
#[derive(Debug, Clone, Default)]
pub struct HexaticResult {
    /// Rotational symmetry order `k` the parameter was computed for.
    pub k: u32,
    /// Per-particle complex `ψ_k(i)`, one entry per particle in frame order.
    /// Dimensionless, with `|ψ_k| ≤ 1`; its argument is the local lattice
    /// orientation (radians, modulo `2π/k`).
    pub psi: Vec<Complex>,
}

impl ComputeResult for HexaticResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

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
        frame.simbox = Some(
            SimBox::ortho(
                array![box_len, box_len, 1.0_f64],
                array![0.0 as F, 0.0 as F, 0.0 as F],
                [false, false, false],
            )
            .unwrap(),
        );
        frame
    }

    /// Six neighbors on a regular hexagon around a centre particle.
    fn hex_environment(box_len: F) -> Frame {
        let c = box_len * 0.5;
        let mut positions = vec![[c, c, 0.0]];
        for k in 0..6 {
            let theta = 2.0_f64 * std::f64::consts::PI * k as F / 6.0;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        frame_with(&positions, box_len)
    }

    /// Four neighbors on a square around a centre particle.
    fn square_environment(box_len: F) -> Frame {
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
        let res = Hexatic::new(6).unwrap().compute(&[&frame], &[nl]).unwrap();
        // ψ_6 at the centre should have magnitude ≈ 1.
        let psi_center = res[0].psi[0];
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
        let res = Hexatic::new(4).unwrap().compute(&[&frame], &[nl]).unwrap();
        let psi_center = res[0].psi[0];
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
        let res = Hexatic::new(6).unwrap().compute(&[&frame], &[nl]).unwrap();
        let psi_center = res[0].psi[0];
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
        let res = Hexatic::new(6).unwrap().compute(&[&frame], &[nl]).unwrap();
        assert_eq!(res[0].psi[0], Complex::ZERO);
    }

    #[test]
    fn k_zero_is_error() {
        assert!(Hexatic::new(0).is_err());
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Hexatic::new(6)
            .unwrap()
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
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

        let frame = hex_environment(20.0);
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
            .compute(&[&frame], &[nl])
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

        let frame = hex_environment(20.0);
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
        let Err(err) = Hexatic::new(6).unwrap().compute(&[&frame], &[nl]) else {
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
        let a = Hexatic::new(6).unwrap().compute(&[&frame], &[nl]).unwrap();
        let mag = a[0].psi[0].norm();

        let c = 10.0_f64;
        let shift = 0.3_f64;
        let mut positions = vec![[c, c, 0.0]];
        for kk in 0..6 {
            let theta = 2.0 * std::f64::consts::PI * kk as F / 6.0 + shift;
            positions.push([c + theta.cos(), c + theta.sin(), 0.0]);
        }
        let frame2 = frame_with(&positions, 20.0);
        let nl2 = nlist_from_frame(&frame2, 1.2);
        let b = Hexatic::new(6)
            .unwrap()
            .compute(&[&frame2], &[nl2])
            .unwrap();
        assert!(
            (b[0].psi[0].norm() - mag).abs() < 1e-10,
            "|ψ_6| should be rotation-invariant: {} vs {}",
            mag,
            b[0].psi[0].norm(),
        );
    }

    /// The `frames.len() >= PAR_THRESHOLD` rayon branch must return exactly
    /// what the serial path does, in frame order: two identical frames →
    /// two identical results, each equal to the single-frame result.
    #[test]
    fn parallel_matches_serial() {
        let frame = hex_environment(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let hx = Hexatic::new(6).unwrap();
        let solo = hx.compute(&[&frame], std::slice::from_ref(&nl)).unwrap();
        let par = hx.compute(&[&frame, &frame], &[nl.clone(), nl]).unwrap();
        assert_eq!(par.len(), 2);
        assert_eq!(par[0].psi, solo[0].psi);
        assert_eq!(par[1].psi, solo[0].psi);
    }
}
