//! Frenkel–ten Wolde solid/liquid classification from `q_ℓm` bond correlations.

// Index-based loops over flat qℓm arrays read naturally; iterator forms
// would require chunks_exact + enumerate without gaining clarity.
#![allow(clippy::needless_range_loop)]

use crate::compute::ComputeResult;
use molrs::core::Complex;
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;

use super::steinhardt::steinhardt_qlm;
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;

/// Frenkel-ten Wolde solid/liquid classifier.
///
/// Mirrors `freud.order.SolidLiquid`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/SolidLiquid.cc)).
///
/// For each neighbor pair `(i, j)` we compute the normalised dot product
/// of their Steinhardt qℓm vectors:
///
/// ```text
///   d_ij = ( Σ_m q_ℓm(i) · conj q_ℓm(j) ) / ( |q_ℓm(i)| · |q_ℓm(j)| )
/// ```
///
/// A bond is **solid-like** when `Re(d_ij) > q_threshold` (typically `0.7`).
/// A particle is **solid** when it has at least `n_threshold` solid-like
/// bonds. The output is a per-particle solid-bond count plus the boolean
/// solid mask.
///
/// This phase reuses [`steinhardt_qlm`] directly
/// — no qℓm recomputation, no duplicate spherical-harmonic evaluations.
#[derive(Debug, Clone, Copy)]
pub struct SolidLiquid {
    l: u32,
    q_threshold: F,
    n_threshold: u32,
    normalize_q: bool,
}

impl SolidLiquid {
    /// Steinhardt order `l`; defaults: q_threshold 0.7, n_threshold 6, normalized `q_ℓm`.
    pub fn new(l: u32) -> Self {
        Self {
            l,
            q_threshold: 0.7,
            n_threshold: 6,
            normalize_q: true,
        }
    }

    /// Dot-product threshold for a solid-like bond (default 0.7).
    pub fn with_q_threshold(mut self, t: F) -> Self {
        self.q_threshold = t;
        self
    }

    /// Minimum solid-like bonds for a solid particle (default 6).
    pub fn with_n_threshold(mut self, n: u32) -> Self {
        self.n_threshold = n;
        self
    }

    /// If false, use the raw (unnormalised) dot product Σ_m qℓm·conj(qℓm)
    /// instead of cosine similarity. Default `true` matches freud.
    pub fn with_normalize_q(mut self, on: bool) -> Self {
        self.normalize_q = on;
        self
    }

    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<SolidLiquidResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();
        let m_count = (2 * self.l + 1) as usize;

        let qlm = steinhardt_qlm(frame, nlist, self.l)?;

        // |qℓm(i)| for normalisation. Skip particles with no neighbors → norm = 0.
        let mut norms = vec![0.0_f64; n];
        for i in 0..n {
            let off = i * m_count;
            let mut s: F = 0.0;
            for m in 0..m_count {
                s += qlm[off + m].norm_sqr();
            }
            norms[i] = s.sqrt();
        }

        let i_idx = nlist.query_point_indices();
        let j_idx = nlist.point_indices();
        let n_pairs = nlist.n_pairs();
        let mut n_solid_bonds = vec![0_u32; n];

        for k in 0..n_pairs {
            let i = i_idx[k] as usize;
            let j = j_idx[k] as usize;
            let mut dot = Complex::ZERO;
            let off_i = i * m_count;
            let off_j = j * m_count;
            for m in 0..m_count {
                // conj(a) * b = (a.re·b.re + a.im·b.im) + i(a.re·b.im − a.im·b.re)
                dot += qlm[off_i + m].conj() * qlm[off_j + m];
            }
            let real = if self.normalize_q {
                let denom = norms[i] * norms[j];
                if denom > 0.0 { dot.re / denom } else { 0.0 }
            } else {
                dot.re
            };
            if real > self.q_threshold {
                n_solid_bonds[i] += 1;
                n_solid_bonds[j] += 1;
            }
        }

        let is_solid: Vec<bool> = n_solid_bonds
            .iter()
            .map(|&c| c >= self.n_threshold)
            .collect();
        Ok(SolidLiquidResult {
            l: self.l,
            n_solid_bonds,
            is_solid,
        })
    }
}

impl LegacyCompute for SolidLiquid {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<SolidLiquidResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<SolidLiquidResult>, ComputeError> {
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

/// Per-frame solid/liquid classification.
#[derive(Debug, Clone, Default)]
pub struct SolidLiquidResult {
    /// ℓ used for the qℓm dot product.
    pub l: u32,
    /// Solid-like bond count per particle.
    pub n_solid_bonds: Vec<u32>,
    /// `true` if the particle has ≥ `n_threshold` solid-like bonds.
    pub is_solid: Vec<bool>,
}

impl ComputeResult for SolidLiquidResult {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::nlist_from_frame;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use ndarray::{Array1 as A1, array};

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Frame {
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
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap());
        frame
    }

    /// Two octahedra sharing the same orientation: every neighbor pair across
    /// the symmetry-mate has qℓm dot ≈ 1 (identical environments).
    fn paired_octahedra(box_len: F) -> Frame {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0_f64, 0.0, 0.0),
                (0.0_f64, 1.0, 0.0),
                (0.0_f64, -1.0, 0.0),
                (0.0_f64, 0.0, 1.0),
                (0.0_f64, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]);
            }
        }
        frame_with(&p, box_len, [false; 3])
    }

    #[test]
    fn dot_product_self_is_one() {
        // For a single octahedron, the centre particle has neighbours whose
        // |qℓm| is zero (they each have only one neighbour). So check that
        // the bond between the centre and a neighbour gives a real-valued
        // (possibly negative) dot, and that with q_threshold < -∞ both
        // counts go up.
        let frame = frame_with(
            &[
                [5.0, 5.0, 5.0],
                [6.0, 5.0, 5.0],
                [4.0, 5.0, 5.0],
                [5.0, 6.0, 5.0],
                [5.0, 4.0, 5.0],
                [5.0, 5.0, 6.0],
                [5.0, 5.0, 4.0],
            ],
            20.0,
            [false; 3],
        );
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &SolidLiquid::new(6)
            .with_q_threshold(-2.0) // count every bond as "solid"
            .with_n_threshold(1)
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        // Centre particle has 6 bonds, each counted once.
        assert_eq!(r.n_solid_bonds[0], 6);
        // Outer particles each have 1 bond.
        for i in 1..7 {
            assert_eq!(r.n_solid_bonds[i], 1);
        }
    }

    /// The bond between two identical-environment particles is solid-like: its
    /// cosine similarity is 1, the largest value `d_ij` can take.
    ///
    /// The fixture is [`paired_octahedra`] — two identically oriented octahedra
    /// whose centres (particles 0 and 7) sit 5 Å apart along `x̂`, so both
    /// centres see the same six bond directions. The centre-centre bond is
    /// therefore the one this test needs, and no cutoff that keeps the 1 Å
    /// octahedral bonds also reports a 5 Å one; the table is a hand-written bond
    /// list instead (a [`Neighbors`] is an *input* here, not necessarily the
    /// output of a cutoff search): the 12 octahedral bonds plus that one.
    ///
    /// The expected counts are derived by hand, not read back from a run. Write
    /// `A = Y₆ₘ(x̂)`, `B = Y₆ₘ(ŷ)`, `C = Y₆ₘ(ẑ)`. ℓ = 6 is even, so
    /// `Y₆ₘ(−d̂) = Y₆ₘ(d̂)` and each centre accumulates `3A + 2B + 2C` over its
    /// 7 bonds — the six spokes give `2A + 2B + 2C`, and the centre-centre bond
    /// adds one more `A` at *both* ends (it leaves 0 along `+x̂` and 7 along
    /// `−x̂`, the same harmonic). Every outer atom has a single bond, so its
    /// `q₆ₘ` is exactly `A`, `B` or `C`. With the addition theorem
    /// `Σₘ Y*ℓₘ(d̂₁) Yℓₘ(d̂₂) = ((2ℓ+1)/4π)·Pℓ(d̂₁·d̂₂)`, `u = 13/4π` and
    /// `P₆(0) = −5/16`, the centre norm is `|q_c|² = (u/49)(17 + 32·(−5/16))
    /// = u/7` and every outer atom's is `|q_o|² = u`. That leaves three
    /// distinct bond scores:
    ///
    /// - centre·centre: identical vectors → `d = 1`, above the 0.7 default.
    /// - centre·(x-spoke): `⟨q_c, A⟩ = (u/7)(3 + 4·(−5/16)) = u/4`, so
    ///   `d = (u/4)/√(u²/7) = √7/4 ≈ 0.6614` — just *below* 0.7. (The extra
    ///   `A` is what singles the x-spokes out; without the centre-centre bond
    ///   all six spokes would tie.)
    /// - centre·(y- or z-spoke): `⟨q_c, B⟩ = (u/7)(2 + 5·(−5/16)) = u/16`, so
    ///   `d = √7/16 ≈ 0.1654`.
    ///
    /// So the whole frame contains exactly one solid-like bond, and it is the
    /// one between the two identical environments. The `0.6614` figure is not
    /// idle: dropping `q_threshold` to `0.6` turns the four x-spoke bonds solid
    /// as well (`n_solid_bonds` becomes `[3, 1, 1, 0, …, 3, 1, 1, 0, …]`), which
    /// is what pins the default-threshold result to the *value* of the score
    /// rather than to the mere presence of the centre-centre bond.
    #[test]
    fn identical_environments_score_one() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        let frame = paired_octahedra(20.0);

        // The six octahedral spokes, in the order `paired_octahedra` lays the
        // outer atoms out around each centre: `r_outer − r_centre` (Å).
        const SPOKES: [[F; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let mut pairs: Vec<NeighborPair> = Vec::with_capacity(13);
        for centre in [0_u32, 7] {
            for (k, disp) in SPOKES.iter().enumerate() {
                pairs.push(NeighborPair {
                    i: centre,
                    j: centre + 1 + k as u32,
                    dist_sq: 1.0,
                    disp: *disp,
                });
            }
        }
        // The bond the test is about: centre 0 → centre 7, 5 Å along +x̂.
        pairs.push(NeighborPair {
            i: 0,
            j: 7,
            dist_sq: 25.0,
            disp: [5.0, 0.0, 0.0],
        });
        let nl = Neighbors::from_pairs(
            pairs,
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 14 },
        );

        // Premise of the derivation above, checked rather than assumed: the two
        // centres really do end up with the same q₆ₘ vector.
        let qlm = steinhardt_qlm(&frame, &nl, 6).unwrap();
        let m = 13_usize; // 2·6 + 1
        for k in 0..m {
            let a = qlm[k];
            let b = qlm[7 * m + k];
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "centre q₆ₘ components must match across symmetric octahedra"
            );
        }

        // Default q_threshold (0.7) keeps only the centre-centre bond;
        // n_threshold 1 then makes the two centres — and nothing else — solid.
        let r = &SolidLiquid::new(6)
            .with_n_threshold(1)
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        let expected_bonds: Vec<u32> = (0..14).map(|i| u32::from(i == 0 || i == 7)).collect();
        assert_eq!(
            r.n_solid_bonds, expected_bonds,
            "only the centre-centre bond clears 0.7: the x-spokes score √7/4 ≈ 0.6614 and the \
             y-/z-spokes √7/16 ≈ 0.1654"
        );
        let expected_solid: Vec<bool> = (0..14).map(|i| i == 0 || i == 7).collect();
        assert_eq!(
            r.is_solid, expected_solid,
            "with n_threshold 1 the solid particles are exactly the two identical centres"
        );
    }

    #[test]
    fn deterministic_across_calls() {
        let frame = paired_octahedra(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let sl = SolidLiquid::new(6);
        let a = sl.compute(&[&frame], std::slice::from_ref(&nl)).unwrap();
        let b = sl.compute(&[&frame], &[nl]).unwrap();
        assert_eq!(a[0].n_solid_bonds, b[0].n_solid_bonds);
        assert_eq!(a[0].is_solid, b[0].is_solid);
    }

    #[test]
    fn empty_frames_is_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = SolidLiquid::new(6)
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn high_threshold_makes_nothing_solid() {
        let frame = paired_octahedra(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &SolidLiquid::new(6)
            .with_q_threshold(2.0) // impossible — cosine ≤ 1
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        assert!(r.is_solid.iter().all(|&s| !s));
    }
}
