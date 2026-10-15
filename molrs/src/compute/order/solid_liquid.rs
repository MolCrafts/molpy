//! Frenkel–ten Wolde solid/liquid classification from `q_ℓm` bond correlations.

use molrs::core::Neighbors;
use molrs::op::F;

use super::bond_average::QlmTable;
use crate::compute::{Compute, ComputeError};

/// Frenkel-ten Wolde solid/liquid classifier.
///
/// Mirrors `freud.order.SolidLiquid`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/SolidLiquid.cc)).
///
/// Configuration: the two thresholds and the normalisation. Data: a
/// [`QlmTable`] (from [`SteinhardtResult::qlm`](crate::compute::SteinhardtResult::qlm))
/// and the half-shell neighbor table whose bonds are classified — the `q_ℓm`
/// are read, never recomputed.
///
/// For each neighbor pair `(i, j)` we compute the normalised dot product
/// of their Steinhardt qℓm vectors (the table's
/// [`bond_coherence`](QlmTable::bond_coherence) over its
/// [`norms`](QlmTable::norms)):
///
/// ```text
///   d_ij = ( Σ_m q_ℓm(i) · conj q_ℓm(j) ) / ( |q_ℓm(i)| · |q_ℓm(j)| )
/// ```
///
/// A bond is **solid-like** when `Re(d_ij) > q_threshold` (typically `0.7`).
/// A particle is **solid** when it has at least `n_threshold` solid-like
/// bonds. The output is a per-particle solid-bond count plus the boolean
/// solid mask. A pair with a particle whose `q_ℓm` all vanish scores 0.
#[derive(Debug, Clone, Copy)]
pub struct SolidLiquid {
    q_threshold: F,
    n_threshold: u32,
    normalize_q: bool,
}

/// The data a [`SolidLiquid`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct SolidLiquidInput<'a> {
    /// The particles' `q_ℓm`.
    pub qlm: &'a QlmTable,
    /// The bonds to classify: a half-shell self-query over the same particles.
    pub neighbors: &'a Neighbors,
}

impl Default for SolidLiquid {
    fn default() -> Self {
        Self::new()
    }
}

impl SolidLiquid {
    /// Defaults: q_threshold 0.7, n_threshold 6, normalized `q_ℓm`.
    pub fn new() -> Self {
        Self {
            q_threshold: 0.7,
            n_threshold: 6,
            normalize_q: true,
        }
    }

    /// Dot-product threshold for a solid-like bond (default 0.7).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a NaN threshold, which no bond could
    /// ever be compared against.
    pub fn with_q_threshold(mut self, t: F) -> Result<Self, ComputeError> {
        if t.is_nan() {
            return Err(ComputeError::OutOfRange {
                field: "SolidLiquid::q_threshold",
                value: t.to_string(),
            });
        }
        self.q_threshold = t;
        Ok(self)
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
}

impl Compute for SolidLiquid {
    type Input<'a> = SolidLiquidInput<'a>;
    type Output = SolidLiquidResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a cross-query table;
    /// [`ComputeError::DimensionMismatch`] when the table is over a different
    /// number of particles than `qlm`.
    fn compute(&self, input: SolidLiquidInput<'_>) -> Result<SolidLiquidResult, ComputeError> {
        let qlm = input.qlm;
        let nlist = input.neighbors;
        let coherence = qlm.bond_coherence(nlist)?;
        let norms = qlm.norms();
        let mut n_solid_bonds = vec![0_u32; qlm.n_particles()];
        for ((&i, &j), &dot) in nlist
            .query_point_indices()
            .iter()
            .zip(nlist.point_indices())
            .zip(&coherence)
        {
            let (i, j) = (i as usize, j as usize);
            let real = if self.normalize_q {
                let denom = norms[i] * norms[j];
                if denom > 0.0 { dot / denom } else { 0.0 }
            } else {
                dot
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
            l: qlm.order(),
            n_solid_bonds,
            is_solid,
        })
    }
}

/// One snapshot's solid/liquid classification.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct SolidLiquidResult {
    /// ℓ of the `q_ℓm` the bonds were scored with.
    pub l: u32,
    /// Solid-like bond count per particle.
    pub n_solid_bonds: Vec<u32>,
    /// `true` if the particle has ≥ `n_threshold` solid-like bonds.
    pub is_solid: Vec<bool>,
}

crate::compute::spec::compute_spec! {
    SolidLiquid as "SolidLiquid";
    params |s| {
        q_threshold: Float = SolidLiquid::new().q_threshold => s.q_threshold;
        n_threshold: Int = SolidLiquid::new().n_threshold => s.n_threshold;
        normalize_q: Bool = SolidLiquid::new().normalize_q => s.normalize_q;
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

    /// A snapshot: positions in a cube of side `box_len` at the origin.
    struct Snapshot {
        positions: Vec<[F; 3]>,
        simbox: SimBox,
    }

    fn frame_with(positions: &[[F; 3]], box_len: F, pbc: [bool; 3]) -> Snapshot {
        Snapshot {
            positions: positions.to_vec(),
            simbox: SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], pbc).unwrap(),
        }
    }

    fn nlist_from_frame(frame: &Snapshot, cutoff: F) -> Neighbors {
        neighbors_of(&frame.positions, &frame.simbox, cutoff)
    }

    fn q6(nl: &Neighbors) -> QlmTable {
        BondAverage::new(SphericalHarmonic { l: 6 })
            .compute(nl)
            .unwrap()
    }

    fn classify(sl: SolidLiquid, nl: &Neighbors) -> SolidLiquidResult {
        sl.compute(SolidLiquidInput {
            qlm: &q6(nl),
            neighbors: nl,
        })
        .unwrap()
    }

    /// Two octahedra sharing the same orientation: every neighbor pair across
    /// the symmetry-mate has qℓm dot ≈ 1 (identical environments).
    fn paired_octahedra(box_len: F) -> Snapshot {
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
        let r = &classify(
            SolidLiquid::new()
                .with_q_threshold(-2.0) // count every bond as "solid"
                .unwrap()
                .with_n_threshold(1),
            &nl,
        );
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
        let qlm = q6(&nl);
        let m = 13_usize; // 2·6 + 1
        for k in 0..m {
            let a = qlm.values()[[0, k]];
            let b = qlm.values()[[7, k]];
            assert!(
                (a.re - b.re).abs() < 1e-12 && (a.im - b.im).abs() < 1e-12,
                "centre q₆ₘ components must match across symmetric octahedra"
            );
        }

        // Default q_threshold (0.7) keeps only the centre-centre bond;
        // n_threshold 1 then makes the two centres — and nothing else — solid.
        let r = &classify(SolidLiquid::new().with_n_threshold(1), &nl);
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
        let sl = SolidLiquid::new();
        let a = classify(sl, &nl);
        let b = classify(sl, &nl);
        assert_eq!(a.n_solid_bonds, b.n_solid_bonds);
        assert_eq!(a.is_solid, b.is_solid);
    }

    #[test]
    fn nan_threshold_is_out_of_range() {
        assert!(matches!(
            SolidLiquid::new().with_q_threshold(F::NAN),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// The q_ℓm table and the bond table must describe the same particles.
    #[test]
    fn a_table_over_other_particles_is_a_dimension_mismatch() {
        let frame = paired_octahedra(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let small = nlist_from_frame(&frame_with(&[[5.0, 5.0, 5.0]], 20.0, [false; 3]), 1.2);
        let err = SolidLiquid::new()
            .compute(SolidLiquidInput {
                qlm: &q6(&small),
                neighbors: &nl,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::DimensionMismatch { .. }));
    }

    #[test]
    fn high_threshold_makes_nothing_solid() {
        let frame = paired_octahedra(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &classify(
            SolidLiquid::new()
                .with_q_threshold(2.0) // impossible — cosine ≤ 1
                .unwrap(),
            &nl,
        );
        assert!(r.is_solid.iter().all(|&s| !s));
    }
}
