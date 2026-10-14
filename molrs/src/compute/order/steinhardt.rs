//! Steinhardt bond-orientational order parameters `q_ℓ` and `w_ℓ`.

use molrs::core::Neighbors;
use molrs::op::F;
use ndarray::{Array2, ArrayView1, ArrayView2, s};

use super::bond_average::{BondAverage, QlmTable, SphericalHarmonic};
use crate::compute::{Compute, ComputeError, FeatureMatrix};

/// Steinhardt order-parameter calculator.
///
/// Configuration: the ℓ values and the variant flags. Data: one snapshot's
/// half-shell neighbor table, carrying the `disp` column (Å); see *Required
/// neighbor columns* below. Many snapshots go through
/// [`Batch`](crate::compute::Batch).
///
/// Mirrors `freud.order.Steinhardt`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/order/Steinhardt.cc)).
/// Implements:
///
/// - per-particle `q_ℓm(i) = (1/N_i) Σ_{j ∈ neigh(i)} Y_ℓm(r̂_ij)` — a
///   [`QlmTable`], the one bond-harmonic average
/// - the **averaged** variant (``average = true``) `q̄_ℓm(i) = (1/(N_i+1))
///   (q_ℓm(i) + Σ_{j ∈ neigh(i)} q_ℓm(j))` — the "near-shell" Steinhardt
/// - the rotational invariant
///   `q_ℓ(i) = √( (4π/(2ℓ+1)) Σ_m |q_ℓm(i)|² )`
/// - the cubic invariant
///   `w_ℓ(i) = Σ_{m1+m2+m3=0} (ℓ ℓ ℓ; m1 m2 m3) q_ℓm1(i) q_ℓm2(i) q_ℓm3(i)`
///   with an optional normalization
///   `ŵ_ℓ(i) = w_ℓ(i) / ( Σ_m |q_ℓm(i)|² )^{3/2}`.
///
/// # Conventions
///
/// - Self-query [`Neighbors`]: each pair `(i, j)` with `i < j` carries the
///   vector `r_j − r_i`. The average visits each pair once and updates both
///   particles, exploiting `Y_ℓm(−r̂) = (−1)^ℓ Y_ℓm(r̂)`. The particle count
///   is the table's `SelfQuery { n_points }`.
/// - `Y_ℓm` follows the Condon-Shortley + physics-normalization convention
///   (see [`molrs::core::ylm_all`]).
///
/// # Required neighbor columns
///
/// `q_ℓm` is built from bond *directions*, so the table must carry the
/// minimum-image displacement column `disp` (Å) — materialize it with
/// [`NeighborColumns::DISP`](molrs::core::NeighborColumns::DISP)
/// or [`FULL`](molrs::core::NeighborColumns::FULL). Distances
/// alone are not enough: a `DIST_SQ` or `INDICES_ONLY` table stores no
/// directions, and reads back `None` rather than zeros, so `compute` answers
/// [`ComputeError::BadShape`] naming the missing column instead of indexing an
/// empty view. A cross-query table is `BadShape` too: it would credit every
/// bond twice.
///
/// # References
///
/// - Steinhardt, Nelson & Ronchetti, *Phys. Rev. B* **28**, 784 (1983).
/// - Lechner & Dellago, *J. Chem. Phys.* **129**, 114707 (2008) — averaged
///   variant.
#[derive(Debug, Clone)]
pub struct Steinhardt {
    l: Vec<u32>,
    average: bool,
    wl: bool,
    wl_normalize: bool,
}

/// The data a [`Steinhardt`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct SteinhardtInput<'a> {
    /// One snapshot's half-shell self-query table with the `disp` column.
    pub neighbors: &'a Neighbors,
}

impl Steinhardt {
    /// Build a calculator for the listed ℓ values (must be non-empty).
    ///
    /// ℓ is the degree of the spherical harmonic the parameter is built from —
    /// dimensionless, and conventionally 4 or 6, the degrees that distinguish
    /// cubic and hexagonal close-packed environments most sharply.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] when `l` is empty: there would be no
    /// parameter to compute.
    pub fn new(l: &[u32]) -> Result<Self, ComputeError> {
        if l.is_empty() {
            return Err(ComputeError::OutOfRange {
                field: "Steinhardt::l",
                value: "[]".into(),
            });
        }
        Ok(Self {
            l: l.to_vec(),
            average: false,
            wl: false,
            wl_normalize: false,
        })
    }

    /// Enable Lechner-Dellago averaged variant (`q̄_ℓm`).
    pub fn with_average(mut self, on: bool) -> Self {
        self.average = on;
        self
    }

    /// Also compute the third-order invariant `w_ℓ`.
    pub fn with_wl(mut self, on: bool) -> Self {
        self.wl = on;
        self
    }

    /// Normalize `w_ℓ` by `( Σ_m |q_ℓm|² )^{3/2}`.
    pub fn with_wl_normalize(mut self, on: bool) -> Self {
        self.wl_normalize = on;
        self
    }

    /// The configured ℓ values, in the order they were requested — the same
    /// order every per-ℓ column of [`SteinhardtResult`] is in.
    pub fn l(&self) -> &[u32] {
        &self.l
    }
}

impl Compute for Steinhardt {
    type Input<'a> = SteinhardtInput<'a>;
    type Output = SteinhardtResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp` or a cross-query
    /// table.
    fn compute(&self, input: SteinhardtInput<'_>) -> Result<SteinhardtResult, ComputeError> {
        let nlist = input.neighbors;
        let mut tables = Vec::with_capacity(self.l.len());
        for &l in &self.l {
            let raw = BondAverage::new(SphericalHarmonic { l }).compute(nlist)?;
            tables.push(if self.average {
                raw.neighbor_average(nlist)?
            } else {
                raw
            });
        }
        let n = tables[0].n_particles();
        let n_l = self.l.len();
        let n_cols = if self.wl { 2 * n_l } else { n_l };
        let mut features = Array2::<F>::zeros((n, n_cols));
        for (c, table) in tables.iter().enumerate() {
            features.column_mut(c).assign(&table.ql());
            if self.wl {
                features
                    .column_mut(n_l + c)
                    .assign(&table.wl(self.wl_normalize)?);
            }
        }
        Ok(SteinhardtResult {
            l: self.l.clone(),
            tables,
            features,
        })
    }
}

/// One snapshot's Steinhardt parameters for one or more ℓ values.
///
/// As a [`FeatureMatrix`] it is one row per particle: the `q_ℓ` columns in
/// the order of [`l`](Self::l), followed by the `w_ℓ` columns when they were
/// requested.
#[derive(Debug, Clone)]
pub struct SteinhardtResult {
    /// ℓ values, in the order requested.
    pub l: Vec<u32>,
    /// The `q_ℓm` table per ℓ (averaged when the variant is on), parallel to
    /// `l`.
    pub tables: Vec<QlmTable>,
    /// `n_particles × (n_l [+ n_l])`: `q_ℓ`, then `w_ℓ`.
    features: Array2<F>,
}

impl SteinhardtResult {
    /// The `q_ℓm` table of degree `l`; `None` when `l` was not configured.
    /// This is the table [`SolidLiquid`](crate::compute::SolidLiquid) and
    /// [`ContinuousCoordination`](crate::compute::ContinuousCoordination)
    /// read, so the harmonics are evaluated once.
    pub fn qlm(&self, l: u32) -> Option<&QlmTable> {
        self.l
            .iter()
            .position(|&have| have == l)
            .map(|k| &self.tables[k])
    }

    /// `q_ℓ`, `n_particles × n_l` (column `k` is ℓ = `l[k]`).
    pub fn ql(&self) -> ArrayView2<'_, F> {
        self.features.slice(s![.., ..self.l.len()])
    }

    /// `w_ℓ`, `n_particles × n_l`; `None` unless
    /// [`Steinhardt::with_wl`] was set.
    pub fn wl(&self) -> Option<ArrayView2<'_, F>> {
        (self.features.ncols() > self.l.len()).then(|| self.features.slice(s![.., self.l.len()..]))
    }
}

impl FeatureMatrix for SteinhardtResult {
    fn n_rows(&self) -> usize {
        self.features.nrows()
    }

    fn n_cols(&self) -> usize {
        self.features.ncols()
    }

    fn row(&self, row: usize) -> ArrayView1<'_, F> {
        self.features.row(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Batch;
    use crate::compute::fixtures::neighbors_of;
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

    fn run(s: &Steinhardt, nl: &Neighbors) -> SteinhardtResult {
        s.compute(SteinhardtInput { neighbors: nl }).unwrap()
    }

    // -- 1) Trivial single-pair --------------------------------------------------

    #[test]
    fn ql_isolated_particle_is_zero() {
        // Single particle: no neighbors → q_ℓ = 0
        let frame = frame_with(&[[5.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = nlist_from_frame(&frame, 1.0);
        let s = Steinhardt::new(&[4, 6]).unwrap();
        let r = run(&s, &nl);
        assert_eq!(r.ql()[[0, 0]], 0.0);
        assert_eq!(r.ql()[[0, 1]], 0.0);
    }

    // -- 2) Identical environments must give identical q_ℓ ----------------------
    //
    // For an FCC lattice's 12-coordinate environment, every particle has the same
    // q_ℓ values. We don't construct a full lattice here; instead, two particles
    // with identical *bond* distributions (a regular octahedron of neighbors
    // around each) suffice.

    /// Build an octahedron-coordinated central particle: 6 neighbors at
    /// distance 1 along ±x, ±y, ±z.
    fn octahedron(box_len: F) -> Snapshot {
        let c = box_len * 0.5;
        let positions = [
            [c, c, c],
            [c + 1.0, c, c],
            [c - 1.0, c, c],
            [c, c + 1.0, c],
            [c, c - 1.0, c],
            [c, c, c + 1.0],
            [c, c, c - 1.0],
        ];
        frame_with(&positions, box_len, [false; 3])
    }

    #[test]
    fn q6_octahedral_environment_finite_and_invariant_to_rotation() {
        // Central particle has 6 neighbors. Compute q_6 at center. Then
        // rotate the same octahedron by π/4 around z and confirm q_6 unchanged.
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);
        let q6_center = r.ql()[[0, 0]];
        assert!(
            q6_center > 0.0,
            "q_6(center) should be > 0; got {q6_center}"
        );
        // Domain golden, analytic — no third-party oracle involved.
        //
        // The addition theorem (DLMF 14.30.9,
        // Σ_m Y_ℓm(â) Y*_ℓm(b̂) = ((2ℓ+1)/4π) P_ℓ(â·b̂)) collapses
        // q_ℓ = √((4π/(2ℓ+1)) Σ_m |q_ℓm|²) into
        // q_ℓ = √( (1/N²) Σ_{a,b} P_ℓ(â·b̂) ) over the N = 6 bond directions.
        // The 36 direction pairs of {±x̂, ±ŷ, ±ẑ} split into
        //   6 parallel      P_6(+1) = 1
        //   6 antiparallel  P_6(−1) = 1        (ℓ even)
        //  24 orthogonal    P_6( 0) = −5/16
        // so Σ = 6 + 6 − 24·(5/16) = 9/2 and q_6 = √(4.5/36) = 1/(2√2).
        // Measured 2026-08-10: 0.3535533905932732 (4 ulp below the analytic
        // value; the 1e-12 window is float slack, not a fudge factor).
        const Q6_OCTAHEDRON: F = 0.3535533905932738; // 1/(2√2)
        assert!(
            (q6_center - Q6_OCTAHEDRON).abs() < 1e-12,
            "q_6(octahedron) must equal the analytic 1/(2√2) = {Q6_OCTAHEDRON}; got {q6_center}"
        );

        // Rotated octahedron: same positions but apply yaw φ=π/4 around z.
        let theta = std::f64::consts::FRAC_PI_4;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let mut positions = vec![[10.0_f64, 10.0, 10.0]];
        for &(dx, dy, dz) in &[
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ] {
            positions.push([
                10.0 + cos_t * dx - sin_t * dy,
                10.0 + sin_t * dx + cos_t * dy,
                10.0 + dz,
            ]);
        }
        let frame2 = frame_with(&positions, 20.0, [false; 3]);
        let nl2 = nlist_from_frame(&frame2, 1.2);
        let r2 = &run(&s, &nl2);
        let q6_rotated = r2.ql()[[0, 0]];
        assert!(
            (q6_rotated - q6_center).abs() < 1e-10,
            "q_6 should be rotation-invariant; got {q6_center} vs {q6_rotated}"
        );
    }

    // -- 3) Multiple ℓ ---------------------------------------------------------

    #[test]
    fn multiple_l_values_independent() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s_solo = Steinhardt::new(&[6]).unwrap();
        let s_pair = Steinhardt::new(&[4, 6]).unwrap();

        let r_solo = &run(&s_solo, &nl);
        let r_pair = &run(&s_pair, &nl);

        assert!((r_solo.ql()[[0, 0]] - r_pair.ql()[[0, 1]]).abs() < 1e-12);
        // q_4 and q_6 in general differ for an octahedron.
        assert!((r_pair.ql()[[0, 0]] - r_pair.ql()[[0, 1]]).abs() > 1e-3);

        // Domain golden for the second ℓ, so "they differ" is anchored on *two*
        // known numbers rather than one free value. Same analytic route as
        // `q6_octahedral_environment_finite_and_invariant_to_rotation`
        // (DLMF 14.30.9 over the 36 direction pairs), with P_4 instead of P_6:
        //   6·P_4(+1) + 6·P_4(−1) + 24·P_4(0) = 6 + 6 + 24·(3/8) = 21
        // so q_4 = √(21/36) = √(7/12) = 0.763762615825973…
        // NB: √(7/3)/4 ≈ 0.66145 is *not* this number — do not "correct" the
        // constant to it. Measured 2026-08-10: 0.7637626158259730.
        const Q4_OCTAHEDRON: F = 0.7637626158259734; // √(7/12)
        const Q6_OCTAHEDRON: F = 0.3535533905932738; // 1/(2√2)
        let q4_center = r_pair.ql()[[0, 0]];
        assert!(
            (q4_center - Q4_OCTAHEDRON).abs() < 1e-12,
            "q_4(octahedron) must equal the analytic √(7/12) = {Q4_OCTAHEDRON}; got {q4_center}"
        );
        let q6_center = r_pair.ql()[[0, 1]];
        assert!(
            (q6_center - Q6_OCTAHEDRON).abs() < 1e-12,
            "q_6(octahedron) must equal the analytic 1/(2√2) = {Q6_OCTAHEDRON}; got {q6_center}"
        );
    }

    // -- 4) w_ℓ third-order invariant ------------------------------------------

    #[test]
    fn wl_present_when_requested() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap().with_wl(true);
        let r = &run(&s, &nl);
        assert!(r.wl().is_some());
        let wl = r.wl().unwrap();
        assert_eq!(wl.ncols(), 1);
        assert_eq!(wl.nrows(), 7);
        // For a perfect octahedral environment w_6 is a definite (nonzero) number.
        assert!(
            wl[[0, 0]].abs() > 1e-6,
            "w_6(octahedron) should be nonzero; got {}",
            wl[[0, 0]]
        );
    }

    #[test]
    fn wl_normalize_scales_into_unit_range() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6])
            .unwrap()
            .with_wl(true)
            .with_wl_normalize(true);
        let r = &run(&s, &nl);
        let wl = r.wl().unwrap();
        // Normalised ŵ_ℓ ∈ [-1, 1] for any ℓ ≥ 1 (Cauchy-Schwarz on triple sum).
        assert!(
            wl[[0, 0]].abs() <= 1.0 + 1e-9,
            "|ŵ_6| should be ≤ 1; got {}",
            wl[[0, 0]]
        );
    }

    #[test]
    fn wl_absent_by_default() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);
        assert!(r.wl().is_none());
    }

    // -- 5) Average variant (Lechner-Dellago) ----------------------------------

    #[test]
    fn average_variant_changes_ql() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s_plain = Steinhardt::new(&[6]).unwrap();
        let s_avg = Steinhardt::new(&[6]).unwrap().with_average(true);
        let r_plain = &run(&s_plain, &nl);
        let r_avg = &run(&s_avg, &nl);
        // Outer-shell particles see different neighborhoods in averaged mode.
        let neighbor_idx = 1;
        assert!(
            (r_plain.ql()[[neighbor_idx, 0]] - r_avg.ql()[[neighbor_idx, 0]]).abs() > 1e-9
                || r_plain.ql()[[neighbor_idx, 0]] == 0.0
        );
    }

    // -- 6) Deterministic --------------------------------------------------------

    #[test]
    fn deterministic_across_calls() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[4, 6]).unwrap().with_wl(true);
        let r1 = run(&s, &nl);
        let r2 = run(&s, &nl);
        for (x, y) in r1.ql().iter().zip(r2.ql().iter()) {
            assert!((x - y).abs() < 1e-15);
        }
    }

    // -- 7) The exposed q_ℓm table -------------------------------------------------

    #[test]
    fn qlm_table_is_the_bond_average() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let qlm_raw = BondAverage::new(SphericalHarmonic { l: 6 })
            .compute(&nl)
            .unwrap();

        // Plain (non-averaged) Steinhardt exposes the same q_lm.
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);
        for (a, b) in qlm_raw
            .values()
            .iter()
            .zip(r.qlm(6).unwrap().values().iter())
        {
            assert!((a.re - b.re).abs() < 1e-14 && (a.im - b.im).abs() < 1e-14);
        }
        assert!(r.qlm(4).is_none(), "ℓ = 4 was not configured");
    }

    /// One row per particle, the q_ℓ then w_ℓ columns.
    #[test]
    fn result_is_a_feature_matrix() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let r = run(&Steinhardt::new(&[4, 6]).unwrap().with_wl(true), &nl);
        let f: &dyn FeatureMatrix = &r;
        assert_eq!((f.n_rows(), f.n_cols()), (7, 4));
        assert_eq!(f.row(0)[1], r.ql()[[0, 1]]);
        assert_eq!(f.row(0)[3], r.wl().unwrap()[[0, 1]]);
    }

    // -- 8) Empty / error paths ------------------------------------------------

    #[test]
    fn empty_l_is_error() {
        assert!(Steinhardt::new(&[]).is_err());
    }

    /// Dropping the `disp` column is a legal downgrade — the table still has
    /// pairs and distances. Steinhardt is a bond-*direction* order parameter,
    /// so it must refuse that table loudly instead of reading zeros.
    #[test]
    fn nlist_without_displacement_vectors_is_error() {
        use molrs::core::NeighborColumns;
        let frame = octahedron(20.0);
        let nl_full = nlist_from_frame(&frame, 1.2);
        assert!(nl_full.n_pairs() > 0);
        // Lean list: distances kept, displacements dropped (a caller that only
        // asked for d², e.g. an RDF-shaped query, reused for an order kernel).
        let nl_lean = nl_full.repack(NeighborColumns::DIST_SQ);
        assert_eq!(nl_lean.n_pairs(), nl_full.n_pairs());
        assert!(
            nl_lean.disp().is_none(),
            "downgrade drops the disp column; it must not fabricate zeros"
        );
        let err = Steinhardt::new(&[6])
            .unwrap()
            .compute(SteinhardtInput {
                neighbors: &nl_lean,
            })
            .unwrap_err();
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "expected BadShape when disp is missing; got {err:?}"
        );
    }

    /// ac-002 (spec neighborlist-03-compute): an **indices-only** table with
    /// real pairs is the strictest missing-column case — neither physical
    /// column was ever stored, so there is nothing to fall back on. Steinhardt
    /// must answer `BadShape` rather than index an empty `disp` view (which
    /// would be an out-of-bounds panic, or a WASM `unreachable`).
    ///
    /// The pairs are the octahedron's own bonds, hard-coded rather than
    /// searched: centre 0 bonded to its six neighbours 1..=6 at unit distance
    /// along ±x, ±y, ±z. Every pair satisfies the half-shell contract `i < j`,
    /// so `SelfQuery { n_points: 7 }` is a legal label for them.
    #[test]
    fn steinhardt_indices_only_neighbors_is_bad_shape() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        let bonds: [[F; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        let pairs: Vec<NeighborPair> = bonds
            .iter()
            .enumerate()
            .map(|(k, disp)| NeighborPair {
                i: 0,
                j: (k + 1) as u32,
                dist_sq: 1.0,
                disp: *disp,
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

        let err = Steinhardt::new(&[6])
            .unwrap()
            .compute(SteinhardtInput { neighbors: &nl })
            .unwrap_err();
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "indices-only Neighbors must be BadShape, not a panic; got {err:?}"
        );
    }

    // -- 9) Multi-frame --------------------------------------------------------

    #[test]
    fn multi_frame_returns_one_result_per_frame() {
        let frame1 = octahedron(20.0);
        let frame2 = octahedron(20.0);
        let nl1 = nlist_from_frame(&frame1, 1.2);
        let nl2 = nlist_from_frame(&frame2, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = Batch(s)
            .compute(vec![
                SteinhardtInput { neighbors: &nl1 },
                SteinhardtInput { neighbors: &nl2 },
            ])
            .unwrap();
        assert_eq!(r.len(), 2);
        // Same frame, same nlist → same q_6 at center
        assert!((r[0].ql()[[0, 0]] - r[1].ql()[[0, 0]]).abs() < 1e-12);
    }

    // -- 10) q_ℓm shape sanity --------------------------------------------------

    #[test]
    fn qlm_shape_is_n_times_2lp1() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);
        assert_eq!(r.tables[0].values().len(), 7 * (2 * 6 + 1));
    }

    // -- 11) Antiparallel pair: parity check -----------------------------------
    //
    // For two particles at ±x_hat (just one pair, antiparallel bonds), the
    // contributions on each particle differ only by (-1)^ℓ. For ℓ=6 (even),
    // both particles must end up with identical q_ℓm — and therefore identical q_6.

    #[test]
    fn parity_two_particle_pair() {
        let frame = frame_with(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = nlist_from_frame(&frame, 1.5);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);
        assert!(
            (r.ql()[[0, 0]] - r.ql()[[1, 0]]).abs() < 1e-12,
            "even-ℓ q_ℓ must be parity-symmetric across an antiparallel pair"
        );
    }

    // -- 12) Antiparallel pair, odd ℓ: parity flip -----------------------------

    #[test]
    fn parity_odd_l_two_particle_pair() {
        // For ℓ=3 (odd), q_ℓm at particle 1 = -q_ℓm at particle 0 → same magnitude.
        let frame = frame_with(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], 10.0, [false; 3]);
        let nl = nlist_from_frame(&frame, 1.5);
        let s = Steinhardt::new(&[3]).unwrap();
        let r = &run(&s, &nl);
        assert!((r.ql()[[0, 0]] - r.ql()[[1, 0]]).abs() < 1e-12);
    }

    // -- 13) Multi-l result struct ordering -----------------------------------

    #[test]
    fn result_l_field_preserves_input_order() {
        let frame = octahedron(20.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let s = Steinhardt::new(&[6, 4, 8]).unwrap();
        let r = &run(&s, &nl);
        assert_eq!(r.l, vec![6, 4, 8]);
    }

    // -- 14) PBC: same lattice in wrapped vs unwrapped boxes gives same q_ℓ ---

    /// Six-coordinate environment centred on the *origin* of a periodic box, so
    /// the neighbors at (±1, 0, 0) etc. straddle the boundary and are only
    /// found through the minimum image.
    ///
    /// Shared by `pbc_consistent_with_open_box`, which reads the centre, and
    /// `parity_visible_at_mixed_role_particle`, which reads the shell.
    fn wrapped_octahedron() -> Snapshot {
        let positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0], // wrapped equivalent of (-1, 0, 0)
            [0.0, 1.0, 0.0],
            [0.0, 9.0, 0.0], // wrapped equivalent of (0, -1, 0)
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 9.0], // wrapped equivalent of (0, 0, -1)
        ];
        frame_with(&positions, 10.0, [true, true, true])
    }

    #[test]
    fn pbc_consistent_with_open_box() {
        let frame = wrapped_octahedron();
        let nl = nlist_from_frame(&frame, 1.5);

        let frame_open = octahedron(20.0);
        let nl_open = nlist_from_frame(&frame_open, 1.2);

        let s = Steinhardt::new(&[6]).unwrap();
        let r_pbc = &run(&s, &nl);
        let r_open = &run(&s, &nl_open);
        assert!(
            (r_pbc.ql()[[0, 0]] - r_open.ql()[[0, 0]]).abs() < 1e-10,
            "q_6 should match between PBC-wrapped and open-box octahedra: got {} vs {}",
            r_pbc.ql()[[0, 0]],
            r_open.ql()[[0, 0]]
        );
    }

    // -- 15) Parity is only observable at a mixed-role particle ----------------

    /// Domain: pin `q_6` where the half-shell parity factor `(-1)^ℓ` can
    /// actually be seen.
    ///
    /// Every other assertion in this module reads a particle that is *only* the
    /// `i` side of its pairs (the octahedron centre) or *only* the `j` side (a
    /// shell atom at cutoff 1.2, whose single bond is to the centre). On such a
    /// particle the parity multiplies every accumulated `Y_ℓm` by the same
    /// factor, and `|(-1)^ℓ| = 1` cancels out of `Σ_m |q_ℓm|²` — so `q_ℓ` is
    /// blind to the sign and those anchors cannot fail on a parity bug.
    ///
    /// The wrapped fixture at cutoff 1.5 fixes that: the second-shell diagonals
    /// (√2 ≈ 1.414 Å < 1.5 Å) put each shell atom on *both* sides of the
    /// half-shell table, with a different split per atom —
    ///
    /// * particle 1 — `j` in (0,1); `i` in (1,3), (1,4), (1,5), (1,6) → 1 / 4
    /// * particle 3 — `j` in (0,3), (1,3), (2,3); `i` in (3,5), (3,6) → 3 / 2
    ///
    /// — so a wrong parity sign weights the two roles differently *and*
    /// differently for the two particles. Their bond sets are exact rotations of
    /// one another (−90° about ẑ maps particle 3's five directions onto
    /// particle 1's), so the rotationally invariant `q_6` must agree.
    ///
    /// Bite-proof (2026-08-10, this fixture, ℓ = 6, correct parity `(-1)^ℓ` =
    /// +1 vs. a sign-flipped parity):
    ///
    /// | parity  | q_6(1)          | q_6(3)          |
    /// |---------|-----------------|-----------------|
    /// | correct | 0.4538033715168 | 0.4538033715168 |
    /// | flipped | 0.5485777064373 | 0.2157834562704 |
    ///
    /// Both assertions below therefore bite, in both directions: the equality
    /// splits by ~0.33 and the pinned value moves by ~0.09, against tolerances
    /// of 1e-12 and 1e-9. The counterfactual row was reproduced offline with an
    /// independent implementation of the same accumulation (scipy
    /// `sph_harm_y`), which also reproduces the correct-parity value — that
    /// cross-check is not part of the gate and nothing here imports it.
    #[test]
    fn parity_visible_at_mixed_role_particle() {
        let frame = wrapped_octahedron();
        let nl = nlist_from_frame(&frame, 1.5);
        let s = Steinhardt::new(&[6]).unwrap();
        let r = &run(&s, &nl);

        let q6_p1 = r.ql()[[1, 0]];
        let q6_p3 = r.ql()[[3, 0]];
        assert!(
            (q6_p1 - q6_p3).abs() < 1e-12,
            "particles 1 and 3 have rotationally equivalent bond sets but opposite \
             half-shell role splits, so q_6 must agree: got {q6_p1} vs {q6_p3}"
        );
        // Golden of the correct-parity accumulation (see the table above).
        const Q6_MIXED_ROLE: F = 0.453803371517;
        assert!(
            (q6_p1 - Q6_MIXED_ROLE).abs() < 1e-9,
            "q_6 at a mixed-role shell particle must be {Q6_MIXED_ROLE}; got {q6_p1}"
        );
    }

    // -- 16) Query mode: a cross-query table is not a Steinhardt input --------

    /// Edge: a **cross-query** [`Neighbors`] table must be refused with
    /// [`ComputeError::BadShape`], not consumed.
    ///
    /// The bond average visits each row once and updates *both* endpoints, using
    /// `Y_ℓm(−r̂) = (−1)^ℓ Y_ℓm(r̂)` for the `j` side. That double update is only
    /// correct for a half-shell [`QueryMode::SelfQuery`], where each unordered
    /// pair appears exactly once. On a full-shell or cross table every pair is
    /// present in both orderings, so:
    ///
    /// * `neighbor_count` is doubled — `q_ℓm` survives by luck only because the
    ///   numerator doubles with it;
    /// * the averaged (Lechner-Dellago) variant does **not** survive:
    ///   the neighbour average counts `1 + 2·N_i` neighbours instead of `1 + N_i`, so
    ///   `q̄_ℓ` is silently wrong rather than absent;
    /// * a genuine cross-query also indexes two *different* point sets, so `j`
    ///   need not even be a particle of `frame`.
    ///
    /// The table below is well-formed in every other respect (FULL storage,
    /// finite displacements, in-range indices), so the mode is the only thing
    /// left to refuse. The guard belongs in the bond average, the one place
    /// `q_ℓm` is built, so every order parameter inherits it.
    #[test]
    fn steinhardt_cross_query_table_is_bad_shape() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        // Same six bonds as `steinhardt_indices_only_neighbors_is_bad_shape`:
        // centre 0 to its neighbours 1..=6 at unit distance along ±x, ±y, ±z.
        let bonds: [[F; 3]; 6] = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        // Full shell: both orderings of every bond, as a cross-query reports
        // them. `disp` stays r_j − r_i, so the reversed row carries −disp.
        let mut pairs: Vec<NeighborPair> = Vec::with_capacity(2 * bonds.len());
        for (k, disp) in bonds.iter().enumerate() {
            let j = (k + 1) as u32;
            pairs.push(NeighborPair {
                i: 0,
                j,
                dist_sq: 1.0,
                disp: *disp,
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

        // Deliberately not `expect_err`: the Ok payload is the whole q_ℓm
        // buffer, and dumping it buries the one thing the failure says.
        let Err(err) = BondAverage::new(SphericalHarmonic { l: 6 }).compute(&nl) else {
            panic!("the bond average must refuse a cross-query table outright, but returned Ok");
        };
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "cross-query Neighbors must be BadShape; got {err:?}"
        );

        let Err(err) = Steinhardt::new(&[6])
            .unwrap()
            .compute(SteinhardtInput { neighbors: &nl })
        else {
            panic!("Steinhardt::compute must inherit the bond average's cross-query guard");
        };
        assert!(
            matches!(err, ComputeError::BadShape { .. }),
            "cross-query Neighbors must be BadShape; got {err:?}"
        );
    }
}
