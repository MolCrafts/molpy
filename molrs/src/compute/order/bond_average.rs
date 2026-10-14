//! The per-particle bond-harmonic average — the one engine behind every
//! bond-orientational order parameter.
//!
//! For each particle `i` with neighbours `j`, the average
//!
//! ```text
//!   q_c(i) = (1/N_i) Σ_{j ∈ neigh(i)} B_c(r̂_ij)
//! ```
//!
//! of a [`BondBasis`] `B` evaluated on the bond directions is the table every
//! order parameter reads: the 3-D spherical harmonics `Y_ℓm` give Steinhardt's
//! `q_ℓm` (Steinhardt, Nelson & Ronchetti, *Phys. Rev. B* **28**, 784 (1983)),
//! the planar harmonic `e^{ikθ}` gives the 2-D `ψ_k`. [`BondAverage`] builds it
//! from a half-shell neighbour table; [`QlmTable`] holds it and derives the
//! rotational invariants and the bond coherence the solid/liquid kernels read.

use molrs::core::{Complex, FOUR_PI, Neighbors, wigner_3j, ylm_all};
use molrs::op::{F, Vec3};
use ndarray::{Array1, Array2, ArrayView2};

use crate::compute::{Compute, ComputeError, require_disp, require_self_query};

/// A family of functions of a bond direction whose neighbour average is an
/// order parameter.
///
/// A basis of order `n` has `n_components` complex components and parity
/// `(−1)^n` under bond reversal — which is what lets [`BondAverage`] visit a
/// half-shell pair once and credit both of its particles.
pub(crate) trait BondBasis {
    /// The order: `ℓ` of `Y_ℓm`, `k` of `e^{ikθ}`.
    fn order(&self) -> u32;

    /// Components per particle: `2ℓ + 1` for `Y_ℓm`, 1 for `e^{ikθ}`.
    fn n_components(&self) -> usize;

    /// Whether the basis is planar (`e^{ikθ}`) rather than spherical.
    fn planar(&self) -> bool;

    /// Evaluate the basis on the bond `disp` (Å, not normalised) into `out`
    /// (`n_components` long). `false` when the bond has no direction in this
    /// basis (zero length, or zero in-plane length); `out` is then untouched.
    fn evaluate(&self, disp: Vec3, out: &mut [Complex]) -> bool;

    /// `(−1)^order`: the factor a component picks up when the bond reverses.
    fn parity(&self) -> F {
        if self.order() & 1 == 0 { 1.0 } else { -1.0 }
    }
}

/// The 3-D spherical harmonics `Y_ℓm`, `m = −ℓ … ℓ`, of the bond direction
/// (Condon–Shortley phase, physics normalisation; [`ylm_all`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SphericalHarmonic {
    /// Degree `ℓ`.
    pub l: u32,
}

impl SphericalHarmonic {
    /// The polar and azimuthal angles `(θ, φ)` of a bond (radians):
    /// `θ = arccos(dz / r) ∈ [0, π]`, `φ = atan2(dy, dx) ∈ [−π, π]`. `None`
    /// for a zero-length bond, which has no direction.
    pub(crate) fn angles(disp: Vec3) -> Option<(F, F)> {
        let [dx, dy, dz] = disp;
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        if r == 0.0 {
            return None;
        }
        Some(((dz / r).clamp(-1.0, 1.0).acos(), dy.atan2(dx)))
    }
}

impl BondBasis for SphericalHarmonic {
    fn order(&self) -> u32 {
        self.l
    }

    fn n_components(&self) -> usize {
        (2 * self.l + 1) as usize
    }

    fn planar(&self) -> bool {
        false
    }

    fn evaluate(&self, disp: Vec3, out: &mut [Complex]) -> bool {
        let Some((theta, phi)) = Self::angles(disp) else {
            return false;
        };
        ylm_all(self.l, theta, phi, out);
        true
    }
}

/// The planar harmonic `e^{ikθ}` of the in-plane bond angle
/// `θ = atan2(dy, dx)`; the bond's `z` component is ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PlanarHarmonic {
    /// Rotational symmetry `k`.
    pub k: u32,
}

impl BondBasis for PlanarHarmonic {
    fn order(&self) -> u32 {
        self.k
    }

    fn n_components(&self) -> usize {
        1
    }

    fn planar(&self) -> bool {
        true
    }

    fn evaluate(&self, disp: Vec3, out: &mut [Complex]) -> bool {
        let [dx, dy, _] = disp;
        if dx == 0.0 && dy == 0.0 {
            return false;
        }
        out[0] = Complex::from_polar(1.0, self.k as F * dy.atan2(dx));
        true
    }
}

/// The neighbour average of a [`BondBasis`] over one snapshot's half-shell
/// table: the one implementation of `q_ℓm` / `ψ_k`.
///
/// Each row of a [`SelfQuery`](molrs::core::QueryMode::SelfQuery) table is
/// visited once and credited to both of its particles — to `j` with the
/// basis parity, since from `j` the bond points the other way. A particle with
/// no neighbour (or only zero-length bonds) has an all-zero row.
#[derive(Debug, Clone, Copy)]
pub(crate) struct BondAverage<B> {
    basis: B,
}

impl<B: BondBasis> BondAverage<B> {
    /// The average of `basis`.
    pub(crate) fn new(basis: B) -> Self {
        Self { basis }
    }
}

impl<B: BondBasis> Compute for BondAverage<B> {
    type Input<'a> = &'a Neighbors;
    type Output = QlmTable;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp` or a cross-query
    /// table (it would credit every bond twice); [`ComputeError::OutOfRange`]
    /// for a pair index past the table's point count.
    fn compute(&self, nlist: &Neighbors) -> Result<QlmTable, ComputeError> {
        let disp = require_disp(nlist)?;
        let n = require_self_query(nlist)?;
        let m_count = self.basis.n_components();
        let parity = self.basis.parity();

        let mut values = Array2::from_elem((n, m_count), Complex::ZERO);
        let mut count = vec![0_u32; n];
        let mut buf = vec![Complex::ZERO; m_count];
        for (k, (&i, &j)) in nlist
            .query_point_indices()
            .iter()
            .zip(nlist.point_indices())
            .enumerate()
        {
            let (i, j) = pair_index(i, j, n, k)?;
            if !self
                .basis
                .evaluate([disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]], &mut buf)
            {
                continue;
            }
            for (m, &b) in buf.iter().enumerate() {
                values[[i, m]] += b;
                values[[j, m]] += b.scale(parity);
            }
            count[i] += 1;
            count[j] += 1;
        }
        for (mut row, &c) in values.outer_iter_mut().zip(&count) {
            if c > 0 {
                let inv = 1.0 / c as F;
                row.mapv_inplace(|v| v.scale(inv));
            }
        }
        Ok(QlmTable {
            order: self.basis.order(),
            planar: self.basis.planar(),
            values,
        })
    }
}

/// `(i, j)` of pair `k` as particle indices below `n`.
fn pair_index(i: u32, j: u32, n: usize, k: usize) -> Result<(usize, usize), ComputeError> {
    let (i, j) = (i as usize, j as usize);
    if i >= n || j >= n {
        return Err(ComputeError::OutOfRange {
            field: "pair index",
            value: format!("({i}, {j}) of pair {k} past the table's {n} points"),
        });
    }
    Ok((i, j))
}

/// Per-particle bond-harmonic averages of one order: Steinhardt's `q_ℓm`
/// (`2ℓ + 1` components, column `m + ℓ`) or the planar `ψ_k` (one component).
///
/// Produced by [`Steinhardt`](crate::compute::Steinhardt) (through
/// [`SteinhardtResult::qlm`](crate::compute::SteinhardtResult::qlm)) and read
/// by [`SolidLiquid`](crate::compute::SolidLiquid) and
/// [`ContinuousCoordination`](crate::compute::ContinuousCoordination), so the
/// harmonics are evaluated once. Values are dimensionless.
#[derive(Debug, Clone, PartialEq)]
pub struct QlmTable {
    order: u32,
    planar: bool,
    values: Array2<Complex>,
}

impl QlmTable {
    /// The order: `ℓ` of a spherical table, `k` of a planar one.
    pub fn order(&self) -> u32 {
        self.order
    }

    /// Whether the table holds the planar `ψ_k` rather than `q_ℓm`.
    pub fn is_planar(&self) -> bool {
        self.planar
    }

    /// Number of particles (rows).
    pub fn n_particles(&self) -> usize {
        self.values.nrows()
    }

    /// The averages, `n_particles × n_components`; a spherical table's column
    /// `m + ℓ` holds `q_ℓm`.
    pub fn values(&self) -> ArrayView2<'_, Complex> {
        self.values.view()
    }

    /// Per particle, the norm `√(Σ_m |q_ℓm|²)`.
    pub fn norms(&self) -> Array1<F> {
        self.values
            .outer_iter()
            .map(|row| row.iter().map(|q| q.norm_sqr()).sum::<F>().sqrt())
            .collect()
    }

    /// Per particle, the rotational invariant
    /// `q_ℓ = √( (4π/(2ℓ+1)) Σ_m |q_ℓm|² )`; on a planar table, `|ψ_k|`.
    pub fn ql(&self) -> Array1<F> {
        let pref = if self.planar {
            1.0
        } else {
            FOUR_PI / (2.0 * self.order as F + 1.0)
        };
        self.values
            .outer_iter()
            .map(|row| (pref * row.iter().map(|q| q.norm_sqr()).sum::<F>()).sqrt())
            .collect()
    }

    /// Per particle, the third-order invariant
    /// `w_ℓ = Σ_{m1+m2+m3=0} (ℓ ℓ ℓ; m1 m2 m3) q_ℓm1 q_ℓm2 q_ℓm3`, and with
    /// `normalize` divided by `(Σ_m |q_ℓm|²)^{3/2}` (0 for a particle whose
    /// `q_ℓm` all vanish).
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] on a planar table: `w_ℓ` is defined
    /// through the 3-D Wigner 3-j symbols.
    pub fn wl(&self, normalize: bool) -> Result<Array1<F>, ComputeError> {
        if self.planar {
            return Err(ComputeError::OutOfRange {
                field: "QlmTable::wl",
                value: format!("planar table of order {}; w_ℓ needs q_ℓm", self.order),
            });
        }
        let l = self.order;
        let l_i32 = l as i32;
        // The triples (m1, m2, m3 = −m1−m2) with their 3-j coefficient, so the
        // per-particle loop is complex products and one real factor.
        let mut triples: Vec<(usize, usize, usize, F)> = Vec::new();
        for m1 in -l_i32..=l_i32 {
            for m2 in -l_i32..=l_i32 {
                let m3 = -m1 - m2;
                if m3.abs() > l_i32 {
                    continue;
                }
                let w = wigner_3j(l, l, l, m1, m2, m3);
                if w == 0.0 {
                    continue;
                }
                triples.push((
                    (m1 + l_i32) as usize,
                    (m2 + l_i32) as usize,
                    (m3 + l_i32) as usize,
                    w,
                ));
            }
        }
        Ok(self
            .values
            .outer_iter()
            .map(|q| {
                let wl: F = triples
                    .iter()
                    .map(|&(a, b, c, w)| w * (q[a] * q[b] * q[c]).re)
                    .sum();
                if !normalize {
                    return wl;
                }
                let denom = q.iter().map(|v| v.norm_sqr()).sum::<F>().powf(1.5);
                if denom > 0.0 { wl / denom } else { 0.0 }
            })
            .collect())
    }

    /// Per pair of `neighbors`, in table row order, the bond coherence
    /// `Re Σ_m conj(q_ℓm(i)) · q_ℓm(j)`. Dividing it by
    /// `norms[i] · norms[j]` gives the cosine similarity of the two
    /// environments.
    ///
    /// `neighbors` must be a half-shell table over the same particles.
    ///
    /// # Errors
    /// [`ComputeError::BadShape`] for a cross-query table;
    /// [`ComputeError::DimensionMismatch`] when its point count is not the
    /// table's; [`ComputeError::OutOfRange`] for a pair index past it.
    pub fn bond_coherence(&self, neighbors: &Neighbors) -> Result<Array1<F>, ComputeError> {
        let n = require_self_query(neighbors)?;
        if n != self.n_particles() {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_particles(),
                got: n,
                what: "neighbor-table points vs q_lm table particles",
            });
        }
        neighbors
            .query_point_indices()
            .iter()
            .zip(neighbors.point_indices())
            .enumerate()
            .map(|(k, (&i, &j))| {
                let (i, j) = pair_index(i, j, n, k)?;
                let mut dot = Complex::ZERO;
                for (a, b) in self.values.row(i).iter().zip(self.values.row(j)) {
                    dot += a.conj() * *b;
                }
                Ok(dot.re)
            })
            .collect()
    }

    /// The Lechner–Dellago neighbour average
    /// `q̄(i) = (q(i) + Σ_{j ∈ neigh(i)} q(j)) / (N_i + 1)` over the
    /// half-shell `neighbors` the table was built from.
    ///
    /// # Errors
    /// As [`bond_coherence`](Self::bond_coherence).
    pub(crate) fn neighbor_average(&self, neighbors: &Neighbors) -> Result<QlmTable, ComputeError> {
        let n = require_self_query(neighbors)?;
        if n != self.n_particles() {
            return Err(ComputeError::DimensionMismatch {
                expected: self.n_particles(),
                got: n,
                what: "neighbor-table points vs q_lm table particles",
            });
        }
        let mut acc = self.values.clone();
        let mut count = vec![1_u32; n]; // include self
        for (k, (&i, &j)) in neighbors
            .query_point_indices()
            .iter()
            .zip(neighbors.point_indices())
            .enumerate()
        {
            let (i, j) = pair_index(i, j, n, k)?;
            for m in 0..self.values.ncols() {
                acc[[i, m]] += self.values[[j, m]];
                acc[[j, m]] += self.values[[i, m]];
            }
            count[i] += 1;
            count[j] += 1;
        }
        for (mut row, &c) in acc.outer_iter_mut().zip(&count) {
            let inv = 1.0 / c as F;
            row.mapv_inplace(|v| v.scale(inv));
        }
        Ok(QlmTable {
            order: self.order,
            planar: self.planar,
            values: acc,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use molrs::core::{NeighborColumns, NeighborPair, QueryMode, SimBox};
    use ndarray::array;

    fn open_box(len: F) -> SimBox {
        SimBox::cube(len, array![0.0, 0.0, 0.0], [false; 3]).unwrap()
    }

    fn q6(nl: &Neighbors) -> QlmTable {
        BondAverage::new(SphericalHarmonic { l: 6 })
            .compute(nl)
            .unwrap()
    }

    /// The centre of a lattice cluster: `centre` plus its nearest-neighbour
    /// shell, searched at `cutoff`.
    fn cluster(shell: &[[F; 3]], cutoff: F) -> Neighbors {
        let c = 10.0;
        let mut positions = vec![[c, c, c]];
        positions.extend(shell.iter().map(|d| [c + d[0], c + d[1], c + d[2]]));
        let nl = neighbors_of(&positions, &open_box(20.0), cutoff);
        // Keep only the centre's bonds: the shell–shell pairs would make the
        // shell particles' environments differ from the lattice's.
        let pairs: Vec<NeighborPair> = nl
            .query_point_indices()
            .iter()
            .zip(nl.point_indices())
            .enumerate()
            .filter(|(_, (i, _))| **i == 0)
            .map(|(k, (&i, &j))| NeighborPair {
                i,
                j,
                dist_sq: nl.dist_sq().unwrap()[k],
                disp: [
                    nl.disp().unwrap()[[k, 0]],
                    nl.disp().unwrap()[[k, 1]],
                    nl.disp().unwrap()[[k, 2]],
                ],
            })
            .collect();
        Neighbors::from_pairs(
            pairs,
            NeighborColumns::FULL,
            QueryMode::SelfQuery {
                n_points: positions.len(),
            },
        )
    }

    /// Domain goldens (Steinhardt, Nelson & Ronchetti 1983, Table I; carried
    /// over from the Steinhardt tests): the 12-neighbour FCC shell has
    /// `q_4 = 0.19094`, `q_6 = 0.57452`; the 8-neighbour BCC shell
    /// `q_4 = 0.50918`, `q_6 = 0.62854`; the 6-neighbour simple-cubic shell
    /// `q_4 = √(7/12)`, `q_6 = 1/(2√2)` (exact, by the addition theorem).
    #[test]
    fn lattice_shells_give_the_tabulated_invariants() {
        let fcc: Vec<[F; 3]> = [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, 1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, 1.0],
            [0.0, -1.0, -1.0],
        ]
        .to_vec();
        let mut bcc = Vec::new();
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                for sz in [-1.0, 1.0] {
                    bcc.push([sx, sy, sz]);
                }
            }
        }
        let sc: Vec<[F; 3]> = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
        .to_vec();

        let q = |shell: &[[F; 3]], cutoff: F, l: u32| {
            BondAverage::new(SphericalHarmonic { l })
                .compute(&cluster(shell, cutoff))
                .unwrap()
                .ql()[0]
        };
        for (got, want, tol) in [
            (q(&fcc, 1.5, 4), 0.19094, 1e-5),
            (q(&fcc, 1.5, 6), 0.57452, 1e-5),
            (q(&bcc, 1.8, 4), 0.50918, 1e-5),
            (q(&bcc, 1.8, 6), 0.62854, 1e-5),
            (q(&sc, 1.2, 4), 0.7637626158259734, 1e-12),
            (q(&sc, 1.2, 6), 0.3535533905932738, 1e-12),
        ] {
            assert!((got - want).abs() < tol, "got {got}, want {want}");
        }
    }

    /// `ψ_k` of a regular `k`-gon is a unit phase; the planar table's
    /// invariant is its modulus.
    #[test]
    fn planar_average_of_a_regular_hexagon_is_unit() {
        let mut positions = vec![[10.0, 10.0, 0.0]];
        for k in 0..6 {
            let t = std::f64::consts::TAU * k as F / 6.0;
            positions.push([10.0 + t.cos(), 10.0 + t.sin(), 0.0]);
        }
        let simbox =
            SimBox::ortho(array![20.0, 20.0, 1.0], array![0.0, 0.0, 0.0], [false; 3]).unwrap();
        let nl = neighbors_of(&positions, &simbox, 1.2);
        let table = BondAverage::new(PlanarHarmonic { k: 6 })
            .compute(&nl)
            .unwrap();
        assert!(table.is_planar());
        assert_eq!(table.values().ncols(), 1);
        assert!((table.ql()[0] - 1.0).abs() < 1e-10);
        assert!(matches!(
            table.wl(false),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// Two identical environments are perfectly coherent: the bond
    /// coherence over their norms is 1.
    #[test]
    fn coherence_of_a_particle_with_itself_is_its_squared_norm() {
        let nl = neighbors_of(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], &open_box(20.0), 1.5);
        let table = q6(&nl);
        let coherence = table.bond_coherence(&nl).unwrap();
        let norms = table.norms();
        assert_eq!(coherence.len(), 1);
        // ℓ even: both ends hold the same q_6m, so the cosine is 1.
        assert!((coherence[0] / (norms[0] * norms[1]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn coherence_over_another_point_set_is_a_dimension_mismatch() {
        let nl = neighbors_of(&[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0]], &open_box(20.0), 1.5);
        let other = neighbors_of(
            &[[5.0, 5.0, 5.0], [6.0, 5.0, 5.0], [7.0, 5.0, 5.0]],
            &open_box(20.0),
            1.5,
        );
        assert!(matches!(
            q6(&nl).bond_coherence(&other),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    /// A pair index past the table's declared point count is an error, not
    /// an out-of-bounds panic.
    #[test]
    fn a_pair_past_the_point_count_is_out_of_range() {
        let nl = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 5,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 2 },
        );
        assert!(matches!(
            BondAverage::new(SphericalHarmonic { l: 6 }).compute(&nl),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
