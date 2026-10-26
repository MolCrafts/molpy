//! Environment matching by neighbor-bond fingerprint, with optional
//! rotation-invariant registration.

use std::collections::BTreeMap;

use crate::op::{DEFAULT_GAP_TOL, SuperpositionError, superpose};
use molrs::core::{Neighbors, QueryMode, Topology};
use molrs::op::{F, Vec3};

use crate::compute::{Compute, ComputeError, Partition, is_half_shell, require_disp};

/// `EnvironmentMatch` analyzer.
///
/// Environment matching / clustering by neighbor-vector geometry.
///
/// Mirrors `freud.environment.EnvironmentCluster` / `EnvironmentMatch`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/EnvironmentMatch.cc)).
///
/// Configuration: the RMSD threshold, the matching mode and the registration
/// neighbor limit. Data: one snapshot's neighbor table with the `disp`
/// column. Two modes:
///
/// - **No-rotation** (`with_registration(false)`, the default): two
///   particles match when their **sorted bond magnitudes** agree
///   pair-wise within `rmsd_threshold`.
/// - **Registration** (`with_registration(true)`): two particles match
///   when there is a **rotation** and a **permutation** of one bond
///   set that minimises the RMSD vs the other to within
///   `rmsd_threshold`. Optimal rotation per permutation is found by
///   Horn's quaternion method ([`crate::op::superpose`]), rotation only:
///   the bond vectors are already relative to the centre particle, so no
///   translation is fitted.
///   Permutations are enumerated by Heap's algorithm, so registration costs
///   up to `n!` per comparison; an environment with more neighbors than
///   [`max_neighbors_for_registration`](Self::with_max_neighbors_for_registration)
///   (default 8 → 40 320 permutations) is an error rather than a silent
///   switch to magnitude-only matching.
///
/// After per-pair match decisions, particles are clustered into
/// environment classes: the connected components of the "matches" graph
/// ([`Topology::connected_components`]), labelled in order of their lowest
/// particle index. The result is a [`Partition`].
///
/// A half-shell table credits each bond to both particles (the `j` side
/// reversed); a cross-query table to the query particle only.
///
/// # Performance
///
/// Per-particle bond fingerprints are built in a **single O(n_pairs)
/// pass** over the neighbor list. Particles are then bucketed by neighbor
/// count, and matching compares every pair *within* a bucket: O(b²) per
/// bucket of size `b`. That quadratic comparison is intrinsic to environment
/// matching (freud does the same) and is the dominant cost at scale.
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentMatch {
    rmsd_threshold: F,
    registration: bool,
    max_neighbors_for_registration: usize,
}

/// The data an [`EnvironmentMatch`] analyses.
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentMatchInput<'a> {
    /// One snapshot's neighbor table with the `disp` column.
    pub neighbors: &'a Neighbors,
}

impl EnvironmentMatch {
    /// Rotation-invariant registration when not set: off (compare bond
    /// vectors as they are).
    pub const DEFAULT_REGISTRATION: bool = false;
    /// The largest environment registration accepts when not set.
    pub const DEFAULT_MAX_NEIGHBORS_FOR_REGISTRATION: usize = 8;

    /// Environments match when their best-fit RMSD ≤ `rmsd_threshold` (Å).
    /// Registration off; registration limit 8 neighbors.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a negative or NaN threshold.
    pub fn new(rmsd_threshold: F) -> Result<Self, ComputeError> {
        if rmsd_threshold.is_nan() || rmsd_threshold < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "EnvironmentMatch::rmsd_threshold",
                value: rmsd_threshold.to_string(),
            });
        }
        Ok(Self {
            rmsd_threshold,
            registration: Self::DEFAULT_REGISTRATION,
            max_neighbors_for_registration: Self::DEFAULT_MAX_NEIGHBORS_FOR_REGISTRATION,
        })
    }

    /// Toggle rotation-invariant registration (Horn's quaternion +
    /// brute-force permutation matching).
    pub fn with_registration(mut self, on: bool) -> Self {
        self.registration = on;
        self
    }

    /// The largest environment registration accepts (default
    /// [`DEFAULT_MAX_NEIGHBORS_FOR_REGISTRATION`](Self::DEFAULT_MAX_NEIGHBORS_FOR_REGISTRATION)). An
    /// environment with more neighbors makes [`compute`](Compute::compute)
    /// fail with [`ComputeError::OutOfRange`].
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for 0.
    pub fn with_max_neighbors_for_registration(mut self, n: usize) -> Result<Self, ComputeError> {
        if n == 0 {
            return Err(ComputeError::OutOfRange {
                field: "EnvironmentMatch::max_neighbors_for_registration",
                value: "0".into(),
            });
        }
        self.max_neighbors_for_registration = n;
        Ok(self)
    }

    /// The match threshold, Å.
    pub fn rmsd_threshold(&self) -> F {
        self.rmsd_threshold
    }

    /// The registration neighbor limit.
    pub fn max_neighbors_for_registration(&self) -> usize {
        self.max_neighbors_for_registration
    }

    /// Whether environments `a` and `b` (same bond count) match.
    fn matches(&self, a: &[Vec3], b: &[Vec3], mags: (&[F], &[F])) -> Result<bool, ComputeError> {
        let rmsd = if self.registration {
            rmsd_with_registration(a, b, self.rmsd_threshold)?
        } else {
            rmsd_no_rotation(mags.0, mags.1)
        };
        Ok(rmsd <= self.rmsd_threshold)
    }
}

/// Bond-vector fingerprints for every particle, built in a single
/// O(n_pairs) pass over the neighbor list. In self-query mode each pair
/// contributes to both endpoints (the `j`-side bond reversed); in
/// cross-query only the query-side (`i`) endpoint. Within each
/// particle's list bonds land in ascending pair-index order.
///
/// # Errors
/// [`ComputeError::BadShape`] for a table without `disp`;
/// [`ComputeError::NonFinite`] for a non-finite bond;
/// [`ComputeError::OutOfRange`] for a pair index past the point counts.
fn build_all_bond_vectors(nlist: &Neighbors) -> Result<Vec<Vec<Vec3>>, ComputeError> {
    let disp = require_disp(nlist)?;
    let symmetric = is_half_shell(nlist.mode());
    let n = match nlist.mode() {
        QueryMode::SelfQuery { n_points } => n_points,
        QueryMode::CrossQuery { n_query_points, .. } => n_query_points,
    };
    let mut bonds: Vec<Vec<Vec3>> = vec![Vec::new(); n];
    for (k, ((&i, &j), row)) in nlist
        .query_point_indices()
        .iter()
        .zip(nlist.point_indices())
        .zip(disp.outer_iter())
        .enumerate()
    {
        let v = [row[0], row[1], row[2]];
        if let Some(c) = v.iter().position(|x| !x.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "pair disp",
                index: 3 * k + c,
            });
        }
        let (i, j) = (i as usize, j as usize);
        if i >= n || (symmetric && j >= n) {
            return Err(ComputeError::OutOfRange {
                field: "pair index",
                value: format!("({i}, {j}) of pair {k} past the table's {n} points"),
            });
        }
        bonds[i].push(v);
        // A self-query has no self-pairs (i < j), so the reversed bond is
        // never the same row twice.
        if symmetric {
            bonds[j].push([-v[0], -v[1], -v[2]]);
        }
    }
    Ok(bonds)
}

/// Magnitude-only fingerprint derived from bond vectors (for the
/// no-rotation mode). Bonds are finite, so the order is total.
fn magnitudes_sorted(bonds: &[Vec3]) -> Vec<F> {
    let mut mags: Vec<F> = bonds
        .iter()
        .map(|b| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt())
        .collect();
    mags.sort_by(F::total_cmp);
    mags
}

/// RMSD between sorted-magnitude fingerprints of equal length (no rotation).
fn rmsd_no_rotation(a: &[F], b: &[F]) -> F {
    if a.is_empty() {
        return 0.0;
    }
    let s: F = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (s / a.len() as F).sqrt()
}

/// Optimal-rotation RMSD between two bond-vector sets `a[i] ↔ b[i]`,
/// `√(min_R Σ |R aᵢ − bᵢ|² / n)` over proper rotations `R` only (no
/// translation: the vectors are relative to the centre particle, as in freud's
/// `MatchEnv` registration).
///
/// [`superpose`] centres both sets, which would add a translation degree of
/// freedom. It is fed the point-reflection-symmetric sets `{aᵢ} ∪ {−aᵢ}` and
/// `{bᵢ} ∪ {−bᵢ}` instead: both centroids are then exactly 0, the
/// cross-covariance `S = Σ a bᵀ` merely doubles (same optimal `R`), and the
/// weighted RMSD² over the `2n` points, `2 Σ |R aᵢ − bᵢ|² / 2n`, equals the
/// rotation-only RMSD² over the original `n`.
///
/// # Errors
/// [`ComputeError::OutOfRange`] carrying any superposition failure other than
/// "no points" (two empty environments coincide); the bonds are finite and
/// the weights unit, so none is expected.
fn rmsd_horn(a: &[Vec3], b: &[Vec3]) -> Result<F, ComputeError> {
    let reference: Vec<Vec3> = a
        .iter()
        .copied()
        .chain(a.iter().map(|v| [-v[0], -v[1], -v[2]]))
        .collect();
    let target: Vec<Vec3> = b
        .iter()
        .copied()
        .chain(b.iter().map(|v| [-v[0], -v[1], -v[2]]))
        .collect();
    let weights = vec![1.0; reference.len()];
    match superpose(&reference, &target, &weights, DEFAULT_GAP_TOL) {
        Ok(fit) => Ok(fit.rmsd),
        Err(SuperpositionError::NoPoints) => Ok(0.0),
        Err(e) => Err(ComputeError::OutOfRange {
            field: "EnvironmentMatch registration",
            value: e.to_string(),
        }),
    }
}

/// Heap's algorithm: enumerate every permutation of `b`, callback per perm.
/// Stops when `cb` returns `Ok(false)` or an error.
fn for_each_permutation(
    b: &mut [Vec3],
    cb: &mut impl FnMut(&[Vec3]) -> Result<bool, ComputeError>,
) -> Result<(), ComputeError> {
    let n = b.len();
    let mut c = vec![0usize; n];
    if !cb(b)? {
        return Ok(());
    }
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i & 1 == 0 {
                b.swap(0, i);
            } else {
                b.swap(c[i], i);
            }
            if !cb(b)? {
                return Ok(());
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    Ok(())
}

/// Best RMSD over all permutations of `b` (same length as `a`), using Horn's
/// optimal rotation per permutation. Early-exits when a perm meets
/// `threshold`.
fn rmsd_with_registration(a: &[Vec3], b: &[Vec3], threshold: F) -> Result<F, ComputeError> {
    if a.is_empty() {
        return Ok(0.0);
    }
    let mut b_work = b.to_vec();
    let mut best = F::INFINITY;
    for_each_permutation(&mut b_work, &mut |perm| {
        best = best.min(rmsd_horn(a, perm)?);
        // Continue while we haven't yet matched.
        Ok(best > threshold)
    })?;
    Ok(best)
}

impl Compute for EnvironmentMatch {
    type Input<'a> = EnvironmentMatchInput<'a>;
    type Output = EnvironmentMatchResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for a table without `disp`;
    /// [`ComputeError::NonFinite`] for a non-finite bond;
    /// [`ComputeError::OutOfRange`] in registration mode for an environment
    /// with more neighbors than the registration limit.
    fn compute(
        &self,
        input: EnvironmentMatchInput<'_>,
    ) -> Result<EnvironmentMatchResult, ComputeError> {
        let bonds = build_all_bond_vectors(input.neighbors)?;
        let n = bonds.len();
        let fingerprints: Vec<Vec<F>> = bonds.iter().map(|b| magnitudes_sorted(b)).collect();

        // Only same-length bond sets can match: bucket by neighbor count.
        let mut by_len: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (i, b) in bonds.iter().enumerate() {
            by_len.entry(b.len()).or_default().push(i);
        }
        if self.registration
            && let Some((&too_many, bucket)) = by_len
                .range(self.max_neighbors_for_registration + 1..)
                .next()
        {
            return Err(ComputeError::OutOfRange {
                field: "EnvironmentMatch::max_neighbors_for_registration",
                value: format!(
                    "particle {} has {too_many} neighbors, above the registration limit {} \
                     (raise it with with_max_neighbors_for_registration, or match without \
                     registration)",
                    bucket[0], self.max_neighbors_for_registration
                ),
            });
        }

        let mut edges: Vec<[usize; 2]> = Vec::new();
        for bucket in by_len.values() {
            for (ai, &a) in bucket.iter().enumerate() {
                for &b in &bucket[ai + 1..] {
                    let mags = (fingerprints[a].as_slice(), fingerprints[b].as_slice());
                    if self.matches(&bonds[a], &bonds[b], mags)? {
                        edges.push([a, b]);
                    }
                }
            }
        }
        let mut matches = Topology::with_atoms(n);
        matches.add_bonds(&edges);
        // Components are labelled 0, 1, … in order of their lowest particle.
        let cluster_idx: Vec<u32> = matches
            .connected_components()
            .into_iter()
            .map(|label| label as u32)
            .collect();
        let n_clusters = cluster_idx.iter().max().map_or(0, |&m| m as usize + 1);

        Ok(EnvironmentMatchResult {
            cluster_idx,
            n_clusters,
            fingerprints,
        })
    }
}

/// One snapshot's environment classes; a [`Partition`] of the particles.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct EnvironmentMatchResult {
    /// Particle → environment-class label (`0..n_clusters`), classes numbered
    /// in order of their lowest particle.
    pub cluster_idx: Vec<u32>,
    /// Number of distinct environment classes.
    pub n_clusters: usize,
    /// Per-particle sorted bond magnitudes, Å (the raw feature).
    pub fingerprints: Vec<Vec<F>>,
}

impl Partition for EnvironmentMatchResult {
    fn n_items(&self) -> usize {
        self.cluster_idx.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        Some(u64::from(self.cluster_idx[item]))
    }
}

crate::compute::spec::compute_spec! {
    EnvironmentMatch as "EnvironmentMatch";
    params |s| {
        rmsd_threshold: Float;
        registration: Bool = EnvironmentMatch::DEFAULT_REGISTRATION => s.registration;
        max_neighbors_for_registration: Int
            = EnvironmentMatch::DEFAULT_MAX_NEIGHBORS_FOR_REGISTRATION
            => s.max_neighbors_for_registration;
    }
    inputs {
        neighbors: Neighbors;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::fixtures::neighbors_of;
    use crate::core::Cell;
    use molrs::core::SimBox;

    /// A snapshot: positions in an open cube of side `box_len` at the origin.
    struct Snapshot {
        positions: Vec<[F; 3]>,
        simbox: SimBox,
    }

    fn frame_with(positions: &[[F; 3]], box_len: F) -> Snapshot {
        Snapshot {
            positions: positions.to_vec(),
            simbox: SimBox::new(
                Cell::Lengths([box_len; 3]),
                [0.0 as F, 0.0 as F, 0.0 as F],
                [false; 3],
            )
            .unwrap(),
        }
    }

    fn nlist_from_frame(frame: &Snapshot, cutoff: F) -> Neighbors {
        neighbors_of(&frame.positions, &frame.simbox, cutoff)
    }

    fn classify(m: EnvironmentMatch, nl: &Neighbors) -> EnvironmentMatchResult {
        m.compute(EnvironmentMatchInput { neighbors: nl }).unwrap()
    }

    /// Two identical octahedra at different centres should be classed
    /// together; their satellite atoms (each with a single bond) form a
    /// second class.
    fn paired_octahedra() -> Snapshot {
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (10.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]); // centre (6 bonds of length 1)
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                p.push([cx + dx, cy + dy, cz + dz]); // satellite (1 bond)
            }
        }
        frame_with(&p, 20.0)
    }

    #[test]
    fn two_octahedra_centres_share_a_class() {
        let frame = paired_octahedra();
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &classify(EnvironmentMatch::new(1e-9).unwrap(), &nl);
        // Centres are particles 0 and 7. Their fingerprints are both
        // [1,1,1,1,1,1] → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
        // Each satellite has fingerprint [1] → another (single) class
        // shared by all 12 satellites.
        for k in (1..=6).chain(8..=13) {
            assert_eq!(r.cluster_idx[k], r.cluster_idx[1]);
        }
        // Centres vs satellites must differ (different fingerprint length).
        assert_ne!(r.cluster_idx[0], r.cluster_idx[1]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn distinct_fingerprints_get_distinct_classes() {
        // Three particles forming a chain: 0 has 1 neighbour, 1 has 2, 2 has 1.
        let frame = frame_with(
            &[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            10.0,
        );
        let nl = nlist_from_frame(&frame, 1.5);
        let r = &classify(EnvironmentMatch::new(1e-9).unwrap(), &nl);
        // 0 and 2 each have one bond of length 1 → same class.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[2]);
        // 1 has two bonds → different class.
        assert_ne!(r.cluster_idx[1], r.cluster_idx[0]);
        assert_eq!(r.n_clusters, 2);
    }

    #[test]
    fn threshold_merges_near_matches() {
        // Two centres with slightly perturbed bond lengths: 1.00 vs 1.01.
        // Threshold 0.05 merges them; threshold 0.005 keeps them apart.
        let mut p: Vec<[F; 3]> = Vec::new();
        for (cx, off) in &[(5.0_f64, 1.00_f64), (10.0_f64, 1.01_f64)] {
            p.push([*cx, 5.0, 5.0]);
            for &(dx, dy, dz) in &[
                (*off, 0.0, 0.0),
                (-*off, 0.0, 0.0),
                (0.0, *off, 0.0),
                (0.0, -*off, 0.0),
            ] {
                p.push([*cx + dx, 5.0 + dy, 5.0 + dz]);
            }
        }
        let frame = frame_with(&p, 20.0);
        let nl = nlist_from_frame(&frame, 1.5);
        let merged = &classify(EnvironmentMatch::new(0.05).unwrap(), &nl);
        assert_eq!(merged.cluster_idx[0], merged.cluster_idx[5]);

        let split = &classify(EnvironmentMatch::new(0.001).unwrap(), &nl);
        assert_ne!(split.cluster_idx[0], split.cluster_idx[5]);
    }

    /// Regression guard for the single-pass `build_all_bond_vectors`
    /// dispatch: an asymmetric chain with two *distinct* bond lengths
    /// pins each particle's collected bond set via the public
    /// `fingerprints`. A mis-dispatch — dropping the reversed `j`-side
    /// bond, assigning to the wrong endpoint, or double-counting a pair —
    /// changes at least one fingerprint here.
    ///
    /// Geometry: three collinear atoms at x = 0.0, 1.0, 2.5 with cutoff
    /// 1.6 Å. Neighbor pairs: (0,1) at 1.0 and (1,2) at 1.5; (0,2) at 2.5
    /// is outside the cutoff. Expected sorted magnitudes:
    ///   atom 0 → [1.0]   atom 1 → [1.0, 1.5]   atom 2 → [1.5]
    #[test]
    fn fingerprints_dispatch_forward_and_reversed_bonds() {
        let frame = frame_with(
            &[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
            10.0,
        );
        let nl = nlist_from_frame(&frame, 1.6);
        let r = &classify(EnvironmentMatch::new(1e-9).unwrap(), &nl);

        let approx = |a: &[F], b: &[F]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-9)
        };
        assert!(
            approx(&r.fingerprints[0], &[1.0]),
            "atom 0 fingerprint = {:?}",
            r.fingerprints[0]
        );
        assert!(
            approx(&r.fingerprints[1], &[1.0, 1.5]),
            "atom 1 fingerprint = {:?}",
            r.fingerprints[1]
        );
        assert!(
            approx(&r.fingerprints[2], &[1.5]),
            "atom 2 fingerprint = {:?}",
            r.fingerprints[2]
        );
    }

    #[test]
    fn invalid_threshold_errors() {
        assert!(EnvironmentMatch::new(-1.0).is_err());
    }

    /// The result is a partition: one group key per particle.
    #[test]
    fn result_is_a_partition() {
        let frame = paired_octahedra();
        let nl = nlist_from_frame(&frame, 1.2);
        let r = classify(EnvironmentMatch::new(1e-9).unwrap(), &nl);
        let p: &dyn Partition = &r;
        assert_eq!(p.n_items(), 14);
        assert_eq!(p.group_of(0), p.group_of(7));
        assert_ne!(p.group_of(0), p.group_of(1));
    }

    /// Registration above its neighbor limit used to fall back silently to
    /// magnitude-only matching; it is an error now.
    #[test]
    fn registration_above_the_limit_is_out_of_range() {
        let frame = paired_octahedra();
        let nl = nlist_from_frame(&frame, 1.2);
        let err = EnvironmentMatch::new(1e-6)
            .unwrap()
            .with_registration(true)
            .with_max_neighbors_for_registration(5)
            .unwrap()
            .compute(EnvironmentMatchInput { neighbors: &nl })
            .unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
        // Without registration the limit does not apply.
        let r = classify(
            EnvironmentMatch::new(1e-6)
                .unwrap()
                .with_max_neighbors_for_registration(5)
                .unwrap(),
            &nl,
        );
        assert_eq!(r.n_clusters, 2);
        assert!(matches!(
            EnvironmentMatch::new(1e-6)
                .unwrap()
                .with_max_neighbors_for_registration(0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    /// A non-finite bond used to sort with a panic or match nothing; it is
    /// an error now.
    #[test]
    fn non_finite_bond_is_non_finite() {
        use molrs::core::{NeighborColumns, NeighborPair};
        let nl = Neighbors::from_pairs(
            [NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [F::NAN, 0.0, 0.0],
            }],
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let err = EnvironmentMatch::new(0.1)
            .unwrap()
            .compute(EnvironmentMatchInput { neighbors: &nl })
            .unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { .. }));
    }

    /// Two octahedra at different orientations: the magnitude-only mode
    /// already merges them (their sorted magnitudes are equal). Add a
    /// case that ONLY registration mode catches: equal bond-magnitude
    /// sets in two different relative orientations whose lab-frame
    /// bond vectors differ but RMSD-with-rotation is zero.
    #[test]
    fn registration_matches_rotated_octahedra() {
        // Particle 0: octahedron, axes (±x, ±y, ±z).
        // Particle 7: same octahedron rotated 45° about z. Lab-frame bond
        // vectors differ from particle 0's; registration finds the
        // rotation that aligns them.
        let mut p: Vec<[F; 3]> = Vec::new();
        for &(cx, cy, cz) in &[(5.0_f64, 5.0, 5.0), (12.0_f64, 5.0, 5.0)] {
            p.push([cx, cy, cz]);
            for &(dx, dy, dz) in &[
                (1.0_f64, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ] {
                if cx > 10.0 {
                    // Rotate 45° about z for the second octahedron.
                    let c = std::f64::consts::FRAC_PI_4.cos();
                    let s = std::f64::consts::FRAC_PI_4.sin();
                    p.push([cx + c * dx - s * dy, cy + s * dx + c * dy, cz + dz]);
                } else {
                    p.push([cx + dx, cy + dy, cz + dz]);
                }
            }
        }
        let frame = frame_with(&p, 30.0);
        let nl = nlist_from_frame(&frame, 1.2);
        let r = &classify(
            EnvironmentMatch::new(1e-6).unwrap().with_registration(true),
            &nl,
        );
        // The two centres (indices 0 and 7) must share a class under
        // registration mode.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
    }
}
