// Union-Find traversal and per-bucket O(b²) compare loops read more
// clearly with explicit indexing than iterator combinators.
#![allow(clippy::needless_range_loop, clippy::if_same_then_else)]

//! Environment matching by neighbor-bond fingerprint, with optional
//! rotation-invariant registration.

use crate::compute::ComputeResult;
use std::collections::HashMap;

use crate::op::{DEFAULT_GAP_TOL, SuperpositionError, superpose};
use molrs::core::FrameAccess;
use molrs::core::Neighbors;
use molrs::op::F;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref;
use crate::compute::require_disp;

/// `EnvironmentMatch` analyzer.
///
/// See below for the two matching modes and
/// their complexity (linear fingerprint build; quadratic per-bucket
/// comparison).
///
/// Environment matching / clustering by neighbor-vector geometry.
///
/// Mirrors `freud.environment.EnvironmentCluster` / `EnvironmentMatch`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/environment/EnvironmentMatch.cc)).
/// Two modes:
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
///   Permutations are enumerated by Heap's algorithm — viable for the
///   typical neighborhood sizes `n ≤ 12` (12! ≈ 4.8 × 10⁸ but with
///   early-exit on `rmsd > threshold` the practical count is much lower).
///
/// After per-pair match decisions, particles are clustered into
/// environment classes by union-find.
///
/// # Performance
///
/// Per-particle bond fingerprints are built in a **single O(n_pairs)
/// pass** over the neighbor list — linear in the total neighbor count
/// (≈ O(N·k) for `N` particles with `k` neighbors each). Particles are
/// then bucketed by neighbor count, and matching compares every pair
/// *within* a bucket: O(b²) per bucket of size `b`. That quadratic
/// comparison is intrinsic to environment matching (freud does the same)
/// and is the dominant cost at scale. In registration mode each
/// comparison additionally enumerates up to `n!` permutations, bounded by
/// [`EnvironmentMatch::with_max_neighbors_for_registration`].
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentMatch {
    rmsd_threshold: F,
    registration: bool,
    /// Hard cap on `n_neighbors` accepted in registration mode — guards
    /// against accidentally invoking O(n!) brute force for huge
    /// neighborhoods. Defaults to 8 (matches freud's safety bound).
    max_neighbors_for_registration: usize,
}

impl EnvironmentMatch {
    /// Environments match when their best-fit RMSD ≤ `rmsd_threshold` (Å).
    pub fn new(rmsd_threshold: F) -> Result<Self, ComputeError> {
        if rmsd_threshold.is_nan() || rmsd_threshold < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "EnvironmentMatch::rmsd_threshold",
                value: rmsd_threshold.to_string(),
            });
        }
        Ok(Self {
            rmsd_threshold,
            registration: false,
            max_neighbors_for_registration: 8,
        })
    }

    /// Toggle rotation-invariant registration (Horn's quaternion +
    /// brute-force permutation matching).
    pub fn with_registration(mut self, on: bool) -> Self {
        self.registration = on;
        self
    }

    /// Set the hard cap on neighborhood size in registration mode
    /// (`n!` cost grows fast — default 8 → 40 320 perms).
    pub fn with_max_neighbors_for_registration(mut self, n: usize) -> Self {
        self.max_neighbors_for_registration = n;
        self
    }

    pub fn rmsd_threshold(&self) -> F {
        self.rmsd_threshold
    }
}

/// Bond-vector fingerprints for every particle, built in a single
/// O(n_pairs) pass over the neighbor list. In self-query mode each pair
/// contributes to both endpoints (the `j`-side bond reversed); in
/// cross-query only the query-side (`i`) endpoint. Within each
/// particle's list bonds land in ascending pair-index order — the same
/// order a per-particle scan would produce — so downstream fingerprints,
/// magnitude sorting, and registration RMSD are unchanged. Pair indices
/// outside `0..n` are ignored.
///
/// A table without the `disp` column carries no bond vectors at all, so it is
/// rejected as [`ComputeError::BadShape`] — the caller's neighbor list is an
/// input, and a bad input is an error rather than a panic.
fn build_all_bond_vectors(n: usize, nlist: &Neighbors) -> Result<Vec<Vec<[F; 3]>>, ComputeError> {
    let mut bonds: Vec<Vec<[F; 3]>> = vec![Vec::new(); n];
    let i_idx = nlist.query_point_indices();
    let j_idx = nlist.point_indices();
    let disp = require_disp(nlist)?;
    let symmetric = matches!(nlist.mode(), molrs::core::QueryMode::SelfQuery { .. });
    for k in 0..nlist.n_pairs() {
        let i = i_idx[k] as usize;
        let v = [disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]];
        if i < n {
            bonds[i].push(v);
        }
        if symmetric {
            let j = j_idx[k] as usize;
            // Skip self-pairs so the reversed bond is not double-counted.
            if j != i && j < n {
                bonds[j].push([-v[0], -v[1], -v[2]]);
            }
        }
    }
    Ok(bonds)
}

/// Magnitude-only fingerprint derived from bond vectors (for the
/// no-rotation mode).
fn magnitudes_sorted(bonds: &[[F; 3]]) -> Vec<F> {
    let mut mags: Vec<F> = bonds
        .iter()
        .map(|b| (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt())
        .collect();
    mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
    mags
}

/// RMSD between sorted-magnitude fingerprints (no rotation).
fn rmsd_no_rotation(a: &[F], b: &[F]) -> F {
    if a.len() != b.len() {
        return F::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let mut s: F = 0.0;
    for k in 0..a.len() {
        let d = a[k] - b[k];
        s += d * d;
    }
    (s / a.len() as F).sqrt()
}

/// Optimal-rotation RMSD between two bond-vector sets `a[i] ↔ b[i]`,
/// `√(min_R Σ |R aᵢ − bᵢ|² / n)` over proper rotations `R` only (no
/// translation: the vectors are relative to the centre particle, as in freud's
/// `MatchEnv` registration). Assumes `a.len() == b.len()`.
///
/// [`superpose`] centres both sets, which would add a translation degree of
/// freedom. It is fed the point-reflection-symmetric sets `{aᵢ} ∪ {−aᵢ}` and
/// `{bᵢ} ∪ {−bᵢ}` instead: both centroids are then exactly 0, the
/// cross-covariance `S = Σ a bᵀ` merely doubles (same optimal `R`), and the
/// weighted RMSD² over the `2n` points, `2 Σ |R aᵢ − bᵢ|² / 2n`, equals the
/// rotation-only RMSD² over the original `n`.
fn rmsd_horn(a: &[[F; 3]], b: &[[F; 3]]) -> F {
    let reference: Vec<[F; 3]> = a
        .iter()
        .copied()
        .chain(a.iter().map(|v| [-v[0], -v[1], -v[2]]))
        .collect();
    let target: Vec<[F; 3]> = b
        .iter()
        .copied()
        .chain(b.iter().map(|v| [-v[0], -v[1], -v[2]]))
        .collect();
    let weights = vec![1.0; reference.len()];
    match superpose(&reference, &target, &weights, DEFAULT_GAP_TOL) {
        Ok(fit) => fit.rmsd,
        // No bonds: two empty environments coincide.
        Err(SuperpositionError::NoPoints) => 0.0,
        // A non-finite bond vector never matches.
        Err(_) => F::INFINITY,
    }
}

/// Heap's algorithm: enumerate every permutation of `b`, callback per perm.
/// Early-terminates when `cb` returns `false`.
fn for_each_permutation(b: &mut [[F; 3]], cb: &mut impl FnMut(&[[F; 3]]) -> bool) -> bool {
    let n = b.len();
    let mut c = vec![0usize; n];
    if !cb(b) {
        return false;
    }
    let mut i = 0;
    while i < n {
        if c[i] < i {
            if i & 1 == 0 {
                b.swap(0, i);
            } else {
                b.swap(c[i], i);
            }
            if !cb(b) {
                return false;
            }
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
    true
}

/// Best RMSD over all permutations of `b`, using Horn's optimal rotation
/// per permutation. Early-exits when a perm meets `threshold`.
fn rmsd_with_registration(a: &[[F; 3]], b: &[[F; 3]], threshold: F) -> F {
    if a.len() != b.len() {
        return F::INFINITY;
    }
    if a.is_empty() {
        return 0.0;
    }
    let mut b_work = b.to_vec();
    let mut best = F::INFINITY;
    for_each_permutation(&mut b_work, &mut |perm| {
        let r = rmsd_horn(a, perm);
        if r < best {
            best = r;
        }
        // Continue while we haven't yet matched.
        best > threshold
    });
    best
}

/// Single-pass disjoint-set union for environment clustering.
struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }
    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let next = self.parent[x as usize];
            self.parent[x as usize] = self.parent[next as usize];
            x = next;
        }
        x
    }
    fn union(&mut self, a: u32, b: u32) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent[ra as usize] = rb;
        }
    }
}

impl EnvironmentMatch {
    fn one_frame<FA: FrameAccess>(
        &self,
        frame: &FA,
        nlist: &Neighbors,
    ) -> Result<EnvironmentMatchResult, ComputeError> {
        let (xs_p, _, _) = get_positions_ref(frame)?;
        let n = xs_p.slice().len();

        let bonds = build_all_bond_vectors(n, nlist)?;

        // Group particles by neighbor count first — only same-length
        // bond sets can match.
        let mut by_len: HashMap<usize, Vec<usize>> = HashMap::new();
        for (i, b) in bonds.iter().enumerate() {
            by_len.entry(b.len()).or_default().push(i);
        }

        let mut dsu = Dsu::new(n);
        for bucket in by_len.values() {
            if bucket.is_empty() {
                continue;
            }
            let bucket_len = bonds[bucket[0]].len();
            // Decide mode for this bucket: registration is off iff
            // (a) caller disabled it, or
            // (b) bucket_len exceeds the safety cap.
            let use_registration =
                self.registration && bucket_len <= self.max_neighbors_for_registration;
            // Pre-compute sorted magnitudes for non-registration path.
            let mags: Vec<Vec<F>> = if !use_registration {
                bucket
                    .iter()
                    .map(|&i| magnitudes_sorted(&bonds[i]))
                    .collect()
            } else {
                Vec::new()
            };
            for ai in 0..bucket.len() {
                let a = bucket[ai];
                for bi in (ai + 1)..bucket.len() {
                    let b = bucket[bi];
                    let r = if use_registration {
                        rmsd_with_registration(&bonds[a], &bonds[b], self.rmsd_threshold)
                    } else {
                        rmsd_no_rotation(&mags[ai], &mags[bi])
                    };
                    if r <= self.rmsd_threshold {
                        dsu.union(a as u32, b as u32);
                    }
                }
            }
        }

        // Compact root labels into 0..n_clusters.
        let mut label_for: HashMap<u32, u32> = HashMap::new();
        let mut cluster_idx = vec![0_u32; n];
        for i in 0..n {
            let r = dsu.find(i as u32);
            let next = label_for.len() as u32;
            let lbl = *label_for.entry(r).or_insert(next);
            cluster_idx[i] = lbl;
        }
        let n_clusters = label_for.len();

        // Expose magnitude fingerprints for inspection (same shape as
        // before; bond-vector fingerprints are internal-only).
        let fingerprints: Vec<Vec<F>> = bonds.iter().map(|b| magnitudes_sorted(b)).collect();

        Ok(EnvironmentMatchResult {
            cluster_idx,
            n_clusters,
            fingerprints,
        })
    }
}

impl LegacyCompute for EnvironmentMatch {
    type Args<'a> = &'a [Neighbors];
    type Output = Vec<EnvironmentMatchResult>;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        nlists: &'a [Neighbors],
    ) -> Result<Vec<EnvironmentMatchResult>, ComputeError> {
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
        let mut out = Vec::with_capacity(frames.len());
        for (f, nl) in frames.iter().zip(nlists.iter()) {
            out.push(self.one_frame(*f, nl)?);
        }
        Ok(out)
    }
}

/// Per-frame environment-matching result.
#[derive(Debug, Clone, Default)]
pub struct EnvironmentMatchResult {
    /// Particle → environment-class label (`0..n_clusters`).
    pub cluster_idx: Vec<u32>,
    /// Number of distinct environment classes.
    pub n_clusters: usize,
    /// Per-particle sorted-bond fingerprint (the raw feature).
    pub fingerprints: Vec<Vec<F>>,
}

impl ComputeResult for EnvironmentMatchResult {}

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
        frame.simbox =
            Some(SimBox::cube(box_len, array![0.0 as F, 0.0 as F, 0.0 as F], [false; 3]).unwrap());
        frame
    }

    /// Two identical octahedra at different centres should be classed
    /// together; their satellite atoms (each with a single bond) form a
    /// second class.
    fn paired_octahedra() -> Frame {
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
        let r = &EnvironmentMatch::new(1e-9)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
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
        let r = &EnvironmentMatch::new(1e-9)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
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
        let merged = &EnvironmentMatch::new(0.05)
            .unwrap()
            .compute(&[&frame], std::slice::from_ref(&nl))
            .unwrap()[0];
        assert_eq!(merged.cluster_idx[0], merged.cluster_idx[5]);

        let split = &EnvironmentMatch::new(0.001)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];
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
        let r = &EnvironmentMatch::new(1e-9)
            .unwrap()
            .compute(&[&frame], &[nl])
            .unwrap()[0];

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

    #[test]
    fn empty_frames_error() {
        let frames: Vec<&Frame> = Vec::new();
        let err = EnvironmentMatch::new(0.1)
            .unwrap()
            .compute(&frames, &Vec::<Neighbors>::new())
            .unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
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
        let r = &EnvironmentMatch::new(1e-6)
            .unwrap()
            .with_registration(true)
            .compute(&[&frame], &[nl])
            .unwrap()[0];
        // The two centres (indices 0 and 7) must share a class under
        // registration mode.
        assert_eq!(r.cluster_idx[0], r.cluster_idx[7]);
    }
}
