//! Post-processing filters that prune a [`Neighbors`] table using geometric criteria: [`filter_sann`] and [`filter_rad`].

use crate::core::Neighbors;
use crate::op::F;

/// SANN filter: keep the smallest set of nearest neighbors whose solid
/// angles sum to `4π`.
///
/// SANN stands for *Solid-Angle Nearest Neighbours*. The idea is that the `m`
/// closest neighbors of a particle should between them account for the whole
/// `4π` steradians of directions around it; adding more neighbors would
/// over-cover the sphere, so `m` is chosen as the smallest count that just
/// covers it. No cutoff or coordination number is supplied by the user.
///
/// # The published criterion
///
/// van Meel et al., *J. Chem. Phys.* **136**, 234107 (2012). Sort the neighbors
/// of a query point by distance, `r_1 ≤ r_2 ≤ … ≤ r_n` (Å). Neighbor `j` claims
/// a spherical cap of solid angle `2π(1 − r_j/R)` for a trial shell radius `R`,
/// and requiring the first `m` caps to tile the whole sphere,
/// `Σ_{j≤m} 2π(1 − r_j/R) = 4π`, gives the shell radius implied by that count:
///
/// ```text
///   R(m) = ( Σ_{j=1..m} r_j ) / (m - 2)
/// ```
///
/// The chosen count is the smallest `m ≥ 3` for which that shell closes before
/// the next neighbor,
///
/// ```text
///   R(m) < r_{m+1}
/// ```
///
/// and the first `m` neighbors are kept. Reading the inequality the other way
/// round: while `R(m) ≥ r_{m+1}` the neighbor at `r_{m+1}` still lies inside the
/// shell that the first `m` imply, so it belongs to the set and `m` must grow.
///
/// # When the shell does not close
///
/// The test for count `m` needs `r_{m+1}`, so with `n` candidates only
/// `m ≤ n - 1` can be evaluated. If no such `m` closes the shell, the criterion
/// is *unconverged* on this candidate set — the answer would need neighbors the
/// input does not contain — and the query point's entire neighbor list is kept.
/// A query point with fewer than 3 neighbors is the same case and is likewise
/// kept whole. Supply a more generous search cutoff if unconverged points
/// matter.
///
/// Distances are Å; the implementation sorts on the stored `dist_sq` (Å²), which
/// is order-equivalent because the square root is monotonic, then takes square
/// roots to evaluate `R(m)` and to compare it against `r_{m+1}`.
///
/// The output inherits the input's `mode` and column policy, and query points
/// are visited in ascending index order so the result is deterministic.
///
/// # Why a filter
///
/// A cutoff-based neighbor list answers "which particles are within `r_c`?",
/// which is a question about a number the user picked, not about the structure.
/// The filters here answer "which particles are *actually* nearest neighbors?"
/// — they take an over-generous cutoff list and discard the pairs that a
/// parameter-free geometric criterion says are not part of the first
/// coordination shell (the set of particles directly touching a given one).
///
/// A port of `freud.locality.FilterSANN`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/FilterSANN.cc));
/// [`filter_rad`] is the other filter. Both preserve the query-point ordering of
/// pairs but may shrink the list.
///
/// # Panics
/// Panics if `nlist` lacks the `dist_sq` or `disp` column. `dist_sq` carries the
/// criterion itself; `disp` is needed to copy surviving pairs into the output
/// table. Neither can be reconstructed from indices, and substituting zeros
/// would fabricate physically meaningful values.
///
/// Both filters require a fully populated input. Each reads `dist_sq` for its criterion and `disp` to copy surviving
/// pairs into the output, so a table built without either column is rejected
/// with a panic naming the missing column. That is deliberate: a lean table
/// reports an absent column as `None`, and the only alternatives to panicking
/// would be to invent zeros — a zero displacement is a physically meaningful
/// value, so it would be indistinguishable from data — or to return an empty
/// list, which reads downstream as "this particle has no neighbors". Rerun the
/// search with [`NeighborColumns::FULL`](crate::core::NeighborColumns::FULL)
/// instead.
pub fn filter_sann(nlist: &Neighbors) -> Neighbors {
    let (offsets, mut order) = pairs_by_query(nlist);

    let mut out = Neighbors::empty(nlist.mode(), nlist.columns());
    out.simbox = nlist.simbox.clone();
    let dist_sq = nlist
        .dist_sq()
        .expect("filter_sann needs the dist_sq column; rerun the search with it stored");
    let vectors = nlist
        .disp()
        .expect("filter_sann needs the disp column; rerun the search with it stored");
    let j_idx = nlist.point_indices();

    // Query points come out in ascending index order: the grouping is a
    // counting sort, so the output is deterministic.
    for q in 0..offsets.len().saturating_sub(1) {
        let pair_ks = &mut order[offsets[q]..offsets[q + 1]];
        if pair_ks.is_empty() {
            continue;
        }
        pair_ks.sort_unstable_by(|&a, &b| dist_sq[a].partial_cmp(&dist_sq[b]).unwrap());
        let m = sann_cutoff(pair_ks, dist_sq);
        for &k in &pair_ks[..m] {
            out.push(
                q as u32,
                j_idx[k],
                dist_sq[k],
                [vectors[[k, 0]], vectors[[k, 1]], vectors[[k, 2]]],
            );
        }
    }
    out
}

/// Pair rows grouped by query point, as CSR: `order[offsets[q]..offsets[q + 1]]`
/// are the rows of query `q`, in ascending row order. A counting sort — one
/// pass to count, one to place — with no per-query allocation.
fn pairs_by_query(nlist: &Neighbors) -> (Vec<usize>, Vec<usize>) {
    let queries = nlist.query_point_indices();
    let n_queries = queries.iter().map(|&q| q as usize + 1).max().unwrap_or(0);
    let mut offsets = vec![0usize; n_queries + 1];
    for &q in queries {
        offsets[q as usize + 1] += 1;
    }
    for q in 0..n_queries {
        offsets[q + 1] += offsets[q];
    }
    let mut next = offsets.clone();
    let mut order = vec![0usize; queries.len()];
    for (row, &q) in queries.iter().enumerate() {
        order[next[q as usize]] = row;
        next[q as usize] += 1;
    }
    (offsets, order)
}

/// Find the smallest `m ≥ 3` with `R(m) = (Σ_{i=1..m} r_i) / (m - 2) < r_{m+1}`
/// over distance-sorted neighbors — the SANN criterion of van Meel et al.
/// (see [`filter_sann`]). If no such `m` exists the shell never closes within
/// the candidate set, and the full length is returned so the caller keeps
/// everything.
///
/// `pair_ks_sorted` indexes `dist_sq` in ascending distance. Because the test
/// for `m` reads `r_{m+1}`, the loop can only reach `m = n - 1`; `m = n` would
/// need a candidate beyond the list, so it falls through to the keep-all result.
fn sann_cutoff(pair_ks_sorted: &[usize], dist_sq: &[F]) -> usize {
    let n = pair_ks_sorted.len();
    if n < 3 {
        return n;
    }
    // 1-based r_i is the 0-based entry i - 1.
    let r = |i: usize| dist_sq[pair_ks_sorted[i - 1]].sqrt();
    // Running Σ_{i=1..m}, seeded with the two terms that precede the first
    // testable count m = 3.
    let mut sum_r: F = r(1) + r(2);
    for m in 3..n {
        sum_r += r(m);
        if sum_r / ((m - 2) as F) < r(m + 1) {
            return m;
        }
    }
    n
}

/// RAD filter: drop a neighbor `j` whenever there is a strictly closer
/// accepted neighbor `k` such that the bond `r_ij` lies inside the cone
/// of half-angle `acceptance` around `r_ik`.
///
/// RAD stands for *Relative Angular Distance*. The picture is one of blocking:
/// a nearby neighbor occludes the directions behind it, so a farther particle
/// sitting in roughly the same direction is judged to be in the second shell
/// rather than in contact. Neighbors are considered in order of increasing
/// distance, and each accepted one starts occluding for those that follow.
///
/// Concretely, for the unit vectors `r̂_ik` (already accepted, closer) and
/// `r̂_ij` (candidate), the candidate is dropped when
/// `r̂_ik · r̂_ij > cos(acceptance)`, which is exactly the statement that the
/// angle between them is smaller than `acceptance`.
///
/// `acceptance` is in **radians**; `std::f64::consts::FRAC_PI_3` (`60°`) is the
/// value this port carries as its default. A smaller angle occludes less and so
/// keeps more neighbors. Note that the criterion here is a *fixed* cone: unlike
/// the original relative-angular-distance formulation it does not additionally
/// weight the test by how much closer the occluding neighbor is, so agreement
/// with the freud implementation
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/FilterRAD.cc))
/// should be re-established by test before the two are treated as interchangeable.
///
/// A candidate at exactly zero distance has no direction, so it is skipped
/// entirely — it neither survives the filter nor occludes anything. This
/// matters for a cross-query whose query and reference sets are the same
/// points, where every point is its own neighbor at distance zero.
///
/// The output inherits the input's `mode` and column policy, and query points
/// are visited in ascending index order so the result is deterministic.
///
/// # Panics
/// Panics if `nlist` lacks the `dist_sq` or `disp` column. Both are load-bearing
/// here: `disp` supplies the directions the criterion compares and `dist_sq` the
/// order they are considered in. See [`filter_sann`] for why a missing column is a
/// panic rather than an empty result.
pub fn filter_rad(nlist: &Neighbors, acceptance: F) -> Neighbors {
    let cos_thresh = acceptance.cos();
    let (offsets, mut order) = pairs_by_query(nlist);

    let mut out = Neighbors::empty(nlist.mode(), nlist.columns());
    out.simbox = nlist.simbox.clone();
    let dist_sq = nlist
        .dist_sq()
        .expect("filter_rad needs the dist_sq column; rerun the search with it stored");
    let vectors = nlist
        .disp()
        .expect("filter_rad needs the disp column; rerun the search with it stored");
    let j_idx = nlist.point_indices();

    for q in 0..offsets.len().saturating_sub(1) {
        let pair_ks = &mut order[offsets[q]..offsets[q + 1]];
        if pair_ks.is_empty() {
            continue;
        }
        pair_ks.sort_unstable_by(|&a, &b| dist_sq[a].partial_cmp(&dist_sq[b]).unwrap());

        // Walk neighbors in increasing distance and accept those that are
        // not "shadowed" by an already-accepted closer one.
        let mut accepted: Vec<[F; 3]> = Vec::new();
        for &k in pair_ks.iter() {
            let r = dist_sq[k].sqrt();
            if r == 0.0 {
                continue;
            }
            let nx = vectors[[k, 0]] / r;
            let ny = vectors[[k, 1]] / r;
            let nz = vectors[[k, 2]] / r;
            let mut shadowed = false;
            for a in &accepted {
                let dot = a[0] * nx + a[1] * ny + a[2] * nz;
                if dot > cos_thresh {
                    shadowed = true;
                    break;
                }
            }
            if !shadowed {
                accepted.push([nx, ny, nz]);
                out.push(
                    q as u32,
                    j_idx[k],
                    dist_sq[k],
                    [vectors[[k, 0]], vectors[[k, 1]], vectors[[k, 2]]],
                );
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{NeighborColumns, NeighborPair, QueryMode};

    /// Cross-query table over `n_points × n_points`, both physical columns
    /// stored (a filter needs them).
    fn make_nlist(pairs: &[(u32, u32, F, [F; 3])], n_points: usize) -> Neighbors {
        Neighbors::from_pairs(
            pairs.iter().map(|&(i, j, dist_sq, disp)| NeighborPair {
                i,
                j,
                dist_sq,
                disp,
            }),
            NeighborColumns::FULL,
            QueryMode::CrossQuery {
                n_query_points: n_points,
                n_points,
            },
        )
    }

    // ---- SANN -------------------------------------------------------------

    /// Regression: the SANN criterion must cap a query point whose neighbors
    /// sit at *distinct* distances.
    ///
    /// van Meel, Filion, Valeriani & Frenkel, *J. Chem. Phys.* **136**, 234107
    /// (2012): with distances sorted ascending, the neighbor count is the
    /// smallest `m ≥ 3` such that `R(m) = (Σ_{i≤m} r_i) / (m − 2) < r_{m+1}`.
    ///
    /// Fixture — central query atom at the origin, four candidates that a
    /// 5.0 Å cutoff all admits (largest separation 3.5 Å), so the filter sees
    /// the full four-pair list:
    ///
    /// | j | position (Å)   | r (Å) | `dist_sq` (Å²) |
    /// |---|----------------|-------|----------------|
    /// | 1 | ( 0.9, 0,   0) | 0.9   | 0.81           |
    /// | 2 | ( 0,   1.0, 0) | 1.0   | 1.00           |
    /// | 3 | ( 0,   0,  1.1)| 1.1   | 1.21           |
    /// | 4 | (−3.5, 0,   0) | 3.5   | 12.25          |
    ///
    /// Hand derivation:
    ///   m = 3 → R(3) = (0.9 + 1.0 + 1.1) / (3 − 2) = 3.0 < r_4 = 3.5 ✓
    /// so the shell closes at m = 3 and the 3.5 Å candidate is dropped.
    #[test]
    fn sann_distinct_distances_matches_published_criterion() {
        let pairs = [
            (0u32, 1u32, 0.81_f64, [0.9_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.21, [0.0, 0.0, 1.1]),
            (0, 4, 12.25, [-3.5, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 5);
        let f = filter_sann(&nl);

        assert_eq!(f.n_pairs(), 3, "shell closes at m = 3 (R(3) = 3.0 < 3.5)");
        let kept: Vec<u32> = (0..f.n_pairs()).map(|k| f.point_indices()[k]).collect();
        assert_eq!(kept, vec![1, 2, 3], "the three closest survive, in order");
        assert!(
            !kept.contains(&4),
            "the 3.5 Å candidate is outside the closed shell and must be dropped"
        );
    }

    /// Four neighbors at distances 1, 1, 1, 2 — the published criterion never
    /// converges here, so the whole list survives.
    ///
    /// Hand derivation (van Meel et al., see above):
    ///   m = 3 → R(3) = (1 + 1 + 1) / (3 − 2) = 3.0 < r_4 = 2.0 ✗
    ///   m = 4 → would need r_5, which does not exist (n = 4)
    /// No `m < n` closes the shell → unconverged → keep all 4.
    ///
    /// 3 would be the wrong answer: the inequality `Σ_{j≤m} r_j ≥ m·r_m`
    /// fires at m = 3 purely because the first three distances are exactly
    /// equal. The golden is the published value.
    #[test]
    fn sann_keeps_all_when_shell_never_closes() {
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.0, [0.0, 0.0, 1.0]),
            (0, 4, 4.0, [2.0, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 5);
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 4, "R(3) = 3.0 ≥ r_4 = 2.0 → no shell closes");
    }

    #[test]
    fn sann_keeps_all_when_below_three() {
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 2);
    }

    /// Two query points, each carrying the 0.9 / 1.0 / 1.1 / 3.5 Å fixture:
    /// the filter closes each shell independently at m = 3.
    ///
    /// Hand derivation, per query point (van Meel et al.):
    ///   m = 3 → R(3) = (0.9 + 1.0 + 1.1) / (3 − 2) = 3.0 < r_4 = 3.5 ✓
    ///
    /// The distances are deliberately *distinct*. The earlier fixture used
    /// 1, 1, 1, 2 per query, whose published answer is "keep all 4" — with
    /// that fixture a `filter_sann` that returned its input untouched would
    /// still satisfy the assertion, so the test could not tell the filter
    /// apart from a no-op.
    #[test]
    fn sann_preserves_query_split() {
        let pairs = [
            (0u32, 1u32, 0.81_f64, [0.9_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.21, [0.0, 0.0, 1.1]),
            (0, 4, 12.25, [-3.5, 0.0, 0.0]),
            (1, 0, 0.81, [-0.9, 0.0, 0.0]),
            (1, 5, 1.0, [0.0, 1.0, 0.0]),
            (1, 6, 1.21, [0.0, 0.0, 1.1]),
            (1, 7, 12.25, [3.5, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 8);
        let f = filter_sann(&nl);
        let kept_by = |q: u32| -> Vec<u32> {
            (0..f.n_pairs())
                .filter(|&k| f.query_point_indices()[k] == q)
                .map(|k| f.point_indices()[k])
                .collect()
        };
        assert_eq!(kept_by(0), vec![1, 2, 3], "query 0 caps at m = 3");
        assert_eq!(kept_by(1), vec![0, 5, 6], "query 1 caps at m = 3");
    }

    // ---- RAD --------------------------------------------------------------

    #[test]
    fn rad_keeps_orthogonal_neighbors() {
        // Three neighbors along +x, +y, +z (pairwise angles 90°, so cosines
        // are 0 < cos(60°) = 0.5). All three should be retained.
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
            (0, 3, 1.0, [0.0, 0.0, 1.0]),
        ];
        let nl = make_nlist(&pairs, 4);
        let f = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(f.n_pairs(), 3);
    }

    #[test]
    fn rad_drops_a_second_neighbor_in_the_same_cone() {
        // Two neighbors along +x; the second is at twice the distance and
        // lies inside the 60° cone of the first → dropped.
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 4.0, [2.0_f64, 0.0, 0.0]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(f.n_pairs(), 1);
        assert_eq!(f.point_indices()[0], 1);
    }

    #[test]
    fn rad_threshold_change_alters_acceptance() {
        // Two neighbors 30° apart along +x and a tilted +x direction.
        // With acceptance=60°, the second falls inside the cone (cos 30° ≈
        // 0.866 > cos 60° = 0.5) → dropped. With acceptance=10° (cos ≈
        // 0.985), it survives.
        let c = (30.0_f64).to_radians().cos();
        let s = (30.0_f64).to_radians().sin();
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0_f64, [c, s, 0.0_f64]),
        ];
        let nl = make_nlist(&pairs, 3);
        let f60 = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        let f10 = filter_rad(&nl, 10.0_f64.to_radians());
        assert_eq!(f60.n_pairs(), 1);
        assert_eq!(f10.n_pairs(), 2);
    }

    #[test]
    fn empty_input_returns_empty() {
        let nl = Neighbors::from_pairs(
            std::iter::empty(),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 0 },
        );
        let f = filter_sann(&nl);
        assert_eq!(f.n_pairs(), 0);
        let g = filter_rad(&nl, std::f64::consts::FRAC_PI_3);
        assert_eq!(g.n_pairs(), 0);
    }

    // ---- lean input is rejected, not silently filtered to nothing ----------

    #[test]
    #[should_panic(expected = "filter_sann needs the dist_sq column")]
    fn sann_on_indices_only_panics() {
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
        ];
        let lean = make_nlist(&pairs, 3).repack(NeighborColumns::INDICES_ONLY);
        let _ = filter_sann(&lean);
    }

    #[test]
    #[should_panic(expected = "filter_rad needs the disp column")]
    fn rad_without_disp_panics() {
        let pairs = [
            (0u32, 1u32, 1.0_f64, [1.0_f64, 0.0, 0.0]),
            (0, 2, 1.0, [0.0, 1.0, 0.0]),
        ];
        let lean = make_nlist(&pairs, 3).repack(NeighborColumns::DIST_SQ);
        let _ = filter_rad(&lean, std::f64::consts::FRAC_PI_3);
    }
}
