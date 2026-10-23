//! Axis-Aligned Bounding-Box tree neighbor search: [`AabbQuery`].
// Tight 3-coord AABB loops read naturally with index-based access.
#![allow(clippy::needless_range_loop)]

use super::{Backend, PairVisitor};
use crate::core::bvh::Bvh;
use crate::core::{Points, SimBox};
use crate::op::{F, Fnx3};

/// AABB-tree k-nearest-neighbor query.
///
/// A bounding-volume hierarchy over one point set. [`build`](Self::build) indexes the points, and the index
/// then answers two questions.
///
/// [`query_knn`](Self::query_knn) answers *which `k` points lie closest to this
/// position* under the minimum-image convention — a question with no radius,
/// which a cutoff-based engine cannot express. The same tree also backs cutoff
/// searches as the `Aabb` backend of
/// [`NeighborList`](crate::core::NeighborList) (construct one
/// with [`NeighborList::aabb`](crate::core::NeighborList::aabb)),
/// which is where pair enumeration and table materialization live — this type
/// exposes neither directly.
///
/// # Algorithm
///
/// Mirrors `freud.locality.AABBQuery`
/// ([source](https://github.com/glotzerlab/freud/blob/main/freud/locality/AABBQuery.cc)).
///
/// An *axis-aligned bounding box* (AABB) is the smallest box with faces
/// parallel to the coordinate axes that contains a set of points; it is cheap
/// to store (two corners) and cheap to test a distance against. A *bounding
/// volume hierarchy* (BVH) is a binary tree of such boxes: each leaf wraps a
/// single point, each internal node owns the union of its two children's boxes.
/// A query descends from the root and prunes any subtree whose box is farther
/// from the query point than the worst candidate found so far, so most of the
/// tree is never visited. Average cost is `O(log N + k)` for `k` neighbors.
///
/// ## PBC handling — MIC-based, no ghost atoms
///
/// Under periodic boundary conditions (PBC) the box tiles space, so a particle
/// near one face may be close to a particle near the opposite face. The
/// *minimum-image convention* (MIC) is the rule that only the shortest of those
/// separations counts. Periodicity is handled the same way
/// [`LinkCell`](crate::core::LinkCell) does it: the tree is built
/// **only on the original `N` points**, never on a ghost-expanded set. For each
/// query point we enumerate the lattice-image shifts that could bring a tree
/// point within reach of the query, run one (non-periodic) descent per shift,
/// then pin every hit's displacement to the canonical minimum-image vector
/// returned by [`Mic::apply`](crate::core::Mic::apply).
///
/// Both the indexed points and the query are folded into the primary cell
/// first ([`SimBox::wrap`]), because the image-shift range below is derived
/// from the cell's own geometry and therefore only reaches the images of a
/// point that lies in it. Folding is free of consequence: a minimum-image
/// separation is invariant when either endpoint moves by a lattice vector, so
/// indices and distances are what they would have been. The
/// range on a periodic axis is `n_k = ceil(r / d_k)`, where `d_k` is the
/// **perpendicular plane spacing** [`SimBox::nearest_plane_distance`] and `r`
/// is the reach being probed, giving `2·n_k + 1` shifts on that axis:
///
/// - `r ≤ d_k` (the typical MD case): `n_k = 1`, so `3` shifts per periodic
///   axis and `27` tree queries per particle in a fully periodic 3-D box.
/// - `r > d_k`: `n_k ≥ 2`, so `125` or more.
/// - A non-periodic axis contributes only the zero shift, so a fully
///   free-boundary system needs exactly one tree query per particle.
///
/// **The range is sized from `d_k`, never from the lattice-vector length
/// `‖a_k‖`.** For a tilted cell `‖a_k‖` over-estimates the usable width
/// (`d_k ≤ ‖a_k‖`, with equality only when the cell is orthogonal along `k`),
/// so dividing the reach by `‖a_k‖` yields *fewer* images than are needed and
/// pairs go missing with nothing raised. `LinkCell` sizes its cells the same
/// way, for the same reason.
///
/// Duplicate hits across shifts are collapsed to the shortest separation, so a
/// point is reported once regardless of how many images found it.
///
/// This keeps memory bounded by the original `N` points (no
/// `O(N · n_images)` ghost copies) and aligns the PBC story with the rest
/// of `crate::core::neighbors`: every algorithm gets its periodicity from
/// `SimBox`, never from a ghost-expanded point set.
#[derive(Debug, Clone)]
pub struct AabbQuery {
    cutoff: F,
    bx: Option<SimBox>,
    tree: Bvh,
    stored_pos: Fnx3,
}

impl AabbQuery {
    /// Create a query with the given cutoff distance (Å).
    ///
    /// The tree itself is built later, by [`build`](Self::build). The
    /// cutoff is not validated here; a non-positive value is rejected by
    /// `build`, which panics.
    ///
    /// The cutoff is what the `Aabb` [`NeighborList`] backend searches at.
    /// [`query_knn`](Self::query_knn) ignores it: a k-nearest question has no
    /// radius and derives its own image-shift bound from the cell.
    ///
    /// [`NeighborList`]: crate::core::NeighborList
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            tree: Bvh::build(&[]),
            stored_pos: Fnx3::zeros((0, 3)),
        }
    }

    /// The cutoff distance (Å) this query was constructed with.
    pub fn cutoff(&self) -> F {
        self.cutoff
    }

    /// Enumerate lattice-image shifts whose magnitude could bring a tree point
    /// within `reach` of a query point.
    ///
    /// The range on periodic axis `k` is `-n_k ..= n_k` with
    /// `n_k = ceil(reach / d_k)`, where `d_k` is the perpendicular plane
    /// spacing from [`SimBox::nearest_plane_distance`]; non-PBC axes contribute
    /// only the zero shift. The counts multiply across axes, so a fully
    /// periodic box with `reach ≤ min_k d_k` yields `3³ = 27` shifts.
    ///
    /// Sizing from `d_k` rather than the lattice-vector length `‖a_k‖` is
    /// load-bearing: `d_k ≤ ‖a_k‖` for every cell, so edge-length sizing
    /// under-counts images on a tilted cell and silently drops pairs.
    fn enumerate_shifts(bx: &SimBox, reach: F) -> Vec<[F; 3]> {
        let pbc = bx.pbc();
        let d = bx.nearest_plane_distance();
        let range = |width: F, periodic: bool| -> (i32, i32) {
            if !periodic || width <= 0.0 {
                (0, 0)
            } else {
                let n = (reach / width).ceil() as i32;
                (-n, n)
            }
        };
        let (nxn, nxp) = range(d[0], pbc[0]);
        let (nyn, nyp) = range(d[1], pbc[1]);
        let (nzn, nzp) = range(d[2], pbc[2]);

        // `lattice` reads the columns of H for both box kinds, so the shift is
        // built the same way for orthorhombic and triclinic cells.
        let a = [bx.lattice(0), bx.lattice(1), bx.lattice(2)];
        let mut shifts: Vec<[F; 3]> = Vec::new();
        for ix in nxn..=nxp {
            for iy in nyn..=nyp {
                for iz in nzn..=nzp {
                    shifts.push([
                        ix as F * a[0][0] + iy as F * a[1][0] + iz as F * a[2][0],
                        ix as F * a[0][1] + iy as F * a[1][1] + iz as F * a[2][1],
                        ix as F * a[0][2] + iy as F * a[1][2] + iz as F * a[2][2],
                    ]);
                }
            }
        }
        shifts
    }

    /// Circumradius of the cell: an upper bound on any minimum-image
    /// separation, for a query that carries no radius of its own.
    ///
    /// The minimum-image representative of a displacement lies in the centred
    /// cell `{ Σ f_k a_k : |f_k| ≤ ½ }`, a convex polytope whose vertices are
    /// `½ Σ_k s_k a_k` over `s ∈ {−1,+1}³`. The Euclidean norm is convex, so it
    /// attains its maximum over that polytope at a vertex — half the longest
    /// body diagonal. Four sign patterns suffice; the other four are global
    /// negations with the same norm.
    ///
    /// For an orthogonal cell every combination has norm `√(Lx²+Ly²+Lz²)`, so
    /// this collapses to the familiar `diag/2`. For a tilted cell
    /// `√(Σ‖a_k‖²)/2` is **not** an upper bound — it can fall below the true
    /// circumradius, which would let a descent miss the true nearest neighbour.
    fn circumradius(bx: &SimBox) -> F {
        let a = [bx.lattice(0), bx.lattice(1), bx.lattice(2)];
        let mut best = 0.0_f64;
        for &(s1, s2, s3) in &[
            (1.0, 1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ] {
            let v = [
                s1 * a[0][0] + s2 * a[1][0] + s3 * a[2][0],
                s1 * a[0][1] + s2 * a[1][1] + s3 * a[2][1],
                s1 * a[0][2] + s2 * a[1][2] + s3 * a[2][2],
            ];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if n > best {
                best = n;
            }
        }
        0.5 * best
    }

    /// Find the `k` nearest neighbors of `query` in the most-recent build,
    /// honoring PBC via the same image-shift enumeration as
    /// [`build`](Self::build).
    ///
    /// `query` is a point in Cartesian coordinates (Å) and need not be one of
    /// the indexed points. Returns up to `k` pairs `(j_index, mic_dist_sq)`,
    /// where `mic_dist_sq` is the squared minimum-image distance in Å², sorted
    /// ascending by distance. Fewer than `k` entries may come back if the system
    /// has fewer than `k` points, if `k = 0`, or if nothing has been built yet —
    /// in which case the result is empty rather than an error.
    ///
    /// The cutoff plays no role here: a k-nearest-neighbor question has no
    /// radius, so the image-shift enumeration uses the cell's circumradius —
    /// half its longest body diagonal, an upper bound on any minimum-image
    /// separation — and therefore cannot miss a candidate.
    pub fn query_knn(&self, query: [F; 3], k: usize) -> Vec<(u32, F)> {
        if k == 0 || self.stored_pos.nrows() == 0 {
            return Vec::new();
        }
        let bx = self.bx.as_ref().expect("query_knn before build");
        // Fold the query into the cell for the same reason `build` folds the
        // points: the shift range is derived from the cell, so both endpoints
        // must live there. MIC distances are unaffected.
        let query = bx.wrap_row(query);
        // A k-nearest question carries no cutoff, so the reach is the cell's
        // circumradius — an upper bound on any minimum-image separation.
        let shifts = Self::enumerate_shifts(bx, Self::circumradius(bx));

        // Every (point, image) hit, reduced below to the nearest image of
        // each point. A flat Vec sorted once beats a map keyed per hit: the
        // candidate count is `k × n_shifts`, small and known.
        let mut found: Vec<(u32, F)> = Vec::with_capacity(k * shifts.len());
        let pts = self.stored_pos.view();

        for shift in &shifts {
            let shifted = [
                query[0] + shift[0],
                query[1] + shift[1],
                query[2] + shift[2],
            ];
            let candidates = self.tree.knn(&shifted, k, |i| {
                let q = [
                    pts[[i as usize, 0]] - shifted[0],
                    pts[[i as usize, 1]] - shifted[1],
                    pts[[i as usize, 2]] - shifted[2],
                ];
                (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
            });
            for &(j, _) in &candidates {
                let r_j = [
                    pts[[j as usize, 0]],
                    pts[[j as usize, 1]],
                    pts[[j as usize, 2]],
                ];
                let dr = bx.mic().apply(crate::op::vec3::sub(r_j, query));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                found.push((j, d2));
            }
        }
        // Nearest image per point, then by distance with the point index as
        // the tie-break — so equal distances come out in the same order on
        // every run.
        found.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
        found.dedup_by_key(|c| c.0);
        found.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        found.truncate(k);
        found
    }

    /// Build the bounding-volume hierarchy over `points` in `bx`.
    ///
    /// `points` is an `N × 3` view of Cartesian coordinates (Å); `bx` supplies
    /// the periodicity used for minimum-image distances. Both are retained, so
    /// [`query_knn`](Self::query_knn) can answer against them.
    ///
    /// No pairs are enumerated and no table is allocated: this indexes, and
    /// [`query_knn`](Self::query_knn) is the only question the index answers.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    pub fn build<'a>(&mut self, points: impl Into<Points<'a>>, bx: &SimBox) {
        self.index(super::checked(points.into()), bx);
    }

    /// [`build`](Self::build) on points already checked to be `N × 3`.
    fn index(&mut self, points: Points<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        // Fold into the primary cell first. The image-shift range is bounded by
        // the cell's own geometry, so it can only reach the images of a point
        // that already lies in (or beside) the cell — a point several cells out
        // would need a range nothing here derives. Wrapping costs nothing the
        // answer depends on: the minimum-image separation is invariant under
        // shifting either endpoint by a lattice vector, so indices and MIC
        // distances are unchanged. Non-periodic axes are left alone by `wrap`.
        let (wrapped, _) = match points {
            Points::Rows(rows) => bx.wrap(rows),
            Points::Columns(_) => bx.wrap(points.to_rows().view()),
        };
        // One box per point: a BVH over degenerate boxes is the point tree,
        // and it is the same tree the mesh and sphere-union regions descend.
        let boxes: Vec<([F; 3], [F; 3])> = (0..wrapped.nrows())
            .map(|i| {
                let p = [wrapped[[i, 0]], wrapped[[i, 1]], wrapped[[i, 2]]];
                (p, p)
            })
            .collect();
        self.tree = Bvh::build(&boxes);
        self.bx = Some(bx.clone());
        self.stored_pos = wrapped;
    }
}

impl Backend for AabbQuery {
    fn cutoff(&self) -> F {
        self.cutoff
    }

    fn build_index(&mut self, points: Points<'_>, bx: &SimBox) {
        self.index(points, bx);
    }

    fn update_index(&mut self, points: Points<'_>, bx: &SimBox) {
        self.index(points, bx);
    }

    /// Half-shell cutoff traversal over the tree.
    ///
    /// For each point `i` the tree is descended once per lattice-image shift
    /// that the cutoff can reach (`enumerate_shifts`), which is how this
    /// backend gets periodicity without ever materialising a ghost point. A
    /// pair can surface under more than one shift — two points can be within
    /// the cutoff of each other through two different images when the cutoff
    /// approaches half the cell width — so the hits for one `i` are collapsed
    /// before anything is emitted, and each unordered pair reaches the visitor
    /// exactly once with the canonical minimum-image displacement.
    ///
    /// Without a prior index there is no box to fold against and the visitor is
    /// never called, which is the [`Backend`] contract.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let Some(bx) = &self.bx else {
            return;
        };
        let n = self.stored_pos.nrows();
        if n == 0 {
            return;
        }
        let cutoff2 = self.cutoff * self.cutoff;
        let shifts = Self::enumerate_shifts(bx, self.cutoff);
        let pts = self.stored_pos.view();
        let mut hits: Vec<u32> = Vec::new();

        for i in 0..n {
            let pi = [pts[[i, 0]], pts[[i, 1]], pts[[i, 2]]];
            hits.clear();
            for shift in &shifts {
                let probe = [pi[0] + shift[0], pi[1] + shift[1], pi[2] + shift[2]];
                self.tree.for_each_within(
                    &probe,
                    self.cutoff,
                    |t| {
                        let t = t as usize;
                        let q = [
                            pts[[t, 0]] - probe[0],
                            pts[[t, 1]] - probe[1],
                            pts[[t, 2]] - probe[2],
                        ];
                        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
                    },
                    |t, _| {
                        // Half shell: the pair is owned by its lower index.
                        if t as usize > i {
                            hits.push(t);
                        }
                    },
                );
            }
            hits.sort_unstable();
            hits.dedup();
            for &j in &hits {
                let pj = [
                    pts[[j as usize, 0]],
                    pts[[j as usize, 1]],
                    pts[[j as usize, 2]],
                ];
                let dr = bx.mic().apply(crate::op::vec3::sub(pj, pi));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    visitor.visit_pair(i as u32, j, d2, dr);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    fn cube_bx(l: F, pbc: [bool; 3]) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], pbc).unwrap()
    }

    /// A strongly tilted cell whose three lattice vectors are all exactly
    /// 10 Å long, but whose plane spacings are not: `d = [8, 8, 10]`.
    ///
    /// `a1 = (10,0,0)`, `a2 = (6,8,0)`, `a3 = (0,0,10)`; `V = 800 Å³`,
    /// `‖a2 x a3‖ = ‖(80,-60,0)‖ = 100` so `d_1 = 8`, `‖a3 x a1‖ = 100` so
    /// `d_2 = 8`, `‖a1 x a2‖ = 80` so `d_3 = 10`. Sizing an image range from
    /// the lengths instead of the spacings therefore divides by 10 where it
    /// should divide by 8 — the under-count this cell exists to catch.
    fn tilted_bx() -> SimBox {
        let h = [[10.0_f64, 6.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]];
        SimBox::new(Cell::Matrix(h), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap()
    }

    /// The image range comes from the plane spacings, not the lattice-vector
    /// lengths. On [`tilted_bx`] at reach 9.0 the correct ranges are
    /// `ceil(9/8), ceil(9/8), ceil(9/10) = 2, 2, 1` — 75 shifts. Sizing from
    /// `lengths() = [10, 10, 10]` gives `1, 1, 1` — 27 shifts — and drops the
    /// `|i| = 2` rings entirely.
    #[test]
    fn shift_range_sizes_from_plane_distance_not_lengths() {
        let bx = tilted_bx();

        // The premise, checked two ways: the spacings are what the derivation
        // says, and they are strictly below the lattice-vector lengths.
        let d = bx.nearest_plane_distance();
        let l = bx.lengths();
        for (k, (&dk, &lk)) in d.iter().zip(l.iter()).enumerate() {
            let a_i = bx.lattice((k + 1) % 3);
            let a_j = bx.lattice((k + 2) % 3);
            let cross = crate::op::vec3::cross(
                crate::op::to_vec3(a_i.view()),
                crate::op::to_vec3(a_j.view()),
            );
            let hand = bx.volume() / crate::op::vec3::norm(cross);
            assert!((dk - hand).abs() < 1e-12, "axis {k}: {dk} vs {hand}");
            assert!(dk <= lk + 1e-12, "axis {k}: d {dk} exceeds |a| {lk}");
        }
        assert!((d[0] - 8.0).abs() < 1e-12);
        assert!((d[1] - 8.0).abs() < 1e-12);
        assert!((d[2] - 10.0).abs() < 1e-12);
        assert!((l[0] - 10.0).abs() < 1e-12);
        assert!((l[1] - 10.0).abs() < 1e-12);

        assert_eq!(
            AabbQuery::enumerate_shifts(&bx, 9.0).len(),
            75,
            "reach 9.0 needs ranges 2/2/1 = 75 shifts; sizing from lengths gives 27"
        );
    }

    /// The k-NN reach is the cell circumradius — half the *longest body
    /// diagonal*, maximised over sign combinations — not `sqrt(sum |a_k|^2)/2`.
    ///
    /// On [`tilted_bx`] the maximum is at `(+,+,+)`: `a1+a2+a3 = (16,8,10)`,
    /// `‖.‖ = sqrt(420)`, so the circumradius is `sqrt(420)/2 = 10.2470`. The
    /// half-diagonal of the edge lengths gives `sqrt(300)/2 = 8.6603`, which is **below**
    /// the true bound — an under-estimate, so a descent could miss the true
    /// nearest neighbour.
    #[test]
    fn circumradius_is_half_longest_body_diagonal() {
        let bx = tilted_bx();
        let r = AabbQuery::circumradius(&bx);
        assert!(
            (r - (420.0_f64).sqrt() / 2.0).abs() < 1e-12,
            "circumradius {r}, expected sqrt(420)/2"
        );
        let edge_half_diagonal = {
            let l = bx.lengths();
            (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt() / 2.0
        };
        assert!(
            r > edge_half_diagonal,
            "the edge-length half-diagonal {edge_half_diagonal} must be an under-estimate of {r}"
        );

        // On an orthogonal cell every sign pattern has the same norm, so the
        // circumradius is exactly the edge-length half-diagonal.
        let cube = cube_bx(10.0, [true; 3]);
        assert!((AabbQuery::circumradius(&cube) - (300.0_f64).sqrt() / 2.0).abs() < 1e-12);
    }

    /// The consequence, observed through the public return value: on a tilted
    /// cell `query_knn` must agree with a brute-force minimum-image scan.
    ///
    /// The circumradius and the shift range have no public surface of their
    /// own, so this is the assertion that actually protects a caller.
    #[test]
    fn knn_matches_brute_force_triclinic() {
        let bx = tilted_bx();
        // A deterministic spread, including points close to every face.
        // Points inside the cell, plus points several cells outside it: the
        // tree is built on the coordinates exactly as given, and an
        // under-estimated reach loses its outermost ring of shifts, which only
        // shows up once a query sits far outside the primary cell.
        let pts: Fnx3 = array![
            [0.2_f64, 0.3, 0.4],
            [9.6, 0.5, 9.5],
            [5.0, 7.7, 5.0],
            [15.5, 7.9, 0.2],
            [3.1, 1.2, 9.8],
            [13.0, 6.0, 4.4],
            [1.0, 7.5, 0.1],
            [8.0, 2.0, 5.5],
            [37.4, 25.9, 31.2],
            [-24.6, -17.1, -22.8],
            [41.0, -23.3, 18.7],
            [-33.9, 30.2, -29.4],
        ];
        let mut q = AabbQuery::new(1.0);
        q.build(pts.view(), &bx);

        for qi in 0..pts.nrows() {
            let query = [pts[[qi, 0]], pts[[qi, 1]], pts[[qi, 2]]];
            let k = 3;
            let got = q.query_knn(query, k);

            let mut want: Vec<(u32, F)> = (0..pts.nrows())
                .map(|j| {
                    let r_j = [pts[[j, 0]], pts[[j, 1]], pts[[j, 2]]];
                    let dr = bx.mic().apply(crate::op::vec3::sub(r_j, query));
                    (j as u32, dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])
                })
                .collect();
            want.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            want.truncate(k);

            assert_eq!(got.len(), want.len(), "query {qi}");
            for (g, w) in got.iter().zip(want.iter()) {
                assert_eq!(g.0, w.0, "query {qi}: index");
                assert!((g.1 - w.1).abs() < 1e-12, "query {qi}: dist2");
            }
        }
    }

    /// The `Aabb` backend must produce the same pair *set* as the O(N²)
    /// oracle — that is the whole contract a backend owes. Checked on all
    /// three box kinds, because the image-shift enumeration is the part most
    /// likely to differ and it is the part that depends on the cell.
    #[test]
    fn aabb_backend_matches_brute_force() {
        use crate::core::{NeighborColumns, NeighborList};

        let pts: Fnx3 = array![
            [0.2_f64, 0.3, 0.4],
            [9.6, 0.5, 9.5],
            [5.0, 7.7, 5.0],
            [2.5, 7.9, 0.2],
            [3.1, 1.2, 9.8],
            [7.0, 6.0, 4.4],
            [1.0, 7.5, 0.1],
            [8.0, 2.0, 5.5],
            [4.4, 4.4, 4.4],
            [9.9, 9.9, 9.9],
        ];

        for (label, bx) in [
            ("cube pbc", cube_bx(10.0, [true; 3])),
            ("cube free", cube_bx(10.0, [false; 3])),
            ("mixed pbc", cube_bx(10.0, [true, false, true])),
            ("triclinic", tilted_bx()),
        ] {
            for &cutoff in &[1.5_f64, 3.0, 4.5] {
                let mut want = NeighborList::brute_force(cutoff).unwrap();
                want.build(pts.view(), &bx);
                let mut want_pairs: Vec<(u32, u32)> = want
                    .neighbors(NeighborColumns::INDICES_ONLY)
                    .query_point_indices()
                    .iter()
                    .copied()
                    .zip(
                        want.neighbors(NeighborColumns::INDICES_ONLY)
                            .point_indices()
                            .iter()
                            .copied(),
                    )
                    .collect();
                want_pairs.sort_unstable();

                let mut got = NeighborList::aabb(cutoff).unwrap();
                got.build(pts.view(), &bx);
                let got_tbl = got.neighbors(NeighborColumns::FULL);
                let mut got_pairs: Vec<(u32, u32)> = got_tbl
                    .query_point_indices()
                    .iter()
                    .copied()
                    .zip(got_tbl.point_indices().iter().copied())
                    .collect();
                got_pairs.sort_unstable();

                assert_eq!(
                    got_pairs, want_pairs,
                    "{label} cutoff {cutoff}: pair set differs from the oracle"
                );

                // Every pair is emitted once, and half-shell.
                let mut uniq = got_pairs.clone();
                uniq.dedup();
                assert_eq!(
                    uniq.len(),
                    got_pairs.len(),
                    "{label} cutoff {cutoff}: duplicate pair"
                );
                for (i, j) in &got_pairs {
                    assert!(i < j, "{label}: pair ({i},{j}) is not half-shell");
                }

                // And the geometry the visitor handed over is the canonical
                // minimum image, not whichever image the descent happened to
                // find the point through.
                let d2 = got_tbl.dist_sq().unwrap();
                let disp = got_tbl.disp().unwrap();
                for (row, (i, j)) in got_pairs.iter().enumerate() {
                    let pi = [
                        pts[[*i as usize, 0]],
                        pts[[*i as usize, 1]],
                        pts[[*i as usize, 2]],
                    ];
                    let pj = [
                        pts[[*j as usize, 0]],
                        pts[[*j as usize, 1]],
                        pts[[*j as usize, 2]],
                    ];
                    let mic = bx.mic().apply(crate::op::vec3::sub(pj, pi));
                    let mic2 = mic[0] * mic[0] + mic[1] * mic[1] + mic[2] * mic[2];
                    assert!((d2[row] - mic2).abs() < 1e-12, "{label}: dist2 row {row}");
                    for c in 0..3 {
                        assert!(
                            (disp[[row, c]] - mic[c]).abs() < 1e-12,
                            "{label}: disp row {row} comp {c}"
                        );
                    }
                }
            }
        }
    }

    /// An empty point set builds a valid (empty) tree, and a k-nearest-neighbor
    /// question against it answers "nothing" instead of panicking.
    #[test]
    fn empty_input_yields_empty_knn() {
        let pts: Fnx3 = ndarray::Array2::zeros((0, 3));
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        assert_eq!(aabb.query_knn([0.0, 0.0, 0.0], 3).len(), 0);
    }

    #[test]
    fn cutoff_below_box_size_enumerates_27_pbc_images() {
        // Tree is built on raw positions (no wrapping), so the +1 and −1
        // image shifts are needed to catch wrap-pairs near each boundary,
        // even when cutoff < L/2.
        let bx = cube_bx(10.0, [true; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 2.0);
        assert_eq!(shifts.len(), 27);
    }

    #[test]
    fn non_pbc_box_has_only_zero_shift() {
        let bx = cube_bx(10.0, [false; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 2.0);
        assert_eq!(shifts.len(), 1);
        assert_eq!(shifts[0], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn cutoff_above_full_box_enumerates_more_images() {
        // With cutoff > L, the second image ring is needed too.
        let bx = cube_bx(10.0, [true; 3]);
        let shifts = AabbQuery::enumerate_shifts(&bx, 12.0);
        // ceil(12/10) = 2 → [-2, 2] = 5 per axis → 125 total
        assert_eq!(shifts.len(), 125);
    }

    #[test]
    fn knn_finds_k_closest_on_line() {
        // Points at x = 0, 1, 2, 3, 4. Query at x = 0.2 with k = 3 should
        // return {0, 1, 2} sorted by distance.
        let pts = array![
            [0.0_f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];
        let bx = cube_bx(100.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.2, 0.0, 0.0], 3);
        assert_eq!(knn.len(), 3);
        assert_eq!(knn[0].0, 0);
        assert_eq!(knn[1].0, 1);
        assert_eq!(knn[2].0, 2);
        // Distances ascending: 0.2² < 0.8² < 1.8².
        assert!(knn[0].1 < knn[1].1);
        assert!(knn[1].1 < knn[2].1);
    }

    #[test]
    fn knn_with_pbc_finds_wrap_neighbor() {
        // x = 0.1 and 9.9 in a PBC box of length 10 → wrap distance 0.2.
        // Query at x = 0 should find 0 (distance 0.1) and 1 (wrap distance 0.1).
        let pts = array![[0.1_f64, 0.0, 0.0], [9.9, 0.0, 0.0]];
        let bx = cube_bx(10.0, [true, true, true]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.0, 0.0, 0.0], 2);
        assert_eq!(knn.len(), 2);
        // Both at distance² = 0.01 (approximately).
        for &(_, d2) in &knn {
            assert!(
                (d2 - 0.01).abs() < 1e-9,
                "expected wrap distance ~0.1; got d²={d2}"
            );
        }
    }

    #[test]
    fn knn_k_larger_than_n_returns_all() {
        let pts = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.5, 0.0, 0.0], 10);
        assert_eq!(knn.len(), 2);
    }

    #[test]
    fn knn_zero_k_returns_empty() {
        let pts = array![[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let bx = cube_bx(10.0, [false; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);
        let knn = aabb.query_knn([0.5, 0.0, 0.0], 0);
        assert_eq!(knn.len(), 0);
    }

    #[test]
    fn knn_matches_brute_force_random() {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(11);
        let n = 100;
        let mut pts = Fnx3::zeros((n, 3));
        for i in 0..n {
            pts[[i, 0]] = rng.random::<F>() * 10.0;
            pts[[i, 1]] = rng.random::<F>() * 10.0;
            pts[[i, 2]] = rng.random::<F>() * 10.0;
        }
        let bx = cube_bx(10.0, [true; 3]);
        let mut aabb = AabbQuery::new(1.0);
        aabb.build(pts.view(), &bx);

        let q = [5.0_f64, 5.0, 5.0];
        let k = 7;
        let aabb_knn = aabb.query_knn(q, k);

        // Brute-force reference.
        let mut bf: Vec<(u32, F)> = (0..n)
            .map(|j| {
                let r_j = [pts[[j, 0]], pts[[j, 1]], pts[[j, 2]]];
                let dr = bx.mic().apply(crate::op::vec3::sub(r_j, q));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                (j as u32, d2)
            })
            .collect();
        bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        bf.truncate(k);

        assert_eq!(aabb_knn.len(), bf.len());
        for (a, b) in aabb_knn.iter().zip(bf.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-12);
        }
    }
}
