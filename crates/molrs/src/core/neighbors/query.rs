//! Cross-query neighbor search, one point set against another: [`NeighborQuery`].

use super::{Backend, CutoffError, checked};
use crate::core::LinkCell;
use crate::core::{Cell, Points, SimBox};
use crate::core::{NeighborColumns, Neighbors, QueryMode};
use crate::op::{F, Fnx3, Fnx3View};

/// Cross-query search over a fixed set of reference points.
///
/// This is the door for the *cross* question — "which reference points lie
/// within the cutoff of **these other** points?" — and that is the reason the
/// type exists. Searching one point set against itself is a different question
/// and belongs to [`NeighborList`](crate::core::NeighborList),
/// which additionally can re-index moved coordinates, stream its pairs without
/// materializing a table, and choose which columns a table keeps.
///
/// Wraps a [`LinkCell`] cell-list spatial index built once at construction, so
/// that many queries can be answered against the same reference points without
/// re-indexing.
///
/// - [`query`](NeighborQuery::query) — cross-query: all pairs `(i, j)` where `i`
///   indexes the query points and `j` the reference points. Directed and
///   full-shell.
/// - [`query_self`](NeighborQuery::query_self) — self-query: unique pairs
///   `(i, j)` with `i < j` within the reference points. Half-shell; a
///   convenience over the same reference points.
///
/// Every returned [`Neighbors`] table carries both physical columns
/// ([`NeighborColumns::FULL`]): distances in Å² and minimum-image
/// displacements in Å. There is no column policy here — a caller who wants a
/// leaner table uses
/// [`NeighborList::neighbors`](crate::core::NeighborList::neighbors), or drops
/// columns afterwards with [`Neighbors::repack`].
///
/// # Example
///
/// ```ignore
/// let nq = NeighborQuery::new(&simbox, points.view(), 3.0)?;
/// let cross = nq.query(query_points.view());  // the question this type is for
/// let selfq = nq.query_self();                // convenience; prefer NeighborList
/// ```
#[derive(Debug, Clone)]
pub struct NeighborQuery {
    /// The underlying cell-list spatial index (built once at construction).
    lc: LinkCell,
    /// Copy of the reference points (owned for self-query).
    points: Fnx3,
    /// Copy of the simulation box.
    simbox: SimBox,
}

impl NeighborQuery {
    /// Build a spatial index from reference points.
    ///
    /// `points` are the Cartesian reference coordinates (Å) — an `N × 3`
    /// array (the view converts) or three `x`/`y`/`z` columns
    /// ([`Points::Columns`]); `simbox` supplies the periodicity used for
    /// minimum-image distances, and `cutoff` (Å) is the interaction radius
    /// every later query uses — it is fixed here, not per query, because it
    /// sets the cell size of the index.
    ///
    /// # Errors
    /// [`CutoffError`] if `cutoff` is not a positive finite length.
    ///
    /// # Panics
    /// Panics if `points` is not `N × 3`.
    pub fn new<'a>(
        simbox: &SimBox,
        points: impl Into<Points<'a>>,
        cutoff: F,
    ) -> Result<Self, CutoffError> {
        let cutoff = super::checked_cutoff(cutoff)?;
        let points = checked(points.into());

        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build_index(points, simbox);

        Ok(Self {
            lc,
            // Owned interleaved copy for self-query and `points()`.
            points: points.to_rows(),
            simbox: simbox.clone(),
        })
    }

    /// Build from free-boundary points (no periodic box).
    ///
    /// "Free boundary" means the system is an isolated cluster rather than a
    /// periodically repeated cell, so no pair may wrap around an edge. The
    /// constructor derives a non-periodic bounding box from the point cloud
    /// ([`SimBox::bounding`]), padded by `cutoff` (Å) on every side so that
    /// every particle sits comfortably inside it and no minimum-image folding
    /// can occur.
    ///
    /// An empty point set is not an error: it has no bounds, so it is
    /// indexed in a non-periodic cube of edge `cutoff` at the origin, and its
    /// queries return no pairs.
    ///
    /// # Errors
    /// [`CutoffError`] if `cutoff` is not a positive finite length.
    ///
    /// # Panics
    /// Panics if `points` is not `N × 3`.
    pub fn unbounded<'a>(points: impl Into<Points<'a>>, cutoff: F) -> Result<Self, CutoffError> {
        let cutoff = super::checked_cutoff(cutoff)?;
        let points = checked(points.into());
        let bx = if points.is_empty() {
            SimBox::new(Cell::Lengths([cutoff; 3]), [0.0; 3], [false; 3])
        } else {
            SimBox::bounding(points, [cutoff; 3], [false; 3])
        }
        .expect("a cutoff-padded bounding box of N x 3 points has positive extent");
        Self::new(&bx, points, cutoff)
    }

    /// Cross-query: find all pairs `(i, j)` where `i` indexes `query_points`
    /// (an `N × 3` array or [`Points::Columns`], Å) and `j` indexes the
    /// reference `points`.
    ///
    /// Results are full-shell, not half-shell: every query point reports all of
    /// its reference neighbors, with no `i < j` constraint. If `query_points`
    /// holds the same coordinates as the reference set, each point therefore
    /// finds itself at distance zero and each pair appears in both orderings —
    /// use [`query_self`](Self::query_self) when you want each unordered pair
    /// once instead.
    ///
    /// The returned table is tagged [`QueryMode::CrossQuery`] with both point
    /// counts, and carries both physical columns
    /// ([`NeighborColumns::FULL`]): `dist_sq` in Å², `disp` in Å.
    ///
    /// # Panics
    /// Panics if `query_points` is not `N × 3`.
    pub fn query<'a>(&self, query_points: impl Into<Points<'a>>) -> Neighbors {
        let query_points = checked(query_points.into());
        let n_query = query_points.len();
        let cutoff_sq = self.lc.cutoff * self.lc.cutoff;

        let mut nlist = Neighbors::empty(
            QueryMode::CrossQuery {
                n_query_points: n_query,
                n_points: self.points.nrows(),
            },
            NeighborColumns::FULL,
        );

        for qi in 0..n_query {
            self.lc.visit_neighbors_of_pt(
                query_points.point(qi),
                &self.simbox,
                |rj, dist_sq, diff| {
                    if dist_sq <= cutoff_sq {
                        nlist.push(qi as u32, rj, dist_sq, diff);
                    }
                },
            );
        }

        nlist.simbox = Some(self.simbox.clone());
        nlist
    }

    /// Self-query: find unique pairs `(i, j)` with `i < j` within the
    /// reference point set.
    ///
    /// This is the standard half-shell neighbor list: each unordered pair is
    /// reported exactly once and no point pairs with itself. The returned table
    /// is tagged `QueryMode::SelfQuery { n_points }` with the size of the
    /// reference set, and carries both physical columns
    /// ([`NeighborColumns::FULL`]).
    ///
    /// Reuses the index this object already holds — the cell index is the same
    /// for both traversals; only the walk differs (forward half-shell here,
    /// full per-point neighborhood for cross), so no re-sort happens.
    ///
    /// A caller that only ever asks this question should hold a
    /// [`super::NeighborList`] instead: it can re-index moved coordinates with
    /// [`update`](super::NeighborList::update), stream pairs without materializing a
    /// table, and choose which columns a table keeps.
    pub fn query_self(&self) -> Neighbors {
        let mut out = Neighbors::empty(
            QueryMode::SelfQuery {
                n_points: self.points.nrows(),
            },
            NeighborColumns::FULL,
        );
        self.lc.materialize_into(&mut out);
        out.simbox = Some(self.simbox.clone());
        out
    }

    /// Reference to the stored simulation box.
    pub fn simbox(&self) -> &SimBox {
        &self.simbox
    }

    /// Reference to the stored reference points (`N × 3`, Å).
    pub fn points(&self) -> Fnx3View<'_> {
        self.points.view()
    }

    /// The cutoff distance (Å) fixed at construction — read from the index,
    /// which is its single owner.
    pub fn cutoff(&self) -> F {
        self.lc.cutoff
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SimBox;
    use ndarray::array;

    #[test]
    fn self_query_matches_linkcell() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];

        let nq = NeighborQuery::new(&bx, pts.view(), 0.5).unwrap();
        let nlist = nq.query_self();

        assert_eq!(nlist.mode(), QueryMode::SelfQuery { n_points: 3 });
        assert_eq!(nlist.n_pairs(), 1);
        assert_eq!(nlist.query_point_indices()[0], 0);
        assert_eq!(nlist.point_indices()[0], 1);
        assert_eq!(nlist.n_points(), 3);
        assert_eq!(nlist.n_query_points(), 3);
    }

    #[test]
    fn cross_query_finds_all_neighbors() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let ref_pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let query_pts = array![[0.5, 0.0, 0.0]];

        let nq = NeighborQuery::new(&bx, ref_pts.view(), 0.6).unwrap();
        let nlist = nq.query(query_pts.view());

        assert_eq!(
            nlist.mode(),
            QueryMode::CrossQuery {
                n_query_points: 1,
                n_points: 3,
            }
        );
        assert_eq!(nlist.n_query_points(), 1);
        assert_eq!(nlist.n_points(), 3);
        // query point at 0.5 is within 0.6 of ref points 0 (at 0.0) and 1 (at 1.0)
        assert_eq!(nlist.n_pairs(), 2);
    }

    /// The table stores \(d^2\); a caller that wants a distance takes the square
    /// root itself (there is no `distances()` accessor to hide the cost or to
    /// return an empty slice when the column is absent).
    #[test]
    fn dist_sq_square_roots_to_distance() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = NeighborQuery::new(&bx, pts.view(), 1.5).unwrap();
        let nlist = nq.query_self();

        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("self-query materializes dist_sq");
        assert!((d2[0] - 1.0).abs() < 1e-6);
        assert!((d2[0].sqrt() - 1.0).abs() < 1e-6);
    }

    /// `disp()` hands back the stored MIC displacement \(r_j - r_i\)
    /// (unnormalized), or `None` when the column was never stored.
    #[test]
    fn disp_returns_stored_displacements() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = NeighborQuery::new(&bx, pts.view(), 1.5).unwrap();
        let nlist = nq.query_self();

        let disp = nlist.disp().expect("self-query materializes disp");
        assert_eq!(disp.nrows(), 1);
        assert_eq!(disp.ncols(), 3);
        assert!((disp[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(disp[[0, 1]].abs() < 1e-6);
        assert!(disp[[0, 2]].abs() < 1e-6);
    }

    #[test]
    fn self_query_pbc_boundary() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]];

        let nq = NeighborQuery::new(&bx, pts.view(), 0.5).unwrap();
        let nlist = nq.query_self();

        assert_eq!(nlist.n_pairs(), 1);
    }

    #[test]
    fn cross_query_self_overlap_produces_full_shell() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];

        // Cross-query with the same point set should produce pairs in both
        // directions (0->1 and 1->0), plus self-pairs (0->0, 1->1), unlike
        // self-query which only produces (0, 1) once.
        let nq = NeighborQuery::new(&bx, pts.view(), 0.6).unwrap();
        let nlist = nq.query(pts.view());

        // Each query point finds both reference points (including itself at dist=0)
        // Total: 2 query x 2 ref = 4 pairs
        assert_eq!(nlist.n_pairs(), 4);

        // Compare with self-query which only produces unique pairs (i<j)
        let self_nlist = nq.query_self();
        assert_eq!(self_nlist.n_pairs(), 1);
    }

    #[test]
    fn free_boundary_self_query() {
        let pts = array![[0.0 as F, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 10.0, 10.0],];
        let nq = NeighborQuery::unbounded(pts.view(), 1.0).unwrap();
        let nlist = nq.query_self();

        // Only pts[0] and pts[1] are within cutoff=1.0
        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("self-query materializes dist_sq");
        assert!((d2[0].sqrt() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn free_boundary_cross_query() {
        let ref_pts = array![[0.0 as F, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0],];
        let query_pts = array![[0.3 as F, 0.0, 0.0]];

        let nq = NeighborQuery::unbounded(ref_pts.view(), 0.5).unwrap();
        let nlist = nq.query(query_pts.view());

        // query point at 0.3 is within 0.5 of ref[0] (dist=0.3) but not ref[1] (dist=0.7)
        assert_eq!(nlist.n_pairs(), 1);
    }

    #[test]
    fn free_boundary_no_wrap() {
        // Points far apart — should NOT be neighbors (no PBC wrapping)
        let pts = array![[0.0 as F, 0.0, 0.0], [5.0, 5.0, 5.0],];
        let nq = NeighborQuery::unbounded(pts.view(), 1.0).unwrap();
        let nlist = nq.query_self();
        assert_eq!(nlist.n_pairs(), 0);
    }

    // --- SoA cross-query is bit-identical to the Array2 cross-query ---

    /// Split an `Array2` (N×3) into three column vectors (SoA layout).
    fn columns(pts: &ndarray::Array2<F>) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = pts.nrows();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        for i in 0..n {
            xs.push(pts[[i, 0]]);
            ys.push(pts[[i, 1]]);
            zs.push(pts[[i, 2]]);
        }
        (xs, ys, zs)
    }

    /// Bitwise (not approximate) equality of two neighbor lists.
    fn assert_bitwise_equal(a: &Neighbors, b: &Neighbors) {
        assert_eq!(a.n_pairs(), b.n_pairs(), "n_pairs differ");
        let d2a = a.dist_sq().expect("query materializes dist_sq");
        let d2b = b.dist_sq().expect("query materializes dist_sq");
        let da = a.disp().expect("query materializes disp");
        let db = b.disp().expect("query materializes disp");
        for k in 0..a.n_pairs() {
            assert_eq!(
                a.query_point_indices()[k],
                b.query_point_indices()[k],
                "idx_i"
            );
            assert_eq!(a.point_indices()[k], b.point_indices()[k], "idx_j");
            assert_eq!(d2a[k], d2b[k], "dist_sq bitwise");
            for d in 0..3 {
                assert_eq!(da[[k, d]], db[[k, d]], "disp[{}] bitwise", d);
            }
        }
    }

    #[test]
    fn columns_query_matches_query_bitwise() {
        // Periodic cube fixture.
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let refp = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
            [2.2, 8.1, 3.3],
        ];
        let qp = array![[1.2, 1.0, 1.0], [5.1, 5.0, 5.0], [9.9, 1.1, 1.0],];
        let (rx, ry, rz) = columns(&refp);
        let (qx, qy, qz) = columns(&qp);

        let nq_a = NeighborQuery::new(&bx, refp.view(), 2.0).unwrap();
        let nl_a = nq_a.query(qp.view());

        let nq_s = NeighborQuery::new(&bx, Points::Columns([&rx, &ry, &rz]), 2.0).unwrap();
        let nl_s = nq_s.query(Points::Columns([&qx, &qy, &qz]));

        assert!(nl_a.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&nl_a, &nl_s);

        // Free / non-periodic fixture (query points overlap the reference set).
        let nq_af = NeighborQuery::unbounded(refp.view(), 2.0).unwrap();
        let nl_af = nq_af.query(qp.view());

        let nq_sf = NeighborQuery::unbounded(Points::Columns([&rx, &ry, &rz]), 2.0).unwrap();
        let nl_sf = nq_sf.query(Points::Columns([&qx, &qy, &qz]));

        assert!(nl_af.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&nl_af, &nl_sf);
    }
}
