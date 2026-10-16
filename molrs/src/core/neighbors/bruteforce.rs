//! O(N^2) brute-force neighbor search — reference implementation: [`BruteForce`].

use super::{Backend, PairVisitor};
use crate::core::{Points, SimBox};
use crate::op::{F, Fnx3};

/// Brute-force O(N^2) neighbor search — the reference
/// [`NeighborList`](crate::core::NeighborList) backend.
///
/// Iterates over all unique pairs and keeps those within the cutoff. Search
/// from outside this crate goes through
/// [`NeighborList::brute_force`](crate::core::NeighborList::brute_force),
/// which drives this type.
///
/// It stores coordinates, not pairs: every traversal rescans them, so there is
/// no cached table to go stale and no column policy to configure. Which columns
/// a materialized table keeps is named by the caller at
/// [`NeighborList::neighbors`](crate::core::NeighborList::neighbors),
/// exactly as it is for the cell list.
///
/// Checks every pair `(i, j)` with `i < j` — the half-shell convention, so each
/// unordered pair is examined exactly once. Useful for correctness testing
/// against the cell-list algorithm and for very small systems where the
/// overhead of cell construction is not worthwhile.
///
/// Because it consults the simulation box directly for every pair and shares no
/// cell-assignment code with [`LinkCell`](crate::core::LinkCell),
/// a defect in the cell partition cannot cancel against a defect here; that is
/// what makes it usable as an oracle in tests.
#[derive(Debug, Clone)]
pub struct BruteForce {
    /// Interaction cutoff distance (Å). Pairs are kept when their minimum-image
    /// separation is less than or equal to this.
    pub cutoff: F,
    /// Simulation box from the last index call.
    bx: Option<SimBox>,
    /// Stored positions for visit_pairs (set by update_index).
    stored_pos: Fnx3,
}

impl BruteForce {
    /// Create a new `BruteForce` with the given cutoff distance (Å).
    ///
    /// The cutoff is not validated here; a non-positive value is rejected by
    /// the first index call (`build_index` / `update_index`), which panics.
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            stored_pos: Fnx3::zeros((0, 3)),
        }
    }
}

impl Backend for BruteForce {
    /// Store positions and box without computing pairs.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    fn cutoff(&self) -> F {
        self.cutoff
    }

    fn build_index(&mut self, points: Points<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    /// Store positions and box without computing pairs.
    ///
    /// There is no spatial index to build for an all-pairs search, so this
    /// simply keeps a copy of the coordinates for `visit_pairs` to rescan.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    fn update_index(&mut self, points: Points<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.stored_pos = points.to_rows();
        self.bx = Some(bx.clone());
    }

    /// On-demand O(N^2) pair traversal using the stored positions.
    ///
    /// The visitor receives the same two quantities a table would store:
    /// `dist_sq` (Å²) and the minimum-image displacement `r_j - r_i` (Å), for
    /// every pair with `i < j` inside the cutoff.
    ///
    /// There is exactly one source of pairs — a fresh rescan of the
    /// coordinates the last `build_index` / `update_index` stored — so nothing
    /// here can hand back a stale answer. Without such a call there is no box
    /// to fold minimum images against, and the visitor is simply never called,
    /// which is the [`Backend`] contract for an unindexed backend.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let Some(bx) = &self.bx else {
            return;
        };
        let n = self.stored_pos.nrows();
        let cutoff2 = self.cutoff * self.cutoff;
        let mic = bx.mic();
        for i in 0..n {
            let pi = [
                self.stored_pos[[i, 0]],
                self.stored_pos[[i, 1]],
                self.stored_pos[[i, 2]],
            ];
            for j in (i + 1)..n {
                let pj = [
                    self.stored_pos[[j, 0]],
                    self.stored_pos[[j, 1]],
                    self.stored_pos[[j, 2]],
                ];
                let dr = mic.apply(crate::op::vec3::sub(pj, pi));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    visitor.visit_pair(i as u32, j as u32, d2, dr);
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

    struct Collect(Vec<(u32, u32, F, [F; 3])>);

    impl PairVisitor for Collect {
        fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]) {
            self.0.push((i, j, dist_sq, diff));
        }
    }

    fn pairs(bf: &BruteForce) -> Vec<(u32, u32, F, [F; 3])> {
        let mut out = Collect(Vec::new());
        bf.visit_pairs(&mut out);
        out.0
    }

    /// Three points on the x axis of a 10 Å cube: 0, 1 and 9.
    fn line() -> Fnx3 {
        array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]
    }

    fn cube(pbc: bool) -> SimBox {
        SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [pbc; 3]).unwrap()
    }

    #[test]
    fn visits_every_pair_within_the_cutoff_once_with_i_less_than_j() {
        let mut bf = BruteForce::new(1.5);
        bf.build_index(Points::Rows(line().view()), &cube(true));
        let got = pairs(&bf);
        // (0,1) directly and (0,2) through the periodic image; (1,2) is 2 Å apart.
        assert_eq!(got.len(), 2);
        assert_eq!(got[0], (0, 1, 1.0, [1.0, 0.0, 0.0]));
        assert_eq!(
            got[1],
            (0, 2, 1.0, [-1.0, 0.0, 0.0]),
            "minimum image, not the raw +9"
        );
    }

    #[test]
    fn the_cutoff_is_inclusive() {
        let mut bf = BruteForce::new(1.0);
        bf.build_index(Points::Rows(line().view()), &cube(true));
        assert_eq!(pairs(&bf).len(), 2, "d == cutoff is kept");
        let mut bf = BruteForce::new(0.999);
        bf.build_index(Points::Rows(line().view()), &cube(true));
        assert!(pairs(&bf).is_empty());
    }

    #[test]
    fn a_non_periodic_axis_does_not_wrap() {
        let mut bf = BruteForce::new(1.5);
        bf.build_index(Points::Rows(line().view()), &cube(false));
        let got = pairs(&bf);
        assert_eq!(got.len(), 1);
        assert_eq!((got[0].0, got[0].1), (0, 1));
    }

    #[test]
    fn nothing_is_visited_before_an_index_call() {
        let bf = BruteForce::new(1.5);
        assert!(pairs(&bf).is_empty());
    }

    #[test]
    #[should_panic(expected = "cutoff must be positive")]
    fn a_non_positive_cutoff_is_refused_at_index_time() {
        let mut bf = BruteForce::new(0.0);
        bf.build_index(Points::Rows(line().view()), &cube(true));
    }
}
