//! Cell-list neighbor search — O(N) with sorted-particle layout and half-shell iteration: [`LinkCell`].

use super::{Backend, PairVisitor};
use crate::core::Cell;
use crate::core::{CellGrid, Neighbors};
use crate::core::{Points, SimBox};
use crate::op::{F, Fnx3View};

/// Occupied-cell count from which materializing the pair table in parallel pays
/// for itself.
///
/// Below it rayon's split/merge overhead exceeds the pair work: at N = 1000 with
/// ρ = 0.8 and cutoff 2.5 Å there are ~10 occupied cells and the serial walk is
/// 30–50% faster. Above it the fold wins decisively — measured on a uniform
/// periodic system at cutoff 4 Å, serial → parallel materialization is
/// 10.1 ms → 4.2 ms at 20k particles (2.4×) and 110 ms → 26 ms at 100k (4.2×).
/// Empirical, and it gates only *how* the pairs are found: both sides of the
/// branch produce the same pair set.
#[cfg(feature = "rayon")]
const PAR_MIN_CELLS: usize = 64;

/// Cell-list neighbor search algorithm — the default
/// [`NeighborList`](crate::core::NeighborList) backend.
///
/// Partitions space into a regular grid of cells whose width >= cutoff, then
/// searches only neighboring cells for pair interactions.  Uses half-shell
/// iteration so each pair is found exactly once.
///
/// This type owns the spatial index and the pair traversal, and nothing else:
/// it caches no pair table and carries no column policy. What it produces is a
/// half-shell self list — every pair satisfies `i < j` — and deciding what
/// becomes of those pairs belongs to the engine, which either streams them
/// ([`NeighborList::for_each_pair`](crate::core::NeighborList::for_each_pair))
/// or writes them into a [`Neighbors`] table whose columns the *caller* names
/// ([`NeighborList::neighbors`](crate::core::NeighborList::neighbors)).
///
/// With the `rayon` feature (default), materialization is parallelized across
/// occupied cells via `rayon::par_iter`.
///
/// Search from outside this crate goes through
/// [`NeighborList::new`](crate::core::NeighborList::new), which
/// drives this type.
///
/// # Algorithm
///
/// A *cell list* divides the simulation box into a grid of cells at least one
/// cutoff wide. A particle can then only have neighbors in its own cell and the
/// 26 cells touching it, which turns the O(N²) all-pairs scan into an O(N) one.
///
/// Particles are *counting-sorted* by cell index — a linear-time sort that
/// works by tallying how many particles fall in each cell and then scattering
/// them into the resulting offsets — so that all particles in the same cell
/// occupy a contiguous slice of `sorted_idx` / `sorted_pos`. This gives
/// excellent cache locality during the pair search compared to a linked-list
/// layout.
///
/// Only **occupied cells** are visited during pair search, so sparse systems
/// (few particles, many cells) pay O(N), not O(n_cells).
///
/// *Half-shell* iteration means each cell only looks at a forward half of its
/// neighboring cells, so every unordered pair is discovered exactly once and
/// the resulting table satisfies `i < j`.
#[derive(Debug, Clone)]
pub struct LinkCell {
    /// Interaction cutoff distance (Å). Also sets the minimum cell width, so
    /// changing it invalidates any previously built index.
    pub cutoff: F,
    /// Cell partition of the box: dimensions + per-axis periodicity.
    grid: CellGrid,
    /// `cell_start[c]` = index into `sorted_idx` where cell `c` begins.
    /// Length = n_cells + 1 (sentinel at end).
    cell_start: Vec<u32>,
    /// Original particle indices, sorted by cell assignment.
    sorted_idx: Vec<u32>,
    /// Particle positions in sorted order, flat `[x0,y0,z0, x1,y1,z1, ...]`.
    /// Stored as raw `Vec<F>` (not `Array2`) to eliminate per-row view overhead
    /// in the tight pair loop.
    sorted_pos: Vec<F>,
    /// Indices of non-empty cells — pair search only iterates these.
    occupied_cells: Vec<u32>,
    /// Simulation box from the last build/update.
    bx: SimBox,
    /// Reusable cursor buffer for counting-sort scatter (avoids allocation).
    cursor: Vec<u32>,
    /// Reusable cell-assignment buffer (avoids cloning sorted_idx).
    cell_of: Vec<u32>,
}

impl Default for LinkCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkCell {
    /// Create a new `LinkCell` with zero cutoff (must be set via [`cutoff`](Self::cutoff)).
    ///
    /// The index starts empty; the first `build_index` fills it.
    pub fn new() -> Self {
        Self {
            cutoff: 0.0,
            grid: CellGrid::with_dims([1; 3], [false; 3]),
            cell_start: Vec::new(),
            sorted_idx: Vec::new(),
            sorted_pos: Vec::new(),
            occupied_cells: Vec::new(),
            bx: SimBox::new(
                Cell::Lengths([1.0; 3]),
                [0.0 as F, 0.0, 0.0],
                [false, false, false],
            )
            .expect("dummy box"),
            cursor: Vec::new(),
            cell_of: Vec::new(),
        }
    }

    /// Set the cutoff distance in Å (builder pattern; consumes and returns
    /// `self`).
    ///
    /// Must end up positive before any build, which asserts on it.
    pub fn cutoff(mut self, cutoff: F) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Visit the reference points in the cell neighborhood of an arbitrary
    /// query point, read as a stack `[F; 3]`.
    ///
    /// The cross-query kernel: used by
    /// [`NeighborQuery`](super::NeighborQuery)'s query paths and the RDF cross
    /// stream. Calls `callback(ref_index, dist_sq, [dx, dy, dz])` with
    /// `dist_sq` in Å² and the minimum-image displacement `r_ref - r_query` in
    /// Å.
    ///
    /// The callback fires for **every** point in the query point's own cell and
    /// in its surrounding cells (up to 26 distinct ones, fewer when an axis has
    /// only one or two cells), including points beyond the cutoff: a cell is at
    /// least one cutoff wide, so its corners reach farther than that. Applying
    /// the cutoff test is the caller's job.
    pub(crate) fn visit_neighbors_of_pt<C>(&self, query_point: [F; 3], bx: &SimBox, mut callback: C)
    where
        C: FnMut(u32, F, [F; 3]),
    {
        if self.cell_start.is_empty() {
            return;
        }

        let query_cell = self.grid.cell_of(bx, query_point);
        let qp = query_point;
        let mic = bx.mic();

        // Check the query cell itself + all 26 neighbor cells
        let mut buf = [0usize; 27];
        let n_all = self.grid.stencil_all(query_cell, &mut buf);
        let all_cells = &buf[..n_all];
        for nc in std::iter::once(query_cell).chain(all_cells.iter().copied()) {
            let start = self.cell_start[nc] as usize;
            let end = self.cell_start[nc + 1] as usize;
            for si in start..end {
                let oj = self.sorted_idx[si];
                let pj = pos_at(&self.sorted_pos, si);
                let dr = mic.apply(crate::op::vec3::sub(pj, qp));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                callback(oj, d2, dr);
            }
        }
    }
}

impl Backend for LinkCell {
    fn cutoff(&self) -> F {
        self.cutoff
    }

    fn build_index(&mut self, points: Points<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    /// Sort the points into cells. Columns are sorted directly — no
    /// interleaved copy — through the same counting-sort core as rows, so the
    /// index is identical to the one the `N × 3` path produces.
    fn update_index(&mut self, points: Points<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        match points {
            Points::Rows(rows) => self.counting_sort(rows, bx),
            Points::Columns([xs, ys, zs]) => self.counting_sort_soa(xs, ys, zs, bx),
        }
    }

    /// On-demand pair traversal — zero allocation.
    ///
    /// Half-shell iteration over the occupied cells, calling the visitor
    /// instead of building a [`Neighbors`] table, so no column policy is
    /// involved: the visitor always receives `dist_sq` (Å²) and the
    /// minimum-image displacement `r_j - r_i` (Å) for each pair, with `i < j`.
    ///
    /// Requires a prior `build_index` / `update_index`. Without one there are
    /// no occupied cells and the visitor is simply never called.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        if self.occupied_cells.is_empty() {
            return;
        }
        let cutoff2 = self.cutoff * self.cutoff;
        let mut fwd_buf = [0usize; 27];

        for &cell in &self.occupied_cells {
            self.pairs_in_cell(cell as usize, cutoff2, &mut fwd_buf, &mut |i, j, d2, dr| {
                visitor.visit_pair(i, j, d2, dr)
            });
        }
    }

    /// Materialize the pairs into `out`, in parallel when that pays.
    ///
    /// Above `PAR_MIN_CELLS` occupied cells the search folds over the cells with
    /// rayon and concatenates the partial tables — 2.4× faster than the serial
    /// walk at 20k particles and 4.2× at 100k. Below that threshold, and without
    /// the `rayon` feature, it falls back to the trait default, which drives
    /// [`visit_pairs`](Self::visit_pairs).
    ///
    /// Both paths emit the identical pair set with `i < j`; only the row order
    /// differs, which is why [`NeighborList::neighbors`](crate::core::NeighborList::neighbors)
    /// promises a pair *set* and not an order.
    fn materialize_into(&self, out: &mut Neighbors) {
        #[cfg(feature = "rayon")]
        {
            if self.occupied_cells.len() >= PAR_MIN_CELLS {
                let mut partial = self.par_pairs(out.mode(), out.columns());
                out.append(&mut partial);
                return;
            }
        }
        self.visit_pairs(&mut |i, j, d2, dr| out.push(i, j, d2, dr));
    }
}

impl LinkCell {
    /// Half-shell pairs of one occupied cell, streamed into `sink`: the
    /// self-cell double loop plus the forward-stencil walk. Single home for
    /// the cell-walk kernel — [`Backend::visit_pairs`] instantiates it with a
    /// visitor sink, the rayon fold in [`par_pairs`](Self::par_pairs) with a
    /// table-push sink; the generic `sink` is monomorphized at each call site,
    /// so neither pays dispatch the hand-inlined loops did not.
    #[inline]
    fn pairs_in_cell(
        &self,
        cell: usize,
        cutoff2: F,
        fwd_buf: &mut [usize; 27],
        sink: &mut impl FnMut(u32, u32, F, [F; 3]),
    ) {
        let start = self.cell_start[cell] as usize;
        let end = self.cell_start[cell + 1] as usize;
        let mic = self.bx.mic();

        // Self-cell pairs: sorted order within a cell is ascending original
        // index (stable counting sort), so `oi < oj` holds structurally.
        for si in start..end {
            let pi = pos_at(&self.sorted_pos, si);
            let oi = self.sorted_idx[si];
            for sj in (si + 1)..end {
                let pj = pos_at(&self.sorted_pos, sj);
                let dr = mic.apply(crate::op::vec3::sub(pj, pi));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    sink(oi, self.sorted_idx[sj], d2, dr);
                }
            }
        }

        // Forward neighbor cells (stack buffer — no alloc).
        let n_fwd = self.grid.stencil_forward(cell, fwd_buf);
        for si in start..end {
            let pi = pos_at(&self.sorted_pos, si);
            let oi = self.sorted_idx[si];

            for &nc in &fwd_buf[..n_fwd] {
                let nc_start = self.cell_start[nc] as usize;
                let nc_end = self.cell_start[nc + 1] as usize;

                for sj in nc_start..nc_end {
                    let oj = self.sorted_idx[sj];
                    let pj = pos_at(&self.sorted_pos, sj);
                    let dr = mic.apply(crate::op::vec3::sub(pj, pi));
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        // Half-shell output contract: the pair is emitted with
                        // the smaller original index first. `dr` was computed
                        // as MIC(r_sj - r_si) for the *sorted* slots; when the
                        // roles swap, the sign must swap too, so the sink
                        // always receives r_j - r_i for the emitted (i, j).
                        if oi < oj {
                            sink(oi, oj, d2, dr);
                        } else {
                            sink(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                        }
                    }
                }
            }
        }
    }
}

impl LinkCell {
    /// Counting sort particles by cell index (interleaved `Array2` input).
    ///
    /// Thin adapter over [`counting_sort_impl`](Self::counting_sort_impl).
    fn counting_sort(&mut self, points: Fnx3View<'_>, bx: &SimBox) {
        let n_points = points.nrows();
        self.counting_sort_impl(
            n_points,
            |i| [points[[i, 0]], points[[i, 1]], points[[i, 2]]],
            bx,
        );
    }

    /// Counting sort particles by cell index (SoA `x`/`y`/`z` slices).
    ///
    /// SoA sibling of [`counting_sort`](Self::counting_sort); shares the same
    /// [`counting_sort_impl`](Self::counting_sort_impl) core, so the resulting
    /// `sorted_pos` / `cell_start` / `occupied_cells` are byte-identical to the
    /// `Array2` path for the same coordinates.
    fn counting_sort_soa(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        let n_points = xs.len();
        self.counting_sort_impl(n_points, |i| [xs[i], ys[i], zs[i]], bx);
    }

    /// Shared counting-sort core: cell assignment + prefix sum + scatter.
    ///
    /// Reads each particle position via `get_pt(i) -> [x, y, z]`, so both the
    /// interleaved (`Array2`) and column-major (SoA) entry points funnel
    /// through identical arithmetic. Sets up `grid`, `bx`, `cell_start`,
    /// `sorted_idx`, `sorted_pos`, and `occupied_cells`. Does NOT compute pairs.
    fn counting_sort_impl<G>(&mut self, n_points: usize, get_pt: G, bx: &SimBox)
    where
        G: Fn(usize) -> [F; 3],
    {
        let grid = CellGrid::for_cutoff(bx, self.cutoff);
        let n_cells = grid.n_cells();

        self.grid = grid;
        self.bx = bx.clone();

        // 1) Compute cell per particle, count per cell.
        //    `cell_of[i]` holds the final cell index for particle i; this buffer
        //    is never aliased with `sorted_idx` (which gets the reordering).
        self.cell_start.clear();
        self.cell_start.resize(n_cells + 1, 0);
        self.cell_of.resize(n_points, 0);
        for i in 0..n_points {
            let cell = grid.cell_of(bx, get_pt(i));
            self.cell_of[i] = cell as u32;
            self.cell_start[cell] += 1;
        }

        // 2) Prefix sum -> cell_start[c] = offset where cell c begins.
        //    Collect occupied cells while scanning.
        self.occupied_cells.clear();
        let mut acc = 0u32;
        for c in 0..n_cells {
            let count = self.cell_start[c];
            if count > 0 {
                self.occupied_cells.push(c as u32);
            }
            self.cell_start[c] = acc;
            acc += count;
        }
        self.cell_start[n_cells] = acc;
        debug_assert_eq!(acc as usize, n_points);

        // 3) Scatter particles into sorted order.
        //    `cursor[c]` starts at `cell_start[c]`; incremented as each particle
        //    in cell c is placed. Flat `sorted_pos` is 3N floats.
        self.cursor.resize(n_cells, 0);
        self.cursor.copy_from_slice(&self.cell_start[..n_cells]);

        self.sorted_idx.resize(n_points, 0);
        self.sorted_pos.resize(n_points * 3, 0.0);

        for i in 0..n_points {
            let cell = self.cell_of[i] as usize;
            let dst = self.cursor[cell] as usize;
            self.cursor[cell] += 1;
            self.sorted_idx[dst] = i as u32;
            let base = dst * 3;
            let p = get_pt(i);
            self.sorted_pos[base] = p[0];
            self.sorted_pos[base + 1] = p[1];
            self.sorted_pos[base + 2] = p[2];
        }
    }

    /// Half-shell pair search folded over occupied cells with rayon, into a
    /// fresh table keeping the columns `columns` asks for.
    ///
    /// Each worker accumulates its cells into a private [`Neighbors`], and the
    /// partial tables are concatenated by [`Neighbors::append`], which is where
    /// the column-alignment contract of the merge is stated. Every worker starts
    /// from the same `columns`, so every partial table has the same columns and
    /// concatenating them cannot misalign anything.
    ///
    /// Partial tables carry the caller's `mode` from the start, so no
    /// placeholder tag ever exists to launder.
    ///
    /// Only worth calling above `PAR_MIN_CELLS` occupied cells; below that,
    /// rayon's split/merge overhead exceeds the pair work.
    #[cfg(feature = "rayon")]
    fn par_pairs(
        &self,
        mode: crate::core::QueryMode,
        columns: crate::core::NeighborColumns,
    ) -> Neighbors {
        #[cfg(test)]
        crate::core::test_thread_pool::ensure();
        use rayon::prelude::*;

        let cutoff2 = self.cutoff * self.cutoff;

        self.occupied_cells
            .par_iter()
            .fold(
                || Neighbors::empty(mode, columns),
                |mut acc, &cell_u32| {
                    let mut fwd_buf = [0usize; 27];
                    self.pairs_in_cell(
                        cell_u32 as usize,
                        cutoff2,
                        &mut fwd_buf,
                        &mut |i, j, d2, dr| acc.push(i, j, d2, dr),
                    );
                    acc
                },
            )
            .reduce(
                || Neighbors::empty(mode, columns),
                |mut a, mut b| {
                    a.append(&mut b);
                    a
                },
            )
    }
}

// ---------------------------------------------------------------------------
// Inline stencil computation (zero-alloc: writes into caller-provided buffer)
// ---------------------------------------------------------------------------

/// Flat-slab position accessor: `[x, y, z]` of particle at sorted slot `si`.
#[inline(always)]
fn pos_at(sorted_pos: &[F], si: usize) -> [F; 3] {
    let base = si * 3;
    [sorted_pos[base], sorted_pos[base + 1], sorted_pos[base + 2]]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// The cell-list backend, driven through [`NeighborList`] — the only sanctioned
/// door to a self search.
///
/// Every fixture is hard-coded and every comparison sorts by `(i, j)` first:
/// [`NeighborList::neighbors`] promises a pair *set*, not a row order, because
/// the backend folds over occupied cells in parallel above a threshold.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use crate::core::SimBox;
    use crate::core::neighbors::test_fixtures::table_rows_sorted;
    use crate::core::{NeighborColumns, NeighborList};
    use ndarray::array;

    /// Half-shell self pairs of `pts` from the cell-list backend, every column
    /// present.
    fn cell_pairs(cutoff: F, pts: Fnx3View<'_>, bx: &SimBox) -> Neighbors {
        let mut nl = NeighborList::new(cutoff);
        nl.build(pts, bx);
        nl.neighbors(NeighborColumns::FULL)
    }

    /// The same, from the O(N²) reference backend — the oracle the cell list is
    /// checked against.
    fn brute_pairs(cutoff: F, pts: Fnx3View<'_>, bx: &SimBox) -> Neighbors {
        let mut nl = NeighborList::brute_force(cutoff);
        nl.build(pts, bx);
        nl.neighbors(NeighborColumns::FULL)
    }

    #[test]
    fn linked_cell_basic_pairs() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];
        let res = cell_pairs(0.5, pts.view(), &bx);
        assert_eq!(res.n_pairs(), 1);
        assert_eq!(res.query_point_indices()[0], 0);
        assert_eq!(res.point_indices()[0], 1);
    }

    #[test]
    fn linked_cell_pbc_boundary() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]];
        let res = cell_pairs(0.5, pts.view(), &bx);
        assert_eq!(res.n_pairs(), 1);
    }

    #[test]
    fn linked_cell_no_duplicates() {
        let bx = SimBox::new(Cell::Lengths([3.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]];
        let res = cell_pairs(1.0, pts.view(), &bx);
        let mut seen = std::collections::HashSet::new();
        for k in 0..res.n_pairs() {
            let i = res.query_point_indices()[k];
            let j = res.point_indices()[k];
            assert!(i < j);
            assert!(seen.insert((i, j)));
        }
    }

    #[test]
    fn linked_cell_cutoff_edge_included() {
        let bx = SimBox::new(Cell::Lengths([3.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let res = cell_pairs(1.0, pts.view(), &bx);
        assert_eq!(res.n_pairs(), 1);
    }

    /// Two independent searches over the same coordinates agree pair for pair,
    /// bit for bit.
    ///
    /// This replaces a test that read one *cached* table twice and compared it
    /// with itself — an assertion no defect could break. Materializing twice
    /// runs the search twice, so a search that depended on uninitialized buffer
    /// state or on iteration order of a hash container would be caught here.
    #[test]
    fn repeated_search_gives_the_same_pairs() {
        let bx = SimBox::new(Cell::Lengths([4.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [1.1, 1.2, 1.3]];

        let first = table_rows_sorted(&cell_pairs(0.5, pts.view(), &bx));
        let second = table_rows_sorted(&cell_pairs(0.5, pts.view(), &bx));

        assert_eq!(first.len(), 1, "fixture should produce exactly one pair");
        assert_eq!(first, second, "repeated searches must agree bitwise");
    }

    /// The materialized table is the streamed pairs — the `push` path and the
    /// visitor path must not diverge.
    #[test]
    fn stream_matches_materialized_table() {
        let bx = SimBox::new(Cell::Lengths([3.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.9, 2.8, 2.7]
        ];

        let mut nl = NeighborList::new(0.6);
        nl.build(pts.view(), &bx);

        let table = table_rows_sorted(&nl.neighbors(NeighborColumns::FULL));

        let mut streamed: Vec<(u32, u32, F, [F; 3])> = Vec::new();
        nl.for_each_pair(|p| streamed.push((p.i, p.j, p.dist_sq, p.disp)));
        streamed.sort_by_key(|r| (r.0, r.1));

        assert!(!table.is_empty(), "fixture should produce pairs");
        assert_eq!(table.len(), streamed.len());
        for (a, b) in table.iter().zip(streamed.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-6);
            for d in 0..3 {
                assert!((a.3[d] - b.3[d]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn neighborlist_algorithm_switch_consistency() {
        let bx = SimBox::new(Cell::Lengths([3.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.9, 2.8, 2.7]
        ];

        let lc_pairs = table_rows_sorted(&cell_pairs(0.6, pts.view(), &bx));
        let bf_pairs = table_rows_sorted(&brute_pairs(0.6, pts.view(), &bx));

        assert_eq!(lc_pairs.len(), bf_pairs.len());
        for (a, b) in lc_pairs.iter().zip(bf_pairs.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-6);
            for d in 0..3 {
                assert!((a.3[d] - b.3[d]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn non_periodic_no_wrap() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]];

        let result = cell_pairs(1.0, pts.view(), &bx);

        assert_eq!(result.n_pairs(), 0, "non-periodic box should not wrap");
    }

    #[test]
    fn periodic_does_wrap() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]];

        let result = cell_pairs(1.0, pts.view(), &bx);

        assert_eq!(result.n_pairs(), 1, "periodic box should wrap");
    }

    #[test]
    fn non_periodic_matches_brute_force() {
        let bx = SimBox::new(
            Cell::Lengths([20.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .expect("invalid box");
        let pts = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
        ];
        let cutoff = 1.0;

        let lc_result = cell_pairs(cutoff, pts.view(), &bx);
        let bf_result = brute_pairs(cutoff, pts.view(), &bx);

        assert_eq!(
            lc_result.n_pairs(),
            bf_result.n_pairs(),
            "cell list and reference should agree for non-periodic"
        );

        let mut lc_pairs: Vec<(u32, u32)> = lc_result
            .query_point_indices()
            .iter()
            .zip(lc_result.point_indices().iter())
            .map(|(&i, &j)| if i < j { (i, j) } else { (j, i) })
            .collect();
        lc_pairs.sort();

        let mut bf_pairs: Vec<(u32, u32)> = bf_result
            .query_point_indices()
            .iter()
            .zip(bf_result.point_indices().iter())
            .map(|(&i, &j)| if i < j { (i, j) } else { (j, i) })
            .collect();
        bf_pairs.sort();

        assert_eq!(lc_pairs, bf_pairs);
    }

    // --- sparse system: 3 particles in large box with small cutoff ---

    /// A sparse system pays for its particles, not for its empty cells.
    ///
    /// Box 20 Å at cutoff 0.5 Å is 64 000 cells for three particles; the search
    /// walks `occupied_cells` (three of them), so this configuration does not
    /// time out.
    ///
    /// `occupied_cells` is a backend internal that no engine accessor exposes,
    /// so this single test stays on [`LinkCell`] — through the trait methods the
    /// engine itself drives, `Backend::build_index` and `Backend::visit_pairs`.
    #[test]
    fn sparse_system_fast() {
        let bx = SimBox::new(
            Cell::Lengths([20.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .expect("invalid box");
        let pts = array![[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let mut lc = LinkCell::new().cutoff(0.5);
        lc.build_index(Points::Rows(pts.view()), &bx);
        assert_eq!(lc.occupied_cells.len(), 3);

        let mut n_pairs = 0usize;
        lc.visit_pairs(&mut |_i: u32, _j: u32, _d2: F, _dr: [F; 3]| n_pairs += 1);
        assert_eq!(n_pairs, 0);
    }

    // --- freud: 4 collinear, count pairs ---

    #[test]
    fn collinear_pair_counts() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let pts = array![
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ];
        assert_eq!(cell_pairs(2.01, pts.view(), &bx).n_pairs(), 5);
    }

    // --- all pairs within cutoff ---

    #[test]
    fn all_pairs_within_cutoff() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let pts = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [5.5, 5.5, 5.0],
        ];
        assert_eq!(cell_pairs(1.5, pts.view(), &bx).n_pairs(), 6);
    }

    // --- brute force vs linkcell with PBC ---

    #[test]
    fn exhaustive_pbc() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let pts = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
        ];
        let cutoff = 2.0;

        assert_eq!(
            cell_pairs(cutoff, pts.view(), &bx).n_pairs(),
            brute_pairs(cutoff, pts.view(), &bx).n_pairs()
        );
    }

    // --- edge cases ---

    #[test]
    fn no_pairs_large_separation() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let pts = array![[1.0, 1.0, 1.0], [8.0, 8.0, 8.0],];
        assert_eq!(cell_pairs(1.0, pts.view(), &bx).n_pairs(), 0);
    }

    #[test]
    fn single_particle_no_pairs() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let pts = array![[5.0, 5.0, 5.0],];
        assert_eq!(cell_pairs(3.0, pts.view(), &bx).n_pairs(), 0);
    }

    #[test]
    fn distances_correct() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let pts = array![[3.0, 5.0, 5.0], [5.0, 5.0, 5.0],];
        let nlist = cell_pairs(3.0, pts.view(), &bx);
        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("FULL storage materializes dist_sq");
        assert!((d2[0].sqrt() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn pbc_distance_correct() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let pts = array![[0.5, 5.0, 5.0], [9.5, 5.0, 5.0],]; // MIC dist = 1.0
        let nlist = cell_pairs(2.0, pts.view(), &bx);
        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("FULL storage materializes dist_sq");
        assert!((d2[0].sqrt() - 1.0).abs() < 1e-5);
    }

    // --- column path is bit-identical to the Array2 path ---

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

    /// Bitwise (not approximate) equality of two neighbor lists, compared as
    /// sorted pair sets.
    fn assert_bitwise_equal(a: &Neighbors, b: &Neighbors) {
        assert_eq!(a.n_pairs(), b.n_pairs(), "n_pairs differ");
        let rows_a = table_rows_sorted(a);
        let rows_b = table_rows_sorted(b);
        for (ra, rb) in rows_a.iter().zip(rows_b.iter()) {
            assert_eq!(ra.0, rb.0, "idx_i");
            assert_eq!(ra.1, rb.1, "idx_j");
            // Bitwise f64 equality — the arithmetic is identical, so it must match.
            assert_eq!(ra.2, rb.2, "dist_sq bitwise");
            for d in 0..3 {
                assert_eq!(ra.3[d], rb.3[d], "disp[{}] bitwise", d);
            }
        }
    }

    /// The column input ([`Points::Columns`]) indexes into the
    /// same cells as the `N × 3` one, so the two tables match *bitwise* — not
    /// merely within a tolerance.
    #[test]
    fn column_build_matches_row_build_bitwise() {
        // Periodic cube fixture: several pairs, including a PBC-wrap pair.
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let pts = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
            [2.2, 8.1, 3.3],
            [7.7, 2.4, 9.6],
        ];
        let (xs, ys, zs) = columns(&pts);
        let aos = cell_pairs(2.0, pts.view(), &bx);
        let mut soa_nl = NeighborList::new(2.0);
        soa_nl.build(Points::Columns([&xs, &ys, &zs]), &bx);
        let soa = soa_nl.neighbors(NeighborColumns::FULL);
        assert!(aos.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&aos, &soa);

        // Free / non-periodic fixture.
        let bxf = SimBox::new(
            Cell::Lengths([20.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        let ptsf = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
            [1.0, 18.0, 3.0],
        ];
        let (fxs, fys, fzs) = columns(&ptsf);
        let aos_free = cell_pairs(1.0, ptsf.view(), &bxf);
        let mut soa_free_nl = NeighborList::new(1.0);
        soa_free_nl.build(Points::Columns([&fxs, &fys, &fzs]), &bxf);
        let soa_free = soa_free_nl.neighbors(NeighborColumns::FULL);
        assert!(aos_free.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&aos_free, &soa_free);
    }
}

// ---------------------------------------------------------------------------
// Equivalence matrix
// ---------------------------------------------------------------------------

/// The cell-list backend against an independent O(N²) oracle, over the full
/// configuration space the cell partition has to survive.
///
/// Both sides are driven through [`NeighborList`](super::NeighborList) — the
/// engine is the only door to a self search — with
/// [`NeighborList::new`](super::NeighborList::new) selecting the cell list and
/// [`NeighborList::brute_force`](super::NeighborList::brute_force) the O(N²)
/// reference.
///
/// The reference shares no cell-assignment or stencil code with the algorithm
/// under test, so a defect in either cannot cancel out; its own contract is
/// pinned in `bruteforce.rs`.
///
/// Both traversal modes are exercised. They fail differently: the pair path's
/// forward filter covers an unordered cell pair from whichever side is cheaper,
/// so a stencil that loses a neighbour in one direction still produces the
/// right pair set, while a query rooted at the cell with the broken stencil
/// silently returns nothing. Testing pairs alone leaves that entire class
/// unobserved.
#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::core::Cell;
    use crate::core::CellGrid;
    use crate::core::NeighborColumns;
    use crate::core::NeighborList;
    use crate::core::NeighborQuery;
    use crate::core::SimBox;

    use ndarray::{Array2, array};
    use std::collections::{BTreeMap, BTreeSet};

    /// Deterministic point source — no RNG crate, no version drift, same
    /// stream on every platform.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> F {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as F) / ((1u64 << 53) as F)
        }
    }

    fn cells() -> Vec<(&'static str, Cell)> {
        vec![
            (
                "ortho",
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [90.0, 90.0, 90.0],
                },
            ),
            (
                "hex",
                Cell::LengthsAngles {
                    lengths: [10.0, 10.0, 12.0],
                    angles: [90.0, 90.0, 120.0],
                },
            ),
            (
                "tilted",
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [70.0, 80.0, 65.0],
                },
            ),
        ]
    }

    const PBCS: [[bool; 3]; 4] = [
        [true, true, true],
        [true, true, false],
        [true, false, false],
        [false, false, false],
    ];

    /// Points in Cartesian space: bulk sampling inside the cell, plus the two
    /// populations that break a naive partition — coordinates outside the cell
    /// (an optimiser's intermediate iterates leave the box routinely) and
    /// coordinates sitting exactly on a cell boundary.
    fn points(bx: &SimBox, seed: u64) -> Array2<F> {
        let mut rng = Lcg(seed);
        let mut frac: Vec<[F; 3]> = (0..40)
            .map(|_| [rng.unit(), rng.unit(), rng.unit()])
            .collect();

        for k in 0..3 {
            for &out in &[-1.7 as F, -0.15, 1.05, 2.4] {
                let mut p = [rng.unit(), rng.unit(), rng.unit()];
                p[k] = out;
                frac.push(p);
            }
        }
        for k in 0..3 {
            for &edge in &[0.0 as F, 1.0] {
                let mut p = [0.5 as F, 0.5, 0.5];
                p[k] = edge;
                frac.push(p);
            }
        }

        let mut pts = Array2::<F>::zeros((frac.len(), 3));
        for (i, f) in frac.iter().enumerate() {
            let cart = bx.make_cartesian(array![f[0], f[1], f[2]].view());
            for d in 0..3 {
                pts[[i, d]] = cart[d];
            }
        }
        pts
    }

    fn oracle_cross(
        query: &Array2<F>,
        refs: &Array2<F>,
        bx: &SimBox,
        cutoff: F,
    ) -> BTreeSet<(u32, u32)> {
        let c2 = cutoff * cutoff;
        let mut out = BTreeSet::new();
        for i in 0..query.nrows() {
            let qi = [query[[i, 0]], query[[i, 1]], query[[i, 2]]];
            for j in 0..refs.nrows() {
                let rj = [refs[[j, 0]], refs[[j, 1]], refs[[j, 2]]];
                let dr = bx.mic().apply(crate::op::vec3::sub(rj, qi));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= c2 {
                    out.insert((i as u32, j as u32));
                }
            }
        }
        out
    }

    /// Unordered pair multiset from a [`Neighbors`] table. A `BTreeMap` keyed on
    /// the ordered pair would silently swallow a duplicate emission, so
    /// duplicates are counted explicitly.
    ///
    /// The two physical columns are also checked against each other here rather
    /// than in a test of their own. `dist_sq` and `disp` are produced by the
    /// same minimum-image evaluation, so `dist_sq == ‖disp‖²` must hold for
    /// every pair — and collecting inside the matrix loop is what makes that
    /// claim executable across orthorhombic, hexagonal and tilted cells and all
    /// four periodicity combinations, i.e. through the triclinic MIC branch as
    /// well as the orthorhombic one. A cell partition that folded the two
    /// columns to different periodic images would fail here.
    fn collect(res: &Neighbors) -> (BTreeMap<(u32, u32), F>, usize) {
        let mut map = BTreeMap::new();
        let n = res.n_pairs();
        let d2 = res.dist_sq().expect("search materializes dist_sq");
        let disp = res.disp().expect("search materializes disp");
        for k in 0..n {
            let (i, j) = (res.query_point_indices()[k], res.point_indices()[k]);
            let norm_sq = disp[[k, 0]] * disp[[k, 0]]
                + disp[[k, 1]] * disp[[k, 1]]
                + disp[[k, 2]] * disp[[k, 2]];
            assert!(
                (d2[k] - norm_sq).abs() <= 1e-12,
                "pair ({i},{j}): dist_sq {} != ||disp||^2 {norm_sq}",
                d2[k]
            );
            let key = if i < j { (i, j) } else { (j, i) };
            map.insert(key, d2[k]);
        }
        (map, n)
    }

    /// Cutoffs chosen to drive `celldim` through 1, 2, 3 and 5 cells per axis.
    fn cutoffs(bx: &SimBox) -> Vec<F> {
        let npd = bx.nearest_plane_distance();
        let min = npd[0].min(npd[1]).min(npd[2]);
        vec![min / 1.0, min / 2.0, min / 3.0, min / 5.0]
    }

    #[test]
    fn pair_traversal_matches_the_oracle_over_the_matrix() {
        let mut seen_dims: BTreeSet<u32> = BTreeSet::new();
        let mut configs = 0usize;

        for (name, h) in cells() {
            for pbc in PBCS {
                let bx = SimBox::new(h, [0.0, 0.0, 0.0], pbc).expect("box");
                let pts = points(&bx, 0x5EED);

                for cutoff in cutoffs(&bx) {
                    let tag = format!("{name} pbc={pbc:?} cutoff={cutoff:.3}");
                    seen_dims.extend(CellGrid::for_cutoff(&bx, cutoff).celldim());
                    configs += 1;

                    let mut bf = NeighborList::brute_force(cutoff);
                    bf.build(pts.view(), &bx);
                    let (want, _) = collect(&bf.neighbors(NeighborColumns::FULL));

                    let mut lc = NeighborList::new(cutoff);
                    lc.build(pts.view(), &bx);
                    let (got, emitted) = collect(&lc.neighbors(NeighborColumns::FULL));

                    assert_eq!(emitted, want.len(), "{tag}: pair count (duplicate or gap)");
                    assert_eq!(
                        got.keys().collect::<Vec<_>>(),
                        want.keys().collect::<Vec<_>>(),
                        "{tag}: pair set"
                    );
                    for (key, d2) in &got {
                        assert!(
                            (d2 - want[key]).abs() <= 1e-12,
                            "{tag}: dist_sq for {key:?}: {d2} vs {}",
                            want[key]
                        );
                    }
                }
            }
        }

        assert!(configs >= 48, "matrix shrank to {configs} configurations");
        for target in [1u32, 2, 3, 5] {
            assert!(
                seen_dims.contains(&target),
                "matrix never produced celldim {target}; saw {seen_dims:?}"
            );
        }
    }

    #[test]
    fn cross_query_matches_the_oracle_over_the_matrix() {
        for (name, h) in cells() {
            for pbc in PBCS {
                let bx = SimBox::new(h, [0.0, 0.0, 0.0], pbc).expect("box");
                let refs = points(&bx, 0x5EED);
                let query = points(&bx, 0xC0FFEE);

                for cutoff in cutoffs(&bx) {
                    let tag = format!("{name} pbc={pbc:?} cutoff={cutoff:.3}");
                    let want = oracle_cross(&query, &refs, &bx, cutoff);

                    let nq = NeighborQuery::new(&bx, refs.view(), cutoff);
                    let res = nq.query(query.view());
                    let got: BTreeSet<(u32, u32)> = (0..res.n_pairs())
                        .map(|k| (res.query_point_indices()[k], res.point_indices()[k]))
                        .collect();

                    assert_eq!(got.len(), res.n_pairs(), "{tag}: duplicate query hit");
                    assert_eq!(got, want, "{tag}: cross-query set");
                }
            }
        }
    }
}
