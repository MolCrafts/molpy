//! [`CellGrid`] — the PBC-aware lattice cell partition.

use crate::core::SimBox;
use crate::op::F;

/// A regular partition of a [`SimBox`] into cells, indexed in fractional space.
///
/// Construct with [`for_cutoff`](Self::for_cutoff) (cells at least one cutoff
/// wide) or [`with_dims`](Self::with_dims) (explicit dimensions, for tests and
/// for callers that manage their own sizing).
///
/// One place answers "which cell does this point fall in, and which cells are
/// its neighbours". [`LinkCell`](crate::core::LinkCell) uses it, and so can external
/// consumers that need Packmol-style cell indexing without inheriting the
/// neighbour-list machinery around it.
///
/// Two properties are load-bearing and are pinned by its tests:
///
/// **Cell sizing uses plane distances.** A cell-list search is valid only if
/// every cell is at least one cutoff wide along every lattice direction,
/// measured as a *plane distance*. For a tilted cell `|a_k|` overestimates that
/// width, so sizing by lattice-vector length would make cells thinner than the
/// cutoff and pairs would be silently missed. [`CellGrid::for_cutoff`] sizes
/// from [`SimBox::nearest_plane_distance`], which is correct for orthorhombic
/// and triclinic alike. Everything downstream then works in *fractional* index
/// space, where the stencil is identical for both — the lattice re-enters only
/// through the minimum-image displacement.
///
/// **Wrap on periodic axes, clamp on non-periodic ones.** A point outside the
/// cell along a non-periodic axis must land in the nearest edge cell, not on
/// the opposite face. This matters for optimisation workloads (packing) where
/// intermediate iterates routinely leave the box — wrapping there would hide a
/// particle from its true neighbours while showing it to unrelated ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellGrid {
    celldim: [u32; 3],
    pbc: [bool; 3],
}

impl CellGrid {
    /// Partition sized so every lattice direction is at least `cutoff` wide:
    /// `celldim[k] = max(1, floor(nearest_plane_distance[k] / cutoff))`.
    ///
    /// Periodicity is taken from the box.
    ///
    /// The `max(1)` marks the one case the 3×3×3 stencil cannot cover: if the
    /// box is itself narrower than `cutoff` along some direction, a single cell
    /// is the most the partition can offer and that cell is narrower than the
    /// cutoff. Pair search then rests on the minimum image alone, so pairs
    /// beyond half the box width along that direction are missed. That is the
    /// usual minimum-image restriction, not something the partition can repair
    /// — callers who need a cutoff above half the box must enlarge the box (or
    /// search explicit images). The width guarantee therefore reads: cell width
    /// ≥ cutoff **whenever the box is at least one cutoff wide** along that
    /// direction.
    ///
    /// # Panics
    ///
    /// If `cutoff` is not strictly positive.
    #[inline]
    pub fn for_cutoff(bx: &SimBox, cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        let npd = bx.nearest_plane_distance();
        Self {
            celldim: [
                ((npd[0] / cutoff).floor() as u32).max(1),
                ((npd[1] / cutoff).floor() as u32).max(1),
                ((npd[2] / cutoff).floor() as u32).max(1),
            ],
            pbc: bx.pbc(),
        }
    }

    /// [`for_cutoff`](Self::for_cutoff) with a ceiling on the total cell
    /// count: never more than `max_cells` cells.
    ///
    /// If the cutoff-sized grid already fits, it is returned unchanged.
    /// Otherwise every axis is coarsened by one **common** factor `s`,
    /// `celldim[k] = max(1, floor(fine[k] / s))`, with `s` chosen so the
    /// product fits the budget. An axis the common factor would drive below one
    /// cell is pinned at one and the factor is re-derived over the remaining
    /// axes, so a very flat box keeps its resolution along the long directions
    /// instead of blowing the budget. A final trim of the longest axis absorbs
    /// floating-point rounding in `s`, so `n_cells() <= max_cells` holds
    /// exactly, not approximately.
    ///
    /// Coarsening only widens cells, so the width guarantee of `for_cutoff`
    /// (cell width ≥ `cutoff` wherever the box allows it) still holds and a
    /// ±1 stencil still finds every pair: a cap costs search selectivity,
    /// never correctness. Use it when the box is large relative to the cutoff
    /// and the point count (e.g. a free-boundary bounding box), where the raw
    /// grid would allocate far more cells than there are points to bin.
    ///
    /// # Panics
    ///
    /// If `cutoff` is not strictly positive or `max_cells` is zero.
    pub fn for_cutoff_capped(bx: &SimBox, cutoff: F, max_cells: usize) -> Self {
        assert!(max_cells > 0, "max_cells must be positive");
        let fine = Self::for_cutoff(bx, cutoff);
        if fine.n_cells() <= max_cells {
            return fine;
        }
        let raw = fine.celldim.map(|d| d as F);
        let mut dims = [1u32; 3];
        // Axes still coarsened by the common factor; the others are pinned at 1.
        let mut free = [true; 3];
        loop {
            let nfree = free.iter().filter(|f| **f).count();
            if nfree == 0 {
                break;
            }
            // Pinned axes contribute one cell each, so the whole budget goes
            // to the free ones: prod(raw[free]) / s^nfree = max_cells.
            let raw_free: F = (0..3).filter(|k| free[*k]).map(|k| raw[k]).product();
            let s = (raw_free / max_cells as F).powf(1.0 / nfree as F).max(1.0);
            let mut pinned = false;
            for k in 0..3 {
                if free[k] {
                    let d = (raw[k] / s).floor();
                    if d < 1.0 {
                        free[k] = false;
                        dims[k] = 1;
                        pinned = true;
                    } else {
                        dims[k] = d as u32;
                    }
                }
            }
            if !pinned {
                break;
            }
        }
        // Rounding in `s` can leave the product a hair over budget.
        let total = |d: &[u32; 3]| d.iter().map(|v| *v as usize).product::<usize>();
        while total(&dims) > max_cells {
            let k = (0..3).max_by_key(|k| dims[*k]).expect("three axes");
            dims[k] -= 1;
        }
        Self::with_dims(dims, fine.pbc)
    }

    /// Partition with explicit dimensions and periodicity.
    ///
    /// # Panics
    ///
    /// If any dimension is zero.
    #[inline]
    pub fn with_dims(celldim: [u32; 3], pbc: [bool; 3]) -> Self {
        assert!(
            celldim[0] > 0 && celldim[1] > 0 && celldim[2] > 0,
            "cell dimensions must be non-zero"
        );
        Self { celldim, pbc }
    }

    #[inline(always)]
    pub fn celldim(&self) -> [u32; 3] {
        self.celldim
    }

    #[inline(always)]
    pub fn pbc(&self) -> [bool; 3] {
        self.pbc
    }

    #[inline(always)]
    pub fn n_cells(&self) -> usize {
        (self.celldim[0] as usize) * (self.celldim[1] as usize) * (self.celldim[2] as usize)
    }

    /// Cell triple for a point.
    ///
    /// Wraps into `[0, celldim[k])` on periodic axes; clamps into
    /// `[0, celldim[k] - 1]` on non-periodic axes.
    #[inline(always)]
    pub fn cell3(&self, bx: &SimBox, r: [F; 3]) -> [u32; 3] {
        let f = bx.make_fractional_raw_arr3(r);
        [
            self.axis_cell(f[0], 0),
            self.axis_cell(f[1], 1),
            self.axis_cell(f[2], 2),
        ]
    }

    /// Flat cell index for a point — [`cell3`](Self::cell3) then
    /// [`flat`](Self::flat).
    ///
    /// Keep this inlined at the call site. The per-axis wrap/clamp dispatch is
    /// loop-invariant, so an inlining compiler hoists it out of a binning loop
    /// entirely; behind a call boundary it costs about 10% on an orthorhombic
    /// box (measured, 1000 points, `neighbors/cellgrid`). Inlined it is faster
    /// than an unconditional wrap on both box kinds, because the triclinic path
    /// does not wrap three components it is about to overwrite.
    #[inline(always)]
    pub fn cell_of(&self, bx: &SimBox, r: [F; 3]) -> usize {
        self.flat(self.cell3(bx, r))
    }

    /// Flatten a cell triple. Layout is x-fastest: `cz*ny*nx + cy*nx + cx`.
    #[inline(always)]
    pub fn flat(&self, c: [u32; 3]) -> usize {
        (c[2] * self.celldim[1] * self.celldim[0] + c[1] * self.celldim[0] + c[0]) as usize
    }

    /// Inverse of [`flat`](Self::flat).
    #[inline(always)]
    pub fn unflat(&self, icell: usize) -> [u32; 3] {
        let idx = icell as u32;
        let nxy = self.celldim[0] * self.celldim[1];
        [
            (idx % nxy) % self.celldim[0],
            (idx % nxy) / self.celldim[0],
            idx / nxy,
        ]
    }

    /// Neighbouring cells, self excluded — up to 26 distinct indices.
    ///
    /// Use for one-sided queries ("everything near this point"). For pair
    /// enumeration use [`stencil_forward`](Self::stencil_forward) instead.
    #[inline]
    pub fn stencil_all(&self, icell: usize, out: &mut [usize; 27]) -> usize {
        self.collect(icell, |nc| nc != icell, out)
    }

    /// Forward half of the stencil — neighbouring cells with a strictly greater
    /// index.
    ///
    /// Sweeping every cell and visiting its forward neighbours produces each
    /// unordered pair of adjacent cells exactly once, which is what a pair loop
    /// needs. The `nc > icell` filter is what makes this hold even when an axis
    /// has only one or two cells and several stencil offsets alias to the same
    /// neighbour.
    ///
    /// The buffer is 27 wide, not 13: on a small grid the pre-dedup candidate
    /// set can exceed the eventual count.
    #[inline]
    pub fn stencil_forward(&self, icell: usize, out: &mut [usize; 27]) -> usize {
        self.collect(icell, |nc| nc > icell, out)
    }

    // -----------------------------------------------------------------------
    // internals
    // -----------------------------------------------------------------------

    /// Cell index along one axis from a **raw** (unwrapped) fractional
    /// coordinate.
    #[inline(always)]
    fn axis_cell(&self, frac: F, k: usize) -> u32 {
        let dim = self.celldim[k];
        if self.pbc[k] {
            let w = frac - frac.floor();
            ((w * dim as F).floor() as u32) % dim
        } else {
            let s = (frac * dim as F).floor();
            if s <= 0.0 { 0 } else { (s as u32).min(dim - 1) }
        }
    }

    /// Shared stencil walk: the 3×3×3 offset block around `icell`, skipping
    /// out-of-range offsets on non-periodic axes, wrapping on periodic ones,
    /// then sorted and deduplicated in place.
    ///
    /// The offset block is symmetric on every axis regardless of how few cells
    /// that axis has. Special-casing small dimensions to a forward-only offset
    /// set looks like an optimisation but drops a genuine neighbour: with two
    /// non-periodic cells, cell 1's backward neighbour (cell 0) is the only one
    /// it has, and a `{0, +1}` offset set would report that cell 1 has no
    /// neighbours at all.
    #[inline]
    fn collect(
        &self,
        icell: usize,
        filter: impl Fn(usize) -> bool,
        out: &mut [usize; 27],
    ) -> usize {
        let [dx, dy, dz] = self.celldim;
        let c = self.unflat(icell);
        let nxy = dx * dy;

        let mut len = 0usize;
        for nk in (c[2] as i32 - 1)..=(c[2] as i32 + 1) {
            if !self.pbc[2] && (nk < 0 || nk >= dz as i32) {
                continue;
            }
            for nj in (c[1] as i32 - 1)..=(c[1] as i32 + 1) {
                if !self.pbc[1] && (nj < 0 || nj >= dy as i32) {
                    continue;
                }
                for ni in (c[0] as i32 - 1)..=(c[0] as i32 + 1) {
                    if !self.pbc[0] && (ni < 0 || ni >= dx as i32) {
                        continue;
                    }
                    let nc = (wrap(nk, dz) * nxy + wrap(nj, dy) * dx + wrap(ni, dx)) as usize;
                    if filter(nc) {
                        out[len] = nc;
                        len += 1;
                    }
                }
            }
        }

        out[..len].sort_unstable();
        let mut w = 0usize;
        for r in 0..len {
            if w == 0 || out[r] != out[w - 1] {
                out[w] = out[r];
                w += 1;
            }
        }
        w
    }
}

/// Wrap a signed cell index into `[0, dim)`.
#[inline(always)]
fn wrap(idx: i32, dim: u32) -> u32 {
    let d = dim as i32;
    let mut v = idx % d;
    if v < 0 {
        v += d;
    }
    v as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;

    use std::collections::HashSet;

    fn ortho(len: F, pbc: [bool; 3]) -> SimBox {
        SimBox::new(Cell::Lengths([len, len, len]), [0.0, 0.0, 0.0], pbc).expect("box")
    }

    /// Independent adjacency oracle: two cell indices along one axis are
    /// adjacent when they are equal, differ by one, or (periodic, and the axis
    /// has at least two cells) sit on opposite ends.
    fn axis_adjacent(a: u32, b: u32, dim: u32, periodic: bool) -> bool {
        if a == b {
            return true;
        }
        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
        if hi - lo == 1 {
            return true;
        }
        periodic && dim >= 2 && lo == 0 && hi == dim - 1
    }

    fn neighbours_of(g: &CellGrid, icell: usize) -> HashSet<usize> {
        let dim = g.celldim();
        let pbc = g.pbc();
        let c = g.unflat(icell);
        let mut out = HashSet::new();
        for z in 0..dim[2] {
            for y in 0..dim[1] {
                for x in 0..dim[0] {
                    let other = g.flat([x, y, z]);
                    if other == icell {
                        continue;
                    }
                    if axis_adjacent(c[0], x, dim[0], pbc[0])
                        && axis_adjacent(c[1], y, dim[1], pbc[1])
                        && axis_adjacent(c[2], z, dim[2], pbc[2])
                    {
                        out.insert(other);
                    }
                }
            }
        }
        out
    }

    const DIMS: [u32; 4] = [1, 2, 3, 5];
    const PBCS: [[bool; 3]; 4] = [
        [true, true, true],
        [true, true, false],
        [true, false, false],
        [false, false, false],
    ];

    // -- flat / unflat -----------------------------------------------------

    #[test]
    fn flat_unflat_round_trips_over_every_cell() {
        for &dx in &DIMS {
            for &dy in &DIMS {
                for &dz in &DIMS {
                    let g = CellGrid::with_dims([dx, dy, dz], [true; 3]);
                    for i in 0..g.n_cells() {
                        assert_eq!(g.flat(g.unflat(i)), i, "dims {dx},{dy},{dz} cell {i}");
                    }
                }
            }
        }
    }

    #[test]
    fn flat_layout_is_x_fastest() {
        let g = CellGrid::with_dims([4, 3, 2], [true; 3]);
        assert_eq!(g.flat([1, 0, 0]), 1);
        assert_eq!(g.flat([0, 1, 0]), 4);
        assert_eq!(g.flat([0, 0, 1]), 12);
    }

    // -- AC-001: wrap vs clamp --------------------------------------------

    #[test]
    fn periodic_axis_wraps_non_periodic_axis_clamps() {
        // 10 Å box, 4 cells per axis => 2.5 Å per cell. x periodic, y/z not.
        let bx = ortho(10.0, [true, false, false]);
        let g = CellGrid::with_dims([4, 4, 4], [true, false, false]);

        // x: 12.5 Å is one box + 2.5 Å, so it must land in the same cell as
        // 2.5 Å (cell 1), not be clamped to the edge.
        assert_eq!(g.cell3(&bx, [2.5, 5.0, 5.0])[0], 1);
        assert_eq!(g.cell3(&bx, [12.5, 5.0, 5.0])[0], 1);
        assert_eq!(g.cell3(&bx, [-7.5, 5.0, 5.0])[0], 1);

        // y/z: outside points clamp to the edge cells and never wrap.
        assert_eq!(g.cell3(&bx, [5.0, -30.0, 5.0])[1], 0);
        assert_eq!(g.cell3(&bx, [5.0, -0.001, 5.0])[1], 0);
        assert_eq!(g.cell3(&bx, [5.0, 90.0, 5.0])[1], 3);
        assert_eq!(g.cell3(&bx, [5.0, 10.001, 5.0])[1], 3);
        assert_eq!(g.cell3(&bx, [5.0, 5.0, -1.0])[2], 0);
        assert_eq!(g.cell3(&bx, [5.0, 5.0, 11.0])[2], 3);
    }

    #[test]
    fn axes_are_independent() {
        let bx = ortho(10.0, [true, false, true]);
        let g = CellGrid::with_dims([4, 4, 4], [true, false, true]);
        // Out of the box on all three axes at once: x and z wrap, y clamps.
        assert_eq!(g.cell3(&bx, [12.5, 90.0, -7.5]), [1, 3, 1]);
    }

    #[test]
    fn boundary_fractions_use_the_half_open_convention() {
        let bx = ortho(8.0, [true, false, true]);
        let g = CellGrid::with_dims([4, 4, 4], [true, false, true]);
        // Lower edge belongs to cell 0 on both kinds of axis.
        assert_eq!(g.cell3(&bx, [0.0, 0.0, 0.0]), [0, 0, 0]);
        // Upper edge: periodic wraps to 0, non-periodic clamps to the last cell.
        assert_eq!(g.cell3(&bx, [8.0, 8.0, 8.0]), [0, 3, 0]);
        // Cell boundaries belong to the upper cell.
        assert_eq!(g.cell3(&bx, [2.0, 2.0, 2.0]), [1, 1, 1]);
    }

    #[test]
    fn cell_of_agrees_with_flat_of_cell3() {
        let bx = ortho(10.0, [true, false, true]);
        let g = CellGrid::with_dims([3, 4, 5], [true, false, true]);
        for r in [[1.0, 2.0, 3.0], [-5.0, 12.0, 25.0], [9.99, 0.0, 0.0]] {
            assert_eq!(g.cell_of(&bx, r), g.flat(g.cell3(&bx, r)));
        }
    }

    // -- for_cutoff --------------------------------------------------------

    #[test]
    fn for_cutoff_sizes_from_plane_distances() {
        let bx = ortho(10.0, [true; 3]);
        assert_eq!(CellGrid::for_cutoff(&bx, 2.5).celldim(), [4, 4, 4]);
        assert_eq!(CellGrid::for_cutoff(&bx, 3.0).celldim(), [3, 3, 3]);
        // A cutoff larger than the box still leaves one cell per axis.
        assert_eq!(CellGrid::for_cutoff(&bx, 50.0).celldim(), [1, 1, 1]);
    }

    #[test]
    fn for_cutoff_on_a_tilted_cell_uses_plane_distance_not_edge_length() {
        // Hexagonal cell: |a| = |b| = 10, but the plane distance along the
        // tilted direction is 10 * sin(120°) = 8.66. Sizing off the 10 Å edge
        // length would give cells thinner than the cutoff and silently drop
        // pairs, so the guarantee is stated against the plane distance.
        let hex = Cell::LengthsAngles {
            lengths: [10.0, 10.0, 10.0],
            angles: [90.0, 90.0, 120.0],
        };
        let bx = SimBox::new(hex, [0.0, 0.0, 0.0], [true; 3]).expect("hex box");
        let npd = bx.nearest_plane_distance();
        assert!(
            npd.iter().any(|&d| d < 10.0 - 1e-9),
            "expected a plane distance below the edge length, got {npd:?}"
        );

        let cutoff = 2.0;
        let g = CellGrid::for_cutoff(&bx, cutoff);
        for k in 0..3 {
            let width = npd[k] / g.celldim()[k] as F;
            assert!(
                width >= cutoff - 1e-12,
                "axis {k}: cell width {width} below cutoff {cutoff}"
            );
            // And sizing is off the plane distance, not the edge length: an
            // edge-length partition would be coarser on the tilted axis.
            assert!(g.celldim()[k] as F <= npd[k] / cutoff + 1e-12);
        }
    }

    #[test]
    fn a_box_narrower_than_the_cutoff_degrades_to_one_cell() {
        // Documented limit of the 3x3x3 stencil: the partition cannot make a
        // cell wider than the box, so a cutoff above the box width leaves one
        // cell and the search falls back on the minimum image alone.
        let bx = ortho(5.0, [true; 3]);
        let g = CellGrid::for_cutoff(&bx, 9.0);
        assert_eq!(g.celldim(), [1, 1, 1]);
    }

    // -- for_cutoff_capped -------------------------------------------------

    #[test]
    fn for_cutoff_capped_returns_the_fine_grid_when_it_fits() {
        let bx = ortho(10.0, [true; 3]);
        assert_eq!(
            CellGrid::for_cutoff_capped(&bx, 2.5, 64),
            CellGrid::for_cutoff(&bx, 2.5)
        );
    }

    #[test]
    fn for_cutoff_capped_coarsens_every_axis_by_one_factor() {
        // Fine grid 100^3; a budget of 1000 is a common factor of 10.
        let bx = ortho(100.0, [false; 3]);
        let g = CellGrid::for_cutoff_capped(&bx, 1.0, 1000);
        assert_eq!(g.celldim(), [10, 10, 10]);
        assert_eq!(g.pbc(), [false; 3]);
    }

    #[test]
    fn for_cutoff_capped_never_exceeds_the_budget() {
        let bx = ortho(97.0, [true; 3]);
        for cap in [1, 2, 7, 26, 27, 28, 999, 1000, 1001, 50_000] {
            let g = CellGrid::for_cutoff_capped(&bx, 1.0, cap);
            assert!(g.n_cells() <= cap, "cap {cap}: {:?}", g.celldim());
        }
    }

    #[test]
    fn for_cutoff_capped_pins_a_thin_axis_and_spends_the_budget_elsewhere() {
        // Fine grid [1000, 1000, 1]: a naive common factor cbrt(1e6/100) = 21.5
        // would give [46, 46, 1] = 2116 cells, far over a budget of 100. The
        // thin axis is pinned and the factor re-derived over the other two.
        let bx = SimBox::new(
            Cell::Lengths([1000.0, 1000.0, 1.0]),
            [0.0, 0.0, 0.0],
            [false; 3],
        )
        .expect("slab");
        let g = CellGrid::for_cutoff_capped(&bx, 1.0, 100);
        assert_eq!(g.celldim(), [10, 10, 1]);
    }

    #[test]
    fn for_cutoff_capped_cells_stay_at_least_one_cutoff_wide() {
        let bx = ortho(50.0, [true; 3]);
        let cutoff = 3.0;
        let g = CellGrid::for_cutoff_capped(&bx, cutoff, 20);
        for k in 0..3 {
            assert!(50.0 / g.celldim()[k] as F >= cutoff);
        }
    }

    #[test]
    #[should_panic(expected = "max_cells must be positive")]
    fn for_cutoff_capped_rejects_a_zero_budget() {
        CellGrid::for_cutoff_capped(&ortho(10.0, [true; 3]), 1.0, 0);
    }

    // -- AC-003 / AC-002: stencils ----------------------------------------

    #[test]
    fn stencils_are_distinct_and_exclude_self() {
        let mut buf = [0usize; 27];
        for &dx in &DIMS {
            for &dy in &DIMS {
                for &dz in &DIMS {
                    for &pbc in &PBCS {
                        let g = CellGrid::with_dims([dx, dy, dz], pbc);
                        for i in 0..g.n_cells() {
                            let n = g.stencil_all(i, &mut buf);
                            let slice = &buf[..n];
                            let uniq: HashSet<_> = slice.iter().copied().collect();
                            assert_eq!(uniq.len(), n, "all: dup in {dx},{dy},{dz} {pbc:?}");
                            assert!(!slice.contains(&i), "all: self in stencil");

                            let n = g.stencil_forward(i, &mut buf);
                            let slice = &buf[..n];
                            let uniq: HashSet<_> = slice.iter().copied().collect();
                            assert_eq!(uniq.len(), n, "fwd: dup in {dx},{dy},{dz} {pbc:?}");
                            assert!(slice.iter().all(|&nc| nc > i), "fwd: not forward-only");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stencil_all_matches_the_adjacency_oracle() {
        let mut buf = [0usize; 27];
        for &dx in &DIMS {
            for &dy in &DIMS {
                for &dz in &DIMS {
                    for &pbc in &PBCS {
                        let g = CellGrid::with_dims([dx, dy, dz], pbc);
                        for i in 0..g.n_cells() {
                            let n = g.stencil_all(i, &mut buf);
                            let got: HashSet<usize> = buf[..n].iter().copied().collect();
                            assert_eq!(
                                got,
                                neighbours_of(&g, i),
                                "dims {dx},{dy},{dz} pbc {pbc:?} cell {i}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn two_non_periodic_cells_still_see_each_other() {
        // Regression: a forward-only offset set on a short axis leaves the
        // upper cell with an empty full stencil.
        let mut buf = [0usize; 27];
        let g = CellGrid::with_dims([2, 1, 1], [false, false, false]);
        let n = g.stencil_all(1, &mut buf);
        assert_eq!(&buf[..n], &[0], "cell 1 must see cell 0");
        let n = g.stencil_all(0, &mut buf);
        assert_eq!(&buf[..n], &[1]);
    }

    #[test]
    fn forward_sweep_yields_every_adjacent_pair_exactly_once() {
        let mut buf = [0usize; 27];
        for &dx in &DIMS {
            for &dy in &DIMS {
                for &dz in &DIMS {
                    for &pbc in &PBCS {
                        let g = CellGrid::with_dims([dx, dy, dz], pbc);
                        let mut seen: HashSet<(usize, usize)> = HashSet::new();
                        let mut count = 0usize;
                        for i in 0..g.n_cells() {
                            let n = g.stencil_forward(i, &mut buf);
                            for &nc in &buf[..n] {
                                assert!(
                                    seen.insert((i, nc)),
                                    "pair ({i},{nc}) emitted twice for {dx},{dy},{dz} {pbc:?}"
                                );
                                count += 1;
                            }
                        }
                        // Same set as the oracle's unordered adjacency.
                        let mut expected: HashSet<(usize, usize)> = HashSet::new();
                        for i in 0..g.n_cells() {
                            for j in neighbours_of(&g, i) {
                                let pair = if i < j { (i, j) } else { (j, i) };
                                expected.insert(pair);
                            }
                        }
                        assert_eq!(count, expected.len());
                        assert_eq!(seen, expected, "dims {dx},{dy},{dz} pbc {pbc:?}");
                    }
                }
            }
        }
    }
}
