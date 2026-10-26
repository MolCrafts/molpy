//! Lattice/grid occupancy tracking for self-avoidance.

use std::collections::HashSet;

use crate::core::SimBox;
use crate::op::{F, Pbc3};

/// How cell occupancy decides whether a candidate position overlaps.
///
/// Overlap is decided **purely by cell occupancy** — never by pairwise distance
/// or a neighbour list. Two occupancy models are supported, selected per
/// [`GrowthStrategy`](super::GrowthStrategy):
///
/// - [`OccupancyMode::SameCell`] — reject only if the candidate's own cell is
///   already taken. Used by lattice strategies whose step geometry already
///   guarantees a minimum separation (e.g. FCC: any two distinct sites are
///   `>= bond_length` apart), so the grid only has to forbid re-occupying a
///   site. The cell edge is chosen so each lattice site maps to a unique cell.
/// - [`OccupancyMode::BlockClear`] — reject if the candidate's cell or any of
///   its 26 neighbours (excluding the bonding tip's cell) is occupied. With a
///   cell edge equal to the excluded radius this guarantees every pair of
///   non-bonded monomers is at least one cell — i.e. `excluded_radius` — apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OccupancyMode {
    /// Reject iff the candidate's own cell is occupied. `cell` is the grid edge.
    SameCell {
        /// Grid cell edge length.
        cell: F,
    },
    /// Reject iff the candidate's cell or any 26-neighbour (minus the tip's
    /// cell) is occupied. `cell` equals the guaranteed minimum separation.
    BlockClear {
        /// Grid cell edge length, equal to the enforced minimum separation.
        cell: F,
    },
}

impl OccupancyMode {
    fn cell(self) -> F {
        match self {
            OccupancyMode::SameCell { cell } | OccupancyMode::BlockClear { cell } => cell,
        }
    }
}

/// A sparse periodic/reflective occupancy grid over the simulation box.
pub(crate) struct OccupancyGrid {
    mode: OccupancyMode,
    ncells: [i64; 3],
    pbc: Pbc3,
    occupied: HashSet<[i64; 3]>,
}

impl OccupancyGrid {
    /// Build an empty grid sized to `simbox` with the given boundary flags.
    pub fn new(mode: OccupancyMode, simbox: &SimBox, pbc: Pbc3) -> Self {
        let l = simbox.lengths();
        let cell = mode.cell();
        let ncells = [
            (l[0] / cell).floor().max(1.0) as i64,
            (l[1] / cell).floor().max(1.0) as i64,
            (l[2] / cell).floor().max(1.0) as i64,
        ];
        Self {
            mode,
            ncells,
            pbc,
            occupied: HashSet::new(),
        }
    }

    fn normalize_axis(&self, mut i: i64, ax: usize) -> i64 {
        let n = self.ncells[ax];
        if self.pbc[ax] {
            i = i.rem_euclid(n);
        } else {
            i = i.clamp(0, n - 1);
        }
        i
    }

    fn cell_of(&self, p: [F; 3]) -> [i64; 3] {
        let cell = self.mode.cell();
        [
            self.normalize_axis((p[0] / cell).floor() as i64, 0),
            self.normalize_axis((p[1] / cell).floor() as i64, 1),
            self.normalize_axis((p[2] / cell).floor() as i64, 2),
        ]
    }

    fn normalize_cell(&self, c: [i64; 3]) -> [i64; 3] {
        [
            self.normalize_axis(c[0], 0),
            self.normalize_axis(c[1], 1),
            self.normalize_axis(c[2], 2),
        ]
    }

    /// Is `p` free to occupy? `tip` (the bonding partner) is exempt from the
    /// neighbour scan so a bonded step is never self-blocked.
    pub fn is_free(&self, p: [F; 3], tip: Option<[F; 3]>) -> bool {
        let c = self.cell_of(p);
        match self.mode {
            OccupancyMode::SameCell { .. } => !self.occupied.contains(&c),
            OccupancyMode::BlockClear { .. } => {
                let tip_cell = tip.map(|t| self.cell_of(t));
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let nc = self.normalize_cell([c[0] + dx, c[1] + dy, c[2] + dz]);
                            if Some(nc) == tip_cell {
                                continue;
                            }
                            if self.occupied.contains(&nc) {
                                return false;
                            }
                        }
                    }
                }
                true
            }
        }
    }

    /// Mark `p`'s cell occupied.
    pub fn insert(&mut self, p: [F; 3]) {
        let c = self.cell_of(p);
        self.occupied.insert(c);
    }

    /// Free `p`'s cell (used when backtracking).
    pub fn remove(&mut self, p: [F; 3]) {
        let c = self.cell_of(p);
        self.occupied.remove(&c);
    }
}
