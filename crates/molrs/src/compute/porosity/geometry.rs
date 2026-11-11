//! Periodic clearance of a point against hard spheres.
//!
//! The clearance at a point is its distance to the nearest atom surface,
//! negative when the point lies inside a sphere. Distances use the box
//! minimum image. An orthogonal, fully periodic box is indexed by a cell
//! list; every other box is scanned directly, with the same distance.

use molrs::core::{BoxKind, Mic, SimBox};
use molrs::op::F;
use ndarray::{ArrayView1, ArrayView2};

use crate::compute::ComputeError;

/// Atoms as hard spheres: Cartesian centres and van der Waals radii, both Å.
pub(crate) struct Spheres {
    pub pos: Vec<[F; 3]>,
    pub rad: Vec<F>,
    pub max_r: F,
}

/// Read `N × 3` positions and one radius per atom.
pub(crate) fn read_spheres(
    positions: ArrayView2<'_, F>,
    radii: ArrayView1<'_, F>,
) -> Result<Spheres, ComputeError> {
    if positions.ncols() != 3 {
        return Err(ComputeError::BadShape {
            expected: "positions N × 3".into(),
            got: format!("N × {}", positions.ncols()),
        });
    }
    let n = positions.nrows();
    if n == 0 {
        return Err(ComputeError::EmptyInput);
    }
    if radii.len() != n {
        return Err(ComputeError::DimensionMismatch {
            expected: n,
            got: radii.len(),
            what: "radii length",
        });
    }
    let mut pos = Vec::with_capacity(n);
    let mut rad = Vec::with_capacity(n);
    let mut max_r: F = 0.0;
    for i in 0..n {
        let point = [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
        if point.iter().any(|v| !v.is_finite()) {
            return Err(ComputeError::NonFinite {
                where_: "positions",
                index: i,
            });
        }
        let radius = radii[i];
        if !radius.is_finite() || radius < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "radii",
                value: format!("index {i}: {radius}"),
            });
        }
        max_r = max_r.max(radius);
        pos.push(point);
        rad.push(radius);
    }
    Ok(Spheres { pos, rad, max_r })
}

/// A box with a positive volume, plus the geometry clearance queries need.
pub(crate) struct CellGeom {
    pub origin: [F; 3],
    pub matrix: [[F; 3]; 3],
    pub lengths: [F; 3],
    pub pbc: [bool; 3],
    pub volume: F,
    pub mic: Mic,
    pub orthogonal: bool,
}

pub(crate) fn read_cell(simbox: &SimBox) -> Result<CellGeom, ComputeError> {
    let volume = simbox.volume();
    if !volume.is_finite() || volume <= 0.0 {
        return Err(ComputeError::OutOfRange {
            field: "simbox volume",
            value: volume.to_string(),
        });
    }
    let lengths = simbox.lengths();
    if lengths.iter().any(|l| !l.is_finite() || *l <= 0.0) {
        return Err(ComputeError::OutOfRange {
            field: "simbox lengths",
            value: format!("{lengths:?}"),
        });
    }
    Ok(CellGeom {
        origin: simbox.origin(),
        matrix: simbox.matrix(),
        lengths: [lengths[0], lengths[1], lengths[2]],
        pbc: simbox.pbc(),
        volume,
        mic: simbox.mic(),
        orthogonal: matches!(simbox.kind(), &BoxKind::Ortho { .. }),
    })
}

/// `origin + H · frac` for `frac` in `[0, 1)³`.
pub(crate) fn cartesian(cell: &CellGeom, frac: [F; 3]) -> [F; 3] {
    let mut point = cell.origin;
    for (row, slot) in point.iter_mut().enumerate() {
        for (col, component) in frac.iter().enumerate() {
            *slot += cell.matrix[row][col] * component;
        }
    }
    point
}

/// Nearest-surface queries against [`Spheres`].
pub(crate) struct SurfaceIndex {
    spheres: Spheres,
    pub cell: CellGeom,
    dims: [usize; 3],
    head: Vec<i32>,
    next: Vec<i32>,
}

impl SurfaceIndex {
    pub(crate) fn build(spheres: Spheres, cell: CellGeom) -> Self {
        let dims = cell_dims(&cell, spheres.max_r);
        let n_cells = dims[0] * dims[1] * dims[2];
        let mut head = vec![-1; n_cells];
        let mut next = vec![-1; spheres.pos.len()];
        for (i, point) in spheres.pos.iter().enumerate() {
            let slot = flat(dims, cell_of(&cell, dims, *point));
            next[i] = head[slot];
            head[slot] = i as i32;
        }
        Self {
            spheres,
            cell,
            dims,
            head,
            next,
        }
    }

    /// Distance to the nearest sphere surface, Å. Negative inside a sphere.
    pub(crate) fn clearance(&self, point: [F; 3]) -> F {
        let home = cell_of(&self.cell, self.dims, point);
        let mut best = F::INFINITY;
        let min_edge = self.edge();
        let shells = self.dims.iter().copied().max().unwrap_or(1);
        for shell in 0..=shells {
            self.visit_shell(home, shell, point, &mut best);
            if best <= 0.0 {
                return best;
            }
            if shell as F * min_edge > best + self.spheres.max_r {
                break;
            }
        }
        best
    }

    fn edge(&self) -> F {
        self.cell
            .lengths
            .iter()
            .zip(self.dims)
            .map(|(length, n)| length / n as F)
            .fold(F::INFINITY, F::min)
    }

    fn visit_shell(&self, home: [usize; 3], shell: usize, point: [F; 3], best: &mut F) {
        let shell = shell as isize;
        for di in -shell..=shell {
            for dj in -shell..=shell {
                for dk in -shell..=shell {
                    if di.abs().max(dj.abs()).max(dk.abs()) != shell {
                        continue;
                    }
                    let Some(slot) = self.shift(home, [di, dj, dk]) else {
                        continue;
                    };
                    let mut atom = self.head[slot];
                    while atom >= 0 {
                        let i = atom as usize;
                        let clear = self.distance(point, self.spheres.pos[i]) - self.spheres.rad[i];
                        if clear < *best {
                            *best = clear;
                        }
                        atom = self.next[i];
                    }
                }
            }
        }
    }

    fn distance(&self, a: [F; 3], b: [F; 3]) -> F {
        let d = self.cell.mic.apply([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    fn shift(&self, home: [usize; 3], delta: [isize; 3]) -> Option<usize> {
        let mut idx = [0; 3];
        for k in 0..3 {
            let n = self.dims[k] as isize;
            let i = home[k] as isize + delta[k];
            if self.cell.pbc[k] {
                idx[k] = i.rem_euclid(n) as usize;
            } else if (0..n).contains(&i) {
                idx[k] = i as usize;
            } else {
                return None;
            }
        }
        Some(flat(self.dims, idx))
    }
}

/// Cell counts for an orthogonal periodic box. Anything else is one cell,
/// which the shell walk scans in full.
fn cell_dims(cell: &CellGeom, max_r: F) -> [usize; 3] {
    if !(cell.orthogonal && cell.pbc == [true, true, true]) {
        return [1, 1, 1];
    }
    let scale = max_r.max(1.0);
    std::array::from_fn(|k| (cell.lengths[k] / scale).floor().max(1.0) as usize)
}

fn cell_of(cell: &CellGeom, dims: [usize; 3], point: [F; 3]) -> [usize; 3] {
    // The one-cell index holds every atom, so the bin does not matter.
    // An orthogonal periodic box bins by the wrapped fractional coordinate
    // along each lattice axis, which is `(point − origin) / length`.
    if dims == [1, 1, 1] {
        return [0, 0, 0];
    }
    std::array::from_fn(|k| {
        let mut t = (point[k] - cell.origin[k]) / cell.lengths[k];
        t -= t.floor();
        let mut i = (t * dims[k] as F).floor() as isize;
        if i >= dims[k] as isize {
            i = dims[k] as isize - 1;
        }
        if i < 0 {
            i = 0;
        }
        i as usize
    })
}

fn flat(dims: [usize; 3], idx: [usize; 3]) -> usize {
    (idx[0] * dims[1] + idx[1]) * dims[2] + idx[2]
}
