//! [`RegularGrid`] — a lattice of voxels: origin, basis, dimensions and
//! per-axis periodicity.

use molrs::core::{BoxKind, SimBox};
use molrs::op::F;

use super::SpreadKernel;
use crate::compute::ComputeError;
use crate::compute::named_enum::named_enum;

named_enum! {
    /// How far a [`RegularGrid::stencil`] looks for voxels within `r` along
    /// each grid axis, in voxels around the point.
    #[derive(Default)]
    pub enum StencilReach {
        /// Every voxel whose centre lies within `r`: the candidate range is
        /// `⌈·⌉`-wide enough to hold the whole ball.
        #[default]
        Ceil => "ceil",
        /// Only voxels at most `⌊r / Δ⌋` voxels from the one holding the
        /// point (`Δ` the voxel spacing along the axis), and within `r` —
        /// freud's `GaussianDensity` stencil, which skips the outermost shell
        /// of voxel centres inside `r`.
        Floor => "floor",
    }
}

/// A regular lattice of `dims[0] × … × dims[D−1]` voxels in `D` dimensions.
///
/// Voxel `(i₀, …)` spans the parallelepiped `origin + Σ_d (i_d + t_d) ·
/// basis[d]`, `t_d ∈ [0, 1)`; `basis[d]` is the edge vector of one voxel along
/// grid axis `d`. Its centre is at `t_d = ½`. An axis marked periodic wraps:
/// index `dims[d]` is index 0 again, so a kernel spread across the boundary
/// reappears on the other side. A non-periodic axis has no voxels outside
/// `[0, dims[d])`.
///
/// The one implementation of "which voxel holds this point" and "which voxels
/// lie within `r` of it" in `compute`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize),
    serde(bound(
        serialize = "[F; D]: serde::Serialize, [[F; D]; D]: serde::Serialize, \
        [usize; D]: serde::Serialize, [bool; D]: serde::Serialize"
    ))
)]
pub struct RegularGrid<const D: usize = 3> {
    origin: [F; D],
    basis: [[F; D]; D],
    dims: [usize; D],
    pbc: [bool; D],
    /// Whether `basis` is diagonal (an axis-aligned grid), in which case the
    /// fractional coordinate is a plain division along each axis.
    diagonal: bool,
    /// `M⁻¹` for `M` with the basis vectors as columns: the fractional
    /// coordinate is `t = M⁻¹ (p − origin)`.
    inverse: [[F; D]; D],
}

impl<const D: usize> RegularGrid<D> {
    /// A grid with voxel edge vectors `basis[d]`, `dims[d]` voxels along axis
    /// `d` from `origin`, periodic along the axes `pbc` marks.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a zero dimension, a non-finite origin
    /// or basis entry, or a singular (degenerate) basis.
    pub fn new(
        origin: [F; D],
        basis: [[F; D]; D],
        dims: [usize; D],
        pbc: [bool; D],
    ) -> Result<Self, ComputeError> {
        if dims.contains(&0) {
            return Err(ComputeError::OutOfRange {
                field: "RegularGrid::dims",
                value: format!("{dims:?}"),
            });
        }
        if origin
            .iter()
            .chain(basis.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(ComputeError::OutOfRange {
                field: "RegularGrid::origin/basis",
                value: format!("origin={origin:?}, basis={basis:?}"),
            });
        }
        let diagonal = (0..D).all(|d| (0..D).all(|k| d == k || basis[d][k] == 0.0));
        let mut columns = [[0.0; D]; D];
        for (d, edge) in basis.iter().enumerate() {
            for (k, &v) in edge.iter().enumerate() {
                columns[k][d] = v;
            }
        }
        let inverse = invert(columns).ok_or_else(|| ComputeError::OutOfRange {
            field: "RegularGrid::basis (singular)",
            value: format!("{basis:?}"),
        })?;
        Ok(Self {
            origin,
            basis,
            dims,
            pbc,
            diagonal,
            inverse,
        })
    }

    /// An axis-aligned, non-periodic grid of `dims` voxels spanning
    /// `[−extent/2, +extent/2]` along each axis — a grid centred on the
    /// origin of a body-fixed frame.
    ///
    /// # Errors
    /// As [`new`](Self::new), plus a non-positive extent.
    pub fn centered(extent: [F; D], dims: [usize; D]) -> Result<Self, ComputeError> {
        if extent.iter().any(|&e| !(e.is_finite() && e > 0.0)) || dims.contains(&0) {
            return Err(ComputeError::OutOfRange {
                field: "RegularGrid::centered",
                value: format!("extent={extent:?}, dims={dims:?}"),
            });
        }
        let mut origin = [0.0; D];
        let mut basis = [[0.0; D]; D];
        for d in 0..D {
            origin[d] = -0.5 * extent[d];
            basis[d][d] = extent[d] / dims[d] as F;
        }
        Self::new(origin, basis, dims, [false; D])
    }

    /// The corner of voxel `(0, …, 0)`.
    pub fn origin(&self) -> [F; D] {
        self.origin
    }

    /// The voxel edge vectors, one per grid axis.
    pub fn basis(&self) -> [[F; D]; D] {
        self.basis
    }

    /// Voxels per axis.
    pub fn dims(&self) -> [usize; D] {
        self.dims
    }

    /// Periodicity per axis.
    pub fn pbc(&self) -> [bool; D] {
        self.pbc
    }

    /// The volume of one voxel, `|det basis|` (an area in 2-D).
    pub fn voxel_volume(&self) -> F {
        if self.diagonal {
            return (0..D).map(|d| self.basis[d][d]).product::<F>().abs();
        }
        determinant(self.basis).abs()
    }

    /// The centre of voxel `index`.
    pub fn voxel_center(&self, index: [usize; D]) -> [F; D] {
        self.image_center(index.map(|i| i as isize))
    }

    /// The corner of voxel `index`, `origin + Σ_d index[d] · basis[d]` — the
    /// lattice node point-sampled data sits on. A Gaussian cube stores its
    /// value `(i, j, k)` there (not at the voxel centre), each sample standing
    /// for the voxel volume around it.
    pub fn voxel_corner(&self, index: [usize; D]) -> [F; D] {
        self.lattice_point(index.map(|i| i as isize), 0.0)
    }

    /// The voxel holding `point`, wrapped along periodic axes; `None` when the
    /// point lies outside the grid along a non-periodic axis.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a non-finite coordinate.
    pub fn index_of(&self, point: [F; D]) -> Result<Option<[usize; D]>, ComputeError> {
        check_finite(&point)?;
        let mut index = [0usize; D];
        for (d, slot) in index.iter_mut().enumerate() {
            let i = self.fractional(point, d).floor() as isize;
            match self.wrap(d, i) {
                Some(w) => *slot = w,
                None => return Ok(None),
            }
        }
        Ok(Some(index))
    }

    /// Visit every voxel whose centre lies within `r` of `center` (boundary
    /// included) as `(index, r²)`, with `r²` the squared distance from
    /// `center` to that voxel image's centre. Indices wrap along periodic
    /// axes; images beyond a non-periodic edge are skipped.
    ///
    /// With [`StencilReach::Ceil`] the candidate range along each axis
    /// covers every centre within `r` (the reach is `r` times the length of
    /// the matching row of `M⁻¹`), so the distance test alone decides
    /// membership. [`StencilReach::Floor`] narrows it to `⌊reach⌋` voxels on
    /// either side of the voxel holding `center`.
    ///
    /// # Errors
    /// [`ComputeError::NonFinite`] for a non-finite centre or radius;
    /// [`ComputeError::OutOfRange`] for a negative radius.
    pub fn stencil(
        &self,
        center: [F; D],
        r: F,
        reach: StencilReach,
        mut visit: impl FnMut([usize; D], F),
    ) -> Result<(), ComputeError> {
        check_finite(&center)?;
        if !r.is_finite() {
            return Err(ComputeError::NonFinite {
                where_: "RegularGrid::stencil radius",
                index: 0,
            });
        }
        if r < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "RegularGrid::stencil radius",
                value: r.to_string(),
            });
        }
        let r_sq_max = r * r;
        let mut lo = [0isize; D];
        let mut hi = [0isize; D];
        for d in 0..D {
            let t = self.fractional(center, d);
            let voxels = r * self.inverse[d].iter().map(|v| v * v).sum::<F>().sqrt();
            (lo[d], hi[d]) = match reach {
                StencilReach::Ceil => (
                    (t - voxels - 0.5).floor() as isize,
                    (t + voxels - 0.5).ceil() as isize,
                ),
                StencilReach::Floor => {
                    let (home, half) = (t.floor() as isize, voxels.floor() as isize);
                    (home - half, home + half)
                }
            };
        }
        let mut image = lo;
        loop {
            let c = self.image_center(image);
            let r_sq = (0..D).fold(0.0, |acc, k| {
                let v = c[k] - center[k];
                acc + v * v
            });
            if r_sq <= r_sq_max {
                let mut index = [0usize; D];
                if (0..D).all(|d| match self.wrap(d, image[d]) {
                    Some(w) => {
                        index[d] = w;
                        true
                    }
                    None => false,
                }) {
                    visit(index, r_sq);
                }
            }
            // Odometer over the candidate box, last axis fastest.
            let mut d = D;
            loop {
                if d == 0 {
                    return Ok(());
                }
                d -= 1;
                if image[d] < hi[d] {
                    image[d] += 1;
                    break;
                }
                image[d] = lo[d];
            }
        }
    }

    /// Spread `kernel` centred at `center`: visit every voxel within its
    /// cutoff as `(index, weight)`, `weight = normalization(D) · profile(r²)`.
    ///
    /// # Errors
    /// As [`stencil`](Self::stencil).
    pub fn spread<K: SpreadKernel + ?Sized>(
        &self,
        kernel: &K,
        center: [F; D],
        reach: StencilReach,
        mut visit: impl FnMut([usize; D], F),
    ) -> Result<(), ComputeError> {
        let norm = kernel.normalization(D);
        self.stencil(center, kernel.cutoff(), reach, |index, r_sq| {
            visit(index, norm * kernel.profile(r_sq));
        })
    }

    /// Fractional coordinate of `point` along grid axis `d`.
    fn fractional(&self, point: [F; D], d: usize) -> F {
        if self.diagonal {
            return (point[d] - self.origin[d]) / self.basis[d][d];
        }
        (0..D)
            .map(|k| self.inverse[d][k] * (point[k] - self.origin[k]))
            .sum()
    }

    /// Centre of the (possibly out-of-range) voxel image `index`.
    fn image_center(&self, index: [isize; D]) -> [F; D] {
        self.lattice_point(index, 0.5)
    }

    /// `origin + Σ_d (index[d] + t) · basis[d]`: the point at fractional
    /// offset `t` along every axis of the voxel image `index`.
    fn lattice_point(&self, index: [isize; D], t: F) -> [F; D] {
        let mut c = self.origin;
        if self.diagonal {
            for d in 0..D {
                c[d] += (index[d] as F + t) * self.basis[d][d];
            }
            return c;
        }
        for (d, edge) in self.basis.iter().enumerate() {
            let t = index[d] as F + t;
            for k in 0..D {
                c[k] += t * edge[k];
            }
        }
        c
    }

    /// Map an unwrapped index along axis `d` onto the grid: wrapped when the
    /// axis is periodic, `None` outside a non-periodic axis.
    fn wrap(&self, d: usize, i: isize) -> Option<usize> {
        let n = self.dims[d] as isize;
        if self.pbc[d] {
            Some(i.rem_euclid(n) as usize)
        } else if (0..n).contains(&i) {
            Some(i as usize)
        } else {
            None
        }
    }
}

impl RegularGrid<3> {
    /// The grid that divides `simbox` into `dims` voxels: the box origin, the
    /// box edge vectors over `dims`, and the box's periodicity.
    ///
    /// # Errors
    /// As [`new`](Self::new).
    pub fn over_box(simbox: &SimBox, dims: [usize; 3]) -> Result<Self, ComputeError> {
        if dims.contains(&0) {
            return Err(ComputeError::OutOfRange {
                field: "RegularGrid::dims",
                value: format!("{dims:?}"),
            });
        }
        let o = simbox.origin();
        let origin = [o[0], o[1], o[2]];
        let mut basis = [[0.0; 3]; 3];
        match simbox.kind() {
            BoxKind::Ortho { len, .. } => {
                for d in 0..3 {
                    basis[d][d] = len[d] / dims[d] as F;
                }
            }
            BoxKind::Triclinic => {
                let h = ndarray::arr2(&simbox.matrix());
                for (d, edge) in basis.iter_mut().enumerate() {
                    for (k, v) in edge.iter_mut().enumerate() {
                        *v = h[[k, d]] / dims[d] as F;
                    }
                }
            }
        }
        Self::new(origin, basis, dims, simbox.pbc())
    }
}

fn check_finite<const D: usize>(point: &[F; D]) -> Result<(), ComputeError> {
    match point.iter().position(|v| !v.is_finite()) {
        Some(index) => Err(ComputeError::NonFinite {
            where_: "grid point",
            index,
        }),
        None => Ok(()),
    }
}

/// Gauss–Jordan inverse with partial pivoting; `None` for a singular matrix.
fn invert<const D: usize>(m: [[F; D]; D]) -> Option<[[F; D]; D]> {
    let mut a = m;
    let mut inv = [[0.0; D]; D];
    for (d, row) in inv.iter_mut().enumerate() {
        row[d] = 1.0;
    }
    let scale = m.iter().flatten().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    for col in 0..D {
        let pivot = (col..D).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))?;
        if a[pivot][col].abs() <= 1e-12 * scale {
            return None;
        }
        a.swap(col, pivot);
        inv.swap(col, pivot);
        let p = a[col][col];
        for k in 0..D {
            a[col][k] /= p;
            inv[col][k] /= p;
        }
        for row in 0..D {
            if row != col {
                let f = a[row][col];
                for k in 0..D {
                    a[row][k] -= f * a[col][k];
                    inv[row][k] -= f * inv[col][k];
                }
            }
        }
    }
    Some(inv)
}

/// Determinant by Gaussian elimination.
fn determinant<const D: usize>(m: [[F; D]; D]) -> F {
    let mut a = m;
    let mut det = 1.0;
    for col in 0..D {
        let Some(pivot) = (col..D).max_by(|&x, &y| a[x][col].abs().total_cmp(&a[y][col].abs()))
        else {
            return 0.0;
        };
        if a[pivot][col] == 0.0 {
            return 0.0;
        }
        if pivot != col {
            a.swap(col, pivot);
            det = -det;
        }
        det *= a[col][col];
        let pivot_row = a[col];
        for row in a.iter_mut().skip(col + 1) {
            let f = row[col] / pivot_row[col];
            for (v, p) in row.iter_mut().zip(pivot_row.iter()).skip(col) {
                *v -= f * p;
            }
        }
    }
    det
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::{Gaussian, Indicator};
    use crate::core::Cell;

    fn cube(n: usize, pbc: bool) -> RegularGrid<3> {
        RegularGrid::new(
            [0.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [n; 3],
            [pbc; 3],
        )
        .unwrap()
    }

    #[test]
    fn rejects_degenerate_grids() {
        let id = [[1.0, 0.0], [0.0, 1.0]];
        assert!(RegularGrid::new([0.0; 2], id, [0, 3], [false; 2]).is_err());
        assert!(RegularGrid::new([F::NAN, 0.0], id, [3, 3], [false; 2]).is_err());
        assert!(RegularGrid::new([0.0; 2], [[1.0, 1.0], [2.0, 2.0]], [3, 3], [false; 2]).is_err());
        assert!(RegularGrid::<2>::centered([1.0, -1.0], [2, 2]).is_err());
    }

    #[test]
    fn centers_indices_and_volume() {
        let g = RegularGrid::<2>::centered([4.0, 2.0], [4, 2]).unwrap();
        assert_eq!(g.origin(), [-2.0, -1.0]);
        assert_eq!(g.voxel_center([0, 0]), [-1.5, -0.5]);
        assert_eq!(g.voxel_center([3, 1]), [1.5, 0.5]);
        assert_eq!(g.voxel_volume(), 1.0);
        assert_eq!(g.index_of([1.9, -0.1]).unwrap(), Some([3, 0]));
        assert_eq!(g.index_of([2.1, 0.0]).unwrap(), None);
        assert!(matches!(
            g.index_of([0.0, F::NAN]),
            Err(ComputeError::NonFinite { index: 1, .. })
        ));
    }

    /// A sheared basis: fractional coordinates come from the inverse, the
    /// voxel volume from the determinant.
    #[test]
    fn sheared_grid_indexes_through_the_inverse() {
        let g = RegularGrid::new([0.0; 2], [[2.0, 0.0], [1.0, 1.0]], [4, 4], [false; 2]).unwrap();
        assert!((g.voxel_volume() - 2.0).abs() < 1e-12);
        // origin + 1.5·(2, 0) + 2.5·(1, 1) = (5.5, 2.5): voxel (1, 2) centre.
        assert_eq!(g.voxel_center([1, 2]), [5.5, 2.5]);
        assert_eq!(g.index_of([5.5, 2.5]).unwrap(), Some([1, 2]));
    }

    /// Periodic axes wrap a stencil across the boundary; non-periodic axes
    /// drop the voxels beyond the edge.
    #[test]
    fn stencil_wraps_only_periodic_axes() {
        let mut periodic = Vec::new();
        cube(10, true)
            .stencil([0.1, 5.0, 5.0], 1.0, StencilReach::Ceil, |i, _| {
                periodic.push(i)
            })
            .unwrap();
        assert!(periodic.contains(&[9, 4, 4]), "wrapped image of voxel -1");
        assert!(periodic.contains(&[0, 4, 4]));

        let mut open = Vec::new();
        cube(10, false)
            .stencil([0.1, 5.0, 5.0], 1.0, StencilReach::Ceil, |i, _| {
                open.push(i)
            })
            .unwrap();
        assert!(
            open.iter().all(|i| i[0] <= 1),
            "no voxel past the open edge"
        );
        assert!(open.contains(&[0, 4, 4]));
        assert!(open.len() < periodic.len());
    }

    /// The stencil holds exactly the voxels whose centres lie within `r`,
    /// boundary included — a brute-force scan agrees.
    #[test]
    fn stencil_is_exactly_the_ball() {
        let g = cube(12, false);
        for (center, r) in [
            ([5.3, 6.9, 4.2], 2.7),
            ([5.5, 5.5, 5.5], 2.0),
            ([0.2, 11.7, 6.0], 3.1),
        ] {
            let mut got = Vec::new();
            g.stencil(center, r, StencilReach::Ceil, |i, r_sq| got.push((i, r_sq)))
                .unwrap();
            let mut want = Vec::new();
            for i in 0..12 {
                for j in 0..12 {
                    for k in 0..12 {
                        let c = g.voxel_center([i, j, k]);
                        let r_sq: F = (0..3).map(|a| (c[a] - center[a]).powi(2)).sum();
                        if r_sq <= r * r {
                            want.push([i, j, k]);
                        }
                    }
                }
            }
            let mut got_index: Vec<_> = got.iter().map(|(i, _)| *i).collect();
            got_index.sort();
            assert_eq!(got_index, want, "centre {center:?}, r {r}");
        }
    }

    #[test]
    fn spread_weights_are_the_kernel() {
        let g = cube(8, true);
        let mut total = 0.0;
        g.spread(
            &Indicator::new(1.0).unwrap(),
            [4.5, 4.5, 4.5],
            StencilReach::Ceil,
            |_, w| total += w,
        )
        .unwrap();
        // The centre voxel and its six face neighbours.
        assert_eq!(total, 7.0);

        let k = Gaussian::new(0.5).unwrap();
        let mut at_center = 0.0;
        g.spread(&k, [4.5, 4.5, 4.5], StencilReach::Ceil, |i, w| {
            if i == [4, 4, 4] {
                at_center = w;
            }
        })
        .unwrap();
        assert!((at_center - k.normalization(3)).abs() < 1e-15);
    }

    /// Voxel corners are the lattice nodes, in index order (the former
    /// Voronoi `DensityGrid::point` golden).
    #[test]
    fn voxel_corners_are_the_lattice_nodes() {
        let g = cube(2, false);
        assert!((g.voxel_volume() - 1.0).abs() < 1e-12);
        // m = ((i*ny)+j)*nz + k ; m=5 → i=1,j=0,k=1
        assert_eq!(g.voxel_corner([1, 0, 1]), [1.0, 0.0, 1.0]);
        let sheared =
            RegularGrid::new([0.5, 0.0], [[2.0, 0.0], [1.0, 1.0]], [4, 4], [false; 2]).unwrap();
        assert_eq!(sheared.voxel_corner([1, 2]), [4.5, 2.0]);
        assert_eq!(sheared.voxel_corner([0, 0]), sheared.origin());
    }

    #[test]
    fn over_box_divides_the_cell() {
        let simbox = SimBox::new(
            Cell::Lengths([10.0, 20.0, 30.0]),
            [1.0, 2.0, 3.0],
            [true, false, true],
        )
        .unwrap();
        let g = RegularGrid::over_box(&simbox, [10, 10, 10]).unwrap();
        assert_eq!(g.origin(), [1.0, 2.0, 3.0]);
        assert_eq!(g.basis()[1], [0.0, 2.0, 0.0]);
        assert_eq!(g.pbc(), [true, false, true]);
        assert!((g.voxel_volume() - 6.0).abs() < 1e-12);
    }
}
