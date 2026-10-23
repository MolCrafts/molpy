//! Triclinic simulation box and periodic operations: [`SimBox`].

use crate::op::vec3::{cross, dot, norm};
use crate::op::{F, F3, F3View, Fnx3, Fnx3View, I, Pbc3};
use crate::op::{Vec3, to_mat3, to_vec3};
use crate::op::{det3, inv3};
use ndarray::{Array1, Array2, Array3, ArrayView2, array};

/// Box geometry descriptor, detected once at construction.
/// Consumers with shape-specific domains may inspect it. Coordinate transforms,
/// wrapping and minimum-image distances do not dispatch on this descriptor.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxKind {
    /// Orthorhombic (diagonal H): lengths, inverse lengths cached.
    Ortho { len: F3, inv_len: F3 },
    /// General triclinic.
    Triclinic,
}

/// Simulation box: triclinic cell with origin and per-axis PBC mask.
///
/// Conventions (fractional/cartesian):
/// - cart = origin + H * frac
/// - frac = H^{-1} * (cart - origin)
/// - Lattice vectors are the columns of H.
#[derive(Debug, Clone)]
pub struct SimBox {
    /// Triclinic cell matrix H (columns are lattice vectors)
    h: Fnx3,
    /// Precomputed inverse of H
    inv: Fnx3,
    /// Origin of the cell in Cartesian coordinates
    origin: F3,
    /// Per-axis periodic boundary condition flags (x, y, z)
    pbc: Pbc3,
    /// Cached geometry kind
    kind: BoxKind,
    /// The minimum-image convention of this cell, captured once
    mic: Mic,
    /// Whether the cell is geometrically defined. `false` marks a "no-cell"
    /// box (an undefined / zero-volume cell) — distinct from `pbc`, which only
    /// describes periodicity. A defined non-periodic box (e.g. a free-boundary
    /// bounding box) keeps `cell_defined = true`; only a box with no meaningful
    /// cell at all sets it `false`, and such a box always carries the identity
    /// matrix (see [`Cell::Undefined`]).
    cell_defined: bool,
}

/// Error type for simulation box construction.
#[derive(Debug, Clone, PartialEq)]
pub enum BoxError {
    /// The cell matrix H is singular (determinant ≈ 0).
    SingularCell,
    /// The periodic lattice could not be reduced to a certified basis.
    LatticeReduction,
    /// The matrix does not have shape 3x3.
    InvalidMatrixShape { rows: usize, cols: usize },
    /// A vector does not have the expected length.
    InvalidVectorLength { len: usize },
    /// A required array is not contiguous in memory.
    NonContiguous(&'static str),
    /// Cell lengths/angles do not describe a physical triclinic cell.
    InvalidAngles,
    /// A [`Cell::Lengths`] / [`Cell::LengthsTilts`] edge is not a positive
    /// finite number.
    NonPositiveLength { lengths: [F; 3] },
    /// [`SimBox::bounding`] was given no points: an empty set has no bounds.
    EmptyPoints,
    /// [`SimBox::bounding`] was given a negative or non-finite padding.
    InvalidPadding { padding: [F; 3] },
    /// The bounding box of [`SimBox::bounding`] has zero extent along `axis`
    /// (every point shares that coordinate and the padding is zero).
    ZeroExtent { axis: usize },
    /// The [`Points::Columns`] slices (or a [`Points::Rows`] view's column
    /// count) do not describe `N` three-dimensional points.
    PointsShape { lens: [usize; 3] },
    /// A [`Cell::Undefined`] box was given a periodic axis: an undefined cell
    /// has no lattice to repeat along, so it is periodic on no axis.
    PeriodicWithoutCell { pbc: Pbc3 },
}

impl std::fmt::Display for BoxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoxError::SingularCell => write!(f, "the cell matrix is singular"),
            BoxError::LatticeReduction => {
                write!(f, "periodic lattice reduction could not be certified")
            }
            BoxError::InvalidMatrixShape { rows, cols } => {
                write!(f, "a cell matrix is 3x3, got {rows}x{cols}")
            }
            BoxError::InvalidVectorLength { len } => {
                write!(f, "expected a length-3 vector, got length {len}")
            }
            BoxError::NonContiguous(what) => write!(f, "{what} is not contiguous in memory"),
            BoxError::InvalidAngles => {
                write!(
                    f,
                    "cell lengths/angles do not describe a physical triclinic cell"
                )
            }
            BoxError::NonPositiveLength { lengths } => write!(
                f,
                "cell edge lengths must be positive and finite, got {lengths:?}; \
                 a box without a cell is Cell::Undefined"
            ),
            BoxError::EmptyPoints => write!(
                f,
                "a bounding box needs at least one point; build an empty system's box \
                 with SimBox::new"
            ),
            BoxError::InvalidPadding { padding } => write!(
                f,
                "bounding-box padding must be finite and >= 0, got {padding:?}"
            ),
            BoxError::ZeroExtent { axis } => write!(
                f,
                "the bounding box has zero extent along axis {axis}; pass a positive \
                 padding for that axis"
            ),
            BoxError::PeriodicWithoutCell { pbc } => write!(
                f,
                "an undefined cell is periodic on no axis, but this box carries boundary {pbc:?}"
            ),
            BoxError::PointsShape { lens } => write!(
                f,
                "points must be N x 3 (three equal-length x/y/z columns), got {lens:?}"
            ),
        }
    }
}

impl std::error::Error for BoxError {}

/// The geometry a [`SimBox`] is built from — the one argument of
/// [`SimBox::new`] that says what the cell is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    /// The cell matrix `H`, `h[row][col]`, lattice vectors as **columns**:
    /// the first lattice vector is `a = (h[0][0], h[1][0], h[2][0])`. The
    /// layout [`SimBox::matrix`] returns.
    Matrix([[F; 3]; 3]),
    /// An orthorhombic cell with edge lengths `[a, b, c]` (Å), all positive.
    Lengths([F; 3]),
    /// A restricted-triclinic cell from edge lengths `[a, b, c]` (Å) and the
    /// angles `[alpha, beta, gamma]` between them in degrees.
    LengthsAngles { lengths: [F; 3], angles: [F; 3] },
    /// A restricted-triclinic cell from the diagonal sizes `[lx, ly, lz]` (Å)
    /// and the tilts `[xy, xz, yz]` (Å) — the LAMMPS form.
    LengthsTilts { lengths: [F; 3], tilts: [F; 3] },
    /// No geometric cell: a "no-cell" box of undefined, zero volume. The box
    /// carries the identity matrix so it is never inverted, a writer emits the
    /// identity, and [`SimBox::volume`] / [`SimBox::is_cell_defined`] report
    /// the undefined cell. Distinct from periodicity: a defined non-periodic
    /// box (a free-boundary bounding box) has a cell.
    Undefined,
}

impl Cell {
    /// The cell matrix, or `None` for [`Cell::Undefined`].
    pub(crate) fn resolve(self) -> Result<Option<Fnx3>, BoxError> {
        let positive = |lengths: [F; 3]| {
            if lengths.iter().all(|l| l.is_finite() && *l > 0.0) {
                Ok(())
            } else {
                Err(BoxError::NonPositiveLength { lengths })
            }
        };
        Ok(match self {
            Cell::Matrix(h) => Some(ndarray::arr2(&h)),
            Cell::Lengths(lengths) => {
                positive(lengths)?;
                let [a, b, c] = lengths;
                Some(array![[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]])
            }
            Cell::LengthsAngles { lengths, angles } => {
                Some(matrix_from_lengths_angles(lengths, angles)?)
            }
            Cell::LengthsTilts { lengths, tilts } => {
                positive(lengths)?;
                Some(array![
                    [lengths[0], tilts[0], tilts[1]],
                    [0.0, lengths[1], tilts[2]],
                    [0.0, 0.0, lengths[2]],
                ])
            }
            Cell::Undefined => None,
        })
    }
}

/// Restricted-triclinic matrix from edge lengths and angles in degrees.
fn matrix_from_lengths_angles(lengths: [F; 3], angles: [F; 3]) -> Result<Fnx3, BoxError> {
    let [a, b, c] = lengths;
    let [alpha, beta, gamma] = angles.map(F::to_radians);
    if [a, b, c].iter().any(|value| *value <= 0.0)
        || [alpha, beta, gamma]
            .iter()
            .any(|angle| !(*angle > 0.0 && *angle < std::f64::consts::PI))
    {
        return Err(BoxError::InvalidAngles);
    }
    let (cos_a, cos_b, cos_c) = (alpha.cos(), beta.cos(), gamma.cos());
    let xy = b * cos_c;
    let xz = c * cos_b;
    let ly = (b * b - xy * xy).sqrt();
    if !ly.is_finite() || ly <= 0.0 {
        return Err(BoxError::InvalidAngles);
    }
    let yz = (b * c * cos_a - xy * xz) / ly;
    let lz2 = c * c - xz * xz - yz * yz;
    if !lz2.is_finite() || lz2 <= 0.0 {
        return Err(BoxError::InvalidAngles);
    }
    Ok(array![[a, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz2.sqrt()]])
}

/// A borrowed set of `N` Cartesian points (Å), in either memory layout.
///
/// The one point argument of [`SimBox::bounding`] and of the neighbor searches
/// ([`NeighborList::build`](crate::core::NeighborList::build),
/// [`NeighborQuery::new`](crate::core::NeighborQuery::new), …): a caller holding
/// an `N × 3` array passes [`Points::Rows`] (or the view itself — it converts),
/// a caller holding a column store (`atoms.x` / `atoms.y` / `atoms.z`) passes
/// [`Points::Columns`] without interleaving them first.
#[derive(Debug, Clone, Copy)]
pub enum Points<'a> {
    /// An `N × 3` array, one row per point.
    Rows(Fnx3View<'a>),
    /// Three length-`N` coordinate columns `[x, y, z]`; point `i` is
    /// `(x[i], y[i], z[i])`.
    Columns([&'a [F]; 3]),
}

impl<'a> From<Fnx3View<'a>> for Points<'a> {
    fn from(rows: Fnx3View<'a>) -> Self {
        Points::Rows(rows)
    }
}

impl<'a> From<&'a Fnx3> for Points<'a> {
    fn from(rows: &'a Fnx3) -> Self {
        Points::Rows(rows.view())
    }
}

impl Points<'_> {
    /// The number of points.
    pub fn len(&self) -> usize {
        match self {
            Points::Rows(rows) => rows.nrows(),
            Points::Columns([x, _, _]) => x.len(),
        }
    }

    /// Whether there are no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Point `i` as `[x, y, z]`.
    ///
    /// # Panics
    /// If `i` is out of range.
    #[inline]
    pub fn point(&self, i: usize) -> [F; 3] {
        match self {
            Points::Rows(rows) => [rows[[i, 0]], rows[[i, 1]], rows[[i, 2]]],
            Points::Columns([x, y, z]) => [x[i], y[i], z[i]],
        }
    }

    /// Refuse a set that is not `N` three-dimensional points: a row view
    /// without exactly 3 columns, or columns of unequal length.
    pub(crate) fn check(&self) -> Result<(), BoxError> {
        match self {
            Points::Rows(rows) if rows.ncols() != 3 => Err(BoxError::PointsShape {
                lens: [rows.nrows(), rows.ncols(), 0],
            }),
            Points::Columns([x, y, z]) if x.len() != y.len() || y.len() != z.len() => {
                Err(BoxError::PointsShape {
                    lens: [x.len(), y.len(), z.len()],
                })
            }
            _ => Ok(()),
        }
    }

    /// The points as an owned `N × 3` array.
    pub(crate) fn to_rows(self) -> Fnx3 {
        match self {
            Points::Rows(rows) => rows.to_owned(),
            Points::Columns(_) => {
                let n = self.len();
                let mut out = Fnx3::zeros((n, 3));
                for i in 0..n {
                    let p = self.point(i);
                    out[[i, 0]] = p[0];
                    out[[i, 1]] = p[1];
                    out[[i, 2]] = p[2];
                }
                out
            }
        }
    }
}

impl SimBox {
    /// Refinement passes [`wrap`](Self::wrap) may take to land a crossed atom
    /// strictly inside the cell. Two is already generous: the first pass fixes
    /// the ordinary "rounded onto the far face" landing, the second covers a
    /// triclinic cross-axis perturbation. More would only mask a real defect.
    const WRAP_REFINE_PASSES: usize = 2;

    /// Construct a box from its `cell`, its `origin` (Å) and per-axis
    /// periodicity `pbc`.
    ///
    /// ```
    /// use molrs::core::{Cell, SimBox};
    ///
    /// let cube = SimBox::new(Cell::Lengths([10.0; 3]), [0.0; 3], [true; 3]).unwrap();
    /// assert_eq!(cube.volume(), 1000.0);
    /// let none = SimBox::new(Cell::Undefined, [0.0; 3], [false; 3]).unwrap();
    /// assert!(!none.is_cell_defined());
    /// ```
    ///
    /// # Errors
    /// [`BoxError::SingularCell`] for a singular [`Cell::Matrix`],
    /// [`BoxError::NonPositiveLength`] for a non-positive or non-finite edge,
    /// [`BoxError::InvalidAngles`] for lengths/angles that describe no cell,
    /// [`BoxError::PeriodicWithoutCell`] for a [`Cell::Undefined`] with a
    /// periodic axis.
    pub fn new(cell: Cell, origin: [F; 3], pbc: Pbc3) -> Result<Self, BoxError> {
        let h = cell.resolve()?;
        Self::from_parts(h, ndarray::arr1(&origin), pbc)
    }

    /// The box over cell matrix `h` (`None`: no cell), origin and pbc.
    fn from_parts(h: Option<Fnx3>, origin: F3, pbc: Pbc3) -> Result<Self, BoxError> {
        let cell_defined = h.is_some();
        if !cell_defined && pbc.iter().any(|&periodic| periodic) {
            return Err(BoxError::PeriodicWithoutCell { pbc });
        }
        let (h, inv) = match h {
            Some(h) => {
                let inv = inv3(&to_mat3(h.view())).ok_or(BoxError::SingularCell)?;
                (h, ndarray::arr2(&inv))
            }
            None => (Fnx3::eye(3), Fnx3::eye(3)),
        };
        let kind = detect_box_kind(&h);
        let mic = Mic::new(to_mat3(h.view()), pbc)?;
        Ok(Self {
            h,
            inv,
            origin,
            pbc,
            kind,
            mic,
            cell_defined,
        })
    }

    /// Whether the cell is geometrically defined (`false` ⇒ a no-cell box of
    /// undefined / zero volume, [`Cell::Undefined`]). Distinct from
    /// periodicity ([`is_free`]).
    ///
    /// [`is_free`]: SimBox::is_free
    pub fn is_cell_defined(&self) -> bool {
        self.cell_defined
    }

    /// The cell matrix `H` as a plain `[[F; 3]; 3]`, `h[row][col]`, lattice
    /// vectors as **columns** — the layout [`Cell::Matrix`] takes, so
    /// `SimBox::new(Cell::Matrix(b.matrix()), b.origin(), b.pbc())` rebuilds
    /// the cell of `b` exactly.
    pub fn matrix(&self) -> [[F; 3]; 3] {
        std::array::from_fn(|r| std::array::from_fn(|c| self.h[[r, c]]))
    }

    /// The origin of the cell (Å).
    pub fn origin(&self) -> [F; 3] {
        [self.origin[0], self.origin[1], self.origin[2]]
    }

    /// Whether two boxes describe the same cell within an **absolute**
    /// tolerance `tol` (length units).
    ///
    /// True iff every entry of the two cell matrices and every origin
    /// component differ by at most `tol`, the PBC flags are identical, and
    /// both boxes agree on [`is_cell_defined`](Self::is_cell_defined). The
    /// flags are compared exactly — periodicity has no "nearly". `tol = 0.0`
    /// is bitwise equality of the numbers (with `-0.0 == 0.0`); a NaN entry
    /// never compares equal.
    ///
    /// The comparison is per entry, not up to a lattice reduction: two
    /// matrices that span the same lattice with a different basis are *not*
    /// equal here.
    ///
    /// # Panics
    /// If `tol` is negative or NaN.
    pub fn approx_eq(&self, other: &SimBox, tol: F) -> bool {
        assert!(tol >= 0.0, "tolerance must be non-negative");
        let close = |a: F, b: F| (a - b).abs() <= tol;
        self.pbc == other.pbc
            && self.cell_defined == other.cell_defined
            && self
                .h
                .iter()
                .zip(other.h.iter())
                .all(|(a, b)| close(*a, *b))
            && self
                .origin
                .iter()
                .zip(other.origin.iter())
                .all(|(a, b)| close(*a, *b))
    }

    /// Convert a general cell matrix to LAMMPS restricted-triclinic form.
    pub fn restricted_matrix(matrix: Fnx3View<'_>) -> Result<Fnx3, BoxError> {
        if matrix.dim() != (3, 3) {
            return Err(BoxError::InvalidMatrixShape {
                rows: matrix.nrows(),
                cols: matrix.ncols(),
            });
        }
        let a = to_vec3(matrix.column(0));
        let b = to_vec3(matrix.column(1));
        let c = to_vec3(matrix.column(2));
        let ax = norm(a);
        if ax <= 0.0 {
            return Err(BoxError::SingularCell);
        }
        let ua = [a[0] / ax, a[1] / ax, a[2] / ax];
        let bx = dot(b, ua);
        let cross_ab = cross(a, b);
        let cross_norm = norm(cross_ab);
        if cross_norm <= 0.0 {
            return Err(BoxError::SingularCell);
        }
        let by = norm(cross(ua, b));
        let uab = [
            cross_ab[0] / cross_norm,
            cross_ab[1] / cross_norm,
            cross_ab[2] / cross_norm,
        ];
        let cx = dot(c, ua);
        let cy = dot(c, cross(uab, ua));
        let cz = dot(c, uab);
        Ok(array![[ax, bx, cx], [0.0, by, cy], [0.0, 0.0, cz]])
    }

    /// The orthorhombic box enclosing `points`: their axis-aligned bounds
    /// widened by `padding[d]` on each side of axis `d`, with periodicity
    /// `pbc`.
    ///
    /// A free-boundary neighbor search pads by (at least) its cutoff so every
    /// particle sits well inside the box; a tight box passes a zero padding.
    ///
    /// ```
    /// use molrs::core::{Points, SimBox};
    ///
    /// let xs = [0.0, 2.0];
    /// let ys = [0.0, 1.0];
    /// let zs = [0.0, 3.0];
    /// let bx = SimBox::bounding(Points::Columns([&xs, &ys, &zs]), [1.0; 3], [false; 3]).unwrap();
    /// assert_eq!(bx.origin(), [-1.0, -1.0, -1.0]);
    /// assert_eq!(bx.matrix()[0][0], 4.0);
    /// ```
    ///
    /// # Errors
    /// [`BoxError::EmptyPoints`] when there are no points,
    /// [`BoxError::InvalidPadding`] for a negative or non-finite padding,
    /// [`BoxError::ZeroExtent`] when an axis has zero extent (all points share
    /// the coordinate and its padding is zero), [`BoxError::PointsShape`] when
    /// `points` is not `N × 3`.
    pub fn bounding<'a>(
        points: impl Into<Points<'a>>,
        padding: [F; 3],
        pbc: Pbc3,
    ) -> Result<Self, BoxError> {
        let points = points.into();
        points.check()?;
        if points.is_empty() {
            return Err(BoxError::EmptyPoints);
        }
        if padding.iter().any(|p| !p.is_finite() || *p < 0.0) {
            return Err(BoxError::InvalidPadding { padding });
        }
        let mut min = points.point(0);
        let mut max = min;
        for i in 1..points.len() {
            let p = points.point(i);
            for d in 0..3 {
                min[d] = min[d].min(p[d]);
                max[d] = max[d].max(p[d]);
            }
        }
        let mut lengths = [0.0; 3];
        for d in 0..3 {
            lengths[d] = max[d] - min[d] + 2.0 * padding[d];
            if lengths[d] <= 0.0 {
                return Err(BoxError::ZeroExtent { axis: d });
            }
        }
        let origin = std::array::from_fn(|d| min[d] - padding[d]);
        Self::new(Cell::Lengths(lengths), origin, pbc)
    }

    /// View of the inverse cell matrix
    pub fn inv_view(&self) -> Fnx3View<'_> {
        self.inv.view()
    }

    /// Per-axis PBC flags
    pub fn pbc(&self) -> Pbc3 {
        self.pbc
    }

    /// Cell volume (|det(H)|); `0.0` for a no-cell box.
    ///
    /// A no-cell box carries the identity matrix only so geometry ops are
    /// no-ops; its determinant (1.0) is not a volume, and reporting it would
    /// silently normalize densities by a made-up cell. Consumers that need a
    /// volume (RDF, van Hove) already refuse a non-positive one.
    pub fn volume(&self) -> F {
        if !self.cell_defined {
            return 0.0;
        }
        det3(&to_mat3(self.h.view())).abs()
    }

    /// `true` when the box is free (non-periodic on every axis).
    pub fn is_free(&self) -> bool {
        self.pbc.iter().all(|&p| !p)
    }

    /// Geometry style label: `"free"` (no periodic axis — every no-cell box
    /// is one), `"orthogonal"` (diagonal H), or `"triclinic"`.
    pub fn style(&self) -> &'static str {
        if self.is_free() {
            "free"
        } else {
            match self.kind {
                BoxKind::Ortho { .. } => "orthogonal",
                BoxKind::Triclinic => "triclinic",
            }
        }
    }

    /// Off-diagonal tilts [xy, xz, yz] of the cell matrix
    pub fn tilts(&self) -> F3 {
        array![self.h[[0, 1]], self.h[[0, 2]], self.h[[1, 2]]]
    }

    /// Lattice vector lengths
    pub fn lengths(&self) -> F3 {
        let a = to_vec3(self.lattice(0).view());
        let b = to_vec3(self.lattice(1).view());
        let c = to_vec3(self.lattice(2).view());
        array![norm(a), norm(b), norm(c)]
    }

    /// Lattice angles `[alpha, beta, gamma]` in degrees.
    pub fn angles(&self) -> F3 {
        let a = to_vec3(self.lattice(0).view());
        let b = to_vec3(self.lattice(1).view());
        let c = to_vec3(self.lattice(2).view());
        let angle = |u: &Vec3, v: &Vec3| {
            (dot(*u, *v) / (norm(*u) * norm(*v)))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        };
        array![angle(&b, &c), angle(&a, &c), angle(&a, &b)]
    }

    /// Spacing between the two parallel faces normal to each reciprocal
    /// direction: `d_k = V / ||a_i x a_j|| = 1 / ||(H^-1)_{k,:}||`.
    ///
    /// This is the **full** width, not half of it: for a cube of side `L` it
    /// returns `L`. Callers that need the minimum-image validity bound halve it
    /// themselves (see [`crate::core::VerletSkin`]). For a tilted
    /// cell `|a_k|` *overestimates* this width, so any cutoff or image-range
    /// calculation must size from here and never from
    /// [`lengths`](Self::lengths) — otherwise pairs are silently missed.
    pub fn nearest_plane_distance(&self) -> F3 {
        let v = self.volume();
        let a1 = to_vec3(self.lattice(0).view());
        let a2 = to_vec3(self.lattice(1).view());
        let a3 = to_vec3(self.lattice(2).view());

        array![
            v / norm(cross(a2, a3)),
            v / norm(cross(a3, a1)),
            v / norm(cross(a1, a2))
        ]
    }

    pub fn kind(&self) -> &BoxKind {
        &self.kind
    }

    /// Lattice vector by index (0,1,2) — columns of H
    pub fn lattice(&self, index: usize) -> F3 {
        assert!(index < 3, "lattice index must be 0..2");
        self.h.column(index).to_owned()
    }

    /// Fractional coordinates **without** the wrap into `[0, 1)`.
    ///
    /// Folding every axis back into the primitive cell unconditionally is
    /// right for a fully periodic box but destroys the information a caller needs on
    /// a **non-periodic** axis: a point above the box must stay above it, so
    /// that a cell-list assignment can clamp it to the edge cell instead of
    /// wrapping it to the opposite face. Callers that dispatch on
    /// [`pbc`](Self::pbc) per axis — see `CellGrid` — start from this raw value
    /// and apply wrap or clamp themselves.
    ///
    /// Folding this per component with `f - f.floor()` gives the fractional
    /// coordinates of a point inside the cell.
    #[inline(always)]
    pub(crate) fn make_fractional_raw_arr3(&self, r: [F; 3]) -> [F; 3] {
        let d = std::array::from_fn::<_, 3, _>(|i| r[i] - self.origin[i]);
        std::array::from_fn(|i| (0..3).map(|j| self.inv[[i, j]] * d[j]).sum())
    }

    /// Convert one fractional point to Cartesian coordinates. The batch form
    /// is [`to_cart`](Self::to_cart).
    pub(crate) fn make_cartesian(&self, frac: F3View<'_>) -> F3 {
        &self.origin + &self.h.dot(&frac)
    }

    /// The box's minimum-image convention as a detached, `Copy` kernel.
    ///
    /// The one minimum-image entry point: `bx.mic().apply(b - a)` is the
    /// minimum-image displacement from `a` to `b`. `Mic` carries everything it
    /// needs on the stack, so a hot loop captures it once — even while it
    /// holds the owning structure mutably — and applies it per pair with no
    /// allocation. The batch row-wise form is [`delta`](Self::delta).
    #[inline]
    pub fn mic(&self) -> Mic {
        self.mic
    }

    /// Convert Cartesian points to fractional coordinates (N×3)
    pub fn to_frac(&self, xyz: Fnx3View<'_>) -> Fnx3 {
        let n = xyz.nrows();
        let mut result = Fnx3::zeros((n, 3));
        for i in 0..n {
            let dr = &xyz.row(i) - &self.origin.view();
            result.row_mut(i).assign(&self.inv.dot(&dr));
        }
        result
    }

    /// Convert fractional coordinates to Cartesian points (N×3)
    pub fn to_cart(&self, frac: Fnx3View<'_>) -> Fnx3 {
        let n = frac.nrows();
        let mut result = Fnx3::zeros((n, 3));
        for i in 0..n {
            let cart = &self.origin + &self.h.dot(&frac.row(i));
            result.row_mut(i).assign(&cart);
        }
        result
    }

    /// Check if points lie within [0,1) in fractional space.
    pub fn contains(&self, xyz: Fnx3View<'_>) -> Array1<bool> {
        let n = xyz.nrows();
        let mut mask = Vec::with_capacity(n);
        for i in 0..n {
            let dr = &xyz.row(i) - &self.origin.view();
            let frac = self.inv.dot(&dr);
            let inside = (0..3).all(|d| frac[d] >= 0.0 && frac[d] < 1.0);
            mask.push(inside);
        }
        Array1::from_vec(mask)
    }

    /// Row-wise displacement vectors `xyzu2 - xyzu1` (N×3), minimum-imaged
    /// when `minimum_image` is set.
    ///
    /// # Panics
    /// If the two arrays have different row counts.
    pub fn delta(&self, xyzu1: Fnx3View<'_>, xyzu2: Fnx3View<'_>, minimum_image: bool) -> Fnx3 {
        assert_eq!(xyzu1.nrows(), xyzu2.nrows());
        let mut out = &xyzu2 - &xyzu1;
        if minimum_image {
            for mut row in out.rows_mut() {
                let dr = self.mic.apply([row[0], row[1], row[2]]);
                row[0] = dr[0];
                row[1] = dr[1];
                row[2] = dr[2];
            }
        }
        out
    }

    /// Row-wise minimum-image distances between equally sized point arrays.
    pub fn distances(&self, points1: Fnx3View<'_>, points2: Fnx3View<'_>) -> Array1<F> {
        assert_eq!(points1.raw_dim(), points2.raw_dim());
        let values = points1
            .rows()
            .into_iter()
            .zip(points2.rows())
            .map(|(a, b)| {
                let dr = self.mic.apply([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt()
            })
            .collect();
        Array1::from_vec(values)
    }

    /// All pairwise minimum-image displacement vectors (`points2 - points1`).
    pub fn pairwise_delta(&self, points1: Fnx3View<'_>, points2: Fnx3View<'_>) -> Array3<F> {
        let mut out = Array3::zeros((points1.nrows(), points2.nrows(), 3));
        for (i, a) in points1.rows().into_iter().enumerate() {
            for (j, b) in points2.rows().into_iter().enumerate() {
                let dr = self.mic.apply([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                for d in 0..3 {
                    out[[i, j, d]] = dr[d];
                }
            }
        }
        out
    }

    /// All pairwise minimum-image distances.
    pub fn pairwise_distances(&self, points1: Fnx3View<'_>, points2: Fnx3View<'_>) -> Array2<F> {
        let mut out = Array2::zeros((points1.nrows(), points2.nrows()));
        for (i, a) in points1.rows().into_iter().enumerate() {
            for (j, b) in points2.rows().into_iter().enumerate() {
                let dr = self.mic.apply([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                out[[i, j]] = (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2]).sqrt();
            }
        }
        out
    }

    /// Return a box with its cell matrix right-multiplied by a transform.
    pub fn transformed(&self, transformation: &Fnx3) -> Result<Self, BoxError> {
        Self::from_parts(
            self.cell_defined.then(|| self.h.dot(transformation)),
            self.origin.clone(),
            self.pbc,
        )
    }

    /// The accumulated shift as exact integers of the schema's image type
    /// ([`I`], i32). Every component is a whole number by construction — it
    /// only ever gains `floor` or `round` results. A single wrap shifts a point
    /// by a handful of cells, far inside the i32 range.
    #[inline]
    fn as_images(n: [F; 3]) -> [I; 3] {
        [n[0] as I, n[1] as I, n[2] as I]
    }

    /// Shift `r` by the integer lattice vector `n`, always from the original
    /// point so repeated refinement never accumulates rounding.
    #[inline]
    fn shifted_by_images(&self, r: [F; 3], n: [F; 3]) -> [F; 3] {
        let mut out = r;
        for (d, nd) in n.iter().enumerate() {
            if *nd != 0.0 {
                for (k, o) in out.iter_mut().enumerate() {
                    *o -= nd * self.h[[k, d]];
                }
            }
        }
        out
    }

    /// Wrap one point and report the integer lattice shift applied — the
    /// single-point kernel of [`wrap`](Self::wrap), no allocation.
    ///
    /// Returns `(wrapped, m)` with `wrapped = r - H·m`. It is the *same*
    /// kernel `wrap` runs per row, so the one-point and array forms agree bit
    /// for bit by construction. Use it on hot single-point paths (region
    /// membership, per-probe wrapping) that would otherwise pay a `(1, 3)`
    /// array round-trip.
    ///
    /// All the guarantees of `wrap` hold: a point already inside the cell is
    /// returned untouched with `m = [0, 0, 0]`, and `m` is the shift actually
    /// applied (including a snap off the far face).
    #[inline]
    pub(crate) fn wrap_row_shift(&self, r: [F; 3]) -> ([F; 3], [I; 3]) {
        let f = self.make_fractional_raw_arr3(r);
        let mut n = [0.0; 3];
        let mut crossed = false;
        for d in 0..3 {
            if self.pbc[d] {
                n[d] = f[d].floor();
                if n[d] != 0.0 {
                    crossed = true;
                }
            }
        }
        // A point already in the cell is returned untouched, bit for bit: no
        // arithmetic is performed on it at all.
        if !crossed {
            return (r, [0; 3]);
        }

        let mut out = self.shifted_by_images(r, n);
        // Subtracting whole lattice vectors is exact in real arithmetic but not
        // in floating point: a coordinate a hair below the origin gains one
        // lattice vector and can land *exactly* on the far face (`f == 1.0`),
        // outside the `[0, 1)` postcondition. In a triclinic cell it is worse —
        // shifting along one axis perturbs all three fractional components, so
        // a single correction pass is not enough.
        //
        // Refine the integer shift instead of the point: each pass re-derives
        // the shift from `out` but rebuilds `out` from the original `r`, so the
        // answer is one rounding away from exact however many passes it takes.
        // MD wraps every step for 10^8 steps, so "rare" here means "often".
        for _ in 0..Self::WRAP_REFINE_PASSES {
            let g = self.make_fractional_raw_arr3(out);
            let mut adjusted = false;
            for d in 0..3 {
                if !self.pbc[d] {
                    continue;
                }
                let m = g[d].floor();
                if m != 0.0 {
                    n[d] += m;
                    adjusted = true;
                }
            }
            if !adjusted {
                return (out, Self::as_images(n));
            }
            out = self.shifted_by_images(r, n);
        }

        // The refinement can fail to converge, and not because it is wrong: for
        // a point a hair below the origin the true wrapped value (`L - 1e-18`)
        // is simply **not representable** — `ulp(10 A) = 1.8e-15`. Both
        // candidates, the original point (`f < 0`) and the shifted one
        // (`f == 1.0`), sit outside the cell, so no integer shift satisfies the
        // postcondition and the two alternate forever.
        //
        // Decide it instead of looping: snap to the origin face. The atom moves
        // by less than one ulp of a cell edge — the error already inherent in
        // the shift — and `f == 0` is inside where `f == 1.0` is not.
        let (snapped, absorbed) = self.snap_to_origin_face(out);
        for d in 0..3 {
            n[d] += absorbed[d];
        }
        (snapped, Self::as_images(n))
    }

    /// Wrap one point into the cell on the periodic axes, discarding the
    /// shift — [`wrap_row_shift`](Self::wrap_row_shift)`.0`, bit for bit what
    /// [`wrap`](Self::wrap) returns for the same row.
    #[inline]
    pub(crate) fn wrap_row(&self, r: [F; 3]) -> [F; 3] {
        self.wrap_row_shift(r).0
    }

    /// Last-resort clamp for a point the integer refinement could not land
    /// inside: force every escaped periodic axis onto the origin face.
    ///
    /// Returns the snapped point **and the integer lattice shift the snap
    /// absorbed**, because the snap is not always a sub-ulp nudge. The case it
    /// exists for — a point a hair below the origin, shifted up to land exactly
    /// on the far face (`f == 1.0`) — is pinned back to the origin face, which
    /// moves it by a whole lattice vector. A caller accumulating image flags
    /// that missed that would have its flags disagree with its positions by a
    /// full cell, and nothing would say so until an unwrapped trajectory came
    /// back wrong. The adjustment is `round(f)`, not `floor(f)`: `f == 1.0` is
    /// one cell out, while `f == -5e-17` is the origin seen from below and is
    /// zero cells out.
    #[inline]
    fn snap_to_origin_face(&self, out: [F; 3]) -> ([F; 3], [F; 3]) {
        let g = self.make_fractional_raw_arr3(out);
        let escaped = (0..3).any(|d| self.pbc[d] && !(0.0..1.0).contains(&g[d]));
        if !escaped {
            return (out, [0.0; 3]);
        }
        let mut absorbed = [0.0; 3];
        for d in 0..3 {
            if self.pbc[d] && !(0.0..1.0).contains(&g[d]) {
                absorbed[d] = g[d].round();
            }
        }
        let mut frac = g;
        for (d, fd) in frac.iter_mut().enumerate() {
            if self.pbc[d] {
                *fd -= fd.floor();
                if !(0.0..1.0).contains(fd) {
                    *fd = 0.0;
                }
            }
        }
        let cart = self.make_cartesian(ndarray::arr1(&frac).view());
        ([cart[0], cart[1], cart[2]], absorbed)
    }

    /// Wrap Cartesian points into the unit cell on the periodic axes **and**
    /// report the integer lattice shift applied to each.
    ///
    /// Returns `(wrapped, m)` with `wrapped = r - H·m` row by row. A
    /// non-periodic axis is left alone (its shift is 0).
    ///
    /// Whole lattice vectors are subtracted from the Cartesian coordinate, so a
    /// point **already inside the cell is returned bit for bit** with
    /// `m = [0, 0, 0]`, and only the atoms that actually crossed a face are
    /// touched. This is deliberately *not* a `to_frac` → fold → `to_cart`
    /// round-trip: that form rewrites every coordinate of every atom through
    /// two matrix products, so it perturbs points that never left the cell by
    /// a rounding error each call — and MD wraps after every integration step.
    ///
    /// `m` serves both uses of image flags:
    ///
    /// - **MD**: a caller maintaining canonical state accumulates `n += m`
    ///   each step and reconstructs the continuous trajectory as
    ///   `r^u = wrapped + H·n` ([`unwrap`](Self::unwrap)). `m` is the shift
    ///   **actually applied**, not `floor(H⁻¹r)` recomputed: the kernel refines
    ///   the shift and may finish by snapping a point off an
    ///   exactly-representable far face, and flags derived independently would
    ///   silently disagree with the positions they belong to.
    /// - **Import**: for a configuration that arrived as continuous coordinates
    ///   (a file, an analysis that never wrapped), `m` recovers its image
    ///   flags. A wrapped configuration correctly yields all zeros.
    ///
    /// `m` is the schema's image integer type ([`I`], i32) — the same type as
    /// the `ix`/`iy`/`iz` columns it is banked into.
    pub fn wrap(&self, xyz: Fnx3View<'_>) -> (Fnx3, Array2<I>) {
        let n = xyz.nrows();
        let mut out = xyz.to_owned();
        let mut shifts = Array2::<I>::zeros((n, 3));
        for i in 0..n {
            let (w, m) = self.wrap_row_shift([xyz[[i, 0]], xyz[[i, 1]], xyz[[i, 2]]]);
            for d in 0..3 {
                out[[i, d]] = w[d];
                shifts[[i, d]] = m[d];
            }
        }
        (out, shifts)
    }

    /// Reconstruct unwrapped coordinates from wrapped points and image flags.
    ///
    /// `images` is the schema's image integer type ([`I`], i32), so the
    /// `ix`/`iy`/`iz` columns of a frame pass straight in.
    pub fn unwrap(&self, xyz: Fnx3View<'_>, images: ArrayView2<'_, I>) -> Fnx3 {
        assert_eq!(xyz.raw_dim(), images.raw_dim());
        assert_eq!(xyz.ncols(), 3);
        let mut result = xyz.to_owned();
        for i in 0..xyz.nrows() {
            let image = array![
                images[[i, 0]] as F,
                images[[i, 1]] as F,
                images[[i, 2]] as F,
            ];
            let shift = self.h.dot(&image);
            for d in 0..3 {
                result[[i, d]] += shift[d];
            }
        }
        result
    }

    pub fn corners(&self) -> Fnx3 {
        self.to_cart(
            array![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
            .view(),
        )
    }

    /// Axis-aligned bounding box of the cell geometry.
    ///
    /// Layout: rows = x/y/z, col 0 = min, col 1 = max — the AABB of the eight
    /// corners. For triclinic cells this is larger than the true cell volume;
    /// use [`contains`](Self::contains) for membership. Geometric region types that
    /// describe the same volume live in the regions (`crate::core::Region`)
    /// (`Cuboid` / `Parallelepiped`) — not on this type.
    pub fn bounds(&self) -> Fnx3 {
        let corners = self.corners();
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            let mut lo = corners[[0, d]];
            let mut hi = lo;
            for i in 1..corners.nrows() {
                lo = lo.min(corners[[i, d]]);
                hi = hi.max(corners[[i, d]]);
            }
            b[[d, 0]] = lo;
            b[[d, 1]] = hi;
        }
        b
    }
}

/// Prepared periodic-distance kernel, shared by every cell geometry.
///
/// Construction reduces the periodic lattice and caches its dual basis and
/// neighboring image translations. `apply` is always the same matrix/vector
/// calculation plus a bounded minimum over at most 27 images; it never reduces
/// a basis or searches an integer lattice at pair-evaluation time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mic {
    // Basis vectors are stored as rows for contiguous dot products.
    basis: [[F; 3]; 3],
    dual: [[F; 3]; 3],
    translations: [[F; 3]; 27],
    n_translations: usize,
}

impl Mic {
    /// Prepare distances for a cell matrix (lattice vectors as columns).
    /// Only periodic vectors enter the lattice; nonperiodic components remain
    /// unrestricted. Orthogonal cells use this identical preparation and kernel.
    ///
    /// # Errors
    /// Refuses singular/nonfinite matrices and a lattice whose reduction cannot
    /// be certified within the construction iteration limit.
    pub fn new(h: [[F; 3]; 3], pbc: Pbc3) -> Result<Self, BoxError> {
        inv3(&h).ok_or(BoxError::SingularCell)?;
        let rank = pbc.iter().filter(|&&periodic| periodic).count();
        let mut basis = [[0.0; 3]; 3];
        for (i, axis) in (0..3).filter(|&axis| pbc[axis]).enumerate() {
            basis[i] = [h[0][axis], h[1][axis], h[2][axis]];
        }
        Self::reduce_basis(&mut basis, rank)?;
        let mut unit = [[0.0; 3]; 3];
        let mut lengths = [1.0; 3];
        for i in 0..rank {
            lengths[i] = norm(basis[i]);
            unit[i] = basis[i].map(|value| value / lengths[i]);
        }
        // Normalize before forming the Gram matrix so unequal cell lengths do
        // not square its condition number. Unused dimensions form an identity
        // block. The resulting dual is the left inverse of the periodic basis,
        // including oblique partial-PBC cells (not rows of the full H inverse).
        let gram = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < rank && j < rank {
                    dot(unit[i], unit[j])
                } else {
                    F::from(i == j)
                }
            })
        });
        let gram_inv = inv3(&gram).ok_or(BoxError::LatticeReduction)?;
        let dual = std::array::from_fn(|i| {
            std::array::from_fn(|axis| {
                (0..rank).map(|j| gram_inv[i][j] * unit[j][axis]).sum::<F>() / lengths[i]
            })
        });
        let mut result = Self {
            basis,
            dual,
            ..Self::free()
        };
        // A Minkowski-reduced lattice in dimension <=3 needs only translations
        // with coefficients -1,0,1. Keep zero first for deterministic ties.
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    let coefficients = [i, j, k];
                    if coefficients == [0; 3] || (rank..3).any(|axis| coefficients[axis] != 0) {
                        continue;
                    }
                    result.translations[result.n_translations] = std::array::from_fn(|axis| {
                        (0..rank)
                            .map(|a| coefficients[a] as F * basis[a][axis])
                            .sum()
                    });
                    result.n_translations += 1;
                }
            }
        }
        Ok(result)
    }

    /// The zero-dimensional periodic lattice. It uses the same matrix kernel,
    /// with zero projection and a single zero translation.
    pub const fn free() -> Self {
        Self {
            basis: [[0.; 3]; 3],
            dual: [[0.; 3]; 3],
            translations: [[0.; 3]; 27],
            n_translations: 1,
        }
    }

    /// Prepare a diagonal cell from per-axis lengths. A non-positive length
    /// disables that periodic direction, as for callers carrying frame lengths.
    pub fn ortho(lengths: [F; 3]) -> Self {
        let pbc = lengths.map(|length| length > 0.0);
        let h = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i == j {
                    if pbc[i] { lengths[i] } else { 1.0 }
                } else {
                    0.0
                }
            })
        });
        Self::new(h, pbc).expect("finite, nonsingular cell lengths")
    }

    /// Minkowski's low-dimensional reduction inequalities are sufficient in
    /// rank <=3: after sorting, each b_j must be no longer than any
    /// b_j + sum_{i<j} c_i b_i, c_i in {-1,0,1}. Replacing b_j by a shorter
    /// such vector is unimodular (its b_j coefficient is one). Integer pair
    /// reduction accelerates large shears; the full inequalities certify exit.
    fn reduce_basis(basis: &mut [[F; 3]; 3], rank: usize) -> Result<(), BoxError> {
        let norm2 = |v: [F; 3]| dot(v, v);
        let shorter = |a: [F; 3], b: [F; 3]| norm2(a) < norm2(b) * (1.0 - 64.0 * F::EPSILON);
        for _ in 0..10_000 {
            basis[..rank].sort_by(|a, b| norm2(*a).total_cmp(&norm2(*b)));
            let mut changed = false;
            for j in 1..rank {
                for i in 0..j {
                    let multiple = (dot(basis[j], basis[i]) / norm2(basis[i])).round();
                    let candidate =
                        std::array::from_fn(|axis| basis[j][axis] - multiple * basis[i][axis]);
                    if shorter(candidate, basis[j]) {
                        basis[j] = candidate;
                        changed = true;
                    }
                }
            }
            if changed {
                continue;
            }
            'certificate: for j in 1..rank {
                for c0 in -1..=1 {
                    for c1 in -1..=1 {
                        if j == 1 && c1 != 0 {
                            continue;
                        }
                        let coefficients = [c0, c1];
                        let candidate = std::array::from_fn(|axis| {
                            basis[j][axis]
                                + (0..j)
                                    .map(|i| coefficients[i] as F * basis[i][axis])
                                    .sum::<F>()
                        });
                        if shorter(candidate, basis[j]) {
                            basis[j] = candidate;
                            changed = true;
                            break 'certificate;
                        }
                    }
                }
            }
            if !changed {
                return Ok(());
            }
        }
        Err(BoxError::LatticeReduction)
    }

    /// Nearest periodic displacement. Project to the prepared periodic basis,
    /// fold coefficients, evaluate the fixed candidate translations, and select
    /// their minimum squared norm. All cell shapes and PBC ranks share this path.
    #[inline]
    pub fn apply(&self, d: [F; 3]) -> [F; 3] {
        let coefficients = self.dual.map(|row| dot(row, d));
        let integers = coefficients.map(F::round);
        let reduced = std::array::from_fn::<_, 3, _>(|axis| {
            (0..3)
                .map(|i| (coefficients[i] - integers[i]) * self.basis[i][axis])
                .sum::<F>()
        });
        let base = std::array::from_fn::<_, 3, _>(|axis| {
            d[axis] - (0..3).map(|i| integers[i] * self.basis[i][axis]).sum::<F>()
        });
        let mut best = 0;
        let mut best_sq = dot(reduced, reduced);
        for i in 1..self.n_translations {
            let candidate = std::array::from_fn(|axis| reduced[axis] - self.translations[i][axis]);
            let sq = dot(candidate, candidate);
            if sq < best_sq {
                best = i;
                best_sq = sq;
            }
        }
        std::array::from_fn(|axis| base[axis] - self.translations[best][axis])
    }
}

fn detect_box_kind(h: &Fnx3) -> BoxKind {
    let eps: F = 1e-12;
    let is_ortho = h[[0, 1]].abs() < eps
        && h[[0, 2]].abs() < eps
        && h[[1, 0]].abs() < eps
        && h[[1, 2]].abs() < eps
        && h[[2, 0]].abs() < eps
        && h[[2, 1]].abs() < eps;
    if is_ortho {
        let len = array![h[[0, 0]], h[[1, 1]], h[[2, 2]]];
        let inv_len = array![1.0 / len[0], 1.0 / len[1], 1.0 / len[2]];
        BoxKind::Ortho { len, inv_len }
    } else {
        BoxKind::Triclinic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::I;

    fn assert_close(a: F, b: F) {
        assert!((a - b).abs() < 1e-6 as F, "{} != {}", a, b);
    }

    #[test]
    fn cell_defined_distinct_from_pbc() {
        // A defined box (any pbc) reports cell_defined = true (the default).
        let defined = SimBox::new(
            Cell::Lengths([2.0, 2.0, 2.0]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .unwrap();
        assert!(defined.is_cell_defined());
        assert!(defined.is_free()); // non-periodic => free, but cell IS defined
        assert_close(defined.volume(), 8.0); // real cell volume preserved (RDF)

        // A no-cell box carries the identity cell (so geometry is a no-op) but
        // is marked cell_defined = false.
        let nocell = SimBox::new(Cell::Undefined, [0.0, 0.0, 0.0], [false, false, false]).unwrap();
        assert!(!nocell.is_cell_defined());
        // No cell, no volume: the placeholder identity's det(H) = 1 is not one.
        assert_eq!(nocell.volume(), 0.0);
        assert_eq!(nocell.style(), "free");
        // geometry no-ops on the identity cell:
        let pts = array![[1.0, 2.0, 3.0]];
        assert_eq!(nocell.wrap(pts.view()).0, pts);
    }

    /// An undefined cell carries the identity matrix in place of a cell: it is
    /// never inverted, and the origin is kept.
    #[test]
    fn an_undefined_cell_carries_the_identity() {
        let nocell = SimBox::new(Cell::Undefined, [1.0, 2.0, 3.0], [false; 3]).unwrap();
        assert!(!nocell.is_cell_defined());
        assert_eq!(
            nocell.matrix(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(nocell.inv_view(), ndarray::Array2::<F>::eye(3));
        assert_eq!(nocell.origin(), [1.0, 2.0, 3.0]);
        assert_eq!(nocell.volume(), 0.0);
        // A *defined* singular cell is still refused.
        assert!(matches!(
            SimBox::new(
                Cell::Matrix([[0.0; 3]; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true]
            ),
            Err(BoxError::SingularCell)
        ));
    }

    #[test]
    fn roundtrip_frac_cart() {
        let bx = SimBox::new(
            Cell::Lengths([2.0, 3.0, 4.0]),
            [0.5, -1.0, 2.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let pts = array![[0.5, -1.0, 2.0], [2.5, 2.0, 6.0]];
        let frac = bx.to_frac(pts.view());
        let cart = bx.to_cart(frac.view());
        assert!((&pts - &cart).iter().all(|v| v.abs() < 1e-5));
    }

    #[test]
    fn wrap_row_matches_wrap_on_every_axis_kind() {
        let bx = SimBox::bounding(
            array![[0.0, 0.0, 0.0], [4.0, 5.0, 6.0]].view(),
            [0.0; 3],
            [true, false, true],
        )
        .unwrap();
        for r in [[4.5, 5.5, -0.5], [-1.0, 2.0, 13.0], [1.0, 1.0, 1.0]] {
            let one = bx.wrap_row(r);
            let many = bx.wrap(array![[r[0], r[1], r[2]]].view()).0;
            // Same arithmetic, so the kernel and the array form agree exactly —
            // not merely to a tolerance.
            for d in 0..3 {
                assert_eq!(one[d], many[[0, d]], "{r:?} axis {d}");
            }
        }
        // The non-periodic axis passes through untouched.
        assert_eq!(bx.wrap_row([4.5, 5.5, -0.5])[1], 5.5);
    }

    /// The property MD depends on: wrapping runs after every integration step,
    /// so an atom that never left the cell must come back with its coordinate
    /// unchanged to the last bit. A `to_frac` → fold → `to_cart` round-trip
    /// fails this.
    #[test]
    fn wrap_is_bit_exact_for_points_already_inside() {
        for bx in [
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::Lengths([7.0, 11.0, 13.0]),
                [-3.0, 0.5, 2.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [70.0, 80.0, 65.0],
                },
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ] {
            // Fractional points strictly inside the cell, mapped to Cartesian.
            let frac = array![
                [0.1, 0.2, 0.3],
                [0.5, 0.5, 0.5],
                [0.999_999, 0.000_001, 0.7],
                [0.123_456_789, 0.987_654_321, 0.314_159_265],
            ];
            let pts = bx.to_cart(frac.view());
            assert_eq!(bx.wrap(pts.view()).0, pts);
        }
    }

    /// A coordinate a hair below the origin must not come back sitting exactly
    /// on the far face. The naive `r - floor(f)*L` rounds `-1e-18 + L` up to
    /// `L`, i.e. `f == 1.0` — outside the cell, and on the next step it leaves
    /// through the opposite face. `wrap` owes a strict `f in [0, 1)`.
    #[test]
    fn wrap_postcondition_holds_just_below_the_origin() {
        for bx in [
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::Lengths([7.0, 11.0, 13.0]),
                [-3.0, 0.5, 2.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [70.0, 80.0, 65.0],
                },
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ]
        .into_iter()
        .enumerate()
        {
            let (bi, bx) = (bx.0, bx.1);
            let origin = bx.origin().to_owned();
            for eps in [1e-18, 1e-16, 1e-13, F::MIN_POSITIVE] {
                let below = array![[origin[0] - eps, origin[1] - eps, origin[2] - eps]];
                let wrapped = bx.wrap(below.view()).0;
                let frac = bx.to_frac(wrapped.view());
                for d in 0..3 {
                    let f = frac[[0, d]];
                    assert!(
                        (0.0..1.0).contains(&f),
                        "box {bi} eps {eps:e} axis {d}: f = {f:?} escaped [0, 1)"
                    );
                }
                // And it is still inside by the box's own predicate.
                assert!(
                    bx.contains(wrapped.view())[0],
                    "box {bi} eps {eps:e} not contained"
                );
            }
        }
    }

    /// The postcondition has to hold for *every* atom, not for the cases a
    /// hand-written test happens to pick. Sweep a deterministic pseudo-random
    /// cloud spanning many cells, including coordinates pushed right up against
    /// a face, over all three box kinds.
    #[test]
    fn wrap_postcondition_holds_over_a_random_cloud() {
        let boxes = [
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::Lengths([7.0, 11.0, 13.0]),
                [-3.0, 0.5, 2.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [70.0, 80.0, 65.0],
                },
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ];
        // xorshift64* — deterministic, no dev-dependency.
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as F / (1u64 << 53) as F
        };

        for (bi, bx) in boxes.iter().enumerate() {
            let lengths = bx.lengths();
            for k in 0..20_000 {
                // Span ~±50 cells, and every 4th point is pinned within a few
                // ulps of a face — the only regime where this can break.
                let mut r = [0.0; 3];
                for (d, rd) in r.iter_mut().enumerate() {
                    let cells = (next() - 0.5) * 100.0;
                    *rd = bx.origin()[d] + cells * lengths[d];
                    if k % 4 == 0 {
                        let face = (cells.round()) * lengths[d];
                        *rd = bx.origin()[d] + face;
                        for _ in 0..(k % 3) {
                            *rd = rd.next_down();
                        }
                    }
                }
                let pts = array![[r[0], r[1], r[2]]];
                let w = bx.wrap(pts.view()).0;
                let frac = bx.to_frac(w.view());
                for d in 0..3 {
                    let f = frac[[0, d]];
                    assert!(
                        (0.0..1.0).contains(&f),
                        "box {bi} point {k} {r:?} axis {d}: f = {f:?} escaped [0, 1)"
                    );
                }
                // The wrapped point differs from the input by a whole number of
                // cells, to within the rounding the shift itself carries.
                let back = bx.to_frac(pts.view());
                for d in 0..3 {
                    let shift = back[[0, d]] - frac[[0, d]];
                    assert!(
                        (shift - shift.round()).abs() < 1e-9,
                        "box {bi} point {k} axis {d}: shift {shift} is not integral"
                    );
                }
            }
        }
    }

    /// The shift `wrap` reports reconstructs the input — including through
    /// the refine-and-snap path, the one place a shift derived independently
    /// of the position would silently diverge.
    #[test]
    fn wrap_shift_reconstructs_the_input() {
        for bx in [
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::Lengths([7.0, 11.0, 13.0]),
                [-3.0, 0.5, 2.0],
                [true, true, true],
            )
            .unwrap(),
            SimBox::new(
                Cell::LengthsAngles {
                    lengths: [10.0, 11.0, 12.0],
                    angles: [70.0, 80.0, 65.0],
                },
                [0.0, 0.0, 0.0],
                [true, true, true],
            )
            .unwrap(),
        ] {
            let o = bx.origin().to_owned();
            // Row 0 is built in *fractional* space so it is genuinely inside
            // every box kind: a Cartesian point near the origin can fall
            // outside a skewed cell.
            let inside = bx.to_cart(array![[0.1_f64, 0.2, 0.3]].view());
            let pts = array![
                [inside[[0, 0]], inside[[0, 1]], inside[[0, 2]]],
                [o[0] + 43.7, o[1] - 19.2, o[2] + 101.5],
                [o[0] - 1e-18, o[1] - 1e-18, o[2] - 1e-18],
                [o[0] - 1e-13, o[1] + 55.0, o[2] - 77.7],
                [o[0] - F::MIN_POSITIVE, o[1], o[2]],
            ];

            let (w, m) = bx.wrap(pts.view());

            // The shift is the one that was applied: putting it back returns
            // the input. A shift derived independently as floor(H^-1 r) would
            // be off by a whole cell on the snapped row.
            let back = bx.unwrap(w.view(), m.view());
            for i in 0..pts.nrows() {
                for d in 0..3 {
                    assert!(
                        (back[[i, d]] - pts[[i, d]]).abs() < 1e-9,
                        "row {i} axis {d}: {} vs {}",
                        back[[i, d]],
                        pts[[i, d]]
                    );
                }
            }

            // In-cell points are untouched and carry no shift.
            assert_eq!(m.row(0).to_vec(), vec![0 as I; 3]);
        }
    }

    /// `images` reads the flag off the fractional coordinate with no fudge.
    ///
    /// Adding `1e-8` before flooring would be no small number on a
    /// *fractional* coordinate: it moves the boundary by `1e-8 · L`, which is
    /// 5e-7 Å in a 50 Å cell. Every atom in the half-open band just below a
    /// face would be reported one cell further along than it is, and the
    /// reconstruction `xyz + H·images` would put it a whole cell away from
    /// where it started. A tolerance that shifts a boundary is not a tolerance.
    #[test]
    fn images_has_no_boundary_fudge() {
        let l = 50.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap();

        // Fractional coordinates inside the band a 1e-8 epsilon would swallow:
        // just below 1.0, i.e. just inside the far face of cell 0.
        for eps_frac in [1e-9_f64, 5e-9, 9e-9] {
            let pts = array![[(1.0 - eps_frac) * l, 0.5 * l, 0.5 * l]];
            let im = bx.wrap(pts.view()).1;
            assert_eq!(
                im[[0, 0]],
                0,
                "a point {eps_frac} of a cell below the face is still in cell 0"
            );
            // And the round trip puts it back where it was, not a cell away.
            let back = bx.unwrap(bx.wrap(pts.view()).0.view(), im.view());
            assert!(
                (back[[0, 0]] - pts[[0, 0]]).abs() < 1e-9,
                "round trip moved it from {} to {}",
                pts[[0, 0]],
                back[[0, 0]]
            );
        }

        // The ordinary cases are unaffected.
        let pts = array![[-0.5 * l, 1.5 * l, 2.5 * l]];
        let im = bx.wrap(pts.view()).1;
        assert_eq!(im.row(0).to_vec(), vec![-1 as I, 1, 2]);
    }

    /// Wrapping is idempotent: a second pass moves nothing, bit for bit.
    #[test]
    fn wrap_is_idempotent() {
        let bx = SimBox::new(
            Cell::LengthsAngles {
                lengths: [10.0, 11.0, 12.0],
                angles: [70.0, 80.0, 65.0],
            },
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap();
        let pts = array![
            [43.7, -19.2, 101.5],
            [-0.000_1, 55.0, -77.7],
            [3.0, 4.0, 5.0],
        ];
        let once = bx.wrap(pts.view()).0;
        let twice = bx.wrap(once.view()).0;
        assert_eq!(once, twice);
    }

    #[test]
    fn wrap_into_cell() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[2.1, -0.1, 3.9], [-1.9, 4.2, 0.0]];
        let wrapped = bx.wrap(pts.view()).0;
        let frac = bx.to_frac(wrapped.view());
        for i in 0..wrapped.nrows() {
            let fx = frac[[i, 0]];
            let fy = frac[[i, 1]];
            let fz = frac[[i, 2]];
            assert!((0.0..1.0).contains(&fx));
            assert!((0.0..1.0).contains(&fy));
            assert!((0.0..1.0).contains(&fz));
        }
    }

    #[test]
    fn calc_distance_matches_components() {
        let bx = SimBox::new(Cell::Lengths([3.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let a = array![[0.1, 0.2, 0.3]];
        let b = array![[2.9, 0.2, 0.3]];
        let dr = bx.mic().apply([2.9 - 0.1, 0.0, 0.0]);
        let delta = bx.delta(a.view(), b.view(), true);
        assert_eq!(delta.row(0).to_vec(), dr.to_vec());
        let d2 = bx.distances(a.view(), b.view())[0].powi(2);
        assert!((d2 - (dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2])).abs() < 1e-6);
    }

    #[test]
    fn test_lengths_ortho() {
        let bx = SimBox::new(
            Cell::Lengths([2.0, 4.0, 5.0]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let lengths = bx.lengths();
        assert_close(lengths[0], 2.0);
        assert_close(lengths[1], 4.0);
        assert_close(lengths[2], 5.0);
    }

    #[test]
    fn test_tilts_values() {
        let h = [[2.0, 1.0, 2.0], [0.0, 4.0, 3.0], [0.0, 0.0, 5.0]];
        let bx =
            SimBox::new(Cell::Matrix(h), [0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let tilts = bx.tilts();
        assert_close(tilts[0], 1.0);
        assert_close(tilts[1], 2.0);
        assert_close(tilts[2], 3.0);
    }

    #[test]
    fn test_volume() {
        let bx = SimBox::new(
            Cell::Lengths([2.0, 3.0, 4.0]),
            [0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        assert_close(bx.volume(), 24.0);
    }

    #[test]
    fn test_wrap_single_and_multi() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[10.0, -5.0, -5.0], [0.0, 0.5, 0.0]];
        let wrapped = bx.wrap(pts.view()).0;
        assert_close(wrapped[[0, 0]], 0.0);
        assert_close(wrapped[[0, 1]], 1.0);
        assert_close(wrapped[[0, 2]], 1.0);
        assert_close(wrapped[[1, 0]], 0.0);
        assert_close(wrapped[[1, 1]], 0.5);
        assert_close(wrapped[[1, 2]], 0.0);
    }

    #[test]
    fn test_fractional_and_cartesian() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let p = array![[-1.0, -1.0, -1.0]];
        let frac = bx.to_frac(p.view());
        assert_close(frac[[0, 0]], -0.5);
        assert_close(frac[[0, 1]], -0.5);
        assert_close(frac[[0, 2]], -0.5);
        let cart = bx.to_cart(frac.view());
        assert_close(cart[[0, 0]], -1.0);
        assert_close(cart[[0, 1]], -1.0);
        assert_close(cart[[0, 2]], -1.0);
        // Folded into the cell, the point sits at its centre.
        let folded = bx.make_cartesian(array![0.5, 0.5, 0.5].view());
        assert_eq!(bx.wrap(p.view()).0.row(0).to_vec(), folded.to_vec());
    }

    #[test]
    fn test_to_frac_to_cart_roundtrip() {
        let bx = SimBox::new(
            Cell::Lengths([2.0, 3.0, 4.0]),
            [1.0, 2.0, 3.0],
            [true, true, true],
        )
        .expect("invalid box lengths");
        let pts = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
        let frac = bx.to_frac(pts.view());
        let cart = bx.to_cart(frac.view());
        for i in 0..pts.nrows() {
            for j in 0..3 {
                assert_close(pts[[i, j]], cart[[i, j]]);
            }
        }
    }

    #[test]
    fn mic_and_distance_agree() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let dr = bx.mic().apply([1.9 - 0.1, 0.0, 0.0]);
        assert_close(dr[0], -0.2);
        assert_close(dr[1], 0.0);
        assert_close(dr[2], 0.0);
        let d = bx.distances(
            array![[0.1, 0.0, 0.0]].view(),
            array![[1.9, 0.0, 0.0]].view(),
        );
        assert_close(d[0] * d[0], 0.04);
    }

    #[test]
    fn test_isin_point_non_pbc() {
        let bx = SimBox::new(
            Cell::Lengths([2.0; 3]),
            [0.0, 0.0, 0.0],
            [false, false, false],
        )
        .expect("invalid box length");
        let pts = array![[0.5, 0.5, 0.5], [-0.1, 0.5, 0.5], [2.1, 0.5, 0.5]];
        let mask = bx.contains(pts.view());
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn test_isin_mask() {
        let bx = SimBox::new(Cell::Lengths([2.0; 3]), [0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [2.1, 0.0, 0.0], [-0.1, 0.0, 0.0]];
        let mask = bx.contains(pts.view());
        assert!(mask[0]);
        assert!(!mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn bounding_pads_the_extent() {
        let pts = array![[1.0 as F, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let bx = SimBox::bounding(pts.view(), [1.0; 3], [false; 3]).unwrap();
        assert_eq!(bx.pbc(), [false, false, false]);
        // origin should be min - padding = [0.0, 1.0, 2.0]
        let o = bx.origin();
        assert!((o[0] - 0.0).abs() < 1e-5);
        assert!((o[1] - 1.0).abs() < 1e-5);
        assert!((o[2] - 2.0).abs() < 1e-5);
        // lengths should be (max-min) + 2*padding = [5.0, 5.0, 5.0]
        let l = bx.lengths();
        assert!((l[0] - 5.0).abs() < 1e-5);
        assert!((l[1] - 5.0).abs() < 1e-5);
        assert!((l[2] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn bounding_a_single_point_spans_twice_the_padding() {
        let pts = array![[1.0 as F, 2.0, 3.0]];
        let bx = SimBox::bounding(pts.view(), [2.0; 3], [false; 3]).unwrap();
        assert_eq!(bx.pbc(), [false, false, false]);
        // lengths = max(0 + 4, 2) = 4 on each axis
        let l = bx.lengths();
        assert!(l[0] >= 2.0);
        assert!(l[1] >= 2.0);
        assert!(l[2] >= 2.0);
    }

    #[test]
    fn bounding_refuses_what_has_no_extent() {
        use ndarray::Array2;
        // No points: no bounds.
        let empty = Array2::<F>::zeros((0, 3));
        assert!(matches!(
            SimBox::bounding(empty.view(), [1.0; 3], [false; 3]),
            Err(BoxError::EmptyPoints)
        ));
        // Padding may be zero, but a zero-extent axis is refused by name.
        let flat = array![[0.0 as F, 0.0, 1.0], [2.0, 3.0, 1.0]];
        assert!(matches!(
            SimBox::bounding(flat.view(), [0.0; 3], [false; 3]),
            Err(BoxError::ZeroExtent { axis: 2 })
        ));
        let tight = SimBox::bounding(flat.view(), [0.0, 0.0, 0.5], [false; 3]).unwrap();
        assert_eq!(tight.origin(), [0.0, 0.0, 0.5]);
        assert_eq!(tight.matrix()[2][2], 1.0);
        // Negative padding is refused.
        assert!(matches!(
            SimBox::bounding(flat.view(), [-1.0, 0.0, 0.0], [false; 3]),
            Err(BoxError::InvalidPadding { .. })
        ));
        // Ragged columns are refused.
        let (x, y, z) = ([0.0 as F, 1.0], [0.0 as F], [0.0 as F, 1.0]);
        assert!(matches!(
            SimBox::bounding(Points::Columns([&x, &y, &z]), [1.0; 3], [false; 3]),
            Err(BoxError::PointsShape { .. })
        ));
    }

    #[test]
    fn test_simbox_pbc_accessor() {
        let bx = SimBox::new(
            Cell::Lengths([1.0; 3]),
            [0.0 as F, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        assert_eq!(bx.pbc(), [true, false, true]);
    }

    #[test]
    fn bounding_columns_matches_rows_bitwise() {
        let pts = array![[1.0 as F, 2.0, 3.0], [4.0, -5.0, 6.0], [-2.5, 5.5, 0.25]];
        let xs = vec![1.0 as F, 4.0, -2.5];
        let ys = vec![2.0 as F, -5.0, 5.5];
        let zs = vec![3.0 as F, 6.0, 0.25];
        let a = SimBox::bounding(pts.view(), [1.5; 3], [false; 3]).unwrap();
        let b = SimBox::bounding(Points::Columns([&xs, &ys, &zs]), [1.5; 3], [false; 3]).unwrap();

        assert_eq!(a.origin(), b.origin(), "origin bitwise");
        assert_eq!(a.matrix(), b.matrix(), "H bitwise");
        assert_eq!(a.pbc(), b.pbc());
    }

    #[test]
    fn mic_matches_the_batch_delta_on_both_box_kinds() {
        // `Mic` is the single-pair kernel and `delta` the batch form; they
        // must not become two drifting definitions of the convention.
        let ortho = SimBox::new(
            Cell::Lengths([10.0, 11.0, 12.0]),
            [0.5, -1.0, 2.0],
            [true, false, true],
        )
        .unwrap();
        let tri = SimBox::new(
            Cell::LengthsAngles {
                lengths: [10.0, 11.0, 12.0],
                angles: [70.0, 80.0, 65.0],
            },
            [0.3, -0.2, 1.1],
            [true, true, false],
        )
        .unwrap();

        for bx in [&ortho, &tri] {
            let mic = bx.mic();
            for (a, b) in [
                ([0.0 as F, 0.0, 0.0], [9.0 as F, 1.0, -3.0]),
                ([1.5, -2.5, 3.5], [-7.25, 8.125, 0.0625]),
                ([4.0, 4.0, 4.0], [4.0, 4.0, 4.0]),
            ] {
                let want = bx.delta(ndarray::arr2(&[a]).view(), ndarray::arr2(&[b]).view(), true);
                let want = [want[[0, 0]], want[[0, 1]], want[[0, 2]]];
                let got = mic.apply([b[0] - a[0], b[1] - a[1], b[2] - a[2]]);
                for d in 0..3 {
                    assert_eq!(got[d], want[d], "component {d} for {a:?} -> {b:?}");
                }
            }
        }
    }

    #[test]
    fn a_box_with_no_periodic_axis_simplifies_to_free() {
        let free = SimBox::new(Cell::Lengths([10.0; 3]), [0.0, 0.0, 0.0], [false; 3]).unwrap();
        assert_eq!(free.mic(), Mic::free());
        let d = [123.0 as F, -456.0, 789.0];
        assert_eq!(Mic::free().apply(d), d);

        let periodic = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0, 0.0, 0.0],
            [false, true, false],
        )
        .unwrap();
        assert_ne!(periodic.mic(), Mic::free());
    }

    #[test]
    fn triclinic_mic_matches_independent_image_enumeration() {
        let bx = SimBox::new(
            Cell::LengthsAngles {
                lengths: [10., 11., 12.],
                angles: [70., 80., 65.],
            },
            [0.3, -0.2, 1.1],
            [true, true, false],
        )
        .unwrap();
        for d in [[9., 1., -3.], [-8.75, 10.625, -3.4375], [0., 0., 0.]] {
            let got = bx.mic().apply(d);
            let mut want = F::INFINITY;
            for i in -3..=3 {
                for j in -3..=3 {
                    let image: [F; 3] = std::array::from_fn(|axis| {
                        d[axis] - bx.h[[axis, 0]] * i as F - bx.h[[axis, 1]] * j as F
                    });
                    want = want.min(dot(image, image));
                }
            }
            assert!((dot(got, got) - want).abs() < 1e-11, "{d:?}: {got:?}");
        }
    }

    #[test]
    fn every_geometry_uses_the_prepared_matrix_kernel() {
        for (h, pbc, count) in [
            ([[2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], [true; 3], 27),
            ([[2., 1., 0.], [0., 3., 1.], [0., 0., 4.]], [true; 3], 27),
            (
                [[2., 1., 0.], [0., 3., 1.], [0., 0., 4.]],
                [true, false, true],
                9,
            ),
            ([[2., 1., 0.], [0., 3., 1.], [0., 0., 4.]], [false; 3], 1),
        ] {
            let kernel = Mic::new(h, pbc).unwrap();
            assert_eq!(kernel.n_translations, count);
        }
    }

    #[test]
    fn partial_pbc_uses_the_periodic_subspace_not_the_full_inverse() {
        let mic = Mic::new(
            [[1., 1000., 0.], [0., 1., 0.], [0., 0., 1.]],
            [true, false, false],
        )
        .unwrap();
        assert_eq!(mic.apply([1000., 0.6, 0.]), [0., 0.6, 0.]);
        // A large perpendicular component must not hide small differences in
        // periodic squared distance through floating-point cancellation.
        assert_eq!(mic.apply([0.7, 1e12, 0.]), [-0.30000000000000004, 1e12, 0.]);
    }

    #[test]
    fn from_matrix_reads_lattice_vectors_as_columns() {
        let h = [[2.0, 0.5, 0.25], [0.0, 3.0, 0.75], [0.0, 0.0, 4.0]];
        let bx = SimBox::new(Cell::Matrix(h), [1.0, 2.0, 3.0], [true, false, true]).unwrap();
        assert_eq!(bx.lattice(1).to_vec(), vec![0.5, 3.0, 0.0]);
        assert_eq!(bx.matrix(), h);
        assert_eq!(bx.origin().to_vec(), vec![1.0, 2.0, 3.0]);
        assert_eq!(bx.pbc(), [true, false, true]);
    }

    #[test]
    fn from_matrix_rejects_a_singular_cell() {
        let h = [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(matches!(
            SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3]),
            Err(BoxError::SingularCell)
        ));
    }

    #[test]
    fn matrix_columns_are_the_lattice_vectors() {
        let bx = SimBox::new(
            Cell::LengthsAngles {
                lengths: [10.0, 11.0, 12.0],
                angles: [70.0, 80.0, 65.0],
            },
            [0.0, 0.0, 0.0],
            [true; 3],
        )
        .unwrap();
        for (r, row) in bx.matrix().iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                assert_eq!(*v, bx.lattice(c)[r]);
            }
        }
    }

    #[test]
    fn approx_eq_bounds_every_entry_by_an_absolute_tolerance() {
        let h = [[10.0, 1.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let a = SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3]).unwrap();
        let mut hb = h;
        hb[0][1] += 1e-3;
        let b = SimBox::new(Cell::Matrix(hb), [0.0; 3], [true; 3]).unwrap();
        assert!(a.approx_eq(&a.clone(), 0.0));
        assert!(a.approx_eq(&b, 2e-3));
        assert!(!a.approx_eq(&b, 5e-4));
        let shifted = SimBox::new(Cell::Matrix(h), [0.0, 0.0, 1e-3], [true; 3]).unwrap();
        assert!(a.approx_eq(&shifted, 2e-3));
        assert!(!a.approx_eq(&shifted, 5e-4));
    }

    /// An undefined cell has no lattice to repeat along: a periodic flag on
    /// it is refused, whichever axis carries it.
    #[test]
    fn an_undefined_cell_refuses_a_periodic_axis() {
        for pbc in [[true, false, false], [false, true, false], [true; 3]] {
            let err = SimBox::new(Cell::Undefined, [0.0; 3], pbc).unwrap_err();
            assert!(matches!(err, BoxError::PeriodicWithoutCell { pbc: got } if got == pbc));
        }
    }

    #[test]
    fn approx_eq_compares_periodicity_and_cell_definition_exactly() {
        let h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let o = [0.0, 0.0, 0.0];
        let a = SimBox::new(Cell::Matrix(h), o, [true; 3]).unwrap();
        let other_pbc = SimBox::new(Cell::Matrix(h), o, [true, true, false]).unwrap();
        let no_cell = SimBox::new(Cell::Undefined, o, [false; 3]).unwrap();
        assert_eq!(no_cell.volume(), 0.0);
        assert_eq!(no_cell.style(), "free");
        assert!(!a.approx_eq(&other_pbc, 1.0));
        assert!(!a.approx_eq(&no_cell, 1.0));
    }

    #[test]
    #[should_panic(expected = "tolerance must be non-negative")]
    fn approx_eq_rejects_a_negative_tolerance() {
        let bx = SimBox::new(Cell::Lengths([1.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
        bx.approx_eq(&bx, -1.0);
    }

    #[test]
    fn wrap_row_shift_matches_wrap_bit_for_bit() {
        let bx = SimBox::new(
            Cell::LengthsAngles {
                lengths: [10.0, 11.0, 12.0],
                angles: [70.0, 80.0, 65.0],
            },
            [0.3, -0.2, 1.1],
            [true, true, false],
        )
        .unwrap();
        for r in [[25.0, -13.0, 40.0], [-1e-17, 0.5, 0.5], [1.0, 1.0, 1.0]] {
            let (w, m) = bx.wrap_row_shift(r);
            let (ws, ms) = bx.wrap(array![[r[0], r[1], r[2]]].view());
            for d in 0..3 {
                assert_eq!(w[d], ws[[0, d]], "{r:?} axis {d}");
                assert_eq!(m[d], ms[[0, d]], "{r:?} axis {d}");
            }
        }
    }
    /// Frozen offline ASE 3.29.0 / freud-analysis 3.6.1 results. Scientific
    /// software is deliberately absent from the default gate (AGENTS.md).
    #[test]
    fn prepared_mic_matches_external_static_goldens() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("simbox_oracle.json")).unwrap();
        for case in data["cases"].as_array().unwrap() {
            let h: [[F; 3]; 3] = serde_json::from_value(case["h"].clone()).unwrap();
            let pbc: [bool; 3] = serde_json::from_value(case["pbc"].clone()).unwrap();
            let vectors: Vec<[F; 3]> = serde_json::from_value(case["vectors"].clone()).unwrap();
            let expected: Vec<[F; 3]> = serde_json::from_value(case["mic"].clone()).unwrap();
            let tolerance = case["atol"].as_f64().unwrap_or(2e-9);
            let mic = Mic::new(h, pbc).unwrap();
            for (v, want) in vectors.into_iter().zip(expected) {
                let got = mic.apply(v);
                for axis in 0..3 {
                    assert!(
                        (got[axis] - want[axis]).abs() <= tolerance,
                        "{}: {v:?}, got {got:?}, expected {want:?}",
                        case["source"]
                    );
                }
            }
        }
    }

    #[test]
    fn matrix_wrapping_matches_external_static_goldens() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("simbox_oracle.json")).unwrap();
        for case in data["wrap_cases"].as_array().unwrap() {
            let h: [[F; 3]; 3] = serde_json::from_value(case["h"].clone()).unwrap();
            let pbc: [bool; 3] = serde_json::from_value(case["pbc"].clone()).unwrap();
            let origin: [F; 3] = serde_json::from_value(case["origin"].clone()).unwrap();
            let vectors: Vec<[F; 3]> = serde_json::from_value(case["vectors"].clone()).unwrap();
            let expected: Vec<[F; 3]> = serde_json::from_value(case["wrapped"].clone()).unwrap();
            let tolerance = case["atol"].as_f64().unwrap();
            let bx = SimBox::new(Cell::Matrix(h), origin, pbc).unwrap();
            let points = Array2::from_shape_fn((vectors.len(), 3), |(i, j)| vectors[i][j]);
            let (got, images) = bx.wrap(points.view());
            let restored = bx.unwrap(got.view(), images.view());
            for i in 0..vectors.len() {
                for j in 0..3 {
                    assert!(
                        (got[[i, j]] - expected[i][j]).abs() <= tolerance,
                        "{}",
                        case["source"]
                    );
                    assert!((restored[[i, j]] - vectors[i][j]).abs() <= 2e-9);
                }
            }
        }
    }

    #[test]
    fn periodic_basis_preparation_satisfies_minkowski_inequalities() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("simbox_oracle.json")).unwrap();
        for case in data["cases"].as_array().unwrap() {
            let h: [[F; 3]; 3] = serde_json::from_value(case["h"].clone()).unwrap();
            let pbc: [bool; 3] = serde_json::from_value(case["pbc"].clone()).unwrap();
            let rank = pbc.iter().filter(|&&p| p).count();
            let mic = Mic::new(h, pbc).unwrap();
            for j in 1..rank {
                let length = norm(mic.basis[j]);
                assert!(length >= norm(mic.basis[j - 1]) * (1. - 1e-13));
                for c0 in -1..=1 {
                    for c1 in -1..=1 {
                        if j == 1 && c1 != 0 {
                            continue;
                        }
                        let coefficients = [c0, c1];
                        let candidate = std::array::from_fn(|axis| {
                            mic.basis[j][axis]
                                + (0..j)
                                    .map(|i| coefficients[i] as F * mic.basis[i][axis])
                                    .sum::<F>()
                        });
                        assert!(norm(candidate) >= length * (1. - 1e-13));
                    }
                }
            }
        }
    }
}
