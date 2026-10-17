//! Python wrappers for the simulation box: `molrs.core.Cell` and
//! `molrs.core.Box`.
//!
//! [`PyCell`] mirrors the core [`Cell`] enum — the one argument that says what
//! the cell is — as a frozen class with one staticmethod per variant.
//! [`PyBox`] wraps [`SimBox`], which holds a resolved [`Cell`]:
//! `Box(cell, origin, pbc)` mirroring `SimBox::new(Cell, origin, pbc)` (the
//! cell a `Cell` or a 3×3 matrix), the factories `Box.cube` / `Box.ortho` /
//! `Box.from_lengths_tilts` / `Box.from_lengths_angles` for the common cells,
//! `Box.bounding(points, padding, pbc)` mirroring `SimBox::bounding`,
//! coordinate transformations (Cartesian <-> fractional), wrapping, and
//! displacements with an optional minimum-image convention.
//!
//! All length quantities are in the unit of the stored coordinates (Å in
//! molrs).

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::core::{BoxError, Cell, SimBox};
use molrs::op::{F, I};
use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Coerce a Python array to N×3 points. The bool is true when the input was shape `(3,)`.
fn delta_points_nx3(arg: &Bound<'_, PyAny>) -> PyResult<(Array2<F>, bool)> {
    if let Ok(arr2) = arg.extract::<ArrayArg2<'_, f64>>() {
        let view = arr2.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N, 3) or (3,)"));
        }
        return Ok((view.to_owned(), false));
    }
    if let Ok(arr1) = arg.extract::<ArrayArg1<'_, f64>>() {
        let view = arr1.as_array();
        if view.len() != 3 {
            return Err(PyValueError::new_err("expected shape (N, 3) or (3,)"));
        }
        return Ok((view.to_owned().insert_axis(Axis(0)), true));
    }
    Err(PyValueError::new_err("expected shape (N, 3) or (3,)"))
}

/// `Box.wrap`'s `(wrapped, shifts)`.
type WrapResult<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<I>>);

/// Reject a coordinate array that is not `(N, 3)`.
fn check_nx3(view: &ndarray::ArrayView2<'_, F>, label: &str) -> PyResult<()> {
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

/// What `Box(cell, …)` accepts as its cell: a [`PyCell`], or the 3×3 cell
/// matrix itself (lattice vectors as columns), which is `Cell.matrix(h)`.
#[derive(FromPyObject)]
enum CellArg {
    Cell(PyCell),
    Matrix([[F; 3]; 3]),
}

impl CellArg {
    fn into_cell(self) -> Cell {
        match self {
            CellArg::Cell(cell) => cell.inner,
            CellArg::Matrix(h) => Cell::Matrix(h),
        }
    }
}

/// The geometry of a simulation cell, exposed to Python as `molrs.core.Cell`.
///
/// Mirrors the core `Cell` enum: one staticmethod per variant, each building
/// a frozen value that :class:`Box` and :class:`Parallelepiped` take. The
/// geometry is checked where it is used (``Box(...)`` raises for a singular
/// matrix, a non-positive length or impossible angles).
///
/// Examples
/// --------
/// >>> Box(Cell.lengths([10.0, 10.0, 10.0])).volume()
/// 1000.0
/// >>> Box(Cell.undefined(), pbc=(False, False, False)).is_cell_defined
/// False
#[pyclass(module = "molrs.core", name = "Cell", frozen, from_py_object, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyCell {
    pub(crate) inner: Cell,
}

#[pymethods]
impl PyCell {
    /// The cell matrix ``h`` (3×3, Å), lattice vectors as **columns** —
    /// the layout :attr:`Box.matrix` returns.
    #[staticmethod]
    fn matrix(h: [[F; 3]; 3]) -> Self {
        Self {
            inner: Cell::Matrix(h),
        }
    }

    /// An orthorhombic cell with edge lengths ``[a, b, c]`` (Å), all positive.
    #[staticmethod]
    fn lengths(lengths: [F; 3]) -> Self {
        Self {
            inner: Cell::Lengths(lengths),
        }
    }

    /// A restricted-triclinic cell from edge lengths ``[a, b, c]`` (Å) and the
    /// angles ``[alpha, beta, gamma]`` between them, in degrees.
    #[staticmethod]
    fn lengths_angles(lengths: [F; 3], angles: [F; 3]) -> Self {
        Self {
            inner: Cell::LengthsAngles { lengths, angles },
        }
    }

    /// A restricted-triclinic cell from the diagonal sizes ``[lx, ly, lz]``
    /// (Å) and the tilts ``[xy, xz, yz]`` (Å) — the LAMMPS form.
    #[staticmethod]
    fn lengths_tilts(lengths: [F; 3], tilts: [F; 3]) -> Self {
        Self {
            inner: Cell::LengthsTilts { lengths, tilts },
        }
    }

    /// No geometric cell: a box of undefined, zero volume
    /// (``Box.is_cell_defined`` is ``False``). Distinct from periodicity.
    #[staticmethod]
    fn undefined() -> Self {
        Self {
            inner: Cell::Undefined,
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let cls = py.get_type::<Self>();
        let (name, args) = match slf.get().inner {
            Cell::Matrix(h) => ("matrix", PyTuple::new(py, [h])?),
            Cell::Lengths(lengths) => ("lengths", PyTuple::new(py, [lengths])?),
            Cell::LengthsAngles { lengths, angles } => {
                ("lengths_angles", PyTuple::new(py, [lengths, angles])?)
            }
            Cell::LengthsTilts { lengths, tilts } => {
                ("lengths_tilts", PyTuple::new(py, [lengths, tilts])?)
            }
            Cell::Undefined => ("undefined", PyTuple::empty(py)),
        };
        PyTuple::new(py, [cls.getattr(name)?, args.into_any()])
    }

    fn __repr__(&self) -> String {
        match self.inner {
            Cell::Matrix(h) => format!("Cell.matrix({h:?})"),
            Cell::Lengths(lengths) => format!("Cell.lengths({lengths:?})"),
            Cell::LengthsAngles { lengths, angles } => {
                format!("Cell.lengths_angles({lengths:?}, {angles:?})")
            }
            Cell::LengthsTilts { lengths, tilts } => {
                format!("Cell.lengths_tilts({lengths:?}, {tilts:?})")
            }
            Cell::Undefined => "Cell.undefined()".to_string(),
        }
    }
}

impl PyBox {
    /// The one place every constructor and factory resolves its cell.
    fn from_cell(cell: Cell, origin: [F; 3], pbc: [bool; 3]) -> PyResult<Self> {
        let inner = SimBox::new(cell, origin, pbc).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }
}

/// Simulation box with periodic boundary conditions, exposed to Python as
/// `molrs.core.Box`.
///
/// The box holds a :class:`Cell` (resolved to a 3x3 matrix **H** whose
/// columns are the lattice vectors), an origin point and per-axis PBC flags.
/// The factories build the common cells without spelling the :class:`Cell`.
///
/// # Python Examples
///
/// ```python
/// from molrs.core import Box, Cell
///
/// box = Box.cube(10.0)                          # cubic, periodic
/// box = Box.ortho([10.0, 20.0, 30.0])           # orthorhombic
/// print(box.volume())                           # 6000.0
/// slab = Box.cube(20.0, pbc=(True, True, False))
/// free = Box(Cell.undefined(), pbc=(False, False, False))
/// ```
#[pyclass(module = "molrs.core", name = "Box", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyBox {
    pub(crate) inner: SimBox,
}

#[pymethods]
impl PyBox {
    /// Create a simulation box from its cell, origin and periodicity.
    ///
    /// Parameters
    /// ----------
    /// cell : Cell or array_like, shape (3, 3)
    ///     The cell geometry (``Cell.matrix`` / ``lengths`` /
    ///     ``lengths_angles`` / ``lengths_tilts`` / ``undefined``), or the
    ///     cell matrix itself (Å, lattice vectors as columns).
    /// origin : array_like, shape (3,)
    ///     Origin of the box in Cartesian coordinates (Å).
    /// pbc : array_like of bool, shape (3,)
    ///     Periodic boundary flags for x, y, z.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A singular cell matrix, a non-positive or non-finite length, or
    ///     lengths/angles that describe no cell.
    #[new]
    #[pyo3(signature = (cell, origin = [0.0; 3], pbc = [true; 3]))]
    fn new(cell: CellArg, origin: [F; 3], pbc: [bool; 3]) -> PyResult<Self> {
        Self::from_cell(cell.into_cell(), origin, pbc)
    }

    /// A cubic box of edge ``a`` (Å): ``Box(Cell.lengths([a, a, a]), …)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A non-positive or non-finite ``a``.
    #[staticmethod]
    #[pyo3(signature = (a, origin = [0.0; 3], pbc = [true; 3]))]
    fn cube(a: F, origin: [F; 3], pbc: [bool; 3]) -> PyResult<Self> {
        Self::from_cell(Cell::Lengths([a; 3]), origin, pbc)
    }

    /// An orthorhombic box with edge lengths ``[a, b, c]`` (Å):
    /// ``Box(Cell.lengths(lengths), …)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A non-positive or non-finite length.
    #[staticmethod]
    #[pyo3(signature = (lengths, origin = [0.0; 3], pbc = [true; 3]))]
    fn ortho(lengths: [F; 3], origin: [F; 3], pbc: [bool; 3]) -> PyResult<Self> {
        Self::from_cell(Cell::Lengths(lengths), origin, pbc)
    }

    /// A restricted-triclinic box from the diagonal sizes ``[lx, ly, lz]``
    /// (Å) and the tilts ``[xy, xz, yz]`` (Å), the LAMMPS form:
    /// ``Box(Cell.lengths_tilts(lengths, tilts), …)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A non-positive or non-finite length.
    #[staticmethod]
    #[pyo3(signature = (lengths, tilts, origin = [0.0; 3], pbc = [true; 3]))]
    fn from_lengths_tilts(
        lengths: [F; 3],
        tilts: [F; 3],
        origin: [F; 3],
        pbc: [bool; 3],
    ) -> PyResult<Self> {
        Self::from_cell(Cell::LengthsTilts { lengths, tilts }, origin, pbc)
    }

    /// A restricted-triclinic box from edge lengths ``[a, b, c]`` (Å) and the
    /// angles ``[alpha, beta, gamma]`` between them, in degrees:
    /// ``Box(Cell.lengths_angles(lengths, angles), …)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A non-positive length, or angles that describe no cell.
    #[staticmethod]
    #[pyo3(signature = (lengths, angles, origin = [0.0; 3], pbc = [true; 3]))]
    fn from_lengths_angles(
        lengths: [F; 3],
        angles: [F; 3],
        origin: [F; 3],
        pbc: [bool; 3],
    ) -> PyResult<Self> {
        Self::from_cell(Cell::LengthsAngles { lengths, angles }, origin, pbc)
    }

    /// The orthorhombic box enclosing ``points``: their axis-aligned bounds
    /// widened by ``padding[d]`` on each side of axis ``d``, with periodicity
    /// ``pbc``.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float64
    ///     The points to enclose (Å).
    /// padding : array_like, shape (3,)
    ///     Margin on each side of each axis (Å), finite and ``>= 0``.
    /// pbc : array_like of bool, shape (3,)
    ///     Periodic boundary flags.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     No points, ``points`` not ``(N, 3)``, a negative or non-finite
    ///     padding, or an axis of zero extent (pass a positive padding).
    #[staticmethod]
    fn bounding(points: ArrayArg2<'_, f64>, padding: [F; 3], pbc: [bool; 3]) -> PyResult<Self> {
        let view = points.as_array();
        check_nx3(&view, "points")?;
        let inner = SimBox::bounding(view, padding, pbc).map_err(box_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether the cell is geometrically defined. ``False`` marks a box built
    /// from ``Cell.undefined()`` (undefined / zero volume), distinct from
    /// ``is_free`` (periodicity).
    #[getter]
    fn is_cell_defined(&self) -> bool {
        self.inner.is_cell_defined()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let this = slf.borrow();
        let cell = if this.inner.is_cell_defined() {
            Cell::Matrix(this.inner.matrix())
        } else {
            Cell::Undefined
        };
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                PyCell { inner: cell },
                this.inner.origin(),
                this.inner.pbc(),
            ),
        )
    }

    /// Whether ``other`` describes the same cell within an absolute tolerance.
    ///
    /// True iff every cell-matrix entry and origin component differ by at most
    /// ``tol`` (length units), the PBC flags are identical and both boxes agree
    /// on ``is_cell_defined``. ``tol=0.0`` is exact equality.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``tol`` is negative or NaN.
    fn approx_eq(&self, other: PyRef<'_, PyBox>, tol: F) -> PyResult<bool> {
        if tol.is_nan() || tol < 0.0 {
            return Err(PyValueError::new_err("tol must be non-negative"));
        }
        Ok(self.inner.approx_eq(&other.inner, tol))
    }

    /// Volume of the simulation box (Å³); ``0.0`` for an undefined cell.
    fn volume(&self) -> F {
        self.inner.volume()
    }

    /// ``True`` when the box is free (non-periodic on every axis).
    #[getter]
    fn is_free(&self) -> bool {
        self.inner.is_free()
    }

    /// Geometry style label: ``"free"``, ``"orthogonal"``, or ``"triclinic"``.
    #[getter]
    fn style(&self) -> &'static str {
        self.inner.style()
    }

    /// Return a lattice vector (a column of :attr:`matrix`) by index.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``index`` is not 0, 1, or 2.
    fn lattice<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if index >= 3 {
            return Err(PyValueError::new_err("index must be 0, 1, or 2"));
        }
        Ok(self.inner.lattice(index).into_pyarray(py))
    }

    /// Cell matrix **H** (3x3, Å), lattice vectors as columns — the layout
    /// :meth:`Cell.matrix` takes.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        ndarray::arr2(&self.inner.matrix()).into_pyarray(py)
    }

    /// Inverse cell matrix.
    #[getter]
    fn inverse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.inv_view().to_owned().into_pyarray(py)
    }

    /// Box origin in Cartesian coordinates, shape ``(3,)``.
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        ndarray::arr1(&self.inner.origin()).into_pyarray(py)
    }

    /// Periodic boundary condition flags ``[pbc_x, pbc_y, pbc_z]``.
    #[getter]
    fn pbc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        ndarray::arr1(&self.inner.pbc()).into_pyarray(py)
    }

    /// Lengths of the three lattice vectors ``[|a|, |b|, |c|]``.
    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.lengths().into_pyarray(py)
    }

    /// Lattice angles ``[alpha, beta, gamma]`` in degrees.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.angles().into_pyarray(py)
    }

    /// A general cell matrix (columns = lattice vectors) in LAMMPS
    /// restricted-triclinic form.
    #[staticmethod]
    fn restricted_matrix<'py>(
        py: Python<'py>,
        matrix: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(SimBox::restricted_matrix(matrix.as_array())
            .map_err(box_error_to_pyerr)?
            .into_pyarray(py))
    }

    /// LAMMPS-convention tilt factors ``(xy, xz, yz)``. Zero on
    /// orthogonal boxes.
    #[getter]
    fn tilts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.tilts().into_pyarray(py)
    }

    /// Perpendicular distances between opposite cell faces.
    #[getter]
    fn nearest_plane_distance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.nearest_plane_distance().into_pyarray(py)
    }

    /// Eight Cartesian cell corners.
    fn corners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.corners().into_pyarray(py)
    }

    /// Per-axis coordinate bounds as ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``.
    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.bounds().into_pyarray(py)
    }

    /// Convert Cartesian coordinates ``(N, 3)`` to fractional coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyz`` does not have 3 columns.
    fn to_frac<'py>(
        &self,
        py: Python<'py>,
        xyz: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = xyz.as_array();
        check_nx3(&view, "xyz")?;
        Ok(self.inner.to_frac(view).into_pyarray(py))
    }

    /// Convert fractional coordinates ``(N, 3)`` to Cartesian coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``frac`` does not have 3 columns.
    fn to_cart<'py>(
        &self,
        py: Python<'py>,
        frac: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = frac.as_array();
        check_nx3(&view, "frac")?;
        Ok(self.inner.to_cart(view).into_pyarray(py))
    }

    /// Wrap coordinates into the cell on the periodic axes **and** report the
    /// integer lattice shift applied to each point.
    ///
    /// Returns
    /// -------
    /// tuple[numpy.ndarray, numpy.ndarray]
    ///     ``(wrapped, shifts)``: the wrapped ``(N, 3)`` float64 coordinates
    ///     and the ``(N, 3)`` int32 shifts ``m`` with
    ///     ``wrapped = xyz - (matrix @ m.T).T``. For continuous coordinates
    ///     ``shifts`` are their image flags (the ``ix``/``iy``/``iz`` columns).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyz`` does not have 3 columns.
    fn wrap<'py>(&self, py: Python<'py>, xyz: ArrayArg2<'_, f64>) -> PyResult<WrapResult<'py>> {
        let view = xyz.as_array();
        check_nx3(&view, "xyz")?;
        let (wrapped, shifts) = self.inner.wrap(view);
        Ok((wrapped.into_pyarray(py), shifts.into_pyarray(py)))
    }

    /// Reconstruct unwrapped coordinates from wrapped coordinates and images.
    ///
    /// ``images`` is ``int32``, the schema's integer type, so a frame's
    /// ``ix``/``iy``/``iz`` columns pass in without a cast.
    fn unwrap<'py>(
        &self,
        py: Python<'py>,
        xyz: ArrayArg2<'_, f64>,
        images: ArrayArg2<'_, I>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let xyz = xyz.as_array();
        let images = images.as_array();
        if xyz.ncols() != 3 || xyz.raw_dim() != images.raw_dim() {
            return Err(PyValueError::new_err(
                "xyz and images must have identical shape (N,3)",
            ));
        }
        Ok(self.inner.unwrap(xyz, images).into_pyarray(py))
    }

    /// Displacement vectors ``xyzu2 - xyzu1``, minimum-imaged when
    /// ``minimum_image`` is set.
    ///
    /// Parameters
    /// ----------
    /// xyzu1 : numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     First set of Cartesian coordinates.
    /// xyzu2 : numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     Second set, of the same shape as ``xyzu1``.
    /// minimum_image : bool, optional
    ///     Apply the minimum-image convention. Default ``False``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     Rank matches the inputs.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ranks differ, 1-D length is not 3, columns != 3, or N differs.
    #[pyo3(signature = (xyzu1, xyzu2, minimum_image=false))]
    fn delta<'py>(
        &self,
        py: Python<'py>,
        xyzu1: &Bound<'py, PyAny>,
        xyzu2: &Bound<'py, PyAny>,
        minimum_image: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let (v1, squeezed1) = delta_points_nx3(xyzu1)?;
        let (v2, squeezed2) = delta_points_nx3(xyzu2)?;
        if squeezed1 != squeezed2 {
            return Err(PyValueError::new_err(
                "xyzu1 and xyzu2 must both be shape (N, 3) or both be shape (3,)",
            ));
        }
        if v1.raw_dim() != v2.raw_dim() {
            return Err(PyValueError::new_err(
                "xyzu1 and xyzu2 must have the same shape",
            ));
        }
        let d = self.inner.delta(v1.view(), v2.view(), minimum_image);
        if squeezed1 {
            Ok(d.remove_axis(Axis(0)).into_pyarray(py).into_any())
        } else {
            Ok(d.into_pyarray(py).into_any())
        }
    }

    /// Row-wise minimum-image distances between equally sized point arrays.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        points1: ArrayArg2<'_, f64>,
        points2: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        if points1.raw_dim() != points2.raw_dim() || points1.ncols() != 3 {
            return Err(PyValueError::new_err(
                "points1 and points2 must have identical shape (N,3)",
            ));
        }
        Ok(self.inner.distances(points1, points2).into_pyarray(py))
    }

    /// All pairwise minimum-image displacement vectors (`points2 - points1`).
    fn pairwise_delta<'py>(
        &self,
        py: Python<'py>,
        points1: ArrayArg2<'_, f64>,
        points2: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        check_nx3(&points1, "points1")?;
        check_nx3(&points2, "points2")?;
        Ok(self.inner.pairwise_delta(points1, points2).into_pyarray(py))
    }

    /// All pairwise minimum-image distances.
    fn pairwise_distances<'py>(
        &self,
        py: Python<'py>,
        points1: ArrayArg2<'_, f64>,
        points2: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        check_nx3(&points1, "points1")?;
        check_nx3(&points2, "points2")?;
        Ok(self
            .inner
            .pairwise_distances(points1, points2)
            .into_pyarray(py))
    }

    /// Return a box whose cell matrix is right-multiplied by `transformation`.
    fn transformed(&self, transformation: ArrayArg2<'_, f64>) -> PyResult<Self> {
        let transformation = transformation.as_array();
        if transformation.dim() != (3, 3) {
            return Err(PyValueError::new_err(
                "transformation must have shape (3,3)",
            ));
        }
        let inner = self
            .inner
            .transformed(&transformation.to_owned())
            .map_err(box_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether each point ``(N, 3)`` lies inside the primary cell
    /// (fractional coordinates in ``[0, 1)``).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyz`` does not have 3 columns.
    fn contains<'py>(
        &self,
        py: Python<'py>,
        xyz: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let view = xyz.as_array();
        check_nx3(&view, "xyz")?;
        Ok(self.inner.contains(view).into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Box(style={:?}, volume={:.2})",
            self.inner.style(),
            self.inner.volume()
        )
    }
}

/// Convert a [`BoxError`] to a Python `ValueError` carrying its message.
pub(crate) fn box_error_to_pyerr(e: BoxError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
