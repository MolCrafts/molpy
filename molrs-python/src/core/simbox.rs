//! Python wrapper for the simulation box (periodic boundary conditions).
//!
//! [`PyBox`] wraps the Rust [`SimBox`] and exposes construction helpers for
//! cubic, orthorhombic, and fully triclinic cells, plus coordinate
//! transformations (Cartesian <-> fractional), wrapping, and displacement
//! calculations with optional minimum-image convention.
//!
//! All length quantities are in the same units as the stored coordinates
//! (typically angstroms).

use crate::array::{ArrayArg1, ArrayArg2};
use crate::core::frame::PyFrame;
use molrs::core::keys;
use molrs::core::schema::block_names;
use molrs::core::{BoxError, SimBox};
use molrs::op::{F, I};
use ndarray::{Array1, Array2, Axis, array};
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

/// Points for [`PyBox::from_bounds`]: an `(N, 3)` array, or a `Frame`'s atom
/// coordinates.
fn bounds_points(arg: &Bound<'_, PyAny>) -> PyResult<Array2<F>> {
    if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
        return frame.with_frame(|f| {
            if let Some(atoms) = f.get(block_names::ATOMS)
                && let Some(key) = keys::COORDS
                    .into_iter()
                    .find(|k| atoms.validity(k).is_some())
            {
                return Err(PyValueError::new_err(format!(
                    "atoms column '{key}' has missing rows"
                )));
            }
            f.coords().map_err(|e| PyValueError::new_err(e.to_string()))
        })?;
    }
    let arr = arg
        .extract::<ArrayArg2<'_, f64>>()
        .map_err(|_| PyValueError::new_err("points must be an (N,3) float array or a Frame"))?;
    let view = arr.as_array();
    if view.ncols() != 3 {
        return Err(PyValueError::new_err("points must have shape (N,3)"));
    }
    Ok(view.to_owned())
}

/// Simulation box with periodic boundary conditions, exposed to Python as
/// `molrs.core.Box`.
///
/// The box is defined by a 3x3 cell matrix **H** whose columns are the
/// lattice vectors, an origin point, and per-axis PBC flags.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs.core import Box
///
/// box = Box.cube(10.0)                       # 10 x 10 x 10 cubic
/// box = Box.ortho(np.array([10, 20, 30]))    # orthorhombic
/// print(box.volume())                        # 6000.0
/// ```
#[pyclass(module = "molrs.core", name = "Box", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyBox {
    pub(crate) inner: SimBox,
}

#[pymethods]
impl PyBox {
    /// Create a simulation box from a cell matrix, a diagonal, or no cell.
    ///
    /// Parameters
    /// ----------
    /// h : array_like, shape (3, 3) or (3,), optional
    ///     Cell matrix with lattice vectors as **columns**, or the diagonal of
    ///     an orthorhombic one. ``None`` or an all-zero matrix is no cell: a
    ///     free box carrying the identity as a placeholder
    ///     (``cell_defined`` false).
    /// origin : array_like, shape (3,), optional
    ///     Origin of the box in Cartesian coordinates. Defaults to
    ///     ``[0, 0, 0]``.
    /// pbc : array_like of bool, shape (3,), optional
    ///     Periodic boundary flags for x, y, z. Defaults to periodic on every
    ///     axis when there is a cell and on none when there is not.
    /// cell_defined : bool, optional
    ///     Whether the cell is geometrically defined; inferred from ``h`` when
    ///     omitted. ``False`` ignores ``h`` (a store's undefined cell).
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``h`` is neither ``(3, 3)`` nor ``(3,)``, or a defined cell
    ///     matrix is singular.
    ///
    /// Examples
    /// --------
    /// >>> box = Box(np.eye(3) * 10.0)
    /// >>> Box([10.0, 20.0, 30.0]).style
    /// 'orthogonal'
    /// >>> Box().is_free
    /// True
    #[new]
    #[pyo3(signature = (h=None, origin=None, pbc=None, cell_defined=None))]
    fn new(
        h: Option<PyArrayLikeDyn<'_, F, AllowTypeChange>>,
        origin: Option<ArrayArg1<'_, f64>>,
        pbc: Option<ArrayArg1<'_, bool>>,
        cell_defined: Option<bool>,
    ) -> PyResult<Self> {
        let h_matrix = match h {
            None => None,
            Some(h) => {
                let view = h.as_array();
                Some(match view.shape() {
                    [3, 3] => Array2::from_shape_fn((3, 3), |(i, j)| view[[i, j]]),
                    [3] => Array2::from_diag(&Array1::from_shape_fn(3, |i| view[[i]])),
                    shape => {
                        return Err(PyValueError::new_err(format!(
                            "h must be (3, 3) or (3,), got {shape:?}"
                        )));
                    }
                })
            }
        };
        // No matrix, or an all-zero one, is no cell.
        let cell_defined = cell_defined.unwrap_or_else(|| {
            h_matrix
                .as_ref()
                .is_some_and(|h| h.iter().any(|&x| x != 0.0))
        });
        let pbc_array = match pbc {
            Some(_) => parse_pbc(pbc)?,
            None => [cell_defined; 3],
        };
        let inner = SimBox::new_cell(
            h_matrix.unwrap_or_else(|| Array2::eye(3)),
            parse_origin(origin)?,
            pbc_array,
            cell_defined,
        )
        .map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Whether the cell is geometrically defined. ``False`` marks a "no-cell"
    /// box (undefined / zero-volume), distinct from ``is_free`` (periodicity).
    #[getter]
    fn cell_defined(&self) -> bool {
        self.inner.is_cell_defined()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.h(py),
                this.origin(py),
                this.pbc(py),
                this.cell_defined(),
            ),
        )
    }

    /// Create a cubic simulation box.
    ///
    /// Parameters
    /// ----------
    /// a : float
    ///     Side length of the cube in the same length unit as coordinates.
    /// origin : numpy.ndarray, shape (3,), optional
    ///     Box origin. Defaults to ``[0, 0, 0]``.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags. Defaults to ``[True, True, True]``.
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Examples
    /// --------
    /// >>> box = Box.cube(10.0)
    /// >>> box.volume()
    /// 1000.0
    #[staticmethod]
    #[pyo3(signature = (a, origin=None, pbc=None))]
    fn cube(
        a: f64,
        origin: Option<ArrayArg1<'_, f64>>,
        pbc: Option<ArrayArg1<'_, bool>>,
    ) -> PyResult<Self> {
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;
        let inner = SimBox::cube(a, origin_vec, pbc_array).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Create an orthorhombic (rectangular) simulation box.
    ///
    /// Parameters
    /// ----------
    /// lengths : numpy.ndarray, shape (3,), dtype float
    ///     Side lengths ``[Lx, Ly, Lz]``.
    /// origin : numpy.ndarray, shape (3,), optional
    ///     Box origin. Defaults to ``[0, 0, 0]``.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags. Defaults to ``[True, True, True]``.
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``lengths`` does not have exactly 3 elements.
    ///
    /// Examples
    /// --------
    /// >>> box = Box.ortho(np.array([10.0, 20.0, 30.0]))
    #[staticmethod]
    #[pyo3(signature = (lengths, origin=None, pbc=None))]
    fn ortho(
        lengths: ArrayArg1<'_, f64>,
        origin: Option<ArrayArg1<'_, f64>>,
        pbc: Option<ArrayArg1<'_, bool>>,
    ) -> PyResult<Self> {
        let lv = lengths.as_slice()?;
        if lv.len() != 3 {
            return Err(PyValueError::new_err("lengths must have length 3"));
        }
        let lengths_arr = array![lv[0], lv[1], lv[2]];
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;
        let inner =
            SimBox::ortho(lengths_arr, origin_vec, pbc_array).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Create a tight orthorhombic box around a point cloud.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), or Frame
    ///     The points to enclose. A ``Frame`` contributes the ``x``/``y``/``z``
    ///     columns of its ``atoms`` block.
    /// padding : float or numpy.ndarray, shape (3,)
    ///     Margin added on each side, one value for all axes or one per axis.
    ///     Must be non-negative.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags. Defaults to ``[True, True, True]``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there are no points, a coordinate column is missing or has
    ///     holes, ``padding`` is negative or not of length 3, or the box is
    ///     degenerate.
    #[staticmethod]
    #[pyo3(signature = (points, padding, pbc=None))]
    fn from_bounds(
        points: &Bound<'_, PyAny>,
        padding: &Bound<'_, PyAny>,
        pbc: Option<ArrayArg1<'_, bool>>,
    ) -> PyResult<Self> {
        let points = bounds_points(points)?;
        let padding = if let Ok(scalar) = padding.extract::<F>() {
            [scalar; 3]
        } else {
            // A sequence or a 1-D array alike; numpy arrays iterate as floats.
            let v = padding.extract::<Vec<F>>().map_err(|_| {
                PyValueError::new_err("padding must be a float or a sequence of length 3")
            })?;
            if v.len() != 3 {
                return Err(PyValueError::new_err("padding must have length 3"));
            }
            [v[0], v[1], v[2]]
        };
        let pbc = parse_pbc(pbc)?;
        let inner = SimBox::from_bounds(points.view(), padding, pbc).map_err(box_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Whether ``other`` describes the same cell within an absolute tolerance.
    ///
    /// True iff every cell-matrix entry and origin component differ by at most
    /// ``tol`` (length units), the PBC flags are identical and both boxes agree
    /// on ``cell_defined``. ``tol=0.0`` is exact equality.
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

    /// Volume of the simulation box.
    ///
    /// Returns
    /// -------
    /// float
    ///     Volume in length_unit^3 (e.g. angstrom^3).
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

    /// Return a lattice vector by index.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Lattice vector index: 0, 1, or 2.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,)
    ///     The lattice vector as a Cartesian 3-vector.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``index`` is not 0, 1, or 2.
    fn lattice<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if index >= 3 {
            return Err(PyValueError::new_err("index must be 0, 1, or 2"));
        }
        let vec = self.inner.lattice(index);
        Ok(vec.into_pyarray(py))
    }

    /// Cell matrix **H** (3x3), lattice vectors as columns.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3, 3), dtype float
    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.h_view().to_owned().into_pyarray(py)
    }

    /// Inverse cell matrix.
    #[getter]
    fn inverse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.inv_view().to_owned().into_pyarray(py)
    }

    /// Box origin in Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), dtype float
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.origin_view().to_owned().into_pyarray(py)
    }

    /// Periodic boundary condition flags ``[pbc_x, pbc_y, pbc_z]``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), dtype bool
    #[getter]
    fn pbc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.inner.pbc_view().to_owned().into_pyarray(py)
    }

    /// Lengths of the three lattice vectors (property; ``[|a|, |b|, |c|]``).
    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.lengths().into_pyarray(py)
    }

    /// Lattice angles ``[alpha, beta, gamma]`` in degrees.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.angles().into_pyarray(py)
    }

    #[staticmethod]
    fn matrix_from_lengths_angles<'py>(
        py: Python<'py>,
        lengths: [f64; 3],
        angles: [f64; 3],
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(SimBox::matrix_from_lengths_angles(lengths, angles)
            .map_err(box_error_to_pyerr)?
            .into_pyarray(py))
    }

    #[staticmethod]
    fn matrix_from_lengths_tilts<'py>(
        py: Python<'py>,
        lengths: [f64; 3],
        tilts: [f64; 3],
    ) -> Bound<'py, PyArray2<f64>> {
        SimBox::matrix_from_lengths_tilts(lengths, tilts).into_pyarray(py)
    }

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
        let h = self.inner.h_view();
        let arr = ndarray::array![h[(0, 1)], h[(0, 2)], h[(1, 2)]];
        arr.into_pyarray(py)
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

    /// Minimum-image displacement from ``r1`` to ``r2``.
    fn shortest_vector<'py>(
        &self,
        py: Python<'py>,
        r1: ArrayArg1<'_, f64>,
        r2: ArrayArg1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let r1 = r1.as_array();
        let r2 = r2.as_array();
        if r1.len() != 3 || r2.len() != 3 {
            return Err(PyValueError::new_err("r1 and r2 must have length 3"));
        }
        Ok(self.inner.shortest_vector(r1, r2).into_pyarray(py))
    }

    /// Squared minimum-image distance between two points.
    fn distance_squared(&self, r1: ArrayArg1<'_, f64>, r2: ArrayArg1<'_, f64>) -> PyResult<f64> {
        let r1 = r1.as_array();
        let r2 = r2.as_array();
        if r1.len() != 3 || r2.len() != 3 {
            return Err(PyValueError::new_err("r1 and r2 must have length 3"));
        }
        Ok(self.inner.calc_distance2(r1, r2))
    }

    /// Minimum-image distance between two points.
    fn distance(&self, r1: ArrayArg1<'_, f64>, r2: ArrayArg1<'_, f64>) -> PyResult<f64> {
        Ok(self.distance_squared(r1, r2)?.sqrt())
    }

    /// Convert Cartesian coordinates to fractional coordinates.
    ///
    /// Parameters
    /// ----------
    /// xyz : numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Fractional coordinates in the range ``[0, 1)`` for wrapped points.
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
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let frac = self.inner.to_frac(view);
        Ok(frac.into_pyarray(py))
    }

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// Parameters
    /// ----------
    /// xyzs : numpy.ndarray, shape (N, 3), dtype float
    ///     Fractional coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyzs`` does not have 3 columns.
    fn to_cart<'py>(
        &self,
        py: Python<'py>,
        xyzs: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = xyzs.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let cart = self.inner.to_cart(view);
        Ok(cart.into_pyarray(py))
    }

    /// Wrap coordinates into the primary simulation cell.
    ///
    /// Applies periodic wrapping along axes where PBC is enabled.
    ///
    /// Parameters
    /// ----------
    /// xyzu : numpy.ndarray, shape (N, 3), dtype float
    ///     Unwrapped Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Wrapped Cartesian coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyzu`` does not have 3 columns.
    fn wrap<'py>(
        &self,
        py: Python<'py>,
        xyzu: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let view = xyzu.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let wrapped = self.inner.wrap(view);
        Ok(wrapped.into_pyarray(py))
    }

    /// Integer periodic image flags for Cartesian coordinates.
    ///
    /// The flags are ``int32``, the schema's integer type for the
    /// ``ix``/``iy``/``iz`` columns.
    fn images<'py>(
        &self,
        py: Python<'py>,
        xyz: ArrayArg2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<I>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        Ok(self.inner.images(view).into_pyarray(py))
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

    /// Compute displacement vectors between two point sets.
    ///
    /// Calculates ``xyzu2 - xyzu1`` with optional minimum-image convention
    /// for periodic systems.
    ///
    /// Parameters
    /// ----------
    /// xyzu1 : numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     First set of Cartesian coordinates.
    /// xyzu2 : numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     Second set of Cartesian coordinates. Must have the same rank as
    ///     ``xyzu1`` (both 1-D or both 2-D).
    /// minimum_image : bool, optional
    ///     If ``True``, apply the minimum-image convention to displacements.
    ///     Default is ``False``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3) or (3,), dtype float
    ///     Displacement vectors ``xyzu2 - xyzu1``. Rank matches the inputs.
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
        if points1.ncols() != 3 || points2.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
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
        if points1.ncols() != 3 || points2.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
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

    /// Test whether each point lies inside the primary simulation cell.
    ///
    /// Parameters
    /// ----------
    /// xyz : numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype bool
    ///     ``True`` for points inside the cell.
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
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let inside = self.inner.contains(view);
        Ok(inside.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Box(volume={:.2})", self.inner.volume())
    }
}

/// Parse an optional origin array, defaulting to `[0, 0, 0]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
fn parse_origin(origin: Option<ArrayArg1<'_, f64>>) -> PyResult<Array1<F>> {
    match origin {
        Some(o) => {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            Ok(array![s[0], s[1], s[2]])
        }
        None => Ok(array![0.0 as F, 0.0 as F, 0.0 as F]),
    }
}

/// Parse an optional PBC flag array, defaulting to `[true, true, true]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
fn parse_pbc(pbc: Option<ArrayArg1<'_, bool>>) -> PyResult<[bool; 3]> {
    match pbc {
        Some(p) => {
            let s = p.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            Ok([s[0], s[1], s[2]])
        }
        None => Ok([true, true, true]),
    }
}

/// Convert a [`BoxError`] to a Python `ValueError`.
fn box_error_to_pyerr(e: BoxError) -> PyErr {
    PyValueError::new_err(format!("{:?}", e))
}
