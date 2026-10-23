//! Python wrappers for geometric regions.
//!
//! A region is a solid with a signed distance to its boundary. Every class
//! answers ``contains(points)``, ``distance(points)`` (negative inside,
//! positive outside, zero on the boundary) and ``bounds()``, and composes
//! with ``&`` (intersection), ``|`` (union) and ``~`` (complement). Every
//! shape describes its inside: outside a sphere is ``~Sphere(...)``, a shell
//! is ``outer & ~inner``.
//!
//! | Class            | Description                                   |
//! |------------------|-----------------------------------------------|
//! | `Sphere`         | Solid sphere                                  |
//! | `Cuboid`         | Axis-aligned box (incl. cubes)                |
//! | `Parallelepiped` | General (triclinic) box volume                |
//! | `HalfSpace`      | One side of a plane                           |
//! | `Cylinder`       | Finite capped cylinder                        |
//! | `Ellipsoid`      | Axis-aligned ellipsoid                        |
//! | `Polyhedron`     | Solid bounded by a watertight `TriMesh`       |
//! | `SphereUnion`    | Union of spheres, optionally periodic         |
//! | `Region`         | Composed region (from ``&`` / ``|`` / ``~``)  |
//!
//! All lengths are in the same unit as the input coordinates (typically
//! angstroms).
//!
//! Every region exports its shared geometry as a ``PyCapsule`` named
//! ``molrs.RegionRef/<major.minor>`` through ``_ffi_regionref_capsule()``, so
//! a downstream extension built on the same molrs line (molpack) can evaluate
//! it in place without marshalling.

use crate::array::{ArrayArg1, ArrayArg2};
use crate::core::mesh::PyTriMesh;
use crate::core::simbox::{PyBox, PyCell};
use molrs::core::{
    AndRegion, Cell, Cuboid, Cylinder, Ellipsoid, HalfSpace, NotRegion, OrRegion, Parallelepiped,
    Polyhedron, Region, Sphere, SphereUnion,
};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyTuple};
use std::sync::Arc;

/// Type-erased region handle for dynamic dispatch.
type DynRegion = Arc<dyn Region + Send + Sync>;

// ---------------------------------------------------------------------------
// Shared implementation
// ---------------------------------------------------------------------------

/// Test which points are inside a region, returning a boolean mask.
fn contains_impl<'py>(
    region: &dyn Region,
    py: Python<'py>,
    points: ArrayArg2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let arr = points.as_array().to_owned();
    if arr.ncols() != 3 {
        return Err(PyValueError::new_err("points must have shape (N, 3)"));
    }
    let mask = region.contains(&arr);
    Ok(mask.into_pyarray(py))
}

/// Signed distance of each row of an ``(N, 3)`` array to the boundary:
/// negative inside, positive outside.
fn distance_impl<'py>(
    region: &dyn Region,
    py: Python<'py>,
    points: ArrayArg2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = points.as_array();
    if arr.ncols() != 3 {
        return Err(PyValueError::new_err("points must have shape (N, 3)"));
    }
    let d: Vec<f64> = arr
        .rows()
        .into_iter()
        .map(|r| region.distance(&[r[0], r[1], r[2]]))
        .collect();
    Ok(d.into_pyarray(py))
}

/// Return the axis-aligned bounding box of a region as a ``(3, 2)`` array.
fn bounds_impl<'py>(region: &dyn Region, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
    region.bounds().into_pyarray(py)
}

/// `[F; 3]` from a length-3 sequence argument (a list, a tuple, or a 1-D
/// array all extract).
fn vec3(v: Vec<f64>, what: &str) -> PyResult<[f64; 3]> {
    if v.len() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have length 3")));
    }
    Ok([v[0], v[1], v[2]])
}

/// The three-element numpy array of `v`.
fn np3<'py>(py: Python<'py>, v: [f64; 3]) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(vec![v[0], v[1], v[2]]).into_pyarray(py)
}

/// `Send` wrapper for the raw handle pointer a capsule carries.
///
/// The capsule's ``void*`` is ``*mut RegionRefPtr`` ≡ ``*mut *mut RegionRef``
/// — the same double indirection as ``Frame._ffi_frameref_capsule``. A bare
/// raw pointer is not `Send`, which `PyCapsule::new` requires of its payload;
/// the handle behind it is `Send + Sync` (an `Arc<dyn Region>`), so the
/// assertion is sound.
#[repr(transparent)]
struct RegionRefPtr(*mut molrs_ffi::RegionRef);
// SAFETY: see the type's doc comment.
unsafe impl Send for RegionRefPtr {}

/// Export a region as a ``molrs.RegionRef/<major.minor>`` capsule.
fn region_capsule<'py>(py: Python<'py>, region: DynRegion) -> PyResult<Bound<'py, PyCapsule>> {
    let raw = RegionRefPtr(Box::into_raw(Box::new(molrs_ffi::RegionRef::new(region))));
    let name = molrs_ffi::abi::regionref_capsule_name().to_owned();
    PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: RegionRefPtr, _ctx| {
        // SAFETY: `ptr.0` is the pointer produced by `Box::into_raw` above
        // and is reclaimed exactly once when the capsule dies.
        drop(unsafe { Box::from_raw(ptr.0) });
    })
}

/// Extract the shared region behind any supported Python region object.
pub(crate) fn extract_region(obj: &Bound<'_, PyAny>) -> PyResult<DynRegion> {
    if let Ok(r) = obj.extract::<PySphere>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyCuboid>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyParallelepiped>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyHalfSpace>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyCylinder>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyEllipsoid>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyPolyhedron>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PySphereUnion>() {
        return Ok(r.inner);
    }
    if let Ok(r) = obj.extract::<PyRegion>() {
        return Ok(r.inner);
    }
    Err(PyTypeError::new_err(
        "expected a region: Sphere, Cuboid, Parallelepiped, HalfSpace, Cylinder, Ellipsoid, \
         Polyhedron, SphereUnion, or a composed Region",
    ))
}

/// `left & right`, `left | right`, or `~left`, keeping the operand objects
/// as the composition tree so the result pickles through them.
fn compose(
    op: &str,
    left: &Bound<'_, PyAny>,
    right: Option<&Bound<'_, PyAny>>,
) -> PyResult<PyRegion> {
    let py = left.py();
    let a = extract_region(left)?;
    let (inner, tree): (DynRegion, Bound<'_, PyTuple>) = match (op, right) {
        ("and", Some(b_obj)) => {
            let b = extract_region(b_obj)?;
            (
                Arc::new(AndRegion::new(a, b)),
                PyTuple::new(
                    py,
                    [
                        op.into_pyobject(py)?.into_any(),
                        left.clone(),
                        b_obj.clone(),
                    ],
                )?,
            )
        }
        ("or", Some(b_obj)) => {
            let b = extract_region(b_obj)?;
            (
                Arc::new(OrRegion::new(a, b)),
                PyTuple::new(
                    py,
                    [
                        op.into_pyobject(py)?.into_any(),
                        left.clone(),
                        b_obj.clone(),
                    ],
                )?,
            )
        }
        ("not", None) => (
            Arc::new(NotRegion::new(a)),
            PyTuple::new(py, [op.into_pyobject(py)?.into_any(), left.clone()])?,
        ),
        _ => return Err(PyValueError::new_err("unknown region composition")),
    };
    Ok(PyRegion {
        inner,
        tree: tree.into_any().unbind(),
    })
}

/// [`contains_impl`] over a block's `x` / `y` / `z` columns.
fn mask_impl<'py>(
    region: &dyn Region,
    block: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let py = block.py();
    let xyz = block.get_item(("x", "y", "z"))?;
    let points: ArrayArg2<'_, f64> = xyz.extract()?;
    contains_impl(region, py, points)
}

/// `left op right` as a composed region, or `NotImplemented` when `right` is
/// not a region (so Python asks `right.__rand__` / `__ror__`).
fn compose_or_defer(
    op: &str,
    left: &Bound<'_, PyAny>,
    right: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let py = left.py();
    if extract_region(right).is_err() {
        return Ok(py.NotImplemented());
    }
    Ok(Py::new(py, compose(op, left, Some(right))?)?.into_any())
}

/// The methods every region class shares: the two queries, the bounds, the
/// block mask, the three operators, and the FFI capsule.
macro_rules! region_methods {
    ($ty:ident) => {
        #[pymethods]
        impl $ty {
            /// Test which points are inside this region.
            ///
            /// Parameters
            /// ----------
            /// points : numpy.ndarray, shape (N, 3), dtype float
            ///     Test points.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray, shape (N,), dtype bool
            ///     ``True`` for points inside (the boundary counts as inside).
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``points`` does not have 3 columns.
            fn contains<'py>(
                &self,
                py: Python<'py>,
                points: ArrayArg2<'_, f64>,
            ) -> PyResult<Bound<'py, PyArray1<bool>>> {
                contains_impl(self.inner.as_ref(), py, points)
            }

            /// Signed distance of each point to the boundary.
            ///
            /// Parameters
            /// ----------
            /// points : numpy.ndarray, shape (N, 3), dtype float
            ///     Query points.
            ///
            /// Returns
            /// -------
            /// numpy.ndarray, shape (N,), dtype float
            ///     Negative inside, positive outside, zero on the boundary; same
            ///     unit as the coordinates.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If ``points`` does not have 3 columns.
            fn distance<'py>(
                &self,
                py: Python<'py>,
                points: ArrayArg2<'_, f64>,
            ) -> PyResult<Bound<'py, PyArray1<f64>>> {
                distance_impl(self.inner.as_ref(), py, points)
            }

            /// Axis-aligned bounding box, shape ``(3, 2)``:
            /// ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
            fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
                bounds_impl(self.inner.as_ref(), py)
            }

            /// Which rows of ``block`` lie inside: :meth:`contains` of its
            /// ``x`` / ``y`` / ``z`` columns.
            ///
            /// Raises
            /// ------
            /// KeyError
            ///     If ``block`` lacks ``x``, ``y`` or ``z``.
            fn mask<'py>(&self, block: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<bool>>> {
                mask_impl(self.inner.as_ref(), block)
            }

            /// The rows of ``block`` inside the region: ``block[self.mask(block)]``.
            fn __call__<'py>(&self, block: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
                let mask = mask_impl(self.inner.as_ref(), block)?;
                block.get_item(mask)
            }

            /// Intersection: ``self & other``. An operand that is not a region
            /// gets ``NotImplemented``, so its own ``__rand__`` (a selector's)
            /// decides.
            fn __and__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                compose_or_defer("and", slf.as_any(), other)
            }

            /// Union: ``self | other``; ``NotImplemented`` for a non-region.
            fn __or__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
                compose_or_defer("or", slf.as_any(), other)
            }

            /// Complement: ``~self``, every point ``self`` does not contain.
            /// A point on the boundary is ``self``'s, so ``self`` and
            /// ``~self`` never share a point.
            fn __invert__(slf: &Bound<'_, Self>) -> PyResult<PyRegion> {
                compose("not", slf.as_any(), None)
            }

            /// Export this region's shared geometry as a ``PyCapsule`` named
            /// ``"molrs.RegionRef/<major.minor>"``.
            ///
            /// The capsule's pointer is ``*mut *mut`` :class:`molrs_ffi.RegionRef`
            /// (a cloned handle over the same ``Arc``). A consumer built on the
            /// same molrs minor line resolves it and evaluates the region in
            /// place; one on another line fails the capsule name check.
            fn _ffi_regionref_capsule<'py>(
                &self,
                py: Python<'py>,
            ) -> PyResult<Bound<'py, PyCapsule>> {
                region_capsule(py, self.inner.clone())
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Sphere
// ---------------------------------------------------------------------------

/// Solid sphere region.
///
/// Exposed to Python as `molpy.core.Sphere`.
///
/// Parameters
/// ----------
/// center : numpy.ndarray, shape (3,), dtype float
///     Center of the sphere.
/// radius : float
///     Radius of the sphere.
///
/// Examples
/// --------
/// >>> s = Sphere(np.array([0, 0, 0]), 5.0)
/// >>> mask = s.contains(points)
/// >>> shell = s & ~Sphere(np.array([0, 0, 0]), 2.0)
/// >>> d = s.distance(points)  # negative inside
#[pyclass(module = "molpy.core", name = "Sphere", from_py_object, subclass)]
#[derive(Clone)]
pub struct PySphere {
    inner: Arc<Sphere>,
}

#[pymethods]
impl PySphere {
    /// Create a solid sphere region.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``center`` does not have length 3.
    #[new]
    fn new(center: Vec<f64>, radius: f64) -> PyResult<Self> {
        let c = vec3(center, "center")?;
        Ok(Self {
            inner: Arc::new(Sphere::new(Array1::from_vec(c.to_vec()), radius)),
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.inner.center.to_owned().into_pyarray(py),
                this.inner.radius,
            ),
        )
    }

    /// Center, shape ``(3,)``.
    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.center.to_owned().into_pyarray(py)
    }

    /// Radius.
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius
    }

    fn __repr__(&self) -> String {
        format!(
            "Sphere(center=[{:.2}, {:.2}, {:.2}], radius={:.2})",
            self.inner.center[0], self.inner.center[1], self.inner.center[2], self.inner.radius
        )
    }
}
region_methods!(PySphere);

// ---------------------------------------------------------------------------
// Cuboid
// ---------------------------------------------------------------------------

/// Axis-aligned cuboid (box) region, exposed to Python as `molpy.core.Cuboid`.
///
/// A point is inside when `origin[d] <= p[d] <= origin[d] + lengths[d]` on
/// every axis.
#[pyclass(module = "molpy.core", name = "Cuboid", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyCuboid {
    inner: Arc<Cuboid>,
}

#[pymethods]
impl PyCuboid {
    /// Create an axis-aligned cuboid region from its minimum corner
    /// ``origin`` and edge ``lengths``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``origin`` or ``lengths`` does not have length 3.
    #[new]
    fn new(origin: Vec<f64>, lengths: Vec<f64>) -> PyResult<Self> {
        let o = vec3(origin, "origin")?;
        let l = vec3(lengths, "lengths")?;
        Ok(Self {
            inner: Arc::new(Cuboid::new(
                Array1::from_vec(o.to_vec()),
                Array1::from_vec(l.to_vec()),
            )),
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                this.inner.origin.to_owned().into_pyarray(py),
                this.inner.lengths.to_owned().into_pyarray(py),
            ),
        )
    }

    /// Minimum corner, shape ``(3,)``.
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.origin.to_owned().into_pyarray(py)
    }

    /// Edge lengths, shape ``(3,)``.
    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.lengths.to_owned().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "Cuboid(origin=[{:.2}, {:.2}, {:.2}], lengths=[{:.2}, {:.2}, {:.2}])",
            self.inner.origin[0],
            self.inner.origin[1],
            self.inner.origin[2],
            self.inner.lengths[0],
            self.inner.lengths[1],
            self.inner.lengths[2],
        )
    }
}
region_methods!(PyCuboid);

// ---------------------------------------------------------------------------
// Parallelepiped
// ---------------------------------------------------------------------------

/// General parallelepiped (oblique box) region, exposed as `molpy.core.Parallelepiped`.
///
/// Defined by an origin corner and a 3×3 edge matrix ``h`` whose **columns**
/// are the three edge vectors. A point is inside when its fractional
/// coordinates lie in the closed cell ``[0, 1]³``; ``distance`` is measured
/// perpendicular to the bounding planes, in the input length unit.
///
/// This is pure geometric containment — **not** a periodic simulation box.
/// For PBC / MIC / wrap, use :class:`molpy.core.Box`.
#[pyclass(
    module = "molpy.core",
    name = "Parallelepiped",
    from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyParallelepiped {
    inner: Arc<Parallelepiped>,
}

#[pymethods]
impl PyParallelepiped {
    /// Create a parallelepiped from a :class:`Cell` — the edge matrix
    /// (columns = edges, Å), orthorhombic lengths, lengths and angles, or
    /// lengths and tilts, as a :class:`Box` takes one — and ``origin``, the
    /// corner the edges hang from (length 3, Å). A cube of edge ``a`` is
    /// ``Parallelepiped(Cell.lengths([a, a, a]), origin)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``Cell.undefined()`` (no edges), a cell the :class:`Cell` form
    ///     refuses (a non-positive length, an impossible angle), a singular
    ///     or non-finite edge matrix, or an ``origin`` not of length 3.
    #[new]
    fn new(cell: PyCell, origin: Vec<f64>) -> PyResult<Self> {
        let o = vec3(origin, "origin")?;
        let inner = Parallelepiped::new(cell.inner, Array1::from_vec(o.to_vec()))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Edge matrix ``h`` (3×3, Å), the edge vectors as columns.
    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.h().to_owned().into_pyarray(py)
    }

    /// Origin corner, shape ``(3,)`` (Å).
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.origin().to_owned().into_pyarray(py)
    }

    /// Signed volume ``det(h)``.
    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                PyCell {
                    inner: Cell::Matrix(std::array::from_fn(|r| {
                        std::array::from_fn(|c| this.inner.h()[[r, c]])
                    })),
                },
                this.inner.origin().to_vec(),
            ),
        )
    }

    fn __repr__(&self) -> String {
        let o = self.inner.origin();
        format!(
            "Parallelepiped(origin=[{:.2}, {:.2}, {:.2}], volume={:.2})",
            o[0],
            o[1],
            o[2],
            self.inner.volume()
        )
    }
}
region_methods!(PyParallelepiped);

// ---------------------------------------------------------------------------
// HalfSpace
// ---------------------------------------------------------------------------

/// One side of a plane, exposed as `molpy.core.HalfSpace`.
///
/// Inside is the side the ``normal`` points *away* from (``n · (x − p) <= 0``);
/// the other side is ``~HalfSpace(...)``. ``distance`` is the exact distance
/// to the plane.
///
/// Parameters
/// ----------
/// normal : numpy.ndarray, shape (3,), dtype float
///     Outward normal (any non-zero length).
/// point : numpy.ndarray, shape (3,), dtype float
///     A point on the plane.
#[pyclass(module = "molpy.core", name = "HalfSpace", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyHalfSpace {
    inner: Arc<HalfSpace>,
}

#[pymethods]
impl PyHalfSpace {
    /// Create the half-space behind the plane through ``point`` with outward
    /// ``normal``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``normal`` is zero or a length is wrong.
    #[new]
    fn new(normal: Vec<f64>, point: Vec<f64>) -> PyResult<Self> {
        let n = vec3(normal, "normal")?;
        let p = vec3(point, "point")?;
        let inner = HalfSpace::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Unit outward normal, shape ``(3,)``.
    fn normal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np3(py, self.inner.normal())
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        let n = this.inner.normal();
        let point = [
            n[0] * this.inner.offset(),
            n[1] * this.inner.offset(),
            n[2] * this.inner.offset(),
        ];
        crate::pickle::reduce_via_type(slf.as_any(), (np3(py, n), np3(py, point)))
    }

    fn __repr__(&self) -> String {
        let n = self.inner.normal();
        format!(
            "HalfSpace(normal=[{:.2}, {:.2}, {:.2}], offset={:.2})",
            n[0],
            n[1],
            n[2],
            self.inner.offset()
        )
    }
}
region_methods!(PyHalfSpace);

// ---------------------------------------------------------------------------
// Cylinder
// ---------------------------------------------------------------------------

/// Finite capped cylinder, exposed as `molpy.core.Cylinder`.
///
/// Parameters
/// ----------
/// base : numpy.ndarray, shape (3,), dtype float
///     Centre of the first cap.
/// axis : numpy.ndarray, shape (3,), dtype float
///     Direction from the first cap to the second (any non-zero length).
/// radius : float
/// length : float
///     Distance between the caps.
#[pyclass(module = "molpy.core", name = "Cylinder", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyCylinder {
    inner: Arc<Cylinder>,
}

#[pymethods]
impl PyCylinder {
    /// Create a finite cylinder.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``axis`` is zero, or ``radius`` / ``length`` is not positive.
    #[new]
    fn new(base: Vec<f64>, axis: Vec<f64>, radius: f64, length: f64) -> PyResult<Self> {
        let b = vec3(base, "base")?;
        let a = vec3(axis, "axis")?;
        let inner = Cylinder::new(b, a, radius, length)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                np3(py, this.inner.base()),
                np3(py, this.inner.axis()),
                this.inner.radius(),
                this.inner.length(),
            ),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Cylinder(radius={:.2}, length={:.2})",
            self.inner.radius(),
            self.inner.length()
        )
    }
}
region_methods!(PyCylinder);

// ---------------------------------------------------------------------------
// Ellipsoid
// ---------------------------------------------------------------------------

/// Axis-aligned ellipsoid, exposed as `molpy.core.Ellipsoid`.
///
/// ``distance`` has the exact sign; its magnitude is a lower bound on the
/// Euclidean distance (exact along the shortest semi-axis).
///
/// Parameters
/// ----------
/// center : numpy.ndarray, shape (3,), dtype float
/// semi_axes : numpy.ndarray, shape (3,), dtype float
///     Semi-axes along x, y, z.
#[pyclass(module = "molpy.core", name = "Ellipsoid", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyEllipsoid {
    inner: Arc<Ellipsoid>,
}

#[pymethods]
impl PyEllipsoid {
    /// Create an axis-aligned ellipsoid.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a semi-axis is not positive.
    #[new]
    fn new(center: Vec<f64>, semi_axes: Vec<f64>) -> PyResult<Self> {
        let c = vec3(center, "center")?;
        let a = vec3(semi_axes, "semi_axes")?;
        let inner = Ellipsoid::new(c, a).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (
                np3(py, this.inner.center()),
                np3(py, this.inner.semi_axes()),
            ),
        )
    }

    fn __repr__(&self) -> String {
        let a = self.inner.semi_axes();
        format!(
            "Ellipsoid(semi_axes=[{:.2}, {:.2}, {:.2}])",
            a[0], a[1], a[2]
        )
    }
}
region_methods!(PyEllipsoid);

// ---------------------------------------------------------------------------
// Polyhedron
// ---------------------------------------------------------------------------

/// Solid bounded by a watertight :class:`TriMesh`, exposed as `molpy.core.Polyhedron`.
///
/// Where the mesh came from is not the region's concern — an STL read with
/// :func:`molpy.io.read_stl`, or a mesh built in memory. Scale the mesh
/// first (``mesh.scaled(s)``) when the file is not in your working unit.
///
/// Examples
/// --------
/// >>> cavity = molpy.core.Polyhedron(molpy.io.read_stl("cavity.stl").scaled(4.18))
/// >>> inside = cavity.contains(points)
#[pyclass(module = "molpy.core", name = "Polyhedron", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyPolyhedron {
    inner: Arc<Polyhedron>,
}

#[pymethods]
impl PyPolyhedron {
    /// Bound a solid by ``mesh``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the mesh is empty, has a non-finite vertex, a zero-area face, or
    ///     is not watertight.
    #[new]
    fn new(mesh: &PyTriMesh) -> PyResult<Self> {
        let inner = Polyhedron::new(mesh.inner.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// The bounding surface (a copy).
    fn mesh(&self) -> PyTriMesh {
        PyTriMesh {
            inner: self.inner.mesh().clone(),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let this = slf.borrow();
        let mesh = this.mesh();
        crate::pickle::reduce_via_type(slf.as_any(), (mesh,))
    }

    fn __repr__(&self) -> String {
        format!("Polyhedron(n_faces={})", self.inner.mesh().n_faces())
    }
}
region_methods!(PyPolyhedron);

// ---------------------------------------------------------------------------
// SphereUnion
// ---------------------------------------------------------------------------

/// Union of spheres, exposed as `molpy.core.SphereUnion` — atoms as a region.
///
/// One sphere per row of ``centers`` with its own radius (a scalar is
/// broadcast). With ``radii = r_vdw + r_probe`` the union is the
/// solvent-accessible volume and ``~SphereUnion(...)`` the space a probe
/// centre can occupy. Pass ``box`` to make the union periodic on the box's
/// periodic axes (minimum image); without it the spheres sit in open space.
///
/// Examples
/// --------
/// >>> polymer = molpy.core.SphereUnion(centers, 0.5 * sigma + 1.0, box=frame.box)
/// >>> void = ~polymer
#[pyclass(module = "molpy.core", name = "SphereUnion", from_py_object, subclass)]
#[derive(Clone)]
pub struct PySphereUnion {
    inner: Arc<SphereUnion>,
}

#[pymethods]
impl PySphereUnion {
    /// Create a union of spheres.
    ///
    /// Parameters
    /// ----------
    /// centers : numpy.ndarray, shape (N, 3), dtype float
    /// radii : float or numpy.ndarray, shape (N,), dtype float
    /// box : molpy.core.Box or None
    ///     Periodicity; ``None`` means open space.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If there are no spheres, the counts disagree, or a radius is not
    ///     positive.
    #[new]
    #[pyo3(signature = (centers, radii, r#box=None))]
    fn new(
        centers: ArrayArg2<'_, f64>,
        radii: &Bound<'_, PyAny>,
        r#box: Option<&PyBox>,
    ) -> PyResult<Self> {
        let c = centers.as_array();
        if c.ncols() != 3 {
            return Err(PyValueError::new_err("centers must have shape (N, 3)"));
        }
        let r: Vec<f64> = if let Ok(scalar) = radii.extract::<f64>() {
            vec![scalar; c.nrows()]
        } else {
            let arr: ArrayArg1<'_, f64> = radii.extract()?;
            arr.as_slice()?.to_vec()
        };
        let inner = match r#box {
            Some(b) => SphereUnion::new(c.view(), &r, &b.inner),
            None => SphereUnion::free(c.view(), &r),
        }
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    /// Number of spheres.
    #[getter]
    fn n_spheres(&self) -> usize {
        self.inner.n_spheres()
    }

    /// Sphere centres as stored (wrapped on periodic axes), shape ``(N, 3)``.
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let cs = self.inner.centers();
        let mut a = Array2::zeros((cs.len(), 3));
        for (i, c) in cs.iter().enumerate() {
            for k in 0..3 {
                a[[i, k]] = c[k];
            }
        }
        a.into_pyarray(py)
    }

    /// Sphere radii, shape ``(N,)``.
    fn radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.radii().to_vec().into_pyarray(py)
    }

    /// The box the union lives in (free when constructed without one).
    #[getter]
    fn r#box(&self) -> PyBox {
        PyBox {
            inner: self.inner.simbox().clone(),
        }
    }

    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let this = slf.borrow();
        crate::pickle::reduce_via_type(
            slf.as_any(),
            (this.centers(py), this.radii(py), this.r#box()),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "SphereUnion(n_spheres={}, periodic={})",
            self.inner.n_spheres(),
            self.inner.simbox().pbc().iter().any(|&p| p)
        )
    }
}
region_methods!(PySphereUnion);

// ---------------------------------------------------------------------------
// Composed Region
// ---------------------------------------------------------------------------

/// Composed region produced by ``&``, ``|``, or ``~`` operators.
///
/// Exposed to Python as `molpy.core.Region`. ``Region(source)`` clones any region
/// object into this class; otherwise instances come from the operators.
///
/// Pickling records the composition as a tree of the operand objects, so a
/// composed region round-trips through whatever its leaves pickle as.
///
/// Examples
/// --------
/// >>> shell = Sphere(c, 5.0) & ~Sphere(c, 3.0)
/// >>> shell.contains(points)
#[pyclass(module = "molpy.core", name = "Region", from_py_object, subclass)]
pub struct PyRegion {
    inner: DynRegion,
    /// ``(op, *operands)`` with the operand region objects, or ``("id", source)``.
    tree: Py<PyAny>,
}

impl Clone for PyRegion {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone(),
            tree: self.tree.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyRegion {
    /// Clone an existing region object into a ``Region``.
    #[new]
    fn new(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = source.py();
        let inner = extract_region(source)?;
        let tree = PyTuple::new(py, ["id".into_pyobject(py)?.into_any(), source.clone()])?;
        Ok(Self {
            inner,
            tree: tree.into_any().unbind(),
        })
    }

    /// Rebuild a composed region from its pickled composition tree.
    #[staticmethod]
    fn _from_tree(tree: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let tag: String = tree.get_item(0)?.extract()?;
        match (tag.as_str(), tree.len()) {
            ("id", 2) => Self::new(&tree.get_item(1)?),
            ("not", 2) => compose("not", &tree.get_item(1)?, None),
            ("and", 3) | ("or", 3) => {
                compose(tag.as_str(), &tree.get_item(1)?, Some(&tree.get_item(2)?))
            }
            _ => Err(PyValueError::new_err("malformed region composition tree")),
        }
    }

    fn __reduce__<'py>(
        slf: &Bound<'py, Self>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.borrow();
        let from_tree = slf.get_type().getattr("_from_tree")?;
        Ok((from_tree, PyTuple::new(py, [this.tree.bind(py).clone()])?))
    }

    fn __repr__(&self) -> String {
        "Region(composed)".to_string()
    }
}
region_methods!(PyRegion);
