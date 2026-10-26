//! `molpy.op` — the pure numeric base (`molrs::op`): weighted superposition,
//! centroids and NeRF placement over `float64` numpy arrays.
//!
//! Registered as a submodule of `_core`, like `md`. Points cross as `(k, 3)`
//! arrays and rotations as `(3, 3)` row-major matrices. A
//! wrong shape is a `ValueError` naming the argument; a
//! [`SuperpositionError`](molrs::op::SuperpositionError) is a `ValueError`
//! carrying the Rust message.

use crate::array::ArrayArgDyn;
use molrs::op;
use molrs::op::Rigid;
use molrs::op::{DEFAULT_GAP_TOL, Freedom, Superposition};
use molrs::op::{Mat3, Vec3};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::error::py_value_err;

// ---------------------------------------------------------------------------
// Array seams shared with the leaf `replicate`
// ---------------------------------------------------------------------------

/// Read an `(n, 3)` float64 array as points, refusing any other shape.
fn points_from_array(array: &ArrayArgDyn<'_, f64>, name: &str) -> PyResult<Vec<Vec3>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3), got {shape:?}"
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// Read an `(n, 3, 3)` float64 array as row-major rotation matrices.
fn matrices_from_array(array: &ArrayArgDyn<'_, f64>, name: &str) -> PyResult<Vec<Mat3>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[1] != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3, 3), got {shape:?}"
        )));
    }
    Ok((0..shape[0])
        .map(|c| std::array::from_fn(|i| std::array::from_fn(|j| view[[c, i, j]])))
        .collect())
}

/// Pair `rotations (N, 3, 3)` with `translations (N, 3)` into rigid motions.
pub(crate) fn rigids_from_arrays(
    rotations: &ArrayArgDyn<'_, f64>,
    translations: &ArrayArgDyn<'_, f64>,
) -> PyResult<Vec<Rigid>> {
    let rotations = matrices_from_array(rotations, "rotations")?;
    let translations = points_from_array(translations, "translations")?;
    if rotations.len() != translations.len() {
        return Err(PyValueError::new_err(format!(
            "rotations and translations disagree in count: {} vs {}",
            rotations.len(),
            translations.len()
        )));
    }
    Ok(rotations
        .into_iter()
        .zip(translations)
        .map(|(rotation, translation)| Rigid {
            rotation,
            translation,
        })
        .collect())
}

fn matrix_to_py<'py>(py: Python<'py>, m: &Mat3) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((3, 3), |(i, j)| m[i][j]).into_pyarray(py)
}

pub(crate) fn vector_to_py<'py>(py: Python<'py>, v: &Vec3) -> Bound<'py, PyArray1<f64>> {
    Array1::from(v.to_vec()).into_pyarray(py)
}

/// Read `weights`, or `k` ones when `None`.
fn weights_or_uniform(weights: Option<ArrayArgDyn<'_, f64>>, k: usize) -> PyResult<Vec<f64>> {
    match weights {
        None => Ok(vec![1.0; k]),
        Some(w) => {
            let view = w.as_array();
            if view.ndim() != 1 {
                return Err(PyValueError::new_err(format!(
                    "weights must have shape (k,), got {:?}",
                    view.shape()
                )));
            }
            Ok(view.iter().copied().collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Superposition
// ---------------------------------------------------------------------------

/// The best-fit proper rigid motion ``target ≈ rotation @ reference + translation``
/// returned by :func:`molpy.op.superpose`. Frozen.
///
/// Attributes
/// ----------
/// rotation : ndarray, shape (3, 3), float64
///     A proper rotation (orthogonal, determinant +1), row-major.
/// translation : ndarray, shape (3,), float64
///     In the coordinates' length unit (Å in molrs), applied after the
///     rotation.
/// rmsd : float
///     Weighted root-mean-square deviation of the fit, in the coordinates'
///     length unit (Å).
/// rho : float
///     Scale-free eigen-gap (dimensionless); 0 when ``freedom == "free"``.
/// center : ndarray, shape (3,), float64
///     Weighted target centroid in Å (the point a spin axis passes through).
/// freedom : str
///     ``"unique"``, ``"spin"`` (rotation about ``axis`` is undetermined) or
///     ``"free"`` (no rotation determined; ``rotation`` is the identity).
/// axis : ndarray, shape (3,), float64, or None
///     The unit spin axis; ``None`` unless ``freedom == "spin"``.
#[pyclass(
    module = "molpy.op",
    name = "Superposition",
    frozen,
    skip_from_py_object
)]
pub struct PySuperposition {
    pub(crate) inner: Superposition,
}

impl PySuperposition {
    pub(crate) fn from_core(inner: Superposition) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySuperposition {
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_py(py, &self.inner.rigid.rotation)
    }

    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vector_to_py(py, &self.inner.rigid.translation)
    }

    #[getter]
    fn rmsd(&self) -> f64 {
        self.inner.rmsd
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vector_to_py(py, &self.inner.center)
    }

    #[getter]
    fn freedom(&self) -> &'static str {
        match self.inner.freedom {
            Freedom::Unique => "unique",
            Freedom::Spin { .. } => "spin",
            Freedom::Free => "free",
        }
    }

    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        match self.inner.freedom {
            Freedom::Spin { axis } => Some(vector_to_py(py, &axis)),
            Freedom::Unique | Freedom::Free => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Superposition(freedom='{}', rmsd={}, rho={})",
            self.freedom(),
            self.inner.rmsd,
            self.inner.rho
        )
    }
}

// ---------------------------------------------------------------------------
// Free operations
// ---------------------------------------------------------------------------

/// Best-fit proper rigid motion mapping ``reference`` onto ``target``.
///
/// Parameters
/// ----------
/// reference, target : ndarray, shape (k, 3), float64
/// weights : ndarray, shape (k,), float64, optional
///     Per-point weights; uniform when omitted. Zero-weight points are dropped.
/// gap_tol : float, keyword-only
///     Eigen-gap below which the rotation is reported ``"spin"``;
///     :data:`DEFAULT_GAP_TOL` by default.
///
/// Returns
/// -------
/// Superposition
///
/// Raises
/// ------
/// ValueError
///     On a shape other than ``(k, 3)``, a length mismatch, a negative or
///     non-finite weight, a non-finite coordinate, or no positive weight.
#[pyfunction(name = "superpose")]
#[pyo3(signature = (reference, target, weights=None, *, gap_tol=DEFAULT_GAP_TOL))]
fn py_superpose(
    reference: ArrayArgDyn<'_, f64>,
    target: ArrayArgDyn<'_, f64>,
    weights: Option<ArrayArgDyn<'_, f64>>,
    gap_tol: f64,
) -> PyResult<PySuperposition> {
    let reference = points_from_array(&reference, "reference")?;
    let target = points_from_array(&target, "target")?;
    let weights = weights_or_uniform(weights, reference.len())?;
    op::superpose(&reference, &target, &weights, gap_tol)
        .map(PySuperposition::from_core)
        .map_err(py_value_err)
}

/// Weighted centroid ``Σ wᵢ pᵢ / Σ wᵢ``.
///
/// Parameters
/// ----------
/// points : ndarray, shape (k, 3), float64
/// weights : ndarray, shape (k,), float64, optional
///     Uniform when omitted.
///
/// Returns
/// -------
/// ndarray, shape (3,), float64, or None
///     The centroid in the length unit of ``points`` (Å in molrs); ``None``
///     when the lengths differ or the total weight is not positive and
///     finite.
///
/// Raises
/// ------
/// ValueError
///     If ``points`` is not shape ``(k, 3)`` or ``weights`` is not 1-D.
#[pyfunction(name = "centroid")]
#[pyo3(signature = (points, weights=None))]
fn py_centroid<'py>(
    py: Python<'py>,
    points: ArrayArgDyn<'_, f64>,
    weights: Option<ArrayArgDyn<'_, f64>>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let points = points_from_array(&points, "points")?;
    let weights = weights_or_uniform(weights, points.len())?;
    Ok(op::centroid(&points, &weights).map(|c| vector_to_py(py, &c)))
}

/// The point ``d`` at distance ``bond`` from ``c`` that makes the angle
/// ``angle`` at ``c`` with ``b`` (∠b–c–d) and the dihedral ``torsion`` about
/// ``b → c`` with ``a`` (a–b–c–d): the natural-extension reference frame
/// (NeRF; Parsons et al., *J. Comput. Chem.* **26** (2005) 1063).
///
/// The inverse of the bond angle and the IUPAC dihedral: for the returned
/// ``d`` they are ``angle`` and ``torsion`` to rounding. Collinear
/// ``a, b, c`` leave the dihedral plane undefined; every off-axis component
/// then vanishes.
///
/// Parameters
/// ----------
/// a, b, c : array_like, shape (3,), float64
///     The three placed points, in any one length unit.
/// bond : float
///     Distance ``|d − c|``, in the points' length unit.
/// angle, torsion : float
///     In radians (``molpy.core.UnitRegistry().factor("deg", "rad")`` takes
///     degrees there).
///
/// Returns
/// -------
/// ndarray, shape (3,), float64
#[pyfunction(name = "place_from_internal_coords")]
fn py_place_from_internal_coords<'py>(
    py: Python<'py>,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    bond: f64,
    angle: f64,
    torsion: f64,
) -> Bound<'py, PyArray1<f64>> {
    vector_to_py(
        py,
        &op::place_from_internal_coords(a, b, c, bond, angle, torsion),
    )
}

/// Populate the `molpy.op` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySuperposition>()?;
    m.add("DEFAULT_GAP_TOL", DEFAULT_GAP_TOL)?;
    m.add_function(wrap_pyfunction!(py_superpose, m)?)?;
    m.add_function(wrap_pyfunction!(py_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(py_place_from_internal_coords, m)?)?;
    Ok(())
}
