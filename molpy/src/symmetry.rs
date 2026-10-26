//! `molpy.builder.SymOp` / `SpaceGroup` — molrs `core::{SymOp, SpaceGroup}`.

use molrs::core::{SpaceGroup, SymOp};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::core::frame::PyFrame;

/// One space-group operator ``x' = R x + t`` on fractional coordinates.
#[pyclass(module = "molpy.builder", name = "SymOp", frozen, skip_from_py_object)]
pub struct PySymOp {
    inner: SymOp,
}

#[pymethods]
impl PySymOp {
    /// Parse a Jones-faithful triplet (``"-y+1/2, x, z+1/4"``, the form CIFs
    /// use); each translation is reduced into ``[0, 1)``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A triplet without three components, or a term that is no
    ///     coordinate and no number.
    #[staticmethod]
    fn parse(triplet: &str) -> PyResult<Self> {
        Ok(Self {
            inner: SymOp::parse(triplet).map_err(PyValueError::new_err)?,
        })
    }

    /// The ``(3, 3)`` rotation / reflection.
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        ndarray::Array2::from_shape_fn((3, 3), |(i, j)| self.inner.rotation[i][j]).into_pyarray(py)
    }

    /// The ``(3,)`` translation, each component in ``[0, 1)``.
    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.translation.to_vec().into_pyarray(py)
    }
}

/// A space group as its list of affine operators (molrs ``SpaceGroup``).
#[pyclass(
    module = "molpy.builder",
    name = "SpaceGroup",
    frozen,
    skip_from_py_object
)]
pub struct PySpaceGroup {
    pub(crate) inner: SpaceGroup,
}

#[pymethods]
impl PySpaceGroup {
    /// The group of an explicit, complete operator list, taken verbatim.
    #[staticmethod]
    fn from_triplets(triplets: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            inner: SpaceGroup::from_triplets(triplets.iter().map(String::as_str))
                .map_err(PyValueError::new_err)?,
        })
    }

    /// The group closed from ``generators`` (the identity added when absent).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A triplet that does not parse, or a set that does not close.
    #[staticmethod]
    fn from_generators(generators: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            inner: SpaceGroup::from_generators(generators.iter().map(String::as_str))
                .map_err(PyValueError::new_err)?,
        })
    }

    /// The group a CIF frame's ``symops`` meta states (``mp.io.read_cif``
    /// keeps the file's operator loop there).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A frame without ``symops`` (a CIF without an operator loop is not
    ///     taken to be P1), or a triplet that does not parse.
    #[staticmethod]
    fn from_frame(frame: &PyFrame) -> PyResult<Self> {
        let inner = frame.with_frame(SpaceGroup::from_frame)?;
        Ok(Self {
            inner: inner.map_err(PyValueError::new_err)?,
        })
    }

    /// Number of operators.
    #[getter]
    fn order(&self) -> usize {
        self.inner.order()
    }

    /// The operators, in group order.
    #[getter]
    fn operators(&self) -> Vec<PySymOp> {
        self.inner
            .operators()
            .iter()
            .map(|&inner| PySymOp { inner })
            .collect()
    }

    /// Every symmetry image of fractional site ``frac`` in ``[0, 1)``,
    /// de-duplicated within ``symprec`` under the periodic boundary: an
    /// ``(M, 3)`` array, ``M`` the site's multiplicity.
    #[pyo3(signature = (frac, *, symprec=1e-5))]
    fn equivalent_positions<'py>(
        &self,
        py: Python<'py>,
        frac: [f64; 3],
        symprec: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let images = self.inner.equivalent_positions(frac, symprec);
        ndarray::Array2::from_shape_fn((images.len(), 3), |(i, k)| images[i][k]).into_pyarray(py)
    }
}

/// Register the symmetry classes on `molpy.builder`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySymOp>()?;
    m.add_class::<PySpaceGroup>()?;
    Ok(())
}
