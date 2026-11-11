//! Percolation of a bonded network under periodic boundaries
//! (`molrs::compute::Percolation`, Livraghi et al. 2021).

use molrs::compute::{Compute, Percolation, PercolationInput, PercolationResult};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::spec::described;
use crate::array::ArrayArg2;
use crate::core::simbox::PyBox;

/// How a bonded network percolates (``molrs::compute::PercolationResult``).
///
/// ``component_idx[i]`` is atom ``i``'s connected component;
/// ``percolation_dims[c]`` is component ``c``'s percolation dimension (0 finite,
/// 1 strand, 2 sheet, 3 spanning the box in every direction) and
/// ``translations[c]`` the basis of its translation lattice, in lattice-vector
/// units. ``unwrap_images`` (``N × 3``) makes every non-percolating component
/// whole when ``H · image`` is added to the positions.
#[pyclass(module = "molpy.compute", name = "PercolationResult", frozen)]
pub(crate) struct PyPercolationResult {
    inner: PercolationResult,
}

impl From<PercolationResult> for PyPercolationResult {
    fn from(inner: PercolationResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPercolationResult {
    #[getter]
    fn component_idx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        self.inner.component_idx.clone().into_pyarray(py)
    }

    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components
    }

    #[getter]
    fn component_sizes(&self) -> Vec<usize> {
        self.inner.component_sizes.clone()
    }

    #[getter]
    fn percolation_dims(&self) -> Vec<usize> {
        self.inner.percolation_dims.clone()
    }

    #[getter]
    fn translations(&self) -> Vec<Vec<(i64, i64, i64)>> {
        self.inner
            .translations
            .iter()
            .map(|basis| basis.iter().map(|t| (t[0], t[1], t[2])).collect())
            .collect()
    }

    #[getter]
    fn unwrap_images<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        self.inner.unwrap_images.clone().into_pyarray(py)
    }

    /// The largest percolation dimension over all components.
    fn max_dim(&self) -> usize {
        self.inner.max_dim()
    }

    fn __repr__(&self) -> String {
        format!(
            "PercolationResult(n_components={}, percolation_dims={:?})",
            self.inner.n_components, self.inner.percolation_dims
        )
    }
}

/// Percolation of a bonded network in a periodic cell (Livraghi et al., J.
/// Chem. Theory Comput. 17, 6449 (2021)).
///
/// ``compute(positions=, simbox=, bonds=)`` reads every bond by minimum image (as
/// LAMMPS does), walks each connected component and reduces the translations
/// of its closed walks to a lattice basis; the basis rank is the percolation
/// dimension. A bond as long as half the nearest plane distance is refused:
/// its image is ambiguous. Returns a ``PercolationResult``.
#[pyclass(module = "molpy.compute", name = "Percolation", frozen)]
pub(crate) struct PyPercolation {
    inner: Percolation,
}

#[pymethods]
impl PyPercolation {
    #[new]
    fn new() -> Self {
        Self {
            inner: Percolation::new(),
        }
    }

    #[pyo3(signature = (*, positions, simbox, bonds))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        positions: ArrayArg2<'py, f64>,
        simbox: PyRef<'py, PyBox>,
        bonds: Vec<Vec<usize>>,
    ) -> Result<PyPercolationResult> {
        let table = args::tuples(bonds, "Percolation bonds")?;
        let pairs: Vec<[usize; 2]> = table
            .rows()
            .into_iter()
            .map(|row| [row[0], row[1]])
            .collect();
        let out = {
            let input = PercolationInput {
                positions: positions.as_array(),
                simbox: &simbox.inner,
                bonds: &pairs,
            };
            py.detach(|| self.inner.compute(input))
        }?;
        Ok(out.into())
    }
}

described!(PyPercolation, Percolation);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPercolationResult>()?;
    m.add_class::<PyPercolation>()?;
    Ok(())
}
