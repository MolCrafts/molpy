//! Voronoi analyses (`molrs::compute::voronoi`): the radical (power)
//! tessellation `RadicalVoronoi`, the density integration
//! `VoronoiIntegration`, the cell analyses `VoronoiDomainAnalysis` and
//! `VoronoiVoidAnalysis`, and `FiniteFieldPolarizability`; their outputs
//! `VoronoiCells`, `VoronoiFace`, `MolecularMoments`, `VoronoiDomainResult`
//! and `VoronoiVoidResult`.

use crate::array::{ArrayArg1, ArrayArg2, ArrayArg3};
use molrs::compute::{
    Compute, FiniteFieldPolarizability, FiniteFieldPolarizabilityInput, MolecularMoments,
    RadicalVoronoi, RadicalVoronoiInput, VoronoiCells, VoronoiDomainAnalysis, VoronoiDomainInput,
    VoronoiDomainResult, VoronoiFace, VoronoiIntegration, VoronoiIntegrationInput,
    VoronoiVoidAnalysis, VoronoiVoidInput, VoronoiVoidResult,
};
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::{ToPy, list, py_result};
use super::spec::described;
use super::values::PyRegularGrid;
use crate::core::simbox::PyBox;

py_result! {
    /// One face of a Voronoi cell (``molrs::compute::VoronoiFace``): its
    /// ``area`` and the ``neighbor`` across it (negative for a boundary).
    PyVoronoiFace(VoronoiFace) as "VoronoiFace" { area, neighbor }
}

impl ToPy for Vec<VoronoiFace> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        list(py, self)
    }
}

impl ToPy for Vec<Vec<VoronoiFace>> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        list(py, self)
    }
}

py_result! {
    /// A tessellation (``molrs::compute::VoronoiCells``): the ``volumes`` of
    /// the cells and the ``faces`` of each.
    PyVoronoiCells(VoronoiCells) as "VoronoiCells" { volumes, faces }
}

#[pymethods]
impl PyVoronoiCells {
    /// The summed volume of every cell.
    fn total_volume(&self) -> f64 {
        self.inner.total_volume()
    }

    /// The neighbours of cell ``i`` across its faces.
    fn neighbors(&self, i: usize) -> PyResult<Vec<i64>> {
        if i >= self.inner.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "cell {i} out of range for {} cells",
                self.inner.len()
            )));
        }
        Ok(self.inner.neighbors(i))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
}

py_result! {
    /// Per-group moments from an integrated density
    /// (``molrs::compute::MolecularMoments``): the ``charges``, ``dipoles``
    /// and ``references`` of each group key in ``groups``.
    PyMolecularMoments(MolecularMoments) as "MolecularMoments" {
        groups, charges, dipoles, references
    }
}

py_result! {
    /// Voronoi domains (``molrs::compute::VoronoiDomainResult``): the ``sizes``
    /// of the connected domains, their ``count``, the ``largest_fraction``
    /// and each cell's domain (``None`` outside every domain). A partition.
    PyVoronoiDomainResult(VoronoiDomainResult) as "VoronoiDomainResult" {
        sizes, count, largest_fraction, domain_of
    }
}

py_result! {
    /// Voids (``molrs::compute::VoronoiVoidResult``): the cavity volumes
    /// (descending), their total and the void fraction of the box.
    PyVoronoiVoidResult(VoronoiVoidResult) as "VoronoiVoidResult" {
        cavity_volumes, total_void_volume, void_fraction
    }
}

/// The radical (power) Voronoi tessellation of one snapshot
/// (``molrs::compute::RadicalVoronoi``).
///
/// ``compute(positions=, simbox=, radii=)``: ``(N, 3)`` positions, the box
/// and ``(N,)`` radii. Returns ``VoronoiCells``.
#[pyclass(module = "molpy.compute", name = "RadicalVoronoi", frozen)]
pub(crate) struct PyRadicalVoronoi {
    inner: RadicalVoronoi,
}

#[pymethods]
impl PyRadicalVoronoi {
    #[new]
    fn new() -> Self {
        Self {
            inner: RadicalVoronoi::new(),
        }
    }

    #[pyo3(signature = (*, positions, simbox, radii))]
    fn compute(
        &self,
        positions: ArrayArg2<'_, f64>,
        simbox: PyRef<'_, PyBox>,
        radii: ArrayArg1<'_, f64>,
    ) -> Result<PyVoronoiCells> {
        let out = self.inner.compute(RadicalVoronoiInput {
            positions: positions.as_array(),
            simbox: &simbox.inner,
            radii: radii.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyRadicalVoronoi, RadicalVoronoi);

/// Integrate a density over the radical Voronoi cells of the atoms and sum
/// the cells of each group (``molrs::compute::VoronoiIntegration``).
///
/// Parameters
/// ----------
/// unit : str
///     The length unit of the positions and grid: ``"angstrom"`` or
///     ``"bohr"``.
///
/// ``compute(positions=, radii=, nuclear_charges=, partition=, simbox=,
/// grid=, density=)``: ``(N, 3)`` positions, ``(N,)`` radii and nuclear
/// charges, a partition into molecules, the box, the ``RegularGrid`` the
/// density lives on and the ``(nx, ny, nz)`` density. Returns
/// ``MolecularMoments``.
#[pyclass(module = "molpy.compute", name = "VoronoiIntegration", frozen)]
pub(crate) struct PyVoronoiIntegration {
    inner: VoronoiIntegration,
}

#[pymethods]
impl PyVoronoiIntegration {
    #[new]
    fn new(unit: &str) -> Result<Self> {
        Ok(Self {
            inner: VoronoiIntegration::new(unit.parse()?),
        })
    }

    #[allow(clippy::too_many_arguments, reason = "the core input's fields")]
    #[pyo3(signature = (*, positions, radii, nuclear_charges, partition, simbox, grid, density))]
    fn compute<'py>(
        &self,
        positions: ArrayArg2<'py, f64>,
        radii: ArrayArg1<'py, f64>,
        nuclear_charges: ArrayArg1<'py, f64>,
        partition: &Bound<'py, PyAny>,
        simbox: PyRef<'py, PyBox>,
        grid: PyRef<'py, PyRegularGrid>,
        density: ArrayArg3<'py, f64>,
    ) -> Result<PyMolecularMoments> {
        let partition = args::PartitionArg::extract(partition)?;
        let out = self.inner.compute(VoronoiIntegrationInput {
            positions: positions.as_array(),
            radii: radii.as_array(),
            nuclear_charges: nuclear_charges.as_array(),
            partition: partition.partition(),
            simbox: &simbox.inner,
            grid: &grid.inner,
            density: density.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyVoronoiIntegration, VoronoiIntegration);

/// Connected domains of the cells a partition selects
/// (``molrs::compute::VoronoiDomainAnalysis``).
///
/// ``compute(cells=, partition=)``: the ``VoronoiCells`` and a partition of
/// the cells (an unlabelled cell is in no domain). Returns a
/// ``VoronoiDomainResult``.
#[pyclass(module = "molpy.compute", name = "VoronoiDomainAnalysis", frozen)]
pub(crate) struct PyVoronoiDomainAnalysis {
    inner: VoronoiDomainAnalysis,
}

#[pymethods]
impl PyVoronoiDomainAnalysis {
    #[new]
    fn new() -> Self {
        Self {
            inner: VoronoiDomainAnalysis::new(),
        }
    }

    #[pyo3(signature = (*, cells, partition))]
    fn compute<'py>(
        &self,
        cells: PyRef<'py, PyVoronoiCells>,
        partition: &Bound<'py, PyAny>,
    ) -> Result<PyVoronoiDomainResult> {
        let partition = args::PartitionArg::extract(partition)?;
        let out = self.inner.compute(VoronoiDomainInput {
            cells: &cells.inner,
            partition: partition.partition(),
        })?;
        Ok(out.into())
    }
}

described!(PyVoronoiDomainAnalysis, VoronoiDomainAnalysis);

/// Cavities formed by connected void cells
/// (``molrs::compute::VoronoiVoidAnalysis``).
///
/// ``compute(cells=, is_void=, box_volume=)``: the ``VoronoiCells``, a
/// per-cell flag marking the void (probe) cells and the box volume. Returns a
/// ``VoronoiVoidResult``.
#[pyclass(module = "molpy.compute", name = "VoronoiVoidAnalysis", frozen)]
pub(crate) struct PyVoronoiVoidAnalysis {
    inner: VoronoiVoidAnalysis,
}

#[pymethods]
impl PyVoronoiVoidAnalysis {
    #[new]
    fn new() -> Self {
        Self {
            inner: VoronoiVoidAnalysis::new(),
        }
    }

    #[pyo3(signature = (*, cells, is_void, box_volume))]
    fn compute(
        &self,
        cells: PyRef<'_, PyVoronoiCells>,
        is_void: Vec<bool>,
        box_volume: f64,
    ) -> Result<PyVoronoiVoidResult> {
        let out = self.inner.compute(VoronoiVoidInput {
            cells: &cells.inner,
            is_void: &is_void,
            box_volume,
        })?;
        Ok(out.into())
    }
}

described!(PyVoronoiVoidAnalysis, VoronoiVoidAnalysis);

/// Polarizability by central finite differences of the group dipoles under
/// ``±field`` (``molrs::compute::FiniteFieldPolarizability``).
///
/// Parameters
/// ----------
/// field : float
///     The applied field strength.
///
/// ``compute(plus=, minus=)``: the ``MolecularMoments`` under ``+field`` and
/// ``-field``. Returns the per-group polarizability array.
#[pyclass(module = "molpy.compute", name = "FiniteFieldPolarizability", frozen)]
pub(crate) struct PyFiniteFieldPolarizability {
    inner: FiniteFieldPolarizability,
}

#[pymethods]
impl PyFiniteFieldPolarizability {
    #[new]
    fn new(field: f64) -> Result<Self> {
        Ok(Self {
            inner: FiniteFieldPolarizability::new(field)?,
        })
    }

    #[pyo3(signature = (*, plus, minus))]
    fn compute<'py>(
        &self,
        py: Python<'py>,
        plus: PyRef<'py, PyMolecularMoments>,
        minus: PyRef<'py, PyMolecularMoments>,
    ) -> Result<Bound<'py, PyArray2<f64>>> {
        let out = self.inner.compute(FiniteFieldPolarizabilityInput {
            plus: &plus.inner,
            minus: &minus.inner,
        })?;
        Ok(out.into_pyarray(py))
    }
}

described!(PyFiniteFieldPolarizability, FiniteFieldPolarizability);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVoronoiFace>()?;
    m.add_class::<PyVoronoiCells>()?;
    m.add_class::<PyMolecularMoments>()?;
    m.add_class::<PyVoronoiDomainResult>()?;
    m.add_class::<PyVoronoiVoidResult>()?;
    m.add_class::<PyRadicalVoronoi>()?;
    m.add_class::<PyVoronoiIntegration>()?;
    m.add_class::<PyVoronoiDomainAnalysis>()?;
    m.add_class::<PyVoronoiVoidAnalysis>()?;
    m.add_class::<PyFiniteFieldPolarizability>()?;
    Ok(())
}
