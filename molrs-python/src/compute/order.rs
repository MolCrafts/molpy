//! Order parameters (`molrs::compute::order`): the bond-average table
//! `QlmTable`, `Steinhardt`, `Hexatic`, `SolidLiquid`,
//! `ContinuousCoordination`, `Nematic` and `Cubatic`.

use molrs::compute::{
    Compute, ContinuousCoordination, ContinuousCoordinationInput, ContinuousCoordinationResult,
    Cubatic, CubaticInput, CubaticResult, Hexatic, HexaticInput, HexaticResult, Nematic,
    NematicInput, NematicResult, QlmTable, SolidLiquid, SolidLiquidInput, SolidLiquidResult,
    Steinhardt, SteinhardtInput, SteinhardtResult,
};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use super::error::Result;
use super::output::{ToPy, complex_array, list, py_result};
use super::spec::described;
use crate::core::neighborlist::PyNeighbors;

/// Per-particle bond averages ``q_lm`` (or planar ``ψ_k``) of one order
/// (``molrs::compute::QlmTable``).
///
/// Read from ``SteinhardtResult.qlm(l)``; the input of ``SolidLiquid`` and
/// ``ContinuousCoordination``.
#[pyclass(module = "molrs.compute", name = "QlmTable", frozen)]
pub(crate) struct PyQlmTable {
    pub(crate) inner: QlmTable,
}

impl ToPy for QlmTable {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(Bound::new(
            py,
            PyQlmTable {
                inner: self.clone(),
            },
        )?
        .into_any())
    }
}

impl ToPy for Vec<QlmTable> {
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        list(py, self)
    }
}

#[pymethods]
impl PyQlmTable {
    /// The order ``l`` (or ``k`` for a planar table).
    #[getter]
    fn order(&self) -> u32 {
        self.inner.order()
    }

    #[getter]
    fn is_planar(&self) -> bool {
        self.inner.is_planar()
    }

    #[getter]
    fn n_particles(&self) -> usize {
        self.inner.n_particles()
    }

    /// The ``(n_particles, n_m)`` complex128 table.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        complex_array(py, self.inner.values())
    }

    /// The per-particle norm of the ``m`` vector.
    fn norms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.norms().to_py(py)
    }

    /// The rotation-invariant ``q_l`` per particle.
    fn ql<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.ql().to_py(py)
    }

    /// The third-order invariant ``w_l`` per particle.
    fn wl<'py>(&self, py: Python<'py>, normalize: bool) -> Result<Bound<'py, PyAny>> {
        Ok(self.inner.wl(normalize)?.to_py(py)?)
    }

    /// The per-particle bond coherence over ``neighbors``.
    fn bond_coherence<'py>(
        &self,
        py: Python<'py>,
        neighbors: PyRef<'py, PyNeighbors>,
    ) -> Result<Bound<'py, PyAny>> {
        Ok(self.inner.bond_coherence(&neighbors.inner)?.to_py(py)?)
    }

    fn __repr__(&self) -> String {
        format!(
            "QlmTable(order={}, n_particles={})",
            self.inner.order(),
            self.inner.n_particles()
        )
    }
}

#[pymethods]
impl PySteinhardtResult {
    /// One ``QlmTable`` per order, aligned with ``l``.
    #[getter]
    fn tables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.tables.to_py(py)
    }

    /// ``q_l`` per particle and order, ``(n_particles, len(l))``.
    #[getter]
    fn ql<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.ql().to_owned().to_py(py)
    }

    /// ``w_l`` per particle and order, or ``None`` when not computed.
    #[getter]
    fn wl<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.wl().map(|wl| wl.to_owned()).to_py(py)
    }

    /// The table of order ``l``, or ``None``.
    fn qlm(&self, l: u32) -> Option<PyQlmTable> {
        self.inner.qlm(l).map(|table| PyQlmTable {
            inner: table.clone(),
        })
    }
}

py_result! {
    /// Steinhardt order parameters (``molrs::compute::SteinhardtResult``); a
    /// feature matrix.
    PySteinhardtResult(SteinhardtResult) as "SteinhardtResult" { l }
}

py_result! {
    /// The k-atic order ``ψ_k`` per particle (``molrs::compute::HexaticResult``).
    PyHexaticResult(HexaticResult) as "HexaticResult" { k, psi }
}

py_result! {
    /// Solid-like bonds and particles (``molrs::compute::SolidLiquidResult``).
    PySolidLiquidResult(SolidLiquidResult) as "SolidLiquidResult" { l, n_solid_bonds, is_solid }
}

py_result! {
    /// Continuous coordination per particle
    /// (``molrs::compute::ContinuousCoordinationResult``).
    PyContinuousCoordinationResult(ContinuousCoordinationResult) as "ContinuousCoordinationResult" {
        l, coord
    }
}

py_result! {
    /// The nematic order tensor and director (``molrs::compute::NematicResult``).
    PyNematicResult(NematicResult) as "NematicResult" { order, eigenvalues, director, q_tensor }
}

py_result! {
    /// The cubatic order and its director basis (``molrs::compute::CubaticResult``).
    PyCubaticResult(CubaticResult) as "CubaticResult" { order, director_basis }
}

/// Steinhardt bond-orientational order (``molrs::compute::Steinhardt``).
///
/// Parameters
/// ----------
/// l : list[int]
///     The spherical-harmonic orders.
/// average, wl, wl_normalize : bool, optional
///     Neighbour-averaged ``q_l``; also ``w_l``; normalize ``w_l``.
///
/// ``compute(neighbors=)`` over a half-shell self-query ``Neighbors``.
#[pyclass(module = "molrs.compute", name = "Steinhardt", frozen)]
pub(crate) struct PySteinhardt {
    inner: Steinhardt,
}

#[pymethods]
impl PySteinhardt {
    #[new]
    #[pyo3(signature = (l, *, average = None, wl = None, wl_normalize = None))]
    fn new(
        l: Vec<u32>,
        average: Option<bool>,
        wl: Option<bool>,
        wl_normalize: Option<bool>,
    ) -> Result<Self> {
        let mut inner = Steinhardt::new(&l)?;
        if let Some(average) = average {
            inner = inner.with_average(average);
        }
        if let Some(wl) = wl {
            inner = inner.with_wl(wl);
        }
        if let Some(wl_normalize) = wl_normalize {
            inner = inner.with_wl_normalize(wl_normalize);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, neighbors))]
    fn compute(&self, neighbors: PyRef<'_, PyNeighbors>) -> Result<PySteinhardtResult> {
        let out = self.inner.compute(SteinhardtInput {
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PySteinhardt, Steinhardt);

/// The k-atic bond order ``ψ_k`` of a 2-D system (``molrs::compute::Hexatic``).
///
/// ``compute(neighbors=)``.
#[pyclass(module = "molrs.compute", name = "Hexatic", frozen)]
pub(crate) struct PyHexatic {
    inner: Hexatic,
}

#[pymethods]
impl PyHexatic {
    #[new]
    fn new(k: u32) -> Result<Self> {
        Ok(Self {
            inner: Hexatic::new(k)?,
        })
    }

    #[pyo3(signature = (*, neighbors))]
    fn compute(&self, neighbors: PyRef<'_, PyNeighbors>) -> Result<PyHexaticResult> {
        let out = self.inner.compute(HexaticInput {
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyHexatic, Hexatic);

/// Solid-liquid classification by bond coherence
/// (``molrs::compute::SolidLiquid``).
///
/// Parameters
/// ----------
/// q_threshold : float, optional
/// n_threshold : int, optional
/// normalize_q : bool, optional
///
/// ``compute(qlm=, neighbors=)`` with a ``QlmTable``.
#[pyclass(module = "molrs.compute", name = "SolidLiquid", frozen)]
pub(crate) struct PySolidLiquid {
    inner: SolidLiquid,
}

#[pymethods]
impl PySolidLiquid {
    #[new]
    #[pyo3(signature = (*, q_threshold = None, n_threshold = None, normalize_q = None))]
    fn new(
        q_threshold: Option<f64>,
        n_threshold: Option<u32>,
        normalize_q: Option<bool>,
    ) -> Result<Self> {
        let mut inner = SolidLiquid::new();
        if let Some(q_threshold) = q_threshold {
            inner = inner.with_q_threshold(q_threshold)?;
        }
        if let Some(n_threshold) = n_threshold {
            inner = inner.with_n_threshold(n_threshold);
        }
        if let Some(normalize_q) = normalize_q {
            inner = inner.with_normalize_q(normalize_q);
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, qlm, neighbors))]
    fn compute(
        &self,
        qlm: PyRef<'_, PyQlmTable>,
        neighbors: PyRef<'_, PyNeighbors>,
    ) -> Result<PySolidLiquidResult> {
        let out = self.inner.compute(SolidLiquidInput {
            qlm: &qlm.inner,
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PySolidLiquid, SolidLiquid);

/// Continuous coordination number (``molrs::compute::ContinuousCoordination``).
///
/// ``compute(qlm=, neighbors=)`` with a ``QlmTable``.
#[pyclass(module = "molrs.compute", name = "ContinuousCoordination", frozen)]
pub(crate) struct PyContinuousCoordination {
    inner: ContinuousCoordination,
}

#[pymethods]
impl PyContinuousCoordination {
    #[new]
    fn new(power: f64) -> Result<Self> {
        Ok(Self {
            inner: ContinuousCoordination::new(power)?,
        })
    }

    #[pyo3(signature = (*, qlm, neighbors))]
    fn compute(
        &self,
        qlm: PyRef<'_, PyQlmTable>,
        neighbors: PyRef<'_, PyNeighbors>,
    ) -> Result<PyContinuousCoordinationResult> {
        let out = self.inner.compute(ContinuousCoordinationInput {
            qlm: &qlm.inner,
            neighbors: &neighbors.inner,
        })?;
        Ok(out.into())
    }
}

described!(PyContinuousCoordination, ContinuousCoordination);

/// Nematic order of ``(N, 3)`` directors (``molrs::compute::Nematic``).
///
/// ``compute(directors=)``.
#[pyclass(module = "molrs.compute", name = "Nematic", frozen)]
pub(crate) struct PyNematic {
    inner: Nematic,
}

#[pymethods]
impl PyNematic {
    #[new]
    fn new() -> Self {
        Self {
            inner: Nematic::new(),
        }
    }

    #[pyo3(signature = (*, directors))]
    fn compute(&self, directors: PyReadonlyArray2<'_, f64>) -> Result<PyNematicResult> {
        let out = self.inner.compute(NematicInput {
            directors: directors.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyNematic, Nematic);

/// Cubatic order of ``(N, 3)`` directors by simulated annealing
/// (``molrs::compute::Cubatic``).
///
/// Parameters
/// ----------
/// seed, n_steps, n_chains : int, optional
/// initial_temp, cooling_rate, min_temp : float, optional
///
/// ``compute(directors=)``.
#[pyclass(module = "molrs.compute", name = "Cubatic", frozen)]
pub(crate) struct PyCubatic {
    inner: Cubatic,
}

#[pymethods]
impl PyCubatic {
    #[new]
    #[pyo3(signature = (
        *,
        seed = None,
        initial_temp = None,
        cooling_rate = None,
        min_temp = None,
        n_steps = None,
        n_chains = None,
    ))]
    fn new(
        seed: Option<u64>,
        initial_temp: Option<f64>,
        cooling_rate: Option<f64>,
        min_temp: Option<f64>,
        n_steps: Option<usize>,
        n_chains: Option<usize>,
    ) -> Result<Self> {
        let mut inner = Cubatic::new();
        if let Some(seed) = seed {
            inner = inner.with_seed(seed);
        }
        if let Some(initial_temp) = initial_temp {
            inner = inner.with_initial_temp(initial_temp)?;
        }
        if let Some(cooling_rate) = cooling_rate {
            inner = inner.with_cooling_rate(cooling_rate)?;
        }
        if let Some(min_temp) = min_temp {
            inner = inner.with_min_temp(min_temp)?;
        }
        if let Some(n_steps) = n_steps {
            inner = inner.with_n_steps(n_steps)?;
        }
        if let Some(n_chains) = n_chains {
            inner = inner.with_n_chains(n_chains)?;
        }
        Ok(Self { inner })
    }

    #[pyo3(signature = (*, directors))]
    fn compute(&self, directors: PyReadonlyArray2<'_, f64>) -> Result<PyCubaticResult> {
        let out = self.inner.compute(CubaticInput {
            directors: directors.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyCubatic, Cubatic);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyQlmTable>()?;
    m.add_class::<PySteinhardtResult>()?;
    m.add_class::<PyHexaticResult>()?;
    m.add_class::<PySolidLiquidResult>()?;
    m.add_class::<PyContinuousCoordinationResult>()?;
    m.add_class::<PyNematicResult>()?;
    m.add_class::<PyCubaticResult>()?;
    m.add_class::<PySteinhardt>()?;
    m.add_class::<PyHexatic>()?;
    m.add_class::<PySolidLiquid>()?;
    m.add_class::<PyContinuousCoordination>()?;
    m.add_class::<PyNematic>()?;
    m.add_class::<PyCubatic>()?;
    Ok(())
}
