//! Group shape (`molrs::compute::shape`): `CenterOfMass`, `GyrationTensor`
//! and `InertiaTensor` of the groups of a partition, and their outputs.

use crate::array::{ArrayArg1, ArrayArg2};
use molrs::compute::{
    CenterOfMass, CenterOfMassInput, CenterOfMassResult, Compute, GyrationTensor,
    GyrationTensorInput, GyrationTensorResult, InertiaTensor, InertiaTensorInput,
    InertiaTensorResult,
};

use pyo3::prelude::*;

use super::args;
use super::error::Result;
use super::output::{ToPy, py_result};
use super::spec::described;
use crate::core::simbox::PyBox;

py_result! {
    /// Centres of mass (``molrs::compute::CenterOfMassResult``): one row of
    /// ``centers`` and one ``total_masses`` entry per group key in ``groups``.
    PyCenterOfMassResult(CenterOfMassResult) as "CenterOfMassResult" {
        groups, centers, total_masses
    }
}

py_result! {
    /// Gyration tensors (``molrs::compute::GyrationTensorResult``): a
    /// ``(3, 3)`` tensor per group key in ``groups``.
    PyGyrationTensorResult(GyrationTensorResult) as "GyrationTensorResult" { groups, tensors }
}

#[pymethods]
impl PyGyrationTensorResult {
    /// The radius of gyration of each group, ``sqrt(trace)``.
    fn radius_of_gyration<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.radius_of_gyration().to_py(py)
    }
}

py_result! {
    /// Inertia tensors (``molrs::compute::InertiaTensorResult``): a ``(3, 3)``
    /// tensor per group key in ``groups``.
    PyInertiaTensorResult(InertiaTensorResult) as "InertiaTensorResult" { groups, tensors }
}

/// Mass-weighted centre of each group (``molrs::compute::CenterOfMass``).
///
/// ``compute(positions=, simbox=None, partition=, masses=)``: ``(N, 3)``
/// positions, the box (groups are unwrapped across it when given), a
/// partition (an unsigned-int label array or a partition result) and ``(N,)``
/// masses. Returns a ``CenterOfMassResult``.
#[pyclass(module = "molrs.compute", name = "CenterOfMass", frozen)]
pub(crate) struct PyCenterOfMass {
    inner: CenterOfMass,
}

#[pymethods]
impl PyCenterOfMass {
    #[new]
    fn new() -> Self {
        Self {
            inner: CenterOfMass::new(),
        }
    }

    #[pyo3(signature = (*, positions, simbox = None, partition, masses))]
    fn compute<'py>(
        &self,
        positions: ArrayArg2<'py, f64>,
        simbox: Option<PyRef<'py, PyBox>>,
        partition: &Bound<'py, PyAny>,
        masses: ArrayArg1<'py, f64>,
    ) -> Result<PyCenterOfMassResult> {
        let partition = args::PartitionArg::extract(partition)?;
        let out = self.inner.compute(CenterOfMassInput {
            positions: positions.as_array(),
            simbox: simbox.as_ref().map(|simbox| &simbox.inner),
            partition: partition.partition(),
            masses: masses.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyCenterOfMass, CenterOfMass);

/// Weighted gyration tensor of each group (``molrs::compute::GyrationTensor``).
///
/// ``compute(positions=, simbox=None, partition=, weights=)``: as
/// ``CenterOfMass`` with ``(N,)`` weights. Returns a ``GyrationTensorResult``.
#[pyclass(module = "molrs.compute", name = "GyrationTensor", frozen)]
pub(crate) struct PyGyrationTensor {
    inner: GyrationTensor,
}

#[pymethods]
impl PyGyrationTensor {
    #[new]
    fn new() -> Self {
        Self {
            inner: GyrationTensor::new(),
        }
    }

    #[pyo3(signature = (*, positions, simbox = None, partition, weights))]
    fn compute<'py>(
        &self,
        positions: ArrayArg2<'py, f64>,
        simbox: Option<PyRef<'py, PyBox>>,
        partition: &Bound<'py, PyAny>,
        weights: ArrayArg1<'py, f64>,
    ) -> Result<PyGyrationTensorResult> {
        let partition = args::PartitionArg::extract(partition)?;
        let out = self.inner.compute(GyrationTensorInput {
            positions: positions.as_array(),
            simbox: simbox.as_ref().map(|simbox| &simbox.inner),
            partition: partition.partition(),
            weights: weights.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyGyrationTensor, GyrationTensor);

/// Moment of inertia tensor of each group (``molrs::compute::InertiaTensor``).
///
/// ``compute(positions=, simbox=None, partition=, masses=)``: as
/// ``CenterOfMass``. Returns an ``InertiaTensorResult``.
#[pyclass(module = "molrs.compute", name = "InertiaTensor", frozen)]
pub(crate) struct PyInertiaTensor {
    inner: InertiaTensor,
}

#[pymethods]
impl PyInertiaTensor {
    #[new]
    fn new() -> Self {
        Self {
            inner: InertiaTensor::new(),
        }
    }

    #[pyo3(signature = (*, positions, simbox = None, partition, masses))]
    fn compute<'py>(
        &self,
        positions: ArrayArg2<'py, f64>,
        simbox: Option<PyRef<'py, PyBox>>,
        partition: &Bound<'py, PyAny>,
        masses: ArrayArg1<'py, f64>,
    ) -> Result<PyInertiaTensorResult> {
        let partition = args::PartitionArg::extract(partition)?;
        let out = self.inner.compute(InertiaTensorInput {
            positions: positions.as_array(),
            simbox: simbox.as_ref().map(|simbox| &simbox.inner),
            partition: partition.partition(),
            masses: masses.as_array(),
        })?;
        Ok(out.into())
    }
}

described!(PyInertiaTensor, InertiaTensor);

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCenterOfMassResult>()?;
    m.add_class::<PyGyrationTensorResult>()?;
    m.add_class::<PyInertiaTensorResult>()?;
    m.add_class::<PyCenterOfMass>()?;
    m.add_class::<PyGyrationTensor>()?;
    m.add_class::<PyInertiaTensor>()?;
    Ok(())
}
