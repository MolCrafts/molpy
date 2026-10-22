//! Python bindings for `molrs::ff::clpol_scaling` (`molpy.ff.clpol_scaling`): CL&Pol
//! fragment scaling of Lennard-Jones parameters — the fragment table
//! ([`PyFragmentScaling`]; the shipped table is `fragment_table`), the SAPT pair factor
//! (`compute_k_ij`) and the scaled force field (`scale_lj`).

use std::collections::HashMap;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::py_value_err;
use crate::ff::forcefield::PyForceField;

/// CL&Pol fragment scaling data backed by the native force-field layer.
#[pyclass(
    module = "molpy.ff.clpol_scaling",
    name = "FragmentScaling",
    frozen,
    get_all,
    skip_from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyFragmentScaling {
    name: String,
    q: f64,
    mu: f64,
    alpha: f64,
    polarizable: bool,
}

impl From<PyFragmentScaling> for molrs::ff::clpol_scaling::FragmentScaling {
    fn from(value: PyFragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

impl From<molrs::ff::clpol_scaling::FragmentScaling> for PyFragmentScaling {
    fn from(value: molrs::ff::clpol_scaling::FragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

#[pymethods]
impl PyFragmentScaling {
    #[new]
    #[pyo3(signature = (name, q, mu, alpha, polarizable=false))]
    fn new(name: String, q: f64, mu: f64, alpha: f64, polarizable: bool) -> Self {
        Self {
            name,
            q,
            mu,
            alpha,
            polarizable,
        }
    }

    fn __repr__(&self) -> String {
        format!("FragmentScaling(name='{}')", self.name)
    }
}

/// Native SAPT epsilon-scaling factor.
#[pyfunction(name = "compute_k_ij")]
pub fn compute_k_ij_py(
    fr_i: PyRef<'_, PyFragmentScaling>,
    fr_j: PyRef<'_, PyFragmentScaling>,
    r: f64,
) -> PyResult<f64> {
    molrs::ff::clpol_scaling::compute_k_ij(&fr_i.clone().into(), &fr_j.clone().into(), r)
        .map_err(py_value_err)
}

/// Clone and scale LJ parameters using native COM and force-field transforms.
#[pyfunction(name = "scale_lj")]
#[pyo3(signature = (ff, fragments, fragment_table=None, scale_sigma=false))]
pub fn scale_lj_py(
    py: Python<'_>,
    ff: &Bound<'_, PyForceField>,
    fragments: &Bound<'_, PyDict>,
    fragment_table: Option<&Bound<'_, PyDict>>,
    scale_sigma: bool,
) -> PyResult<Py<PyForceField>> {
    let mut native_fragments = Vec::with_capacity(fragments.len());
    for (label, value) in fragments.iter() {
        let name = label.extract::<String>()?;
        let (atom_types, coords, masses) =
            value.extract::<(Vec<String>, Vec<[f64; 3]>, Vec<f64>)>()?;
        native_fragments.push(molrs::ff::clpol_scaling::FragmentAtoms {
            name,
            atom_types,
            coords,
            masses,
        });
    }

    let mut scaling = HashMap::new();
    if let Some(data) = fragment_table {
        for (label, value) in data.iter() {
            let item = value.extract::<PyRef<'_, PyFragmentScaling>>()?;
            scaling.insert(label.extract::<String>()?, item.clone().into());
        }
    } else {
        scaling = molrs::ff::clpol_scaling::fragment_table();
    }

    let inner = molrs::ff::clpol_scaling::scale_lj(
        &ff.borrow().inner,
        &native_fragments,
        &scaling,
        scale_sigma,
    )
    .map_err(|error| match error {
        molrs::ff::clpol_scaling::ScaleLjError::MissingFragment(name) => {
            PyKeyError::new_err(format!("no scaling data for fragment '{name}'"))
        }
        other => py_value_err(other),
    })?;
    PyForceField::from_core(py, inner)
}

/// CL&Pol's fragment scaling table (paduagroup/clandpol ``fragment.ff``):
/// each fragment's charge, dipole and polarizability, by fragment name, as
/// :func:`scale_lj` reads it.
#[pyfunction]
fn fragment_table(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, scaling) in molrs::ff::clpol_scaling::fragment_table() {
        result.set_item(name, Py::new(py, PyFragmentScaling::from(scaling))?)?;
    }
    Ok(result)
}

/// Register `molpy.ff.clpol_scaling`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFragmentScaling>()?;
    crate::add_function(
        m,
        "molpy.ff.clpol_scaling",
        wrap_pyfunction!(compute_k_ij_py, m)?,
    )?;
    crate::add_function(
        m,
        "molpy.ff.clpol_scaling",
        wrap_pyfunction!(scale_lj_py, m)?,
    )?;
    crate::add_function(
        m,
        "molpy.ff.clpol_scaling",
        wrap_pyfunction!(fragment_table, m)?,
    )?;
    Ok(())
}
