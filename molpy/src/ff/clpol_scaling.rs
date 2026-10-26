//! Python bindings for `molrs::ff::clpol_scaling` (`molpy.ff.clpol_scaling`): CL&Pol
//! fragment scaling of Lennard-Jones parameters — one fragment's data
//! ([`PyFragmentScaling`], its SAPT pair factor `k_ij`) and the scaling
//! ([`PyClpolScaling`]: its fragment table, `apply` to a force field).

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

    /// The SAPT ε-scaling factor of this fragment and ``other`` at
    /// centre-of-mass distance ``r`` (Å).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A fragment whose polarizability is not positive.
    fn k_ij(&self, other: PyRef<'_, PyFragmentScaling>, r: f64) -> PyResult<f64> {
        molrs::ff::clpol_scaling::FragmentScaling::from(self.clone())
            .k_ij(&other.clone().into(), r)
            .map_err(py_value_err)
    }

    fn __repr__(&self) -> String {
        format!("FragmentScaling(name='{}')", self.name)
    }
}

/// The scaling of ``ff``'s fragments' Lennard-Jones rows, in the core's
/// error spelling: a fragment the table lacks is a ``KeyError``.
fn scaling_err(error: molrs::ff::clpol_scaling::ClpolScalingError) -> PyErr {
    match error {
        molrs::ff::clpol_scaling::ClpolScalingError::MissingFragment(name) => {
            PyKeyError::new_err(format!("no scaling data for fragment '{name}'"))
        }
        other => py_value_err(other),
    }
}

/// CL&Pol's SAPT-derived Lennard-Jones scaling.
///
/// Parameters
/// ----------
/// fragments : dict[str, FragmentScaling], optional
///     Each fragment's charge, dipole and polarizability, by name. ``None``:
///     CL&Pol's own table (paduagroup/clandpol ``fragment.ff``).
/// scale_sigma : bool
///     Scale each cross-fragment σ by 0.985 as well as ε.
#[pyclass(module = "molpy.ff.clpol_scaling", name = "ClpolScaling", subclass)]
pub struct PyClpolScaling {
    inner: molrs::ff::clpol_scaling::ClpolScaling,
}

#[pymethods]
impl PyClpolScaling {
    #[new]
    #[pyo3(signature = (fragments=None, *, scale_sigma=false))]
    fn new(fragments: Option<&Bound<'_, PyDict>>, scale_sigma: bool) -> PyResult<Self> {
        let mut inner = molrs::ff::clpol_scaling::ClpolScaling {
            scale_sigma,
            ..Default::default()
        };
        if let Some(data) = fragments {
            let mut table = HashMap::with_capacity(data.len());
            for (label, value) in data.iter() {
                let item = value.extract::<PyRef<'_, PyFragmentScaling>>()?;
                table.insert(label.extract::<String>()?, item.clone().into());
            }
            inner.fragments = table;
        }
        Ok(Self { inner })
    }

    /// The fragment table, by fragment name.
    #[getter]
    fn fragments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (name, scaling) in &self.inner.fragments {
            result.set_item(name, Py::new(py, PyFragmentScaling::from(scaling.clone()))?)?;
        }
        Ok(result)
    }

    /// Whether σ is scaled as well as ε.
    #[getter]
    fn scale_sigma(&self) -> bool {
        self.inner.scale_sigma
    }

    /// A copy of ``ff`` whose cross-fragment ``pair`` rows are scaled by the
    /// SAPT factor of their two fragments at the distance of the fragments'
    /// centres of mass. ``ff`` is not changed.
    ///
    /// Parameters
    /// ----------
    /// ff : ForceField
    /// fragment_atoms : dict[str, tuple[list[str], list[tuple[float, float, float]], list[float]]]
    ///     Per fragment name, its atom types, coordinates (Å) and masses.
    ///
    /// Raises
    /// ------
    /// KeyError
    ///     A fragment the table has no row for.
    /// ValueError
    ///     Types, coordinates and masses of unequal length, a non-finite mass,
    ///     or a non-positive polarizability.
    fn apply(
        &self,
        py: Python<'_>,
        ff: &Bound<'_, PyForceField>,
        fragment_atoms: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyForceField>> {
        let mut native = Vec::with_capacity(fragment_atoms.len());
        for (label, value) in fragment_atoms.iter() {
            let (atom_types, coords, masses) =
                value.extract::<(Vec<String>, Vec<[f64; 3]>, Vec<f64>)>()?;
            native.push(molrs::ff::clpol_scaling::FragmentAtoms {
                name: label.extract::<String>()?,
                atom_types,
                coords,
                masses,
            });
        }
        let inner = self
            .inner
            .apply(&ff.borrow().inner, &native)
            .map_err(scaling_err)?;
        PyForceField::from_core(py, inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "ClpolScaling(fragments={}, scale_sigma={})",
            self.inner.fragments.len(),
            if self.inner.scale_sigma {
                "True"
            } else {
                "False"
            }
        )
    }
}

/// Register `molpy.ff.clpol_scaling`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFragmentScaling>()?;
    m.add_class::<PyClpolScaling>()?;
    Ok(())
}
