//! `ForceField.canonical`, `ForceField.to_form`, `ForceField.fit_form`: the
//! force-field IR's form conversions (`molrs::ff::ir::form`) on the Python
//! force field.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use molrs::ff::form_conversion::{FitMetric, FitResidual};

use super::PyForceField;
use crate::ff::ir::refuse;

fn residual_dict<'py>(py: Python<'py>, residual: &FitResidual) -> PyResult<Bound<'py, PyDict>> {
    let types = PyList::empty(py);
    for t in &residual.types {
        let d = PyDict::new(py);
        d.set_item("style", &t.style)?;
        d.set_item("type", &t.type_)?;
        d.set_item("exact", t.exact)?;
        d.set_item("sum_sq", t.sum_sq)?;
        d.set_item("rms", t.rms())?;
        d.set_item("max_abs", t.max_abs)?;
        d.set_item("offset", t.offset)?;
        types.append(d)?;
    }
    let out = PyDict::new(py);
    out.set_item("sum_sq", residual.sum_sq())?;
    out.set_item("rms", residual.rms())?;
    out.set_item("max_abs", residual.max_abs())?;
    out.set_item("types", types)?;
    Ok(out)
}

#[pymethods]
impl PyForceField {
    /// The force field in canonical form: every style of a form family in
    /// its canonical style's category mapped onto that style, exactly
    /// (``dihedral opls``/``charmm``/``rb``/… → ``dihedral periodic``,
    /// ``bond class2`` → ``bond harmonic``, ``pair lj/class2`` →
    /// ``pair lj/cut``). Raises ``ValueError`` naming the type and the
    /// condition for a row the canonical style cannot hold.
    fn canonical(&self) -> PyResult<PyForceField> {
        let inner = self.inner.canonical().map_err(refuse)?;
        Ok(PyForceField { inner })
    }

    /// The force field with every style of ``category`` in the form family
    /// of ``style`` converted to ``style``, exactly; raises ``ValueError``
    /// naming the first row outside ``style``'s image and the condition.
    fn to_form(&self, category: &str, style: &str) -> PyResult<PyForceField> {
        let inner = self.inner.to_form(category, style).map_err(refuse)?;
        Ok(PyForceField { inner })
    }

    /// Fit every other style of ``category`` to ``style`` by least squares
    /// over the sample points ``q`` of the category's coordinate (``r``;
    /// ``θ``, ``φ`` in radians) with weights ``w`` (default 1), Boltzmann
    /// factors of each row's source energy at ``kt`` when given, and a free
    /// constant offset when ``offset``. Returns ``(forcefield, residual)``,
    /// ``residual`` a dict: ``sum_sq``, ``rms``, ``max_abs`` and ``types``
    /// (per fitted row: ``style``, ``type``, ``exact``, ``sum_sq``, ``rms``,
    /// ``max_abs``, ``offset``).
    #[pyo3(signature = (category, style, q, w=None, *, kt=None, offset=false))]
    #[allow(clippy::too_many_arguments)]
    fn fit_form<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        style: &str,
        q: Vec<f64>,
        w: Option<Vec<f64>>,
        kt: Option<f64>,
        offset: bool,
    ) -> PyResult<(PyForceField, Bound<'py, PyDict>)> {
        let mut metric = FitMetric::new(q);
        if let Some(w) = w {
            metric = metric.weights(w);
        }
        metric.kt = kt;
        metric.offset = offset;
        let (inner, residual) = self
            .inner
            .fit_form(category, style, &metric)
            .map_err(refuse)?;
        Ok((PyForceField { inner }, residual_dict(py, &residual)?))
    }
}
