//! `molpy.builder.DrudeBuilder` / `Tip4pBuilder` — molrs `ff::virtual_sites`.

use std::collections::HashMap;

use molrs::ff::virtual_sites::{DrudeBuilder, Polarizability, TIP4P_OM, Tip4pBuilder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyMapping;

use crate::core::molgraph::PyAtomistic;
use crate::core::simbox::PyBox;
use crate::error::molrs_error_to_pyerr;

/// The columns of one CL&Pol ``alpha.ff`` row, less ``type_name``.
const CLPOL_KEYS: [&str; 5] = ["m_D", "q_D_sign", "k_D", "alpha", "a_thole"];

/// One row's parameters, every column present and numeric.
fn polarizability(name: &str, params: &Bound<'_, PyAny>) -> PyResult<Polarizability> {
    let params = params.cast::<PyMapping>()?;
    let missing: Vec<&str> = CLPOL_KEYS
        .into_iter()
        .filter(|k| !params.contains(*k).unwrap_or(false))
        .collect();
    if !missing.is_empty() {
        return Err(PyValueError::new_err(format!(
            "CL&Pol parameters for type {name:?} lack {}",
            missing.join(", ")
        )));
    }
    let num = |k: &str| -> PyResult<f64> { params.get_item(k)?.extract() };
    Ok(Polarizability {
        m_d: num("m_D")?,
        q_d_sign: num("q_D_sign")?,
        k_d: num("k_D")?,
        alpha: num("alpha")?,
        a_thole: num("a_thole")?,
    })
}

/// ``type -> params`` from either CL&Pol table shape: a mapping (as
/// ``DrudeBuilder().polarizability`` holds it) or rows each carrying a
/// ``type_name`` (as ``mp.io.read_clpol_alpha`` returns). A type given twice
/// keeps its later row.
fn polarizability_table(table: &Bound<'_, PyAny>) -> PyResult<HashMap<String, Polarizability>> {
    let mut out = HashMap::new();
    if let Ok(mapping) = table.cast::<PyMapping>() {
        for item in mapping.items()?.iter() {
            let (name, params): (String, Bound<'_, PyAny>) = item.extract()?;
            let row = polarizability(&name, &params)?;
            out.insert(name, row);
        }
        return Ok(out);
    }
    for row in table.try_iter()? {
        let row = row?;
        let mapping = row.cast::<PyMapping>()?;
        if !mapping.contains("type_name")? {
            return Err(PyValueError::new_err(format!(
                "CL&Pol row {} has no 'type_name'",
                row.repr()?
            )));
        }
        let name: String = mapping.get_item("type_name")?.extract()?;
        let params = polarizability(&name, &row)?;
        out.insert(name, params);
    }
    Ok(out)
}

/// CL&Pol polarizer: a Drude shell on every atom whose CL&P type has
/// ``k_D > 0`` in the table (molrs ``DrudeBuilder``).
///
/// The shell takes charge ``q_D = ±sqrt(4πε₀ · k_D · α)`` and mass ``m_D``,
/// sits on its core and is typed ``<drude_prefix><core type>``; the core's
/// charge and mass lose the shell's, so the net charge is conserved. The
/// shell is tied to its core by a ``drudes`` row of style ``harmonic`` with
/// ``k = k_D / 2`` in kcal/mol/Å² (molrs's ``k r²`` against alpha.ff's
/// ``½ k_D r²`` in kJ), ``alpha`` and ``thole`` (CL&Pol's ``a_thole``).
///
/// Parameters
/// ----------
/// polarizability
///     The CL&Pol table: a ``type -> params`` mapping or ``read_clpol_alpha``
///     rows; ``None`` is molrs's shipped ``alpha.ff``.
/// drude_prefix
///     The prefix of a shell's type.
#[pyclass(
    module = "molpy.builder",
    name = "DrudeBuilder",
    frozen,
    skip_from_py_object
)]
pub struct PyDrudeBuilder {
    inner: DrudeBuilder,
}

#[pymethods]
impl PyDrudeBuilder {
    #[new]
    #[pyo3(signature = (polarizability=None, *, drude_prefix="D"))]
    fn new(polarizability: Option<&Bound<'_, PyAny>>, drude_prefix: &str) -> PyResult<Self> {
        let inner = match polarizability {
            None => DrudeBuilder::clpol(),
            Some(table) => DrudeBuilder::new(polarizability_table(table)?),
        };
        Ok(Self {
            inner: inner.with_prefix(drude_prefix),
        })
    }

    /// The CL&Pol table this builder reads, ``type -> {"m_D", "q_D_sign",
    /// "k_D", "alpha", "a_thole"}`` in u, e (sign), kJ/mol/Å² (``k/2 r²``
    /// form), Å³ and dimensionless. A type with ``k_D == 0`` carries no Drude
    /// particle.
    #[getter]
    fn polarizability(&self) -> HashMap<String, HashMap<&'static str, f64>> {
        self.inner
            .polarizability()
            .iter()
            .map(|(name, row)| {
                let values = [row.m_d, row.q_d_sign, row.k_d, row.alpha, row.a_thole];
                (name.clone(), CLPOL_KEYS.into_iter().zip(values).collect())
            })
            .collect()
    }

    /// A new molecule with a Drude shell on every polarizable atom; ``mol``
    /// is untouched.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     An atom that is no site has no ``type``, or a polarizable atom has
    ///     no ``charge``.
    fn apply(&self, py: Python<'_>, mol: &Bound<'_, PyAtomistic>) -> PyResult<Py<PyAtomistic>> {
        let out = self
            .inner
            .apply(mol.borrow().core())
            .map_err(molrs_error_to_pyerr)?;
        mol.borrow().derive(py, out)
    }
}

/// A TIP4P M-site on each water's HOH bisector (molrs ``Tip4pBuilder``).
///
/// The massless site sits at ``d_om`` Å from O along the bisector of the O–H
/// vectors (minimum images in ``box`` when given); O's charge moves onto it.
/// The construction is a ``virtual_sites`` row of style ``average3``
/// (M, O, H, H) with ``w1 = 1 − 2a``, ``w2 = w3 = a``,
/// ``a = d_om / |r_OH1 + r_OH2|``.
#[pyclass(
    module = "molpy.builder",
    name = "Tip4pBuilder",
    frozen,
    skip_from_py_object
)]
pub struct PyTip4pBuilder {
    inner: Tip4pBuilder,
}

#[pymethods]
impl PyTip4pBuilder {
    #[new]
    #[pyo3(signature = (d_om=TIP4P_OM, *, r#box=None), text_signature = "(d_om=0.1546, *, box=None)")]
    fn new(d_om: f64, r#box: Option<PyRef<'_, PyBox>>) -> Self {
        let builder = Tip4pBuilder::new(d_om);
        Self {
            inner: match r#box {
                Some(b) => builder.with_box(b.inner.clone()),
                None => builder,
            },
        }
    }

    /// A new molecule with an M-site on every water; ``mol`` is untouched.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     A water atom without coordinates or charge, or a linear water.
    fn apply(&self, py: Python<'_>, mol: &Bound<'_, PyAtomistic>) -> PyResult<Py<PyAtomistic>> {
        let out = self
            .inner
            .apply(mol.borrow().core())
            .map_err(molrs_error_to_pyerr)?;
        mol.borrow().derive(py, out)
    }
}

/// Register the virtual-site builders on `molpy.builder`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDrudeBuilder>()?;
    m.add_class::<PyTip4pBuilder>()?;
    Ok(())
}
