//! `BigSMILES` / generative `BigSMILES` (`molrs::io::bigsmiles`):
//! `read_bigsmiles(text)` reads a string into a [`PyBigSmilesIr`], whose
//! `to_molecules()` / `to_system()` give the `molpy.builder` polymer model.

use molrs::io::bigsmiles::BigSmilesIr;
use pyo3::prelude::*;

use crate::error::smiles_error_to_pyerr;
use crate::polymer::{PyMolecule, PySystem};

/// A parsed `BigSMILES` or generative-`BigSMILES` string, handed out by
/// :func:`read_bigsmiles`.
#[pyclass(module = "molpy.io.bigsmiles", name = "BigSmilesIr", frozen)]
pub struct PyBigSmilesIr {
    inner: BigSmilesIr,
}

#[pymethods]
impl PyBigSmilesIr {
    /// The number of molecules (``.``-separated) the string writes.
    #[getter]
    fn n_molecules(&self) -> usize {
        self.inner.molecules.len()
    }

    /// Each molecule as a :class:`molpy.builder.Molecule`. Fragments are named by
    /// position: ``"m.p"`` for SMILES piece ``p`` of molecule ``m``,
    /// ``"m.p.rk"`` / ``"m.p.ek"`` for repeat unit / end group ``k`` of
    /// object ``p``. Size distributions count heavy-atom mass.
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     (``notation == "bigsmiles"``, ``kind == "BigBuild"``) for a string
    ///     the polymer model refuses.
    fn to_molecules(&self) -> PyResult<Vec<PyMolecule>> {
        let molecules = self.inner.to_molecules().map_err(smiles_error_to_pyerr)?;
        Ok(molecules
            .into_iter()
            .map(|inner| PyMolecule { inner })
            .collect())
    }

    /// The string as a :class:`molpy.builder.System`, every molecule in its
    /// written amount ``.|N|`` / ``.|p%|``, amounts counting heavy-atom mass.
    ///
    /// Raises
    /// ------
    /// SmilesError
    ///     (``kind == "BigBuild"``) for a molecule without an amount, or
    ///     amounts the system refuses.
    fn to_system(&self) -> PyResult<PySystem> {
        let inner = self.inner.to_system().map_err(smiles_error_to_pyerr)?;
        Ok(PySystem { inner })
    }
}

/// Read a `BigSMILES` or generative-`BigSMILES` string.
///
/// Parameters
/// ----------
/// text : str
///     The string: SMILES with stochastic objects ``{[] … ; … []}``,
///     generation weights ``[<|0.5|]``, size distributions
///     ``|schulz_zimm(1500, 1400)|`` and system amounts ``.|5e5|``.
///
/// Returns
/// -------
/// BigSmilesIr
///
/// Raises
/// ------
/// SmilesError
///     (a ``ValueError``, ``notation == "bigsmiles"``) for text that is not
///     (generative) `BigSMILES`, or uses notation this reader does not model
///     (``kind == "BigUnsupported"``: nested objects, objects in branches,
///     fragment declarations, ladder and non-covalent descriptors).
///
/// Examples
/// --------
/// >>> ir = molpy.io.bigsmiles.read_bigsmiles("{[][<]CC[>];[<][H],[>][H][]}|poisson(300)|.|3000|")
/// >>> ir.n_molecules
/// 1
#[pyfunction]
fn read_bigsmiles(text: &str) -> PyResult<PyBigSmilesIr> {
    let inner = BigSmilesIr::parse(text).map_err(smiles_error_to_pyerr)?;
    Ok(PyBigSmilesIr { inner })
}

/// Register `read_bigsmiles` and `BigSmilesIr`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(
        m,
        "molpy.io.bigsmiles",
        wrap_pyfunction!(read_bigsmiles, m)?,
    )?;
    m.add_class::<PyBigSmilesIr>()
}
