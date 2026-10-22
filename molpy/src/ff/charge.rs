//! Python bindings for the charge models (`molrs::ff::charge`).
//!
//! Three classes — `molpy.ff.charge.BccModel`, `MullikenModel`, `GasteigerModel`
//! — behind the one Rust trait, and behind **one Python calling convention**:
//!
//! ```text
//! model.needs_equivalencing() -> bool
//! model.assign(mol, qm=None)  -> ndarray[float64], shape (n_atoms,)
//! ```
//!
//! `GasteigerModel` takes the `qm` argument and ignores it. That is the row that
//! keeps the trait honest — a binding that gave it a different method name would put
//! the branch the trait exists to remove back into every caller — so the argument is
//! optional on every model here, and a model that *needs* QM charges refuses when it
//! has none rather than inventing a base to correct.
//!
//! # Precision
//!
//! Charges cross as `f64` end to end: in as a `float64` numpy array (read, never
//! written), out as a fresh `float64` one. Nothing here narrows to `f32`, and nothing
//! renormalizes. Both matter and neither is visible against antechamber's six printed
//! decimals: BCC increments are pairwise antisymmetric, so the corrected charges sum
//! to `sum(am1)` to ~1e-16 in `f64` and to ~1e-8 through an `f32` — and antechamber
//! itself carries the AM1 rounding residual straight through rather than rescaling to
//! the integer net charge.
//!
//! # Nothing is written into the molecule
//!
//! Every model takes the graph by shared borrow and returns an array. The models'
//! internal atom types (BCC codes like `11` / `91`) stay internal, so a caller keeps
//! their GAFF or OPLS types in `keys.TYPE` *and* gets BCC charges — which is exactly
//! what the standard AM1-BCC workflow needs.

use crate::array::ArrayArg1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::charge::{
    BccModel, BccParameterSet, ChargeError, ChargeModel, GasteigerModel, MullikenModel,
};

use crate::core::molgraph::PyAtomistic;

/// The correction family named by an antechamber `-c` flag
/// ([`BccParameterSet::from_name`]), as a Python `ValueError` when it names none.
fn bcc_set_from_name(name: &str) -> PyResult<BccParameterSet> {
    BccParameterSet::from_name(name).map_err(PyValueError::new_err)
}

/// A charge failure, as a Python `ValueError`.
///
/// Every [`ChargeError`] variant is a refusal — a missing QM base, a count mismatch,
/// an atom the table cannot name, a bond it cannot correct — so all of them are the
/// caller's to see, and none of them may become a plausible-looking charge.
fn charge_err(e: ChargeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// The QM charges a model was handed, as an owned `f64` vector.
///
/// `as_array().to_vec()` rather than `as_slice()`: a non-contiguous view is a legal
/// argument (a strided slice of a bigger charge table), and copying is what keeps the
/// caller's array read-only — `correct` must not consume the buffer it was given.
fn qm_charges(qm: Option<ArrayArg1<'_, f64>>) -> Option<Vec<f64>> {
    qm.map(|array| array.as_array().to_vec())
}

/// One charge per atom, as a fresh `float64` numpy array of shape `(n_atoms,)`.
fn charges_out(py: Python<'_>, charges: Vec<f64>) -> Bound<'_, PyArray1<f64>> {
    charges.into_pyarray(py)
}

/// AM1-BCC / ABCG2 bond-charge corrections — `molpy.ff.charge.BccModel`.
///
/// The correction stage of AM1-BCC: base QM charges in, corrected charges out. The
/// increments are pairwise antisymmetric, so the total charge is conserved to machine
/// precision — this model does not renormalize, and neither does antechamber.
///
/// Parameters
/// ----------
/// parameter_set : str
///     The antechamber ``-c`` flag naming the correction family: ``"bcc"``
///     (``BCCPARM.DAT``) or ``"abcg2"`` (``BCCPARM_ABCG2.DAT``). Required — there is
///     no default table, and the two disagree.
///
/// Raises
/// ------
/// ValueError
///     If ``parameter_set`` names no correction family.
///
/// Examples
/// --------
/// >>> model = molpy.ff.charge.BccModel(parameter_set="bcc")
/// >>> model.correct(methane, np.zeros(5))[0]     # 4 x the C-H increment
/// 0.1572
/// >>> model.assign(methanol, raw_sqm_charges)    # equivalences first, then corrects
#[pyclass(module = "molpy.ff.charge", name = "BccModel", subclass)]
#[derive(Debug)]
pub struct PyBccModel {
    inner: BccModel,
}

#[pymethods]
impl PyBccModel {
    /// Bind the model to the correction family `parameter_set` names.
    #[new]
    #[pyo3(signature = (*, parameter_set))]
    fn new(parameter_set: &str) -> PyResult<Self> {
        let inner = BccModel::new(bcc_set_from_name(parameter_set)?);
        Ok(Self { inner })
    }

    /// The antechamber ``-c`` flag of this model's correction family.
    #[getter]
    fn parameter_set(&self) -> &'static str {
        self.inner.parameter_set().name()
    }

    /// ``True`` — AM1-BCC averages its base charges over the topological-equivalence
    /// classes before correcting them (antechamber's ``-eq 1``, whose default is per
    /// charge *method*).
    fn needs_equivalencing(&self) -> bool {
        ChargeModel::needs_equivalencing(&self.inner)
    }

    /// The final AM1-BCC charges: equivalence the QM base charges, then correct them.
    ///
    /// This is the **whole** model — the raw Mulliken charges an AM1 backend really
    /// hands over (un-averaged, one conformer) in, final charges out. Use
    /// :meth:`correct` when the caller has already applied the class-mean.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule; left untouched, including its ``type`` column.
    /// qm : ndarray of float64, optional
    ///     Raw QM base charges in graph atom order.
    ///
    /// Returns
    /// -------
    /// ndarray of float64, shape (n_atoms,)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``qm`` is absent, is not one charge per atom, or the molecule carries an
    ///     atom the parameter set cannot type or a bond it cannot correct.
    #[pyo3(signature = (mol, qm=None))]
    fn assign<'py>(
        &self,
        py: Python<'py>,
        mol: &PyAtomistic,
        qm: Option<ArrayArg1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qm = qm_charges(qm);
        let charges = self
            .inner
            .assign(mol.core(), qm.as_deref())
            .map_err(charge_err)?;
        Ok(charges_out(py, charges))
    }

    /// Correct base charges that are already equivalenced.
    ///
    /// The stage AM1-BCC *is*: one bond-charge increment per bond, added to the atom
    /// at one end and subtracted at the other. It does **not** equivalence — feeding
    /// it raw `sqm` charges gives a different (and asymmetric) answer, which is what
    /// :meth:`assign` exists for.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule; left untouched, including its ``type`` column.
    /// am1 : ndarray of float64
    ///     Equivalenced AM1 base charges in graph atom order. Read, never written.
    ///
    /// Returns
    /// -------
    /// ndarray of float64, shape (n_atoms,)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``am1`` is not one charge per atom, or the molecule carries an atom the
    ///     parameter set cannot type or a bond it cannot correct.
    fn correct<'py>(
        &self,
        py: Python<'py>,
        mol: &PyAtomistic,
        am1: ArrayArg1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let am1 = am1.as_array().to_vec();
        let charges = self.inner.correct(mol.core(), &am1).map_err(charge_err)?;
        Ok(charges_out(py, charges))
    }

    fn __repr__(&self) -> String {
        format!("BccModel(parameter_set='{}')", self.parameter_set())
    }
}

/// Mulliken population charges, unchanged — `molpy.ff.charge.MullikenModel`.
///
/// The pass-through: it hands back the QM charges it was given, bit for bit. No
/// correction, and no equivalencing (antechamber's ``-eq`` default is ``0`` for every
/// method but ``bcc`` / ``abcg2`` / ``resp``).
///
/// Examples
/// --------
/// >>> molpy.ff.charge.MullikenModel().assign(mol, am1)
#[pyclass(module = "molpy.ff.charge", name = "MullikenModel", subclass)]
#[derive(Debug)]
pub struct PyMullikenModel;

#[pymethods]
impl PyMullikenModel {
    /// Create the pass-through model. It has no parameter table.
    #[new]
    fn new() -> Self {
        Self
    }

    /// ``False`` — Mulliken charges are handed back as they arrived.
    fn needs_equivalencing(&self) -> bool {
        ChargeModel::needs_equivalencing(&MullikenModel)
    }

    /// The QM charges, unchanged.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule; used only for its atom count.
    /// qm : ndarray of float64, optional
    ///     QM charges in graph atom order.
    ///
    /// Returns
    /// -------
    /// ndarray of float64, shape (n_atoms,)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``qm`` is absent or is not one charge per atom.
    #[pyo3(signature = (mol, qm=None))]
    fn assign<'py>(
        &self,
        py: Python<'py>,
        mol: &PyAtomistic,
        qm: Option<ArrayArg1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qm = qm_charges(qm);
        let charges = MullikenModel
            .assign(mol.core(), qm.as_deref())
            .map_err(charge_err)?;
        Ok(charges_out(py, charges))
    }

    fn __repr__(&self) -> String {
        "MullikenModel()".to_string()
    }
}

/// Gasteiger / PEOE partial charges — `molpy.ff.charge.GasteigerModel`.
///
/// The model with **no QM input**: it is handed a molecule and nothing else, and
/// still lands on antechamber's ``-c gas`` column. The ``qm`` argument exists so that
/// a caller holding an unknown model can call every model the same way; this one
/// ignores it.
///
/// The loop runs to convergence (antechamber's ``CONVERG`` 1e-5, ``GASMAXITER`` 500)
/// — a sweep count is not a knob, because the damping is geometric and where the loop
/// stops *is* the answer.
///
/// Examples
/// --------
/// >>> molpy.ff.charge.GasteigerModel().assign(mol)
#[pyclass(module = "molpy.ff.charge", name = "GasteigerModel", subclass)]
#[derive(Debug)]
pub struct PyGasteigerModel;

#[pymethods]
impl PyGasteigerModel {
    /// Create the PEOE model. It reads ``ATOMTYPE_GAS.DEF`` and ``GASPARM.DAT``.
    #[new]
    fn new() -> Self {
        Self
    }

    /// ``False`` — the PEOE charges are already symmetric by construction.
    fn needs_equivalencing(&self) -> bool {
        ChargeModel::needs_equivalencing(&GasteigerModel)
    }

    /// The Gasteiger/PEOE charges of every atom, hydrogens included.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecule; left untouched.
    /// qm : ndarray of float64, optional
    ///     Ignored — this model needs no QM base charges. The argument exists only so
    ///     that every charge model answers the same call.
    ///
    /// Returns
    /// -------
    /// ndarray of float64, shape (n_atoms,)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``ATOMTYPE_GAS.DEF`` cannot type an atom, or ``GASPARM.DAT`` has no row
    ///     for the type it assigned. An atom with no χ curve has no charge — it is not
    ///     defaulted to zero.
    #[pyo3(signature = (mol, qm=None))]
    fn assign<'py>(
        &self,
        py: Python<'py>,
        mol: &PyAtomistic,
        qm: Option<ArrayArg1<'_, f64>>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let qm = qm_charges(qm);
        let charges = GasteigerModel
            .assign(mol.core(), qm.as_deref())
            .map_err(charge_err)?;
        Ok(charges_out(py, charges))
    }

    fn __repr__(&self) -> String {
        "GasteigerModel()".to_string()
    }
}

/// Register `molpy.ff.charge`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBccModel>()?;
    m.add_class::<PyMullikenModel>()?;
    m.add_class::<PyGasteigerModel>()?;
    Ok(())
}
