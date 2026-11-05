//! Python wrapper for 2D depiction (`molrs::depict`).
//!
//! The public surface is [`PyDepictor`] (`molpy.depict.Depictor`): its
//! constructor takes the bond length to draw and [`PyDepictor::layout`]
//! returns a copy of the molecule with `x` / `y` drawing coordinates and
//! `z = 0`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use molrs::depict::Depictor;

use crate::core::molgraph::PyAtomistic;
use crate::error::molrs_error_to_pyerr;

/// 2D drawing coordinates for a molecular graph — RDKit's
/// ``Compute2DCoords`` in purpose.
///
/// Rings are regular polygons (fused rings share an edge, spiro rings an
/// atom), chains a 120° zigzag, sp centres straight, substituents spread into
/// the free angle, and disconnected species side by side. The layout is
/// deterministic.
///
/// Parameters
/// ----------
/// bond_length : float, optional
///     Length every bond is drawn at (default ``1.5``, RDKit's unit).
///
/// Raises
/// ------
/// ValueError
///     If ``bond_length`` is not finite and positive.
///
/// Examples
/// --------
/// >>> mol = molpy.io.smiles.read_smiles("c1ccccc1O").to_atomistic()
/// >>> flat = Depictor().layout(mol)
/// >>> frame = flat.to_frame()   # x, y for a 2D viewer; z is 0
#[pyclass(
    module = "molpy.depict",
    name = "Depictor",
    subclass,
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDepictor {
    inner: Depictor,
}

#[pymethods]
impl PyDepictor {
    #[new]
    #[pyo3(signature = (bond_length=1.5))]
    fn new(bond_length: f64) -> PyResult<Self> {
        Ok(Self {
            inner: Depictor::new(bond_length).map_err(molrs_error_to_pyerr)?,
        })
    }

    /// The bond length this depictor draws.
    #[getter]
    fn bond_length(&self) -> f64 {
        self.inner.bond_length()
    }

    /// Lay a molecule out in 2D.
    ///
    /// The input is not modified. The result is an :class:`~molpy.core.Atomistic`
    /// with ``x`` / ``y`` drawing coordinates and ``z = 0`` on every atom,
    /// replacing any coordinates the input had; every other property
    /// (formal charges, types, …) and every relation (ports included) is
    /// carried across. Hydrogens present as atoms are laid out too.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     The molecular graph.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///     The laid-out copy.
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``mol`` is not an ``Atomistic``.
    fn layout(&self, py: Python<'_>, mol: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(leaf) = mol.cast::<PyAtomistic>() {
            let out = self
                .inner
                .layout(leaf.borrow().core())
                .map_err(molrs_error_to_pyerr)?;
            return Ok(leaf.borrow().derive(py, out)?.into_any());
        }
        Err(PyTypeError::new_err(format!(
            "Depictor.layout expects an Atomistic, got {}",
            mol.get_type().name()?
        )))
    }

    fn __repr__(&self) -> String {
        format!("Depictor(bond_length={})", self.inner.bond_length())
    }
}

/// Register `molpy.depict`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDepictor>()?;
    Ok(())
}
