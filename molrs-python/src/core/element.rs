//! Python binding for the Rust-owned periodic table.
//!
//! An `Element` is looked up by exactly one key per call —
//! [`PyElement::by_symbol`] or [`PyElement::by_number`] — and the frame-level
//! species conversions are the two explicit static steps
//! [`PyElement::fill_atomic_numbers`] / [`PyElement::fill_symbols`]. molrs
//! converts between an `element` symbol and an `atomic_number` nowhere else.

use molrs::core::Element;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyTuple};

use crate::core::frame::PyFrame;
use crate::error::molrs_error_to_pyerr;

/// Immutable chemical-element record backed by the Rust periodic table.
///
/// Not constructed directly: look one up with :meth:`by_symbol` or
/// :meth:`by_number`.
#[pyclass(
    module = "molrs.core",
    name = "Element",
    frozen,
    eq,
    hash,
    skip_from_py_object
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PyElement {
    inner: Element,
}

#[pymethods]
impl PyElement {
    /// The element whose symbol is ``symbol`` (case-insensitive).
    ///
    /// Parameters
    /// ----------
    /// symbol : str
    ///     A chemical symbol, e.g. ``"C"`` or ``"fe"``.
    ///
    /// Returns
    /// -------
    /// Element
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``symbol`` is no element's symbol (the wildcard ``"*"`` is
    ///     none: its atomic number 0 names no element).
    #[staticmethod]
    fn by_symbol(symbol: &str) -> PyResult<Self> {
        Element::by_symbol(symbol)
            .map(|inner| Self { inner })
            .ok_or_else(|| PyValueError::new_err(format!("'{symbol}' is not an element symbol")))
    }

    /// The element whose atomic number is ``z``.
    ///
    /// Parameters
    /// ----------
    /// z : int
    ///     An atomic number, ``1`` to ``118``.
    ///
    /// Returns
    /// -------
    /// Element
    ///
    /// Raises
    /// ------
    /// TypeError
    ///     If ``z`` is a ``bool`` rather than an ``int``.
    /// ValueError
    ///     If ``z`` is outside ``1..=118`` (``0``, the wildcard, names no
    ///     element).
    #[staticmethod]
    fn by_number(z: &Bound<'_, PyAny>) -> PyResult<Self> {
        if z.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err(
                "an atomic number is an int, not a bool",
            ));
        }
        let number: i64 = z.extract()?;
        u8::try_from(number)
            .ok()
            .and_then(Element::by_number)
            .map(|inner| Self { inner })
            .ok_or_else(|| {
                PyValueError::new_err(format!("{number} is not an atomic number (1..=118)"))
            })
    }

    /// Write ``atoms.atomic_number`` from the ``atoms.element`` symbols, in
    /// place.
    ///
    /// The explicit step from a symbol-spelling file (XYZ, PDB, SDF, …) to
    /// the field every molrs consumer reads. Symbols match
    /// case-insensitively; ``"*"`` is the wildcard, atomic number 0. An
    /// existing ``atomic_number`` column is replaced; ``element`` is kept.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     The frame to convert; its ``atoms`` block gains the column.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no ``atoms`` block, the block has no string
    ///     ``element`` column, or a row holds a symbol that is no element
    ///     (the message names the row).
    #[staticmethod]
    fn fill_atomic_numbers(frame: &PyFrame) -> PyResult<()> {
        frame
            .with_frame_mut(Element::fill_atomic_numbers)?
            .map_err(molrs_error_to_pyerr)
    }

    /// Write ``atoms.element`` symbols from ``atoms.atomic_number``, in place.
    ///
    /// The inverse explicit step, for a writer whose format spells species
    /// as symbols; atomic number 0 becomes the wildcard ``"*"``. An existing
    /// ``element`` column is replaced; ``atomic_number`` is kept.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     The frame to convert; its ``atoms`` block gains the column.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame has no ``atoms`` block, the block has no unsigned
    ///     ``atomic_number`` column, or a row holds a number outside
    ///     ``0..=118`` (the message names the row).
    #[staticmethod]
    fn fill_symbols(frame: &PyFrame) -> PyResult<()> {
        frame
            .with_frame_mut(Element::fill_symbols)?
            .map_err(molrs_error_to_pyerr)
    }

    /// Atomic number.
    #[getter]
    fn z(&self) -> u8 {
        self.inner.z()
    }

    /// Canonical English name.
    #[getter]
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    /// Canonical chemical symbol.
    #[getter]
    fn symbol(&self) -> &'static str {
        self.inner.symbol()
    }

    /// Standard atomic mass in daltons.
    #[getter]
    fn mass(&self) -> f64 {
        self.inner.atomic_mass() as f64
    }

    /// Van der Waals radius in angstroms.
    #[getter]
    fn vdw(&self) -> f64 {
        self.inner.vdw_radius() as f64
    }

    /// Single-bond covalent radius in angstroms.
    #[getter]
    fn covalent(&self) -> f64 {
        self.inner.covalent_radius() as f64
    }

    /// Pickle as ``Element.by_number(z)``.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyTuple>> {
        let py = slf.py();
        let by_number = py
            .get_type::<Self>()
            .getattr(pyo3::intern!(py, "by_number"))?;
        let args = PyTuple::new(py, [slf.get().inner.z()])?;
        PyTuple::new(py, [by_number, args.into_any()])
    }

    fn __repr__(&self) -> String {
        format!("<Element {}>", self.inner.symbol())
    }
}
