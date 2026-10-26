//! Python binding for the ATD atom typifier (`molrs::ff::typifier::atd`).
//!
//! One rule engine, seven `ATOMTYPE_*.DEF` tables. The table is chosen by name at
//! construction and there is **no default**: seven exist, they disagree, and picking
//! one silently would be picking for the caller.
//!
//! # The names are antechamber's `-at` flags, not the tables' file names
//!
//! `antechamber -at gaff` walks `ATOMTYPE_GFF.DEF`, and the Rust enum is named after
//! the *table* ([`AtdParameterSet::Gff`]) while the acceptance contract words the
//! Python argument as the *flag* (`AtdTypifier(parameter_set="gaff")`). The two
//! spellings differ for exactly the two GAFF columns, which is precisely where a
//! wrong-set binding would hide — so the mapping is written down once, in
//! [`AtdParameterSet::name`] / [`AtdParameterSet::from_name`], and read back
//! out by the `parameter_set` getter.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use molrs::ff::typifier::{AtdParameterSet, AtdTypifier};

use crate::perceive::PyBondOrderSource;

use super::PyTypifier;

/// Antechamber atom typifier — `molpy.ff.typifier.AtdTypifier`.
///
/// One rule engine bound to one `ATOMTYPE_*.DEF` table. :meth:`typify` perceives
/// antechamber bond types, derives the facts each rule can ask about, and labels
/// every atom with the first rule of the table that matches it.
///
/// Graph in / graph out: :meth:`typify` returns a clone of ``mol`` with the
/// table's atom type in ``keys.TYPE`` on every atom and the perceived
/// ``bcc_bond_type`` on every bond; the caller's handles still address their
/// atoms on it, and ``mol`` is left untouched — the standard AM1-BCC workflow
/// needs the caller's force-field types *and* these at once. It raises
/// ``ValueError`` if the molecule's facts cannot be derived, or if an atom
/// matches no rule of the table — including one the table's catch-all row
/// would label ``"DU"``, which names no type.
///
/// Parameters
/// ----------
/// parameter_set : str
///     The antechamber ``-at`` flag naming the table: ``"bcc"``, ``"abcg2"``,
///     ``"gas"``, ``"gaff"``, ``"gaff2"``, ``"amber"`` or ``"sybyl"``. Required —
///     there is no default table.
/// bond_orders : BondOrderSource, default ``BondOrderSource.Connectivity``
///     Which bond orders the types follow (``AtdTypifier::with_bond_orders``;
///     the core's default is ``Connectivity``). ``Connectivity`` judges them
///     from the connectivity alone, as antechamber does by default
///     (``bondtype -j full``): the molecule's own orders are ignored, and on a
///     molecule with two Kekulé structures (azulene, cyclooctatetraene) the
///     colouring (``cc`` / ``cd`` …) is the one antechamber's search settles
///     on for the same atom and bond order. Every hydrogen must be drawn.
///     ``Stated`` keeps the molecule's own orders (aromatic bonds without one
///     are kekulized).
///
/// Raises
/// ------
/// ValueError
///     If ``parameter_set`` names no table.
///
/// Examples
/// --------
/// >>> typed = molpy.ff.typifier.AtdTypifier(parameter_set="gaff").typify(benzene)
/// >>> typed.get(carbon, molpy.core.keys.TYPE)
/// 'ca'
#[pyclass(module = "molpy.ff.typifier", name = "AtdTypifier", extends = PyTypifier, subclass)]
#[derive(Debug)]
pub struct PyAtdTypifier {
    parameter_set: AtdParameterSet,
    bond_orders: PyBondOrderSource,
}

#[pymethods]
impl PyAtdTypifier {
    /// Bind the engine to the table `parameter_set` names.
    #[new]
    #[pyo3(signature = (*, parameter_set, bond_orders = PyBondOrderSource::Connectivity))]
    fn new(parameter_set: &str, bond_orders: PyBondOrderSource) -> PyResult<(Self, PyTypifier)> {
        let parameter_set =
            AtdParameterSet::from_name(parameter_set).map_err(PyValueError::new_err)?;
        Ok((
            Self {
                parameter_set,
                bond_orders: bond_orders.clone(),
            },
            PyTypifier::native(
                AtdTypifier::new(parameter_set).with_bond_orders(bond_orders.into()),
            ),
        ))
    }

    /// The antechamber ``-at`` flag of the table this typifier walks.
    #[getter]
    fn parameter_set(&self) -> &'static str {
        self.parameter_set.name()
    }

    /// Which bond orders the types follow.
    #[getter]
    fn bond_orders(&self) -> PyBondOrderSource {
        self.bond_orders.clone()
    }

    fn __repr__(&self) -> String {
        let source = match self.bond_orders {
            PyBondOrderSource::Stated => "Stated",
            PyBondOrderSource::Connectivity => "Connectivity",
        };
        format!(
            "AtdTypifier(parameter_set='{}', bond_orders=BondOrderSource.{source})",
            self.parameter_set(),
        )
    }
}
