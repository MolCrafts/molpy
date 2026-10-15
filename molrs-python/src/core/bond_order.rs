//! `molrs.core.BondOrder` / `molrs.core.BondNumber` — `molrs::core`'s two
//! orthogonal facts about a bond: its chemical class and the integer bond
//! number of a localized (Kekulé) structure. Each is stored as its integer
//! code (`keys.BOND_TYPE` / `keys.BOND_NUMBER`); `int(member)` is that code.

use molrs::core::{BondNumber, BondOrder};
use pyo3::prelude::*;

/// The chemical class of a bond — ``molrs::core::BondOrder``.
///
/// Stored under ``keys.BOND_TYPE`` as its code: ``Unknown`` 0, ``Single`` 1,
/// ``Double`` 2, ``Triple`` 3, ``Aromatic`` 4. ``Aromatic`` is a class, peer
/// to single / double / triple, not a number: an aromatic bond's localized
/// number is a :class:`BondNumber`.
#[pyclass(
    module = "molrs.core",
    name = "BondOrder",
    eq,
    eq_int,
    frozen,
    hash,
    skip_from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyBondOrder {
    Unknown = 0,
    Single = 1,
    Double = 2,
    Triple = 3,
    Aromatic = 4,
}

impl From<BondOrder> for PyBondOrder {
    fn from(order: BondOrder) -> Self {
        match order {
            BondOrder::Unknown => Self::Unknown,
            BondOrder::Single => Self::Single,
            BondOrder::Double => Self::Double,
            BondOrder::Triple => Self::Triple,
            BondOrder::Aromatic => Self::Aromatic,
        }
    }
}

impl From<PyBondOrder> for BondOrder {
    fn from(order: PyBondOrder) -> Self {
        BondOrder::from_code(order as u32)
    }
}

#[pymethods]
impl PyBondOrder {
    /// The stored code (``keys.BOND_TYPE``).
    #[getter]
    fn code(&self) -> u32 {
        BondOrder::from(*self).code()
    }

    /// The class a stored code names; a code outside ``0..=4`` is ``Unknown``.
    #[staticmethod]
    fn from_code(code: u32) -> Self {
        BondOrder::from_code(code).into()
    }

    /// Whether this is an aromatic bond.
    fn is_aromatic(&self) -> bool {
        BondOrder::from(*self).is_aromatic()
    }

    /// The bond number a non-aromatic class implies; ``None`` for
    /// ``Aromatic`` and ``Unknown``.
    fn implied_number(&self) -> Option<PyBondNumber> {
        BondOrder::from(*self).implied_number().map(Into::into)
    }
}

/// The integer bond number of a localized Lewis / Kekulé structure —
/// ``molrs::core::BondNumber``.
///
/// Stored under ``keys.BOND_NUMBER`` as its code: ``Unknown`` 0, ``Single``
/// 1, ``Double`` 2, ``Triple`` 3, ``Quadruple`` 4.
#[pyclass(
    module = "molrs.core",
    name = "BondNumber",
    eq,
    eq_int,
    frozen,
    hash,
    skip_from_py_object
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyBondNumber {
    Unknown = 0,
    Single = 1,
    Double = 2,
    Triple = 3,
    Quadruple = 4,
}

impl From<BondNumber> for PyBondNumber {
    fn from(number: BondNumber) -> Self {
        match number {
            BondNumber::Unknown => Self::Unknown,
            BondNumber::Single => Self::Single,
            BondNumber::Double => Self::Double,
            BondNumber::Triple => Self::Triple,
            BondNumber::Quadruple => Self::Quadruple,
        }
    }
}

#[pymethods]
impl PyBondNumber {
    /// The stored code (``keys.BOND_NUMBER``).
    #[getter]
    fn code(&self) -> u32 {
        BondNumber::from_code(*self as u32).code()
    }

    /// The number a stored code names; a code outside ``0..=4`` is
    /// ``Unknown``.
    #[staticmethod]
    fn from_code(code: u32) -> Self {
        BondNumber::from_code(code).into()
    }

    /// The bond class this number implies — the inverse of
    /// :meth:`BondOrder.implied_number`. ``Quadruple`` is classed ``Double``;
    /// ``Unknown`` implies ``Unknown``.
    fn implied_type(&self) -> PyBondOrder {
        BondNumber::from_code(*self as u32).implied_type().into()
    }

    /// The number as a count, for valence sums; ``Unknown`` counts as 0.
    fn count(&self) -> u32 {
        BondNumber::from_code(*self as u32).count()
    }
}
