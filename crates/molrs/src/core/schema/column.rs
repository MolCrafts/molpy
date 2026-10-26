//! Canonical column vocabulary: what a column key means and how it is stored.

use crate::core::DType;
use crate::core::PresetDim;
use crate::core::UnitPreset;

/// Structural shape of a column beyond axis 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColShape {
    /// `(nrows,)` — one value per row.
    Scalar,
    /// `(nrows, n)` — a fixed-width vector per row.
    Vec(usize),
}

impl ColShape {
    /// Whether an ndarray shape is admissible for this spec.
    pub fn admits(&self, shape: &[usize]) -> bool {
        match self {
            ColShape::Scalar => shape.len() == 1,
            ColShape::Vec(n) => shape.len() == 2 && shape[1] == *n,
        }
    }
}

impl std::fmt::Display for ColShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColShape::Scalar => write!(f, "scalar"),
            ColShape::Vec(n) => write!(f, "vec({n})"),
        }
    }
}

/// Physical dimension of a column, over the ten [`PresetDim`]s.
///
/// This is the single truth for what a column measures: the schema document
/// derives the unit it displays from it. molrs stores raw numbers; the
/// dimension says what they measure, and the unit is that dimension's unit in
/// whichever preset the frame is in (its `units` meta entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnDimension {
    /// Not a physical quantity: identifiers, labels, flags, codes, endpoints.
    NotAQuantity,
    /// A pure number with the same value in every preset (quaternion
    /// components).
    Dimensionless,
    /// One preset dimension.
    Of(PresetDim),
    /// The product of two preset dimensions (a dipole is charge × length).
    Product(PresetDim, PresetDim),
}

impl ColumnDimension {
    /// The unit this dimension takes in `preset`.
    ///
    /// `None` for [`NotAQuantity`](Self::NotAQuantity), `""` for
    /// [`Dimensionless`](Self::Dimensionless), the preset's unit for
    /// [`Of`](Self::Of), and `"<a> * <b>"` for [`Product`](Self::Product).
    /// `None` also when the preset lacks a named dimension.
    pub fn unit_in(&self, preset: &UnitPreset) -> Option<String> {
        match *self {
            ColumnDimension::NotAQuantity => None,
            ColumnDimension::Dimensionless => Some(String::new()),
            ColumnDimension::Of(d) => preset.unit(d.name()).map(str::to_owned),
            ColumnDimension::Product(a, b) => {
                let ua = preset.unit(a.name())?;
                let ub = preset.unit(b.name())?;
                Some(format!("{ua} * {ub}"))
            }
        }
    }
}

impl std::fmt::Display for ColumnDimension {
    /// Lower-case dimension names: `""`, `"dimensionless"`, `"length"`,
    /// `"charge * length"`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ColumnDimension::NotAQuantity => Ok(()),
            ColumnDimension::Dimensionless => write!(f, "dimensionless"),
            ColumnDimension::Of(d) => write!(f, "{}", d.name()),
            ColumnDimension::Product(a, b) => write!(f, "{} * {}", a.name(), b.name()),
        }
    }
}

/// One canonical column of the Frame vocabulary.
///
/// A spec binds a **column key**, wherever that key appears — in `atoms`, in
/// `bonds`, in a relation block `MolGraph::to_frame` minted on the fly. That is
/// what lets [`Block::insert`](crate::core::Block::insert) enforce
/// dtype without knowing which block it is about to live in.
///
/// The vocabulary is closed; the *block* set is open. A key with no spec is
/// unconstrained — that is the extension point for perceived facts,
/// per-instance force-field parameters, and format-local columns.
#[derive(Debug, Clone, Copy)]
pub struct ColumnSpec {
    /// Canonical key as it appears in a `Block` (`"x"`, `"atomi"`).
    pub key: &'static str,
    /// Rust/Python constant name (`"X"`, `"ATOMI"`). Emitted with the column
    /// table from the same declaration (`stringify!` of that identifier), and
    /// exported as `molrs.core.keys.<CONST>` by the Python binding.
    pub const_name: &'static str,
    /// The one admissible storage dtype. Not a set — see the module doc on
    /// [`super`] for why a key that needs two dtypes is two keys.
    pub dtype: DType,
    /// Shape beyond axis 0.
    pub shape: ColShape,
    /// Physical dimension. Drives the displayed unit; never enforced on
    /// write — molrs stores raw numbers.
    pub dimension: ColumnDimension,
    /// One-line meaning. Never empty (asserted by the vocabulary gate).
    pub doc: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::PresetDim;
    use crate::core::UnitPreset;

    #[test]
    fn unit_in_not_a_quantity_is_none() {
        assert_eq!(
            ColumnDimension::NotAQuantity.unit_in(&UnitPreset::real()),
            None
        );
    }

    #[test]
    fn unit_in_dimensionless_is_empty() {
        assert_eq!(
            ColumnDimension::Dimensionless
                .unit_in(&UnitPreset::real())
                .as_deref(),
            Some("")
        );
    }

    #[test]
    fn unit_in_of_is_the_preset_unit() {
        let real = UnitPreset::real();
        assert_eq!(
            ColumnDimension::Of(PresetDim::Length)
                .unit_in(&real)
                .as_deref(),
            Some("angstrom")
        );
        assert_eq!(
            ColumnDimension::Of(PresetDim::Velocity)
                .unit_in(&UnitPreset::metal())
                .as_deref(),
            Some("angstrom / picosecond")
        );
    }

    #[test]
    fn unit_in_product_joins_both_units_with_a_star() {
        assert_eq!(
            ColumnDimension::Product(PresetDim::Charge, PresetDim::Length)
                .unit_in(&UnitPreset::real())
                .as_deref(),
            Some("elementary_charge * angstrom")
        );
    }
}
