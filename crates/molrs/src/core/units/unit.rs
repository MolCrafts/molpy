//! A resolved unit value, detached from the registry after parse.

use std::fmt;
use std::str::FromStr;

use crate::op::F;

use super::dimension::Dimension;
use super::error::UnitsError;

/// A resolved unit: conversion to SI base is `si = value * factor + offset`.
///
/// A `Unit` is self-contained — produced by [`UnitRegistry::parse`] (or
/// `str::parse`) and usable without further registry access. The `offset`
/// is non-zero only for affine units (e.g. `degC`, offset 273.15 K), which
/// are rejected by all multiplicative operations.
///
/// [`UnitRegistry::parse`]: super::registry::UnitRegistry::parse
#[derive(Clone, PartialEq, Debug)]
pub struct Unit {
    pub(crate) factor: F,
    pub(crate) offset: F,
    pub(crate) dimension: Dimension,
    pub(crate) name: String,
}

impl Unit {
    /// Construct a unit from its parts. Primarily for internal/parser use.
    pub(crate) fn new(factor: F, offset: F, dimension: Dimension, name: String) -> Unit {
        Unit {
            factor,
            offset,
            dimension,
            name,
        }
    }

    /// Rebuild a resolved unit from its complete, registry-independent state.
    ///
    /// This is primarily useful for persistence layers. Normal callers should
    /// obtain units from [`UnitRegistry`](super::UnitRegistry), which validates
    /// names and compound expressions.
    pub fn from_parts(factor: F, offset: F, dimension: Dimension, name: String) -> Unit {
        Unit::new(factor, offset, dimension, name)
    }

    /// Multiplicative factor from this unit to its SI base representation.
    pub fn factor(&self) -> F {
        self.factor
    }

    /// Additive offset in the SI base representation.
    pub fn offset(&self) -> F {
        self.offset
    }

    /// Canonical display name retained by this resolved unit.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The dimension of this unit.
    pub fn dimension(&self) -> Dimension {
        self.dimension
    }

    /// True when this unit carries a non-zero additive offset (e.g. °C).
    pub fn is_affine(&self) -> bool {
        self.offset != 0.0
    }

    /// Multiplicative factor converting a value in `self` to a value in `other`.
    ///
    /// Use this for bulk conversion: compute the scalar once, then scale an
    /// entire array, instead of building a [`Quantity`](super::Quantity) per
    /// element.
    ///
    /// # Errors
    ///
    /// - [`UnitsError::DimensionMismatch`] — the two units have different
    ///   dimensions.
    /// - [`UnitsError::AffineUnit`] — either unit carries a non-zero offset
    ///   (affine conversion is not a pure scaling; use
    ///   [`Quantity::to`](super::Quantity::to) instead).
    ///
    /// # Examples
    ///
    /// Batch-rescale coordinates from nm to Å (factor 10):
    ///
    /// ```
    /// use molrs::core::{UnitRegistry, UnitsError};
    ///
    /// let reg = UnitRegistry::new();
    /// let nm = reg.parse("nm")?;
    /// let ang = reg.parse("angstrom")?;
    /// let scale = nm.factor_to(&ang)?;
    ///
    /// let coords_nm = [0.1, 0.2, 0.3]; // flat [x0, y0, z0] in nm
    /// let coords_ang: Vec<f64> = coords_nm.iter().map(|x| x * scale).collect();
    /// assert!((coords_ang[0] - 1.0).abs() < 1e-12);
    /// # Ok::<(), UnitsError>(())
    /// ```
    pub fn factor_to(&self, other: &Unit) -> Result<F, UnitsError> {
        if self.dimension != other.dimension {
            return Err(UnitsError::DimensionMismatch {
                left: self.dimension,
                right: other.dimension,
            });
        }
        if self.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: self.name.clone(),
                operation: "factor_to",
            });
        }
        if other.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: other.name.clone(),
                operation: "factor_to",
            });
        }
        Ok(self.factor / other.factor)
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl FromStr for Unit {
    type Err = UnitsError;

    /// Parse against the global preloaded registry
    /// ([`UnitRegistry::global`](super::registry::UnitRegistry::global)).
    ///
    /// # Errors
    ///
    /// Same as [`UnitRegistry::parse`](super::registry::UnitRegistry::parse).
    fn from_str(s: &str) -> Result<Unit, UnitsError> {
        super::registry::UnitRegistry::global().parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::UnitRegistry;

    fn unit(expr: &str) -> Unit {
        UnitRegistry::new().parse(expr).unwrap()
    }

    #[test]
    fn factor_to_same_unit_is_exactly_one() {
        // x/x == 1.0 exactly in IEEE-754 for any finite nonzero factor.
        for expr in ["m", "kcal/mol", "eV", "bohr", "atm"] {
            let u = unit(expr);
            assert_eq!(u.factor_to(&u).unwrap(), 1.0, "factor_to self for {expr}");
        }
    }

    #[test]
    fn factor_to_dimension_mismatch_errors() {
        let err = unit("m").factor_to(&unit("s")).unwrap_err();
        assert!(
            matches!(err, UnitsError::DimensionMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn factor_to_affine_self_errors() {
        let err = unit("degC").factor_to(&unit("K")).unwrap_err();
        assert!(
            matches!(
                err,
                UnitsError::AffineUnit {
                    operation: "factor_to",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn factor_to_affine_other_errors() {
        let err = unit("K").factor_to(&unit("degC")).unwrap_err();
        assert!(
            matches!(
                err,
                UnitsError::AffineUnit {
                    operation: "factor_to",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn from_str_parses_via_global_registry() {
        let u: Unit = "nm".parse().unwrap();
        assert_eq!(u.dimension(), Dimension::LENGTH);
        assert!((u.factor - 1e-9).abs() < 1e-23);
    }

    #[test]
    fn from_str_unknown_unit_errors() {
        let err = "zorp".parse::<Unit>().unwrap_err();
        assert!(matches!(err, UnitsError::UnknownUnit { .. }), "got {err:?}");
    }

    #[test]
    fn display_prints_canonical_name() {
        assert_eq!(unit("kcal/mol").to_string(), "kcal * mol^-1");
        assert_eq!(unit("m").to_string(), "m");
    }

    #[test]
    fn is_affine_predicate() {
        assert!(unit("degC").is_affine());
        assert!(!unit("K").is_affine());
    }
}
