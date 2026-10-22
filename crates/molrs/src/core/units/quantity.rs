//! A magnitude paired with a unit, with dimension-checked operations.

use std::fmt;
use std::ops::{Div, Mul, Neg};

use crate::op::F;

use super::dimension::Dimension;
use super::error::UnitsError;
use super::registry::UnitRegistry;
use super::unit::Unit;

/// A value with an attached unit.
///
/// All arithmetic is dimension-checked: additive operations convert the
/// right-hand side into `self`'s unit first, multiplicative operations
/// compose dimensions and reject affine units. Scaling by a bare scalar
/// (`Mul<F>` / `Div<F>` / `Neg`) keeps the unit unchanged.
#[derive(Clone, PartialEq, Debug)]
pub struct Quantity {
    pub(crate) value: F,
    pub(crate) unit: Unit,
}

impl Quantity {
    /// Construct a quantity from a magnitude and a unit.
    pub fn new(value: F, unit: Unit) -> Quantity {
        Quantity { value, unit }
    }

    /// The magnitude expressed in `self.unit()`.
    pub fn value(&self) -> F {
        self.value
    }

    /// The unit of this quantity.
    pub fn unit(&self) -> &Unit {
        &self.unit
    }

    /// Convert to a target unit (handles affine °C↔K correctly).
    ///
    /// Conversion goes through SI base: `si = value * factor + offset`,
    /// then `target = (si - offset') / factor'`.
    ///
    /// # Errors
    ///
    /// [`UnitsError::DimensionMismatch`] if the target unit has a different
    /// dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::{UnitRegistry, UnitsError};
    ///
    /// let reg = UnitRegistry::new();
    /// let t = reg.quantity(25.0, "degC")?;
    /// let kelvin = t.to(&reg.parse("K")?)?;
    /// assert!((kelvin.value() - 298.15).abs() < 1e-12);
    /// # Ok::<(), UnitsError>(())
    /// ```
    pub fn to(&self, unit: &Unit) -> Result<Quantity, UnitsError> {
        if self.unit.dimension != unit.dimension {
            return Err(UnitsError::DimensionMismatch {
                left: self.unit.dimension,
                right: unit.dimension,
            });
        }
        let si = self.value * self.unit.factor + self.unit.offset;
        let value = (si - unit.offset) / unit.factor;
        Ok(Quantity::new(value, unit.clone()))
    }

    /// Convert to a target unit named by a string (uses the global registry).
    ///
    /// # Errors
    ///
    /// - Parse errors from [`UnitRegistry::parse`] (unknown unit, malformed
    ///   expression, affine unit in a compound).
    /// - [`UnitsError::DimensionMismatch`] if the parsed unit has a
    ///   different dimension.
    pub fn to_parsed(&self, expr: &str) -> Result<Quantity, UnitsError> {
        self.to(&UnitRegistry::global().parse(expr)?)
    }

    /// Convert to the SI base representation of this quantity's dimension.
    pub fn to_base_units(&self) -> Quantity {
        let dim = self.unit.dimension;
        let base = Unit::new(1.0, 0.0, dim, si_base_name(dim));
        let si = self.value * self.unit.factor + self.unit.offset;
        Quantity::new(si, base)
    }

    /// Add `rhs` (converted to `self`'s unit); the result keeps `self`'s unit.
    ///
    /// Affine operands are treated as absolute points on their scale and
    /// converted through SI before adding (`37 degC + 1 K` reads the kelvin
    /// operand as an absolute temperature, not a temperature difference).
    /// This differs from pint, which requires explicit delta units; molrs
    /// has no delta semantics yet.
    ///
    /// # Errors
    ///
    /// [`UnitsError::DimensionMismatch`] if `rhs` has a different dimension.
    pub fn try_add(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let rhs = rhs.to(&self.unit)?;
        Ok(Quantity::new(self.value + rhs.value, self.unit.clone()))
    }

    /// Subtract `rhs` (converted to `self`'s unit); the result keeps `self`'s unit.
    ///
    /// # Errors
    ///
    /// [`UnitsError::DimensionMismatch`] if `rhs` has a different dimension.
    pub fn try_sub(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let rhs = rhs.to(&self.unit)?;
        Ok(Quantity::new(self.value - rhs.value, self.unit.clone()))
    }

    /// Multiply two quantities, composing dimensions and factors.
    ///
    /// # Errors
    ///
    /// [`UnitsError::AffineUnit`] if either operand's unit carries an offset
    /// (e.g. `degC`) — products of affine units are ill-defined.
    pub fn try_mul(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let (lhs_unit, rhs_unit) = check_multiplicative(&self.unit, &rhs.unit, "mul")?;
        let unit = Unit::new(
            lhs_unit.factor * rhs_unit.factor,
            0.0,
            lhs_unit.dimension * rhs_unit.dimension,
            format!("{} * {}", lhs_unit.name, rhs_unit.name),
        );
        Ok(Quantity::new(self.value * rhs.value, unit))
    }

    /// Divide two quantities, composing dimensions and factors.
    ///
    /// # Errors
    ///
    /// [`UnitsError::AffineUnit`] if either operand's unit carries an offset
    /// (e.g. `degC`) — ratios of affine units are ill-defined.
    pub fn try_div(&self, rhs: &Quantity) -> Result<Quantity, UnitsError> {
        let (lhs_unit, rhs_unit) = check_multiplicative(&self.unit, &rhs.unit, "div")?;
        let unit = Unit::new(
            lhs_unit.factor / rhs_unit.factor,
            0.0,
            lhs_unit.dimension / rhs_unit.dimension,
            format!("{} / ({})", lhs_unit.name, rhs_unit.name),
        );
        Ok(Quantity::new(self.value / rhs.value, unit))
    }
}

/// Reject affine operands for multiplicative operations.
fn check_multiplicative<'a>(
    lhs: &'a Unit,
    rhs: &'a Unit,
    operation: &'static str,
) -> Result<(&'a Unit, &'a Unit), UnitsError> {
    for unit in [lhs, rhs] {
        if unit.is_affine() {
            return Err(UnitsError::AffineUnit {
                name: unit.name.clone(),
                operation,
            });
        }
    }
    Ok((lhs, rhs))
}

/// Canonical SI-base display name for a dimension, e.g. `m^2 * kg * s^-2`.
fn si_base_name(dim: Dimension) -> String {
    const BASE_SYMBOLS: [&str; 7] = ["m", "kg", "s", "A", "K", "mol", "cd"];
    let parts: Vec<String> = BASE_SYMBOLS
        .iter()
        .zip(dim.exponents())
        .filter(|(_, exp)| *exp != 0)
        .map(|(sym, exp)| {
            if exp == 1 {
                (*sym).to_string()
            } else {
                format!("{}^{}", sym, exp)
            }
        })
        .collect();
    if parts.is_empty() {
        "1".to_string()
    } else {
        parts.join(" * ")
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

impl Neg for Quantity {
    type Output = Quantity;
    fn neg(self) -> Quantity {
        Quantity::new(-self.value, self.unit)
    }
}

impl Mul<F> for Quantity {
    type Output = Quantity;
    fn mul(self, rhs: F) -> Quantity {
        Quantity::new(self.value * rhs, self.unit)
    }
}

impl Div<F> for Quantity {
    type Output = Quantity;
    fn div(self, rhs: F) -> Quantity {
        Quantity::new(self.value / rhs, self.unit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(expr: &str) -> Unit {
        UnitRegistry::new().parse(expr).unwrap()
    }

    fn qty(value: F, expr: &str) -> Quantity {
        Quantity::new(value, unit(expr))
    }

    #[test]
    fn try_sub_autoconverts_units() {
        // 1 kcal - 500 cal = 0.5 kcal; the 500/1000 ratio is exact in IEEE-754.
        let a = qty(1.0, "kcal");
        let b = qty(500.0, "cal");
        let diff = a.try_sub(&b).unwrap();
        assert_eq!(diff.value(), 0.5);
        assert_eq!(diff.unit(), a.unit());
    }

    #[test]
    fn try_sub_dimension_mismatch_errors() {
        let err = qty(1.0, "kcal").try_sub(&qty(1.0, "m")).unwrap_err();
        assert!(
            matches!(err, UnitsError::DimensionMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn try_mul_affine_operand_errors() {
        let err = qty(25.0, "degC").try_mul(&qty(2.0, "m")).unwrap_err();
        assert!(
            matches!(
                err,
                UnitsError::AffineUnit {
                    operation: "mul",
                    ..
                }
            ),
            "got {err:?}"
        );
        // affine on the rhs as well
        let err = qty(2.0, "m").try_mul(&qty(25.0, "degC")).unwrap_err();
        assert!(matches!(err, UnitsError::AffineUnit { .. }), "got {err:?}");
    }

    #[test]
    fn try_div_affine_operand_errors() {
        let err = qty(25.0, "degC").try_div(&qty(2.0, "m")).unwrap_err();
        assert!(
            matches!(
                err,
                UnitsError::AffineUnit {
                    operation: "div",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn try_div_by_zero_value_yields_infinity() {
        // No panic in the public API: IEEE-754 semantics, value becomes +inf.
        let q = qty(6.0, "m").try_div(&qty(0.0, "s")).unwrap();
        assert!(
            q.value().is_infinite() && q.value() > 0.0,
            "got {}",
            q.value()
        );
        assert_eq!(q.unit().dimension(), Dimension::LENGTH / Dimension::TIME);
    }

    #[test]
    fn to_same_unit_is_identity() {
        // Dimension equal, factor cancels exactly for the metre (factor 1.0).
        let q = qty(5.0, "m");
        assert_eq!(q.to(q.unit()).unwrap().value(), 5.0);
        // Non-trivial factor: identity within 1 ulp-scale relative error.
        let q = qty(5.0, "kcal/mol");
        let back = q.to(q.unit()).unwrap();
        assert!(
            ((back.value() - 5.0) / 5.0).abs() <= 1e-15,
            "got {}",
            back.value()
        );
    }

    #[test]
    fn to_parsed_converts_via_global_registry() {
        let q = qty(1.0, "kcal").to_parsed("kJ").unwrap();
        assert!(
            ((q.value() - 4.184) / 4.184).abs() <= 1e-12,
            "got {}",
            q.value()
        );
    }

    #[test]
    fn to_parsed_unknown_unit_errors() {
        let err = qty(1.0, "kcal").to_parsed("zorp").unwrap_err();
        assert!(matches!(err, UnitsError::UnknownUnit { .. }), "got {err:?}");
    }

    #[test]
    fn neg_negates_value_keeps_unit() {
        let q = -qty(2.5, "eV");
        assert_eq!(q.value(), -2.5);
        assert_eq!(q.unit(), &unit("eV"));
    }

    #[test]
    fn scalar_mul_and_div() {
        let q = qty(2.0, "m") * 3.0;
        assert_eq!(q.value(), 6.0);
        let q = qty(6.0, "m") / 4.0;
        assert_eq!(q.value(), 1.5);
        assert_eq!(q.unit().dimension(), Dimension::LENGTH);
    }

    #[test]
    fn display_shows_value_and_unit() {
        assert_eq!(qty(2.0, "m").to_string(), "2 m");
    }

    #[test]
    fn to_base_units_dimensionless_renders_one() {
        // deg is dimensionless (π/180); SI-base display collapses to "1".
        let base = qty(180.0, "deg").to_base_units();
        assert!(base.unit().dimension().is_dimensionless());
        assert_eq!(base.unit().to_string(), "1");
        assert!(
            ((base.value() - std::f64::consts::PI) / std::f64::consts::PI).abs() <= 1e-15,
            "got {}",
            base.value()
        );
    }

    #[test]
    fn to_base_units_si_name_composes() {
        // kcal/mol → m^2 * kg * s^-2 * mol^-1 in SI-base symbols.
        let base = qty(1.0, "kcal/mol").to_base_units();
        assert_eq!(base.unit().to_string(), "m^2 * kg * s^-2 * mol^-1");
    }
}
