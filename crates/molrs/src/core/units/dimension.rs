//! Dimensional analysis: exponent vector over the 7 SI base dimensions.

use std::fmt;
use std::ops::{Div, Mul};

/// Exponents over the 7 SI base dimensions, in order:
/// `[length, mass, time, current, temperature, amount, luminous intensity]`.
///
/// `Mul`/`Div` add/subtract exponents, so dimensions compose algebraically.
///
/// # Examples
///
/// ```
/// use molrs::core::Dimension;
///
/// // Energy = M·L²·T⁻²
/// let derived = Dimension::MASS * Dimension::LENGTH.pow(2) / Dimension::TIME.pow(2);
/// assert_eq!(derived, Dimension::ENERGY);
/// assert!((Dimension::ENERGY / Dimension::ENERGY).is_dimensionless());
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct Dimension([i32; 7]);

impl Dimension {
    /// Dimensionless (all exponents zero).
    pub const DIMENSIONLESS: Dimension = Dimension([0, 0, 0, 0, 0, 0, 0]);
    /// Length (L).
    pub const LENGTH: Dimension = Dimension([1, 0, 0, 0, 0, 0, 0]);
    /// Mass (M).
    pub const MASS: Dimension = Dimension([0, 1, 0, 0, 0, 0, 0]);
    /// Time (T).
    pub const TIME: Dimension = Dimension([0, 0, 1, 0, 0, 0, 0]);
    /// Electric current (I).
    pub const CURRENT: Dimension = Dimension([0, 0, 0, 1, 0, 0, 0]);
    /// Temperature (Θ).
    pub const TEMPERATURE: Dimension = Dimension([0, 0, 0, 0, 1, 0, 0]);
    /// Amount of substance (N).
    pub const AMOUNT: Dimension = Dimension([0, 0, 0, 0, 0, 1, 0]);
    /// Energy: M L² T⁻².
    pub const ENERGY: Dimension = Dimension([2, 1, -2, 0, 0, 0, 0]);
    /// Force: M L T⁻².
    pub const FORCE: Dimension = Dimension([1, 1, -2, 0, 0, 0, 0]);
    /// Pressure: M L⁻¹ T⁻².
    pub const PRESSURE: Dimension = Dimension([-1, 1, -2, 0, 0, 0, 0]);
    /// Electric charge: T I (current × time).
    pub const CHARGE: Dimension = Dimension([0, 0, 1, 1, 0, 0, 0]);
    /// Frequency: T⁻¹.
    pub const FREQUENCY: Dimension = Dimension([0, 0, -1, 0, 0, 0, 0]);

    /// Construct from a raw exponent array.
    pub const fn from_exponents(exps: [i32; 7]) -> Dimension {
        Dimension(exps)
    }

    /// The raw exponent array.
    pub const fn exponents(&self) -> [i32; 7] {
        self.0
    }

    /// True when every exponent is zero.
    pub fn is_dimensionless(&self) -> bool {
        self.0 == [0; 7]
    }

    /// Raise the dimension to an integer power (scales all exponents).
    pub fn pow(self, n: i32) -> Dimension {
        Dimension(self.0.map(|e| e * n))
    }
}

impl Mul for Dimension {
    type Output = Dimension;
    // Multiplying dimensions adds their exponents (L²·L = L³).
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, rhs: Dimension) -> Dimension {
        let mut out = self.0;
        for (o, r) in out.iter_mut().zip(rhs.0) {
            *o += r;
        }
        Dimension(out)
    }
}

impl Div for Dimension {
    type Output = Dimension;
    // Dividing dimensions subtracts their exponents (L³/L = L²).
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Dimension) -> Dimension {
        let mut out = self.0;
        for (o, r) in out.iter_mut().zip(rhs.0) {
            *o -= r;
        }
        Dimension(out)
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SYMBOLS: [&str; 7] = ["L", "M", "T", "I", "Θ", "N", "J"];
        if self.is_dimensionless() {
            return write!(f, "dimensionless");
        }
        let mut first = true;
        for (sym, &exp) in SYMBOLS.iter().zip(&self.0) {
            if exp == 0 {
                continue;
            }
            if !first {
                write!(f, "·")?;
            }
            first = false;
            if exp == 1 {
                write!(f, "{}", sym)?;
            } else {
                write!(f, "{}^{}", sym, exp)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn energy_is_mass_length2_per_time2() {
        // ENERGY = M·L²·T⁻²
        let derived = Dimension::MASS * Dimension::LENGTH.pow(2) / Dimension::TIME.pow(2);
        assert_eq!(derived, Dimension::ENERGY);
    }

    #[test]
    fn force_is_energy_per_length() {
        let derived = Dimension::ENERGY / Dimension::LENGTH;
        assert_eq!(derived, Dimension::FORCE);
    }

    #[test]
    fn pressure_is_force_per_area() {
        let derived = Dimension::FORCE / Dimension::LENGTH.pow(2);
        assert_eq!(derived, Dimension::PRESSURE);
    }

    #[test]
    fn mul_adds_exponents() {
        let d = Dimension::LENGTH * Dimension::LENGTH;
        assert_eq!(d, Dimension::LENGTH.pow(2));
    }

    #[test]
    fn div_subtracts_exponents() {
        let d = Dimension::ENERGY / Dimension::ENERGY;
        assert!(d.is_dimensionless());
    }

    #[test]
    fn pow_scales_exponents() {
        let area = Dimension::LENGTH.pow(2);
        assert_eq!(area.exponents(), [2, 0, 0, 0, 0, 0, 0]);
        let inv = Dimension::TIME.pow(-2);
        assert_eq!(inv.exponents(), [0, 0, -2, 0, 0, 0, 0]);
    }

    #[test]
    fn default_is_dimensionless() {
        assert_eq!(Dimension::default(), Dimension::DIMENSIONLESS);
        assert!(Dimension::default().is_dimensionless());
    }

    #[test]
    fn dimensionless_predicate() {
        assert!(Dimension::DIMENSIONLESS.is_dimensionless());
        assert!(!Dimension::LENGTH.is_dimensionless());
    }

    #[test]
    fn hash_and_eq_consistent() {
        let mut set = HashSet::new();
        set.insert(Dimension::ENERGY);
        let derived = Dimension::MASS * Dimension::LENGTH.pow(2) / Dimension::TIME.pow(2);
        assert!(set.contains(&derived));
    }

    #[test]
    fn display_renders_exponents() {
        // Order follows the base-dimension array: L, M, T, ...
        assert_eq!(Dimension::ENERGY.to_string(), "L^2·M·T^-2");
        assert_eq!(Dimension::LENGTH.to_string(), "L");
        assert_eq!(Dimension::DIMENSIONLESS.to_string(), "dimensionless");
    }
}
