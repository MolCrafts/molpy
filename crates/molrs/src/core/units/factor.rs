//! [`UnitFactor`]: a conversion factor named by its two units and resolved
//! once from the global [`UnitRegistry`].

use std::sync::OnceLock;

use crate::op::F;

use super::registry::UnitRegistry;

/// The factor between two units, named by the pair of unit expressions it
/// converts between and resolved once (on first use) from
/// [`UnitRegistry::global`]. molrs defines each one it uses once, in
/// [`unit_factors`](crate::core::unit_factors):
///
/// ```
/// use molrs::core::UnitFactor;
/// use molrs::core::unit_factors::{KCAL_TO_KJ, NM_TO_ANGSTROM};
///
/// let k_kj = 2.0 * KCAL_TO_KJ.get(); // kcal/mol → kJ/mol
/// assert_eq!(k_kj, 8.368);
/// assert_eq!(0.15 * NM_TO_ANGSTROM.get(), 1.5);
///
/// // A conversion of one's own: two unit expressions, resolved once.
/// static HARTREE_TO_EV: UnitFactor = UnitFactor::new("hartree", "eV");
/// assert!((HARTREE_TO_EV.get() - 27.211_386_245_988).abs() < 1e-9);
/// ```
///
/// Every unit conversion in molrs goes through the unit registry — this
/// type, [`UnitRegistry::factor`] or [`Quantity::to`](super::Quantity::to) —
/// never a hand-written factor, so each unit has one definition. A `static`
/// `UnitFactor` costs one atomic load after the first call, which keeps it
/// out of the way of a hot loop.
#[derive(Debug)]
pub struct UnitFactor {
    from: &'static str,
    to: &'static str,
    value: OnceLock<F>,
}

impl UnitFactor {
    /// The factor converting a value in `from` to `to`, resolved on first
    /// [`get`](Self::get).
    pub const fn new(from: &'static str, to: &'static str) -> Self {
        Self {
            from,
            to,
            value: OnceLock::new(),
        }
    }

    /// `value_in_to = value_in_from × get()` ([`UnitRegistry::factor`]).
    ///
    /// # Panics
    ///
    /// When the expressions do not parse, differ in dimension or are affine:
    /// they are literals of the calling code, so that is a bug there, found
    /// by the first call.
    pub fn get(&self) -> F {
        *self.value.get_or_init(|| {
            UnitRegistry::global()
                .factor(self.from, self.to)
                .unwrap_or_else(|e| panic!("unit factor {} -> {}: {e}", self.from, self.to))
        })
    }

    /// The unit expression converted from.
    pub fn from_unit(&self) -> &'static str {
        self.from
    }

    /// The unit expression converted to.
    pub fn to_unit(&self) -> &'static str {
        self.to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_power_of_ten_is_the_correctly_rounded_factor() {
        assert_eq!(UnitFactor::new("angstrom", "nm").get(), 0.1);
        assert_eq!(UnitFactor::new("nm", "angstrom").get(), 10.0);
        assert_eq!(UnitFactor::new("nm^2", "angstrom^2").get(), 100.0);
        assert_eq!(UnitFactor::new("fs", "s").get(), 1e-15);
        assert_eq!(UnitFactor::new("m", "cm").get(), 100.0);
    }

    #[test]
    fn the_calorie_is_thermochemical() {
        assert_eq!(UnitFactor::new("kcal", "kJ").get(), 4.184);
    }

    #[test]
    #[should_panic(expected = "unit factor kcal -> nm")]
    fn a_dimension_mismatch_panics_naming_both_units() {
        UnitFactor::new("kcal", "nm").get();
    }
}
