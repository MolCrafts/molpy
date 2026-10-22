//! Pint-inspired unit system: dimensions, units, registry, parser, quantities.
//!
//! A [`UnitRegistry`] holds [`UnitDef`] entries over the 7 SI base dimensions,
//! an internal recursive-descent parser turns compound expressions
//! (`"kcal/mol/angstrom"`) into self-contained [`Unit`] values, and
//! [`Quantity`] provides dimension-checked arithmetic and conversion.
//!
//! Every unit is reduced at parse time to `(factor, offset, Dimension)` with
//! conversion to SI base defined as `si = value * factor + offset`; the
//! `offset` is non-zero only for affine units such as `degC`. Conversion
//! factors are SI-2019 exact or CODATA 2018 recommended values (see
//! [`constants`] and the registry preload tables in `registry.rs`).
//!
//! # One definition per unit
//!
//! Every unit conversion in molrs — a reader's nm → Å, a writer's kcal/mol →
//! kJ/mol, an analysis's Å³ → cm³ — goes through this module: a
//! [`UnitFactor`] (a `static` resolved once, for hot paths),
//! [`UnitRegistry::factor`] or [`Quantity::to`]. No module writes a factor
//! by hand (`* 4.184`, `/ 10.0`), and [`constants`](crate::core::constants)
//! holds physical constants and the constants engines define, never a
//! unit-conversion factor. `module_boundaries` checks both.
//!
//! Reference: design follows pint (Python),
//! <https://pint.readthedocs.io/en/stable/>.
//!
//! # Examples
//!
//! Energy conversion and dimension checking:
//!
//! ```
//! use molrs::core::{UnitRegistry, UnitsError};
//!
//! let reg = UnitRegistry::new();
//!
//! // 1 kcal/mol = 4.184 kJ/mol (thermochemical calorie, exact).
//! let e = reg.quantity(1.0, "kcal/mol")?;
//! let kj = e.to(&reg.parse("kJ/mol")?)?;
//! assert!((kj.value() - 4.184).abs() < 1e-12);
//!
//! // Converting an energy to a length is a dimension error.
//! let err = e.to(&reg.parse("angstrom")?).unwrap_err();
//! assert!(matches!(err, UnitsError::DimensionMismatch { .. }));
//! # Ok::<(), UnitsError>(())
//! ```

mod dimension;
mod error;
mod factor;
mod parse;
mod preset;
mod quantity;
mod registry;
mod unit;

pub use dimension::Dimension;
pub use error::UnitsError;
pub use factor::UnitFactor;
pub use preset::{
    PresetDim, UnitPreset, UnitPresetRegistry, lookup_unit_preset, register_unit_preset,
    replace_unit_preset, unit_preset_names,
};
pub use quantity::Quantity;
pub use registry::{UnitDef, UnitRegistry};
pub use unit::Unit;
