//! The unit conversions molrs makes, each defined once as a
//! [`UnitFactor`]: named by its two units, resolved from the global
//! [`UnitRegistry`](super::UnitRegistry) on first use.
//!
//! A module that converts between two units uses the factor here; a
//! conversion no module has made yet is added here, not written where it is
//! used (`molrs::module_boundaries` fails on a `UnitFactor` defined anywhere
//! else).
//!
//! ```
//! use molrs::core::unit_factors::{KCAL_TO_KJ, NM_TO_ANGSTROM};
//!
//! assert_eq!(2.0 * KCAL_TO_KJ.get(), 8.368);
//! assert_eq!(0.15 * NM_TO_ANGSTROM.get(), 1.5);
//! ```

use super::UnitFactor;

/// kcal → kJ (thermochemical calorie, 4.184 J): kcal/mol → kJ/mol.
pub static KCAL_TO_KJ: UnitFactor = UnitFactor::new("kcal", "kJ");
/// J → kcal.
pub static J_TO_KCAL: UnitFactor = UnitFactor::new("J", "kcal");
/// J → kJ.
pub static J_TO_KJ: UnitFactor = UnitFactor::new("J", "kJ");
/// nm → Å.
pub static NM_TO_ANGSTROM: UnitFactor = UnitFactor::new("nm", "angstrom");
/// Å → nm.
pub static ANGSTROM_TO_NM: UnitFactor = UnitFactor::new("angstrom", "nm");
/// Å → m.
pub static ANGSTROM_TO_M: UnitFactor = UnitFactor::new("angstrom", "m");
/// bohr → Å.
pub static BOHR_TO_ANGSTROM: UnitFactor = UnitFactor::new("bohr", "angstrom");
/// bohr³ → Å³.
pub static BOHR3_TO_ANGSTROM3: UnitFactor = UnitFactor::new("bohr^3", "angstrom^3");
/// fs → s.
pub static FS_TO_S: UnitFactor = UnitFactor::new("fs", "s");
/// THz → fs⁻¹.
pub static THZ_TO_PER_FS: UnitFactor = UnitFactor::new("THz", "1/fs");
/// m/s → cm/fs.
pub static M_PER_S_TO_CM_PER_FS: UnitFactor = UnitFactor::new("m/s", "cm/fs");
/// kcal/Å² → kJ/nm²: a harmonic force constant, kcal·mol⁻¹·Å⁻² → kJ·mol⁻¹·nm⁻².
pub static KCAL_ANGSTROM2_TO_KJ_NM2: UnitFactor = UnitFactor::new("kcal/angstrom^2", "kJ/nm^2");
/// kcal·Å → kJ·nm: a Coulomb constant, kcal·Å·mol⁻¹·e⁻² → kJ·nm·mol⁻¹·e⁻².
pub static KCAL_ANGSTROM_TO_KJ_NM: UnitFactor = UnitFactor::new("kcal*angstrom", "kJ*nm");
/// kJ·nm → kcal·Å: a Coulomb constant, kJ·nm·mol⁻¹·e⁻² → kcal·Å·mol⁻¹·e⁻².
pub static KJ_NM_TO_KCAL_ANGSTROM: UnitFactor = UnitFactor::new("kJ*nm", "kcal*angstrom");
