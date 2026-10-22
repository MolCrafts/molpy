//! Physical constants (CODATA 2018 / SI-2019 exact where applicable) and the
//! constants molecular engines and force fields define as data (Coulomb
//! prefactors, 1-4 scale factors). Every such number molrs uses lives here,
//! once; [`ALL`] lists them by name, which is what the bindings expose.
//!
//! A unit-conversion factor is not a constant: kcal ↔ kJ, nm ↔ Å, bohr ↔ Å,
//! cm³ ↔ Å³ and every other conversion is the unit module's
//! ([`UnitFactor`](crate::core::UnitFactor),
//! [`UnitRegistry::factor`](crate::core::UnitRegistry::factor)), whose unit
//! definitions are built from the constants here.
//!
//! Reference: SI Brochure, 9th edition (2019) for the exact defining
//! constants; CODATA 2018 recommended values,
//! <https://physics.nist.gov/cuu/Constants/>.
//!
//! # Examples
//!
//! ```
//! use molrs::core::constants::GAS_CONSTANT;
//!
//! // R = N_A · k_B = 8.314 462 618... J/(mol·K), exact under SI-2019.
//! assert!((GAS_CONSTANT - 8.314_462_618_153_24).abs() < 1e-12);
//! ```

use crate::op::F;

/// Avogadro constant `N_A` (exact, SI-2019), in mol⁻¹.
pub const AVOGADRO: F = 6.022_140_76e23;

/// Boltzmann constant `k_B` (exact, SI-2019), in J/K. In a unit system's
/// own units it is [`UnitPreset::boltzmann`](crate::core::UnitPreset::boltzmann)
/// (`real`: `k_B` in kcal·mol⁻¹·K⁻¹, the molar gas constant over 4184).
pub const BOLTZMANN: F = 1.380_649e-23;

/// Molar gas constant `R = N_A · k_B` (exact, SI-2019), in J/(mol·K).
pub const GAS_CONSTANT: F = AVOGADRO * BOLTZMANN;

/// Elementary charge `e` (exact, SI-2019), in coulombs.
pub const ELEMENTARY_CHARGE: F = 1.602_176_634e-19;

/// Planck constant `h` (exact, SI-2019), in J·s.
pub const PLANCK: F = 6.626_070_15e-34;

/// Bohr radius `a₀` (CODATA 2018), in metres: the `bohr` unit, the
/// Gaussian cube format's length unit.
pub const BOHR_RADIUS: F = 5.291_772_109_03e-11;

/// Hartree energy `E_h` (CODATA 2018), in joules: the `hartree` unit.
pub const HARTREE_ENERGY: F = 4.359_744_722_207_1e-18;

/// Atomic mass constant `m_u` (CODATA 2018), in kilograms: the `dalton`
/// unit.
pub const ATOMIC_MASS_CONSTANT: F = 1.660_539_066_60e-27;

/// Coulomb constant `k_e = 1/(4π·ε₀)` (CODATA 2018), in N·m²·C⁻².
pub const COULOMB_CONSTANT: F = 8.987_551_792_3e9;

/// Coulomb constant `k_e = 1/(4π·ε₀)` in MD "real" units — LAMMPS `real`'s `qqr2e`
/// (kcal·Å·mol⁻¹·e⁻²), CODATA-derived. It is what the dielectric/conductivity
/// analyses use, and what the OPLS and LAMMPS force fields declare on their
/// `pair/coul/cut` style.
///
/// It is **not** a constant of the Coulomb kernel. MMFF rounds it differently
/// (Halgren's 332.0716, [`MMFF_COULOMB`]), and the
/// 2.4e-5 difference is above the RDKit parity tolerance on caffeine. Both values
/// are correct: the *force field* chooses one and states it on its `pair/coul/cut`
/// style, and the kernel has no default.
pub const COULOMB_REAL: F = 332.063_71;

/// Coulomb constant in LAMMPS `metal` units (eV·Å·e⁻²): LAMMPS's own `qqr2e`
/// for that unit style, so a `metal` force field read from LAMMPS prices its
/// electrostatics as LAMMPS does.
pub const COULOMB_METAL: F = 14.399_645;

/// Coulomb constant AMBER (sander/pmemd) evaluates, kcal·Å·mol⁻¹·e⁻².
///
/// Equal to `18.2223² = 332.05221729`, where `18.2223` is
/// [`AMBER_CHARGE_FACTOR`], the factor Amber writes into the prmtop `CHARGE`
/// section (Amber [FileFormats](https://ambermd.org/FileFormats.php), ParmEd
/// `AMBER_ELECTROSTATIC`). AmberTools `sander` single-points on acetate,
/// methylammonium and imidazolium recover it to printed precision.
/// [`COULOMB_REAL`] differs by a relative 3.46e-5 — a documented cross-engine
/// offset, not an error.
pub const AMBER_COULOMB: F = 332.052_217_29;

/// AMBER's prmtop charge factor: `CHARGE` stores `q · 18.2223`.
///
/// The literal is Amber's own (`18.2223² = `[`AMBER_COULOMB`]); do not re-derive
/// it from `√`[`COULOMB_REAL`], which would shift every charge by ~1.7e-5.
pub const AMBER_CHARGE_FACTOR: F = 18.2223;

/// CHARMM's `CCELEC`, kcal·Å·mol⁻¹·e⁻²: the Coulomb constant CHARMM (and a
/// chamber prmtop, which stores `CHARGE` as `q·√332.0716`, ParmEd's
/// `CHARMM_ELECTROSTATIC`) evaluates.
pub const CHARMM_COULOMB: F = 332.0716;

/// OpenMM's Coulomb constant `ONE_4PI_EPS0`, kJ·nm·mol⁻¹·e⁻², as OpenMM
/// states it (`SimTKOpenMMRealType.h`): every OpenMM energy is computed with
/// it, [`COULOMB_REAL`]'s 332.06371 kcal·Å·mol⁻¹·e⁻² × (1 + 9.9·10⁻⁹).
pub const OPENMM_ONE_4PI_EPS0: F = 138.935_457_644_381_98;

/// GROMACS's Coulomb constant `ONE_4PI_EPS0`, kJ·nm·mol⁻¹·e⁻²: 1/(4π ε₀)
/// from CODATA 2018 in GROMACS's own expression (`units.h`), one ulp below
/// [`OPENMM_ONE_4PI_EPS0`].
pub const GROMACS_ONE_4PI_EPS0: F = 138.935_457_644_381_96;

/// OpenMM's Coulomb constant in `real` units (kcal·Å·mol⁻¹·e⁻²):
/// [`OPENMM_ONE_4PI_EPS0`] converted from kJ·nm, 332.06371329919216. What
/// an OpenMM force field read into molrs states on its Coulomb styles, so it
/// prices its electrostatics as OpenMM does.
#[cfg(feature = "ff")]
pub(crate) fn openmm_coulomb_real() -> F {
    OPENMM_ONE_4PI_EPS0 * crate::core::unit_factors::KJ_NM_TO_KCAL_ANGSTROM.get()
}

/// GROMACS's Coulomb constant in `real` units (kcal·Å·mol⁻¹·e⁻²):
/// [`GROMACS_ONE_4PI_EPS0`] converted from kJ·nm, 332.06371329919205 (one ulp
/// below [`openmm_coulomb_real`]). What a GROMACS topology read into molrs
/// states on its Coulomb styles.
#[cfg(feature = "ff")]
pub(crate) fn gromacs_coulomb_real() -> F {
    GROMACS_ONE_4PI_EPS0 * crate::core::unit_factors::KJ_NM_TO_KCAL_ANGSTROM.get()
}

/// Speed of light in vacuum `c` (exact, SI-2019), in m/s.
pub const SPEED_OF_LIGHT: F = 299_792_458.0;

/// Second radiation constant `c₂ = h·c / k_B` (exact under SI-2019,
/// CODATA 2018), in cm·K: the `hcν̃ / k_BT` of a Boltzmann factor at
/// wavenumber `ν̃` (cm⁻¹).
pub const SECOND_RADIATION_CONSTANT: F = 1.438_776_877;

/// MMFF94's mdyne·Å → kcal·mol⁻¹ factor, `143.9325`: the prefactor of
/// Halgren's bond, angle, stretch-bend and out-of-plane terms (Halgren 1996,
/// J. Comput. Chem. 17, 490), and RDKit's `MDYNE_A_TO_KCAL_MOL`
/// (`ForceField/MMFF/Params.h`). It is engine data, not a unit conversion:
/// the exact factor (1 mdyne·Å = 10⁻¹⁸ J, × N_A / 4184 J) is 143.93263…,
/// and MMFF's energies are defined with the rounded literal.
pub const MMFF_MDYNE_A_TO_KCAL_MOL: F = 143.9325;

/// π as AmberTools `parmchk2` writes it, `3.1415926` (eight figures): its
/// empirical angle force constant (`empangle`) converts θ₀ to radians as
/// `θ₀ · PARMCHK2_PI / 180`, and molrs's GAFF estimate does the same, digit
/// for digit.
#[allow(clippy::approx_constant)]
pub const PARMCHK2_PI: F = 3.1415926;

/// An angle in degrees in radians as `parmchk2` converts it,
/// `degrees · PARMCHK2_PI / 180` (in that order), so a GAFF angle estimate is
/// parmchk2's to the last digit.
#[cfg(feature = "ff")]
pub(crate) fn parmchk2_radians(degrees: F) -> F {
    degrees * PARMCHK2_PI / 180.0
}

/// Relative permittivity of vacuum, `ε_r = 1` — the medium OPLS, GAFF/AMBER
/// and MMFF were each parameterised in. A force field still has to choose
/// vacuum: every one that does states `dielectric` on its `coul/cut` style,
/// and the kernel has no default for it.
pub const VACUUM_DIELECTRIC: F = 1.0;

/// The Coulomb constant UFF's bond force-constant rule is written with,
/// kcal·Å·mol⁻¹·e⁻² (Rappé et al. 1992, Eq. 6, `k_ij = 664.12 Z_i Z_j / r³`
/// with `664.12 = 2 · 332.06`; RDKit `Params::G`).
pub const UFF_COULOMB: F = 332.06;

/// MMFF94's Coulomb constant, kcal·Å·mol⁻¹·e⁻² (Halgren 1996, the
/// `332.0716` of the buffered-Coulomb term; RDKit's). Not CODATA's
/// [`COULOMB_REAL`]: the 2.4e-5 difference is above the RDKit parity
/// tolerance, so MMFF states its own.
pub const MMFF_COULOMB: F = 332.0716;

/// OPLS-AA's 1-4 Lennard-Jones scale weight (GROMACS `[ defaults ]`
/// fudgeLJ): `lj_14 = OPLS_LJ_14`.
pub const OPLS_LJ_14: F = 0.5;

/// OPLS-AA's 1-4 Coulomb scale weight (GROMACS `[ defaults ]` fudgeQQ):
/// `coul_14 = OPLS_COULOMB_14`.
pub const OPLS_COULOMB_14: F = 0.5;

/// AMBER's 1-4 Coulomb **divisor** (`SCEE`): `coul_14 = 1 / AMBER_SCEE`.
///
/// The GAFF/GAFF2 typifier force field's 1-4 Coulomb scale, and the
/// force-field prmtop reader's value when `SCEE_SCALE_FACTOR` is absent
/// (pre-Amber-11 files, whose force field is AMBER's). A prmtop that carries
/// the section never touches it, so a GLYCAM file (`SCEE = 1.0`) reads
/// correctly.
pub const AMBER_SCEE: F = 1.2;

/// AMBER's 1-4 Lennard-Jones **divisor** (`SCNB`): `lj_14 = 1 / AMBER_SCNB`.
/// The GAFF/GAFF2 typifier force field's 1-4 LJ scale, and the force-field
/// prmtop reader's value when `SCNB_SCALE_FACTOR` is absent.
pub const AMBER_SCNB: F = 2.0;

/// Every constant of this module, by its name: the table the bindings
/// expose (Python `molrs.core.constants`), so a constant added here needs
/// no binding edit.
pub const ALL: &[(&str, F)] = &[
    ("AVOGADRO", AVOGADRO),
    ("BOLTZMANN", BOLTZMANN),
    ("GAS_CONSTANT", GAS_CONSTANT),
    ("ELEMENTARY_CHARGE", ELEMENTARY_CHARGE),
    ("PLANCK", PLANCK),
    ("BOHR_RADIUS", BOHR_RADIUS),
    ("HARTREE_ENERGY", HARTREE_ENERGY),
    ("ATOMIC_MASS_CONSTANT", ATOMIC_MASS_CONSTANT),
    ("COULOMB_CONSTANT", COULOMB_CONSTANT),
    ("COULOMB_REAL", COULOMB_REAL),
    ("COULOMB_METAL", COULOMB_METAL),
    ("AMBER_COULOMB", AMBER_COULOMB),
    ("AMBER_CHARGE_FACTOR", AMBER_CHARGE_FACTOR),
    ("CHARMM_COULOMB", CHARMM_COULOMB),
    ("OPENMM_ONE_4PI_EPS0", OPENMM_ONE_4PI_EPS0),
    ("GROMACS_ONE_4PI_EPS0", GROMACS_ONE_4PI_EPS0),
    ("SPEED_OF_LIGHT", SPEED_OF_LIGHT),
    ("SECOND_RADIATION_CONSTANT", SECOND_RADIATION_CONSTANT),
    ("MMFF_MDYNE_A_TO_KCAL_MOL", MMFF_MDYNE_A_TO_KCAL_MOL),
    ("PARMCHK2_PI", PARMCHK2_PI),
    ("VACUUM_DIELECTRIC", VACUUM_DIELECTRIC),
    ("UFF_COULOMB", UFF_COULOMB),
    ("MMFF_COULOMB", MMFF_COULOMB),
    ("OPLS_LJ_14", OPLS_LJ_14),
    ("OPLS_COULOMB_14", OPLS_COULOMB_14),
    ("AMBER_SCEE", AMBER_SCEE),
    ("AMBER_SCNB", AMBER_SCNB),
];

#[cfg(test)]
mod tests {
    use super::*;

    /// AMBER's Coulomb constant is its prmtop charge factor squared, 3.46e-5
    /// below CODATA's `real` constant.
    #[test]
    fn amber_coulomb_is_its_charge_factor_squared() {
        assert!((AMBER_COULOMB - AMBER_CHARGE_FACTOR.powi(2)).abs() < 1e-9);
        let rel = (COULOMB_REAL - AMBER_COULOMB) / COULOMB_REAL;
        assert!((rel - 3.4610e-5).abs() < 1e-8, "relative offset {rel}");
    }

    /// `c₂ = h·c / k_B` from the exact SI-2019 constants, in cm·K.
    #[test]
    fn second_radiation_constant_is_hc_over_kb() {
        let c2 =
            PLANCK * SPEED_OF_LIGHT / BOLTZMANN * crate::core::UnitFactor::new("m", "cm").get();
        assert!((c2 - SECOND_RADIATION_CONSTANT).abs() < 1e-9, "{c2}");
    }
}

#[cfg(test)]
mod all_tests {
    use super::*;

    /// [`ALL`] names every `pub const` of this file, once.
    #[test]
    fn all_lists_every_constant() {
        let text = include_str!("constants.rs");
        let mut declared: Vec<&str> = text
            .lines()
            .filter_map(|l| l.strip_prefix("pub const "))
            .filter_map(|l| l.split(':').next())
            .filter(|name| *name != "ALL")
            .collect();
        declared.sort_unstable();
        let mut listed: Vec<&str> = ALL.iter().map(|(name, _)| *name).collect();
        listed.sort_unstable();
        assert_eq!(declared, listed);
    }

    /// The Bohr radius, CODATA 2018: one value, the `bohr` unit's.
    #[test]
    fn the_bohr_unit_is_the_bohr_radius() {
        let factor = crate::core::UnitRegistry::global()
            .factor("bohr", "angstrom")
            .unwrap();
        assert!((factor - 0.529_177_210_903).abs() < 1e-15, "{factor}");
    }
}
