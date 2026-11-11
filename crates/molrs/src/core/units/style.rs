//! The LAMMPS `units` styles: [`UnitStyle`] and the constants each one sets.

use std::fmt;
use std::str::FromStr;

use super::{UnitPreset, UnitsError};
use crate::core::constants::{COULOMB_METAL, COULOMB_REAL};
use crate::op::F;

/// One LAMMPS `units` style. The default is [`Lj`](Self::Lj), as in LAMMPS.
///
/// A style fixes the unit of every quantity a run reads and writes, and the
/// conversion constants the integrator, thermostats and barostats apply
/// ([`constants`](Self::constants)). Nothing is converted between styles: a
/// run is in exactly one, and every input is taken in it.
///
/// The names are LAMMPS's, matched exactly (LAMMPS compares with
/// `strcmp`): `lj`, `real`, `metal`, `si`, `cgs`, `electron`, `micro`,
/// `nano`.
///
/// Reference: LAMMPS `units` command, <https://docs.lammps.org/units.html>;
/// `Update::set_units` in LAMMPS `src/update.cpp`.
///
/// ```
/// use molrs::core::UnitStyle;
///
/// let real: UnitStyle = "real".parse()?;
/// assert_eq!(real.constants().boltz, 0.0019872067);
/// assert_eq!(real.constants().dt, 1.0); // fs
/// assert_eq!(UnitStyle::default(), UnitStyle::Lj);
/// # Ok::<(), molrs::core::UnitsError>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum UnitStyle {
    /// Reduced Lennard-Jones units: every constant is 1.
    #[default]
    Lj,
    /// g/mol, Å, fs, kcal/mol, K, e, atm.
    Real,
    /// g/mol, Å, ps, eV, K, e, bar.
    Metal,
    /// kg, m, s, J, K, C, Pa.
    Si,
    /// g, cm, s, erg, K, statcoulomb, dyne/cm².
    Cgs,
    /// amu, Bohr, fs, Hartree, K, e, Pa.
    Electron,
    /// pg, µm, µs, pg·µm²/µs², K, pC.
    Micro,
    /// ag, nm, ns, ag·nm²/ns², K, e.
    Nano,
}

/// Every value LAMMPS `Update::set_units` assigns for one style: the
/// `force->…` conversion constants, the default `update->dt` and the
/// default `neighbor->skin`.
///
/// Each field carries LAMMPS's name and holds the literal `src/update.cpp`
/// writes (an expression such as `48.88821291 * 48.88821291` is evaluated as
/// written there, so the value is the same `double`); the field docs are
/// `src/force.h`'s. Fields LAMMPS leaves at `0.0` ("not yet set") are `0.0`
/// here too.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitConstants {
    /// Boltzmann constant (energy / temperature).
    pub boltz: F,
    /// Planck's constant (energy · time).
    pub hplanck: F,
    /// Conversion of m v² to energy.
    pub mvv2e: F,
    /// Conversion of f t / m to velocity.
    pub ftm2v: F,
    /// Conversion of mass / volume to density.
    pub mv2d: F,
    /// Conversion of N k T / V to pressure.
    pub nktv2p: F,
    /// Conversion of q² / r to energy; LAMMPS divides it by the
    /// `dielectric` to form `qqrd2e` (`Force::init`).
    pub qqr2e: F,
    /// Conversion of q E to force.
    pub qe2f: F,
    /// Conversion of v x · dynamic viscosity to force.
    pub vxmu2f: F,
    /// Conversion of x x / t to kinematic viscosity.
    pub xxt2kmu: F,
    /// Electron mass (set in `real` only).
    pub e_mass: F,
    /// Conversion of ħ² / (m r²) to energy (set in `real` only).
    pub hhmrr2e: F,
    /// Conversion of m v / ħ to distance (set in `real` only).
    pub mvh2r: F,
    /// One Ångström in the style's distance unit.
    pub angstrom: F,
    /// One femtosecond in the style's time unit.
    pub femtosecond: F,
    /// One elementary charge in the style's charge unit.
    pub qelectron: F,
    /// Default timestep (`update->dt`), in the time unit.
    pub dt: F,
    /// Default neighbour skin (`neighbor->skin`), in the distance unit.
    pub skin: F,
}

// The eight blocks of LAMMPS `Update::set_units` (src/update.cpp), in its
// order and with its literals.

const LJ: UnitConstants = UnitConstants {
    boltz: 1.0,
    hplanck: 1.0,
    mvv2e: 1.0,
    ftm2v: 1.0,
    mv2d: 1.0,
    nktv2p: 1.0,
    qqr2e: 1.0,
    qe2f: 1.0,
    vxmu2f: 1.0,
    xxt2kmu: 1.0,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0,
    femtosecond: 1.0,
    qelectron: 1.0,
    dt: 0.005,
    skin: 0.3,
};

const REAL: UnitConstants = UnitConstants {
    boltz: 0.0019872067,
    hplanck: 95.306976368,
    mvv2e: 48.88821291 * 48.88821291,
    ftm2v: 1.0 / 48.88821291 / 48.88821291,
    mv2d: 1.0 / 0.602214129,
    nktv2p: 68568.415,
    qqr2e: COULOMB_REAL,
    qe2f: 23.060549,
    vxmu2f: 1.4393264316e4,
    xxt2kmu: 0.1,
    e_mass: 1.0 / 1836.1527556560675,
    hhmrr2e: 0.0957018663603261,
    mvh2r: 1.5339009481951,
    angstrom: 1.0,
    femtosecond: 1.0,
    qelectron: 1.0,
    dt: 1.0,
    skin: 2.0,
};

const METAL: UnitConstants = UnitConstants {
    boltz: 8.617343e-5,
    hplanck: 4.135667403e-3,
    mvv2e: 1.0364269e-4,
    ftm2v: 1.0 / 1.0364269e-4,
    mv2d: 1.0 / 0.602214129,
    nktv2p: 1.6021765e6,
    qqr2e: COULOMB_METAL,
    qe2f: 1.0,
    vxmu2f: 0.6241509647,
    xxt2kmu: 1.0e-4,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0,
    femtosecond: 1.0e-3,
    qelectron: 1.0,
    dt: 0.001,
    skin: 2.0,
};

const SI: UnitConstants = UnitConstants {
    boltz: 1.3806504e-23,
    hplanck: 6.62606896e-34,
    mvv2e: 1.0,
    ftm2v: 1.0,
    mv2d: 1.0,
    nktv2p: 1.0,
    qqr2e: 8.9876e9,
    qe2f: 1.0,
    vxmu2f: 1.0,
    xxt2kmu: 1.0,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0e-10,
    femtosecond: 1.0e-15,
    qelectron: 1.6021765e-19,
    dt: 1.0e-8,
    skin: 0.001,
};

const CGS: UnitConstants = UnitConstants {
    boltz: 1.3806504e-16,
    hplanck: 6.62606896e-27,
    mvv2e: 1.0,
    ftm2v: 1.0,
    mv2d: 1.0,
    nktv2p: 1.0,
    qqr2e: 1.0,
    qe2f: 1.0,
    vxmu2f: 1.0,
    xxt2kmu: 1.0,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0e-8,
    femtosecond: 1.0e-15,
    qelectron: 4.8032044e-10,
    dt: 1.0e-8,
    skin: 0.1,
};

const ELECTRON: UnitConstants = UnitConstants {
    boltz: 3.16681534e-6,
    hplanck: 0.1519829846,
    mvv2e: 1.06657236,
    ftm2v: 0.937582899,
    mv2d: 1.0,
    nktv2p: 2.94210108e13,
    qqr2e: 1.0,
    qe2f: 1.94469051e-10,
    vxmu2f: 3.39893149e1,
    xxt2kmu: 3.13796367e-2,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.88972612,
    femtosecond: 1.0,
    qelectron: 1.0,
    dt: 0.001,
    skin: 2.0,
};

const MICRO: UnitConstants = UnitConstants {
    boltz: 1.3806504e-8,
    hplanck: 6.62606896e-13,
    mvv2e: 1.0,
    ftm2v: 1.0,
    mv2d: 1.0,
    nktv2p: 1.0,
    qqr2e: 8.987556e6,
    qe2f: 1.0,
    vxmu2f: 1.0,
    xxt2kmu: 1.0,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0e-4,
    femtosecond: 1.0e-9,
    qelectron: 1.6021765e-7,
    dt: 2.0,
    skin: 0.1,
};

const NANO: UnitConstants = UnitConstants {
    boltz: 0.013806504,
    hplanck: 6.62606896e-4,
    mvv2e: 1.0,
    ftm2v: 1.0,
    mv2d: 1.0,
    nktv2p: 1.0,
    qqr2e: 230.7078669,
    qe2f: 1.0,
    vxmu2f: 1.0,
    xxt2kmu: 1.0,
    e_mass: 0.0,
    hhmrr2e: 0.0,
    mvh2r: 0.0,
    angstrom: 1.0e-1,
    femtosecond: 1.0e-6,
    qelectron: 1.0,
    dt: 0.00045,
    skin: 0.1,
};

impl UnitStyle {
    /// Every style, in the order of LAMMPS `Update::set_units`.
    pub const ALL: [UnitStyle; 8] = [
        UnitStyle::Lj,
        UnitStyle::Real,
        UnitStyle::Metal,
        UnitStyle::Si,
        UnitStyle::Cgs,
        UnitStyle::Electron,
        UnitStyle::Micro,
        UnitStyle::Nano,
    ];

    /// The LAMMPS name (`"lj"`, `"real"`, …).
    pub fn name(self) -> &'static str {
        match self {
            UnitStyle::Lj => "lj",
            UnitStyle::Real => "real",
            UnitStyle::Metal => "metal",
            UnitStyle::Si => "si",
            UnitStyle::Cgs => "cgs",
            UnitStyle::Electron => "electron",
            UnitStyle::Micro => "micro",
            UnitStyle::Nano => "nano",
        }
    }

    /// What LAMMPS `units <style>` sets: the conversion constants, the
    /// default timestep and the default neighbour skin.
    pub fn constants(self) -> &'static UnitConstants {
        match self {
            UnitStyle::Lj => &LJ,
            UnitStyle::Real => &REAL,
            UnitStyle::Metal => &METAL,
            UnitStyle::Si => &SI,
            UnitStyle::Cgs => &CGS,
            UnitStyle::Electron => &ELECTRON,
            UnitStyle::Micro => &MICRO,
            UnitStyle::Nano => &NANO,
        }
    }

    /// The style's [`UnitPreset`]: the unit of each quantity, by name, with
    /// this style's constants.
    pub fn preset(self) -> UnitPreset {
        match self {
            UnitStyle::Lj => UnitPreset::lj(),
            UnitStyle::Real => UnitPreset::real(),
            UnitStyle::Metal => UnitPreset::metal(),
            UnitStyle::Si => UnitPreset::si(),
            UnitStyle::Cgs => UnitPreset::cgs(),
            UnitStyle::Electron => UnitPreset::electron(),
            UnitStyle::Micro => UnitPreset::micro(),
            UnitStyle::Nano => UnitPreset::nano(),
        }
    }
}

impl FromStr for UnitStyle {
    type Err = UnitsError;

    /// The style named `name`, exactly as LAMMPS spells it.
    ///
    /// # Errors
    ///
    /// [`UnitsError::UnknownStyle`] for any other name.
    fn from_str(name: &str) -> Result<Self, UnitsError> {
        UnitStyle::ALL
            .into_iter()
            .find(|style| style.name() == name)
            .ok_or_else(|| UnitsError::UnknownStyle {
                name: name.to_owned(),
            })
    }
}

impl fmt::Display for UnitStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The literals of LAMMPS `Update::set_units` (src/update.cpp, develop
    /// 30 Mar 2026), transcribed by hand, in the field order of
    /// [`UnitConstants`]: boltz, hplanck, mvv2e, ftm2v, mv2d, nktv2p, qqr2e,
    /// qe2f, vxmu2f, xxt2kmu, e_mass, hhmrr2e, mvh2r, angstrom, femtosecond,
    /// qelectron, dt, skin. Compared bit for bit.
    #[rustfmt::skip]
    const LAMMPS: [(&str, [F; 18]); 8] = [
        ("lj", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
                1.0, 1.0, 1.0, 0.005, 0.3]),
        ("real", [0.0019872067, 95.306976368, 48.88821291 * 48.88821291,
                  1.0 / 48.88821291 / 48.88821291, 1.0 / 0.602214129, 68568.415,
                  332.06371, 23.060549, 1.4393264316e4, 0.1, 1.0 / 1836.1527556560675,
                  0.0957018663603261, 1.5339009481951, 1.0, 1.0, 1.0, 1.0, 2.0]),
        ("metal", [8.617343e-5, 4.135667403e-3, 1.0364269e-4, 1.0 / 1.0364269e-4,
                   1.0 / 0.602214129, 1.6021765e6, 14.399645, 1.0, 0.6241509647,
                   1.0e-4, 0.0, 0.0, 0.0, 1.0, 1.0e-3, 1.0, 0.001, 2.0]),
        ("si", [1.3806504e-23, 6.62606896e-34, 1.0, 1.0, 1.0, 1.0, 8.9876e9, 1.0,
                1.0, 1.0, 0.0, 0.0, 0.0, 1.0e-10, 1.0e-15, 1.6021765e-19, 1.0e-8,
                0.001]),
        ("cgs", [1.3806504e-16, 6.62606896e-27, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                 1.0, 0.0, 0.0, 0.0, 1.0e-8, 1.0e-15, 4.8032044e-10, 1.0e-8, 0.1]),
        ("electron", [3.16681534e-6, 0.1519829846, 1.06657236, 0.937582899, 1.0,
                      2.94210108e13, 1.0, 1.94469051e-10, 3.39893149e1,
                      3.13796367e-2, 0.0, 0.0, 0.0, 1.88972612, 1.0, 1.0, 0.001,
                      2.0]),
        ("micro", [1.3806504e-8, 6.62606896e-13, 1.0, 1.0, 1.0, 1.0, 8.987556e6,
                   1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0e-4, 1.0e-9, 1.6021765e-7, 2.0,
                   0.1]),
        ("nano", [0.013806504, 6.62606896e-4, 1.0, 1.0, 1.0, 1.0, 230.7078669, 1.0,
                  1.0, 1.0, 0.0, 0.0, 0.0, 1.0e-1, 1.0e-6, 1.0, 0.00045, 0.1]),
    ];

    fn fields(c: &UnitConstants) -> [F; 18] {
        [
            c.boltz,
            c.hplanck,
            c.mvv2e,
            c.ftm2v,
            c.mv2d,
            c.nktv2p,
            c.qqr2e,
            c.qe2f,
            c.vxmu2f,
            c.xxt2kmu,
            c.e_mass,
            c.hhmrr2e,
            c.mvh2r,
            c.angstrom,
            c.femtosecond,
            c.qelectron,
            c.dt,
            c.skin,
        ]
    }

    #[test]
    fn every_constant_is_the_lammps_literal() {
        for (style, (name, lammps)) in UnitStyle::ALL.into_iter().zip(LAMMPS) {
            assert_eq!(style.name(), name);
            let ours = fields(style.constants());
            for (k, (a, b)) in ours.iter().zip(lammps).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{name} field {k}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn a_name_parses_to_its_style_and_back() {
        for style in UnitStyle::ALL {
            assert_eq!(style.name().parse::<UnitStyle>().unwrap(), style);
            assert_eq!(style.to_string(), style.name());
        }
    }

    /// LAMMPS compares the name with `strcmp`: no case folding, no aliases,
    /// and `openmm` is no LAMMPS style.
    #[test]
    fn any_other_name_is_refused() {
        for name in ["Real", "REAL", "openmm", "", "lj "] {
            let err = name.parse::<UnitStyle>().unwrap_err();
            assert_eq!(
                err,
                UnitsError::UnknownStyle {
                    name: name.to_owned()
                }
            );
        }
    }

    #[test]
    fn the_default_style_is_lj() {
        assert_eq!(UnitStyle::default(), UnitStyle::Lj);
    }

    /// A style's preset reports that style and the style's own constants.
    #[test]
    fn a_style_preset_carries_the_style_constants() {
        for style in UnitStyle::ALL {
            let preset = style.preset();
            let c = style.constants();
            assert_eq!(preset.name(), style.name());
            assert_eq!(preset.style(), Some(style));
            assert_eq!(preset.boltzmann(), c.boltz);
            assert_eq!(preset.coulomb(), c.qqr2e);
            assert_eq!(preset.ftm2v().unwrap(), c.ftm2v);
            assert_eq!(preset.mvv2e().unwrap(), c.mvv2e);
            assert_eq!(preset.nktv2p().unwrap(), c.nktv2p);
        }
    }
}
