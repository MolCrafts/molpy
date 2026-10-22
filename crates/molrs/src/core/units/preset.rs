//! Engine-neutral unit-system presets: [`UnitPreset`].

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::op::F;

use crate::core::constants::{
    BOLTZMANN, COULOMB_CONSTANT, COULOMB_METAL, COULOMB_REAL, GAS_CONSTANT,
};
use crate::core::unit_factors::{J_TO_KCAL, J_TO_KJ, KCAL_ANGSTROM_TO_KJ_NM};

/// The Boltzmann constant in `energy` per kelvin.
fn boltzmann_in(energy: &str) -> F {
    BOLTZMANN * registry_factor("J", energy)
}

/// The Coulomb constant `1/(4π ε₀)` in `energy · length / charge²`.
fn coulomb_in(energy: &str, length: &str, charge: &str) -> F {
    COULOMB_CONSTANT * registry_factor("J*m/C**2", &format!("({energy})*({length})/({charge})**2"))
}

/// [`UnitRegistry::factor`](super::UnitRegistry::factor) of the global
/// registry over the presets' own (literal) unit expressions.
fn registry_factor(from: &str, to: &str) -> F {
    super::UnitRegistry::global()
        .factor(from, to)
        .unwrap_or_else(|e| panic!("unit preset factor {from} -> {to}: {e}"))
}

/// One of the ten named dimensions every [`UnitPreset`] reports.
///
/// A *dimension* is the kind of quantity (length, energy, …), independent of
/// the unit it is written in; a preset fixes one unit per dimension. Each
/// variant's doc gives its unit in the `real` preset, the preset the schema
/// document displays units in.
///
/// [`name`](Self::name) is the preset table key, so
/// `preset.unit(dim.name())` is the preset's unit for that dimension.
/// [`ALL`](Self::ALL) is the single list of those keys, in the order of the
/// LAMMPS `units` documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PresetDim {
    /// Mass (`real`: g/mol).
    Mass,
    /// Length (`real`: Å).
    Length,
    /// Time (`real`: fs).
    Time,
    /// Energy (`real`: kcal/mol).
    Energy,
    /// Temperature (`real`: K).
    Temperature,
    /// Charge (`real`: e, the elementary charge).
    Charge,
    /// Pressure (`real`: atm).
    Pressure,
    /// Velocity (`real`: Å/fs).
    Velocity,
    /// Force (`real`: kcal/(mol·Å)).
    Force,
    /// Mass density (`real`: g/cm³).
    Density,
}

impl PresetDim {
    /// Every dimension, in preset table order.
    pub const ALL: [PresetDim; 10] = [
        PresetDim::Mass,
        PresetDim::Length,
        PresetDim::Time,
        PresetDim::Energy,
        PresetDim::Temperature,
        PresetDim::Charge,
        PresetDim::Pressure,
        PresetDim::Velocity,
        PresetDim::Force,
        PresetDim::Density,
    ];

    /// The preset table key (`"mass"`, `"length"`, …).
    pub fn name(self) -> &'static str {
        match self {
            PresetDim::Mass => "mass",
            PresetDim::Length => "length",
            PresetDim::Time => "time",
            PresetDim::Energy => "energy",
            PresetDim::Temperature => "temperature",
            PresetDim::Charge => "charge",
            PresetDim::Pressure => "pressure",
            PresetDim::Velocity => "velocity",
            PresetDim::Force => "force",
            PresetDim::Density => "density",
        }
    }
}

/// One unit-system view: ten unit names plus the Boltzmann and Coulomb
/// constants expressed in that system.
///
/// A `UnitPreset` is a named view of the constants in [`crate::core::constants`]
/// plus the ten base-unit names of a LAMMPS-style unit system. Preset **names**
/// keep the familiar `"real"` / `"metal"` / `"lj"` tokens; the type names do not
/// mention LAMMPS. Callers compose conversions themselves — there is no
/// `convert(value, from, to)` façade.
///
/// Reference: LAMMPS `units` command,
/// <https://docs.lammps.org/units.html>; Thompson et al.,
/// *Comput. Phys. Commun.* **271** (2022) 108171.
#[derive(Clone, Debug)]
pub struct UnitPreset {
    name: String,
    units: HashMap<&'static str, String>,
    boltzmann: F,
    coulomb: F,
}

impl UnitPreset {
    /// A preset of the caller's own: `units` gives the unit expression of
    /// each of the ten [`PresetDim`]s (keyed by [`PresetDim::name`]),
    /// `boltzmann` and `coulomb` the two constants in those units.
    ///
    /// # Errors
    ///
    /// `Err` naming the key for a missing dimension, a key that is not a
    /// dimension, or an empty unit expression. (Whether the expressions
    /// parse is the [`UnitRegistry`](super::UnitRegistry)'s question, asked
    /// when a unit is used.)
    pub fn new<K: AsRef<str>, V: Into<String>>(
        name: impl Into<String>,
        units: impl IntoIterator<Item = (K, V)>,
        boltzmann: F,
        coulomb: F,
    ) -> Result<Self, String> {
        let name = name.into();
        let mut table: HashMap<&'static str, String> = HashMap::new();
        for (key, unit) in units {
            let key = key.as_ref();
            let dim = PresetDim::ALL
                .into_iter()
                .find(|d| d.name() == key)
                .ok_or_else(|| {
                    format!("unit preset `{name}`: `{key}` is not a preset dimension")
                })?;
            let unit: String = unit.into();
            if unit.trim().is_empty() {
                return Err(format!("unit preset `{name}`: `{key}` has an empty unit"));
            }
            table.insert(dim.name(), unit);
        }
        if let Some(missing) = PresetDim::ALL
            .iter()
            .find(|d| !table.contains_key(d.name()))
        {
            return Err(format!(
                "unit preset `{name}` does not give a unit for `{}`",
                missing.name()
            ));
        }
        Ok(Self {
            name,
            units: table,
            boltzmann,
            coulomb,
        })
    }

    fn from_table(
        name: &str,
        units: [(&'static str, &'static str); 10],
        boltzmann: F,
        coulomb: F,
    ) -> Self {
        Self {
            name: name.to_owned(),
            units: units.into_iter().map(|(k, v)| (k, v.to_owned())).collect(),
            boltzmann,
            coulomb,
        }
    }

    /// LAMMPS `real`: Å, kcal/mol, fs, e.
    pub fn real() -> Self {
        Self::from_table(
            "real",
            [
                ("mass", "gram_per_mole"),
                ("length", "angstrom"),
                ("time", "femtosecond"),
                ("energy", "kilocalorie_per_mole"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "atmosphere"),
                ("velocity", "angstrom / femtosecond"),
                ("force", "kilocalorie_per_mole / angstrom"),
                ("density", "gram / centimeter ** 3"),
            ],
            // k_B per mole of particles: R in kcal·mol⁻¹·K⁻¹ (exact under
            // SI-2019, not LAMMPS `real`'s rounded `boltz` 0.0019872067).
            GAS_CONSTANT * J_TO_KCAL.get(),
            COULOMB_REAL,
        )
    }

    /// LAMMPS `metal`: Å, eV, ps, e.
    pub fn metal() -> Self {
        Self::from_table(
            "metal",
            [
                ("mass", "gram_per_mole"),
                ("length", "angstrom"),
                ("time", "picosecond"),
                ("energy", "electron_volt"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "bar"),
                ("velocity", "angstrom / picosecond"),
                ("force", "electron_volt / angstrom"),
                ("density", "gram / centimeter ** 3"),
            ],
            boltzmann_in("eV"),
            COULOMB_METAL,
        )
    }

    /// SI: kg, m, s, J.
    pub fn si() -> Self {
        Self::from_table(
            "si",
            [
                ("mass", "kilogram"),
                ("length", "meter"),
                ("time", "second"),
                ("energy", "joule"),
                ("temperature", "kelvin"),
                ("charge", "coulomb"),
                ("pressure", "pascal"),
                ("velocity", "meter / second"),
                ("force", "newton"),
                ("density", "kilogram / meter ** 3"),
            ],
            BOLTZMANN,
            COULOMB_CONSTANT,
        )
    }

    /// CGS.
    pub fn cgs() -> Self {
        Self::from_table(
            "cgs",
            [
                ("mass", "gram"),
                ("length", "centimeter"),
                ("time", "second"),
                ("energy", "erg"),
                ("temperature", "kelvin"),
                ("charge", "statcoulomb"),
                ("pressure", "dyne / centimeter ** 2"),
                ("velocity", "centimeter / second"),
                ("force", "dyne"),
                ("density", "gram / centimeter ** 3"),
            ],
            boltzmann_in("erg"),
            // Gaussian units define the statcoulomb by k_e = 1.
            1.0,
        )
    }

    /// Atomic / `electron` units.
    pub fn electron() -> Self {
        Self::from_table(
            "electron",
            [
                ("mass", "amu"),
                ("length", "bohr"),
                ("time", "femtosecond"),
                ("energy", "hartree"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "pascal"),
                ("velocity", "bohr / femtosecond"),
                ("force", "hartree / bohr"),
                ("density", "amu / bohr ** 3"),
            ],
            boltzmann_in("hartree"),
            // Atomic units define k_e = 1.
            1.0,
        )
    }

    /// Reduced LJ units. Numeric constants are 1; names follow the registry's
    /// `lj_*` definitions (`UnitRegistry::define_lj_units`, or
    /// `define_lj_sigma` for the length scale alone).
    pub fn lj() -> Self {
        Self::from_table(
            "lj",
            [
                ("mass", "lj_mass"),
                ("length", "lj_sigma"),
                ("time", "lj_tau"),
                ("energy", "lj_epsilon"),
                ("temperature", "lj_epsilon_over_kB"),
                ("charge", "lj_charge"),
                ("pressure", "lj_epsilon / lj_sigma ** 3"),
                ("velocity", "lj_sigma / lj_tau"),
                ("force", "lj_epsilon / lj_sigma"),
                ("density", "lj_mass / lj_sigma ** 3"),
            ],
            1.0,
            1.0,
        )
    }

    /// Microscopic (`micro`) style.
    pub fn micro() -> Self {
        Self::from_table(
            "micro",
            [
                ("mass", "picogram"),
                ("length", "micrometer"),
                ("time", "microsecond"),
                ("energy", "picogram * micrometer ** 2 / microsecond ** 2"),
                ("temperature", "kelvin"),
                ("charge", "picocoulomb"),
                ("pressure", "picogram / (micrometer * microsecond ** 2)"),
                ("velocity", "micrometer / microsecond"),
                ("force", "picogram * micrometer / microsecond ** 2"),
                ("density", "picogram / micrometer ** 3"),
            ],
            boltzmann_in("picogram * micrometer ** 2 / microsecond ** 2"),
            coulomb_in(
                "picogram * micrometer ** 2 / microsecond ** 2",
                "micrometer",
                "picocoulomb",
            ),
        )
    }

    /// Nanoscopic (`nano`) style.
    pub fn nano() -> Self {
        Self::from_table(
            "nano",
            [
                ("mass", "attogram"),
                ("length", "nanometer"),
                ("time", "nanosecond"),
                ("energy", "attogram * nanometer ** 2 / nanosecond ** 2"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "attogram / (nanometer * nanosecond ** 2)"),
                ("velocity", "nanometer / nanosecond"),
                ("force", "attogram * nanometer / nanosecond ** 2"),
                ("density", "attogram / nanometer ** 3"),
            ],
            boltzmann_in("attogram * nanometer ** 2 / nanosecond ** 2"),
            coulomb_in(
                "attogram * nanometer ** 2 / nanosecond ** 2",
                "nanometer",
                "elementary_charge",
            ),
        )
    }

    /// OpenMM's (and GROMACS's) unit system: nm, kJ/mol, ps, e.
    ///
    /// Not a LAMMPS style. `k_B` is the exact molar gas constant in
    /// kJ·mol⁻¹·K⁻¹ (`R / 1000`), and the Coulomb constant is
    /// [`COULOMB_REAL`] in kJ·nm·mol⁻¹·e⁻² (through the unit registry), so
    /// it prices charges exactly as the `real` preset does.
    pub fn openmm() -> Self {
        Self::from_table(
            "openmm",
            [
                ("mass", "gram_per_mole"),
                ("length", "nanometer"),
                ("time", "picosecond"),
                ("energy", "kilojoule_per_mole"),
                ("temperature", "kelvin"),
                ("charge", "elementary_charge"),
                ("pressure", "bar"),
                ("velocity", "nanometer / picosecond"),
                ("force", "kilojoule_per_mole / nanometer"),
                ("density", "gram / centimeter ** 3"),
            ],
            GAS_CONSTANT * J_TO_KJ.get(),
            COULOMB_REAL * KCAL_ANGSTROM_TO_KJ_NM.get(),
        )
    }

    /// The built-in preset `name` — a LAMMPS `units` style, or `openmm` —
    /// as its constructor builds it; `None` for any other name. Unlike
    /// [`lookup_unit_preset`], never a preset registered or replaced at run time.
    pub fn builtin(name: &str) -> Option<Self> {
        Some(match name {
            "real" => Self::real(),
            "metal" => Self::metal(),
            "si" => Self::si(),
            "cgs" => Self::cgs(),
            "electron" => Self::electron(),
            "lj" => Self::lj(),
            "micro" => Self::micro(),
            "nano" => Self::nano(),
            "openmm" => Self::openmm(),
            _ => return None,
        })
    }

    /// Preset name (`"real"`, `"metal"`, …).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Boltzmann constant in this system's energy / temperature units.
    pub fn boltzmann(&self) -> F {
        self.boltzmann
    }

    /// Coulomb constant in this system's energy · length / charge² units.
    pub fn coulomb(&self) -> F {
        self.coulomb
    }

    /// Unit name for `dimension`, or `None` if the preset does not define it.
    pub fn unit(&self, dimension: &str) -> Option<&str> {
        self.units.get(dimension).map(String::as_str)
    }

    pub fn mass(&self) -> &str {
        self.unit("mass").expect("preset defines mass")
    }
    pub fn length(&self) -> &str {
        self.unit("length").expect("preset defines length")
    }
    pub fn time(&self) -> &str {
        self.unit("time").expect("preset defines time")
    }
    pub fn energy(&self) -> &str {
        self.unit("energy").expect("preset defines energy")
    }
    pub fn temperature(&self) -> &str {
        self.unit("temperature")
            .expect("preset defines temperature")
    }
    pub fn charge(&self) -> &str {
        self.unit("charge").expect("preset defines charge")
    }
    pub fn pressure(&self) -> &str {
        self.unit("pressure").expect("preset defines pressure")
    }
    pub fn velocity(&self) -> &str {
        self.unit("velocity").expect("preset defines velocity")
    }
    pub fn force(&self) -> &str {
        self.unit("force").expect("preset defines force")
    }
    pub fn density(&self) -> &str {
        self.unit("density").expect("preset defines density")
    }
}

/// Named registry of [`UnitPreset`]s. Built-ins are pre-registered; callers
/// extend it with [`register`](Self::register).
pub struct UnitPresetRegistry {
    inner: HashMap<String, UnitPreset>,
}

impl UnitPresetRegistry {
    /// Empty registry (no built-ins).
    pub fn empty() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Built-in presets: the LAMMPS styles real, metal, si, cgs, electron,
    /// lj, micro and nano, plus openmm.
    pub fn new() -> Self {
        let mut reg = Self::empty();
        for name in [
            "real", "metal", "si", "cgs", "electron", "lj", "micro", "nano", "openmm",
        ] {
            let p = UnitPreset::builtin(name).expect("a built-in preset");
            reg.inner.insert(name.to_owned(), p);
        }
        reg
    }

    /// Insert `data` under `name`. Errors if the name is already taken.
    pub fn register(&mut self, name: impl Into<String>, data: UnitPreset) -> Result<(), String> {
        let name = name.into();
        if self.inner.contains_key(&name) {
            return Err(format!("unit preset `{name}` is already registered"));
        }
        self.inner.insert(name, data);
        Ok(())
    }

    /// Insert `data` under `name`, replacing (and returning) any preset
    /// already there.
    pub fn replace(&mut self, name: impl Into<String>, data: UnitPreset) -> Option<UnitPreset> {
        self.inner.insert(name.into(), data)
    }

    /// Look up a preset by name.
    pub fn get(&self, name: &str) -> Option<&UnitPreset> {
        self.inner.get(name)
    }

    /// Iterate registered `(name, preset)` pairs. Order is the map's.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &UnitPreset)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Number of registered presets.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for UnitPresetRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn global() -> &'static Mutex<UnitPresetRegistry> {
    static REGISTRY: OnceLock<Mutex<UnitPresetRegistry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(UnitPresetRegistry::new()))
}

/// Process-wide preset lookup (built-ins plus anything [`register_unit_preset`] added).
pub fn lookup_unit_preset(name: &str) -> Option<UnitPreset> {
    global().lock().ok()?.get(name).cloned()
}

/// Register an extra preset on the process-wide registry.
pub fn register_unit_preset(name: impl Into<String>, data: UnitPreset) -> Result<(), String> {
    global()
        .lock()
        .map_err(|e| e.to_string())?
        .register(name, data)
}

/// Put `data` under `name` on the process-wide registry, replacing any
/// preset there (a built-in included); returns the replaced one.
pub fn replace_unit_preset(
    name: impl Into<String>,
    data: UnitPreset,
) -> Result<Option<UnitPreset>, String> {
    Ok(global()
        .lock()
        .map_err(|e| e.to_string())?
        .replace(name, data))
}

/// Every preset name on the process-wide registry, sorted.
pub fn unit_preset_names() -> Vec<String> {
    let mut names: Vec<String> = global()
        .lock()
        .map(|reg| reg.iter().map(|(name, _)| name.to_owned()).collect())
        .unwrap_or_default();
    names.sort();
    names
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn micro_and_nano_constants_are_lammps_boltz_and_qqr2e() {
        // LAMMPS `force.cpp`: micro boltz 1.3806504e-8, qqr2e 8.987556e6;
        // nano boltz 0.013806504, qqr2e 230.7078669 (older CODATA, so 1e-5).
        let close = |a: F, b: F| ((a - b) / b).abs() < 1e-5;
        let micro = UnitPreset::micro();
        assert!(
            close(micro.boltzmann(), 1.380_650_4e-8),
            "{}",
            micro.boltzmann()
        );
        assert!(close(micro.coulomb(), 8.987_556e6), "{}", micro.coulomb());
        let nano = UnitPreset::nano();
        assert!(
            close(nano.boltzmann(), 0.013_806_504),
            "{}",
            nano.boltzmann()
        );
        assert!(close(nano.coulomb(), 230.707_866_9), "{}", nano.coulomb());
    }

    #[test]
    fn real_boltzmann_is_the_gas_constant_in_kcal() {
        // k_B in kcal·mol⁻¹·K⁻¹ = R / 4184 J, correctly rounded
        // (SI-2019 exact), not LAMMPS `real`'s rounded `boltz` (0.0019872067).
        assert_eq!(UnitPreset::real().boltzmann(), 1.987_204_258_640_831_6e-3);
    }

    #[test]
    fn real_energy_is_kcal_and_time_is_fs() {
        assert_eq!(UnitPreset::real().energy(), "kilocalorie_per_mole");
        assert_eq!(UnitPreset::real().time(), "femtosecond");
    }

    #[test]
    fn every_builtin_preset_reports_ten_dimensions() {
        let reg = UnitPresetRegistry::new();
        assert!(reg.len() >= 7);
        for (_name, preset) in reg.iter() {
            for dim in PresetDim::ALL {
                assert!(
                    preset.unit(dim.name()).is_some(),
                    "preset `{}` missing dimension `{}`",
                    preset.name(),
                    dim.name()
                );
            }
        }
    }

    #[test]
    fn preset_dim_all_names_are_exactly_the_preset_table_keys() {
        // `PresetDim` is the single list of dimension names; this pins it to
        // the real preset table: every name is a key, and there is no key it
        // misses (distinct names, same count).
        let real = UnitPreset::real();
        for dim in PresetDim::ALL {
            assert!(
                real.unit(dim.name()).is_some(),
                "`{}` is not a preset table key",
                dim.name()
            );
        }
        let names: std::collections::HashSet<&str> =
            PresetDim::ALL.iter().map(|d| d.name()).collect();
        assert_eq!(
            names.len(),
            PresetDim::ALL.len(),
            "duplicate PresetDim name"
        );
        assert_eq!(names.len(), real.units.len());
    }

    #[test]
    fn openmm_constants_are_reals_in_kilojoules_and_nanometres() {
        use crate::core::UnitRegistry;
        let p = UnitPreset::openmm();
        assert_eq!(p.length(), "nanometer");
        let units = UnitRegistry::new();
        // k_B: the real preset's value, converted kcal → kJ.
        let real = units
            .quantity(UnitPreset::real().boltzmann(), "kilocalorie_per_mole")
            .unwrap();
        let kj = real
            .to(&units.parse("kilojoule_per_mole").unwrap())
            .unwrap();
        assert!((p.boltzmann() - kj.value()).abs() < 1e-15);
        assert!((p.coulomb() - 138.935_456).abs() < 1e-4, "{}", p.coulomb());
        assert!(lookup_unit_preset("openmm").is_some());
        assert!(unit_preset_names().contains(&"openmm".to_string()));
    }

    #[test]
    fn a_custom_preset_needs_every_dimension() {
        let real = UnitPreset::real();
        let full: Vec<(&str, &str)> = PresetDim::ALL
            .iter()
            .map(|d| (d.name(), real.unit(d.name()).unwrap()))
            .collect();
        let p = UnitPreset::new("mine", full.clone(), 1.0, 2.0).unwrap();
        assert_eq!(p.name(), "mine");
        assert_eq!(p.energy(), "kilocalorie_per_mole");
        assert!(UnitPreset::new("short", full[..9].to_vec(), 1.0, 1.0).is_err());
        let mut extra = full.clone();
        extra.push(("colour", "red"));
        assert!(UnitPreset::new("extra", extra, 1.0, 1.0).is_err());
    }

    #[test]
    fn replace_overwrites_and_returns_the_old_preset() {
        let mut reg = UnitPresetRegistry::new();
        let old = reg.replace("real", UnitPreset::metal()).unwrap();
        assert_eq!(old.name(), "real");
        assert_eq!(reg.get("real").unwrap().energy(), "electron_volt");
    }

    #[test]
    fn lj_temperature_is_epsilon_over_boltzmann() {
        assert_eq!(UnitPreset::lj().temperature(), "lj_epsilon_over_kB");
    }

    #[test]
    fn every_dimension_of_every_builtin_preset_parses_after_define_lj_units() {
        use crate::core::UnitRegistry;
        let mut units = UnitRegistry::new();
        let mass = units.quantity(100.0, "gram_per_mole").unwrap();
        let sigma = units.quantity(4.2, "angstrom").unwrap();
        let epsilon = units.quantity(1.0, "kilocalorie_per_mole").unwrap();
        units.define_lj_units(&mass, &sigma, &epsilon).unwrap();

        let presets = UnitPresetRegistry::new();
        for (name, preset) in presets.iter() {
            for dim in PresetDim::ALL {
                let expr = preset
                    .unit(dim.name())
                    .unwrap_or_else(|| panic!("preset `{name}` lacks `{}`", dim.name()));
                assert!(
                    units.parse(expr).is_ok(),
                    "preset `{name}` dimension `{}`: `{expr}` does not parse: {:?}",
                    dim.name(),
                    units.parse(expr).err()
                );
            }
        }
    }

    #[test]
    fn register_rejects_a_duplicate_name() {
        let mut reg = UnitPresetRegistry::empty();
        reg.register("real", UnitPreset::real()).unwrap();
        let err = reg.register("real", UnitPreset::metal()).unwrap_err();
        assert!(err.contains("real"));
    }

    #[test]
    fn lookup_by_name_returns_the_real_preset() {
        assert_eq!(
            lookup_unit_preset("real").unwrap().boltzmann(),
            UnitPreset::real().boltzmann()
        );
    }
}
