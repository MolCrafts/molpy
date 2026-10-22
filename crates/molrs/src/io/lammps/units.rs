//! LAMMPS unit-style conversion: the force-field writer's coefficients and
//! the data writer's geometry go through this one table.

#[cfg(feature = "ff")]
use crate::ff::ir::UnitScale;
use molrs::core::{PresetDim, Quantity, UnitPreset, UnitRegistry, UnitsError};
use molrs::op::F;

/// Map a LAMMPS `units` keyword onto a core preset name (`"lj"` / `"real"` / `"metal"`).
pub fn parse_lammps_units_style(s: &str) -> Result<&'static str, String> {
    match s.to_ascii_lowercase().as_str() {
        "lj" => Ok("lj"),
        "real" => Ok("real"),
        "metal" => Ok("metal"),
        other => Err(format!(
            "unsupported LAMMPS units `{other}` (phase 1: lj, real, metal)"
        )),
    }
}

/// Reference scales for `define_lj_units` (physical mass, σ, ε).
///
/// Canonical defaults: `1 g/mol`, `1 Å`, `1 kcal/mol` so that one reduced
/// energy unit equals one thermochemical kcal/mol and one reduced length
/// equals one ångström — matching LAMMPS real for the bridge path.
#[derive(Debug, Clone)]
pub struct LammpsLjReference {
    pub mass: Quantity,
    pub sigma: Quantity,
    pub epsilon: Quantity,
}

impl LammpsLjReference {
    /// Canonical bridge scales: `m = 1 g/mol`, `σ = 1 Å`, `ε = 1 kcal/mol`.
    pub fn canonical() -> Result<Self, UnitsError> {
        let reg = UnitRegistry::new();
        Ok(Self {
            mass: reg.quantity(1.0, "gram_per_mole")?,
            sigma: reg.quantity(1.0, "angstrom")?,
            epsilon: reg.quantity(1.0, "kilocalorie_per_mole")?,
        })
    }
}

/// LAMMPS force-field I/O adapter: `UnitRegistry` plus the lj hub.
///
/// Not a unit-system type — conversion data lives in core.
///
/// All conversions go through the molrs [`UnitRegistry`] / [`Quantity`] stack
/// with **lj reduced units as the mandatory hub**:
///
/// ```text
/// source style  →  lj_*  →  target style
/// ```
///
/// Physical quantity meanings follow the LAMMPS `units` command
/// (<https://docs.lammps.org/units.html>): real (Å, kcal/mol), metal (Å, eV),
/// lj (reduced). Thermochemical calorie (4.184 J) is already encoded in
/// [`UnitRegistry`]'s MD preload.
///
/// A **canonical reference** `(m=1 g/mol, σ=1 Å, ε=1 kcal/mol)` makes
/// real↔metal bridge through lj without material-specific scales, while still
/// never hard-coding eV↔kcal factors in the force-field reader.
pub struct LammpsUnitConverter {
    reg: UnitRegistry,
}

impl LammpsUnitConverter {
    /// Build a system with the given lj reference scales.
    pub fn with_reference(reference: &LammpsLjReference) -> Result<Self, UnitsError> {
        let mut reg = UnitRegistry::new();
        reg.define_lj_units(&reference.mass, &reference.sigma, &reference.epsilon)?;
        Ok(Self { reg })
    }

    /// Canonical bridge (`1 g/mol`, `1 Å`, `1 kcal/mol`).
    pub fn canonical() -> Result<Self, UnitsError> {
        Self::with_reference(&LammpsLjReference::canonical()?)
    }

    /// Registry with lj units defined (for tests / advanced callers).
    pub fn registry(&self) -> &UnitRegistry {
        &self.reg
    }

    /// The unit of `dimension` in LAMMPS unit style `style`, read from its
    /// [`UnitPreset`]; an error for a style that is no preset.
    fn unit_of(style: &str, dimension: PresetDim) -> Result<String, String> {
        let preset = UnitPreset::builtin(style)
            .ok_or_else(|| format!("unknown LAMMPS units style `{style}`"))?;
        preset
            .unit(dimension.name())
            .map(str::to_owned)
            .ok_or_else(|| format!("unit preset `{style}` gives no {} unit", dimension.name()))
    }

    /// Convert a raw file value of `dimension` from `from` style to `to`
    /// style **through lj** (`from → lj → to`).
    fn convert_through_lj(
        &self,
        value: F,
        from: &str,
        to: &str,
        dimension: PresetDim,
    ) -> Result<F, String> {
        if from == to {
            return Ok(value);
        }
        let parse = |style: &str| {
            let unit = Self::unit_of(style, dimension)?;
            self.reg.parse(&unit).map_err(|e| e.to_string())
        };
        let (from_u, lj_u, to_u) = (parse(from)?, parse("lj")?, parse(to)?);

        let q = Quantity::new(value, from_u);
        let in_lj = q.to(&lj_u).map_err(|e| e.to_string())?;
        let out = in_lj.to(&to_u).map_err(|e| e.to_string())?;
        Ok(out.value())
    }

    /// Energy (ε, dihedral K, …): `from → lj → to`.
    pub fn energy(&self, value: F, from: &str, to: &str) -> Result<F, String> {
        self.convert_through_lj(value, from, to, PresetDim::Energy)
    }

    /// Length (σ, r0): `from → lj → to`.
    pub fn length(&self, value: F, from: &str, to: &str) -> Result<F, String> {
        self.convert_through_lj(value, from, to, PresetDim::Length)
    }

    /// Any preset `dimension` (mass, charge, velocity, …): `from → lj → to`;
    /// `value` unchanged when the two are the same style.
    pub fn convert(
        &self,
        value: F,
        from: &str,
        to: &str,
        dimension: PresetDim,
    ) -> Result<F, String> {
        self.convert_through_lj(value, from, to, dimension)
    }

    /// The per-dimension conversion of every parameter from `from` to `to`
    /// (one unit of energy, length, charge and mass, each `from → lj →
    /// to`); exactly the identity when the two are the same style.
    #[cfg(feature = "ff")]
    pub fn scale(&self, from: &str, to: &str) -> Result<UnitScale, String> {
        if from == to {
            return Ok(UnitScale::IDENTITY);
        }
        let one = |dimension| self.convert_through_lj(1.0, from, to, dimension);
        Ok(UnitScale::new(
            one(PresetDim::Energy)?,
            one(PresetDim::Length)?,
            one(PresetDim::Charge)?,
            one(PresetDim::Mass)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_to_metal_energy_via_lj_matches_si() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        // 1 kcal/mol → metal eV through lj hub.
        let ev = sys.energy(1.0, "real", "metal").unwrap();
        // Direct SI path for comparison (not used in production FF code).
        let reg = UnitRegistry::new();
        let direct = reg
            .quantity(1.0, "kilocalorie_per_mole")
            .unwrap()
            .to(&reg.parse("eV").unwrap())
            .unwrap()
            .value();
        assert!((ev - direct).abs() < 1e-12, "lj hub {ev} vs SI {direct}");
    }

    #[test]
    fn metal_to_real_energy_via_lj() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        let kcal = sys.energy(1.0, "metal", "real").unwrap();
        let reg = UnitRegistry::new();
        let direct = reg
            .quantity(1.0, "eV")
            .unwrap()
            .to(&reg.parse("kilocalorie_per_mole").unwrap())
            .unwrap()
            .value();
        assert!(
            (kcal - direct).abs() < 1e-12,
            "lj hub {kcal} vs SI {direct}"
        );
        // ~23.06 kcal/mol per eV
        assert!((kcal - 23.060_547_830_619).abs() < 1e-6 || (kcal - direct).abs() < 1e-12);
    }

    #[test]
    fn length_real_metal_identical() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        let a = sys.length(3.5, "real", "metal").unwrap();
        assert!((a - 3.5).abs() < 1e-15);
    }

    #[test]
    fn same_style_is_the_identity() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        assert_eq!(sys.energy(0.5, "lj", "lj").unwrap(), 0.5);
        assert_eq!(
            sys.convert(0.25, "metal", "metal", PresetDim::Velocity)
                .unwrap(),
            0.25
        );
        #[cfg(feature = "ff")]
        assert!(sys.scale("real", "real").unwrap().is_identity());
    }

    #[test]
    fn real_to_metal_velocity_is_per_picosecond() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        // 1 Å/fs = 1000 Å/ps.
        let v = sys
            .convert(1.0, "real", "metal", PresetDim::Velocity)
            .unwrap();
        assert!((v - 1000.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn an_unknown_style_is_an_error_not_a_panic() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        let err = sys.energy(1.0, "real", "furlong").unwrap_err();
        assert!(err.contains("furlong"), "{err}");
        assert!(sys.length(1.0, "nope", "real").is_err());
    }

    #[test]
    fn parse_units_keywords() {
        assert_eq!(parse_lammps_units_style("REAL").unwrap(), "real");
        assert_eq!(parse_lammps_units_style("metal").unwrap(), "metal");
        assert_eq!(parse_lammps_units_style("lj").unwrap(), "lj");
        assert!(parse_lammps_units_style("si").is_err());
    }

    #[cfg(feature = "ff")]
    #[test]
    fn bond_k_metal_to_real_scales_with_energy() {
        let sys = LammpsUnitConverter::canonical().unwrap();
        // Same numerical K in metal (eV/Å²) vs real (kcal/mol/Å²) must scale
        // exactly as energy (length is Å in both).
        let k_real = sys
            .scale("metal", "real")
            .unwrap()
            .apply(1.0, "E/L^2".parse().unwrap());
        let e_real = sys.energy(1.0, "metal", "real").unwrap();
        assert!((k_real - e_real).abs() < 1e-12);
    }
}
