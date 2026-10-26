//! The unit registry: definitions + prefix table + parse entry points.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::core::constants;
use crate::op::F;

use super::dimension::Dimension;
use super::error::UnitsError;
use super::quantity::Quantity;
use super::unit::Unit;

/// A single unit definition added to a [`UnitRegistry`].
///
/// Conversion to SI base is `si = value * factor + offset`, where `factor`
/// and `offset` are expressed in the SI base units of `dimension` (e.g.
/// `angstrom` has `factor = 1e-10` metres, `degC` has `offset = 273.15`
/// kelvin).
#[derive(Clone, Debug, PartialEq)]
pub struct UnitDef {
    /// Canonical name, e.g. `"calorie"`.
    pub name: String,
    /// Alternative names, e.g. `["cal"]`.
    pub aliases: Vec<String>,
    /// Preferred display symbol.
    pub symbol: String,
    /// Multiplicative factor to SI base (SI base units per one of this unit).
    pub factor: F,
    /// Additive offset to SI base, in SI base units (non-zero only for
    /// affine units such as `degC`).
    pub offset: F,
    /// Dimension of the unit.
    pub dimension: Dimension,
    /// Whether the unit accepts SI prefixes (kcal, nm, fs).
    pub prefixable: bool,
}

/// SI prefix table, longest prefix first so `da` wins over `d`.
///
/// `u` and `µ` (U+00B5) are both accepted for micro.
const PREFIXES: &[(&str, F)] = &[
    ("da", 1e1),
    ("y", 1e-24),
    ("z", 1e-21),
    ("a", 1e-18),
    ("f", 1e-15),
    ("p", 1e-12),
    ("n", 1e-9),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("m", 1e-3),
    ("c", 1e-2),
    ("d", 1e-1),
    ("h", 1e2),
    ("k", 1e3),
    ("M", 1e6),
    ("G", 1e9),
    ("T", 1e12),
    ("P", 1e15),
    ("E", 1e18),
    ("Z", 1e21),
    ("Y", 1e24),
];

/// Spelled-out SI prefixes accepted alongside their symbols.
///
/// This keeps Python attribute-style unit lookup intuitive (``nanometer``,
/// ``femtosecond``) without duplicating prefixed definitions in the registry.
const LONG_PREFIXES: &[(&str, F)] = &[
    ("quetta", 1e30),
    ("ronna", 1e27),
    ("yotta", 1e24),
    ("zetta", 1e21),
    ("exa", 1e18),
    ("peta", 1e15),
    ("tera", 1e12),
    ("giga", 1e9),
    ("mega", 1e6),
    ("kilo", 1e3),
    ("hecto", 1e2),
    ("deca", 1e1),
    ("deci", 1e-1),
    ("centi", 1e-2),
    ("milli", 1e-3),
    ("micro", 1e-6),
    ("nano", 1e-9),
    ("pico", 1e-12),
    ("femto", 1e-15),
    ("atto", 1e-18),
    ("zepto", 1e-21),
    ("yocto", 1e-24),
    ("ronto", 1e-27),
    ("quecto", 1e-30),
];

/// A registry of unit definitions and the SI prefix table.
///
/// The registry resolves unit names (canonical name, symbol, or alias,
/// optionally with one SI prefix) and parses compound expressions into
/// self-contained [`Unit`] values. Preload factors are SI-2019 exact or
/// CODATA 2018 recommended values; each entry in the preload tables below
/// cites its source.
///
/// # Examples
///
/// End-to-end: build a registry, parse units, convert a quantity.
///
/// ```
/// use molrs::core::{UnitRegistry, UnitsError};
///
/// let reg = UnitRegistry::new();
///
/// // A force gradient of 2 kcal/(mol·Å), converted to kJ/(mol·nm).
/// let g = reg.quantity(2.0, "kcal/mol/angstrom")?;
/// let target = reg.parse("kJ/mol/nm")?;
/// let converted = g.to(&target)?;
/// assert!((converted.value() - 83.68).abs() < 1e-9);
/// # Ok::<(), UnitsError>(())
/// ```
pub struct UnitRegistry {
    defs: Vec<UnitDef>,
    /// Canonical name, symbol, and every alias → index into `defs`.
    index: HashMap<String, usize>,
}

static GLOBAL_REGISTRY: OnceLock<UnitRegistry> = OnceLock::new();

impl UnitRegistry {
    /// Rebuild a registry from an exact definition snapshot.
    ///
    /// Unlike [`empty`](Self::empty), this starts with no implicit SI base
    /// definitions. Callers should therefore pass the complete snapshot from
    /// [`definitions`](Self::definitions). This pair is intended for durable
    /// language-binding serialization.
    pub fn from_definitions(defs: Vec<UnitDef>) -> Result<UnitRegistry, UnitsError> {
        let mut registry = UnitRegistry {
            defs: Vec::new(),
            index: HashMap::new(),
        };
        for definition in defs {
            registry.define(definition)?;
        }
        Ok(registry)
    }

    /// Preloaded with SI + molecular-simulation units.
    ///
    /// Covers the SI base set plus the MD working set: `angstrom`, `bohr`,
    /// `calorie`/`kcal`, `electron_volt`, `hartree`, `dalton`/`amu`, `bar`,
    /// `atmosphere`, `degC`, `radian`/`degree`, `elementary_charge`, `debye`,
    /// and all prefixable SI derivatives (`nm`, `fs`, `kJ`, ...).
    pub fn new() -> UnitRegistry {
        let mut r = UnitRegistry::empty();
        for def in md_defs() {
            r.define(def)
                .expect("preloaded MD unit table must be collision-free");
        }
        r
    }

    /// SI base units only — for building a custom system from scratch.
    ///
    /// Includes `gram` as the prefixable mass atom (the SI base unit is
    /// `kilogram`, which itself takes no further prefix).
    pub fn empty() -> UnitRegistry {
        let mut r = UnitRegistry {
            defs: Vec::new(),
            index: HashMap::new(),
        };
        for def in base_defs() {
            r.define(def)
                .expect("SI base unit table must be collision-free");
        }
        r
    }

    /// Shared immutable preloaded registry (OnceLock singleton).
    pub fn global() -> &'static UnitRegistry {
        GLOBAL_REGISTRY.get_or_init(UnitRegistry::new)
    }

    /// The factor that converts a value in `from` to `to`:
    /// `value_in_to = value_in_from × factor`.
    ///
    /// The factor is `from`'s SI factor over `to`'s, except for two cases
    /// where that quotient of rounded doubles is not the correctly rounded
    /// answer:
    ///
    /// - a ratio within a few ulps of a power of ten is that power of ten,
    ///   correctly rounded (`cm^3 → angstrom^3` is exactly `1e24`, not
    ///   `1e24` plus an ulp; `m^3 → angstrom^3` is `1e30`). Every SI prefix
    ///   and the ångström are exact powers of ten, so a product of them is
    ///   one too, however many rounded multiplications built it;
    /// - otherwise, when the inverse ratio is a whole number, the factor is
    ///   that number's reciprocal.
    ///
    /// ```
    /// use molrs::core::UnitRegistry;
    ///
    /// let reg = UnitRegistry::global();
    /// assert_eq!(reg.factor("kcal", "kJ").unwrap(), 4.184);
    /// assert_eq!(reg.factor("nm", "angstrom").unwrap(), 10.0);
    /// assert_eq!(reg.factor("angstrom", "nm").unwrap(), 0.1);
    /// ```
    ///
    /// # Errors
    ///
    /// Either expression does not parse, the two differ in dimension, or
    /// one is affine (`degC`).
    pub fn factor(&self, from: &str, to: &str) -> Result<F, UnitsError> {
        let from = self.parse(from)?;
        let to = self.parse(to)?;
        let direct = from.factor_to(&to)?;
        if let Some(power) = nearest_power_of_ten(direct) {
            return Ok(power);
        }
        let inverse = to.factor_to(&from)?;
        Ok(if inverse >= 1.0 && inverse.fract() == 0.0 {
            1.0 / inverse
        } else {
            direct
        })
    }

    /// Define a unit, registering its name, symbol, and all aliases.
    ///
    /// # Errors
    ///
    /// [`UnitsError::Redefinition`] if the name, symbol, or any alias is
    /// already registered.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::core::{Dimension, UnitDef, UnitRegistry, UnitsError};
    ///
    /// let mut reg = UnitRegistry::empty();
    /// reg.define(UnitDef {
    ///     name: "smoot".to_string(),
    ///     aliases: vec![],
    ///     symbol: "smoot".to_string(),
    ///     factor: 1.7018, // metres per smoot
    ///     offset: 0.0,
    ///     dimension: Dimension::LENGTH,
    ///     prefixable: false,
    /// })?;
    /// assert!(reg.parse("smoot").is_ok());
    /// # Ok::<(), UnitsError>(())
    /// ```
    pub fn define(&mut self, def: UnitDef) -> Result<(), UnitsError> {
        let mut keys: Vec<&str> = Vec::with_capacity(2 + def.aliases.len());
        keys.push(&def.name);
        if def.symbol != def.name {
            keys.push(&def.symbol);
        }
        for alias in &def.aliases {
            if !keys.contains(&alias.as_str()) {
                keys.push(alias);
            }
        }
        for key in &keys {
            if self.index.contains_key(*key) {
                return Err(UnitsError::Redefinition {
                    name: (*key).to_string(),
                });
            }
        }
        let idx = self.defs.len();
        let keys: Vec<String> = keys.into_iter().map(str::to_string).collect();
        self.defs.push(def);
        for key in keys {
            self.index.insert(key, idx);
        }
        Ok(())
    }

    /// Parse a compound unit expression (`"kcal/mol/angstrom"`, `"m s^-2"`).
    ///
    /// Supports `*`, `/`, `·`, whitespace (implicit multiplication),
    /// parentheses, `^`/`**` integer exponents, numeric factors, and one SI
    /// prefix per atom. Affine units (`degC`) are only legal as a single
    /// bare atom, never inside a compound expression.
    ///
    /// # Errors
    ///
    /// - [`UnitsError::UnknownUnit`] — an atom resolves to no definition.
    /// - [`UnitsError::Parse`] — malformed expression (empty input, bad
    ///   exponent, unbalanced parentheses, trailing tokens).
    /// - [`UnitsError::AffineUnit`] — an affine unit appears in a compound
    ///   expression.
    pub fn parse(&self, expr: &str) -> Result<Unit, UnitsError> {
        super::parse::parse_expr(self, expr)
    }

    /// Convenience: `registry.quantity(1.5, "kcal/mol")`.
    ///
    /// # Errors
    ///
    /// Same as [`UnitRegistry::parse`].
    pub fn quantity(&self, value: F, expr: &str) -> Result<Quantity, UnitsError> {
        Ok(Quantity::new(value, self.parse(expr)?))
    }

    /// Define the reduced-LJ length unit `lj_sigma` alone.
    ///
    /// **Reduced Lennard-Jones (LJ) units** measure every quantity in multiples
    /// of the three parameters of the LJ pair potential
    /// `E(r) = 4ε[(σ/r)¹² − (σ/r)⁶]`: the length σ (where `E` crosses zero),
    /// the energy ε (the well depth) and a particle mass m. `lj_sigma` is the
    /// unit whose size is `sigma`.
    ///
    /// With only σ known, exactly the quantities of dimension L^a (no mass,
    /// no time) have a defined reduced scale; every other `lj_*` unit stays
    /// unknown, so a conversion that needs one is refused by the parser.
    ///
    /// # Errors
    ///
    /// - [`UnitsError::DimensionMismatch`] — `sigma` is not a length.
    /// - [`UnitsError::Parse`] — `sigma` is not finite and positive.
    /// - [`UnitsError::Redefinition`] — `lj_sigma` is already defined.
    pub fn define_lj_sigma(&mut self, sigma: &Quantity) -> Result<(), UnitsError> {
        let definition = self.lj_sigma_def(sigma)?;
        self.define(definition)
    }

    /// The `lj_sigma` definition for `sigma`, validated but not yet defined.
    fn lj_sigma_def(&self, sigma: &Quantity) -> Result<UnitDef, UnitsError> {
        let sigma_m = sigma.to(&self.parse("m")?)?.value();
        if !sigma_m.is_finite() || sigma_m <= 0.0 {
            return Err(UnitsError::Parse {
                expr: "Lennard-Jones sigma".to_string(),
                message: "sigma must be finite and positive".to_string(),
            });
        }
        Ok(def(
            "lj_sigma",
            "lj_sigma",
            &[],
            sigma_m,
            0.0,
            Dimension::LENGTH,
            false,
        ))
    }

    /// Define Lennard-Jones reduced units from physical mass, length, and energy scales.
    ///
    /// Defines `lj_sigma` (as [`define_lj_sigma`](Self::define_lj_sigma) does),
    /// `lj_mass`, `lj_epsilon`, `lj_tau`, `lj_epsilon_over_kB` and
    /// `lj_charge`, following the LAMMPS `lj` relations
    /// (<https://docs.lammps.org/units.html>):
    ///
    /// - time `tau = sigma * sqrt(mass / epsilon)`;
    /// - temperature `epsilon / k_B` (`k_B` the Boltzmann constant);
    /// - charge `sqrt(4 pi eps0 sigma epsilon)` (`eps0` the vacuum
    ///   permittivity), evaluated as
    ///   `e * sqrt(sigma[Å] * epsilon[kcal/mol] / COULOMB_REAL)` with
    ///   [`COULOMB_REAL`](crate::core::constants::COULOMB_REAL) and stored in
    ///   coulomb.
    ///
    /// The definitions retain their physical dimensions, so normal checked
    /// conversion works without a separate context engine.
    ///
    /// # Errors
    ///
    /// - [`UnitsError::DimensionMismatch`] — a scale has the wrong dimension.
    /// - [`UnitsError::Parse`] — a scale is not finite and positive.
    /// - [`UnitsError::Redefinition`] — an `lj_*` name is already defined
    ///   (including `lj_sigma` from an earlier `define_lj_sigma`).
    pub fn define_lj_units(
        &mut self,
        mass: &Quantity,
        sigma: &Quantity,
        epsilon: &Quantity,
    ) -> Result<(), UnitsError> {
        let sigma_def = self.lj_sigma_def(sigma)?;
        let sigma_m = sigma_def.factor;
        let mass_kg = mass.to(&self.parse("kg")?)?.value();
        let sigma_angstrom = sigma.to(&self.parse("angstrom")?)?.value();
        let epsilon_j = epsilon.to(&self.parse("J")?)?.value();
        let epsilon_kcal_mol = epsilon.to(&self.parse("kilocalorie_per_mole")?)?.value();
        if !mass_kg.is_finite() || !epsilon_j.is_finite() || mass_kg <= 0.0 || epsilon_j <= 0.0 {
            return Err(UnitsError::Parse {
                expr: "Lennard-Jones scales".to_string(),
                message: "mass and epsilon must be finite and positive".to_string(),
            });
        }

        let tau_s = (mass_kg * sigma_m * sigma_m / epsilon_j).sqrt();
        let temperature_k = epsilon_j / constants::BOLTZMANN;
        let charge_c = crate::core::constants::ELEMENTARY_CHARGE
            * (sigma_angstrom * epsilon_kcal_mol / crate::core::constants::COULOMB_REAL).sqrt();
        let definitions = [
            sigma_def,
            def(
                "lj_mass",
                "lj_mass",
                &[],
                mass_kg,
                0.0,
                Dimension::MASS,
                false,
            ),
            def(
                "lj_epsilon",
                "lj_epsilon",
                &[],
                epsilon_j,
                0.0,
                Dimension::ENERGY,
                false,
            ),
            def("lj_tau", "lj_tau", &[], tau_s, 0.0, Dimension::TIME, false),
            def(
                "lj_epsilon_over_kB",
                "lj_epsilon_over_kB",
                &[],
                temperature_k,
                0.0,
                Dimension::TEMPERATURE,
                false,
            ),
            def(
                "lj_charge",
                "lj_charge",
                &[],
                charge_c,
                0.0,
                Dimension::CHARGE,
                false,
            ),
        ];
        // Refuse a clash before defining anything, so a failed call leaves
        // the registry unchanged. Each `lj_*` def has no alias and its
        // symbol is its name, so the name is its only index key.
        if let Some(clash) = definitions
            .iter()
            .find(|d| self.index.contains_key(&d.name))
        {
            return Err(UnitsError::Redefinition {
                name: clash.name.clone(),
            });
        }
        for definition in definitions {
            self.define(definition)?;
        }
        Ok(())
    }

    /// Iterate all registered definitions (test/introspection support).
    pub fn definitions(&self) -> impl Iterator<Item = &UnitDef> {
        self.defs.iter()
    }

    /// Resolve a single unit atom: exact name/symbol/alias first, then one
    /// SI prefix + prefixable atom. Returns `(factor, offset, dimension)`.
    pub(crate) fn resolve_atom(&self, atom: &str) -> Option<(F, F, Dimension)> {
        if let Some(&idx) = self.index.get(atom) {
            let def = &self.defs[idx];
            return Some((def.factor, def.offset, def.dimension));
        }
        for (prefix, scale) in PREFIXES {
            let Some(rest) = atom.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if let Some(&idx) = self.index.get(rest) {
                let def = &self.defs[idx];
                if def.prefixable && def.offset == 0.0 {
                    return Some((scale * def.factor, 0.0, def.dimension));
                }
            }
        }
        for (prefix, scale) in LONG_PREFIXES {
            let Some(rest) = atom.strip_prefix(prefix) else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if let Some(&idx) = self.index.get(rest) {
                let def = &self.defs[idx];
                if def.prefixable && def.offset == 0.0 {
                    return Some((scale * def.factor, 0.0, def.dimension));
                }
            }
        }
        None
    }
}

/// The power of ten `x` is within rounding error of, correctly rounded, or
/// `None`. The tolerance (8 ulps, relative 1.8e-15) bounds the error of the
/// handful of rounded products and quotients a factor of SI prefixes and
/// powers is built from; no CODATA-valued unit lies that close to a power
/// of ten.
fn nearest_power_of_ten(x: F) -> Option<F> {
    if !x.is_finite() || x <= 0.0 {
        return None;
    }
    let exponent = x.log10().round() as i32;
    // `1eN` parsed from decimal text is the correctly rounded double.
    let power: F = format!("1e{exponent}").parse().ok()?;
    ((x - power).abs() <= 8.0 * F::EPSILON * power).then_some(power)
}

/// Shorthand constructor for the preload tables.
fn def(
    name: &str,
    symbol: &str,
    aliases: &[&str],
    factor: F,
    offset: F,
    dimension: Dimension,
    prefixable: bool,
) -> UnitDef {
    UnitDef {
        name: name.to_string(),
        aliases: aliases.iter().map(|s| s.to_string()).collect(),
        symbol: symbol.to_string(),
        factor,
        offset,
        dimension,
        prefixable,
    }
}

/// The 7 SI base units plus `gram` (the prefixable mass atom).
fn base_defs() -> Vec<UnitDef> {
    const LUMINOSITY: Dimension = Dimension::from_exponents([0, 0, 0, 0, 0, 0, 1]);
    vec![
        def("meter", "m", &[], 1.0, 0.0, Dimension::LENGTH, true),
        def("kilogram", "kg", &[], 1.0, 0.0, Dimension::MASS, false),
        def("gram", "g", &[], 1e-3, 0.0, Dimension::MASS, true),
        def("second", "s", &["sec"], 1.0, 0.0, Dimension::TIME, true),
        def("ampere", "A", &[], 1.0, 0.0, Dimension::CURRENT, true),
        def("kelvin", "K", &[], 1.0, 0.0, Dimension::TEMPERATURE, true),
        def("mole", "mol", &[], 1.0, 0.0, Dimension::AMOUNT, true),
        def("candela", "cd", &[], 1.0, 0.0, LUMINOSITY, true),
    ]
}

/// Derived + molecular-simulation unit set.
///
/// Factors are SI-2019 exact or CODATA 2018 recommended values; each entry
/// cites its source.
fn md_defs() -> Vec<UnitDef> {
    const CHARGE_LENGTH: Dimension = Dimension::from_exponents([1, 0, 1, 1, 0, 0, 0]);
    let l = Dimension::LENGTH;
    let e = Dimension::ENERGY;
    vec![
        // Length. angstrom: exact; bohr: CODATA 2018 a0
        // ([`constants::BOHR_RADIUS`]).
        def("angstrom", "Å", &["ang"], 1e-10, 0.0, l, false),
        def(
            "bohr",
            "bohr",
            &["a0"],
            constants::BOHR_RADIUS,
            0.0,
            l,
            false,
        ),
        // Energy. joule: SI derived; calorie: thermochemical, exact 4.184 J
        // (the one definition of the calorie: kcal ↔ kJ is this unit's);
        // eV: e × 1 V; hartree: CODATA 2018 E_h.
        def("joule", "J", &[], 1.0, 0.0, e, true),
        def("calorie", "cal", &[], 4.184, 0.0, e, true),
        def(
            "kilocalorie_per_mole",
            "kcal_per_mol",
            &[],
            4184.0 / constants::AVOGADRO,
            0.0,
            e,
            false,
        ),
        def(
            "kilojoule_per_mole",
            "kJ_per_mol",
            &[],
            1000.0 / constants::AVOGADRO,
            0.0,
            e,
            false,
        ),
        def("erg", "erg", &[], 1e-7, 0.0, e, false),
        def(
            "electron_volt",
            "eV",
            &[],
            constants::ELEMENTARY_CHARGE,
            0.0,
            e,
            true,
        ),
        def(
            "hartree",
            "Eh",
            &[],
            constants::HARTREE_ENERGY,
            0.0,
            e,
            false,
        ),
        // Boltzmann constant k_B (SI-2019 exact) as a unit, J/K.
        def(
            "boltzmann_constant",
            "k_B",
            &[],
            constants::BOLTZMANN,
            0.0,
            Dimension::ENERGY / Dimension::TEMPERATURE,
            false,
        ),
        // Force / pressure (SI derived, exact).
        def("newton", "N", &[], 1.0, 0.0, Dimension::FORCE, true),
        def("dyne", "dyn", &[], 1e-5, 0.0, Dimension::FORCE, false),
        def("pascal", "Pa", &[], 1.0, 0.0, Dimension::PRESSURE, true),
        def("bar", "bar", &[], 1e5, 0.0, Dimension::PRESSURE, false),
        def(
            "atmosphere",
            "atm",
            &[],
            101_325.0,
            0.0,
            Dimension::PRESSURE,
            false,
        ),
        // Time (exact).
        def("minute", "min", &[], 60.0, 0.0, Dimension::TIME, false),
        def("hour", "h", &[], 3600.0, 0.0, Dimension::TIME, false),
        // Frequency (SI derived, exact). Prefixable: MHz, GHz, THz.
        def("hertz", "Hz", &[], 1.0, 0.0, Dimension::FREQUENCY, true),
        // Mass. dalton: CODATA 2018; prefixable for kDa.
        def(
            "dalton",
            "Da",
            &["amu"],
            constants::ATOMIC_MASS_CONSTANT,
            0.0,
            Dimension::MASS,
            true,
        ),
        def(
            "gram_per_mole",
            "g_per_mol",
            &[],
            1e-3 / constants::AVOGADRO,
            0.0,
            Dimension::MASS,
            false,
        ),
        // Temperature: affine, offset 273.15 K (exact).
        def(
            "degC",
            "degC",
            &["celsius", "°C"],
            1.0,
            273.15,
            Dimension::TEMPERATURE,
            false,
        ),
        // Angle (dimensionless). degree: exact π/180.
        def(
            "radian",
            "rad",
            &[],
            1.0,
            0.0,
            Dimension::DIMENSIONLESS,
            true,
        ),
        def(
            "degree",
            "deg",
            &[],
            std::f64::consts::PI / 180.0,
            0.0,
            Dimension::DIMENSIONLESS,
            false,
        ),
        // Charge. coulomb: SI derived; elementary charge: SI-2019 exact;
        // debye: 1e-21/c C·m, c exact.
        def("coulomb", "C", &[], 1.0, 0.0, Dimension::CHARGE, true),
        def(
            "statcoulomb",
            "statC",
            &[],
            0.1 / constants::SPEED_OF_LIGHT,
            0.0,
            Dimension::CHARGE,
            false,
        ),
        def(
            "elementary_charge",
            "e",
            &[],
            constants::ELEMENTARY_CHARGE,
            0.0,
            Dimension::CHARGE,
            false,
        ),
        def(
            "debye",
            "D",
            &[],
            1e-21 / constants::SPEED_OF_LIGHT,
            0.0,
            CHARGE_LENGTH,
            false,
        ),
    ]
}

impl Default for UnitRegistry {
    fn default() -> UnitRegistry {
        UnitRegistry::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn custom_def(name: &str, aliases: &[&str]) -> UnitDef {
        UnitDef {
            name: name.to_string(),
            aliases: aliases.iter().map(|s| s.to_string()).collect(),
            symbol: name.to_string(),
            factor: 1.0,
            offset: 0.0,
            dimension: Dimension::LENGTH,
            prefixable: false,
        }
    }

    #[test]
    fn powers_of_ten_are_correctly_rounded() {
        let reg = UnitRegistry::global();
        let cases: &[(&str, &str, F)] = &[
            ("angstrom", "nm", 0.1),
            ("nm", "angstrom", 10.0),
            ("cm^3", "angstrom^3", 1e24),
            ("angstrom^3", "cm^3", 1e-24),
            ("m^3", "angstrom^3", 1e30),
            ("angstrom^3", "m^3", 1e-30),
            ("nm^3", "angstrom^3", 1e3),
            ("cm^2", "angstrom^2", 1e16),
            ("m^2", "nm^2", 1e18),
            ("mm^2", "cm^2", 0.01),
            ("km/ms", "m/s", 1e6),
            ("ns/um", "ps/nm", 1.0),
            ("g/cm^3", "kg/m^3", 1e3),
            ("kg/m^3", "g/cm^3", 1e-3),
            ("fs", "ps", 1e-3),
            ("mg*cm^-3", "ug*angstrom^-3", 1e-21),
        ];
        for &(from, to, want) in cases {
            let got = reg.factor(from, to).unwrap();
            assert_eq!(got, want, "{from} -> {to}: {got:e}");
            assert_eq!(got, format!("{want:e}").parse::<F>().unwrap());
        }
    }

    #[test]
    fn non_powers_of_ten_are_not_snapped() {
        let reg = UnitRegistry::global();
        assert_eq!(reg.factor("kcal", "kJ").unwrap(), 4.184);
        let bohr = reg.factor("bohr", "angstrom").unwrap();
        assert!((bohr - 0.529177210903).abs() < 1e-15);
    }

    #[test]
    fn define_new_unit_ok() {
        let mut r = UnitRegistry::empty();
        assert!(r.define(custom_def("smoot", &["sm"])).is_ok());
        // newly defined unit is parseable
        assert!(r.parse("smoot").is_ok());
    }

    #[test]
    fn redefinition_is_error() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        let err = r.define(custom_def("smoot", &[])).unwrap_err();
        assert!(matches!(err, UnitsError::Redefinition { .. }));
    }

    #[test]
    fn alias_collision_is_error() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &["sm"])).unwrap();
        // second unit reuses an existing alias
        let err = r.define(custom_def("widget", &["sm"])).unwrap_err();
        assert!(matches!(err, UnitsError::Redefinition { .. }));
    }

    #[test]
    fn custom_registry_independent_of_global() {
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        // global / default registry must not know the custom unit
        assert!(UnitRegistry::global().parse("smoot").is_err());
        assert!(UnitRegistry::new().parse("smoot").is_err());
    }

    #[test]
    fn spelled_out_prefixes_resolve_without_duplicate_definitions() {
        let r = UnitRegistry::new();
        for (long, short) in [
            ("nanometer", "nm"),
            ("femtosecond", "fs"),
            ("kilocalorie", "kcal"),
        ] {
            let long = r.parse(long).unwrap();
            let short = r.parse(short).unwrap();
            assert_eq!(long.dimension(), short.dimension());
            assert_eq!(long.factor_to(&short).unwrap(), 1.0);
        }
    }

    #[test]
    fn boltzmann_constant_is_a_unit_of_energy_per_temperature() {
        let r = UnitRegistry::new();
        let kt = r.quantity(300.0, "k_B * kelvin").unwrap();
        let kj = kt.to(&r.parse("kilojoule_per_mole").unwrap()).unwrap();
        // R T at 300 K = 2.494 338 785 kJ/mol.
        assert!(
            (kj.value() - 2.494_338_785_445_6).abs() < 1e-9,
            "{}",
            kj.value()
        );
        let long = r.parse("boltzmann_constant").unwrap();
        assert_eq!(long.factor_to(&r.parse("k_B").unwrap()).unwrap(), 1.0);
    }

    #[test]
    fn per_particle_molar_named_units_convert_to_energy_and_mass() {
        let r = UnitRegistry::new();
        let kcal = r.quantity(1.0, "kilocalorie_per_mole").unwrap();
        let kj = kcal.to(&r.parse("kilojoule_per_mole").unwrap()).unwrap();
        assert!((kj.value() - 4.184).abs() < 1e-12);
        assert!(kcal.to(&r.parse("eV").unwrap()).is_ok());
        assert!(
            r.quantity(1.0, "gram_per_mole")
                .unwrap()
                .to(&r.parse("Da").unwrap())
                .is_ok()
        );
    }

    #[test]
    fn lennard_jones_scales_are_native_convertible_units() {
        let source = UnitRegistry::new();
        let mass = source.quantity(39.948, "amu").unwrap();
        let sigma = source.quantity(3.405, "angstrom").unwrap();
        let epsilon = source.quantity(0.2381, "kilocalorie_per_mole").unwrap();
        let mut reduced = UnitRegistry::new();
        reduced.define_lj_units(&mass, &sigma, &epsilon).unwrap();

        let sigma_star = reduced
            .quantity(3.405, "angstrom")
            .unwrap()
            .to(&reduced.parse("lj_sigma").unwrap())
            .unwrap();
        assert!((sigma_star.value() - 1.0).abs() < 1e-12);
        let tau_ps = reduced
            .quantity(1.0, "lj_tau")
            .unwrap()
            .to(&reduced.parse("ps").unwrap())
            .unwrap();
        assert!((tau_ps.value() - 2.16).abs() < 0.01);
        let temperature = reduced
            .quantity(1.0, "lj_epsilon_over_kB")
            .unwrap()
            .to(&reduced.parse("K").unwrap())
            .unwrap();
        assert!((temperature.value() - 119.8).abs() < 0.5);
    }

    #[test]
    fn empty_has_si_base_only() {
        let r = UnitRegistry::empty();
        // SI base present
        assert!(r.parse("meter").is_ok());
        assert!(r.parse("kg").is_ok());
        assert!(r.parse("s").is_ok());
        assert!(r.parse("mol").is_ok());
        assert!(r.parse("K").is_ok());
        // MD-set units absent in empty registry
        assert!(r.parse("kcal").is_err());
        assert!(r.parse("angstrom").is_err());
        assert!(r.parse("eV").is_err());
    }

    #[test]
    fn new_has_md_units() {
        let r = UnitRegistry::new();
        assert!(r.parse("kcal").is_ok());
        assert!(r.parse("angstrom").is_ok());
        assert!(r.parse("eV").is_ok());
        assert!(r.parse("hartree").is_ok());
        assert!(r.parse("bohr").is_ok());
        assert!(r.parse("atm").is_ok());
    }

    #[test]
    fn hertz_is_prefixable_inverse_time() {
        let r = UnitRegistry::new();
        for spelling in ["Hz", "hertz", "GHz", "gigahertz", "THz"] {
            assert!(r.parse(spelling).is_ok(), "{spelling} must parse");
        }
        // 1 GHz = 1 ns⁻¹ (exact).
        let ghz = r.quantity(1.0, "GHz").unwrap();
        let per_ns = ghz.to(&r.parse("1/ns").unwrap()).unwrap();
        assert!(
            (per_ns.value() - 1.0).abs() < 1e-12,
            "1 GHz = {} / ns",
            per_ns.value()
        );
    }

    #[test]
    fn quantity_convenience() {
        let r = UnitRegistry::new();
        let q = r.quantity(1.5, "kcal").unwrap();
        assert_eq!(q.value(), 1.5);
    }

    #[test]
    fn defined_unit_converts_correctly() {
        // define() must yield correct conversions, not merely parse success.
        // 1 smoot = 1.7018 m (MIT bridge convention).
        let mut r = UnitRegistry::empty();
        let mut def = custom_def("smoot", &[]);
        def.factor = 1.7018;
        r.define(def).unwrap();
        let smoot = r.parse("smoot").unwrap();
        let meter = r.parse("m").unwrap();
        let factor = smoot.factor_to(&meter).unwrap();
        assert!(
            ((factor - 1.7018) / 1.7018).abs() <= 1e-15,
            "smoot->m factor = {factor}"
        );
        // and through the Quantity path
        let q = r.quantity(2.0, "smoot").unwrap().to(&meter).unwrap();
        assert!(
            ((q.value() - 3.4036) / 3.4036).abs() <= 1e-15,
            "2 smoot = {} m",
            q.value()
        );
    }

    #[test]
    fn defined_prefixable_unit_accepts_prefix() {
        let mut r = UnitRegistry::empty();
        let mut def = custom_def("blip", &[]);
        def.factor = 2.0;
        def.dimension = Dimension::TIME;
        def.prefixable = true;
        r.define(def).unwrap();
        let kblip = r.parse("kblip").unwrap();
        let blip = r.parse("blip").unwrap();
        assert_eq!(kblip.factor_to(&blip).unwrap(), 1000.0);
        assert_eq!(kblip.dimension(), Dimension::TIME);
    }

    #[test]
    fn definitions_iterates_all_defs() {
        let r = UnitRegistry::empty();
        let base_count = r.definitions().count();
        assert!(base_count >= 8, "SI base set missing: {base_count}");
        let mut r = UnitRegistry::empty();
        r.define(custom_def("smoot", &[])).unwrap();
        assert!(r.definitions().any(|d| d.name == "smoot"));
        // preloaded registry strictly larger than base set
        assert!(UnitRegistry::new().definitions().count() > base_count);
    }

    // ---- Reduced LJ units -------------------------------------------------
    //
    // Goldens hand-derived (spec assembly-02 Domain basis) for sigma = 4.2 A,
    // m = 100 g/mol, epsilon = 1 kcal/mol, from the LAMMPS `lj` relations
    // (docs.lammps.org/units.html): tau = sigma*sqrt(m/eps), T = eps/k_B,
    // q = sqrt(4 pi eps0 sigma eps) = e*sqrt(sigma[A]*eps[kcal/mol]/332.06371),
    // v = sigma/tau, p = eps/sigma^3, rho = m/sigma^3, f = eps/sigma.

    fn lj_registry() -> UnitRegistry {
        let mut r = UnitRegistry::new();
        let mass = r.quantity(100.0, "gram_per_mole").unwrap();
        let sigma = r.quantity(4.2, "angstrom").unwrap();
        let epsilon = r.quantity(1.0, "kilocalorie_per_mole").unwrap();
        r.define_lj_units(&mass, &sigma, &epsilon).unwrap();
        r
    }

    /// Value in `target` of one reduced unit of preset dimension `dim`.
    fn one_lj(dim: &str, target: &str) -> F {
        let r = lj_registry();
        let lj = crate::core::UnitPreset::lj();
        let from = r.parse(lj.unit(dim).unwrap()).unwrap();
        from.factor_to(&r.parse(target).unwrap()).unwrap()
    }

    fn assert_rel(got: F, want: F) {
        let rel = ((got - want) / want).abs();
        assert!(rel < 1e-6, "got {got}, want {want} (rel {rel:e})");
    }

    #[test]
    fn lj_length_golden() {
        assert_rel(1.5 * one_lj("length", "angstrom"), 6.3);
    }

    #[test]
    fn lj_time_golden() {
        assert_rel(one_lj("time", "femtosecond"), 2053.3049);
    }

    #[test]
    fn lj_temperature_golden() {
        assert_rel(one_lj("temperature", "kelvin"), 503.21953);
    }

    #[test]
    fn lj_charge_golden() {
        assert_rel(one_lj("charge", "elementary_charge"), 0.1124641);
    }

    #[test]
    fn lj_velocity_golden() {
        assert_rel(one_lj("velocity", "angstrom / femtosecond"), 2.045483e-3);
    }

    #[test]
    fn lj_pressure_golden() {
        assert_rel(one_lj("pressure", "atmosphere"), 925.4997);
    }

    #[test]
    fn lj_density_golden() {
        assert_rel(one_lj("density", "gram / centimeter ** 3"), 2.241306);
    }

    #[test]
    fn lj_force_golden() {
        assert_rel(
            one_lj("force", "kilocalorie_per_mole / angstrom"),
            0.2380952,
        );
    }

    #[test]
    fn lj_mass_golden() {
        assert_rel(one_lj("mass", "gram_per_mole"), 100.0);
    }

    #[test]
    fn define_lj_units_defines_mass_and_charge_with_their_dimensions() {
        let r = lj_registry();
        assert_eq!(r.parse("lj_mass").unwrap().dimension(), Dimension::MASS);
        assert_eq!(r.parse("lj_charge").unwrap().dimension(), Dimension::CHARGE);
    }

    #[test]
    fn define_lj_sigma_alone_defines_only_sigma() {
        let mut r = UnitRegistry::new();
        let sigma = r.quantity(4.2, "angstrom").unwrap();
        r.define_lj_sigma(&sigma).unwrap();
        let s = r.parse("lj_sigma").unwrap();
        assert_eq!(s.dimension(), Dimension::LENGTH);
        assert_rel(s.factor_to(&r.parse("angstrom").unwrap()).unwrap(), 4.2);
        for absent in ["lj_mass", "lj_epsilon", "lj_tau", "lj_charge"] {
            assert!(
                matches!(r.parse(absent), Err(UnitsError::UnknownUnit { .. })),
                "`{absent}` must stay unknown after define_lj_sigma"
            );
        }
    }

    #[test]
    fn define_lj_units_after_define_lj_sigma_is_a_redefinition() {
        let mut r = UnitRegistry::new();
        let mass = r.quantity(100.0, "gram_per_mole").unwrap();
        let sigma = r.quantity(4.2, "angstrom").unwrap();
        let epsilon = r.quantity(1.0, "kilocalorie_per_mole").unwrap();
        r.define_lj_sigma(&sigma).unwrap();
        let err = r.define_lj_units(&mass, &sigma, &epsilon).unwrap_err();
        assert!(
            matches!(err, UnitsError::Redefinition { ref name } if name == "lj_sigma"),
            "got {err:?}"
        );
    }
}
