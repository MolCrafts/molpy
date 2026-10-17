//! Engine codecs: how a style of the force-field IR is written to, and read
//! from, an engine's format (`ff-ir-02-protocol` §8).

pub(crate) mod lammps;

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

use crate::ff::ir::Params;
use crate::ff::ir::{IrError, ParamDimension, ParamKind, ParamSpec, ParamValue, StyleSpec};
use molrs::op::F;

/// The engine formats molrs reads or writes.
///
/// Engine I/O is driven by the protocol, not by a table per writer:
///
/// * **LAMMPS** — a style carries a [`LammpsForm`] on its [`StyleSpec`].
///   [`LammpsForm::Positional`] is derived from the spec alone:
///   `<category>_style <name>` (`pair_style <name> <cutoff>`) and
///   `<category>_coeff <type> v₁ … vₙ` in `params` order, each value
///   converted by its [`ParamDimension`] ([`UnitScale`]) from the force field's units
///   to the file's — so a style registered with a spec reads and writes
///   with nothing else written. A line LAMMPS does not spell positionally
///   (`fourier`'s term count, `nharmonic`'s `N`, `lj/charmm`'s 1-4 pair and
///   switch, `class2`'s cross-term lines, `fix cmap`) has a
///   [`LammpsForm::Custom`] codec ([`LammpsCodec`]). A style with
///   [`LammpsForm::None`] is refused by name; an expression-only one
///   because the installed LAMMPS has no `LEPTON` package.
/// * **OpenMM XML** — an expression style needs no codec: its expression is
///   the `Custom*Force` energy, rewritten so the parameters stay in IR units
///   (`4.184*(E[r → 10*r])`); see
///   [`OpenmmXmlWriter`](crate::io::openmm_xml::OpenmmXmlWriter).
/// * **GROMACS, AMBER prmtop and frcmod** hold the built-in styles they have
///   directives for and refuse every other style.
///
/// Every refusal is [`IrError::NoEngineForm`] naming the engine, the
/// category, the style and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Engine {
    Lammps,
    OpenmmXml,
    Gromacs,
    AmberPrmtop,
    AmberFrcmod,
}

impl Engine {
    /// The name an error message gives it.
    pub fn name(self) -> &'static str {
        match self {
            Engine::Lammps => "LAMMPS",
            Engine::OpenmmXml => "OpenMM XML",
            Engine::Gromacs => "GROMACS",
            Engine::AmberPrmtop => "AMBER prmtop",
            Engine::AmberFrcmod => "AMBER frcmod",
        }
    }

    /// `lammps`, `openmm`, `gromacs`, `prmtop`, `frcmod` (any case).
    pub fn parse(name: &str) -> Result<Engine, String> {
        match name.to_ascii_lowercase().as_str() {
            "lammps" => Ok(Engine::Lammps),
            "openmm" | "openmm_xml" | "openmmxml" => Ok(Engine::OpenmmXml),
            "gromacs" => Ok(Engine::Gromacs),
            "prmtop" | "amber" => Ok(Engine::AmberPrmtop),
            "frcmod" => Ok(Engine::AmberFrcmod),
            other => Err(format!(
                "engine {other:?}: one of lammps, openmm, gromacs, prmtop, frcmod"
            )),
        }
    }

    /// The refusal of `category` `style` by this engine, for `reason`.
    pub fn refuse(self, category: &str, style: &str, reason: impl Into<String>) -> IrError {
        IrError::NoEngineForm {
            engine: self.name().to_owned(),
            category: category.to_owned(),
            style: style.to_owned(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Engine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// An engine form of one style: what a codec of any engine is.
pub trait EngineCodec: Send + Sync + 'static {
    /// The engine it writes.
    fn engine(&self) -> Engine;

    /// The engine's name of `spec`'s style (`fourier` for `dihedral
    /// periodic`).
    fn name(&self, spec: &StyleSpec) -> String;
}

/// One number of a coefficient line: LAMMPS parses some as integers
/// (`utils::inumeric`), which a decimal point would make it refuse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Token {
    Real(F),
    Int(i64),
}

impl Token {
    pub fn value(self) -> F {
        match self {
            Token::Real(v) => v,
            Token::Int(n) => n as F,
        }
    }

    /// `precision` decimals for a real, none for an integer.
    pub fn render(self, precision: usize) -> String {
        match self {
            Token::Real(v) => format!("{v:.precision$}"),
            Token::Int(n) => n.to_string(),
        }
    }

    /// `v` as an integer token when it is one, else `None`.
    pub fn integral(v: F) -> Option<Token> {
        (v.fract() == 0.0 && v.abs() < 1e15).then_some(Token::Int(v as i64))
    }
}

/// Unit conversion of a parameter by its [`ParamDimension`]: a value of dimension
/// `E^e·L^l·A^a·Q^q·M^m` is multiplied by `f_E^e · f_L^l · f_Q^q · f_M^m`
/// (D2: an angle value — exactly `A` — converts only between `units.angle`
/// values, the degree in every preset, so not at all; a per-radian exponent
/// never converts).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UnitScale {
    /// `None`: the identity (the same units on both sides), exact.
    factors: Option<[F; 4]>,
}

impl UnitScale {
    pub const IDENTITY: UnitScale = UnitScale { factors: None };

    /// One unit of energy, length, charge and mass of the source, in the
    /// target's.
    pub fn new(energy: F, length: F, charge: F, mass: F) -> Self {
        Self {
            factors: Some([energy, length, charge, mass]),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.factors.is_none()
    }

    /// `value` of dimension `dim`, in the target units.
    pub fn apply(&self, value: F, dim: ParamDimension) -> F {
        let Some([e, l, q, m]) = self.factors else {
            return value;
        };
        if dim.is_angle_value() {
            return value;
        }
        let pow = |f: F, n: i8| if n == 0 { 1.0 } else { f.powi(n as i32) };
        value * pow(e, dim.energy) * pow(l, dim.length) * pow(q, dim.charge) * pow(m, dim.mass)
    }
}

/// What a [`LammpsCodec`] writes for one type: the numbers after its type
/// field(s), and any further coefficient lines of the same type, each a
/// keyword and its numbers (`class2`'s `bb`, `ba`, `mbt`, …).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LammpsCoeffs {
    pub values: Vec<Token>,
    pub extra: Vec<(&'static str, Vec<Token>)>,
}

/// A style's LAMMPS lines, both ways. Every method but [`write`] and
/// [`read`] has the positional default.
///
/// Values are written in the file's units (`units` converts from the force
/// field's) and read as the file states them: the force field read is in
/// the file's units.
///
/// [`write`]: LammpsCodec::write
/// [`read`]: LammpsCodec::read
pub trait LammpsCodec: EngineCodec {
    /// The coefficient line(s) of one type, from its parameters.
    fn write(
        &self,
        spec: &StyleSpec,
        params: &Params,
        units: &UnitScale,
    ) -> Result<LammpsCoeffs, String>;

    /// One type's parameters from the numbers after its type field(s).
    fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String>;

    /// The keywords of the further coefficient lines [`write`] emits.
    ///
    /// [`write`]: LammpsCodec::write
    fn extra_keywords(&self) -> &'static [&'static str] {
        &[]
    }

    /// A further coefficient line `<keyword> values…` of a type: `Ok` when
    /// the IR holds it (all it adds is in `params`), an error naming what
    /// it cannot hold otherwise.
    fn parse_extra(
        &self,
        spec: &StyleSpec,
        keyword: &str,
        values: &[&str],
        params: &mut Params,
    ) -> Result<(), String> {
        let _ = (values, params);
        Err(format!(
            "{} {}: no `{keyword}` coefficient line",
            spec.category, spec.name
        ))
    }

    /// Refuse a style whose style-level parameters the LAMMPS lines cannot
    /// carry.
    fn check_style(&self, spec: &StyleSpec, style: &Params) -> Result<(), String> {
        positional::check_style(spec, style)
    }

    /// The numbers after the name on the `<category>_style` line (a pair
    /// style's global cutoff), in the file's units.
    fn style_args(
        &self,
        spec: &StyleSpec,
        style: &Params,
        units: &UnitScale,
    ) -> Result<Vec<Token>, String> {
        positional::style_args(spec, style, units)
    }

    /// The style-level parameters a `<category>_style` line's numbers
    /// state.
    fn parse_style_args(&self, spec: &StyleSpec, args: &[&str]) -> Result<Params, String> {
        positional::parse_style_args(spec, args)
    }

    /// The mixing rule the LAMMPS style always uses, whatever `pair_modify`
    /// says (`lj/class2`: `sixthpower`); `None` when `pair_modify mix`
    /// decides.
    fn fixed_mixing(&self) -> Option<&'static str> {
        None
    }

    /// The `pair_modify` keywords (`mix <rule>`, `shift yes`) the style's
    /// parameters need; empty for a bonded style.
    fn pair_modify(&self, spec: &StyleSpec, style: &Params) -> Result<Vec<String>, String> {
        positional::pair_modify(self.fixed_mixing(), spec, style)
    }
}

/// How a style is written to and read from LAMMPS.
#[derive(Clone, Default)]
pub enum LammpsForm {
    /// LAMMPS has no style of this form: refused by name.
    #[default]
    None,
    /// The LAMMPS style `name` (the style's own when `None`), its
    /// coefficients the spec's `params` in order, converted per dimension.
    Positional { name: Option<Cow<'static, str>> },
    /// A codec of its own.
    Custom(Arc<dyn LammpsCodec>),
}

impl PartialEq for LammpsForm {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (LammpsForm::None, LammpsForm::None) => true,
            (LammpsForm::Positional { name: a }, LammpsForm::Positional { name: b }) => a == b,
            (LammpsForm::Custom(a), LammpsForm::Custom(b)) => {
                std::ptr::addr_eq(Arc::as_ptr(a), Arc::as_ptr(b))
            }
            _ => false,
        }
    }
}

impl fmt::Debug for LammpsForm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LammpsForm::None => f.write_str("None"),
            LammpsForm::Positional { name } => {
                f.debug_struct("Positional").field("name", name).finish()
            }
            LammpsForm::Custom(_) => f.write_str("Custom(..)"),
        }
    }
}

/// The positional codec: [`LammpsForm::Positional`] as a [`LammpsCodec`].
static POSITIONAL: positional::Positional = positional::Positional;

impl LammpsForm {
    /// The positional form under the style's own name.
    pub fn positional() -> Self {
        LammpsForm::Positional { name: None }
    }

    /// The positional form under the LAMMPS name `name`.
    pub fn positional_named(name: impl Into<Cow<'static, str>>) -> Self {
        LammpsForm::Positional {
            name: Some(name.into()),
        }
    }

    /// The codec, `None` for [`LammpsForm::None`].
    pub fn codec(&self) -> Option<&dyn LammpsCodec> {
        match self {
            LammpsForm::None => None,
            LammpsForm::Positional { .. } => Some(&POSITIONAL),
            LammpsForm::Custom(c) => Some(c.as_ref()),
        }
    }

    /// The LAMMPS style name `spec` is written under, `None` without a
    /// form.
    pub fn lammps_name(&self, spec: &StyleSpec) -> Option<String> {
        match self {
            LammpsForm::None => None,
            LammpsForm::Positional { name } => {
                Some(name.as_deref().unwrap_or(&spec.name).to_owned())
            }
            LammpsForm::Custom(c) => Some(c.name(spec)),
        }
    }

    /// The codec, or the refusal naming why `spec` has none.
    pub fn require(&self, spec: &StyleSpec) -> Result<&dyn LammpsCodec, IrError> {
        self.codec().ok_or_else(|| {
            let reason = if spec.expression.is_some() {
                "an expression style needs LAMMPS's LEPTON package, which the installed LAMMPS \
                 has not (register it with a LAMMPS form when LAMMPS has a style of its form)"
            } else {
                "it is registered without a LAMMPS form"
            };
            Engine::Lammps.refuse(&spec.category, &spec.name, reason)
        })
    }

    /// Refuse a positional form the spec cannot have (`ff-ir-02-protocol`
    /// §8): a category with no `<category>_style`, a Text, Array or
    /// indexed parameter, a style parameter other than `cutoff` and
    /// `mixing`.
    pub fn check_spec(&self, spec: &StyleSpec) -> Result<(), IrError> {
        match self {
            LammpsForm::Positional { .. } => positional::check_spec(spec),
            _ => Ok(()),
        }
    }
}

/// The LAMMPS categories with a `<category>_style` command.
pub const LAMMPS_STYLE_CATEGORIES: [&str; 5] = ["pair", "bond", "angle", "dihedral", "improper"];

/// The positional codec and the pieces custom codecs build on.
pub mod positional {
    use super::*;

    /// [`LammpsForm::Positional`]'s codec.
    pub struct Positional;

    impl EngineCodec for Positional {
        fn engine(&self) -> Engine {
            Engine::Lammps
        }

        /// The style's own name; a [`LammpsForm::Positional`] with a name of
        /// its own answers [`LammpsForm::lammps_name`] instead.
        fn name(&self, spec: &StyleSpec) -> String {
            spec.name.to_string()
        }
    }

    impl LammpsCodec for Positional {
        fn write(
            &self,
            spec: &StyleSpec,
            params: &Params,
            units: &UnitScale,
        ) -> Result<LammpsCoeffs, String> {
            check_spec(spec).map_err(|e| e.to_string())?;
            Ok(LammpsCoeffs {
                values: write(spec, params, units)?,
                extra: Vec::new(),
            })
        }

        fn read(&self, spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
            check_spec(spec).map_err(|e| e.to_string())?;
            read(spec, values)
        }
    }

    pub(super) fn check_spec(spec: &StyleSpec) -> Result<(), IrError> {
        let refuse = |reason: String| Engine::Lammps.refuse(&spec.category, &spec.name, reason);
        if !LAMMPS_STYLE_CATEGORIES.contains(&spec.category.as_ref()) {
            return Err(refuse(format!(
                "LAMMPS has no `{}_style` command",
                spec.category
            )));
        }
        for p in &spec.params {
            if p.kind != ParamKind::Scalar {
                return Err(refuse(format!(
                    "parameter `{}` is no number, and a positional coefficient line holds \
                     numbers",
                    p.name
                )));
            }
            if p.indexed {
                return Err(refuse(format!(
                    "parameter `{}` is indexed, which a positional line cannot count (a codec of \
                     its own spells the count)",
                    p.name
                )));
            }
        }
        for p in &spec.style_params {
            if !matches!(p.name.as_ref(), "cutoff" | "mixing") {
                return Err(refuse(format!(
                    "style parameter `{}`: a positional style line carries only `cutoff` (and \
                     `pair_modify mix` the `mixing`)",
                    p.name
                )));
            }
        }
        Ok(())
    }

    /// A row's value of `p`, else its default, else an error naming it.
    pub fn value(spec: &StyleSpec, p: &ParamSpec, params: &Params) -> Result<F, String> {
        params
            .get(&p.name)
            .and_then(|v| v.as_num())
            .or_else(|| p.default.as_ref().and_then(ParamValue::as_num))
            .ok_or_else(|| {
                format!(
                    "{} {}: missing parameter `{}` (it has no default)",
                    spec.category, spec.name, p.name
                )
            })
    }

    /// One parameter's token: a dimensionless integral value as an integer
    /// (LAMMPS reads multiplicities and signs with `inumeric`), every other
    /// value converted by its dimension.
    pub fn token(p: &ParamSpec, v: F, units: &UnitScale) -> Token {
        if p.dim == ParamDimension::NONE
            && let Some(t) = Token::integral(v)
        {
            return t;
        }
        Token::Real(units.apply(v, p.dim))
    }

    /// Refuse a numeric parameter of `params` the spec does not declare:
    /// writing the line would drop it. A key that prices nothing (an
    /// annotation, an estimate's provenance) is no parameter, and passes
    /// ([`StyleSpec::undeclared_param`]).
    pub fn refuse_undeclared_params(spec: &StyleSpec, params: &Params) -> Result<(), String> {
        match spec.undeclared_param(params) {
            Some(key) => Err(format!(
                "{} {}: parameter `{key}` has no place on the LAMMPS coefficient line",
                spec.category, spec.name
            )),
            None => Ok(()),
        }
    }

    /// `params` in the spec's order, converted.
    pub fn write(
        spec: &StyleSpec,
        params: &Params,
        units: &UnitScale,
    ) -> Result<Vec<Token>, String> {
        refuse_undeclared_params(spec, params)?;
        spec.params
            .iter()
            .map(|p| Ok(token(p, value(spec, p, params)?, units)))
            .collect()
    }

    /// Exactly one number per parameter, in the spec's order.
    pub fn read(spec: &StyleSpec, values: &[&str]) -> Result<Params, String> {
        let names: Vec<&str> = spec.params.iter().map(|p| p.name.as_ref()).collect();
        parse_named(spec, &names, values)
    }

    /// Exactly `names.len()` numbers, stored under `names`.
    pub fn parse_named(
        spec: &StyleSpec,
        names: &[&str],
        values: &[&str],
    ) -> Result<Params, String> {
        if values.len() != names.len() {
            return Err(format!(
                "{} {} takes {} coefficients ({}), got {}{}",
                spec.category,
                spec.name,
                names.len(),
                names.join(" "),
                values.len(),
                match names.get(values.len()) {
                    Some(missing) if values.len() < names.len() => format!(": missing `{missing}`"),
                    _ => String::new(),
                }
            ));
        }
        let mut params = Params::new();
        for (name, raw) in names.iter().zip(values) {
            params.set(name, number(name, raw)?);
        }
        Ok(params)
    }

    /// `raw` as a number, an error naming `what` otherwise.
    pub fn number(what: &str, raw: &str) -> Result<F, String> {
        raw.parse::<F>()
            .map_err(|_| format!("{what} is not a number: {raw:?}"))
    }

    /// The style-level keys a LAMMPS file never carries, which a writer
    /// passes over: the instance expression (the IR's own), and the 1-4
    /// weights a compile projects.
    const PASSED: [&str; 3] = ["expression", "lj14scale", "coulomb14scale"];

    /// Every style-level parameter is `cutoff`, `mixing`, or at its declared
    /// default.
    pub fn check_style(spec: &StyleSpec, style: &Params) -> Result<(), String> {
        check_style_except(spec, style, &[])
    }

    /// [`check_style`], with `also` carried by the codec.
    pub fn check_style_except(
        spec: &StyleSpec,
        style: &Params,
        also: &[&str],
    ) -> Result<(), String> {
        let carried = |k: &str| matches!(k, "cutoff" | "mixing") || also.contains(&k);
        let stated = spec.style_params_off_default(style);
        let refuse = |key: &str, value: String| {
            Err(format!(
                "{} {}: style parameter `{key}` = {value} has no place on a LAMMPS line",
                spec.category, spec.name
            ))
        };
        if let Some((key, value)) = stated
            .iter()
            .find(|(key, _)| !carried(key) && !PASSED.contains(key))
        {
            return refuse(key, format!("{value:?}"));
        }
        Ok(())
    }

    /// A pair style's cutoff, required when the spec declares one (the
    /// writer never invents a cutoff).
    pub fn style_args(
        spec: &StyleSpec,
        style: &Params,
        units: &UnitScale,
    ) -> Result<Vec<Token>, String> {
        let Some(p) = spec.style_param("cutoff") else {
            return Ok(Vec::new());
        };
        let cutoff = style
            .get("cutoff")
            .and_then(|v| v.as_num())
            .ok_or_else(|| {
                format!(
                    "{} style '{}' has no cutoff: declare one on the style or write with \
                 skip_pair_style",
                    spec.category, spec.name
                )
            })?;
        Ok(vec![Token::Real(units.apply(cutoff, p.dim))])
    }

    /// The cutoff of a `pair_style <name> <cutoff>` line.
    pub fn parse_style_args(spec: &StyleSpec, args: &[&str]) -> Result<Params, String> {
        let names: &[&str] = if spec.style_param("cutoff").is_some() {
            &["cutoff"]
        } else {
            &[]
        };
        if args.len() != names.len() {
            return Err(format!(
                "{}_style {} takes {} argument(s) ({}), got {}",
                spec.category,
                spec.name,
                names.len(),
                names.join(" "),
                args.len()
            ));
        }
        let mut params = Params::new();
        for (name, raw) in names.iter().zip(args) {
            params.set(name, number(name, raw)?);
        }
        Ok(params)
    }

    /// `mix <rule>` for a style that mixes by `pair_modify`: its declared
    /// rule, else the spec's default.
    pub fn pair_modify(
        fixed: Option<&'static str>,
        spec: &StyleSpec,
        style: &Params,
    ) -> Result<Vec<String>, String> {
        let Some(p) = spec.style_param("mixing") else {
            return Ok(Vec::new());
        };
        let rule = style
            .get("mixing")
            .and_then(|v| v.as_text())
            .or_else(|| p.default.as_ref().and_then(ParamValue::as_text))
            .unwrap_or("arithmetic");
        match fixed {
            Some(fixed) if fixed == rule => Ok(Vec::new()),
            Some(fixed) => Err(format!(
                "{} {}: mixing {rule:?}, and LAMMPS's style always mixes {fixed}",
                spec.category, spec.name
            )),
            None => Ok(vec![format!("mix {rule}")]),
        }
    }
}
