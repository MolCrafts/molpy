//! What a style is: its ordered per-type parameters, its style parameters,
//! where its numbers come from, and (optionally) its energy as an
//! expression.

use std::borrow::Cow;

use ndarray::ArrayD;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::LammpsForm;
use crate::ff::ir::Params;
use crate::ff::ir::{CoulombRange, PairRole, ParamSource, SpecialClass};
use crate::ff::ir::{IrError, ParamDimension};
use molrs::op::F;

/// A parameter value: a number, a string, or an N-dimensional array.
///
/// The value of one key of a [`Params`] bag, and a [`ParamSpec`]'s default.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamValue {
    /// A number.
    Num(F),
    /// A string.
    Text(Cow<'static, str>),
    /// An N-dimensional `f64` array (a CMAP correction's `grid`).
    Array(ArrayD<F>),
}

impl ParamValue {
    /// The number, or `None` when the value is not [`ParamValue::Num`].
    pub fn as_num(&self) -> Option<F> {
        match self {
            ParamValue::Num(v) => Some(*v),
            _ => None,
        }
    }

    /// The string, or `None` when the value is not [`ParamValue::Text`].
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ParamValue::Text(s) => Some(s),
            _ => None,
        }
    }

    /// The array, or `None` when the value is not [`ParamValue::Array`].
    pub fn as_array(&self) -> Option<&ArrayD<F>> {
        match self {
            ParamValue::Array(a) => Some(a),
            _ => None,
        }
    }
}

impl From<F> for ParamValue {
    fn from(v: F) -> Self {
        ParamValue::Num(v)
    }
}

impl From<&str> for ParamValue {
    fn from(v: &str) -> Self {
        ParamValue::Text(Cow::Owned(v.to_owned()))
    }
}

impl From<&String> for ParamValue {
    fn from(v: &String) -> Self {
        ParamValue::Text(Cow::Owned(v.clone()))
    }
}

impl From<String> for ParamValue {
    fn from(v: String) -> Self {
        ParamValue::Text(Cow::Owned(v))
    }
}

impl From<ArrayD<F>> for ParamValue {
    fn from(v: ArrayD<F>) -> Self {
        ParamValue::Array(v)
    }
}

/// The shape of one parameter's value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// One number per row.
    Scalar,
    /// An `f64` array of this rank per row (`cmap` `grid`: rank 2), the
    /// table's column `f64[T, S…]`. Not an expression variable.
    Array { rank: u8 },
    /// A string, optionally one of `choices` (`mixing`, `one_four`). Not an
    /// expression variable.
    Text {
        choices: Option<Vec<Cow<'static, str>>>,
    },
}

/// How a pair style's parameter combines from the two self rows when no
/// cross row gives the pair its own value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamCombination {
    /// It does not: an unlike pair needs a cross row stating it (LAMMPS
    /// `buck`, `morse`). Every bonded parameter.
    None,
    /// `½(pᵢ + pⱼ)`.
    Arithmetic,
    /// `√(pᵢ pⱼ)`.
    Geometric,
    /// The well depth of the joint (ε, σ) rule the style's `mixing` names
    /// ([`CombiningRule`]; absent:
    /// the style's default), combined with the length parameter `sigma`.
    LjEpsilon { sigma: Cow<'static, str> },
    /// The length of that rule, combined with the depth `epsilon`.
    LjSigma { epsilon: Cow<'static, str> },
}

/// One parameter of a style.
#[derive(Clone, Debug, PartialEq)]
pub struct ParamSpec {
    /// `^[A-Za-z_][A-Za-z0-9_]*$`, not a reserved name.
    pub name: Cow<'static, str>,
    pub dim: ParamDimension,
    pub kind: ParamKind,
    /// The value a row (or the style) that lacks the parameter takes; `None`
    /// makes it required.
    pub default: Option<ParamValue>,
    /// Pair styles only.
    pub mix: ParamCombination,
    /// A numbered family `<name>1 … <name>M`, contiguous from 1, one `M`
    /// per table shared by every indexed parameter of the style.
    pub indexed: bool,
}

impl ParamSpec {
    /// A required scalar parameter that does not mix.
    pub fn new(name: impl Into<Cow<'static, str>>, dim: ParamDimension) -> Self {
        Self {
            name: name.into(),
            dim,
            kind: ParamKind::Scalar,
            default: None,
            mix: ParamCombination::None,
            indexed: false,
        }
    }

    /// A text parameter, one of `choices` when there are any.
    pub fn text(name: impl Into<Cow<'static, str>>, choices: &[&'static str]) -> Self {
        let choices = (!choices.is_empty()).then(|| choices.iter().map(|&c| c.into()).collect());
        Self::new(name, ParamDimension::NONE).kind(ParamKind::Text { choices })
    }

    pub fn kind(mut self, kind: ParamKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn default_value(mut self, value: ParamValue) -> Self {
        self.default = Some(value);
        self
    }

    pub fn default_num(self, value: F) -> Self {
        self.default_value(ParamValue::Num(value))
    }

    pub fn mix(mut self, mix: ParamCombination) -> Self {
        self.mix = mix;
        self
    }

    pub fn indexed(mut self) -> Self {
        self.indexed = true;
        self
    }
}

/// A set of registration sample points (`ff-ir-02-protocol` §5): the
/// derivative and agreement checks evaluate the style at 16 seeded points
/// with these parameter values and the coordinate drawn from `q` (for a
/// compound category, `q` bounds the step lengths of the seeded atoms).
#[derive(Clone, Debug, PartialEq)]
pub struct ConformanceSample {
    pub params: Vec<(Cow<'static, str>, ParamValue)>,
    pub q: (F, F),
}

/// One style of the force-field IR: `(category, name)`, its parameters and
/// where they come from.
///
/// The kernel that prices it is registered beside it
/// ([`Registry::register_style`](crate::ff::style_registry::Registry::register_style)).
///
/// The parameter order is the force-field IR's: where LAMMPS has a style of
/// the name, it is that style's `*_coeff` order and the meanings are
/// LAMMPS's (the IR adopts the LAMMPS standard). [`builtin_styles`] states it
/// for every style molrs registers, so an engine codec can write a style it
/// has no arm for from the spec alone.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleSpec {
    pub category: Cow<'static, str>,
    pub name: Cow<'static, str>,
    /// Per-type parameters, in the IR's order (LAMMPS's `*_coeff` order
    /// where LAMMPS has the style).
    pub params: Vec<ParamSpec>,
    /// Style-level parameters. `cutoff` (`L`), `mixing` and `special` (text)
    /// keep their reserved meanings.
    pub style_params: Vec<ParamSpec>,
    pub source: ParamSource,
    /// Which special-bonds weights scale a pair style; `None` on a pair style
    /// means molrec's default, `lj` ([`SpecialClass::Vdw`]).
    pub special: Option<SpecialClass>,
    /// What a pair style is to the nonbonded treatment; `None` follows its
    /// [`special`](Self::special) class ([`pair_role`](Self::pair_role)).
    pub role: Option<PairRole>,
    /// The energy as a Lepton expression (molrec `docs/spec/forcefield.md`,
    /// Expressions), kept byte for byte.
    pub expression: Option<String>,
    /// Whether the kernel's force is the gradient of its energy. `false`
    /// only for `pair coul/charmm` (LAMMPS's switched force): an expression
    /// beside it agrees on the energy alone.
    pub force_is_gradient: bool,
    /// The bare spelling of the indexed parameters (`k`, `periodicity`,
    /// `phase`) is accepted as term 1 (`dihedral periodic`).
    pub unindexed_one_term: bool,
    /// How LAMMPS writes and reads the style ([`LammpsForm`]); `None`
    /// refuses it by name.
    pub lammps: LammpsForm,
    /// Points the conformance checks run on at registration; without any
    /// they run once per process at the style's first compile.
    pub samples: Vec<ConformanceSample>,
}

impl StyleSpec {
    /// A table-driven style with no parameters declared yet.
    pub fn new(category: impl Into<Cow<'static, str>>, name: impl Into<Cow<'static, str>>) -> Self {
        Self {
            category: category.into(),
            name: name.into(),
            params: Vec::new(),
            style_params: Vec::new(),
            source: ParamSource::TypeRows,
            special: None,
            role: None,
            expression: None,
            force_is_gradient: true,
            unindexed_one_term: false,
            lammps: LammpsForm::None,
            samples: Vec::new(),
        }
    }

    pub fn params(mut self, params: Vec<ParamSpec>) -> Self {
        self.params = params;
        self
    }

    pub fn style_params(mut self, params: Vec<ParamSpec>) -> Self {
        self.style_params = params;
        self
    }

    pub fn source(mut self, source: ParamSource) -> Self {
        self.source = source;
        self
    }

    pub fn special(mut self, special: SpecialClass) -> Self {
        self.special = Some(special);
        self
    }

    pub fn role(mut self, role: PairRole) -> Self {
        self.role = Some(role);
        self
    }

    pub fn expression(mut self, expression: impl Into<String>) -> Self {
        self.expression = Some(expression.into());
        self
    }

    pub fn lammps(mut self, form: LammpsForm) -> Self {
        self.lammps = form;
        self
    }

    pub fn sample(mut self, sample: ConformanceSample) -> Self {
        self.samples.push(sample);
        self
    }

    /// The per-type parameter `name`, if declared.
    pub fn param(&self, name: &str) -> Option<&ParamSpec> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The style parameter `name`, if declared.
    pub fn style_param(&self, name: &str) -> Option<&ParamSpec> {
        self.style_params.iter().find(|p| p.name == name)
    }

    /// The declaration a parameter key of this style names, and whether it
    /// is a style parameter: a declared parameter, a member `<name><m>` of an
    /// indexed family, or — on a `pair` style — `q1`, `q2`, `<name>1`,
    /// `<name>2`, the per-atom values (declaration `None`). `None` for a key
    /// the style does not declare.
    pub fn declaration(&self, key: &str, pair: bool) -> Option<(Option<&ParamSpec>, bool)> {
        if let Some(p) = self.param(key) {
            return Some((Some(p), false));
        }
        if let Some(p) = self.style_param(key) {
            return Some((Some(p), true));
        }
        let base = key.trim_end_matches(|c: char| c.is_ascii_digit());
        if base.len() < key.len()
            && let Some(p) = self.param(base).filter(|p| p.indexed)
        {
            return Some((Some(p), false));
        }
        let per_atom = matches!(key, "q1" | "q2")
            || key
                .strip_suffix(['1', '2'])
                .is_some_and(|b| self.param(b).is_some());
        (pair && per_atom).then_some((None, false))
    }

    /// The special-bonds weights that scale this (pair) style: its own, else
    /// molrec's default `lj`.
    pub fn special_class(&self) -> SpecialClass {
        self.special.unwrap_or(SpecialClass::Vdw)
    }

    /// What this (pair) style is to the nonbonded treatment: its declared
    /// role, else a truncated Coulomb for a style scaled by the electrostatic
    /// weights and dispersion for any other.
    pub fn pair_role(&self) -> PairRole {
        self.role.unwrap_or(match self.special_class() {
            SpecialClass::Coulomb => PairRole::Coulomb(CoulombRange::Cut),
            SpecialClass::Vdw => PairRole::Dispersion,
        })
    }

    /// The style params and type rows a kernel is built from: every value
    /// present checked against its declaration, every declared default
    /// filled in where the style (or a row) lacks the parameter.
    ///
    /// The one place a [`ParamSpec::default`] takes effect.
    /// [`PotentialCompiler`](crate::ff::compile::PotentialCompiler) gathers
    /// through it before any kernel — a Tier-3 constructor, a form kernel
    /// or an expression — sees a parameter, so no kernel states a default of
    /// its own and every tier prices an absent parameter alike. A row is
    /// filled whatever it is: a pair style's cross row lacking a parameter
    /// takes the default too, as an optional trailing `pair_coeff i j`
    /// argument takes LAMMPS's. An indexed family's default fills each of
    /// the row's terms.
    ///
    /// A value of the wrong kind is [`IrError::BadValue`]: text or an array
    /// where the spec declares a number, an array of another rank, text
    /// outside its declared choices.
    pub fn gather(
        &self,
        style: &Params,
        rows: &[(&str, &Params)],
    ) -> Result<(Params, Vec<(String, Params)>), IrError> {
        let mut gathered = style.clone();
        self.fill("", &self.style_params, &mut gathered)?;
        let rows = rows
            .iter()
            .map(|&(label, row)| {
                let mut row = row.clone();
                self.fill(label, &self.params, &mut row)?;
                Ok((label.to_owned(), row))
            })
            .collect::<Result<_, IrError>>()?;
        Ok((gathered, rows))
    }

    /// The style params a kernel of this style reads: every declared style
    /// parameter, its default filled where `style` lacks it, and nothing
    /// else. A key the spec does not declare is dropped.
    ///
    /// Two force fields that state a style differently but alike — one
    /// spells out a default (`coul/cut`'s `delta = 0`), the other leaves it
    /// out — give the same params here.
    ///
    /// # Errors
    ///
    /// A value of the wrong kind, as [`gather`](Self::gather).
    pub fn priced_style_params(&self, style: &Params) -> Result<Params, IrError> {
        self.priced("", &self.style_params, style)
    }

    /// The style params of `style` that are not at their declared default:
    /// what a writer must find a place for. A parameter stated at its
    /// default (`coul/cut`'s `delta = 0`, `lj/cut`'s `mixing =
    /// "arithmetic"`) prices as its absence does — [`gather`](Self::gather)
    /// fills that same value — so a format with no place for it loses
    /// nothing by leaving it out; [`priced_style_params`] gives the two the
    /// same params for the same reason. A key the spec does not declare, and
    /// any array, is kept: it is the writer's to place or refuse.
    ///
    /// [`priced_style_params`]: Self::priced_style_params
    pub fn style_params_off_default(&self, style: &Params) -> Params {
        let mut defaults = Params::new();
        self.fill("", &self.style_params, &mut defaults)
            .expect("an empty params holds no value of the wrong kind");
        style
            .iter()
            .filter(|(key, value)| defaults.get(key) != Some(*value))
            .map(|(key, value)| (key, value.clone()))
            .collect()
    }

    /// The first numeric parameter of `row` this spec does not declare (a
    /// member of an indexed family is declared), passing over the keys that
    /// price nothing (annotations and estimate provenance): what a writer of this
    /// style's rows would drop, and so refuses.
    pub fn undeclared_param<'a>(&self, row: &'a Params) -> Option<&'a str> {
        row.iter_parameters()
            .map(|(key, _)| key)
            .find(|key| !declares(&self.params, key))
    }

    /// The row params a kernel of this style reads off `row`: every declared
    /// per-type parameter (each member of an indexed family) with its
    /// default filled, and nothing else. What the spec does not declare —
    /// an annotation (`element`, `desc`), an estimate's provenance, a
    /// reader's bookkeeping (`id`) — prices nothing and is dropped.
    ///
    /// # Errors
    ///
    /// A value of the wrong kind, as [`gather`](Self::gather).
    pub fn priced_row(&self, label: &str, row: &Params) -> Result<Params, IrError> {
        self.priced(label, &self.params, row)
    }

    /// `params` with the defaults of `decls` filled, restricted to the keys
    /// `decls` declare.
    fn priced(&self, label: &str, decls: &[ParamSpec], params: &Params) -> Result<Params, IrError> {
        let mut filled = params.clone();
        self.fill(label, decls, &mut filled)?;
        let declared = |key: &str| declares(decls, key);
        Ok(filled
            .iter()
            .filter(|(key, _)| declared(key))
            .map(|(key, value)| (key, value.clone()))
            .collect())
    }

    /// A row that states nothing, with every per-type default: what a
    /// per-instance term without a type row reads.
    pub fn default_row(&self) -> Params {
        let mut row = Params::new();
        self.fill("", &self.params, &mut row)
            .expect("an empty row holds no value of the wrong kind");
        row
    }

    /// Check `row`'s values of `decls` and fill their defaults: a plain
    /// parameter's under its name, an indexed family's under every member
    /// `<name><m>` the row's terms reach (`m` up to the longest family the
    /// row states), and under the bare name when the row spells term 1
    /// bare ([`unindexed_one_term`](Self::unindexed_one_term)).
    fn fill(&self, label: &str, decls: &[ParamSpec], row: &mut Params) -> Result<(), IrError> {
        let indexed = || decls.iter().filter(|d| d.indexed);
        let terms = indexed()
            .map(|d| {
                (1..)
                    .take_while(|m| {
                        row.get(&format!("{}{m}", d.name))
                            .and_then(|v| v.as_num())
                            .is_some()
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        let bare = self.unindexed_one_term
            && indexed().any(|d| row.get(&d.name).and_then(|v| v.as_num()).is_some());
        for decl in decls {
            let keys: Vec<String> = if decl.indexed {
                for key in family(row, &decl.name) {
                    self.check(label, decl, &key, row)?;
                }
                (1..=terms)
                    .map(|m| format!("{}{m}", decl.name))
                    .chain(bare.then(|| decl.name.to_string()))
                    .collect()
            } else {
                vec![decl.name.to_string()]
            };
            for key in keys {
                let present = self.check(label, decl, &key, row)?;
                if let (Some(default), false) = (&decl.default, present) {
                    row.set(&key, default.clone());
                }
            }
        }
        Ok(())
    }

    /// Whether `row` states `key` (of the declaration `decl`), refusing a
    /// value of another kind.
    fn check(
        &self,
        label: &str,
        decl: &ParamSpec,
        key: &str,
        row: &Params,
    ) -> Result<bool, IrError> {
        let bad = |reason: String| IrError::BadValue {
            style: self.name.to_string(),
            type_: label.to_owned(),
            param: key.to_owned(),
            reason,
        };
        let (num, text, array) = (
            row.get(key).and_then(|v| v.as_num()),
            row.get(key).and_then(|v| v.as_text()),
            row.get(key).and_then(|v| v.as_array()),
        );
        let found = || match (num, text, array) {
            (_, Some(t), _) => format!("is the text {t:?}"),
            (_, _, Some(a)) => format!("is an array of rank {}", a.ndim()),
            _ => "is a number".to_owned(),
        };
        match &decl.kind {
            ParamKind::Scalar if text.is_some() || array.is_some() => {
                Err(bad(format!("{}; the spec declares a number", found())))
            }
            ParamKind::Scalar => Ok(num.is_some()),
            ParamKind::Text { .. } if num.is_some() || array.is_some() => {
                Err(bad(format!("{}; the spec declares text", found())))
            }
            ParamKind::Text {
                choices: Some(choices),
            } => match text {
                Some(t) if !choices.iter().any(|c| c == t) => Err(bad(format!(
                    "is {t:?}, not one of {}",
                    choices
                        .iter()
                        .map(|c| format!("{c:?}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                ))),
                t => Ok(t.is_some()),
            },
            ParamKind::Text { choices: None } => Ok(text.is_some()),
            ParamKind::Array { rank } => match array {
                Some(a) if a.ndim() != *rank as usize => Err(bad(format!(
                    "has rank {}; the spec declares rank {rank}",
                    a.ndim()
                ))),
                Some(_) => Ok(true),
                None if num.is_some() || text.is_some() => Err(bad(format!(
                    "{}; the spec declares an array of rank {rank}",
                    found()
                ))),
                None => Ok(false),
            },
        }
    }
}

/// Whether `key` is a member `<base><m>` of the indexed family `base`, or
/// `base` itself.
/// Whether `decls` declare `key`: a plain parameter by its name, an indexed
/// family by any member `<name><m>` (or the bare name).
fn declares(decls: &[ParamSpec], key: &str) -> bool {
    decls.iter().any(|d| {
        if d.indexed {
            is_member(key, &d.name)
        } else {
            key == d.name
        }
    })
}

fn is_member(key: &str, base: &str) -> bool {
    key.strip_prefix(base)
        .is_some_and(|m| m.is_empty() || m.bytes().all(|b| b.is_ascii_digit()))
}

/// Every key of `row` (numeric, text or array) that is a member
/// `<base><m>` of the indexed family `base`, or `base` itself.
fn family(row: &Params, base: &str) -> Vec<String> {
    row.iter()
        .map(|(key, _)| key)
        .filter(|key| is_member(key, base))
        .map(str::to_owned)
        .collect()
}

/// A scalar parameter of dimension `dim` (a [`ParamDimension`] spelling); the built-in
/// table is written with it, and a test parses every spelling.
fn p(name: &'static str, dim: &str) -> ParamSpec {
    ParamSpec::new(
        name,
        dim.parse()
            .unwrap_or_else(|e| panic!("built-in param {name}: {e}")),
    )
}

fn ps(names: &[&'static str], dim: &str) -> Vec<ParamSpec> {
    names.iter().map(|n| p(n, dim)).collect()
}

fn cutoff() -> ParamSpec {
    p("cutoff", "L")
}

/// `mixing`, absent the IR's undeclared rule ([`CombiningRule::UNDECLARED`],
/// arithmetic). Not every engine's default: LAMMPS mixes `lj/cut`
/// geometrically, which its reader states explicitly.
fn mixing() -> ParamSpec {
    mixing_by(CombiningRule::UNDECLARED)
}

/// `mixing`, absent `rule`.
fn mixing_by(rule: CombiningRule) -> ParamSpec {
    ParamSpec::text("mixing", &["arithmetic", "geometric", "sixthpower"])
        .default_value(ParamValue::Text(rule.name().into()))
}

/// CHARMM's switch `S(r)` from `inner` to `cutoff`, as an expression
/// definition.
fn charmm_switch() -> String {
    "S=select(step(inner-r),1,(cutoff^2-r^2)^2*(cutoff^2+2*r^2-3*inner^2)/(cutoff^2-inner^2)^3)"
        .to_owned()
}

/// `(ε₁₄, σ₁₄)` of one `pair lj/charmm` row whose regular pair is
/// `(epsilon, sigma)`: the row's `epsilon14` / `sigma14`, each absent one the
/// regular value (CHARMM's and LAMMPS's rule — a type without 1-4 parameters
/// is priced at its own). The one place the rule is written; the kernel and
/// the LAMMPS codec both read it.
pub(crate) fn lj_charmm_one_four(row: &Params, epsilon: f64, sigma: f64) -> (f64, f64) {
    let own = |key: &str, regular: f64| row.get(key).and_then(|v| v.as_num()).unwrap_or(regular);
    (own("epsilon14", epsilon), own("sigma14", sigma))
}

/// The built-in spec of `(category, name)` ([`builtin_styles`], built once).
pub(crate) fn builtin_style(category: &str, name: &str) -> Option<&'static StyleSpec> {
    static SPECS: std::sync::LazyLock<Vec<StyleSpec>> = std::sync::LazyLock::new(builtin_styles);
    SPECS
        .iter()
        .find(|s| s.category == category && s.name == name)
}

/// The spec of every style molrs registers: each built-in kernel (the
/// crate-private table `ff::style_registry::BuiltinKernels`), `dihedral rb`
/// (expression only), and the styles of the categories that price no energy.
///
/// Names, order and dimensions are the force-field IR's
/// (`docs/developer/forcefield-ir.md`, Style reference; molrec
/// `docs/spec/forcefield.md`, Style registry); the expressions are
/// Appendix A of `ff-ir-02-protocol`, byte for byte (the generated ones,
/// `dihedral periodic` and `nharmonic`, are left to the expression engine).
/// The per-instance styles (MMFF, UFF, the per-atom-charge Coulomb styles)
/// list the Frame columns their forms read.
pub fn builtin_styles() -> Vec<StyleSpec> {
    use crate::ff::ir::engine_codec::lammps::{self as lc, custom};
    use ParamSource::PerInstance;
    let positional = LammpsForm::positional;
    use SpecialClass::{Coulomb, Vdw};
    let s = StyleSpec::new;
    let eps = |sigma: &'static str| {
        p("epsilon", "E").mix(ParamCombination::LjEpsilon {
            sigma: sigma.into(),
        })
    };
    let sig = |epsilon: &'static str| {
        p("sigma", "L").mix(ParamCombination::LjSigma {
            epsilon: epsilon.into(),
        })
    };
    let coulomb = || p("coulomb", "E*L/Q^2");
    // Declared, never defaulted: a force field states its dielectric as it
    // states its Coulomb constant.
    let dielectric = || p("dielectric", "1");
    let zero = |name: &'static str, dim: &str| p(name, dim).default_num(0.0);
    let mut periodic = s("dihedral", "periodic")
        .params(vec![
            p("k", "E").indexed(),
            p("periodicity", "1").indexed(),
            zero("phase", "A").indexed(),
        ])
        .lammps(custom(&lc::FOURIER));
    periodic.unindexed_one_term = true;
    let mut coul_charmm = s("pair", "coul/charmm")
        .style_params(vec![coulomb(), dielectric(), p("inner", "L"), cutoff()])
        .source(PerInstance)
        .special(Coulomb)
        .expression(format!(
            "coulomb*q1*q2/(dielectric*r)*S; {}",
            charmm_switch()
        ))
        .lammps(custom(&lc::COUL_CHARMM));
    coul_charmm.force_is_gradient = false;
    vec![
        // ---- atom, and the categories that price no energy
        s("atom", "full").params(vec![p("mass", "M"), p("charge", "Q")]),
        s("constraint", "fixed").params(vec![p("r0", "L")]),
        s("virtual_site", "average2").params(ps(&["w1", "w2"], "1")),
        s("virtual_site", "average3").params(ps(&["w1", "w2", "w3"], "1")),
        s("virtual_site", "outofplane3").params(vec![
            p("w12", "1"),
            p("w13", "1"),
            p("wcross", "1/L"),
        ]),
        // ---- drude: a spec, priced by its expression alone
        s("drude", "harmonic")
            .params(vec![p("k", "E/L^2"), p("alpha", "L^3"), p("thole", "1")])
            .expression("k*r^2"),
        // ---- bond
        s("bond", "harmonic")
            .params(vec![p("k", "E/L^2"), p("r0", "L")])
            .expression("k*(r-r0)^2")
            .lammps(positional()),
        s("bond", "morse")
            .params(vec![p("d0", "E"), p("alpha", "1/L"), p("r0", "L")])
            .expression("d0*(1-exp(-alpha*(r-r0)))^2")
            .lammps(positional()),
        s("bond", "class2")
            .params(vec![
                p("r0", "L"),
                p("k2", "E/L^2"),
                p("k3", "E/L^3"),
                p("k4", "E/L^4"),
            ])
            .expression("k2*d^2+k3*d^3+k4*d^4; d=r-r0")
            .lammps(positional()),
        s("bond", "mmff_bond")
            .params(vec![p("kb", "E/L^2"), p("r0", "L")])
            .source(PerInstance),
        s("bond", "uff_bond")
            .params(vec![p("kb", "E/L^2"), p("r0", "L")])
            .source(PerInstance),
        // ---- angle
        s("angle", "harmonic")
            .params(vec![p("k", "E/A^2"), p("theta0", "A")])
            .expression("k*(theta-theta0*(pi/180))^2; pi=3.141592653589793")
            .lammps(positional()),
        s("angle", "charmm")
            .params(vec![
                p("k", "E/A^2"),
                p("theta0", "A"),
                p("k_ub", "E/L^2"),
                p("r_ub", "L"),
            ])
            .expression(
                "k*(theta-theta0*(pi/180))^2+k_ub*(distance(p1,p3)-r_ub)^2; pi=3.141592653589793",
            )
            .lammps(positional()),
        s("angle", "class2")
            .params(vec![
                p("theta0", "A"),
                p("k2", "E/A^2"),
                p("k3", "E/A^3"),
                p("k4", "E/A^4"),
            ])
            .expression("k2*d^2+k3*d^3+k4*d^4; d=theta-theta0*(pi/180); pi=3.141592653589793")
            .lammps(custom(&lc::ANGLE_CLASS2)),
        s("angle", "mmff_angle")
            .params(vec![p("ka", "E/A^2"), p("theta0", "A"), p("linear", "1")])
            .source(PerInstance),
        s("angle", "mmff_stbn")
            .params(vec![
                p("kba_ijk", "E/L/A"),
                p("kba_kji", "E/L/A"),
                p("r0_ij", "L"),
                p("r0_kj", "L"),
                p("theta0", "A"),
                p("linear", "1"),
            ])
            .source(PerInstance),
        s("angle", "uff_angle")
            .params(vec![
                p("ka", "E"),
                p("order", "1"),
                p("c0", "1"),
                p("c1", "1"),
                p("c2", "1"),
            ])
            .source(PerInstance),
        // ---- dihedral
        periodic,
        s("dihedral", "charmm")
            .params(vec![
                p("k", "E"),
                p("periodicity", "1"),
                zero("phase", "A"),
                zero("w", "1"),
            ])
            .expression("k*(1+cos(periodicity*phi-phase*(pi/180))); pi=3.141592653589793")
            .lammps(custom(&lc::DIHEDRAL_CHARMM)),
        s("dihedral", "opls")
            .params(["k1", "k2", "k3", "k4"].map(|k| zero(k, "E")).to_vec())
            .expression(
                "0.5*(k1*(1+cos(phi))+k2*(1-cos(2*phi))+k3*(1+cos(3*phi))+k4*(1-cos(4*phi)))",
            )
            .lammps(positional()),
        s("dihedral", "multi/harmonic")
            .params(["a1", "a2", "a3", "a4", "a5"].map(|a| zero(a, "E")).to_vec())
            .expression("a1+a2*c+a3*c^2+a4*c^3+a5*c^4; c=cos(phi)")
            .lammps(positional()),
        s("dihedral", "nharmonic")
            .params(vec![p("a", "E").indexed()])
            .lammps(custom(&lc::NHARMONIC)),
        s("dihedral", "harmonic")
            .params(vec![p("k", "E"), p("sign", "1"), p("periodicity", "1")])
            .expression("k*(1+sign*cos(periodicity*phi))")
            .lammps(custom(&lc::SIGNED_COSINE)),
        s("dihedral", "class2")
            .params(vec![
                zero("k1", "E"),
                zero("phi1", "A"),
                zero("k2", "E"),
                zero("phi2", "A"),
                zero("k3", "E"),
                zero("phi3", "A"),
            ])
            .expression(
                "k1*(1-cos(phi-phi1*(pi/180)))+k2*(1-cos(2*phi-phi2*(pi/180)))+k3*(1-cos(3*phi-phi3*(pi/180))); pi=3.141592653589793",
            )
            .lammps(custom(&lc::DIHEDRAL_CLASS2)),
        // molrec's registry style, registered by its expression alone: the
        // first built-in that is pure protocol.
        s("dihedral", "rb")
            .params(ps(&["c0", "c1", "c2", "c3", "c4", "c5"], "E"))
            .expression("c0+c1*c+c2*c^2+c3*c^3+c4*c^4+c5*c^5; c=-cos(phi)"),
        s("dihedral", "mmff_torsion")
            .params(ps(&["v1", "v2", "v3"], "E"))
            .source(PerInstance),
        s("dihedral", "uff_torsion")
            .params(vec![p("V", "E"), p("order", "1"), p("cosTerm", "1")])
            .source(PerInstance),
        // ---- improper
        s("improper", "harmonic")
            .params(vec![p("k", "E/A^2"), zero("chi0", "A")])
            .expression("k*(chi-chi0*(pi/180))^2; pi=3.141592653589793")
            .lammps(positional()),
        s("improper", "cvff")
            .params(vec![p("k", "E"), p("sign", "1"), p("periodicity", "1")])
            .expression("k*(1+sign*cos(periodicity*phi))")
            .lammps(custom(&lc::SIGNED_COSINE)),
        s("improper", "periodic")
            .params(vec![p("k", "E"), p("periodicity", "1"), zero("phase", "A")])
            .expression("k*(1+cos(periodicity*phi-phase*(pi/180))); pi=3.141592653589793")
            .lammps(custom(&lc::PERIODIC_AS_CVFF)),
        s("improper", "mmff_oop")
            .params(vec![p("koop", "E/A^2")])
            .source(PerInstance),
        s("improper", "uff_inversion")
            .params(vec![p("K", "E"), p("c0", "1"), p("c1", "1"), p("c2", "1")])
            .source(PerInstance),
        // ---- cmap
        s("cmap", "charmm")
            .params(vec![p(super::CMAP_GRID, "E").kind(ParamKind::Array { rank: 2 })])
            .lammps(custom(&lc::FIX_CMAP)),
        // ---- pair
        s("pair", "lj/cut")
            .params(vec![eps("sigma"), sig("epsilon")])
            .style_params(vec![
                cutoff(),
                mixing(),
                p("n", "1").default_num(12.0),
                p("m", "1").default_num(6.0),
                p("shift", "1").default_num(0.0),
            ])
            .special(Vdw)
            .expression(
                "C*epsilon*((sigma/r)^n-(sigma/r)^m-select(shift,(sigma/cutoff)^n-(sigma/cutoff)^m,0)); \
                 C=n/(n-m)*(n/m)^(m/(n-m))",
            )
            .lammps(custom(&lc::LJ_CUT)),
        s("pair", "lj/class2")
            .params(vec![eps("sigma"), sig("epsilon")])
            // LAMMPS mixes `lj/class2` sixthpower unless told otherwise.
            .style_params(vec![cutoff(), mixing_by(CombiningRule::SixthPower)])
            .special(Vdw)
            .expression("epsilon*(2*(sigma/r)^9-3*(sigma/r)^6)")
            .lammps(custom(&lc::LJ_CLASS2)),
        s("pair", "buck")
            .params(vec![p("a", "E"), p("rho", "L"), p("c", "E*L^6")])
            .style_params(vec![cutoff()])
            .special(Vdw)
            .expression("a*exp(-r/rho)-c/r^6")
            .lammps(positional()),
        s("pair", "morse")
            .params(vec![p("d0", "E"), p("alpha", "1/L"), p("r0", "L")])
            .style_params(vec![cutoff()])
            .special(Vdw)
            .expression("d0*((1-exp(-alpha*(r-r0)))^2-1)")
            .lammps(positional()),
        s("pair", "lj/charmm")
            .params(vec![
                eps("sigma"),
                sig("epsilon"),
                p("epsilon14", "E").mix(ParamCombination::LjEpsilon {
                    sigma: "sigma14".into(),
                }),
                p("sigma14", "L").mix(ParamCombination::LjSigma {
                    epsilon: "epsilon14".into(),
                }),
            ])
            .style_params(vec![
                p("inner", "L"),
                cutoff(),
                mixing(),
                ParamSpec::text("one_four", &["regular", "epsilon14"])
                    .default_value(ParamValue::Text("regular".into())),
            ])
            .special(Vdw)
            .expression(format!(
                "4*epsilon*((sigma/r)^12-(sigma/r)^6)*S; {}",
                charmm_switch()
            ))
            .lammps(custom(&lc::LJ_CHARMM)),
        s("pair", "coul/cut")
            .style_params(vec![
                coulomb(),
                dielectric(),
                p("delta", "L").default_num(0.0),
                cutoff(),
            ])
            .source(PerInstance)
            .special(Coulomb)
            .expression("coulomb*q1*q2/(dielectric*(r+delta))")
            .lammps(custom(&lc::COUL_CUT)),
        coul_charmm,
        s("pair", "coul/long/pme")
            .style_params(vec![
                coulomb(),
                dielectric(),
                cutoff(),
                p("alpha", "1/L"),
                p("order", "1"),
                p("grid_x", "1"),
                p("grid_y", "1"),
                p("grid_z", "1"),
            ])
            .source(PerInstance)
            .special(Coulomb)
            .role(PairRole::Coulomb(CoulombRange::Long))
            .lammps(custom(&lc::COUL_LONG)),
        s("pair", "thole")
            .params(vec![p("charge", "Q"), p("alpha", "L^3"), p("damp", "1")])
            .style_params(vec![coulomb(), dielectric()])
            .special(Coulomb)
            .role(PairRole::Damping),
        s("pair", "coul/tt")
            .params(vec![p("charge", "Q")])
            .style_params(vec![
                coulomb(),
                dielectric(),
                p("b", "1/L").default_num(4.5),
                p("c", "1").default_num(1.0),
                p("order", "1").default_num(4.0),
            ])
            .special(Coulomb)
            .role(PairRole::Damping),
        s("pair", "uff_lj")
            .params(vec![p("x1", "L"), p("D1", "E")])
            .source(PerInstance)
            .special(Vdw),
        s("pair", "mmff_vdw")
            .params(vec![
                p("alpha", "L^3"),
                p("n_eff", "1"),
                p("a_i", "1"),
                p("g_i", "1"),
                // MMFF's own default role, neither donor nor acceptor.
                p("da", "1").default_num(f64::from(crate::ff::params::mmff::DA_NEITHER)),
            ])
            .style_params(vec![
                p("B", "1").default_num(0.2),
                p("Beta", "1").default_num(12.0),
                p("DARAD", "1").default_num(0.8),
                p("DAEPS", "1").default_num(0.5),
            ])
            .special(Vdw),
    ]
}

#[cfg(test)]
mod tests {
    /// Lepton has no named constants: an expression that converts a
    /// degree-valued parameter writes `(pi/180)` and defines `pi` as its
    /// last sub-definition, the shortest decimal of `f64::consts::PI`, so
    /// `x*(pi/180)` is `x.to_radians()` to the bit.
    #[test]
    fn a_degree_conversion_defines_pi_as_std_pi() {
        let pi = format!("; pi={}", std::f64::consts::PI);
        assert_eq!(pi, "; pi=3.141592653589793");
        assert_eq!(std::f64::consts::PI / 180.0, 1f64.to_radians());
        let mut n = 0;
        for spec in super::builtin_styles() {
            let Some(expression) = spec.expression.as_deref() else {
                continue;
            };
            if expression.contains("pi/180") || expression.contains("pi=") {
                n += 1;
                assert!(expression.contains("*(pi/180)"), "{expression}");
                assert!(expression.ends_with(&pi), "{expression}");
                assert_eq!(expression.matches("; pi=").count(), 1, "{expression}");
            }
        }
        assert_eq!(n, 7);
    }
}
