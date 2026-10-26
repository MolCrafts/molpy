//! The [`ForceField`] data model. Force-field files are mapped to and from it
//! by [`crate::io`]; no file format is here.
//!
//! Provides a declarative layer for defining atom types, bond types, pair types,
//! etc. with their parameters. A [`ForceField`] holds [`Style`]s, each of which
//! holds its [`TypeDef`] rows — one row shape for every category — in its
//! [`StyleDefs`]. The forcefield is compiled
//! into computational [`Potential`](super::potential::Potential) objects by
//! [`PotentialCompiler`](super::compile::PotentialCompiler).
//!
//! Its vocabulary — [`Params`], [`SpecialBonds`], the combining rule — is the
//! IR's ([`crate::ff::ir`]); which categories exist is the style registry's
//! ([`crate::ff::style_registry`]).

pub mod param_columns;

use std::sync::Arc;

use smallvec::SmallVec;

use molrs::op::F;

use crate::ff::ir::{Arity, Params, SpecialBonds, builtin_categories, pair_key};
use crate::ff::style_registry::conformance::is_category_name;
use crate::ff::style_registry::{Registry, RegistryRef};

// ---------------------------------------------------------------------------
// Type definitions
// ---------------------------------------------------------------------------

/// One type of a style — the one row shape of every category: its name, the
/// atom types it is defined on, and its parameters.
///
/// `endpoints` are as many atom-type names as the category's arity, in order
/// (`itom`, `jtom`, …): none for an atom type, two for a pair type (a self
/// pair names its one atom type twice), five for a CMAP type (the two
/// consecutive backbone dihedrals `itom-jtom-ktom-ltom` and
/// `jtom-ktom-ltom-mtom`, its correction table the array param `grid`).
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub endpoints: SmallVec<[String; MAX_ARITY]>,
    pub params: Params,
}

/// The type definitions of one style, under its category: the category's
/// name, its [`Arity`] and the [`TypeDef`] rows in definition order.
///
/// The arity is the category's declaration's
/// ([`CategorySpec`](crate::ff::ir::CategorySpec), in the force field's
/// registry or among the IR's
/// [built-in categories](crate::ff::ir::builtin_categories)), or the one
/// stated for a category nothing declares
/// ([`ForceField::def_style_with_arity`]). A pair category
/// ([`Arity::SelfOrPair`]) is the one whose rows are identified by their
/// unordered endpoints (see [`Style`]).
#[derive(Debug, Clone, PartialEq)]
pub struct StyleDefs {
    category: Arc<str>,
    arity: Arity,
    types: Vec<TypeDef>,
}

impl StyleDefs {
    /// No definitions, under `category` of `arity`.
    fn new(category: &str, arity: Arity) -> Self {
        Self {
            category: Arc::from(category),
            arity,
            types: Vec::new(),
        }
    }

    /// No definitions, under the same category as `self`.
    fn empty_like(&self) -> Self {
        Self {
            category: self.category.clone(),
            arity: self.arity,
            types: Vec::new(),
        }
    }

    /// Category string for registry lookups.
    pub fn category(&self) -> &str {
        &self.category
    }

    /// How many endpoint columns a type row of the category carries in a
    /// record (a pair two, a self pair restating its one atom type).
    pub fn arity(&self) -> usize {
        self.arity.endpoints()
    }

    /// The type rows, in definition order.
    pub fn types(&self) -> &[TypeDef] {
        &self.types
    }

    /// Whether the category is a non-bonded pair: a row is found by its
    /// unordered endpoints, and a self pair is defined on one.
    fn is_pair(&self) -> bool {
        self.arity == Arity::SelfOrPair
    }

    /// Collect `(type_name, params)` pairs for kernel construction.
    pub fn collect_type_params(&self) -> Vec<(String, Params)> {
        self.types
            .iter()
            .map(|t| (t.name.clone(), t.params.clone()))
            .collect()
    }

    /// The rows a kernel resolves parameters from: a bonded or atom type by its
    /// name (the label a frame row carries), a pair type by the atom types it
    /// was defined between. A pair kernel meets an atom-type pair, never a
    /// pair-type name, so the key is built from the endpoints in byte order
    /// ([`pair_key`]) — a type named anything, defined in either order, is
    /// found through them. A style holds at most one row per unordered pair
    /// (the conflict rule, see [`Style`]), so the keys are distinct.
    pub fn kernel_type_params(&self) -> Result<Vec<(String, Params)>, String> {
        if !self.is_pair() {
            return Ok(self.collect_type_params());
        }
        self.types
            .iter()
            .map(|t| {
                Ok((
                    pair_key(&t.endpoints[0], &t.endpoints[1])?,
                    t.params.clone(),
                ))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Style
// ---------------------------------------------------------------------------

/// A style groups a named interaction method with its type definitions.
///
/// A style is identified by its `(category, name)` pair. It is defined through
/// [`ForceField::def_style`], and its types through [`Style::def_type`] (name
/// with its endpoints given). The fields are private: a writable name could collide with another style, and a
/// direct push onto the definitions would bypass the arity check and the
/// conflict rule.
///
/// # The conflict rule
///
/// A type is identified by `(category, style, name)`. Defining a name this
/// style already holds with the same endpoints and exactly equal [`Params`] is
/// a no-op; anything else is `Err(DefError::TypeConflict)` and leaves the first
/// definition in place. A pair is also identified by its unordered
/// `{itom, jtom}` (a kernel finds it by its endpoints, never by its name): a
/// pair type restating a pair another one holds, in either order and under
/// any name, is a no-op when the two have
/// [the same parameters](Params::same_parameters) and
/// `Err(DefError::PairConflict)` otherwise, so a pair style holds at most one
/// row per pair. The edits `set_type_param`, `set_param` and `del_type`
/// change an existing definition and are outside the rule;
/// `rename_type` can land on an existing name and therefore carries it.
#[derive(Debug, Clone)]
pub struct Style {
    name: String,
    params: Params,
    defs: StyleDefs,
}

impl Style {
    fn new(defs: StyleDefs, name: &str, params: Params) -> Self {
        Self {
            name: name.to_owned(),
            params,
            defs,
        }
    }

    /// The style name (e.g. `harmonic`, `lj/cut`).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Style-level params, numeric and string (`cutoff`, `mixing`, …).
    pub fn params(&self) -> &Params {
        &self.params
    }

    /// The type definitions this style holds.
    pub fn defs(&self) -> &StyleDefs {
        &self.defs
    }

    /// The category of the style's definitions.
    pub fn category(&self) -> &str {
        self.defs.category()
    }

    /// How many endpoints a type of this style names in a record (see
    /// [`StyleDefs::arity`]).
    pub fn arity(&self) -> usize {
        self.defs.arity()
    }

    /// The atom type named `name`, when this is an atom style.
    pub fn get_atomtype(&self, name: &str) -> Option<&TypeDef> {
        if self.category() != "atom" {
            return None;
        }
        self.defs.types.iter().find(|t| t.name == name)
    }

    /// The bond type defined on `itom`, `jtom` in either order, when this is
    /// a bond style.
    pub fn get_bondtype(&self, itom: &str, jtom: &str) -> Option<&TypeDef> {
        if self.category() != "bond" {
            return None;
        }
        self.defs.types.iter().find(|t| {
            let e = &t.endpoints;
            (e[0] == itom && e[1] == jtom) || (e[0] == jtom && e[1] == itom)
        })
    }

    /// The pair type defined on `{itom, jtom}` in either order (`jtom`
    /// `None`: the self pair of `itom`), when this is a pair style.
    pub fn get_pairtype(&self, itom: &str, jtom: Option<&str>) -> Option<&TypeDef> {
        if !self.defs.is_pair() {
            return None;
        }
        let jtom = jtom.unwrap_or(itom);
        self.defs.types.iter().find(|t| {
            let e = &t.endpoints;
            (e[0] == itom && e[1] == jtom) || (e[0] == jtom && e[1] == itom)
        })
    }

    /// Define a type named `name` on the given `endpoints` (atom-type names).
    ///
    /// The name is an opaque identifier, stored verbatim: it may be built from
    /// the endpoint labels ([`TypeName::join`](molrs::core::TypeName::join)
    /// is the convention) but it is never read back into endpoints — a `-` in
    /// a name is just a character. The endpoint count follows the category:
    /// an atom style takes none; a pair style one (a self pair) or two; bond
    /// two; angle three; dihedral and improper four; cmap five; any other
    /// category its arity. Any other count is `Err(DefError::Arity)` and
    /// never panics. Re-defining a stored name, or
    /// restating a stored pair, follows the conflict rule (see [`Style`]):
    /// identical is a no-op, different is `Err(DefError::TypeConflict)` /
    /// `Err(DefError::PairConflict)`.
    pub fn def_type(
        &mut self,
        name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> Result<&mut Self, DefError> {
        if !self.accepts_endpoint_count(endpoints.len()) {
            return Err(DefError::Arity {
                category: self.category().to_owned(),
                expected: self.endpoint_arity(),
                name: name.to_owned(),
                got: endpoints.len(),
            });
        }
        // An endpoint label — and an atom type's name, the label every other
        // type is defined on — is joined into type names and pair keys by
        // `TypeName::join`, so it may not contain what `join` refuses (`@`).
        // Checked here, where the label enters, not at the kernel that reads
        // it back: a force field that accepts a label it cannot compile is a
        // trap sprung later.
        let labels: &[&str] = if endpoints.is_empty() {
            &[name]
        } else {
            endpoints
        };
        molrs::core::TypeName::join(labels).map_err(DefError::Name)?;
        self.insert_type(name.to_owned(), endpoints, params)?;
        Ok(self)
    }

    fn accepts_endpoint_count(&self, n: usize) -> bool {
        if self.defs.is_pair() {
            n == 1 || n == 2
        } else {
            n == self.arity()
        }
    }

    /// The `def_type` endpoint count this category expects.
    fn endpoint_arity(&self) -> String {
        match self.arity() {
            _ if self.defs.is_pair() => "1 or 2 endpoints".to_owned(),
            0 => "no endpoints".to_owned(),
            n => format!("{n} endpoints"),
        }
    }

    /// The conflict rule, and the one place it lives: whether defining `name`
    /// with `endpoints` and `params` here would be accepted, without defining
    /// it.
    ///
    /// `Ok(false)` when the definition is new (it would be appended);
    /// `Ok(true)` when it restates one already here (a no-op): `name` defined
    /// with the same endpoints and equal params, or — in a pair style — the
    /// same unordered `{itom, jtom}` under another name with
    /// [the same parameters](Params::same_parameters).
    /// `Err(DefError::TypeConflict)` when `name` is defined with anything
    /// else; `Err(DefError::PairConflict)` when another pair type prices
    /// `{itom, jtom}` differently (molrec forcefield, linking rule 3: a pair is
    /// found by its endpoints in either order, so two rows on one pair cannot
    /// both apply). `endpoints` are the stored form: the endpoint count has
    /// been checked against the category, and a one-endpoint pair is a
    /// self-pair.
    pub(crate) fn check_type(
        &self,
        name: &str,
        endpoints: &[&str],
        params: &Params,
    ) -> Result<bool, DefError> {
        self.check_type_by(
            name,
            endpoints,
            params,
            |a, b| a == b,
            Params::same_parameters,
        )
    }

    /// [`Self::check_type`] under the given sameness: `same_type` decides
    /// whether `name`'s stored params are the given ones, `same_pair_params`
    /// whether those of a pair type stated under another name are. The
    /// conflict rule is `==` and [`Params::same_parameters`];
    /// [`ForceField::merge`] compares what the kernel prices ([`Pricing`]).
    fn check_type_by(
        &self,
        name: &str,
        endpoints: &[&str],
        params: &Params,
        same_type: impl Fn(&Params, &Params) -> bool,
        same_pair_params: impl Fn(&Params, &Params) -> bool,
    ) -> Result<bool, DefError> {
        let pair = match endpoints {
            [only] if self.defs.is_pair() => Some((*only, *only)),
            [i, j] if self.defs.is_pair() => Some((*i, *j)),
            _ => None,
        };
        let same_pair =
            |i: &str, j: &str| pair.is_some_and(|(a, b)| (i, j) == (a, b) || (i, j) == (b, a));
        if let Some(stored) = self.defs.types.iter().find(|t| t.name == name) {
            let same_endpoints = match pair {
                Some(_) => same_pair(&stored.endpoints[0], &stored.endpoints[1]),
                None => stored
                    .endpoints
                    .iter()
                    .map(String::as_str)
                    .eq(endpoints.iter().copied()),
            };
            return if same_endpoints && same_type(&stored.params, params) {
                Ok(true)
            } else {
                Err(DefError::TypeConflict {
                    category: self.category().to_owned(),
                    style: self.name.clone(),
                    name: name.to_owned(),
                })
            };
        }
        if pair.is_none() {
            return Ok(false);
        }
        let Some(stated) = self
            .defs
            .types
            .iter()
            .find(|t| same_pair(&t.endpoints[0], &t.endpoints[1]))
        else {
            return Ok(false);
        };
        if same_pair_params(&stated.params, params) {
            Ok(true)
        } else {
            Err(DefError::PairConflict {
                style: self.name.clone(),
                name: name.to_owned(),
                stated: stated.name.clone(),
                itom: stated.endpoints[0].clone(),
                jtom: stated.endpoints[1].clone(),
            })
        }
    }

    /// Every type this style holds as `(name, endpoints, params)`, in
    /// definition order, with endpoints in the form [`Style::def_type`]
    /// accepts (a pair row always has two). Replaying the rows through
    /// `def_type` reproduces the definitions. The one-pass read of a whole
    /// style; [`type_params`](Self::type_params) and
    /// [`type_endpoints`](Self::type_endpoints) answer for one type.
    pub fn type_rows(&self) -> Vec<(&str, Vec<&str>, &Params)> {
        self.defs
            .types
            .iter()
            .map(|t| {
                let e = t.endpoints.iter().map(String::as_str).collect();
                (t.name.as_str(), e, &t.params)
            })
            .collect()
    }

    /// The one insert path: [`Style::check_type`] decides, then a name not
    /// yet defined is appended.
    ///
    /// The endpoint count has been checked against the category by the
    /// caller; a one-endpoint pair is a self-pair, stored as its one atom
    /// type twice. On `Err(DefError::TypeConflict)` the stored definition is
    /// unchanged.
    fn insert_type(&mut self, name: String, e: &[&str], params: Params) -> Result<(), DefError> {
        if self.check_type(&name, e, &params)? {
            return Ok(());
        }
        let endpoints = match e {
            [only] if self.defs.is_pair() => {
                [*only, *only].map(str::to_owned).into_iter().collect()
            }
            _ => e.iter().map(|&end| end.to_owned()).collect(),
        };
        self.defs.types.push(TypeDef {
            name,
            endpoints,
            params,
        });
        Ok(())
    }
}

/// Error from defining a style or a type: [`ForceField::def_style`] and
/// [`Style::def_type`] (and [`Style::rename_type`],
/// the one edit that can land on an existing name), and from
/// [`ForceField::merge`].
#[derive(Debug, Clone, PartialEq)]
pub enum DefError {
    /// The type `name` was given the wrong number of endpoints for
    /// `category`.
    Arity {
        category: String,
        expected: String,
        name: String,
        got: usize,
    },
    /// Unknown style category string: neither declared by the force
    /// field's registry nor one of the IR's built-in categories, nor held
    /// by the force field already.
    UnknownCategory(String),
    /// A style of `category` declared with `got` endpoints per type, where
    /// the category (its registration, or the styles of it the force field
    /// already holds) has `expected`.
    CategoryArity {
        category: String,
        expected: usize,
        got: usize,
    },
    /// No style `name` is defined under `category`.
    UnknownStyle { category: String, name: String },
    /// The `category` style `style` already defines a type `name` with other
    /// endpoints or other params. The first definition is kept.
    TypeConflict {
        category: String,
        style: String,
        name: String,
    },
    /// The pair style `style` already prices the pair `{itom, jtom}` (in
    /// either order) as the type `stated`, and the type `name` restates it
    /// with other parameters. The first definition is kept.
    PairConflict {
        style: String,
        name: String,
        stated: String,
        itom: String,
        jtom: String,
    },
    /// The `category` style `name` is already defined with other style params.
    /// The first definition is kept.
    StyleConflict { category: String, name: String },
    /// [`ForceField::merge`]: both force fields declare units, and they differ.
    UnitsConflict { ours: String, theirs: String },
    /// [`ForceField::merge`]: both force fields declare [`SpecialBonds`], and
    /// they differ.
    SpecialBondsConflict {
        ours: SpecialBonds,
        theirs: SpecialBonds,
    },
    /// A type name could not be built from its endpoint labels (a label
    /// containing `@`, or a malformed qualifier; see
    /// [`TypeName`](molrs::core::TypeName)).
    Name(String),
}

impl std::fmt::Display for DefError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DefError::Arity {
                category,
                expected,
                name,
                got,
            } => write!(
                f,
                "{category} type \"{name}\": expected {expected}, got {got} endpoint(s)"
            ),
            DefError::UnknownCategory(category) => write!(
                f,
                "unknown style category '{category}': no force-field IR registry \
                 declares it (register_category)"
            ),
            DefError::CategoryArity {
                category,
                expected,
                got,
            } => write!(
                f,
                "category '{category}' has {expected} endpoint(s) per type, not {got}"
            ),
            DefError::UnknownStyle { category, name } => {
                write!(f, "no {category} style named '{name}'")
            }
            DefError::TypeConflict {
                category,
                style,
                name,
            } => write!(
                f,
                "{category} style '{style}' already defines type \"{name}\" with \
                 different endpoints or params"
            ),
            DefError::PairConflict {
                style,
                name,
                stated,
                itom,
                jtom,
            } => write!(
                f,
                "pair style '{style}': type \"{name}\" restates the pair {{{itom}, {jtom}}} \
                 that type \"{stated}\" prices, with different params"
            ),
            DefError::StyleConflict { category, name } => write!(
                f,
                "{category} style '{name}' is already defined with different style params"
            ),
            DefError::UnitsConflict { ours, theirs } => write!(
                f,
                "force field declares units '{ours}', the merged one declares '{theirs}'"
            ),
            DefError::SpecialBondsConflict { ours, theirs } => write!(
                f,
                "force field declares special_bonds lj {:?} coul {:?}, the merged one \
                 declares lj {:?} coul {:?}",
                ours.lj, ours.coul, theirs.lj, theirs.coul
            ),
            DefError::Name(reason) => write!(f, "invalid type name: {reason}"),
        }
    }
}

impl std::error::Error for DefError {}

impl DefError {
    /// The force-field IR refusal this is, when it is one: a type with the
    /// wrong number of endpoints, or a style of the wrong arity, is
    /// [`IrError::Arity`](crate::ff::ir::IrError::Arity) (`arity` the count
    /// given); an undeclared category is
    /// [`IrError::UnknownCategory`](crate::ff::ir::IrError::UnknownCategory).
    pub fn ir(&self) -> Option<crate::ff::ir::IrError> {
        use crate::ff::ir::IrError;
        match self {
            DefError::Arity { category, got, .. }
            | DefError::CategoryArity { category, got, .. } => Some(IrError::Arity {
                category: category.clone(),
                arity: *got,
            }),
            DefError::UnknownCategory(category) => Some(IrError::UnknownCategory {
                category: category.clone(),
            }),
            _ => None,
        }
    }
}

/// In-place mutators backing the Python handle-view layer (Style/Type views read
/// through [`collect_type_params`](StyleDefs::collect_type_params) and write
/// through these). Each operates on the type identified by its dash-form name.
impl Style {
    /// The params of the type named `name`, or `None` if no such type.
    pub fn type_params(&self, name: &str) -> Option<&Params> {
        self.defs
            .types
            .iter()
            .find(|t| t.name == name)
            .map(|t| &t.params)
    }

    /// Endpoint atom-type names of the type named `name` (e.g. `["CT","CT"]`),
    /// or `None` if no such type. Atom styles return an empty vec.
    pub fn type_endpoints(&self, name: &str) -> Option<Vec<String>> {
        self.defs
            .types
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.endpoints.to_vec())
    }

    /// Set (or add) a single param on the type named `name` — a number, a
    /// string, or an array (a CMAP type's `grid`), whatever
    /// [`ParamValue`](crate::ff::ir::ParamValue) converts from. Returns
    /// `false` if no such type exists.
    ///
    /// An edit of an existing definition, not a definition: it changes the
    /// stored value in place and is outside the conflict rule (see [`Style`]).
    pub fn set_type_param(
        &mut self,
        name: &str,
        key: &str,
        value: impl Into<crate::ff::ir::ParamValue>,
    ) -> bool {
        let Some(t) = self.defs.types.iter_mut().find(|t| t.name == name) else {
            return false;
        };
        t.params.set(key, value.into());
        true
    }

    /// Set (or add) one style-level param, e.g. a pair style's `cutoff` (a
    /// number) or `mixing` (a string).
    ///
    /// An edit of an existing definition, not a definition (see [`Style`]):
    /// the way a caller declares a run setting — the cutoff no reader or
    /// typifier invents — on a style it already has.
    pub fn set_param(&mut self, key: &str, value: impl Into<crate::ff::ir::ParamValue>) {
        self.params.set(key, value);
    }

    /// Rename the type named `old` to `new`. Returns `Ok(false)` if no type
    /// is named `old`.
    ///
    /// The one edit that can land on an existing name, so it carries the
    /// conflict rule (see [`Style`]): if `new` is already defined with the same
    /// endpoints and equal params, the two are one definition and `old` is
    /// dropped (`Ok(true)`); if it is defined with anything else, the rename is
    /// `Err(DefError::TypeConflict)` and nothing changes.
    pub fn rename_type(&mut self, old: &str, new: &str) -> Result<bool, DefError> {
        let types = &mut self.defs.types;
        let Some(from) = types.iter().position(|t| t.name == old) else {
            return Ok(false);
        };
        if old == new {
            return Ok(true);
        }
        let mut renamed = types[from].clone();
        renamed.name = new.to_owned();
        match types.iter().find(|t| t.name == new) {
            Some(existing) if *existing == renamed => {
                types.remove(from);
            }
            Some(_) => {
                return Err(DefError::TypeConflict {
                    category: self.defs.category().to_owned(),
                    style: self.name.clone(),
                    name: new.to_owned(),
                });
            }
            None => types[from] = renamed,
        }
        Ok(true)
    }

    /// Remove the type named `name`. Returns the count removed (`0` or `1`:
    /// a name is defined at most once per style).
    ///
    /// An edit of an existing definition, not a definition: it is outside the
    /// conflict rule (see [`Style`]).
    pub fn del_type(&mut self, name: &str) -> usize {
        let types = &mut self.defs.types;
        let before = types.len();
        types.retain(|t| t.name != name);
        before - types.len()
    }
}

// ---------------------------------------------------------------------------
// ForceField
// ---------------------------------------------------------------------------

/// The nonbonded treatment a force field's pair styles state
/// ([`ForceField::nonbonded`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nonbonded {
    /// How far its electrostatics reach; `None` when it has no Coulomb style.
    pub coulomb: Option<crate::ff::ir::CoulombRange>,
    /// The Coulomb styles' largest `cutoff`; `None` when none states one.
    pub coulomb_cutoff: Option<F>,
    /// The dispersion styles' largest `cutoff`; `None` when none states one.
    pub vdw_cutoff: Option<F>,
}

/// Top-level forcefield container holding styles and their type definitions.
///
/// A force field is built through exactly two fallible primitives:
/// [`ForceField::def_style`] and [`Style::def_type`] (a type's name and its
/// endpoints are both given).
///
/// # Example
///
/// ```
/// use molrs::ff::forcefield::ForceField;
/// use molrs::ff::ir::Params;
///
/// let mut ff = ForceField::new("example");
/// ff.def_style("bond", "harmonic", Params::new())
///     .unwrap()
///     .def_type("A-B", &["A", "B"], Params::from_iter([("k", 300.0), ("r0", 1.5)]))
///     .unwrap();
/// ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
///     .unwrap()
///     .def_type("A", &["A"], Params::from_iter([("epsilon", 0.5), ("sigma", 1.0)]))
///     .unwrap();
///
/// // Compile into Potentials against a typed topology
/// // let potentials = PotentialCompiler::new(&ff).compile(&frame).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ForceField {
    pub name: String,
    styles: Vec<Style>,
    /// `None` until declared; [`Self::units`] reads the default then.
    units: Option<String>,
    /// `None` until declared; [`Self::special_bonds`] reads the default then.
    special_bonds: Option<SpecialBonds>,
    /// The style registry this force field's categories, kernels and codecs
    /// are looked up in: the process-wide one unless
    /// [`with_registry`](Self::with_registry) gave it its own.
    pub(crate) registry: RegistryRef,
}

/// The most endpoints a type names (`itom` … `mtom`).
const MAX_ARITY: usize = 5;

/// The relative difference within which [`ForceField::merge`] takes two
/// numbers for one value.
///
/// Two sources of one force field store its numbers at their own precision:
/// an AMBER prmtop keeps eight significant digits (`6.66666667` for
/// `gaff.dat`'s `20.000 / 3`), and tleap converts θ₀ to radians with
/// π = 3.141594, so every angle it writes reads back 4.3·10⁻⁷ above the
/// degrees `gaff.dat` states (`109.8000471` for `c3-c3-hc`'s `109.80`). A
/// value that differs by more than this is another parameter: no force field
/// states one to seven significant digits.
pub const MERGE_REL_TOL: f64 = 1e-6;

/// What [`ForceField::merge`] compares two definitions of one style on:
/// the parameters its kernel prices, defaults filled
/// ([`StyleSpec::priced_style_params`](crate::ff::ir::StyleSpec::priced_style_params),
/// [`StyleSpec::priced_row`](crate::ff::ir::StyleSpec::priced_row)) — or,
/// for a style no registry declares, every parameter column — within
/// [`MERGE_REL_TOL`].
struct Pricing {
    spec: Option<crate::ff::ir::StyleSpec>,
}

impl Pricing {
    /// The pricing of the style `(category, name)`, from `registry` (the
    /// merged force field's own, or the process-wide one).
    fn of(registry: &RegistryRef, category: &str, name: &str) -> Self {
        let spec = registry.with(|r| r.style(category, name).map(|(spec, _)| spec.clone()));
        Self { spec }
    }

    /// Whether two style-param sets of this style price alike.
    fn styles_alike(&self, a: &Params, b: &Params) -> bool {
        self.alike(a, b, |spec, p| spec.priced_style_params(p).ok())
    }

    /// Whether two params of the type `name` price alike.
    fn rows_alike(&self, name: &str, a: &Params, b: &Params) -> bool {
        self.alike(a, b, |spec, p| spec.priced_row(name, p).ok())
    }

    /// `priced` the two through the spec and compare within tolerance; a
    /// value the spec refuses (of the wrong kind) is alike only when
    /// identical.
    fn alike(
        &self,
        a: &Params,
        b: &Params,
        priced: impl Fn(&crate::ff::ir::StyleSpec, &Params) -> Option<Params>,
    ) -> bool {
        match &self.spec {
            Some(spec) => match (priced(spec, a), priced(spec, b)) {
                (Some(a), Some(b)) => a.approx_eq(&b, MERGE_REL_TOL),
                _ => a == b,
            },
            None => a.parameters().approx_eq(&b.parameters(), MERGE_REL_TOL),
        }
    }
}

impl ForceField {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            styles: Vec::new(),
            units: None,
            special_bonds: None,
            registry: RegistryRef::Global,
        }
    }

    /// Look this force field's categories, kernels and codecs up in
    /// `registry` instead of the process-wide one — a test's own styles,
    /// seen by nothing else. [`def_style`](Self::def_style), the
    /// [`PotentialCompiler`](crate::ff::compile::PotentialCompiler) and the
    /// force-field writers all read it.
    pub fn with_registry(mut self, registry: Arc<Registry>) -> Self {
        self.registry = RegistryRef::Own(registry);
        self
    }

    /// The registry this force field is read against.
    pub fn registry(&self) -> &RegistryRef {
        &self.registry
    }

    /// Run `f` on this force field's registry under no copy: the process-wide
    /// one is read under its lock, so `f` must not register anything.
    pub(crate) fn read_registry<R>(&self, f: impl FnOnce(&Registry) -> R) -> R {
        match &self.registry {
            RegistryRef::Global => crate::ff::style_registry::GlobalRegistry.with(f),
            RegistryRef::Own(r) => f(r),
        }
    }

    /// The unit system the parameters are expressed in (a LAMMPS `units`
    /// name: `real`, `metal`, `lj`, …), or `None` when none is declared.
    pub fn units(&self) -> Option<&str> {
        self.units.as_deref()
    }

    /// The declared unit system, or the error naming the force field that a
    /// reader of its numbers (a unit conversion, a record) needs: undeclared
    /// units are never assumed.
    pub fn required_units(&self) -> Result<&str, String> {
        self.units.as_deref().ok_or_else(|| {
            format!(
                "force field '{}' declares no units: the unit system its numbers are in \
                 must be stated (set_units), never assumed",
                self.name
            )
        })
    }

    /// Declare the unit system the parameters are expressed in.
    pub fn set_units(&mut self, units: &str) {
        self.units = Some(units.to_owned());
    }

    /// The force field's declared [`SpecialBonds`] 1-2 / 1-3 / 1-4
    /// nonbonded scale weights, or `None` when it declares none. Pair kernels
    /// weigh each special pair (a `pairs` row) by its `n_bonds` class;
    /// compiling a field with a pair style and no weights is an error, never
    /// a default.
    pub fn special_bonds(&self) -> Option<&SpecialBonds> {
        self.special_bonds.as_ref()
    }

    /// The declared [`SpecialBonds`], or the error naming the force field
    /// that a reader of its pair weights (a compile, an engine writer) needs.
    pub fn required_special_bonds(&self) -> Result<&SpecialBonds, String> {
        self.special_bonds.as_ref().ok_or_else(|| {
            format!(
                "force field '{}' declares no special_bonds: its 1-2 / 1-3 / 1-4 pair weights \
                 must be stated (set_special_bonds), never assumed",
                self.name
            )
        })
    }

    /// Declare the [`SpecialBonds`] weights. Force-field readers set these per
    /// force field (Amber/GAFF, OPLS, …). Declaring the default value is still
    /// a declaration.
    pub fn set_special_bonds(&mut self, special_bonds: SpecialBonds) {
        self.special_bonds = Some(special_bonds);
    }

    /// Merge `other` into `self`: the union of both definitions.
    ///
    /// A style is identified by `(category, name)` and a type by its name
    /// within its style, as for the definition primitives
    /// ([`Self::def_style`], [`Style::def_type`]); `other`'s new styles and
    /// types are defined through them. Where both define one, the two are
    /// one definition when they **price alike** — what the style's kernel
    /// reads from them is the same:
    ///
    /// - only the parameters the style's [spec](crate::ff::ir::StyleSpec)
    ///   declares are compared, each declared default filled in where a
    ///   definition leaves it out ([`StyleSpec::priced_style_params`],
    ///   [`StyleSpec::priced_row`]). A `coul/cut` that states `delta = 0`
    ///   and one that leaves it out are one style; a type's annotations,
    ///   an estimate's provenance and a reader's bookkeeping (a prmtop atom
    ///   type's `id`) take no part. A style no registry declares compares
    ///   every parameter column ([`Params::parameters`]).
    /// - numbers agree within a relative [`MERGE_REL_TOL`]
    ///   ([`Params::approx_eq`]): two sources store one value at their own
    ///   precision (tleap writes θ₀ in radians converted with π = 3.141594,
    ///   4.3·10⁻⁷ off `gaff.dat`'s degrees, and eight significant digits of
    ///   every number).
    ///
    /// Alike, `self`'s definition is kept; otherwise it is
    /// `Err(DefError::StyleConflict)` / `Err(DefError::TypeConflict)` /
    /// `Err(DefError::PairConflict)`, as from the primitives (a pair type is
    /// also identified by its unordered endpoints). `self`'s styles come
    /// first, then `other`'s new styles in `other`'s order; types keep their
    /// order.
    ///
    /// Declared `units` and `special_bonds` are adopted when `self` declares
    /// none; two declared values that differ are
    /// `Err(DefError::UnitsConflict)` / `Err(DefError::SpecialBondsConflict)`.
    ///
    /// All-or-nothing: on `Err`, `self` is unchanged. `self.name` is kept.
    ///
    /// [`StyleSpec::priced_style_params`]: crate::ff::ir::StyleSpec::priced_style_params
    /// [`StyleSpec::priced_row`]: crate::ff::ir::StyleSpec::priced_row
    pub fn merge(&mut self, other: &ForceField) -> Result<(), DefError> {
        let mut out = self.clone();
        match (&out.units, &other.units) {
            (Some(ours), Some(theirs)) if ours != theirs => {
                return Err(DefError::UnitsConflict {
                    ours: ours.clone(),
                    theirs: theirs.clone(),
                });
            }
            (None, Some(theirs)) => out.units = Some(theirs.clone()),
            _ => {}
        }
        match (out.special_bonds, other.special_bonds) {
            (Some(ours), Some(theirs)) if ours != theirs => {
                return Err(DefError::SpecialBondsConflict { ours, theirs });
            }
            (None, Some(theirs)) => out.special_bonds = Some(theirs),
            _ => {}
        }
        for style in &other.styles {
            let pricing = Pricing::of(&out.registry, style.category(), &style.name);
            let target = out.merge_style(style, &pricing)?;
            for (name, endpoints, params) in style.type_rows() {
                let alike = |a: &Params, b: &Params| pricing.rows_alike(name, a, b);
                if !target.check_type_by(name, &endpoints, params, alike, alike)? {
                    target.def_type(name, &endpoints, params.clone())?;
                }
            }
        }
        *self = out;
        Ok(())
    }

    /// [`Self::merge`]'s style step: `self`'s style of `theirs`'s
    /// `(category, name)` when it prices alike, `theirs`'s style (with no
    /// types yet) when `self` has none, `Err(DefError::StyleConflict)`
    /// otherwise.
    fn merge_style(&mut self, theirs: &Style, pricing: &Pricing) -> Result<&mut Style, DefError> {
        let params = match self.get_style(theirs.category(), &theirs.name) {
            Some(ours) if !pricing.styles_alike(&ours.params, &theirs.params) => {
                return Err(DefError::StyleConflict {
                    category: theirs.category().to_owned(),
                    name: theirs.name.clone(),
                });
            }
            Some(ours) => ours.params.clone(),
            None => theirs.params.clone(),
        };
        self.define_style(theirs.defs.empty_like(), &theirs.name, &params)
    }

    /// Define the `category` style named `name`, or return the existing one.
    ///
    /// `category` names a **declared** category, its arity taken from its
    /// declaration: one this force field's registry declares (the built-in
    /// ones, and [`GlobalRegistry::register_category`](crate::ff::style_registry::GlobalRegistry::register_category)'s for the
    /// process-wide registry), one of the IR's
    /// [built-in categories](crate::ff::ir::builtin_categories), or a
    /// category this force field already holds a style of (one read from a
    /// record that no registry declares). Any other category is
    /// `Err(DefError::UnknownCategory)`;
    /// [`def_style_with_arity`](Self::def_style_with_arity) brings one in.
    ///
    /// A style is identified by `(category, name)`: a repeated definition with
    /// exactly equal `params` returns the style already defined; with
    /// different `params` it is `Err(DefError::StyleConflict)` and the first
    /// definition is kept.
    pub fn def_style(
        &mut self,
        category: &str,
        name: &str,
        params: Params,
    ) -> Result<&mut Style, DefError> {
        let defs = self.empty_defs(category)?;
        self.define_style(defs, name, &params)
    }

    /// [`def_style`](Self::def_style), stating that the category's types
    /// name `arity` endpoints — how a reader brings in a category that
    /// nothing declares, its arity taken from the endpoint columns of the
    /// record (molrec: an unknown category's arity is its endpoint prefix).
    ///
    /// A declared category keeps its own arity, so `arity` must equal it
    /// (`Err(DefError::CategoryArity)`; a pair's is 2). An undeclared
    /// category spelled like one (`^[a-z][a-z0-9_]*$`) is brought in with
    /// `arity` (0 ..= 5, else `Err(DefError::CategoryArity)`); any other is
    /// `Err(DefError::UnknownCategory)`.
    ///
    /// ```
    /// use molrs::ff::forcefield::ForceField;
    /// use molrs::ff::ir::Params;
    ///
    /// let mut ff = ForceField::new("example");
    /// ff.def_style_with_arity("bond", 2, "harmonic", Params::new()).unwrap();
    /// ff.def_style_with_arity("bespoke", 4, "x", Params::new()).unwrap();
    /// assert_eq!(ff.get_style("bespoke", "x").unwrap().arity(), 4);
    /// assert!(ff.def_style_with_arity("bond", 3, "other", Params::new()).is_err());
    /// ```
    pub fn def_style_with_arity(
        &mut self,
        category: &str,
        arity: usize,
        name: &str,
        params: Params,
    ) -> Result<&mut Style, DefError> {
        let defs = match self.empty_defs(category) {
            Ok(defs) if defs.arity() == arity => defs,
            Ok(defs) => {
                return Err(DefError::CategoryArity {
                    category: category.to_owned(),
                    expected: defs.arity(),
                    got: arity,
                });
            }
            Err(DefError::UnknownCategory(_)) if is_category_name(category) => {
                let arity = u8::try_from(arity)
                    .ok()
                    .filter(|&n| usize::from(n) <= MAX_ARITY)
                    .ok_or_else(|| DefError::CategoryArity {
                        category: category.to_owned(),
                        expected: MAX_ARITY,
                        got: arity,
                    })?;
                StyleDefs::new(category, Arity::Exact(arity))
            }
            Err(e) => return Err(e),
        };
        self.define_style(defs, name, &params)
    }

    /// No definitions under `category`, of the arity its declaration gives
    /// it — in this force field's registry, else among the IR's built-in
    /// categories — else of the arity of the styles of `category` this
    /// force field holds.
    fn empty_defs(&self, category: &str) -> Result<StyleDefs, DefError> {
        let declared = self
            .read_registry(|r| r.category(category).map(|spec| spec.arity))
            .or_else(|| {
                builtin_categories()
                    .into_iter()
                    .find(|spec| spec.name == category)
                    .map(|spec| spec.arity)
            });
        if let Some(arity) = declared {
            return Ok(StyleDefs::new(category, arity));
        }
        self.styles
            .iter()
            .find(|s| s.category() == category)
            .map(|s| s.defs.empty_like())
            .ok_or_else(|| DefError::UnknownCategory(category.to_owned()))
    }

    /// The one path a style is defined through: `defs` (empty) is its
    /// category; an existing `(category, name)` with equal params is
    /// returned, with other params refused.
    fn define_style(
        &mut self,
        defs: StyleDefs,
        name: &str,
        params: &Params,
    ) -> Result<&mut Style, DefError> {
        let category = defs.category();
        if let Some(other) = self.styles.iter().find(|s| s.category() == category)
            && other.arity() != defs.arity()
        {
            return Err(DefError::CategoryArity {
                category: category.to_owned(),
                expected: other.arity(),
                got: defs.arity(),
            });
        }
        if let Some(idx) = self
            .styles
            .iter()
            .position(|s| s.category() == category && s.name == name)
        {
            if self.styles[idx].params != *params {
                return Err(DefError::StyleConflict {
                    category: category.to_owned(),
                    name: name.to_owned(),
                });
            }
            return Ok(&mut self.styles[idx]);
        }
        self.styles.push(Style::new(defs, name, params.clone()));
        Ok(self.styles.last_mut().expect("a style was just pushed"))
    }

    /// The rule [`Self::def_style`] applies, checked without defining: an
    /// unknown `category` is `Err(DefError::UnknownCategory)`, and a style
    /// `(category, name)` already defined with other `params` is
    /// `Err(DefError::StyleConflict)`. `Ok(())` means `def_style` would
    /// either define the style or return the identical one.
    pub(crate) fn check_style(
        &self,
        category: &str,
        name: &str,
        params: &Params,
    ) -> Result<(), DefError> {
        self.empty_defs(category)?;
        match self.get_style(category, name) {
            Some(existing) if existing.params != *params => Err(DefError::StyleConflict {
                category: category.to_owned(),
                name: name.to_owned(),
            }),
            _ => Ok(()),
        }
    }

    /// An empty force field with this one's name and its **declared** units
    /// and special_bonds, and no styles or types.
    ///
    /// The seed of a typing output: a typifier's output starts as
    /// `library.empty_like()` and gains only the definitions typing assigns.
    /// Undeclared state stays undeclared — the defaults are not declared.
    pub fn empty_like(&self) -> ForceField {
        ForceField {
            name: self.name.clone(),
            styles: Vec::new(),
            units: self.units.clone(),
            special_bonds: self.special_bonds,
            registry: self.registry.clone(),
        }
    }

    // -- queries --

    pub fn styles(&self) -> &[Style] {
        &self.styles
    }

    pub fn get_style(&self, category: &str, name: &str) -> Option<&Style> {
        self.styles
            .iter()
            .find(|s| s.category() == category && s.name == name)
    }

    /// Mutable style lookup, for the explicit edits (`set_type_param`,
    /// `set_param`, `rename_type`, `del_type`) of an existing definition.
    pub fn get_style_mut(&mut self, category: &str, name: &str) -> Option<&mut Style> {
        self.styles
            .iter_mut()
            .find(|s| s.category() == category && s.name == name)
    }

    /// The nonbonded treatment this force field's pair styles state, each
    /// style read by its declared [`PairRole`](crate::ff::ir::PairRole): its
    /// Coulomb range (long-range when any Coulomb style is), the Coulomb
    /// styles' largest `cutoff` and the dispersion styles' largest `cutoff`.
    /// Damping styles set neither.
    ///
    /// Cutoffs are in `length_unit` (a unit-registry name, `"nanometer"`),
    /// converted from the length unit of the force field's declared unit
    /// preset; `None` keeps them in the force field's own unit.
    ///
    /// # Errors
    ///
    /// A pair style the registry does not know, a `cutoff` that is no number,
    /// or — when there is a cutoff to convert — undeclared units, a preset
    /// the registry has no length unit for, or a `length_unit` it cannot
    /// convert to.
    pub fn nonbonded(&self, length_unit: Option<&str>) -> Result<Nonbonded, String> {
        use crate::ff::ir::{CoulombRange, PairRole};
        let mut coulomb: Option<CoulombRange> = None;
        let (mut coulomb_cutoff, mut vdw_cutoff): (Option<F>, Option<F>) = (None, None);
        for style in self.get_styles("pair") {
            let name = style.name();
            let role = self
                .registry()
                .with(|r| r.style("pair", name).map(|(spec, _)| spec.pair_role()))
                .ok_or_else(|| format!("pair style '{name}' is not registered"))?;
            let cutoff = match style.params().get("cutoff") {
                None => None,
                Some(v) => Some(
                    v.as_num()
                        .ok_or_else(|| format!("pair style '{name}': cutoff is not a number"))?,
                ),
            };
            let widen = |held: Option<F>| match (held, cutoff) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
            match role {
                PairRole::Coulomb(range) => {
                    coulomb = Some(match (coulomb, range) {
                        (Some(CoulombRange::Long), _) | (_, CoulombRange::Long) => {
                            CoulombRange::Long
                        }
                        _ => CoulombRange::Cut,
                    });
                    coulomb_cutoff = widen(coulomb_cutoff);
                }
                PairRole::Dispersion => vdw_cutoff = widen(vdw_cutoff),
                PairRole::Damping => {}
            }
        }
        let factor = match length_unit {
            Some(target) if coulomb_cutoff.is_some() || vdw_cutoff.is_some() => {
                let units = self.required_units()?;
                let preset = crate::core::UnitPreset::builtin(units)
                    .ok_or_else(|| format!("force field units '{units}' are no unit preset"))?;
                crate::core::UnitRegistry::global()
                    .factor(preset.length(), target)
                    .map_err(|e| format!("cutoffs from {} to {target}: {e}", preset.length()))?
            }
            _ => 1.0,
        };
        Ok(Nonbonded {
            coulomb,
            coulomb_cutoff: coulomb_cutoff.map(|c| c * factor),
            vdw_cutoff: vdw_cutoff.map(|c| c * factor),
        })
    }

    pub fn get_styles(&self, category: &str) -> Vec<&Style> {
        self.styles
            .iter()
            .filter(|s| s.category() == category)
            .collect()
    }

    /// The types of every style of `category`, style by style, each in
    /// definition order.
    pub fn get_types(&self, category: &str) -> Vec<&TypeDef> {
        self.styles
            .iter()
            .filter(|s| s.category() == category)
            .flat_map(|s| s.defs.types())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use molrs::core::TypeName;
    use ndarray::ArrayD;

    /// Each pair style is read by its declared role: the Coulomb range and
    /// cutoff from the Coulomb styles, the dispersion cutoff from the rest,
    /// nothing from a damping; cutoffs converted through the units preset.
    #[test]
    fn nonbonded_reads_each_pair_style_by_its_role() {
        let mut ff = ForceField::new("t");
        ff.set_units("real");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 12.0)]))
            .unwrap();
        ff.def_style("pair", "thole", Params::from_iter([("cutoff", 99.0)]))
            .unwrap();
        let nb = ff.nonbonded(Some("nanometer")).unwrap();
        assert_eq!(nb.coulomb, Some(crate::ff::ir::CoulombRange::Cut));
        assert!((nb.coulomb_cutoff.unwrap() - 1.2).abs() < 1e-12, "{nb:?}");
        assert!((nb.vdw_cutoff.unwrap() - 1.0).abs() < 1e-12, "{nb:?}");
        assert_eq!(ff.nonbonded(None).unwrap().vdw_cutoff, Some(10.0));

        ff.set_units("lj");
        let err = ff.nonbonded(Some("nanometer")).unwrap_err();
        assert!(err.contains("nanometer"), "{err}");
    }

    #[test]
    fn test_params() {
        let p = Params::from_iter([("k", 300.0), ("r0", 1.5)]);
        assert_eq!(p.get("k").and_then(|v| v.as_num()), Some(300.0));
        assert_eq!(p.get("r0").and_then(|v| v.as_num()), Some(1.5));
        assert_eq!(p.get("missing"), None);
    }

    // -- ForceField::def_style ---------------------------------------------

    #[test]
    fn def_style_returns_the_existing_style_for_a_repeated_category_and_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_iter([("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap();

        let styles = ff.get_styles("bond");
        assert_eq!(styles.len(), 1);
        let types = styles[0].defs().types();
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn def_style_keys_a_style_by_category_and_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new()).unwrap();
        ff.def_style("angle", "harmonic", Params::new()).unwrap();

        assert_eq!(ff.styles().len(), 2);
        assert!(ff.get_style("bond", "harmonic").is_some());
        assert!(ff.get_style("angle", "harmonic").is_some());
    }

    #[test]
    fn def_style_rejects_an_unknown_category() {
        let mut ff = ForceField::new("test");
        assert!(matches!(
            ff.def_style("bogus", "x", Params::new()),
            Err(DefError::UnknownCategory(_))
        ));
        assert!(ff.styles().is_empty());
    }

    /// A category beyond the seven holds its rows under its declared arity,
    /// and every edit reads and writes it like any other: its rows,
    /// params, renames, removals, the conflict rule and a merge.
    #[test]
    fn a_relation_style_is_edited_like_any_other() {
        let mut ff = ForceField::new("t");
        // molrec's `drude`, which the built-in registry declares (arity 2).
        let style = ff.def_style("drude", "harmonic", Params::new()).unwrap();
        assert_eq!((style.category(), style.arity()), ("drude", 2));
        style
            .def_type("C-D", &["C", "D"], Params::from_iter([("k", 500.0)]))
            .unwrap();
        assert!(matches!(
            style.def_type("C-D", &["C", "E"], Params::from_iter([("k", 500.0)])),
            Err(DefError::TypeConflict { category, .. }) if category == "drude"
        ));
        assert!(matches!(
            style.def_type("X", &["C"], Params::new()),
            Err(DefError::Arity { got: 1, .. })
        ));
        assert_eq!(
            style.type_endpoints("C-D"),
            Some(vec!["C".into(), "D".into()])
        );
        assert!(style.set_type_param("C-D", "k", 400.0));
        assert!(style.set_type_param("C-D", "desc", "core"));
        assert!(style.set_type_param("C-D", "table", ArrayD::zeros(vec![3])));
        assert_eq!(
            style
                .type_params("C-D")
                .unwrap()
                .get("k")
                .and_then(|v| v.as_num()),
            Some(400.0)
        );
        assert_eq!(style.rename_type("C-D", "core"), Ok(true));
        assert_eq!(
            style.type_rows(),
            vec![("core", vec!["C", "D"], style.type_params("core").unwrap())]
        );
        assert_eq!(ff.get_types("drude").len(), 1);

        let mut other = ForceField::new("u");
        other
            .def_style("drude", "harmonic", Params::new())
            .unwrap()
            .def_type("O-D", &["O", "D"], Params::from_iter([("k", 300.0)]))
            .unwrap();
        ff.merge(&other).unwrap();
        assert_eq!(
            ff.get_style("drude", "harmonic").unwrap().type_rows().len(),
            2
        );
        assert_eq!(
            ff.get_style_mut("drude", "harmonic")
                .unwrap()
                .del_type("core"),
            1
        );
    }

    /// `kspace` is not a category a force field can declare a style under.
    ///
    /// Where PME actually *is* registered — `pair/coul/long/pme`, and nothing
    /// under `kspace` — is a registry claim, and `registry.rs` asserts it.
    #[test]
    fn kspace_is_not_a_style_category() {
        let mut ff = ForceField::new("test");
        assert!(matches!(
            ff.def_style("kspace", "pme", Params::new()),
            Err(DefError::UnknownCategory(_))
        ));
    }

    // -- Style accessors ---------------------------------------------------

    #[test]
    fn style_name_is_the_defined_name() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        assert_eq!(style.name(), "lj/cut");
    }

    #[test]
    fn style_params_carry_numeric_and_string_values_of_the_definition() {
        let mut params = Params::from_iter([("cutoff", 10.0)]);
        params.set("mixing", "geometric");
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", params).unwrap();

        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            style.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );
        assert_eq!(
            style.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    #[test]
    fn style_defs_hold_the_defined_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap();

        let style = ff.get_style("atom", "full").unwrap();
        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "CT");
        assert_eq!(
            types[0].params.get("mass").and_then(|v| v.as_num()),
            Some(12.011)
        );
    }

    // -- Style::def_type -----------------------------------------------------

    #[test]
    fn set_param_declares_a_cutoff_on_an_existing_style() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .set_param("cutoff", 12.0);
        assert_eq!(
            ff.get_style("pair", "lj/cut")
                .unwrap()
                .params()
                .get("cutoff")
                .and_then(|v| v.as_num()),
            Some(12.0)
        );
    }

    #[test]
    fn def_type_atom() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "CT",
                &[],
                Params::from_iter([("mass", 12.011), ("charge", -0.12)]),
            )
            .unwrap()
            .def_type(
                "HC",
                &[],
                Params::from_iter([("mass", 1.008), ("charge", 0.06)]),
            )
            .unwrap();

        let style = ff.get_style("atom", "full").unwrap();
        let types = style.defs().types();
        assert_eq!(types.len(), 2);

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(ct.params.get("mass").and_then(|v| v.as_num()), Some(12.011));
        assert_eq!(
            ct.params.get("charge").and_then(|v| v.as_num()),
            Some(-0.12)
        );
    }

    #[test]
    fn def_type_bond() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_iter([("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        let bt = style.get_bondtype("CT", "OH").unwrap();
        assert_eq!(bt.endpoints[0], "CT");
        assert_eq!(bt.endpoints[1], "OH");
        assert_eq!(bt.params.get("k").and_then(|v| v.as_num()), Some(300.0));
        assert_eq!(bt.params.get("r0").and_then(|v| v.as_num()), Some(1.4));
    }

    #[test]
    fn get_bondtype_is_order_independent() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-HC",
                &["CT", "HC"],
                Params::from_iter([("k", 340.0), ("r0", 1.09)]),
            )
            .unwrap();

        let bt = style.get_bondtype("HC", "CT").unwrap();
        assert_eq!(bt.params.get("r0").and_then(|v| v.as_num()), Some(1.09));
    }

    #[test]
    fn def_type_angle() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("angle", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_iter([("k", 33.0), ("theta0", 107.8)]),
            )
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].endpoints[0], "HC");
        assert_eq!(types[0].endpoints[1], "CT");
        assert_eq!(types[0].endpoints[2], "HC");
        assert_eq!(
            types[0].params.get("theta0").and_then(|v| v.as_num()),
            Some(107.8)
        );
    }

    #[test]
    fn def_type_dihedral() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("dihedral", "opls", Params::new()).unwrap();
        style
            .def_type(
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.3)]),
            )
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].endpoints[0], "HC");
        assert_eq!(types[0].endpoints[3], "HC");
    }

    #[test]
    fn def_type_pair_self() {
        let mut ff = ForceField::new("test");
        let style = ff
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        style
            .def_type(
                "Ar",
                &["Ar"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 3.4)]),
            )
            .unwrap();

        let pt = style.get_pairtype("Ar", None).unwrap();
        assert_eq!(pt.endpoints[0], "Ar");
        assert_eq!(pt.endpoints[1], "Ar");
        assert_eq!(pt.params.get("epsilon").and_then(|v| v.as_num()), Some(1.0));
    }

    #[test]
    fn def_type_pair_cross() {
        let mut ff = ForceField::new("test");
        let style = ff
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        style
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.3)]),
            )
            .unwrap();

        let pt = style.get_pairtype("OH", Some("CT")).unwrap();
        assert_eq!(pt.endpoints[0], "CT");
        assert_eq!(pt.endpoints[1], "OH");
        assert_eq!(pt.params.get("epsilon").and_then(|v| v.as_num()), Some(0.1));
    }

    #[test]
    fn def_type_carries_string_params_with_the_definition() {
        let mut params = Params::from_iter([("mass", 12.011)]);
        params.set("element", "C");
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style.def_type("CT", &[], params).unwrap();

        let ct = style.get_atomtype("CT").unwrap();
        assert_eq!(
            ct.params.get("element").and_then(|v| v.as_text()),
            Some("C")
        );
    }

    #[test]
    fn def_type_chains_on_the_returned_style() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_iter([("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 2);
    }

    // -- Style::def_type: endpoints given, name stored verbatim ----------------

    /// The name is an opaque identifier: a bond named `anything` holds exactly
    /// the endpoints it was given, under exactly that name.
    #[test]
    fn def_type_stores_the_given_endpoints_under_the_verbatim_name() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "anything",
                &["CT", "OH"],
                Params::from_iter([("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "anything");
        assert_eq!(types[0].endpoints[0], "CT");
        assert_eq!(types[0].endpoints[1], "OH");
        assert_eq!(
            style.type_endpoints("anything"),
            Some(vec!["CT".to_string(), "OH".to_string()])
        );
    }

    /// A `-` in a name is a character, never a separator: the bond named
    /// `CT-OH` defined on `HC`, `OS` holds `HC`, `OS`; the atom type `C-1`
    /// takes no endpoints; the self pair `tip3p-O` holds its one label twice.
    #[test]
    fn def_type_never_splits_a_dashed_name() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("CT-OH", &["HC", "OS"], Params::new())
            .unwrap();
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("C-1", &[], Params::new())
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("tip3p-O", &["tip3p-O"], Params::new())
            .unwrap();

        assert_eq!(
            ff.get_style("bond", "harmonic")
                .unwrap()
                .type_endpoints("CT-OH"),
            Some(vec!["HC".to_string(), "OS".to_string()])
        );
        assert_eq!(
            ff.get_style("atom", "full").unwrap().type_endpoints("C-1"),
            Some(Vec::new())
        );
        assert_eq!(
            ff.get_style("pair", "lj/cut")
                .unwrap()
                .type_endpoints("tip3p-O"),
            Some(vec!["tip3p-O".to_string(), "tip3p-O".to_string()])
        );
    }

    /// A reversed name is its own name: `OH-CT` is stored as written, not
    /// reoriented.
    #[test]
    fn def_type_keeps_a_reversed_name_as_written() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type("OH-CT", &["OH", "CT"], Params::new())
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types[0].name, "OH-CT");
        assert_eq!(
            (
                types[0].endpoints[0].as_str(),
                types[0].endpoints[1].as_str()
            ),
            ("OH", "CT")
        );
    }

    /// An endpoint count off the category's arity is `Err(Arity)` (never a
    /// panic), whatever the name looks like; nothing is stored.
    /// `@` starts a [`TypeName`](molrs::core::TypeName)
    /// qualifier, so no endpoint label may carry it — nor an atom type's name,
    /// which is the endpoint every other type is defined on. A qualified
    /// *type name* (`C_3-C_R@1.5`) is fine: names are never joined.
    #[test]
    fn def_type_refuses_an_endpoint_label_containing_at() {
        let mut ff = ForceField::new("t");
        let err = ff
            .def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type("x", &["a@b"], Params::new())
            .unwrap_err();
        assert!(matches!(err, DefError::Name(_)), "{err}");
        let err = ff
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("a@b", &[], Params::new())
            .unwrap_err();
        assert!(matches!(err, DefError::Name(_)), "{err}");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type("C_3-C_R@1.5", &["C_3", "C_R"], Params::new())
            .unwrap();
    }

    #[test]
    fn type_params_finds_one_type_by_name() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type("A-B", &["A", "B"], Params::from_iter([("k", 1.0)]))
            .unwrap();
        assert_eq!(
            style
                .type_params("A-B")
                .and_then(|p| p.get("k").and_then(|v| v.as_num())),
            Some(1.0)
        );
        assert!(style.type_params("B-A").is_none());
    }

    #[test]
    fn set_param_writes_a_string_style_param() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style.set_param("mixing", "geometric");
        assert_eq!(
            style.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    #[test]
    fn def_type_endpoint_count_off_the_category_arity_is_an_arity_error() {
        let cases: [(&str, &str, &[&str]); 10] = [
            ("atom", "full", &["CT"]),
            ("bond", "harmonic", &["CT"]),
            ("bond", "harmonic", &["CT", "CT", "CT"]),
            ("angle", "harmonic", &["CT", "CT"]),
            ("dihedral", "opls", &["CT", "CT", "CT"]),
            ("improper", "harmonic", &["CT", "CT", "CT", "CT", "CT"]),
            ("pair", "lj/cut", &[]),
            ("pair", "lj/cut", &["CT", "CT", "CT"]),
            ("cmap", "charmm", &["CT", "CT", "CT", "CT"]),
            ("cmap", "charmm", &["CT", "CT", "CT", "CT", "CT", "CT"]),
        ];
        for (category, style_name, endpoints) in cases {
            let mut ff = ForceField::new("test");
            let style = ff.def_style(category, style_name, Params::new()).unwrap();
            let got = style
                .def_type("CT-CT", endpoints, Params::new())
                .map(|_| ());
            assert!(
                matches!(
                    got,
                    Err(DefError::Arity { got: n, .. }) if n == endpoints.len()
                ),
                "{category} with {} endpoints: {got:?}",
                endpoints.len()
            );
            assert!(style.defs().collect_type_params().is_empty(), "{category}");
        }
    }

    #[test]
    fn def_type_atom_accepts_empty_endpoints() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap();

        assert_eq!(style.type_endpoints("CT"), Some(Vec::new()));
    }

    // -- Params equality -------------------------------------------------------

    /// Identity, not closeness: one ulp apart is a different parameter set, and
    /// the string side counts as much as the numeric side.
    #[test]
    fn params_equality_is_exact_on_both_sides() {
        let mut a = Params::from_iter([("k", 300.0)]);
        a.set("element", "C");
        let mut same = Params::from_iter([("k", 300.0)]);
        same.set("element", "C");
        let mut next_ulp = Params::from_iter([("k", f64::from_bits(300.0_f64.to_bits() + 1))]);
        next_ulp.set("element", "C");
        let mut other_string = Params::from_iter([("k", 300.0)]);
        other_string.set("element", "N");

        assert_eq!(a, same);
        assert_ne!(a, next_ulp);
        assert_ne!(a, other_string);
    }

    fn grid(values: &[f64]) -> ArrayD<f64> {
        let n = (values.len() as f64).sqrt() as usize;
        ArrayD::from_shape_vec(vec![n, n], values.to_vec()).unwrap()
    }

    /// The array side is compared like the others: same keys, same shape,
    /// every element equal under `==`.
    #[test]
    fn params_equality_is_exact_on_the_array_side() {
        let mut a = Params::new();
        a.set("grid", grid(&[0.0, 1.0, 2.0, 3.0]));
        let same = a.clone();
        let mut next_ulp = Params::new();
        next_ulp.set(
            "grid",
            grid(&[0.0, 1.0, 2.0, f64::from_bits(3.0_f64.to_bits() + 1)]),
        );
        let mut reshaped = Params::new();
        reshaped.set(
            "grid",
            ArrayD::from_shape_vec(vec![4], vec![0.0, 1.0, 2.0, 3.0]).unwrap(),
        );

        assert_eq!(a, same);
        assert_ne!(a, next_ulp);
        assert_ne!(a, reshaped);
        assert_ne!(a, Params::new());
        assert_eq!(
            a.get("grid").and_then(|v| v.as_array()).map(|g| g.shape()),
            Some(&[2, 2][..])
        );
        assert_eq!(
            a.iter()
                .filter_map(|(k, v)| v.as_array().map(|a| (k, a)))
                .count(),
            1
        );
        assert!(a.same_parameters(&same));
        assert!(!a.same_parameters(&next_ulp));
    }

    /// A pair restating a stored pair is compared on its arrays too.
    #[test]
    fn a_pair_restatement_compares_array_parameters() {
        let with = |g: &[f64]| {
            let mut p = Params::from_iter([("epsilon", 0.1)]);
            p.set("table", grid(g));
            p
        };
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "table", Params::new()).unwrap();
        style
            .def_type("A-B", &["A", "B"], with(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        style
            .def_type("B-A", &["B", "A"], with(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        assert_eq!(
            style.type_rows().len(),
            1,
            "an equal restatement is a no-op"
        );
        assert!(matches!(
            style.def_type("B-A", &["B", "A"], with(&[1.0, 2.0, 3.0, 5.0])),
            Err(DefError::PairConflict { .. })
        ));
    }

    #[test]
    fn def_type_cmap_takes_five_endpoints_and_keeps_its_grid() {
        let mut params = Params::new();
        params.set("grid", grid(&[0.0, 0.5, -0.5, 0.25]));
        let mut ff = ForceField::new("test");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        style
            .def_type("C-NH1-CT1-C-NH1", &["C", "NH1", "CT1", "C", "NH1"], params)
            .unwrap();

        assert_eq!(style.category(), "cmap");
        assert_eq!(
            style.type_endpoints("C-NH1-CT1-C-NH1"),
            Some(["C", "NH1", "CT1", "C", "NH1"].map(String::from).to_vec())
        );
        let cmaps = ff.get_types("cmap");
        assert_eq!(cmaps.len(), 1);
        assert_eq!(
            (
                cmaps[0].endpoints[0].as_str(),
                cmaps[0].endpoints[4].as_str()
            ),
            ("C", "NH1")
        );
        assert_eq!(
            cmaps[0].params.get("grid").and_then(|v| v.as_array()),
            Some(&grid(&[0.0, 0.5, -0.5, 0.25]))
        );
        assert_eq!(
            TypeName::join(&["C", "NH1", "CT1", "C", "NH1"])
                .unwrap()
                .as_str(),
            "C-NH1-CT1-C-NH1"
        );
    }

    /// A different grid under a stored name is a type conflict, and the edits
    /// carry arrays: `set_type_param` replaces one in place and
    /// `rename_type` keeps it.
    #[test]
    fn a_cmap_grid_takes_part_in_the_conflict_rule_and_the_edits() {
        let ends = ["C", "N", "CA", "C", "N"];
        let with = |g: &[f64]| {
            let mut p = Params::new();
            p.set("grid", grid(g));
            p
        };
        let mut ff = ForceField::new("t");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        style.def_type("c1", &ends, with(&[1.0; 4])).unwrap();
        style.def_type("c1", &ends, with(&[1.0; 4])).unwrap();
        assert!(matches!(
            style.def_type("c1", &ends, with(&[2.0; 4])),
            Err(DefError::TypeConflict { .. })
        ));
        assert!(style.set_type_param("c1", "grid", grid(&[3.0; 4])));
        assert!(!style.set_type_param("nope", "grid", grid(&[3.0; 4])));
        assert!(style.rename_type("c1", "c2").unwrap());
        assert_eq!(
            style
                .type_params("c2")
                .and_then(|p| p.get("grid").and_then(|v| v.as_array())),
            Some(&grid(&[3.0; 4]))
        );
        assert_eq!(style.del_type("c2"), 1);
    }

    #[test]
    fn merge_carries_array_params() {
        let mut params = Params::new();
        params.set("grid", grid(&[1.0, 2.0, 3.0, 4.0]));
        let mut other = ForceField::new("other");
        other
            .def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["A", "B", "C", "D", "E"], params.clone())
            .unwrap();
        let mut ff = ForceField::new("ff");
        ff.merge(&other).unwrap();
        assert_eq!(ff.get_types("cmap")[0].params, params);
        ff.merge(&other).unwrap();
        assert_eq!(ff.get_types("cmap").len(), 1, "an identical overlap");
    }

    // -- the conflict rule: Style::def_type ---------------------------------------

    #[test]
    fn def_type_identical_redefinition_leaves_one_type() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 1);
    }

    /// A different numeric param under one name is rejected, and the first
    /// definition is the one left standing.
    #[test]
    fn def_type_with_a_different_numeric_param_is_a_type_conflict() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();

        let second = style.def_type(
            "CT-CT",
            &["CT", "CT"],
            Params::from_iter([("k", 310.0), ("r0", 1.529)]),
        );

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0].params.get("k").and_then(|v| v.as_num()),
            Some(268.0)
        );
    }

    /// A different string param under one name is rejected, and the first
    /// definition is the one left standing.
    #[test]
    fn def_type_with_a_different_string_param_is_a_type_conflict() {
        let mut carbon = Params::from_iter([("mass", 12.011)]);
        carbon.set("element", "C");
        let mut nitrogen = Params::from_iter([("mass", 12.011)]);
        nitrogen.set("element", "N");
        let mut ff = ForceField::new("test");
        let style = ff.def_style("atom", "full", Params::new()).unwrap();
        style.def_type("CT", &[], carbon).unwrap();

        let second = style.def_type("CT", &[], nitrogen);

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0].params.get("element").and_then(|v| v.as_text()),
            Some("C")
        );
    }

    /// One name, equal params, different endpoints: still a different
    /// definition, and the first endpoints are the ones left standing.
    #[test]
    fn def_type_same_name_with_different_endpoints_is_a_type_conflict() {
        let mut ff = ForceField::new("test");
        let style = ff.def_style("bond", "mmff", Params::new()).unwrap();
        style
            .def_type("0_1_5", &["1", "5"], Params::from_iter([("kb", 4.258)]))
            .unwrap();

        let second = style.def_type("0_1_5", &["1", "6"], Params::from_iter([("kb", 4.258)]));

        assert!(
            matches!(second, Err(DefError::TypeConflict { .. })),
            "{second:?}"
        );
        assert_eq!(
            style.type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    // -- the conflict rule: a pair is its unordered endpoints ----------------------

    fn lj_params(epsilon: f64) -> Params {
        Params::from_iter([("epsilon", epsilon), ("sigma", 2.0)])
    }

    /// `B-A` restating `A-B` with another epsilon: refused, whatever either
    /// is named, and the first row stands.
    #[test]
    fn def_type_restating_a_reversed_pair_with_other_params_is_a_pair_conflict() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style.def_type("A-B", &["A", "B"], lj_params(0.9)).unwrap();

        let second = style.def_type("B-A", &["B", "A"], lj_params(0.8));

        assert!(
            matches!(&second, Err(DefError::PairConflict { stated, .. }) if stated == "A-B"),
            "{second:?}"
        );
        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0].params.get("epsilon").and_then(|v| v.as_num()),
            Some(0.9)
        );
    }

    /// Two names on one ordered pair, and a self pair restated under another
    /// name, are one pair as well.
    #[test]
    fn def_type_restating_a_pair_in_the_same_order_with_other_params_is_a_pair_conflict() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style
            .def_type("nbfix-1", &["A", "B"], lj_params(0.9))
            .unwrap()
            .def_type("A", &["A"], lj_params(0.1))
            .unwrap();

        let second = style.def_type("nbfix-2", &["A", "B"], lj_params(0.8));
        assert!(
            matches!(second, Err(DefError::PairConflict { .. })),
            "{second:?}"
        );
        let self_pair = style.def_type("A-again", &["A", "A"], lj_params(0.2));
        assert!(
            matches!(self_pair, Err(DefError::PairConflict { .. })),
            "{self_pair:?}"
        );
    }

    /// An equal restatement, in either order and under any name, is a no-op:
    /// the annotations take no part, and the first row is the one kept.
    #[test]
    fn def_type_restating_a_pair_with_equal_params_is_a_no_op() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        let mut first = lj_params(0.9);
        first.set("desc", "NBFIX");
        style.def_type("A-B", &["A", "B"], first).unwrap();
        let mut restated = lj_params(0.9);
        restated.set("desc", "restated");
        restated.set("doi", "10.0/x");
        style
            .def_type("B-A", &["B", "A"], restated)
            .unwrap()
            .def_type("A-B-again", &["A", "B"], lj_params(0.9))
            .unwrap()
            .def_type("A-B", &["B", "A"], {
                let mut same = lj_params(0.9);
                same.set("desc", "NBFIX");
                same
            })
            .unwrap();

        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(types[0].name, "A-B");
        assert_eq!(
            types[0].params.get("desc").and_then(|v| v.as_text()),
            Some("NBFIX")
        );
    }

    /// A parameter one row carries and the other lacks is a difference, and a
    /// string parameter is a parameter.
    #[test]
    fn a_missing_or_string_parameter_is_a_pair_conflict() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style.def_type("A-B", &["A", "B"], lj_params(0.9)).unwrap();
        let lacking = Params::from_iter([("epsilon", 0.9)]);
        assert!(style.def_type("B-A", &["B", "A"], lacking).is_err());
        let mut flavoured = lj_params(0.9);
        flavoured.set("flavour", "x");
        assert!(style.def_type("B-A", &["B", "A"], flavoured).is_err());
    }

    /// Only a pair is found by its endpoints: two bond types on one pair are
    /// two definitions.
    #[test]
    fn a_bond_restating_its_endpoints_is_not_a_pair_conflict() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("bond", "harmonic", Params::new()).unwrap();
        style
            .def_type("CT-HC", &["CT", "HC"], Params::from_iter([("k", 1.0)]))
            .unwrap()
            .def_type("HC-CT", &["HC", "CT"], Params::from_iter([("k", 2.0)]))
            .unwrap();
        assert_eq!(style.type_rows().len(), 2);
    }

    /// The kernel finds a pair defined in either order under one key.
    #[test]
    fn kernel_type_params_key_a_pair_in_byte_order() {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style
            .def_type("nbfix", &["B", "A"], lj_params(0.9))
            .unwrap()
            .def_type("C", &["C"], lj_params(0.1))
            .unwrap();
        let keys: Vec<String> = style
            .defs()
            .kernel_type_params()
            .unwrap()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        assert_eq!(keys, ["A-B", "C"]);
        assert_eq!(pair_key("B", "A").unwrap(), pair_key("A", "B").unwrap());
    }

    // -- the conflict rule: ForceField::def_style --------------------------------

    /// Different style params under one `(category, name)` are rejected, and the
    /// first style params are the ones left standing.
    #[test]
    fn def_style_with_different_params_is_a_style_conflict() {
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();

        let second = ff
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 12.0)]))
            .map(|_| ());

        assert!(
            matches!(second, Err(DefError::StyleConflict { .. })),
            "{second:?}"
        );
        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            style.params().get("cutoff").and_then(|v| v.as_num()),
            Some(10.0)
        );
    }

    #[test]
    fn def_style_with_identical_params_returns_the_existing_style() {
        let mut ff = ForceField::new("test");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "Ar",
                &["Ar"],
                Params::from_iter([("epsilon", 0.238), ("sigma", 3.4)]),
            )
            .unwrap();

        let again = ff
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();

        assert!(again.get_pairtype("Ar", None).is_some());
        assert_eq!(ff.styles().len(), 1);
    }

    // -- edits: rename_type carries the collision rule, set_type_param does not --

    #[test]
    fn rename_type_to_an_unused_name_renames_it() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CX", &[], Params::from_iter([("mass", 13.0)]))
            .unwrap();
        let style = ff.get_style_mut("atom", "full").unwrap();

        assert_eq!(style.rename_type("CX", "CY"), Ok(true));
        assert!(style.get_atomtype("CY").is_some());
        assert!(style.get_atomtype("CX").is_none());
    }

    /// Renaming onto a name already defined with different params would leave
    /// two definitions under one name: a conflict, and nothing is renamed.
    #[test]
    fn rename_type_onto_an_existing_name_with_different_params_is_an_error() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.011)]))
            .unwrap()
            .def_type("CX", &[], Params::from_iter([("mass", 13.0)]))
            .unwrap();
        let style = ff.get_style_mut("atom", "full").unwrap();

        let renamed = style.rename_type("CX", "CT");

        assert!(
            matches!(renamed, Err(DefError::TypeConflict { .. })),
            "{renamed:?}"
        );
        assert_eq!(
            style
                .get_atomtype("CT")
                .unwrap()
                .params
                .get("mass")
                .and_then(|v| v.as_num()),
            Some(12.011)
        );
        assert!(style.get_atomtype("CX").is_some());
    }

    /// An edit of an existing definition is not a definition: it changes the
    /// value in place and never meets the conflict rule.
    #[test]
    fn set_type_param_changes_an_existing_type_in_place() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let style = ff.get_style_mut("bond", "harmonic").unwrap();

        let changed: bool = style.set_type_param("CT-CT", "k", 310.0);

        assert!(changed);
        let types = style.defs().types();
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0].params.get("k").and_then(|v| v.as_num()),
            Some(310.0)
        );
    }

    // -- ForceField queries ----------------------------------------------------

    #[test]
    fn test_get_all_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("CT", &[], Params::from_iter([("mass", 12.0)]))
            .unwrap()
            .def_type("OH", &[], Params::from_iter([("mass", 16.0)]))
            .unwrap();
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-OH",
                &["CT", "OH"],
                Params::from_iter([("k", 300.0), ("r0", 1.4)]),
            )
            .unwrap();

        assert_eq!(ff.get_types("atom").len(), 2);
        assert_eq!(ff.get_types("bond").len(), 1);
        assert_eq!(ff.get_types("pair").len(), 0);
    }

    #[test]
    fn get_angletypes_returns_the_defined_angle_params() {
        let mut ff = ForceField::new("test");
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_iter([("k", 33.0), ("theta0", 107.8)]),
            )
            .unwrap();

        let types = ff.get_types("angle");
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0].params.get("theta0").and_then(|v| v.as_num()),
            Some(107.8)
        );
    }

    // -- declared state: units / special_bonds ---------------------------------

    const AMBER_SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.8333],
    };

    const OPLS_SB: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 0.5],
        coul: [0.0, 0.0, 0.5],
    };

    #[test]
    fn units_is_none_until_set_units() {
        let mut ff = ForceField::new("test");
        assert_eq!(ff.units(), None);
        assert!(
            ff.required_units()
                .unwrap_err()
                .contains("declares no units")
        );

        ff.set_units("lj");

        assert_eq!(ff.units(), Some("lj"));
        assert_eq!(ff.required_units(), Ok("lj"));
    }

    #[test]
    fn special_bonds_is_none_until_set_special_bonds() {
        let mut ff = ForceField::new("test");
        assert_eq!(ff.special_bonds(), None);
        assert!(
            ff.required_special_bonds()
                .unwrap_err()
                .contains("declares no special_bonds")
        );

        ff.set_special_bonds(AMBER_SB);

        assert_eq!(ff.special_bonds(), Some(&AMBER_SB));
        assert_eq!(ff.required_special_bonds(), Ok(&AMBER_SB));
    }

    /// Declaring the 1-2 / 1-3 exclusion weights is a declaration like any
    /// other.
    #[test]
    fn set_special_bonds_to_exclude_12_13_declares_it() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        assert_eq!(ff.special_bonds(), Some(&SpecialBonds::EXCLUDE_12_13));
    }

    // -- ForceField::empty_like ------------------------------------------------

    /// The seed of a typing output: same name and declared state, no styles.
    #[test]
    fn empty_like_keeps_name_and_declarations_and_drops_every_style() {
        let mut library = ForceField::new("library");
        library.set_units("metal");
        library.set_special_bonds(AMBER_SB);
        library
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        library
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();

        let empty = library.empty_like();

        assert_eq!(empty.name, "library");
        assert_eq!(empty.units(), Some("metal"));
        assert_eq!(empty.special_bonds(), Some(&AMBER_SB));
        assert!(empty.styles().is_empty(), "{:?}", empty.styles());
    }

    /// Undeclared stays undeclared: `empty_like` does not declare the defaults.
    #[test]
    fn empty_like_of_an_undeclared_force_field_declares_nothing() {
        let library = ForceField::new("bare");

        let empty = library.empty_like();

        assert_eq!(empty.name, "bare");
        assert_eq!(empty.units(), None);
        assert_eq!(empty.special_bonds(), None);
        assert!(empty.styles().is_empty());
    }

    /// `empty_like` reads `self`; the library keeps its definitions.
    #[test]
    fn empty_like_leaves_the_source_unchanged() {
        let mut library = ForceField::new("library");
        library.set_special_bonds(OPLS_SB);
        library
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("X", &[], Params::from_iter([("mass", 1.0)]))
            .unwrap();
        let before = library.clone();

        let _ = library.empty_like();

        assert_same_definitions(&library, &before);
    }

    // -- ForceField::merge -----------------------------------------------------

    /// Every definition of `a` equals `b`'s, in order: name, declared state,
    /// and per style its category, name, params and types (endpoints and
    /// params included).
    pub(crate) fn assert_same_definitions(a: &ForceField, b: &ForceField) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.units(), b.units());
        assert_eq!(a.special_bonds(), b.special_bonds());
        assert_eq!(a.styles().len(), b.styles().len(), "style count");
        for (x, y) in a.styles().iter().zip(b.styles()) {
            assert_eq!(
                (x.category(), x.name(), x.params()),
                (y.category(), y.name(), y.params())
            );
            assert_eq!(
                x.defs(),
                y.defs(),
                "{}:{} types differ",
                x.category(),
                x.name()
            );
        }
    }

    fn style_keys(ff: &ForceField) -> Vec<(&str, &str)> {
        ff.styles()
            .iter()
            .map(|s| (s.category(), s.name()))
            .collect()
    }

    fn bond_names(style: &Style) -> Vec<&str> {
        let types = style.defs().types();
        types.iter().map(|t| t.name.as_str()).collect()
    }

    /// `self`'s styles first, then `other`'s new styles in `other`'s order; a
    /// shared style gains `other`'s new types after its own.
    #[test]
    fn merge_is_the_union_with_self_styles_first_then_new_styles_in_order() {
        let mut ff = ForceField::new("target");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B",
                &["A", "B"],
                Params::from_iter([("k", 1.0), ("r0", 1.0)]),
            )
            .unwrap();
        ff.def_style("pair", "lj/cut", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .unwrap();

        let mut other = ForceField::new("source");
        other
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-B-C",
                &["A", "B", "C"],
                Params::from_iter([("k", 50.0), ("theta0", 108.9)]),
            )
            .unwrap();
        other
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "C-D",
                &["C", "D"],
                Params::from_iter([("k", 2.0), ("r0", 2.0)]),
            )
            .unwrap()
            .def_type(
                "B-C",
                &["B", "C"],
                Params::from_iter([("k", 3.0), ("r0", 3.0)]),
            )
            .unwrap();
        other
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("X", &[], Params::from_iter([("mass", 1.0)]))
            .unwrap();

        ff.merge(&other).unwrap();

        assert_eq!(ff.name, "target");
        assert_eq!(
            style_keys(&ff),
            vec![
                ("bond", "harmonic"),
                ("pair", "lj/cut"),
                ("angle", "harmonic"),
                ("atom", "full"),
            ]
        );
        assert_eq!(
            bond_names(ff.get_style("bond", "harmonic").unwrap()),
            vec!["A-B", "C-D", "B-C"]
        );
        assert!(
            ff.get_style("angle", "harmonic")
                .unwrap()
                .type_endpoints("A-B-C")
                .is_some()
        );
        assert!(
            ff.get_style("atom", "full")
                .unwrap()
                .get_atomtype("X")
                .is_some()
        );
    }

    /// A type keeps its given endpoints through the merge.
    #[test]
    fn merge_keeps_the_given_endpoints() {
        let mut other = ForceField::new("mmff");
        other
            .def_style("bond", "mmff", Params::new())
            .unwrap()
            .def_type("0_1_5", &["1", "5"], Params::from_iter([("kb", 4.258)]))
            .unwrap();
        let mut ff = ForceField::new("target");

        ff.merge(&other).unwrap();

        assert_eq!(
            ff.get_style("bond", "mmff")
                .unwrap()
                .type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    #[test]
    fn merge_of_an_identical_overlap_is_a_no_op() {
        let mut ff = ForceField::new("same");
        ff.set_units("real");
        ff.set_special_bonds(AMBER_SB);
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let mut lj = Params::from_iter([("cutoff", 10.0)]);
        lj.set("mixing", "geometric");
        ff.def_style("pair", "lj/cut", lj)
            .unwrap()
            .def_type(
                "CT",
                &["CT"],
                Params::from_iter([("epsilon", 0.066), ("sigma", 3.5)]),
            )
            .unwrap();
        let before = ff.clone();

        ff.merge(&before).unwrap();

        assert_same_definitions(&ff, &before);
    }

    /// All-or-nothing: `other` defines a new style and a new type *before* the
    /// conflicting type and declares units and special_bonds; none of it may
    /// land.
    #[test]
    fn merge_type_conflict_leaves_self_equal_to_its_pre_merge_clone() {
        let mut ff = ForceField::new("target");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 268.0), ("r0", 1.529)]),
            )
            .unwrap();
        let before = ff.clone();

        let mut other = ForceField::new("source");
        other.set_units("real");
        other.set_special_bonds(AMBER_SB);
        other
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT",
                &["CT", "CT", "CT"],
                Params::from_iter([("k", 58.35), ("theta0", 111.7)]),
            )
            .unwrap();
        other
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-HC",
                &["CT", "HC"],
                Params::from_iter([("k", 340.0), ("r0", 1.09)]),
            )
            .unwrap()
            .def_type(
                "CT-CT",
                &["CT", "CT"],
                Params::from_iter([("k", 310.0), ("r0", 1.529)]),
            )
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::TypeConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_keeps_lj_cut_cutoff_and_mixing() {
        let mut lj = Params::from_iter([("cutoff", 12.0)]);
        lj.set("mixing", "geometric");
        let mut other = ForceField::new("source");
        other
            .def_style("pair", "lj/cut", lj)
            .unwrap()
            .def_type(
                "OW",
                &["OW"],
                Params::from_iter([("epsilon", 0.1553), ("sigma", 3.166)]),
            )
            .unwrap();
        let mut ff = ForceField::new("target");

        ff.merge(&other).unwrap();

        let style = ff.get_style("pair", "lj/cut").unwrap();
        assert_eq!(
            style.params().get("cutoff").and_then(|v| v.as_num()),
            Some(12.0)
        );
        assert_eq!(
            style.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    #[test]
    fn merge_with_a_different_coul_cut_coulomb_is_a_style_conflict() {
        let mut ff = ForceField::new("target");
        ff.def_style(
            "pair",
            "coul/cut",
            Params::from_iter([("coulomb", 332.06371), ("dielectric", 1.0)]),
        )
        .unwrap();
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other
            .def_style(
                "pair",
                "coul/cut",
                Params::from_iter([("coulomb", 332.0716), ("dielectric", 1.0)]),
            )
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::StyleConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_with_a_different_pair_cutoff_is_a_style_conflict() {
        let mut ff = ForceField::new("target");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other
            .def_style("pair", "lj/cut", Params::from_iter([("cutoff", 12.0)]))
            .unwrap();

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::StyleConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    /// A style param one force field states at its spec default and the
    /// other leaves out is one style param: `coul/cut`'s `delta = 0`,
    /// `lj/cut`'s `mixing = arithmetic`. A non-default value is another.
    #[test]
    fn merge_fills_spec_defaults_before_comparing_style_params() {
        let coulomb = Params::from_iter(
            [("coulomb", 332.0522173), ("dielectric", 1.0)]
                .iter()
                .copied(),
        );
        let mut stated = coulomb.clone();
        stated.set("delta", 0.0);
        let mut arithmetic = Params::new();
        arithmetic.set("mixing", "arithmetic");
        let mut ff = ForceField::new("target");
        ff.def_style("pair", "coul/cut", coulomb.clone()).unwrap();
        ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        let mut other = ForceField::new("source");
        other.def_style("pair", "coul/cut", stated).unwrap();
        other.def_style("pair", "lj/cut", arithmetic).unwrap();

        ff.merge(&other).unwrap();
        assert_eq!(ff.styles().len(), 2);
        assert_eq!(
            ff.get_style("pair", "coul/cut").unwrap().params(),
            &coulomb,
            "self's statement is kept"
        );

        let mut geometric = Params::new();
        geometric.set("mixing", "geometric");
        let mut other = ForceField::new("source");
        other.def_style("pair", "lj/cut", geometric).unwrap();
        let merged = ff.merge(&other);
        assert!(
            matches!(merged, Err(DefError::StyleConflict { .. })),
            "{merged:?}"
        );
    }

    /// Two sources store one value at their own precision: a prmtop's θ₀ is
    /// `gaff.dat`'s degrees through tleap's π = 3.141594, 4.3e-7 above them,
    /// and its torsion k eight significant digits of `PK / IDIVF`. Within
    /// `MERGE_REL_TOL` they are one value and self's is kept; beyond it they
    /// are two.
    #[test]
    fn merge_takes_numbers_within_its_tolerance_for_one_value() {
        let angle =
            |theta0: f64| Params::from_iter([("k", 46.3), ("theta0", theta0)].iter().copied());
        let torsion = |k: f64| {
            Params::from_iter(
                [("k1", k), ("periodicity1", 2.0), ("phase1", 180.0)]
                    .iter()
                    .copied(),
            )
        };
        let define = |theta0: f64, k: f64| {
            let mut ff = ForceField::new("ff");
            ff.def_style("angle", "harmonic", Params::new())
                .unwrap()
                .def_type("c3-c3-hc", &["c3", "c3", "hc"], angle(theta0))
                .unwrap();
            ff.def_style("dihedral", "periodic", Params::new())
                .unwrap()
                .def_type("c3-s6-ne-sy", &["c3", "s6", "ne", "sy"], torsion(k))
                .unwrap();
            ff
        };
        let mut ff = define(109.80, 20.0 / 3.0);
        ff.merge(&define(109.80004705760963, 6.66666667)).unwrap();
        assert_eq!(ff.get_types("angle").len(), 1);
        assert_eq!(
            ff.get_types("angle")[0].params,
            angle(109.80),
            "self's kept"
        );
        assert_eq!(ff.get_types("dihedral").len(), 1);

        let merged = ff.merge(&define(109.81, 20.0 / 3.0));
        assert!(
            matches!(merged, Err(DefError::TypeConflict { .. })),
            "{merged:?}"
        );
    }

    /// What the style's spec does not declare prices nothing, and a merge
    /// does not compare it: a prmtop atom type's LJ-class `id`, an estimate's
    /// provenance, an annotation. A declared parameter one lacks is a
    /// difference.
    #[test]
    fn merge_compares_only_what_the_spec_declares() {
        let mut ff = ForceField::new("native");
        let mut estimated = Params::from_iter([("k", 63.1), ("theta0", 109.68)].iter().copied());
        estimated.set("estimated", 1.0);
        estimated.set("estimate_method", "analogy");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type(
                "c3",
                &[],
                Params::from_iter([("mass", 12.01)].iter().copied()),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type("f-c3-sy", &["f", "c3", "sy"], estimated.clone())
            .unwrap();
        let mut prmtop = ForceField::new("prmtop");
        let mut with_id = Params::from_iter([("id", 4.0), ("mass", 12.01)].iter().copied());
        with_id.set("desc", "sp3 carbon");
        prmtop
            .def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("c3", &[], with_id)
            .unwrap();
        prmtop
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "f-c3-sy",
                &["f", "c3", "sy"],
                Params::from_iter(
                    [("k", 63.1), ("theta0", 109.68004677699743)]
                        .iter()
                        .copied(),
                ),
            )
            .unwrap();

        ff.merge(&prmtop).unwrap();
        assert_eq!(ff.get_types("atom").len(), 1);
        assert_eq!(ff.get_types("angle")[0].params, estimated, "self's kept");

        let mut no_theta = ForceField::new("partial");
        no_theta
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "f-c3-sy",
                &["f", "c3", "sy"],
                Params::from_iter([("k", 63.1)].iter().copied()),
            )
            .unwrap();
        let merged = ff.merge(&no_theta);
        assert!(
            matches!(merged, Err(DefError::TypeConflict { .. })),
            "{merged:?}"
        );
    }

    #[test]
    fn merge_into_undeclared_special_bonds_adopts_the_others() {
        let mut ff = ForceField::new("target");
        let mut other = ForceField::new("source");
        other.set_special_bonds(AMBER_SB);

        ff.merge(&other).unwrap();

        assert_eq!(ff.special_bonds(), Some(&AMBER_SB));
    }

    #[test]
    fn merge_into_undeclared_units_adopts_the_others() {
        let mut ff = ForceField::new("target");
        let mut other = ForceField::new("source");
        other.set_units("lj");

        ff.merge(&other).unwrap();

        assert_eq!(ff.units(), Some("lj"));
        assert_eq!(ff.units(), Some("lj"));
    }

    /// An undeclared `other` leaves `self`'s declarations alone, and an
    /// undeclared pair stays undeclared.
    #[test]
    fn merge_of_undeclared_state_keeps_self_declarations() {
        let mut declared = ForceField::new("target");
        declared.set_units("metal");
        declared.set_special_bonds(OPLS_SB);
        declared.merge(&ForceField::new("source")).unwrap();
        assert_eq!(declared.units(), Some("metal"));
        assert_eq!(declared.special_bonds(), Some(&OPLS_SB));

        let mut undeclared = ForceField::new("target");
        undeclared.merge(&ForceField::new("source")).unwrap();
        assert_eq!(undeclared.units(), None);
        assert_eq!(undeclared.special_bonds(), None);
    }

    /// MMFF-style and OPLS-style 1-4 rules must refuse to combine.
    #[test]
    fn merge_with_a_different_declared_special_bonds_is_a_special_bonds_conflict() {
        let mut ff = ForceField::new("target");
        ff.set_special_bonds(OPLS_SB);
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other.set_special_bonds(AMBER_SB);

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::SpecialBondsConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }

    #[test]
    fn merge_with_a_different_declared_units_is_a_units_conflict() {
        let mut ff = ForceField::new("target");
        ff.set_units("real");
        let before = ff.clone();
        let mut other = ForceField::new("source");
        other.set_units("lj");

        let merged = ff.merge(&other);

        assert!(
            matches!(merged, Err(DefError::UnitsConflict { .. })),
            "{merged:?}"
        );
        assert_same_definitions(&ff, &before);
    }
}
