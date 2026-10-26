//! Form codecs: the exact maps between the styles of one physics
//! (`ff-ir-01` P3, on the protocol of `ff-ir-02` §9).
//!
//! A **form family** is a set of styles whose energies are one function
//! space parametrised differently — the torsions (Fourier series in φ), the
//! harmonic bonds and angles, the Lennard-Jones pairs. Each member registers a
//! [`FormCodec`] beside its kernel
//! ([`Registry::register_form`](crate::ff::style_registry::Registry::register_form)):
//! its family, and two exact maps between its own parameters and the family's
//! **canonical** style's:
//!
//! * `embed` — this style → canonical (exact: the same energy, constant
//!   included), refusing a row the canonical style cannot hold;
//! * `project` — canonical → this style, exact on its image, refusing with a
//!   [`FormRefusal`] that names the condition otherwise (a Fourier series with
//!   `bₙ ≠ 0` has no RB form);
//! * optionally `seed` — canonical → the nearest member of this style's
//!   image by a rule of thumb, where a fit starts.
//!
//! The codecs are vocabulary; converting a force field with them is
//! [`crate::ff::form_conversion`]'s.
//!
//! The built-in families ([`builtin_forms`]):
//! | family | canonical | members |
//! |---|---|---|
//! | `torsion` | `dihedral periodic` | `dihedral` `charmm` (w = 0), `opls`, `multi/harmonic`, `nharmonic`, `harmonic`, `class2`, `rb`; `improper` `cvff`, `periodic` |
//! | `bond` | `bond harmonic` | `bond class2` (k3 = k4 = 0) |
//! | `angle` | `angle harmonic` | `angle class2` (k3 = k4 = 0), `angle charmm` (k_ub = 0) |
//! | `lj` | `pair lj/cut` | `pair lj/class2` (= lj/cut n = 9, m = 6, σ' = (2/3)^⅓ σ) |
//!
//! No codec, by design: `improper harmonic` and `bond`/`pair` `morse` are no
//! member of any family's function space (fit them with `fit_form`);
//! `pair lj/charmm` always switches between `inner < cutoff` (its kernel
//! refuses `inner = cutoff`), so it is never exactly `lj/cut`.

mod builtin;
pub mod torsion;

use std::borrow::Cow;
use std::fmt;
use std::sync::Arc;

pub use builtin::builtin_forms;

use crate::ff::ir::{Params, StyleSpec};

/// One type's full parameter set: its style's parameters and its own row.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TypeParams {
    /// The style parameters (`cutoff`, `mixing`, …).
    pub style: Params,
    /// The type's row.
    pub row: Params,
}

impl TypeParams {
    pub fn new(style: Params, row: Params) -> Self {
        Self { style, row }
    }

    /// A row of a style without style parameters.
    pub fn row(row: Params) -> Self {
        Self {
            style: Params::new(),
            row,
        }
    }
}

/// Why a form map refused a row: the condition, in words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormRefusal {
    pub reason: String,
}

impl FormRefusal {
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
        }
    }
}

impl fmt::Display for FormRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)
    }
}

impl std::error::Error for FormRefusal {}

/// A map between one style's parameters and its family's canonical ones.
pub type FormFn = Arc<dyn Fn(&TypeParams) -> Result<TypeParams, FormRefusal> + Send + Sync>;

/// A style's membership of a form family (`ff-ir-02-protocol` §9).
#[derive(Clone)]
pub struct FormCodec {
    /// The family, e.g. `torsion`.
    pub family: Cow<'static, str>,
    /// Whether this is the family's canonical style (exactly one is).
    pub canonical: bool,
    /// This style → the canonical style's parameters, exactly (the same
    /// energy, constant included), or a refusal naming the condition. On the
    /// canonical style itself: its canonical spelling.
    pub embed: FormFn,
    /// Canonical parameters → this style's, exact on its image, else a
    /// refusal naming the condition.
    pub project: FormFn,
    /// Canonical parameters → the nearest member of this style's image by a
    /// rule of thumb (the dominant order of a one-term torsion): the start of
    /// [`ForceField::fit_form`](crate::ff::forcefield::ForceField::fit_form). `None`: the fit starts from the source row's
    /// same-named values.
    pub seed: Option<FormFn>,
}

impl FormCodec {
    /// A member of `family` (not its canonical style) with exact maps
    /// `embed` and `project`.
    pub fn new(
        family: impl Into<Cow<'static, str>>,
        embed: impl Fn(&TypeParams) -> Result<TypeParams, FormRefusal> + Send + Sync + 'static,
        project: impl Fn(&TypeParams) -> Result<TypeParams, FormRefusal> + Send + Sync + 'static,
    ) -> Self {
        Self {
            family: family.into(),
            canonical: false,
            embed: Arc::new(embed),
            project: Arc::new(project),
            seed: None,
        }
    }

    /// Mark this as its family's canonical style.
    pub fn as_canonical(mut self) -> Self {
        self.canonical = true;
        self
    }

    /// Give the fit a starting point.
    pub fn seed(
        mut self,
        seed: impl Fn(&TypeParams) -> Result<TypeParams, FormRefusal> + Send + Sync + 'static,
    ) -> Self {
        self.seed = Some(Arc::new(seed));
        self
    }

    /// Whether `self` and `other` are the same registration: the same family
    /// and role, the same map objects.
    pub(crate) fn same(&self, other: &FormCodec) -> bool {
        fn thin(f: &FormFn) -> *const () {
            Arc::as_ptr(f) as *const ()
        }
        self.family == other.family
            && self.canonical == other.canonical
            && thin(&self.embed) == thin(&other.embed)
            && thin(&self.project) == thin(&other.project)
            && match (&self.seed, &other.seed) {
                (None, None) => true,
                (Some(a), Some(b)) => thin(a) == thin(b),
                _ => false,
            }
    }
}

impl fmt::Debug for FormCodec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FormCodec")
            .field("family", &self.family)
            .field("canonical", &self.canonical)
            .field("seed", &self.seed.is_some())
            .finish()
    }
}

/// Whether `key` is one of `spec`'s per-type parameters: declared, a member
/// `<name><m>` of a declared indexed family, or a `<name>` of an indexed
/// family the style also spells unindexed.
pub(crate) fn declared(spec: &StyleSpec, key: &str) -> bool {
    if spec.param(key).is_some() {
        return true;
    }
    let stem = key.trim_end_matches(|c: char| c.is_ascii_digit());
    stem.len() < key.len() && spec.param(stem).is_some_and(|p| p.indexed)
}
