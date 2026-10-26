//! The style registry: which kernel prices each style of the force-field IR
//! ([`crate::ff::ir`]) — categories, styles, the kernels that price them,
//! their form codecs and engine forms, in one table open to any caller that
//! conforms.
//!
//! The IR states the shape of a category and a style as data; this module
//! binds that shape to the kernels of [`crate::ff::potential`], so it sits
//! above both, and the force field ([`crate::ff::forcefield`]) and the
//! compiler ([`crate::ff::compile`]) read it:
//!
//! * a style is registered as a [`StyleSpec`] and a [`Kernel`] in one of
//!   three tiers: an expression ([`ExpressionKernel`]), a batch form of one
//!   coordinate or of the atoms' positions
//!   ([`ScalarForm`], [`CompoundForm`], built
//!   into the form kernels of [`crate::ff::potential::form_kernel`] — every
//!   built-in whose energy is such a function, including the real-space
//!   erfc of `pair coul/long/pme`), or a constructor that builds the rest
//!   of a kernel (PME's reciprocal-space sum, `cmap charmm`'s splines;
//!   `dihedral rb` is a built-in priced by its expression alone);
//! * the [`Registry`] refuses anything that does not conform
//!   ([`conformance`], [`IrError`]) and seals the built-ins;
//! * [`compile_expression`] compiles a style's Lepton `expression` (the
//!   engine of [`crate::ff::ir::expression`]) into its kernel, with exact
//!   derivatives — installed in every registry [`Registry::builtin`] makes.
//!
//! ```
//! use std::sync::Arc;
//! use molrs::ff::ir::{ParamDimension, ParamSpec, StyleSpec};
//! use molrs::ff::potential::form_kernel::{ParamColumns, ScalarForm};
//! use molrs::ff::style_registry::{Kernel, Registry};
//!
//! /// LAMMPS `bond_style harmonic`, as a third party would write it.
//! struct Harmonic;
//! impl ScalarForm for Harmonic {
//!     fn eval(&self, r: &[f64], p: &ParamColumns<'_>, e: &mut [f64], de_dr: &mut [f64]) {
//!         let (k, r0) = (p.get("k").unwrap(), p.get("r0").unwrap());
//!         for t in 0..r.len() {
//!             e[t] = k[t] * (r[t] - r0[t]).powi(2);
//!             de_dr[t] = 2.0 * k[t] * (r[t] - r0[t]);
//!         }
//!     }
//! }
//!
//! let mut registry = Registry::builtin();
//! let spec = StyleSpec::new("bond", "my_harmonic").params(vec![
//!     ParamSpec::new("k", "E/L^2".parse().unwrap()),
//!     ParamSpec::new("r0", ParamDimension::LENGTH),
//! ]);
//! registry
//!     .register_style(spec, Some(Kernel::Scalar(Arc::new(Harmonic))))
//!     .unwrap();
//! ```

mod batch;
#[cfg(test)]
mod builtin_conformance;
mod builtin_kernels;
mod compiled_expression;
pub mod conformance;

pub use batch::{
    BatchCoordinates, BatchError, BatchForm, BatchGradient, BatchParamKind, BatchValue,
};
pub(crate) use builtin_kernels::BuiltinKernels;
pub(crate) use compiled_expression::fallback_spec;
pub use compiled_expression::{CompiledExpression, compile_expression};

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock, RwLock};

use crate::ff::ir::torsion::{FAMILY, FourierSeries, canonical_series};
use crate::ff::ir::{CategorySpec, IrError, StyleSpec, builtin_categories, builtin_styles};
use crate::ff::ir::{Engine, LammpsCodec, LammpsForm};
use crate::ff::ir::{FormCodec, ParamSource, Params, SpecialClass, TypeParams};
use crate::ff::potential::form_kernel::{
    CompoundForm, CompoundTerms, Probe, ScalarBonded, ScalarForm, ScalarPair,
};
use crate::ff::potential::pair::CandidatePairs;
use crate::ff::potential::{ForceTerm, SpecialWeights};
use conformance::{PROBE_TERMS, form_id};
use molrs::core::Frame;
use molrs::op::F;

// ---------------------------------------------------------------------------
// Registration vocabulary
// ---------------------------------------------------------------------------

/// Builds a molecule-bound [`ForceTerm`] from a style's params, its per-type
/// params (`(type_label, params)`), a typed [`Frame`], and the electrostatic
/// weights of the frame's special pairs (what a long-range sum corrects its
/// 1-2, 1-3 and 1-4 pairs by). Every kernel constructor in the crate matches
/// this signature.
pub type KernelConstructor = fn(
    &Params,
    &[(&str, &Params)],
    &Frame,
    &SpecialWeights,
) -> Result<ForceTerm, crate::ff::potential::CompileError>;

/// What a pair style's compiled kernel is built from beside the frame: the
/// pairs it prices and the electrostatic weights of the special pairs.
pub(crate) struct PairContext<'a> {
    pub(crate) candidates: &'a CandidatePairs,
    pub(crate) coulomb: &'a SpecialWeights,
}

/// Which `Frame` block decides whether a style has any rows to act on.
///
/// `PotentialCompiler::compile` skips a style whose topology is absent — a bond style
/// with no bonds contributes nothing, and letting the kernel fault on the
/// missing block instead would be a worse way to say so. Which block that is,
/// is a property of the **kernel**, not of its category.
///
/// PME is the case that proves it: registered under `pair` because that is
/// where an electrostatic style belongs, it reads per-atom charges and
/// finds its own pairs (the minimum-image loop, or the neighbour table); the
/// special pairs only weigh them. Its row source is the atoms: gated on a
/// topology block, a system without one would lose the entire long-range
/// electrostatics, silently, to exactly zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RowSource {
    /// The category's own topology block: `bonds`, `angles`, `dihedrals` or
    /// `impropers`; absent or empty means the style contributes nothing. For
    /// a pair style, the candidate pairs (every pair of the atoms, or a
    /// neighbour table's), weighed by the frame's special pairs.
    #[default]
    CategoryBlock,
    /// The atoms, or rows the kernel finds for itself. Nothing gates it.
    Atoms,
}

/// A style's energy written as an expression, compiled.
///
/// The seam between the registry and the expression engine
/// (`ff::ir::expression`, whose `Compiled` implements it): the registry never
/// parses an expression, it asks this trait. The registry checks it like
/// any other kernel — its variables against the style's declarations
/// ([`variables`](Self::variables)), its derivative against a central
/// difference ([`form`](Self::form)) — and builds it into the same generic
/// kernels a native form gets.
pub trait ExpressionKernel: Send + Sync + 'static {
    /// The expression exactly as declared (molrec: kept byte for byte).
    fn source(&self) -> &str;

    /// Every free name the expression reads that its own `;` definitions do
    /// not bind and that is not a point (`p1` … in `distance(p1, p2)`).
    /// Registration refuses ([`IrError::UnboundVariable`]) one that is not
    /// a variable of the category ([`Coordinate::variables`]), a declared
    /// numeric per-type or style parameter (an indexed family by any member
    /// `k1`, `k2`, …), or on a pair `q1`, `q2`, `<param>1`, `<param>2`.
    ///
    /// [`Coordinate::variables`]: crate::ff::ir::Coordinate::variables
    fn variables(&self) -> Vec<String>;

    /// The kernel the expression evaluates as: a [`ScalarForm`] of the
    /// category's coordinate, or a [`CompoundForm`] of the points (a
    /// `Compound` category, or any use of a point function). A pair form
    /// reads `x1`/`x2` (self rows) from columns of those names, which the
    /// pair kernel does not supply to Tier 2: an expression engine that
    /// binds them resolves them itself.
    fn form(&self) -> ExpressionForm;
}

/// What an [`ExpressionKernel`] evaluates as.
#[derive(Clone)]
pub enum ExpressionForm {
    Scalar(Arc<dyn ScalarForm>),
    Compound(Arc<dyn CompoundForm>),
}

/// Compiles a style's [`expression`](StyleSpec::expression) for its
/// category. The expression engine installs one
/// ([`Registry::set_expression_compiler`]); without one, an expression-only
/// style is priced by nothing ([`IrError::NoKernel`]) and an expression
/// beside a native kernel is not checked.
pub type ExpressionCompiler =
    fn(&CategorySpec, &StyleSpec) -> Result<Arc<dyn ExpressionKernel>, IrError>;

/// What prices a style: the three kernel tiers.
#[derive(Clone)]
pub enum Kernel {
    /// Tier 1: the energy as an expression, compiled.
    Expression(Arc<dyn ExpressionKernel>),
    /// Tier 2: a batch function of the category's coordinate, built into
    /// [`ScalarBonded`] (a bonded category) or [`ScalarPair`] (a pair
    /// category).
    Scalar(Arc<dyn ScalarForm>),
    /// Tier 2: a batch N-body function of positions, built into
    /// [`CompoundTerms`].
    Compound(Arc<dyn CompoundForm>),
    /// Tier 3: a constructor that builds the whole [`ForceTerm`] itself.
    /// PME's reciprocal-space sum and CMAP's splined grids are built here.
    /// PME's real-space erfc term is the pair form beside the reciprocal sum.
    Constructor {
        compiled: KernelConstructor,
        /// The neighbour-driven form of a pair style, and the special-bonds
        /// weights that scale it.
        typed: Option<(KernelConstructor, SpecialClass)>,
        /// Which block's emptiness means the style contributes nothing.
        rows: RowSource,
    },
}

impl Kernel {
    /// A Tier-3 constructor with no neighbour-driven form, gated on its
    /// category's block.
    pub fn constructor(compiled: KernelConstructor) -> Self {
        Kernel::Constructor {
            compiled,
            typed: None,
            rows: RowSource::CategoryBlock,
        }
    }

    /// The tier, for messages.
    pub(crate) fn tier(&self) -> &'static str {
        match self {
            Kernel::Expression(_) => "expression",
            Kernel::Scalar(_) => "scalar form",
            Kernel::Compound(_) => "compound form",
            Kernel::Constructor { .. } => "constructor",
        }
    }

    /// Whether `self` and `other` are the same kernel: the same object, or
    /// the same constructors with the same declarations.
    fn same(&self, other: &Kernel) -> bool {
        fn thin<T: ?Sized>(p: &Arc<T>) -> *const () {
            Arc::as_ptr(p) as *const ()
        }
        match (self, other) {
            (Kernel::Expression(a), Kernel::Expression(b)) => thin(a) == thin(b),
            (Kernel::Scalar(a), Kernel::Scalar(b)) => thin(a) == thin(b),
            (Kernel::Compound(a), Kernel::Compound(b)) => thin(a) == thin(b),
            (
                Kernel::Constructor {
                    compiled: a,
                    typed: ta,
                    rows: ra,
                },
                Kernel::Constructor {
                    compiled: b,
                    typed: tb,
                    rows: rb,
                },
            ) => {
                std::ptr::fn_addr_eq(*a, *b)
                    && ra == rb
                    && match (ta, tb) {
                        (None, None) => true,
                        (Some((a, ca)), Some((b, cb))) => std::ptr::fn_addr_eq(*a, *b) && ca == cb,
                        _ => false,
                    }
            }
            _ => false,
        }
    }
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kernel::{}", self.tier())
    }
}

/// A built-in form's kernel built monomorphized on the form's type: the
/// same form kernel ([`ScalarPair`], [`ScalarBonded`], [`CompoundTerms`]) a
/// [`Kernel::Scalar`] / [`Kernel::Compound`] is built into, its force law
/// inlined (a pair style [`fused`](ScalarPair::fused) into one pass), where
/// the registered `dyn` form is called through a vtable a batch at a time.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Native {
    compiled: NativeBuild,
    /// The neighbour-driven form of a pair style.
    typed: Option<NativeTyped>,
}

type NativeBuild = fn(
    &CategorySpec,
    &StyleSpec,
    &Params,
    &[(&str, &Params)],
    &Frame,
    Option<&PairContext<'_>>,
) -> Result<ForceTerm, crate::ff::potential::CompileError>;

type NativeTyped = fn(
    &CategorySpec,
    &StyleSpec,
    &Params,
    &[(&str, &Params)],
    &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError>;

/// The candidate pairs of a pair style's compile, or the refusal that it was
/// compiled without them.
fn candidates_of<'a>(
    spec: &StyleSpec,
    pairs: Option<&'a PairContext<'_>>,
) -> Result<&'a CandidatePairs, crate::ff::potential::CompileError> {
    pairs.map(|p| p.candidates).ok_or_else(|| {
        format!(
            "pair style '{}' compiled without the pairs it prices",
            spec.name
        )
        .into()
    })
}

impl Native {
    /// The kernels of the scalar form `Fm`.
    pub(crate) fn scalar<Fm: ScalarForm + Default>() -> Self {
        Self {
            compiled: |category, spec, style, tp, frame, pairs| {
                let form = Arc::new(Fm::default());
                Ok(if category.is_pair_driven() {
                    let candidates = candidates_of(spec, pairs)?;
                    ForceTerm::pair(
                        ScalarPair::compiled(form, spec, style, tp, frame, candidates)?.fused(),
                    )
                } else {
                    ForceTerm::indexed(ScalarBonded::build(form, category, spec, style, tp, frame)?)
                })
            },
            typed: Some(|_, spec, style, tp, frame| {
                let form = Arc::new(Fm::default());
                Ok(ForceTerm::pair(
                    ScalarPair::typed(form, spec, style, tp, frame)?.fused(),
                ))
            }),
        }
    }

    /// The kernel of the compound form `Fm`.
    pub(crate) fn compound<Fm: CompoundForm + Default>() -> Self {
        Self {
            compiled: |category, spec, style, tp, frame, _| {
                let form = Arc::new(Fm::default());
                Ok(ForceTerm::indexed(CompoundTerms::build(
                    form, category, spec, style, tp, frame,
                )?))
            },
            typed: None,
        }
    }
}

/// One registered style: its spec and its kernel (none for a style priced
/// by its expression alone, or in a category that prices nothing).
#[derive(Clone, Debug)]
pub(crate) struct RegisteredStyle {
    spec: StyleSpec,
    kernel: Option<Kernel>,
    /// A sealed built-in form's kernel monomorphized on its type, which
    /// prices it in place of `kernel`'s `dyn` form.
    native: Option<Native>,
    sealed: bool,
    /// What the first-compile verdict is keyed on instead of the form's
    /// address: the registry expression an instance expression is checked
    /// against, whose compiled form is new at every compile.
    probe_id: Option<usize>,
}

/// What a style's form kernel is, and where it came from.
struct Form {
    form: ExpressionForm,
    /// A native form (Tier 2), which an expression beside it must agree
    /// with.
    native: bool,
}

impl RegisteredStyle {
    pub fn spec(&self) -> &StyleSpec {
        &self.spec
    }

    /// The entry an **unregistered** style that carries an `expression` is
    /// priced under: the compile fallback (protocol §4), through the
    /// installed expression engine.
    pub(crate) fn fallback(
        category: &CategorySpec,
        name: &str,
        style: &Params,
        tp: &[(&str, &Params)],
        expressions: Option<ExpressionCompiler>,
    ) -> Option<Result<RegisteredStyle, IrError>> {
        let expression = style.get("expression").and_then(|v| v.as_text())?;
        let spec = fallback_spec(category, name, style, tp, expression);
        let Some(compile) = expressions else {
            return Some(Err(IrError::NoKernel {
                category: category.name.to_string(),
                style: name.to_owned(),
            }));
        };
        Some(compile(category, &spec).map(|x| RegisteredStyle {
            spec,
            kernel: Some(Kernel::Expression(x)),
            native: None,
            sealed: false,
            probe_id: None,
        }))
    }

    /// The entry a style instance carrying its own `expression` is priced
    /// under (protocol §4, D16).
    ///
    /// The registered kernel prices it, a registered kernel taking priority
    /// over an expression; an instance expression that differs from the
    /// registry's is checked for agreement with that kernel at first compile
    /// ([`IrError::Disagree`]), like an expression registered beside a native
    /// kernel. A registered style with neither kernel nor expression is
    /// priced by the instance's. A sealed built-in, and a Tier-3
    /// constructor (never sampled), is priced as registered, unchecked.
    pub(crate) fn with_instance_expression(
        &self,
        category: &CategorySpec,
        instance: Option<&str>,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<std::borrow::Cow<'_, RegisteredStyle>, IrError> {
        use std::borrow::Cow;
        let Some(instance) = instance else {
            return Ok(Cow::Borrowed(self));
        };
        if self.sealed || self.spec.expression.as_deref() == Some(instance) {
            return Ok(Cow::Borrowed(self));
        }
        let native = |form: ExpressionForm| match form {
            ExpressionForm::Scalar(f) => Kernel::Scalar(f),
            ExpressionForm::Compound(f) => Kernel::Compound(f),
        };
        let keyed = |source: &str| {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            source.hash(&mut h);
            Some(h.finish() as usize)
        };
        let (kernel, probe_id) = match (&self.kernel, &self.spec.expression) {
            (Some(Kernel::Constructor { .. }), _) => return Ok(Cow::Borrowed(self)),
            (Some(k @ (Kernel::Scalar(_) | Kernel::Compound(_))), _) => (Some(k.clone()), None),
            (Some(Kernel::Expression(x)), _) => (Some(native(x.form())), keyed(x.source())),
            (None, Some(registered)) => match expressions {
                Some(compile) => (
                    Some(native(compile(category, &self.spec)?.form())),
                    keyed(registered),
                ),
                None => return Ok(Cow::Borrowed(self)),
            },
            (None, None) => (None, None),
        };
        let mut spec = self.spec.clone();
        spec.expression = Some(instance.to_owned());
        spec.samples.clear();
        Ok(Cow::Owned(RegisteredStyle {
            spec,
            kernel,
            native: None,
            sealed: false,
            probe_id,
        }))
    }

    /// Where the style's rows come from.
    pub fn row_source(&self) -> RowSource {
        match self.kernel {
            Some(Kernel::Constructor { rows, .. }) => rows,
            _ => RowSource::CategoryBlock,
        }
    }

    /// The form a Tier-1 or Tier-2 kernel evaluates as; for an
    /// expression-only style, its expression compiled now.
    fn form(
        &self,
        category: &CategorySpec,
        expressions: Option<ExpressionCompiler>,
    ) -> Result<Form, IrError> {
        match &self.kernel {
            Some(Kernel::Expression(x)) => Ok(Form {
                form: x.form(),
                native: false,
            }),
            Some(Kernel::Scalar(f)) => Ok(Form {
                form: ExpressionForm::Scalar(f.clone()),
                native: true,
            }),
            Some(Kernel::Compound(f)) => Ok(Form {
                form: ExpressionForm::Compound(f.clone()),
                native: true,
            }),
            Some(Kernel::Constructor { .. }) => unreachable!("a constructor is no form"),
            None => match (expressions, &self.spec.expression) {
                (Some(compile), Some(_)) => Ok(Form {
                    form: compile(category, &self.spec)?.form(),
                    native: false,
                }),
                _ => Err(IrError::NoKernel {
                    category: self.spec.category.to_string(),
                    style: self.spec.name.to_string(),
                }),
            },
        }
    }

    /// Run the first-compile conformance check of a form kernel. A sealed
    /// built-in is not sampled at run time: its gate is
    /// `style_registry::builtin_conformance` (every built-in against its
    /// Appendix-A expression on seeded configurations, both compile doors),
    /// which holds it on more points than one compile's terms would.
    fn first_compile(
        &self,
        category: &CategorySpec,
        form: &Form,
        expressions: Option<ExpressionCompiler>,
        probe: impl FnOnce() -> Probe,
    ) -> Result<(), IrError> {
        if self.sealed {
            return Ok(());
        }
        conformance::check_at_first_compile(
            category,
            &self.spec,
            &form.form,
            self.probe_id.unwrap_or_else(|| form_id(&form.form)),
            form.native,
            expressions,
            probe,
        )
    }

    /// Build the style's kernel for a **compiled** evaluation: its terms
    /// resolved against `frame` (a pair style against the pairs of `pairs`).
    pub(crate) fn compiled(
        &self,
        category: &CategorySpec,
        params: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
        expressions: Option<ExpressionCompiler>,
        pairs: Option<&PairContext<'_>>,
    ) -> Result<ForceTerm, crate::ff::potential::CompileError> {
        let (params, rows) = self.spec.gather(params, tp)?;
        let (params, tp) = (&params, &borrowed(&rows));
        if let Some(Kernel::Constructor { compiled, .. }) = &self.kernel {
            let none = SpecialWeights::default();
            return compiled(params, tp, frame, pairs.map_or(&none, |p| p.coulomb));
        }
        if let (true, Some(native)) = (self.sealed, &self.native) {
            return (native.compiled)(category, &self.spec, params, tp, frame, pairs);
        }
        let form = self.form(category, expressions)?;
        let coords = frame_coords(frame);
        let spec = &self.spec;
        Ok(match &form.form {
            ExpressionForm::Scalar(f) if category.is_pair_driven() => {
                let candidates = candidates_of(spec, pairs)?;
                let k = ScalarPair::compiled(f.clone(), spec, params, tp, frame, candidates)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                ForceTerm::pair(k)
            }
            ExpressionForm::Scalar(f) => {
                let k = ScalarBonded::build(f.clone(), category, spec, params, tp, frame)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                ForceTerm::indexed(k)
            }
            ExpressionForm::Compound(f) => {
                let k = CompoundTerms::build(f.clone(), category, spec, params, tp, frame)?;
                self.first_compile(category, &form, expressions, || {
                    probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
                })?;
                ForceTerm::indexed(k)
            }
        })
    }

    /// Build a pair style's kernel for a **neighbour-driven** evaluation,
    /// with the special-bonds weights that scale it; `None` when the style
    /// has no such form.
    pub(crate) fn typed(
        &self,
        category: &CategorySpec,
        params: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
        expressions: Option<ExpressionCompiler>,
        coulomb: &SpecialWeights,
    ) -> Result<Option<(ForceTerm, SpecialClass)>, crate::ff::potential::CompileError> {
        let (params, rows) = self.spec.gather(params, tp)?;
        let (params, tp) = (&params, &borrowed(&rows));
        match &self.kernel {
            Some(Kernel::Constructor { typed: None, .. }) => return Ok(None),
            Some(Kernel::Constructor {
                typed: Some((constructor, class)),
                ..
            }) => return Ok(Some((constructor(params, tp, frame, coulomb)?, *class))),
            _ => {}
        }
        if let (true, Some(native)) = (self.sealed, &self.native) {
            return Ok(match native.typed {
                Some(typed) => Some((
                    typed(category, &self.spec, params, tp, frame)?,
                    self.spec.special_class(),
                )),
                None => None,
            });
        }
        let form = self.form(category, expressions)?;
        let ExpressionForm::Scalar(f) = &form.form else {
            return Ok(None);
        };
        let k = ScalarPair::typed(f.clone(), &self.spec, params, tp, frame)?;
        let coords = frame_coords(frame);
        self.first_compile(category, &form, expressions, || {
            probe_on(&coords, |c| k.probe(c, PROBE_TERMS))
        })?;
        Ok(Some((ForceTerm::pair(k), self.spec.special_class())))
    }
}

/// `rows` as the `(label, row)` slice a kernel takes.
fn borrowed(rows: &[(String, Params)]) -> Vec<(&str, &Params)> {
    rows.iter().map(|(l, r)| (l.as_str(), r)).collect()
}

/// The probe of a kernel's real terms at `coords`, or none (the check
/// waits for a compile whose frame has positions) when the frame has none.
fn probe_on(coords: &[F], probe: impl FnOnce(&[F]) -> Probe) -> Probe {
    if coords.is_empty() {
        Probe::default()
    } else {
        probe(coords)
    }
}

/// The atoms' coordinates of `frame`, flat; empty without them.
fn frame_coords(frame: &Frame) -> Vec<F> {
    let Some(atoms) = frame.get("atoms") else {
        return Vec::new();
    };
    let axis = |k: &str| atoms.get(k).and_then(|c| c.get::<f64>());
    let (Some(x), Some(y), Some(z)) = (axis("x"), axis("y"), axis("z")) else {
        return Vec::new();
    };
    x.iter()
        .zip(y.iter())
        .zip(z.iter())
        .flat_map(|((x, y), z)| [*x, *y, *z])
        .collect()
}

#[derive(Clone, Debug)]
struct RegisteredCategory {
    spec: CategorySpec,
    sealed: bool,
}

/// One registered form codec.
#[derive(Clone)]
struct RegisteredForm {
    codec: FormCodec,
    sealed: bool,
}

/// Categories and styles of the force-field IR, the kernels that price the
/// styles, and the form codecs that convert between them
/// ([`crate::ff::ir::FormCodec`]).
///
/// [`Registry::builtin`] holds molrs's own categories and styles, registered
/// through the same spec types a third party uses and then **sealed**: a
/// built-in cannot be overridden or removed ([`IrError::Sealed`]).
/// Re-registering anything identically is a no-op; registering something
/// else under a taken name is [`IrError::Conflict`], mirroring the force
/// field's own conflict rule.
///
/// A force field is read against the process-wide registry
/// ([`GlobalRegistry`]) unless it carries its
/// own ([`ForceField::with_registry`](crate::ff::forcefield::ForceField::with_registry)),
/// which is how a test extends the IR without touching anything another test
/// sees: [`ForceField::def_style`](crate::ff::forcefield::ForceField::def_style),
/// the [`PotentialCompiler`](crate::ff::compile::PotentialCompiler) and the
/// force-field writers all read the force field's registry.
#[derive(Clone, Default)]
pub struct Registry {
    categories: BTreeMap<String, RegisteredCategory>,
    styles: BTreeMap<(String, String), RegisteredStyle>,
    forms: BTreeMap<(String, String), RegisteredForm>,
    expressions: Option<ExpressionCompiler>,
}

impl fmt::Debug for Registry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Registry")
            .field("categories", &self.categories.len())
            .field("styles", &self.styles.len())
            .field("forms", &self.forms.len())
            .finish()
    }
}

impl Registry {
    /// An empty registry: no categories, no styles.
    pub fn new() -> Self {
        Self::default()
    }

    /// Every built-in category and style, sealed.
    ///
    /// The kernels are the built-in kernels (`BuiltinKernels`: forms, and
    /// the constructors for PME's reciprocal sum and CMAP's splines), each
    /// with its spec from
    /// [`builtin_styles`]; a spec without a kernel is priced by its
    /// expression or prices nothing (`dihedral rb`, `drude harmonic`, `atom
    /// full`, …). A test holds the two tables to one set of names.
    pub fn builtin() -> Self {
        let mut r = Self::new();
        r.set_expression_compiler(Some(compile_expression));
        // The built-in categories are molrec's table, which the custom rules
        // (`register_category`) are not: `atom` and `virtual_site` name no
        // endpoints, a pair's block is its atoms.
        for c in builtin_categories() {
            r.categories.insert(
                c.name.to_string(),
                RegisteredCategory {
                    spec: c,
                    sealed: true,
                },
            );
        }
        let builtin_kernels = BuiltinKernels::builtin();
        for spec in builtin_styles() {
            let kernel = builtin_kernels.kernel(&spec.category, &spec.name);
            let native = builtin_kernels.native(&spec.category, &spec.name);
            let (category, name) = (spec.category.clone(), spec.name.clone());
            r.register_style(spec, kernel)
                .unwrap_or_else(|e| panic!("built-in {category} `{name}`: {e}"));
            if let Some(entry) = r.styles.get_mut(&(category.to_string(), name.to_string())) {
                entry.native = native;
            }
        }
        for (category, style, codec) in crate::ff::ir::form::builtin_forms() {
            r.register_form(category, style, codec)
                .unwrap_or_else(|e| panic!("built-in form of {category} `{style}`: {e}"));
        }
        for s in r.styles.values_mut() {
            s.sealed = true;
        }
        for f in r.forms.values_mut() {
            f.sealed = true;
        }
        r
    }

    /// Register a category added at run time ([`CategorySpec::custom`]).
    ///
    /// Refuses a malformed name, an arity outside 2..=5, a block other than
    /// `<name>s`, a coordinate its arity cannot carry, and a name a
    /// different category holds (a built-in with [`IrError::Sealed`]); an
    /// identical re-registration is a no-op.
    pub fn register_category(&mut self, c: CategorySpec) -> Result<(), IrError> {
        if let Some(existing) = self.categories.get(c.name.as_ref()) {
            if existing.spec == c {
                return Ok(());
            }
            let category = c.name.into_owned();
            let style = String::new();
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        conformance::check_category(&c)?;
        self.categories.insert(
            c.name.to_string(),
            RegisteredCategory {
                spec: c,
                sealed: false,
            },
        );
        Ok(())
    }

    /// Register a style and its kernel, after checking that both conform
    /// ([`conformance`]). `kernel` `None`
    /// registers a style priced by its `expression` alone (Tier 1 through
    /// the installed expression engine), or a style of a category that
    /// prices nothing.
    ///
    /// An identical re-registration (equal spec, the same kernel) is a
    /// no-op; anything else under a taken name is [`IrError::Sealed`] for a
    /// built-in and [`IrError::Conflict`] otherwise.
    pub fn register_style(
        &mut self,
        spec: StyleSpec,
        kernel: Option<Kernel>,
    ) -> Result<(), IrError> {
        let key = (spec.category.to_string(), spec.name.to_string());
        if let Some(existing) = self.styles.get(&key) {
            let same_kernel = match (&existing.kernel, &kernel) {
                (None, None) => true,
                (Some(a), Some(b)) => a.same(b),
                _ => false,
            };
            if existing.spec == spec && same_kernel {
                return Ok(());
            }
            let (category, style) = key;
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        let category = self
            .category(&spec.category)
            .ok_or_else(|| IrError::UnknownCategory {
                category: spec.category.to_string(),
            })?;
        conformance::check_style(category, &spec, kernel.as_ref(), self.expressions)?;
        spec.lammps.check_spec(&spec)?;
        self.styles.insert(
            key,
            RegisteredStyle {
                spec,
                kernel,
                native: None,
                sealed: false,
                probe_id: None,
            },
        );
        Ok(())
    }

    /// Remove a style registered at run time. A built-in is
    /// [`IrError::Sealed`]; a style that is not registered,
    /// [`IrError::NoKernel`].
    pub fn unregister_style(&mut self, category: &str, name: &str) -> Result<(), IrError> {
        let key = (category.to_owned(), name.to_owned());
        match self.styles.get(&key) {
            None => Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            }),
            Some(e) if e.sealed => Err(IrError::Sealed {
                category: key.0,
                style: key.1,
            }),
            Some(_) => {
                self.forms.remove(&key);
                self.styles.remove(&key);
                Ok(())
            }
        }
    }

    /// Register the form codec of a registered style: its family, and its
    /// exact maps to and from the family's canonical parameters
    /// (`ff-ir-02-protocol` §9).
    ///
    /// Refuses a style that is not registered ([`IrError::NoKernel`]), a
    /// second canonical style of one family ([`IrError::FormConflict`]), and
    /// a different codec under a style that has one ([`IrError::Sealed`]
    /// for a built-in, [`IrError::Conflict`] otherwise); the same codec again
    /// is a no-op.
    pub fn register_form(
        &mut self,
        category: &str,
        style: &str,
        codec: FormCodec,
    ) -> Result<(), IrError> {
        let key = (category.to_owned(), style.to_owned());
        if !self.styles.contains_key(&key) {
            return Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            });
        }
        if let Some(existing) = self.forms.get(&key) {
            if existing.codec.same(&codec) {
                return Ok(());
            }
            let (category, style) = key;
            return Err(if existing.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        if codec.canonical
            && let Some(((c, s), _)) = self
                .forms
                .iter()
                .find(|(_, f)| f.codec.canonical && f.codec.family == codec.family)
        {
            return Err(IrError::FormConflict {
                family: codec.family.to_string(),
                reason: format!(
                    "{c} `{s}` is its canonical style already; {category} `{style}` cannot be \
                     a second one"
                ),
            });
        }
        self.forms.insert(
            key,
            RegisteredForm {
                codec,
                sealed: false,
            },
        );
        Ok(())
    }

    /// Give a registered style the engine form `form` of `engine`
    /// (`ff-ir-02-protocol` §8): a LAMMPS form for a style registered
    /// without one.
    ///
    /// Only LAMMPS takes a registered form: the OpenMM XML writer writes an
    /// expression style's `Custom*Force` from its expression, and GROMACS
    /// and AMBER hold the built-in styles only ([`IrError::NoEngineForm`]).
    /// A style that is not registered is [`IrError::NoKernel`]; the same
    /// form again is a no-op; another form under a style that has one is
    /// [`IrError::Sealed`] for a built-in and [`IrError::Conflict`]
    /// otherwise; a positional form the spec cannot have is
    /// [`IrError::NoEngineForm`].
    pub fn register_engine_form(
        &mut self,
        engine: Engine,
        category: &str,
        style: &str,
        form: LammpsForm,
    ) -> Result<(), IrError> {
        let key = (category.to_owned(), style.to_owned());
        let Some(entry) = self.styles.get_mut(&key) else {
            return Err(IrError::NoKernel {
                category: key.0,
                style: key.1,
            });
        };
        if engine != Engine::Lammps {
            return Err(engine.refuse(
                category,
                style,
                "only LAMMPS takes a registered form: the OpenMM XML writer writes an expression \
                 style's Custom*Force from its expression, and GROMACS and AMBER hold the \
                 built-in styles only",
            ));
        }
        if entry.spec.lammps == form {
            return Ok(());
        }
        if !matches!(entry.spec.lammps, LammpsForm::None) || entry.sealed {
            let (category, style) = key;
            return Err(if entry.sealed {
                IrError::Sealed { category, style }
            } else {
                IrError::Conflict { category, style }
            });
        }
        form.check_spec(&entry.spec)?;
        entry.spec.lammps = form;
        Ok(())
    }

    /// The style of `category` a LAMMPS file's style name `lammps` reads
    /// as, with its codec: the style of that name when its form writes the
    /// name, else the one style whose form does (`fourier` is `dihedral
    /// periodic`). `None` when no style, or more than one, claims it.
    pub fn lammps_style(
        &self,
        category: &str,
        lammps: &str,
    ) -> Option<(&StyleSpec, &dyn LammpsCodec)> {
        let writes =
            |e: &&RegisteredStyle| e.spec.lammps.lammps_name(&e.spec).as_deref() == Some(lammps);
        let found = match self.registered_style(category, lammps).filter(writes) {
            Some(e) => e,
            None => {
                let mut claims = self
                    .styles
                    .values()
                    .filter(|e| e.spec.category == category)
                    .filter(writes);
                let first = claims.next()?;
                if claims.next().is_some() {
                    return None;
                }
                first
            }
        };
        Some((&found.spec, found.spec.lammps.codec()?))
    }

    /// The form codec of `(category, style)`, if it registered one.
    pub fn form(&self, category: &str, style: &str) -> Option<&FormCodec> {
        self.forms
            .get(&(category.to_owned(), style.to_owned()))
            .map(|f| &f.codec)
    }

    /// Every registered form codec, by `(category, style)`.
    pub fn forms(&self) -> impl Iterator<Item = (&str, &str, &FormCodec)> + '_ {
        self.forms
            .iter()
            .map(|((c, s), f)| (c.as_str(), s.as_str(), &f.codec))
    }

    /// The canonical style `(category, style)` of `family`;
    /// [`IrError::FormConflict`] when it has none.
    pub fn canonical_form(&self, family: &str) -> Result<(&str, &str), IrError> {
        self.forms()
            .find(|(_, _, f)| f.canonical && f.family == family)
            .map(|(c, s, _)| (c, s))
            .ok_or_else(|| IrError::FormConflict {
                family: family.to_owned(),
                reason: "no style is its canonical one".into(),
            })
    }

    /// Install the expression engine: it prices expression-only styles and
    /// checks an expression declared beside a native kernel.
    pub fn set_expression_compiler(&mut self, compiler: Option<ExpressionCompiler>) {
        self.expressions = compiler;
    }

    /// The installed expression engine, if any.
    pub fn expression_compiler(&self) -> Option<ExpressionCompiler> {
        self.expressions
    }

    pub fn category(&self, name: &str) -> Option<&CategorySpec> {
        self.categories.get(name).map(|c| &c.spec)
    }

    /// Every category, by name.
    pub fn categories(&self) -> impl Iterator<Item = &CategorySpec> + '_ {
        self.categories.values().map(|c| &c.spec)
    }

    /// A registered style's spec and kernel.
    pub fn style(&self, category: &str, name: &str) -> Option<(&StyleSpec, Option<&Kernel>)> {
        self.registered_style(category, name)
            .map(|e| (&e.spec, e.kernel.as_ref()))
    }

    /// Every style, or those of `category`, by `(category, name)`.
    pub fn styles<'a>(
        &'a self,
        category: Option<&'a str>,
    ) -> impl Iterator<Item = (&'a StyleSpec, Option<&'a Kernel>)> + 'a {
        self.styles
            .values()
            .filter(move |e| category.is_none_or(|c| e.spec.category == c))
            .map(|e| (&e.spec, e.kernel.as_ref()))
    }

    /// Whether `(category, name)` is a sealed built-in.
    pub fn is_sealed(&self, category: &str, name: &str) -> bool {
        self.registered_style(category, name)
            .is_some_and(|e| e.sealed)
    }

    pub(crate) fn registered_style(&self, category: &str, name: &str) -> Option<&RegisteredStyle> {
        self.styles.get(&(category.to_owned(), name.to_owned()))
    }

    /// The [`ParamSource`] of a registered style.
    pub fn param_source(&self, category: &str, name: &str) -> Option<ParamSource> {
        self.registered_style(category, name).map(|e| e.spec.source)
    }

    /// Where a registered style's rows come from.
    pub fn row_source(&self, category: &str, name: &str) -> Option<RowSource> {
        self.registered_style(category, name)
            .map(RegisteredStyle::row_source)
    }
}

/// The process-wide registry, [`Registry::builtin`] on first use.
fn global() -> &'static RwLock<Registry> {
    static REGISTRY: OnceLock<RwLock<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(Registry::builtin()))
}

/// The process-wide registry: what every compile, reader and writer reads
/// unless given a registry of its own ([`ForceField::with_registry`],
/// [`RegistryRef::Own`]). A handle that holds nothing: each method locks
/// the registry for its call.
///
/// [`ForceField::with_registry`]: crate::ff::forcefield::ForceField::with_registry
#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalRegistry;

impl GlobalRegistry {
    /// [`Registry::register_category`].
    pub fn register_category(&self, c: CategorySpec) -> Result<(), IrError> {
        global().write().unwrap().register_category(c)
    }

    /// [`Registry::register_style`]: the extension point every compile
    /// reads, with nothing rebuilt.
    pub fn register_style(&self, spec: StyleSpec, kernel: Option<Kernel>) -> Result<(), IrError> {
        global().write().unwrap().register_style(spec, kernel)
    }

    /// [`Registry::register_form`].
    pub fn register_form(
        &self,
        category: &str,
        style: &str,
        codec: FormCodec,
    ) -> Result<(), IrError> {
        global()
            .write()
            .unwrap()
            .register_form(category, style, codec)
    }

    /// [`Registry::register_engine_form`].
    pub fn register_engine_form(
        &self,
        engine: Engine,
        category: &str,
        style: &str,
        form: LammpsForm,
    ) -> Result<(), IrError> {
        global()
            .write()
            .unwrap()
            .register_engine_form(engine, category, style, form)
    }

    /// [`Registry::unregister_style`].
    pub fn unregister_style(&self, category: &str, name: &str) -> Result<(), IrError> {
        global().write().unwrap().unregister_style(category, name)
    }

    /// [`Registry::set_expression_compiler`].
    pub fn set_expression_compiler(&self, compiler: Option<ExpressionCompiler>) {
        global().write().unwrap().set_expression_compiler(compiler);
    }

    /// Read the registry. The lock is held for the call, so `f` must not
    /// register anything.
    pub fn with<R>(&self, f: impl FnOnce(&Registry) -> R) -> R {
        f(&global().read().unwrap())
    }
}

/// The registry an engine reader or writer reads: the process-wide one
/// (the default), or a caller's own — a test's styles, seen by nothing
/// else.
#[derive(Clone, Debug, Default)]
pub enum RegistryRef {
    #[default]
    Global,
    Own(Arc<Registry>),
}

impl RegistryRef {
    /// Run `f` on the registry: a snapshot of the process-wide one (no lock
    /// is held while `f` runs, so `f` may read the process-wide registry
    /// itself), or the caller's.
    pub fn with<R>(&self, f: impl FnOnce(&Registry) -> R) -> R {
        match self {
            RegistryRef::Global => f(&GlobalRegistry.with(Registry::clone)),
            RegistryRef::Own(r) => f(r),
        }
    }
}

/// The exact series of a row of any style of the `torsion` family, through
/// its registered form codec (a built-in's, or a third party's): the row
/// embedded in the canonical `dihedral periodic` parameters, summed.
///
/// # Errors
///
/// The style registers no form of the `torsion` family, or its embedding
/// refuses the row (a charmm `w ≠ 0`, a non-integer periodicity, …).
pub fn torsion_series(
    reg: &Registry,
    category: &str,
    style: &str,
    style_params: &Params,
    row: &Params,
) -> std::result::Result<FourierSeries, String> {
    let what = format!("{category} {style}");
    let codec = reg
        .form(category, style)
        .filter(|c| c.family == FAMILY)
        .ok_or_else(|| format!("{what} has no form of the `{FAMILY}` family"))?;
    let canonical = (codec.embed)(&TypeParams {
        style: style_params.clone(),
        row: row.clone(),
    })
    .map_err(|e| format!("{what}: {e}"))?;
    canonical_series(&canonical).map_err(|e| format!("{what}: {e}"))
}

/// The refusal of a style this engine's writer has no form for, among
/// the formats that hold built-in styles only (GROMACS, AMBER): a
/// built-in without one, or any style that is not built in.
pub fn refuse_style(engine: Engine, category: &str, style: &str) -> IrError {
    let builtin = GlobalRegistry.with(|r| r.is_sealed(category, style));
    let reason = if builtin {
        format!(
            "a built-in style no {} directive or section holds",
            engine.name()
        )
    } else {
        format!(
            "it is not a built-in style: {} holds the built-in styles it has directives \
             for, and a style registered at run time (or an expression's) has none",
            engine.name()
        )
    };
    engine.refuse(category, style, reason)
}
