//! What "conforming to the force-field IR" means, checked at registration
//! and at a style's first compile (`ff-ir-02-protocol` §5, D14).
//!
//! Static checks, at registration, refuse with an [`IrError`] naming the
//! offending item:
//!
//! * a custom category: its name, an arity outside 2..=5, a block that is
//!   not `<name>s`, a coordinate its arity cannot carry;
//! * a style: a parameter name outside `^[A-Za-z_][A-Za-z0-9_]*$` or
//!   reserved (a structural column, a variable, a point, a pair input; not
//!   a function name, since a call always has its parenthesis), declared twice, of a forbidden [`ParamDimension`]; a reserved style
//!   parameter of the wrong kind; a mixing rule or special class off a pair;
//!   a kernel tier the category cannot take; an expression reading what the
//!   style does not declare.
//!
//! Numeric checks evaluate the style's form ([`ScalarForm`], [`CompoundForm`],
//! or an expression's) — on its registration [`ConformanceSample`]s, or, without any,
//! once per process at its first compile on up to [`PROBE_TERMS`] real
//! terms:
//!
//! * `dE/dq` (`∂E/∂x`) against a central difference of `E`, relative
//!   [`DERIVATIVE_RTOL`] ([`IrError::Derivative`]);
//! * a pair's energy under exchanging `q1` and `q2`, relative
//!   [`SYMMETRY_RTOL`] ([`IrError::Asymmetric`]);
//! * an expression beside a native form, with an expression engine
//!   installed: energy (and derivative, unless the style's force is not its
//!   gradient) relative [`AGREEMENT_RTOL`] ([`IrError::Disagree`]).
//!
//! Relative errors are measured against `max(|value|, RMS of the values)`.
//! A Tier-3 constructor builds a whole kernel from a Frame and is never
//! sampled; its declarations are checked like any other.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use crate::ff::ir::Params;
use crate::ff::ir::{
    Arity, CategorySpec, ConformanceSample, Coordinate, IrError, ParamCombination, ParamDimension,
    ParamKind, StyleSpec,
};
use crate::ff::ir::{ENDPOINT_COLUMNS, is_unpriced_column};
use crate::ff::potential::form_kernel::{CompoundForm, Probe, ScalarForm, TermParams, columns};
use crate::ff::style_registry::Kernel;
use crate::ff::style_registry::{ExpressionCompiler, ExpressionForm};
use molrs::op::F;

/// `dE/dq` against a central difference.
pub const DERIVATIVE_RTOL: F = 1e-6;
/// An expression against the native kernel beside it.
pub const AGREEMENT_RTOL: F = 1e-10;
/// A pair energy under exchanging its two atoms.
pub const SYMMETRY_RTOL: F = 1e-12;
/// Seeded points per registration sample.
pub const SAMPLE_POINTS: usize = 16;
/// Real terms a first-compile check evaluates.
pub const PROBE_TERMS: usize = 8;
/// The seed of every sample.
pub(crate) const SEED: u64 = 0x1F1F_0002;

/// `^[a-z][a-z0-9_]*$`, molrec's category name.
fn is_category_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes.next().is_some_and(|b| b.is_ascii_lowercase())
        && bytes.all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
}

/// `^[A-Za-z_][A-Za-z0-9_]*$`, a Lepton identifier.
fn is_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    bytes
        .next()
        .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

/// A category registered at run time ([`CategorySpec::custom`]'s rules).
pub(crate) fn check_category(c: &CategorySpec) -> Result<(), IrError> {
    if !is_category_name(&c.name) {
        return Err(IrError::BadName {
            what: "category (^[a-z][a-z0-9_]*$)",
            name: c.name.to_string(),
        });
    }
    let arity = c.arity.endpoints();
    if !matches!(c.arity, Arity::Exact(2..=5)) {
        return Err(IrError::Arity {
            category: c.name.to_string(),
            arity,
        });
    }
    if c.block != format!("{}s", c.name) {
        return Err(IrError::BlockName {
            category: c.name.to_string(),
            block: c.block.to_string(),
        });
    }
    let fits = match c.coordinate {
        Coordinate::Compound => true,
        Coordinate::None => false,
        scalar => scalar.atoms() == Some(arity),
    };
    if !fits || c.excludes {
        return Err(IrError::CoordinateMismatch {
            category: c.name.to_string(),
            kernel: format!("{:?} coordinate over {arity} atoms", c.coordinate),
        });
    }
    Ok(())
}

/// The registration checks of a style ([module](self)). `kernel` is `None`
/// for a style priced by its expression alone (or, in a category that
/// prices nothing, by nothing).
pub(crate) fn check_style(
    category: &CategorySpec,
    spec: &StyleSpec,
    kernel: Option<&Kernel>,
    expressions: Option<ExpressionCompiler>,
) -> Result<(), IrError> {
    let style = spec.name.to_string();
    let malformed = |reason: String| IrError::Malformed {
        style: style.clone(),
        reason,
    };
    if spec.name.is_empty() {
        return Err(IrError::BadName {
            what: "style",
            name: String::new(),
        });
    }
    let pair = category.is_pair_driven();
    check_param_names(spec, pair)?;
    for p in &spec.style_params {
        let reserved = |ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(IrError::ReservedParam {
                    style: style.clone(),
                    param: p.name.to_string(),
                })
            }
        };
        match p.name.as_ref() {
            "cutoff" => reserved(p.dim == ParamDimension::LENGTH && p.kind == ParamKind::Scalar)?,
            "mixing" | "special" => reserved(matches!(p.kind, ParamKind::Text { .. }))?,
            _ => {}
        }
        if p.mix != ParamCombination::None || p.indexed {
            return Err(malformed(format!(
                "style param `{}` is one value for the style: it neither mixes nor is indexed",
                p.name
            )));
        }
    }
    for p in &spec.params {
        if p.indexed && p.kind != ParamKind::Scalar {
            return Err(malformed(format!("indexed `{}` must be numeric", p.name)));
        }
        match &p.mix {
            ParamCombination::None => {}
            _ if !pair => {
                return Err(malformed(format!(
                    "`{}` declares a mixing rule, and a {} row is no pair",
                    p.name, category.name
                )));
            }
            ParamCombination::LjEpsilon { sigma } => {
                if !spec.param(sigma).is_some_and(|s| {
                    s.mix
                        == ParamCombination::LjSigma {
                            epsilon: p.name.clone(),
                        }
                }) {
                    return Err(malformed(format!(
                        "`{}` mixes with `{sigma}`, which must be its `LjSigma`",
                        p.name
                    )));
                }
            }
            ParamCombination::LjSigma { epsilon } => {
                if !spec.param(epsilon).is_some_and(|e| {
                    e.mix
                        == ParamCombination::LjEpsilon {
                            sigma: p.name.clone(),
                        }
                }) {
                    return Err(malformed(format!(
                        "`{}` mixes with `{epsilon}`, which must be its `LjEpsilon`",
                        p.name
                    )));
                }
            }
            ParamCombination::Arithmetic | ParamCombination::Geometric => {}
        }
    }
    if spec.unindexed_one_term && !spec.params.iter().any(|p| p.indexed) {
        return Err(malformed(
            "`unindexed_one_term` needs an indexed parameter".into(),
        ));
    }
    if !pair && spec.special.is_some() {
        return Err(malformed(format!(
            "a special-bonds class belongs to a pair style, not a {}",
            category.name
        )));
    }
    let mismatch = |kernel: &str| IrError::CoordinateMismatch {
        category: category.name.to_string(),
        kernel: kernel.to_owned(),
    };
    if !category.prices_energy() {
        return match kernel {
            Some(k) => Err(mismatch(k.tier())),
            None => Ok(()),
        };
    }
    let form = match kernel {
        None => {
            // Tier 1 by the spec's expression alone.
            if spec.expression.is_none() {
                return Err(IrError::NoKernel {
                    category: category.name.to_string(),
                    style,
                });
            }
            match expressions {
                Some(compile) => {
                    let x = compile(category, spec)?;
                    check_variables(category, spec, x.variables())?;
                    x.form()
                }
                // Nothing can be checked about it until an engine is
                // installed; it is priced by none until then either.
                None => return Ok(()),
            }
        }
        Some(Kernel::Constructor { typed, .. }) => {
            return match typed {
                Some(_) if !pair => Err(mismatch("neighbour-driven constructor")),
                Some((_, class)) if *class != spec.special_class() => Err(malformed(format!(
                    "the neighbour-driven form is scaled by {class:?}, the spec says {:?}",
                    spec.special_class()
                ))),
                _ => Ok(()),
            };
        }
        Some(Kernel::Expression(x)) => {
            if spec.expression.as_deref().is_some_and(|e| e != x.source()) {
                return Err(malformed(
                    "the compiled expression is not the spec's `expression`".into(),
                ));
            }
            check_variables(category, spec, x.variables())?;
            x.form()
        }
        Some(Kernel::Scalar(f)) => ExpressionForm::Scalar(f.clone()),
        Some(Kernel::Compound(f)) => ExpressionForm::Compound(f.clone()),
    };
    match &form {
        ExpressionForm::Scalar(_) if !category.coordinate.is_scalar() => {
            return Err(mismatch("scalar form"));
        }
        ExpressionForm::Compound(_) if pair => return Err(mismatch("compound form")),
        _ => {}
    }
    if pair && let Some(p) = spec.params.iter().find(|p| p.kind != ParamKind::Scalar) {
        return Err(malformed(format!(
            "a pair form takes numeric per-type parameters; `{}` is not",
            p.name
        )));
    }
    let native = !matches!(kernel, None | Some(Kernel::Expression(_)));
    for sample in &spec.samples {
        let probe = Probe::seeded(category, spec, sample)?;
        check_probe(category, spec, &form, native, expressions, &probe)?;
    }
    Ok(())
}

/// Reserved and duplicated parameter names, across per-type and style
/// parameters alike (an expression reads both by name), and their [`ParamDimension`]s.
fn check_param_names(spec: &StyleSpec, pair: bool) -> Result<(), IrError> {
    let style = spec.name.to_string();
    let per_type: HashSet<&str> = spec.params.iter().map(|p| p.name.as_ref()).collect();
    let mut seen = HashSet::new();
    for p in spec.params.iter().chain(&spec.style_params) {
        let name = p.name.as_ref();
        if !is_identifier(name) {
            return Err(IrError::BadName {
                what: "parameter (^[A-Za-z_][A-Za-z0-9_]*$)",
                name: name.to_owned(),
            });
        }
        let point = name
            .strip_prefix('p')
            .is_some_and(|n| matches!(n, "1" | "2" | "3" | "4" | "5"));
        let pair_input = pair
            && (matches!(name, "q" | "q1" | "q2")
                || name
                    .strip_suffix(['1', '2'])
                    .is_some_and(|base| per_type.contains(base)));
        if matches!(
            name,
            "name" | "type" | "style" | "r" | "theta" | "phi" | "chi"
        ) || ENDPOINT_COLUMNS.contains(&name)
            || molrs::core::keys::ENDPOINTS.contains(&name)
            || is_unpriced_column(name)
            || point
            || pair_input
        {
            return Err(IrError::ReservedParam {
                style: style.clone(),
                param: name.to_owned(),
            });
        }
        if !seen.insert(name) {
            return Err(IrError::DuplicateParam {
                style: style.clone(),
                param: name.to_owned(),
            });
        }
        p.dim.check().map_err(|reason| IrError::Dimension {
            param: name.to_owned(),
            dim: p.dim.to_string(),
            reason,
        })?;
    }
    Ok(())
}

/// Every variable an expression reads must be its category's, a declared
/// numeric parameter, or — on a pair — `q1`, `q2`, `<param>1`, `<param>2`.
fn check_variables(
    category: &CategorySpec,
    spec: &StyleSpec,
    variables: Vec<String>,
) -> Result<(), IrError> {
    let numeric = spec
        .params
        .iter()
        .chain(&spec.style_params)
        .filter(|p| p.kind == ParamKind::Scalar);
    let mut allowed: HashSet<String> = HashSet::new();
    for p in numeric {
        allowed.insert(p.name.to_string());
        if p.indexed {
            // `k1 … kM`: any member of the family.
            allowed.insert(format!("{}#", p.name));
        }
    }
    allowed.extend(
        category
            .coordinate
            .variables()
            .iter()
            .map(|v| v.to_string()),
    );
    if category.is_pair_driven() {
        allowed.extend(["q1".to_owned(), "q2".to_owned()]);
        for p in &spec.params {
            allowed.insert(format!("{}1", p.name));
            allowed.insert(format!("{}2", p.name));
        }
    }
    let member_of_family = |v: &str| {
        let base = v.trim_end_matches(|c: char| c.is_ascii_digit());
        base.len() < v.len() && allowed.contains(&format!("{base}#"))
    };
    match variables
        .into_iter()
        .find(|v| !allowed.contains(v) && !member_of_family(v))
    {
        Some(name) => Err(IrError::UnboundVariable {
            style: spec.name.to_string(),
            name,
        }),
        None => Ok(()),
    }
}

/// A seeded SplitMix64: the sample points are the same on every machine
/// and every run, so a refusal is reproducible.
pub(crate) struct Rng(pub(crate) u64);

impl Rng {
    pub(crate) fn uniform(&mut self, lo: F, hi: F) -> F {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        lo + (hi - lo) * ((z >> 11) as F / (1u64 << 53) as F)
    }
}

impl Probe {
    /// [`SAMPLE_POINTS`] seeded points of one registration sample.
    fn seeded(
        category: &CategorySpec,
        spec: &StyleSpec,
        sample: &ConformanceSample,
    ) -> Result<Self, IrError> {
        let n = SAMPLE_POINTS;
        let mut rng = Rng(SEED);
        let mut row = Params::new();
        let mut style = Params::new();
        let declared = |name: &str| spec.style_param(name).is_some();
        for (name, value) in &sample.params {
            let target = if declared(name) { &mut style } else { &mut row };
            target.set(name, value.clone());
        }
        let (style, gathered) = spec.gather(&style, &[("sample", &row)])?;
        let row = &gathered[0].1;
        let rows = [("sample", row)];
        let mut params = TermParams::default();
        let missing = |param: &str| IrError::MissingParam {
            style: spec.name.to_string(),
            type_: "sample".into(),
            param: param.to_owned(),
        };
        for col in columns(spec, &spec.params, &rows)? {
            let decl = &spec.params[col.param];
            match &decl.kind {
                ParamKind::Scalar => {
                    let v = crate::ff::potential::form_kernel::row_num(spec, &col, row)
                        .ok_or_else(|| missing(&col.name))?;
                    params.nums.push((col.name, vec![v; n]));
                }
                ParamKind::Text { .. } => {
                    let v = row
                        .get(&col.name)
                        .and_then(|v| v.as_text())
                        .ok_or_else(|| missing(&col.name))?;
                    params.texts.push((col.name, vec![v.to_owned(); n]));
                }
                ParamKind::Array { .. } => return Err(missing(&col.name)),
            }
        }
        params.add_style(spec, &style, n)?;
        let (lo, hi) = sample.q;
        if category.is_pair_driven() {
            for name in ["q1", "q2"] {
                params.nums.push((
                    name.into(),
                    (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect(),
                ));
            }
            // The two atoms' self rows, near the pair value and unequal, so
            // an expression that reads them is checked for symmetry.
            let pair_values: Vec<(String, F)> = spec
                .params
                .iter()
                .filter_map(|p| {
                    let col = params.nums.iter().find(|(n, _)| n == p.name.as_ref())?;
                    Some((p.name.to_string(), col.1[0]))
                })
                .collect();
            for (name, v) in pair_values {
                for end in ["1", "2"] {
                    params.nums.push((
                        format!("{name}{end}"),
                        (0..n).map(|_| v * rng.uniform(0.8, 1.2)).collect(),
                    ));
                }
            }
        }
        let arity = category.arity.endpoints();
        let mut probe = Probe {
            params,
            q: Vec::new(),
            x: Vec::new(),
            arity,
        };
        if category.coordinate == Coordinate::Compound {
            // The term's atoms as a random walk whose steps are `q` long:
            // never coincident, rarely collinear.
            for _ in 0..n {
                let mut at = [0.0; 3];
                for _ in 0..arity {
                    probe.x.push(at);
                    let dir = [
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                    ];
                    let norm = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2])
                        .sqrt()
                        .max(1e-3);
                    let step = rng.uniform(lo, hi) / norm;
                    for d in 0..3 {
                        at[d] += step * dir[d];
                    }
                }
            }
        } else {
            probe.q = (0..n).map(|_| rng.uniform(lo, hi)).collect();
        }
        Ok(probe)
    }
}

fn scalar_eval(f: &dyn ScalarForm, probe: &Probe, q: &[F]) -> (Vec<F>, Vec<F>) {
    let mut e = vec![0.0; q.len()];
    let mut de = vec![0.0; q.len()];
    probe
        .params
        .with_cols(0..q.len(), |p| f.eval(q, p, &mut e, &mut de));
    (e, de)
}

fn compound_eval(f: &dyn CompoundForm, probe: &Probe, x: &[[F; 3]]) -> (Vec<F>, Vec<[F; 3]>) {
    let n = probe.n();
    let mut e = vec![0.0; n];
    let mut grad = vec![[0.0; 3]; x.len()];
    probe
        .params
        .with_cols(0..n, |p| f.eval(x, probe.arity, p, &mut e, &mut grad));
    (e, grad)
}

/// `max(|v|, RMS of the finite values)`, the scale a relative error is
/// measured against.
fn rms(values: &[F]) -> F {
    let finite: Vec<F> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return 0.0;
    }
    (finite.iter().map(|v| v * v).sum::<F>() / finite.len() as F).sqrt()
}

fn relative(a: F, b: F, rms: F) -> F {
    let scale = a.abs().max(b.abs()).max(rms);
    if scale == 0.0 {
        0.0
    } else {
        (a - b).abs() / scale
    }
}

/// Whether `q` lies within `h` of an edge of the coordinate's domain, where
/// a central difference straddles it.
fn near_edge(coordinate: Coordinate, q: F, h: F) -> bool {
    use std::f64::consts::PI;
    match coordinate {
        Coordinate::Distance => q < h,
        Coordinate::Angle => q < h || q > PI - h,
        Coordinate::Dihedral | Coordinate::Improper => q.abs() > PI - h || q.abs() < h,
        Coordinate::None | Coordinate::Compound => false,
    }
}

/// The numeric checks of `form` at `probe` ([module](self)).
pub(crate) fn check_probe(
    category: &CategorySpec,
    spec: &StyleSpec,
    form: &ExpressionForm,
    native: bool,
    expressions: Option<ExpressionCompiler>,
    probe: &Probe,
) -> Result<(), IrError> {
    let style = spec.name.to_string();
    match form {
        ExpressionForm::Scalar(f) => {
            let h: Vec<F> = probe.q.iter().map(|q| 1e-5 * q.abs().max(1.0)).collect();
            let up: Vec<F> = probe.q.iter().zip(&h).map(|(q, h)| q + h).collect();
            let down: Vec<F> = probe.q.iter().zip(&h).map(|(q, h)| q - h).collect();
            let (e, de) = scalar_eval(&**f, probe, &probe.q);
            let (e_up, _) = scalar_eval(&**f, probe, &up);
            let (e_down, _) = scalar_eval(&**f, probe, &down);
            let fd: Vec<F> = (0..e.len())
                .map(|k| (e_up[k] - e_down[k]) / (2.0 * h[k]))
                .collect();
            let scale = rms(&de);
            for k in 0..e.len() {
                if near_edge(category.coordinate, probe.q[k], h[k])
                    || ![e[k], e_up[k], e_down[k], de[k]]
                        .iter()
                        .all(|v| v.is_finite())
                {
                    continue;
                }
                let rel = relative(de[k], fd[k], scale);
                if rel > DERIVATIVE_RTOL {
                    return Err(IrError::Derivative {
                        style,
                        at: format!("q = {}", probe.q[k]),
                        rel,
                    });
                }
            }
            if category.is_pair_driven() {
                check_symmetry(&**f, probe, &e, &style)?;
            }
        }
        ExpressionForm::Compound(f) => {
            let (e, grad) = compound_eval(&**f, probe, &probe.x);
            let h = 1e-5;
            let flat: Vec<F> = grad.iter().flatten().copied().collect();
            let scale = rms(&flat);
            for a in 0..probe.arity {
                for d in 0..3 {
                    let shifted = |by: F| {
                        let mut x = probe.x.clone();
                        for t in 0..probe.n() {
                            x[t * probe.arity + a][d] += by;
                        }
                        compound_eval(&**f, probe, &x).0
                    };
                    let (up, down) = (shifted(h), shifted(-h));
                    for t in 0..probe.n() {
                        let g = grad[t * probe.arity + a][d];
                        if ![e[t], up[t], down[t], g].iter().all(|v| v.is_finite()) {
                            continue;
                        }
                        let rel = relative(g, (up[t] - down[t]) / (2.0 * h), scale);
                        if rel > DERIVATIVE_RTOL {
                            return Err(IrError::Derivative {
                                style,
                                at: format!("term {t}, atom {a}, axis {d}"),
                                rel,
                            });
                        }
                    }
                }
            }
        }
    }
    if let (true, Some(_), Some(compile)) = (native, &spec.expression, expressions) {
        let x = compile(category, spec)?;
        check_variables(category, spec, x.variables())?;
        check_agreement(spec, form, &x.form(), probe)?;
    }
    Ok(())
}

/// A pair's energy with `q1` and `q2` exchanged (Tier 2 sees no other
/// per-atom input).
fn check_symmetry(f: &dyn ScalarForm, probe: &Probe, e: &[F], style: &str) -> Result<(), IrError> {
    // Every per-atom column `<x>1` exchanged with its `<x>2` (`q1` ↔ `q2`,
    // `epsilon1` ↔ `epsilon2`): `expression::Input::swapped`, by spelling.
    let mut swapped = probe.clone();
    let lookup: HashMap<String, Vec<F>> = probe.params.nums.iter().cloned().collect();
    for (name, col) in &mut swapped.params.nums {
        let partner = match name.as_bytes().last() {
            Some(b'1') => format!("{}2", &name[..name.len() - 1]),
            Some(b'2') => format!("{}1", &name[..name.len() - 1]),
            _ => continue,
        };
        if let Some(other) = lookup.get(&partner) {
            *col = other.clone();
        }
    }
    let (e2, _) = scalar_eval(f, &swapped, &probe.q);
    let scale = rms(e);
    for (a, b) in e.iter().zip(&e2) {
        if a.is_finite() && b.is_finite() && relative(*a, *b, scale) > SYMMETRY_RTOL {
            return Err(IrError::Asymmetric {
                style: style.to_owned(),
            });
        }
    }
    Ok(())
}

/// `native` and `other` price the same energy — and derivative, when the
/// style's force is the gradient of its energy.
fn check_agreement(
    spec: &StyleSpec,
    native: &ExpressionForm,
    other: &ExpressionForm,
    probe: &Probe,
) -> Result<(), IrError> {
    let (energies, derivatives): ([Vec<F>; 2], [Vec<F>; 2]) = match (native, other) {
        (ExpressionForm::Scalar(n), ExpressionForm::Scalar(o)) => {
            let (en, dn) = scalar_eval(&**n, probe, &probe.q);
            let (eo, d_o) = scalar_eval(&**o, probe, &probe.q);
            ([en, eo], [dn, d_o])
        }
        (ExpressionForm::Compound(n), ExpressionForm::Compound(o)) => {
            let (en, gn) = compound_eval(&**n, probe, &probe.x);
            let (eo, go) = compound_eval(&**o, probe, &probe.x);
            let flat = |g: Vec<[F; 3]>| g.into_iter().flatten().collect();
            ([en, eo], [flat(gn), flat(go)])
        }
        _ => {
            return Err(IrError::Malformed {
                style: spec.name.to_string(),
                reason: "the expression and the kernel are not the same tier of form".into(),
            });
        }
    };
    let mut sets = vec![("energy", energies)];
    if spec.force_is_gradient {
        sets.push(("derivative", derivatives));
    }
    for (what, [a, b]) in sets {
        let scale = rms(&a);
        for (k, (x, y)) in a.iter().zip(&b).enumerate() {
            if !x.is_finite() && !y.is_finite() {
                continue;
            }
            let rel = relative(*x, *y, scale);
            if rel.is_nan() || rel > AGREEMENT_RTOL {
                return Err(IrError::Disagree {
                    style: spec.name.to_string(),
                    at: format!("point {k} ({what})"),
                    rel,
                });
            }
        }
    }
    Ok(())
}

/// The outcome of each style's first-compile check, once per process.
///
/// Keyed by the style, its kernel's address and its expression: a style
/// re-registered with another kernel, or an instance carrying another
/// expression (D16), is checked again.
type Verdicts = Mutex<HashMap<(String, String, usize, Option<String>), Result<(), IrError>>>;

fn first_compile() -> &'static Verdicts {
    static CHECKED: OnceLock<Verdicts> = OnceLock::new();
    CHECKED.get_or_init(Default::default)
}

/// The first-compile check of a style registered without samples: the
/// numeric checks on `probe` (real terms), once per process per kernel; a
/// failure is remembered and refuses every later compile alike.
pub(crate) fn check_at_first_compile(
    category: &CategorySpec,
    spec: &StyleSpec,
    form: &ExpressionForm,
    kernel_id: usize,
    native: bool,
    expressions: Option<ExpressionCompiler>,
    probe: impl FnOnce() -> Probe,
) -> Result<(), IrError> {
    if !spec.samples.is_empty() {
        return Ok(());
    }
    let key = (
        spec.category.to_string(),
        spec.name.to_string(),
        kernel_id,
        spec.expression.clone(),
    );
    if let Some(done) = first_compile().lock().unwrap().get(&key) {
        return done.clone();
    }
    let probe = probe();
    let outcome = if probe.n() == 0 {
        // Nothing to evaluate yet: check at a compile that has terms.
        return Ok(());
    } else {
        check_probe(category, spec, form, native, expressions, &probe)
    };
    first_compile().lock().unwrap().insert(key, outcome.clone());
    outcome
}

/// The address of a form, to key [`check_at_first_compile`] on.
pub(crate) fn form_id(form: &ExpressionForm) -> usize {
    match form {
        ExpressionForm::Scalar(f) => Arc::as_ptr(f) as *const () as usize,
        ExpressionForm::Compound(f) => Arc::as_ptr(f) as *const () as usize,
    }
}
