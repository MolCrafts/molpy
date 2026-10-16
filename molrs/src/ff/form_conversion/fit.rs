//! `fit_form`: the least-squares projection onto a style, under a declared
//! metric, with its residual (`ff-ir-01` P3, design item 6).

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::schema::block_names::ATOMS;
use molrs::op::{F, Idx};
use ndarray::Array1;

use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::form_conversion::{Converted, convert_rows, named, rewrite};
use crate::ff::ir::Params;
use crate::ff::ir::TypeParams;
use crate::ff::ir::form::declared;
use crate::ff::ir::{CategorySpec, Coordinate, IrError, ParamDimension, ParamKind, StyleSpec};
use crate::ff::potential::{Accumulator, Potential};
use crate::ff::style_registry::Registry;

/// The metric a fit minimises under: sample points of the category's
/// coordinate, each with a weight.
///
/// The fit of one row minimises `Σᵢ wᵢ (E_fit(qᵢ) + c − E_src(qᵢ))²` over
/// the target's dimensioned parameters (and the offset `c` when
/// [`offset`](Self::offset), else `c = 0`). The coordinate is the
/// category's: `r` (length, > 0) for a bond, `θ` (radians, in [0, π]) for an
/// angle, the signed dihedral `φ` (radians) for a dihedral or an improper.
///
/// The fit sees each style only through its energy `E(q)` as the registry's
/// kernel prices it — one term of the category on a synthetic geometry at
/// each sample point `q` — so it is generic over every tier: native,
/// expression, Python.
#[derive(Clone, Debug, PartialEq)]
pub struct FitMetric {
    /// The sample points.
    pub q: Vec<F>,
    /// One weight `≥ 0` per point.
    pub w: Vec<F>,
    /// When set, each row's weights are also Boltzmann factors of its source
    /// energy, `wᵢ · exp(−(E_src(qᵢ) − min E_src)/kT)` (`kT` in the field's
    /// energy unit).
    pub kt: Option<F>,
    /// Fit a free constant offset `c` with the parameters: the fit is then of
    /// the forces' physics, a constant being no part of it.
    pub offset: bool,
}

impl FitMetric {
    /// The points `q`, each of weight 1, no offset.
    pub fn new(q: Vec<F>) -> Self {
        let w = vec![1.0; q.len()];
        Self {
            q,
            w,
            kt: None,
            offset: false,
        }
    }

    /// `n` uniform points `lo + i (hi − lo)/n`, `i = 0 … n − 1` (half-open,
    /// so a grid over one period counts no point twice, and the grid of `2n`
    /// holds the grid of `n`).
    pub fn grid(lo: F, hi: F, n: usize) -> Self {
        let step = (hi - lo) / n as F;
        Self::new((0..n).map(|i| lo + i as F * step).collect())
    }

    /// The weights `w`, one per point.
    pub fn weights(mut self, w: Vec<F>) -> Self {
        self.w = w;
        self
    }

    /// Boltzmann-weight each row's points by its source energy at `kt`.
    pub fn boltzmann(mut self, kt: F) -> Self {
        self.kt = Some(kt);
        self
    }

    /// Fit a free constant offset.
    pub fn free_offset(mut self) -> Self {
        self.offset = true;
        self
    }

    /// Refuse a metric that is no metric of `coordinate`.
    fn check(&self, coordinate: Coordinate, style: &str) -> Result<(), IrError> {
        let bad = |reason: String| IrError::Malformed {
            style: style.to_owned(),
            reason: format!("fit_form metric: {reason}"),
        };
        if self.q.is_empty() {
            return Err(bad("no sample points".into()));
        }
        if self.w.len() != self.q.len() {
            return Err(bad(format!(
                "{} weights for {} points",
                self.w.len(),
                self.q.len()
            )));
        }
        if let Some(w) = self.w.iter().find(|w| !(w.is_finite() && **w >= 0.0)) {
            return Err(bad(format!("weight {w} is not finite and ≥ 0")));
        }
        if self.w.iter().all(|w| *w == 0.0) {
            return Err(bad("every weight is 0".into()));
        }
        if let Some(kt) = self.kt
            && !(kt.is_finite() && kt > 0.0)
        {
            return Err(bad(format!("kT = {kt} is not finite and > 0")));
        }
        for &q in &self.q {
            let ok = q.is_finite()
                && match coordinate {
                    Coordinate::Distance => q > 0.0,
                    Coordinate::Angle => (0.0..=std::f64::consts::PI).contains(&q),
                    _ => true,
                };
            if !ok {
                return Err(bad(format!(
                    "q = {q} is outside the {coordinate:?} coordinate's domain"
                )));
            }
        }
        Ok(())
    }
}

/// The residual of one fitted row, under the metric's weights `wᵢ` (the
/// Boltzmann factors included): `rᵢ = E_fit(qᵢ) + offset − E_src(qᵢ)`.
#[derive(Clone, Debug, PartialEq)]
pub struct TypeResidual {
    /// The source style.
    pub style: String,
    /// The type's name.
    pub type_: String,
    /// Whether the row was in the target's image: converted exactly
    /// (`project`), not fitted.
    pub exact: bool,
    /// `Σᵢ wᵢ rᵢ²` — the minimised objective. Monotone in the metric: a
    /// pointwise larger weight vector (a superset of points included) never
    /// lowers it.
    pub sum_sq: F,
    /// `Σᵢ wᵢ`.
    pub weight: F,
    /// `maxᵢ |rᵢ|` over the points of non-zero weight.
    pub max_abs: F,
    /// The fitted constant offset `c` (0 without a free offset).
    pub offset: F,
}

impl TypeResidual {
    /// `√(Σ wᵢ rᵢ² / Σ wᵢ)`.
    pub fn rms(&self) -> F {
        (self.sum_sq / self.weight).sqrt()
    }
}

/// The residual of a fit: one entry per fitted row, in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitResidual {
    pub types: Vec<TypeResidual>,
}

impl FitResidual {
    /// `Σ` of the rows' [`sum_sq`](TypeResidual::sum_sq).
    pub fn sum_sq(&self) -> F {
        self.types.iter().map(|t| t.sum_sq).sum()
    }

    /// `√(Σ sum_sq / Σ weight)` over every row; 0 for no rows.
    pub fn rms(&self) -> F {
        let weight: F = self.types.iter().map(|t| t.weight).sum();
        if weight == 0.0 {
            0.0
        } else {
            (self.sum_sq() / weight).sqrt()
        }
    }

    /// The largest [`max_abs`](TypeResidual::max_abs).
    pub fn max_abs(&self) -> F {
        self.types.iter().map(|t| t.max_abs).fold(0.0, F::max)
    }
}

/// Endpoint columns of a block row, in order.
const ENDS: [&str; 4] = ["atomi", "atomj", "atomk", "atoml"];

/// The positions at which a term of `coordinate` has the value `q`: a bond
/// of length `q`; an angle `q` at atom j; atoms i-j-k-l with signed
/// dihedral `q` (`i = (0,1,0)`, `j = 0`, `k = (1,0,0)`,
/// `l = (1, cos q, sin q)`).
fn geometry(coordinate: Coordinate, q: F) -> Vec<F> {
    let (s, c) = q.sin_cos();
    match coordinate {
        Coordinate::Distance => vec![0.0, 0.0, 0.0, q, 0.0, 0.0],
        Coordinate::Angle => vec![c, s, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        _ => vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s],
    }
}

/// How many parameters a seed takes when neither the source nor the spec
/// gives a value: `1` (a periodicity of 1, a force constant of 1, …).
const SEED_DEFAULT: F = 1.0;
/// Levenberg–Marquardt: the most iterations, and the relative improvement of
/// the objective below which it stops.
const MAX_ITER: usize = 200;
const CONVERGED: F = 1e-14;

impl Registry {
    /// [`ForceField::fit_form`] against this registry.
    pub(crate) fn fit_form(
        &self,
        ff: &ForceField,
        category: &str,
        style: &str,
        metric: &FitMetric,
    ) -> Result<(ForceField, FitResidual), IrError> {
        let cat = self
            .category(category)
            .ok_or_else(|| IrError::UnknownCategory {
                category: category.to_owned(),
            })?;
        let (spec, _) = self
            .style(category, style)
            .ok_or_else(|| IrError::NoKernel {
                category: category.to_owned(),
                style: style.to_owned(),
            })?;
        let to = named(category, style);
        if cat.is_pair_driven() || !cat.coordinate.is_scalar() {
            return Err(IrError::OutOfImage {
                from: category.to_owned(),
                to,
                type_: String::new(),
                reason: if cat.is_pair_driven() {
                    "fit_form fits one term per row, and a pair style's unlike pairs follow its \
                     mixing rule, which a per-row fit cannot hold"
                        .into()
                } else {
                    format!(
                        "the category's coordinate {:?} is no scalar",
                        cat.coordinate
                    )
                },
            });
        }
        metric.check(cat.coordinate, style)?;
        let target = self.form(category, style);
        let target_style = ff.get_style(category, style).map(|s| s.params().clone());
        let sources: Vec<String> = ff
            .get_styles(category)
            .into_iter()
            .filter(|s| s.name() != style && !s.type_rows().is_empty())
            .map(|s| s.name().to_owned())
            .collect();
        let mut residual = FitResidual::default();
        let mut rows: Vec<Converted> = Vec::new();
        for source in &sources {
            let from = named(category, source);
            let source_codec = self
                .form(category, source)
                .filter(|c| target.is_some_and(|t| t.family == c.family));
            let style_of = ff.get_style(category, source).expect("listed");
            rows.extend(convert_rows(self, style_of, |name, tp| {
                let refuse = |reason: String| IrError::OutOfImage {
                    from: from.clone(),
                    to: to.clone(),
                    type_: name.to_owned(),
                    reason,
                };
                let e_src = self
                    .energies(cat, source, &tp, &metric.q)
                    .map_err(|e| refuse(format!("the source cannot be priced: {e}")))?;
                let mut w = metric.w.clone();
                if let Some(kt) = metric.kt {
                    let lowest = e_src.iter().copied().fold(F::INFINITY, F::min);
                    for (w, e) in w.iter_mut().zip(&e_src) {
                        *w *= (-(e - lowest) / kt).exp();
                    }
                }
                // The canonical parameters, when the source embeds in the
                // target's family; the exact conversion when the target holds
                // them.
                let canonical = source_codec.and_then(|c| (c.embed)(&tp).ok());
                let exact = match (&canonical, target) {
                    (Some(c), Some(t)) => (t.project)(c).ok(),
                    _ => None,
                };
                let fixed_style = |p: Params| -> Params { target_style.clone().unwrap_or(p) };
                let (fitted, offset, is_exact) = match exact {
                    Some(t) => (TypeParams::new(fixed_style(t.style), t.row), None, true),
                    None => {
                        let seeded = match (&canonical, target.and_then(|t| t.seed.as_ref())) {
                            (Some(c), Some(seed)) => seed(c).ok(),
                            _ => None,
                        };
                        let start = match seeded {
                            Some(t) => TypeParams::new(fixed_style(t.style), t.row),
                            None => name_matched_seed(spec, &tp, target_style.as_ref()),
                        };
                        let (t, c) = self
                            .refine(
                                cat,
                                style,
                                spec,
                                start,
                                &metric.q,
                                &w,
                                &e_src,
                                metric.offset,
                            )
                            .map_err(refuse)?;
                        (t, Some(c), false)
                    }
                };
                let e_fit = self
                    .energies(cat, style, &fitted, &metric.q)
                    .map_err(|e| refuse(format!("the fitted row cannot be priced: {e}")))?;
                let offset = match offset {
                    Some(c) => c,
                    None if metric.offset => weighted_mean_gap(&w, &e_src, &e_fit),
                    None => 0.0,
                };
                let r: Vec<F> = e_fit
                    .iter()
                    .zip(&e_src)
                    .map(|(f, s)| f + offset - s)
                    .collect();
                residual.types.push(TypeResidual {
                    style: source.clone(),
                    type_: name.to_owned(),
                    exact: is_exact,
                    sum_sq: w.iter().zip(&r).map(|(w, r)| w * r * r).sum(),
                    weight: w.iter().sum(),
                    max_abs: w
                        .iter()
                        .zip(&r)
                        .filter(|(w, _)| **w > 0.0)
                        .map(|(_, r)| r.abs())
                        .fold(0.0, F::max),
                    offset,
                });
                Ok(fitted)
            })?);
        }
        if rows.is_empty() {
            return Ok((ff.clone(), residual));
        }
        let family = target.map_or_else(|| category.to_owned(), |t| t.family.to_string());
        let out = rewrite(ff, &family, category, style, &sources, rows)?;
        Ok((out, residual))
    }

    /// The energy of one term of `style` with parameters `tp` at each `q`,
    /// through this registry's kernel.
    pub(crate) fn energies(
        &self,
        cat: &CategorySpec,
        style: &str,
        tp: &TypeParams,
        q: &[F],
    ) -> Result<Vec<F>, crate::ff::potential::CompileError> {
        let arity = cat.arity.endpoints();
        let ends = &["a", "b", "c", "d"][..arity];
        let mut ff = ForceField::new("fit").with_registry(std::sync::Arc::new(self.clone()));
        ff.def_style(cat.name.as_ref(), style, tp.style.clone())
            .and_then(|s| s.def_type("t", ends, tp.row.clone()).map(|_| ()))
            .map_err(|e| e.to_string())?;
        let mut block = Block::new();
        for (i, key) in ENDS[..arity].iter().enumerate() {
            block
                .insert(*key, Array1::from_vec(vec![i as Idx]).into_dyn(), None)
                .map_err(|e| e.to_string())?;
        }
        block
            .insert(
                "type",
                Array1::from_vec(vec!["t".to_owned()]).into_dyn(),
                None,
            )
            .map_err(|e| e.to_string())?;
        let mut frame = Frame::new();
        frame.insert(cat.block.as_ref(), block);
        // The atoms at the first point, so that a kernel's first-compile
        // conformance check runs on this term.
        if let Some(&q0) = q.first() {
            let at = geometry(cat.coordinate, q0);
            let mut atoms = Block::new();
            for (d, key) in ["x", "y", "z"].into_iter().enumerate() {
                let col: Vec<F> = at.chunks(3).map(|p| p[d]).collect();
                atoms
                    .insert(key, Array1::from_vec(col).into_dyn(), None)
                    .map_err(|e| e.to_string())?;
            }
            frame.insert(ATOMS, atoms);
        }
        let pots = PotentialCompiler::new(&ff).compile(&frame)?;
        Ok(q.iter()
            .map(|&q| {
                let x = geometry(cat.coordinate, q);
                let mut acc = Accumulator::new(x.len());
                pots.evaluate(&x, &mut acc);
                acc.energy()
            })
            .collect())
    }

    /// Levenberg–Marquardt from `start` over its dimensioned scalar
    /// parameters (and the offset when `offset`): the fitted row and the
    /// offset.
    #[allow(clippy::too_many_arguments)]
    fn refine(
        &self,
        cat: &CategorySpec,
        style: &str,
        spec: &StyleSpec,
        start: TypeParams,
        q: &[F],
        w: &[F],
        e_src: &[F],
        offset: bool,
    ) -> Result<(TypeParams, F), String> {
        let mut free: Vec<String> = start
            .row
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .map(|(k, _)| k.to_owned())
            .filter(|k| fitted_param(spec, k))
            .collect();
        free.sort();
        let at = |x: &[F]| -> TypeParams {
            let mut tp = start.clone();
            for (k, v) in free.iter().zip(x) {
                tp.row.set(k, *v);
            }
            tp
        };
        let sw: Vec<F> = w.iter().map(|w| w.sqrt()).collect();
        // The refusal becomes the reason of the `OutOfImage` this fit
        // raises (the caller wraps it), so its message is what travels.
        let model = |x: &[F]| {
            self.energies(cat, style, &at(x), q)
                .map_err(|e| e.to_string())
        };
        let resid = |e: &[F], c: F| -> Vec<F> {
            e.iter()
                .zip(e_src)
                .zip(&sw)
                .map(|((e, s), sw)| sw * (e + c - s))
                .collect()
        };
        let ssr = |r: &[F]| r.iter().map(|r| r * r).sum::<F>();

        let mut x: Vec<F> = free
            .iter()
            .map(|k| start.row.get(k).and_then(|v| v.as_num()).expect("listed"))
            .collect();
        let e = model(&x)?;
        let mut c = if offset {
            weighted_mean_gap(w, e_src, &e)
        } else {
            0.0
        };
        let mut r = resid(&e, c);
        let mut s = ssr(&r);
        let n_par = x.len() + usize::from(offset);
        if n_par == 0 {
            return Ok((at(&x), c));
        }
        let mut lambda = 1e-6;
        for _ in 0..MAX_ITER {
            if s == 0.0 {
                break;
            }
            // The Jacobian of the weighted residual, by central differences.
            let mut jac: Vec<Vec<F>> = Vec::with_capacity(n_par);
            for j in 0..x.len() {
                let h = 1e-6 * x[j].abs().max(1.0);
                let mut up = x.clone();
                let mut down = x.clone();
                up[j] += h;
                down[j] -= h;
                let col = match (model(&up), model(&down)) {
                    (Ok(eu), Ok(ed)) => eu
                        .iter()
                        .zip(&ed)
                        .zip(&sw)
                        .map(|((u, d), sw)| sw * (u - d) / (2.0 * h))
                        .collect(),
                    _ => vec![0.0; q.len()],
                };
                jac.push(col);
            }
            if offset {
                jac.push(sw.clone());
            }
            let diag: Vec<F> = jac
                .iter()
                .map(|col| col.iter().map(|v| v * v).sum::<F>())
                .collect();
            let mut improved = None;
            while lambda < 1e16 {
                let step = damped_step(&jac, &r, &diag, lambda);
                let x2: Vec<F> = x.iter().zip(&step).map(|(x, d)| x + d).collect();
                let c2 = if offset { c + step[x.len()] } else { 0.0 };
                if let Ok(e2) = model(&x2) {
                    let r2 = resid(&e2, c2);
                    let s2 = ssr(&r2);
                    if s2.is_finite() && s2 < s {
                        improved = Some((s - s2) / s);
                        (x, c, r, s) = (x2, c2, r2, s2);
                        lambda = (lambda / 3.0).max(1e-15);
                        break;
                    }
                }
                lambda *= 4.0;
            }
            match improved {
                Some(rel) if rel > CONVERGED => {}
                _ => break,
            }
        }
        Ok((at(&x), c))
    }
}

/// Whether the fit varies `key` of a `spec` row: a declared (or indexed
/// family member) scalar parameter with a dimension.
fn fitted_param(spec: &StyleSpec, key: &str) -> bool {
    if !declared(spec, key) {
        return false;
    }
    let decl = spec.param(key).or_else(|| {
        let stem = key.trim_end_matches(|c: char| c.is_ascii_digit());
        spec.param(stem)
    });
    decl.is_some_and(|p| p.kind == ParamKind::Scalar && p.dim != ParamDimension::NONE)
}

/// The start of a fit without a codec seed: every declared scalar parameter
/// of `spec` (an indexed family as its source members, else member 1) from
/// the source row's same-named value, else the spec's default, else
/// [`SEED_DEFAULT`]. The style parameters are the target's own when it is in
/// the force field, else the source's that `spec` declares.
fn name_matched_seed(
    spec: &StyleSpec,
    source: &TypeParams,
    target_style: Option<&Params>,
) -> TypeParams {
    let mut row = Params::new();
    for p in spec.params.iter().filter(|p| p.kind == ParamKind::Scalar) {
        let default = p
            .default
            .as_ref()
            .and_then(|d| d.as_num())
            .unwrap_or(SEED_DEFAULT);
        if p.indexed {
            let mut m = 1;
            while let Some(v) = source
                .row
                .get(&format!("{}{m}", p.name))
                .and_then(|v| v.as_num())
            {
                row.set(&format!("{}{m}", p.name), v);
                m += 1;
            }
            if m == 1 {
                row.set(&format!("{}1", p.name), default);
            }
        } else {
            row.set(
                &p.name,
                source
                    .row
                    .get(&p.name)
                    .and_then(|v| v.as_num())
                    .unwrap_or(default),
            );
        }
    }
    // An indexed family is one table: every member family as long as the
    // longest.
    let terms = spec
        .params
        .iter()
        .filter(|p| p.indexed)
        .map(|p| {
            (1..)
                .take_while(|m| {
                    row.get(&format!("{}{m}", p.name))
                        .and_then(|v| v.as_num())
                        .is_some()
                })
                .count()
        })
        .max()
        .unwrap_or(0);
    for p in spec.params.iter().filter(|p| p.indexed) {
        for m in 1..=terms {
            let key = format!("{}{m}", p.name);
            if row.get(&key).and_then(|v| v.as_num()).is_none() {
                row.set(&key, SEED_DEFAULT);
            }
        }
    }
    let style = match target_style {
        Some(p) => p.clone(),
        None => {
            let mut style = Params::new();
            for p in &spec.style_params {
                if let Some(v) = source.style.get(&p.name).and_then(|v| v.as_num()) {
                    style.set(&p.name, v);
                }
                if let Some(v) = source.style.get(&p.name).and_then(|v| v.as_text()) {
                    style.set(&p.name, v);
                }
            }
            style
        }
    };
    TypeParams::new(style, row)
}

/// `Σ wᵢ (E_srcᵢ − E_fitᵢ) / Σ wᵢ`: the offset that minimises the residual
/// of fixed parameters.
fn weighted_mean_gap(w: &[F], e_src: &[F], e_fit: &[F]) -> F {
    let total: F = w.iter().sum();
    w.iter()
        .zip(e_src.iter().zip(e_fit))
        .map(|(w, (s, f))| w * (s - f))
        .sum::<F>()
        / total
}

/// The Levenberg–Marquardt step `δ` minimising `‖J δ + r‖² + λ Σⱼ dⱼ δⱼ²`,
/// by Householder QR of `[J; √(λ d)]` (columns of `J` given as `jac`). A
/// column with no effect (`dⱼ = 0`) gets no step.
fn damped_step(jac: &[Vec<F>], r: &[F], diag: &[F], lambda: F) -> Vec<F> {
    let p = jac.len();
    let m = r.len() + p;
    // Column-major A (m × p) and right-hand side b = [−r; 0].
    let mut a: Vec<Vec<F>> = jac
        .iter()
        .enumerate()
        .map(|(j, col)| {
            let mut c = col.clone();
            c.extend((0..p).map(|i| {
                if i == j {
                    (lambda * diag[j]).sqrt()
                } else {
                    0.0
                }
            }));
            c
        })
        .collect();
    let mut b: Vec<F> = r
        .iter()
        .map(|r| -r)
        .chain(std::iter::repeat_n(0.0, p))
        .collect();
    for k in 0..p {
        let norm = (k..m).map(|i| a[k][i] * a[k][i]).sum::<F>().sqrt();
        if norm == 0.0 {
            continue;
        }
        let alpha = if a[k][k] > 0.0 { -norm } else { norm };
        let mut v: Vec<F> = (k..m).map(|i| a[k][i]).collect();
        v[0] -= alpha;
        let vv: F = v.iter().map(|x| x * x).sum();
        if vv == 0.0 {
            continue;
        }
        let reflect = |col: &mut [F]| {
            let dot: F = col[k..].iter().zip(&v).map(|(c, v)| c * v).sum::<F>() * 2.0 / vv;
            for (c, v) in col[k..].iter_mut().zip(&v) {
                *c -= dot * v;
            }
        };
        for col in a.iter_mut().skip(k) {
            reflect(col);
        }
        reflect(&mut b);
    }
    let mut x = vec![0.0; p];
    for k in (0..p).rev() {
        if diag[k] == 0.0 || a[k][k] == 0.0 {
            continue;
        }
        let tail: F = (k + 1..p).map(|j| a[j][k] * x[j]).sum();
        x[k] = (b[k] - tail) / a[k][k];
    }
    x
}
