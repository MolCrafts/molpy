//! [`ScalarPair`]: a pair style priced by a [`ScalarForm`] of `r`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::SpecialClass;
use crate::ff::ir::{IrError, ParamCombination, ParamKind, StyleSpec};
use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::form_kernel::Probe;
use crate::ff::potential::form_kernel::{
    ScalarForm, StyleParamColumn, TermParams, columns, retain_form_inputs, row_num,
};
use crate::ff::potential::pair::{atom_type_index, fold_pairs, type_pair};
use crate::ff::potential::param_reads::{neighbour_cutoff, pair_cutoff};
use crate::ff::potential::{Accumulator, PairDriven, Potential, gather_copies};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use molrs::op::F;

/// Below this squared separation a pair is skipped: its direction is
/// undefined.
const MIN_R2: F = 1e-24;

/// What every term of a pair style shares: its style parameters.
#[derive(Clone, Debug, Default)]
struct StyleInputs {
    nums: Vec<(String, F)>,
    texts: Vec<(String, String)>,
}

/// Where a pair's parameters come from.
enum Source {
    /// Resolved against one fixed pair list at construction, with each
    /// pair's 1-4 weight.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        /// Per-type columns (the pair values) and `q1`, `q2`, per pair.
        terms: TermParams,
        weight: Vec<F>,
        /// The style's `cutoff²` (∞ when it states none): a listed pair
        /// prices only at `r < cutoff`, as at the typed door and in LAMMPS.
        cutoff2: F,
    },
    /// A type-pair table keyed by the two atoms' types (LAMMPS's
    /// `pair_coeff i j`), which answers for whatever pairs a neighbour
    /// search turns up. Under a ghost régime `type_id` and `charge` cover
    /// the copies too, each carrying its owner's entry.
    Typed {
        type_id: Vec<u32>,
        ntypes: usize,
        /// One `ntypes × ntypes` table per per-type column, laid out
        /// `ti * ntypes + tj`.
        table: Vec<(String, Vec<F>)>,
        /// Per atom; empty when the frame has no `atoms.charge`.
        charge: Vec<F>,
        /// The self-row inputs the form asked for (`epsilon1`): spelling,
        /// which atom of the pair (0, 1), the value per atom type.
        own: Vec<(String, usize, Vec<F>)>,
        cutoff2: F,
        n_owned: usize,
    },
}

/// A pair style whose energy is a [`ScalarForm`] of the distance.
///
/// Built at both compile doors from the style's [`StyleSpec`]:
///
/// * [`compiled`](Self::compiled) resolves every row of the frame's `pairs`
///   block — the fixed, intramolecular list — prices a row only at
///   `r < cutoff` (the style's `cutoff`, ∞ when it states none: LAMMPS
///   truncates every pair, special ones included), and weights an `is_14`
///   row by the force field's 1-4 weight of the style's special class
///   (`lj14scale` / `coulomb14scale`, which the compiler projects into the
///   style params).
/// * [`typed`](Self::typed) builds a type-pair table and prices a neighbour
///   table, honouring the style's required `cutoff` (`r < cutoff`), the
///   per-pair weights the caller hands it (zero skips the pair), and
///   tallying the virial `Σ f ⊗ r`.
///
/// The form sees each per-type parameter as the pair's value — the cross
/// row's where the style has one stating it, else the two self rows by the
/// parameter's [`ParamCombination`] — every numeric style parameter, `q1`, `q2` when
/// the frame carries `atoms.charge`, and the self-row values `<x>1`, `<x>2`
/// the form asks for ([`ScalarForm::inputs`]).
pub struct ScalarPair {
    form: Arc<dyn ScalarForm>,
    style: StyleInputs,
    source: Source,
}

/// A style's per-type rows, and how they make a pair's values.
struct PairRows<'a> {
    spec: &'a StyleSpec,
    cols: Vec<StyleParamColumn>,
    rows: HashMap<&'a str, &'a Params>,
    mixing: CombiningRule,
}

impl<'a> PairRows<'a> {
    fn new(
        spec: &'a StyleSpec,
        reads: &[String],
        style: &Params,
        tp: &'a [(&'a str, &'a Params)],
    ) -> Result<Self, IrError> {
        if let Some(p) = spec.params.iter().find(|p| p.kind != ParamKind::Scalar) {
            return Err(IrError::Malformed {
                style: spec.name.to_string(),
                reason: format!(
                    "a pair form takes numeric per-type parameters; `{}` is not",
                    p.name
                ),
            });
        }
        let mixing = match style.get("mixing").and_then(|v| v.as_text()) {
            Some(name) => CombiningRule::parse(name).map_err(|reason| IrError::BadValue {
                style: spec.name.to_string(),
                type_: String::new(),
                param: "mixing".into(),
                reason,
            })?,
            None => CombiningRule::UNDECLARED,
        };
        Ok(Self {
            spec,
            cols: retain_form_inputs(spec, columns(spec, &spec.params, tp)?, reads),
            rows: tp.iter().copied().collect(),
            mixing,
        })
    }

    /// Type `a`'s self-row value of column `c` (its row gathered:
    /// [`StyleSpec::gather`] filled its defaults).
    fn own(&self, a: &str, c: usize) -> Result<F, IrError> {
        let col = &self.cols[c];
        let missing = || IrError::MissingParam {
            style: self.spec.name.to_string(),
            type_: a.to_owned(),
            param: col.name.clone(),
        };
        let row = self.rows.get(a).ok_or_else(missing)?;
        row_num(self.spec, col, row).ok_or_else(missing)
    }

    /// The column of `base`'s family member matching column `c`.
    fn partner(&self, c: usize, base: &str) -> Option<usize> {
        let index = self.cols[c].index;
        self.cols
            .iter()
            .position(|col| self.spec.params[col.param].name == base && col.index == index)
    }

    /// The values of the atom-type pair `{a, b}`, one per column.
    fn resolve(&self, a: &str, b: &str) -> Result<Vec<F>, IrError> {
        let n = self.cols.len();
        if a == b {
            return (0..n).map(|c| self.own(a, c)).collect();
        }
        let key = pair_key(a, b).map_err(|reason| IrError::Malformed {
            style: self.spec.name.to_string(),
            reason,
        })?;
        let cross = self.rows.get(key.as_str());
        (0..n)
            .map(
                |c| match cross.and_then(|row| row_num(self.spec, &self.cols[c], row)) {
                    Some(v) => Ok(v),
                    None => self.mixed(a, b, c),
                },
            )
            .collect()
    }

    /// StyleParamColumn `c` of the unlike pair `{a, b}` by its mixing rule.
    fn mixed(&self, a: &str, b: &str, c: usize) -> Result<F, IrError> {
        Ok(match &self.spec.params[self.cols[c].param].mix {
            ParamCombination::Arithmetic => 0.5 * (self.own(a, c)? + self.own(b, c)?),
            ParamCombination::Geometric => (self.own(a, c)? * self.own(b, c)?).sqrt(),
            ParamCombination::LjEpsilon { sigma: other }
            | ParamCombination::LjSigma { epsilon: other } => {
                let s = self.partner(c, other).ok_or_else(|| IrError::Malformed {
                    style: self.spec.name.to_string(),
                    reason: format!(
                        "`{}` mixes with `{other}`, which is not declared",
                        self.cols[c].name
                    ),
                })?;
                let is_eps = matches!(
                    self.spec.params[self.cols[c].param].mix,
                    ParamCombination::LjEpsilon { .. }
                );
                let (e, s) = if is_eps { (c, s) } else { (s, c) };
                let (eps, sig) = self.mixing.combine(
                    (self.own(a, e)?, self.own(a, s)?),
                    (self.own(b, e)?, self.own(b, s)?),
                );
                if is_eps { eps } else { sig }
            }
            ParamCombination::None => {
                return Err(IrError::NoMixing {
                    style: self.spec.name.to_string(),
                    param: self.cols[c].name.clone(),
                    pair: format!("{{{a}, {b}}}"),
                });
            }
        })
    }
}

/// The self-row inputs `form` asks for: `(spelling, atom 0|1, column)`.
fn self_rows(form: &dyn ScalarForm, cols: &[StyleParamColumn]) -> Vec<(String, usize, usize)> {
    form.inputs()
        .into_iter()
        .filter(|i| !cols.iter().any(|c| &c.name == i))
        .filter_map(|i| {
            let end = match i.as_bytes().last() {
                Some(b'1') => 0,
                Some(b'2') => 1,
                _ => return None,
            };
            let c = cols.iter().position(|c| c.name == i[..i.len() - 1])?;
            Some((i, end, c))
        })
        .collect()
}

/// Refuse a form input none of `supplied` is.
fn check_inputs(
    spec: &StyleSpec,
    form: &dyn ScalarForm,
    supplied: impl Iterator<Item = String>,
) -> Result<(), IrError> {
    let supplied: Vec<String> = supplied.collect();
    match form.inputs().into_iter().find(|i| !supplied.contains(i)) {
        Some(param) => Err(IrError::MissingParam {
            style: spec.name.to_string(),
            type_: String::new(),
            param,
        }),
        None => Ok(()),
    }
}

fn charges(frame: &Frame) -> Vec<F> {
    frame
        .get(ATOMS)
        .and_then(|b| b.get("charge"))
        .and_then(|c| c.get::<f64>())
        .map(|c| c.iter().copied().collect())
        .unwrap_or_default()
}

fn atom_types(frame: &Frame, spec: &StyleSpec) -> Result<Vec<String>, String> {
    let col = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| {
            format!(
                "{} `{}`: atoms block missing \"type\" column",
                spec.category, spec.name
            )
        })?;
    Ok(col.iter().cloned().collect())
}

fn style_inputs(spec: &StyleSpec, style: &Params) -> Result<StyleInputs, IrError> {
    let mut t = TermParams::default();
    t.add_style(spec, style, 1)?;
    Ok(StyleInputs {
        nums: t.nums.into_iter().map(|(n, v)| (n, v[0])).collect(),
        texts: t.style_texts,
    })
}

impl ScalarPair {
    /// The compiled form: every row of the frame's `pairs` block resolved
    /// now, an `is_14` row weighted by the style param `lj14scale`
    /// ([`SpecialClass::Vdw`]) or `coulomb14scale` (`Coulomb`), every row
    /// truncated at the style param `cutoff` (absent: ∞).
    pub fn compiled(
        form: Arc<dyn ScalarForm>,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, crate::ff::potential::CompileError> {
        let who = format!("{} `{}`", spec.category, spec.name);
        let table = PairRows::new(spec, &form.inputs(), style, tp)?;
        let cutoff = pair_cutoff(&spec.name, style)?;
        let charge = charges(frame);
        let types = atom_types(frame, spec)?;
        let scale_14 = match spec.special_class() {
            SpecialClass::Vdw => style.get("lj14scale").and_then(|v| v.as_num()),
            SpecialClass::Coulomb => style.get("coulomb14scale").and_then(|v| v.as_num()),
        }
        .unwrap_or(1.0);
        let block = frame
            .get(PAIRS)
            .ok_or_else(|| format!("{who}: frame missing \"pairs\" block"))?;
        let column = |key: &str| {
            block
                .get(key)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| format!("{who}: pairs block missing \"{key}\" column"))
        };
        let (i_col, j_col) = (column("atomi")?, column("atomj")?);
        let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());
        let n = i_col.len();
        let mut values = vec![Vec::with_capacity(n); table.cols.len()];
        let mut q = [Vec::new(), Vec::new()];
        let own = self_rows(&*form, &table.cols);
        let mut own_values = vec![Vec::with_capacity(n); own.len()];
        let mut resolved: HashMap<(&str, &str), Vec<F>> = HashMap::new();
        let (mut atom_i, mut atom_j, mut weight) = (Vec::new(), Vec::new(), Vec::new());
        for row in 0..n {
            let ends = [i_col[row] as usize, j_col[row] as usize];
            let key = (types[ends[0]].as_str(), types[ends[1]].as_str());
            if let std::collections::hash_map::Entry::Vacant(slot) = resolved.entry(key) {
                slot.insert(table.resolve(key.0, key.1)?);
            }
            for (c, v) in resolved[&key].iter().enumerate() {
                values[c].push(*v);
            }
            if !charge.is_empty() {
                q[0].push(charge[ends[0]]);
                q[1].push(charge[ends[1]]);
            }
            for (k, (_, end, c)) in own.iter().enumerate() {
                own_values[k].push(table.own(&types[ends[*end]], *c)?);
            }
            atom_i.push(ends[0]);
            atom_j.push(ends[1]);
            weight.push(if is_14.is_some_and(|b| b[row]) {
                scale_14
            } else {
                1.0
            });
        }
        let mut terms = TermParams::default();
        for (col, v) in table.cols.iter().zip(values) {
            terms.nums.push((col.name.clone(), v));
        }
        if !charge.is_empty() {
            let [q1, q2] = q;
            terms.nums.push(("q1".into(), q1));
            terms.nums.push(("q2".into(), q2));
        }
        for ((name, _, _), v) in own.into_iter().zip(own_values) {
            terms.nums.push((name, v));
        }
        let style_inputs = style_inputs(spec, style)?;
        check_inputs(
            spec,
            &*form,
            terms
                .nums
                .iter()
                .map(|(n, _)| n.clone())
                .chain(style_inputs.nums.iter().map(|(n, _)| n.clone())),
        )?;
        Ok(Self {
            form,
            style: style_inputs,
            source: Source::Compiled {
                atom_i,
                atom_j,
                terms,
                weight,
                cutoff2: cutoff * cutoff,
            },
        })
    }

    /// The neighbour-driven form: a type-pair table over the atoms' types,
    /// which reads no `pairs` block. The style param `cutoff` is required —
    /// a periodic neighbour sum is not finite without one.
    pub fn typed(
        form: Arc<dyn ScalarForm>,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, crate::ff::potential::CompileError> {
        let table = PairRows::new(spec, &form.inputs(), style, tp)?;
        let cutoff = neighbour_cutoff(&spec.name, style)?;
        let (type_id, labels) = atom_type_index(frame)?;
        let ntypes = labels.len();
        let mut values = vec![vec![0.0; ntypes * ntypes]; table.cols.len()];
        for (ti, a) in labels.iter().enumerate() {
            for (tj, b) in labels.iter().enumerate() {
                let t = type_pair(ti as u32, tj as u32, ntypes);
                for (c, v) in table.resolve(a, b)?.into_iter().enumerate() {
                    values[c][t] = v;
                }
            }
        }
        let mut own = Vec::new();
        for (name, end, c) in self_rows(&*form, &table.cols) {
            let per_type = labels
                .iter()
                .map(|l| table.own(l, c))
                .collect::<Result<Vec<F>, _>>()?;
            own.push((name, end, per_type));
        }
        let charge = charges(frame);
        let style_inputs = style_inputs(spec, style)?;
        let charge_names = ["q1", "q2"].into_iter().filter(|_| !charge.is_empty());
        check_inputs(
            spec,
            &*form,
            table
                .cols
                .iter()
                .map(|c| c.name.clone())
                .chain(style_inputs.nums.iter().map(|(n, _)| n.clone()))
                .chain(charge_names.map(str::to_owned))
                .chain(own.iter().map(|(n, _, _)| n.clone())),
        )?;
        let n_owned = type_id.len();
        Ok(Self {
            form,
            style: style_inputs,
            source: Source::Typed {
                type_id,
                ntypes,
                table: table
                    .cols
                    .iter()
                    .map(|c| c.name.clone())
                    .zip(values)
                    .collect(),
                charge,
                own,
                cutoff2: cutoff * cutoff,
                n_owned,
            },
        })
    }

    /// `terms` with the style's parameters added, `n` long.
    fn with_style(&self, mut terms: TermParams, n: usize) -> TermParams {
        for (name, v) in &self.style.nums {
            terms.nums.push((name.clone(), vec![*v; n]));
        }
        terms.style_texts = self.style.texts.clone();
        terms
    }

    /// `(e, dE/dr)` of a batch: `r[k]` and its pairs' inputs.
    fn eval(&self, r: &[F], terms: &TermParams) -> (Vec<F>, Vec<F>) {
        let mut e = vec![0.0; r.len()];
        let mut de = vec![0.0; r.len()];
        terms.with_cols(0..r.len(), |p| self.form.eval(r, p, &mut e, &mut de));
        (e, de)
    }

    /// The inputs of the typed pairs `(i, j)`.
    fn typed_terms(&self, pairs: &[(usize, usize)]) -> TermParams {
        let Source::Typed {
            type_id,
            ntypes,
            table,
            charge,
            own,
            ..
        } = &self.source
        else {
            unreachable!("a typed kernel");
        };
        let mut terms = TermParams::default();
        for (name, tab) in table {
            let col = pairs
                .iter()
                .map(|&(i, j)| tab[type_pair(type_id[i], type_id[j], *ntypes)])
                .collect();
            terms.nums.push((name.clone(), col));
        }
        if !charge.is_empty() {
            terms
                .nums
                .push(("q1".into(), pairs.iter().map(|&(i, _)| charge[i]).collect()));
            terms
                .nums
                .push(("q2".into(), pairs.iter().map(|&(_, j)| charge[j]).collect()));
        }
        for (name, end, per_type) in own {
            let col = pairs
                .iter()
                .map(|&(i, j)| per_type[type_id[if *end == 0 { i } else { j }] as usize])
                .collect();
            terms.nums.push((name.clone(), col));
        }
        self.with_style(terms, pairs.len())
    }

    /// Up to `limit` of its pairs at `coords` (a typed kernel: the first
    /// pairs of distinct atoms), for the conformance check a style without
    /// registration samples gets at its first compile.
    pub(crate) fn probe(&self, coords: &[F], limit: usize) -> Probe {
        let dist = |i: usize, j: usize| {
            (0..3)
                .map(|c| (coords[j * 3 + c] - coords[i * 3 + c]).powi(2))
                .sum::<F>()
                .sqrt()
        };
        let (pairs, terms): (Vec<(usize, usize)>, TermParams) = match &self.source {
            Source::Compiled {
                atom_i,
                atom_j,
                terms,
                ..
            } => {
                let picked: Vec<usize> = (0..atom_i.len().min(limit)).collect();
                let pairs = picked.iter().map(|&t| (atom_i[t], atom_j[t])).collect();
                (pairs, self.with_style(terms.select(&picked), picked.len()))
            }
            Source::Typed { n_owned, .. } => {
                let pairs: Vec<(usize, usize)> = (0..*n_owned)
                    .flat_map(|i| ((i + 1)..*n_owned).map(move |j| (i, j)))
                    .filter(|&(i, j)| dist(i, j) > 1e-6)
                    .take(limit)
                    .collect();
                let terms = self.typed_terms(&pairs);
                (pairs, terms)
            }
        };
        Probe {
            params: terms,
            q: pairs.iter().map(|&(i, j)| dist(i, j)).collect(),
            x: Vec::new(),
            arity: 2,
        }
    }

    /// One contiguous range of a neighbour table, into `out`.
    fn fold_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let Source::Typed {
            type_id, cutoff2, ..
        } = &self.source
        else {
            unreachable!("only a typed kernel folds a neighbour table");
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (0.0, Virial::ZERO);
        };
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        // Exactly zero *skips*: a bonded pair sits at bond length, where a
        // repulsive term is enormous, and scaling it by zero would be
        // arithmetic on a number that should never have been computed.
        let active: Vec<usize> = rows
            .filter(|&p| factor.is_empty() || factor[p] != 0.0)
            .filter(|&p| d2[p] >= MIN_R2 && d2[p] < *cutoff2)
            .collect();
        let ends: Vec<(usize, usize)> = active
            .iter()
            .map(|&p| (i_col[p] as usize, j_col[p] as usize))
            .collect();
        debug_assert!(
            ends.iter()
                .all(|&(i, j)| i < type_id.len() && j < type_id.len()),
            "a pair names an atom the type table does not cover"
        );
        let r: Vec<F> = active.iter().map(|&p| d2[p].sqrt()).collect();
        let (e, de) = self.eval(&r, &self.typed_terms(&ends));
        let mut energy: F = 0.0;
        let mut virial = Virial::ZERO;
        for (k, &p) in active.iter().enumerate() {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            let (i, j) = ends[k];
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let scale = -de[k] / r[k];
            let f = [w * scale * d[0], w * scale * d[1], w * scale * d[2]];
            energy += w * e[k];
            if VIRIAL {
                virial.add_outer(f, d);
            }
            for c in 0..3 {
                out[j * 3 + c] += f[c];
                out[i * 3 + c] -= f[c];
            }
        }
        (energy, virial)
    }
}

impl Potential for ScalarPair {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let Source::Compiled {
            atom_i,
            atom_j,
            terms,
            weight,
            cutoff2,
        } = &self.source
        else {
            // A type table needs a pair table, and nobody handed one over.
            return;
        };
        let disp: Vec<[F; 3]> = atom_i
            .iter()
            .zip(atom_j)
            .map(|(&i, &j)| {
                [
                    coords[j * 3] - coords[i * 3],
                    coords[j * 3 + 1] - coords[i * 3 + 1],
                    coords[j * 3 + 2] - coords[i * 3 + 2],
                ]
            })
            .collect();
        let d2: Vec<F> = disp
            .iter()
            .map(|d| d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            .collect();
        // The rows inside the cutoff, as the typed fold selects them: the
        // form is never evaluated past it (where it need not be defined).
        let active: Vec<usize> = (0..d2.len())
            .filter(|&t| d2[t] >= MIN_R2 && d2[t] < *cutoff2)
            .collect();
        let r: Vec<F> = active.iter().map(|&t| d2[t].sqrt()).collect();
        let (e, de) = self.eval(&r, &self.with_style(terms.select(&active), r.len()));
        let out = acc.forces_without_virial();
        let mut energy: F = 0.0;
        for (k, &t) in active.iter().enumerate() {
            let w = weight[t];
            energy += w * e[k];
            let scale = -de[k] / r[k];
            for c in 0..3 {
                let f = w * scale * disp[t][c];
                out[atom_j[t] * 3 + c] += f;
                out[atom_i[t] * 3 + c] -= f;
            }
        }
        acc.add_energy(energy);
    }
}

impl PairDriven for ScalarPair {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        if let Source::Compiled { .. } = self.source {
            // A compiled kernel answers for its own list, not for this one,
            // and cannot read a per-pair weight either; both providers refuse
            // one, so this is the free-boundary path and `factor` is empty.
            debug_assert!(factor.is_empty());
            self.evaluate(coords, acc);
            return;
        }
        if pairs.disp().is_none() || pairs.dist_sq().is_none() {
            // No displacements, nothing priced — and no virial claimed for it.
            acc.forces_without_virial();
            return;
        }
        fold_pairs(
            acc,
            pairs.query_point_indices().len(),
            |out: &mut [F], rows| self.fold_rows::<false>(out, factor, pairs, rows),
            |out: &mut [F], rows| self.fold_rows::<true>(out, factor, pairs, rows),
        );
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }

    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::Typed {
            type_id,
            charge,
            n_owned,
            ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(type_id, *n_owned, owner);
        if !charge.is_empty() {
            gather_copies(charge, *n_owned, owner);
        }
    }
}
