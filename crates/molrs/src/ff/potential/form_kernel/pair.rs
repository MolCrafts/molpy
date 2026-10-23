//! [`ScalarPair`]: a pair style priced by a [`ScalarForm`] of `r`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::SpecialClass;
use crate::ff::ir::{IrError, ParamCombination, ParamKind, ParamSource, StyleSpec};
use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::form_kernel::Probe;
use crate::ff::potential::form_kernel::{
    ParamColumns, ScalarForm, StyleParamColumn, TermParams, columns, numeric_column,
    retain_form_inputs, row_num,
};
use crate::ff::potential::pair::{atom_type_index, fold_pairs, pair_columns, type_pair};
use crate::ff::potential::param_reads::{neighbour_cutoff, pair_cutoff, style_num};
use crate::ff::potential::{Accumulator, CompileError, PairDriven, Potential};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::core::schema::block_names::ATOMS;
use molrs::op::F;

/// Below this separation a pair has no direction: it exerts no force, and
/// prices its energy only where the form's is finite there.
const MIN_R: F = 1e-12;

/// [`MIN_R`] squared.
const MIN_R2: F = MIN_R * MIN_R;

/// Pairs per call of the form: large enough that a per-call cost is spread
/// thin, small enough that a batch's columns stay in cache.
const BATCH: usize = 512;

/// What every term of a pair style shares: its style parameters.
#[derive(Clone, Debug, Default)]
struct StyleInputs {
    nums: Vec<(String, F)>,
    texts: Vec<(String, String)>,
    /// Each of `nums` broadcast to a whole batch, sliced per call.
    broadcast: Vec<Vec<F>>,
}

impl StyleInputs {
    fn of(spec: &StyleSpec, style: &Params) -> Result<Self, IrError> {
        let mut t = TermParams::default();
        t.add_style(spec, style, 1)?;
        let broadcast = t.style_nums.iter().map(|(_, v)| vec![*v; BATCH]).collect();
        Ok(Self {
            nums: t.style_nums,
            texts: t.style_texts,
            broadcast,
        })
    }

    /// The style parameters as a form's `check_style` reads them.
    fn check<Fm: ScalarForm + ?Sized>(&self, spec: &StyleSpec, form: &Fm) -> Result<(), IrError> {
        form.check_style(&self.cols())
            .map_err(|r| TermParams::refused(spec, r))
    }

    /// The style parameters alone, as a form's `check_style` and
    /// `pointwise` read them.
    fn cols(&self) -> ParamColumns<'_> {
        let mut cols = ParamColumns::new();
        for (name, v) in &self.nums {
            cols.push_style_num(name, *v);
        }
        for (name, t) in &self.texts {
            cols.push_style_text(name, t);
        }
        cols
    }
}

/// A per-atom input of the form: `q1`/`q2` (the atoms' charges), or the
/// self-row value `<x>1`/`<x>2` of a parameter it asks for.
#[derive(Clone, Debug)]
struct AtomInput {
    name: String,
    /// Which atom of the pair: 0 (`i`) or 1 (`j`).
    end: usize,
    /// Index into [`ScalarPair::per_atom`].
    values: usize,
}

/// Where a pair's parameters come from.
#[derive(Clone)]
enum Source {
    /// Resolved against one fixed pair list at construction, with each
    /// pair's 1-4 weight.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        /// Per pair-valued column, its value per pair.
        values: Vec<Vec<F>>,
        weight: Vec<F>,
        /// The style's `cutoff²` (∞ when it states none): a listed pair
        /// prices only at `r < cutoff`, as at the typed door and in LAMMPS.
        cutoff2: F,
    },
    /// A type-pair table keyed by the two atoms' types (LAMMPS's
    /// `pair_coeff i j`), which answers for whatever pairs a neighbour
    /// search turns up.
    Typed {
        /// The type of each atom; `None`: one type for every atom (a style
        /// with no pair-valued column, or a uniform kernel).
        type_id: Option<Vec<u32>>,
        ntypes: usize,
        /// One `ntypes × ntypes` table per pair-valued column, laid out
        /// `ti * ntypes + tj`.
        table: Vec<Vec<F>>,
        cutoff2: F,
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
///   style params); a row of weight exactly zero is skipped.
/// * [`typed`](Self::typed) builds a type-pair table and prices a neighbour
///   table, honouring the style's `cutoff` (`r < cutoff`; required finite
///   when the style declares one, ∞ when it declares none and its form
///   bounds itself), the per-pair weights the caller hands it (zero skips
///   the pair), and tallying the virial `Σ f ⊗ r`.
///
/// The form sees each per-type parameter it reads by name as the pair's
/// value — the cross row's where the style has one stating it, else the two
/// self rows by the parameter's [`ParamCombination`] — every numeric style
/// parameter, `q1`, `q2` when the frame carries `atoms.charge`, and the
/// self-row values `<x>1`, `<x>2` it asks for ([`ScalarForm::inputs`]): the
/// atom's type row, or for a [`ParamSource::PerInstance`] style the atoms'
/// own column of that name. A parameter the form reads only as `<x>1`,
/// `<x>2` is never combined into a pair value (it needs no mixing rule).
///
/// A pair below 10⁻¹² of separation has no direction: it exerts no force,
/// and prices the form's energy there where that is finite (a buffered
/// Coulomb's `k qᵢqⱼ / (D δ)`), nothing where it is not (a bare `1/r`) — as
/// a bonded term at zero length does.
///
/// # Monomorphized and dynamic
///
/// `ScalarPair<Fm>` is generic over its form's type. Over `dyn ScalarForm`
/// (a third party's form, an expression's) it calls the form once per batch
/// of up to 512 pairs and gathers and scatters around the call. Over a
/// concrete form that provides a [`pointwise`](ScalarForm::pointwise) law
/// (every built-in pair style, [`fused`](Self::fused)) it evaluates the law
/// inside its own per-pair loop, inlined beside the selection, the gather
/// and the scatter — a hand-written kernel's single pass, written once for
/// every form.
pub struct ScalarPair<Fm: ?Sized = dyn ScalarForm> {
    form: Arc<Fm>,
    style: StyleInputs,
    /// The pair-valued columns' names, then the atom inputs', in the order
    /// a batch's columns are gathered.
    names: Vec<String>,
    atoms: Vec<AtomInput>,
    per_atom: Vec<Vec<F>>,
    source: Source,
    /// Where each per-term input of the form's pointwise law comes from, in
    /// the law's order; `None` when one is not a column of the kernel or
    /// there are more than [`MAX_LAW_INPUTS`].
    law_inputs: Option<Vec<LawInput>>,
    strategy: Strategy<Fm>,
}

impl<Fm: ?Sized> Clone for ScalarPair<Fm> {
    fn clone(&self) -> Self {
        Self {
            form: Arc::clone(&self.form),
            style: self.style.clone(),
            names: self.names.clone(),
            atoms: self.atoms.clone(),
            per_atom: self.per_atom.clone(),
            source: self.source.clone(),
            law_inputs: self.law_inputs.clone(),
            strategy: self.strategy,
        }
    }
}

/// The most per-term inputs a pointwise law is gathered for (MMFF's vdW has
/// ten): the fused loop is instantiated once per input count, so the gather
/// unrolls.
const MAX_LAW_INPUTS: usize = 10;

/// Where one per-term input of a pointwise law comes from.
#[derive(Clone, Copy, Debug)]
enum LawInput {
    /// Pair-valued column `c`.
    Paired(usize),
    /// Atom input `k` ([`AtomInput`]).
    Atom(usize),
}

/// One fold over a range of a neighbour table.
type FoldRows<Fm> =
    fn(&ScalarPair<Fm>, &mut [F], &[F], &Neighbors, std::ops::Range<usize>) -> (F, Virial);

/// One pair of a uniform kernel at `r²` along its separation: its energy and
/// the force on its second atom.
type OnePair<Fm> = fn(&ScalarPair<Fm>, F, [F; 3]) -> (F, [F; 3]);

/// How a kernel evaluates: a batch at a time through the form's `eval`, or
/// pair by pair through its pointwise law.
struct Strategy<Fm: ?Sized> {
    plain: FoldRows<Fm>,
    tallied: FoldRows<Fm>,
    compiled: fn(&ScalarPair<Fm>, &[F], &mut Accumulator<'_>),
    one: OnePair<Fm>,
}

impl<Fm: ?Sized> Clone for Strategy<Fm> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Fm: ?Sized> Copy for Strategy<Fm> {}

/// A style's per-type rows, and how they make a pair's values.
struct PairRows<'a> {
    spec: &'a StyleSpec,
    cols: Vec<StyleParamColumn>,
    /// Whether column `c` is a pair value the form reads, rather than read
    /// only as a self row (`<x>1`, `<x>2`).
    paired: Vec<bool>,
    rows: HashMap<&'a str, &'a Params>,
    defaults: Params,
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
        let cols = retain_form_inputs(spec, columns(spec, &spec.params, tp)?, reads);
        let paired = pair_valued(spec, &cols, reads);
        Ok(Self {
            spec,
            cols,
            paired,
            rows: tp.iter().copied().collect(),
            defaults: spec.default_row(),
            mixing,
        })
    }

    /// The pair-valued columns' names.
    fn pair_names(&self) -> Vec<String> {
        self.cols
            .iter()
            .zip(&self.paired)
            .filter(|(_, on)| **on)
            .map(|(c, _)| c.name.clone())
            .collect()
    }

    /// Type `a`'s self-row value of column `c` (its row gathered:
    /// [`StyleSpec::gather`] filled its defaults; a per-instance style with
    /// no row for `a` reads the declared defaults).
    fn own(&self, a: &str, c: usize) -> Result<F, IrError> {
        let col = &self.cols[c];
        let missing = || IrError::MissingParam {
            style: self.spec.name.to_string(),
            type_: a.to_owned(),
            param: col.name.clone(),
        };
        let row = match self.rows.get(a) {
            Some(row) => *row,
            None if self.spec.source == ParamSource::PerInstance => &self.defaults,
            None => return Err(missing()),
        };
        row_num(self.spec, col, row).ok_or_else(missing)
    }

    /// The column of `base`'s family member matching column `c`.
    fn partner(&self, c: usize, base: &str) -> Option<usize> {
        let index = self.cols[c].index;
        self.cols
            .iter()
            .position(|col| self.spec.params[col.param].name == base && col.index == index)
    }

    /// The pair values of the atom-type pair `{a, b}`, one per pair-valued
    /// column.
    fn resolve(&self, a: &str, b: &str) -> Result<Vec<F>, IrError> {
        let paired = (0..self.cols.len()).filter(|&c| self.paired[c]);
        if a == b {
            return paired.map(|c| self.own(a, c)).collect();
        }
        let key = pair_key(a, b).map_err(|reason| IrError::Malformed {
            style: self.spec.name.to_string(),
            reason,
        })?;
        let cross = self.rows.get(key.as_str());
        paired
            .map(
                |c| match cross.and_then(|row| row_num(self.spec, &self.cols[c], row)) {
                    Some(v) => Ok(v),
                    None => self.mixed(a, b, c),
                },
            )
            .collect()
    }

    /// Column `c` of the unlike pair `{a, b}` by its mixing rule.
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

/// Which of `cols` are pair values: every column when the form states no
/// inputs; else those it reads by name, and the mixing partner of one that
/// is.
fn pair_valued(spec: &StyleSpec, cols: &[StyleParamColumn], reads: &[String]) -> Vec<bool> {
    if reads.is_empty() {
        return vec![true; cols.len()];
    }
    let by_name: Vec<bool> = cols.iter().map(|c| reads.contains(&c.name)).collect();
    let partner_of = |c: &StyleParamColumn| match &spec.params[c.param].mix {
        ParamCombination::LjEpsilon { sigma: p } | ParamCombination::LjSigma { epsilon: p } => {
            Some(p.clone())
        }
        _ => None,
    };
    cols.iter()
        .map(|c| {
            cols.iter().zip(&by_name).any(|(other, &on)| {
                on && (other.name == c.name
                    || (partner_of(other).is_some_and(|p| spec.params[c.param].name == p)
                        && other.index == c.index))
            })
        })
        .collect()
}

/// The self-row inputs `form` asks for: `(spelling, atom 0|1, column)`.
fn self_rows<Fm: ScalarForm + ?Sized>(
    form: &Fm,
    cols: &[StyleParamColumn],
) -> Vec<(String, usize, usize)> {
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
fn check_inputs<Fm: ScalarForm + ?Sized>(
    spec: &StyleSpec,
    form: &Fm,
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

fn charges(frame: &Frame) -> Option<Vec<F>> {
    frame
        .get(ATOMS)
        .and_then(|b| b.get("charge"))
        .and_then(|c| c.get::<f64>())
        .map(|c| c.iter().copied().collect())
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

/// The per-atom inputs of a pair kernel over `frame`: the charges as `q1`,
/// `q2`, and the self-row values the form asks for, each atom's — its own
/// cell of a per-instance style's atoms column, else its type's row.
fn atom_inputs<Fm: ScalarForm + ?Sized>(
    table: &PairRows<'_>,
    form: &Fm,
    frame: &Frame,
    types: &mut dyn FnMut() -> Result<Vec<String>, String>,
) -> Result<(Vec<AtomInput>, Vec<Vec<F>>), CompileError> {
    let mut inputs = Vec::new();
    let mut per_atom = Vec::new();
    if let Some(q) = charges(frame) {
        per_atom.push(q);
        for (end, name) in ["q1", "q2"].into_iter().enumerate() {
            inputs.push(AtomInput {
                name: name.into(),
                end,
                values: 0,
            });
        }
    }
    let atoms = frame.get(ATOMS);
    let n_atoms = atoms.and_then(|b| b.n_rows()).unwrap_or(0);
    let mut by_column: HashMap<usize, usize> = HashMap::new();
    for (name, end, c) in self_rows(form, &table.cols) {
        let values = match by_column.get(&c) {
            Some(&v) => v,
            None => {
                let col = &table.cols[c].name;
                let own = (table.spec.source == ParamSource::PerInstance)
                    .then(|| {
                        atoms.and_then(|b| numeric_column(b, col).map(|v| (v, b.validity(col))))
                    })
                    .flatten();
                let mut values = Vec::with_capacity(n_atoms);
                let mut labels: Option<Vec<String>> = None;
                let mut by_type: HashMap<String, F> = HashMap::new();
                for a in 0..n_atoms {
                    if let Some((v, valid)) = &own
                        && valid.is_none_or(|m| m[a])
                    {
                        values.push(v[a]);
                        continue;
                    }
                    if labels.is_none() {
                        labels = Some(types()?);
                    }
                    let label = &labels.as_ref().expect("read above")[a];
                    let v = match by_type.get(label) {
                        Some(&v) => v,
                        None => {
                            let v = table.own(label, c)?;
                            by_type.insert(label.clone(), v);
                            v
                        }
                    };
                    values.push(v);
                }
                per_atom.push(values);
                by_column.insert(c, per_atom.len() - 1);
                per_atom.len() - 1
            }
        };
        inputs.push(AtomInput { name, end, values });
    }
    Ok((inputs, per_atom))
}

/// The columns of one batch of pairs, allocated once per fold: a fold
/// fills the first `n` slots, the form reads (and writes) them.
struct Batch {
    /// The pairs' rows in their source (a neighbour table's, or the compiled
    /// list's).
    rows: Vec<usize>,
    /// The pairs' atoms.
    ai: Vec<usize>,
    aj: Vec<usize>,
    /// The pairs' cells of a type-pair table.
    cells: Vec<usize>,
    r: Vec<F>,
    e: Vec<F>,
    de: Vec<F>,
    cols: Vec<Vec<F>>,
    pending: Pending,
}

impl Batch {
    fn new(n_cols: usize) -> Self {
        Self {
            pending: Pending::default(),
            rows: vec![0; BATCH],
            ai: vec![0; BATCH],
            aj: vec![0; BATCH],
            cells: vec![0; BATCH],
            r: vec![0.0; BATCH],
            e: vec![0.0; BATCH],
            de: vec![0.0; BATCH],
            cols: vec![vec![0.0; BATCH]; n_cols],
        }
    }

    /// Price the batch's first `n` pairs, evaluated: each pair's energy and
    /// force times its weight `weight(row)`, its force along its separation
    /// `disp(k, row)`, onto `out` (`+f` on `j`, `−f` on `i`), tallying the
    /// virial `Σ f ⊗ d` when `VIRIAL`. A pair below [`MIN_R`] exerts no
    /// force and prices its energy where it is finite. Returns the energy.
    fn scatter<const VIRIAL: bool>(
        &mut self,
        n: usize,
        weight: impl Fn(usize) -> F,
        disp: impl Fn(usize, usize) -> [F; 3],
        out: &mut [F],
        virial: &mut Virial,
    ) -> F {
        // The weighted energies, four partial sums, and each pair's force
        // factor `−w dE/dr / r` (zero for a pair with no direction).
        let mut energy = [0.0; 4];
        let pairs = self.rows[..n]
            .iter()
            .zip(self.e[..n].iter().zip(&self.r[..n]))
            .zip(self.de[..n].iter_mut());
        for (k, ((&row, (&e, &r)), de)) in pairs.enumerate() {
            let w = weight(row);
            let (e, scale) = if r < MIN_R {
                (if e.is_finite() { e } else { 0.0 }, 0.0)
            } else {
                (e, -*de / r)
            };
            energy[k % 4] += w * e;
            *de = w * scale;
        }
        let pairs = self.rows[..n]
            .iter()
            .zip(self.ai[..n].iter().zip(&self.aj[..n]))
            .zip(&self.de[..n]);
        for (k, ((&row, (&i, &j)), &scale)) in pairs.enumerate() {
            let d = disp(k, row);
            let f = [scale * d[0], scale * d[1], scale * d[2]];
            if VIRIAL {
                virial.add_outer(f, d);
            }
            self.pending.push(i, j, f, out);
        }
        (energy[0] + energy[1]) + (energy[2] + energy[3])
    }
}

/// The atom `i` the last pairs named and the force they put on it, not yet
/// added: a pair list names one `i` for a run of pairs, and adding the run's
/// force to it once keeps that atom's force out of a chain of
/// read-modify-writes.
#[derive(Default)]
struct Pending(Option<(usize, [F; 3])>);

impl Pending {
    /// The force `f` of the pair `(i, j)`: `+f` on `j` now, `−f` on `i` with
    /// the run it belongs to.
    #[inline(always)]
    fn push(&mut self, i: usize, j: usize, f: [F; 3], out: &mut [F]) {
        let fi = match &mut self.0 {
            Some((cur, fi)) if *cur == i => fi,
            pending => {
                if let Some((cur, fi)) = pending.take() {
                    for c in 0..3 {
                        out[cur * 3 + c] -= fi[c];
                    }
                }
                &mut pending.insert((i, [0.0; 3])).1
            }
        };
        for c in 0..3 {
            out[j * 3 + c] += f[c];
            fi[c] += f[c];
        }
    }

    /// Add the force still pending to its atom.
    fn flush(&mut self, out: &mut [F]) {
        if let Some((i, fi)) = self.0.take() {
            for c in 0..3 {
                out[i * 3 + c] -= fi[c];
            }
        }
    }
}

impl<Fm: ScalarForm + ?Sized> ScalarPair<Fm> {
    /// The compiled form: every row of the frame's `pairs` block resolved
    /// now, an `is_14` row weighted by the style param `lj14scale`
    /// ([`SpecialClass::Vdw`]) or `coulomb14scale` (`Coulomb`), every row
    /// truncated at the style param `cutoff` (absent: ∞).
    pub fn compiled(
        form: Arc<Fm>,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, CompileError> {
        let who = format!("{} `{}`", spec.category, spec.name);
        let table = PairRows::new(spec, &form.inputs(), style, tp)?;
        let cutoff = pair_cutoff(&spec.name, style)?;
        let scale_14 = style_num(
            &spec.name,
            style,
            match spec.special_class() {
                SpecialClass::Vdw => crate::ff::ir::LJ_14_SCALE,
                SpecialClass::Coulomb => crate::ff::ir::COULOMB_14_SCALE,
            },
        )?;
        let (i_col, j_col, is_14) = pair_columns(frame, &who)?;
        let n = i_col.len();
        let names = table.pair_names();
        let mut types: Option<Vec<String>> = None;
        let mut read_types = || -> Result<Vec<String>, String> {
            match &types {
                Some(t) => Ok(t.clone()),
                None => {
                    let t = atom_types(frame, spec)?;
                    types = Some(t.clone());
                    Ok(t)
                }
            }
        };
        let mut values = vec![Vec::with_capacity(n); names.len()];
        if !names.is_empty() {
            let labels = read_types()?;
            let mut resolved: HashMap<(&str, &str), Vec<F>> = HashMap::new();
            for row in 0..n {
                let key = (
                    labels[i_col[row] as usize].as_str(),
                    labels[j_col[row] as usize].as_str(),
                );
                if let std::collections::hash_map::Entry::Vacant(slot) = resolved.entry(key) {
                    slot.insert(table.resolve(key.0, key.1)?);
                }
                for (c, v) in resolved[&key].iter().enumerate() {
                    values[c].push(*v);
                }
            }
        }
        let (atoms, per_atom) = atom_inputs(&table, &*form, frame, &mut read_types)?;
        let weight = (0..n)
            .map(|row| {
                if is_14.is_some_and(|b| b[row]) {
                    scale_14
                } else {
                    1.0
                }
            })
            .collect();
        Self::assemble(
            form,
            spec,
            style,
            names,
            atoms,
            per_atom,
            Source::Compiled {
                atom_i: i_col.iter().map(|&a| a as usize).collect(),
                atom_j: j_col.iter().map(|&a| a as usize).collect(),
                values,
                weight,
                cutoff2: cutoff * cutoff,
            },
        )
    }

    /// The neighbour-driven form: a type-pair table over the atoms' types,
    /// which reads no `pairs` block. A style that declares a `cutoff` needs
    /// a finite one — a periodic neighbour sum is not finite without one; a
    /// style that declares none prices every pair the search turns up, its
    /// form bounding itself.
    pub fn typed(
        form: Arc<Fm>,
        spec: &StyleSpec,
        style: &Params,
        tp: &[(&str, &Params)],
        frame: &Frame,
    ) -> Result<Self, CompileError> {
        let table = PairRows::new(spec, &form.inputs(), style, tp)?;
        let cutoff = if spec.style_param("cutoff").is_some() {
            neighbour_cutoff(&spec.name, style)?
        } else {
            F::INFINITY
        };
        let names = table.pair_names();
        let (type_id, ntypes, values) = if names.is_empty() {
            (None, 1, Vec::new())
        } else {
            let (type_id, labels) = atom_type_index(frame)?;
            let ntypes = labels.len();
            let mut values = vec![vec![0.0; ntypes * ntypes]; names.len()];
            for (ti, a) in labels.iter().enumerate() {
                for (tj, b) in labels.iter().enumerate() {
                    let t = type_pair(ti as u32, tj as u32, ntypes);
                    for (c, v) in table.resolve(a, b)?.into_iter().enumerate() {
                        values[c][t] = v;
                    }
                }
            }
            (Some(type_id), ntypes, values)
        };
        let mut read_types = || atom_types(frame, spec);
        let (atoms, per_atom) = atom_inputs(&table, &*form, frame, &mut read_types)?;
        Self::assemble(
            form,
            spec,
            style,
            names,
            atoms,
            per_atom,
            Source::Typed {
                type_id,
                ntypes,
                table: values,
                cutoff2: cutoff * cutoff,
            },
        )
    }

    /// One parameter row priced for every pair a neighbour table hands it,
    /// whatever atoms it names: the type-pair table of a single type. The
    /// style's `cutoff` is required finite.
    pub(crate) fn uniform(
        form: Arc<Fm>,
        spec: &StyleSpec,
        style: &Params,
        row: &Params,
    ) -> Result<Self, CompileError> {
        let (style, rows) = spec.gather(style, &[("*", row)])?;
        let tp = [("*", &rows[0].1)];
        let table = PairRows::new(spec, &form.inputs(), &style, &tp)?;
        let cutoff = neighbour_cutoff(&spec.name, &style)?;
        let names = table.pair_names();
        let values = table
            .resolve("*", "*")?
            .into_iter()
            .map(|v| vec![v])
            .collect();
        Self::assemble(
            form,
            spec,
            &style,
            names,
            Vec::new(),
            Vec::new(),
            Source::Typed {
                type_id: None,
                ntypes: 1,
                table: values,
                cutoff2: cutoff * cutoff,
            },
        )
    }

    fn assemble(
        form: Arc<Fm>,
        spec: &StyleSpec,
        style: &Params,
        mut names: Vec<String>,
        atoms: Vec<AtomInput>,
        per_atom: Vec<Vec<F>>,
        source: Source,
    ) -> Result<Self, CompileError> {
        let style = StyleInputs::of(spec, style)?;
        names.extend(atoms.iter().map(|a| a.name.clone()));
        check_inputs(
            spec,
            &*form,
            names
                .iter()
                .cloned()
                .chain(style.nums.iter().map(|(n, _)| n.clone())),
        )?;
        style.check(spec, &*form)?;
        let n_paired = names.len() - atoms.len();
        let law_inputs = form
            .inputs()
            .iter()
            .filter(|name| !style.nums.iter().any(|(n, _)| n == *name))
            .map(|name| {
                let c = names.iter().position(|n| n == name)?;
                Some(if c < n_paired {
                    LawInput::Paired(c)
                } else {
                    LawInput::Atom(c - n_paired)
                })
            })
            .collect::<Option<Vec<_>>>()
            .filter(|inputs| inputs.len() <= MAX_LAW_INPUTS);
        Ok(Self {
            form,
            style,
            names,
            atoms,
            per_atom,
            source,
            law_inputs,
            strategy: Strategy {
                plain: Self::fold_rows::<false>,
                tallied: Self::fold_rows::<true>,
                compiled: Self::evaluate_batched,
                one: Self::one_batched,
            },
        })
    }

    /// How many columns are pair values (the rest are atom inputs).
    fn n_paired(&self) -> usize {
        self.names.len() - self.atoms.len()
    }

    /// The type-pair cell of the atoms `(i, j)` (0 for a one-type table).
    #[inline]
    fn cell(&self, i: usize, j: usize) -> usize {
        match &self.source {
            Source::Typed {
                type_id: Some(ids),
                ntypes,
                ..
            } => type_pair(ids[i], ids[j], *ntypes),
            _ => 0,
        }
    }

    /// Gather the inputs of the batch's first `n` pairs, whose `rows`, `ai`,
    /// `aj` and (a typed kernel) `cells` are filled: one pass per column.
    fn gather(&self, batch: &mut Batch, n: usize) {
        let (paired, atoms) = batch.cols.split_at_mut(self.n_paired());
        match &self.source {
            Source::Compiled { values, .. } => {
                for (col, v) in paired.iter_mut().zip(values) {
                    for (dst, &t) in col[..n].iter_mut().zip(&batch.rows[..n]) {
                        *dst = v[t];
                    }
                }
            }
            Source::Typed { table, .. } => {
                for (col, tab) in paired.iter_mut().zip(table) {
                    for (dst, &c) in col[..n].iter_mut().zip(&batch.cells[..n]) {
                        *dst = tab[c];
                    }
                }
            }
        }
        for (col, input) in atoms.iter_mut().zip(&self.atoms) {
            let values = &self.per_atom[input.values];
            let ends = if input.end == 0 {
                &batch.ai[..n]
            } else {
                &batch.aj[..n]
            };
            for (dst, &a) in col[..n].iter_mut().zip(ends) {
                *dst = values[a];
            }
        }
    }

    /// The form at the batch's first `n` pairs: `e`, `dE/dr` in `batch.e`,
    /// `batch.de`.
    fn eval(&self, batch: &mut Batch, n: usize) {
        let mut p = ParamColumns::new();
        for (name, col) in self.names.iter().zip(&batch.cols) {
            p.push(name, &col[..n]);
        }
        for ((name, v), b) in self.style.nums.iter().zip(&self.style.broadcast) {
            p.push(name, &b[..n]);
            p.push_style_num(name, *v);
        }
        for (name, t) in &self.style.texts {
            p.push_style_text(name, t);
        }
        self.form
            .eval(&batch.r[..n], &p, &mut batch.e[..n], &mut batch.de[..n]);
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
        let n_atoms = coords.len() / 3;
        let pairs: Vec<(usize, (usize, usize))> = match &self.source {
            Source::Compiled { atom_i, atom_j, .. } => (0..atom_i.len().min(limit))
                .map(|t| (t, (atom_i[t], atom_j[t])))
                .collect(),
            Source::Typed { .. } => (0..n_atoms)
                .flat_map(|i| ((i + 1)..n_atoms).map(move |j| (i, j)))
                .filter(|&(i, j)| dist(i, j) > 1e-6)
                .take(limit)
                .enumerate()
                .collect(),
        };
        let n = pairs.len().min(BATCH);
        let mut batch = Batch::new(self.names.len());
        for (k, &(row, (i, j))) in pairs.iter().take(n).enumerate() {
            (batch.rows[k], batch.ai[k], batch.aj[k]) = (row, i, j);
            batch.cells[k] = self.cell(i, j);
            batch.r[k] = dist(i, j);
        }
        self.gather(&mut batch, n);
        let mut params = TermParams::default();
        for (name, col) in self.names.iter().zip(&batch.cols) {
            params.nums.push((name.clone(), col[..n].to_vec()));
        }
        for (name, v) in &self.style.nums {
            params.nums.push((name.clone(), vec![*v; n]));
            params.style_nums.push((name.clone(), *v));
        }
        params.style_texts = self.style.texts.clone();
        Probe {
            params,
            q: batch.r[..n].to_vec(),
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
        let disp = disp
            .as_slice()
            .expect("a neighbour table's separations are one contiguous n × 3 block");
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        let mut energy: F = 0.0;
        let mut virial = Virial::ZERO;
        let mut batch = Batch::new(self.names.len());
        let mut next = rows.start;
        while next < rows.end {
            // The batch's pairs: those of weight not exactly zero inside the
            // cutoff. Exactly zero *skips*: a bonded pair sits at bond
            // length, where a repulsive term is enormous, and scaling it by
            // zero would be arithmetic on a number that should never have
            // been computed.
            let mut n = 0;
            {
                let (b_rows, b_ai, b_aj) =
                    (&mut batch.rows[..], &mut batch.ai[..], &mut batch.aj[..]);
                let (b_cells, b_r) = (&mut batch.cells[..], &mut batch.r[..]);
                while next < rows.end && n < BATCH {
                    let p = next;
                    next += 1;
                    if (!factor.is_empty() && factor[p] == 0.0) || d2[p] >= *cutoff2 {
                        continue;
                    }
                    let (i, j) = (i_col[p] as usize, j_col[p] as usize);
                    debug_assert!(
                        type_id.as_ref().is_none_or(|t| i < t.len() && j < t.len()),
                        "a pair names an atom the type table does not cover"
                    );
                    b_rows[n] = p;
                    b_ai[n] = i;
                    b_aj[n] = j;
                    b_cells[n] = self.cell(i, j);
                    b_r[n] = d2[p].sqrt();
                    n += 1;
                }
            }
            if n == 0 {
                continue;
            }
            self.gather(&mut batch, n);
            self.eval(&mut batch, n);
            energy += batch.scatter::<VIRIAL>(
                n,
                |p| if factor.is_empty() { 1.0 } else { factor[p] },
                |_, p| [disp[3 * p], disp[3 * p + 1], disp[3 * p + 2]],
                out,
                &mut virial,
            );
        }
        batch.pending.flush(out);
        (energy, virial)
    }

    /// The energy and the force on `j` of one pair of a uniform kernel at
    /// squared distance `r2` along `disp` (`x_j − x_i`), or `None` outside
    /// its cutoff or below 10⁻¹² of separation.
    pub(crate) fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        let Source::Typed { cutoff2, .. } = &self.source else {
            return None;
        };
        let r = r2.sqrt();
        if r2 >= *cutoff2 || r < MIN_R {
            return None;
        }
        Some((self.strategy.one)(self, r2, disp))
    }

    /// One pair of a uniform kernel at `r²` along `disp`, by the batch path.
    fn one_batched(&self, r2: F, disp: [F; 3]) -> (F, [F; 3]) {
        let r = r2.sqrt();
        let mut batch = Batch::new(self.names.len());
        batch.r[0] = r;
        self.gather(&mut batch, 1);
        self.eval(&mut batch, 1);
        let scale = -batch.de[0] / r;
        (
            batch.e[0],
            [scale * disp[0], scale * disp[1], scale * disp[2]],
        )
    }

    /// The compiled list at `coords`, by the batch path.
    fn evaluate_batched(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let Source::Compiled {
            atom_i,
            atom_j,
            weight,
            cutoff2,
            ..
        } = &self.source
        else {
            // A type table needs a pair table, and nobody handed one over.
            return;
        };
        let out = acc.forces_without_virial();
        let mut energy: F = 0.0;
        let mut batch = Batch::new(self.names.len());
        let mut disp = vec![[0.0; 3]; BATCH];
        // A compiled list is an intramolecular sum with no periodicity: a
        // virial from its raw differences would be a number about nothing.
        let mut untallied = Virial::ZERO;
        let mut next = 0;
        while next < atom_i.len() {
            let mut n = 0;
            while next < atom_i.len() && n < BATCH {
                let t = next;
                next += 1;
                let (i, j) = (atom_i[t], atom_j[t]);
                let d = [
                    coords[j * 3] - coords[i * 3],
                    coords[j * 3 + 1] - coords[i * 3 + 1],
                    coords[j * 3 + 2] - coords[i * 3 + 2],
                ];
                let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                // The rows inside the cutoff, as the typed fold selects them:
                // the form is never evaluated past it (where it need not be
                // defined), nor at a weight of exactly zero.
                if weight[t] == 0.0 || d2 >= *cutoff2 {
                    continue;
                }
                (batch.rows[n], batch.ai[n], batch.aj[n]) = (t, i, j);
                batch.r[n] = d2.sqrt();
                disp[n] = d;
                n += 1;
            }
            if n == 0 {
                continue;
            }
            self.gather(&mut batch, n);
            self.eval(&mut batch, n);
            energy += batch.scatter::<false>(n, |t| weight[t], |k, _| disp[k], out, &mut untallied);
        }
        batch.pending.flush(out);
        acc.add_energy(energy);
    }
}

/// Where a pointwise law's input is read, for the pairs of one fold.
#[derive(Clone, Copy)]
enum Src<'a> {
    /// A type-pair table, by the pair's cell.
    Cell(&'a [F]),
    /// A compiled list's column, by the pair's row.
    Row(&'a [F]),
    /// A per-atom column, by the pair's first atom.
    I(&'a [F]),
    /// A per-atom column, by the pair's second atom.
    J(&'a [F]),
}

impl Src<'_> {
    #[inline(always)]
    fn at(self, cell: usize, row: usize, i: usize, j: usize) -> F {
        match self {
            Src::Cell(v) => v[cell],
            Src::Row(v) => v[row],
            Src::I(v) => v[i],
            Src::J(v) => v[j],
        }
    }
}

impl<Fm: ScalarForm> ScalarPair<Fm> {
    /// This kernel evaluating pair by pair through its form's
    /// [`pointwise`](ScalarForm::pointwise) law — selection, gather, the force
    /// law and the scatter in one pass, the law inlined — when the form
    /// provides one and the kernel supplies each of its per-term inputs;
    /// unchanged (the batch path) otherwise.
    pub fn fused(mut self) -> Self {
        let has_law = self.form.pair_law(&self.style.cols()).is_some();
        if let (true, Some(inputs)) = (has_law, &self.law_inputs) {
            macro_rules! by_count {
                ($($n:literal)*) => {
                    match inputs.len() {
                        $($n => Some(Strategy {
                            plain: Self::fold_fused::<false, $n>,
                            tallied: Self::fold_fused::<true, $n>,
                            compiled: Self::evaluate_fused::<$n>,
                            one: Self::one_fused,
                        }),)*
                        _ => None,
                    }
                };
            }
            if let Some(strategy) = by_count!(0 1 2 3 4 5 6 7 8 9 10) {
                self.strategy = strategy;
            }
        }
        self
    }

    /// Where each input of the law is read.
    fn law_sources(&self) -> Vec<Src<'_>> {
        let inputs = self
            .law_inputs
            .as_deref()
            .expect("a fused kernel's law inputs");
        inputs
            .iter()
            .map(|input| match (*input, &self.source) {
                (LawInput::Paired(c), Source::Typed { table, .. }) => Src::Cell(&table[c]),
                (LawInput::Paired(c), Source::Compiled { values, .. }) => Src::Row(&values[c]),
                (LawInput::Atom(k), _) => {
                    let atom = &self.atoms[k];
                    let values = &self.per_atom[atom.values][..];
                    if atom.end == 0 {
                        Src::I(values)
                    } else {
                        Src::J(values)
                    }
                }
            })
            .collect()
    }

    /// One contiguous range of a neighbour table, into `out`, pair by pair.
    fn fold_fused<const VIRIAL: bool, const N: usize>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let Source::Typed {
            type_id,
            ntypes,
            cutoff2,
            ..
        } = &self.source
        else {
            unreachable!("only a typed kernel folds a neighbour table");
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (0.0, Virial::ZERO);
        };
        let disp = disp
            .as_slice()
            .expect("a neighbour table's separations are one contiguous n × 3 block");
        let (i_col, j_col) = (pairs.query_point_indices(), pairs.point_indices());
        let style = self.style.cols();
        let law = self
            .form
            .pair_law(&style)
            .expect("a fused kernel's form has a pair law");
        let sources: [Src<'_>; N] = self
            .law_sources()
            .try_into()
            .unwrap_or_else(|_| unreachable!("a fold instantiated for its law's input count"));
        let mut inputs = [0.0; N];
        let (cutoff2, ntypes) = (*cutoff2, *ntypes);
        let ids: &[u32] = type_id.as_deref().unwrap_or(&[]);
        let mut energy: F = 0.0;
        let mut virial = Virial::ZERO;
        for p in rows {
            // Exactly zero *skips*: a bonded pair sits at bond length, where
            // a repulsive term is enormous, and scaling it by zero would be
            // arithmetic on a number that should never have been computed.
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            if w == 0.0 || d2[p] >= cutoff2 {
                continue;
            }
            let (i, j) = (i_col[p] as usize, j_col[p] as usize);
            let cell = if ids.is_empty() {
                0
            } else {
                type_pair(ids[i], ids[j], ntypes)
            };
            for k in 0..N {
                inputs[k] = sources[k].at(cell, p, i, j);
            }
            let (e, fpair) = law(d2[p], &inputs);
            if d2[p] < MIN_R2 {
                if e.is_finite() {
                    energy += w * e;
                }
                continue;
            }
            energy += w * e;
            let scale = w * fpair;
            let d = [disp[3 * p], disp[3 * p + 1], disp[3 * p + 2]];
            let f = [scale * d[0], scale * d[1], scale * d[2]];
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

    /// The compiled list at `coords`, pair by pair.
    fn evaluate_fused<const N: usize>(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let Source::Compiled {
            atom_i,
            atom_j,
            weight,
            cutoff2,
            ..
        } = &self.source
        else {
            return;
        };
        let style = self.style.cols();
        let law = self
            .form
            .pair_law(&style)
            .expect("a fused kernel's form has a pair law");
        let sources: [Src<'_>; N] = self
            .law_sources()
            .try_into()
            .unwrap_or_else(|_| unreachable!("a fold instantiated for its law's input count"));
        let mut inputs = [0.0; N];
        let out = acc.forces_without_virial();
        let mut energy: F = 0.0;
        for (t, (&i, &j)) in atom_i.iter().zip(atom_j).enumerate() {
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            // The rows inside the cutoff, as the typed fold selects them: the
            // form is never evaluated past it, nor at a weight of exactly
            // zero.
            let w = weight[t];
            if w == 0.0 || d2 >= *cutoff2 {
                continue;
            }
            for k in 0..N {
                inputs[k] = sources[k].at(0, t, i, j);
            }
            let (e, fpair) = law(d2, &inputs);
            if d2 < MIN_R2 {
                if e.is_finite() {
                    energy += w * e;
                }
                continue;
            }
            energy += w * e;
            let scale = w * fpair;
            for c in 0..3 {
                let f = scale * d[c];
                out[j * 3 + c] += f;
                out[i * 3 + c] -= f;
            }
        }
        acc.add_energy(energy);
    }

    /// One pair of a uniform kernel at `r²` along `disp`, by the law.
    fn one_fused(&self, r2: F, disp: [F; 3]) -> (F, [F; 3]) {
        let style = self.style.cols();
        let law = self
            .form
            .pair_law(&style)
            .expect("a fused kernel's form has a pair law");
        let inputs: Vec<F> = self
            .law_sources()
            .iter()
            .map(|src| src.at(0, 0, 0, 0))
            .collect();
        let (e, fpair) = law(r2, &inputs);
        (e, [fpair * disp[0], fpair * disp[1], fpair * disp[2]])
    }
}

impl<Fm: ScalarForm + ?Sized> Potential for ScalarPair<Fm> {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        if let Source::Compiled { .. } = self.source {
            (self.strategy.compiled)(self, coords, acc);
        }
        // A type table needs a pair table, and nobody handed one over.
    }
}

impl<Fm: ScalarForm + ?Sized> PairDriven for ScalarPair<Fm> {
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
            |out: &mut [F], rows| (self.strategy.plain)(self, out, factor, pairs, rows),
            |out: &mut [F], rows| (self.strategy.tallied)(self, out, factor, pairs, rows),
        );
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }
}
