//! Form kernels: everything a kernel does besides the force law, written
//! once over a [`ScalarForm`] or a [`CompoundForm`].
//!
//! A style registered with a form
//! ([`Kernel::Scalar`](crate::ff::style_registry::Kernel::Scalar),
//! [`Kernel::Compound`](crate::ff::style_registry::Kernel::Compound), or an
//! expression's) is built into one of these by
//! [`PotentialCompiler`](crate::ff::compile::PotentialCompiler). They are
//! the column fetch, type lookup, 1-4 scaling, type-pair table and
//! periodic-copy bookkeeping a hand-written kernel (`pair/morse.rs`)
//! repeats, so a third party writes only the energy:
//!
//! * [`ScalarBonded`] — the rows of a bonded category's block, the chain rule
//!   from `r`, `theta` or `phi` onto the atoms.
//! * [`ScalarPair`] — a pair style at both compile doors: a fixed pair list,
//!   or a type-pair table over a neighbour search with per-parameter mixing,
//!   cross rows, `cutoff`, special-bonds weights and the virial.
//! * [`CompoundTerms`] — an N-body term over any block.

mod bonded;
mod compound;
mod form;
mod pair;
#[cfg(test)]
mod tests;

pub use bonded::ScalarBonded;
pub use compound::CompoundTerms;
pub use form::{CompoundForm, ParamColumns, ScalarForm};
pub use pair::ScalarPair;

use std::collections::HashMap;
use std::ops::Range;

use ndarray::{ArrayD, Axis, Slice};

use crate::ff::ir::ParamSource;
use crate::ff::ir::Params;
use crate::ff::ir::{IrError, ParamKind, ParamSpec, StyleSpec};
use molrs::core::Frame;
use molrs::core::keys::ENDPOINTS;
use molrs::op::F;

/// One concrete column of a style: a declared parameter, or one member
/// `<name><m>` of an indexed family.
#[derive(Clone, Debug)]
pub(crate) struct StyleParamColumn {
    pub name: String,
    /// The declared parameter it comes from.
    pub param: usize,
    /// `Some(m)` for the m-th member of an indexed family.
    pub index: Option<usize>,
}

/// The highest `m` with `<base>1 … <base>m` all present in `row`, and
/// whether a higher one is present too (a gap).
fn contiguous(row: &Params, base: &str) -> (usize, bool) {
    let mut m = 0;
    while row
        .get(&format!("{base}{}", m + 1))
        .and_then(|v| v.as_num())
        .is_some()
    {
        m += 1;
    }
    let gap = row
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        .any(|(key, _)| {
            key.strip_prefix(base)
                .and_then(|n| n.parse::<usize>().ok())
                .is_some_and(|n| n > m)
        });
    (m, gap)
}

/// The concrete columns of `params` over `rows`: every declared parameter,
/// an indexed family expanded to `<name>1 … <name>M` with one `M` for the
/// table (the longest family any row has; a gap is refused).
pub(crate) fn columns(
    spec: &StyleSpec,
    params: &[ParamSpec],
    rows: &[(&str, &Params)],
) -> Result<Vec<StyleParamColumn>, IrError> {
    let mut m_table = 0;
    for p in params.iter().filter(|p| p.indexed) {
        for (label, row) in rows {
            let (m, gap) = contiguous(row, &p.name);
            let bare =
                spec.unindexed_one_term && row.get(&p.name).and_then(|v| v.as_num()).is_some();
            if gap || (bare && m > 0) {
                return Err(IrError::MissingParam {
                    style: spec.name.to_string(),
                    type_: label.to_string(),
                    param: format!("{}{}", p.name, m + 1),
                });
            }
            m_table = m_table.max(if bare { 1 } else { m });
        }
    }
    let mut out = Vec::new();
    for (c, p) in params.iter().enumerate() {
        if p.indexed {
            out.extend((1..=m_table).map(|m| StyleParamColumn {
                name: format!("{}{m}", p.name),
                param: c,
                index: Some(m),
            }));
        } else {
            out.push(StyleParamColumn {
                name: p.name.to_string(),
                param: c,
                index: None,
            });
        }
    }
    Ok(out)
}

/// `cols` cut to those a form reads, when it states what it reads
/// ([`ScalarForm::inputs`]): a parameter it does not read is not required.
/// A column is read by its name, or on a pair by its self-row spelling
/// (`<x>1`, `<x>2`) or as the mixing partner of a column that is.
pub(crate) fn retain_form_inputs(
    spec: &StyleSpec,
    cols: Vec<StyleParamColumn>,
    reads: &[String],
) -> Vec<StyleParamColumn> {
    if reads.is_empty() {
        return cols;
    }
    let read = |c: &StyleParamColumn| {
        reads
            .iter()
            .any(|r| r == &c.name || r.strip_suffix(['1', '2']) == Some(c.name.as_str()))
    };
    let partner = |c: &StyleParamColumn| match &spec.params[c.param].mix {
        crate::ff::ir::ParamCombination::LjEpsilon { sigma: p }
        | crate::ff::ir::ParamCombination::LjSigma { epsilon: p } => Some(p.clone()),
        _ => None,
    };
    let kept: Vec<bool> = cols.iter().map(read).collect();
    cols.iter()
        .enumerate()
        .filter(|&(k, c)| {
            kept[k]
                || cols.iter().zip(&kept).any(|(other, &on)| {
                    on && partner(other).is_some_and(|p| spec.params[c.param].name == p)
                })
        })
        .map(|(_, c)| c.clone())
        .collect()
}

/// `column`'s numeric value in `row`, by its own name or — term 1 of a style
/// that accepts it — the bare name of its family.
pub(crate) fn row_num(spec: &StyleSpec, col: &StyleParamColumn, row: &Params) -> Option<F> {
    row.get(&col.name).and_then(|v| v.as_num()).or_else(|| {
        (col.index == Some(1) && spec.unindexed_one_term)
            .then(|| {
                row.get(&spec.params[col.param].name)
                    .and_then(|v| v.as_num())
            })
            .flatten()
    })
}

/// The per-term inputs of a kernel's terms, owned; [`ParamColumns`] borrows
/// them a batch at a time.
#[derive(Clone, Debug, Default)]
pub(crate) struct TermParams {
    pub nums: Vec<(String, Vec<F>)>,
    /// Stacked, leading axis the terms.
    pub arrays: Vec<(String, ArrayD<F>)>,
    pub texts: Vec<(String, Vec<String>)>,
    pub style_texts: Vec<(String, String)>,
}

impl TermParams {
    /// The inputs of terms `range` as a [`ParamColumns`], for `f`.
    pub fn with_cols<R>(&self, range: Range<usize>, f: impl FnOnce(&ParamColumns<'_>) -> R) -> R {
        let texts: Vec<Vec<&str>> = self
            .texts
            .iter()
            .map(|(_, t)| t[range.clone()].iter().map(String::as_str).collect())
            .collect();
        let mut cols = ParamColumns::new();
        for (name, col) in &self.nums {
            cols.push(name, &col[range.clone()]);
        }
        for (name, a) in &self.arrays {
            cols.push_array(name, a.slice_axis(Axis(0), Slice::from(range.clone())));
        }
        for ((name, _), t) in self.texts.iter().zip(&texts) {
            cols.push_text(name, t);
        }
        for (name, t) in &self.style_texts {
            cols.push_style_text(name, t);
        }
        f(&cols)
    }

    /// The terms `picked`, in that order.
    pub fn select(&self, picked: &[usize]) -> Self {
        Self {
            nums: self
                .nums
                .iter()
                .map(|(n, c)| (n.clone(), picked.iter().map(|&t| c[t]).collect()))
                .collect(),
            arrays: self
                .arrays
                .iter()
                .map(|(n, a)| (n.clone(), a.select(Axis(0), picked)))
                .collect(),
            texts: self
                .texts
                .iter()
                .map(|(n, c)| (n.clone(), picked.iter().map(|&t| c[t].clone()).collect()))
                .collect(),
            style_texts: self.style_texts.clone(),
        }
    }

    /// Refuse a form input no numeric column supplies.
    pub fn require(&self, spec: &StyleSpec, inputs: &[String]) -> Result<(), IrError> {
        match inputs
            .iter()
            .find(|i| !self.nums.iter().any(|(n, _)| n == *i))
        {
            Some(param) => Err(IrError::MissingParam {
                style: spec.name.to_string(),
                type_: String::new(),
                param: param.clone(),
            }),
            None => Ok(()),
        }
    }

    /// Add the style's declared numeric parameters as `n`-long columns, and
    /// its text ones, from `style` as [`StyleSpec::gather`] filled it.
    pub fn add_style(&mut self, spec: &StyleSpec, style: &Params, n: usize) -> Result<(), IrError> {
        for p in &spec.style_params {
            let missing = || IrError::MissingParam {
                style: spec.name.to_string(),
                type_: String::new(),
                param: p.name.to_string(),
            };
            match &p.kind {
                ParamKind::Scalar => {
                    let v = style
                        .get(&p.name)
                        .and_then(|v| v.as_num())
                        .ok_or_else(missing)?;
                    self.nums.push((p.name.to_string(), vec![v; n]));
                }
                ParamKind::Text { .. } => {
                    let v = style
                        .get(&p.name)
                        .and_then(|v| v.as_text())
                        .ok_or_else(missing)?;
                    self.style_texts.push((p.name.to_string(), v.to_owned()));
                }
                ParamKind::Array { .. } => {
                    return Err(IrError::Malformed {
                        style: spec.name.to_string(),
                        reason: format!(
                            "style param `{}` is an array; a style param is one value",
                            p.name
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

/// The terms of `block` and each term's parameters: `arity` atom-index
/// columns, and every declared per-type and style parameter.
///
/// A numeric per-type value is, in order: the block's own column of that
/// name where it is not null (molrec linking rule 4, how a per-instance
/// style carries its numbers), the row's type in `tp` (gathered:
/// [`StyleSpec::gather`] filled its defaults), the parameter's default for a
/// term with no row; none of them is [`IrError::MissingParam`]. A table-driven style
/// needs every row's type in `tp`; a per-instance one only where it reads
/// one.
pub(crate) fn resolve_terms(
    spec: &StyleSpec,
    reads: &[String],
    block_name: &str,
    arity: usize,
    style: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<(Vec<Vec<usize>>, TermParams), crate::ff::potential::CompileError> {
    let who = format!("{} `{}`", spec.category, spec.name);
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(block_name)
        .ok_or_else(|| format!("{who}: frame missing \"{block_name}\" block"))?;
    let mut atoms = Vec::with_capacity(arity);
    for key in &ENDPOINTS[..arity] {
        let col = block
            .get(key)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| format!("{who}: {block_name} block missing \"{key}\" column"))?;
        atoms.push(col.iter().map(|&a| a as usize).collect::<Vec<_>>());
    }
    let n = atoms.first().map_or(0, Vec::len);
    let types = block.get("type").and_then(|c| c.get::<String>());
    if types.is_none() && spec.source == ParamSource::TypeRows {
        return Err(format!("{who}: {block_name} block missing \"type\" column").into());
    }
    let mut rows = Vec::with_capacity(n);
    for t in 0..n {
        let label = types.map_or("", |ty| ty[t].as_str());
        let row = type_map.get(label).copied();
        if row.is_none() && spec.source == ParamSource::TypeRows {
            return Err(format!("{who}: unknown {block_name} type '{label}'").into());
        }
        rows.push((label, row));
    }
    let defaults = spec.default_row();
    let present: Vec<(&str, &Params)> = rows
        .iter()
        .filter_map(|&(l, r)| r.map(|r| (l, r)))
        .collect();
    let cols = retain_form_inputs(spec, columns(spec, &spec.params, &present)?, reads);
    let missing = |label: &str, name: &str| IrError::MissingParam {
        style: spec.name.to_string(),
        type_: label.to_owned(),
        param: name.to_owned(),
    };
    let mut out = TermParams::default();
    for col in &cols {
        let decl = &spec.params[col.param];
        match &decl.kind {
            ParamKind::Scalar => {
                let own = block.get(&col.name).and_then(|c| c.get::<f64>());
                let valid = block.validity(&col.name);
                let mut values = Vec::with_capacity(n);
                for (t, &(label, row)) in rows.iter().enumerate() {
                    let cell = own.and_then(|c| valid.is_none_or(|m| m[t]).then(|| c[t]));
                    let v = cell
                        .or_else(|| row_num(spec, col, row.unwrap_or(&defaults)))
                        .ok_or_else(|| missing(label, &col.name))?;
                    values.push(v);
                }
                out.nums.push((col.name.clone(), values));
            }
            ParamKind::Text { .. } => {
                let mut values = Vec::with_capacity(n);
                for &(label, row) in &rows {
                    let v = row
                        .unwrap_or(&defaults)
                        .get(&col.name)
                        .and_then(|v| v.as_text())
                        .ok_or_else(|| missing(label, &col.name))?;
                    values.push(v.to_owned());
                }
                out.texts.push((col.name.clone(), values));
            }
            ParamKind::Array { rank } => {
                let mut views = Vec::with_capacity(n);
                for &(label, row) in &rows {
                    let a = row
                        .and_then(|r| r.get(&col.name).and_then(|v| v.as_array()))
                        .ok_or_else(|| missing(label, &col.name))?;
                    if a.ndim() != *rank as usize {
                        return Err(format!(
                            "{who} type '{label}': `{}` has rank {}, the spec says {rank}",
                            col.name,
                            a.ndim()
                        )
                        .into());
                    }
                    views.push(a.view());
                }
                let stacked = if views.is_empty() {
                    ArrayD::zeros(vec![0; *rank as usize + 1])
                } else {
                    ndarray::stack(Axis(0), &views).map_err(|_| {
                        format!("{who}: `{}` has rows of different shapes", col.name)
                    })?
                };
                out.arrays.push((col.name.clone(), stacked));
            }
        }
    }
    out.add_style(spec, style, n)?;
    Ok((atoms, out))
}

/// The points a numeric check evaluates a form at: terms' inputs, and their
/// coordinates (`q`, scalar forms) or positions (`x`, compound forms).
#[derive(Clone, Debug, Default)]
pub(crate) struct Probe {
    pub params: TermParams,
    pub q: Vec<F>,
    pub x: Vec<[F; 3]>,
    pub arity: usize,
}

impl Probe {
    /// How many terms the probe holds.
    pub(crate) fn n(&self) -> usize {
        if self.x.is_empty() {
            self.q.len()
        } else {
            self.x.len() / self.arity.max(1)
        }
    }
}
