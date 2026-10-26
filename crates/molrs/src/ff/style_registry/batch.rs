//! A style's form evaluated on a batch of terms, exactly as a compile prices
//! them (unweighted, untruncated): [`Registry::batch_form`] →
//! [`BatchForm::evaluate`].
//!
//! The form is looked up under the registry and evaluated outside it: a
//! style's kernel is arbitrary code (a Python callable's), and one that
//! registered a style while the registry was locked would deadlock.

use std::fmt;

use ndarray::{ArrayD, Axis};

use super::{ExpressionForm, Kernel, Registry};
use crate::ff::ir::{CategorySpec, IrError, ParamKind, ParamValue, StyleSpec};
use crate::ff::potential::form_kernel::ParamColumns;
use molrs::op::F;

/// The kind of value a batch parameter takes ([`BatchForm::param_kind`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchParamKind {
    /// A number per term (a scalar broadcast).
    Num,
    /// A string per term.
    Text,
    /// One string for the style.
    StyleText,
    /// An array of this rank per term.
    Array { rank: u8 },
}

/// One parameter's values, one per term (a style text: one).
#[derive(Clone, Debug, PartialEq)]
pub enum BatchValue {
    Num(Vec<F>),
    Text(Vec<String>),
    StyleText(String),
    /// Stacked: the leading axis is the term.
    Array(ArrayD<F>),
}

/// The coordinates of a batch of terms.
#[derive(Clone, Copy, Debug)]
pub enum BatchCoordinates<'a> {
    /// The category's coordinate per term (a scalar form): `r`, `theta` or
    /// `phi` (radians).
    Q(&'a [F]),
    /// The atoms' positions, `arity` per term (a compound form).
    X(&'a [[F; 3]]),
}

/// What a batch evaluation returns beside the energies.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchGradient {
    /// `dE/dq` per term.
    DeDq(Vec<F>),
    /// `∂E/∂x` (not the force), `arity` points per term.
    Grad(Vec<[F; 3]>),
}

/// Why a batch does not evaluate.
#[derive(Clone, Debug, PartialEq)]
pub enum BatchError {
    /// The registry's refusal: an unknown category or style, or a parameter
    /// the form reads that is neither given nor defaulted.
    Ir(IrError),
    /// An input of the wrong kind: an undeclared parameter, or `Q` / `X`
    /// where the form takes the other.
    Input(String),
    /// An input of the wrong size, or a style with no batch form.
    Value(String),
}

impl fmt::Display for BatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ir(e) => write!(f, "{e}"),
            Self::Input(s) | Self::Value(s) => f.write_str(s),
        }
    }
}

impl std::error::Error for BatchError {}

/// A style's form, bound for batch evaluation.
#[derive(Clone)]
pub struct BatchForm {
    category: CategorySpec,
    spec: StyleSpec,
    form: ExpressionForm,
}

impl Registry {
    /// The form `(category, name)` evaluates on a batch: its expression or
    /// form kernel; a native constructor's expression.
    ///
    /// # Errors
    ///
    /// An unknown category or style, or a style with neither an expression
    /// nor a form kernel.
    pub fn batch_form(&self, category: &str, name: &str) -> Result<BatchForm, BatchError> {
        let cat = self.category(category).cloned().ok_or_else(|| {
            BatchError::Ir(IrError::UnknownCategory {
                category: category.to_owned(),
            })
        })?;
        let (spec, kernel) = self.style(category, name).ok_or_else(|| {
            BatchError::Ir(IrError::NoKernel {
                category: category.to_owned(),
                style: name.to_owned(),
            })
        })?;
        let form = match kernel {
            Some(Kernel::Expression(k)) => k.form(),
            Some(Kernel::Scalar(f)) => ExpressionForm::Scalar(f.clone()),
            Some(Kernel::Compound(f)) => ExpressionForm::Compound(f.clone()),
            // A native constructor builds a whole kernel from a frame; its
            // expression is what evaluates on a batch.
            Some(Kernel::Constructor { .. }) | None => {
                match (self.expression_compiler(), &spec.expression) {
                    (Some(compile), Some(_)) => compile(&cat, spec).map_err(BatchError::Ir)?.form(),
                    _ => {
                        return Err(BatchError::Value(format!(
                            "{category} `{name}` has no expression and no form kernel to \
                             evaluate on a batch (a native constructor prices whole frames: \
                             compile it)"
                        )));
                    }
                }
            }
        };
        Ok(BatchForm {
            category: cat,
            spec: spec.clone(),
            form,
        })
    }
}

impl BatchForm {
    fn who(&self) -> String {
        format!("{} `{}`", self.spec.category, self.spec.name)
    }

    fn is_pair(&self) -> bool {
        self.category.is_pair_driven()
    }

    /// Whether the form is priced over positions ([`BatchCoordinates::X`]).
    pub fn is_compound(&self) -> bool {
        matches!(self.form, ExpressionForm::Compound(_))
    }

    /// The atoms per term.
    pub fn arity(&self) -> usize {
        self.category.arity.endpoints()
    }

    /// The kind of value `key` takes: a declared parameter, a member
    /// `<name><m>` of an indexed family, or on a pair `q1`, `q2`, `<name>1`,
    /// `<name>2` (numbers); `None` for a key the style does not declare.
    pub fn param_kind(&self, key: &str) -> Option<BatchParamKind> {
        let (decl, style_level) = self.spec.declaration(key, self.is_pair())?;
        Some(match decl.map(|p| &p.kind) {
            Some(ParamKind::Text { .. }) if style_level => BatchParamKind::StyleText,
            Some(ParamKind::Text { .. }) => BatchParamKind::Text,
            Some(ParamKind::Array { rank }) => BatchParamKind::Array { rank: *rank },
            Some(ParamKind::Scalar) | None => BatchParamKind::Num,
        })
    }

    /// The energy of each term at `coords`, and its gradient.
    ///
    /// `params` gives each parameter as stored (angle values in degrees), one
    /// value per term ([`param_kind`](Self::param_kind) says which kind); a
    /// missing one takes its default, and on a pair `<x>1`, `<x>2` default to
    /// the pair value `<x>`.
    ///
    /// # Errors
    ///
    /// An undeclared parameter or the wrong coordinate kind
    /// ([`BatchError::Input`]); a value count other than the term count, or
    /// positions not `arity` per term ([`BatchError::Value`]); a parameter
    /// the form reads, not given and with no default
    /// ([`IrError::MissingParam`]).
    pub fn evaluate(
        &self,
        coords: BatchCoordinates<'_>,
        params: Vec<(String, BatchValue)>,
    ) -> Result<(Vec<F>, BatchGradient), BatchError> {
        let who = self.who();
        match (&self.form, coords) {
            (ExpressionForm::Scalar(f), BatchCoordinates::Q(q)) => {
                let n = q.len();
                let mut inputs = self.inputs(params, n)?;
                inputs.require(&self.spec, f.inputs(), self.is_pair())?;
                let (mut e, mut de) = (vec![0.0; n], vec![0.0; n]);
                inputs.with_cols(|p| f.eval(q, p, &mut e, &mut de));
                Ok((e, BatchGradient::DeDq(de)))
            }
            (ExpressionForm::Compound(f), BatchCoordinates::X(points)) => {
                let arity = self.arity();
                if arity == 0 || !points.len().is_multiple_of(arity) {
                    return Err(BatchError::Value(format!(
                        "{who}: {} positions are not {arity} per term",
                        points.len()
                    )));
                }
                let n = points.len() / arity;
                let mut inputs = self.inputs(params, n)?;
                inputs.require(&self.spec, f.inputs(), self.is_pair())?;
                let (mut e, mut grad) = (vec![0.0; n], vec![[0.0; 3]; n * arity]);
                inputs.with_cols(|p| f.eval(points, arity, p, &mut e, &mut grad));
                Ok((e, BatchGradient::Grad(grad)))
            }
            (ExpressionForm::Scalar(_), BatchCoordinates::X(_)) => Err(BatchError::Input(format!(
                "{who} is a function of its coordinate: pass q, not x"
            ))),
            (ExpressionForm::Compound(_), BatchCoordinates::Q(_)) => Err(BatchError::Input(
                format!("{who} is priced over its atoms' positions: pass x, not q"),
            )),
        }
    }

    /// `params` checked against the style for `n` terms, defaults filled in.
    fn inputs(&self, params: Vec<(String, BatchValue)>, n: usize) -> Result<Inputs, BatchError> {
        let who = self.who();
        let mut out = Inputs::default();
        for (key, value) in params {
            let Some(kind) = self.param_kind(&key) else {
                return Err(BatchError::Input(format!("{who} has no parameter `{key}`")));
            };
            let count = |len: usize, what: &str| {
                if len == n {
                    Ok(())
                } else {
                    Err(BatchError::Value(format!(
                        "{who}: `{key}`: {len} {what} for {n} terms"
                    )))
                }
            };
            match (kind, value) {
                (BatchParamKind::Num, BatchValue::Num(col)) => {
                    count(col.len(), "values")?;
                    out.nums.push((key, col));
                }
                (BatchParamKind::Text, BatchValue::Text(col)) => {
                    count(col.len(), "strings")?;
                    out.texts.push((key, col));
                }
                (BatchParamKind::StyleText, BatchValue::StyleText(s)) => {
                    out.style_texts.push((key, s));
                }
                (BatchParamKind::Array { rank }, BatchValue::Array(a)) => {
                    if a.ndim() != rank as usize + 1 || a.shape()[0] != n {
                        return Err(BatchError::Value(format!(
                            "{who}: `{key}`: {n} arrays of rank {rank} stacked; got shape {:?}",
                            a.shape()
                        )));
                    }
                    out.arrays.push((key, a));
                }
                (kind, _) => {
                    return Err(BatchError::Input(format!(
                        "{who}: `{key}` takes a {kind:?} value"
                    )));
                }
            }
        }
        let spec = &self.spec;
        for p in spec.params.iter().chain(&spec.style_params) {
            let style_level = spec.style_param(&p.name).is_some();
            let (Some(default), false) = (&p.default, out.has(&p.name) || p.indexed) else {
                continue;
            };
            let name = p.name.to_string();
            match default {
                ParamValue::Num(v) => out.nums.push((name, vec![*v; n])),
                ParamValue::Text(t) if style_level => out.style_texts.push((name, t.to_string())),
                ParamValue::Text(t) => out.texts.push((name, vec![t.to_string(); n])),
                ParamValue::Array(a) => {
                    let views = vec![a.view(); n];
                    let stacked = ndarray::stack(Axis(0), &views).map_err(|e| {
                        BatchError::Value(format!("{who}: default of `{}`: {e}", p.name))
                    })?;
                    out.arrays.push((name, stacked));
                }
            }
        }
        Ok(out)
    }
}

/// The checked inputs of one batch, owned.
#[derive(Default)]
struct Inputs {
    nums: Vec<(String, Vec<F>)>,
    arrays: Vec<(String, ArrayD<F>)>,
    texts: Vec<(String, Vec<String>)>,
    style_texts: Vec<(String, String)>,
}

impl Inputs {
    fn has(&self, name: &str) -> bool {
        self.nums.iter().any(|(n, _)| n == name)
            || self.arrays.iter().any(|(n, _)| n == name)
            || self.texts.iter().any(|(n, _)| n == name)
            || self.style_texts.iter().any(|(n, _)| n == name)
    }

    fn num(&self, name: &str) -> Option<&Vec<F>> {
        self.nums.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }

    /// Refuse a missing input: those the form states it reads (an
    /// expression), else every declared parameter without a default (a
    /// Python kernel). A pair's self rows `<x>1`, `<x>2` default to the pair
    /// value `<x>`.
    fn require(
        &mut self,
        spec: &StyleSpec,
        reads: Vec<String>,
        pair: bool,
    ) -> Result<(), BatchError> {
        let missing = |param: &str| {
            BatchError::Ir(IrError::MissingParam {
                style: spec.name.to_string(),
                type_: String::new(),
                param: param.to_owned(),
            })
        };
        if reads.is_empty() {
            let needed = spec
                .params
                .iter()
                .chain(&spec.style_params)
                .filter(|p| !p.indexed);
            for p in needed {
                if !self.has(&p.name) {
                    return Err(missing(&p.name));
                }
            }
            return Ok(());
        }
        for name in reads {
            if self.num(&name).is_some() {
                continue;
            }
            let pair_value = pair
                .then(|| name.strip_suffix(['1', '2']))
                .flatten()
                .and_then(|base| self.num(base).cloned());
            match pair_value {
                Some(col) => self.nums.push((name, col)),
                None => return Err(missing(&name)),
            }
        }
        Ok(())
    }

    /// `f` over the inputs as one batch's [`ParamColumns`].
    fn with_cols<R>(&self, f: impl FnOnce(&ParamColumns<'_>) -> R) -> R {
        let texts: Vec<Vec<&str>> = self
            .texts
            .iter()
            .map(|(_, t)| t.iter().map(String::as_str).collect())
            .collect();
        let mut cols = ParamColumns::new();
        for (name, col) in &self.nums {
            cols.push(name, col);
        }
        for (name, a) in &self.arrays {
            cols.push_array(name, a.view());
        }
        for ((name, _), t) in self.texts.iter().zip(&texts) {
            cols.push_text(name, t);
        }
        for (name, t) in &self.style_texts {
            cols.push_style_text(name, t);
        }
        f(&cols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn harmonic() -> BatchForm {
        Registry::builtin().batch_form("bond", "harmonic").unwrap()
    }

    /// `bond harmonic` on two lengths prices `k (r − r0)²` and its
    /// derivative.
    #[test]
    fn a_scalar_form_prices_each_term() {
        let (e, de) = harmonic()
            .evaluate(
                BatchCoordinates::Q(&[1.0, 1.5]),
                vec![
                    ("k".into(), BatchValue::Num(vec![300.0; 2])),
                    ("r0".into(), BatchValue::Num(vec![1.2; 2])),
                ],
            )
            .unwrap();
        assert!(
            (e[0] - 12.0).abs() < 1e-12 && (e[1] - 27.0).abs() < 1e-12,
            "{e:?}"
        );
        let BatchGradient::DeDq(de) = de else {
            panic!("a scalar form returns dE/dq")
        };
        assert!((de[0] - 2.0 * 300.0 * -0.2).abs() < 1e-9, "{de:?}");
    }

    #[test]
    fn inputs_are_refused_by_kind_and_count() {
        let form = harmonic();
        let k = || ("k".to_owned(), BatchValue::Num(vec![300.0]));
        let missing = form
            .evaluate(BatchCoordinates::Q(&[1.0]), vec![k()])
            .unwrap_err();
        assert!(
            matches!(&missing, BatchError::Ir(IrError::MissingParam { param, .. }) if param == "r0"),
            "{missing:?}"
        );
        let undeclared = form
            .evaluate(
                BatchCoordinates::Q(&[1.0]),
                vec![k(), ("bogus".into(), BatchValue::Num(vec![1.0]))],
            )
            .unwrap_err();
        assert!(matches!(undeclared, BatchError::Input(_)), "{undeclared:?}");
        let short = form
            .evaluate(
                BatchCoordinates::Q(&[1.0, 2.0]),
                vec![k(), ("r0".into(), BatchValue::Num(vec![1.0, 1.0]))],
            )
            .unwrap_err();
        assert!(matches!(short, BatchError::Value(_)), "{short:?}");
        let positions = form
            .evaluate(BatchCoordinates::X(&[[0.0; 3]; 2]), vec![])
            .unwrap_err();
        assert!(matches!(positions, BatchError::Input(_)), "{positions:?}");
        let unknown = Registry::builtin().batch_form("bond", "nope").map(|_| ());
        assert!(matches!(
            unknown,
            Err(BatchError::Ir(IrError::NoKernel { .. }))
        ));
    }
}
