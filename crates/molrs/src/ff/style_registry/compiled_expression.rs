//! The expression engine ([`expr`]) as a registry
//! kernel: [`ExpressionKernel`] over a compiled expression, the
//! [`ExpressionCompiler`](crate::ff::style_registry::ExpressionCompiler) every
//! [`Registry::builtin`](crate::ff::style_registry::Registry::builtin) installs, and the
//! spec an unregistered style that carries an expression is priced under.

use std::sync::Arc;

use crate::ff::ir::Params;
use crate::ff::ir::expression::{self, Binding, Compiled, ExpressionError, Geometry};
use crate::ff::ir::{
    CategorySpec, Coordinate, IrError, ParamCombination, ParamDimension, ParamKind, ParamSpec,
    StyleSpec,
};
use crate::ff::ir::{ParamSource, SpecialClass};
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm};
use crate::ff::style_registry::{ExpressionForm, ExpressionKernel};
use molrs::op::F;

/// Members an indexed family binds in an expression: `k1 … k16`. A table
/// whose rows have fewer leaves the rest unread (an expression that reads
/// one is refused at its first compile, [`IrError::MissingParam`]).
const INDEXED_MEMBERS: usize = 16;

/// A compiled [`expression::Compiled`] as the registry's [`ExpressionKernel`].
#[derive(Clone, Debug)]
pub struct CompiledExpression(pub Arc<Compiled>);

impl ExpressionKernel for CompiledExpression {
    fn source(&self) -> &str {
        self.0.source()
    }

    fn variables(&self) -> Vec<String> {
        self.0.variables()
    }

    fn form(&self) -> ExpressionForm {
        if self.0.has_scalar_form() {
            ExpressionForm::Scalar(Arc::new(ScalarProgram(self.0.clone())))
        } else {
            ExpressionForm::Compound(Arc::new(CompoundProgram(self.0.clone())))
        }
    }
}

/// The columns a compiled expression reads, by spelling.
fn spellings(c: &Compiled) -> Vec<String> {
    c.inputs().iter().map(expression::Input::spelling).collect()
}

/// An expression of the category's coordinate (one dual per term).
struct ScalarProgram(Arc<Compiled>);

impl ScalarForm for ScalarProgram {
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de_dq: &mut [F]) {
        // The kernels refuse, at build, a form whose inputs they cannot
        // supply (`inputs`), so a missing column here is a kernel's bug.
        self.0
            .eval_scalar_named(q, |n| p.get(n), e, de_dq)
            .unwrap_or_else(|err| panic!("expression `{}`: {err}", self.0.source()));
    }

    fn inputs(&self) -> Vec<String> {
        spellings(&self.0)
    }
}

/// An expression of the term's points (3·arity duals per term).
struct CompoundProgram(Arc<Compiled>);

impl CompoundForm for CompoundProgram {
    fn eval(
        &self,
        x: &[[F; 3]],
        _arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        self.0
            .eval_compound_named(x, |n| p.get(n), e, grad)
            .unwrap_or_else(|err| panic!("expression `{}`: {err}", self.0.source()));
    }

    fn inputs(&self) -> Vec<String> {
        spellings(&self.0)
    }
}

/// The expression engine's geometry for a category.
fn geometry(category: &CategorySpec) -> Result<Geometry, IrError> {
    if category.is_pair_driven() {
        return Ok(Geometry::Pair);
    }
    Ok(match category.coordinate {
        Coordinate::Distance => Geometry::Bond,
        Coordinate::Angle => Geometry::Angle,
        Coordinate::Dihedral => Geometry::Dihedral,
        Coordinate::Improper => Geometry::Improper,
        Coordinate::Compound => Geometry::Compound {
            arity: category.arity.endpoints(),
        },
        Coordinate::None => {
            return Err(IrError::CoordinateMismatch {
                category: category.name.to_string(),
                kernel: "expression".into(),
            });
        }
    })
}

/// The numeric names of `params` an expression may read, an indexed family
/// as its members.
fn numeric(params: &[ParamSpec]) -> Vec<String> {
    let mut out = Vec::new();
    for p in params.iter().filter(|p| p.kind == ParamKind::Scalar) {
        if p.indexed {
            out.extend((1..=INDEXED_MEMBERS).map(|m| format!("{}{m}", p.name)));
        } else {
            out.push(p.name.to_string());
        }
    }
    out
}

/// An [`ExpressionError`] as the [`IrError`] the protocol names for it.
fn ir_error(spec: &StyleSpec, source: &str, e: ExpressionError) -> IrError {
    let style = spec.name.to_string();
    let parse = |at: usize, e: &ExpressionError| IrError::Parse {
        expression: source.to_owned(),
        at,
        reason: e.to_string(),
    };
    match e {
        ExpressionError::UnexpectedChar { pos, .. }
        | ExpressionError::UnexpectedToken { pos, .. }
        | ExpressionError::BadDefinition { pos, .. } => parse(pos, &e),
        ExpressionError::UnexpectedEnd { .. }
        | ExpressionError::EmptyExpression
        | ExpressionError::CyclicDefinition { .. }
        | ExpressionError::DefinitionOrder { .. }
        | ExpressionError::DuplicateDefinition { .. }
        | ExpressionError::DefinitionShadows { .. }
        | ExpressionError::PointAsNumber { .. } => parse(0, &e),
        ExpressionError::UnknownFunction { name } => IrError::UnknownFunction { name },
        ExpressionError::FunctionArity {
            name,
            expected,
            found,
        } => IrError::FunctionArity {
            name,
            given: found,
            expected,
        },
        ExpressionError::UndeclaredVariable { name, .. } => {
            IrError::UnboundVariable { style, name }
        }
        ExpressionError::NotAPoint { found, points, .. } => IrError::Point {
            style,
            point: found,
            arity: points,
        },
        ExpressionError::BadBinding { reason } => IrError::ReservedParam {
            style,
            param: reason,
        },
        ExpressionError::MissingInput { input } => IrError::KernelShape {
            style,
            reason: format!("no column for the expression's input `{input}`"),
        },
    }
}

/// Compile `spec`'s expression for `category`: the
/// [`ExpressionCompiler`](crate::ff::style_registry::ExpressionCompiler) the built-in
/// registries carry. Only numeric parameters are variables (D9).
pub fn compile_expression(
    category: &CategorySpec,
    spec: &StyleSpec,
) -> Result<Arc<dyn ExpressionKernel>, IrError> {
    let source = spec
        .expression
        .as_deref()
        .ok_or_else(|| IrError::NoKernel {
            category: category.name.to_string(),
            style: spec.name.to_string(),
        })?;
    let (params, style_params) = (numeric(&spec.params), numeric(&spec.style_params));
    let params: Vec<&str> = params.iter().map(String::as_str).collect();
    let style_params: Vec<&str> = style_params.iter().map(String::as_str).collect();
    let binding = Binding::new(geometry(category)?, &params, &style_params);
    let compiled = expression::compile(source, &binding).map_err(|e| ir_error(spec, source, e))?;
    Ok(Arc::new(CompiledExpression(Arc::new(compiled))))
}

/// The spec an **unregistered** style is priced under when it carries an
/// `expression` (the compile fallback, protocol §4): every numeric
/// parameter its type rows and style carry, `epsilon`/`sigma` (and
/// `epsilon14`/`sigma14`) mixing by the style's `mixing` on a pair, every
/// other parameter not mixing; a pair's special class from its `special`
/// style param (`coul`, else `lj`).
pub(crate) fn fallback_spec(
    category: &CategorySpec,
    name: &str,
    style: &Params,
    tp: &[(&str, &Params)],
    expression: &str,
) -> StyleSpec {
    // Compile-time projections, not the style's own parameters.
    const INJECTED: [&str; 2] = ["lj14scale", "coulomb14scale"];
    let mut per_type: Vec<String> = Vec::new();
    for (_, row) in tp {
        let mut keys: Vec<&str> = row
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .map(|(k, _)| k)
            .collect();
        keys.sort_unstable();
        for k in keys {
            if !per_type.iter().any(|p| p == k) {
                per_type.push(k.to_owned());
            }
        }
    }
    let pair = category.is_pair_driven();
    let lj = |e: &str, s: &str, p: &str| -> ParamCombination {
        let has = |n: &str| per_type.iter().any(|q| q == n);
        match p {
            _ if !pair || !(has(e) && has(s)) => ParamCombination::None,
            x if x == e => ParamCombination::LjEpsilon {
                sigma: s.to_owned().into(),
            },
            x if x == s => ParamCombination::LjSigma {
                epsilon: e.to_owned().into(),
            },
            _ => ParamCombination::None,
        }
    };
    let params = per_type
        .iter()
        .map(|p| {
            let mix = match lj("epsilon", "sigma", p) {
                ParamCombination::None => lj("epsilon14", "sigma14", p),
                m => m,
            };
            ParamSpec::new(p.clone(), ParamDimension::NONE).mix(mix)
        })
        .collect();
    let mut style_names: Vec<&str> = style
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        .map(|(k, _)| k)
        .filter(|k| !INJECTED.contains(k) && !per_type.iter().any(|p| p == k))
        .collect();
    style_names.sort_unstable();
    let mut spec = StyleSpec::new(category.name.clone(), name.to_owned())
        .params(params)
        .style_params(
            style_names
                .into_iter()
                .map(|k| ParamSpec::new(k.to_owned(), ParamDimension::NONE))
                .collect(),
        )
        .expression(expression);
    if pair {
        spec = spec.special(match style.get("special").and_then(|v| v.as_text()) {
            Some("coul") => SpecialClass::Coulomb,
            _ => SpecialClass::Vdw,
        });
    }
    if tp.is_empty() {
        spec = spec.source(ParamSource::PerInstance);
    }
    spec
}
