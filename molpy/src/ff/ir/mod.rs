//! `molpy.ff.ir`: the force-field IR's vocabulary, from Python.
//!
//! The IR (`molrs::ff::ir`) states a category, a style with ordered
//! parameters and their dimensions, and its energy as data:
//!
//! * `IrError` and one subclass per Rust variant (D17) — every refusal;
//! * `ParamSpec` — one parameter (Rust `ParamSpec`), its dimension parsed at once;
//! * `CategorySpec` / `StyleSpec` — a registered category and style, as
//!   `molpy.ff.style_registry.categories` / `styles` list them.
//!
//! Registering is `molpy.ff.style_registry`'s ([`crate::ff::style_registry`]).
//! The Python face is `python/molrs/ff/ir.py`; a style's engine forms are
//! [`engine`]'s.

pub(crate) mod engine;

use std::borrow::Cow;

use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use molrs::ff::forcefield::DefError;
use molrs::ff::ir::{
    Arity, CategorySpec, ConformanceSample, Coordinate, EndpointOrder, IrError, ParamCombination,
    ParamDimension, ParamKind, ParamSource, ParamSpec, ParamValue, SpecialClass, StyleSpec,
};
use molrs::ff::potential::CompileError;
use molrs::ff::style_registry::{Kernel, Registry};
use molrs::op::F;

// ---------------------------------------------------------------------------
// Errors: `IrError(ValueError)` and one subclass per Rust variant (D17)
// ---------------------------------------------------------------------------

/// The exception classes of `molpy.ff.ir`.
pub mod errors {
    use pyo3::create_exception;
    use pyo3::prelude::*;

    create_exception!(
        molpy.ff.ir,
        IrError,
        pyo3::exceptions::PyValueError,
        "The force-field IR refused a category, a style, a kernel or a term. \
         Each refusal is raised as the subclass named after the Rust \
         `molrs::ff::ir::IrError` variant (`Sealed`, `NoKernel`, …), its \
         message naming the item; the variant's fields are attributes \
         (`err.category`, `err.style`, `err.param`, …). Subclasses \
         `ValueError`."
    );

    macro_rules! variants {
        ($($name:ident: $doc:literal),* $(,)?) => {
            $(create_exception!(molpy.ff.ir, $name, IrError, $doc);)*

            /// Add `IrError` and every variant class to `m`.
            pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
                let py = m.py();
                m.add("IrError", py.get_type::<IrError>())?;
                $(m.add(stringify!($name), py.get_type::<$name>())?;)*
                Ok(())
            }
        };
    }

    variants! {
        UnknownCategoryError: "A style in a category neither built in nor registered.",
        BadNameError: "A category or parameter name outside its pattern.",
        ArityError: "A custom category's arity outside 2..=5, or a type with the wrong number of endpoints.",
        BlockNameError: "A custom category whose block is not `<name>s`.",
        ReservedParamError: "A parameter named like a structural column, a variable or a pair input.",
        DuplicateParamError: "A parameter declared twice.",
        DimensionError: "An unparsable or forbidden dimension.",
        ParseError: "An expression that does not parse, or whose definitions are out of order.",
        UnboundVariableError: "A name an expression reads that is neither a variable of the category nor a declared numeric parameter.",
        UnknownFunctionError: "A function an expression calls that the expression language does not have.",
        FunctionArityError: "A function called with the wrong number of arguments.",
        PointError: "A point `pk` beyond the category's arity, or any point in a pair style.",
        CoordinateMismatchError: "A kernel tier the category cannot take.",
        DerivativeError: "A kernel's derivative against a central difference of its energy, beyond relative 1e-6.",
        DisagreeError: "An expression against the kernel beside it, beyond relative 1e-10.",
        AsymmetricError: "A pair energy that changes when its two atoms are exchanged.",
        SealedError: "A built-in, which cannot be overridden or removed.",
        ConflictError: "A different registration under a taken name.",
        NoKernelError: "Nothing can price the style: no kernel and no expression.",
        NoMixingError: "An unlike pair with no cross row stating a parameter that does not mix.",
        MissingParamError: "A row (or the style) without a value its kernel needs, and no default.",
        BadValueError: "A row's (or the style's) value of a declared parameter that is not of its declared kind or domain: text for a number, an array of another rank, text outside its choices.",
        KernelShapeError: "A kernel's output of the wrong shape or dtype, or a Python kernel that raised (the exception is the `__cause__`).",
        NoEngineFormError: "An engine with no form for the style.",
        FormConflictError: "A form family without exactly one canonical style.",
        NoFormError: "A style that belongs to no form family.",
        OutOfImageError: "An exact form conversion refused: a row outside the image of the target style.",
        MalformedError: "A spec whose declarations contradict each other.",
    }
}

/// `e` as its `molpy.ff.ir` exception, carrying `message` and the
/// variant's fields as attributes.
pub(crate) fn ir_err(e: &IrError, message: String) -> PyErr {
    use errors as x;
    let s = |v: &str| Field::Str(v.to_owned());
    let n = |v: usize| Field::Int(v);
    let f = |v: F| Field::Float(v);
    let (err, fields): (PyErr, Vec<(&str, Field)>) = match e {
        IrError::UnknownCategory { category } => (
            x::UnknownCategoryError::new_err(message),
            vec![("category", s(category))],
        ),
        IrError::BadName { what, name } => (
            x::BadNameError::new_err(message),
            vec![("what", s(what)), ("name", s(name))],
        ),
        IrError::Arity { category, arity } => (
            x::ArityError::new_err(message),
            vec![("category", s(category)), ("arity", n(*arity))],
        ),
        IrError::BlockName { category, block } => (
            x::BlockNameError::new_err(message),
            vec![("category", s(category)), ("block", s(block))],
        ),
        IrError::ReservedParam { style, param } => (
            x::ReservedParamError::new_err(message),
            vec![("style", s(style)), ("param", s(param))],
        ),
        IrError::DuplicateParam { style, param } => (
            x::DuplicateParamError::new_err(message),
            vec![("style", s(style)), ("param", s(param))],
        ),
        IrError::Dimension { param, dim, reason } => (
            x::DimensionError::new_err(message),
            vec![("param", s(param)), ("dim", s(dim)), ("reason", s(reason))],
        ),
        IrError::Parse {
            expression,
            at,
            reason,
        } => (
            x::ParseError::new_err(message),
            vec![
                ("expression", s(expression)),
                ("at", n(*at)),
                ("reason", s(reason)),
            ],
        ),
        IrError::UnboundVariable { style, name } => (
            x::UnboundVariableError::new_err(message),
            vec![("style", s(style)), ("name", s(name))],
        ),
        IrError::UnknownFunction { name } => (
            x::UnknownFunctionError::new_err(message),
            vec![("name", s(name))],
        ),
        IrError::FunctionArity {
            name,
            given,
            expected,
        } => (
            x::FunctionArityError::new_err(message),
            vec![
                ("name", s(name)),
                ("given", n(*given)),
                ("expected", n(*expected)),
            ],
        ),
        IrError::Point {
            style,
            point,
            arity,
        } => (
            x::PointError::new_err(message),
            vec![
                ("style", s(style)),
                ("point", s(point)),
                ("arity", n(*arity)),
            ],
        ),
        IrError::CoordinateMismatch { category, kernel } => (
            x::CoordinateMismatchError::new_err(message),
            vec![("category", s(category)), ("kernel", s(kernel))],
        ),
        IrError::Derivative { style, at, rel } => (
            x::DerivativeError::new_err(message),
            vec![("style", s(style)), ("at", s(at)), ("rel", f(*rel))],
        ),
        IrError::Disagree { style, at, rel } => (
            x::DisagreeError::new_err(message),
            vec![("style", s(style)), ("at", s(at)), ("rel", f(*rel))],
        ),
        IrError::Asymmetric { style } => (
            x::AsymmetricError::new_err(message),
            vec![("style", s(style))],
        ),
        IrError::Sealed { category, style } => (
            x::SealedError::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::Conflict { category, style } => (
            x::ConflictError::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::NoKernel { category, style } => (
            x::NoKernelError::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::NoMixing { style, param, pair } => (
            x::NoMixingError::new_err(message),
            vec![("style", s(style)), ("param", s(param)), ("pair", s(pair))],
        ),
        IrError::MissingParam {
            style,
            type_,
            param,
        } => (
            x::MissingParamError::new_err(message),
            vec![("style", s(style)), ("type", s(type_)), ("param", s(param))],
        ),
        IrError::BadValue {
            style,
            type_,
            param,
            reason,
        } => (
            x::BadValueError::new_err(message),
            vec![
                ("style", s(style)),
                ("type", s(type_)),
                ("param", s(param)),
                ("reason", s(reason)),
            ],
        ),
        IrError::KernelShape { style, reason } => (
            x::KernelShapeError::new_err(message),
            vec![("style", s(style)), ("reason", s(reason))],
        ),
        IrError::NoEngineForm {
            engine,
            category,
            style,
            reason,
        } => (
            x::NoEngineFormError::new_err(message),
            vec![
                ("engine", s(engine)),
                ("category", s(category)),
                ("style", s(style)),
                ("reason", s(reason)),
            ],
        ),
        IrError::FormConflict { family, reason } => (
            x::FormConflictError::new_err(message),
            vec![("family", s(family)), ("reason", s(reason))],
        ),
        IrError::NoForm { category, style } => (
            x::NoFormError::new_err(message),
            vec![("category", s(category)), ("style", s(style))],
        ),
        IrError::OutOfImage {
            from,
            to,
            type_,
            reason,
        } => (
            x::OutOfImageError::new_err(message),
            vec![
                ("from_", s(from)),
                ("to", s(to)),
                ("type", s(type_)),
                ("reason", s(reason)),
            ],
        ),
        IrError::Malformed { style, reason } => (
            x::MalformedError::new_err(message),
            vec![("style", s(style)), ("reason", s(reason))],
        ),
    };
    Python::attach(|py| {
        let value = err.value(py);
        for (name, field) in fields {
            // An attribute that cannot be set leaves the message, which
            // names the item too.
            let _ = match field {
                Field::Str(v) => value.setattr(name, v),
                Field::Int(v) => value.setattr(name, v),
                Field::Float(v) => value.setattr(name, v),
            };
        }
    });
    err
}

/// A field of an [`IrError`] variant, as a Python attribute.
enum Field {
    Str(String),
    Int(usize),
    Float(F),
}

/// `e` as its `molpy.ff.ir` exception, with its own message.
pub(crate) fn refuse(e: IrError) -> PyErr {
    let message = e.to_string();
    ir_err(&e, message)
}

/// A compile's error as a Python exception: a Python kernel's parked
/// exception first (it is the cause of whatever followed), else an IR
/// refusal as its `IrError` subclass, else a plain `ValueError`.
pub(crate) fn compile_err(e: CompileError) -> PyErr {
    if let Err(parked) = crate::ff::style_registry::take_kernel_err() {
        return parked;
    }
    match e {
        CompileError::Ir(e) => refuse(e),
        CompileError::Invalid(message) => PyValueError::new_err(message),
        other @ CompileError::NoBox { .. } => PyValueError::new_err(other.to_string()),
    }
}

/// A writer's error: its typed refusal (`NoEngineForm`) as the `IrError`
/// subclass, anything else a `ValueError`.
pub(crate) fn writer_err(e: molrs::io::writer::ForceFieldWriteError) -> PyErr {
    match e.ir() {
        Some(refusal) => ir_err(refusal, e.to_string()),
        None => PyValueError::new_err(e.to_string()),
    }
}

/// A force-field definition error as a Python exception: the IR refusal
/// it is (`Arity`, `UnknownCategory`) as its subclass, with the
/// definition's own message; else a plain `ValueError`.
pub(crate) fn def_err(e: DefError) -> PyErr {
    match e.ir() {
        Some(refusal) => ir_err(&refusal, e.to_string()),
        None => PyValueError::new_err(e.to_string()),
    }
}

/// `shape` as Python prints it: `(2,)`, `(2, 3, 3)`.
pub(crate) fn py_shape(shape: &[usize]) -> String {
    match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// What `obj` is, for a shape error.
pub(crate) fn describe(obj: &Bound<'_, PyAny>) -> String {
    match obj.cast::<numpy::PyUntypedArray>() {
        Ok(a) => format!(
            "an ndarray of dtype {} and shape {}",
            a.dtype(),
            py_shape(a.shape())
        ),
        Err(_) => obj
            .get_type()
            .name()
            .map(|n| format!("a {n}"))
            .unwrap_or_else(|_| "an object".into()),
    }
}

// ---------------------------------------------------------------------------
// Param
// ---------------------------------------------------------------------------

/// `text` as the [`ParamDimension`] of parameter `param`, refused as `IrError.ParamDimension`.
fn parse_dim(param: &str, text: &str) -> PyResult<ParamDimension> {
    let dim_err = |reason: String| {
        refuse(IrError::Dimension {
            param: param.to_owned(),
            dim: text.to_owned(),
            reason,
        })
    };
    let dim: ParamDimension = text.parse().map_err(dim_err)?;
    dim.check().map_err(dim_err)?;
    Ok(dim)
}

fn parse_mix(mix: Option<&Bound<'_, PyAny>>) -> PyResult<ParamCombination> {
    let Some(mix) = mix.filter(|m| !m.is_none()) else {
        return Ok(ParamCombination::None);
    };
    if let Ok(rule) = mix.extract::<String>() {
        return match rule.as_str() {
            "none" => Ok(ParamCombination::None),
            "arithmetic" => Ok(ParamCombination::Arithmetic),
            "geometric" => Ok(ParamCombination::Geometric),
            _ => Err(PyValueError::new_err(format!(
                "mix={rule:?}: one of None, 'arithmetic', 'geometric', \
                 ('lj_epsilon', <sigma>), ('lj_sigma', <epsilon>)"
            ))),
        };
    }
    let (rule, partner): (String, String) = mix.extract().map_err(|_| {
        PyTypeError::new_err(
            "mix: None, 'arithmetic', 'geometric', ('lj_epsilon', <sigma>) or \
             ('lj_sigma', <epsilon>)",
        )
    })?;
    match rule.as_str() {
        "lj_epsilon" => Ok(ParamCombination::LjEpsilon {
            sigma: partner.into(),
        }),
        "lj_sigma" => Ok(ParamCombination::LjSigma {
            epsilon: partner.into(),
        }),
        _ => Err(PyValueError::new_err(format!(
            "mix=({rule:?}, {partner:?}): the joint rule is 'lj_epsilon' or 'lj_sigma'"
        ))),
    }
}

fn mix_to_py<'py>(py: Python<'py>, mix: &ParamCombination) -> PyResult<Bound<'py, PyAny>> {
    Ok(match mix {
        ParamCombination::None => py.None().into_bound(py),
        ParamCombination::Arithmetic => PyString::new(py, "arithmetic").into_any(),
        ParamCombination::Geometric => PyString::new(py, "geometric").into_any(),
        ParamCombination::LjEpsilon { sigma } => {
            ("lj_epsilon", sigma.as_ref()).into_pyobject(py)?.into_any()
        }
        ParamCombination::LjSigma { epsilon } => {
            ("lj_sigma", epsilon.as_ref()).into_pyobject(py)?.into_any()
        }
    })
}

/// A number or a string, as a [`ParamValue`].
fn value_of(what: &str, obj: &Bound<'_, PyAny>) -> PyResult<ParamValue> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ParamValue::Text(s.into()));
    }
    obj.extract::<F>()
        .map(ParamValue::Num)
        .map_err(|_| PyTypeError::new_err(format!("{what}: a number or a string")))
}

/// One parameter of a force-field IR style: its name, its dimension, its
/// kind, its default, its mixing rule (pair styles) and whether it is an
/// indexed family.
///
/// Parameters
/// ----------
/// name : str
///     ``^[A-Za-z_][A-Za-z0-9_]*$``, not a reserved name (checked when the
///     style registers).
/// dim : str, default "1"
///     The dimension, as exponents of ``E`` (energy), ``L`` (length), ``A``
///     (angle), ``Q`` (charge) and ``M`` (mass): ``"E/L^2"``, ``"E*L^6"``,
///     ``"A"`` (an angle value, stored in degrees), ``"E/A^2"`` (per
///     radian), ``"1"``. Refused at once as :class:`DimensionError`.
/// kind : {"scalar", "array", "text"}, default "scalar"
///     One number per row; an ``f64`` array of ``rank`` per row (not an
///     expression variable); a string, one of ``choices`` when given.
/// rank : int, optional
///     The array rank (``kind="array"`` only).
/// choices : list of str, optional
///     The allowed strings (``kind="text"`` only).
/// default : float or str, optional
///     The value a row (or the style) without the parameter takes; none
///     makes it required.
/// mix : None, "arithmetic", "geometric", ("lj_epsilon", sigma) or ("lj_sigma", epsilon)
///     How a pair style's parameter combines from the two self rows when no
///     cross row states it.
/// indexed : bool, default False
///     A numbered family ``<name>1 … <name>M``.
///
/// Examples
/// --------
/// >>> from molpy.ff import ir
/// >>> ir.ParamSpec("k", "E/L^2")
/// ParamSpec('k', 'E/L^2')
/// >>> ir.ParamSpec("epsilon", "E", mix=("lj_epsilon", "sigma")).mix
/// ('lj_epsilon', 'sigma')
/// >>> ir.ParamSpec("k", "E/L^^2")
/// Traceback (most recent call last):
///     ...
/// molpy.ff.ir.DimensionError: parameter `k`: dimension "E/L^^2": ...
#[pyclass(
    module = "molpy.ff.ir",
    name = "ParamSpec",
    frozen,
    eq,
    skip_from_py_object
)]
#[derive(Clone, PartialEq)]
pub struct PyParamSpec {
    inner: ParamSpec,
}

#[pymethods]
impl PyParamSpec {
    #[new]
    #[pyo3(signature = (name, dim="1", *, kind="scalar", rank=None, choices=None, default=None, mix=None, indexed=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: String,
        dim: &str,
        kind: &str,
        rank: Option<u8>,
        choices: Option<Vec<String>>,
        default: Option<Bound<'_, PyAny>>,
        mix: Option<Bound<'_, PyAny>>,
        indexed: bool,
    ) -> PyResult<Self> {
        let dim = parse_dim(&name, dim)?;
        let bad = |why: String| PyValueError::new_err(format!("ParamSpec {name:?}: {why}"));
        let kind = match (kind, rank, &choices) {
            ("scalar", None, None) => ParamKind::Scalar,
            ("array", Some(rank @ 1..), None) => ParamKind::Array { rank },
            ("array", _, None) => return Err(bad("an array parameter needs rank >= 1".into())),
            ("text", None, _) => ParamKind::Text {
                choices: choices
                    .clone()
                    .map(|c| c.into_iter().map(Cow::Owned).collect()),
            },
            ("scalar" | "array" | "text", ..) => {
                return Err(bad(format!(
                    "rank= is for kind='array' and choices= for kind='text', not kind={kind:?}"
                )));
            }
            _ => {
                return Err(bad(format!(
                    "kind={kind:?}: one of 'scalar', 'array', 'text'"
                )));
            }
        };
        let default = match default.filter(|d| !d.is_none()) {
            None => None,
            Some(d) => Some(value_of("default", &d)?),
        };
        match (&kind, &default) {
            (_, None) | (ParamKind::Scalar, Some(ParamValue::Num(_))) => {}
            (ParamKind::Text { choices }, Some(ParamValue::Text(t))) => {
                if choices.as_ref().is_some_and(|c| !c.contains(t)) {
                    return Err(bad(format!("default {t:?} is not one of its choices")));
                }
            }
            _ => return Err(bad("the default does not fit the parameter's kind".into())),
        }
        let mut inner = ParamSpec::new(name, dim)
            .kind(kind)
            .mix(parse_mix(mix.as_ref())?);
        inner.default = default;
        inner.indexed = indexed;
        Ok(Self { inner })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The dimension, printed in its canonical spelling (``"E/L^2"``).
    #[getter]
    fn dim(&self) -> String {
        self.inner.dim.to_string()
    }

    /// ``"scalar"``, ``"array"`` or ``"text"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ParamKind::Scalar => "scalar",
            ParamKind::Array { .. } => "array",
            ParamKind::Text { .. } => "text",
        }
    }

    #[getter]
    fn rank(&self) -> Option<u8> {
        match self.inner.kind {
            ParamKind::Array { rank } => Some(rank),
            _ => None,
        }
    }

    #[getter]
    fn choices(&self) -> Option<Vec<String>> {
        match &self.inner.kind {
            ParamKind::Text { choices } => choices
                .as_ref()
                .map(|c| c.iter().map(|s| s.to_string()).collect()),
            _ => None,
        }
    }

    #[getter]
    fn default<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .default
            .as_ref()
            .map(|v| crate::ff::forcefield::param_to_py(py, v))
            .transpose()
    }

    #[getter]
    fn mix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        mix_to_py(py, &self.inner.mix)
    }

    #[getter]
    fn indexed(&self) -> bool {
        self.inner.indexed
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut out = format!("ParamSpec({:?}, {:?}", self.inner.name, self.dim());
        match &self.inner.kind {
            ParamKind::Scalar => {}
            ParamKind::Array { rank } => out.push_str(&format!(", kind='array', rank={rank}")),
            ParamKind::Text { choices } => {
                out.push_str(", kind='text'");
                if let Some(c) = choices {
                    out.push_str(&format!(", choices={c:?}"));
                }
            }
        }
        if let Some(d) = &self.inner.default {
            out.push_str(&format!(
                ", default={}",
                crate::ff::forcefield::param_to_py(py, d)?.repr()?
            ));
        }
        if self.inner.mix != ParamCombination::None {
            out.push_str(&format!(
                ", mix={}",
                mix_to_py(py, &self.inner.mix)?.repr()?
            ));
        }
        if self.inner.indexed {
            out.push_str(", indexed=True");
        }
        out.push(')');
        Ok(out.replace('"', "'"))
    }
}

/// `params=`: a list of :class:`ParamSpec`, or a ``{name: dim}`` dict of
/// required scalars.
pub(crate) fn parse_params(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<ParamSpec>> {
    let Some(arg) = arg.filter(|a| !a.is_none()) else {
        return Ok(Vec::new());
    };
    if let Ok(dict) = arg.cast::<PyDict>() {
        return dict
            .iter()
            .map(|(k, v)| {
                let name: String = k.extract()?;
                let dim: String = v.extract().map_err(|_| {
                    PyTypeError::new_err(format!(
                        "params: {{name: dim}} takes a dimension string for {name:?}"
                    ))
                })?;
                Ok(ParamSpec::new(name.clone(), parse_dim(&name, &dim)?))
            })
            .collect();
    }
    arg.try_iter()?
        .map(|item| {
            let item = item?;
            item.cast::<PyParamSpec>()
                .map(|p| p.get().inner.clone())
                .map_err(|_| {
                    PyTypeError::new_err(format!(
                        "params: a list of molpy.ff.ir.ParamSpec or a {{name: dim}} dict, got {}",
                        describe(&item)
                    ))
                })
        })
        .collect()
}

/// `samples=`: dicts ``{"q": (lo, hi), <param>: value, …}``.
pub(crate) fn parse_samples(arg: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<ConformanceSample>> {
    let Some(arg) = arg.filter(|a| !a.is_none()) else {
        return Ok(Vec::new());
    };
    arg.try_iter()?
        .map(|item| {
            let item = item?;
            let dict = item.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("samples: a list of dicts {'q': (lo, hi), <param>: value}")
            })?;
            let mut q = None;
            let mut params = Vec::new();
            for (k, v) in dict.iter() {
                let key: String = k.extract()?;
                if key == "q" {
                    q = Some(v.extract::<(F, F)>().map_err(|_| {
                        PyTypeError::new_err("samples: 'q' is a (lo, hi) pair of numbers")
                    })?);
                } else {
                    let value = value_of(&format!("samples[{key:?}]"), &v)?;
                    params.push((Cow::Owned(key), value));
                }
            }
            let q = q.ok_or_else(|| PyValueError::new_err("samples: every sample needs 'q'"))?;
            Ok(ConformanceSample { params, q })
        })
        .collect()
}

pub(crate) fn parse_coordinate(text: &str) -> PyResult<Coordinate> {
    Ok(match text {
        "compound" => Coordinate::Compound,
        "distance" => Coordinate::Distance,
        "angle" => Coordinate::Angle,
        "dihedral" => Coordinate::Dihedral,
        "improper" => Coordinate::Improper,
        "none" => Coordinate::None,
        _ => {
            return Err(PyValueError::new_err(format!(
                "coordinate={text:?}: one of 'compound', 'distance', 'angle', 'dihedral', \
                 'improper'"
            )));
        }
    })
}

fn coordinate_name(c: Coordinate) -> &'static str {
    match c {
        Coordinate::None => "none",
        Coordinate::Distance => "distance",
        Coordinate::Angle => "angle",
        Coordinate::Dihedral => "dihedral",
        Coordinate::Improper => "improper",
        Coordinate::Compound => "compound",
    }
}

pub(crate) fn parse_order(text: &str) -> PyResult<EndpointOrder> {
    Ok(match text {
        "reversible" => EndpointOrder::Reversible,
        "ordered" => EndpointOrder::Ordered,
        "unordered" => EndpointOrder::Unordered,
        _ => {
            return Err(PyValueError::new_err(format!(
                "order={text:?}: one of 'reversible', 'ordered', 'unordered'"
            )));
        }
    })
}

fn order_name(o: EndpointOrder) -> &'static str {
    match o {
        EndpointOrder::Reversible => "reversible",
        EndpointOrder::Ordered => "ordered",
        EndpointOrder::Unordered => "unordered",
    }
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

/// One category of the force-field IR, as :func:`categories` lists it.
#[pyclass(module = "molpy.ff.ir", name = "CategorySpec", frozen)]
pub struct PyCategorySpec {
    pub(crate) spec: CategorySpec,
    pub(crate) builtin: bool,
}

#[pymethods]
impl PyCategorySpec {
    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    /// Endpoints per type row (2 for a pair).
    #[getter]
    fn arity(&self) -> usize {
        self.spec.arity.endpoints()
    }

    /// Whether its terms are the pairs a neighbour search finds.
    #[getter]
    fn pair(&self) -> bool {
        self.spec.arity == Arity::SelfOrPair
    }

    /// The Frame block its terms live in.
    #[getter]
    fn block(&self) -> &str {
        &self.spec.block
    }

    #[getter]
    fn coordinate(&self) -> &'static str {
        coordinate_name(self.spec.coordinate)
    }

    #[getter]
    fn order(&self) -> &'static str {
        order_name(self.spec.order)
    }

    /// Whether it is one of molrs's own (sealed).
    #[getter]
    fn builtin(&self) -> bool {
        self.builtin
    }

    fn __repr__(&self) -> String {
        format!(
            "CategorySpec('{}', arity={}, block='{}', coordinate='{}')",
            self.spec.name,
            self.arity(),
            self.spec.block,
            self.coordinate()
        )
    }
}

/// One style of the force-field IR, as :func:`styles` lists it.
#[pyclass(module = "molpy.ff.ir", name = "StyleSpec", frozen)]
pub struct PyStyleSpec {
    spec: StyleSpec,
    kernel: Option<&'static str>,
    builtin: bool,
}

impl PyStyleSpec {
    pub(crate) fn of(r: &Registry, spec: &StyleSpec, kernel: Option<&Kernel>) -> Self {
        let tier = match kernel {
            Some(Kernel::Expression(_)) => Some("expression"),
            Some(Kernel::Scalar(_)) => Some("scalar"),
            Some(Kernel::Compound(_)) => Some("compound"),
            Some(Kernel::Constructor { .. }) => Some("constructor"),
            None => spec.expression.as_ref().map(|_| "expression"),
        };
        Self {
            spec: spec.clone(),
            kernel: tier,
            builtin: r.is_sealed(&spec.category, &spec.name),
        }
    }
}

#[pymethods]
impl PyStyleSpec {
    #[getter]
    fn category(&self) -> &str {
        &self.spec.category
    }

    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    /// The per-type parameters, in order.
    #[getter]
    fn params(&self) -> Vec<PyParamSpec> {
        self.spec
            .params
            .iter()
            .map(|p| PyParamSpec { inner: p.clone() })
            .collect()
    }

    #[getter]
    fn style_params(&self) -> Vec<PyParamSpec> {
        self.spec
            .style_params
            .iter()
            .map(|p| PyParamSpec { inner: p.clone() })
            .collect()
    }

    /// The energy as an expression, byte for byte as registered.
    #[getter]
    fn expression(&self) -> Option<&str> {
        self.spec.expression.as_deref()
    }

    /// What prices it: ``"expression"``, ``"scalar"`` / ``"compound"`` (a
    /// form kernel: Rust or a Python callable), ``"constructor"`` (a native
    /// kernel), or ``None`` (a category that prices nothing).
    #[getter]
    fn kernel(&self) -> Option<&'static str> {
        self.kernel
    }

    /// Whether it is one of molrs's own (sealed).
    #[getter]
    fn builtin(&self) -> bool {
        self.builtin
    }

    /// ``"type_rows"`` or ``"per_instance"``: where its numbers come from.
    #[getter]
    fn source(&self) -> &'static str {
        match self.spec.source {
            ParamSource::TypeRows => "type_rows",
            ParamSource::PerInstance => "per_instance",
        }
    }

    /// The style's LAMMPS form: ``"positional"``, ``"positional:<name>"``,
    /// ``"custom:<name>"`` (a codec of its own) or ``None``.
    #[getter]
    fn lammps(&self) -> Option<String> {
        engine::lammps_form_name(&self.spec)
    }

    /// A pair style's special-bonds class: ``"lj"`` or ``"coul"``.
    #[getter]
    fn special(&self) -> Option<&'static str> {
        self.spec.special.map(|s| match s {
            SpecialClass::Vdw => "lj",
            SpecialClass::Coulomb => "coul",
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "StyleSpec('{}', '{}', kernel={})",
            self.spec.category,
            self.spec.name,
            self.kernel.map_or("None".into(), |k| format!("'{k}'"))
        )
    }
}

/// Register the classes of `molpy.ff.ir` on `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(m)?;
    m.add_class::<PyParamSpec>()?;
    m.add_class::<PyStyleSpec>()?;
    m.add_class::<PyCategorySpec>()?;
    Ok(())
}
