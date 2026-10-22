//! [`IrError`]: every way the force-field IR refuses something, each naming
//! what it refused.

use std::fmt;

use molrs::op::F;

/// Why the registry, the expression engine, a compile or an engine codec
/// refused a category, a style or a term (`ff-ir-02-protocol` §5). Python
/// raises a `molrs.ff.ir.IrError` subclass of the same name.
#[derive(Clone, Debug, PartialEq)]
pub enum IrError {
    /// A style in a category neither built in nor registered.
    UnknownCategory {
        category: String,
    },
    /// A category or parameter name outside its pattern.
    BadName {
        what: &'static str,
        name: String,
    },
    /// A custom category's arity outside 2..=5, or a type with the wrong
    /// number of endpoints.
    Arity {
        category: String,
        arity: usize,
    },
    /// A custom category whose block is not `<name>s`.
    BlockName {
        category: String,
        block: String,
    },
    /// A parameter named like a structural column, a variable, a function
    /// or a pair input; or a reserved style param of the wrong kind.
    ReservedParam {
        style: String,
        param: String,
    },
    DuplicateParam {
        style: String,
        param: String,
    },
    /// An unparsable or forbidden dimension.
    Dimension {
        param: String,
        dim: String,
        reason: String,
    },
    /// An expression that does not parse.
    Parse {
        expression: String,
        at: usize,
        reason: String,
    },
    /// A free name that is neither a variable of the category nor a
    /// declared numeric parameter (`theta` in a bond).
    UnboundVariable {
        style: String,
        name: String,
    },
    UnknownFunction {
        name: String,
    },
    FunctionArity {
        name: String,
        given: usize,
        expected: usize,
    },
    /// A point `pk` beyond the category's arity, or any point in a pair.
    Point {
        style: String,
        point: String,
        arity: usize,
    },
    /// A kernel tier the category cannot take: a scalar form on a
    /// `Compound` / `None` category, a compound form on a pair, any kernel
    /// on a category that prices no energy.
    CoordinateMismatch {
        category: String,
        kernel: String,
    },
    /// A form's `dE/dq` (or gradient) against a central difference of its
    /// own energy, beyond relative 1e-6.
    Derivative {
        style: String,
        at: String,
        rel: F,
    },
    /// An expression against the kernel beside it, beyond relative 1e-10.
    Disagree {
        style: String,
        at: String,
        rel: F,
    },
    /// A pair energy not symmetric under exchanging the two atoms.
    Asymmetric {
        style: String,
    },
    /// A built-in, which cannot be overridden or removed.
    Sealed {
        category: String,
        style: String,
    },
    /// A different registration under a taken name.
    Conflict {
        category: String,
        style: String,
    },
    /// Nothing can price the style: no kernel and no expression (or no
    /// expression engine).
    NoKernel {
        category: String,
        style: String,
    },
    /// An unlike pair with no cross row stating a parameter that does not
    /// mix.
    NoMixing {
        style: String,
        param: String,
        pair: String,
    },
    /// A row (or the style) without a value its kernel needs, and no
    /// default.
    MissingParam {
        style: String,
        type_: String,
        param: String,
    },
    /// A row (or the style) whose value of a declared parameter is not of
    /// the declared kind: text where the spec declares a number, an array
    /// of another rank, a text outside the declared choices, a number out
    /// of its domain (a non-integer `periodicity`). Not in the protocol's
    /// table, which has no row for an ill-typed value.
    BadValue {
        style: String,
        type_: String,
        param: String,
        reason: String,
    },
    /// A Tier-2 kernel's output of the wrong shape, or a kernel that raised.
    KernelShape {
        style: String,
        reason: String,
    },
    NoEngineForm {
        engine: String,
        category: String,
        style: String,
        reason: String,
    },
    FormConflict {
        family: String,
        reason: String,
    },
    NoForm {
        category: String,
        style: String,
    },
    /// An exact form conversion refused: the row named `type_` of `from`
    /// (`"<category> <style>"`) is outside the image of `to`, for the named
    /// `reason` (`ff-ir-01` P3: a Fourier series with `bₙ ≠ 0` has no RB
    /// form). Also a `fit_form` that cannot evaluate the category.
    OutOfImage {
        from: String,
        to: String,
        type_: String,
        reason: String,
    },
    /// A spec whose declarations contradict each other (a mixing rule on a
    /// bonded parameter, an ε without its σ, an expression kernel that is
    /// not the spec's expression). Not in the protocol's table: the table
    /// has no row for an internally inconsistent spec.
    Malformed {
        style: String,
        reason: String,
    },
}

impl fmt::Display for IrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use IrError::*;
        match self {
            UnknownCategory { category } => write!(
                f,
                "category `{category}` is not registered (molrs.ff.style_registry.register_category)"
            ),
            BadName { what, name } => write!(f, "{name:?} is no {what} name"),
            Arity { category, arity } => write!(
                f,
                "category `{category}`: arity {arity} (a custom category names 2 to 5 atoms)"
            ),
            BlockName { category, block } => write!(
                f,
                "category `{category}`: block `{block}`; a custom category's block is `{category}s`"
            ),
            ReservedParam { style, param } => {
                write!(f, "style `{style}`: parameter name `{param}` is reserved")
            }
            DuplicateParam { style, param } => {
                write!(f, "style `{style}`: parameter `{param}` declared twice")
            }
            Dimension { param, dim, reason } => {
                write!(f, "parameter `{param}`: dimension {dim:?}: {reason}")
            }
            Parse {
                expression,
                at,
                reason,
            } => write!(f, "expression {expression:?} at {at}: {reason}"),
            UnboundVariable { style, name } => write!(
                f,
                "style `{style}`: `{name}` is neither a variable of its category nor a declared \
                 numeric parameter"
            ),
            UnknownFunction { name } => write!(f, "unknown function `{name}`"),
            FunctionArity {
                name,
                given,
                expected,
            } => write!(f, "`{name}` takes {expected} arguments, given {given}"),
            Point {
                style,
                point,
                arity,
            } => write!(
                f,
                "style `{style}`: point `{point}` (the category has {arity} points)"
            ),
            CoordinateMismatch { category, kernel } => {
                write!(f, "category `{category}` cannot take a {kernel} kernel")
            }
            Derivative { style, at, rel } => write!(
                f,
                "style `{style}`: the derivative disagrees with a central difference of its \
                 energy at {at} (relative {rel:.1e} > 1e-6)"
            ),
            Disagree { style, at, rel } => write!(
                f,
                "style `{style}`: the expression disagrees with the kernel at {at} (relative \
                 {rel:.1e} > 1e-10)"
            ),
            Asymmetric { style } => write!(
                f,
                "style `{style}`: the pair energy changes when its two atoms are exchanged"
            ),
            Sealed { category, style } if style.is_empty() => {
                write!(f, "category `{category}` is built in and sealed")
            }
            Sealed { category, style } => write!(
                f,
                "{category} `{style}` is built in and sealed: register the new form under a \
                 name of its own"
            ),
            Conflict { category, style } if style.is_empty() => write!(
                f,
                "category `{category}` is already registered with a different spec"
            ),
            Conflict { category, style } => write!(
                f,
                "{category} `{style}` is already registered with a different spec or kernel"
            ),
            NoKernel { category, style } => write!(
                f,
                "no kernel for {category} `{style}`: register it (molrs.ff.style_registry.register_style) \
                 or give it an expression"
            ),
            NoMixing { style, param, pair } => write!(
                f,
                "style `{style}`: `{param}` does not mix, and the pair {pair} has no cross row \
                 stating it"
            ),
            MissingParam {
                style,
                type_,
                param,
            } if type_.is_empty() => write!(f, "style `{style}`: missing `{param}`"),
            MissingParam {
                style,
                type_,
                param,
            } => write!(f, "style `{style}` type '{type_}': missing `{param}`"),
            BadValue {
                style,
                type_,
                param,
                reason,
            } if type_.is_empty() => write!(f, "style `{style}`: `{param}` {reason}"),
            BadValue {
                style,
                type_,
                param,
                reason,
            } => write!(f, "style `{style}` type '{type_}': `{param}` {reason}"),
            KernelShape { style, reason } => write!(f, "style `{style}`: {reason}"),
            NoEngineForm {
                engine,
                category,
                style,
                reason,
            } => write!(f, "{engine} has no form for {category} `{style}`: {reason}"),
            FormConflict { family, reason } => write!(f, "form family `{family}`: {reason}"),
            NoForm { category, style } => {
                write!(f, "{category} `{style}` belongs to no form family")
            }
            OutOfImage {
                from,
                to,
                type_,
                reason,
            } if type_.is_empty() => write!(f, "{from} has no exact `{to}` form: {reason}"),
            OutOfImage {
                from,
                to,
                type_,
                reason,
            } => write!(
                f,
                "{from} type '{type_}' has no exact `{to}` form: {reason}"
            ),
            Malformed { style, reason } => write!(f, "style `{style}`: {reason}"),
        }
    }
}

impl std::error::Error for IrError {}
