//! The expression engine's errors.

use std::fmt;

/// Why an expression does not parse or does not compile.
///
/// Each variant names what is wrong — the token and its byte offset for a
/// syntax error, the identifier for a semantic one — so a refusal at
/// registration or at compile can say which expression to fix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionError {
    // -- syntax (parse) --
    /// A character the grammar has no token for, at byte offset `pos`.
    UnexpectedChar { pos: usize, ch: char },
    /// A token where the grammar wants something else.
    UnexpectedToken {
        pos: usize,
        found: String,
        expected: &'static str,
    },
    /// The source (or a `;` segment) ends where the grammar wants more.
    UnexpectedEnd { expected: &'static str },
    /// No energy expression before the first `;` (or an empty string).
    EmptyExpression,
    /// A `;` segment that is not `name=expression`.
    BadDefinition { pos: usize, text: String },

    // -- semantics (compile) --
    /// A call to a function the grammar does not have.
    UnknownFunction { name: String },
    /// A call with the wrong number of arguments.
    FunctionArity {
        name: String,
        expected: usize,
        found: usize,
    },
    /// An identifier that is neither a variable of the category, a declared
    /// numeric parameter nor a sub-definition (e.g. `theta` in a bond).
    /// `allowed` lists the variables.
    UndeclaredVariable { name: String, allowed: Vec<String> },
    /// Sub-definitions that refer to each other in a cycle (a self
    /// reference included), in cycle order, the first name repeated.
    CyclicDefinition { cycle: Vec<String> },
    /// A sub-definition `used_in` that uses `name`, defined to its left:
    /// as in Lepton, a definition sees only the definitions to its right.
    DefinitionOrder { name: String, used_in: String },
    /// Two sub-definitions with one name.
    DuplicateDefinition { name: String },
    /// A sub-definition named like a variable (a geometric variable, a
    /// point, a parameter, a pair's `x1`/`q1`).
    DefinitionShadows { name: String },
    /// A compound function's argument that is not a point `p1`…`pN` of the
    /// category (`points` = N; a pair has none).
    NotAPoint {
        function: String,
        found: String,
        points: usize,
    },
    /// A point `pN` used as a number.
    PointAsNumber { name: String },
    /// A binding the engine cannot compile against: a parameter name that
    /// is no identifier, repeated, reserved (a geometric variable, a point;
    /// in a pair `q`, `q1`, `q2` or `x1` beside `x`); a compound
    /// arity outside 2..=5.
    BadBinding { reason: String },
    /// A column the evaluation needs and the caller did not supply.
    MissingInput { input: String },
}

impl fmt::Display for ExpressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExpressionError::UnexpectedChar { pos, ch } => {
                write!(f, "expression: unexpected character `{ch}` at byte {pos}")
            }
            ExpressionError::UnexpectedToken {
                pos,
                found,
                expected,
            } => write!(
                f,
                "expression: unexpected `{found}` at byte {pos}, expected {expected}"
            ),
            ExpressionError::UnexpectedEnd { expected } => {
                write!(f, "expression: unexpected end, expected {expected}")
            }
            ExpressionError::EmptyExpression => write!(f, "expression: no energy expression"),
            ExpressionError::BadDefinition { pos, text } => write!(
                f,
                "expression: `{text}` at byte {pos} is no sub-definition `name=expression`"
            ),
            ExpressionError::UnknownFunction { name } => write!(
                f,
                "expression: unknown function `{name}` (the grammar has exp log sqrt sin cos \
                 tan asin acos atan abs min max step delta select, and distance angle \
                 dihedral over points)"
            ),
            ExpressionError::FunctionArity {
                name,
                expected,
                found,
            } => write!(
                f,
                "expression: `{name}` takes {expected} argument{}, given {found}",
                if *expected == 1 { "" } else { "s" }
            ),
            ExpressionError::UndeclaredVariable { name, allowed } => write!(
                f,
                "expression: `{name}` is no variable here (the variables are: {})",
                allowed.join(", ")
            ),
            ExpressionError::CyclicDefinition { cycle } => write!(
                f,
                "expression: sub-definitions refer to each other in a cycle: {}",
                cycle.join(" -> ")
            ),
            ExpressionError::DuplicateDefinition { name } => {
                write!(f, "expression: sub-definition `{name}` is defined twice")
            }
            ExpressionError::DefinitionOrder { name, used_in } => write!(
                f,
                "expression: sub-definition `{used_in}` uses `{name}`, defined before it \
                 (a definition sees only the definitions to its right)"
            ),
            ExpressionError::DefinitionShadows { name } => {
                write!(f, "expression: sub-definition `{name}` shadows a variable")
            }
            ExpressionError::NotAPoint {
                function,
                found,
                points: 0,
            } => write!(
                f,
                "expression: `{function}({found}, …)`: a pair style has no points"
            ),
            ExpressionError::NotAPoint {
                function,
                found,
                points,
            } => write!(
                f,
                "expression: `{function}` takes points p1…p{points}, given `{found}`"
            ),
            ExpressionError::PointAsNumber { name } => write!(
                f,
                "expression: point `{name}` is no number, only an argument of \
                 distance/angle/dihedral"
            ),
            ExpressionError::BadBinding { reason } => write!(f, "expression binding: {reason}"),
            ExpressionError::MissingInput { input } => {
                write!(f, "expression: no column for input `{input}`")
            }
        }
    }
}

impl std::error::Error for ExpressionError {}
