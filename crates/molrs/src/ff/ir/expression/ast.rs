//! The syntax tree of an expression: what the parser builds, the printer
//! prints and the compiler lowers.

use molrs::op::F;

/// A binary operator of the grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    /// `^`, right-associative and binding tighter than unary minus.
    Pow,
}

impl BinOp {
    /// The operator's character in the source.
    pub fn symbol(self) -> char {
        match self {
            BinOp::Add => '+',
            BinOp::Sub => '-',
            BinOp::Mul => '*',
            BinOp::Div => '/',
            BinOp::Pow => '^',
        }
    }

    /// Lepton's binding strength: `+ -` 0, `* /` 1, unary minus 2, `^` 3.
    pub(crate) fn precedence(self) -> u8 {
        match self {
            BinOp::Add | BinOp::Sub => 0,
            BinOp::Mul | BinOp::Div => 1,
            BinOp::Pow => 3,
        }
    }

    pub(crate) fn right_associative(self) -> bool {
        matches!(self, BinOp::Pow)
    }
}

/// Binding strength of unary minus (Lepton's: tighter than `*`, looser than `^`).
pub(crate) const NEG_PRECEDENCE: u8 = 2;

/// A function of the grammar: molrec's Lepton subset plus the three geometric
/// functions of OpenMM's `CustomCompoundBondForce`, which only a compound
/// category may call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Func {
    Exp,
    Log,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Abs,
    Min,
    Max,
    /// `step(x)`: 1 for x ≥ 0, else 0 (Lepton's).
    Step,
    /// `delta(x)`: 1 for x = 0, else 0.
    Delta,
    /// `select(x, y, z)`: y when x ≠ 0, else z.
    Select,
    /// `distance(p1, p2)`: |x₂ − x₁|.
    Distance,
    /// `angle(p1, p2, p3)`: the angle at p2, radians in [0, π].
    Angle,
    /// `dihedral(p1, p2, p3, p4)`: the signed dihedral, radians in (−π, π].
    Dihedral,
}

impl Func {
    /// Every function, in the order the grammar lists them.
    pub const ALL: [Func; 18] = [
        Func::Exp,
        Func::Log,
        Func::Sqrt,
        Func::Sin,
        Func::Cos,
        Func::Tan,
        Func::Asin,
        Func::Acos,
        Func::Atan,
        Func::Abs,
        Func::Min,
        Func::Max,
        Func::Step,
        Func::Delta,
        Func::Select,
        Func::Distance,
        Func::Angle,
        Func::Dihedral,
    ];

    /// The function's name in the source.
    pub fn name(self) -> &'static str {
        match self {
            Func::Exp => "exp",
            Func::Log => "log",
            Func::Sqrt => "sqrt",
            Func::Sin => "sin",
            Func::Cos => "cos",
            Func::Tan => "tan",
            Func::Asin => "asin",
            Func::Acos => "acos",
            Func::Atan => "atan",
            Func::Abs => "abs",
            Func::Min => "min",
            Func::Max => "max",
            Func::Step => "step",
            Func::Delta => "delta",
            Func::Select => "select",
            Func::Distance => "distance",
            Func::Angle => "angle",
            Func::Dihedral => "dihedral",
        }
    }

    /// The function a name denotes, if any.
    pub fn from_name(name: &str) -> Option<Func> {
        Func::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The number of arguments the function takes.
    pub fn arity(self) -> usize {
        match self {
            Func::Min | Func::Max | Func::Distance => 2,
            Func::Select | Func::Angle => 3,
            Func::Dihedral => 4,
            _ => 1,
        }
    }

    /// Whether the function takes points (`p1`, `p2`, …) rather than numbers.
    pub fn is_geometric(self) -> bool {
        matches!(self, Func::Distance | Func::Angle | Func::Dihedral)
    }
}

/// An expression tree.
///
/// The tree is syntax only. A call keeps its function name as written, so an
/// unknown function parses and is refused by the compiler with a named error;
/// a variable is any identifier, resolved against a
/// [`Binding`](crate::ff::ir::expression::Binding) only at compile time.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A numeric literal. The parser only builds non-negative finite ones (a
    /// leading `-` is [`Expr::Neg`]); a rewrite may build any value.
    Num(F),
    /// An identifier: the coordinate, a parameter or a sub-definition.
    Var(String),
    /// Unary minus.
    Neg(Box<Expr>),
    /// A binary operation.
    Bin(BinOp, Box<Expr>, Box<Expr>),
    /// A call, the function name as written.
    Call(String, Vec<Expr>),
}

impl Expr {
    pub fn num(v: F) -> Expr {
        Expr::Num(v)
    }

    pub fn var(name: impl Into<String>) -> Expr {
        Expr::Var(name.into())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn neg(e: Expr) -> Expr {
        Expr::Neg(Box::new(e))
    }

    pub fn bin(op: BinOp, l: Expr, r: Expr) -> Expr {
        Expr::Bin(op, Box::new(l), Box::new(r))
    }

    pub fn call(name: impl Into<String>, args: Vec<Expr>) -> Expr {
        Expr::Call(name.into(), args)
    }

    /// Replace every `Var(name)` with `with`.
    pub fn substitute(&self, name: &str, with: &Expr) -> Expr {
        match self {
            Expr::Num(v) => Expr::Num(*v),
            Expr::Var(v) if v == name => with.clone(),
            Expr::Var(v) => Expr::Var(v.clone()),
            Expr::Neg(e) => Expr::neg(e.substitute(name, with)),
            Expr::Bin(op, l, r) => {
                Expr::bin(*op, l.substitute(name, with), r.substitute(name, with))
            }
            Expr::Call(f, args) => Expr::Call(
                f.clone(),
                args.iter().map(|a| a.substitute(name, with)).collect(),
            ),
        }
    }

    /// Every identifier the tree names, in first-appearance order, without
    /// repeats (function names excluded).
    pub fn identifiers(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.walk_identifiers(&mut out);
        out
    }

    fn walk_identifiers<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Expr::Num(_) => {}
            Expr::Var(v) => {
                if !out.contains(&v.as_str()) {
                    out.push(v);
                }
            }
            Expr::Neg(e) => e.walk_identifiers(out),
            Expr::Bin(_, l, r) => {
                l.walk_identifiers(out);
                r.walk_identifiers(out);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.walk_identifiers(out)),
        }
    }
}

/// A `name=expr` sub-definition after a `;`.
#[derive(Debug, Clone, PartialEq)]
pub struct Definition {
    pub name: String,
    pub expr: Expr,
}

/// A whole expression: the energy and its sub-definitions, with the source it
/// came from.
///
/// [`source`](Parsed::source) is the string as given, byte for byte: it is
/// what a reader keeps and a writer writes back. [`Display`](std::fmt::Display)
/// is the printer, for an engine that rewrites the expression (e.g.
/// [`substitute`](Parsed::substitute) `r → 10*r`); it prints the tree, not the
/// source, so it may differ from it in spacing, parentheses and number
/// spelling, never in value.
#[derive(Debug, Clone)]
pub struct Parsed {
    pub(crate) source: String,
    pub(crate) main: Expr,
    pub(crate) defs: Vec<Definition>,
}

impl Parsed {
    /// Build an expression from a tree (for a writer that constructs one).
    /// Its [`source`](Parsed::source) is its printed form.
    pub fn from_tree(main: Expr, defs: Vec<Definition>) -> Parsed {
        let mut p = Parsed {
            source: String::new(),
            main,
            defs,
        };
        p.source = p.to_string();
        p
    }

    /// The source string, byte for byte as it was parsed.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The energy expression (before the first `;`).
    pub fn main(&self) -> &Expr {
        &self.main
    }

    /// The sub-definitions, in source order.
    pub fn defs(&self) -> &[Definition] {
        &self.defs
    }

    /// The same expression with every free occurrence of `name` replaced by
    /// `with`, in the energy and in every sub-definition. A sub-definition
    /// named `name` (which the compiler refuses anyway) is left alone.
    pub fn substitute(&self, name: &str, with: &Expr) -> Parsed {
        if self.defs.iter().any(|d| d.name == name) {
            return self.clone();
        }
        Parsed::from_tree(
            self.main.substitute(name, with),
            self.defs
                .iter()
                .map(|d| Definition {
                    name: d.name.clone(),
                    expr: d.expr.substitute(name, with),
                })
                .collect(),
        )
    }

    /// Whether two expressions have the same trees (sources may differ).
    pub fn same_tree(&self, other: &Parsed) -> bool {
        self.main == other.main && self.defs == other.defs
    }
}
