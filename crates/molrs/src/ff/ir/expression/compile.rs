//! Static checks and lowering to bytecode.

use std::collections::HashMap;

use super::ast::{BinOp, Expr, Func, Parsed};
use super::error::ExpressionError;
use super::parse::parse;
use molrs::op::F;

/// What a category's rows give an expression: its geometric variables and
/// its points.
///
/// | geometry | variables | points |
/// |---|---|---|
/// | `Bond` (bond, drude) | `r` = `distance(p1,p2)` | `p1`, `p2` |
/// | `Angle` | `theta` = `angle(p1,p2,p3)` | `p1`…`p3` |
/// | `Dihedral` | `phi` = `dihedral(p1,p2,p3,p4)` | `p1`…`p4` |
/// | `Improper` | `phi` as above, `chi` = `abs(phi)` | `p1`…`p4` |
/// | `Pair` | `r`, `q1`, `q2`; `x1`/`x2` self rows | none |
/// | `Compound { arity }` (cmap, a custom category) | none | `p1`…`p`*arity* |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Geometry {
    Bond,
    Angle,
    Dihedral,
    Improper,
    Pair,
    /// 2 ≤ arity ≤ 5.
    Compound {
        arity: usize,
    },
}

/// A geometric variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coord {
    R,
    Theta,
    Phi,
    Chi,
}

/// Names no parameter may take: the geometric variables and the points of
/// any category. A function name is not reserved: a call is a name followed
/// by `(`, so `delta` (`pair coul/cut`'s style parameter) and `delta(x)`
/// never meet.
fn reserved(name: &str) -> bool {
    matches!(name, "r" | "theta" | "phi" | "chi")
        || point_index(name.strip_prefix('p').unwrap_or(""), 5).is_some()
}

impl Geometry {
    /// The geometry of a category by its name, for the categories molrec's
    /// variable table names; `None` for one whose rows price no energy
    /// (`atom`, `constraint`, `virtual_site`) or a custom category (bind
    /// it as `Compound` with its arity).
    pub fn of_category(category: &str) -> Option<Geometry> {
        Some(match category {
            "bond" | "drude" => Geometry::Bond,
            "angle" => Geometry::Angle,
            "dihedral" => Geometry::Dihedral,
            "improper" => Geometry::Improper,
            "pair" => Geometry::Pair,
            "cmap" => Geometry::Compound { arity: 5 },
            _ => return None,
        })
    }

    /// How many points a term has (0 for a pair).
    pub fn points(self) -> usize {
        match self {
            Geometry::Bond => 2,
            Geometry::Angle => 3,
            Geometry::Dihedral | Geometry::Improper => 4,
            Geometry::Pair => 0,
            Geometry::Compound { arity } => arity,
        }
    }

    /// The scalar coordinate a scalar evaluation differentiates by (`None`
    /// for a compound category).
    pub fn coordinate(self) -> Option<&'static str> {
        match self {
            Geometry::Bond | Geometry::Pair => Some("r"),
            Geometry::Angle => Some("theta"),
            Geometry::Dihedral | Geometry::Improper => Some("phi"),
            Geometry::Compound { .. } => None,
        }
    }

    fn coord(self, name: &str) -> Option<Coord> {
        match (self, name) {
            (Geometry::Bond | Geometry::Pair, "r") => Some(Coord::R),
            (Geometry::Angle, "theta") => Some(Coord::Theta),
            (Geometry::Dihedral | Geometry::Improper, "phi") => Some(Coord::Phi),
            (Geometry::Improper, "chi") => Some(Coord::Chi),
            _ => None,
        }
    }

    /// The point an identifier `pk` names (0-based), if within the arity.
    pub(crate) fn point(self, name: &str) -> Option<usize> {
        point_index(name.strip_prefix('p')?, self.points())
    }

    fn variable_names(self) -> Vec<String> {
        let mut out: Vec<String> = match self {
            Geometry::Bond | Geometry::Pair => vec!["r".into()],
            Geometry::Angle => vec!["theta".into()],
            Geometry::Dihedral => vec!["phi".into()],
            Geometry::Improper => vec!["phi".into(), "chi".into()],
            Geometry::Compound { .. } => vec![],
        };
        if self == Geometry::Pair {
            out.extend(["q1".to_owned(), "q2".to_owned()]);
        }
        out
    }
}

/// `"3"` → 2 when 3 ≤ arity: a 1-based point number without leading zeros.
fn point_index(digits: &str, arity: usize) -> Option<usize> {
    if digits.is_empty() || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: usize = digits.parse().ok()?;
    (1..=arity).contains(&n).then(|| n - 1)
}

/// What an expression is compiled against: the category's [`Geometry`] and
/// the style's numeric parameters (array and text parameters are no
/// expression variables).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    pub geometry: Geometry,
    /// Numeric per-type parameter names, in the style's declared (`*_coeff`)
    /// order.
    pub params: Vec<String>,
    /// Numeric style-level parameter names (one value per style, e.g.
    /// `coulomb`, `cutoff`).
    pub style_params: Vec<String>,
}

impl Binding {
    pub fn new(geometry: Geometry, params: &[&str], style_params: &[&str]) -> Binding {
        Binding {
            geometry,
            params: params.iter().map(|s| s.to_string()).collect(),
            style_params: style_params.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn bad(reason: String) -> ExpressionError {
        ExpressionError::BadBinding { reason }
    }

    fn validate(&self) -> Result<(), ExpressionError> {
        if let Geometry::Compound { arity } = self.geometry
            && !(2..=5).contains(&arity)
        {
            return Err(Self::bad(format!(
                "a compound category has 2..=5 points, not {arity}"
            )));
        }
        let mut seen: Vec<&str> = Vec::new();
        for name in self.params.iter().chain(&self.style_params) {
            if !is_identifier(name) {
                return Err(Self::bad(format!("parameter `{name}` is no identifier")));
            }
            if seen.contains(&name.as_str()) {
                return Err(Self::bad(format!("parameter `{name}` is declared twice")));
            }
            if reserved(name) {
                return Err(Self::bad(format!(
                    "parameter `{name}` is a geometric variable or a point"
                )));
            }
            if self.geometry == Geometry::Pair {
                if matches!(name.as_str(), "q" | "q1" | "q2") {
                    return Err(Self::bad(format!(
                        "pair parameter `{name}`: `q1`, `q2` are the charges"
                    )));
                }
                if let Some(stem) = name.strip_suffix(['1', '2'])
                    && self.params.iter().any(|p| p == stem)
                {
                    return Err(Self::bad(format!(
                        "pair parameter `{name}` is `{stem}`'s self-row spelling"
                    )));
                }
            }
            seen.push(name);
        }
        Ok(())
    }

    /// Whether `name` is a variable of the binding (a definition may not
    /// take it).
    fn takes(&self, name: &str) -> bool {
        self.geometry.coord(name).is_some()
            || self.geometry.point(name).is_some()
            || self.params.iter().any(|p| p == name)
            || self.style_params.iter().any(|p| p == name)
            || (self.geometry == Geometry::Pair && pair_self(name, self).is_some())
    }

    fn allowed(&self) -> Vec<String> {
        let mut out = self.geometry.variable_names();
        out.extend(self.params.iter().cloned());
        out.extend(self.style_params.iter().cloned());
        if self.geometry == Geometry::Pair {
            out.extend(self.params.iter().map(|p| format!("{p}1")));
            out.extend(self.params.iter().map(|p| format!("{p}2")));
        }
        out
    }
}

/// `[A-Za-z_][A-Za-z0-9_]*`.
pub fn is_identifier(s: &str) -> bool {
    let mut b = s.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_alphabetic() || c == b'_')
        && b.all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

/// A column the evaluation reads, per term (or one value broadcast).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Input {
    /// A per-type parameter of the row (for a pair: the pair value — the
    /// cross row's, else mixed).
    Param(String),
    /// A per-type parameter of the self row of the pair's first (`atom` 1)
    /// or second (`atom` 2) atom's type: `name1`, `name2`.
    SelfParam { name: String, atom: u8 },
    /// The charge of the pair's first or second atom: `q1`, `q2`.
    Charge(u8),
    /// A style-level parameter.
    StyleParam(String),
}

impl Input {
    /// The identifier the expression spells it with.
    pub fn spelling(&self) -> String {
        match self {
            Input::Param(n) | Input::StyleParam(n) => n.clone(),
            Input::SelfParam { name, atom } => format!("{name}{atom}"),
            Input::Charge(atom) => format!("q{atom}"),
        }
    }

    /// The same input with the pair's atoms exchanged (`epsilon1` ↔
    /// `epsilon2`, `q1` ↔ `q2`): the pair-symmetry check evaluates an
    /// expression on swapped columns and compares.
    pub fn swapped(&self) -> Input {
        match self {
            Input::SelfParam { name, atom } => Input::SelfParam {
                name: name.clone(),
                atom: 3 - atom,
            },
            Input::Charge(atom) => Input::Charge(3 - atom),
            other => other.clone(),
        }
    }

    fn order_key(&self, b: &Binding) -> (u8, usize, u8) {
        let pos = |n: &str, list: &[String]| list.iter().position(|p| p == n).unwrap_or(usize::MAX);
        match self {
            Input::Param(n) => (0, pos(n, &b.params), 0),
            Input::SelfParam { name, atom } => (*atom, pos(name, &b.params), 0),
            Input::Charge(atom) => (3, 0, *atom),
            Input::StyleParam(n) => (4, pos(n, &b.style_params), 0),
        }
    }
}

/// An elementary one-argument function of the bytecode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum F1 {
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
    Step,
    Delta,
}

impl F1 {
    fn of(f: Func) -> Option<F1> {
        Some(match f {
            Func::Exp => F1::Exp,
            Func::Log => F1::Log,
            Func::Sqrt => F1::Sqrt,
            Func::Sin => F1::Sin,
            Func::Cos => F1::Cos,
            Func::Tan => F1::Tan,
            Func::Asin => F1::Asin,
            Func::Acos => F1::Acos,
            Func::Atan => F1::Atan,
            Func::Abs => F1::Abs,
            Func::Step => F1::Step,
            Func::Delta => F1::Delta,
            _ => return None,
        })
    }

    pub(crate) fn apply(self, x: F) -> F {
        match self {
            F1::Exp => x.exp(),
            F1::Log => x.ln(),
            F1::Sqrt => x.sqrt(),
            F1::Sin => x.sin(),
            F1::Cos => x.cos(),
            F1::Tan => x.tan(),
            F1::Asin => x.asin(),
            F1::Acos => x.acos(),
            F1::Atan => x.atan(),
            F1::Abs => x.abs(),
            F1::Step => step(x),
            F1::Delta => delta(x),
        }
    }
}

pub(crate) fn step(x: F) -> F {
    if x >= 0.0 { 1.0 } else { 0.0 }
}

pub(crate) fn delta(x: F) -> F {
    if x == 0.0 { 1.0 } else { 0.0 }
}

/// One instruction of a stack program.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Op {
    Const(F),
    Coord(u16),
    Input(u16),
    Load(u16),
    /// Pop into a temporary.
    Store(u16),
    Neg,
    Add,
    Sub,
    Mul,
    Div,
    /// `x^y`, both operands on the stack.
    Pow,
    /// `x^n`, n a constant integer.
    PowI(i32),
    /// `x^c`, c a constant.
    PowF(F),
    Call1(F1),
    Min,
    Max,
    Select,
    Distance([u8; 2]),
    Angle([u8; 3]),
    Dihedral([u8; 4]),
}

/// A stack program and the scratch it needs.
#[derive(Debug, Clone)]
pub(crate) struct Program {
    pub ops: Vec<Op>,
    pub n_tmps: usize,
    pub max_stack: usize,
}

/// A compiled expression: the kernel of a style that carries an `expression`.
///
/// It keeps the source byte for byte ([`source`](Compiled::source)) and
/// evaluates the energy with exact derivatives (forward-mode duals) over a
/// batch of terms: [`eval_scalar`](Compiled::eval_scalar) when the energy is
/// a function of the scalar coordinate alone
/// ([`has_scalar_form`](Compiled::has_scalar_form)),
/// [`eval_compound`](Compiled::eval_compound) over the points whenever the
/// category has points.
///
/// [`compile`] resolves every identifier of a [`Parsed`] expression against a
/// [`Binding`] — the category's geometric variables and points, the style's
/// numeric parameters, the sub-definitions — refusing what does not resolve
/// with a named [`ExpressionError`]. What resolves is lowered to stack programs
/// over three kinds of slot:
///
/// - **coordinates**, dual numbers the evaluator seeds: the scalar
///   coordinate (`r`; `theta`; `phi`, and `chi = |phi|` for an improper) in
///   the scalar program, the 3·arity point coordinates in the compound one;
/// - **inputs**, the parameter columns the caller supplies, listed by
///   [`Compiled::inputs`];
/// - **temporaries**, the sub-definitions, each evaluated once per term.
///
/// Two programs, as the protocol's §4 has it: a **scalar** one (N = 1 dual
/// on the coordinate) when the expression is a function of the coordinate
/// alone, and a **compound** one (N = 3·arity duals on the points, the
/// geometric variable lowered to its compound function) whenever the
/// category has points. Every subtree with no coordinate and no input is
/// folded to a constant, so `2^(1/6)` costs nothing per term and
/// `(sigma/r)^12` is an integer power.
#[derive(Debug, Clone)]
pub struct Compiled {
    pub(crate) parsed: Parsed,
    pub(crate) binding: Binding,
    pub(crate) inputs: Vec<Input>,
    pub(crate) scalar: Option<Program>,
    pub(crate) compound: Option<Program>,
}

impl Compiled {
    /// The source string, byte for byte.
    pub fn source(&self) -> &str {
        self.parsed.source()
    }

    /// The parsed tree (for the printer and engine rewriting).
    pub fn parsed(&self) -> &Parsed {
        &self.parsed
    }

    pub fn binding(&self) -> &Binding {
        &self.binding
    }

    pub fn geometry(&self) -> Geometry {
        self.binding.geometry
    }

    /// Whether the energy is a function of the scalar coordinate alone
    /// (no point function, not a compound category), so
    /// [`eval_scalar`](Self::eval_scalar) prices it: a `ScalarForm`.
    /// Otherwise it is a `CompoundForm` over the points.
    pub fn has_scalar_form(&self) -> bool {
        self.scalar.is_some()
    }

    /// The columns the evaluation reads, in the order `cols` passes them:
    /// only those the expression uses — per-type (pair-value) parameters in
    /// declared order, then the first atom's self-row parameters, the
    /// second's, the charges `q1`, `q2`, and the style parameters.
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    /// Every variable the expression reads that its own sub-definitions do
    /// not bind, in first-appearance order: the geometric variables it uses
    /// and the [`spelling`](Input::spelling) of every input; not the points
    /// (`p1` in `distance(p1, p2)`), which are no variables.
    pub fn variables(&self) -> Vec<String> {
        let defs: Vec<&str> = self.parsed.defs.iter().map(|d| d.name.as_str()).collect();
        let mut out: Vec<String> = Vec::new();
        let exprs =
            std::iter::once(&self.parsed.main).chain(self.parsed.defs.iter().map(|d| &d.expr));
        for e in exprs {
            for id in e.identifiers() {
                if !defs.contains(&id)
                    && self.binding.geometry.point(id).is_none()
                    && !out.iter().any(|o| o == id)
                {
                    out.push(id.to_owned());
                }
            }
        }
        out
    }

    /// The inputs a pair kernel binds besides the per-type pair values —
    /// `x1`/`x2` self rows and `q1`/`q2` charges — by spelling, in
    /// [`inputs`](Self::inputs) order.
    pub fn pair_inputs(&self) -> Vec<String> {
        self.inputs
            .iter()
            .filter(|i| matches!(i, Input::SelfParam { .. } | Input::Charge(_)))
            .map(Input::spelling)
            .collect()
    }

    /// Gather the input columns through `lookup`, in [`inputs`](Self::inputs)
    /// order, refusing the first one it cannot supply.
    pub fn gather<'a>(
        &self,
        mut lookup: impl FnMut(&Input) -> Option<&'a [F]>,
    ) -> Result<Vec<&'a [F]>, ExpressionError> {
        self.inputs
            .iter()
            .map(|i| {
                lookup(i).ok_or_else(|| ExpressionError::MissingInput {
                    input: i.spelling(),
                })
            })
            .collect()
    }
}

/// Lowering mode: which program is being built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Scalar,
    Compound,
}

/// Why a lowering stopped: a refusal, or (scalar mode only) a point
/// function, which means the expression has no scalar form.
enum Stop {
    Refused(ExpressionError),
    NeedsPoints,
}

impl From<ExpressionError> for Stop {
    fn from(e: ExpressionError) -> Self {
        Stop::Refused(e)
    }
}

/// The lowered, constant-folded tree.
#[derive(Debug, Clone)]
enum Node {
    Const(F),
    Coord(u16),
    Input(u16),
    Tmp(u16),
    Neg(Box<Node>),
    Bin(BinOp, Box<Node>, Box<Node>),
    Call(F1, Box<Node>),
    Min(Box<Node>, Box<Node>),
    Max(Box<Node>, Box<Node>),
    Select(Box<Node>, Box<Node>, Box<Node>),
    Distance([u8; 2]),
    Angle([u8; 3]),
    Dihedral([u8; 4]),
}

fn konst(n: &Node) -> Option<F> {
    match n {
        Node::Const(v) => Some(*v),
        _ => None,
    }
}

struct Lowerer<'a> {
    b: &'a Binding,
    mode: Mode,
    def_index: &'a HashMap<&'a str, usize>,
    /// Inputs by provisional slot, shared by both modes (re-sorted at the end).
    inputs: &'a mut Vec<Input>,
    /// Each lowered definition: inlined (constant, slot) or a temporary.
    def_value: Vec<Option<Node>>,
}

impl Lowerer<'_> {
    fn input(&mut self, input: Input) -> Node {
        let slot = match self.inputs.iter().position(|i| *i == input) {
            Some(s) => s,
            None => {
                self.inputs.push(input);
                self.inputs.len() - 1
            }
        };
        Node::Input(slot as u16)
    }

    fn coord(&self, c: Coord) -> Node {
        match (self.mode, c) {
            (Mode::Scalar, Coord::Chi) => Node::Coord(1),
            (Mode::Scalar, _) => Node::Coord(0),
            (Mode::Compound, Coord::R) => Node::Distance([0, 1]),
            (Mode::Compound, Coord::Theta) => Node::Angle([0, 1, 2]),
            (Mode::Compound, Coord::Phi) => Node::Dihedral([0, 1, 2, 3]),
            (Mode::Compound, Coord::Chi) => {
                Node::Call(F1::Abs, Box::new(Node::Dihedral([0, 1, 2, 3])))
            }
        }
    }

    fn var(&mut self, name: &str) -> Result<Node, Stop> {
        let g = self.b.geometry;
        if let Some(c) = g.coord(name) {
            return Ok(self.coord(c));
        }
        if g.point(name).is_some() {
            return Err(ExpressionError::PointAsNumber {
                name: name.to_owned(),
            }
            .into());
        }
        if let Some(&d) = self.def_index.get(name) {
            return Ok(self.def_value[d]
                .clone()
                .expect("a definition is lowered before the ones to its left"));
        }
        let input = if self.b.params.iter().any(|p| p == name) {
            Input::Param(name.to_owned())
        } else if self.b.style_params.iter().any(|p| p == name) {
            Input::StyleParam(name.to_owned())
        } else if let Some(i) = (g == Geometry::Pair)
            .then(|| pair_self(name, self.b))
            .flatten()
        {
            i
        } else {
            return Err(ExpressionError::UndeclaredVariable {
                name: name.to_owned(),
                allowed: self.b.allowed(),
            }
            .into());
        };
        Ok(self.input(input))
    }

    fn lower(&mut self, e: &Expr) -> Result<Node, Stop> {
        Ok(match e {
            Expr::Num(v) => Node::Const(*v),
            Expr::Var(name) => self.var(name)?,
            Expr::Neg(x) => {
                let x = self.lower(x)?;
                match konst(&x) {
                    Some(v) => Node::Const(-v),
                    None => Node::Neg(Box::new(x)),
                }
            }
            Expr::Bin(op, l, r) => {
                let l = self.lower(l)?;
                let r = self.lower(r)?;
                match (konst(&l), konst(&r)) {
                    (Some(a), Some(b)) => Node::Const(fold_bin(*op, a, b)),
                    _ => Node::Bin(*op, Box::new(l), Box::new(r)),
                }
            }
            Expr::Call(name, args) => self.call(name, args)?,
        })
    }

    fn call(&mut self, name: &str, args: &[Expr]) -> Result<Node, Stop> {
        // Name and arity were checked over the whole tree before lowering.
        let f = Func::from_name(name).expect("checked");
        if f.is_geometric() {
            let g = self.b.geometry;
            let mut pts = [0u8; 4];
            for (slot, a) in pts.iter_mut().zip(args) {
                let p = match a {
                    Expr::Var(v) => g.point(v),
                    _ => None,
                };
                *slot = p.ok_or_else(|| ExpressionError::NotAPoint {
                    function: name.to_owned(),
                    found: a.to_string(),
                    points: g.points(),
                })? as u8;
            }
            if self.mode == Mode::Scalar {
                return Err(Stop::NeedsPoints);
            }
            return Ok(match f {
                Func::Distance => Node::Distance([pts[0], pts[1]]),
                Func::Angle => Node::Angle([pts[0], pts[1], pts[2]]),
                _ => Node::Dihedral(pts),
            });
        }
        let mut a = args
            .iter()
            .map(|x| self.lower(x))
            .collect::<Result<Vec<_>, _>>()?;
        if let Some(f1) = F1::of(f) {
            let x = a.pop().expect("arity 1");
            return Ok(match konst(&x) {
                Some(v) => Node::Const(f1.apply(v)),
                None => Node::Call(f1, Box::new(x)),
            });
        }
        let consts: Option<Vec<F>> = a.iter().map(konst).collect();
        Ok(match f {
            Func::Min | Func::Max => {
                let y = Box::new(a.pop().expect("arity 2"));
                let x = Box::new(a.pop().expect("arity 2"));
                match (consts, f) {
                    (Some(v), Func::Min) => Node::Const(if v[0] <= v[1] { v[0] } else { v[1] }),
                    (Some(v), _) => Node::Const(if v[0] >= v[1] { v[0] } else { v[1] }),
                    (None, Func::Min) => Node::Min(x, y),
                    (None, _) => Node::Max(x, y),
                }
            }
            Func::Select => {
                let z = a.pop().expect("arity 3");
                let y = a.pop().expect("arity 3");
                let x = a.pop().expect("arity 3");
                // A constant condition picks its branch now.
                match konst(&x) {
                    Some(c) if c != 0.0 => y,
                    Some(_) => z,
                    None => Node::Select(Box::new(x), Box::new(y), Box::new(z)),
                }
            }
            _ => unreachable!("every other function is F1 or geometric"),
        })
    }
}

/// `epsilon1` → the first atom's `epsilon` self row; `q2` → the second
/// atom's charge.
fn pair_self(name: &str, b: &Binding) -> Option<Input> {
    let atom = match name.as_bytes().last()? {
        b'1' => 1u8,
        b'2' => 2u8,
        _ => return None,
    };
    let stem = &name[..name.len() - 1];
    if stem == "q" {
        Some(Input::Charge(atom))
    } else if b.params.iter().any(|p| p == stem) {
        Some(Input::SelfParam {
            name: stem.to_owned(),
            atom,
        })
    } else {
        None
    }
}

fn fold_bin(op: BinOp, a: F, b: F) -> F {
    match op {
        BinOp::Add => a + b,
        BinOp::Sub => a - b,
        BinOp::Mul => a * b,
        BinOp::Div => a / b,
        BinOp::Pow => pow_const(a, b),
    }
}

/// `a^b` for a constant `b` the way the evaluator computes it, so folding
/// and evaluating agree bit for bit.
pub(crate) fn pow_const(a: F, b: F) -> F {
    match as_small_int(b) {
        Some(n) => a.powi(n),
        None => a.powf(b),
    }
}

/// An exponent that is an exact small integer is taken by `powi`.
pub(crate) fn as_small_int(b: F) -> Option<i32> {
    (b.fract() == 0.0 && b.abs() <= 64.0).then_some(b as i32)
}

/// Emit the program of a node; track the stack depth.
struct Emitter {
    ops: Vec<Op>,
    depth: usize,
    max: usize,
}

impl Emitter {
    fn push(&mut self, op: Op, delta: isize) {
        self.ops.push(op);
        self.depth = (self.depth as isize + delta) as usize;
        self.max = self.max.max(self.depth);
    }

    fn emit(&mut self, n: &Node) {
        match n {
            Node::Const(v) => self.push(Op::Const(*v), 1),
            Node::Coord(s) => self.push(Op::Coord(*s), 1),
            Node::Input(s) => self.push(Op::Input(*s), 1),
            Node::Tmp(s) => self.push(Op::Load(*s), 1),
            Node::Neg(x) => {
                self.emit(x);
                self.push(Op::Neg, 0);
            }
            Node::Bin(BinOp::Pow, l, r) if konst(r).is_some() => {
                self.emit(l);
                let c = konst(r).expect("checked");
                match as_small_int(c) {
                    Some(n) => self.push(Op::PowI(n), 0),
                    None => self.push(Op::PowF(c), 0),
                }
            }
            Node::Bin(op, l, r) => {
                self.emit(l);
                self.emit(r);
                let op = match op {
                    BinOp::Add => Op::Add,
                    BinOp::Sub => Op::Sub,
                    BinOp::Mul => Op::Mul,
                    BinOp::Div => Op::Div,
                    BinOp::Pow => Op::Pow,
                };
                self.push(op, -1);
            }
            Node::Call(f, x) => {
                self.emit(x);
                self.push(Op::Call1(*f), 0);
            }
            Node::Min(a, b) | Node::Max(a, b) => {
                self.emit(a);
                self.emit(b);
                let op = if matches!(n, Node::Min(..)) {
                    Op::Min
                } else {
                    Op::Max
                };
                self.push(op, -1);
            }
            Node::Select(x, y, z) => {
                self.emit(x);
                self.emit(y);
                self.emit(z);
                self.push(Op::Select, -2);
            }
            Node::Distance(p) => self.push(Op::Distance(*p), 1),
            Node::Angle(p) => self.push(Op::Angle(*p), 1),
            Node::Dihedral(p) => self.push(Op::Dihedral(*p), 1),
        }
    }
}

/// Check every call in a tree (function known, arity right).
fn check_calls(e: &Expr) -> Result<(), ExpressionError> {
    match e {
        Expr::Num(_) | Expr::Var(_) => Ok(()),
        Expr::Neg(x) => check_calls(x),
        Expr::Bin(_, l, r) => {
            check_calls(l)?;
            check_calls(r)
        }
        Expr::Call(name, args) => {
            let f = Func::from_name(name)
                .ok_or_else(|| ExpressionError::UnknownFunction { name: name.clone() })?;
            if args.len() != f.arity() {
                return Err(ExpressionError::FunctionArity {
                    name: name.clone(),
                    expected: f.arity(),
                    found: args.len(),
                });
            }
            args.iter().try_for_each(check_calls)
        }
    }
}

/// The definitions each definition names directly.
fn def_deps(parsed: &Parsed, def_index: &HashMap<&str, usize>) -> Vec<Vec<usize>> {
    parsed
        .defs
        .iter()
        .map(|d| {
            d.expr
                .identifiers()
                .into_iter()
                .filter_map(|id| def_index.get(id).copied())
                .collect()
        })
        .collect()
}

/// A cycle among the definitions, in cycle order (first name repeated).
fn find_cycle(parsed: &Parsed, deps: &[Vec<usize>]) -> Option<Vec<String>> {
    // 0 = unvisited, 1 = on the path, 2 = done
    fn visit(
        d: usize,
        deps: &[Vec<usize>],
        state: &mut [u8],
        path: &mut Vec<usize>,
    ) -> Option<usize> {
        match state[d] {
            2 => return None,
            1 => return Some(d),
            _ => {}
        }
        state[d] = 1;
        path.push(d);
        for &u in &deps[d] {
            if let Some(hit) = visit(u, deps, state, path) {
                return Some(hit);
            }
        }
        path.pop();
        state[d] = 2;
        None
    }
    let mut state = vec![0u8; deps.len()];
    for d in 0..deps.len() {
        let mut path = Vec::new();
        if let Some(hit) = visit(d, deps, &mut state, &mut path) {
            let from = path.iter().position(|&p| p == hit).expect("on the path");
            let mut cycle: Vec<String> = path[from..]
                .iter()
                .map(|&p| parsed.defs[p].name.clone())
                .collect();
            cycle.push(parsed.defs[hit].name.clone());
            return Some(cycle);
        }
    }
    None
}

/// The checks before lowering: calls, definition names, cycles and
/// Lepton's definition order.
fn check(parsed: &Parsed, b: &Binding) -> Result<(), ExpressionError> {
    check_calls(&parsed.main)?;
    for d in &parsed.defs {
        check_calls(&d.expr)?;
    }
    let mut def_index: HashMap<&str, usize> = HashMap::new();
    for (i, d) in parsed.defs.iter().enumerate() {
        if def_index.insert(d.name.as_str(), i).is_some() {
            return Err(ExpressionError::DuplicateDefinition {
                name: d.name.clone(),
            });
        }
        if b.takes(&d.name) {
            return Err(ExpressionError::DefinitionShadows {
                name: d.name.clone(),
            });
        }
    }
    let deps = def_deps(parsed, &def_index);
    if let Some(cycle) = find_cycle(parsed, &deps) {
        return Err(ExpressionError::CyclicDefinition { cycle });
    }
    // Lepton: a definition sees only the definitions to its right.
    for (i, ds) in deps.iter().enumerate() {
        if let Some(&j) = ds.iter().find(|&&j| j < i) {
            return Err(ExpressionError::DefinitionOrder {
                name: parsed.defs[j].name.clone(),
                used_in: parsed.defs[i].name.clone(),
            });
        }
    }
    Ok(())
}

/// Lower every definition (right to left, so each sees its right-hand
/// neighbours) and the energy. Every definition is lowered, used or not,
/// so a refusal in an unused one is not missed; only the used ones are
/// emitted. Returns the program over its own input slots and those inputs.
fn lower_program(
    parsed: &Parsed,
    b: &Binding,
    mode: Mode,
    def_index: &HashMap<&str, usize>,
) -> Result<(Program, Vec<Input>), Stop> {
    let mut inputs = Vec::new();
    let mut lw = Lowerer {
        b,
        mode,
        def_index,
        inputs: &mut inputs,
        def_value: vec![None; parsed.defs.len()],
    };
    let n_defs = parsed.defs.len();
    let mut nodes: Vec<Option<Node>> = vec![None; n_defs];
    for d in (0..n_defs).rev() {
        let node = lw.lower(&parsed.defs[d].expr)?;
        lw.def_value[d] = Some(match node {
            Node::Const(_) | Node::Coord(_) | Node::Input(_) | Node::Tmp(_) => node,
            other => {
                // Provisional slot = the definition's index; renumbered below.
                nodes[d] = Some(other);
                Node::Tmp(d as u16)
            }
        });
    }
    let main = lw.lower(&parsed.main)?;

    // The temporaries the energy uses, transitively. A definition only uses
    // definitions to its right, so a left-to-right pass sees every user
    // before what it uses.
    fn mark(n: &Node, used: &mut [bool]) {
        match n {
            Node::Tmp(s) => used[*s as usize] = true,
            Node::Neg(x) | Node::Call(_, x) => mark(x, used),
            Node::Bin(_, x, y) | Node::Min(x, y) | Node::Max(x, y) => {
                mark(x, used);
                mark(y, used);
            }
            Node::Select(x, y, z) => {
                mark(x, used);
                mark(y, used);
                mark(z, used);
            }
            _ => {}
        }
    }
    let mut used = vec![false; n_defs];
    mark(&main, &mut used);
    for d in 0..n_defs {
        if used[d]
            && let Some(n) = &nodes[d]
        {
            mark(n, &mut used);
        }
    }

    let mut em = Emitter {
        ops: Vec::new(),
        depth: 0,
        max: 0,
    };
    let mut slot_of = vec![0u16; n_defs];
    let mut n_tmps = 0u16;
    for d in (0..n_defs).rev() {
        if let (true, Some(n)) = (used[d], &nodes[d]) {
            em.emit(n);
            em.push(Op::Store(d as u16), -1);
            slot_of[d] = n_tmps;
            n_tmps += 1;
        }
    }
    em.emit(&main);
    for op in &mut em.ops {
        if let Op::Load(s) | Op::Store(s) = op {
            *s = slot_of[*s as usize];
        }
    }
    Ok((
        Program {
            ops: em.ops,
            n_tmps: n_tmps as usize,
            max_stack: em.max.max(1),
        },
        inputs,
    ))
}

/// Parse and compile `src` against `binding`.
pub fn compile(src: &str, binding: &Binding) -> Result<Compiled, ExpressionError> {
    compile_parsed(parse(src)?, binding)
}

/// Compile an already parsed expression (e.g. a rewritten one).
pub fn compile_parsed(parsed: Parsed, binding: &Binding) -> Result<Compiled, ExpressionError> {
    binding.validate()?;
    check(&parsed, binding)?;
    let def_index: HashMap<&str, usize> = parsed
        .defs
        .iter()
        .enumerate()
        .map(|(i, d)| (d.name.as_str(), i))
        .collect();

    let g = binding.geometry;
    // The compound program first: it sees the points, so every refusal
    // about them is raised there.
    let compound = if g.points() > 0 {
        match lower_program(&parsed, binding, Mode::Compound, &def_index) {
            Ok(p) => Some(p),
            Err(Stop::Refused(e)) => return Err(e),
            Err(Stop::NeedsPoints) => unreachable!("the compound mode has points"),
        }
    } else {
        None
    };
    let scalar = if g.coordinate().is_some() {
        match lower_program(&parsed, binding, Mode::Scalar, &def_index) {
            Ok(p) => Some(p),
            Err(Stop::NeedsPoints) => None,
            Err(Stop::Refused(e)) => return Err(e),
        }
    } else {
        None
    };

    // The inputs: what an emitted program reads, in canonical order.
    let mut inputs: Vec<Input> = Vec::new();
    for (prog, local) in compound.iter().chain(scalar.iter()) {
        for op in &prog.ops {
            if let Op::Input(s) = op
                && !inputs.contains(&local[*s as usize])
            {
                inputs.push(local[*s as usize].clone());
            }
        }
    }
    inputs.sort_by_key(|i| i.order_key(binding));
    let renumber = |p: Option<(Program, Vec<Input>)>| {
        p.map(|(mut p, local)| {
            for op in &mut p.ops {
                if let Op::Input(s) = op {
                    let i = &local[*s as usize];
                    *s = inputs.iter().position(|x| x == i).expect("collected") as u16;
                }
            }
            p
        })
    };
    let (scalar, compound) = (renumber(scalar), renumber(compound));
    Ok(Compiled {
        parsed,
        binding: binding.clone(),
        inputs,
        scalar,
        compound,
    })
}
