//! Batch evaluation of a [`Compiled`] expression over many terms.
//!
//! Parameters arrive as columns, one per [`Compiled::inputs`] entry, each of
//! length `n_terms` (a value per term) or 1 (one value for every term — a
//! style parameter, or a type shared by all rows).

use super::compile::{Compiled, F1, Geometry, Op, Program, delta, step};
use super::dual::{self, Dual, P3};
use super::error::ExpressionError;
use molrs::op::F;

/// The value of column `c` at term `t`, a length-1 column broadcast.
#[inline]
fn at(c: &[F], t: usize) -> F {
    if c.len() == 1 { c[0] } else { c[t] }
}

impl Compiled {
    fn check_cols(&self, cols: &[&[F]], n: usize) {
        assert_eq!(
            cols.len(),
            self.inputs.len(),
            "expression `{}`: {} input columns given, {} expected ({:?})",
            self.source(),
            cols.len(),
            self.inputs.len(),
            self.inputs
        );
        for (c, i) in cols.iter().zip(&self.inputs) {
            assert!(
                c.len() == n || c.len() == 1,
                "expression `{}`: column `{}` has {} values for {n} terms",
                self.source(),
                i.spelling(),
                c.len()
            );
        }
    }

    /// Energy and dE/dq of every term of a scalar coordinate: `q[t]` is the
    /// term's coordinate (`r`; `theta` or `phi` in radians), `cols` its
    /// parameter columns in [`inputs`](Self::inputs) order, as stored in the
    /// IR (angle-valued parameters in degrees: the expression converts them).
    ///
    /// For an improper, `q` is the signed dihedral φ and `de_dq` is dE/dφ,
    /// whether the expression names `phi` or `chi = abs(phi)`.
    ///
    /// # Panics
    /// When the expression has no scalar form
    /// ([`has_scalar_form`](Self::has_scalar_form): it uses a point function,
    /// or the category is compound), or columns that do not match `inputs`
    /// and `q.len()`.
    pub fn eval_scalar(&self, q: &[F], cols: &[&[F]], e: &mut [F], de_dq: &mut [F]) {
        let Some(prog) = &self.scalar else {
            panic!(
                "expression `{}` is a function of the points, not of one coordinate: \
                 eval_compound prices it",
                self.source()
            )
        };
        let n = q.len();
        assert!(
            e.len() == n && de_dq.len() == n,
            "eval_scalar: output lengths"
        );
        self.check_cols(cols, n);
        let improper = self.geometry() == Geometry::Improper;
        let mut stack: Vec<Dual<1>> = Vec::with_capacity(prog.max_stack);
        let mut tmps = vec![Dual::<1>::constant(0.0); prog.n_tmps];
        let mut coords = [Dual::<1>::constant(0.0); 2];
        for t in 0..n {
            coords[0] = Dual::var(q[t], 0);
            if improper {
                coords[1] = coords[0].abs();
            }
            let r = run(prog, &coords, cols, t, &mut stack, &mut tmps, &[]);
            e[t] = r.v;
            de_dq[t] = r.d[0];
        }
    }

    /// [`eval_scalar`](Self::eval_scalar) with the columns looked up by
    /// [`spelling`](super::Input::spelling) (`k`, `epsilon1`, `q2`,
    /// `coulomb`) — the shape of a named-column batch such as the generic
    /// kernels' parameter columns.
    pub fn eval_scalar_named<'a>(
        &self,
        q: &[F],
        lookup: impl Fn(&str) -> Option<&'a [F]>,
        e: &mut [F],
        de_dq: &mut [F],
    ) -> Result<(), ExpressionError> {
        let cols = self.gather(|i| lookup(&i.spelling()))?;
        self.eval_scalar(q, &cols, e, de_dq);
        Ok(())
    }

    /// [`eval_compound`](Self::eval_compound) with the columns looked up by
    /// spelling.
    pub fn eval_compound_named<'a>(
        &self,
        x: &[[F; 3]],
        lookup: impl Fn(&str) -> Option<&'a [F]>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) -> Result<(), ExpressionError> {
        let cols = self.gather(|i| lookup(&i.spelling()))?;
        self.eval_compound(x, &cols, e, grad);
        Ok(())
    }

    /// One term of a scalar coordinate: `(E, dE/dq)`, `params` one value per
    /// [`inputs`](Self::inputs) entry.
    pub fn eval_scalar_one(&self, q: F, params: &[F]) -> (F, F) {
        let cols: Vec<&[F]> = params.iter().map(std::slice::from_ref).collect();
        let (mut e, mut d) = ([0.0], [0.0]);
        self.eval_scalar(&[q], &cols, &mut e, &mut d);
        (e[0], d[0])
    }

    /// Energy and gradient of every term over its points: `x` holds
    /// `n_terms · arity` positions (term-major, the row's atoms in order,
    /// minimum-imaged), `grad` receives ∂E/∂x in the same layout (the force
    /// is its negative). Any category with points: a geometric variable is
    /// its compound function (`r` = `distance(p1,p2)`, …).
    ///
    /// # Panics
    /// For a pair (no points), or lengths that do not match.
    pub fn eval_compound(&self, x: &[[F; 3]], cols: &[&[F]], e: &mut [F], grad: &mut [[F; 3]]) {
        let Some(prog) = &self.compound else {
            panic!(
                "expression `{}`: a pair has no points to evaluate over",
                self.source()
            )
        };
        let arity = self.geometry().points();
        assert_eq!(
            x.len() % arity,
            0,
            "eval_compound: points not a multiple of the arity"
        );
        let n = x.len() / arity;
        assert!(
            e.len() == n && grad.len() == x.len(),
            "eval_compound: output lengths"
        );
        self.check_cols(cols, n);
        match arity {
            2 => compound::<6>(prog, x, cols, e, grad),
            3 => compound::<9>(prog, x, cols, e, grad),
            4 => compound::<12>(prog, x, cols, e, grad),
            5 => compound::<15>(prog, x, cols, e, grad),
            _ => unreachable!("a binding has 2..=5 points"),
        }
    }
}

fn compound<const N: usize>(
    prog: &Program,
    x: &[[F; 3]],
    cols: &[&[F]],
    e: &mut [F],
    grad: &mut [[F; 3]],
) {
    let arity = N / 3;
    let mut stack: Vec<Dual<N>> = Vec::with_capacity(prog.max_stack);
    let mut tmps = vec![Dual::<N>::constant(0.0); prog.n_tmps];
    let mut points = [[Dual::<N>::constant(0.0); 3]; 5];
    for t in 0..e.len() {
        for (p, pt) in points.iter_mut().enumerate().take(arity) {
            for (a, c) in pt.iter_mut().enumerate() {
                *c = Dual::var(x[t * arity + p][a], 3 * p + a);
            }
        }
        let r = run(prog, &[], cols, t, &mut stack, &mut tmps, &points[..arity]);
        e[t] = r.v;
        for (p, g) in grad[t * arity..(t + 1) * arity].iter_mut().enumerate() {
            g.copy_from_slice(&r.d[3 * p..3 * p + 3]);
        }
    }
}

/// Run the program for term `t`.
fn run<const N: usize>(
    prog: &Program,
    coords: &[Dual<N>],
    cols: &[&[F]],
    t: usize,
    stack: &mut Vec<Dual<N>>,
    tmps: &mut [Dual<N>],
    points: &[P3<N>],
) -> Dual<N> {
    stack.clear();
    for op in &prog.ops {
        match *op {
            Op::Const(v) => stack.push(Dual::constant(v)),
            Op::Coord(s) => stack.push(coords[s as usize]),
            Op::Input(s) => stack.push(Dual::constant(at(cols[s as usize], t))),
            Op::Load(s) => stack.push(tmps[s as usize]),
            Op::Store(s) => tmps[s as usize] = stack.pop().expect("stack"),
            Op::Neg => {
                let a = stack.pop().expect("stack");
                stack.push(-a);
            }
            Op::PowI(n) => {
                let a = stack.pop().expect("stack");
                stack.push(a.powi(n));
            }
            Op::PowF(c) => {
                let a = stack.pop().expect("stack");
                stack.push(a.powf(c));
            }
            Op::Call1(f) => {
                let a = stack.pop().expect("stack");
                stack.push(match f {
                    F1::Exp => a.exp(),
                    F1::Log => a.ln(),
                    F1::Sqrt => a.sqrt(),
                    F1::Sin => a.sin(),
                    F1::Cos => a.cos(),
                    F1::Tan => a.tan(),
                    F1::Asin => a.asin(),
                    F1::Acos => a.acos(),
                    F1::Atan => a.atan(),
                    F1::Abs => a.abs(),
                    F1::Step => Dual::constant(step(a.v)),
                    F1::Delta => Dual::constant(delta(a.v)),
                });
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow | Op::Min | Op::Max => {
                let b = stack.pop().expect("stack");
                let a = stack.pop().expect("stack");
                stack.push(match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                    Op::Pow => a.pow(b),
                    Op::Min => a.min(b),
                    _ => a.max(b),
                });
            }
            Op::Select => {
                let z = stack.pop().expect("stack");
                let y = stack.pop().expect("stack");
                let x = stack.pop().expect("stack");
                stack.push(if x.v != 0.0 { y } else { z });
            }
            Op::Distance([a, b]) => {
                stack.push(dual::distance(points[a as usize], points[b as usize]))
            }
            Op::Angle([a, b, c]) => stack.push(dual::angle(
                points[a as usize],
                points[b as usize],
                points[c as usize],
            )),
            Op::Dihedral([a, b, c, d]) => stack.push(dual::dihedral(
                points[a as usize],
                points[b as usize],
                points[c as usize],
                points[d as usize],
            )),
        }
    }
    stack.pop().expect("a program leaves its value")
}
