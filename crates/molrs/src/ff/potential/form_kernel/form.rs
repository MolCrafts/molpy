//! The Tier-2 calling conventions (`ff-ir-02-protocol` §3): an energy as a
//! function of one coordinate ([`ScalarForm`]) or of the atoms' positions
//! ([`CompoundForm`]), evaluated for a whole batch of terms per call.

use ndarray::ArrayViewD;

use molrs::op::F;

/// The per-term inputs of one batch.
///
/// Parameters arrive **exactly as stored** — IR units, angle values in
/// degrees; no tier converts them:
///
/// * [`get`](Self::get): each numeric per-type parameter as an `n_terms`
///   column (an indexed one as `k1 … kM`), each numeric style parameter
///   broadcast to one, and on a pair the resolved pair value of each
///   parameter (cross row, else its mixing rule) plus `q1`, `q2` when the
///   frame has `atoms.charge`;
/// * [`array`](Self::array): an array parameter, leading axis `n_terms`;
/// * [`text`](Self::text): a text per-type parameter, one string per term;
/// * [`style_text`](Self::style_text): a text style parameter.
#[derive(Clone, Debug, Default)]
pub struct ParamColumns<'a> {
    scalars: Vec<(&'a str, &'a [F])>,
    arrays: Vec<(&'a str, ArrayViewD<'a, F>)>,
    texts: Vec<(&'a str, &'a [&'a str])>,
    style_texts: Vec<(&'a str, &'a str)>,
}

impl<'a> ParamColumns<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add the numeric column `name`.
    pub fn push(&mut self, name: &'a str, col: &'a [F]) {
        self.scalars.push((name, col));
    }

    /// Add the array parameter `name` (leading axis: the terms).
    pub fn push_array(&mut self, name: &'a str, values: ArrayViewD<'a, F>) {
        self.arrays.push((name, values));
    }

    /// Add the text per-type parameter `name`.
    pub fn push_text(&mut self, name: &'a str, values: &'a [&'a str]) {
        self.texts.push((name, values));
    }

    /// Add the text style parameter `name`.
    pub fn push_style_text(&mut self, name: &'a str, value: &'a str) {
        self.style_texts.push((name, value));
    }

    /// The numeric column `name`.
    pub fn get(&self, name: &str) -> Option<&'a [F]> {
        self.scalars
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
    }

    pub fn array(&self, name: &str) -> Option<ArrayViewD<'a, F>> {
        self.arrays
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, a)| a.clone())
    }

    pub fn text(&self, name: &str) -> Option<&'a [&'a str]> {
        self.texts.iter().find(|(n, _)| *n == name).map(|(_, t)| *t)
    }

    pub fn style_text(&self, name: &str) -> Option<&'a str> {
        self.style_texts
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, t)| *t)
    }

    /// The numeric columns' names, in order.
    pub fn names(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.scalars.iter().map(|(n, _)| *n)
    }
}

/// An energy `E(q; p)` of one coordinate per term.
///
/// `q[t]` is term `t`'s coordinate, the category's
/// ([`Coordinate`](crate::ff::ir::Coordinate)): the distance `r`, the angle
/// `theta` ∈ [0, π], or the signed dihedral `phi` ∈ (−π, π] of the atoms in
/// row order (an improper too: a form wanting χ takes |φ|) — radians. The
/// form **writes** (does not add) the **unweighted** energy `e[t]` and its
/// derivative `de_dq[t]`; the kernel applies any pair weight and cutoff and
/// owns the chain rule onto Cartesian forces.
///
/// The derivative is checked against a central difference of `e`, at
/// registration on the style's samples or at its first compile.
///
/// Batch, so that a kernel written in a language with a per-call cost (a
/// numpy callable, an expression interpreter) pays it once per evaluation
/// and not once per term.
pub trait ScalarForm: Send + Sync + 'static {
    fn eval(&self, q: &[F], p: &ParamColumns<'_>, e: &mut [F], de_dq: &mut [F]);

    /// The numeric columns the form reads, by name, when it states them; a
    /// kernel refuses at build a form whose input it cannot supply
    /// ([`IrError::MissingParam`](crate::ff::ir::IrError::MissingParam)).
    /// A pair kernel also binds what only a pair has on request: `<x>1`,
    /// `<x>2`, the self-row values of the two atoms' types (OpenMM's
    /// per-particle binding, an expression feature). Default: nothing
    /// stated, nothing checked.
    fn inputs(&self) -> Vec<String> {
        Vec::new()
    }
}

/// An N-body energy `E(x; p)` of the term's atoms' positions.
///
/// `x` holds `n_terms · arity` points, term `t`'s atoms at
/// `x[t·arity .. (t+1)·arity]` in the order its block row lists them. The
/// form **writes** `e[t]` and the gradient `grad` (`∂E/∂x`, not the force,
/// laid out as `x`).
pub trait CompoundForm: Send + Sync + 'static {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    );

    /// The numeric columns the form reads; see [`ScalarForm::inputs`].
    fn inputs(&self) -> Vec<String> {
        Vec::new()
    }
}
