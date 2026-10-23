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
    style_nums: Vec<(&'a str, F)>,
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

    /// Add the numeric style parameter `name`, one value for the batch. Its
    /// broadcast column is pushed separately ([`push`](Self::push)).
    pub fn push_style_num(&mut self, name: &'a str, value: F) {
        self.style_nums.push((name, value));
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

    /// The numeric column `name` of a form that states it among its
    /// [`inputs`](ScalarForm::inputs): every kernel refuses at build a form
    /// whose stated input it cannot supply, so a stated input is always
    /// here.
    ///
    /// # Panics
    ///
    /// When `name` is absent: the form read an input it did not state.
    pub fn input(&self, name: &str) -> &'a [F] {
        self.get(name).unwrap_or_else(|| {
            panic!("form input `{name}` is absent: a form states every column it reads")
        })
    }

    /// The numeric style parameter `name` of a form that states it among its
    /// inputs; see [`input`](Self::input).
    ///
    /// # Panics
    ///
    /// When `name` is absent: the form read an input it did not state.
    pub fn style_input(&self, name: &str) -> F {
        self.style_num(name).unwrap_or_else(|| {
            panic!("style input `{name}` is absent: a form states every parameter it reads")
        })
    }

    /// The numeric style parameter `name`: one value for every term of the
    /// batch.
    pub fn style_num(&self, name: &str) -> Option<F> {
        self.style_nums
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| *v)
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

/// A style parameter a form refuses at build: which, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyleRefusal {
    pub param: String,
    pub reason: String,
}

impl StyleRefusal {
    pub fn new(param: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            param: param.into(),
            reason: reason.into(),
        }
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

    /// Refuse, once at build, style parameters outside the form's domain (an
    /// exponent that is no integer, a switch whose inner radius is not below
    /// its outer one). `p` holds the style parameters only
    /// ([`ParamColumns::style_num`], [`ParamColumns::style_text`]); the
    /// kernel reports a refusal as [`IrError::BadValue`] of the style.
    /// Default: nothing refused.
    ///
    /// [`IrError::BadValue`]: crate::ff::ir::IrError::BadValue
    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        let _ = p;
        Ok(())
    }

    /// The force law of one term, prepared once from the style parameters
    /// `style` ([`ParamColumns::style_num`], [`ParamColumns::style_text`]):
    /// `law(q, inputs)` is `(e, dE/dq)` of a term at `q` whose per-term
    /// inputs — the form's [`inputs`](Self::inputs) that are not style
    /// parameters, in that order — are `inputs`. It must price exactly what
    /// [`eval`](Self::eval) prices ([`eval_by_term`] is `eval` written over
    /// it).
    ///
    /// A kernel monomorphized on the form's type (a built-in's) calls it
    /// inside its own per-term loop, the force law inlined beside the gather
    /// and the scatter; a kernel over `dyn ScalarForm` cannot, and calls
    /// [`eval`](Self::eval) a batch at a time. Default: `None`, the batch
    /// path everywhere.
    fn pointwise(&self, style: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_>
    where
        Self: Sized,
    {
        let _ = style;
        None::<fn(F, &[F]) -> (F, F)>
    }

    /// The force law of one **pair**, in the squared distance: `law(r², inputs)`
    /// is `(e, −(dE/dr)/r)` — LAMMPS's `fpair`, the factor of the separation
    /// that is the force on the second atom — with the per-term inputs of
    /// [`pointwise`](Self::pointwise). A monomorphized pair kernel calls this
    /// one; a law that is a function of `r²` (a 12-6 Lennard-Jones, CHARMM's
    /// switch) overrides it and writes [`pointwise`](Self::pointwise) over it,
    /// so no `√r²` is taken. Default: [`pointwise`](Self::pointwise) at
    /// `r = √r²`.
    fn pair_law(&self, style: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_>
    where
        Self: Sized,
    {
        let law = self.pointwise(style)?;
        Some(move |r2: F, inputs: &[F]| {
            let r = r2.sqrt();
            let (e, de) = law(r, inputs);
            (e, -de / r)
        })
    }
}

/// `form`'s [`ScalarForm::eval`] written over its
/// [`pointwise`](ScalarForm::pointwise) law: each term's per-term inputs
/// gathered by name, the law applied term by term — what a form that
/// provides a law states as its batch evaluation.
///
/// # Panics
///
/// When `form` provides no law, or an input it states is absent from `p`.
pub fn eval_by_term<Fm: ScalarForm>(
    form: &Fm,
    q: &[F],
    p: &ParamColumns<'_>,
    e: &mut [F],
    de_dq: &mut [F],
) {
    let law = form
        .pointwise(p)
        .expect("a form evaluated by term provides a pointwise law");
    let names: Vec<String> = form
        .inputs()
        .into_iter()
        .filter(|name| p.style_num(name).is_none())
        .collect();
    let cols: Vec<&[F]> = names.iter().map(|name| p.input(name)).collect();
    let mut inputs = vec![0.0; cols.len()];
    for t in 0..q.len() {
        for (v, col) in inputs.iter_mut().zip(&cols) {
            *v = col[t];
        }
        (e[t], de_dq[t]) = law(q[t], &inputs);
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

    /// Refuse style parameters outside the form's domain; see
    /// [`ScalarForm::check_style`].
    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        let _ = p;
        Ok(())
    }
}
