//! The expression engine: a style's `expression` as its kernel.
//!
//! A style that carries an `expression` (molrec's Lepton subset) is priceable
//! with nothing registered: [`compile`] it against the category's
//! [`Binding`] and evaluate the [`Compiled`] program over a batch of terms,
//! with exact derivatives from forward-mode dual numbers — nobody hand-writes
//! dE/dq.
//!
//! # Grammar
//!
//! Exactly molrec `docs/spec/forcefield.md` § Expressions, the protocol's
//! §4 and its decision D7: numbers, names, `+ - * / ^`, unary minus,
//! parentheses, the functions
//! `exp log sqrt sin cos tan asin acos atan abs min max step delta select`,
//! and `;`-separated sub-definitions `energy; a=…; b=…`. `^` is
//! right-associative and binds tighter than unary minus, which binds
//! tighter than `*` and `/` (`-a^2` is −(a²), `a^-2` is a⁻²). As in Lepton,
//! a definition uses only the definitions to its right; no named constants.
//! The compound functions `distance(pa,pb)`, `angle(pa,pb,pc)`,
//! `dihedral(pa,pb,pc,pd)` take points (see [`parse`](parse::parse) for the
//! lexer's details).
//!
//! # Variables ([`Geometry`])
//!
//! | category | geometric variables | points |
//! |---|---|---|
//! | bond, drude | `r` | `p1`, `p2` |
//! | angle | `theta` (radians, [0, π]) | `p1`…`p3` |
//! | dihedral | `phi` (signed, radians, IUPAC: molrs `compute_dihedral`) | `p1`…`p4` |
//! | improper | `phi` (signed) and `chi = abs(phi)` (D4) | `p1`…`p4` |
//! | pair | `r`, `q1`, `q2` | none |
//! | cmap, a custom category (2–5 atoms) | none | `p1`…`pN` |
//!
//! plus the style's numeric per-type and style-level parameter names
//! ([`Binding`]) and the sub-definitions. A geometric variable is shorthand
//! for its compound function, and the compound functions are available in
//! every category with points (D5): `angle charmm` written out is
//! `k*(theta-theta0*(pi/180))^2+k_ub*(distance(p1,p3)-r_ub)^2; pi=3.141592653589793`.
//!
//! For a pair (D6), a bare parameter name binds the pair value (the cross
//! row, else the mixing rule), `x1`/`x2` the self rows of the first/second
//! atom's type, `q1`/`q2` the charges; the expression is the energy before
//! the special weight and the cutoff, and must be symmetric under 1 ↔ 2
//! ([`Input::swapped`] is the exchange).
//!
//! # Evaluation
//!
//! A [`Compiled`] expression holds up to two programs: a **scalar** one
//! (N = 1 dual on the coordinate: a `ScalarForm`) when the energy is a
//! function of the coordinate alone, and a **compound** one (N = 3·arity
//! duals on the points: a `CompoundForm`) whenever the category has points.
//! Non-smooth functions differentiate as the protocol says: `step`, `delta`
//! 0; `abs` sign(x), 0 at 0; `min`/`max` the chosen argument (ties: the
//! first); `select` the chosen branch; a degenerate geometry (coincident or
//! collinear points) its value as computed, gradient 0.
//!
//! # Angles: parameters bound as stored, in degrees (D3)
//!
//! Coordinates are radians, as Lepton's trigonometry is; an angle-valued
//! parameter (`theta0`, `phase`, `chi0`, …) is bound **as stored in the IR,
//! in degrees**, and the expression converts it explicitly with `(pi/180)`.
//! The grammar has no named constants (Lepton has none), so `pi` is the
//! expression's last sub-definition, `pi=3.141592653589793`, as Lepton users
//! write it: `angle harmonic` is
//! `k*(theta-theta0*(pi/180))^2; pi=3.141592653589793`, never
//! `k*(theta-theta0)^2`. Parenthesised, `(pi/180)` is the double nearest
//! π/180, the factor of `f64::to_radians`, so `theta0*(pi/180)` is
//! `theta0.to_radians()` to the bit (`theta0*pi/180` rounds differently).
//! No parameter is converted behind the expression's back, so the
//! expression a reader keeps is the whole truth.
//!
//! # Source and printer
//!
//! [`Compiled::source`] is the string as given, byte for byte. The printer
//! ([`Parsed`]'s `Display`) exists for engine rewriting only
//! ([`Parsed::substitute`], e.g. `r → 10*r`) and parses back to the same
//! tree.

mod ast;
mod compile;
mod dual;
mod error;
mod eval;
mod parse;
mod print;

#[cfg(test)]
mod tests;

pub use ast::{BinOp, Definition, Expr, Func, Parsed};
pub use compile::{Binding, Compiled, Geometry, Input, compile, compile_parsed, is_identifier};
pub use error::ExpressionError;
pub use parse::parse;

/// Compile a style's `expression` for a category molrec's variable table
/// names (`bond`, `drude`, `angle`, `dihedral`, `improper`, `pair`,
/// `cmap`), with its numeric per-type parameters in declared order and its
/// numeric style-level parameters.
///
/// This is the compile fallback's entry point: a style no kernel is
/// registered for, carrying an expression, prices through the result. A
/// custom category binds [`Geometry::Compound`] with its arity through
/// [`compile`].
pub fn compile_style(
    category: &str,
    params: &[&str],
    style_params: &[&str],
    expression: &str,
) -> Result<Compiled, ExpressionError> {
    let geometry = Geometry::of_category(category).ok_or_else(|| ExpressionError::BadBinding {
        reason: format!(
            "category `{category}` is no category of molrec's variable table; \
             a custom one binds Geometry::Compound with its arity"
        ),
    })?;
    compile(expression, &Binding::new(geometry, params, style_params))
}
