//! The Lennard-Jones rows of a force field read outside a kernel: what the
//! 1-4 exceptions, `materialize_one_four` and the GROMACS writer price a
//! close pair with, by the same rules the pair styles' form kernel prices it
//! (an explicit cross row, else the two self rows by the style's `mixing`).

use std::collections::HashMap;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::IrError;
use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::{CompileError, param_reads};
use molrs::op::F;

/// `(ε, σ)` of the row keyed *key* of the Lennard-Jones style `style`, or
/// `None` when there is none.
fn lj_row(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    key: &str,
) -> Result<Option<(F, F)>, IrError> {
    let Some(p) = type_map.get(key) else {
        return Ok(None);
    };
    let eps = param_reads::type_num(style, key, p, "epsilon")?;
    let sigma = param_reads::type_num(style, key, p, "sigma")?;
    Ok(Some((eps, sigma)))
}

/// `(ε, σ)` of the atom-type pair `(a, b)` under the Lennard-Jones style
/// `style` (`lj/cut`, `lj/class2`): its explicit cross row — keyed by
/// [`pair_key`], as `kernel_type_params` keys it, so it answers in either
/// order — else the two self rows mixed by `mixing`.
pub(crate) fn lj_pair_params(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    mixing: CombiningRule,
    a: &str,
    b: &str,
) -> Result<(F, F), CompileError> {
    if a != b
        && let Some(row) = lj_row(style, type_map, &pair_key(a, b)?)?
    {
        return Ok(row);
    }
    let own = |t: &str| -> Result<(F, F), CompileError> {
        lj_row(style, type_map, t)?
            .ok_or_else(|| format!("{style}: unknown atom type '{t}'").into())
    };
    Ok(mixing.combine(own(a)?, own(b)?))
}

/// The `mixing` rule of the Lennard-Jones style `style`, from its params
/// gathered under its spec ([`StyleSpec::gather`]: declared, else the
/// spec's default).
///
/// [`StyleSpec::gather`]: crate::ff::ir::StyleSpec::gather
pub(crate) fn mixing_of(style: &str, style_params: &Params) -> Result<CombiningRule, IrError> {
    let name = param_reads::style_text(style, style_params, "mixing")?;
    CombiningRule::parse(name).map_err(|e| param_reads::bad(style, "", "mixing", e))
}

/// `((ε, σ), (ε₁₄, σ₁₄))`: a pair's regular and 1-4 Lennard-Jones parameters.
pub(crate) type CharmmParams = ((F, F), (F, F));

/// `((ε, σ), (ε₁₄, σ₁₄))` of one `lj/charmm` row.
fn charmm_row(p: &Params, key: &str) -> Result<CharmmParams, IrError> {
    let need = |k: &str| param_reads::type_num("lj/charmm", key, p, k);
    let (eps, sigma) = (need("epsilon")?, need("sigma")?);
    Ok((
        (eps, sigma),
        crate::ff::ir::lj_charmm_one_four(p, eps, sigma),
    ))
}

/// The regular and 1-4 `(ε, σ)` of the atom-type pair `(a, b)` under an
/// `lj/charmm` style: its explicit cross row when it has one (a two-number
/// cross row is its own 1-4 row), the two self rows mixed by `mixing`
/// otherwise — LAMMPS's `init_one`.
pub(crate) fn charmm_pair_params(
    rows: &HashMap<&str, &Params>,
    mixing: CombiningRule,
    a: &str,
    b: &str,
) -> Result<CharmmParams, CompileError> {
    if a != b {
        let key = pair_key(a, b)?;
        if let Some(p) = rows.get(key.as_str()) {
            return Ok(charmm_row(p, &key)?);
        }
    }
    let own = |t: &str| -> Result<CharmmParams, CompileError> {
        let p = rows
            .get(t)
            .ok_or_else(|| format!("lj/charmm: unknown atom type '{t}'"))?;
        Ok(charmm_row(p, t)?)
    };
    let (ra, r14a) = own(a)?;
    let (rb, r14b) = own(b)?;
    Ok((mixing.combine(ra, rb), mixing.combine(r14a, r14b)))
}

/// Lennard-Jones `A`/`B` coefficients to `(σ, ε)`.
///
/// The same 12-6 interaction is written two ways: Amber files carry
/// `E = A/r¹² − B/r⁶`, and `pair lj/cut` takes `E = 4ε[(σ/r)¹² − (σ/r)⁶]`.
///
/// `r_min = (2A/B)^{1/6}`, `ε = B²/(4A)`, `σ = 2^{−1/6} r_min`. A row with
/// either coefficient zero is a non-interacting type — Amber writes those for
/// types that carry no vdW term — and yields `(1.0, 0.0)`: ε of zero switches
/// the term off, and σ is given a benign non-zero value so that no consumer
/// divides by it.
///
/// ```
/// # use molrs::ff::potential::pair::lj_ab_to_sigma_epsilon;
/// // GAFF c3: R* = 1.9080 Å, ε = 0.1094 kcal/mol.
/// let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
/// assert!((epsilon - 0.1094).abs() < 1e-6);
/// assert!((sigma - 3.399669).abs() < 1e-5);
/// assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
/// ```
#[must_use]
pub fn lj_ab_to_sigma_epsilon(a: F, b: F) -> (F, F) {
    if a == 0.0 || b == 0.0 {
        return (1.0, 0.0);
    }
    let r_min = (2.0 * a / b).powf(1.0 / 6.0);
    let epsilon = 0.25 * b * b / a;
    let sigma = (2.0 as F).powf(-1.0 / 6.0) * r_min;
    (sigma, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epsilon14_mixes_like_epsilon_and_a_cross_row_wins() {
        let a = Params::from_iter([
            ("epsilon", 0.1),
            ("sigma", 3.0),
            ("epsilon14", 0.05),
            ("sigma14", 2.8),
        ]);
        let b = Params::from_iter([("epsilon", 0.4), ("sigma", 3.6)]);
        let ab = Params::from_iter([("epsilon", 0.9), ("sigma", 2.0)]);
        let key = pair_key("A", "B").unwrap();
        let mut rows: HashMap<&str, &Params> = HashMap::new();
        rows.insert("A", &a);
        rows.insert("B", &b);
        let (reg, r14) = charmm_pair_params(&rows, CombiningRule::Arithmetic, "A", "B").unwrap();
        assert_eq!(reg, ((0.1_f64 * 0.4).sqrt(), 3.3));
        assert_eq!(r14, ((0.05_f64 * 0.4).sqrt(), 0.5 * (2.8 + 3.6)));
        rows.insert(key.as_str(), &ab);
        let (reg, r14) = charmm_pair_params(&rows, CombiningRule::Arithmetic, "B", "A").unwrap();
        assert_eq!(reg, (0.9, 2.0));
        assert_eq!(r14, (0.9, 2.0), "a 2-number cross row is its own 1-4 row");
    }

    #[test]
    fn zero_coefficient_is_a_non_interacting_type() {
        assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
        assert_eq!(lj_ab_to_sigma_epsilon(1.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn round_trips_the_gaff_c3_row() {
        let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
        assert!((epsilon - 0.1094).abs() < 1e-6, "{epsilon}");
        // sigma = 2^(-1/6) * R* * 2, with R* the Amber half-distance 1.9080 Å.
        assert!((sigma - 3.399669).abs() < 1e-5, "{sigma}");
    }

    #[test]
    fn inverts_the_molrs_form() {
        // Build A/B from a known (sigma, epsilon) and recover it.
        let (sigma, epsilon) = (3.4_f64, 0.238_f64);
        let a = 4.0 * epsilon * sigma.powi(12);
        let b = 4.0 * epsilon * sigma.powi(6);
        let (got_sigma, got_epsilon) = lj_ab_to_sigma_epsilon(a, b);
        assert!((got_sigma - sigma).abs() < 1e-12, "{got_sigma}");
        assert!((got_epsilon - epsilon).abs() < 1e-12, "{got_epsilon}");
    }
}
