//! MMFF94's Van der Waals form — the buffered 14-7 potential.
//!
//! The style is **per-type**: its 95 rows (one per MMFF atom type) carry the
//! polarizabilities the combining rules need, and the style-level block
//! (`<VdWParams B= Beta= DARAD= DAEPS=>`) shapes those rules.
//!
//! # MMFF's electrostatics is not here, and owns no kernel
//!
//! It is a **buffered Coulomb** — `E = k·qᵢqⱼ / (D·(r + δ))` — which is the *generic*
//! [`CoulCut`](super::coul_cut::CoulCut) form at `k = 332.0716`, `D = 1.0`,
//! `δ = 0.05 Å`: a parameterization, not a kernel of its own. The numbers live in
//! [`MMFF_ELE_STYLE`](crate::ff::params::mmff::MMFF_ELE_STYLE) and reach the kernel
//! through the style. vdW is the opposite case — a genuine per-type table — which is
//! why it stays.
//!
//! # 1-4 scaling (house convention)
//!
//! MMFF scales the 1-4 **electrostatic** interaction by 0.75 and does **not**
//! scale the 1-4 **van der Waals** interaction at all (Halgren 1996; RDKit
//! `Nonbonded.cpp`). Both weights arrive through [`SpecialBonds`], which
//! [`PotentialCompiler::compile`] projects into the pair params as `coulomb14scale` /
//! `lj14scale` — so neither form hardcodes a scale factor. MMFF's `lj14scale`
//! is **1.0** (set by the reader), and applying it is a deliberate no-op: its
//! torsion parameters were fitted against *unscaled* 1-4 vdW. Do not "fix"
//! this to 0.5 by analogy with Amber.
//!
//! [`SpecialBonds`]: crate::ff::ir::SpecialBonds
//! [`PotentialCompiler::compile`]: crate::ff::compile::PotentialCompiler::compile

use crate::ff::params::mmff::{DA_ACCEPTOR, DA_DONOR};
use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// Buffered 14-7 van der Waals (MMFF.I eq. 8):
///
/// ```text
/// E = ε_ij (1.07 R*_ij / (r + 0.07 R*_ij))⁷ (1.12 R*_ij⁷ / (r⁷ + 0.12 R*_ij⁷) − 2)
/// ```
///
/// The style is **per-type**: its 95 rows (one per MMFF atom type) carry the
/// polarizability `alpha` (Å³), the Slater-Kirkwood effective electron number
/// `n_eff`, the scale factors `a_i` / `g_i` and the hydrogen-bond role `da`,
/// and the pair's `R*_ij` and `ε_ij` are MMFF's combining rules over both
/// atoms' rows (`alpha1`, `alpha2`, …), shaped by the style's `B`, `Beta`,
/// `DARAD`, `DAEPS` — no mixing rule and no cross row.
#[derive(Clone, Copy, Debug, Default)]
pub struct MmffVdw;

/// One atom's vdW row.
struct Atom {
    alpha: F,
    n_eff: F,
    a_i: F,
    g_i: F,
    /// [`DA_NEITHER`](crate::ff::params::mmff::DA_NEITHER), [`DA_DONOR`] or
    /// [`DA_ACCEPTOR`].
    da: F,
}

/// The style's global vdW line (MMFF94.PAR's `<VdWParams>`).
struct Shape {
    b: F,
    beta: F,
    darad: F,
    daeps: F,
}

impl ScalarForm for MmffVdw {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let shape = Shape {
            b: p.style_input("B"),
            beta: p.style_input("Beta"),
            darad: p.style_input("DARAD"),
            daeps: p.style_input("DAEPS"),
        };
        Some(move |r: F, x: &[F]| {
            let atom = |k: usize| Atom {
                alpha: x[k],
                n_eff: x[2 + k],
                a_i: x[4 + k],
                g_i: x[6 + k],
                da: x[8 + k],
            };
            let (rs, eps) = vdw_combining(&atom(0), &atom(1), &shape);
            buffered_14_7(r, rs, eps)
        })
    }

    fn inputs(&self) -> Vec<String> {
        let mut out: Vec<String> = ["alpha", "n_eff", "a_i", "g_i", "da"]
            .iter()
            .flat_map(|p| [format!("{p}1"), format!("{p}2")])
            .collect();
        out.extend(["B", "Beta", "DARAD", "DAEPS"].map(String::from));
        out
    }
}

/// `(E, dE/dr)` of the buffered 14-7 at `r` for the combined `R*` and `ε`.
fn buffered_14_7(r: F, rs: F, eps: F) -> (F, F) {
    let rho = r + 0.07 * rs;
    let u = 1.07 * rs / rho;
    let u6 = u * u * u * u * u * u;
    let u7 = u6 * u;
    let r7 = r * r * r * r * r * r * r;
    let rs7 = rs * rs * rs * rs * rs * rs * rs;
    let v = 1.12 * rs7 / (r7 + 0.12 * rs7);
    let energy = eps * u7 * (v - 2.0);
    let du_dr = -u / rho;
    let dv_dr = -7.0 * r * r * r * r * r * r * v / (r7 + 0.12 * rs7);
    // `7·u⁷/u` rather than `7·u⁶` was `0/0` for a zero vdW radius — a dummy
    // site, a lone pair, an unparameterised type — and one NaN force
    // poisons the whole trajectory in a single step. Same value, exactly,
    // one multiplication cheaper, and defined everywhere.
    let de_dr = eps * (7.0 * u6 * du_dr * (v - 2.0) + u7 * dv_dr);
    (energy, de_dr)
}

/// MMFF's van der Waals combining rules (Halgren, *MMFF.II*, J. Comput. Chem.
/// **17**, 520–552 (1996), eqs. 5–7), returning `(R*_ij [Å], eps_ij [kcal/mol])`.
///
/// ```text
/// R*_ii = A_i * alpha_i^(1/4)
/// gamma = (R*_ii - R*_jj) / (R*_ii + R*_jj)
/// R*_ij = 0.5 * (R*_ii + R*_jj) * (1 + B * (1 - exp(-Beta * gamma^2)))
/// eps_ij = 181.16 * G_i * G_j * alpha_i * alpha_j
///          / ((sqrt(alpha_i/N_i) + sqrt(alpha_j/N_j)) * R*_ij^6)
/// ```
///
/// # The two donor/acceptor rules
///
/// Both are part of the *definition* of MMFF's vdW term, not an optional
/// refinement, and omitting them costs (measured, vs RDKit): urea +0.588, e_big
/// +1.261, N-methylacetamide +0.116 kcal/mol.
///
/// 1. **Donor R\* suppression** — if either partner is a donor, the `B` shape
///    term is dropped (`R*_ij` is the plain arithmetic mean). A polar hydrogen
///    approaches its acceptor far closer than the empirical expansion allows.
/// 2. **Donor-acceptor scaling** — for a donor *paired with* an acceptor,
///    `R*_ij *= DARAD` (0.8) and `eps_ij *= DAEPS` (0.5): the hydrogen bond is
///    shorter and softer than dispersion alone would give.
///
/// `eps_ij` is evaluated at the **unscaled** `R*_ij`, then scaled — the order is
/// load-bearing (`R*^-6` would otherwise pick up a spurious `DARAD^-6` = 3.8x).
fn vdw_combining(pi: &Atom, pj: &Atom, sp: &Shape) -> (F, F) {
    let (donor, acceptor) = (F::from(DA_DONOR), F::from(DA_ACCEPTOR));
    let rs_i = pi.a_i * pi.alpha.powf(0.25);
    let rs_j = pj.a_i * pj.alpha.powf(0.25);
    let gamma = (rs_i - rs_j) / (rs_i + rs_j);

    // Rule 1: a donor suppresses the B shape term entirely.
    let is_donor = pi.da == donor || pj.da == donor;
    let shape = if is_donor {
        0.0
    } else {
        sp.b * (1.0 - (-sp.beta * gamma * gamma).exp())
    };
    let mut r_star = 0.5 * (rs_i + rs_j) * (1.0 + shape);

    let denom = (pi.alpha / pi.n_eff).sqrt() + (pj.alpha / pj.n_eff).sqrt();
    let mut epsilon = 181.16 * pi.g_i * pj.g_i * pi.alpha * pj.alpha / (denom * r_star.powi(6));

    // Rule 2: donor-acceptor pairs get a shorter, softer minimum.
    let da_pair = (pi.da == donor && pj.da == acceptor) || (pi.da == acceptor && pj.da == donor);
    if da_pair {
        r_star *= sp.darad;
        epsilon *= sp.daeps;
    }
    (r_star, epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::params::mmff::DA_NEITHER;

    /// MMFF94's global vdW line (`<VdWParams B="0.2" Beta="12.0" DARAD="0.8"
    /// DAEPS="0.5">`).
    const MMFF94: Shape = Shape {
        b: 0.2,
        beta: 12.0,
        darad: 0.8,
        daeps: 0.5,
    };

    #[test]
    fn test_mmff_vdw_combining() {
        let p = Atom {
            alpha: 1.050,
            n_eff: 2.490,
            a_i: 3.890,
            g_i: 1.282,
            da: F::from(DA_NEITHER),
        };
        let (rs, eps) = vdw_combining(&p, &p, &MMFF94);
        assert!(rs > 0.0, "r_star should be positive");
        assert!(eps > 0.0, "epsilon should be positive");
    }

    /// The buffered 14-7 of a pair whose combined `R*` is 1.94 Å and `ε`
    /// 0.02 kcal/mol, at 3 Å: finite, Newton's third law.
    #[test]
    fn test_mmff_vdw_energy() {
        let (e, de) = buffered_14_7(3.0, 1.94, 0.02);
        assert!(e.is_finite() && de.is_finite());
        // E = ε (1.07 R*/(r + 0.07 R*))⁷ (1.12 R*⁷/(r⁷ + 0.12 R*⁷) − 2)
        let (r, rs, eps) = (3.0_f64, 1.94_f64, 0.02_f64);
        let want = eps
            * (1.07 * rs / (r + 0.07 * rs)).powi(7)
            * (1.12 * rs.powi(7) / (r.powi(7) + 0.12 * rs.powi(7)) - 2.0);
        assert!((e - want).abs() < 1e-14, "{e} vs {want}");
        let h = 1e-6;
        let fd = (buffered_14_7(r + h, rs, eps).0 - buffered_14_7(r - h, rs, eps).0) / (2.0 * h);
        assert!((de - fd).abs() < 1e-9, "{de} vs {fd}");
    }
}
