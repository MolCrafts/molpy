//! The last resort: bond / angle constants from an empirical formula.
//!
//! When no row of the table can be reached by substituting an atom type, there is
//! nothing left to copy, and the estimator computes a force constant from the
//! element pair instead (Wang et al., *J. Comput. Chem.* 2004, 25:1157–1174 —
//! the paper GAFF itself was parameterised in, Eqs. 3 and 5). **No ab-initio /
//! QM fitting is ever performed.**
//!
//! # Units
//!
//! [`angle_k`] takes θ₀ in **radians** (the formula's own unit); lengths are Å.
//!
//! These functions return AMBER's **un-halved** `K`, because they are calibrated
//! to reproduce `gaff.dat`'s own numbers and AMBER writes a harmonic term as
//! `E = K·(x − x₀)²` — which is LAMMPS's form and therefore molrs's: a formula
//! and a table hit reach a consumer in the same convention with no factor in
//! between.

/// 143.9 prefactor in the empirical angle force-constant formula (Wang 2004,
/// Eq. 5). Units bake out to kcal/mol/rad².
const ANGLE_K_PREFACTOR: f64 = 143.9;

/// Badger's-rule bond force constant (Wang 2004, Eq. 3):
/// `K_r = exp(ln_Kij) / r^m` (kcal/mol/Å²), `r` in Å.
///
/// `m` is the table's own exponent (`PARM PC`, 4.5 upstream) and `r` the
/// tabulated reference length of the element pair — the formula is evaluated at
/// the equilibrium length, which is therefore also the estimate's `r₀`.
pub fn bond_k(ln_kij: f64, r: f64, m: f64) -> f64 {
    ln_kij.exp() / r.powf(m)
}

/// Empirical equilibrium angle (Wang 2004): the mean of the two shared-centre
/// reference angles `θ(A-B-A)` and `θ(C-B-C)`, in their unit (degrees, as
/// molrs stores them).
pub fn angle_theta0(theta_aba: f64, theta_cbc: f64) -> f64 {
    0.5 * (theta_aba + theta_cbc)
}

/// Empirical angle force constant (Wang 2004, Eq. 5, in parmchk2's source form):
///
/// ```text
/// K_θ = 143.9 · Z_i · C_j · Z_k · exp(-2·D) / (r_ij + r_jk) / sqrt(θ₀)
/// D   = (r_ij − r_jk)² / (r_ij + r_jk)²
/// ```
///
/// `θ₀` is in **radians** (the `/ sqrt(θ)` matches parmchk2's
/// `sqrt(angle·π/180)`), bond lengths in Å; result in kcal/mol/rad².
pub fn angle_k(zi: f64, cj: f64, zk: f64, r_ij: f64, r_jk: f64, theta0_rad: f64) -> f64 {
    let sum = r_ij + r_jk;
    let d = (r_ij - r_jk).powi(2) / sum.powi(2);
    ANGLE_K_PREFACTOR * zi * cj * zk * (-2.0 * d).exp() / sum / theta0_rad.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bond_k_matches_gaff_reference() {
        // ac-001: K = exp(ln_Kij)/r^4.5 reproduces gaff.dat force constants.
        // C-C: ln_Kij 7.643, r 1.5375 → ~300.9 (gaff.dat c3-c3 K=300.9).
        let cc = bond_k(7.643, 1.5375, 4.5);
        assert!((cc - 300.9).abs() / 300.9 < 1e-3, "C-C k {cc}");
        // C-H: ln_Kij 6.217, r 1.0969 → ~330.6 (gaff.dat c3-hc K=330.6).
        let ch = bond_k(6.217, 1.0969, 4.5);
        assert!((ch - 330.6).abs() / 330.6 < 1e-3, "C-H k {ch}");
    }

    #[test]
    fn angle_theta0_is_the_mean_of_the_two_reference_angles() {
        // ac-002: θ₀(A-B-C) = (θ(A-B-A) + θ(C-B-C)) / 2.
        let t = angle_theta0(1.90, 2.00);
        assert!((t - 1.95).abs() < 1e-6);
    }

    #[test]
    fn angle_k_matches_gaff_reference() {
        // ac-003: Eq.5 reproduces gaff.dat angle force constants.
        // c3-c3-c3: Z_C=1.183, C_C=1.339, Z_C=1.183, r=1.5375 (both), θ=111.51°.
        let theta = 111.51_f64.to_radians();
        let k = angle_k(1.183, 1.339, 1.183, 1.5375, 1.5375, theta);
        assert!((k - 62.9).abs() / 62.9 < 1e-3, "c3-c3-c3 K_θ {k}");
        // hc-c3-hc: Z_H=0.784, C_C=1.339, Z_H=0.784, r=1.0969 (both), θ=107.58°.
        let theta2 = 107.58_f64.to_radians();
        let k2 = angle_k(0.784, 1.339, 0.784, 1.0969, 1.0969, theta2);
        assert!((k2 - 39.4).abs() / 39.4 < 1e-3, "hc-c3-hc K_θ {k2}");
    }
}
