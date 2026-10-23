//! The last resort: an angle constant from an empirical formula.
//!
//! When no row of the table can be reached by substituting an atom type, there is
//! nothing left to copy, and GAFF's analogue search computes the force constant
//! from the atoms' elements instead (Wang et al., *J. Comput. Chem.* 2004, 25:1157–1174 —
//! the paper GAFF itself was parameterised in, Eq. 5). **No ab-initio /
//! QM fitting is ever performed.**
//!
//! # Units
//!
//! [`angle_k`] takes θ₀ in **radians** (the formula's own unit); lengths are Å.
//!
//! It returns AMBER's **un-halved** `K`, because it is calibrated
//! to reproduce `gaff.dat`'s own numbers and AMBER writes a harmonic term as
//! `E = K·(x − x₀)²` — which is LAMMPS's form and therefore molrs's: a formula
//! and a table hit reach a consumer in the same convention with no factor in
//! between.

/// 143.9 prefactor in the empirical angle force-constant formula (Wang 2004,
/// Eq. 5). Units bake out to kcal/mol/rad².
const ANGLE_K_PREFACTOR: f64 = 143.9;

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
