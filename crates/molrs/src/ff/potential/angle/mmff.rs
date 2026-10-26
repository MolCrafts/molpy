//! MMFF94 angle bending and stretch-bend coupling.

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;
use crate::ff::potential::flat_coords::{accumulate_angle_forces, compute_angle, sub3};
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns, ScalarForm};
use crate::op::vec3::norm;
use molrs::op::F;

/// Cubic bend constant `cb` (rad⁻¹) — **exactly -0.4**.
///
/// MMFF94 (Halgren 1996, eq. 3) defines the anharmonic bend in *degrees*:
///
/// ```text
/// EA = 0.043844 * (ka/2) * dth_deg^2 * (1 + cb_deg * dth_deg),  cb_deg = -0.006981317 deg^-1
/// ```
///
/// molrs works in radians, so the cubic coefficient converts once:
///
/// ```text
/// cb_rad = cb_deg * 180/pi = -(pi/450) * 180/pi = -180/450 = -0.4   (exact, by construction)
/// ```
///
/// This constant was `-0.40107`, commented `= -0.007 * 180/pi`: the degree
/// constant was **rounded to -0.007 before** being converted. The resulting 0.27%
/// error in the anharmonic term is worth 0.0073 kcal/mol on the e_big fixture's
/// angle energy alone — 7x the 1e-3 parity tolerance, i.e. enough to fail parity
/// on its own even with electrostatics and vdW correct.
const CB_RAD: f64 = -0.4;

/// MMFF angle bending, with the linear-centre branch:
///
/// ```text
/// cubic:  E = ½ · 143.9325 · ka · dθ² · (1 + cb · dθ)
/// linear: E = 143.9325 · ka · (1 + cos θ)
/// ```
///
/// `ka` (md·Å·rad⁻²), `theta0` (**degrees**, as MMFF's own tables and every
/// molrs angle parameter are) and `linear` (0/1) are per-instance columns
/// the MMFF typifier bakes onto each angle.
///
/// # Linear centres
///
/// At a linear centre (acetylene, nitrile, allene — MMFF atom-property
/// `linh != 0`) the cubic bend is not merely inaccurate, it is the wrong
/// functional form: it is a Taylor expansion about `theta0`, and there is no
/// `theta0` to expand about when the equilibrium is 180 degrees and the angle is
/// free to bend either way. MMFF switches to a cosine form there and skips the
/// stretch-bend coupling entirely. The typifier bakes a `linear` flag on each
/// angle row (from the *central* atom's `linh`), an integer column —
/// `MolGraph::to_frame` materializes only f64 / i32 / string properties, so a
/// boolean would be dropped on the way to the frame and every nitrile would
/// silently bend cubically; its absence is refused (`MissingParam`) like any
/// other parameter's. This form and [`MmffStretchBend`] both read it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mmff;

impl ScalarForm for Mmff {
    fn eval(&self, theta: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (ka, theta0, linear) = (p.input("ka"), p.input("theta0"), p.input("linear"));
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;
        let cb = CB_RAD as F;
        for t in 0..theta.len() {
            if linear[t] != 0.0 {
                // E = 143.9325 * ka * (1 + cos t)  =>  dE/dt = -143.9325 * ka * sin t.
                // Minimised at t = 180 deg and smooth through it — unlike the
                // cubic form, which has a cusp there.
                e[t] = conv * ka[t] * (1.0 + theta[t].cos());
                de[t] = -conv * ka[t] * theta[t].sin();
            } else {
                let dth = theta[t] - theta0[t].to_radians();
                e[t] = 0.5 * conv * ka[t] * dth * dth * (1.0 + cb * dth);
                de[t] = conv * ka[t] * dth * (1.0 + 1.5 * cb * dth);
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["ka", "theta0", "linear"].map(String::from).to_vec()
    }
}

/// MMFF stretch-bend coupling:
/// `E = 143.9325 · (kba_ijk · dr_ij + kba_kji · dr_kj) · dθ`, an N-body form of
/// the angle's three positions (both bond lengths and the angle).
///
/// `kba_ijk`, `kba_kji` (md/rad), `r0_ij`, `r0_kj` (Å), `theta0` (degrees)
/// and `linear` are per-instance columns.
///
/// # Linear centres carry no stretch-bend
///
/// MMFF defines no stretch-bend coupling at a linear centre (`linear = 1`): with
/// the bend energy a cosine of theta rather than a quadratic in `dtheta`, the
/// `(dr) * (dtheta)` cross term has no reference angle to be measured from. Those
/// rows price **nothing**, exactly as the reference implementation skips them.
///
/// Only the linear centre is skipped, not the molecule: acetonitrile's three
/// methyl angles (H-C-H, H-C-C) contribute stretch-bend like any sp3 centre
/// (-0.010941 kcal/mol in total). "Nitriles have zero stretch-bend" is wrong.
#[derive(Clone, Copy, Debug, Default)]
pub struct MmffStretchBend;

impl CompoundForm for MmffStretchBend {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        debug_assert_eq!(arity, 3);
        let (kba_ijk, kba_kji) = (p.input("kba_ijk"), p.input("kba_kji"));
        let (r0_ij, r0_kj) = (p.input("r0_ij"), p.input("r0_kj"));
        let (theta0, linear) = (p.input("theta0"), p.input("linear"));
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;
        for t in 0..e.len() {
            let g = &mut grad[t * 3..t * 3 + 3];
            if linear[t] != 0.0 {
                e[t] = 0.0;
                g.fill([0.0; 3]);
                continue;
            }
            let flat: [F; 9] = std::array::from_fn(|c| x[t * 3 + c / 3][c % 3]);
            let mut f = [0.0; 9];
            let rij_vec = sub3(&flat, 0, &flat, 1);
            let rkj_vec = sub3(&flat, 2, &flat, 1);
            let rij = norm(rij_vec);
            let rkj = norm(rkj_vec);
            let dr_ij = rij - r0_ij[t];
            let dr_kj = rkj - r0_kj[t];
            let dth = compute_angle(&flat, 0, 1, 2) - theta0[t].to_radians();

            let term = kba_ijk[t] * dr_ij + kba_kji[t] * dr_kj;
            e[t] = conv * term * dth;

            // dE/ddth = conv * term
            accumulate_angle_forces(&flat, 0, 1, 2, conv * term, &mut f);
            // dE/dr_ij = conv * kba_ijk * dth
            if rij > 1e-12 as F {
                let f_r = -conv * kba_ijk[t] * dth / rij;
                for dim in 0..3 {
                    f[dim] += f_r * rij_vec[dim];
                    f[3 + dim] -= f_r * rij_vec[dim];
                }
            }
            // dE/dr_kj = conv * kba_kji * dth
            if rkj > 1e-12 as F {
                let f_r = -conv * kba_kji[t] * dth / rkj;
                for dim in 0..3 {
                    f[6 + dim] += f_r * rkj_vec[dim];
                    f[3 + dim] -= f_r * rkj_vec[dim];
                }
            }
            for (a, ga) in g.iter_mut().enumerate() {
                *ga = [-f[a * 3], -f[a * 3 + 1], -f[a * 3 + 2]];
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["kba_ijk", "kba_kji", "r0_ij", "r0_kj", "theta0", "linear"]
            .map(String::from)
            .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;

    /// ac-008 — the cubic bend constant is **-0.4 rad^-1, exactly**.
    ///
    /// MMFF94 (Halgren 1996) defines the anharmonic bend in DEGREES:
    ///
    /// ```text
    /// EA = 0.043844 * (ka/2) * dth_deg^2 * (1 + cb_deg * dth_deg),  cb_deg = -0.006981317 deg^-1
    /// ```
    ///
    /// Converting the cubic coefficient to radians is one multiplication:
    ///
    /// ```text
    /// cb_rad = cb_deg * 180/pi = -0.006981317 * 57.29577951... = -0.4 exactly
    /// ```
    ///
    /// (`0.006981317 = pi/450`, so the product is `-180/450 = -0.4` — it is exact
    /// by construction, not by luck.)
    ///
    /// The original constant was `-0.40107`, commented `// = -0.007 * 180/pi`:
    /// the author ROUNDED `-0.006981317` to `-0.007` and only then converted. The
    /// resulting 0.27% error in the anharmonic term costs e_big 0.0073 kcal/mol
    /// in the angle energy alone — 7x the 1e-3 parity tolerance, i.e. e_big fails
    /// on this defect even after electrostatics and vdW are fixed.
    #[test]
    fn cubic_bend_constant_is_exactly_minus_zero_point_four() {
        assert_eq!(
            CB_RAD, -0.4,
            "CB_RAD is {CB_RAD}; MMFF's cb = -0.006981317 deg^-1 = -pi/450 deg^-1, and \
             -pi/450 * 180/pi = -0.4 rad^-1 EXACTLY. -0.40107 is what you get by rounding \
             the degree constant to -0.007 BEFORE converting."
        );

        // The derivation, executed: convert the degree constant and land on -0.4.
        let cb_deg: f64 = -0.006981317;
        let cb_rad = cb_deg.to_degrees();
        assert!(
            (cb_rad - CB_RAD).abs() < 1e-7,
            "CB_RAD ({CB_RAD}) is not cb_deg * 180/pi ({cb_rad})"
        );

        // And the rounded-first path is measurably NOT the same number, which is
        // the whole bug: 0.27% of the cubic term.
        let rounded_first = (-0.007_f64).to_degrees();
        assert!(
            (rounded_first - CB_RAD).abs() > 1e-3,
            "rounding cb to -0.007 before converting gives {rounded_first}, which must not \
             coincide with the exact -0.4"
        );
    }

    #[test]
    fn test_mmff_angle_at_equilibrium() {
        let theta0: F = (109.5 as F).to_radians();
        let pot = ExplicitTerms::new("angle", "mmff_angle")
            .term(
                &[0, 1, 2],
                Params::from_iter([("ka", 0.608), ("theta0", 109.5), ("linear", 0.0)]),
            )
            .compile()
            .unwrap();
        let r = 1.5 as F;
        let half = theta0 / 2.0;
        let coords: Vec<F> = vec![
            r * half.cos(),
            r * half.sin(),
            0.0,
            0.0,
            0.0,
            0.0,
            r * half.cos(),
            -r * half.sin(),
            0.0,
        ];
        let (e, _) = pot.energy_forces_at(&coords);
        assert!(e.abs() < 1e-4, "angle energy at eq should be ~0, got {}", e);
    }
}
