//! MMFF94 angle bending and stretch-bend coupling kernels.

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::ANGLES;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{
    accumulate_angle_forces, compute_angle, sub3, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::norm;
use molrs::core::Frame;
use molrs::op::F;

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;

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

// ---------------------------------------------------------------------------
// AngleMmff
//   cubic:  E = (1/2)*143.9325*ka*dth^2*(1 + cb*dth)
//   linear: E = 143.9325*ka*(1 + cos(theta))
// ---------------------------------------------------------------------------

/// MMFF angle bending, with the linear-centre branch.
///
/// `ka` is in md·Å·rad⁻² and `theta0` in radians (143.9325 converts md·Å →
/// kcal·mol⁻¹). `linear[idx]` selects the cosine form for angles whose *central*
/// atom is a linear centre.
///
/// `theta0` is a per-instance column in **degrees**, as MMFF's own tables and
/// every molrs angle parameter are; the kernels convert it to radians once.
///
/// # Linear centres
///
/// At a linear centre (acetylene, nitrile, allene — MMFF atom-property
/// `linh != 0`) the cubic bend is not merely inaccurate, it is the wrong
/// functional form: it is a Taylor expansion about `theta0`, and there is no
/// `theta0` to expand about when the equilibrium is 180 degrees and the angle is
/// free to bend either way. MMFF switches to a cosine form there and skips the
/// stretch-bend coupling entirely. The typifier bakes a `linear` flag on each
/// angle row (from the *central* atom's `linh`), and this kernel and
/// [`AngleMmffStretchBend`] both read it.
pub struct AngleMmff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    ka: Vec<F>,
    theta0: Vec<F>,
    linear: Vec<bool>,
}

impl AngleMmff {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;
        let cb = CB_RAD as F;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let theta = compute_angle(coords, i, j, k);
            let ka = self.ka[idx];

            // dE/dtheta (kcal·mol⁻¹·rad⁻¹); `accumulate_angle_forces` projects it
            // onto the three atoms and negates it into forces.
            let de_dth = if self.linear[idx] {
                // E = 143.9325 * ka * (1 + cos t)  =>  dE/dt = -143.9325 * ka * sin t.
                // Minimised at t = 180 deg and smooth through it — unlike the cubic
                // form, which has a cusp there.
                energy += conv * ka * (1.0 + theta.cos());
                -conv * ka * theta.sin()
            } else {
                let dth = theta - self.theta0[idx];
                energy += 0.5 * conv * ka * dth * dth * (1.0 + cb * dth);
                conv * ka * dth * (1.0 + 1.5 * cb * dth)
            };
            accumulate_angle_forces(coords, i, j, k, de_dth, forces);
        }
        energy
    }
}

impl Potential for AngleMmff {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t], self.atom_k[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for AngleMmff {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k])
    }
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Build the MMFF angle-bend potential from the typifier's baked columns.
///
/// Reads `ka` (md·Å·rad⁻²), `theta0` (degrees) and `linear` (0/1) off the
/// `"angles"` block. All three are per-instance: MMFF resolves them through
/// table → equivalence → empirical rules that this kernel deliberately does not
/// re-implement.
pub fn angle_mmff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    // Per-instance parameters: the MMFF typifier baked ka and theta0 (degrees)
    // onto each angle (table → equivalence → empirical). This kernel only reads
    // the columns and evaluates — no force-field-specific resolution lives here.
    let block = frame
        .get(ANGLES)
        .ok_or("mmff_angle: missing \"angles\" block")?;
    let ic = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomi")?;
    let jc = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomj")?;
    let kc = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomk")?;
    let kac = param_reads::instance_col("mmff_angle", block, "ka")?;
    let th0c = param_reads::instance_col("mmff_angle", block, "theta0")?;
    let linc = linear_column(block, "mmff_angle")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut ka, mut th0, mut lin) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        ka.push(kac[idx] as F);
        th0.push(th0c[idx].to_radians() as F); // degrees → radians
        lin.push(linc[idx] != 0);
    }
    Ok(ForceTerm::indexed(AngleMmff {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        ka,
        theta0: th0,
        linear: lin,
    }))
}

/// The `linear` flag column of an `"angles"` block (0 = ordinary centre, 1 =
/// linear centre), baked by `typifier::mmff::frame_builder::annotate_mmff` from
/// the *central* atom's MMFF `linh` property.
///
/// It is an integer column rather than a boolean one because `MolGraph::to_frame`
/// materializes only f64 / i32 / string properties — a `PropValue::Bool` would be
/// dropped on the way to the [`Frame`], and the flag would silently read as
/// "nothing is linear", which is precisely the defect this column exists to fix.
///
/// Without it every nitrile / alkyne / allene angle would silently use the
/// cubic bend form, so its absence is refused ([`IrError::MissingParam`]).
///
/// [`IrError::MissingParam`]: crate::ff::ir::IrError::MissingParam
fn linear_column<'a>(
    block: &'a molrs::core::Block,
    style: &str,
) -> Result<&'a ndarray::ArrayD<molrs::op::I>, crate::ff::ir::IrError> {
    block
        .get("linear")
        .and_then(|c| c.get::<i32>())
        .ok_or_else(|| param_reads::missing(style, "", "linear"))
}

// ---------------------------------------------------------------------------
// AngleMmffStretchBend: E = 143.9325*(kba_ijk*dr_ij + kba_kji*dr_kj)*dth
// ---------------------------------------------------------------------------

pub struct AngleMmffStretchBend {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    kba_ijk: Vec<F>,
    kba_kji: Vec<F>,
    r0_ij: Vec<F>,
    r0_kj: Vec<F>,
    theta0: Vec<F>,
}

impl AngleMmffStretchBend {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let rij_vec = sub3(coords, i, coords, j);
            let rkj_vec = sub3(coords, k, coords, j);
            let rij = norm(rij_vec);
            let rkj = norm(rkj_vec);
            let theta = compute_angle(coords, i, j, k);
            let dr_ij = rij - self.r0_ij[idx];
            let dr_kj = rkj - self.r0_kj[idx];
            let dth = theta - self.theta0[idx];

            let term = self.kba_ijk[idx] * dr_ij + self.kba_kji[idx] * dr_kj;
            energy += conv * term * dth;

            // dE/ddth = conv * term
            accumulate_angle_forces(coords, i, j, k, conv * term, forces);
            // dE/dr_ij = conv * kba_ijk * dth
            if rij > 1e-12 as F {
                let f_r = -conv * self.kba_ijk[idx] * dth / rij;
                for dim in 0..3 {
                    forces[i * 3 + dim] += f_r * rij_vec[dim];
                    forces[j * 3 + dim] -= f_r * rij_vec[dim];
                }
            }
            // dE/dr_kj = conv * kba_kji * dth
            if rkj > 1e-12 as F {
                let f_r = -conv * self.kba_kji[idx] * dth / rkj;
                for dim in 0..3 {
                    forces[k * 3 + dim] += f_r * rkj_vec[dim];
                    forces[j * 3 + dim] -= f_r * rkj_vec[dim];
                }
            }
        }
        energy
    }
}

impl Potential for AngleMmffStretchBend {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t], self.atom_k[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for AngleMmffStretchBend {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k])
    }
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Build the MMFF stretch-bend potential from the typifier's baked columns.
///
/// # Linear centres carry no stretch-bend
///
/// MMFF defines no stretch-bend coupling at a linear centre (`linear = 1`): with
/// the bend energy a cosine of theta rather than a quadratic in `dtheta`, the
/// `(dr) * (dtheta)` cross term has no reference angle to be measured from. Those
/// rows are **skipped** here, exactly as the reference implementation skips them.
///
/// Only the linear centre is skipped, not the molecule: acetonitrile's three
/// methyl angles (H-C-H, H-C-C) contribute stretch-bend like any sp3 centre
/// (-0.010941 kcal/mol in total). "Nitriles have zero stretch-bend" is wrong.
pub fn angle_mmff_stretch_bend_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    // Per-instance parameters: the MMFF typifier baked the stretch-bend force
    // constants (kba_ijk/kba_kji, via the dfsb period-row default-row fallback
    // that the shared-table path lacked) plus the two reference bond lengths and
    // theta0 (degrees) onto each angle. This kernel only reads the columns.
    let block = frame.get(ANGLES).ok_or("mmff_stbn: missing \"angles\"")?;
    let ic = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomi")?;
    let jc = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomj")?;
    let kc = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomk")?;
    let kba_ijk_c = param_reads::instance_col("mmff_stbn", block, "kba_ijk")?;
    let kba_kji_c = param_reads::instance_col("mmff_stbn", block, "kba_kji")?;
    let r0ij = param_reads::instance_col("mmff_stbn", block, "r0_ij")?;
    let r0kj = param_reads::instance_col("mmff_stbn", block, "r0_kj")?;
    let th0 = param_reads::instance_col("mmff_stbn", block, "theta0")?;
    let linc = linear_column(block, "mmff_stbn")?;

    let n = ic.len();
    let mut pot = AngleMmffStretchBend {
        atom_i: Vec::with_capacity(n),
        atom_j: Vec::with_capacity(n),
        atom_k: Vec::with_capacity(n),
        kba_ijk: Vec::with_capacity(n),
        kba_kji: Vec::with_capacity(n),
        r0_ij: Vec::with_capacity(n),
        r0_kj: Vec::with_capacity(n),
        theta0: Vec::with_capacity(n),
    };
    for idx in 0..n {
        // A linear centre has no stretch-bend term at all.
        if linc[idx] != 0 {
            continue;
        }
        pot.atom_i.push(ic[idx] as usize);
        pot.atom_j.push(jc[idx] as usize);
        pot.atom_k.push(kc[idx] as usize);
        pot.kba_ijk.push(kba_ijk_c[idx] as F);
        pot.kba_kji.push(kba_kji_c[idx] as F);
        pot.r0_ij.push(r0ij[idx] as F);
        pot.r0_kj.push(r0kj[idx] as F);
        pot.theta0.push(th0[idx].to_radians() as F); // degrees → radians
    }
    Ok(ForceTerm::indexed(pot))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let pot = AngleMmff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            ka: vec![0.608],
            theta0: vec![theta0],
            linear: vec![false],
        };
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
