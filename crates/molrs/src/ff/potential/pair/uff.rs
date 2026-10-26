//! UFF van der Waals (RDKit `vdWContrib`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm, eval_by_term};
use molrs::op::F;

/// UFF van der Waals: `E = D_ij · [(x_ij/r)¹² − 2·(x_ij/r)⁶]` (RDKit
/// `vdWContrib`), truncated beyond `2 x_ij` (RDKit's threshold multiplier).
///
/// `x1` and `D1` are per-instance columns the UFF typifier bakes onto each
/// atom; the pair's `x_ij = √(x1ᵢ x1ⱼ)`, `D_ij = √(D1ᵢ D1ⱼ)` combine them
/// as RDKit does, so the form reads each atom's own (`x11`, `x12`, `D11`,
/// `D12`).
#[derive(Clone, Copy, Debug, Default)]
pub struct UffLj;

impl ScalarForm for UffLj {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let _ = p;
        Some(|r: F, x: &[F]| {
            let xij = (x[0] * x[1]).sqrt();
            let dij = (x[2] * x[3]).sqrt();
            // cutoff at 2 * xij (RDKit default threshMultiplier ≈ 2 when used that way)
            if r > 2.0 * xij {
                return (0.0, 0.0);
            }
            let inv_r = 1.0 / r;
            let rr = xij * inv_r;
            let r2u = rr * rr;
            let r6 = r2u * r2u * r2u;
            let r12 = r6 * r6;
            // RDKit: preFactor = 12·D/x · ((x/r)⁷ − (x/r)¹³) = dE/dr.
            let r7 = r6 * rr;
            let r13 = r12 * rr;
            (dij * (r12 - 2.0 * r6), 12.0 * dij / xij * (r7 - r13))
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["x11", "x12", "D11", "D12"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::{Params, SpecialBonds};
    use crate::ff::potential::fixtures::{OneShot, assert_forces_are_negative_gradient};
    use molrs::core::{Block, Frame};
    use molrs::op::F;
    use ndarray::Array1;

    /// Two atoms with their own `x1`, `D1` columns, one `pairs` row.
    fn pair(x1: [F; 2], d1: [F; 2]) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert("x1", Array1::from_vec(x1.to_vec()).into_dyn(), None)
            .unwrap();
        atoms
            .insert("D1", Array1::from_vec(d1.to_vec()).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame
    }

    /// `D_ij [(x_ij/r)¹² − 2 (x_ij/r)⁶]` with the geometric `x_ij`, `D_ij`
    /// of the atoms' columns, and nothing beyond `2 x_ij`; the type rows a
    /// per-instance style is handed price nothing.
    #[test]
    fn energy_combines_the_atoms_columns_geometrically() {
        let mut ff = ForceField::new("uff");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "uff_lj", Params::new()).unwrap();
        let (x1, d1) = ([3.5, 3.1], [0.06, 0.09]);
        let pots = PotentialCompiler::new(&ff).compile(&pair(x1, d1)).unwrap();
        let (xij, dij) = ((x1[0] * x1[1]).sqrt(), (d1[0] * d1[1]).sqrt());
        for r in [2.9, 3.4, 4.8] {
            let s6 = (xij / r).powi(6);
            let want = dij * (s6 * s6 - 2.0 * s6);
            let got = pots.energy_at(&[0.0, 0.0, 0.0, r, 0.0, 0.0]);
            assert!((got - want).abs() < 1e-12, "r = {r}: {got} vs {want}");
        }
        assert_eq!(
            pots.energy_at(&[0.0, 0.0, 0.0, 2.0 * xij + 0.1, 0.0, 0.0]),
            0.0
        );
        let coords = [0.1, -0.2, 0.3, 2.9, 0.5, -0.6];
        assert_forces_are_negative_gradient(&pots, &coords, 1e-6);
    }
}
