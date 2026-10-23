//! MMFF94 bond stretching:
//! `E = ½ · 143.9325 · kb · dr² · (1 + cs·dr + 7/12·cs²·dr²)`.

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;
use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Cubic stretch constant (Å⁻¹).
const CS: f64 = -2.0;

/// MMFF94 bond stretching (MMFF.I eq. 2), `kb` in md/Å and `r0` in Å, both
/// per-instance columns the MMFF typifier bakes onto each bond (143.9325
/// converts md·Å to kcal/mol).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mmff;

impl ScalarForm for Mmff {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (kb, r0) = (p.input("kb"), p.input("r0"));
        let cs = CS as F;
        let cs2 = cs * cs;
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;
        for t in 0..r.len() {
            let dr = r[t] - r0[t];
            e[t] = 0.5 * conv * kb[t] * dr * dr * (1.0 + cs * dr + (7.0 / 12.0) * cs2 * dr * dr);
            de[t] = conv * kb[t] * dr * (1.0 + 1.5 * cs * dr + (7.0 / 6.0) * cs2 * dr * dr);
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["kb", "r0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn bond() -> Potentials {
        ExplicitTerms::new("bond", "mmff_bond")
            .term(&[0, 1], Params::from_iter([("kb", 4.258), ("r0", 1.508)]))
            .compile()
            .unwrap()
    }

    #[test]
    fn test_mmff_bond_at_equilibrium() {
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.508, 0.0, 0.0];
        let (e, _) = bond().energy_forces_at(&coords);
        assert!(
            e.abs() < 1e-6,
            "energy at equilibrium should be ~0, got {}",
            e
        );
    }

    #[test]
    fn test_mmff_bond_stretched() {
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.608, 0.0, 0.0];
        let (e, forces) = bond().energy_forces_at(&coords);
        assert!(e > 0.0, "stretched bond should have positive energy");
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "force sum should be ~0, got {}", sum);
        }
    }
}
