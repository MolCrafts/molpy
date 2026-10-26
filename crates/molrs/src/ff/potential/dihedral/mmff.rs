//! MMFF94 torsional rotation:
//! `E = ½ (V1 (1 + cos φ) + V2 (1 − cos 2φ) + V3 (1 + cos 3φ))`.

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// MMFF94 torsion (MMFF.I eq. 6), `v1`, `v2`, `v3` (kcal/mol) per-instance
/// columns the MMFF typifier bakes onto each dihedral.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mmff;

impl ScalarForm for Mmff {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (v1, v2, v3) = (p.input("v1"), p.input("v2"), p.input("v3"));
        for t in 0..phi.len() {
            let (s1, c1) = phi[t].sin_cos();
            let (s2, c2) = (2.0 * phi[t]).sin_cos();
            let (s3, c3) = (3.0 * phi[t]).sin_cos();
            e[t] = 0.5 * (v1[t] * (1.0 + c1) + v2[t] * (1.0 - c2) + v3[t] * (1.0 + c3));
            de[t] = 0.5 * (-v1[t] * s1 + 2.0 * v2[t] * s2 - 3.0 * v3[t] * s3);
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["v1", "v2", "v3"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    #[test]
    fn test_mmff_torsion() {
        let pot: Potentials = ExplicitTerms::new("dihedral", "mmff_torsion")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("v1", 0.0), ("v2", 0.0), ("v3", 0.3)]),
            )
            .compile()
            .unwrap();
        let coords: Vec<F> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e.is_finite());
        let fx: F = forces.iter().step_by(3).sum();
        let fy: F = forces.iter().skip(1).step_by(3).sum();
        let fz: F = forces.iter().skip(2).step_by(3).sum();
        assert!(
            (fx.abs() + fy.abs() + fz.abs()) < 0.1,
            "force sum too large"
        );
    }
}
