//! Class2 (quartic) bond (LAMMPS `bond_style class2`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// LAMMPS `bond_style class2`, the COMPASS anharmonic bond:
/// `E = k2 (r − r0)² + k3 (r − r0)³ + k4 (r − r0)⁴`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Class2;

impl ScalarForm for Class2 {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (r0, k2, k3, k4) = (p.input("r0"), p.input("k2"), p.input("k3"), p.input("k4"));
        for t in 0..r.len() {
            let dr = r[t] - r0[t];
            let dr2 = dr * dr;
            e[t] = k2[t] * dr2 + k3[t] * dr2 * dr + k4[t] * dr2 * dr2;
            // dE/dr = 2 k2 dr + 3 k3 dr² + 4 k4 dr³
            de[t] = 2.0 * k2[t] * dr + 3.0 * k3[t] * dr2 + 4.0 * k4[t] * dr2 * dr;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["r0", "k2", "k3", "k4"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn class2() -> Potentials {
        ExplicitTerms::new("bond", "class2")
            .term(
                &[0, 1],
                Params::from_iter([("r0", 1.5), ("k2", 100.0), ("k3", 10.0), ("k4", 5.0)]),
            )
            .compile()
            .unwrap()
    }

    fn numerical_forces(pot: &Potentials, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.energy_at(&cp) - pot.energy_at(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn energy_matches_closed_form() {
        // r0=1.5, r=2.0 -> dr=0.5; E = 100*.25 + 10*.125 + 5*.0625
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let dr: F = 0.5;
        let expected = 100.0 * dr * dr + 10.0 * dr.powi(3) + 5.0 * dr.powi(4);
        assert!((class2().energy_at(&coords) - expected).abs() < 1e-9);
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = class2();
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
