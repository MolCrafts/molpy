//! Class2 (quartic) angle (LAMMPS `angle_style class2`, bend term).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// LAMMPS `angle_style class2`'s bend:
/// `E = k2 (θ − θ0)² + k3 (θ − θ0)³ + k4 (θ − θ0)⁴`, `theta0` in degrees.
///
/// The bond-bond and bond-angle cross terms of LAMMPS's style are not part of
/// this style (its codec writes them at zero and refuses them otherwise).
#[derive(Clone, Copy, Debug, Default)]
pub struct Class2;

impl ScalarForm for Class2 {
    fn eval(&self, theta: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (theta0, k2, k3, k4) = (
            p.input("theta0"),
            p.input("k2"),
            p.input("k3"),
            p.input("k4"),
        );
        for t in 0..theta.len() {
            let dt = theta[t] - theta0[t].to_radians();
            let dt2 = dt * dt;
            e[t] = k2[t] * dt2 + k3[t] * dt2 * dt + k4[t] * dt2 * dt2;
            // dE/dθ = 2 k2 dt + 3 k3 dt² + 4 k4 dt³
            de[t] = 2.0 * k2[t] * dt + 3.0 * k3[t] * dt2 + 4.0 * k4[t] * dt2 * dt;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["theta0", "k2", "k3", "k4"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn class2(theta0_deg: F) -> Potentials {
        ExplicitTerms::new("angle", "class2")
            .term(
                &[0, 1, 2],
                Params::from_iter([
                    ("theta0", theta0_deg),
                    ("k2", 50.0),
                    ("k3", 10.0),
                    ("k4", 5.0),
                ]),
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
    fn energy_zero_at_equilibrium() {
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        assert!(class2(90.0).energy_at(&coords).abs() < 1e-9);
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = class2(1.9_f64.to_degrees());
        // bent, off-axis geometry
        let coords: Vec<F> = vec![1.0, 0.2, 0.0, 0.0, 0.0, 0.1, -0.3, 1.0, 0.0];
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
