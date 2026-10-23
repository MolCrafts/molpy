//! Harmonic angle (LAMMPS `angle_style harmonic`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// LAMMPS `angle_style harmonic`: `E = k (θ − θ0)²`.
///
/// `k` is LAMMPS's `K` (energy/rad², the ½ included) and `theta0` is in
/// **degrees**, as the `angle_coeff` line writes it; the form converts it to
/// radians.
#[derive(Clone, Copy, Debug, Default)]
pub struct Harmonic;

impl ScalarForm for Harmonic {
    fn eval(&self, theta: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, theta0) = (p.input("k"), p.input("theta0"));
        for t in 0..theta.len() {
            let dtheta = theta[t] - theta0[t].to_radians();
            e[t] = k[t] * dtheta * dtheta;
            de[t] = 2.0 * k[t] * dtheta;
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "theta0"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Frame;
    use molrs::op::F;

    /// Atoms at (1,0,0), (0,0,0), (0,1,0): one 90° angle of type `label`.
    fn right_angle_frame(label: &str) -> Frame {
        use molrs::core::Block;
        use molrs::op::Idx;
        use ndarray::Array1;
        let mut atoms = Block::new();
        for (key, v) in [
            ("x", [1.0, 0.0, 0.0]),
            ("y", [0.0, 0.0, 1.0]),
            ("z", [0.0; 3]),
        ] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut angles = Block::new();
        for (key, a) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            angles
                .insert(key, Array1::from_vec(vec![a as Idx]).into_dyn(), None)
                .unwrap();
        }
        angles
            .insert(
                "type",
                Array1::from_vec(vec![label.to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", angles);
        frame
    }

    /// LAMMPS `angle_style harmonic`: E = K(θ − θ0)², θ0 given in degrees.
    #[test]
    fn energy_is_the_lammps_formula_with_theta0_in_degrees() {
        let mut ff = crate::ff::forcefield::ForceField::new("t");
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A",
                &["A", "A", "A"],
                Params::from_iter([("k", 50.0), ("theta0", 100.0)]),
            )
            .unwrap();
        let frame = right_angle_frame("A-A-A");
        let pots = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap();
        // The frame's angle is 90 degrees.
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let want = 50.0 * (90.0_f64.to_radians() - 100.0_f64.to_radians()).powi(2);
        let got = pots.energy_at(&coords);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn test_angle_harmonic_energy() {
        let pot = ExplicitTerms::new("angle", "harmonic")
            .term(
                &[0, 1, 2],
                Params::from_iter([("k", 50.0), ("theta0", 90.0)]),
            )
            .compile()
            .unwrap();
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let (e, _) = pot.energy_forces_at(&coords);
        assert!(e.abs() < 1e-4);
    }
}
