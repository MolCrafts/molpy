//! Polynomial-in-cos φ proper dihedrals (LAMMPS `multi/harmonic`, `nharmonic`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Polynomial-in-cos φ proper dihedrals: LAMMPS `dihedral_style
/// multi/harmonic` and `dihedral_style nharmonic`,
///
/// E(φ) = Σ_{i=1..N} A_i · cos^(i−1)(φ)
///
/// `multi/harmonic` is the N = 5 case (`a1..a5`, an absent one 0);
/// `nharmonic` takes any N ≥ 1 (the indexed family `a1..aN`, contiguous —
/// LAMMPS's `dihedral_coeff t N A1 … AN`). The coefficients are energies.
/// One form prices both: it reads `a1, a2, …` as far as the columns go.
#[derive(Clone, Copy, Debug, Default)]
pub struct MultiHarmonic;

impl ScalarForm for MultiHarmonic {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let a: Vec<&[F]> = (1..).map_while(|m| p.get(&format!("a{m}"))).collect();
        for t in 0..phi.len() {
            let c = phi[t].cos();
            // E = Σ A_i c^(i−1), dE/dc = Σ (i−1) A_i c^(i−2)  (Horner, from the top)
            let (mut energy, mut de_dc): (F, F) = (0.0, 0.0);
            for (q, ai) in a.iter().enumerate().rev() {
                energy = ai[t] + c * energy;
                if q > 0 {
                    de_dc = q as F * ai[t] + c * de_dc;
                }
            }
            e[t] = energy;
            // dE/dφ = dE/dc · (−sinφ)
            de[t] = -phi[t].sin() * de_dc;
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(a: [F; 5]) -> Potentials {
        let names = ["a1", "a2", "a3", "a4", "a5"];
        ExplicitTerms::new("dihedral", "multi/harmonic")
            .term(&[0, 1, 2, 3], names.into_iter().zip(a).collect())
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_matches_series() {
        // At φ=0, cos=1 → E = ΣA_n. At φ=π/2, cos=0 → E = A₁.
        let a = [0.5, 1.0, -0.3, 0.2, 0.1];
        let e0 = single(a).energy_forces_at(&quad(0.0)).0;
        assert!((e0 - a.iter().sum::<F>()).abs() < 1e-9, "E(0) got {e0}");
        let e90 = single(a)
            .energy_forces_at(&quad(std::f64::consts::FRAC_PI_2))
            .0;
        assert!((e90 - a[0]).abs() < 1e-9, "E(pi/2) got {e90}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single([0.5, 1.3, -0.7, 0.9, 0.4]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.energy_forces_at(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let ep = pot.energy_forces_at(&cp).0;
            let em = pot.energy_forces_at(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single([0.5, 1.0, 0.5, 0.3, 0.2]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}

#[cfg(test)]
mod nharmonic_tests {
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::op::{F, Idx};
    use ndarray::Array1;

    fn one_dihedral(
        style: &str,
        params: Params,
    ) -> Result<crate::ff::potential::Potentials, crate::ff::potential::CompileError> {
        let mut ff = ForceField::new("t");
        ff.def_style("dihedral", style, Params::new())
            .unwrap()
            .def_type("t", &["a", "b", "c", "d"], params)
            .unwrap();
        let mut block = Block::new();
        for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
            block
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                .unwrap();
        }
        block
            .insert(
                "type",
                Array1::from_vec(vec!["t".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("dihedrals", block);
        PotentialCompiler::new(&ff).compile(&frame)
    }

    /// LAMMPS `dihedral_style nharmonic`, E = Σ A_i cos^(i−1) φ: N = 7,
    /// A = (1, −2, 3, −4, 5, −6, 7) at φ = 40°.
    #[test]
    fn energy_is_the_lammps_formula() {
        let a = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0];
        let pairs: Vec<(String, F)> = a
            .iter()
            .enumerate()
            .map(|(i, &v)| (format!("a{}", i + 1), v))
            .collect();
        let refs: Vec<(&str, F)> = pairs.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let pots = one_dihedral("nharmonic", Params::from_iter(refs.iter().copied())).unwrap();
        let phi: F = 40.0_f64.to_radians();
        let (s, c) = phi.sin_cos();
        let coords = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s];
        let want: F = a
            .iter()
            .enumerate()
            .map(|(i, v)| v * c.powi(i as i32))
            .sum();
        let got = pots.energy_at(&coords);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        crate::ff::potential::fixtures::assert_forces_are_negative_gradient(&pots, &coords, 1e-5);
    }

    /// No `a1`, or a coefficient past a gap, is refused at compile time.
    #[test]
    fn a_missing_or_gapped_coefficient_is_refused() {
        let missing = |err: crate::ff::potential::CompileError| match err.ir() {
            Some(crate::ff::ir::IrError::MissingParam { param, .. }) => param.clone(),
            _ => panic!("{err}"),
        };
        let err = one_dihedral("nharmonic", Params::from_iter([("a2", 1.0)])).unwrap_err();
        assert_eq!(missing(err), "a1");
        let err =
            one_dihedral("nharmonic", Params::from_iter([("a1", 1.0), ("a3", 1.0)])).unwrap_err();
        assert_eq!(missing(err), "a2");
    }
}
