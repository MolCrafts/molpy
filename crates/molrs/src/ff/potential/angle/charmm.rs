//! CHARMM angle with Urey–Bradley (LAMMPS `angle_style charmm`).

use crate::ff::potential::flat_coords::{accumulate_angle_forces, compute_angle, sub3};
use crate::ff::potential::form_kernel::{CompoundForm, ParamColumns};
use crate::op::vec3::norm;
use molrs::op::F;

/// LAMMPS `angle_style charmm`: `E = k (θ − θ0)² + k_ub (r₁₃ − r_ub)²`.
///
/// `k` is LAMMPS's `K` (energy/rad², no ½), `theta0` is in **degrees**, `k_ub`
/// is LAMMPS's `K_ub` (energy/length², no ½) and `r_ub` a length — the four
/// numbers of an `angle_coeff t K theta0 K_ub r_ub` line, in that order.
///
/// r₁₃ is the distance between the angle's end atoms `i` and `k`, so the
/// energy is no function of θ alone: an N-body form of the three positions.
/// The Urey–Bradley term is a 1-3 harmonic spring that this angle owns: it
/// adds no special pair (how a non-bonded style prices a 1-3 pair is its
/// `pairs` row under `special_bonds`, as for any angle).
#[derive(Clone, Copy, Debug, Default)]
pub struct Charmm;

impl CompoundForm for Charmm {
    fn eval(
        &self,
        x: &[[F; 3]],
        arity: usize,
        p: &ParamColumns<'_>,
        e: &mut [F],
        grad: &mut [[F; 3]],
    ) {
        debug_assert_eq!(arity, 3);
        let (k, theta0) = (p.input("k"), p.input("theta0"));
        let (k_ub, r_ub) = (p.input("k_ub"), p.input("r_ub"));
        for t in 0..e.len() {
            let flat: [F; 9] = std::array::from_fn(|c| x[t * 3 + c / 3][c % 3]);
            let mut f = [0.0; 9];
            // Bend: dE/dθ = 2k(θ − θ0).
            let dtheta = compute_angle(&flat, 0, 1, 2) - theta0[t].to_radians();
            let mut energy = k[t] * dtheta * dtheta;
            accumulate_angle_forces(&flat, 0, 1, 2, 2.0 * k[t] * dtheta, &mut f);
            // Urey–Bradley: a spring along r₁₃ = x_i − x_k.
            let rik = sub3(&flat, 0, &flat, 2);
            let r = norm(rik);
            let dr = r - r_ub[t];
            energy += k_ub[t] * dr * dr;
            if r > 0.0 {
                // F_i = −dE/dr · r̂, F_k = −F_i.
                let f_over_r = -2.0 * k_ub[t] * dr / r;
                for (d, &xd) in rik.iter().enumerate() {
                    f[d] += f_over_r * xd;
                    f[6 + d] -= f_over_r * xd;
                }
            }
            e[t] = energy;
            for (a, g) in grad[t * 3..t * 3 + 3].iter_mut().enumerate() {
                *g = [-f[a * 3], -f[a * 3 + 1], -f[a * 3 + 2]];
            }
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "theta0", "k_ub", "r_ub"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::{ExplicitTerms, PotentialCompiler};
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::flat_coords::{compute_angle, sub3};
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::op::F;
    use molrs::op::Idx;
    use ndarray::Array1;

    const K: F = 33.43;
    const THETA0_DEG: F = 110.1;
    const K_UB: F = 22.53;
    const R_UB: F = 2.179;

    fn pot(theta0_deg: F) -> Potentials {
        one(K, theta0_deg, K_UB, R_UB)
    }

    /// One `angle charmm` term over atoms 0-1-2, `theta0` in degrees.
    fn one(k: F, theta0_deg: F, k_ub: F, r_ub: F) -> Potentials {
        ExplicitTerms::new("angle", "charmm")
            .term(
                &[0, 1, 2],
                Params::from_iter([
                    ("k", k),
                    ("theta0", theta0_deg),
                    ("k_ub", k_ub),
                    ("r_ub", r_ub),
                ]),
            )
            .compile()
            .unwrap()
    }

    /// Atom 0 at distance `a` along x, vertex at the origin, atom 2 at
    /// distance `b` at angle `theta_deg` in the xy-plane.
    fn geometry(a: F, b: F, theta_deg: F) -> Vec<F> {
        let t = theta_deg.to_radians();
        vec![a, 0.0, 0.0, 0.0, 0.0, 0.0, b * t.cos(), b * t.sin(), 0.0]
    }

    /// The LAMMPS manual's formula, by hand: law of cosines for r₁₃.
    fn hand_energy(a: F, b: F, theta_deg: F) -> F {
        let t = theta_deg.to_radians();
        let r13 = (a * a + b * b - 2.0 * a * b * t.cos()).sqrt();
        K * (t - THETA0_DEG.to_radians()).powi(2) + K_UB * (r13 - R_UB).powi(2)
    }

    #[test]
    fn energy_is_the_lammps_formula_over_several_geometries() {
        let p = pot(THETA0_DEG);
        for &(a, b, theta) in &[
            (1.53, 1.09, 110.1),
            (1.53, 1.09, 95.0),
            (1.40, 1.60, 130.0),
            (1.0, 1.0, 60.0),
            (1.2, 0.9, 175.0),
        ] {
            let got = p.energy_at(&geometry(a, b, theta));
            let want = hand_energy(a, b, theta);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "a={a} b={b} θ={theta}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn at_both_equilibria_the_energy_and_force_vanish() {
        // θ = θ0 and r₁₃ = r_ub: a = b = r_ub / (2 sin(θ0/2)).
        let a = R_UB / (2.0 * (THETA0_DEG.to_radians() / 2.0).sin());
        let (e, f) = pot(THETA0_DEG).energy_forces_at(&geometry(a, a, THETA0_DEG));
        assert!(e.abs() < 1e-24, "{e}");
        assert!(f.iter().all(|x| x.abs() < 1e-10), "{f:?}");
    }

    #[test]
    fn forces_match_finite_difference_and_sum_to_zero() {
        let p = pot(THETA0_DEG);
        let coords: Vec<F> = vec![1.3, 0.2, -0.1, 0.05, 0.0, 0.1, -0.4, 1.1, 0.3];
        let (_, analytic) = p.energy_forces_at(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let (mut cp, mut cm) = (coords.clone(), coords.clone());
            cp[d] += h;
            cm[d] -= h;
            let numeric = -(p.energy_at(&cp) - p.energy_at(&cm)) / (2.0 * h);
            assert!(
                (analytic[d] - numeric).abs() < 1e-6,
                "dof {d}: {} vs {numeric}",
                analytic[d]
            );
        }
        // Newton's third law: no net force, and no net torque.
        for d in 0..3 {
            let net: F = (0..3).map(|a| analytic[a * 3 + d]).sum();
            assert!(net.abs() < 1e-10, "net force {d}: {net}");
        }
        let mut torque = [0.0; 3];
        for a in 0..3 {
            let (x, f) = (&coords[a * 3..a * 3 + 3], &analytic[a * 3..a * 3 + 3]);
            torque[0] += x[1] * f[2] - x[2] * f[1];
            torque[1] += x[2] * f[0] - x[0] * f[2];
            torque[2] += x[0] * f[1] - x[1] * f[0];
        }
        assert!(torque.iter().all(|t| t.abs() < 1e-10), "{torque:?}");
    }

    #[test]
    fn the_urey_bradley_force_lies_along_r13() {
        // k = 0 isolates the UB term: the force on i is parallel to x_i − x_k
        // and the vertex feels nothing.
        let p = one(0.0, 0.0, K_UB, R_UB);
        let coords = geometry(1.5, 1.2, 100.0);
        let (_, f) = p.energy_forces_at(&coords);
        let r13 = sub3(&coords, 0, &coords, 2);
        let cross = [
            f[1] * r13[2] - f[2] * r13[1],
            f[2] * r13[0] - f[0] * r13[2],
            f[0] * r13[1] - f[1] * r13[0],
        ];
        assert!(cross.iter().all(|c| c.abs() < 1e-12), "{cross:?}");
        assert!(f[3..6].iter().all(|x| *x == 0.0), "{f:?}");
    }

    fn frame(label: &str, coords: &[F]) -> Frame {
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].into_iter().enumerate() {
            let col: Vec<F> = (0..3).map(|a| coords[a * 3 + d]).collect();
            atoms
                .insert(key, Array1::from_vec(col).into_dyn(), None)
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
        // The two bonds the angle spans: a pair's 1-3 class is its bond
        // distance.
        let mut bonds = Block::new();
        for (key, ends) in [("atomi", vec![0 as Idx, 1]), ("atomj", vec![1, 2])] {
            bonds
                .insert(key, Array1::from_vec(ends).into_dyn(), None)
                .unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        frame.insert("angles", angles);
        frame
    }

    fn charmm_ff() -> ForceField {
        let mut ff = ForceField::new("t");
        ff.def_style("angle", "charmm", Params::new())
            .unwrap()
            .def_type(
                "HA-CT-HA",
                &["HA", "CT", "HA"],
                Params::from_iter([
                    ("k", K),
                    ("theta0", THETA0_DEG),
                    ("k_ub", K_UB),
                    ("r_ub", R_UB),
                ]),
            )
            .unwrap();
        ff
    }

    /// `theta0` is stored in degrees, as the `angle_coeff` line writes it, and
    /// the compiled energy is the formula with θ0 in radians.
    #[test]
    fn compiled_theta0_is_degrees() {
        let coords = geometry(1.53, 1.09, 104.0);
        let pots = PotentialCompiler::new(&charmm_ff())
            .compile(&frame("HA-CT-HA", &coords))
            .unwrap();
        let got = pots.energy_at(&coords);
        let want = hand_energy(1.53, 1.09, 104.0);
        assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
    }

    #[test]
    fn both_compile_doors_price_it_the_same() {
        let coords: Vec<F> = vec![1.3, 0.2, -0.1, 0.05, 0.0, 0.1, -0.4, 1.1, 0.3];
        let ff = charmm_ff();
        let compiler = PotentialCompiler::new(&ff);
        let f = frame("HA-CT-HA", &coords);
        let (e1, f1) = compiler.compile(&f).unwrap().energy_forces_at(&coords);
        let typed = compiler.compile_typed(&f).unwrap();
        assert_eq!(typed.len(), 1);
        let (member, weights) = &typed[0];
        assert!(
            weights.is_empty(),
            "a bonded member takes no special weights"
        );
        let (e2, f2) = member.energy_forces_at(&coords);
        assert_eq!(e1, e2);
        assert_eq!(f1, f2);
        assert_eq!(e1, pot(THETA0_DEG).energy_at(&coords));
    }

    #[test]
    fn a_type_missing_a_urey_bradley_param_is_refused() {
        let mut ff = ForceField::new("t");
        ff.def_style("angle", "charmm", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A", "A", "A"],
                Params::from_iter([("k", K), ("theta0", THETA0_DEG), ("k_ub", K_UB)]),
            )
            .unwrap();
        let coords = geometry(1.0, 1.0, 100.0);
        let err = PotentialCompiler::new(&ff)
            .compile(&frame("A", &coords))
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                err.ir(),
                Some(crate::ff::ir::IrError::MissingParam { style, type_, param })
                    if style == "charmm" && type_ == "A" && param == "r_ub"
            ),
            "{err}"
        );
    }

    /// A frame of `xyz` with one `angles` row per `(i, j, k, type)`.
    fn molecule(xyz: &[[F; 3]], angles: &[(Idx, Idx, Idx, &str)]) -> Frame {
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].into_iter().enumerate() {
            let col: Vec<F> = xyz.iter().map(|p| p[d]).collect();
            atoms
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let mut block = Block::new();
        for (slot, key) in ["atomi", "atomj", "atomk"].into_iter().enumerate() {
            let col: Vec<Idx> = angles.iter().map(|a| [a.0, a.1, a.2][slot]).collect();
            block
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let types: Vec<String> = angles.iter().map(|a| a.3.to_owned()).collect();
        block
            .insert("type", Array1::from_vec(types).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", block);
        frame
    }

    /// Beside another angle style, each style prices its own rows (the
    /// compiler cuts the `angles` block per style), both doors agree, and a
    /// row whose type neither style defines is still refused.
    #[test]
    fn a_mixed_angle_field_prices_each_row_under_its_own_style() {
        let mut ff = charmm_ff();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT",
                &["CT", "CT", "CT"],
                Params::from_iter([("k", 58.35), ("theta0", 113.6)]),
            )
            .unwrap();
        let xyz = [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.05, 1.44, 0.10],
            [-0.36, 1.02, 0.05],
        ];
        let coords: Vec<F> = xyz.iter().flatten().copied().collect();
        let both = molecule(&xyz, &[(0, 1, 2, "CT-CT-CT"), (3, 0, 1, "HA-CT-HA")]);
        let compiler = PotentialCompiler::new(&ff);
        let e = compiler.compile(&both).unwrap().energy_at(&coords);
        let typed: F = compiler
            .compile_typed(&both)
            .unwrap()
            .iter()
            .map(|(m, _)| m.energy_at(&coords))
            .sum();
        assert_eq!(e, typed);
        let harmonic = 58.35 * (compute_angle(&coords, 0, 1, 2) - 113.6_f64.to_radians()).powi(2);
        let ub = pot(THETA0_DEG).energy_at(&[
            xyz[3][0], xyz[3][1], xyz[3][2], 0.0, 0.0, 0.0, 1.53, 0.0, 0.0,
        ]);
        assert!(
            (e - (harmonic + ub)).abs() <= 1e-12 * e,
            "{e} vs {}",
            harmonic + ub
        );

        let stray = molecule(&xyz, &[(0, 1, 2, "CT-CT-CT"), (3, 0, 1, "XX-XX-XX")]);
        let err = compiler.compile(&stray).map(|_| ()).unwrap_err();
        assert!(
            err.to_string()
                .contains("'XX-XX-XX' is defined by no angle style"),
            "{err}"
        );
    }
}
