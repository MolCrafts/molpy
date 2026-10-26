//! Periodic improper (AMBER / GAFF / OpenMM impropers).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Periodic improper (AMBER / GAFF / OpenMM impropers):
///
/// E(φ) = k · [1 + cos(n·φ − γ)]
///
/// `k` is the force constant (energy), `periodicity` the multiplicity, and
/// `phase` the phase γ in **degrees**.
/// No LAMMPS style has this name: one term with γ ∈ {0°, 180°} is LAMMPS
/// `improper_style cvff` (`K = k`, `d = cos γ`, `n`), which is how the LAMMPS
/// writer emits it.
///
/// # Atom order
///
/// The improper angle φ is the dihedral I-J-K-L **of the stored order**, as in
/// every LAMMPS improper style that is a dihedral (`cvff`, `harmonic`), and as
/// AMBER, GROMACS and LAMMPS each compute it from their topology files. An
/// AMBER-family improper is stored in AMBER's order — the centre **third**
/// (`atomk`) — because that is the order whose dihedral is AMBER's angle: no
/// order with the centre first has the same dihedral (the axis of an AMBER
/// improper runs through the centre). It is therefore also the order a LAMMPS
/// data file lists such an improper in for `cvff` to reproduce AMBER's energy.
/// OpenMM's XML lists the centre first and evaluates `(c2, c3, c1, c4)`; its
/// reader and writer map between the two.
#[derive(Clone, Copy, Debug, Default)]
pub struct Periodic;

impl ScalarForm for Periodic {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, n, d) = (p.input("k"), p.input("periodicity"), p.input("phase"));
        for t in 0..phi.len() {
            let arg = n[t] * phi[t] - d[t].to_radians();
            e[t] = k[t] * (1.0 + arg.cos());
            de[t] = -k[t] * n[t] * arg.sin();
        }
    }

    fn inputs(&self) -> Vec<String> {
        ["k", "periodicity", "phase"].map(String::from).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use crate::ff::compile::ExplicitTerms;
    use crate::ff::ir::Params;
    use crate::ff::potential::Potentials;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::op::F;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(k: F, n: F, d_deg: F) -> Potentials {
        ExplicitTerms::new("improper", "periodic")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k", k), ("periodicity", n), ("phase", d_deg)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_phase() {
        let p = single(1.5, 1.0, 0.0);
        assert!((p.energy_forces_at(&quad(0.0)).0 - 3.0).abs() < 1e-9);
        assert!(p.energy_forces_at(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, 2.0, 30.0);
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
        let pot = single(1.0, 3.0, 45.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
