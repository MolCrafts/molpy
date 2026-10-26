//! CHARMM proper dihedral (LAMMPS `dihedral_style charmm`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// CHARMM proper dihedral (LAMMPS `dihedral_style charmm`):
///
/// E(φ) = k·[1 + cos(n·φ − d)]
///
/// `k` is LAMMPS's `K` (energy), `periodicity` its integer `n`, and `phase` its
/// `d` in **degrees** (LAMMPS takes an integer number of degrees; any value is
/// accepted here).
///
/// # The weight `w`
///
/// LAMMPS's fourth coefficient `w` weights a 1-4 non-bonded pair that the
/// *dihedral* computes — its end atoms, with the `epsilon14` / `sigma14` of
/// the `lj/charmm` pair style and the full Coulomb, beside `special_bonds`
/// 1-4 weights of zero. This form prices the torsion alone and does not read
/// `w`; the compiler routes every dihedral's `w` pair to the 1-4 exceptions
/// kernel ([`PairExceptions`](crate::ff::potential::pair::PairExceptions)),
/// which also makes LAMMPS's checks (`0 ≤ w ≤ 1`, `special_bonds` 1-4 = 0, a
/// `lj/charmm` pair style). `w = 0` (or absent) is the AMBER use of the style.
#[derive(Clone, Copy, Debug, Default)]
pub struct Charmm;

impl ScalarForm for Charmm {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let (k, n, d) = (p.input("k"), p.input("periodicity"), p.input("phase"));
        for t in 0..phi.len() {
            let arg = n[t] * phi[t] - d[t].to_radians();
            e[t] = k[t] * (1.0 + arg.cos());
            // dE/dφ = −K·n·sin(n·φ − d)
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
    use molrs::core::Frame;
    use molrs::op::F;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    /// A LAMMPS-read `dihedral_style charmm` field and a frame with its one
    /// dihedral at φ = 60°.
    fn lammps_charmm(w: &str) -> (crate::ff::forcefield::ForceField, Frame) {
        use crate::io::reader::ForceFieldReader;
        use molrs::core::Block;
        use molrs::op::Idx;
        use ndarray::Array1;
        let text = format!(
            "special_bonds charmm\ndihedral_style charmm\ndihedral_coeff a-b-c-d 0.2 3 180 {w}\n"
        );
        let ff = crate::io::lammps::forcefield_reader::LammpsForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap();
        let mut dihedrals = Block::new();
        for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
            dihedrals
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                .unwrap();
        }
        dihedrals
            .insert(
                "type",
                Array1::from_vec(vec!["a-b-c-d".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("dihedrals", dihedrals);
        (ff, frame)
    }

    /// A non-zero `w` prices its pair with `lj/charmm`'s `epsilon14` /
    /// `sigma14`; a field without that pair style is refused, as LAMMPS
    /// refuses it ("Dihedral charmm is incompatible with Pair style"), and so
    /// is a `w` outside `[0, 1]`.
    #[test]
    fn a_nonzero_weight_needs_lj_charmm_and_a_weight_in_range() {
        let (ff, frame) = lammps_charmm("1.0");
        let err = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap_err();
        assert!(err.to_string().contains("lj/charmm"), "{err}");
        let (ff, frame) = lammps_charmm("1.5");
        let err = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap_err();
        assert!(
            err.to_string().contains("a-b-c-d") && err.to_string().contains("[0, 1]"),
            "{err}"
        );
    }

    /// `w = 0` is the AMBER use of the style and prices LAMMPS's
    /// `K[1 + cos(nφ − d)]`, `d` in degrees.
    #[test]
    fn a_zero_weight_compiles_to_the_lammps_energy() {
        let (ff, frame) = lammps_charmm("0.0");
        let pots = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap();
        let phi = 60.0_f64.to_radians();
        let want = 0.2 * (1.0 + (3.0 * phi - std::f64::consts::PI).cos());
        let got = pots.energy_at(&quad(phi));
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    fn single(k: F, n: F, d_deg: F) -> Potentials {
        ExplicitTerms::new("dihedral", "charmm")
            .term(
                &[0, 1, 2, 3],
                Params::from_iter([("k", k), ("periodicity", n), ("phase", d_deg)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn energy_phase() {
        // E = K[1 + cos(nφ − d)]. With n=1, d=0: E(0)=2K, E(π)=0.
        let e0 = single(1.5, 1.0, 0.0).energy_forces_at(&quad(0.0)).0;
        let epi = single(1.5, 1.0, 0.0)
            .energy_forces_at(&quad(std::f64::consts::PI))
            .0;
        assert!((e0 - 3.0).abs() < 1e-9, "E(0) got {e0}");
        assert!(epi.abs() < 1e-9, "E(pi) got {epi}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, 2.0, 30.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.energy_forces_at(&coords);
        let h = 1e-6;
        for dd in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[dd] += h;
            cm[dd] -= h;
            let ep = pot.energy_forces_at(&cp).0;
            let em = pot.energy_forces_at(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[dd] - fd).abs() < 1e-5,
                "comp {dd}: analytic {} vs fd {fd}",
                forces[dd]
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
