//! Periodic / Fourier proper dihedral (AMBER / GAFF; LAMMPS
//! `dihedral_style fourier`).

use crate::ff::potential::form_kernel::{ParamColumns, ScalarForm};
use molrs::op::F;

/// Periodic / Fourier proper dihedral (AMBER / GAFF):
///
/// E(φ) = Σ_m k_m · [1 + cos(n_m·φ − γ_m)]
///
/// AMBER-family torsions are a sum of cosine terms per quadruple, the
/// indexed family `k{m}`, `periodicity{m}`, `phase{m}` (1-indexed, the phase
/// in **degrees**, as LAMMPS `dihedral_style fourier` writes it). A single
/// unindexed `k`/`periodicity`/`phase` triple is accepted as the one-term
/// case (the common GAFF default), keeping the form identical to one CHARMM
/// term. Types of fewer terms than the table's longest hold their absent
/// terms at `k = 0`, which price nothing.
#[derive(Clone, Copy, Debug, Default)]
pub struct Periodic;

impl ScalarForm for Periodic {
    fn eval(&self, phi: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        let terms: Vec<[&[F]; 3]> = (1..)
            .map_while(|m| {
                Some([
                    p.get(&format!("k{m}"))?,
                    p.input(&format!("periodicity{m}")),
                    p.input(&format!("phase{m}")),
                ])
            })
            .collect();
        for t in 0..phi.len() {
            let (mut energy, mut de_dphi): (F, F) = (0.0, 0.0);
            for [k, n, d] in &terms {
                let arg = n[t] * phi[t] - d[t].to_radians();
                energy += k[t] * (1.0 + arg.cos());
                de_dphi += -k[t] * n[t] * arg.sin();
            }
            e[t] = energy;
            de[t] = de_dphi;
        }
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

    fn single(terms: Vec<(F, F, F)>) -> Potentials {
        let mut row = Params::new();
        for (m, (k, n, d)) in terms.into_iter().enumerate() {
            row.set(&format!("k{}", m + 1), k);
            row.set(&format!("periodicity{}", m + 1), n);
            row.set(&format!("phase{}", m + 1), d);
        }
        ExplicitTerms::new("dihedral", "periodic")
            .term(&[0, 1, 2, 3], row)
            .compile()
            .unwrap()
    }

    /// `(k, n, phase in degrees)`.
    fn term(k: F, n: F, d_deg: F) -> (F, F, F) {
        (k, n, d_deg)
    }

    #[test]
    fn single_term_energy() {
        // K[1+cos(nφ−d)], n=1,d=0: E(0)=2K, E(π)=0.
        let p = single(vec![term(1.5, 1.0, 0.0)]);
        assert!((p.energy_forces_at(&quad(0.0)).0 - 3.0).abs() < 1e-9);
        assert!(p.energy_forces_at(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn multi_term_energy_sums() {
        // Two terms add: at φ=0, E = K1[1+cos(−d1)] + K2[1+cos(−d2)].
        let t = vec![term(1.0, 1.0, 0.0), term(0.5, 2.0, 180.0)];
        let e = single(t).energy_forces_at(&quad(0.0)).0;
        // K1[1+1] + K2[1+cos(180°)] = 2.0 + 0.5*0 = 2.0
        assert!((e - 2.0).abs() < 1e-9, "got {e}");
    }

    /// The indexed `k<m>`/`periodicity<m>`/`phase<m>` terms, or one term
    /// spelled bare; both spellings at once is refused, and so is a gap.
    #[test]
    fn indexed_and_single_spellings() {
        let mut p = Params::new();
        p.set("k1", 1.0);
        p.set("periodicity1", 1.0);
        p.set("phase1", 0.0);
        p.set("k2", 0.5);
        p.set("periodicity2", 2.0);
        p.set("phase2", 180.0);
        let indexed = ExplicitTerms::new("dihedral", "periodic")
            .term(&[0, 1, 2, 3], p.clone())
            .compile()
            .unwrap();
        // K1[1+1] + K2[1+cos(180°)] = 2.0 + 0.5*0 = 2.0
        assert!((indexed.energy_at(&quad(0.0)) - 2.0).abs() < 1e-9);

        let mut q = Params::new();
        q.set("k", 2.0);
        q.set("periodicity", 3.0);
        q.set("phase", 0.0);
        let bare = ExplicitTerms::new("dihedral", "periodic")
            .term(&[0, 1, 2, 3], q.clone())
            .compile()
            .unwrap();
        let phi = 0.3;
        let want = 2.0 * (1.0 + (3.0 * phi as F).cos());
        assert!((bare.energy_at(&quad(phi)) - want).abs() < 1e-12);

        let refusal = |row: Params| {
            ExplicitTerms::new("dihedral", "periodic")
                .term(&[0, 1, 2, 3], row)
                .compile()
                .map(|_| ())
                .unwrap_err()
        };
        let mut both = p.clone();
        both.set("k", 2.0);
        let err = refusal(both);
        assert!(
            matches!(err.ir(), Some(crate::ff::ir::IrError::BadValue { param, .. }) if param == "k"),
            "{err}"
        );
        let mut gap = Params::new();
        gap.set("k1", 1.0);
        gap.set("periodicity1", 1.0);
        gap.set("k3", 1.0);
        gap.set("periodicity3", 1.0);
        let err = refusal(gap);
        assert!(
            matches!(err.ir(), Some(crate::ff::ir::IrError::MissingParam { param, .. }) if param == "k2"),
            "{err}"
        );
    }

    /// Types of different term counts share one table: a type's absent terms
    /// price nothing.
    #[test]
    fn types_of_different_term_counts_price_their_own_terms() {
        let mut ff = crate::ff::forcefield::ForceField::new("t");
        let style = ff.def_style("dihedral", "periodic", Params::new()).unwrap();
        style
            .def_type(
                "one",
                &["a", "b", "c", "d"],
                Params::from_iter([("k1", 1.3), ("periodicity1", 1.0), ("phase1", 0.0)]),
            )
            .unwrap()
            .def_type(
                "three",
                &["a", "b", "c", "e"],
                Params::from_iter([
                    ("k1", 0.4),
                    ("periodicity1", 1.0),
                    ("k2", -0.7),
                    ("periodicity2", 2.0),
                    ("phase2", 180.0),
                    ("k3", 0.2),
                    ("periodicity3", 3.0),
                ]),
            )
            .unwrap();
        let mut block = molrs::core::Block::new();
        for (key, col) in [
            ("atomi", [0, 0]),
            ("atomj", [1, 1]),
            ("atomk", [2, 2]),
            ("atoml", [3, 3]),
        ] {
            block
                .insert(
                    key,
                    ndarray::Array1::from_vec(col.map(|a| a as molrs::op::Idx).to_vec()).into_dyn(),
                    None,
                )
                .unwrap();
        }
        block
            .insert(
                "type",
                ndarray::Array1::from_vec(vec!["one".to_owned(), "three".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = molrs::core::Frame::new();
        frame.insert("dihedrals", block);
        let pots = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap();
        let phi: F = 0.7;
        let want = 1.3 * (1.0 + phi.cos()) + 0.4 * (1.0 + phi.cos())
            - 0.7 * (1.0 + (2.0 * phi - std::f64::consts::PI).cos())
            + 0.2 * (1.0 + (3.0 * phi).cos());
        let got = pots.energy_at(&quad(phi));
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn numerical_gradient_multiterm() {
        let pot = single(vec![
            term(1.3, 1.0, 0.0),
            term(-0.7, 2.0, 180.0),
            term(0.4, 3.0, 0.0),
        ]);
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
        let pot = single(vec![term(1.0, 1.0, 0.0), term(0.5, 2.0, 90.0)]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
