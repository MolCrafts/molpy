//! Polynomial-in-cos φ proper dihedrals (LAMMPS `multi/harmonic`, `nharmonic`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::DIHEDRALS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::ir::torsion::nharmonic_coefficients;
use crate::ff::potential::flat_coords::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Multi/harmonic (or nharmonic) proper dihedral with pre-resolved flat arrays.
///
/// Polynomial-in-cos φ proper dihedrals: LAMMPS `dihedral_style
/// multi/harmonic` and `dihedral_style nharmonic`,
///
/// E(φ) = Σ_{i=1..N} A_i · cos^(i−1)(φ)
///
/// `multi/harmonic` is the N = 5 case (`a1..a5`, an absent one 0);
/// `nharmonic` takes any N ≥ 1 (`a1..aN`, contiguous — LAMMPS's
/// `dihedral_coeff t N A1 … AN`). The coefficients are energies. One kernel
/// prices both.
pub struct DihedralMultiHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    /// A₁..A_N per dihedral instance.
    a: Vec<Vec<F>>,
}

impl DihedralMultiHarmonic {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
            let phi = compute_dihedral(coords, i, j, k, l);
            let c = phi.cos();
            let a = &self.a[idx];

            // E = Σ A_i c^(i−1), dE/dc = Σ (i−1) A_i c^(i−2)  (Horner, from the top)
            let (mut e, mut de_dc): (F, F) = (0.0, 0.0);
            for (p, &ai) in a.iter().enumerate().rev() {
                e = ai + c * e;
                if p > 0 {
                    de_dc = p as F * ai + c * de_dc;
                }
            }
            energy += e;
            // dE/dφ = dE/dc · (−sinφ)
            let de_dphi = -phi.sin() * de_dc;
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralMultiHarmonic {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| {
                (
                    self.atom_i[t],
                    self.atom_j[t],
                    self.atom_k[t],
                    self.atom_l[t],
                )
            },
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for DihedralMultiHarmonic {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k, &self.atom_l])
    }
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
                terms[[t, 3]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Construct a [`DihedralMultiHarmonic`] from per-type params (`a1`..`a5`,
/// an absent one 0) and a Frame's `"dihedrals"` block
/// (`atomi/atomj/atomk/atoml/type`).
pub fn dihedral_multi_harmonic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    cos_polynomial_constructor("dihedral_multi_harmonic", tp, frame, |label, p| {
        ["a1", "a2", "a3", "a4", "a5"]
            .iter()
            .map(|key| param_reads::type_num("multi/harmonic", label, p, key))
            .collect()
    })
}

/// Construct the LAMMPS `dihedral_style nharmonic` kernel from per-type
/// params `a1..aN` (contiguous, N ≥ 1) and a Frame's `"dihedrals"` block.
pub fn dihedral_nharmonic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    cos_polynomial_constructor("dihedral_nharmonic", tp, frame, |label, p| {
        // The first coefficient absent: `a1`, or the one a gap skips.
        nharmonic_coefficients(p).map_err(|_| {
            let m = (1..)
                .take_while(|m| p.get(&format!("a{m}")).and_then(|v| v.as_num()).is_some())
                .count();
            param_reads::missing("nharmonic", label, &format!("a{}", m + 1))
        })
    })
}

/// The one constructor of both styles; `coefficients` reads a type's `A_i`.
fn cos_polynomial_constructor(
    what: &str,
    tp: &[(&str, &Params)],
    frame: &Frame,
    coefficients: impl Fn(&str, &Params) -> Result<Vec<F>, crate::ff::ir::IrError>,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(DIHEDRALS)
        .ok_or_else(|| format!("{what}: missing \"dihedrals\" block"))?;
    let ic = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomi")?;
    let jc = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomj")?;
    let kc = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomk")?;
    let lc = block
        .get("atoml")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atoml")?;
    let tc = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut a) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("{what}: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        a.push(coefficients(&tc[idx], p)?);
    }
    Ok(ForceTerm::indexed(DihedralMultiHarmonic {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        a,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(a: [F; 5]) -> DihedralMultiHarmonic {
        DihedralMultiHarmonic {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            a: vec![a.to_vec()],
        }
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
