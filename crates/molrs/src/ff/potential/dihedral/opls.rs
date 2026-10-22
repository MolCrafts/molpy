//! OPLS 4-cosine (Fourier) proper dihedral.

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::DIHEDRALS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// OPLS 4-cosine proper dihedral with pre-resolved flat arrays.
///
/// OPLS 4-cosine (Fourier) proper dihedral:
///
/// E(φ) = ½[ F1(1+cos φ) + F2(1−cos 2φ) + F3(1+cos 3φ) + F4(1−cos 4φ) ]
///
/// This is the OPLS-AA torsion form (Jorgensen, Maxwell & Tirado-Rives,
/// J. Am. Chem. Soc. 1996, 118, 11225). Coefficients F1..F4 are in kcal/mol.
/// The kernel is topology-blind: it consumes pre-resolved dihedral quadruples
/// and coefficients, mirroring the MMFF torsion kernel.
pub struct DihedralOpls {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    f1: Vec<F>,
    f2: Vec<F>,
    f3: Vec<F>,
    f4: Vec<F>,
}

impl DihedralOpls {
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

            let (s1, c1) = phi.sin_cos();
            let (s2, c2) = (2.0 * phi).sin_cos();
            let (s3, c3) = (3.0 * phi).sin_cos();
            let (s4, c4) = (4.0 * phi).sin_cos();

            let (f1, f2, f3, f4) = (self.f1[idx], self.f2[idx], self.f3[idx], self.f4[idx]);

            energy += 0.5 * (f1 * (1.0 + c1) + f2 * (1.0 - c2) + f3 * (1.0 + c3) + f4 * (1.0 - c4));

            // dE/dφ = ½[ −F1 sinφ + 2F2 sin2φ − 3F3 sin3φ + 4F4 sin4φ ]
            let de_dphi = 0.5 * (-f1 * s1 + 2.0 * f2 * s2 - 3.0 * f3 * s3 + 4.0 * f4 * s4);
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralOpls {
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

impl IndexedTerms for DihedralOpls {
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

/// Construct a [`DihedralOpls`] from style params, per-type params (F1..F4),
/// and a Frame's `"dihedrals"` block (`atomi/atomj/atomk/atoml/type`).
pub fn dihedral_opls_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(DIHEDRALS)
        .ok_or("dihedral_opls: missing \"dihedrals\" block")?;
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
    let (mut ai, mut aj, mut ak, mut al) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut f1, mut f2, mut f3, mut f4) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("dihedral_opls: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        // Every coefficient is required, as LAMMPS's `dihedral_coeff` requires
        // all four: reading a missing one as 0, or a mis-spelled bag
        // (`c1..c4`), would make a whole torsion vanish in silence.
        // `Params` is flat scalars, so the multi-term `periodic` style spells
        // its terms `k{m}`/`periodicity{m}`/`phase{m}` — the same `k1..k4` keys
        // this style uses for the OPLS quartet, with a different meaning (LAMMPS
        // `K_n` already carries the 1/2). A bag that also names
        // `periodicity1`/`phase1` is a periodic bag on the wrong style, and
        // reading it here would price a plain barrier as a half barrier, silently.
        let label = tc[idx].as_str();
        if let Some(key) = ["periodicity1", "phase1"]
            .into_iter()
            .find(|key| p.get(key).and_then(|v| v.as_num()).is_some())
        {
            return Err(param_reads::bad(
                "opls",
                label,
                key,
                "is no `dihedral opls` parameter: the row is a `dihedral periodic` \
                 term table; declare the periodic style for it",
            )
            .into());
        }
        let need = |key: &str| param_reads::type_num("opls", label, p, key);
        f1.push(need("k1")?);
        f2.push(need("k2")?);
        f3.push(need("k3")?);
        f4.push(need("k4")?);
    }
    Ok(ForceTerm::indexed(DihedralOpls {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        f1,
        f2,
        f3,
        f4,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// Build a 4-atom geometry at a target dihedral φ in the xy/z layout used
    /// by the MMFF torsion test, then read it back.
    fn quad(phi: F) -> Vec<F> {
        // i=(0,1,0) j=(0,0,0) k=(1,0,0); l rotates around the j-k (x) axis.
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(f1: F, f2: F, f3: F, f4: F) -> DihedralOpls {
        DihedralOpls {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            f1: vec![f1],
            f2: vec![f2],
            f3: vec![f3],
            f4: vec![f4],
        }
    }

    #[test]
    fn per_term_phase() {
        // φ at the reference geometry: l=(1, cos φ, sin φ) gives dihedral = φ.
        let at = |c: &[F]| compute_dihedral(c, 0, 1, 2, 3);
        // F1 term: E(0)=F1, E(π)=0.
        let e0 = single(2.0, 0.0, 0.0, 0.0).energy_forces_at(&quad(0.0)).0;
        let epi = single(2.0, 0.0, 0.0, 0.0)
            .energy_forces_at(&quad(std::f64::consts::PI))
            .0;
        assert!((e0 - 2.0).abs() < 1e-9, "F1 E(0) got {e0}");
        assert!(epi.abs() < 1e-9, "F1 E(pi) got {epi}");
        // F2 term: E(0)=0, E(π/2)=F2.
        let c = quad(std::f64::consts::FRAC_PI_2);
        assert!((at(&c) - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
        let e = single(0.0, 3.0, 0.0, 0.0).energy_forces_at(&c).0;
        assert!((e - 3.0).abs() < 1e-9, "F2 E(pi/2) got {e}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(1.3, -0.7, 0.9, 0.4);
        // A generic, non-degenerate geometry.
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
            let fd = -(ep - em) / (2.0 * h); // force = -dE/dx
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(1.0, 0.5, 0.3, 0.2);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
