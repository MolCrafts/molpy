//! Harmonic improper (LAMMPS `improper_style harmonic`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::IMPROPERS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Harmonic improper with pre-resolved flat arrays.
///
/// Harmonic improper (LAMMPS `improper_style harmonic`):
///
/// E(χ) = K · (χ − χ₀)²
///
/// Following LAMMPS, the improper angle χ is the **unsigned** dihedral of the
/// quadruple I-J-K-L: χ = |φ| ∈ [0, π], where φ = atan2(…) is the signed
/// dihedral. Hence dχ/dφ = sign(φ), so dE/dφ = 2K(χ − χ₀)·sign(φ), projected
/// onto Cartesian forces by the shared dihedral routine. `chi0` is the
/// equilibrium angle in **degrees** (0 for a planar centre), as LAMMPS takes
/// it; the kernel converts it to radians once. `k` is LAMMPS's `K`
/// (energy/rad², the ½ included).
///
/// # Atom order
///
/// LAMMPS's: the improper angle is the dihedral I-J-K-L of the stored order,
/// and the first atom I is the centre (LAMMPS's "atom of symmetry" for this
/// style; CHARMM writes its impropers this way).
pub struct ImproperHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    /// equilibrium angle in radians (the parameter is degrees)
    chi0: Vec<F>,
}

impl ImproperHarmonic {
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
            let chi = phi.abs();
            let (ki, c0) = (self.k[idx], self.chi0[idx]);
            let dchi = chi - c0;
            energy += ki * dchi * dchi;
            // dE/dφ = 2K(χ − χ₀)·sign(φ). sign(0) → 0 (stationary, planar).
            let sign = if phi > 0.0 {
                1.0
            } else if phi < 0.0 {
                -1.0
            } else {
                0.0
            };
            let de_dphi = 2.0 * ki * dchi * sign;
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for ImproperHarmonic {
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

impl IndexedTerms for ImproperHarmonic {
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

/// Construct an [`ImproperHarmonic`] from per-type params (`k`, `chi0` degrees)
/// and a Frame's `"impropers"` block (`atomi/atomj/atomk/atoml/type`).
pub fn improper_harmonic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(IMPROPERS)
        .ok_or("improper_harmonic: missing \"impropers\" block")?;
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
    let (mut kk, mut cc) = (Vec::with_capacity(n), Vec::with_capacity(n));

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("improper_harmonic: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        kk.push(param_reads::type_num("harmonic", &tc[idx], p, "k")?);
        // degrees → radians
        cc.push(param_reads::type_num("harmonic", &tc[idx], p, "chi0")?.to_radians());
    }
    Ok(ForceTerm::indexed(ImproperHarmonic {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        k: kk,
        chi0: cc,
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

    fn single(k: F, chi0_deg: F) -> ImproperHarmonic {
        ImproperHarmonic {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![k],
            chi0: vec![chi0_deg.to_radians()],
        }
    }

    #[test]
    fn energy_minimum_at_chi0() {
        // χ₀ = 60°, geometry at φ = 60° → E = 0.
        let p = single(3.0, 60.0);
        let e = p.energy_forces_at(&quad(60.0_f64.to_radians())).0;
        assert!(e.abs() < 1e-9, "E at χ₀ got {e}");
        // φ = 90° → E = K(30°)² in radians.
        let e2 = p.energy_forces_at(&quad(90.0_f64.to_radians())).0;
        let want = 3.0 * (30.0_f64.to_radians()).powi(2);
        assert!((e2 - want).abs() < 1e-9, "E got {e2} want {want}");
    }

    #[test]
    fn numerical_gradient() {
        // χ₀ chosen so φ stays well away from 0 and ±π under perturbation.
        let pot = single(5.0, 20.0);
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
        let pot = single(5.0, 20.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
