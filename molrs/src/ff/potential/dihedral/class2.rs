//! Class2 (COMPASS / PCFF) proper dihedral, core torsion term.

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

/// Class2 proper dihedral (core 3-term cosine) with pre-resolved flat arrays.
///
/// Class2 (COMPASS / PCFF) proper dihedral — core torsion term:
///
/// E(φ) = Σ_{n=1..3} K_n · [1 − cos(n·φ − φ_n)]
///
/// `K_n` are force constants (energy) and `φ_n` the per-term phases in
/// **degrees**, as LAMMPS `dihedral_style class2` takes them (the kernel
/// converts once). This kernel covers the *core*
/// three-term cosine expansion that the
/// molpy `class2` dihedral data model carries (`k1`,`phi1`,…,`k3`,`phi3`). The
/// optional class2 cross terms (mbt / ebt / at / aat / bb13), which are emitted
/// as separate LAMMPS coeff lines and not part of this style's per-type params,
/// are out of scope here.
pub struct DihedralClass2 {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    /// (K_n, φ_n in radians) for n = 1..3 per dihedral instance.
    terms: Vec<[(F, F); 3]>,
}

impl DihedralClass2 {
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
            let mut de_dphi: F = 0.0;
            for (m, &(kn, pn)) in self.terms[idx].iter().enumerate() {
                let n = (m + 1) as F;
                let arg = n * phi - pn;
                energy += kn * (1.0 - arg.cos());
                // dE/dφ = K_n·n·sin(n·φ − φ_n)
                de_dphi += kn * n * arg.sin();
            }
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralClass2 {
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

impl IndexedTerms for DihedralClass2 {
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

/// Construct a [`DihedralClass2`] from per-type params (`k1`,`phi1`,…,`k3`,
/// `phi3`, the phases in degrees as LAMMPS takes them) and a Frame's
/// `"dihedrals"` block.
pub fn dihedral_class2_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(DIHEDRALS)
        .ok_or("dihedral_class2: missing \"dihedrals\" block")?;
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
    let (mut ai, mut aj, mut ak, mut al, mut terms) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("dihedral_class2: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        let mut t = [(0.0 as F, 0.0 as F); 3];
        for (m, slot) in t.iter_mut().enumerate() {
            let label = tc[idx].as_str();
            let kn = param_reads::type_num("class2", label, p, &format!("k{}", m + 1))?;
            // degrees (LAMMPS `dihedral_style class2`) → radians
            let pn =
                param_reads::type_num("class2", label, p, &format!("phi{}", m + 1))?.to_radians();
            *slot = (kn, pn);
        }
        terms.push(t);
    }
    Ok(ForceTerm::indexed(DihedralClass2 {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        terms,
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

    fn single(terms: [(F, F); 3]) -> DihedralClass2 {
        DihedralClass2 {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            terms: vec![terms],
        }
    }

    #[test]
    fn energy_phase() {
        // Single term K1[1−cos(φ−φ1)], φ1=0: E(0)=0, E(π)=2K1.
        let t = [(1.5, 0.0), (0.0, 0.0), (0.0, 0.0)];
        assert!(single(t).energy_forces_at(&quad(0.0)).0.abs() < 1e-9);
        let epi = single(t).energy_forces_at(&quad(std::f64::consts::PI)).0;
        assert!((epi - 3.0).abs() < 1e-9, "E(pi) got {epi}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single([(1.3, 0.0), (-0.7, std::f64::consts::PI), (0.4, 0.0)]);
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
        let pot = single([(1.0, 0.0), (0.5, 1.0), (0.3, 0.5)]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
