//! CVFF improper (LAMMPS `improper_style cvff`).

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

/// CVFF improper with pre-resolved flat arrays.
///
/// CVFF improper (LAMMPS `improper_style cvff`):
///
/// E(χ) = k · [1 + s · cos(n·χ)]
///
/// `sign` is s = ±1 (a sign, **not** a phase — hence its own canonical name)
/// and `periodicity` is the integer multiplicity n. The improper angle χ is the
/// dihedral angle defined by the quadruple I-J-K-L of the stored order, I the
/// centre (LAMMPS's symmetry atom for this style), so the geometry reuses the
/// shared dihedral routines.
///
/// The same function of the dihedral is LAMMPS's `dihedral_style harmonic`
/// (`K[1 + d cos(nφ)]`); one shared constructor builds either from its block.
pub struct ImproperCvff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    d: Vec<F>,
    n: Vec<F>,
}

impl ImproperCvff {
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
            let chi = compute_dihedral(coords, i, j, k, l);
            let (ki, di, ni) = (self.k[idx], self.d[idx], self.n[idx]);
            energy += ki * (1.0 + di * (ni * chi).cos());
            // dE/dχ = −K·d·n·sin(n·χ)
            let de_dchi = -ki * di * ni * (ni * chi).sin();
            accumulate_dihedral_forces(coords, i, j, k, l, de_dchi, forces);
        }
        energy
    }
}

impl Potential for ImproperCvff {
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

impl IndexedTerms for ImproperCvff {
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

/// Construct an [`ImproperCvff`] from per-type params (`k`, `sign`,
/// `periodicity`) and a Frame's `"impropers"` block
/// (`atomi/atomj/atomk/atoml/type`).
pub fn improper_cvff_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    signed_cosine_constructor(IMPROPERS, "cvff", tp, frame)
}

/// `k·[1 + sign·cos(periodicity·φ)]` over the quadruples of `block_name`
/// (`"impropers"` for `improper cvff`, `"dihedrals"` for `dihedral
/// harmonic`); `style` is the style's name (`cvff`, `harmonic`).
pub fn signed_cosine_constructor(
    block_name: &str,
    style: &str,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(block_name)
        .ok_or_else(|| format!("{style}: missing \"{block_name}\" block"))?;
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
    let (mut kk, mut dd, mut nn) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("{block_name} {style}: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        let need = |key: &str| param_reads::type_num(style, &tc[idx], p, key);
        kk.push(need("k")?);
        dd.push(need("sign")?);
        nn.push(need("periodicity")?);
    }
    Ok(ForceTerm::indexed(ImproperCvff {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        k: kk,
        d: dd,
        n: nn,
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

    fn single(k: F, d: F, n: F) -> ImproperCvff {
        ImproperCvff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![k],
            d: vec![d],
            n: vec![n],
        }
    }

    #[test]
    fn energy_phase() {
        // K[1 + d cos(nχ)], k=2,d=1,n=1: E(0)=4, E(π)=0.
        let p = single(2.0, 1.0, 1.0);
        assert!((p.energy_forces_at(&quad(0.0)).0 - 4.0).abs() < 1e-9);
        assert!(p.energy_forces_at(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, -1.0, 2.0);
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
        let pot = single(1.0, 1.0, 3.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
