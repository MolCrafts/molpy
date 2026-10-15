//! Periodic improper (AMBER / GAFF / OpenMM impropers).

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

/// Periodic improper with pre-resolved flat arrays.
///
/// Periodic improper (AMBER / GAFF / OpenMM impropers):
///
/// E(φ) = k · [1 + cos(n·φ − γ)]
///
/// `k` is the force constant (energy), `periodicity` the multiplicity, and
/// `phase` the phase γ in **degrees**; the kernel converts it to radians once.
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
pub struct ImproperPeriodic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    n: Vec<F>,
    /// phase in radians (the parameter is degrees)
    d: Vec<F>,
}

impl ImproperPeriodic {
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
            let (ki, ni, di) = (self.k[idx], self.n[idx], self.d[idx]);
            let arg = ni * phi - di;
            energy += ki * (1.0 + arg.cos());
            let de_dphi = -ki * ni * arg.sin();
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for ImproperPeriodic {
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

impl IndexedTerms for ImproperPeriodic {
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

/// Construct an [`ImproperPeriodic`] from per-type params (`k`,
/// `periodicity`, `phase` in degrees) and a Frame's `"impropers"` block
/// (`atomi/atomj/atomk/atoml/type`, in AMBER order — see [`ImproperPeriodic`]).
pub fn improper_periodic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(IMPROPERS)
        .ok_or("improper_periodic: missing \"impropers\" block")?;
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
    let (mut kk, mut nn, mut dd) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("improper_periodic: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        kk.push(param_reads::type_num("periodic", &tc[idx], p, "k")?);
        nn.push(param_reads::type_num(
            "periodic",
            &tc[idx],
            p,
            "periodicity",
        )?);
        // degrees → radians
        dd.push(param_reads::type_num("periodic", &tc[idx], p, "phase")?.to_radians());
    }
    Ok(ForceTerm::indexed(ImproperPeriodic {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        k: kk,
        n: nn,
        d: dd,
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

    fn single(k: F, n: F, d_deg: F) -> ImproperPeriodic {
        ImproperPeriodic {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![k],
            n: vec![n],
            d: vec![d_deg.to_radians()],
        }
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
