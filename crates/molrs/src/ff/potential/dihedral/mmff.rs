//! MMFF94 torsional rotation: E = 0.5*(V1*(1+cos phi) + V2*(1-cos 2phi) + V3*(1+cos 3phi))

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::DIHEDRALS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

pub struct DihedralMmff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    v1: Vec<F>,
    v2: Vec<F>,
    v3: Vec<F>,
}

impl DihedralMmff {
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

            energy += 0.5
                * (self.v1[idx] * (1.0 + c1)
                    + self.v2[idx] * (1.0 - c2)
                    + self.v3[idx] * (1.0 + c3));

            let de_dphi =
                0.5 * (-self.v1[idx] * s1 + 2.0 * self.v2[idx] * s2 - 3.0 * self.v3[idx] * s3);
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralMmff {
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

impl IndexedTerms for DihedralMmff {
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

pub fn dihedral_mmff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    // Per-instance parameters: the MMFF typifier baked v1/v2/v3 onto each
    // dihedral (table → empirical). This kernel only reads the columns and
    // evaluates — no force-field-specific resolution lives here.
    let block = frame
        .get(DIHEDRALS)
        .ok_or("mmff_torsion: missing \"dihedrals\"")?;
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
    let v1c = param_reads::instance_col("mmff_torsion", block, "v1")?;
    let v2c = param_reads::instance_col("mmff_torsion", block, "v2")?;
    let v3c = param_reads::instance_col("mmff_torsion", block, "v3")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut v1, mut v2, mut v3) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        v1.push(v1c[idx] as F);
        v2.push(v2c[idx] as F);
        v3.push(v3c[idx] as F);
    }
    Ok(ForceTerm::indexed(DihedralMmff {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        v1,
        v2,
        v3,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    #[test]
    fn test_mmff_torsion() {
        let pot = DihedralMmff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            v1: vec![0.0],
            v2: vec![0.0],
            v3: vec![0.3],
        };
        let coords: Vec<F> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e.is_finite());
        let fx: F = forces.iter().step_by(3).sum();
        let fy: F = forces.iter().skip(1).step_by(3).sum();
        let fz: F = forces.iter().skip(2).step_by(3).sum();
        assert!(
            (fx.abs() + fy.abs() + fz.abs()) < 0.1,
            "force sum too large"
        );
    }
}
