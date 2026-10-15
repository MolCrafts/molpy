//! MMFF94 out-of-plane bending (Wilson angle).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::IMPROPERS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::{cross, dot, norm};
use molrs::core::Frame;
use molrs::op::F;

use crate::core::constants::MMFF_MDYNE_A_TO_KCAL_MOL;

/// MMFF94 out-of-plane bending: E = 0.5*143.9325*koop*chi^2 (Wilson angle)
///
/// No LAMMPS style has this name. Its atom order is that of LAMMPS's
/// out-of-plane styles (`improper_style fourier`, `umbrella`): the centre is
/// **first** (`atomi`), and χ is the angle between the bond centre→`atoml` and
/// the plane (centre, `atomj`, `atomk`) — MMFF's `I J K L` with `J` central,
/// read centre first.
pub struct ImproperMmff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    koop: Vec<F>,
}

impl ImproperMmff {
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
        let conv = MMFF_MDYNE_A_TO_KCAL_MOL as F;

        for idx in 0..n_terms {
            // Stored centre first; the Wilson math below names the centre `j`.
            let (j, i, k, l) = atoms(idx);
            let a = sub3(coords, i, coords, j); // ji
            let b = sub3(coords, k, coords, j); // jk
            let c = sub3(coords, l, coords, j); // jl
            let n = cross(a, b);
            let n_mag = norm(n);
            let c_mag = norm(c);
            if n_mag < 1e-12 as F || c_mag < 1e-12 as F {
                continue;
            }

            let sin_chi = (dot(c, n) / (c_mag * n_mag)).clamp(-1.0, 1.0);
            let chi = sin_chi.asin();
            let cos_chi = chi.cos();
            energy += 0.5 * conv * self.koop[idx] * chi * chi;

            if cos_chi.abs() < 1e-12 as F {
                continue;
            }
            let prefactor = conv * self.koop[idx] * chi / cos_chi;

            // d(sin chi)/d(r_l)
            let inv_cn = 1.0 / (c_mag * n_mag);
            for dim in 0..3 {
                let dsdc = n[dim] * inv_cn - sin_chi * c[dim] / (c_mag * c_mag);
                forces[l * 3 + dim] += -prefactor * dsdc;
                forces[j * 3 + dim] -= -prefactor * dsdc;
            }
            // d(sin chi)/d(r_i)  (a = r_i - r_j, affects n = a x b)
            let bxc = cross(b, c);
            let bxn = cross(b, n);
            for dim in 0..3 {
                let dsda = bxc[dim] * inv_cn - sin_chi * bxn[dim] / (n_mag * n_mag);
                forces[i * 3 + dim] += -prefactor * dsda;
                forces[j * 3 + dim] -= -prefactor * dsda;
            }
            // d(sin chi)/d(r_k)  (b = r_k - r_j, affects n = a x b)
            let axc = cross(a, c);
            let axn = cross(a, n);
            for dim in 0..3 {
                let dsdb = -axc[dim] * inv_cn + sin_chi * axn[dim] / (n_mag * n_mag);
                forces[k * 3 + dim] += -prefactor * dsdb;
                forces[j * 3 + dim] -= -prefactor * dsdb;
            }
        }
        energy
    }
}

impl Potential for ImproperMmff {
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

impl IndexedTerms for ImproperMmff {
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

pub fn improper_mmff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    // Per-instance parameters: the MMFF typifier baked koop onto each improper.
    // This kernel only reads the column and evaluates.
    let block = frame
        .get(IMPROPERS)
        .ok_or("mmff_oop: missing \"impropers\"")?;
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
    let koopc = param_reads::instance_col("mmff_oop", block, "koop")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut koop) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        koop.push(koopc[idx] as F);
    }
    Ok(ForceTerm::indexed(ImproperMmff {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        koop,
    }))
}
