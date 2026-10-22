//! CHARMM proper dihedral (LAMMPS `dihedral_style charmm`).

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

/// CHARMM proper dihedral with pre-resolved flat arrays.
///
/// CHARMM proper dihedral (LAMMPS `dihedral_style charmm`):
///
/// E(φ) = k·[1 + cos(n·φ − d)]
///
/// `k` is LAMMPS's `K` (energy), `periodicity` its integer `n`, and `phase` its
/// `d` in **degrees** (LAMMPS takes an integer number of degrees; any value is
/// accepted here). The kernel converts the phase to radians once.
///
/// # The weight `w`
///
/// LAMMPS's fourth coefficient `w` weights a 1-4 non-bonded pair that the
/// *dihedral* computes — its end atoms, with the `epsilon14` / `sigma14` of
/// the `lj/charmm` pair style and the full Coulomb, beside `special_bonds`
/// 1-4 weights of zero. This kernel prices the torsion alone; the compiler
/// routes every dihedral's `w` pair to the 1-4 exceptions kernel
/// ([`PairExceptions`](crate::ff::potential::pair::PairExceptions)), which also makes LAMMPS's
/// checks (`0 ≤ w ≤ 1`, `special_bonds` 1-4 = 0, a `lj/charmm` pair style).
/// `w = 0` (or absent) is the AMBER use of the style.
pub struct DihedralCharmm {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    n: Vec<F>,
    /// phase in radians (the parameter is degrees)
    d: Vec<F>,
}

impl DihedralCharmm {
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
            // dE/dφ = −K·n·sin(n·φ − d)
            let de_dphi = -ki * ni * arg.sin();
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralCharmm {
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

impl IndexedTerms for DihedralCharmm {
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

/// Construct a [`DihedralCharmm`] from per-type params (`k`, `periodicity`,
/// `phase` in degrees) and a Frame's `"dihedrals"` block
/// (`atomi/atomj/atomk/atoml/type`). `w` is the compiler's (see
/// [`DihedralCharmm`]).
pub fn dihedral_charmm_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(DIHEDRALS)
        .ok_or("dihedral_charmm: missing \"dihedrals\" block")?;
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
            .ok_or_else(|| format!("dihedral_charmm: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        let label = tc[idx].as_str();
        kk.push(param_reads::type_num("charmm", label, p, "k")?);
        nn.push(param_reads::type_num("charmm", label, p, "periodicity")?);
        // degrees → radians
        dd.push(param_reads::type_num("charmm", label, p, "phase")?.to_radians());
    }
    Ok(ForceTerm::indexed(DihedralCharmm {
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

    fn single(k: F, n: F, d_deg: F) -> DihedralCharmm {
        DihedralCharmm {
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
