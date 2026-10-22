//! Harmonic angle (LAMMPS `angle_style harmonic`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::ANGLES;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{compute_angle, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// Harmonic angle potential with pre-resolved flat arrays. Its own `theta0`
/// array is in radians (the parameter is degrees; see [`angle_harmonic_constructor`]).
///
/// LAMMPS `angle_style harmonic`: E = k·(θ − θ0)².
///
/// `k` is LAMMPS's `K` (energy/rad², the ½ included) and `theta0` is in
/// **degrees**, as in an `angle_coeff t K theta0` line; the kernel converts it
/// to radians once, at construction.
pub struct AngleHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    k: Vec<F>,
    theta0: Vec<F>,
}

impl AngleHarmonic {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        k: Vec<F>,
        theta0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(atom_k.len(), n);
        assert_eq!(k.len(), n);
        assert_eq!(theta0.len(), n);
        Self {
            atom_i,
            atom_j,
            atom_k,
            k,
            theta0,
        }
    }
}

impl AngleHarmonic {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize),
    ) -> F {
        let _n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let k_spring = self.k[idx];
            let theta0 = self.theta0[idx];

            let theta = compute_angle(coords, i, j, k);
            let dtheta = theta - theta0;
            energy += k_spring * dtheta * dtheta;

            // dE/dtheta = 2k (theta - theta0)
            super::accumulate_angle_forces(coords, i, j, k, 2.0 * k_spring * dtheta, forces);
        }

        energy
    }
}

impl Potential for AngleHarmonic {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t], self.atom_k[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for AngleHarmonic {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k])
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
            )
        });
        acc.add_energy(energy);
    }
}

/// Construct an [`AngleHarmonic`] from style params, type params, and Frame topology.
pub fn angle_harmonic_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get(ANGLES)
        .ok_or_else(|| "AngleHarmonic: frame missing \"angles\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomj\" column".to_string())?;
    let k_col = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomk\" column".to_string())?;
    let type_col = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "AngleHarmonic: angles block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut atom_k = Vec::with_capacity(i_col.len());
    let mut k_vec = Vec::with_capacity(i_col.len());
    let mut theta0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("AngleHarmonic: unknown angle type '{}'", label))?;
        // `k` is LAMMPS's `K`: E = k(θ − θ0)², no ½.
        let k = param_reads::type_num("harmonic", label, params, "k")?;
        // theta0 is a parameter in degrees (LAMMPS); the kernel works in radians.
        let theta0_rad = param_reads::type_num("harmonic", label, params, "theta0")?.to_radians();

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        atom_k.push(k_col[idx] as usize);
        k_vec.push(k);
        theta0_vec.push(theta0_rad);
    }

    Ok(ForceTerm::indexed(AngleHarmonic::new(
        atom_i, atom_j, atom_k, k_vec, theta0_vec,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// Atoms at (1,0,0), (0,0,0), (0,1,0): one 90° angle of type `label`.
    fn right_angle_frame(label: &str) -> Frame {
        use molrs::core::Block;
        use molrs::op::Idx;
        use ndarray::Array1;
        let mut atoms = Block::new();
        for (key, v) in [
            ("x", [1.0, 0.0, 0.0]),
            ("y", [0.0, 0.0, 1.0]),
            ("z", [0.0; 3]),
        ] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut angles = Block::new();
        for (key, a) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            angles
                .insert(key, Array1::from_vec(vec![a as Idx]).into_dyn(), None)
                .unwrap();
        }
        angles
            .insert(
                "type",
                Array1::from_vec(vec![label.to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", angles);
        frame
    }

    /// LAMMPS `angle_style harmonic`: E = K(θ − θ0)², θ0 given in degrees.
    #[test]
    fn energy_is_the_lammps_formula_with_theta0_in_degrees() {
        let mut ff = crate::ff::forcefield::ForceField::new("t");
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A",
                &["A", "A", "A"],
                Params::from_iter([("k", 50.0), ("theta0", 100.0)]),
            )
            .unwrap();
        let frame = right_angle_frame("A-A-A");
        let pots = crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap();
        // The frame's angle is 90 degrees.
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let want = 50.0 * (90.0_f64.to_radians() - 100.0_f64.to_radians()).powi(2);
        let got = pots.energy_at(&coords);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }

    #[test]
    fn test_angle_harmonic_energy() {
        let theta0: F = std::f64::consts::FRAC_PI_2 as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![50.0], vec![theta0]);
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let (e, _) = pot.energy_forces_at(&coords);
        assert!(e.abs() < 1e-4);
    }
}
