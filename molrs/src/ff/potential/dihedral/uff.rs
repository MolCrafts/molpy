//! UFF torsion: E = V/2 · (1 − cosTerm · cos(n·φ)) (RDKit `TorsionAngleContrib`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::DIHEDRALS;
use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::{cross, dot, norm};
use molrs::core::Frame;
use molrs::op::F;

pub struct DihedralUff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    v: Vec<F>,
    order: Vec<u8>,
    cos_term: Vec<F>,
}

impl DihedralUff {
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
        let mut energy = 0.0 as F;
        let forces = out;

        for idx in 0..n_terms {
            let (a, b, c, d) = atoms(idx);
            // RDKit: r1=p1-p2, r2=p3-p2, r3=p2-p3, r4=p4-p3
            let r1 = sub3(coords, a, coords, b);
            let r2 = sub3(coords, c, coords, b);
            let r3 = sub3(coords, b, coords, c);
            let r4 = sub3(coords, d, coords, c);
            let t1 = cross(r1, r2);
            let t2 = cross(r3, r4);
            let d1 = norm(t1);
            let d2 = norm(t2);
            if d1 < 1e-12 as F || d2 < 1e-12 as F {
                continue;
            }
            let cos_phi = (dot(t1, t2) / (d1 * d2)).clamp(-1.0, 1.0);
            let sin_sq = (1.0 - cos_phi * cos_phi).max(0.0);
            let sin_phi = sin_sq.sqrt();

            let order = self.order[idx];
            let cos_n = match order {
                2 => 1.0 - 2.0 * sin_sq,
                3 => cos_phi * (cos_phi * cos_phi - 3.0 * sin_sq),
                6 => 1.0 + sin_sq * (-32.0 * sin_sq * sin_sq + 48.0 * sin_sq - 18.0),
                _ => cos_phi,
            };
            let v = self.v[idx];
            let cos_term = self.cos_term[idx];
            energy += v / 2.0 * (1.0 - cos_term * cos_n);

            // RDKit `getThetaDeriv`: dE/dφ with E = V/2 · (1 − cosTerm · cos(nφ)).
            let mut d_e_d_phi = match order {
                2 => 2.0 * sin_phi * cos_phi,
                3 => sin_phi * (3.0 - 4.0 * sin_phi * sin_phi),
                6 => cos_phi * sin_phi * (32.0 * sin_sq * (sin_sq - 1.0) + 6.0),
                _ => 0.0,
            };
            // E = (V/2)(1 − cosTerm·cos nφ)  ⇒  dE/dφ = +(V/2)·cosTerm·n·sin nφ.
            d_e_d_phi *= (v / 2.0 * cos_term) * (order as F);
            let sin_term = if sin_phi.abs() < 1e-8 {
                d_e_d_phi / cos_phi.signum().max(1e-8).abs().copysign(cos_phi).max(1e-8)
            } else {
                d_e_d_phi / sin_phi
            };

            // Project using RDKit calcTorsionGrad (simplified via force on planes)
            // Use standard dihedral force accumulation.
            accumulate_torsion_forces(
                coords, a, b, c, d, sin_term, cos_phi, &t1, &t2, d1, d2, &r1, &r2, &r3, &r4, forces,
            );
        }
        energy
    }
}

impl Potential for DihedralUff {
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

impl IndexedTerms for DihedralUff {
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

/// RDKit-style torsion gradient (adapted from `calcTorsionGrad`).
#[allow(clippy::too_many_arguments)]
fn accumulate_torsion_forces(
    _coords: &[F],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    sin_term: F,
    cos_phi: F,
    t1: &[F; 3],
    t2: &[F; 3],
    d1: F,
    d2: F,
    r1: &[F; 3],
    r2: &[F; 3],
    r3: &[F; 3],
    r4: &[F; 3],
    forces: &mut [F],
) {
    // r vectors as RDKit: r[0]=r1, r[1]=r2, r[2]=r3, r[3]=r4
    // t[0]=t1, t[1]=t2, d[0]=d1, d[1]=d2
    // cosφ = (t₁·t₂)/(|t₁||t₂|) ⇒ ∂cosφ/∂t₁ = (t̂₂ − cosφ·t̂₁)/|t₁|. Both terms
    // must be *unit* normals; feeding raw t scales them differently and the
    // expression is only the derivative when |t₁| = |t₂| = 1.
    let (u1, u2) = (
        [t1[0] / d1, t1[1] / d1, t1[2] / d1],
        [t2[0] / d2, t2[1] / d2, t2[2] / d2],
    );
    let d_cos_dt = [
        1.0 / d1 * (u2[0] - cos_phi * u1[0]),
        1.0 / d1 * (u2[1] - cos_phi * u1[1]),
        1.0 / d1 * (u2[2] - cos_phi * u1[2]),
        1.0 / d2 * (u1[0] - cos_phi * u2[0]),
        1.0 / d2 * (u1[1] - cos_phi * u2[1]),
        1.0 / d2 * (u1[2] - cos_phi * u2[2]),
    ];

    // g[0] (atom a)
    forces[a * 3] += sin_term * (d_cos_dt[2] * r2[1] - d_cos_dt[1] * r2[2]);
    forces[a * 3 + 1] += sin_term * (d_cos_dt[0] * r2[2] - d_cos_dt[2] * r2[0]);
    forces[a * 3 + 2] += sin_term * (d_cos_dt[1] * r2[0] - d_cos_dt[0] * r2[1]);

    // g[1] (atom b)
    forces[b * 3] += sin_term
        * (d_cos_dt[1] * (r2[2] - r1[2])
            + d_cos_dt[2] * (r1[1] - r2[1])
            + d_cos_dt[4] * (-r4[2])
            + d_cos_dt[5] * r4[1]);
    forces[b * 3 + 1] += sin_term
        * (d_cos_dt[0] * (r1[2] - r2[2])
            + d_cos_dt[2] * (r2[0] - r1[0])
            + d_cos_dt[3] * r4[2]
            + d_cos_dt[5] * (-r4[0]));
    forces[b * 3 + 2] += sin_term
        * (d_cos_dt[0] * (r2[1] - r1[1])
            + d_cos_dt[1] * (r1[0] - r2[0])
            + d_cos_dt[3] * (-r4[1])
            + d_cos_dt[4] * r4[0]);

    // g[2] (atom c)
    forces[c * 3] += sin_term
        * (d_cos_dt[1] * r1[2]
            + d_cos_dt[2] * (-r1[1])
            + d_cos_dt[4] * (r4[2] - r3[2])
            + d_cos_dt[5] * (r3[1] - r4[1]));
    forces[c * 3 + 1] += sin_term
        * (d_cos_dt[0] * (-r1[2])
            + d_cos_dt[2] * r1[0]
            + d_cos_dt[3] * (r3[2] - r4[2])
            + d_cos_dt[5] * (r4[0] - r3[0]));
    forces[c * 3 + 2] += sin_term
        * (d_cos_dt[0] * r1[1]
            + d_cos_dt[1] * (-r1[0])
            + d_cos_dt[3] * (r4[1] - r3[1])
            + d_cos_dt[4] * (r3[0] - r4[0]));

    // g[3] (atom d)
    forces[d * 3] += sin_term * (d_cos_dt[4] * r3[2] - d_cos_dt[5] * r3[1]);
    forces[d * 3 + 1] += sin_term * (d_cos_dt[5] * r3[0] - d_cos_dt[3] * r3[2]);
    forces[d * 3 + 2] += sin_term * (d_cos_dt[3] * r3[1] - d_cos_dt[4] * r3[0]);
}

pub fn dihedral_uff_constructor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let block = frame
        .get(DIHEDRALS)
        .ok_or("uff_torsion: missing \"dihedrals\" block")?;
    let i = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_torsion: missing atomi")?;
    let j = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_torsion: missing atomj")?;
    let k = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_torsion: missing atomk")?;
    let l = block
        .get("atoml")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_torsion: missing atoml")?;
    let v = param_reads::instance_col("uff_torsion", block, "V")?;
    let order = param_reads::instance_col("uff_torsion", block, "order")?;
    let cos_term = param_reads::instance_col("uff_torsion", block, "cosTerm")?;
    let n = i.len();
    Ok(ForceTerm::indexed(DihedralUff {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        atom_l: (0..n).map(|t| l[t] as usize).collect(),
        v: v.iter().map(|&x| x as F).collect(),
        order: order.iter().map(|&x| x as u8).collect(),
        cos_term: cos_term.iter().map(|&x| x as F).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::fixtures::assert_forces_are_negative_gradient;

    fn torsion(v: F, order: u8, cos_term: F) -> DihedralUff {
        DihedralUff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            v: vec![v],
            order: vec![order],
            cos_term: vec![cos_term],
        }
    }

    /// Planar anti arrangement: φ = 180°.
    fn anti() -> Vec<F> {
        vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 2.1, 1.4, 0.0, 3.6, 1.4, 0.0]
    }

    #[test]
    fn energy_is_half_v_times_one_minus_cos_term_cos_n_phi() {
        // cos(3·180°) = −1, so E = V/2·(1 + cosTerm).
        let (e_plus, _) = torsion(4.0, 3, 1.0).energy_forces_at(&anti());
        let (e_minus, _) = torsion(4.0, 3, -1.0).energy_forces_at(&anti());
        assert!((e_plus - 4.0).abs() < 1e-9);
        assert!(e_minus.abs() < 1e-9);
    }

    #[test]
    fn forces_are_the_negative_energy_gradient() {
        // A skew chain, no three atoms collinear, nothing planar.
        let coords = vec![0.0, 0.1, 0.2, 1.5, 0.0, 0.0, 2.1, 1.4, 0.3, 3.4, 1.7, 1.1];
        for (order, cos_term) in [(3, 1.0), (2, -1.0), (6, 1.0)] {
            assert_forces_are_negative_gradient(&torsion(4.0, order, cos_term), &coords, 1e-5);
        }
    }
}
