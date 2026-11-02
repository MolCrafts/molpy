//! The second ETKDG stage: 3D experimental-torsion refinement.
//!
//! Port of RDKit's `construct3DForceField` / `minimizeWithExpTorsions`
//! (`$RDBASE/Code/DistGeom/DistGeomUtils.cpp`,
//! `$RDBASE/Code/GraphMol/DistGeomHelpers/Embedder.cpp`, BSD-3, Copyright
//! (C) 2004-2025 Greg Landrum / Sereina Riniker and other RDKit
//! contributors), legacy (sequential) implementation, ETKDGv3 force
//! constants (`ETKDGForceConsts::SEQ::Cosine`). The torsions and the
//! inversions are `ff::potential` kernels; the distance and angle
//! constraints, which only exist against the first stage's geometry and the
//! bounds matrix, are owned here. The minimizer is the crate's one L-BFGS,
//! [`crate::optimize::minimize_lbfgs_rms`], as for RDKit's own BFGS.

use crate::conformer::distgeom::{
    BondAngle, BoundsMatrix, DgConstraints, ImproperConstraint, TorsionConstraint,
};
use crate::ff::compile::ExplicitTerms;
use crate::ff::ir::Params;
use crate::ff::potential::{Accumulator, Potential, Potentials};
use crate::ff::typifier::uff::inversion_coeffs;

/// Half-width of the window a known (1-2, 1-3) distance is held in (RDKit
/// `KNOWN_DIST_TOL`).
const KNOWN_DIST_TOL: f64 = 0.01;
/// Force constant of a known distance (RDKit `KNOWN_DIST_FORCE_CONSTANT`).
const KNOWN_DIST_FORCE: f64 = 100.0;
/// Force constant of the remaining bounds-matrix pairs
/// (`boundsMatForceScaling · 10`, `boundsMatForceScaling = 1`).
const BOUNDS_FORCE: f64 = 10.0;
/// Scaling of the UFF inversion terms (`kTermImproper`).
const IMPROPER_FORCE: f64 = 10.0;
/// Force constant of a linear angle held in [179°, 180°] (`kTermAngle`).
const LINEAR_ANGLE_FORCE: f64 = 1.0;

/// Second-stage (3D) objective, RDKit's `construct3DForceField`:
///
/// * the experimental (CrystalFF M6) and flat-ring torsions, `Σₘ Vₘ (1 + sₘ
///   cos mφ)` — LAMMPS `dihedral_style fourier` (`dihedral periodic` here)
///   with `kₘ = Vₘ`, `nₘ = m`, phase 0° for `sₘ = +1` and 180° for `sₘ = −1`,
///   one per central bond; their end atoms get no distance term;
/// * the UFF inversion of every planar sp2 centre, three permutations, `K`
///   scaled by `kTermImproper`;
/// * every bond held at its first-stage length ± 0.01 Å, `k = 100`;
/// * every angle's 1-3 distance held the same way — unless the centre is
///   sp (the angle held in [179°, 180°]) or an inversion centre (the 1-3
///   distance kept in its bounds-matrix range, `k = 100`);
/// * every other pair kept in its bounds-matrix range, `k = 10`.
///
/// Without the basic-knowledge terms (no inversions, no linear angles) this
/// is RDKit's `constructPlain3DForceField`.
pub(super) struct TorsionRefinement {
    /// Flat-bottom distance constraints `½ k (d − bound)²` outside
    /// `[lower, upper]` (RDKit `DistanceConstraintContribs`).
    distances: Vec<DistanceBound>,
    /// Flat-bottom angle constraints `k (θ − bound)²`, θ in degrees, outside
    /// `[lower, upper]` (RDKit `AngleConstraintContribs`).
    angles: Vec<AngleBound>,
    /// The torsions, as one `dihedral periodic` kernel.
    torsions: Potentials,
    /// The inversions, as one `uff_inversion` kernel.
    planarity: Potentials,
}

#[derive(Clone, Copy)]
struct DistanceBound {
    i: usize,
    j: usize,
    lower: f64,
    upper: f64,
    k: f64,
}

#[derive(Clone, Copy)]
struct AngleBound {
    atoms: [usize; 3],
    lower: f64,
    upper: f64,
    k: f64,
}

impl TorsionRefinement {
    /// Build over the first-stage geometry `coords` (flat `n*3`): the
    /// constraint set's bonds, angles and bounds, the given torsions, and
    /// the inversions and linear angles when `basic_knowledge` is on.
    pub(super) fn build<'a>(
        constraints: &DgConstraints,
        coords: &[f64],
        torsions: impl IntoIterator<Item = &'a TorsionConstraint>,
        basic_knowledge: bool,
    ) -> Self {
        let bounds = &constraints.bounds;
        let n = bounds.len();
        let mut paired = vec![false; n * n];
        let mut mark = |i: usize, j: usize| {
            paired[i * n + j] = true;
            paired[j * n + i] = true;
        };

        let mut fourier = ExplicitTerms::new("dihedral", "periodic");
        let mut central = std::collections::HashSet::new();
        for t in torsions {
            let [i, j, k, l] = t.atoms;
            if !central.insert((j.min(k), j.max(k))) {
                continue;
            }
            mark(i, l);
            let mut row = Params::new();
            for m in 0..6 {
                row.set(&format!("k{}", m + 1), t.force_constants[m]);
                row.set(&format!("periodicity{}", m + 1), (m + 1) as f64);
                let phase = if t.signs[m] < 0 { 180.0 } else { 0.0 };
                row.set(&format!("phase{}", m + 1), phase);
            }
            fourier = fourier.term(&t.atoms, row);
        }
        let torsions = compile_or_empty(&fourier, !central.is_empty());

        let impropers: &[ImproperConstraint] = if basic_knowledge {
            &constraints.improper
        } else {
            &[]
        };
        let mut inversion_centre = vec![false; n];
        let mut inversions = ExplicitTerms::new("improper", "uff_inversion");
        for im in impropers {
            let [a0, c, a2, a3] = im.atoms;
            let (k, c0, c1, c2) = inversion_coeffs(im.center_z, im.c_bound_to_o);
            let row = Params::from_iter([
                ("K", k * IMPROPER_FORCE),
                ("c0", c0),
                ("c1", c1),
                ("c2", c2),
            ]);
            // RDKit's permutations (0,1,2,3), (0,1,3,2), (2,1,3,0) of
            // [n0, centre, n2, n3]; the kernel takes the centre first.
            for outer in [[a0, a2, a3], [a0, a3, a2], [a2, a3, a0]] {
                inversions = inversions.term(&[c, outer[0], outer[1], outer[2]], row.clone());
            }
            inversion_centre[c] = true;
        }
        let planarity = compile_or_empty(&inversions, !impropers.is_empty());

        let distance = |i: usize, j: usize| {
            let d: f64 = (0..3)
                .map(|a| (coords[i * 3 + a] - coords[j * 3 + a]).powi(2))
                .sum();
            d.sqrt()
        };
        let known = |i: usize, j: usize| {
            let d = distance(i, j);
            DistanceBound {
                i,
                j,
                lower: d - KNOWN_DIST_TOL,
                upper: d + KNOWN_DIST_TOL,
                k: KNOWN_DIST_FORCE,
            }
        };
        let mut distances = Vec::new();
        for &[i, j] in &constraints.bonds {
            mark(i, j);
            distances.push(known(i, j));
        }
        let mut angles = Vec::new();
        for &BondAngle {
            atoms: [i, j, k],
            linear,
        } in &constraints.angles
        {
            mark(i, k);
            if basic_knowledge && linear {
                angles.push(AngleBound {
                    atoms: [i, j, k],
                    lower: 179.0,
                    upper: 180.0,
                    k: LINEAR_ANGLE_FORCE,
                });
            } else if inversion_centre[j] {
                distances.push(DistanceBound {
                    i,
                    j: k,
                    lower: bounds.lower(i, k),
                    upper: bounds.upper(i, k),
                    k: KNOWN_DIST_FORCE,
                });
            } else {
                distances.push(known(i, k));
            }
        }
        for i in 1..n {
            for j in 0..i {
                if !paired[i * n + j] {
                    distances.push(long_range(bounds, i, j));
                }
            }
        }
        Self {
            distances,
            angles,
            torsions,
            planarity,
        }
    }

    /// Energy + gradient over a flat `n*3` coordinate buffer.
    pub(super) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let mut energy = 0.0;
        for c in &self.distances {
            energy += distance_bound(p, c, grad);
        }
        for c in &self.angles {
            energy += angle_bound(p, c, grad);
        }
        // Torsions and inversions: forces are −gradient, so they are
        // tallied into one buffer and subtracted once.
        let mut acc = Accumulator::new(p.len());
        for kernel in [&self.torsions as &dyn Potential, &self.planarity] {
            kernel.evaluate(p, &mut acc);
        }
        energy += acc.energy();
        for (g, f) in grad.iter_mut().zip(acc.forces()) {
            *g -= f;
        }
        energy
    }
}

/// The kernel of `terms`, or none when there is no term.
fn compile_or_empty(terms: &ExplicitTerms, any: bool) -> Potentials {
    if any {
        terms
            .compile()
            .expect("a built-in style prices any row the second stage writes")
    } else {
        Potentials::new()
    }
}

/// A pair kept in its bounds-matrix range (RDKit
/// `addLongRangeDistanceConstraints`).
fn long_range(bounds: &BoundsMatrix, i: usize, j: usize) -> DistanceBound {
    DistanceBound {
        i,
        j,
        lower: bounds.lower(i, j),
        upper: bounds.upper(i, j),
        k: BOUNDS_FORCE,
    }
}

/// `½ k (d − bound)²` outside `[lower, upper]`, gradient added into `grad`.
fn distance_bound(p: &[f64], c: &DistanceBound, grad: &mut [f64]) -> f64 {
    let r: [f64; 3] = std::array::from_fn(|a| p[c.i * 3 + a] - p[c.j * 3 + a]);
    let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
    let term = if d < c.lower {
        d - c.lower
    } else if d > c.upper {
        d - c.upper
    } else {
        return 0.0;
    };
    let pre = c.k * term / d.max(1e-8);
    for a in 0..3 {
        grad[c.i * 3 + a] += pre * r[a];
        grad[c.j * 3 + a] -= pre * r[a];
    }
    0.5 * c.k * term * term
}

/// `k (θ − bound)²`, θ in degrees, outside `[lower, upper]`, gradient added
/// into `grad`.
fn angle_bound(p: &[f64], c: &AngleBound, grad: &mut [f64]) -> f64 {
    let [i, j, k] = c.atoms;
    let u: [f64; 3] = std::array::from_fn(|a| p[i * 3 + a] - p[j * 3 + a]);
    let v: [f64; 3] = std::array::from_fn(|a| p[k * 3 + a] - p[j * 3 + a]);
    let (lu, lv) = (norm(u).max(1e-8), norm(v).max(1e-8));
    let cos = ((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (lu * lv)).clamp(-1.0, 1.0);
    let theta = cos.acos().to_degrees();
    let term = if theta < c.lower {
        theta - c.lower
    } else if theta > c.upper {
        theta - c.upper
    } else {
        return 0.0;
    };
    // dE/dθ (radians) · dθ/dcos, with RDKit's floor on sin θ.
    let sin = (1.0 - cos * cos).sqrt().max(1e-8);
    let pre = -2.0 * c.k * term * 180.0 / std::f64::consts::PI / sin;
    for a in 0..3 {
        let gi = pre * (v[a] / (lu * lv) - cos * u[a] / (lu * lu));
        let gk = pre * (u[a] / (lu * lv) - cos * v[a] / (lv * lv));
        grad[i * 3 + a] += gi;
        grad[k * 3 + a] += gk;
        grad[j * 3 + a] -= gi + gk;
    }
    c.k * term * term
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Central differences of `f` at `p`, against the analytic gradient.
    fn check_gradient(p: &[f64], f: impl Fn(&[f64], &mut [f64]) -> f64) {
        let mut grad = vec![0.0; p.len()];
        f(p, &mut grad);
        let h = 1e-6;
        for x in 0..p.len() {
            let (mut a, mut b) = (p.to_vec(), p.to_vec());
            a[x] += h;
            b[x] -= h;
            let mut scratch = vec![0.0; p.len()];
            let numeric = (f(&a, &mut scratch) - f(&b, &mut scratch)) / (2.0 * h);
            assert!(
                (numeric - grad[x]).abs() < 1e-5,
                "component {x}: {numeric} vs {}",
                grad[x]
            );
        }
    }

    #[test]
    fn distance_bound_gradient_is_the_energy_derivative() {
        let p = [0.0, 0.0, 0.0, 1.1, 0.3, -0.2];
        for (lower, upper) in [(1.5, 1.6), (0.5, 0.6), (1.0, 1.3)] {
            let c = DistanceBound {
                i: 0,
                j: 1,
                lower,
                upper,
                k: 100.0,
            };
            check_gradient(&p, |q, g| distance_bound(q, &c, g));
        }
    }

    #[test]
    fn angle_bound_gradient_is_the_energy_derivative() {
        let p = [1.0, 0.1, 0.0, 0.0, 0.0, 0.0, -0.8, 0.7, 0.2];
        for (lower, upper) in [(179.0, 180.0), (60.0, 70.0), (100.0, 160.0)] {
            let c = AngleBound {
                atoms: [0, 1, 2],
                lower,
                upper,
                k: 1.0,
            };
            check_gradient(&p, |q, g| angle_bound(q, &c, g));
        }
    }
}
