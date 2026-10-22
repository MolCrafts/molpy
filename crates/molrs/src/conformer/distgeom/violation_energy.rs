//! The distance-geometry error function: how far an embedding violates its
//! distance bounds and chiral volumes.
//!
//! Crippen & Havel's error function (*Distance Geometry and Molecular
//! Conformation*, 1988), as RDKit evaluates it (BSD-3, Copyright (C)
//! 2004-2025 Greg Landrum / Sereina Riniker and other RDKit contributors):
//!   * `$RDBASE/Code/DistGeom/DistViolationContribs.cpp` — distance-bound
//!     violations ([`DistanceViolations`]),
//!   * `$RDBASE/Code/DistGeom/ChiralViolationContribs.cpp` — signed
//!     chiral-volume violations (`calcChiralVolume`, [`chiral_volume`]),
//!   * `$RDBASE/Code/DistGeom/FourthDimContribs.h` — the fourth-dimension
//!     penalty that squeezes a 4D embedding back to 3D.
//!
//! It is not a force field: its terms are squared violations of geometric
//! constraints (bounds, volumes, a dimension), with no physical parameters
//! and no unit of energy, and they are only defined against a
//! [`BoundsMatrix`]. So it lives with the bounds it measures, here in
//! distance geometry, not in `ff::potential`. Every function takes a flat
//! coordinate buffer of stride `dim` (3 or 4).

use crate::conformer::distgeom::{BoundsMatrix, ChiralConstraint};
use crate::op::vec3::{cross, dot};

/// One distance-bound violation term (RDKit `DistViolationContribs`).
#[derive(Clone, Copy)]
struct BoundPair {
    i: usize,
    j: usize,
    ub2: f64,
    lb2: f64,
    weight: f64,
}

/// The distance-bound violations of every atom pair of a bounds matrix:
/// `(d²/u² − 1)²` above the upper bound `u`, `(2l²/(l² + d²) − 1)²` below the
/// lower bound `l`, zero between.
pub(crate) struct DistanceViolations {
    dim: usize,
    pairs: Vec<BoundPair>,
}

impl DistanceViolations {
    /// Every pair `j < i` of `bounds`, unit weight, on coordinates of stride
    /// `dim`.
    pub(crate) fn new(bounds: &BoundsMatrix, dim: usize) -> Self {
        let n = bounds.len();
        let mut pairs = Vec::new();
        for i in 1..n {
            for j in 0..i {
                let u = bounds.upper(i, j);
                let l = bounds.lower(i, j);
                pairs.push(BoundPair {
                    i,
                    j,
                    ub2: u * u,
                    lb2: l * l,
                    weight: 1.0,
                });
            }
        }
        Self { dim, pairs }
    }

    /// The violation energy at `p`; its gradient is added into `grad`.
    pub(crate) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        let dim = self.dim;
        let mut energy = 0.0;
        for c in &self.pairs {
            let mut d2 = 0.0;
            for k in 0..dim {
                let d = p[c.i * dim + k] - p[c.j * dim + k];
                d2 += d * d;
            }
            let mut val = 0.0;
            if d2 > c.ub2 {
                val = d2 / c.ub2 - 1.0;
            } else if d2 < c.lb2 {
                val = 2.0 * c.lb2 / (c.lb2 + d2) - 1.0;
            }
            if val > 0.0 {
                energy += c.weight * val * val;
            }
            // Gradient (RDKit DistViolationContribs::getGrad).
            let mut pre = 0.0;
            let mut d = 0.0;
            if d2 > c.ub2 {
                d = d2.sqrt();
                pre = 4.0 * (d2 / c.ub2 - 1.0) * (d / c.ub2);
            } else if d2 < c.lb2 {
                d = d2.sqrt();
                let l2d2 = d2 + c.lb2;
                pre = 8.0 * c.lb2 * d * (1.0 - 2.0 * c.lb2 / l2d2) / (l2d2 * l2d2);
            }
            if pre != 0.0 {
                for k in 0..dim {
                    let p1 = c.i * dim + k;
                    let p2 = c.j * dim + k;
                    let dgrad = if d > 0.0 {
                        c.weight * pre * (p[p1] - p[p2]) / d
                    } else {
                        c.weight * pre * (p[p1] - p[p2])
                    };
                    grad[p1] += dgrad;
                    grad[p2] -= dgrad;
                }
            }
        }
        energy
    }
}

/// One chiral-volume violation term (RDKit `ChiralViolationContribs`).
#[derive(Clone, Copy)]
struct ChiralVolume {
    idx: [usize; 4],
    vol_lower: f64,
    vol_upper: f64,
    weight: f64,
}

/// The first-stage error function of an ETKDG embedding: distance-bound
/// violations, chiral-volume violations and the fourth-dimension penalty
/// (RDKit `DistGeom::constructForceField` with `weightChiral`,
/// `weightFourthDim`).
pub(crate) struct ViolationEnergy {
    n: usize,
    dim: usize,
    distances: DistanceViolations,
    chiral: Vec<ChiralVolume>,
    fourth_weight: f64,
}

impl ViolationEnergy {
    /// The error function of `bounds` and the `chiral` constraints on
    /// coordinates of stride `dim`; the chiral terms weigh `weight_chiral`
    /// (none at or below `1e-8`), the fourth dimension `weight_fourth` (only
    /// when `dim == 4`).
    pub(crate) fn new(
        bounds: &BoundsMatrix,
        chiral: &[ChiralConstraint],
        dim: usize,
        weight_chiral: f64,
        weight_fourth: f64,
    ) -> Self {
        let mut volumes = Vec::new();
        if weight_chiral > 1e-8 {
            for c in chiral {
                volumes.push(ChiralVolume {
                    idx: c.neighbors,
                    vol_lower: c.volume_lower,
                    vol_upper: c.volume_upper,
                    weight: weight_chiral,
                });
            }
        }
        Self {
            n: bounds.len(),
            dim,
            distances: DistanceViolations::new(bounds, dim),
            chiral: volumes,
            fourth_weight: if dim == 4 { weight_fourth } else { 0.0 },
        }
    }

    /// The error at `p` (a flat `n*dim` buffer); its gradient overwrites
    /// `grad`.
    pub(crate) fn energy_grad(&self, p: &[f64], grad: &mut [f64]) -> f64 {
        for g in grad.iter_mut() {
            *g = 0.0;
        }
        let dim = self.dim;
        let mut energy = self.distances.energy_grad(p, grad);

        // Chiral-volume violations (computed using only the first 3 dims).
        for c in &self.chiral {
            energy += chiral_violation(p, dim, c, grad);
        }

        // Fourth-dimension penalty.
        if self.fourth_weight > 1e-8 && dim == 4 {
            for i in 0..self.n {
                let pid = i * dim + 3;
                energy += self.fourth_weight * p[pid] * p[pid];
                grad[pid] += self.fourth_weight * p[pid];
            }
        }
        energy
    }
}

/// The three edge vectors `p_a − p_4` (`a` = 1, 2, 3) of a chiral quadruple,
/// in the first three dimensions of stride `dim`.
fn edges(p: &[f64], idx: [usize; 4], dim: usize) -> [[f64; 3]; 3] {
    let [i1, i2, i3, i4] = idx;
    let edge = |a: usize| {
        [
            p[a * dim] - p[i4 * dim],
            p[a * dim + 1] - p[i4 * dim + 1],
            p[a * dim + 2] - p[i4 * dim + 2],
        ]
    };
    [edge(i1), edge(i2), edge(i3)]
}

/// One chiral term's violation energy at `p`; its gradient is added into
/// `grad` (RDKit `ChiralViolationContribs::getGrad`, 12 components).
fn chiral_violation(p: &[f64], dim: usize, c: &ChiralVolume, grad: &mut [f64]) -> f64 {
    let [i1, i2, i3, i4] = c.idx;
    let [v1, v2, v3] = edges(p, c.idx, dim);
    let vol = dot(v1, cross(v2, v3));

    let (energy, pre) = if vol < c.vol_lower {
        (
            c.weight * (vol - c.vol_lower) * (vol - c.vol_lower),
            c.weight * (vol - c.vol_lower),
        )
    } else if vol > c.vol_upper {
        (
            c.weight * (vol - c.vol_upper) * (vol - c.vol_upper),
            c.weight * (vol - c.vol_upper),
        )
    } else {
        return 0.0;
    };

    grad[dim * i1] += pre * (v2[1] * v3[2] - v3[1] * v2[2]);
    grad[dim * i1 + 1] += pre * (v3[0] * v2[2] - v2[0] * v3[2]);
    grad[dim * i1 + 2] += pre * (v2[0] * v3[1] - v3[0] * v2[1]);

    grad[dim * i2] += pre * (v3[1] * v1[2] - v3[2] * v1[1]);
    grad[dim * i2 + 1] += pre * (v3[2] * v1[0] - v3[0] * v1[2]);
    grad[dim * i2 + 2] += pre * (v3[0] * v1[1] - v3[1] * v1[0]);

    grad[dim * i3] += pre * (v2[2] * v1[1] - v2[1] * v1[2]);
    grad[dim * i3 + 1] += pre * (v2[0] * v1[2] - v2[2] * v1[0]);
    grad[dim * i3 + 2] += pre * (v2[1] * v1[0] - v2[0] * v1[1]);

    grad[dim * i4] += pre
        * (p[i1 * dim + 2] * (p[i2 * dim + 1] - p[i3 * dim + 1])
            + p[i2 * dim + 2] * (p[i3 * dim + 1] - p[i1 * dim + 1])
            + p[i3 * dim + 2] * (p[i1 * dim + 1] - p[i2 * dim + 1]));
    grad[dim * i4 + 1] += pre
        * (p[i1 * dim] * (p[i2 * dim + 2] - p[i3 * dim + 2])
            + p[i2 * dim] * (p[i3 * dim + 2] - p[i1 * dim + 2])
            + p[i3 * dim] * (p[i1 * dim + 2] - p[i2 * dim + 2]));
    grad[dim * i4 + 2] += pre
        * (p[i1 * dim + 1] * (p[i2 * dim] - p[i3 * dim])
            + p[i2 * dim + 1] * (p[i3 * dim] - p[i1 * dim])
            + p[i3 * dim + 1] * (p[i1 * dim] - p[i2 * dim]));
    energy
}

/// Signed chiral volume `v₁ · (v₂ × v₃)`, `v_a = p_a − p₄`, of four points in
/// their first three dimensions (RDKit `DistGeom::calcChiralVolume`). `dim`
/// is the coordinate stride.
pub(crate) fn chiral_volume(p: &[f64], idx: [usize; 4], dim: usize) -> f64 {
    let [v1, v2, v3] = edges(p, idx, dim);
    dot(v1, cross(v2, v3))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A right-handed tetrahedron: apex above the origin, base on the axes.
    fn tetrahedron(dim: usize) -> Vec<f64> {
        let pts = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ];
        let mut p = vec![0.0; 4 * dim];
        for (a, xyz) in pts.iter().enumerate() {
            p[a * dim..a * dim + 3].copy_from_slice(xyz);
        }
        p
    }

    #[test]
    fn the_chiral_volume_is_the_triple_product_about_the_fourth_point() {
        // v1 = e_x, v2 = e_y, v3 = e_z relative to the origin: e_x · (e_y × e_z) = 1.
        assert_eq!(chiral_volume(&tetrahedron(3), [0, 1, 2, 3], 3), 1.0);
        // Swapping two substituents flips the sign.
        assert_eq!(chiral_volume(&tetrahedron(3), [1, 0, 2, 3], 3), -1.0);
    }

    #[test]
    fn the_stride_skips_the_fourth_dimension() {
        let mut p = tetrahedron(4);
        for a in 0..4 {
            p[a * 4 + 3] = 9.0 * a as f64; // anything in dimension 4 is ignored
        }
        assert_eq!(chiral_volume(&p, [0, 1, 2, 3], 4), 1.0);
    }

    #[test]
    fn coplanar_points_have_no_volume() {
        let mut p = tetrahedron(3);
        p[2] = 0.0; // put the apex into the z = 0 plane
        p[8] = 0.0;
        assert_eq!(chiral_volume(&p, [0, 1, 2, 3], 3), 0.0);
    }
}
