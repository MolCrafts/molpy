//! Out-of-plane distance improper (LAMMPS `improper_style distance`).

use crate::ff::potential::{Accumulator, Potential};
use molrs::op::F;
use molrs::op::vec3::{add, cross, dot, norm, scale, sub};

/// `K₂ d² + K₄ d⁴` over explicit `[i (centre), j, k, l]` quadruples.
///
/// Out-of-plane distance improper, LAMMPS `improper_style distance`:
///
/// E = K₂ d² + K₄ d⁴
///
/// where `d` is the distance of the central atom `i` from the plane of the
/// other three, `j`, `k`, `l`. It is the planarity restraint of the ETKDG
/// conformer pipeline (RDKit's out-of-plane term at
/// `oobForceScalingFactor = 10`, `K₂ = 10`, `K₄ = 0`), priced here once so the
/// conformer stages share the force-field kernels instead of carrying their
/// own.
///
/// A degenerate plane (the three outer atoms collinear) has no normal; the
/// term then contributes neither energy nor force.
#[derive(Debug, Clone, Default)]
pub struct ImproperDistance {
    atoms: Vec<[usize; 4]>,
    k2: Vec<F>,
    k4: Vec<F>,
}

impl ImproperDistance {
    /// No terms yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one term: centre `atoms[0]` out of the plane of `atoms[1..4]`,
    /// with `k2` (energy / length²) and `k4` (energy / length⁴).
    pub fn term(mut self, atoms: [usize; 4], k2: F, k4: F) -> Self {
        self.atoms.push(atoms);
        self.k2.push(k2);
        self.k4.push(k4);
        self
    }

    /// Number of terms.
    pub fn len(&self) -> usize {
        self.atoms.len()
    }

    /// Whether there is no term.
    pub fn is_empty(&self) -> bool {
        self.atoms.is_empty()
    }
}

fn point(coords: &[F], a: usize) -> [F; 3] {
    [coords[3 * a], coords[3 * a + 1], coords[3 * a + 2]]
}

impl Potential for ImproperDistance {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let forces = acc.forces_without_virial();
        let mut energy = 0.0;
        for (t, &[i, j, k, l]) in self.atoms.iter().enumerate() {
            let (c, a, b, d) = (
                point(coords, i),
                point(coords, j),
                point(coords, k),
                point(coords, l),
            );
            let (u, v, w) = (sub(b, a), sub(d, a), sub(c, a));
            let normal = cross(u, v);
            let len = norm(normal);
            if len < 1e-9 {
                continue;
            }
            let n_hat = scale(normal, 1.0 / len);
            let h = dot(w, n_hat);
            let (k2, k4) = (self.k2[t], self.k4[t]);
            energy += k2 * h * h + k4 * h * h * h * h;
            let de_dh = 2.0 * k2 * h + 4.0 * k4 * h * h * h;
            // ∂h/∂c = n̂; ∂h/∂b and ∂h/∂d through the normal; ∂h/∂a closes
            // the sum to zero (h is translation invariant).
            let dh_dc = n_hat;
            let dh_db = scale(sub(cross(v, w), scale(cross(v, n_hat), h)), 1.0 / len);
            let dh_dd = scale(sub(cross(w, u), scale(cross(n_hat, u), h)), 1.0 / len);
            let dh_da = scale(add(add(dh_dc, dh_db), dh_dd), -1.0);
            for (atom, g) in [(i, dh_dc), (j, dh_da), (k, dh_db), (l, dh_dd)] {
                for x in 0..3 {
                    forces[3 * atom + x] -= de_dh * g[x];
                }
            }
        }
        acc.add_energy(energy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    const COORDS: [F; 12] = [
        0.1, 0.2, 0.7, // centre, above the plane
        0.0, 0.0, 0.0, //
        1.0, 0.1, 0.0, //
        0.2, 1.1, 0.1,
    ];

    #[test]
    fn energy_is_k2_d_squared_plus_k4_d_fourth() {
        // Plane z = 0, centre at height 0.5.
        let coords = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let pot = ImproperDistance::new().term([0, 1, 2, 3], 10.0, 2.0);
        let e = pot.energy_at(&coords);
        assert!((e - (10.0 * 0.25 + 2.0 * 0.0625)).abs() < 1e-12, "{e}");
    }

    #[test]
    fn forces_are_the_negative_gradient() {
        let pot = ImproperDistance::new().term([0, 1, 2, 3], 10.0, 3.0);
        let (_, forces) = pot.energy_forces_at(&COORDS);
        let h = 1e-6;
        for idx in 0..COORDS.len() {
            let mut p = COORDS;
            p[idx] += h;
            let ep = pot.energy_at(&p);
            p[idx] -= 2.0 * h;
            let em = pot.energy_at(&p);
            let numeric = -(ep - em) / (2.0 * h);
            assert!(
                (forces[idx] - numeric).abs() < 1e-6,
                "coordinate {idx}: analytic {} numeric {numeric}",
                forces[idx]
            );
        }
        let total: F = forces.iter().sum();
        assert!(total.abs() < 1e-10, "forces must sum to zero: {total}");
    }

    #[test]
    fn a_collinear_plane_prices_nothing() {
        let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let pot = ImproperDistance::new().term([0, 1, 2, 3], 10.0, 0.0);
        assert_eq!(pot.energy_forces_at(&coords), (0.0, vec![0.0; 12]));
    }
}
