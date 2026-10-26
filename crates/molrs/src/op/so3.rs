//! Uniform sampling of directions on S², the unit sphere in 3D.

use crate::op::{F, Vec3};

use std::f64::consts::TAU;

/// A unit vector from `u ∈ [0, 1]²`: `z = 2u₀ − 1`, `φ = 2πu₁`,
/// `(√(1−z²) cos φ, √(1−z²) sin φ, z)` (dimensionless). Uniform on the sphere
/// when `u` is uniform on `[0, 1)²`. Why a uniform `z` is right: by
/// Archimedes' hat-box theorem, the area of a unit-sphere band between heights
/// `z` and `z + dz` is `2π dz`, independent of `z`, so equal steps in `z`
/// cover equal areas; the azimuth `φ` is uniform by symmetry.
pub fn unit_vector_from_uniform(u: [F; 2]) -> Vec3 {
    let z = 2.0 * u[0] - 1.0;
    let phi = TAU * u[1];
    let r = (1.0 - z * z).max(0.0).sqrt();
    [r * phi.cos(), r * phi.sin(), z]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::F;

    const MAP_TOL: F = 1e-15;

    #[test]
    fn unit_vector_from_uniform_equator_at_zero_azimuth_is_x() {
        // z = 2·0.5 − 1 = 0, φ = 0 → (1, 0, 0).
        let v = unit_vector_from_uniform([0.5, 0.0]);
        for (d, want) in [1.0, 0.0, 0.0].iter().enumerate() {
            assert!((v[d] - want).abs() < MAP_TOL, "v = {v:?}");
        }
    }
}
