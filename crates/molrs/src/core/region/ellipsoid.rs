//! An axis-aligned ellipsoid.

use super::primitives::Region;
use crate::op::{F, Fnx3};
use ndarray::Array2;

/// A solid axis-aligned ellipsoid with semi-axes `(a, b, c)` about `center`.
///
/// With `q = (x − center) ⊘ (a, b, c)`, the point is inside where `‖q‖ ≤ 1`.
/// The Euclidean distance to an ellipsoid has no closed form, so
/// `distance(x) = (‖q‖ − 1) · min(a, b, c)`: the sign is exact and the
/// magnitude is a lower bound on the true distance on both sides (equal to
/// it along the shortest semi-axis). The gradient is the unit vector along
/// `q ⊘ (a, b, c)`, the normal of the level set, zero at the centre.
#[derive(Debug, Clone)]
pub struct Ellipsoid {
    center: [F; 3],
    semi_axes: [F; 3],
}

impl Ellipsoid {
    /// Ellipsoid about `center` (Å) with semi-axes `(a, b, c)` along x, y, z (Å).
    ///
    /// # Errors
    ///
    /// Returns `Err` if any semi-axis is not positive or any value is not
    /// finite.
    pub fn new(center: [F; 3], semi_axes: [F; 3]) -> Result<Self, String> {
        if semi_axes.iter().any(|a| !(*a > 0.0 && a.is_finite())) {
            return Err(format!(
                "Ellipsoid: semi-axes must be > 0, got {semi_axes:?}"
            ));
        }
        if center.iter().any(|c| !c.is_finite()) {
            return Err("Ellipsoid: center must be finite".to_string());
        }
        Ok(Self { center, semi_axes })
    }

    /// Centre, Å.
    pub fn center(&self) -> [F; 3] {
        self.center
    }

    /// Semi-axes `(a, b, c)`, Å.
    pub fn semi_axes(&self) -> [F; 3] {
        self.semi_axes
    }

    fn scaled(&self, point: &[F; 3]) -> ([F; 3], F) {
        let q = [
            (point[0] - self.center[0]) / self.semi_axes[0],
            (point[1] - self.center[1]) / self.semi_axes[1],
            (point[2] - self.center[2]) / self.semi_axes[2],
        ];
        (q, (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt())
    }
}

impl Region for Ellipsoid {
    fn bounds(&self) -> Fnx3 {
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = self.center[d] - self.semi_axes[d];
            b[[d, 1]] = self.center[d] + self.semi_axes[d];
        }
        b
    }

    fn distance(&self, point: &[F; 3]) -> F {
        let (_, k) = self.scaled(point);
        let a_min = self.semi_axes[0]
            .min(self.semi_axes[1])
            .min(self.semi_axes[2]);
        (k - 1.0) * a_min
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (q, k) = self.scaled(point);
        if k < 1e-12 {
            return [0.0; 3];
        }
        let g = [
            q[0] / self.semi_axes[0],
            q[1] / self.semi_axes[1],
            q[2] / self.semi_axes[2],
        ];
        let n = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        [g[0] / n, g[1] / n, g[2] / n]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_is_exact_and_shortest_axis_is_euclidean() {
        let e = Ellipsoid::new([1.0, 0.0, 0.0], [2.0, 3.0, 1.0]).unwrap();
        assert_eq!(e.distance(&[1.0, 0.0, 0.0]), -1.0);
        assert_eq!(e.distance(&[1.0, 0.0, 1.0]), 0.0);
        assert_eq!(e.distance(&[1.0, 3.0, 0.0]), 0.0);
        // Along the shortest semi-axis the value is the Euclidean distance.
        assert!((e.distance(&[1.0, 0.0, 2.5]) - 1.5).abs() < 1e-12);
        // Along a longer axis it is a lower bound on the Euclidean 2.0.
        let d = e.distance(&[5.0, 0.0, 0.0]);
        assert!(d > 0.0 && d <= 2.0 + 1e-12);
        assert!(e.contains_point(&[2.0, 1.0, 0.5]));
        assert!(!e.contains_point(&[3.1, 0.0, 0.0]));
        let b = e.bounds();
        assert_eq!(b[[1, 0]], -3.0);
        assert_eq!(b[[1, 1]], 3.0);
    }

    #[test]
    fn gradient_is_the_level_set_normal() {
        let e = Ellipsoid::new([0.0; 3], [2.0, 3.0, 1.0]).unwrap();
        let p = [1.5, 1.2, 0.4];
        let g = e.distance_grad(&p);
        let mut fd = [0.0; 3];
        for k in 0..3 {
            let mut plus = p;
            plus[k] += 1e-6;
            let mut minus = p;
            minus[k] -= 1e-6;
            fd[k] = (e.distance(&plus) - e.distance(&minus)) / 2e-6;
        }
        // The analytic gradient is the unit normal; the finite difference of
        // the scaled implicit function shares its direction.
        let fdn = (fd[0] * fd[0] + fd[1] * fd[1] + fd[2] * fd[2]).sqrt();
        for k in 0..3 {
            assert!((g[k] - fd[k] / fdn).abs() < 1e-5, "{g:?} vs {fd:?}");
        }
        assert_eq!(e.distance_grad(&[0.0; 3]), [0.0; 3]);
    }

    #[test]
    fn named_rejects() {
        assert!(Ellipsoid::new([0.0; 3], [1.0, 0.0, 1.0]).is_err());
        assert!(Ellipsoid::new([F::INFINITY, 0.0, 0.0], [1.0; 3]).is_err());
    }
}
