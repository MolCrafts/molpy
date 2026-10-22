//! A finite, capped cylinder.

use super::primitives::Region;
use crate::op::{F, Fnx3};
use ndarray::Array2;

/// A solid cylinder of radius `r` and length `L`, closed at both ends,
/// starting at `base` and running along the unit `axis`.
///
/// With `t = (x − base) · axis` and `x⊥ = (x − base) − t · axis`,
/// `distance(x) = max(‖x⊥‖ − r, −t, t − L)`: the perpendicular distance to
/// the nearest of the wall and the two caps when the point is inside or past
/// exactly one of them, and the larger of the two excesses past a rim (a
/// lower bound on the Euclidean distance there). The gradient is the radial
/// unit vector when the wall wins and `∓axis` when a cap wins; on the axis
/// the radial direction is undefined and reported as zero.
#[derive(Debug, Clone)]
pub struct Cylinder {
    base: [F; 3],
    axis: [F; 3],
    radius: F,
    length: F,
}

impl Cylinder {
    /// Cylinder from the centre of one cap (`base`, Å) along `axis` (any
    /// non-zero length, normalised here), with `radius` and `length` in Å.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `axis` is zero or not finite, or if `radius` or
    /// `length` is not positive.
    pub fn new(base: [F; 3], axis: [F; 3], radius: F, length: F) -> Result<Self, String> {
        let len = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if !(len > 0.0 && len.is_finite()) {
            return Err("Cylinder: axis must be a finite non-zero vector".to_string());
        }
        if !(radius > 0.0 && radius.is_finite()) {
            return Err(format!("Cylinder: radius must be > 0, got {radius}"));
        }
        if !(length > 0.0 && length.is_finite()) {
            return Err(format!("Cylinder: length must be > 0, got {length}"));
        }
        if base.iter().any(|b| !b.is_finite()) {
            return Err("Cylinder: base must be finite".to_string());
        }
        Ok(Self {
            base,
            axis: [axis[0] / len, axis[1] / len, axis[2] / len],
            radius,
            length,
        })
    }

    /// Centre of the first cap, Å.
    pub fn base(&self) -> [F; 3] {
        self.base
    }

    /// Unit axis direction.
    pub fn axis(&self) -> [F; 3] {
        self.axis
    }

    /// Radius, Å.
    pub fn radius(&self) -> F {
        self.radius
    }

    /// Length between the caps, Å.
    pub fn length(&self) -> F {
        self.length
    }

    /// `(distance, gradient)` in one pass over the three terms.
    fn nearest(&self, point: &[F; 3]) -> (F, [F; 3]) {
        let v = [
            point[0] - self.base[0],
            point[1] - self.base[1],
            point[2] - self.base[2],
        ];
        let t = v[0] * self.axis[0] + v[1] * self.axis[1] + v[2] * self.axis[2];
        let perp = [
            v[0] - t * self.axis[0],
            v[1] - t * self.axis[1],
            v[2] - t * self.axis[2],
        ];
        let rho = (perp[0] * perp[0] + perp[1] * perp[1] + perp[2] * perp[2]).sqrt();
        let mut best = (rho - self.radius, [0.0; 3]);
        if rho > 1e-12 {
            best.1 = [perp[0] / rho, perp[1] / rho, perp[2] / rho];
        }
        let a = self.axis;
        if -t > best.0 {
            best = (-t, [-a[0], -a[1], -a[2]]);
        }
        if t - self.length > best.0 {
            best = (t - self.length, a);
        }
        best
    }
}

impl Region for Cylinder {
    /// AABB of the two end discs.
    fn bounds(&self) -> Fnx3 {
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            let reach = self.radius * (1.0 - self.axis[d] * self.axis[d]).max(0.0).sqrt();
            let far = self.base[d] + self.length * self.axis[d];
            b[[d, 0]] = self.base[d].min(far) - reach;
            b[[d, 1]] = self.base[d].max(far) + reach;
        }
        b
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.nearest(point).0
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        self.nearest(point).1
    }
}

#[cfg(test)]
mod tests {
    use super::super::primitives::tests::check_region_contract;
    use super::*;

    fn z_cylinder() -> Cylinder {
        Cylinder::new([1.0, 1.0, 0.0], [0.0, 0.0, 3.0], 2.0, 5.0).unwrap()
    }

    #[test]
    fn wall_and_cap_goldens() {
        let c = z_cylinder();
        assert_eq!(c.axis(), [0.0, 0.0, 1.0]);
        // On the axis, mid-height: the wall is 2 away, the caps 2.5.
        assert_eq!(c.distance(&[1.0, 1.0, 2.5]), -2.0);
        // On the wall.
        assert_eq!(c.distance(&[3.0, 1.0, 2.5]), 0.0);
        assert_eq!(c.distance_grad(&[3.0, 1.0, 2.5]), [1.0, 0.0, 0.0]);
        // Past the top cap, inside the wall radius.
        assert_eq!(c.distance(&[1.0, 1.0, 7.0]), 2.0);
        assert_eq!(c.distance_grad(&[1.0, 1.0, 7.0]), [0.0, 0.0, 1.0]);
        // Below the bottom cap.
        assert_eq!(c.distance(&[1.0, 2.0, -1.5]), 1.5);
        assert_eq!(c.distance_grad(&[1.0, 2.0, -1.5]), [0.0, 0.0, -1.0]);
        // Past a rim: the larger excess wins (lower bound on the Euclidean 5).
        assert_eq!(c.distance(&[5.0, 1.0, 8.0]), 3.0);
        let b = c.bounds();
        assert_eq!(b[[0, 0]], -1.0);
        assert_eq!(b[[0, 1]], 3.0);
        assert_eq!(b[[2, 0]], 0.0);
        assert_eq!(b[[2, 1]], 5.0);
    }

    #[test]
    fn oblique_axis_keeps_the_contract() {
        let c = Cylinder::new([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 1.5, 4.0).unwrap();
        let probes: Vec<[F; 3]> = (0..40)
            .map(|i| {
                let t = i as F / 39.0;
                [t * 4.0 - 0.5, t * 3.0 + 0.2, 0.7 + t * 1.9]
            })
            .collect();
        check_region_contract(&c, &probes);
        // The bounds contain every point the contract found inside.
        let b = c.bounds();
        for p in &probes {
            if c.contains_point(p) {
                for d in 0..3 {
                    assert!(p[d] >= b[[d, 0]] - 1e-12 && p[d] <= b[[d, 1]] + 1e-12);
                }
            }
        }
    }

    #[test]
    fn named_rejects() {
        assert!(Cylinder::new([0.0; 3], [0.0; 3], 1.0, 1.0).is_err());
        assert!(Cylinder::new([0.0; 3], [0.0, 0.0, 1.0], 0.0, 1.0).is_err());
        assert!(Cylinder::new([0.0; 3], [0.0, 0.0, 1.0], 1.0, -1.0).is_err());
    }
}
