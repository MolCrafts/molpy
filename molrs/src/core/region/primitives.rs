//! Geometric regions: the [`Region`] trait, the primitive shapes and their Boolean composition.

use crate::op::to_mat3;
use crate::op::{F, F3, Fnx3};
use crate::op::{det3, inv3};
use ndarray::{Array1, Array2, array};
use std::sync::Arc;

/// Step of the central finite difference behind the default
/// [`Region::distance_grad`], Å.
const FD_STEP: F = 1e-6;

/// A solid with a signed distance to its boundary.
///
/// A region is a solid. Every shape describes its *inside* ([`Sphere`],
/// [`Cuboid`], [`Parallelepiped`], [`HalfSpace`](crate::core::HalfSpace),
/// [`Cylinder`](crate::core::Cylinder), [`Ellipsoid`](crate::core::Ellipsoid),
/// [`Polyhedron`](crate::core::Polyhedron),
/// [`SphereUnion`](crate::core::SphereUnion)); outside, shells and voids are
/// Boolean compositions ([`AndRegion`], [`OrRegion`], [`NotRegion`]): "outside a
/// sphere" is [`NotRegion`] over a [`Sphere`], a shell is a sphere `&` the
/// complement of a smaller one. There are no `Inside*` / `Outside*` pairs.
///
/// Pure geometry — **not** the periodic simulation cell, which is
/// [`SimBox`](crate::core::SimBox) (PBC / MIC / wrap); it is not a region type.
///
/// Type layout conventions:
/// - Points: N×3 row-major [`Fnx3`], each row is `(x, y, z)`, Å.
/// - Bounds: 3×2 [`Fnx3`], col 0 = min, col 1 = max, rows = x/y/z.
///
/// `distance` is the one method a shape has to write. It is **negative
/// inside, positive outside, zero on the boundary**, and its sign together
/// with the direction of [`distance_grad`](Self::distance_grad) is what every
/// consumer relies on; the magnitude is Euclidean where a shape can afford it
/// and documented where it is not (a lower bound is always acceptable).
/// A shape contains `distance <= 0`, so its boundary belongs to it. The
/// combinators are exact set algebra on containment, not on the distance:
/// [`NotRegion`] contains what its operand does not (so a boundary point is in
/// the region and never in its complement — the complement is open), and
/// [`AndRegion`] / [`OrRegion`] contain what both / either operand does. Off
/// the boundary that is the same as testing their distance; on it, it is what
/// keeps `R` and `~R` disjoint.
///
/// Trait objects are `Arc<dyn Region + Send + Sync>`: the combinators take
/// them, and a region is shared across threads by evaluators such as a
/// packer's rayon loop.
pub trait Region: Send + Sync + std::fmt::Debug {
    /// Axis-aligned bounding box, Å.
    ///
    /// Layout: rows = x/y/z; col 0 = min, col 1 = max. An unbounded region
    /// reports `±∞` on its open sides.
    fn bounds(&self) -> Fnx3;

    /// Signed distance from `point` to the boundary, Å: negative inside,
    /// positive outside, zero on it.
    fn distance(&self, point: &[F; 3]) -> F;

    /// Gradient of [`distance`](Self::distance) at `point` (Å/Å, a direction).
    ///
    /// Points from the inside toward the outside. The default is a central
    /// finite difference with step `1e-6` Å; shapes with a closed form
    /// override it.
    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let mut g = [0.0; 3];
        for k in 0..3 {
            let mut plus = *point;
            plus[k] += FD_STEP;
            let mut minus = *point;
            minus[k] -= FD_STEP;
            g[k] = (self.distance(&plus) - self.distance(&minus)) / (2.0 * FD_STEP);
        }
        g
    }

    /// Does the region repeat along `shift` — is it the same set of points
    /// after being translated by it?
    ///
    /// A half-space `z > 5` repeats along any shift in x or y, and not along
    /// one in z. A sphere, a box, a mesh — anything bounded — repeats along
    /// nothing: translating it moves it.
    ///
    /// This is what makes a region usable under periodic boundaries along a
    /// lattice vector `a`. A region that neither repeats along `a` nor is
    /// contained in one image has no well-defined inside there: which side of
    /// it a point falls on would depend on which image the point was wrapped
    /// into.
    ///
    /// The default is `false` — a shape that has not thought about it does
    /// not claim to repeat. Overriding it is a claim about the *set*, not
    /// about the distance function: `distance` may legitimately differ while
    /// the region is the same.
    fn repeats_along(&self, shift: [F; 3]) -> bool {
        let _ = shift;
        false
    }

    /// Whether `point` is inside the region: `distance(point) <= 0` for a
    /// shape.
    ///
    /// Override only with a test that provably agrees with the distance off
    /// the boundary (`distance != 0`); the combinators override it to keep
    /// their set algebra exact on the boundary (see the trait docs).
    fn contains_point(&self, point: &[F; 3]) -> bool {
        self.distance(point) <= 0.0
    }

    /// Batched [`contains_point`](Self::contains_point) over the rows of an
    /// N×3 array.
    ///
    /// # Panics
    ///
    /// Panics if `points` does not have exactly 3 columns.
    fn contains(&self, points: &Fnx3) -> Array1<bool> {
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");
        points
            .rows()
            .into_iter()
            .map(|row| self.contains_point(&[row[0], row[1], row[2]]))
            .collect()
    }
}

fn aabb(lo: [F; 3], hi: [F; 3]) -> Fnx3 {
    let mut b = Array2::zeros((3, 2));
    for d in 0..3 {
        b[[d, 0]] = lo[d];
        b[[d, 1]] = hi[d];
    }
    b
}

/// A solid sphere.
///
/// `distance(x) = ‖x − c‖ − r` (exact Euclidean); the gradient is the radial
/// unit vector, zero at the centre.
#[derive(Debug, Clone)]
pub struct Sphere {
    /// Center of the sphere, Å.
    pub center: F3,
    /// Radius of the sphere, Å.
    pub radius: F,
}

impl Sphere {
    /// Creates a sphere with a given center and radius (Å).
    pub fn new(center: F3, radius: F) -> Self {
        Self { center, radius }
    }

    /// Creates a sphere centered at the origin with the given radius (Å).
    pub fn with_radius(radius: F) -> Self {
        Self {
            center: Array1::zeros(3),
            radius,
        }
    }

    fn offset(&self, point: &[F; 3]) -> ([F; 3], F) {
        let d = [
            point[0] - self.center[0],
            point[1] - self.center[1],
            point[2] - self.center[2],
        ];
        (d, (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt())
    }
}

impl Region for Sphere {
    fn bounds(&self) -> Fnx3 {
        let r = self.radius;
        let c = &self.center;
        aabb(
            [c[0] - r, c[1] - r, c[2] - r],
            [c[0] + r, c[1] + r, c[2] + r],
        )
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.offset(point).1 - self.radius
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (d, norm) = self.offset(point);
        if norm < 1e-12 {
            [0.0; 3]
        } else {
            [d[0] / norm, d[1] / norm, d[2] / norm]
        }
    }
}

/// An axis-aligned cuboid (box): inside when
/// `origin[d] <= p[d] <= origin[d] + lengths[d]` on every axis.
///
/// `distance(x) = max_k max(o_k − x_k, x_k − o_k − L_k)`: the perpendicular
/// distance to the nearest face plane when the point is inside or past one
/// face, and the largest single-axis excess past an edge or corner (a lower
/// bound on the Euclidean distance there). The gradient is `±e_k` of the
/// winning face.
#[derive(Debug, Clone)]
pub struct Cuboid {
    /// Minimum corner (lower bound on each axis), Å.
    pub origin: F3,
    /// Edge lengths along x, y, z, Å.
    pub lengths: F3,
}

impl Cuboid {
    /// Creates a cuboid with the given origin (min corner) and edge lengths (Å).
    pub fn new(origin: F3, lengths: F3) -> Self {
        Self { origin, lengths }
    }

    /// The winning face: `(distance, axis, sign)` with `sign = -1` for the
    /// low face and `+1` for the high face of `axis`.
    fn nearest_face(&self, point: &[F; 3]) -> (F, usize, F) {
        let mut best = (F::NEG_INFINITY, 0usize, 1.0 as F);
        for (k, &xk) in point.iter().enumerate() {
            let below = self.origin[k] - xk;
            if below > best.0 {
                best = (below, k, -1.0);
            }
            let above = xk - self.origin[k] - self.lengths[k];
            if above > best.0 {
                best = (above, k, 1.0);
            }
        }
        best
    }
}

impl Region for Cuboid {
    fn bounds(&self) -> Fnx3 {
        let o = &self.origin;
        let l = &self.lengths;
        aabb([o[0], o[1], o[2]], [o[0] + l[0], o[1] + l[1], o[2] + l[2]])
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.nearest_face(point).0
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (_, axis, sign) = self.nearest_face(point);
        let mut g = [0.0; 3];
        g[axis] = sign;
        g
    }
}

/// A general parallelepiped (oblique box) defined by an origin and three
/// edge vectors (columns of `H`).
///
/// This is the geometric counterpart of a triclinic simulation cell volume —
/// **without** PBC, wrap, or MIC. For the periodic cell, use
/// [`crate::core::SimBox`].
///
/// A point `p` is inside when the fractional coordinates
/// `f = H⁻¹ · (p − origin)` satisfy `0 ≤ f_d ≤ 1` on every axis.
///
/// `distance` is measured perpendicular to the bounding lattice planes, in
/// Å: with `s_k = 1 / ‖row_k(H⁻¹)‖` the spacing of the `k`-th plane pair,
/// `distance(x) = max_k max(−f_k · s_k, (f_k − 1) · s_k)`. That makes the
/// value comparable across tilted cells and equal to the [`Cuboid`] rule on an
/// orthorhombic `H`. The gradient is the unit outward normal of the winning
/// plane, `±row_k(H⁻¹) / ‖row_k(H⁻¹)‖`.
///
/// Axis-aligned boxes prefer [`Cuboid`] (cheaper, no inverse).
#[derive(Debug, Clone)]
pub struct Parallelepiped {
    /// One corner of the parallelepiped, Å.
    origin: F3,
    /// Edge matrix `H` (columns are the three edge vectors), Å.
    h: Fnx3,
    /// Cached `H⁻¹`.
    inv: Fnx3,
    /// Interplanar spacing of each face pair, Å: turns a fractional offset
    /// into a perpendicular distance.
    spacing: [F; 3],
    /// Unit outward normal of each face pair: the normalised rows of `H⁻¹`.
    normal: [[F; 3]; 3],
}

impl Parallelepiped {
    /// Construct from edge matrix `H` (columns = edges, Å) and `origin` (Å).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `H` is singular (zero volume) or not finite.
    pub fn new(h: Fnx3, origin: F3) -> Result<Self, String> {
        let inv: Fnx3 =
            ndarray::arr2(&inv3(&to_mat3(h.view())).ok_or_else(|| {
                "Parallelepiped: singular edge matrix H (zero volume)".to_string()
            })?);
        let mut spacing = [0.0; 3];
        let mut normal = [[0.0; 3]; 3];
        for k in 0..3 {
            let row = [inv[[k, 0]], inv[[k, 1]], inv[[k, 2]]];
            let norm = (row[0] * row[0] + row[1] * row[1] + row[2] * row[2]).sqrt();
            if !(norm > 0.0 && norm.is_finite()) {
                return Err("Parallelepiped: edge matrix H is not finite".to_string());
            }
            spacing[k] = 1.0 / norm;
            normal[k] = [row[0] / norm, row[1] / norm, row[2] / norm];
        }
        Ok(Self {
            origin,
            h,
            inv,
            spacing,
            normal,
        })
    }

    /// Cubic region of edge length `a` (Å) with the given origin (min corner).
    ///
    /// # Errors
    ///
    /// Returns `Err` if `a <= 0`.
    pub fn cube(a: F, origin: F3) -> Result<Self, String> {
        if a <= 0.0 {
            return Err(format!(
                "Parallelepiped::cube: edge length must be > 0, got {a}"
            ));
        }
        let h = array![[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]];
        Self::new(h, origin)
    }

    /// Axis-aligned orthorhombic region with the given edge lengths (Å).
    ///
    /// Prefer [`Cuboid`] when you only need axis-aligned containment — this
    /// constructor exists so a single `Parallelepiped` API covers cube → ortho
    /// → triclinic.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `lengths` is not three positive numbers.
    pub fn ortho(lengths: F3, origin: F3) -> Result<Self, String> {
        if lengths.len() != 3 {
            return Err(format!(
                "Parallelepiped::ortho: lengths must have length 3, got {}",
                lengths.len()
            ));
        }
        if (0..3).any(|d| lengths[d] <= 0.0) {
            return Err("Parallelepiped::ortho: all edge lengths must be > 0".into());
        }
        let h = array![
            [lengths[0], 0.0, 0.0],
            [0.0, lengths[1], 0.0],
            [0.0, 0.0, lengths[2]]
        ];
        Self::new(h, origin)
    }

    /// Construct from three explicit edge vectors `a`, `b`, `c` and `origin` (Å).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the edges are coplanar.
    pub fn from_edges(a: [F; 3], b: [F; 3], c: [F; 3], origin: F3) -> Result<Self, String> {
        let h = array![[a[0], b[0], c[0]], [a[1], b[1], c[1]], [a[2], b[2], c[2]]];
        Self::new(h, origin)
    }

    /// Origin corner, Å.
    pub fn origin(&self) -> &F3 {
        &self.origin
    }

    /// Edge matrix `H` (columns are edge vectors), Å.
    pub fn h(&self) -> &Fnx3 {
        &self.h
    }

    /// Signed volume `det(H)`, Å³.
    pub fn volume(&self) -> F {
        det3(&to_mat3(self.h.view()))
    }

    fn frac_of(&self, point: &[F; 3]) -> [F; 3] {
        let dr = [
            point[0] - self.origin[0],
            point[1] - self.origin[1],
            point[2] - self.origin[2],
        ];
        let mut f = [0.0; 3];
        for (k, fk) in f.iter_mut().enumerate() {
            *fk = self.inv[[k, 0]] * dr[0] + self.inv[[k, 1]] * dr[1] + self.inv[[k, 2]] * dr[2];
        }
        f
    }

    /// The winning bounding plane: `(distance, axis, sign)` with `sign = -1`
    /// past the `f = 0` plane and `+1` past the `f = 1` plane of `axis`.
    fn nearest_face(&self, point: &[F; 3]) -> (F, usize, F) {
        let f = self.frac_of(point);
        let mut best = (F::NEG_INFINITY, 0usize, 1.0 as F);
        for (k, &fk) in f.iter().enumerate() {
            let below = -fk * self.spacing[k];
            if below > best.0 {
                best = (below, k, -1.0);
            }
            let above = (fk - 1.0) * self.spacing[k];
            if above > best.0 {
                best = (above, k, 1.0);
            }
        }
        best
    }
}

impl Region for Parallelepiped {
    fn bounds(&self) -> Fnx3 {
        // AABB of the eight corners: origin + Σ ε_i · edge_i, ε ∈ {0,1}.
        let o = [self.origin[0], self.origin[1], self.origin[2]];
        let edge = |i: usize| [self.h[[0, i]], self.h[[1, i]], self.h[[2, i]]];
        let (e0, e1, e2) = (edge(0), edge(1), edge(2));
        let mut lo = o;
        let mut hi = o;
        for mask in 0u8..8 {
            for d in 0..3 {
                let p = o[d]
                    + if mask & 1 != 0 { e0[d] } else { 0.0 }
                    + if mask & 2 != 0 { e1[d] } else { 0.0 }
                    + if mask & 4 != 0 { e2[d] } else { 0.0 };
                lo[d] = lo[d].min(p);
                hi[d] = hi[d].max(p);
            }
        }
        aabb(lo, hi)
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.nearest_face(point).0
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (_, axis, sign) = self.nearest_face(point);
        let n = self.normal[axis];
        [sign * n[0], sign * n[1], sign * n[2]]
    }
}

/// Intersection of two regions (AND): inside iff inside both.
///
/// `distance = max(d_a, d_b)`; the gradient is that of the larger term.
#[derive(Debug, Clone)]
pub struct AndRegion {
    a: Arc<dyn Region + Send + Sync>,
    b: Arc<dyn Region + Send + Sync>,
}

impl AndRegion {
    /// Creates an intersection of two regions.
    pub fn new(a: Arc<dyn Region + Send + Sync>, b: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a, b }
    }
}

impl Region for AndRegion {
    /// An intersection repeats along a shift both members repeat along.
    fn repeats_along(&self, shift: [F; 3]) -> bool {
        self.a.repeats_along(shift) && self.b.repeats_along(shift)
    }

    fn bounds(&self) -> Fnx3 {
        // Intersection bounds: max of mins, min of maxs
        let a_bounds = self.a.bounds();
        let b_bounds = self.b.bounds();
        let mut result = Array2::zeros((3, 2));
        for d in 0..3 {
            result[[d, 0]] = a_bounds[[d, 0]].max(b_bounds[[d, 0]]);
            result[[d, 1]] = a_bounds[[d, 1]].min(b_bounds[[d, 1]]);
        }
        result
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.a.distance(point).max(self.b.distance(point))
    }

    /// In both operands, by their own containment.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        self.a.contains_point(point) && self.b.contains_point(point)
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        if self.a.distance(point) >= self.b.distance(point) {
            self.a.distance_grad(point)
        } else {
            self.b.distance_grad(point)
        }
    }
}

/// Complement of a region (NOT): inside iff not inside the original.
///
/// Containment is the exact complement: a point on the original's boundary
/// (`d_a == 0`) is the original's, so it is not in the complement, which is
/// therefore open. `distance = −d_a`; the gradient is negated. `bounds` reports the inner
/// region's box, since the complement is unbounded.
#[derive(Debug, Clone)]
pub struct NotRegion {
    a: Arc<dyn Region + Send + Sync>,
}

impl NotRegion {
    /// Creates a complement of a region.
    pub fn new(a: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a }
    }
}

impl Region for NotRegion {
    /// A complement repeats exactly where the region it complements does.
    fn repeats_along(&self, shift: [F; 3]) -> bool {
        self.a.repeats_along(shift)
    }

    fn bounds(&self) -> Fnx3 {
        self.a.bounds()
    }

    fn distance(&self, point: &[F; 3]) -> F {
        -self.a.distance(point)
    }

    /// Not in the operand: `distance <= 0` here would also hold the
    /// operand's boundary, putting it in both a region and its complement.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        !self.a.contains_point(point)
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let g = self.a.distance_grad(point);
        [-g[0], -g[1], -g[2]]
    }
}

/// Union of two regions (OR): inside iff inside either.
///
/// `distance = min(d_a, d_b)`; the gradient is that of the smaller term.
#[derive(Debug, Clone)]
pub struct OrRegion {
    a: Arc<dyn Region + Send + Sync>,
    b: Arc<dyn Region + Send + Sync>,
}

impl OrRegion {
    /// Creates a union of two regions.
    pub fn new(a: Arc<dyn Region + Send + Sync>, b: Arc<dyn Region + Send + Sync>) -> Self {
        Self { a, b }
    }
}

impl Region for OrRegion {
    /// A union repeats along a shift both members repeat along.
    fn repeats_along(&self, shift: [F; 3]) -> bool {
        self.a.repeats_along(shift) && self.b.repeats_along(shift)
    }

    fn bounds(&self) -> Fnx3 {
        // Union bounds: min of mins, max of maxs
        let a_bounds = self.a.bounds();
        let b_bounds = self.b.bounds();
        let mut result = Array2::zeros((3, 2));
        for d in 0..3 {
            result[[d, 0]] = a_bounds[[d, 0]].min(b_bounds[[d, 0]]);
            result[[d, 1]] = a_bounds[[d, 1]].max(b_bounds[[d, 1]]);
        }
        result
    }

    fn distance(&self, point: &[F; 3]) -> F {
        self.a.distance(point).min(self.b.distance(point))
    }

    /// In either operand, by its own containment.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        self.a.contains_point(point) || self.b.contains_point(point)
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        if self.a.distance(point) <= self.b.distance(point) {
            self.a.distance_grad(point)
        } else {
            self.b.distance_grad(point)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Every shape and combinator must keep `contains_point` equal to
    /// `distance < 0` off the boundary (`distance == 0`, where a complement
    /// is open and a shape closed), its analytic gradient equal to the finite
    /// difference (off the ties where the max/min switches branch), and
    /// every `repeats_along` claim true of the set it describes.
    pub(crate) fn check_region_contract(region: &dyn Region, probes: &[[F; 3]]) {
        check_repeat_claims(region, probes);
        for p in probes {
            let d = region.distance(p);
            if d != 0.0 {
                assert_eq!(
                    region.contains_point(p),
                    d < 0.0,
                    "contains_point disagrees with distance at {p:?} (d = {d})"
                );
            }
            let g = region.distance_grad(p);
            let mut fd = [0.0; 3];
            for k in 0..3 {
                let mut plus = *p;
                plus[k] += FD_STEP;
                let mut minus = *p;
                minus[k] -= FD_STEP;
                fd[k] = (region.distance(&plus) - region.distance(&minus)) / (2.0 * FD_STEP);
            }
            for k in 0..3 {
                assert!(
                    (g[k] - fd[k]).abs() < 1e-5,
                    "gradient {g:?} vs finite difference {fd:?} at {p:?}"
                );
            }
        }
    }

    /// A claim to repeat along a shift is a claim about membership: every
    /// point must keep its side when translated by it, and by a few
    /// multiples of it. A shape that does not claim to repeat is not
    /// checked — the claim is what carries the obligation.
    pub(crate) fn check_repeat_claims(region: &dyn Region, probes: &[[F; 3]]) {
        let shifts = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.7, -0.3, 0.2],
            [-2.4, 0.0, 0.0],
        ];
        for shift in shifts {
            if !region.repeats_along(shift) {
                continue;
            }
            for p in probes {
                for n in [1.0, 2.0, -3.0] {
                    let q = [
                        p[0] + n * shift[0],
                        p[1] + n * shift[1],
                        p[2] + n * shift[2],
                    ];
                    assert_eq!(
                        region.contains_point(p),
                        region.contains_point(&q),
                        "claims to repeat along {shift:?}, but {p:?} and {q:?} \
                         (shifted by {n}x) are on opposite sides"
                    );
                }
            }
        }
    }

    /// A bounded shape repeats along nothing: translating it moves it.
    #[test]
    fn a_bounded_shape_repeats_along_nothing() {
        let sphere = Sphere::new(Array1::from_vec(vec![0.0, 0.0, 0.0]), 2.0);
        let cuboid = Cuboid::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![2.0, 2.0, 2.0]),
        );
        for shift in [[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.5, 0.5, 0.5]] {
            assert!(!sphere.repeats_along(shift), "sphere, {shift:?}");
            assert!(!cuboid.repeats_along(shift), "cuboid, {shift:?}");
        }
    }

    /// A composition repeats only where every member does — and a complement
    /// repeats exactly where the region it complements does, since swapping
    /// inside for outside does not move the boundary.
    #[test]
    fn a_composition_repeats_where_every_member_does() {
        use super::super::half_space::HalfSpace;

        let slab_lo: Arc<dyn Region + Send + Sync> =
            Arc::new(HalfSpace::new([0.0, 0.0, 1.0], [0.0, 0.0, 5.0]).expect("plane"));
        let slab_hi: Arc<dyn Region + Send + Sync> =
            Arc::new(HalfSpace::new([0.0, 0.0, -1.0], [0.0, 0.0, 1.0]).expect("plane"));
        let wall: Arc<dyn Region + Send + Sync> =
            Arc::new(HalfSpace::new([1.0, 0.0, 0.0], [3.0, 0.0, 0.0]).expect("plane"));
        let ball: Arc<dyn Region + Send + Sync> =
            Arc::new(Sphere::new(Array1::from_vec(vec![0.0, 0.0, 0.0]), 2.0));

        // A slab: two planes with the same normal, so it still repeats in x/y.
        let slab = AndRegion::new(Arc::clone(&slab_lo), slab_hi);
        assert!(slab.repeats_along([2.0, 0.0, 0.0]));
        assert!(slab.repeats_along([0.0, 1.0, 0.0]));
        assert!(!slab.repeats_along([0.0, 0.0, 1.0]));

        // Add a wall across x and the x direction stops repeating.
        let boxed_in = AndRegion::new(Arc::new(slab), wall);
        assert!(!boxed_in.repeats_along([2.0, 0.0, 0.0]));
        assert!(boxed_in.repeats_along([0.0, 1.0, 0.0]));

        // A bounded member removes every direction, in a union as in an
        // intersection.
        let with_ball = OrRegion::new(Arc::clone(&slab_lo), ball);
        assert!(!with_ball.repeats_along([0.0, 1.0, 0.0]));

        // The complement keeps the claim.
        let outside = NotRegion::new(Arc::clone(&slab_lo));
        assert!(outside.repeats_along([0.0, 1.0, 0.0]));
        assert!(!outside.repeats_along([0.0, 0.0, 1.0]));

        let probes = sweep();
        check_repeat_claims(&boxed_in, &probes);
        check_repeat_claims(&with_ball, &probes);
        check_repeat_claims(&outside, &probes);
    }

    fn sweep() -> Vec<[F; 3]> {
        let mut probes = Vec::new();
        for i in 0..7 {
            for j in 0..5 {
                let t = i as F / 6.0;
                let u = j as F / 4.0;
                probes.push([t * 4.0 - 0.7, u * 3.1 - 0.3, 0.9 + 0.31 * t + 0.17 * u]);
            }
        }
        probes
    }

    #[test]
    fn cuboid_contains_and_bounds() {
        let c = Cuboid::new(
            Array1::from_vec(vec![0.0, 0.0, 0.0]),
            Array1::from_vec(vec![2.0, 2.0, 2.0]),
        );
        let pts = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 1.0, 1.0, // inside
                3.0, 1.0, 1.0, // outside (x > 2)
                2.0, 2.0, 2.0, // on the max corner (inclusive)
                -0.1, 0.0, 0.0, // outside (x < 0)
            ],
        )
        .unwrap();
        let mask = c.contains(&pts);
        assert_eq!(mask.to_vec(), vec![true, false, true, false]);
        assert!(c.contains_point(&[0.5, 1.0, 2.0]));
        assert!(!c.contains_point(&[2.5, 1.0, 1.0]));
        let b = c.bounds();
        assert_eq!(b[[0, 0]], 0.0);
        assert_eq!(b[[0, 1]], 2.0);
    }

    #[test]
    fn cuboid_distance_goldens() {
        let c = Cuboid::new(Array1::zeros(3), Array1::from_vec(vec![2.0, 4.0, 6.0]));
        // Inside: minus the distance to the nearest face (x = 0 at 0.5).
        assert!((c.distance(&[0.5, 2.0, 3.0]) + 0.5).abs() < 1e-12);
        // Past one face: the perpendicular excess.
        assert!((c.distance(&[3.0, 2.0, 3.0]) - 1.0).abs() < 1e-12);
        assert_eq!(c.distance_grad(&[3.0, 2.0, 3.0]), [1.0, 0.0, 0.0]);
        assert_eq!(c.distance_grad(&[-1.0, 2.0, 3.0]), [-1.0, 0.0, 0.0]);
        // Past a corner: the largest single-axis excess, a lower bound.
        assert!((c.distance(&[3.0, 6.0, 3.0]) - 2.0).abs() < 1e-12);
        assert_eq!(c.distance_grad(&[3.0, 6.0, 3.0]), [0.0, 1.0, 0.0]);
        // On a face.
        assert_eq!(c.distance(&[2.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn parallelepiped_cube_matches_cuboid_closed() {
        let p = Parallelepiped::cube(2.0, Array1::zeros(3)).unwrap();
        assert!(p.contains_point(&[1.0, 1.0, 1.0]));
        assert!(p.contains_point(&[0.0, 0.0, 0.0]));
        // The boundary belongs to the region: `distance <= 0`.
        assert!(p.contains_point(&[2.0, 0.0, 0.0]));
        assert!(!p.contains_point(&[2.01, 0.0, 0.0]));
        assert!(!p.contains_point(&[-0.01, 0.0, 0.0]));
        let b = p.bounds();
        assert!((b[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((b[[0, 1]] - 2.0).abs() < 1e-12);
        assert!((p.volume() - 8.0).abs() < 1e-12);
        let c = Cuboid::new(Array1::zeros(3), Array1::from_vec(vec![2.0; 3]));
        for q in sweep() {
            assert!((p.distance(&q) - c.distance(&q)).abs() < 1e-12, "at {q:?}");
        }
    }

    #[test]
    fn parallelepiped_skewed_contains_and_aabb() {
        // Edges: a=(2,0,0), b=(1,2,0), c=(0,0,3) — a skewed prism in xy.
        let p = Parallelepiped::from_edges(
            [2.0, 0.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            Array1::zeros(3),
        )
        .unwrap();
        // Interior fractional (0.25, 0.25, 0.25) → cart = 0.25*a + 0.25*b + 0.25*c
        // = (0.75, 0.5, 0.75)
        assert!(p.contains_point(&[0.75, 0.5, 0.75]));
        // Just outside along a
        assert!(!p.contains_point(&[2.1, 0.0, 0.0]));
        // AABB spans x in [0, 3], y in [0, 2], z in [0, 3]
        let b = p.bounds();
        assert!((b[[0, 0]] - 0.0).abs() < 1e-12);
        assert!((b[[0, 1]] - 3.0).abs() < 1e-12);
        assert!((b[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((b[[1, 1]] - 2.0).abs() < 1e-12);
        assert!((b[[2, 1]] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn parallelepiped_distance_is_perpendicular_angstrom() {
        // Hexagonal cell a = b = c = 26 Å, γ = 120°: the `a` face pair is
        // 26·sin(120°) apart, so fractional 1.1 along `a` is 0.1 of that.
        let gamma = 120.0_f64.to_radians();
        let p = Parallelepiped::from_edges(
            [26.0, 0.0, 0.0],
            [26.0 * gamma.cos(), 26.0 * gamma.sin(), 0.0],
            [0.0, 0.0, 26.0],
            Array1::zeros(3),
        )
        .unwrap();
        let h = p.h().clone();
        let frac = array![1.1, 0.5, 0.5];
        let x = h.dot(&frac);
        let d = p.distance(&[x[0], x[1], x[2]]);
        assert!((d - 0.1 * 26.0 * gamma.sin()).abs() < 1e-9, "d = {d}");
        // The gradient is the unit outward normal of the `a` face pair.
        let g = p.distance_grad(&[x[0], x[1], x[2]]);
        let norm = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);
        // Interior points are negative with the same perpendicular metric.
        let inside = h.dot(&array![0.5, 0.5, 0.5]);
        let di = p.distance(&[inside[0], inside[1], inside[2]]);
        assert!((di + 0.5 * 26.0 * gamma.sin()).abs() < 1e-9, "d = {di}");
    }

    #[test]
    fn parallelepiped_singular_is_err() {
        let h = array![[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]];
        assert!(Parallelepiped::new(h, Array1::zeros(3)).is_err());
    }

    #[test]
    fn sphere_bounds_are_correct() {
        let s = Sphere::new(Array1::from_vec(vec![1.0, 2.0, 3.0]), 2.0);
        let b = s.bounds();
        // Row-major [ [min_x,max_x], [min_y,max_y], [min_z,max_z] ]
        assert_eq!(b[[0, 0]], -1.0);
        assert_eq!(b[[1, 0]], 0.0);
        assert_eq!(b[[2, 0]], 1.0);
        assert_eq!(b[[0, 1]], 3.0);
        assert_eq!(b[[1, 1]], 4.0);
        assert_eq!(b[[2, 1]], 5.0);
    }

    #[test]
    fn sphere_contains_points() {
        let s = Sphere::with_radius(2.0);
        let pts: Fnx3 = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.0, 0.0, 0.0, // inside (center)
                2.0, 0.0, 0.0, // on surface
                2.1, 0.0, 0.0, // outside
            ],
        )
        .unwrap();
        let mask = s.contains(&pts);
        assert_eq!(mask.len(), 3);
        assert!(mask[0]);
        assert!(mask[1]);
        assert!(!mask[2]);
    }

    #[test]
    fn sphere_distance_goldens() {
        let s = Sphere::with_radius(2.0);
        assert_eq!(s.distance(&[0.0; 3]), -2.0);
        assert_eq!(s.distance(&[2.0, 0.0, 0.0]), 0.0);
        assert!((s.distance(&[0.0, 5.0, 0.0]) - 3.0).abs() < 1e-12);
        assert_eq!(s.distance_grad(&[0.0, 5.0, 0.0]), [0.0, 1.0, 0.0]);
        assert_eq!(s.distance_grad(&[0.0; 3]), [0.0; 3]);
    }

    #[test]
    fn contains_matches_distance_and_grad_on_every_shape() {
        let sphere: Arc<dyn Region + Send + Sync> =
            Arc::new(Sphere::new(Array1::from_vec(vec![1.0, 1.0, 1.0]), 1.3));
        let cuboid: Arc<dyn Region + Send + Sync> = Arc::new(Cuboid::new(
            Array1::from_vec(vec![0.2, 0.1, 0.4]),
            Array1::from_vec(vec![2.3, 1.7, 1.9]),
        ));
        let cell: Arc<dyn Region + Send + Sync> = Arc::new(
            Parallelepiped::from_edges(
                [2.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [0.3, 0.2, 3.0],
                Array1::from_vec(vec![0.1, 0.2, 0.3]),
            )
            .unwrap(),
        );
        let shell = AndRegion::new(
            sphere.clone(),
            Arc::new(NotRegion::new(Arc::new(Sphere::new(
                Array1::from_vec(vec![1.0, 1.0, 1.0]),
                0.6,
            )))),
        );
        let either = OrRegion::new(cuboid.clone(), cell.clone());
        let outside = NotRegion::new(cuboid.clone());
        let probes = sweep();
        for region in [
            sphere.as_ref(),
            cuboid.as_ref(),
            cell.as_ref(),
            &shell,
            &either,
            &outside,
        ] {
            check_region_contract(region, &probes);
        }
    }

    #[test]
    fn combinators_are_max_min_and_negation() {
        let a: Arc<dyn Region + Send + Sync> = Arc::new(Sphere::with_radius(2.0));
        let b: Arc<dyn Region + Send + Sync> = Arc::new(Cuboid::new(
            Array1::zeros(3),
            Array1::from_vec(vec![3.0; 3]),
        ));
        let p = [1.0, 1.0, 2.5];
        let (da, db) = (a.distance(&p), b.distance(&p));
        assert_eq!(
            AndRegion::new(a.clone(), b.clone()).distance(&p),
            da.max(db)
        );
        assert_eq!(OrRegion::new(a.clone(), b.clone()).distance(&p), da.min(db));
        assert_eq!(NotRegion::new(a.clone()).distance(&p), -da);
    }

    #[test]
    fn de_morgan_holds_for_distances() {
        let a: Arc<dyn Region + Send + Sync> = Arc::new(Sphere::with_radius(2.0));
        let b: Arc<dyn Region + Send + Sync> = Arc::new(Cuboid::new(
            Array1::zeros(3),
            Array1::from_vec(vec![3.0; 3]),
        ));
        let not_and = NotRegion::new(Arc::new(AndRegion::new(a.clone(), b.clone())));
        let or_not = OrRegion::new(
            Arc::new(NotRegion::new(a.clone())),
            Arc::new(NotRegion::new(b.clone())),
        );
        for p in sweep() {
            assert_eq!(not_and.distance(&p), or_not.distance(&p), "at {p:?}");
            assert_eq!(not_and.contains_point(&p), or_not.contains_point(&p));
        }
    }

    /// A region and its complement partition space: no point is in both and
    /// every point is in one. A point on the boundary (`distance == 0`)
    /// belongs to the region, so `~cube` refuses the atoms a lattice puts on
    /// the faces of a cube — the case that read as "`~cube` is everything".
    #[test]
    fn a_region_and_its_complement_partition_space_even_on_the_boundary() {
        let cube: Arc<dyn Region + Send + Sync> = Arc::new(Cuboid::new(
            Array1::zeros(3),
            Array1::from_vec(vec![1.5; 3]),
        ));
        let ball: Arc<dyn Region + Send + Sync> = Arc::new(Sphere::with_radius(1.0));
        // Corners, edge and face centres: every one on the cube's boundary.
        let on_cube = [
            [0.0, 0.0, 0.0],
            [1.5, 1.5, 1.5],
            [0.75, 0.75, 0.0],
            [0.0, 0.75, 0.75],
            [1.5, 0.75, 0.0],
        ];
        let probes: Vec<[F; 3]> = on_cube
            .iter()
            .copied()
            .chain([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [3.0, 3.0, 3.0]])
            .chain(sweep())
            .collect();
        let composites: Vec<Arc<dyn Region + Send + Sync>> = vec![
            cube.clone(),
            ball.clone(),
            Arc::new(AndRegion::new(cube.clone(), ball.clone())),
            Arc::new(OrRegion::new(cube.clone(), ball.clone())),
            Arc::new(AndRegion::new(
                cube.clone(),
                Arc::new(NotRegion::new(ball.clone())),
            )),
        ];
        for region in composites {
            let complement = NotRegion::new(region.clone());
            for p in &probes {
                assert_ne!(
                    region.contains_point(p),
                    complement.contains_point(p),
                    "{region:?} and its complement disagree at {p:?}"
                );
            }
        }
        let outside = NotRegion::new(cube.clone());
        for p in on_cube {
            assert!(cube.contains_point(&p), "the boundary is the cube's: {p:?}");
            assert!(!outside.contains_point(&p), "~cube holds {p:?}");
        }
        // `box & ~ball` drops the ball's surface, as `~ball` does.
        let shell = AndRegion::new(cube.clone(), Arc::new(NotRegion::new(ball)));
        assert!(!shell.contains_point(&[1.0, 0.0, 0.0]));
        assert!(shell.contains_point(&[1.2, 0.5, 0.5]));
    }

    #[test]
    fn composed_bounds_intersect_and_unite() {
        let a: Arc<dyn Region + Send + Sync> = Arc::new(Sphere::with_radius(2.0));
        let b: Arc<dyn Region + Send + Sync> = Arc::new(Cuboid::new(
            Array1::from_vec(vec![1.0, 1.0, 1.0]),
            Array1::from_vec(vec![3.0; 3]),
        ));
        let both = AndRegion::new(a.clone(), b.clone()).bounds();
        assert_eq!(both[[0, 0]], 1.0);
        assert_eq!(both[[0, 1]], 2.0);
        let either = OrRegion::new(a, b).bounds();
        assert_eq!(either[[0, 0]], -2.0);
        assert_eq!(either[[0, 1]], 4.0);
    }
}
