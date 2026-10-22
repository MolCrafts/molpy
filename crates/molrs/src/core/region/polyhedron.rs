//! A solid bounded by a watertight triangle mesh: [`Polyhedron`].

use std::f64::consts::{PI, SQRT_2};

use super::primitives::Region;
use crate::core::TriMesh;
use crate::core::bvh::{Bvh, triangle_box};
use crate::core::mesh::DEGENERATE_AREA2;
use crate::op::vec3::{add, cross, dot, norm, scale, sub};
use crate::op::{F, Fnx3};
use ndarray::Array2;

/// Points closer than this to the surface count as on it: `distance` is
/// `0` and the gradient is undefined (reported as zero).
const SURFACE_EPS: F = 1e-9;

/// Direction of the parity ray. Irrational components keep it off every
/// mesh edge and vertex of a mesh authored in rational coordinates, and no
/// zero component keeps the BVH slab test's reciprocal finite.
const PARITY_RAY: [F; 3] = [1.0, SQRT_2, PI];

/// Why a [`TriMesh`] cannot bound a [`Polyhedron`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyhedronError {
    /// The mesh has no faces.
    Empty,
    /// Vertex `index` has a non-finite coordinate.
    NonFiniteVertex {
        /// Index into [`TriMesh::vertices`].
        index: usize,
    },
    /// Face `index` has vanishing area.
    DegenerateFace {
        /// Index into [`TriMesh::faces`].
        index: usize,
    },
    /// Some directed half-edges have no opposite: the surface is open, has a
    /// duplicated face, or a flipped neighbour, so parity cannot decide
    /// inside from outside.
    NotWatertight {
        /// Number of unpaired directed half-edges.
        unpaired: usize,
    },
}

impl std::fmt::Display for PolyhedronError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "mesh has no faces"),
            Self::NonFiniteVertex { index } => {
                write!(f, "mesh vertex {index} has a non-finite coordinate")
            }
            Self::DegenerateFace { index } => write!(f, "mesh face {index} has zero area"),
            Self::NotWatertight { unpaired } => write!(
                f,
                "mesh is not watertight: {unpaired} directed half-edge(s) lack an opposite"
            ),
        }
    }
}

impl std::error::Error for PolyhedronError {}

/// The solid bounded by a watertight [`TriMesh`].
///
/// Where the mesh came from is not this type's concern: an STL read by
/// `molrs::io::read_stl` (feature `io`), a marching-cubes extraction, a
/// hand-built [`TriMesh`] — anything closed. Unit conversion is the caller's
/// composition, `Polyhedron::new(mesh.scaled(s))`.
///
/// Containment is the even-odd rule along one fixed ray, which for a closed
/// 2-manifold is the Jordan–Brouwer separation: a point is inside iff a ray
/// from it crosses the surface an odd number of times. The distance is the
/// Euclidean distance to the closest point on any triangle, signed by that
/// parity. Both walk a bounding-volume hierarchy built once at construction.
#[derive(Debug, Clone)]
pub struct Polyhedron {
    mesh: TriMesh,
    /// Resolved corners, in face order, for the kernels.
    triangles: Vec<[[F; 3]; 3]>,
    bvh: Bvh,
    aabb: ([F; 3], [F; 3]),
}

impl Polyhedron {
    /// Bound a solid by `mesh`, in whatever length unit the mesh is in.
    ///
    /// # Errors
    ///
    /// Returns the first gate the mesh fails, in this order: no faces, a
    /// non-finite vertex, a face of zero area, an unpaired half-edge.
    pub fn new(mesh: TriMesh) -> Result<Self, PolyhedronError> {
        if mesh.is_empty() {
            return Err(PolyhedronError::Empty);
        }
        if let Some(index) = mesh
            .vertices()
            .iter()
            .position(|v| v.iter().any(|x| !x.is_finite()))
        {
            return Err(PolyhedronError::NonFiniteVertex { index });
        }
        if let Some(index) = mesh.first_degenerate_face(DEGENERATE_AREA2) {
            return Err(PolyhedronError::DegenerateFace { index });
        }
        let unpaired = mesh.unpaired_half_edges();
        if unpaired > 0 {
            return Err(PolyhedronError::NotWatertight { unpaired });
        }
        let aabb = mesh.aabb().ok_or(PolyhedronError::Empty)?;
        let triangles = mesh.to_triangles();
        let boxes: Vec<_> = triangles.iter().map(triangle_box).collect();
        let bvh = Bvh::build(&boxes);
        Ok(Self {
            mesh,
            triangles,
            bvh,
            aabb,
        })
    }

    /// The bounding surface.
    pub fn mesh(&self) -> &TriMesh {
        &self.mesh
    }

    /// Distance to the surface and the closest point on it.
    fn closest(&self, x: &[F; 3]) -> (F, [F; 3]) {
        let (i, d) = self
            .bvh
            .nearest(x, |i| {
                let t = &self.triangles[i as usize];
                norm(sub(*x, closest_point_triangle(*x, t[0], t[1], t[2])))
            })
            .expect("a Polyhedron has at least one face");
        let t = &self.triangles[i as usize];
        (d, closest_point_triangle(*x, t[0], t[1], t[2]))
    }

    /// Whether `x` is outside the mesh's own bounding box, which for a
    /// watertight mesh settles the sign on its own.
    fn outside_aabb(&self, x: &[F; 3]) -> bool {
        (0..3).any(|k| x[k] < self.aabb.0[k] || x[k] > self.aabb.1[k])
    }

    fn even_odd_inside(&self, x: &[F; 3]) -> bool {
        let dir = scale(PARITY_RAY, 1.0 / norm(PARITY_RAY));
        let inv = [1.0 / dir[0], 1.0 / dir[1], 1.0 / dir[2]];
        let hits = self.bvh.count_hits(x, &inv, |i| {
            let t = &self.triangles[i as usize];
            ray_hits_triangle(*x, dir, t[0], t[1], t[2])
        });
        hits % 2 == 1
    }
}

impl Region for Polyhedron {
    fn bounds(&self) -> Fnx3 {
        let mut b = Array2::zeros((3, 2));
        for d in 0..3 {
            b[[d, 0]] = self.aabb.0[d];
            b[[d, 1]] = self.aabb.1[d];
        }
        b
    }

    fn distance(&self, point: &[F; 3]) -> F {
        let (d, _) = self.closest(point);
        if d < SURFACE_EPS {
            0.0
        } else if self.outside_aabb(point) {
            // Watertight and the box is already stored: outside the box is
            // outside the surface, for six comparisons instead of a ray walk.
            d
        } else if self.even_odd_inside(point) {
            -d
        } else {
            d
        }
    }

    fn distance_grad(&self, point: &[F; 3]) -> [F; 3] {
        let (d, c) = self.closest(point);
        if d < SURFACE_EPS {
            return [0.0; 3];
        }
        let s = if !self.outside_aabb(point) && self.even_odd_inside(point) {
            -1.0
        } else {
            1.0
        };
        scale(sub(*point, c), s / d)
    }

    /// `distance <= 0` without paying for the distance: membership is the
    /// parity, and the closest point only ever contributes the surface
    /// tolerance — a threshold query that stops at the first hit.
    fn contains_point(&self, point: &[F; 3]) -> bool {
        // The AABB only short-circuits the parity: a point a hair outside the
        // box can still be within the tolerance of the surface.
        (!self.outside_aabb(point) && self.even_odd_inside(point))
            || self.bvh.any_within(point, SURFACE_EPS, |i| {
                let t = &self.triangles[i as usize];
                norm(sub(
                    *point,
                    closest_point_triangle(*point, t[0], t[1], t[2]),
                ))
            })
    }
}

/// Closest point on triangle `abc` to `p`.
///
/// Region test over the barycentric Voronoi regions of the triangle: Ericson,
/// *Real-Time Collision Detection* (2005) §5.1.5, which implements Eberly's
/// "Distance Between Point and Triangle in 3D" (1999).
fn closest_point_triangle(p: [F; 3], a: [F; 3], b: [F; 3], c: [F; 3]) -> [F; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

/// Whether the ray from `orig` along unit `dir` crosses triangle `v0 v1 v2`.
///
/// Möller & Trumbore, "Fast, Minimum Storage Ray-Triangle Intersection",
/// *J. Graphics Tools* 2(1):21–28 (1997). The barycentric domain is
/// half-open (`u ∈ [0, 1)`, `v ≥ 0`, `u + v < 1`) so a ray through a shared
/// edge is counted for exactly one of the two faces, and `t > EPS` keeps a
/// point on the surface from counting its own face.
fn ray_hits_triangle(orig: [F; 3], dir: [F; 3], v0: [F; 3], v1: [F; 3], v2: [F; 3]) -> bool {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let pvec = cross(dir, e2);
    let det = dot(e1, pvec);
    if det.abs() < SURFACE_EPS {
        return false;
    }
    let inv = 1.0 / det;
    let tvec = sub(orig, v0);
    let u = dot(tvec, pvec) * inv;
    if !(0.0..1.0).contains(&u) {
        return false;
    }
    let qvec = cross(tvec, e1);
    let v = dot(dir, qvec) * inv;
    if v < 0.0 || u + v >= 1.0 {
        return false;
    }
    let t = dot(e2, qvec) * inv;
    t > SURFACE_EPS
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Closed axis-aligned box from `lo` to `hi`, 12 triangles, outward winding.
    pub(crate) fn cube_tris(lo: [F; 3], hi: [F; 3]) -> Vec<[[F; 3]; 3]> {
        let [x0, y0, z0] = lo;
        let [x1, y1, z1] = hi;
        let p = |x, y, z| [x, y, z];
        vec![
            // -x
            [p(x0, y0, z0), p(x0, y0, z1), p(x0, y1, z1)],
            [p(x0, y0, z0), p(x0, y1, z1), p(x0, y1, z0)],
            // +x
            [p(x1, y0, z0), p(x1, y1, z0), p(x1, y1, z1)],
            [p(x1, y0, z0), p(x1, y1, z1), p(x1, y0, z1)],
            // -y
            [p(x0, y0, z0), p(x1, y0, z0), p(x1, y0, z1)],
            [p(x0, y0, z0), p(x1, y0, z1), p(x0, y0, z1)],
            // +y
            [p(x0, y1, z0), p(x0, y1, z1), p(x1, y1, z1)],
            [p(x0, y1, z0), p(x1, y1, z1), p(x1, y1, z0)],
            // -z
            [p(x0, y0, z0), p(x0, y1, z0), p(x1, y1, z0)],
            [p(x0, y0, z0), p(x1, y1, z0), p(x1, y0, z0)],
            // +z
            [p(x0, y0, z1), p(x1, y0, z1), p(x1, y1, z1)],
            [p(x0, y0, z1), p(x1, y1, z1), p(x0, y1, z1)],
        ]
    }

    fn unit_cube() -> Polyhedron {
        Polyhedron::new(TriMesh::from_triangles(&cube_tris([0.0; 3], [1.0; 3]))).expect("cube")
    }

    /// `contains_point` is a shortcut past the distance, so it has to keep
    /// agreeing with the distance — including on the surface, where the
    /// tolerance decides, and just outside the box, where the AABB
    /// short-circuit must not overrule that tolerance.
    #[test]
    fn contains_agrees_with_distance() {
        let r = unit_cube();
        let mut probes = vec![
            [0.5, 0.5, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [1.0 + 1e-12, 0.5, 0.5],
            [-1e-12, 0.5, 0.5],
            [1.0 + 1e-6, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ];
        for i in 0..12 {
            let t = i as F / 11.0;
            probes.push([t * 1.4 - 0.2, t * 0.7, 0.5]);
            probes.push([0.5, t * 1.4 - 0.2, t * 0.3]);
        }
        for p in probes {
            assert_eq!(
                r.contains_point(&p),
                r.distance(&p) <= 0.0,
                "contains_point disagrees with distance at {p:?} (d = {})",
                r.distance(&p)
            );
        }
    }

    #[test]
    fn unit_cube_distance_goldens() {
        let r = unit_cube();
        assert!(r.contains_point(&[0.5, 0.5, 0.5]));
        assert!((r.distance(&[0.5, 0.5, 0.5]) + 0.5).abs() < 1e-12);
        assert!(r.contains_point(&[1.0, 0.5, 0.5]));
        assert!(r.distance(&[1.0, 0.5, 0.5]).abs() < 1e-9);
        assert!(!r.contains_point(&[2.0, 0.5, 0.5]));
        assert!((r.distance(&[2.0, 0.5, 0.5]) - 1.0).abs() < 1e-12);
        assert!((r.distance(&[0.5, 0.5, -1.0]) - 1.0).abs() < 1e-12);
        let corner = r.distance(&[3.0, 3.0, 3.0]);
        assert!((corner - 2.0 * 3.0_f64.sqrt()).abs() < 1e-12);
        let g = r.distance_grad(&[2.0, 0.5, 0.5]);
        assert!((g[0] - 1.0).abs() < 1e-9);
        assert!(g[1].abs() < 1e-9 && g[2].abs() < 1e-9);
        let b = r.bounds();
        assert_eq!(b[[0, 0]], 0.0);
        assert_eq!(b[[2, 1]], 1.0);
        assert_eq!(r.mesh().n_faces(), 12);
    }

    #[test]
    fn grad_matches_finite_difference_away_from_edges() {
        let r = unit_cube();
        for p in [
            [0.5, 0.5, 0.2],
            [0.3, 0.5, 0.5],
            [1.6, 0.5, 0.5],
            [0.5, -0.4, 0.5],
        ] {
            let g = r.distance_grad(&p);
            for k in 0..3 {
                let mut plus = p;
                plus[k] += 1e-6;
                let mut minus = p;
                minus[k] -= 1e-6;
                let fd = (r.distance(&plus) - r.distance(&minus)) / 2e-6;
                assert!((g[k] - fd).abs() < 1e-5, "{g:?} at {p:?}");
            }
        }
    }

    #[test]
    fn nested_cavity_even_odd() {
        let mut tris = cube_tris([-2.0; 3], [2.0; 3]);
        tris.extend(cube_tris([-1.0; 3], [1.0; 3]));
        let r = Polyhedron::new(TriMesh::from_triangles(&tris)).expect("nested");
        assert!(!r.contains_point(&[0.0; 3]));
        assert!((r.distance(&[0.0; 3]) - 1.0).abs() < 1e-9);
        assert!(r.contains_point(&[1.5, 0.0, 0.0]));
        assert!((r.distance(&[1.5, 0.0, 0.0]) + 0.5).abs() < 1e-9);
        assert!(!r.contains_point(&[3.0, 0.0, 0.0]));
        assert!((r.distance(&[3.0, 0.0, 0.0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn named_rejects() {
        assert_eq!(
            Polyhedron::new(TriMesh::from_triangles(&[])).err(),
            Some(PolyhedronError::Empty)
        );
        let deg = [[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]];
        assert_eq!(
            Polyhedron::new(TriMesh::from_triangles(&deg)).err(),
            Some(PolyhedronError::DegenerateFace { index: 0 })
        );
        let mut open = cube_tris([0.0; 3], [1.0; 3]);
        open.pop();
        assert!(matches!(
            Polyhedron::new(TriMesh::from_triangles(&open)),
            Err(PolyhedronError::NotWatertight { .. })
        ));
        let mut nan = cube_tris([0.0; 3], [1.0; 3]);
        nan[0][0][0] = F::NAN;
        assert!(matches!(
            Polyhedron::new(TriMesh::from_triangles(&nan)),
            Err(PolyhedronError::NonFiniteVertex { .. })
        ));
    }

    #[test]
    fn scaled_mesh_doubles_the_cube() {
        let mesh = TriMesh::from_triangles(&cube_tris([0.0; 3], [1.0; 3])).scaled(2.0);
        let r = Polyhedron::new(mesh).expect("scaled");
        assert!((r.distance(&[1.0, 1.0, 1.0]) + 1.0).abs() < 1e-9);
        assert_eq!(r.bounds()[[0, 1]], 2.0);
    }
}
