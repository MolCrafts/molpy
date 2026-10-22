//! Triangle meshes — a surface as shared vertices plus indexed faces: [`TriMesh`].

use std::collections::HashMap;

use crate::op::F;
use crate::op::vec3::{cross, sub};

/// Below this twice-area a face has no usable normal and no interior.
///
/// Twice-area is the cross-product magnitude, so this is `2e-12` Å² of actual
/// area — small enough to pass every mesh a CAD tool emits, large enough to
/// catch a repeated or exactly collinear corner.
pub const DEGENERATE_AREA2: F = 1e-12;

/// A triangle surface: shared vertices, and faces indexing into them.
///
/// Sits beside [`Region`](crate::core::Region) because that is what consumes it:
/// molpack's `StlRegion` is a watertight `TriMesh` plus a containment rule,
/// and molvis paints the same mesh as the container a trajectory plays inside.
/// Reading one out of a file is `molrs::io::stl`'s job (feature `io`); this
/// is the geometry it hands back.
///
/// Coordinates carry no units of their own — they are whatever the file said.
/// Callers that know the file's unit convert with [`TriMesh::scaled`] (molpack
/// reads STL as Å per file unit).
///
/// The type is deliberately permissive: it stores whatever triangles it was
/// given and answers questions about them ([`TriMesh::is_watertight`],
/// [`TriMesh::first_degenerate_face`]). Refusing a mesh is the consumer's
/// call — a packing region needs a closed surface to ask `contains`, a viewer
/// only needs something to paint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriMesh {
    vertices: Vec<[F; 3]>,
    faces: Vec<[u32; 3]>,
}

impl TriMesh {
    /// Build from triangle soup, welding corners that are **bit-identical**.
    ///
    /// Bitwise equality, not a tolerance: welding is what makes
    /// [`is_watertight`](Self::is_watertight) answerable at all, and STL
    /// writers emit the shared corner of two faces as the same bytes. A
    /// tolerance would silently merge a genuinely open seam, which is the one
    /// thing the watertight check exists to find. It also means `0.0` and
    /// `-0.0` are different corners — as they are to every other STL reader.
    pub fn from_triangles(triangles: &[[[F; 3]; 3]]) -> Self {
        let mut vertices: Vec<[F; 3]> = Vec::new();
        let mut index_of: HashMap<[u64; 3], u32> = HashMap::new();
        let mut faces: Vec<[u32; 3]> = Vec::with_capacity(triangles.len());

        for triangle in triangles {
            let mut face = [0u32; 3];
            for (slot, corner) in face.iter_mut().zip(triangle) {
                let key = [
                    corner[0].to_bits(),
                    corner[1].to_bits(),
                    corner[2].to_bits(),
                ];
                *slot = *index_of.entry(key).or_insert_with(|| {
                    vertices.push(*corner);
                    (vertices.len() - 1) as u32
                });
            }
            faces.push(face);
        }

        Self { vertices, faces }
    }

    /// Build from an existing vertex table and face list.
    ///
    /// # Errors
    ///
    /// Returns the offending face index when a face points past the end of
    /// `vertices` — an out-of-range index would panic every accessor below.
    pub fn from_indexed(vertices: Vec<[F; 3]>, faces: Vec<[u32; 3]>) -> Result<Self, usize> {
        let n = vertices.len() as u32;
        for (index, face) in faces.iter().enumerate() {
            if face.iter().any(|&v| v >= n) {
                return Err(index);
            }
        }
        Ok(Self { vertices, faces })
    }

    /// The vertex table. Faces index into it.
    pub fn vertices(&self) -> &[[F; 3]] {
        &self.vertices
    }

    /// Corner indices, three per face.
    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.faces.is_empty()
    }

    /// The three corners of face `index`, resolved.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of range. Both constructors reject faces that
    /// point past the vertex table, so a face that exists is always resolvable.
    pub fn triangle(&self, index: usize) -> [[F; 3]; 3] {
        let face = self.faces[index];
        [
            self.vertices[face[0] as usize],
            self.vertices[face[1] as usize],
            self.vertices[face[2] as usize],
        ]
    }

    /// Every face as resolved corners — the flat form a distance query wants.
    pub fn to_triangles(&self) -> Vec<[[F; 3]; 3]> {
        (0..self.faces.len()).map(|i| self.triangle(i)).collect()
    }

    /// Axis-aligned bounds as `(min, max)`, or `None` for an empty mesh.
    pub fn aabb(&self) -> Option<([F; 3], [F; 3])> {
        let first = *self.vertices.first()?;
        let mut min = first;
        let mut max = first;
        for vertex in &self.vertices {
            for (k, &value) in vertex.iter().enumerate() {
                if value < min[k] {
                    min[k] = value;
                }
                if value > max[k] {
                    max[k] = value;
                }
            }
        }
        Some((min, max))
    }

    /// Twice the area of face `index` — the cross-product magnitude.
    pub fn face_area2(&self, index: usize) -> F {
        let [a, b, c] = self.triangle(index);
        let cross = cross(sub(b, a), sub(c, a));
        (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
    }

    /// Unit normal of face `index`, from the winding of its corners.
    ///
    /// Computed, never read off the file: STL records a normal per facet and
    /// writers routinely leave it at `0 0 0` (molpack's does), so a reader
    /// that trusted it would hand back a surface with no lighting at all.
    /// A degenerate face has no normal and gets `[0, 0, 0]`.
    pub fn face_normal(&self, index: usize) -> [F; 3] {
        let [a, b, c] = self.triangle(index);
        let cross = cross(sub(b, a), sub(c, a));
        let length = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if length <= 0.0 || !length.is_finite() {
            return [0.0; 3];
        }
        [cross[0] / length, cross[1] / length, cross[2] / length]
    }

    /// One unit normal per face, in face order.
    pub fn face_normals(&self) -> Vec<[F; 3]> {
        (0..self.faces.len()).map(|i| self.face_normal(i)).collect()
    }

    /// The first face with no usable area, or `None`.
    ///
    /// `min_area2` is compared against [`face_area2`](Self::face_area2);
    /// [`DEGENERATE_AREA2`] is the usual choice. A face whose corners weld to
    /// the same vertex counts too, however large its coordinates say it is.
    pub fn first_degenerate_face(&self, min_area2: F) -> Option<usize> {
        (0..self.faces.len()).find(|&index| {
            let [i, j, k] = self.faces[index];
            i == j || j == k || k == i || self.face_area2(index) < min_area2
        })
    }

    /// Directed edges with no opposite — zero for a closed 2-manifold.
    ///
    /// Each face contributes its three edges in winding order. A closed
    /// surface pairs every `(u, v)` with exactly one `(v, u)`; anything else
    /// is a boundary, a duplicate face, or a flipped neighbour, and the count
    /// is how many such edges there are.
    pub fn unpaired_half_edges(&self) -> usize {
        let mut directed: HashMap<(u32, u32), u32> = HashMap::new();
        for &[a, b, c] in &self.faces {
            for edge in [(a, b), (b, c), (c, a)] {
                *directed.entry(edge).or_insert(0) += 1;
            }
        }
        directed
            .iter()
            .filter(|&(&(u, v), &count)| count != 1 || directed.get(&(v, u)) != Some(&1))
            .count()
    }

    /// Whether the surface is closed: every directed edge has one opposite.
    pub fn is_watertight(&self) -> bool {
        !self.is_empty() && self.unpaired_half_edges() == 0
    }

    /// The same mesh with every coordinate multiplied by `factor`.
    ///
    /// The unit seam: a file's numbers mean whatever its producer meant, and
    /// the caller that knows converts here rather than teaching the reader
    /// about units.
    pub fn scaled(&self, factor: F) -> Self {
        Self {
            vertices: self
                .vertices
                .iter()
                .map(|v| [v[0] * factor, v[1] * factor, v[2] * factor])
                .collect(),
            faces: self.faces.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed unit cube at the origin, 12 triangles, outward winding.
    fn cube(edge: F) -> Vec<[[F; 3]; 3]> {
        let quads: [[[F; 3]; 4]; 6] = [
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        ];
        let scale = |p: [F; 3]| [p[0] * edge, p[1] * edge, p[2] * edge];
        let mut triangles = Vec::with_capacity(12);
        for quad in quads {
            triangles.push([scale(quad[0]), scale(quad[1]), scale(quad[2])]);
            triangles.push([scale(quad[0]), scale(quad[2]), scale(quad[3])]);
        }
        triangles
    }

    #[test]
    fn welds_shared_corners() {
        let mesh = TriMesh::from_triangles(&cube(2.0));
        assert_eq!(mesh.n_faces(), 12);
        // 36 corners collapse onto the cube's 8 vertices.
        assert_eq!(mesh.n_vertices(), 8);
    }

    #[test]
    fn closed_cube_is_watertight() {
        let mesh = TriMesh::from_triangles(&cube(1.0));
        assert_eq!(mesh.unpaired_half_edges(), 0);
        assert!(mesh.is_watertight());
    }

    #[test]
    fn dropping_a_face_opens_the_surface() {
        let mut triangles = cube(1.0);
        triangles.pop();
        let mesh = TriMesh::from_triangles(&triangles);
        assert_eq!(mesh.unpaired_half_edges(), 3);
        assert!(!mesh.is_watertight());
    }

    #[test]
    fn an_empty_mesh_is_not_watertight() {
        assert!(!TriMesh::default().is_watertight());
        assert!(TriMesh::default().aabb().is_none());
    }

    #[test]
    fn normals_come_from_the_winding() {
        let mesh = TriMesh::from_triangles(&[[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]]);
        assert_eq!(mesh.face_normal(0), [0.0, 0.0, 1.0]);
        assert_eq!(mesh.face_area2(0), 4.0);
    }

    #[test]
    fn a_collinear_face_has_no_normal_and_no_area() {
        let mesh = TriMesh::from_triangles(&[[[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]]);
        assert_eq!(mesh.face_normal(0), [0.0; 3]);
        assert_eq!(mesh.first_degenerate_face(DEGENERATE_AREA2), Some(0));
    }

    #[test]
    fn a_face_with_a_repeated_corner_is_degenerate() {
        let mesh = TriMesh::from_triangles(&[[[0.0; 3], [1.0, 0.0, 0.0], [0.0; 3]]]);
        assert_eq!(mesh.first_degenerate_face(DEGENERATE_AREA2), Some(0));
    }

    #[test]
    fn a_sound_mesh_has_no_degenerate_face() {
        let mesh = TriMesh::from_triangles(&cube(3.0));
        assert_eq!(mesh.first_degenerate_face(DEGENERATE_AREA2), None);
    }

    #[test]
    fn aabb_spans_every_vertex() {
        let mesh = TriMesh::from_triangles(&cube(2.5));
        assert_eq!(mesh.aabb(), Some(([0.0; 3], [2.5; 3])));
    }

    #[test]
    fn scaling_moves_vertices_and_keeps_faces() {
        let mesh = TriMesh::from_triangles(&cube(1.0));
        let scaled = mesh.scaled(4.0);
        assert_eq!(scaled.aabb(), Some(([0.0; 3], [4.0; 3])));
        assert_eq!(scaled.faces(), mesh.faces());
    }

    #[test]
    fn indexed_construction_rejects_an_out_of_range_face() {
        let vertices = vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            TriMesh::from_indexed(vertices, vec![[0, 1, 3]]).unwrap_err(),
            0
        );
    }
}
