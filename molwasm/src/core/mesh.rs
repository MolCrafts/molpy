//! WASM binding for the triangle mesh, `molrs::core::TriMesh`.
//!
//! A surface, not a structure: shared vertices plus indexed faces, with no
//! atoms and no simulation box. It is what `readStl` hands
//! back — the container a packing run was confined to, the geometry a viewer
//! paints around the trajectory.
//!
//! Accessors copy into fresh typed arrays rather than viewing WASM memory, so
//! the caller may keep them after `free()`. Coordinates come back as
//! `Float32Array`: a mesh goes to a GPU vertex buffer, which is `f32`
//! regardless, and the double it was read as buys nothing on the way there.

use molrs::core::TriMesh as RsTriMesh;
use wasm_bindgen::prelude::*;

/// Triangle surface: vertices, faces, and the questions worth asking about
/// them.
///
/// # Example (JavaScript)
///
/// ```js
/// const mesh = readStl(new Uint8Array(await file.arrayBuffer()));
/// console.log(mesh.nFaces(), mesh.isWatertight());
/// const vertices = mesh.vertices();    // Float32Array, 3 per vertex
/// const faces    = mesh.faces();       // Uint32Array, 3 indices per face
/// const normals  = mesh.faceNormals(); // Float32Array, 3 per face, from the winding
/// mesh.free();
/// ```
#[wasm_bindgen]
pub struct TriMesh {
    pub(crate) inner: RsTriMesh,
}

#[wasm_bindgen]
impl TriMesh {
    /// Number of vertices in the shared table.
    #[wasm_bindgen(js_name = nVertices)]
    pub fn n_vertices(&self) -> usize {
        self.inner.n_vertices()
    }

    /// Number of triangles.
    #[wasm_bindgen(js_name = nFaces)]
    pub fn n_faces(&self) -> usize {
        self.inner.n_faces()
    }

    /// Vertex coordinates, three per vertex: `[x0, y0, z0, x1, …]`.
    pub fn vertices(&self) -> Vec<f32> {
        self.inner
            .vertices()
            .iter()
            .flat_map(|v| [v[0] as f32, v[1] as f32, v[2] as f32])
            .collect()
    }

    /// Corner indices into the vertex table, three per face.
    pub fn faces(&self) -> Vec<u32> {
        self.inner.faces().iter().flatten().copied().collect()
    }

    /// One unit normal per face, computed from the winding of its corners.
    ///
    /// Not the normal the file recorded: STL writers routinely leave that at
    /// `0 0 0`, so a consumer that lit a surface with it would get a black
    /// mesh.
    #[wasm_bindgen(js_name = faceNormals)]
    pub fn face_normals(&self) -> Vec<f32> {
        (0..self.inner.n_faces())
            .flat_map(|i| {
                let n = self.inner.face_normal(i);
                [n[0] as f32, n[1] as f32, n[2] as f32]
            })
            .collect()
    }

    /// Axis-aligned bounds as `[minX, minY, minZ, maxX, maxY, maxZ]`, or an
    /// empty array when the mesh has no vertices.
    pub fn aabb(&self) -> Vec<f64> {
        match self.inner.aabb() {
            Some((min, max)) => vec![min[0], min[1], min[2], max[0], max[1], max[2]],
            None => Vec::new(),
        }
    }

    /// Whether the surface is closed — every directed edge has one opposite.
    ///
    /// A viewer paints an open mesh happily; a containment test on one answers
    /// "inside" for points that are not, so anything asking that question
    /// should check here first.
    #[wasm_bindgen(js_name = isWatertight)]
    pub fn is_watertight(&self) -> bool {
        self.inner.is_watertight()
    }
}
