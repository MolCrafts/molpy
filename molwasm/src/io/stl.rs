//! STL surface meshes for the WASM API — the face of `molrs::io::stl`.
//!
//! | Function | Format |
//! |----------|--------|
//! | [`readStl`](read_stl) | STL, ASCII or binary (auto-detected) |
//!
//! Unlike every other format's reader, this produces a [`TriMesh`] rather
//! than a `Frame`: an STL carries triangles, not atoms.

use crate::core::mesh::TriMesh;
use crate::io::{JsSource, source_bytes};
use wasm_bindgen::prelude::*;

/// Read an STL file — its bytes, or an ASCII STL's text — into a
/// [`TriMesh`].
///
/// Both shapes of the format are accepted and told apart by length, not by
/// the leading keyword — a binary STL's 80-byte header is free text and
/// routinely starts with `solid` too.
///
/// # Arguments
///
/// * `source` - The whole file: a `Uint8Array` (either shape) or a `string`
///   (the text of an ASCII STL). An internal copy is taken, so the caller may
///   discard the buffer as soon as this returns.
///
/// # Errors
///
/// Throws a `JsValue` string when `source` is neither a string nor a
/// `Uint8Array`, or when the bytes are neither a length-matched
/// binary STL nor readable ASCII. A file with no facets is **not** an error:
/// it reads as a mesh with zero faces, and whether that is acceptable is the
/// caller's decision.
///
/// # Example (JavaScript)
///
/// ```js
/// const mesh = readStl(new Uint8Array(await file.arrayBuffer()));
/// console.log(`${mesh.nFaces()} triangles, watertight: ${mesh.isWatertight()}`);
/// ```
#[wasm_bindgen(js_name = readStl)]
pub fn read_stl(source: &JsSource) -> Result<TriMesh, JsValue> {
    let bytes = source_bytes("readStl", source)?;
    molrs::io::read_stl(bytes.as_slice())
        .map(|inner| TriMesh { inner })
        .map_err(|e| JsValue::from_str(&format!("STL reading error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{bytes, text};
    use wasm_bindgen_test::*;

    /// One ASCII facet, wound so the normal is +z and the recorded one is not
    /// consulted at all (the writer left it at zero, as molpack's does).
    const TRIANGLE: &str = "solid one\n\
          facet normal 0 0 0\n\
            outer loop\n\
              vertex 0 0 0\n\
              vertex 2 0 0\n\
              vertex 0 2 0\n\
            endloop\n\
          endfacet\n\
        endsolid one\n";

    #[wasm_bindgen_test]
    fn reads_a_mesh_for_javascript() {
        let mesh = read_stl(&bytes(TRIANGLE.as_bytes())).unwrap();
        assert_eq!(read_stl(&text(TRIANGLE)).unwrap().n_faces(), 1);
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.faces(), vec![0, 1, 2]);
        assert_eq!(mesh.face_normals(), vec![0.0, 0.0, 1.0]);
        assert_eq!(mesh.aabb(), vec![0.0, 0.0, 0.0, 2.0, 2.0, 0.0]);
        assert!(!mesh.is_watertight());
    }

    #[wasm_bindgen_test]
    fn surfaces_a_parse_failure_as_a_js_string() {
        assert!(read_stl(&bytes(b"ITEM: TIMESTEP\n0\n")).is_err());
    }
}
