//! STL — stereolithography triangle soup, ASCII and binary.

use crate::io::{Source, invalid_data};
use std::io::{Error, Result};

use crate::core::TriMesh;
use crate::op::F;

/// 80-byte header plus the `u32` triangle count.
const BINARY_HEADER_BYTES: usize = 84;
/// Per triangle: a normal and three vertices (12 `f32`), then a `u16` attribute.
const BINARY_TRIANGLE_BYTES: usize = 50;

/// Read an STL file into a triangle mesh — binary if the length says so,
/// ASCII otherwise.
///
/// A file with no facets is not an error: it reads as an empty
/// [`TriMesh`]. Whether that is acceptable is the consumer's call — the same
/// division that leaves the watertight gate to whoever needs a closed surface.
///
/// The format has two shapes and no version marker, so telling them apart is
/// the reader's first job. A binary file is an 80-byte header, a `u32` triangle
/// count, then exactly 50 bytes per triangle; that **length identity** is the
/// discriminator. The leading `solid` keyword is not: a binary header is free
/// text and routinely starts with `solid` too, so keyword-first detection reads
/// binary files as ASCII and finds no vertices in them.
///
/// Recorded facet normals are read past and thrown away. Writers routinely
/// leave them at `0 0 0` — molpack's own STL writer does — so the normal a
/// consumer can trust is the one
/// [`TriMesh::face_normal`](crate::core::TriMesh::face_normal) computes from
/// the winding.
///
/// What comes back is triangles, nothing more: no watertight gate, no
/// degeneracy gate, no unit conversion. A packing region needs a closed surface
/// and refuses without one; a viewer paints whatever it is handed. Both ask the
/// [`TriMesh`] rather than being second-guessed here.
///
/// # Errors
///
/// [`std::io::Error`] if the path cannot be read, or
/// [`std::io::ErrorKind::InvalidData`] when the bytes are not a length-matched binary
/// STL and also not ASCII: not UTF-8, not starting with `solid`, or carrying a
/// vertex count that is not a multiple of three.
pub fn read_stl<'a>(src: impl Into<Source<'a>>) -> Result<TriMesh> {
    let src = src.into();
    parse_stl(&src.bytes()?).map_err(|e| match src.path() {
        Some(path) => Error::new(
            e.kind(),
            format!("could not read STL {}: {e}", path.display()),
        ),
        None => e,
    })
}

/// Parse STL bytes — binary if the length says so, ASCII otherwise.
fn parse_stl(bytes: &[u8]) -> Result<TriMesh> {
    let triangles = match binary_triangle_count(bytes) {
        Some(count) => parse_binary(bytes, count),
        None => parse_ascii(bytes)?,
    };
    Ok(TriMesh::from_triangles(&triangles))
}

/// Triangle count of a length-matched binary STL, or `None` if not one.
fn binary_triangle_count(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < BINARY_HEADER_BYTES {
        return None;
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().ok()?) as usize;
    let expected = BINARY_HEADER_BYTES.checked_add(count.checked_mul(BINARY_TRIANGLE_BYTES)?)?;
    (bytes.len() == expected).then_some(count)
}

fn parse_binary(bytes: &[u8], count: usize) -> Vec<[[F; 3]; 3]> {
    let mut triangles = Vec::with_capacity(count);
    let mut offset = BINARY_HEADER_BYTES;
    for _ in 0..count {
        offset += 12; // recorded facet normal — recomputed from the winding
        let mut triangle = [[0.0; 3]; 3];
        for corner in &mut triangle {
            for axis in corner.iter_mut() {
                let word = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
                *axis = f32::from_bits(word) as F;
                offset += 4;
            }
        }
        offset += 2; // attribute byte count
        triangles.push(triangle);
    }
    triangles
}

fn parse_ascii(bytes: &[u8]) -> Result<Vec<[[F; 3]; 3]>> {
    let text = std::str::from_utf8(bytes)
        .map_err(|_| invalid_data("not UTF-8 and not a length-matched binary STL"))?;
    let body = text.trim_start_matches('\u{feff}').trim_start();
    if !body
        .get(..5)
        .is_some_and(|head| head.eq_ignore_ascii_case("solid"))
    {
        return Err(invalid_data("ASCII STL must start with 'solid'"));
    }

    // Scanning for `vertex` rather than walking facet/loop/endloop: the
    // keyword is the only line that carries geometry, and writers disagree
    // about everything around it (indentation, `endfacet` spelling, whether a
    // solid is named). Three vertices in order are one triangle.
    let mut corners: Vec<[F; 3]> = Vec::new();
    let mut tokens = body.split_ascii_whitespace();
    while let Some(token) = tokens.next() {
        if !token.eq_ignore_ascii_case("vertex") {
            continue;
        }
        let mut corner = [0.0; 3];
        for axis in corner.iter_mut() {
            let word = tokens
                .next()
                .ok_or_else(|| invalid_data("truncated vertex"))?;
            *axis = word
                .parse::<F>()
                .map_err(|_| invalid_data(format!("bad vertex coordinate {word:?}")))?;
        }
        corners.push(corner);
    }

    if !corners.len().is_multiple_of(3) {
        return Err(invalid_data(format!(
            "{} vertices; not a multiple of 3",
            corners.len()
        )));
    }
    Ok(corners.as_chunks::<3>().0.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASCII_TRIANGLE: &str = "solid one\n\
          facet normal 0 0 0\n\
            outer loop\n\
              vertex 0 0 0\n\
              vertex 2 0 0\n\
              vertex 0 2 0\n\
            endloop\n\
          endfacet\n\
        endsolid one\n";

    /// The same triangle as a binary STL; `header` fills the 80-byte preamble.
    fn binary_triangle(header: &str) -> Vec<u8> {
        let mut bytes = vec![0u8; BINARY_HEADER_BYTES + BINARY_TRIANGLE_BYTES];
        let head = header.as_bytes();
        bytes[..head.len().min(80)].copy_from_slice(&head[..head.len().min(80)]);
        bytes[80..84].copy_from_slice(&1u32.to_le_bytes());
        // Recorded normal is deliberately wrong; the winding says +z.
        let values: [f32; 12] = [
            0.0, 0.0, -1.0, // normal
            0.0, 0.0, 0.0, // a
            2.0, 0.0, 0.0, // b
            0.0, 2.0, 0.0, // c
        ];
        for (i, value) in values.iter().enumerate() {
            let at = 84 + i * 4;
            bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn reads_an_ascii_facet() {
        let mesh = read_stl(ASCII_TRIANGLE.as_bytes()).unwrap();
        assert_eq!(mesh.n_faces(), 1);
        assert_eq!(
            mesh.triangle(0),
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        );
    }

    #[test]
    fn reads_the_same_triangle_from_binary() {
        let mesh = read_stl(&binary_triangle("binary")).unwrap();
        assert_eq!(
            mesh.triangle(0),
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        );
    }

    #[test]
    fn the_length_tells_binary_from_ascii_even_when_the_header_says_solid() {
        let mesh = read_stl(&binary_triangle("solid cube")).unwrap();
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn the_recorded_facet_normal_is_ignored() {
        // The fixture records (0, 0, -1); the winding says (0, 0, +1).
        let mesh = read_stl(&binary_triangle("binary")).unwrap();
        assert_eq!(mesh.face_normal(0), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn ascii_is_case_and_whitespace_insensitive() {
        let shouty = "SOLID s\nFACET NORMAL 0 0 0\nOUTER LOOP\nVERTEX 0 0 0\n\
             VERTEX 1 0 0\nVERTEX 0 1 0\nENDLOOP\nENDFACET\nENDSOLID s\n";
        assert_eq!(read_stl(shouty.as_bytes()).unwrap().n_faces(), 1);
    }

    #[test]
    fn refuses_text_that_is_not_an_stl() {
        let err = read_stl(b"ITEM: TIMESTEP\n0\n").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn refuses_a_vertex_count_that_is_not_a_multiple_of_three() {
        let truncated = "solid s\nvertex 0 0 0\nvertex 1 0 0\nendsolid s\n";
        assert!(read_stl(truncated.as_bytes()).is_err());
    }

    #[test]
    fn refuses_a_non_numeric_coordinate() {
        let bad = ASCII_TRIANGLE.replace("vertex 2 0 0", "vertex two 0 0");
        assert!(read_stl(bad.as_bytes()).is_err());
    }

    #[test]
    fn a_solid_with_no_facets_reads_as_an_empty_mesh() {
        // Not an error here: molpack calls it `Empty` and refuses, a viewer
        // paints nothing. Both are decisions this reader does not make.
        let mesh = read_stl(b"solid empty\nendsolid empty\n").unwrap();
        assert!(mesh.is_empty());
    }

    #[test]
    fn refuses_bytes_that_are_neither_utf8_nor_length_matched() {
        assert!(read_stl(&[0xff, 0xfe, 0x00, 0x80, 0x81]).is_err());
    }

    #[test]
    fn welds_a_closed_binary_mesh_into_a_watertight_surface() {
        // Two facets sharing an edge weld into four vertices, not six.
        let mut bytes = vec![0u8; BINARY_HEADER_BYTES + 2 * BINARY_TRIANGLE_BYTES];
        bytes[80..84].copy_from_slice(&2u32.to_le_bytes());
        let facets: [[f32; 12]; 2] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ];
        for (facet, chunk) in facets.iter().enumerate() {
            for (i, value) in chunk.iter().enumerate() {
                let at = BINARY_HEADER_BYTES + facet * BINARY_TRIANGLE_BYTES + i * 4;
                bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        let mesh = read_stl(&bytes).unwrap();
        assert_eq!(mesh.n_faces(), 2);
        assert_eq!(mesh.n_vertices(), 4);
    }
}
