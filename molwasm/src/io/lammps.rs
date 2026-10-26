//! LAMMPS for the WASM API — the face of `molrs::io::lammps`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `LammpsDataStream` | `LammpsDataIndexBuilder` + `read_lammps_data` (the chunk-fed data-file reader) |
//! | `LammpsDumpStream` | `LammpsDumpIndexBuilder` + `read_lammps_dump` (the chunk-fed dump reader) |
//! | `readLammpsData`, `writeLammpsData` | `read_lammps_data`, `write_lammps_data` |
//! | `readLammpsDump`, `writeLammpsDump` | `read_lammps_dump`, `write_lammps_dump` (one snapshot) |
//! | `readLammpsLog`, `isLammpsLog` | [`log`] — `io::lammps::log` |

pub mod log;

pub use self::log::*;

use molrs::io::lammps::{LammpsDataIndexBuilder, LammpsDumpIndexBuilder};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = LammpsDataStream,
    indexer = LammpsDataIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_lammps_data(bytes),
}

impl_wasm_traj_stream! {
    name    = LammpsDumpStream,
    indexer = LammpsDumpIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_lammps_dump(bytes),
}

read_door!(
    /// Read a LAMMPS data file, its text or bytes.
    readLammpsData => read_lammps_data, "LAMMPS data"
);
write_door!(
    /// Write `frame` as LAMMPS data text.
    writeLammpsData => write_lammps_data, "LAMMPS data"
);
read_door!(
    /// Read the first snapshot of a LAMMPS dump, its text or bytes.
    readLammpsDump => read_lammps_dump, "LAMMPS dump"
);

/// Write `frame` as one LAMMPS dump snapshot. `columns` is the `dump custom`
/// column line, in order; omitted, every column the `atoms` block holds.
#[wasm_bindgen(js_name = writeLammpsDump)]
pub fn write_lammps_dump(frame: &Frame, columns: Option<Vec<String>>) -> Result<String, JsValue> {
    let chosen: Option<Vec<&str>> = columns
        .as_ref()
        .map(|c| c.iter().map(String::as_str).collect());
    let what = "LAMMPS dump";
    let bytes = frame.with_frame(|f| {
        crate::io::written(what, |buf| {
            molrs::io::write_lammps_dump(buf, f, chosen.as_deref())
        })
    })?;
    crate::io::text_of(what, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::frame_index::FrameOffset;
    use crate::io::test_fixtures::{float_col, text};
    use wasm_bindgen_test::*;

    /// Two-frame LAMMPS dump (matches the smallest fixture in `molrs::io` tests).
    const LAMMPS_DUMP: &str = "ITEM: TIMESTEP\n0\n\
ITEM: NUMBER OF ATOMS\n2\n\
ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
ITEM: ATOMS id type x y z\n\
1 1 0.0 0.0 0.0\n\
2 1 1.0 1.0 1.0\n\
ITEM: TIMESTEP\n1\n\
ITEM: NUMBER OF ATOMS\n2\n\
ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n\
ITEM: ATOMS id type x y z\n\
1 1 0.5 0.5 0.5\n\
2 1 1.5 1.5 1.5\n";

    /// Smoke test for the LAMMPS dump streaming class: feed → finish →
    /// parse the second frame → the returned Frame carries its data.
    #[wasm_bindgen_test]
    fn lammps_dump_stream_indexes_and_parses_two_frames() {
        let mut stream = LammpsDumpStream::new();
        let bytes = LAMMPS_DUMP.as_bytes();
        let cap = stream.alloc_input_buffer(bytes.len());
        assert!(!cap.is_null());
        // SAFETY: we just allocated this buffer at `len = bytes.len()`,
        // and we hold &mut self exclusively through this scope.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), cap, bytes.len());
        }
        let entries = stream
            .feed_index_chunk(0.0, bytes.len())
            .expect("feed_index_chunk");
        let trailing = stream.finish_index().expect("finish_index");
        let mut all = entries;
        all.extend(trailing);
        assert_eq!(all.len(), 2, "expected exactly 2 frames in the fixture");

        // Decode frame 1 into a Frame: the dump's own column order, its
        // rows, and its box.
        let f1 = &all[1];
        let frame = stream
            .parse_range_in_input(f1.byte_offset() as usize, f1.byte_len() as usize)
            .expect("parse_range_in_input");
        assert_eq!(frame.keys().unwrap(), vec!["atoms".to_string()]);
        let atoms = frame.get("atoms").expect("atoms block");
        let keys = atoms.keys().expect("keys");
        // The dump's `type` column is read as the canonical `type_id`.
        assert_eq!(keys, ["id", "type_id", "x", "y", "z"]);
        assert_eq!(atoms.n_rows().expect("n_rows"), 2);
        let x: js_sys::Float64Array =
            wasm_bindgen::JsCast::unchecked_into(JsValue::from(atoms.copy("x", None).expect("x")));
        assert_eq!(x.to_vec(), vec![0.5, 1.5]);
        assert!(
            frame.get_box().unwrap().is_some(),
            "the dump's box bounds survive"
        );
    }

    /// `writeLammpsDump` writes the chosen columns; `readLammpsDump` reads
    /// the first snapshot back.
    #[wasm_bindgen_test]
    fn write_lammps_dump_reads_back() {
        let first = read_lammps_dump(&text(LAMMPS_DUMP)).expect("read");
        let out = write_lammps_dump(
            &first,
            Some(vec!["id".into(), "x".into(), "y".into(), "z".into()]),
        )
        .expect("write");
        assert!(out.contains("ITEM: ATOMS id x y z"), "{out}");
        let back = read_lammps_dump(&text(&out)).expect("read back");
        assert_eq!(
            float_col(&back.get("atoms").unwrap(), "x").to_vec(),
            vec![0.0, 1.0]
        );
    }

    /// Indexing the same fixture in many small chunks must produce the
    /// same entries as a single-shot feed.
    #[wasm_bindgen_test]
    fn lammps_dump_stream_chunked_index_matches_single_shot() {
        let bytes = LAMMPS_DUMP.as_bytes();

        // Single-shot.
        let mut s1 = LammpsDumpStream::new();
        let p1 = s1.alloc_input_buffer(bytes.len());
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), p1, bytes.len());
        }
        let mut all1 = s1.feed_index_chunk(0.0, bytes.len()).expect("feed");
        all1.extend(s1.finish_index().expect("finish"));

        // 17-byte chunks — small enough to split lines, frames, and
        // ITEM headers across the boundary.
        let mut s2 = LammpsDumpStream::new();
        let mut all2: Vec<FrameOffset> = Vec::new();
        let chunk_size = 17usize;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let len = (bytes.len() - offset).min(chunk_size);
            let p = s2.alloc_input_buffer(len);
            unsafe {
                std::ptr::copy_nonoverlapping(bytes[offset..offset + len].as_ptr(), p, len);
            }
            let entries = s2.feed_index_chunk(offset as f64, len).expect("feed");
            all2.extend(entries);
            offset += len;
        }
        all2.extend(s2.finish_index().expect("finish"));

        assert_eq!(all1.len(), all2.len(), "frame count mismatch");
        for (a, b) in all1.iter().zip(all2.iter()) {
            assert_eq!(a.byte_offset(), b.byte_offset());
            assert_eq!(a.byte_len(), b.byte_len());
        }
    }

    /// A LAMMPS data file is one frame of `LammpsDataStream`.
    #[wasm_bindgen_test]
    fn lammps_data_stream_reads_one_frame() {
        let data = "LAMMPS data\n\n\
                    2 atoms\n1 atom types\n\n\
                    0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n\n\
                    Atoms\n\n\
                    1 1 1.0 2.0 3.0\n2 1 4.0 5.0 6.0\n";
        let mut stream = LammpsDataStream::new();
        let entries = index_whole!(stream, data.as_bytes());
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        let frame = stream
            .parse_range_in_input(e.byte_offset() as usize, e.byte_len() as usize)
            .expect("parse");
        let x = float_col(&frame.get("atoms").expect("atoms"), "x");
        assert_eq!(x.to_vec(), vec![1.0, 4.0]);
    }
}
