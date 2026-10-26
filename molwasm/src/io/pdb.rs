//! Protein Data Bank for the WASM API — the face of `molrs::io::pdb`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `PdbStream` | `PdbIndexBuilder` + `read_pdb` (the chunk-fed PDB reader) |
//! | `readPdb`, `writePdb` | `read_pdb`, `write_pdb` |

use molrs::io::pdb::PdbIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = PdbStream,
    indexer = PdbIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_pdb(bytes),
}

read_door!(
    /// Read the first frame of PDB (a `MODEL` block or a whole file): the
    /// file's text or bytes.
    readPdb => read_pdb, "PDB"
);
write_door!(
    /// Write `frame` as PDB text.
    writePdb => write_pdb, "PDB"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{float_col, text, two_atom_frame};
    use wasm_bindgen_test::*;

    /// A PDB file is read through `PdbStream`, its one reader.
    #[wasm_bindgen_test]
    fn pdb_stream_reads_the_atoms() {
        let pdb = "ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C\n\
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N\n\
END\n";
        let mut stream = PdbStream::new();
        let entries = index_whole!(stream, pdb.as_bytes());
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        let frame = stream
            .parse_range_in_input(e.byte_offset() as usize, e.byte_len() as usize)
            .expect("parse");
        let x = float_col(&frame.get("atoms").expect("atoms"), "x");
        assert_eq!(x.to_vec(), vec![1.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn write_pdb_emits_atom_records_read_pdb_reads_back() {
        let out = write_pdb(&two_atom_frame()).expect("pdb output");
        assert!(out.contains("ATOM"));
        let back = read_pdb(&text(&out)).expect("read back");
        assert_eq!(back.get("atoms").expect("atoms").n_rows().unwrap(), 2);
    }
}
