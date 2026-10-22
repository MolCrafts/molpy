//! CSV tables for the WASM API — the face of `molrs::io`'s CSV doors.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readCsvBlock` | `read_csv_block` |
//! | `writeCsvBlock` | `write_csv_block` |

use wasm_bindgen::prelude::*;

use crate::core::block::Block;
use crate::io::{JsSource, source_bytes, text_of, written};

/// Parse CSV — the file's text or bytes — into a [`Block`]; each column's
/// dtype is inferred int → float → str. With `header`, the text is
/// headerless and those are the column names; without, the first non-empty
/// line names them. `delimiter` defaults to `","`.
///
/// # Errors
///
/// Throws if `source` is neither a string nor a `Uint8Array`, or is not a
/// CSV table.
#[wasm_bindgen(js_name = readCsvBlock)]
pub fn read_csv_block(
    source: &JsSource,
    delimiter: Option<char>,
    header: Option<Vec<String>>,
) -> Result<Block, JsValue> {
    let bytes = source_bytes("readCsvBlock", source)?;
    let block = molrs::io::read_csv_block(
        bytes.as_slice(),
        delimiter.unwrap_or(','),
        header.as_deref(),
    )
    .map_err(|e| JsValue::from_str(&format!("CSV read error: {e}")))?;
    Block::from_rs(block)
}

/// Write `block` as CSV text — the inverse of `readCsvBlock`.
/// `delimiter` defaults to `","`, `header` (write the names line) to `true`.
#[wasm_bindgen(js_name = writeCsvBlock)]
pub fn write_csv_block(
    block: &Block,
    delimiter: Option<char>,
    header: Option<bool>,
) -> Result<String, JsValue> {
    let bytes = block.with_rs(|b| {
        written("CSV", |buf| {
            molrs::io::write_csv_block(buf, b, delimiter.unwrap_or(','), header.unwrap_or(true))
        })
    })??;
    text_of("CSV", bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::text;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_csv_block_reads_back() {
        let block = read_csv_block(&text("a,b\n1,x\n2,y\n"), None, None).expect("csv");
        assert_eq!(block.n_rows().expect("rows"), 2);
        let out = write_csv_block(&block, None, None).expect("text");
        assert!(out.starts_with("a,b\n"), "{out}");
        let back = read_csv_block(&text(&out), None, None).expect("read back");
        assert_eq!(back.keys().expect("keys"), ["a", "b"]);
        assert_eq!(back.n_rows().expect("rows"), 2);
    }
}
