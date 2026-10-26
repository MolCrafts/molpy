//! XCrySDen XSF for the WASM API — the face of `molrs::io::xsf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readXsf` | `read_xsf` |
//! | `writeXsf` | `write_xsf` |
//!
//! molrs reads XSF with functions only, so JS has no reader class.

use wasm_bindgen::prelude::*;

read_door!(
    /// Read XCrySDen XSF (the file's text or bytes): a periodic box from `CRYSTAL` + `PRIMVEC`, a
    /// free one from `MOLECULE`; atoms with `atomic_number` (u64, as the file
    /// states it — no `element` symbols are synthesized) and `x`/`y`/`z` (Å).
    readXsf => read_xsf, "XSF"
);
write_door!(
    /// Write `frame` as XCrySDen XSF text.
    writeXsf => write_xsf, "XSF"
);
