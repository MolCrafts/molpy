//! The Crystallographic Information File (CIF) codec — MVP subset.

use crate::core::Cell;
use crate::io::frame_columns::insert_column;
use crate::io::frame_index::FrameOffset;
use crate::io::{Sink, SinkWriter, Source, invalid_data};
use std::collections::HashMap;
use std::io::{BufRead, Result, Seek, Write};

use ndarray::{Array1, Array2, IxDyn};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MetaMap;
use molrs::core::SimBox;
use molrs::op::{F, Idx};

use crate::io::reader::{FrameReader, ReadSeek, Reader, TrajectoryReader};
use crate::io::writer::{FrameWriter, Writer};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Strip a parenthesised standard-deviation suffix from a numeric token.
/// `5.917(3)` → `5.917`. Non-numeric tokens are returned unchanged.
fn strip_esd(token: &str) -> &str {
    if let Some(idx) = token.find('(') {
        &token[..idx]
    } else {
        token
    }
}

fn parse_float(token: &str) -> Option<F> {
    strip_esd(token).parse::<F>().ok()
}

// ---------------------------------------------------------------------------
// Tokeniser
// ---------------------------------------------------------------------------
//
// We do not aim to be a strict CIF tokeniser — only good enough to feed the
// loop-row parser. Tokens emitted are owned strings (not borrowed) because
// `;`-multi-line tokens span several physical lines.

#[derive(Debug, Clone)]
struct LineSource<R: BufRead> {
    reader: R,
    /// Pending tokens from a single physical line, FIFO.
    pending: Vec<String>,
    /// One-line lookahead used by the block-restart logic. (Reserved for
    /// future use; the current parser only needs token-level pushback.)
    pushed_line: Option<String>,
    /// True after we've seen EOF on the underlying reader.
    eof: bool,
}

impl<R: BufRead> LineSource<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            pending: Vec::new(),
            pushed_line: None,
            eof: false,
        }
    }

    fn read_raw_line(&mut self) -> Result<Option<String>> {
        if let Some(line) = self.pushed_line.take() {
            return Ok(Some(line));
        }
        if self.eof {
            return Ok(None);
        }
        let mut buf = String::new();
        let bytes = self.reader.read_line(&mut buf)?;
        if bytes == 0 {
            self.eof = true;
            return Ok(None);
        }
        Ok(Some(buf))
    }

    /// Pull the next non-empty, non-comment token.
    fn next_token(&mut self) -> Result<Option<String>> {
        loop {
            if let Some(t) = self.pending.pop() {
                return Ok(Some(t));
            }
            let line = match self.read_raw_line()? {
                Some(l) => l,
                None => return Ok(None),
            };

            // Multi-line `;...;` opens when `;` is the first column.
            if let Some(rest) = line.strip_prefix(';') {
                let mut content = String::new();
                content.push_str(rest.trim_end_matches(['\r', '\n']));
                loop {
                    let next = match self.read_raw_line()? {
                        Some(l) => l,
                        None => return Err(invalid_data("unterminated ;-block in CIF")),
                    };
                    if let Some(after) = next.strip_prefix(';') {
                        // Discard this terminator line, push remaining tokens
                        // from after the `;` (rare but allowed).
                        let rest = after.trim();
                        if !rest.is_empty() {
                            self.tokenise_line(rest);
                        }
                        break;
                    }
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(next.trim_end_matches(['\r', '\n']));
                }
                return Ok(Some(content));
            }

            self.tokenise_line(line.trim_end_matches(['\r', '\n']));
            // Loop and pop next token.
        }
    }

    fn tokenise_line(&mut self, line: &str) {
        // Comments: full-line comments only when `#` is at column 0 OR after
        // whitespace (CIF allows `#` to start a comment after whitespace).
        // For simplicity, drop everything from the first `#` whose preceding
        // char is whitespace or start-of-line.
        let mut s = String::from(line);
        let mut cut: Option<usize> = None;
        let mut prev_is_ws = true;
        for (i, c) in s.char_indices() {
            if c == '\'' || c == '"' {
                // Don't cut inside quotes; bail out of comment search until
                // the close quote. (Approximation: track quote state.)
                // For our subset, treat the whole string as no-comment if a
                // quote appears before any '#'.
                cut = None;
                break;
            }
            if c == '#' && prev_is_ws {
                cut = Some(i);
                break;
            }
            prev_is_ws = c.is_whitespace();
        }
        if let Some(idx) = cut {
            s.truncate(idx);
        }
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return;
        }

        // Whitespace-tokenise honoring single & double quotes.
        let mut buf = String::new();
        let mut in_single = false;
        let mut in_double = false;
        let mut tokens: Vec<String> = Vec::new();
        for c in trimmed.chars() {
            if in_single {
                if c == '\'' {
                    in_single = false;
                    tokens.push(std::mem::take(&mut buf));
                } else {
                    buf.push(c);
                }
            } else if in_double {
                if c == '"' {
                    in_double = false;
                    tokens.push(std::mem::take(&mut buf));
                } else {
                    buf.push(c);
                }
            } else if c.is_whitespace() {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
            } else if c == '\'' && buf.is_empty() {
                in_single = true;
            } else if c == '"' && buf.is_empty() {
                in_double = true;
            } else {
                buf.push(c);
            }
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
        // Reverse so we can pop in source order.
        tokens.reverse();
        for t in tokens {
            self.pending.push(t);
        }
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

struct FrameInProgress {
    name: String,
    cell_a: Option<F>,
    cell_b: Option<F>,
    cell_c: Option<F>,
    cell_alpha: Option<F>,
    cell_beta: Option<F>,
    cell_gamma: Option<F>,
    /// Misc exact-dtype metadata copied into the frame.
    meta: MetaMap,
    /// Per-atom column name → list of token values, in row order.
    atom_cols: HashMap<String, Vec<String>>,
}

impl FrameInProgress {
    fn new(name: String) -> Self {
        Self {
            name,
            cell_a: None,
            cell_b: None,
            cell_c: None,
            cell_alpha: None,
            cell_beta: None,
            cell_gamma: None,
            meta: MetaMap::new(),
            atom_cols: HashMap::new(),
        }
    }

    fn set_cell(&mut self, key: &str, value: &str) {
        let v = parse_float(value);
        match key {
            "_cell_length_a" | "_cell.length_a" => self.cell_a = v,
            "_cell_length_b" | "_cell.length_b" => self.cell_b = v,
            "_cell_length_c" | "_cell.length_c" => self.cell_c = v,
            "_cell_angle_alpha" | "_cell.angle_alpha" => self.cell_alpha = v,
            "_cell_angle_beta" | "_cell.angle_beta" => self.cell_beta = v,
            "_cell_angle_gamma" | "_cell.angle_gamma" => self.cell_gamma = v,
            _ => {}
        }
    }

    fn build_frame(self) -> Result<Frame> {
        if self.atom_cols.is_empty() {
            return Err(invalid_data(format!(
                "CIF block '{}' has no _atom_site_* loop",
                self.name
            )));
        }

        let n = self.atom_cols.values().next().map(|v| v.len()).unwrap_or(0);
        if n == 0 {
            return Err(invalid_data(format!(
                "CIF block '{}' atom loop has zero rows",
                self.name
            )));
        }
        for (k, v) in &self.atom_cols {
            if v.len() != n {
                return Err(invalid_data(format!(
                    "CIF block '{}': column '{}' has {} rows, expected {}",
                    self.name,
                    k,
                    v.len(),
                    n
                )));
            }
        }

        let mut atoms = Block::new();
        let mut frame = Frame::new();
        if !self.name.is_empty() {
            frame.meta.insert("title", self.name);
        }
        // `MetaMap` has no owned iterator. Copy in `iter` order so the frame
        // keeps the keys in the order this reader inserted them.
        frame.meta.extend(
            self.meta
                .iter()
                .map(|(key, value)| (key.clone(), value.clone())),
        );

        // Cell → SimBox (only if all six are present)
        let cell_present = self.cell_a.is_some()
            && self.cell_b.is_some()
            && self.cell_c.is_some()
            && self.cell_alpha.is_some()
            && self.cell_beta.is_some()
            && self.cell_gamma.is_some();
        let simbox = if cell_present {
            let lengths = [
                self.cell_a.unwrap(),
                self.cell_b.unwrap(),
                self.cell_c.unwrap(),
            ];
            let angles = [
                self.cell_alpha.unwrap(),
                self.cell_beta.unwrap(),
                self.cell_gamma.unwrap(),
            ];
            Some(
                SimBox::new(Cell::LengthsAngles { lengths, angles }, [0.0; 3], [true; 3])
                    .map_err(|e| invalid_data(format!("CIF cell → SimBox: {e:?}")))?,
            )
        } else {
            None
        };

        // Coordinates: prefer Cartn over fract.
        let cartn_x = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_x", "_atom_site.Cartn_x"],
        );
        let cartn_y = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_y", "_atom_site.Cartn_y"],
        );
        let cartn_z = column_floats(
            &self.atom_cols,
            &["_atom_site_Cartn_z", "_atom_site.Cartn_z"],
        );
        let fract_x = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_x", "_atom_site.fract_x"],
        );
        let fract_y = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_y", "_atom_site.fract_y"],
        );
        let fract_z = column_floats(
            &self.atom_cols,
            &["_atom_site_fract_z", "_atom_site.fract_z"],
        );

        let (xs, ys, zs): (Vec<F>, Vec<F>, Vec<F>) =
            if let (Some(x), Some(y), Some(z)) = (&cartn_x, &cartn_y, &cartn_z) {
                (x.clone(), y.clone(), z.clone())
            } else if let (Some(fx), Some(fy), Some(fz), Some(simbox)) =
                (&fract_x, &fract_y, &fract_z, &simbox)
            {
                let frac = Array2::from_shape_fn((n, 3), |(i, k)| [fx, fy, fz][k][i]);
                let cart = simbox.to_cart(frac.view());
                (
                    cart.column(0).to_vec(),
                    cart.column(1).to_vec(),
                    cart.column(2).to_vec(),
                )
            } else {
                return Err(invalid_data(
                    "CIF atom loop: missing both Cartesian and fractional coordinates",
                ));
            };

        insert_column(&mut atoms, "x", xs)?;
        insert_column(&mut atoms, "y", ys)?;
        insert_column(&mut atoms, "z", zs)?;

        // Optional columns. Naming follows the PDB reader so downstream
        // molvis modifiers (element coloring, backbone ribbon, selection)
        // can consume CIF and PDB frames interchangeably.
        if let Some(ids) = column_u32(&self.atom_cols, &["_atom_site.id", "_atom_site_id"]) {
            insert_column(&mut atoms, "id", ids)?;
        }
        // The species field is kept raw (as the file spells it, e.g. `Fe2+`).
        // A `.` / `?` placeholder is an absent species, not a made-up one: a
        // null row (value `""`) that the explicit `Element::fill_atomic_numbers`
        // step then reports by row.
        if let Some(syms) = string_column(
            &self.atom_cols,
            &["_atom_site.type_symbol", "_atom_site_type_symbol"],
        ) {
            let valid: Vec<bool> = syms.iter().map(|s| !is_placeholder(s)).collect();
            let syms = placeholders_as_empty(syms);
            let n = syms.len();
            let arr = Array1::from_vec(syms)
                .into_shape_with_order(IxDyn(&[n]))
                .map_err(invalid_data)?
                .into_dyn();
            atoms
                .insert("element", arr, Some(valid))
                .map_err(invalid_data)?;
        }
        if let Some(names) = string_column(
            &self.atom_cols,
            &["_atom_site.label_atom_id", "_atom_site_label"],
        ) {
            insert_column(&mut atoms, "name", names)?;
        }
        if let Some(res_names) = string_column(
            &self.atom_cols,
            &["_atom_site.label_comp_id", "_atom_site.auth_comp_id"],
        ) {
            insert_column(&mut atoms, "res_name", res_names)?;
        }
        // The canonical residue number: unsigned, so a negative one is
        // refused here at the boundary (as the PDB reader does), and the CIF
        // placeholders `.` / `?` (a ligand or water with no polymer residue
        // number) are null rows rather than a fake 0.
        if let Some((res_ids, valid)) = column_res_id(
            &self.atom_cols,
            &["_atom_site.label_seq_id", "_atom_site.auth_seq_id"],
        )? {
            let n = res_ids.len();
            let arr = Array1::from_vec(res_ids)
                .into_shape_with_order(IxDyn(&[n]))
                .map_err(invalid_data)?
                .into_dyn();
            atoms
                .insert("res_id", arr, Some(valid))
                .map_err(invalid_data)?;
        }
        if let Some(chains) = string_column(
            &self.atom_cols,
            &["_atom_site.label_asym_id", "_atom_site.auth_asym_id"],
        ) {
            insert_column(&mut atoms, "chain", chains)?;
        }
        if let Some(icodes) = string_column(&self.atom_cols, &["_atom_site.pdbx_PDB_ins_code"]) {
            insert_column(&mut atoms, "icode", placeholders_as_empty(icodes))?;
        }
        if let Some(altlocs) = string_column(&self.atom_cols, &["_atom_site.label_alt_id"]) {
            insert_column(&mut atoms, "altloc", placeholders_as_empty(altlocs))?;
        }
        if let Some(occ) = column_floats(
            &self.atom_cols,
            &["_atom_site.occupancy", "_atom_site_occupancy"],
        ) {
            insert_column(&mut atoms, "occupancy", occ)?;
        }
        if let Some(b) = column_floats(
            &self.atom_cols,
            &["_atom_site.B_iso_or_equiv", "_atom_site_B_iso_or_equiv"],
        ) {
            insert_column(&mut atoms, "b_factor", b)?;
        }

        frame.insert("atoms", atoms);

        frame.simbox = simbox;

        Ok(frame)
    }
}

fn column_floats(map: &HashMap<String, Vec<String>>, keys: &[&str]) -> Option<Vec<F>> {
    for k in keys {
        if let Some(col) = map.get(*k) {
            return Some(col.iter().map(|s| parse_float(s).unwrap_or(0.0)).collect());
        }
    }
    None
}

fn string_column(map: &HashMap<String, Vec<String>>, keys: &[&str]) -> Option<Vec<String>> {
    for k in keys {
        if let Some(col) = map.get(*k) {
            return Some(col.clone());
        }
    }
    None
}

/// Is `token` one of CIF's placeholders: `.` ("not applicable"), `?`
/// ("unknown"), or empty?
fn is_placeholder(token: &str) -> bool {
    matches!(token.trim(), "." | "?" | "")
}

/// A string column with the CIF placeholders read as `""` (none).
fn placeholders_as_empty(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|v| if is_placeholder(&v) { String::new() } else { v })
        .collect()
}

/// Parse the residue-number column from a CIF loop as unsigned `res_id`
/// values plus a validity mask: a placeholder (`.` for ligand / water / metal
/// atoms with no polymer residue numbering, `?`) is a null row.
///
/// # Errors
///
/// `InvalidData` naming the value for a negative or unparseable number —
/// residue ids are unsigned, and a reader renumbers at its boundary rather
/// than storing a fake value.
fn column_res_id(
    map: &HashMap<String, Vec<String>>,
    keys: &[&str],
) -> Result<Option<(Vec<Idx>, Vec<bool>)>> {
    let Some(col) = keys.iter().find_map(|k| map.get(*k)) else {
        return Ok(None);
    };
    let mut values = Vec::with_capacity(col.len());
    let mut valid = Vec::with_capacity(col.len());
    for token in col {
        if is_placeholder(token) {
            values.push(0);
            valid.push(false);
            continue;
        }
        let v = token.trim().parse::<Idx>().map_err(|_| {
            invalid_data(format!(
                "CIF residue number {token:?} is not a non-negative integer; residue ids are \
                 unsigned"
            ))
        })?;
        values.push(v);
        valid.push(true);
    }
    Ok(Some((values, valid)))
}

fn column_u32(map: &HashMap<String, Vec<String>>, keys: &[&str]) -> Option<Vec<Idx>> {
    for k in keys {
        if let Some(col) = map.get(*k) {
            return Some(
                col.iter()
                    .map(|s| s.trim().parse::<Idx>().unwrap_or(0))
                    .collect(),
            );
        }
    }
    None
}

/// Read one CIF data block from `src`. Returns `Ok(None)` at EOF.
fn read_one_block<R: BufRead>(src: &mut LineSource<R>) -> Result<Option<Frame>> {
    let mut current: Option<FrameInProgress> = None;

    while let Some(tok) = src.next_token()? {
        if let Some(name) = tok.strip_prefix("data_") {
            if let Some(prev) = current.take() {
                // Push back: we rebuild a fake "data_" token via the pending stack.
                // Easiest: wrap in a pseudo-line and put back through pending.
                src.pending.push(format!("data_{}", name));
                return Ok(Some(prev.build_frame()?));
            }
            current = Some(FrameInProgress::new(name.to_string()));
            continue;
        }
        let frame = match current.as_mut() {
            Some(f) => f,
            None => continue, // skip stray tokens before any data_ block
        };

        if tok == "loop_" {
            handle_loop(src, frame)?;
            continue;
        }

        if let Some(stripped) = tok.strip_prefix('_') {
            // Key-value: read the next token as value.
            let key = format!("_{}", stripped);
            let value = src
                .next_token()?
                .ok_or_else(|| invalid_data(format!("missing value for {}", key)))?;
            frame.set_cell(&key, &value);
            // Save scalar string values into meta when small.
            if value.len() < 256 && key.starts_with("_chemical_") {
                frame.meta.insert(key, value);
            }
            continue;
        }

        // Stray tokens outside any structure — ignore.
    }

    if let Some(f) = current {
        return Ok(Some(f.build_frame()?));
    }
    Ok(None)
}

/// Handle a `loop_` table. Reads header `_keys` until a non-`_` token, then
/// consumes `n_keys * n_rows` tokens as data values.
fn handle_loop<R: BufRead>(src: &mut LineSource<R>, frame: &mut FrameInProgress) -> Result<()> {
    let mut keys: Vec<String> = Vec::new();

    while let Some(tok) = src.next_token()? {
        if let Some(rest) = tok.strip_prefix('_') {
            keys.push(format!("_{}", rest));
            continue;
        }
        // First non-`_` token is the first data value — push back via pending.
        src.pending.push(tok);
        break;
    }
    if keys.is_empty() {
        return Ok(()); // empty loop_; rare but allowed
    }

    // Only the *coordinate-bearing* atom_site loop is a true atom loop.
    // Anisotropic-displacement loops (`_atom_site_aniso_*`) and similar use
    // the same prefix but lack coordinates and must be skipped. Likewise,
    // ignore any second atom loop in the same block — only the first wins.
    let has_coord_col = keys.iter().any(|k| {
        k == "_atom_site_fract_x"
            || k == "_atom_site.fract_x"
            || k == "_atom_site_Cartn_x"
            || k == "_atom_site.Cartn_x"
    });
    let already_have_atoms = !frame.atom_cols.is_empty();
    let is_atom_loop = has_coord_col && !already_have_atoms;

    let n_cols = keys.len();
    if is_atom_loop {
        // Pre-allocate columns we care about.
        for k in &keys {
            frame.atom_cols.entry(k.clone()).or_default();
        }
    }

    // Collect data rows: read `n_cols` tokens at a time. End the loop when:
    //  - we run out of tokens (EOF), OR
    //  - the next token starts with `_`, `loop_`, `data_`, `save_`, or `global_`.
    'outer: loop {
        let mut row: Vec<String> = Vec::with_capacity(n_cols);
        for col_idx in 0..n_cols {
            let tok = match src.next_token()? {
                Some(t) => t,
                None => {
                    if col_idx == 0 {
                        break 'outer;
                    } else {
                        return Err(invalid_data(format!(
                            "CIF loop ended mid-row (got {} of {} cols)",
                            col_idx, n_cols
                        )));
                    }
                }
            };
            if col_idx == 0 {
                let lower = tok.as_str();
                if lower.starts_with('_')
                    || lower == "loop_"
                    || lower.starts_with("data_")
                    || lower.starts_with("save_")
                    || lower == "global_"
                {
                    src.pending.push(tok);
                    break 'outer;
                }
            }
            row.push(tok);
        }
        if is_atom_loop {
            for (key, val) in keys.iter().zip(row) {
                frame.atom_cols.get_mut(key).unwrap().push(val);
            }
        }
    }

    Ok(())
}

/// `FrameReader`-trait wrapper.
///
/// It covers the CIF subset most molecular work needs:
///
/// - `data_<id>` blocks. Multi-block files yield one [`Frame`] per
///   [`CifReader::read`] call.
/// - Key-value pairs `_key  value`, including parenthesized esd (`5.917(3)`).
/// - `loop_` tables. Only `_atom_site_*` (small-molecule CIF) and
///   `_atom_site.*` (mmCIF) loops are extracted into an atoms block — all
///   other loops are tolerantly skipped.
/// - Cell parameters: `_cell_length_a/b/c`, `_cell_angle_alpha/beta/gamma`.
/// - Atom-site columns: `label`, `type_symbol`, `fract_x/y/z`, `Cartn_x/y/z`,
///   `occupancy`, `B_iso_or_equiv`.
/// - Comments (`#`) and blank lines.
/// - Multi-line strings (`;...;`) are recognised and skipped when they appear
///   as a column value — their content is preserved as a single token.
/// - Quoted strings (`'...'`, `"..."`).
/// - Special values `?` and `.` (treated as empty).
///
/// Out of scope: full CIF dictionary validation, `save_` blocks, `loop_`
/// nesting, and symmetry expansion (the parsed coordinates are exactly what the
/// file holds).
///
/// Over a seekable stream it is also a [`TrajectoryReader`]: `data_` blocks
/// by index, each starting at the line that opens it.
pub struct CifReader<R: BufRead> {
    src: LineSource<R>,
    index: Option<Vec<FrameOffset>>,
}

impl<R: BufRead> Reader for CifReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            src: LineSource::new(reader),
            index: None,
        }
    }
}

impl CifReader<Box<dyn ReadSeek>> {
    /// Open a CIF file, or the bytes of one, for random access by `data_`
    /// block.
    pub fn open<'a>(src: impl Into<Source<'a>>) -> Result<Self> {
        Ok(Self::new(src.into().owned_reader()?))
    }
}

impl<R: BufRead + Seek> TrajectoryReader for CifReader<R> {
    fn build_index(&mut self) -> Result<()> {
        if self.index.is_none() {
            // A `data_` line inside a `;`-delimited text field opens no
            // block: the parser reads the field as one value.
            let mut in_text = false;
            self.index = Some(crate::io::frame_index::index_line_records(
                &mut self.src.reader,
                |line| {
                    if line.starts_with(';') {
                        in_text = !in_text;
                        return false;
                    }
                    !in_text && line.trim_start().starts_with("data_")
                },
            )?);
        }
        Ok(())
    }

    fn read_frame(&mut self, index: usize) -> Result<Option<Frame>> {
        self.build_index()?;
        let Some(at) = self.index.as_ref().and_then(|o| o.get(index).copied()) else {
            return Ok(None);
        };
        let bytes = crate::io::frame_index::frame_bytes(&mut self.src.reader, at)?;
        read_cif(bytes.as_slice()).map(Some)
    }

    fn len(&mut self) -> Result<usize> {
        self.build_index()?;
        Ok(self.index.as_ref().map_or(0, Vec::len))
    }
}

impl<R: BufRead> FrameReader for CifReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(read_one_block(&mut self.src)?)
    }
}

/// Read the first `data_` block of a CIF file.
pub fn read_cif<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let src = src.into();
    CifReader::new(src.reader()?)
        .read()?
        .ok_or_else(|| invalid_data(format!("{}: no CIF data_ block", src.name())))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a frame as a single-block CIF file.
pub fn write_cif<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    let mut w = sink.into().writer()?;
    write_frame_to(&mut w, frame)?;
    w.flush()
}

/// Write a Frame as a single CIF data block. Cell parameters come from
/// `frame.simbox`; atoms from the `atoms` block (Cartesian or fractional).
fn write_frame_to<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("CIF write: frame has no atoms block"))?;
    let n = atoms
        .n_rows()
        .ok_or_else(|| invalid_data("CIF write: atoms block has no rows"))?;

    let title = frame
        .meta
        .get("title")
        .and_then(|value| value.as_str())
        .unwrap_or("molrs");
    writeln!(writer, "data_{}", sanitise_data_name(title))?;

    if let Some(sb) = frame.simbox.as_ref() {
        let lengths = sb.lengths();
        writeln!(writer, "_cell_length_a    {:.6}", lengths[0])?;
        writeln!(writer, "_cell_length_b    {:.6}", lengths[1])?;
        writeln!(writer, "_cell_length_c    {:.6}", lengths[2])?;
        // Compute angles from H columns.
        let h = ndarray::arr2(&sb.matrix());
        let v1 = [h[(0, 0)], h[(1, 0)], h[(2, 0)]];
        let v2 = [h[(0, 1)], h[(1, 1)], h[(2, 1)]];
        let v3 = [h[(0, 2)], h[(1, 2)], h[(2, 2)]];
        let alpha = angle_deg(&v2, &v3);
        let beta = angle_deg(&v1, &v3);
        let gamma = angle_deg(&v1, &v2);
        writeln!(writer, "_cell_angle_alpha {:.6}", alpha)?;
        writeln!(writer, "_cell_angle_beta  {:.6}", beta)?;
        writeln!(writer, "_cell_angle_gamma {:.6}", gamma)?;
    }
    writeln!(writer)?;
    writeln!(writer, "loop_")?;
    writeln!(writer, "_atom_site_label")?;
    writeln!(writer, "_atom_site_type_symbol")?;
    writeln!(writer, "_atom_site_Cartn_x")?;
    writeln!(writer, "_atom_site_Cartn_y")?;
    writeln!(writer, "_atom_site_Cartn_z")?;

    let xs = atoms
        .get("x")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get("y")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get("z")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let labels = atoms.get("name").and_then(|c| c.get::<String>());
    // CIF spells species as `_atom_site_type_symbol`; the writer reads the
    // `element` column and nothing else. An empty value writes CIF's own
    // "unknown" placeholder `?`.
    let symbols = atoms
        .get("element")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| {
            invalid_data(
                "CIF write: atoms block has no string 'element' column; a frame that \
                 carries only atomic_number needs `Element::fill_symbols(&mut frame)` \
                 first",
            )
        })?;

    for i in 0..n {
        let label = labels
            .map(|c| c[[i]].clone())
            .unwrap_or_else(|| format!("X{}", i + 1));
        let sym = match symbols[[i]].trim() {
            "" => "?",
            s => s,
        };
        writeln!(
            writer,
            "{} {} {:.6} {:.6} {:.6}",
            label,
            sym,
            xs[[i]],
            ys[[i]],
            zs[[i]]
        )?;
    }
    Ok(())
}

fn sanitise_data_name(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .collect()
}

fn angle_deg(u: &[F; 3], v: &[F; 3]) -> F {
    let dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    let nu = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
    let nv = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let c = (dot / (nu * nv)).clamp(-1.0, 1.0);
    c.acos().to_degrees()
}

/// A CIF writer: one `data_` block per [`FrameWriter::write`], which
/// [`CifReader`] reads back block by block.
pub struct CifWriter<W: Write> {
    writer: W,
}

impl<'a> CifWriter<SinkWriter<'a>> {
    /// Create a CIF file, or fill a buffer, block by block.
    pub fn create(sink: impl Into<Sink<'a>>) -> Result<Self> {
        Ok(Self {
            writer: sink.into().writer()?,
        })
    }
}

impl<W: Write> Writer for CifWriter<W> {
    type W = W;
    fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W: Write> FrameWriter for CifWriter<W> {
    fn write(&mut self, frame: &Frame) -> Result<()> {
        // Refuse to emit a frame that violates the vocabulary: a bad file
        // looks fine and is found wrong later, by whatever reads it.
        crate::io::writer::check_write_frame(frame)?;
        write_frame_to(&mut self.writer, frame)?;
        self.writer.flush()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const SMALL_CIF: &str = "\
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90.0
_cell_angle_beta  90.0
_cell_angle_gamma 90.0
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.0 0.0 0.0
C2 C 0.5 0.5 0.5
";

    #[test]
    fn reads_small_cif() {
        let mut reader = CifReader::new(Cursor::new(SMALL_CIF.as_bytes()));
        let frame = reader.read().unwrap().unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
        let xs = atoms.get("x").and_then(|c| c.get::<f64>()).unwrap();
        assert!((xs[[1]] - 2.5).abs() < 1e-9, "got {}", xs[[1]]);
        assert!(frame.simbox.is_some());
    }

    #[test]
    fn round_trip_small_cif() {
        let frame = {
            let mut reader = CifReader::new(Cursor::new(SMALL_CIF.as_bytes()));
            reader.read().unwrap().unwrap()
        };
        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let mut reader2 = CifReader::new(Cursor::new(&buf));
        let frame2 = reader2.read().unwrap().unwrap();
        let xs1 = frame
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        let xs2 = frame2
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-4);
        }
    }

    #[test]
    fn a_placeholder_type_symbol_is_a_null_row_and_writes_back_as_unknown() {
        let text = SMALL_CIF.replace("C2 C 0.5", "C2 ? 0.5");
        let frame = CifReader::new(Cursor::new(text.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        let el = atoms
            .get("element")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!((el[[0]].as_str(), el[[1]].as_str()), ("C", ""));
        assert_eq!(atoms.validity("element"), Some(&[true, false][..]));

        let mut buf = Vec::new();
        write_frame_to(&mut buf, &frame).unwrap();
        let out = String::from_utf8(buf).unwrap();
        assert!(out.lines().any(|l| l.starts_with("C2 ? ")), "{out}");
    }

    #[test]
    fn writer_requires_an_element_column() {
        let mut frame = CifReader::new(Cursor::new(SMALL_CIF.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let mut atoms = frame.get("atoms").unwrap().clone();
        atoms.remove("element");
        frame.insert("atoms", atoms);
        let err = write_frame_to(&mut Vec::new(), &frame).unwrap_err();
        assert!(err.to_string().contains("fill_symbols"), "{err}");
    }

    /// `CifReader::open` reads `data_` blocks by index; a `data_` line inside a
    /// `;` text field opens none.
    #[test]
    fn cif_reader_open_reads_blocks_by_index() {
        use crate::io::reader::TrajectoryReader;
        let text = format!(
            "{SMALL_CIF}\n_note\n;\ndata_not_a_block\n;\n{}",
            SMALL_CIF.replacen("data_", "data_second", 1)
        );
        let mut reader = CifReader::open(text.as_bytes()).unwrap();
        assert_eq!(reader.len().unwrap(), 2);
        assert!(reader.read_frame(1).unwrap().is_some());
        assert!(reader.read_frame(2).unwrap().is_none());
    }

    #[test]
    fn esd_strip() {
        assert_eq!(strip_esd("5.917(3)"), "5.917");
        assert_eq!(strip_esd("90.000"), "90.000");
    }

    const MMCIF: &str = "\
data_prot
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N A ALA A 1 B 1.0 2.0 3.0 0.5 10.5
HETATM 2 O O . HOH C . ? 4.0 5.0 6.0 1.0 20.0
";

    #[test]
    fn mmcif_atoms_take_the_canonical_topology_keys() {
        let frame = CifReader::new(Cursor::new(MMCIF.as_bytes()))
            .read()
            .unwrap()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        let strings = |key: &str| -> Vec<String> {
            atoms
                .get(key)
                .and_then(|c| c.get::<String>())
                .unwrap()
                .iter()
                .cloned()
                .collect()
        };
        assert_eq!(strings("chain"), ["A", "C"]);
        assert_eq!(strings("altloc"), ["A", ""]);
        assert_eq!(strings("icode"), ["B", ""]);
        let res_id = atoms.get("res_id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(res_id[[0]], 1);
        assert_eq!(atoms.validity("res_id"), Some(&[true, false][..]));
        let b = atoms.get("b_factor").and_then(|c| c.get::<f64>()).unwrap();
        assert_eq!(b.iter().copied().collect::<Vec<_>>(), [10.5, 20.0]);
        for old in ["chain_id", "res_seq", "b_iso"] {
            assert!(atoms.get(old).is_none(), "{old}");
        }

        let negative = MMCIF.replace("ALA A 1 B", "ALA A -3 B");
        assert!(
            CifReader::new(Cursor::new(negative.as_bytes()))
                .read()
                .is_err()
        );
    }
}
