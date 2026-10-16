//! The Crystallographic Information File (CIF) codec — MVP subset.

use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use molrs::core::MetaValue;
use molrs::core::keys;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, BufWriter, Result, Write};
use std::path::Path;

use ndarray::{Array1, Array2, IxDyn, array};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MetaMap;
use molrs::core::SimBox;
use molrs::op::{F, Idx};

use crate::io::reader::{FrameReader, Reader};
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

/// Data names whose values are the symmetry-operator triplets: core CIF 1.0
/// (`_symmetry_equiv_pos_as_xyz`), core CIF 1.1 / 2.0
/// (`_space_group_symop_operation_xyz`) and their DDL2 (mmCIF) spellings.
const SYMOP_TAGS: &[&str] = &[
    "_symmetry_equiv_pos_as_xyz",
    "_space_group_symop_operation_xyz",
    "_space_group_symop.operation_xyz",
    "_symmetry_equiv.pos_as_xyz",
];

/// Data names whose value is the Hermann–Mauguin space-group symbol, in the
/// same four dialects.
const SPACE_GROUP_TAGS: &[&str] = &[
    "_symmetry_space_group_name_H-M",
    "_space_group_name_H-M_alt",
    "_space_group.name_H-M_alt",
    "_symmetry.space_group_name_H-M",
];

/// Is `key` one of `tags`? CIF data names are case-insensitive.
fn is_tag(key: &str, tags: &[&str]) -> bool {
    tags.iter().any(|tag| key.eq_ignore_ascii_case(tag))
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
    /// The first symmetry-operator list the block states, verbatim.
    symops: Option<Vec<String>>,
    /// The first Hermann–Mauguin symbol the block states.
    space_group: Option<String>,
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
            symops: None,
            space_group: None,
            atom_cols: HashMap::new(),
        }
    }

    /// Keep a key-value symmetry tag: the space-group symbol, or a lone
    /// operator written as a tag instead of a one-row loop. The first
    /// statement wins; a placeholder (`?`, `.`) states nothing.
    fn set_symmetry(&mut self, key: &str, value: &str) {
        if is_placeholder(value) {
            return;
        }
        if is_tag(key, SPACE_GROUP_TAGS) && self.space_group.is_none() {
            self.space_group = Some(value.trim().to_string());
        } else if is_tag(key, SYMOP_TAGS) && self.symops.is_none() {
            self.symops = Some(vec![value.to_string()]);
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
        if let Some(symbol) = self.space_group {
            frame.meta.insert(keys::SPACE_GROUP, symbol);
        }
        if let Some(symops) = self.symops.filter(|ops| !ops.is_empty()) {
            let ops = symops.into_iter().map(serde_json::Value::String).collect();
            frame
                .meta
                .insert(keys::SYMOPS, MetaValue::Json(serde_json::Value::Array(ops)));
        }

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
            let h = SimBox::matrix_from_lengths_angles(lengths, angles)
                .map_err(|e| invalid_data(format!("CIF cell → SimBox: {e:?}")))?;
            Some(
                SimBox::new(h, array![0.0 as F, 0.0, 0.0], [true; 3])
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
        if let Some(syms) = string_column(
            &self.atom_cols,
            &["_atom_site.type_symbol", "_atom_site_type_symbol"],
        ) {
            insert_column(&mut atoms, "element", syms)?;
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
                .insert_nullable("res_id", arr, valid)
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
            frame.set_symmetry(&key, &value);
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

    // The symmetry-operator column of the first loop that has one; any
    // other column of that loop (`_symmetry_equiv_pos_site_id`, …) is not
    // kept.
    let symop_col = keys
        .iter()
        .position(|k| is_tag(k, SYMOP_TAGS))
        .filter(|_| frame.symops.is_none());
    let mut symops = Vec::new();

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
        if let Some(col) = symop_col {
            symops.push(row[col].clone());
        }
        if is_atom_loop {
            for (key, val) in keys.iter().zip(row) {
                frame.atom_cols.get_mut(key).unwrap().push(val);
            }
        }
    }
    if symop_col.is_some() {
        frame.symops = Some(symops);
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
/// - Symmetry, kept as frame metadata: the operator triplets of the first
///   `_symmetry_equiv_pos_as_xyz` / `_space_group_symop_operation_xyz` /
///   `_space_group_symop.operation_xyz` loop under
///   [`keys::SYMOPS`](crate::core::keys::SYMOPS) (a JSON array of strings,
///   verbatim, in file order), and the `_symmetry_space_group_name_H-M` /
///   `_space_group_name_H-M_alt` / `_space_group.name_H-M_alt` symbol under
///   [`keys::SPACE_GROUP`](crate::core::keys::SPACE_GROUP) (a string). The
///   mmCIF `_symmetry_equiv.pos_as_xyz` and `_symmetry.space_group_name_H-M`
///   are read too. [`CifWriter`] writes both back.
/// - Atom-site columns: `label`, `type_symbol`, `fract_x/y/z`, `Cartn_x/y/z`,
///   `occupancy`, `B_iso_or_equiv`.
/// - Comments (`#`) and blank lines.
/// - Multi-line strings (`;...;`) are recognised and skipped when they appear
///   as a column value — their content is preserved as a single token.
/// - Quoted strings (`'...'`, `"..."`).
/// - Special values `?` and `.` (treated as empty).
///
/// Out of scope: full CIF dictionary validation, `save_` blocks, `loop_`
/// nesting, and applying the symmetry operators. The atoms are exactly the
/// sites the file lists (the asymmetric unit, for most crystal CIFs); the
/// operators ride along in the metadata for a caller that expands them.
pub struct CifReader<R: BufRead> {
    src: LineSource<R>,
}

impl<R: BufRead> Reader for CifReader<R> {
    type R = R;
    fn new(reader: R) -> Self {
        Self {
            src: LineSource::new(reader),
        }
    }
}

impl<R: BufRead> FrameReader for CifReader<R> {
    fn read(&mut self) -> Result<Option<Frame>> {
        // Validate on the way out: a frame that violates the vocabulary
        // is a malformed file or a reader bug, not a result to return.
        crate::io::reader::check_read_frame(read_one_block(&mut self.src)?)
    }
}

/// Read every `data_` block of the CIF file at `path`.
pub fn read_cif_trajectory<P: AsRef<Path>>(path: P) -> Result<Vec<Frame>> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut cr = CifReader::new(BufReader::new(file));
    crate::io::reader::collect_frames(&mut cr)
}

/// Read the first block from `path`.
pub fn read_cif<P: AsRef<Path>>(path: P) -> Result<Frame> {
    let file = std::fs::File::open(path.as_ref())?;
    let mut cr = CifReader::new(BufReader::new(file));
    cr.read()?
        .ok_or_else(|| invalid_data("CIF file has no data_ block"))
}

/// Read the first `data_` block of CIF `text` — [`read_cif`] on text in
/// memory.
pub fn read_cif_str(text: &str) -> Result<Frame> {
    CifReader::new(text.as_bytes())
        .read()?
        .ok_or_else(|| invalid_data("CIF text has no data_ block"))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Write a Frame as a single-block CIF file at `path`.
pub fn write_cif<P: AsRef<Path>>(path: P, frame: &Frame) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    let mut w = BufWriter::new(file);
    write_frame_to(&mut w, frame)?;
    w.flush()
}

/// Write a frame as single-block CIF text — [`write_cif`] into memory.
pub fn write_cif_str(frame: &Frame) -> Result<String> {
    let mut buf = Vec::new();
    write_frame_to(&mut buf, frame)?;
    String::from_utf8(buf).map_err(invalid_data)
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
        let h = sb.h_view();
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
    write_symmetry(writer, frame)?;
    writeln!(writer)?;
    writeln!(writer, "loop_")?;
    writeln!(writer, "_atom_site_label")?;
    writeln!(writer, "_atom_site_type_symbol")?;
    writeln!(writer, "_atom_site_Cartn_x")?;
    writeln!(writer, "_atom_site_Cartn_y")?;
    writeln!(writer, "_atom_site_Cartn_z")?;

    let xs = atoms
        .get("x")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.x missing"))?;
    let ys = atoms
        .get("y")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.y missing"))?;
    let zs = atoms
        .get("z")
        .and_then(|c| c.as_float())
        .ok_or_else(|| invalid_data("atoms.z missing"))?;
    let labels = atoms.get("name").and_then(|c| c.as_string());
    let symbols = atoms.get("element").and_then(|c| c.as_string());

    for i in 0..n {
        let label = labels
            .map(|c| c[[i]].clone())
            .unwrap_or_else(|| format!("X{}", i + 1));
        let sym = symbols.map(|c| c[[i]].clone()).unwrap_or("X".to_string());
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

/// Write the frame's [`keys::SPACE_GROUP`] and [`keys::SYMOPS`] metadata
/// back as the `_space_group_name_H-M_alt` tag and the
/// `_space_group_symop_operation_xyz` loop, so a CIF read and written again
/// keeps its symmetry.
///
/// # Errors
///
/// `InvalidData` when either key holds a value of the wrong form: the symbol
/// must be a string, the operators a JSON array of strings.
fn write_symmetry<W: Write>(writer: &mut W, frame: &Frame) -> Result<()> {
    if let Some(value) = frame.meta.get(keys::SPACE_GROUP) {
        let symbol = value.as_str().ok_or_else(|| {
            invalid_data(format!(
                "CIF write: meta '{}' must be a string, got {}",
                keys::SPACE_GROUP,
                value.dtype()
            ))
        })?;
        writeln!(writer, "_space_group_name_H-M_alt {}", quote_value(symbol))?;
    }
    let Some(value) = frame.meta.get(keys::SYMOPS) else {
        return Ok(());
    };
    let ops = symop_strings(value).ok_or_else(|| {
        invalid_data(format!(
            "CIF write: meta '{}' must be a JSON array of operator strings",
            keys::SYMOPS
        ))
    })?;
    if ops.is_empty() {
        return Ok(());
    }
    writeln!(writer)?;
    writeln!(writer, "loop_")?;
    writeln!(writer, "_space_group_symop_operation_xyz")?;
    for op in ops {
        writeln!(writer, "{}", quote_value(op))?;
    }
    Ok(())
}

/// The operator strings of a [`keys::SYMOPS`] value, or `None` when it is
/// not a JSON array of strings.
fn symop_strings(value: &MetaValue) -> Option<Vec<&str>> {
    let MetaValue::Json(serde_json::Value::Array(items)) = value else {
        return None;
    };
    items.iter().map(serde_json::Value::as_str).collect()
}

/// A CIF value token: quoted when it holds whitespace or a quote, or starts
/// with a character that would read as something else.
fn quote_value(value: &str) -> String {
    let plain = !value.is_empty()
        && !value
            .chars()
            .any(|c| c.is_whitespace() || c == '\'' || c == '"')
        && !value.starts_with(['_', '#', '$', ';', '[', ']']);
    if plain {
        value.to_string()
    } else if value.contains('\'') {
        format!("\"{value}\"")
    } else {
        format!("'{value}'")
    }
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

/// `FrameWriter`-trait wrapper.
pub struct CifWriter<W: Write> {
    writer: W,
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
        write_frame_to(&mut self.writer, frame)
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
        let xs = atoms.get("x").and_then(|c| c.as_float()).unwrap();
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
            .and_then(|c| c.as_float())
            .unwrap();
        let xs2 = frame2
            .get("atoms")
            .unwrap()
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap();
        for i in 0..xs1.len() {
            assert!((xs1[[i]] - xs2[[i]]).abs() < 1e-4);
        }
    }

    #[test]
    fn esd_strip() {
        assert_eq!(strip_esd("5.917(3)"), "5.917");
        assert_eq!(strip_esd("90.000"), "90.000");
    }

    const P21C_CIF: &str = "\
data_p21c
_symmetry_space_group_name_H-M   'P 21/c'
_cell_length_a 5.0
_cell_length_b 6.0
_cell_length_c 7.0
_cell_angle_alpha 90.0
_cell_angle_beta  100.0
_cell_angle_gamma 90.0
loop_
_symmetry_equiv_pos_site_id
_symmetry_equiv_pos_as_xyz
1 'x, y, z'
2 '-x, y+1/2, -z+1/2'
3 '-x, -y, -z'
4 'x, -y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
";

    const P21C_OPS: [&str; 4] = [
        "x, y, z",
        "-x, y+1/2, -z+1/2",
        "-x, -y, -z",
        "x, -y+1/2, z+1/2",
    ];

    fn symops(frame: &Frame) -> Vec<String> {
        let value = frame.meta.get(keys::SYMOPS).expect("symops meta");
        symop_strings(value)
            .expect("a JSON array of strings")
            .into_iter()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn keeps_the_symmetry_operators_and_space_group_verbatim() {
        let frame = read_cif_str(P21C_CIF).unwrap();
        assert_eq!(symops(&frame), P21C_OPS);
        assert_eq!(
            frame.meta.get(keys::SPACE_GROUP).and_then(|v| v.as_str()),
            Some("P 21/c")
        );
        // The atoms stay the asymmetric unit: nothing is expanded.
        assert_eq!(frame.get("atoms").unwrap().n_rows(), Some(1));
    }

    #[test]
    fn reads_the_modern_and_ddl2_symmetry_tags() {
        let modern = P21C_CIF
            .replace(
                "_symmetry_space_group_name_H-M",
                "_space_group_name_H-M_alt",
            )
            .replace("_symmetry_equiv_pos_site_id", "_space_group_symop_id")
            .replace(
                "_symmetry_equiv_pos_as_xyz",
                "_space_group_symop_operation_xyz",
            );
        let ddl2 = P21C_CIF
            .replace(
                "_symmetry_space_group_name_H-M",
                "_space_group.name_H-M_alt",
            )
            .replace("_symmetry_equiv_pos_site_id", "_space_group_symop.id")
            .replace(
                "_symmetry_equiv_pos_as_xyz",
                "_space_group_symop.operation_xyz",
            );
        for text in [modern, ddl2] {
            let frame = read_cif_str(&text).unwrap();
            assert_eq!(symops(&frame), P21C_OPS);
            assert_eq!(
                frame.meta.get(keys::SPACE_GROUP).and_then(|v| v.as_str()),
                Some("P 21/c")
            );
        }
    }

    #[test]
    fn a_block_without_symmetry_has_no_symmetry_meta() {
        let frame = read_cif_str(SMALL_CIF).unwrap();
        assert!(frame.meta.get(keys::SYMOPS).is_none());
        assert!(frame.meta.get(keys::SPACE_GROUP).is_none());
        let unknown = SMALL_CIF.replace("data_test\n", "data_test\n_space_group_name_H-M_alt ?\n");
        let frame = read_cif_str(&unknown).unwrap();
        assert!(frame.meta.get(keys::SPACE_GROUP).is_none());
    }

    #[test]
    fn symmetry_round_trips_through_the_writer() {
        let frame = read_cif_str(P21C_CIF).unwrap();
        let text = write_cif_str(&frame).unwrap();
        let back = read_cif_str(&text).unwrap();
        assert_eq!(symops(&back), P21C_OPS);
        assert_eq!(
            back.meta.get(keys::SPACE_GROUP).and_then(|v| v.as_str()),
            Some("P 21/c")
        );
        assert_eq!(back.get("atoms").unwrap().n_rows(), Some(1));
    }

    #[test]
    fn writer_refuses_malformed_symmetry_meta() {
        let mut frame = read_cif_str(SMALL_CIF).unwrap();
        frame.meta.insert(keys::SYMOPS, "x,y,z");
        assert!(write_cif_str(&frame).is_err());
        let mut frame = read_cif_str(SMALL_CIF).unwrap();
        frame.meta.insert(keys::SPACE_GROUP, 14_i64);
        assert!(write_cif_str(&frame).is_err());
    }

    #[test]
    fn quote_value_quotes_only_what_needs_it() {
        assert_eq!(quote_value("x,y,z"), "x,y,z");
        assert_eq!(quote_value("x, y, z"), "'x, y, z'");
        assert_eq!(quote_value("P 21/c"), "'P 21/c'");
        assert_eq!(quote_value("it's"), "\"it's\"");
        assert_eq!(quote_value(""), "''");
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
                .and_then(|c| c.as_string())
                .unwrap()
                .iter()
                .cloned()
                .collect()
        };
        assert_eq!(strings("chain"), ["A", "C"]);
        assert_eq!(strings("altloc"), ["A", ""]);
        assert_eq!(strings("icode"), ["B", ""]);
        let res_id = atoms.get("res_id").and_then(|c| c.as_uint()).unwrap();
        assert_eq!(res_id[[0]], 1);
        assert_eq!(atoms.validity("res_id"), Some(&[true, false][..]));
        let b = atoms.get("b_factor").and_then(|c| c.as_float()).unwrap();
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
