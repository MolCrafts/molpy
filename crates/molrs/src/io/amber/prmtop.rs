//! AMBER prmtop **structure** reader — AMBER files and the CHARMM files ParmEd's `chamber` writes in the same format (`%FLAG CTITLE`).

use crate::core::Cell;
use crate::io::frame_columns::insert_column;
use crate::io::{Source, invalid_data};
use std::collections::HashMap;
use std::io::{BufRead, Error, Result};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeName;
use molrs::core::keys;
use molrs::core::schema::block_names;
use molrs::op::{F, Idx};

use super::prmtop_tables;
use crate::core::constants::AMBER_CHARGE_FACTOR;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split whitespace-joined section lines into tokens and parse as `T`.
fn parse_tokens<T: std::str::FromStr>(lines: &[String]) -> Result<Vec<T>>
where
    T::Err: std::fmt::Display,
{
    prmtop_tables::parse_tokens(lines).map_err(invalid_data)
}

fn unsupported(what: &str) -> Error {
    invalid_data(format!("{what} are not supported"))
}

fn length_mismatch(flag: &str, got: usize, expected: impl std::fmt::Display) -> Error {
    invalid_data(format!("{flag} has {got} entries, expected {expected}"))
}

// ---------------------------------------------------------------------------
// Section map
// ---------------------------------------------------------------------------

/// Field width of a fixed-width integer `%FORMAT` (`10I8`, `1I8`); `None`
/// for real and character formats.
///
/// Only integers need it: an `E16.8` value is at most 15 characters, so real
/// fields are always blank-separated (and some writers print them with more
/// digits than the width, which a fixed-width cut would split).
fn integer_field_width(format_line: &str) -> Option<usize> {
    let spec = format_line.split('(').nth(1)?.split(')').next()?;
    let pos = spec.find(['I', 'i'])?;
    let width: String = spec[pos + 1..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    width.parse().ok().filter(|&w: &usize| w > 0)
}

/// Re-tokenise one `nIw` data line with single spaces.
///
/// Fortran fixed-width fields need not be separated: in `10I8` a negative
/// index of 7 digits fills all 8 columns (`-1000011`) and touches its
/// neighbour, so a whitespace split merges two values — topology sections
/// of systems above ~3×10⁵ atoms hit this. Every downstream parser splits on
/// whitespace, so the fields are cut here once. A line whose fixed-width
/// cut does not yield integers (a free-format writer) keeps its whitespace
/// tokens.
fn split_integer_fields(raw: &str, width: usize) -> String {
    let is_int = |f: &str| {
        let digits = f.strip_prefix(['-', '+']).unwrap_or(f);
        !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
    };
    let free_format = || raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut fields = Vec::new();
    for chunk in raw.as_bytes().chunks(width) {
        let Ok(field) = std::str::from_utf8(chunk) else {
            return free_format();
        };
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        if !is_int(field) {
            return free_format();
        }
        fields.push(field);
    }
    fields.join(" ")
}

/// Parse a prmtop into flag → data lines (sanitized, non-empty).
///
/// Public for the FF reader and for Python helpers that still need raw
/// parameter tables (`BOND_FORCE_CONSTANT`, …) without re-implementing the
/// `%FLAG` / `%COMMENT` scan in molpy.
pub(crate) fn parse_flag_sections<R: BufRead>(
    mut reader: R,
) -> Result<HashMap<String, Vec<String>>> {
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut flag: Option<String> = None;
    let mut data: Vec<String> = Vec::new();
    let mut numeric_width: Option<usize> = None;
    let mut buf = String::new();

    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let line = buf.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("%FORMAT") {
            numeric_width = integer_field_width(line);
            continue;
        }
        if line.starts_with("%FLAG") {
            if let Some(f) = flag.take() {
                sections
                    .entry(f)
                    .or_default()
                    .extend(std::mem::take(&mut data));
            }
            let name = line
                .split_whitespace()
                .nth(1)
                .ok_or_else(|| invalid_data("malformed %FLAG line"))?
                .to_string();
            flag = Some(name);
            data = Vec::new();
            numeric_width = None;
        } else if line.starts_with("%VERSION") || line.starts_with("%COMMENT") {
            // Amber FileFormats: any number of %COMMENT lines may appear
            // between %FLAG and data; ignore them.
        } else if let Some(width) = numeric_width {
            data.push(split_integer_fields(
                buf.trim_end_matches(['\n', '\r']),
                width,
            ));
        } else {
            data.push(line.to_string());
        }
    }
    if let Some(f) = flag {
        sections.entry(f).or_default().extend(data);
    }
    Ok(sections)
}

// ---------------------------------------------------------------------------
// POINTERS
// ---------------------------------------------------------------------------

/// The POINTERS map ([`parse_pointers`](prmtop_tables::parse_pointers)),
/// which a structure read needs to name `NATOM`.
fn read_pointers(lines: &[String]) -> Result<HashMap<String, i64>> {
    let meta = prmtop_tables::parse_pointers(lines).map_err(invalid_data)?;
    if !meta.contains_key("NATOM") {
        return Err(invalid_data("POINTERS missing NATOM"));
    }
    Ok(meta)
}

// ---------------------------------------------------------------------------
// Connectivity decode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BondRow {
    type_id: Idx,
    atomi: Idx,
    atomj: Idx,
    type_name: String,
}

#[derive(Debug, Clone)]
struct AngleRow {
    type_id: Idx,
    atomi: Idx,
    atomj: Idx,
    atomk: Idx,
    type_name: String,
}

#[derive(Debug, Clone)]
struct DihedralRow {
    atomi: Idx,
    atomj: Idx,
    atomk: Idx,
    atoml: Idx,
    type_name: String,
    /// True when the raw 4th pointer was negative (Amber improper flag).
    is_improper: bool,
    /// True when every merged row's raw 3rd pointer was negative (Amber 1-4
    /// suppression).
    exclude_14: bool,
}

fn decode_bonds(pointers: &[i64], atom_types: &[String]) -> Result<Vec<BondRow>> {
    if !pointers.len().is_multiple_of(3) {
        return Err(invalid_data(format!(
            "bond pointer length {} not multiple of 3",
            pointers.len()
        )));
    }
    let mut out = Vec::with_capacity(pointers.len() / 3);
    for chunk in pointers.as_chunks::<3>().0 {
        let a = chunk[0];
        let b = chunk[1];
        if a < 0 || b < 0 {
            return Err(invalid_data(format!(
                "Found negative bonded atom pointers ({a}, {b})"
            )));
        }
        let type_id = chunk[2] as Idx;
        let mut i = (a / 3) as Idx; // 0-based
        let mut j = (b / 3) as Idx;
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        let ti = atom_types
            .get(i as usize)
            .ok_or_else(|| invalid_data(format!("bond atom index {i} out of range")))?;
        let tj = atom_types
            .get(j as usize)
            .ok_or_else(|| invalid_data(format!("bond atom index {j} out of range")))?;
        let type_name = TypeName::join(&TypeName::orient(&[ti, tj]))
            .map_err(invalid_data)?
            .to_string();
        out.push(BondRow {
            type_id,
            atomi: i,
            atomj: j,
            type_name,
        });
    }
    Ok(out)
}

fn decode_angles(pointers: &[i64], atom_types: &[String]) -> Result<Vec<AngleRow>> {
    if !pointers.len().is_multiple_of(4) {
        return Err(invalid_data(format!(
            "angle pointer length {} not multiple of 4",
            pointers.len()
        )));
    }
    let mut out = Vec::with_capacity(pointers.len() / 4);
    for chunk in pointers.as_chunks::<4>().0 {
        let a = chunk[0];
        let b = chunk[1];
        let c = chunk[2];
        if a < 0 || b < 0 || c < 0 {
            return Err(invalid_data(format!(
                "Found negative angle atom pointers ({a}, {b}, {c})"
            )));
        }
        let type_id = chunk[3] as Idx;
        let mut i = (a / 3) as Idx;
        let j = (b / 3) as Idx;
        let mut k = (c / 3) as Idx;
        if i > k {
            std::mem::swap(&mut i, &mut k);
        }
        let ti = atom_types
            .get(i as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {i} out of range")))?;
        let tj = atom_types
            .get(j as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {j} out of range")))?;
        let tk = atom_types
            .get(k as usize)
            .ok_or_else(|| invalid_data(format!("angle atom index {k} out of range")))?;
        // The label is the force-field reader's type name: the types in
        // `TypeName::orient`'s spelling. An angle reads the same both ways,
        // so the row's atom order stays as it is.
        let type_name = TypeName::join(&TypeName::orient(&[ti, tj, tk]))
            .map_err(invalid_data)?
            .to_string();
        out.push(AngleRow {
            type_id,
            atomi: i,
            atomj: j,
            atomk: k,
            type_name,
        });
    }
    Ok(out)
}

/// The `dihedrals` and `impropers` rows of the raw `DIHEDRALS_*` pointer rows
/// (`prmtop_tables::decode_torsions`), the force-field reader's names on
/// every row.
///
/// - A proper is one row per atom quartet: the rows of a multi-term torsion
///   share the quartet, and the force-field reader's `dihedral periodic` type
///   holds every term. A quartet tleap gave two different sets of terms is
///   two types, the second `<quartet>@<n>`
///   (`prmtop_tables::proper_type_names`). It is oriented as the force-field reader names it:
///   reversed when [`TypeName::reads_reversed`] says its types are, atoms and
///   name together. Counts therefore match the torsions, not the cosine terms.
/// - An improper (negative 4th pointer) keeps its prmtop atom order: AMBER
///   puts the central atom third, and reversing it names a different term.
///   A single-term improper is one row; a multi-term one (several rows on one
///   quartet, or a negative-PN chain) is one row per term, named
///   `<quartet>@<n>` (`prmtop_tables::Torsion::improper_rows`), because
///   `improper periodic` holds one term. Without the parameter tables an
///   improper is one row.
/// - `exclude_14` of a merged torsion is set only when every merged row set
///   it: AMBER flags all but one term so the 1-4 pair is counted once.
///
/// Rows keep the order of their quartet's first prmtop row.
fn decode_dihedrals(
    pointers: &[i64],
    atom_types: &[String],
    tables: Option<prmtop_tables::TorsionTables<'_>>,
) -> Result<Vec<DihedralRow>> {
    let torsions =
        prmtop_tables::decode_torsions(pointers, atom_types, tables).map_err(invalid_data)?;
    let proper_names =
        prmtop_tables::proper_type_names(&torsions, atom_types).map_err(invalid_data)?;
    let mut out = Vec::with_capacity(torsions.len());
    for (t, proper_name) in torsions.into_iter().zip(proper_names) {
        let names: Vec<String> = match proper_name {
            Some(name) => vec![name],
            None => t
                .improper_rows(atom_types)
                .map_err(invalid_data)?
                .into_iter()
                .map(|(name, _)| name)
                .collect(),
        };
        for type_name in names {
            out.push(DihedralRow {
                atomi: t.atoms[0] as Idx,
                atomj: t.atoms[1] as Idx,
                atomk: t.atoms[2] as Idx,
                atoml: t.atoms[3] as Idx,
                type_name,
                is_improper: t.improper,
                exclude_14: t.exclude_14,
            });
        }
    }
    Ok(out)
}

fn residue_ids(pointer_lines: Option<&Vec<String>>, n_atoms: usize) -> Result<Vec<Idx>> {
    let Some(lines) = pointer_lines else {
        return Ok(vec![0; n_atoms]);
    };
    let mut ptrs: Vec<i64> = parse_tokens(lines)?;
    if ptrs.is_empty() {
        return Ok(vec![0; n_atoms]);
    }
    // RESIDUE_POINTER is 1-based start atom; append sentinel n_atoms+1.
    ptrs.push((n_atoms as i64) + 1);
    let mut res_id = Vec::with_capacity(n_atoms);
    for (r, window) in ptrs.windows(2).enumerate() {
        let start = (window[0] - 1) as usize;
        let end = (window[1] - 1) as usize;
        if end < start || end > n_atoms {
            return Err(invalid_data(format!(
                "bad RESIDUE_POINTER slice {start}..{end} for n_atoms={n_atoms}"
            )));
        }
        for _ in start..end {
            res_id.push(r as Idx);
        }
    }
    if res_id.len() != n_atoms {
        return Err(invalid_data(format!(
            "RESIDUE_POINTER assigned {} atoms, expected {n_atoms}",
            res_id.len()
        )));
    }
    Ok(res_id)
}

// ---------------------------------------------------------------------------
// Build Frame
// ---------------------------------------------------------------------------

fn build_bond_block(rows: &[BondRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut type_id = Vec::with_capacity(n);
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
    }
    insert_column(&mut block, "type_id", type_id)?;
    insert_column(&mut block, "atomi", atomi)?;
    insert_column(&mut block, "atomj", atomj)?;
    insert_column(&mut block, "type", type_name)?;
    insert_column(&mut block, "id", id)?;
    Ok(block)
}

fn build_angle_block(rows: &[AngleRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut type_id = Vec::with_capacity(n);
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        type_id.push(r.type_id);
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
    }
    insert_column(&mut block, "type_id", type_id)?;
    insert_column(&mut block, "atomi", atomi)?;
    insert_column(&mut block, "atomj", atomj)?;
    insert_column(&mut block, "atomk", atomk)?;
    insert_column(&mut block, "type", type_name)?;
    insert_column(&mut block, "id", id)?;
    Ok(block)
}

fn build_dihedral_block(rows: &[DihedralRow]) -> Result<Block> {
    let n = rows.len();
    let mut block = Block::new();
    let mut atomi = Vec::with_capacity(n);
    let mut atomj = Vec::with_capacity(n);
    let mut atomk = Vec::with_capacity(n);
    let mut atoml = Vec::with_capacity(n);
    let mut type_name = Vec::with_capacity(n);
    let mut id = Vec::with_capacity(n);
    let mut exclude_14 = Vec::with_capacity(n);
    for (idx, r) in rows.iter().enumerate() {
        atomi.push(r.atomi);
        atomj.push(r.atomj);
        atomk.push(r.atomk);
        atoml.push(r.atoml);
        type_name.push(r.type_name.clone());
        id.push((idx as Idx) + 1);
        exclude_14.push(r.exclude_14);
    }
    insert_column(&mut block, "atomi", atomi)?;
    insert_column(&mut block, "atomj", atomj)?;
    insert_column(&mut block, "atomk", atomk)?;
    insert_column(&mut block, "atoml", atoml)?;
    insert_column(&mut block, "type", type_name)?;
    insert_column(&mut block, "id", id)?;
    insert_column(&mut block, keys::EXCLUDE_14, exclude_14)?;
    Ok(block)
}

fn section_ints(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<i64>> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

fn first_i64(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Option<i64>> {
    let Some(lines) = sections.get(key) else {
        return Ok(None);
    };
    let vals: Vec<i64> = parse_tokens(lines)?;
    Ok(vals.first().copied())
}

fn trim_trailing_empty(mut names: Vec<String>) -> Vec<String> {
    while names.last().is_some_and(|s| s.is_empty()) {
        names.pop();
    }
    names
}

fn a4_exact(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    expected: usize,
) -> Result<Option<Vec<String>>> {
    let Some(lines) = sections.get(flag) else {
        return Ok(None);
    };
    let names = trim_trailing_empty(prmtop_tables::parse_a4_names(lines));
    if names.len() != expected {
        return Err(length_mismatch(flag, names.len(), expected));
    }
    Ok(Some(names))
}

fn floats_n(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    expected: usize,
) -> Result<Option<Vec<F>>> {
    let Some(lines) = sections.get(flag) else {
        return Ok(None);
    };
    let vals: Vec<F> = parse_tokens(lines)?;
    if vals.len() != expected {
        return Err(length_mismatch(flag, vals.len(), expected));
    }
    Ok(Some(vals))
}

fn check_dead_int_section(
    sections: &HashMap<String, Vec<String>>,
    flag: &str,
    n_atoms: usize,
) -> Result<()> {
    let Some(lines) = sections.get(flag) else {
        return Ok(());
    };
    let vals: Vec<i64> = parse_tokens(lines)?;
    if vals.len() != n_atoms {
        return Err(length_mismatch(flag, vals.len(), n_atoms));
    }
    Ok(())
}

fn refuse_unsupported(
    sections: &HashMap<String, Vec<String>>,
    meta_map: &HashMap<String, i64>,
) -> Result<()> {
    if meta_map.get("IFPERT").copied().unwrap_or(0) > 0 {
        return Err(unsupported("perturbed (IFPERT>0) prmtop files"));
    }
    if meta_map.get("IFCAP").copied().unwrap_or(0) > 0 {
        return Err(unsupported("solvent-cap (IFCAP>0) prmtop files"));
    }
    if meta_map.get("IFBOX").copied().unwrap_or(0) == 3 {
        return Err(unsupported("IFBOX=3 prmtop files"));
    }
    if sections.contains_key("LENNARD_JONES_CCOEF") {
        return Err(unsupported("12-6-4 Lennard-Jones prmtop files"));
    }
    if first_i64(sections, "IPOL")?.unwrap_or(0) > 0 {
        return Err(unsupported("polarizable (IPOL > 0) prmtop files"));
    }
    if let Some(lines) = sections.get("NONBONDED_PARM_INDEX") {
        let vals: Vec<i64> = parse_tokens(lines)?;
        if vals.iter().any(|&v| v < 0) {
            return Err(unsupported("10-12 hydrogen-bond prmtop files"));
        }
    }
    Ok(())
}

fn residue_names(
    sections: &HashMap<String, Vec<String>>,
    res_ids: &[Idx],
    n_res: usize,
) -> Result<Option<Vec<String>>> {
    let Some(labels) = a4_exact(sections, "RESIDUE_LABEL", n_res)? else {
        return Ok(None);
    };
    let mut names = Vec::with_capacity(res_ids.len());
    for &rid in res_ids {
        let name = labels
            .get(rid as usize)
            .ok_or_else(|| invalid_data(format!("res_id {rid} out of range for RESIDUE_LABEL")))?;
        names.push(name.clone());
    }
    Ok(Some(names))
}

fn molecule_ids(
    sections: &HashMap<String, Vec<String>>,
    n_atoms: usize,
) -> Result<Option<Vec<Idx>>> {
    let Some(lines) = sections.get("ATOMS_PER_MOLECULE") else {
        return Ok(None);
    };
    let counts: Vec<i64> = parse_tokens(lines)?;
    let mut mol_id = Vec::with_capacity(n_atoms);
    for (i, &n) in counts.iter().enumerate() {
        if n < 0 {
            return Err(invalid_data(format!(
                "ATOMS_PER_MOLECULE entry {n} is negative"
            )));
        }
        let id = (i as Idx) + 1;
        for _ in 0..n {
            mol_id.push(id);
        }
    }
    if mol_id.len() != n_atoms {
        return Err(length_mismatch("ATOMS_PER_MOLECULE", mol_id.len(), n_atoms));
    }
    Ok(Some(mol_id))
}

fn build_exclusions_block(
    sections: &HashMap<String, Vec<String>>,
    nnb: i64,
    n_atoms: usize,
) -> Result<Option<Block>> {
    let has_numex = sections.contains_key("NUMBER_EXCLUDED_ATOMS");
    let has_list = sections.contains_key("EXCLUDED_ATOMS_LIST");
    if !has_numex && !has_list {
        return Ok(None);
    }
    if has_numex != has_list {
        return Err(invalid_data(
            "NUMBER_EXCLUDED_ATOMS and EXCLUDED_ATOMS_LIST must appear together",
        ));
    }
    let numex: Vec<i64> = parse_tokens(&sections["NUMBER_EXCLUDED_ATOMS"])?;
    let list: Vec<i64> = parse_tokens(&sections["EXCLUDED_ATOMS_LIST"])?;
    if list.len() != nnb as usize {
        return Err(length_mismatch(
            "EXCLUDED_ATOMS_LIST",
            list.len(),
            format!("NNB={nnb}"),
        ));
    }
    if numex.len() != n_atoms {
        return Err(length_mismatch(
            "NUMBER_EXCLUDED_ATOMS",
            numex.len(),
            n_atoms,
        ));
    }
    let mut atomi = Vec::new();
    let mut atomj = Vec::new();
    let mut cursor = 0usize;
    for (i, &count) in numex.iter().enumerate() {
        if count < 0 {
            return Err(invalid_data("NUMBER_EXCLUDED_ATOMS entry is negative"));
        }
        let n = count as usize;
        if cursor + n > list.len() {
            return Err(length_mismatch(
                "EXCLUDED_ATOMS_LIST",
                list.len(),
                format!("NNB={nnb}"),
            ));
        }
        for &partner in &list[cursor..cursor + n] {
            if partner == 0 {
                continue;
            }
            if partner < 1 {
                return Err(invalid_data(format!(
                    "EXCLUDED_ATOMS_LIST partner {partner} is not a 1-based atom number"
                )));
            }
            let mut a = i as Idx;
            let mut b = (partner as Idx) - 1;
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if a != b {
                atomi.push(a);
                atomj.push(b);
            }
        }
        cursor += n;
    }
    if cursor != list.len() {
        return Err(invalid_data(format!(
            "NUMBER_EXCLUDED_ATOMS sums to {cursor}, expected NNB={nnb}"
        )));
    }
    let mut block = Block::new();
    insert_column(&mut block, keys::ATOMI, atomi)?;
    insert_column(&mut block, keys::ATOMJ, atomj)?;
    Ok(Some(block))
}

fn apply_box(frame: &mut Frame, sections: &HashMap<String, Vec<String>>, ifbox: i64) -> Result<()> {
    let Some(lines) = sections.get("BOX_DIMENSIONS") else {
        return Ok(());
    };
    let vals: Vec<F> = parse_tokens(lines)?;
    if vals.len() < 4 {
        return Err(length_mismatch("BOX_DIMENSIONS", vals.len(), 4usize));
    }
    frame.meta.insert("oldbeta", vals[0]);
    let lengths = [vals[1], vals[2], vals[3]];
    let origin = [0.0 as F, 0.0, 0.0];
    let simbox = match ifbox {
        1 => SimBox::new(Cell::Lengths(lengths), origin, [true; 3])
            .map_err(|e| invalid_data(format!("{e:?}")))?,
        2 => {
            let angle = (-1.0 as F / 3.0).acos().to_degrees();
            let cell = Cell::LengthsAngles {
                lengths,
                angles: [angle; 3],
            };
            SimBox::new(cell, origin, [true; 3]).map_err(|e| invalid_data(format!("{e}")))?
        }
        _ => return Ok(()),
    };
    frame.simbox = Some(simbox);
    Ok(())
}

fn apply_meta_scalars(frame: &mut Frame, sections: &HashMap<String, Vec<String>>) -> Result<()> {
    if let Some(lines) = sections.get("RADIUS_SET")
        && let Some(line) = lines.first()
    {
        frame.meta.insert("radius_set", line.clone());
    }
    if let Some(lines) = sections.get("SOLVENT_POINTERS") {
        let vals: Vec<i64> = parse_tokens(lines)?;
        if vals.len() != 3 {
            return Err(length_mismatch("SOLVENT_POINTERS", vals.len(), 3usize));
        }
        frame.meta.insert("solvent_iptres", vals[0]);
        frame.meta.insert("solvent_nspm", vals[1]);
        frame.meta.insert("solvent_nspsol", vals[2]);
    }
    Ok(())
}

fn section_floats(sections: &HashMap<String, Vec<String>>, key: &str) -> Result<Vec<F>> {
    match sections.get(key) {
        Some(lines) => parse_tokens(lines),
        None => Ok(Vec::new()),
    }
}

/// The `cmaps` block: the five atoms of each crossterm and its type name
/// (`prmtop_tables::cmap_terms`).
fn build_cmap_block(cmap: &prmtop_tables::CmapTerms) -> Result<Block> {
    let mut block = Block::new();
    for (p, key) in [
        keys::ATOMI,
        keys::ATOMJ,
        keys::ATOMK,
        keys::ATOML,
        keys::ATOMM,
    ]
    .into_iter()
    .enumerate()
    {
        let col: Vec<Idx> = cmap.atoms.iter().map(|five| five[p] as Idx).collect();
        insert_column(&mut block, key, col)?;
    }
    insert_column(&mut block, keys::TYPE, cmap.names.clone())?;
    Ok(block)
}

/// The structure [`Frame`] of the parsed `%FLAG` sections (the module docs).
pub(crate) fn frame_from_sections(sections: &HashMap<String, Vec<String>>) -> Result<Frame> {
    let pointers_lines = sections.get("POINTERS").ok_or_else(|| {
        invalid_data(
            "Invalid or empty prmtop file: POINTERS section missing. \
             This typically means the external tool (tleap) failed to create the file.",
        )
    })?;
    let meta_map = read_pointers(pointers_lines)?;
    let n_atoms = *meta_map
        .get("n_atoms")
        .ok_or_else(|| invalid_data("n_atoms missing after POINTERS"))? as usize;
    let n_res = *meta_map.get("NRES").unwrap_or(&0) as usize;
    let nnb = *meta_map.get("NNB").unwrap_or(&0);
    let ifbox = meta_map.get("IFBOX").copied().unwrap_or(0);
    refuse_unsupported(sections, &meta_map)?;
    check_dead_int_section(sections, "JOIN_ARRAY", n_atoms)?;
    check_dead_int_section(sections, "IROTAT", n_atoms)?;

    let names = sections
        .get("ATOM_NAME")
        .map(|l| prmtop_tables::parse_a4_names(l))
        .unwrap_or_default();
    if !names.is_empty() && names.len() != n_atoms {
        // Trailing empty pad fields from a short last line can inflate count;
        // trim to n_atoms when we have at least n_atoms non-pad entries.
        // Prefer exact n_atoms prefix if longer (pad empties at end).
        if names.len() < n_atoms {
            return Err(invalid_data(format!(
                "ATOM_NAME has {} entries, expected {n_atoms}",
                names.len()
            )));
        }
    }
    let mut names = names;
    if names.len() > n_atoms {
        names.truncate(n_atoms);
    }
    if names.len() < n_atoms {
        names.resize(n_atoms, String::new());
    }

    let raw_charges: Vec<F> = sections
        .get("CHARGE")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    if raw_charges.len() != n_atoms {
        return Err(invalid_data(format!(
            "CHARGE has {} entries, expected {n_atoms}",
            raw_charges.len()
        )));
    }
    // A chamber file scales by CHARMM's √332.0716, an AMBER one by 18.2223.
    let charge_factor = if prmtop_tables::is_chamber(sections) {
        crate::core::constants::CHARMM_COULOMB.sqrt()
    } else {
        AMBER_CHARGE_FACTOR
    };
    let charges: Vec<F> = raw_charges.into_iter().map(|q| q / charge_factor).collect();

    let masses: Vec<F> = sections
        .get("MASS")
        .map(|l| parse_tokens(l))
        .transpose()?
        .unwrap_or_default();
    if masses.len() != n_atoms {
        return Err(invalid_data(format!(
            "MASS has {} entries, expected {n_atoms}",
            masses.len()
        )));
    }

    let atom_types = prmtop_tables::atom_type_names(sections, n_atoms).map_err(invalid_data)?;

    let atomic_numbers_raw: Option<Vec<i64>> = sections
        .get("ATOMIC_NUMBER")
        .map(|l| parse_tokens(l))
        .transpose()?;
    // AMBER writes -1 for an unknown element; omit the column rather than
    // fake a species. A number above 118 is not a species at all: refuse it.
    let atomic_numbers: Option<Vec<Idx>> = match atomic_numbers_raw {
        Some(nums) if !nums.is_empty() && nums.iter().all(|&z| z >= 0) => {
            if nums.len() != n_atoms {
                return Err(invalid_data(format!(
                    "ATOMIC_NUMBER has {} entries, expected {n_atoms}",
                    nums.len()
                )));
            }
            if let Some((row, z)) = nums.iter().enumerate().find(|&(_, &z)| z > 118) {
                return Err(invalid_data(format!(
                    "ATOMIC_NUMBER entry {row}: {z} is not an atomic number (0..=118)"
                )));
            }
            Some(nums.into_iter().map(|z| z as Idx).collect())
        }
        _ => None,
    };

    let res_ids = residue_ids(sections.get("RESIDUE_POINTER"), n_atoms)?;
    let res_name = residue_names(sections, &res_ids, n_res)?;
    let mol_id = molecule_ids(sections, n_atoms)?;
    let tree = a4_exact(sections, "TREE_CHAIN_CLASSIFICATION", n_atoms)?;
    let gb_radius = floats_n(sections, "RADII", n_atoms)?;
    let gb_screen = floats_n(sections, "SCREEN", n_atoms)?;
    let exclusions = build_exclusions_block(sections, nnb, n_atoms)?;

    // Connectivity (inc + without H).
    let mut bond_ptrs = section_ints(sections, "BONDS_INC_HYDROGEN")?;
    bond_ptrs.extend(section_ints(sections, "BONDS_WITHOUT_HYDROGEN")?);
    let bonds = decode_bonds(&bond_ptrs, &atom_types)?;

    let mut angle_ptrs = section_ints(sections, "ANGLES_INC_HYDROGEN")?;
    angle_ptrs.extend(section_ints(sections, "ANGLES_WITHOUT_HYDROGEN")?);
    let angles = decode_angles(&angle_ptrs, &atom_types)?;

    let mut dihe_ptrs = section_ints(sections, "DIHEDRALS_INC_HYDROGEN")?;
    dihe_ptrs.extend(section_ints(sections, "DIHEDRALS_WITHOUT_HYDROGEN")?);
    let dih_k: Vec<F> = section_floats(sections, "DIHEDRAL_FORCE_CONSTANT")?;
    let dih_per: Vec<F> = section_floats(sections, "DIHEDRAL_PERIODICITY")?;
    let dih_phase: Vec<F> = section_floats(sections, "DIHEDRAL_PHASE")?;
    // Without the parameter tables (a structure-only head) the terms are
    // unknown, and each improper is one row.
    let tables = (!dih_k.is_empty() || !dih_per.is_empty() || !dih_phase.is_empty()).then_some(
        prmtop_tables::TorsionTables {
            k: &dih_k,
            periodicity: &dih_per,
            phase: &dih_phase,
        },
    );
    let mut dihedrals = decode_dihedrals(&dihe_ptrs, &atom_types, tables)?;
    // A chamber file's CHARMM impropers, in file order (centre first), named
    // as the force-field reader's `improper harmonic` types.
    for imp in prmtop_tables::chamber_impropers(sections, n_atoms).map_err(invalid_data)? {
        let types = imp.atoms.map(|a| atom_types[a].as_str());
        dihedrals.push(DihedralRow {
            atomi: imp.atoms[0] as Idx,
            atomj: imp.atoms[1] as Idx,
            atomk: imp.atoms[2] as Idx,
            atoml: imp.atoms[3] as Idx,
            type_name: TypeName::join(&types).map_err(invalid_data)?.to_string(),
            is_improper: true,
            exclude_14: true,
        });
    }
    let cmaps = prmtop_tables::cmap_terms(sections, &atom_types)
        .map_err(invalid_data)?
        .map(|c| build_cmap_block(&c))
        .transpose()?;

    // ---- atoms block ----
    let mut atoms = Block::new();
    let ids: Vec<Idx> = (1..=n_atoms as Idx).collect();
    insert_column(&mut atoms, "id", ids)?;
    insert_column(&mut atoms, "name", names)?;
    insert_column(&mut atoms, "type", atom_types)?;
    insert_column(&mut atoms, "charge", charges)?;
    insert_column(&mut atoms, "mass", masses)?;
    insert_column(&mut atoms, "res_id", res_ids)?;
    if let Some(names) = res_name {
        insert_column(&mut atoms, keys::RES_NAME, names)?;
    }
    if let Some(ids) = mol_id {
        insert_column(&mut atoms, keys::MOL_ID, ids)?;
    }
    if let Some(vals) = tree {
        insert_column(&mut atoms, "tree", vals)?;
    }
    if let Some(vals) = gb_radius {
        insert_column(&mut atoms, "gb_radius", vals)?;
    }
    if let Some(vals) = gb_screen {
        insert_column(&mut atoms, "gb_screen", vals)?;
    }
    if let Some(zs) = atomic_numbers {
        // The file spells species as Z: keep it raw, synthesize no symbols.
        insert_column(&mut atoms, keys::ATOMIC_NUMBER, zs)?;
    }

    let mut frame = Frame::new();
    // meta
    if let Some(title_lines) = sections.get("TITLE")
        && let Some(t) = title_lines.first()
    {
        frame.meta.insert("title", t.clone());
    }
    for (k, v) in meta_map {
        frame.meta.insert(k, v);
    }

    frame.insert("atoms", atoms);
    frame.insert("bonds", build_bond_block(&bonds)?);
    frame.insert("angles", build_angle_block(&angles)?);
    // One row per torsion, not per cosine term (meta `n_dihedrals` stays the
    // POINTERS row count NPHIH+MPHIA). Propers and impropers are separate
    // terms of separate styles, so each lives in its own block only.
    let (impropers, propers): (Vec<DihedralRow>, Vec<DihedralRow>) =
        dihedrals.into_iter().partition(|r| r.is_improper);
    frame.insert("dihedrals", build_dihedral_block(&propers)?);
    if !impropers.is_empty() {
        frame.insert("impropers", build_dihedral_block(&impropers)?);
    }
    if let Some(excl) = exclusions {
        frame.insert("exclusions", excl);
    }
    if let Some(cmaps) = cmaps {
        frame.insert(block_names::CMAPS, cmaps);
    }
    apply_meta_scalars(&mut frame, sections)?;
    apply_box(&mut frame, sections, ifbox)?;

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Read an AMBER prmtop structure file at `path` into a [`Frame`].
///
/// Parses topology/connectivity into a [`Frame`]. Force-field parameter tables
/// (harmonic constants, LJ coefficients, Fourier terms) are **not** assembled
/// here — that is the force-field reader's product
/// ([`AmberPrmtopForcefieldReader`](crate::io::amber::AmberPrmtopForcefieldReader)), which names its types exactly as the
/// rows below are labelled (the shared `prmtop_tables` helpers decide both).
/// What the frame carries of the tables is per-row: which rows a multi-term
/// improper is. A 1-4 pair's own weight (`SCEE` / `SCNB`) is force-field
/// meaning: `AmberPrmtopForcefieldReader::read_system`
/// returns this frame with those `pairs` rows.
///
/// # Output Frame
///
/// - `"atoms"`: `id` (uint, 1-based), `name` (str), `type` (str,
///   `prmtop_tables::atom_type_names`: the `AMBER_ATOM_TYPE`, or
///   `<type>~<class>` when one type name stands for two LJ classes or masses —
///   a chamber file cuts CHARMM's types to four characters), `charge` (float,
///   electron units — prmtop value / 18.2223, or / √332.0716 in a chamber
///   file), `mass` (float), optional `atomic_number` (uint, the raw
///   `ATOMIC_NUMBER` section; absent when the section is absent or holds a
///   `-1` "unknown" entry — no `element` column is synthesized, call
///   `Element::fill_symbols` for one), `res_id` (uint, 0-based from `RESIDUE_POINTER`), optional
///   `res_name` (from `RESIDUE_LABEL`), optional `mol_id` (1-based, from
///   `ATOMS_PER_MOLECULE`; never inferred from bonds). Format-local
///   unregistered columns — a debt, not a licence, with no cross-format
///   consumer: `tree` (`TREE_CHAIN_CLASSIFICATION`), `gb_radius` (Å, `RADII`),
///   `gb_screen` (`SCREEN`). Each is absent when its section is absent.
/// - `"bonds"` / `"angles"`: connectivity (`atomi`/… 0-based uint), `type`
///   (the force-field reader's type name: the end atom types in sorted order,
///   an angle's vertex in the middle), `type_id` (uint, prmtop index), `id`
///   (uint, 1-based row id).
/// - `"dihedrals"` (propers) / `"impropers"` (negative 4th pointer):
///   connectivity, `type` (the force-field reader's type name), `id` and
///   `exclude_14` (bool: every merged row had a negative 3rd pointer). A
///   proper is one row per torsion, not per cosine term: the prmtop rows of a
///   multi-term torsion share one atom quartet and become one row, and the
///   prmtop parameter index (`type_id`) is dropped because such a torsion has
///   none. An improper keeps its prmtop atom order (centre third); a
///   multi-term improper (several rows on its quartet, or a negative-PN chain)
///   is one row per term, typed `<quartet>@<n>`, since `improper periodic`
///   holds one term. A chamber file's CHARMM impropers (`CHARMM_IMPROPERS`)
///   follow, in file order (CHARMM's centre first), `exclude_14` set. Empty
///   systems still get schema-typed empty blocks; `"impropers"` is absent when
///   there are none.
/// - `"cmaps"`: `atomi` … `atomm`, `type` — the CMAP crossterms
///   (`CHARMM_CMAP_INDEX`, or ff19SB's `CMAP_INDEX`), typed as
///   `prmtop_tables::cmap_terms` names them; absent without CMAP.
/// - `"exclusions"`: `atomi`/`atomj` (uint, 0-based, `atomi < atomj`), the
///   Ewald real-space correction set from `NUMBER_EXCLUDED_ATOMS` /
///   `EXCLUDED_ATOMS_LIST`. `0` placeholders are dropped. An all-placeholder
///   list yields a schema-typed empty block.
/// - `frame.simbox`: from `BOX_DIMENSIONS` when `IFBOX = 1` (ortho) or
///   `IFBOX = 2` (truncated octahedron at `arccos(-1/3)`). An inpcrd box
///   takes precedence over a prmtop box: the prmtop cell is the topology-time
///   default and the inpcrd cell is the current state. The two readers stay
///   independent — callers who read both files apply that rule.
/// - `frame.meta`: POINTERS raw fields (`NATOM`, …) plus derived
///   `n_atoms` / `n_bonds` / `n_angles` / `n_dihedrals` / `n_atomtypes` /
///   `n_bondtypes` / `n_angletypes` / `n_dihedraltypes`, optional `title`,
///   and optional `radius_set` / `oldbeta` / `solvent_iptres` /
///   `solvent_nspm` / `solvent_nspsol`.
///
/// Refused by name: perturbed (`IFPERT > 0`), solvent-cap (`IFCAP > 0`),
/// `IFBOX = 3`, polarizable (`IPOL > 0`), 12-6-4 (`LENNARD_JONES_CCOEF`)
/// and 10-12 (negative ICO) files. A 1-4 row on a negative-PN chain or on a
/// bonded / angle-end pair (sander prices both, the IR has no form) is the
/// force-field reader's refusal, not this one's.
///
/// # Encoding notes (Amber [FileFormats](https://ambermd.org/FileFormats.php))
///
/// - Bonded atom pointers are coordinate-array indexes: true 1-based atom number
///   is `|N|/3 + 1` (0-based index `|N|/3`). The chamber sections
///   (`CHARMM_UREY_BRADLEY`, `CHARMM_IMPROPERS`, `*CMAP_INDEX`) hold 1-based
///   atom numbers instead.
/// - Dihedral: 3rd pointer negative → ignore end-group (1-4) interactions;
///   4th pointer negative → improper torsion (whose 1-4 pair sander never
///   prices, whatever its 3rd pointer). Atom index uses absolute value.
///   Pointer `0` denotes atom 1 — the sign, not the value, carries the flags.
/// - `ATOM_NAME` / `AMBER_ATOM_TYPE` / residue labels are Fortran `20a4`
///   (exactly 4-char fields; may not be whitespace-delimited).
/// - `%COMMENT` lines are optional and skipped; section order is not required
///   to be fixed (we index by `%FLAG` name).
/// - Charges are Amber internal units (`E = q1*q2/r` with kcal/mol, Å);
///   we divide by the literal 18.2223 (a chamber file: √332.0716, ParmEd's
///   `CHARMM_ELECTROSTATIC`) to electron charge for the Frame.
pub fn read_prmtop<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let frame = read_frame_from(src.into().reader()?)?;
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Parse AMBER prmtop structure text from any [`BufRead`].
fn read_frame_from<R: BufRead>(reader: R) -> Result<Frame> {
    let sections = parse_flag_sections(reader)?;
    frame_from_sections(&sections)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod fixed_width_tests {
    use super::*;

    #[test]
    fn adjacent_eight_column_integers_are_split() {
        let line = format!("{:>8}{:>8}{:>8}", -1000011, -1000014, 3);
        assert_eq!(line, "-1000011-1000014       3");
        assert_eq!(split_integer_fields(&line, 8), "-1000011 -1000014 3");
    }

    #[test]
    fn free_format_integer_lines_keep_their_tokens() {
        // Not aligned to 8 columns: the fixed-width cut is not all integers.
        assert_eq!(split_integer_fields("12 345 6789 1", 8), "12 345 6789 1");
    }

    #[test]
    fn integer_sections_parse_by_width_and_reals_by_blanks() {
        let text = format!(
            "%FLAG DIHEDRALS_INC_HYDROGEN\n%FORMAT(10I8)\n{:>8}{:>8}{:>8}{:>8}{:>8}\n\
             %FLAG CHARGE\n%FORMAT(5E16.8)\n  1.00000000E+00 -2.000000000000001E+00\n",
            3000030, -3000033, 3000036, -3000039, 7
        );
        let sections = parse_flag_sections(text.as_bytes()).unwrap();
        let dih: Vec<i64> = parse_tokens(&sections["DIHEDRALS_INC_HYDROGEN"]).unwrap();
        assert_eq!(dih, vec![3000030, -3000033, 3000036, -3000039, 7]);
        let q: Vec<f64> = parse_tokens(&sections["CHARGE"]).unwrap();
        assert_eq!(q, vec![1.0, -2.000000000000001]);
        assert_eq!(integer_field_width("%FORMAT(10I8)"), Some(8));
        assert_eq!(integer_field_width("%FORMAT(5E16.8)"), None);
        assert_eq!(integer_field_width("%FORMAT(20a4)"), None);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Cursor;

    /// A 16-atom LiTFSI prmtop head (TFSI anion + Li+), hand-written against
    /// the Amber PARM spec <https://ambermd.org/FileFormats.php> — no
    /// AmberTools is involved at test time.
    ///
    /// Self-consistency the goldens lean on, all derived from the text itself:
    /// `NATOM = 16`, `NRES = 2`, `NNB = 65`, `IFBOX = 0`. The bond graph is a
    /// chain `0-1(-2,-3)-4(-5,-6)-7-8(-9,-10)-11(-12,-13,-14)` with atom 15
    /// (Li+) isolated, so the 1-2/1-3/1-4 exclusion counts are
    /// `7 7 5 4 7 3 2 7 6 5 4 3 2 1 1 1`, which sum to exactly 65 with the two
    /// trailing `0` placeholders (atoms 14 and 15 exclude nothing) — hence 63
    /// exclusion rows. `DIHEDRALS_INC_HYDROGEN` is empty, so the 27 torsion
    /// rows keep `WITHOUT_HYDROGEN` order. Rows 11–16 are three two-term
    /// torsions (`12 21 24 {27,30,33} 3` then `12 21 -24 {27,30,33} 4`), so
    /// the 27 rows are 24 torsions.
    const LITFSI_HEAD: &str = "\
%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
TFSI
%FLAG POINTERS
%FORMAT(10I8)
      16       6       0      14       0      25       0      27       0       0
      65       2      14      25      27       7      12       4       7       0
       0       0       0       0       0       0       0       0      15       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
F   C   F1  F2  S   O   O3  N   S1  O1  O2  C1  F4  F5  F3  LI
%FLAG CHARGE
%FORMAT(5E16.8)
 -4.94977802E+00  1.01079098E+01 -4.94977802E+00 -4.94977802E+00  2.70364265E+01
 -1.11739144E+01 -1.19684066E+01 -1.92992379E+01  3.06571975E+01 -1.19684066E+01
 -1.19684066E+01  1.01079098E+01 -4.94977802E+00 -4.94977802E+00 -4.94977802E+00
  1.82223000E+01
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       9       6       9       9      16       8       8       7      16       8
       8       6       9       9       9       3
%FLAG MASS
%FORMAT(5E16.8)
  1.90000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  3.20600000E+01
  1.60000000E+01  1.60000000E+01  1.40100000E+01  3.20600000E+01  1.60000000E+01
  1.60000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  1.90000000E+01
  6.94000000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
f   c3  f   f   s6  o   o   ne  sy  o   o   c3  f   f   f   Li+
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       7       7       5       4       7       3       2       7       6       5
       4       3       2       1       1       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       5       6       7       8       3       4       5
       6       7       8       9       4       5       6       7       8       5
       6       7       8       6       7       8       9      10      11      12
       7       8       9       8       9       9      10      11      12      13
      14      15      10      11      12      13      14      15      11      12
      13      14      15      12      13      14      15      13      14      15
      14      15      15       0       0
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
TF  LI
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1      16
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      33      36       1      33      39       1      33      42       1      24
      27       2      24      30       2      24      33       3      21      24
       4      12      15       5      12      18       5      12      21       6
       3       6       1       3       9       1       3      12       7       0
       3       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
      39      33      42       1      36      33      39       1      36      33
      42       1      30      24      33       2      27      24      30       3
      27      24      33       2      24      33      36       4      24      33
      39       4      24      33      42       4      21      24      27       5
      21      24      30       5      21      24      33       6      18      12
      21       7      15      12      18       8      15      12      21       7
      12      21      24       9       9       3      12      10       6       3
       9       1       6       3      12      10       3      12      15      11
       3      12      18      11       3      12      21      12       0       3
       6       1       0       3       9       1       0       3      12      10
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      30      24      33      36       1      30      24      33      39       1
      30      24      33      42       1      27      24      33      36       1
      27      24      33      39       1      27      24      33      42       1
      21      24      33      36       1      21      24      33      39       1
      21      24      33      42       1      18      12      21      24       2
      15      12      21      24       2      12      21      24      27       3
      12      21     -24      27       4      12      21      24      30       3
      12      21     -24      30       4      12      21      24      33       3
      12      21     -24      33       4       9       3      12      15       1
       9       3      12      18       1       9       3      12      21       1
       6       3      12      15       1       6       3      12      18       1
       6       3      12      21       1       3      12      21      24       2
       0       3      12      15       1       0       3      12      18       1
       0       3      12      21       1
%FLAG TREE_CHAIN_CLASSIFICATION
%FORMAT(20a4)
E   M   E   E   M   E   E   M   M   E   E   M   E   E   E   BLA
%FLAG JOIN_ARRAY
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG IROTAT
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG SOLVENT_POINTERS
%FORMAT(3I8)
       2       2       3
%FLAG ATOMS_PER_MOLECULE
%FORMAT(10I8)
      15       1
%FLAG RADIUS_SET
%FORMAT(1a80)
modified Bondi radii (mbondi2)
%FLAG RADII
%FORMAT(5E16.8)
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.80000000E+00
  1.50000000E+00  1.50000000E+00  1.55000000E+00  1.80000000E+00  1.50000000E+00
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.50000000E+00
  1.50000000E+00
%FLAG SCREEN
%FORMAT(5E16.8)
  8.80000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  9.60000000E-01
  8.50000000E-01  8.50000000E-01  7.90000000E-01  9.60000000E-01  8.50000000E-01
  8.50000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  8.80000000E-01
  8.00000000E-01
";

    fn frame_from(s: &str) -> Frame {
        read_frame_from(Cursor::new(s.as_bytes())).expect("parse")
    }

    #[test]
    fn a4_names_chunking() {
        let names =
            prmtop_tables::parse_a4_names(&["F   C   F1  F2  S   O   O3  N   ".to_string()]);
        assert_eq!(names, vec!["F", "C", "F1", "F2", "S", "O", "O3", "N"]);
    }

    #[test]
    fn litfsi_counts() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(frame.meta.get("n_atoms").and_then(|v| v.as_i64()), Some(16));
        assert_eq!(frame.meta.get("n_bonds").and_then(|v| v.as_i64()), Some(14));
        assert_eq!(
            frame.meta.get("n_angles").and_then(|v| v.as_i64()),
            Some(25)
        );
        assert_eq!(
            frame.meta.get("n_dihedrals").and_then(|v| v.as_i64()),
            Some(27)
        );
        assert_eq!(
            frame.meta.get("title").and_then(|v| v.as_str()),
            Some("TFSI")
        );
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(16));
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.n_rows(), Some(14));
        let angles = frame.get("angles").unwrap();
        assert_eq!(angles.n_rows(), Some(25));
        // 27 prmtop rows (meta, from POINTERS) are 24 torsions (rows).
        let dihedrals = frame.get("dihedrals").unwrap();
        assert_eq!(dihedrals.n_rows(), Some(24));
    }

    /// An angle's `type` is the force-field reader's type name: its end atom
    /// types in sorted order around the vertex, so a frame label finds its
    /// type by exact name. The row's atom order is untouched. Rows and types
    /// read by hand from the fixture: row 3 is atoms 10-8-11 (`o`, `sy`,
    /// `c3`), row 6 is 8-11-12 (`sy`, `c3`, `f`), row 11 is 7-8-11 (`ne`,
    /// `sy`, `c3`), row 12 is 6-4-7 (`o`, `s6`, `ne`).
    #[test]
    fn an_angle_label_names_its_end_types_in_sorted_order() {
        let frame = frame_from(LITFSI_HEAD);
        let angles = frame.get("angles").unwrap();
        let labels = angles.get("type").and_then(|c| c.get::<String>()).unwrap();
        let (i, j, k) = (
            angles.get("atomi").and_then(|c| c.get::<u64>()).unwrap(),
            angles.get("atomj").and_then(|c| c.get::<u64>()).unwrap(),
            angles.get("atomk").and_then(|c| c.get::<u64>()).unwrap(),
        );
        assert_eq!((i[[3]], j[[3]], k[[3]]), (10, 8, 11));
        assert_eq!(labels[[3]], "c3-sy-o");
        assert_eq!(labels[[6]], "f-c3-sy");
        assert_eq!(labels[[11]], "c3-sy-ne");
        assert_eq!(labels[[12]], "ne-s6-o");
        assert_eq!(labels[[0]], "f-c3-f");
    }

    #[test]
    fn litfsi_charge_and_li() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let charge = atoms.get("charge").and_then(|c| c.get::<f64>()).unwrap();
        assert!((charge[[15]] - 1.0).abs() < 1e-5);
        let total: F = (0..16).map(|i| charge[[i]]).sum();
        assert!(total.abs() < 0.01);
        let z = atoms
            .get("atomic_number")
            .and_then(|c| c.get::<u64>())
            .unwrap();
        assert_eq!(z[[15]], 3);
        assert_eq!(z[[0]], 9);
        let names = atoms.get("name").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(names[[0]], "F");
        assert_eq!(names[[15]], "LI");
        let types = atoms.get("type").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(types[[0]], "f");
        assert_eq!(types[[1]], "c3");
        assert_eq!(types[[15]], "Li+");
        let res = atoms.get("res_id").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(res[[0]], 0);
        assert_eq!(res[[14]], 0);
        assert_eq!(res[[15]], 1);
    }

    #[test]
    fn bond_first_pair_zero_based() {
        let frame = frame_from(LITFSI_HEAD);
        let bonds = frame.get("bonds").unwrap();
        let ai = bonds.get("atomi").and_then(|c| c.get::<u64>()).unwrap();
        let aj = bonds.get("atomj").and_then(|c| c.get::<u64>()).unwrap();
        let mut found = false;
        for i in 0..ai.len() {
            let a = ai[[i]];
            let b = aj[[i]];
            if (a == 11 && b == 12) || (a == 12 && b == 11) {
                found = true;
                break;
            }
        }
        assert!(found, "expected bond (11,12)");
    }

    #[test]
    fn missing_pointers() {
        let text = "%FLAG TITLE\n%FORMAT(20a4)\ntest\n";
        let err = read_frame_from(Cursor::new(text.as_bytes())).unwrap_err();
        assert!(err.to_string().contains("POINTERS section missing"));
    }

    #[test]
    fn bond_index_encoding_unit() {
        // raw 33,36 → 11,12 0-based; type 1
        let types: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
        let rows = decode_bonds(&[33, 36, 1], &types).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].atomi, 11);
        assert_eq!(rows[0].atomj, 12);
        assert_eq!(rows[0].type_id, 1);
    }

    #[test]
    fn dihedral_negative_k() {
        let types: Vec<String> = (0..16).map(|i| format!("t{i}")).collect();
        // (12, 21, -24, 27, 1): k=abs(-24)/3=8, l=9; j=7
        let rows = decode_dihedrals(&[12, 21, -24, 27, 1], &types, None).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].atomk, 8);
        assert_eq!(rows[0].atoml, 9);
    }
    // -----------------------------------------------------------------
    // Fixtures for amber-prmtop-complete-01-structure
    //
    // Goldens are hand-derived from the Amber PARM/prmtop specification
    // (<https://ambermd.org/FileFormats.php>, accessed 2026-09-04) and from
    // the fixture text itself. No AmberTools is involved at test time.
    // -----------------------------------------------------------------

    /// POINTERS line 3 of [`LITFSI_HEAD`] — fields 21..30
    /// (IFPERT, NBPER, NGPER, NDPER, MBPER, MGPER, MDPER, IFBOX, NMXRS, IFCAP).
    const LITFSI_POINTERS_LINE3: &str =
        "       0       0       0       0       0       0       0       0      15       0";

    /// `LITFSI_HEAD` with IFPERT / IFBOX / IFCAP rewritten in POINTERS.
    fn litfsi_with_pointers3(ifpert: i64, ifbox: i64, ifcap: i64) -> String {
        assert_eq!(
            LITFSI_HEAD.matches(LITFSI_POINTERS_LINE3).count(),
            1,
            "POINTERS line 3 must occur exactly once in LITFSI_HEAD"
        );
        let line = format!(
            "{ifpert:>8}{z:>8}{z:>8}{z:>8}{z:>8}{z:>8}{z:>8}{ifbox:>8}{nmxrs:>8}{ifcap:>8}",
            z = 0,
            nmxrs = 15
        );
        LITFSI_HEAD.replace(LITFSI_POINTERS_LINE3, &line)
    }

    /// Drop one `%FLAG <name>` section (flag line, `%FORMAT` line and data).
    ///
    /// The "absent when the section is absent" half of every column assertion.
    fn without_section(text: &str, flag: &str) -> String {
        assert!(
            text.contains(&format!("%FLAG {flag}\n")),
            "fixture does not carry %FLAG {flag}"
        );
        let mut out = String::with_capacity(text.len());
        let mut skipping = false;
        for line in text.lines() {
            if line.starts_with("%FLAG") {
                skipping = line.split_whitespace().nth(1) == Some(flag);
            }
            if !skipping {
                out.push_str(line);
                out.push('\n');
            }
        }
        out
    }

    /// `(OLDBETA, bx, by, bz)` for a 30 Å cube — IFBOX=1.
    const BOX_ORTHO: &str = "\
%FLAG BOX_DIMENSIONS
%FORMAT(5E16.8)
  9.00000000E+01  3.00000000E+01  3.00000000E+01  3.00000000E+01
";

    /// `(OLDBETA, bx, by, bz)` for a 30 Å truncated octahedron — IFBOX=2.
    ///
    /// Amber writes OLDBETA truncated to `1.09471219E+02`; the cell is built
    /// from `arccos(-1/3) = 109.4712206344907`, never from this number.
    const BOX_OCTAHEDRON: &str = "\
%FLAG BOX_DIMENSIONS
%FORMAT(5E16.8)
  1.09471219E+02  3.00000000E+01  3.00000000E+01  3.00000000E+01
";

    /// Minimal 4-atom prmtop: a star (atom 1 bonded to 0, 2 and 3) whose
    /// torsion table carries one proper row and two Amber impropers, one of
    /// which also has a negative **3rd** pointer (`exclude_14`).
    ///
    /// `nnb` / `numex` / `excluded` are parameters so the exclusion-block edge
    /// cases reuse one legal topology.
    fn star4_prmtop(nnb: i64, numex: &str, excluded: &str) -> String {
        format!(
            "\
%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
STAR
%FLAG POINTERS
%FORMAT(10I8)
       4       2       0       3       0       3       0       3       0       0
{nnb:>8}       1       3       3       3       1       1       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  C4
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       6       6       6       6
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.20100000E+01
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
{numex}
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
{excluded}
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
MOL
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  c3
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1       3       9       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       0       3       9       1       6       3
       9       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3      -6       9       1       0       3      -6      -9       2
       3       0       6      -9       2
"
        )
    }

    /// Exclusion counts of the 4-atom star: 3, 2, 1 and one `0` placeholder.
    const STAR4_NUMEX: &str = "       3       2       1       1";
    /// The matching `EXCLUDED_ATOMS_LIST` — NNB = 7, one `0` placeholder.
    const STAR4_EXCLUDED: &str = "       2       3       4       3       4       4       0";

    /// The 4-atom star with its true exclusion list.
    fn star4() -> String {
        star4_prmtop(7, STAR4_NUMEX, STAR4_EXCLUDED)
    }

    fn refusal_from(text: &str) -> Error {
        read_frame_from(Cursor::new(text.as_bytes()))
            .expect_err("expected the reader to refuse this file")
    }

    /// Every unsupported-feature refusal: `InvalidData`, names the subject and
    /// says `"are not supported"` (a refusal a user cannot act on is not one).
    fn assert_refusal(text: &str, subject: &str) {
        let err = refusal_from(text);
        assert_eq!(
            err.kind(),
            std::io::ErrorKind::InvalidData,
            "refusal for {subject:?} must be InvalidData"
        );
        let msg = err.to_string();
        assert!(
            msg.contains(subject),
            "refusal message {msg:?} does not name {subject:?}"
        );
        assert!(
            msg.contains("are not supported"),
            "refusal message {msg:?} does not say \"are not supported\""
        );
    }

    // -----------------------------------------------------------------
    // Per-atom columns (ac-002)
    // -----------------------------------------------------------------

    #[test]
    fn res_name_follows_residue_label() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let res_name = atoms
            .get("res_name")
            .and_then(|c| c.get::<String>())
            .expect("atoms block must carry res_name from RESIDUE_LABEL");
        assert_eq!(res_name.len(), 16);
        for i in 0..15 {
            assert_eq!(res_name[[i]], "TF", "atom {i} is in residue TF");
        }
        assert_eq!(res_name[[15]], "LI");
    }

    #[test]
    fn res_name_absent_without_residue_label() {
        let text = without_section(LITFSI_HEAD, "RESIDUE_LABEL");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms
                .get("res_name")
                .and_then(|c| c.get::<String>())
                .is_none(),
            "res_name must be absent, never fabricated, when RESIDUE_LABEL is missing"
        );
    }

    #[test]
    fn mol_id_from_atoms_per_molecule() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let mol_id = atoms
            .get("mol_id")
            .and_then(|c| c.get::<u64>())
            .expect("atoms block must carry mol_id from ATOMS_PER_MOLECULE");
        assert_eq!(mol_id.len(), 16);
        for i in 0..15 {
            assert_eq!(mol_id[[i]], 1, "atom {i} is in the TFSI molecule");
        }
        assert_eq!(mol_id[[15]], 2);
    }

    #[test]
    fn mol_id_absent_without_atoms_per_molecule() {
        // Amber writes ATOMS_PER_MOLECULE and SOLVENT_POINTERS only for
        // IFBOX > 0, so "no molecule partition" means both are gone. The
        // partition is never re-derived from bond connectivity.
        let text = without_section(LITFSI_HEAD, "ATOMS_PER_MOLECULE");
        let text = without_section(&text, "SOLVENT_POINTERS");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms.get("mol_id").and_then(|c| c.get::<u64>()).is_none(),
            "mol_id must be absent, never inferred from bonds"
        );
    }

    #[test]
    fn tree_from_tree_chain_classification() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let tree = atoms
            .get("tree")
            .and_then(|c| c.get::<String>())
            .expect("atoms block must carry tree from TREE_CHAIN_CLASSIFICATION");
        assert_eq!(tree.len(), 16);
        assert_eq!(tree[[0]], "E");
        assert_eq!(tree[[1]], "M");
        assert_eq!(tree[[7]], "M");
        assert_eq!(tree[[15]], "BLA");
    }

    #[test]
    fn tree_absent_without_tree_chain_classification() {
        let text = without_section(LITFSI_HEAD, "TREE_CHAIN_CLASSIFICATION");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(atoms.get("tree").and_then(|c| c.get::<String>()).is_none());
    }

    #[test]
    fn gb_radius_from_radii() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let r = atoms
            .get("gb_radius")
            .and_then(|c| c.get::<f64>())
            .expect("atoms block must carry gb_radius from RADII");
        assert_eq!(r.len(), 16);
        assert!((r[[0]] - 1.50).abs() < 1e-12, "F radius");
        assert!((r[[1]] - 1.70).abs() < 1e-12, "C radius");
        assert!((r[[7]] - 1.55).abs() < 1e-12, "N radius");
        assert!((r[[15]] - 1.50).abs() < 1e-12, "Li radius");
    }

    #[test]
    fn gb_radius_absent_without_radii() {
        let text = without_section(LITFSI_HEAD, "RADII");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms
                .get("gb_radius")
                .and_then(|c| c.get::<f64>())
                .is_none()
        );
    }

    #[test]
    fn gb_screen_from_screen() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        let s = atoms
            .get("gb_screen")
            .and_then(|c| c.get::<f64>())
            .expect("atoms block must carry gb_screen from SCREEN");
        assert_eq!(s.len(), 16);
        assert!((s[[0]] - 0.88).abs() < 1e-12, "F screen");
        assert!((s[[1]] - 0.72).abs() < 1e-12, "C screen");
        assert!((s[[4]] - 0.96).abs() < 1e-12, "S screen");
        assert!((s[[15]] - 0.80).abs() < 1e-12, "Li screen");
    }

    #[test]
    fn gb_screen_absent_without_screen() {
        let text = without_section(LITFSI_HEAD, "SCREEN");
        let frame = frame_from(&text);
        let atoms = frame.get("atoms").unwrap();
        assert!(
            atoms
                .get("gb_screen")
                .and_then(|c| c.get::<f64>())
                .is_none()
        );
    }

    // -----------------------------------------------------------------
    // exclusions block (ac-003)
    // -----------------------------------------------------------------

    #[test]
    fn exclusions_rows_are_zero_based_ordered_pairs() {
        let frame = frame_from(LITFSI_HEAD);
        let excl = frame
            .get("exclusions")
            .expect("prmtop must produce the exclusions block PME already reads");
        // NNB = 65 entries, two of which are the `0` placeholders of the two
        // atoms that exclude nothing (F3, index 14; Li, index 15).
        assert_eq!(excl.n_rows(), Some(63));
        let atomi = excl
            .get("atomi")
            .and_then(|c| c.get::<u64>())
            .expect("exclusions.atomi");
        let atomj = excl
            .get("atomj")
            .and_then(|c| c.get::<u64>())
            .expect("exclusions.atomj");
        for row in 0..63 {
            assert!(
                atomi[[row]] < atomj[[row]],
                "row {row}: {} !< {}",
                atomi[[row]],
                atomj[[row]]
            );
            assert!(
                atomj[[row]] <= 14,
                "row {row} references a non-existent partner"
            );
        }
        // File numbers are 1-based: atom 1 excludes 2..8 -> (0,1)..(0,7).
        assert_eq!((atomi[[0]], atomj[[0]]), (0, 1));
        assert_eq!((atomi[[6]], atomj[[6]]), (0, 7));
        assert_eq!((atomi[[7]], atomj[[7]]), (1, 2));
        assert_eq!((atomi[[62]], atomj[[62]]), (13, 14));
    }

    #[test]
    fn exclusions_all_zero_list_is_an_empty_block() {
        let text = star4_prmtop(
            4,
            "       1       1       1       1",
            "       0       0       0       0",
        );
        let frame = frame_from(&text);
        let excl = frame
            .get("exclusions")
            .expect("an all-placeholder list still yields a schema-typed empty block");
        assert_eq!(excl.n_rows(), Some(0));
        assert!(
            excl.get("atomi").and_then(|c| c.get::<u64>()).is_some(),
            "atomi column must exist"
        );
        assert!(
            excl.get("atomj").and_then(|c| c.get::<u64>()).is_some(),
            "atomj column must exist"
        );
    }

    #[test]
    fn exclusions_length_mismatch_names_nnb() {
        // NNB = 7 in POINTERS, but the flattened list carries only 6 entries.
        let text = star4_prmtop(
            7,
            STAR4_NUMEX,
            "       2       3       4       3       4       4",
        );
        let err = refusal_from(&text);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(msg.contains("NNB"), "message {msg:?} must name NNB");
        assert!(
            msg.contains('7') && msg.contains('6'),
            "message {msg:?} must name both counts"
        );
    }

    // -----------------------------------------------------------------
    // exclude_14 (ac-004)
    // -----------------------------------------------------------------

    #[test]
    fn dihedral_exclude_14_marks_the_negative_third_pointer() {
        let frame = frame_from(&chain5_prmtop(
            "       0       3      -6       9       1       3       6       9      12       1",
        ));
        let flag = frame
            .get("dihedrals")
            .unwrap()
            .get("exclude_14")
            .and_then(|c| c.get::<bool>())
            .expect("dihedrals must carry exclude_14");
        assert_eq!((flag[[0]], flag[[1]]), (true, false));
    }

    #[test]
    fn improper_exclude_14_is_its_own_rows_flag() {
        // star4: `0 3 -6 9 1` is the one proper; `0 3 -6 -9 2` and
        // `3 0 6 -9 2` are impropers, only the first with a negative 3rd pointer.
        let frame = frame_from(&star4());
        assert_eq!(frame.get("dihedrals").unwrap().n_rows(), Some(1));
        let impropers = frame
            .get("impropers")
            .expect("a negative 4th pointer puts the row in impropers");
        assert_eq!(quartets(impropers), vec![[0, 1, 2, 3], [1, 0, 2, 3]]);
        let flag = impropers
            .get("exclude_14")
            .and_then(|c| c.get::<bool>())
            .expect("impropers must carry exclude_14");
        assert_eq!((flag[[0]], flag[[1]]), (true, false));
    }

    #[test]
    fn zero_third_pointer_is_atom_zero_and_not_excluded() {
        // Pointer 0 denotes atom 1 (index 0): the *sign*, not the value,
        // carries the 1-4 suppression flag.
        let types: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
        let rows = decode_dihedrals(&[3, 0, 0, 9, 1], &types, None).unwrap();
        let block = build_dihedral_block(&rows).unwrap();
        assert_eq!(
            block.get("atomk").and_then(|c| c.get::<u64>()).unwrap()[[0]],
            0
        );
        let flag = block
            .get("exclude_14")
            .and_then(|c| c.get::<bool>())
            .expect("build_dihedral_block must emit exclude_14");
        assert!(!flag[[0]], "a literal 0 pointer is not a negative pointer");
    }

    // -----------------------------------------------------------------
    // One dihedral per quartet (ff-endpoints-01)
    // -----------------------------------------------------------------

    /// A 5-atom `H1–C1–O1–C2–H2` chain (types `hc c3 os c3 hc`), one molecule,
    /// whose `DIHEDRALS_INC_HYDROGEN` table is `torsions` (raw prmtop pointers,
    /// 5 per row). Its two proper quartets are `0-1-2-3` and `1-2-3-4`.
    ///
    /// Both quartets carry the type name `hc-c3-os-c3`: the force-field reader
    /// orients a proper so that its second type is not after its third
    /// (`c3 ≤ os`), so `1-2-3-4` (`c3-os-c3-hc`) reads as `4-3-2-1`.
    pub(crate) fn chain5_prmtop(torsions: &str) -> String {
        let nphih = torsions.split_whitespace().count() / 5;
        format!(
            "\
%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
CHAIN
%FLAG POINTERS
%FORMAT(10I8)
       5       2       2       2       2       1{nphih:>8}       0       0       0
       0       1       2       1       0       2       2       2       2       0
       0       0       0       0       0       0       0       0       5       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
H1  C1  O1  C2  H2
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.00800000E+00  1.20100000E+01  1.60000000E+01  1.20100000E+01  1.00800000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
hc  c3  os  c3  hc
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
MOL
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG ATOMS_PER_MOLECULE
%FORMAT(10I8)
       5
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       0       3       1       9      12       1
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       3       6       2       6       9       2
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       6       9      12       1
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       3       6       9       2
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
{torsions}
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
"
        )
    }

    /// The `(atomi, atomj, atomk, atoml)` rows of `block`.
    fn quartets(block: &Block) -> Vec<[Idx; 4]> {
        let col = |k: &str| {
            block
                .get(k)
                .and_then(|c| c.get::<u64>())
                .unwrap_or_else(|| panic!("{k} column"))
        };
        let (i, j, k, l) = (col("atomi"), col("atomj"), col("atomk"), col("atoml"));
        (0..block.n_rows().unwrap_or(0))
            .map(|r| [i[[r]], j[[r]], k[[r]], l[[r]]])
            .collect()
    }

    #[test]
    fn a_multi_term_torsion_is_one_dihedral_named_by_its_quartet() {
        // Quartet 0-1-2-3 carries two cosine terms (types 1 and 2); AMBER
        // flags the second row's 3rd pointer so its 1-4 pair is counted once.
        let frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       0       3      -6       9       2",
        ));
        let dihedrals = frame.get("dihedrals").unwrap();
        // `hc-c3-os-c3` reads reversed (`c3-os-c3-hc` is the smaller spelling),
        // so the row is stored the other way round, atoms and label together.
        assert_eq!(quartets(dihedrals), vec![[3, 2, 1, 0]]);
        assert_eq!(
            dihedrals
                .get("type")
                .and_then(|c| c.get::<String>())
                .unwrap()[[0]],
            "c3-os-c3-hc",
            "the label is the quartet's type name"
        );
        assert!(
            !dihedrals
                .get("exclude_14")
                .and_then(|c| c.get::<bool>())
                .unwrap()[[0]],
            "one term keeps the 1-4 pair, so the torsion keeps it"
        );
    }

    #[test]
    fn a_torsion_whose_every_row_is_flagged_excludes_its_1_4_pair() {
        let frame = frame_from(&chain5_prmtop(
            "       0       3      -6       9       1       0       3      -6       9       2",
        ));
        let dihedrals = frame.get("dihedrals").unwrap();
        assert_eq!(dihedrals.n_rows(), Some(1));
        assert!(
            dihedrals
                .get("exclude_14")
                .and_then(|c| c.get::<bool>())
                .unwrap()[[0]]
        );
    }

    #[test]
    fn distinct_quartets_stay_distinct_dihedrals() {
        let frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       3       6       9      12       1",
        ));
        let dihedrals = frame.get("dihedrals").unwrap();
        // One spelling for both: the first quartet reads reversed, the second
        // (`c3-os-c3-hc` in atom order) already reads forward.
        assert_eq!(quartets(dihedrals), vec![[3, 2, 1, 0], [1, 2, 3, 4]]);
        let types = dihedrals
            .get("type")
            .and_then(|c| c.get::<String>())
            .unwrap();
        assert_eq!(types[[0]], "c3-os-c3-hc");
        assert_eq!(types[[1]], "c3-os-c3-hc");
    }

    #[test]
    fn a_merged_dihedral_carries_no_prmtop_parameter_index() {
        // `type_id` was the prmtop row's parameter index; a torsion of several
        // rows has no single one, so the block labels by `type` only.
        let frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       0       3      -6       9       2",
        ));
        assert!(
            frame
                .get("dihedrals")
                .unwrap()
                .get("type_id")
                .and_then(|c| c.get::<u64>())
                .is_none()
        );
    }

    #[test]
    fn an_improper_keeps_its_prmtop_atom_order() {
        // `0 9 3 -6`: i=H1, j=C2, k=C1 (j > k by index and by name order is
        // irrelevant), l=O1. AMBER's centre is the third atom; reversing the
        // row would move it to second place.
        let frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       0       9       3      -6       1",
        ));
        let impropers = frame.get("impropers").expect("impropers block");
        assert_eq!(quartets(impropers), vec![[0, 3, 1, 2]]);
        assert_eq!(
            impropers
                .get("type")
                .and_then(|c| c.get::<String>())
                .unwrap()[[0]],
            "hc-c3-c3-os"
        );
    }

    #[test]
    fn an_improper_is_not_a_proper_dihedral() {
        let frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       0       9       3      -6       1",
        ));
        assert_eq!(
            quartets(frame.get("dihedrals").unwrap()),
            vec![[3, 2, 1, 0]]
        );
    }

    #[test]
    fn a_multi_term_torsion_writes_one_lammps_dihedral() {
        use crate::io::lammps::LammpsDataWriter;
        use crate::io::read_lammps_data;
        use crate::io::writer::FrameWriter;

        let mut frame = frame_from(&chain5_prmtop(
            "       0       3       6       9       1       0       3      -6       9       2",
        ));
        let atoms = frame.get_mut("atoms").unwrap();
        for (key, vals) in [
            (keys::X, vec![0.0, 1.0, 2.0, 3.0, 4.0]),
            (keys::Y, vec![0.0, 1.0, 0.0, 1.0, 0.0]),
            (keys::Z, vec![0.0; 5]),
        ] {
            insert_column(atoms, key, vals).unwrap();
        }
        let mut buf = Vec::new();
        LammpsDataWriter::new(&mut buf)
            .write(&frame)
            .expect("write LAMMPS data");
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("\n1 dihedrals\n"), "{text}");
        let back = read_lammps_data(text.as_bytes()).expect("read back");
        assert_eq!(back.get("dihedrals").unwrap().n_rows(), Some(1));
    }

    // -----------------------------------------------------------------
    // Box (ac-005)
    // -----------------------------------------------------------------

    #[test]
    fn ifbox_1_builds_an_orthogonal_cell() {
        let text = litfsi_with_pointers3(0, 1, 0) + BOX_ORTHO;
        let frame = frame_from(&text);
        let simbox = frame
            .simbox
            .as_ref()
            .expect("IFBOX=1 with BOX_DIMENSIONS must build a SimBox");
        let lengths = simbox.lengths();
        for axis in 0..3 {
            assert!(
                (lengths[axis] - 30.0).abs() < 1e-9,
                "axis {axis} length {} != 30.0",
                lengths[axis]
            );
        }
        assert!(
            (simbox.volume() - 27000.0).abs() < 1e-9,
            "volume {} != 27000.0",
            simbox.volume()
        );
    }

    #[test]
    fn ifbox_2_builds_the_truncated_octahedron_cell() {
        let text = litfsi_with_pointers3(0, 2, 0) + BOX_OCTAHEDRON;
        let frame = frame_from(&text);
        let simbox = frame
            .simbox
            .as_ref()
            .expect("IFBOX=2 with BOX_DIMENSIONS must build a SimBox");
        let h = ndarray::arr2(&simbox.matrix());
        // arccos(-1/3) = 109.4712206344907 deg, so 30*cos(alpha) = -10 exactly.
        assert!((h[[0, 0]] - 30.0).abs() < 1e-9, "h[0][0] = {}", h[[0, 0]]);
        assert!((h[[0, 1]] + 10.0).abs() < 1e-9, "h[0][1] = {}", h[[0, 1]]);
        assert!((h[[0, 2]] + 10.0).abs() < 1e-9, "h[0][2] = {}", h[[0, 2]]);
        // 30^3 * sqrt(16/27) = 20784.6096908265...
        assert!(
            (simbox.volume() - 20784.6096908265).abs() < 1e-6,
            "volume {} != 20784.6096908265",
            simbox.volume()
        );
    }

    #[test]
    fn ifbox_3_is_refused() {
        let text = litfsi_with_pointers3(0, 3, 0) + BOX_ORTHO;
        assert_refusal(&text, "IFBOX=3");
    }

    #[test]
    fn ifbox_0_leaves_simbox_none() {
        let frame = frame_from(LITFSI_HEAD);
        assert!(
            frame.simbox.is_none(),
            "IFBOX=0 and no BOX_DIMENSIONS means no cell — nothing is fabricated"
        );
    }

    #[test]
    fn charge_conversion_keeps_the_amber_literal() {
        // 1.82223000E+01 / 18.2223 == 1.0 exactly; re-deriving the factor from
        // molrs' own C = 332.0637133 would shift this by ~1.7e-5.
        let frame = frame_from(LITFSI_HEAD);
        let charge = frame
            .get("atoms")
            .unwrap()
            .get("charge")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert!(
            (charge[[15]] - 1.0).abs() < 1e-12,
            "Li+ charge {}",
            charge[[15]]
        );
    }

    // -----------------------------------------------------------------
    // meta scalars (ac-006)
    // -----------------------------------------------------------------

    #[test]
    fn meta_radius_set_is_the_verbatim_line() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(
            frame.meta.get("radius_set").and_then(|v| v.as_str()),
            Some("modified Bondi radii (mbondi2)")
        );
    }

    #[test]
    fn meta_solvent_pointers_are_three_integers() {
        let frame = frame_from(LITFSI_HEAD);
        assert_eq!(
            frame.meta.get("solvent_iptres").and_then(|v| v.as_i64()),
            Some(2)
        );
        assert_eq!(
            frame.meta.get("solvent_nspm").and_then(|v| v.as_i64()),
            Some(2)
        );
        assert_eq!(
            frame.meta.get("solvent_nspsol").and_then(|v| v.as_i64()),
            Some(3)
        );
    }

    #[test]
    fn meta_oldbeta_is_the_file_value_not_the_cell_angle() {
        let ortho = frame_from(&(litfsi_with_pointers3(0, 1, 0) + BOX_ORTHO));
        assert_eq!(
            ortho.meta.get("oldbeta").and_then(|v| v.as_f64()),
            Some(90.0)
        );

        let octa = frame_from(&(litfsi_with_pointers3(0, 2, 0) + BOX_OCTAHEDRON));
        let oldbeta = octa
            .meta
            .get("oldbeta")
            .and_then(|v| v.as_f64())
            .expect("oldbeta must be recorded for provenance");
        assert!(
            (oldbeta - 109.4712190).abs() < 1e-9,
            "oldbeta {oldbeta} is not the truncated file value"
        );
        assert!(
            (oldbeta - 109.4712206).abs() > 1e-7,
            "oldbeta must be the file's number, not the arccos(-1/3) used for the cell"
        );
    }

    #[test]
    fn meta_scalars_absent_when_their_sections_are() {
        let text = without_section(LITFSI_HEAD, "RADIUS_SET");
        let text = without_section(&text, "SOLVENT_POINTERS");
        let frame = frame_from(&text);
        assert!(frame.meta.get("radius_set").is_none());
        assert!(frame.meta.get("solvent_iptres").is_none());
        assert!(frame.meta.get("solvent_nspm").is_none());
        assert!(frame.meta.get("solvent_nspsol").is_none());
        assert!(frame.meta.get("oldbeta").is_none());
    }

    // -----------------------------------------------------------------
    // Length-checked and discarded (ac-007)
    // -----------------------------------------------------------------

    #[test]
    fn join_array_and_irotat_produce_no_column() {
        let frame = frame_from(LITFSI_HEAD);
        let atoms = frame.get("atoms").unwrap();
        for dead in ["join_array", "join", "irotat", "rotat"] {
            assert!(
                atoms.get(dead).and_then(|c| c.get::<u64>()).is_none()
                    && atoms.get(dead).and_then(|c| c.get::<i32>()).is_none(),
                "{dead} is a dead Amber field: length-checked, then discarded"
            );
        }
    }

    #[test]
    fn join_array_wrong_length_is_refused() {
        let short = "\
%FLAG JOIN_ARRAY
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0
";
        let text = without_section(LITFSI_HEAD, "JOIN_ARRAY") + short;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("JOIN_ARRAY"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains("15") && msg.contains("16"),
            "message {msg:?} must name both counts"
        );
    }

    #[test]
    fn irotat_wrong_length_is_refused() {
        let short = "\
%FLAG IROTAT
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0
";
        let text = without_section(LITFSI_HEAD, "IROTAT") + short;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("IROTAT"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains("15") && msg.contains("16"),
            "message {msg:?} must name both counts"
        );
    }

    #[test]
    fn section_length_mismatch_with_pointers_is_refused() {
        // NRES = 2, but RESIDUE_LABEL carries three 20a4 fields.
        let three = "\
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
TF  LI  XX
";
        let text = without_section(LITFSI_HEAD, "RESIDUE_LABEL") + three;
        let err = refusal_from(&text);
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        let msg = err.to_string();
        assert!(
            msg.contains("RESIDUE_LABEL"),
            "message {msg:?} must name the section"
        );
        assert!(
            msg.contains('3') && msg.contains('2'),
            "message {msg:?} must name both counts"
        );
    }

    // -----------------------------------------------------------------
    // Loud refusals (ac-007)
    // -----------------------------------------------------------------

    #[test]
    fn perturbed_prmtop_is_refused() {
        assert_refusal(&litfsi_with_pointers3(1, 0, 0), "IFPERT");
    }

    #[test]
    fn solvent_cap_prmtop_is_refused() {
        let cap = "\
%FLAG CAP_INFO
%FORMAT(10I8)
      12
%FLAG CAP_INFO2
%FORMAT(5E16.8)
  8.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
";
        assert_refusal(&(litfsi_with_pointers3(0, 0, 1) + cap), "IFCAP");
    }

    #[test]
    fn polarizable_prmtop_is_refused() {
        let ipol = "\
%FLAG IPOL
%FORMAT(1I8)
       1
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ipol), "IPOL");
    }

    /// The hand-written chamber prmtop of the force-field reader's tests.
    const CHAMBER_MINI: &str = include_str!("testdata/chamber_mini.parm7");

    /// A chamber (CHARMM) prmtop reads: its charges
    /// are de-scaled by CHARMM's √332.0716, its CHARMM impropers join the
    /// `impropers` block in file order (centre first, no 1-4), and its CMAP
    /// crossterm is a `cmaps` row named by its five atom types.
    #[test]
    fn a_chamber_prmtop_reads() {
        let frame = frame_from(CHAMBER_MINI);
        let atoms = frame.get("atoms").unwrap();
        let q = atoms.get("charge").unwrap().get::<f64>().unwrap();
        for v in q.iter() {
            assert!((v - 0.1).abs() < 1e-15, "charge {v}");
        }
        let impropers = frame.get("impropers").expect("impropers");
        assert_eq!(quartets(impropers), vec![[1, 0, 2, 3]]);
        let labels = impropers.get("type").unwrap().get::<String>().unwrap();
        assert_eq!(labels[[0]], "N1-C1-C2-C3");
        assert!(
            impropers
                .get(keys::EXCLUDE_14)
                .unwrap()
                .get::<bool>()
                .unwrap()[[0]]
        );
        let cmaps = frame.get("cmaps").expect("cmaps");
        let col = |k: &str| cmaps.get(k).unwrap().get::<u64>().unwrap()[[0]];
        assert_eq!(
            ["atomi", "atomj", "atomk", "atoml", "atomm"].map(col),
            [0, 1, 2, 3, 4]
        );
        assert_eq!(
            cmaps.get("type").unwrap().get::<String>().unwrap()[[0]],
            "C1-N1-C2-C3-N2"
        );
    }

    #[test]
    fn negative_nonbonded_parm_index_is_refused() {
        // NTYPES = 6 -> 36 entries; a negative ICO indexes HBOND_ACOEF (10-12).
        let ico = "\
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       4       7      11      -1       2       3       5       8
      12      17       4       5       6       9      13      18       7       8
       9      10      14      19      11      12      13      14      15      20
      -1      17      18      19      20      21
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ico), "10-12");
    }

    #[test]
    fn lennard_jones_ccoef_prmtop_is_refused() {
        // NTYPES*(NTYPES+1)/2 = 21 C4 terms of the 12-6-4 model.
        let ccoef = "\
%FLAG LENNARD_JONES_CCOEF
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00
";
        assert_refusal(&(LITFSI_HEAD.to_string() + ccoef), "12-6-4");
    }
}
