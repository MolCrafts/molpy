//! LAMMPS molecule template files (native text + JSON).
use crate::io::{Sink, Source};

use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use crate::io::lammps::drude::{SPRINGS, lammps_type_labels, refuse_virtual_sites};
use std::collections::HashMap;
use std::io::{BufRead, Result, Write};

use serde_json::{Value as JsonValue, json};

use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::MetaValue;
use molrs::core::keys;
use molrs::op::{F, I, Idx};

/// The LAMMPS molecule JSON field naming the unit style. It is the file's
/// vocabulary, not the frame's: the frame meta key is [`keys::UNITS`].
const JSON_UNITS: &str = "units";

// ── public path API ─────────────────────────────────────────────────────────

/// Read a LAMMPS molecule template file (the native text format).
///
/// The native format is header counts plus body sections (`Coords`, `Types`,
/// `Charges`, `Bonds`, …); the JSON format
/// ([`read_lammps_molecule_json`]) follows the LAMMPS molecule JSON schema
/// (`format: "molecule"`). Both read into the same canonical columns:
///
/// - `"atoms"`: `id`, `type` (string), optional `x`/`y`/`z`, `charge`, `mass`,
///   `mol_id`, `diameter`
/// - `"bonds"` / `"angles"` / `"dihedrals"` / `"impropers"`: `id`, `type`,
///   `atomi`… (0-based indices into atoms)
/// - meta: `format=lammps_molecule`, `source_format`, counts, optional
///   `title` / `units` / `center_of_mass` / `total_mass` / `inertia`
///
/// Read from a file, the frame's meta carries its path as `source_file`.
pub fn read_lammps_molecule<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let frame = read_lammps_molecule_native(src.into())?;
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Read a LAMMPS molecule template in its JSON format (`format: "molecule"`).
///
/// Read from a file, the frame's meta carries its path as `source_file`.
pub fn read_lammps_molecule_json<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let frame = read_json(src.into())?;
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Write a frame as a LAMMPS molecule template file (the native text format).
pub fn write_lammps_molecule<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    crate::io::writer::check_write_frame(frame)?;
    sink.into()
        .put_with(|w| write_lammps_molecule_native(w, frame))
}

/// Write a frame as a LAMMPS molecule template in its JSON format.
pub fn write_lammps_molecule_json<'a>(sink: impl Into<Sink<'a>>, frame: &Frame) -> Result<()> {
    crate::io::writer::check_write_frame(frame)?;
    sink.into().put_with(|w| write_json(w, frame))
}

// ── native reader ───────────────────────────────────────────────────────────

fn read_lammps_molecule_native(src: Source<'_>) -> Result<Frame> {
    let mut lines: Vec<String> = src
        .reader()?
        .lines()
        .collect::<std::result::Result<_, _>>()?;
    if lines.is_empty() {
        return Err(invalid_data("Empty molecule file"));
    }
    // First line is title/comment.
    let title = lines[0].trim().trim_start_matches('#').trim().to_string();
    lines = lines.into_iter().skip(1).collect();

    let (header, sections) = parse_native_sections(&lines)?;
    let mut frame = Frame::new();
    frame.meta.insert("format", "lammps_molecule");
    frame.meta.insert("source_format", "native");
    if let Some(path) = src.path() {
        frame.meta.insert("source_file", path.display().to_string());
    }
    frame.meta.insert("title", title);
    for (k, v) in [
        ("n_atoms", header.n_atoms),
        ("n_bonds", header.n_bonds),
        ("n_angles", header.n_angles),
        ("n_dihedrals", header.n_dihedrals),
        ("n_impropers", header.n_impropers),
    ] {
        if let Some(n) = v {
            frame.meta.insert(k, MetaValue::I64(n as i64));
        }
    }
    if let Some(m) = header.total_mass {
        frame.meta.insert("total_mass", MetaValue::F64(m));
    }
    if let Some(com) = header.com {
        frame.meta.insert("center_of_mass", MetaValue::F64x3(com));
    }
    if let Some(inertia) = header.inertia {
        frame.meta.insert("inertia", MetaValue::F64x6(inertia));
    }

    let (atoms, id_to_idx) = parse_native_atoms(&sections)?;
    if atoms.n_rows().unwrap_or(0) > 0 {
        frame.insert("atoms", atoms);
    }
    for (name, arity) in [
        ("Bonds", 2usize),
        ("Angles", 3),
        ("Dihedrals", 4),
        ("Impropers", 4),
    ] {
        if let Some(block) = parse_native_connectivity(&sections, name, arity, &id_to_idx)? {
            frame.insert(name.to_ascii_lowercase(), block);
        }
    }
    Ok(frame)
}

#[derive(Default)]
struct NativeHeader {
    n_atoms: Option<usize>,
    n_bonds: Option<usize>,
    n_angles: Option<usize>,
    n_dihedrals: Option<usize>,
    n_impropers: Option<usize>,
    total_mass: Option<F>,
    com: Option<[F; 3]>,
    inertia: Option<[F; 6]>,
}

fn parse_native_sections(lines: &[String]) -> Result<(NativeHeader, HashMap<String, Vec<String>>)> {
    let mut header = NativeHeader::default();
    let mut sections: HashMap<String, Vec<String>> = HashMap::new();
    let mut current: Option<String> = None;

    for raw in lines {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if try_header_line(line, &mut header) {
            continue;
        }
        if let Some(sec) = section_name(line) {
            current = Some(sec);
            sections.entry(current.clone().unwrap()).or_default();
            continue;
        }
        if let Some(ref sec) = current {
            sections.get_mut(sec).unwrap().push(line.to_string());
        }
    }
    Ok((header, sections))
}

fn try_header_line(line: &str, header: &mut NativeHeader) -> bool {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 2 {
        return false;
    }
    let last = parts[parts.len() - 1];
    match last {
        "atoms" => {
            if let Ok(n) = parts[0].parse() {
                header.n_atoms = Some(n);
                return true;
            }
        }
        "bonds" => {
            if let Ok(n) = parts[0].parse() {
                header.n_bonds = Some(n);
                return true;
            }
        }
        "angles" => {
            if let Ok(n) = parts[0].parse() {
                header.n_angles = Some(n);
                return true;
            }
        }
        "dihedrals" => {
            if let Ok(n) = parts[0].parse() {
                header.n_dihedrals = Some(n);
                return true;
            }
        }
        "impropers" => {
            if let Ok(n) = parts[0].parse() {
                header.n_impropers = Some(n);
                return true;
            }
        }
        "mass" => {
            if let Ok(m) = parts[0].parse() {
                header.total_mass = Some(m);
                return true;
            }
        }
        "com" if parts.len() >= 4 => {
            if let (Ok(x), Ok(y), Ok(z)) = (parts[0].parse(), parts[1].parse(), parts[2].parse()) {
                header.com = Some([x, y, z]);
                return true;
            }
        }
        "inertia" if parts.len() >= 7 => {
            let mut arr = [0.0_f64; 6];
            for (i, p) in parts.iter().take(6).enumerate() {
                if let Ok(v) = p.parse() {
                    arr[i] = v;
                } else {
                    return false;
                }
            }
            header.inertia = Some(arr);
            return true;
        }
        _ => {}
    }
    false
}

fn section_name(line: &str) -> Option<String> {
    let lower = line.to_ascii_lowercase();
    const NAMES: &[&str] = &[
        "coords",
        "types",
        "molecules",
        "charges",
        "diameters",
        "masses",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
    ];
    for n in NAMES {
        if lower == *n {
            // Canonical section key: capitalized as in LAMMPS docs for multiword,
            // single word capitalized first letter.
            return Some(match *n {
                "coords" => "Coords".into(),
                "types" => "Types".into(),
                "molecules" => "Molecules".into(),
                "charges" => "Charges".into(),
                "diameters" => "Diameters".into(),
                "masses" => "Masses".into(),
                "bonds" => "Bonds".into(),
                "angles" => "Angles".into(),
                "dihedrals" => "Dihedrals".into(),
                "impropers" => "Impropers".into(),
                _ => n.to_string(),
            });
        }
    }
    None
}

fn parse_native_atoms(
    sections: &HashMap<String, Vec<String>>,
) -> Result<(Block, HashMap<Idx, Idx>)> {
    let types_lines = sections
        .get("Types")
        .ok_or_else(|| invalid_data("Native molecule file must contain Types section"))?;
    let mut ids: Vec<Idx> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    for line in types_lines {
        let parts: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if parts.len() >= 2 {
            ids.push(parts[0].parse().map_err(invalid_data)?);
            types.push(parts[1].to_string());
        }
    }
    if ids.is_empty() {
        return Err(invalid_data("Types section is empty"));
    }
    let id_to_idx: HashMap<Idx, Idx> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as Idx))
        .collect();
    let n = ids.len();
    let mut block = Block::new();
    insert_column(&mut block, "id", ids)?;
    insert_column(&mut block, "type", types)?;

    if let Some(lines) = sections.get("Coords") {
        let mut x = vec![0.0_f64; n];
        let mut y = vec![0.0_f64; n];
        let mut z = vec![0.0_f64; n];
        for line in lines {
            let parts: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            if parts.len() >= 4 {
                let id: Idx = parts[0].parse().map_err(invalid_data)?;
                if let Some(&idx) = id_to_idx.get(&id) {
                    let i = idx as usize;
                    x[i] = parts[1].parse().map_err(invalid_data)?;
                    y[i] = parts[2].parse().map_err(invalid_data)?;
                    z[i] = parts[3].parse().map_err(invalid_data)?;
                }
            }
        }
        insert_column(&mut block, "x", x)?;
        insert_column(&mut block, "y", y)?;
        insert_column(&mut block, "z", z)?;
    }

    for (sec, col) in [
        ("Charges", "charge"),
        ("Masses", "mass"),
        ("Diameters", "diameter"),
    ] {
        if let Some(lines) = sections.get(sec) {
            let mut vals = vec![0.0_f64; n];
            for line in lines {
                let parts: Vec<&str> = line
                    .split('#')
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .collect();
                if parts.len() >= 2 {
                    let id: Idx = parts[0].parse().map_err(invalid_data)?;
                    if let Some(&idx) = id_to_idx.get(&id) {
                        vals[idx as usize] = parts[1].parse().map_err(invalid_data)?;
                    }
                }
            }
            insert_column(&mut block, col, vals)?;
        }
    }
    if let Some(lines) = sections.get("Molecules") {
        let mut vals = vec![0_u64; n];
        for line in lines {
            let parts: Vec<&str> = line
                .split('#')
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            if parts.len() >= 2 {
                let id: Idx = parts[0].parse().map_err(invalid_data)?;
                if let Some(&idx) = id_to_idx.get(&id) {
                    vals[idx as usize] = parts[1].parse().map_err(invalid_data)?;
                }
            }
        }
        insert_column(&mut block, "mol_id", vals)?;
    }
    Ok((block, id_to_idx))
}

fn parse_native_connectivity(
    sections: &HashMap<String, Vec<String>>,
    section: &str,
    arity: usize,
    id_to_idx: &HashMap<Idx, Idx>,
) -> Result<Option<Block>> {
    let Some(lines) = sections.get(section) else {
        return Ok(None);
    };
    if lines.is_empty() {
        return Ok(None);
    }
    let mut ids = Vec::new();
    let mut types = Vec::new();
    let mut members: Vec<Vec<Idx>> = vec![Vec::new(); arity];
    for line in lines {
        let parts: Vec<&str> = line
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        // id type a1 a2 …  → need 2 + arity fields
        if parts.len() < 2 + arity {
            continue;
        }
        ids.push(parts[0].parse::<Idx>().map_err(invalid_data)?);
        types.push(parts[1].to_string());
        for (k, mcol) in members.iter_mut().enumerate() {
            let atom_id: Idx = parts[2 + k].parse().map_err(invalid_data)?;
            let idx = *id_to_idx
                .get(&atom_id)
                .ok_or_else(|| invalid_data(format!("unknown atom id {atom_id} in {section}")))?;
            mcol.push(idx);
        }
    }
    if ids.is_empty() {
        return Ok(None);
    }
    let mut block = Block::new();
    insert_column(&mut block, "id", ids)?;
    insert_column(&mut block, "type", types)?;
    let keys = ["atomi", "atomj", "atomk", "atoml"];
    for (k, col) in members.into_iter().enumerate() {
        insert_column(&mut block, keys[k], col)?;
    }
    Ok(Some(block))
}

// ── native writer ───────────────────────────────────────────────────────────

fn write_lammps_molecule_native(w: &mut impl Write, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let n = atoms
        .n_rows()
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    if n == 0 {
        return Err(invalid_data("Frame must contain atoms data"));
    }
    let sections = topology(frame)?;
    let title = frame
        .meta
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Molecule template written by molrs");
    writeln!(w, "# {title}")?;
    writeln!(w)?;
    writeln!(w, "{n} atoms")?;
    for section in &sections {
        writeln!(w, "{} {}", section.ids.len(), section.block)?;
    }
    if let Some(MetaValue::F64(m)) = frame.meta.get("total_mass") {
        writeln!(w, "{m:.6} mass")?;
    }
    if let Some(MetaValue::F64x3(c)) = frame.meta.get("center_of_mass") {
        writeln!(w, "{:.6} {:.6} {:.6} com", c[0], c[1], c[2])?;
    }
    if let Some(MetaValue::F64x6(inn)) = frame.meta.get("inertia") {
        write!(w, "{:.6}", inn[0])?;
        for v in &inn[1..] {
            write!(w, " {v:.6}")?;
        }
        writeln!(w, " inertia")?;
    }
    writeln!(w)?;

    let ids = atom_ids(atoms);
    if atoms.contains_key("x") && atoms.contains_key("y") && atoms.contains_key("z") {
        writeln!(w, "Coords")?;
        writeln!(w)?;
        let x = atoms
            .get("x")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("x missing"))?;
        let y = atoms
            .get("y")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("y missing"))?;
        let z = atoms
            .get("z")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("z missing"))?;
        for i in 0..n {
            writeln!(w, "{} {:.6} {:.6} {:.6}", ids[i], x[[i]], y[[i]], z[[i]])?;
        }
        writeln!(w)?;
    }
    writeln!(w, "Types")?;
    writeln!(w)?;
    let type_ids = type_ids_for_block(atoms)?;
    for i in 0..n {
        writeln!(w, "{} {}", ids[i], type_ids[i])?;
    }
    writeln!(w)?;

    if atoms.contains_key("mol_id") {
        writeln!(w, "Molecules")?;
        writeln!(w)?;
        if let Some(vals) = atoms.get("mol_id").and_then(|c| c.get::<u64>()) {
            for i in 0..n {
                writeln!(w, "{} {}", ids[i], vals[[i]])?;
            }
        } else if let Some(vals) = atoms.get("mol_id").and_then(|c| c.get::<i32>()) {
            for i in 0..n {
                writeln!(w, "{} {}", ids[i], vals[[i]])?;
            }
        } else {
            return Err(invalid_data("mol_id"));
        }
        writeln!(w)?;
    }
    for (col, heading) in [
        ("charge", "Charges"),
        ("mass", "Masses"),
        ("diameter", "Diameters"),
    ] {
        if !atoms.contains_key(col) {
            continue;
        }
        writeln!(w, "{heading}")?;
        writeln!(w)?;
        let vals = atoms
            .get(col)
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data(col))?;
        for i in 0..n {
            writeln!(w, "{} {:.6}", ids[i], vals[[i]])?;
        }
        writeln!(w)?;
    }

    for section in &sections {
        writeln!(w, "{}", capitalize(section.block))?;
        writeln!(w)?;
        for (row, members) in section.members.iter().enumerate() {
            write!(w, "{} {}", section.ids[row], section.types[row])?;
            for &atom in members {
                write!(w, " {}", atom_id(&ids, atom)?)?;
            }
            writeln!(w)?;
        }
        writeln!(w)?;
    }
    w.flush()
}

/// The topology sections a template has, in file order: `(block, arity,
/// JSON type field)`.
const SECTIONS: [(&str, usize, &str); 4] = [
    ("bonds", 2, "bond-type"),
    ("angles", 3, "angle-type"),
    ("dihedrals", 4, "dihedral-type"),
    ("impropers", 4, "improper-type"),
];

/// One topology section's rows: each row's id, type id and member atoms
/// (0-based indices).
struct Section {
    block: &'static str,
    type_field: &'static str,
    ids: Vec<Idx>,
    types: Vec<I>,
    members: Vec<Vec<usize>>,
}

/// The frame's non-empty topology sections: `bonds` holds the `drudes` rows
/// after its own ([`SPRINGS`]), numbered on from the bonds' last id. A
/// section's types are its rows' `type_id` (a caller's own numbering, as a
/// bond/react system's), else its sorted labels; bonds and Drude springs
/// share one bond-type id space by label, the data writer's
/// ([`lammps_type_labels`]). A `virtual_sites` row is refused.
fn topology(frame: &Frame) -> Result<Vec<Section>> {
    refuse_virtual_sites(frame)?;
    let springs = frame
        .get(SPRINGS.0)
        .and_then(Block::n_rows)
        .is_some_and(|n| n > 0);
    let mut sections = Vec::new();
    for (name, arity, type_field) in SECTIONS {
        let sources: &[&str] = if name == SPRINGS.1 {
            &[SPRINGS.1, SPRINGS.0]
        } else {
            &[name]
        };
        let mut ids = Vec::new();
        let mut members = Vec::new();
        for &source in sources {
            let Some(block) = frame.get(source) else {
                continue;
            };
            let nb = block.n_rows().unwrap_or(0);
            let next = ids.last().copied().unwrap_or(0);
            match block.get("id") {
                None => ids.extend((1..=nb as Idx).map(|k| next + k)),
                Some(col) => {
                    if let Some(col) = col.get::<u64>() {
                        ids.extend(col.iter().copied());
                    } else if let Some(col) = col.get::<i32>() {
                        ids.extend(col.iter().map(|&v| v as Idx));
                    } else {
                        return Err(invalid_data(format!("{source}.id")));
                    }
                }
            }
            let mut cols = Vec::new();
            for key in keys::ENDPOINTS.iter().take(arity) {
                cols.push(
                    block
                        .get(key)
                        .and_then(|c| c.get::<u64>())
                        .ok_or_else(|| invalid_data(format!("{source}.{key}")))?,
                );
            }
            members.extend((0..nb).map(|i| cols.iter().map(|c| c[[i]] as usize).collect()));
        }
        if members.is_empty() {
            continue;
        }
        let types: Vec<I> = if name == SPRINGS.1 && springs {
            lammps_type_labels(frame)
                .map_err(invalid_data)?
                .block(name)
                .map(|t| t.type_ids().iter().map(|&v| v as I).collect())
                .unwrap_or_default()
        } else {
            type_ids_for_block(frame.get(name).expect("a section has rows of its own"))?
        };
        if types.len() != members.len() {
            return Err(invalid_data(format!(
                "internal: {sources:?} type ids {} != rows {}",
                types.len(),
                members.len()
            )));
        }
        sections.push(Section {
            block: name,
            type_field,
            ids,
            types,
            members,
        });
    }
    Ok(sections)
}

/// The template id of the atom at 0-based index `atom`.
fn atom_id(ids: &[Idx], atom: usize) -> Result<Idx> {
    ids.get(atom)
        .copied()
        .ok_or_else(|| invalid_data("connectivity atom index out of range"))
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

fn atom_ids(atoms: &Block) -> Vec<Idx> {
    let n = atoms.n_rows().unwrap_or(0);
    if let Some(col) = atoms.get("id").and_then(|c| c.get::<u64>()) {
        return (0..n).map(|i| col[[i]]).collect();
    }
    if let Some(col) = atoms.get("id").and_then(|c| c.get::<i32>()) {
        return (0..n).map(|i| col[[i]] as Idx).collect();
    }
    (1..=n as Idx).collect()
}

fn type_ids_for_block(block: &Block) -> Result<Vec<I>> {
    let n = block.n_rows().ok_or_else(|| invalid_data("empty block"))?;
    if let Some(col) = block.get("type_id").and_then(|c| c.get::<i32>()) {
        return Ok((0..n).map(|i| col[[i]]).collect());
    }
    if let Some(col) = block.get("type_id").and_then(|c| c.get::<u64>()) {
        return Ok((0..n).map(|i| col[[i]] as I).collect());
    }
    if let Some(col) = block.get("type").and_then(|c| c.get::<String>()) {
        let labels: Vec<String> = (0..n).map(|i| col[[i]].clone()).collect();
        let mut uniq = labels.clone();
        uniq.sort();
        uniq.dedup();
        if uniq.iter().all(|s| s.parse::<I>().is_ok()) {
            return labels
                .iter()
                .map(|s| s.parse::<I>().map_err(invalid_data))
                .collect();
        }
        let map: HashMap<&str, I> = uniq
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), (i + 1) as I))
            .collect();
        return Ok(labels.iter().map(|s| map[s.as_str()]).collect());
    }
    if let Some(col) = block.get("type").and_then(|c| c.get::<i32>()) {
        return Ok((0..n).map(|i| col[[i]]).collect());
    }
    Err(invalid_data("block has no type / type_id"))
}

// ── JSON ────────────────────────────────────────────────────────────────────

/// Refuse a molecule-file `units` style whose numbers a [`Frame`] cannot hold
/// as written: a frame's lengths are Å, its masses g/mol and its charges e
/// ([`keys::UNITS`]), so only a preset in exactly those units passes
/// (`real`, `metal`). A file in another style would need every coordinate,
/// mass and charge converted, and `lj` names no physical scale to convert by.
fn check_frame_units(style: &str) -> Result<()> {
    let frame_unit = [
        ("length", "angstrom"),
        ("mass", "gram_per_mole"),
        ("charge", "elementary_charge"),
    ];
    let preset = molrs::core::UnitPreset::builtin(style)
        .ok_or_else(|| invalid_data(format!("unknown LAMMPS units style `{style}`")))?;
    if let Some((dimension, _)) = frame_unit
        .iter()
        .find(|(dimension, unit)| preset.unit(dimension) != Some(*unit))
    {
        return Err(invalid_data(format!(
            "LAMMPS units `{style}` states {dimension} in {}, but a frame holds Å, g/mol and e; \
             only `real` and `metal` molecules are read and written as they are",
            preset.unit(dimension).unwrap_or("no unit")
        )));
    }
    Ok(())
}

fn read_json(src: Source<'_>) -> Result<Frame> {
    let text = src.text()?;
    let data: JsonValue = serde_json::from_str(&text).map_err(invalid_data)?;
    if data.get("format").and_then(|v| v.as_str()) != Some("molecule") {
        return Err(invalid_data("JSON file must have format='molecule'"));
    }
    let mut frame = Frame::new();
    frame.meta.insert("format", "lammps_molecule");
    frame.meta.insert("source_format", "json");
    if let Some(path) = src.path() {
        frame.meta.insert("source_file", path.display().to_string());
    }
    frame.meta.insert(
        "title",
        data.get("title").and_then(|v| v.as_str()).unwrap_or(""),
    );
    // `units` is optional in the molecule schema; an unstated one stays
    // absent rather than being guessed, and a non-string one is refused.
    if let Some(units) = data.get(JSON_UNITS) {
        let units = units.as_str().ok_or_else(|| {
            invalid_data(format!(
                "JSON molecule field `{JSON_UNITS}` must be a string, got {units}"
            ))
        })?;
        check_frame_units(units)?;
        // The frame's `units` is the force-field units object
        // (`{"preset": …}`), not a bare string.
        frame.meta.insert(
            keys::UNITS,
            MetaValue::Json(serde_json::json!({ "preset": units })),
        );
    }
    frame.meta.insert(
        "revision",
        MetaValue::I64(data.get("revision").and_then(|v| v.as_i64()).unwrap_or(1)),
    );

    let types_data = data
        .pointer("/types/data")
        .and_then(|v| v.as_array())
        .ok_or_else(|| invalid_data("JSON molecule file must contain 'types' section"))?;
    let mut ids = Vec::new();
    let mut types = Vec::new();
    for entry in types_data {
        let a = entry
            .as_array()
            .ok_or_else(|| invalid_data("types data entry must be array"))?;
        ids.push(a[0].as_i64().ok_or_else(|| invalid_data("atom id"))? as Idx);
        let t = if let Some(s) = a[1].as_str() {
            s.to_string()
        } else if let Some(n) = a[1].as_i64() {
            n.to_string()
        } else {
            return Err(invalid_data("type must be str or int"));
        };
        types.push(t);
    }
    let id_to_idx: HashMap<Idx, Idx> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i as Idx))
        .collect();
    let n = ids.len();
    let mut atoms = Block::new();
    insert_column(&mut atoms, "id", ids.clone())?;
    insert_column(&mut atoms, "type", types)?;

    if let Some(coords) = data.pointer("/coords/data").and_then(|v| v.as_array()) {
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut z = vec![0.0; n];
        for entry in coords {
            let a = entry.as_array().ok_or_else(|| invalid_data("coords row"))?;
            let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
            let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
            x[idx] = a[1].as_f64().ok_or_else(|| invalid_data("x"))?;
            y[idx] = a[2].as_f64().ok_or_else(|| invalid_data("y"))?;
            z[idx] = a[3].as_f64().ok_or_else(|| invalid_data("z"))?;
        }
        insert_column(&mut atoms, "x", x)?;
        insert_column(&mut atoms, "y", y)?;
        insert_column(&mut atoms, "z", z)?;
    }
    for (json_key, col, as_int) in [
        ("charges", "charge", false),
        ("masses", "mass", false),
        ("diameters", "diameter", false),
        ("molecule", "mol_id", true),
    ] {
        if let Some(rows) = data
            .pointer(&format!("/{json_key}/data"))
            .and_then(|v| v.as_array())
        {
            if as_int {
                let mut vals = vec![0_u64; n];
                for entry in rows {
                    let a = entry.as_array().ok_or_else(|| invalid_data(json_key))?;
                    let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
                    let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
                    vals[idx] = a[1].as_i64().ok_or_else(|| invalid_data(col))? as u64;
                }
                insert_column(&mut atoms, col, vals)?;
            } else {
                let mut vals = vec![0.0_f64; n];
                for entry in rows {
                    let a = entry.as_array().ok_or_else(|| invalid_data(json_key))?;
                    let id = a[0].as_i64().ok_or_else(|| invalid_data("id"))? as Idx;
                    let idx = *id_to_idx.get(&id).ok_or_else(|| invalid_data("id"))? as usize;
                    vals[idx] = a[1].as_f64().ok_or_else(|| invalid_data(col))?;
                }
                insert_column(&mut atoms, col, vals)?;
            }
        }
    }
    frame.insert("atoms", atoms);

    for (key, arity) in [
        ("bonds", 2usize),
        ("angles", 3),
        ("dihedrals", 4),
        ("impropers", 4),
    ] {
        if let Some(rows) = data
            .pointer(&format!("/{key}/data"))
            .and_then(|v| v.as_array())
        {
            if rows.is_empty() {
                continue;
            }
            let mut cids = Vec::new();
            let mut ctypes = Vec::new();
            let mut members: Vec<Vec<Idx>> = vec![Vec::new(); arity];
            for (i, entry) in rows.iter().enumerate() {
                let a = entry.as_array().ok_or_else(|| invalid_data(key))?;
                cids.push((i + 1) as Idx);
                let t = if let Some(s) = a[0].as_str() {
                    s.to_string()
                } else if let Some(n) = a[0].as_i64() {
                    n.to_string()
                } else {
                    return Err(invalid_data("connectivity type"));
                };
                ctypes.push(t);
                for k in 0..arity {
                    let atom_id = a[1 + k].as_i64().ok_or_else(|| invalid_data("atom id"))? as Idx;
                    let idx = *id_to_idx
                        .get(&atom_id)
                        .ok_or_else(|| invalid_data(format!("unknown atom {atom_id}")))?;
                    members[k].push(idx);
                }
            }
            let mut block = Block::new();
            insert_column(&mut block, "id", cids)?;
            insert_column(&mut block, "type", ctypes)?;
            let keys = ["atomi", "atomj", "atomk", "atoml"];
            for (k, col) in members.into_iter().enumerate() {
                insert_column(&mut block, keys[k], col)?;
            }
            frame.insert(key, block);
        }
    }

    if let Some(com) = data.get("com").and_then(|v| v.as_array())
        && com.len() >= 3
    {
        frame.meta.insert(
            "center_of_mass",
            MetaValue::F64x3([
                com[0].as_f64().unwrap_or(0.0),
                com[1].as_f64().unwrap_or(0.0),
                com[2].as_f64().unwrap_or(0.0),
            ]),
        );
    }
    if let Some(m) = data.get("masstotal").and_then(|v| v.as_f64()) {
        frame.meta.insert("total_mass", MetaValue::F64(m));
    }
    if let Some(inn) = data.get("inertia").and_then(|v| v.as_array())
        && inn.len() >= 6
    {
        let mut a = [0.0_f64; 6];
        for (i, v) in inn.iter().take(6).enumerate() {
            a[i] = v.as_f64().unwrap_or(0.0);
        }
        frame.meta.insert("inertia", MetaValue::F64x6(a));
    }
    Ok(frame)
}

fn write_json(w: &mut impl Write, frame: &Frame) -> Result<()> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let n = atoms
        .n_rows()
        .ok_or_else(|| invalid_data("Frame must contain atoms data"))?;
    let sections = topology(frame)?;
    let ids = atom_ids(atoms);
    let type_labels: Vec<String> =
        if let Some(col) = atoms.get("type").and_then(|c| c.get::<String>()) {
            (0..n).map(|i| col[[i]].clone()).collect()
        } else {
            type_ids_for_block(atoms)?
                .into_iter()
                .map(|v| v.to_string())
                .collect()
        };
    let types_data: Vec<JsonValue> = (0..n)
        .map(|i| {
            if let Ok(t) = type_labels[i].parse::<i64>() {
                json!([ids[i], t])
            } else {
                json!([ids[i], type_labels[i]])
            }
        })
        .collect();

    let title = frame
        .meta
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("Molecule template written by molrs");
    let mut data = json!({
        "application": "LAMMPS",
        "format": "molecule",
        "revision": 1,
        "title": title,
        "schema": "https://download.lammps.org/json/molecule-schema.json",
        "types": { "format": ["atom-id", "type"], "data": types_data },
    });
    // `units` is optional in the molecule schema: written only when the frame
    // states it, and refused when it names no preset.
    if let Some(units) = frame.meta.get(keys::UNITS) {
        let preset = keys::units_preset(units).ok_or_else(|| {
            invalid_data(format!(
                "frame meta `{}` must name a preset (`{{\"preset\": …}}`) \
                 to write the JSON `{JSON_UNITS}` field, got {units:?}",
                keys::UNITS
            ))
        })?;
        check_frame_units(preset)?;
        data[JSON_UNITS] = json!(preset);
    }

    if atoms.contains_key("x") && atoms.contains_key("y") && atoms.contains_key("z") {
        let x = atoms
            .get("x")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("x"))?;
        let y = atoms
            .get("y")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("y"))?;
        let z = atoms
            .get("z")
            .and_then(|c| c.get::<f64>())
            .ok_or_else(|| invalid_data("z"))?;
        let coords: Vec<JsonValue> = (0..n)
            .map(|i| json!([ids[i], x[[i]], y[[i]], z[[i]]]))
            .collect();
        data["coords"] = json!({ "format": ["atom-id", "x", "y", "z"], "data": coords });
    }
    if let Some(c) = atoms.get("charge").and_then(|c| c.get::<f64>()) {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], c[[i]]])).collect();
        data["charges"] = json!({ "format": ["atom-id", "charge"], "data": rows });
    }
    if let Some(m) = atoms.get("mass").and_then(|c| c.get::<f64>()) {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["masses"] = json!({ "format": ["atom-id", "mass"], "data": rows });
    }
    if let Some(m) = atoms.get("mol_id").and_then(|c| c.get::<u64>()) {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["molecule"] = json!({ "format": ["atom-id", "molecule-id"], "data": rows });
    } else if let Some(m) = atoms.get("mol_id").and_then(|c| c.get::<i32>()) {
        let rows: Vec<JsonValue> = (0..n).map(|i| json!([ids[i], m[[i]]])).collect();
        data["molecule"] = json!({ "format": ["atom-id", "molecule-id"], "data": rows });
    }

    for section in &sections {
        let mut format = vec![section.type_field.to_string()];
        let arity = section.members.first().map_or(0, Vec::len);
        format.extend((1..=arity).map(|k| format!("atom{k}")));
        let mut rows = Vec::new();
        for (row, members) in section.members.iter().enumerate() {
            let mut out = vec![json!(section.types[row])];
            for &atom in members {
                out.push(json!(atom_id(&ids, atom)?));
            }
            rows.push(JsonValue::Array(out));
        }
        data[section.block] = json!({ "format": format, "data": rows });
    }

    if let Some(MetaValue::F64x3(c)) = frame.meta.get("center_of_mass") {
        data["com"] = json!([c[0], c[1], c[2]]);
    }
    if let Some(MetaValue::F64(m)) = frame.meta.get("total_mass") {
        data["masstotal"] = json!(m);
    }
    if let Some(MetaValue::F64x6(inn)) = frame.meta.get("inertia") {
        data["inertia"] = json!(inn.to_vec());
    }

    serde_json::to_writer_pretty(w, &data).map_err(invalid_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WATER: &str = r#"# Water molecule. TIP3P geometry
3 atoms
2 bonds
1 angles

Coords

1    0.00000  -0.06556   0.00000
2    0.75695   0.52032   0.00000
3   -0.75695   0.52032   0.00000

Types

1        1
2        2
3        2

Charges

1       -0.834
2        0.417
3        0.417

Bonds

1   1      1      2
2   1      1      3

Angles

1   1      2      1      3
"#;

    #[test]
    fn read_native_water_molecule() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("water.mol");
        std::fs::write(&path, WATER).unwrap();
        let frame = read_lammps_molecule(&path).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(3));
        assert_eq!(frame.get("bonds").unwrap().n_rows(), Some(2));
        assert_eq!(frame.get("angles").unwrap().n_rows(), Some(1));
        let charge = atoms
            .get("charge")
            .and_then(|c| c.get::<f64>())
            .expect("Charges section");
        assert_eq!(
            charge.iter().copied().collect::<Vec<_>>(),
            vec![-0.834, 0.417, 0.417]
        );
    }

    #[test]
    fn native_write_then_read_keeps_every_section() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("water.mol");
        std::fs::write(&path, WATER).unwrap();
        let frame = read_lammps_molecule(&path).unwrap();
        let out = dir.path().join("water_out.mol");
        write_lammps_molecule(&out, &frame).unwrap();
        let back = read_lammps_molecule(&out).unwrap();
        for (block, rows) in [("atoms", 3), ("bonds", 2), ("angles", 1)] {
            assert_eq!(back.get(block).unwrap().n_rows(), Some(rows), "{block}");
        }
        let charge = back
            .get("atoms")
            .unwrap()
            .get("charge")
            .and_then(|c| c.get::<f64>())
            .unwrap();
        assert_eq!(
            charge.iter().copied().collect::<Vec<_>>(),
            vec![-0.834, 0.417, 0.417]
        );
    }

    /// A relation block of `pairs` (0-based atom indices) typed `types`.
    fn pair_block(pairs: &[(Idx, Idx)], types: &[&str]) -> Block {
        let mut block = Block::new();
        let (i, j): (Vec<Idx>, Vec<Idx>) = pairs.iter().copied().unzip();
        insert_column(&mut block, keys::ATOMI, i).unwrap();
        insert_column(&mut block, keys::ATOMJ, j).unwrap();
        let types: Vec<String> = types.iter().map(|t| t.to_string()).collect();
        insert_column(&mut block, keys::TYPE, types).unwrap();
        block
    }

    /// A polarisable methane-like fragment: core `CT` (1) with its Drude
    /// particle `DCT` (2) and two `HC` (3, 4), bonded `CT-HC`.
    fn drude_fragment() -> Frame {
        let mut atoms = Block::new();
        let types: Vec<String> = ["CT", "DCT", "HC", "HC"].map(String::from).to_vec();
        insert_column(&mut atoms, keys::TYPE, types).unwrap();
        insert_column(&mut atoms, keys::CHARGE, vec![0.5 as F, -0.5, 0.0, 0.0]).unwrap();
        for (key, x) in [
            ("x", [0.0 as F, 0.1, 1.0, -1.0]),
            ("y", [0.0; 4]),
            ("z", [0.0; 4]),
        ] {
            insert_column(&mut atoms, key, x.to_vec()).unwrap();
        }
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", pair_block(&[(0, 2), (0, 3)], &["CT-HC", "CT-HC"]));
        frame.insert("drudes", pair_block(&[(0, 1)], &["CT-DCT"]));
        frame
    }

    /// As the data writer does, the template writes each Drude spring
    /// (`drudes` row) as a `Bonds` row after the bonds, its type a bond type
    /// of its own label numbered with the bonds' (`lammps_type_labels`):
    /// LAMMPS's DRUDE package reads the core-shell springs as bonds.
    #[test]
    fn drude_springs_are_template_bonds_of_their_own_type() {
        let dir = tempfile::tempdir().unwrap();
        let frame = drude_fragment();
        let path = dir.path().join("drude.mol");
        write_lammps_molecule(&path, &frame).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.contains("\n3 bonds\n"), "{text}");
        assert!(
            text.contains("Bonds\n\n1 2 1 3\n2 2 1 4\n3 1 1 2\n"),
            "{text}"
        );
        let back = read_lammps_molecule(&path).unwrap();
        assert_eq!(back.get("bonds").unwrap().n_rows(), Some(3));

        let path = dir.path().join("drude.json");
        write_lammps_molecule_json(&path, &frame).unwrap();
        let back = read_lammps_molecule_json(&path).unwrap();
        let bonds = back.get("bonds").unwrap();
        assert_eq!(bonds.n_rows(), Some(3));
        let j = bonds.get("atomj").and_then(|c| c.get::<u64>()).unwrap();
        assert_eq!(j.iter().copied().collect::<Vec<_>>(), vec![2, 3, 1]);
    }

    /// A template builds no virtual site: a `virtual_sites` row is refused
    /// by both formats, as the data writer refuses it.
    #[test]
    fn virtual_sites_are_refused_by_both_template_formats() {
        let dir = tempfile::tempdir().unwrap();
        let mut frame = drude_fragment();
        let mut sites = Block::new();
        insert_column(&mut sites, "site", vec![1 as Idx]).unwrap();
        frame.insert("virtual_sites", sites);
        for path in ["site.mol", "site.json"] {
            let path = dir.path().join(path);
            let err = if path.extension().is_some_and(|e| e == "json") {
                write_lammps_molecule_json(&path, &frame)
            } else {
                write_lammps_molecule(&path, &frame)
            }
            .expect_err("a virtual site has no template form");
            assert!(err.to_string().contains("virtual_sites"), "{err}");
        }
    }

    #[test]
    fn json_units_that_is_not_a_string_is_refused_on_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atom.json");
        std::fs::write(
            &path,
            r#"{"format": "molecule", "units": 3,
                "types": {"format": ["atom-id", "type"], "data": [[1, 1]]}}"#,
        )
        .unwrap();
        let err = read_lammps_molecule_json(&path).expect_err("a numeric units must be refused");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        assert!(err.to_string().contains("units"), "{err}");
    }

    #[test]
    fn meta_units_that_is_not_a_string_is_refused_on_json_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("water.mol");
        std::fs::write(&path, WATER).unwrap();
        let mut frame = read_lammps_molecule(&path).unwrap();
        frame.meta.insert(keys::UNITS, MetaValue::I64(3));
        let err = write_lammps_molecule_json(dir.path().join("water.json").as_path(), &frame)
            .expect_err("a non-string units meta must be refused");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        assert!(err.to_string().contains("units"), "{err}");
    }

    #[test]
    fn json_units_read_as_the_units_object_and_write_back_as_the_preset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("atom.json");
        std::fs::write(
            &path,
            r#"{"format": "molecule", "units": "real",
                "types": {"format": ["atom-id", "type"], "data": [[1, 1]]}}"#,
        )
        .unwrap();
        let frame = read_lammps_molecule_json(&path).unwrap();
        assert_eq!(
            frame.meta.get(keys::UNITS),
            Some(&MetaValue::Json(serde_json::json!({"preset": "real"})))
        );
        let out = dir.path().join("back.json");
        write_lammps_molecule_json(&out, &frame).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.contains("\"units\": \"real\""), "{text}");
    }

    /// A molecule in `nano` units has nm coordinates; a frame's are Å. It is
    /// refused on read and on write instead of entering a frame unconverted.
    #[test]
    fn json_units_whose_numbers_are_not_frame_units_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nano.json");
        std::fs::write(
            &path,
            r#"{"format": "molecule", "units": "nano",
                "types": {"format": ["atom-id", "type"], "data": [[1, "C"]]},
                "coords": {"format": ["atom-id", "x", "y", "z"], "data": [[1, 0.1, 0.0, 0.0]]}}"#,
        )
        .unwrap();
        let err = read_lammps_molecule_json(&path).unwrap_err().to_string();
        assert!(err.contains("nano") && err.contains("nanometer"), "{err}");

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("type", ndarray::arr1(&["C".to_string()]).into_dyn(), None)
            .unwrap();
        frame.insert("atoms", atoms);
        frame.meta.insert(
            keys::UNITS,
            MetaValue::Json(serde_json::json!({ "preset": "lj" })),
        );
        let mut sink = Vec::new();
        let err = write_lammps_molecule_json(&mut sink, &frame)
            .unwrap_err()
            .to_string();
        assert!(err.contains("`lj`"), "{err}");
    }
}
