//! CP2K input files with an inline cell and atomic coordinates.

use std::collections::BTreeMap;
use std::io::Result;

use serde_json::{Map, Value, json};

use crate::core::constants::BOHR_RADIUS;
use crate::core::{Block, Cell, Element, Frame, SimBox, keys};
use crate::io::frame_columns::insert_column;
use crate::io::reader::check_read_frame;
use crate::io::{Source, invalid_data};

/// A structure and the CP2K sections, arguments, and keyword values it came from.
pub struct Cp2kInput {
    pub frame: Frame,
    pub settings: Map<String, Value>,
}

#[derive(Default)]
struct Section {
    name: String,
    line: usize,
    arguments: Vec<String>,
    rows: Vec<(usize, Vec<String>)>,
    children: Vec<Section>,
}

impl Section {
    fn child(&self, name: &str) -> Result<&Section> {
        let mut found = self.children.iter().filter(|child| child.name == name);
        let first = found
            .next()
            .ok_or_else(|| error(self.line, format!("missing &{name}")))?;
        if found.next().is_some() {
            return Err(error(self.line, format!("multiple &{name} sections")));
        }
        Ok(first)
    }

    fn keyword(&self, name: &str) -> Result<Option<(usize, &[String])>> {
        let mut found = self
            .rows
            .iter()
            .filter(|(_, row)| row[0].eq_ignore_ascii_case(name));
        let result = found.next().map(|(line, row)| (*line, &row[1..]));
        if found.next().is_some() {
            return Err(error(self.line, format!("multiple {name} keywords")));
        }
        Ok(result)
    }

    fn record(&self) -> Value {
        let mut keywords = BTreeMap::<String, Vec<&[String]>>::new();
        for (_, row) in &self.rows {
            keywords
                .entry(row[0].to_ascii_uppercase())
                .or_default()
                .push(&row[1..]);
        }
        json!({
            "name": self.name,
            "arguments": self.arguments,
            "keywords": keywords,
            "sections": self.children.iter().map(Section::record).collect::<Vec<_>>(),
        })
    }
}

/// Read one CP2K input with inline `&CELL` and `&COORD`. Supports Cartesian
/// angstrom/bohr and fractional coordinates. CP2K preprocessor directives
/// and line continuations are rejected rather than expanded.
pub fn read_cp2k_input<'a>(source: impl Into<Source<'a>>) -> Result<Cp2kInput> {
    let text = source.into().text()?;
    let root = parse_sections(&text)?;
    let system = root.child("FORCE_EVAL")?.child("SUBSYS")?;
    let cell = read_cell(system.child("CELL")?)?;
    let coords = system.child("COORD")?;
    let (scale, fractional) = match (coords.keyword("UNIT")?, coords.keyword("SCALED")?) {
        (Some((line, _)), Some((_, values))) if enabled(values, line)? => {
            return Err(error(
                line,
                "UNIT cannot be used with fractional coordinates",
            ));
        }
        (Some((line, values)), _) => (length_scale(single(values, line)?, line)?, false),
        (None, Some((line, values))) => (1.0, enabled(values, line)?),
        (None, None) => (1.0, false),
    };
    let matrix = cell.matrix();
    let mut columns = [Vec::new(), Vec::new(), Vec::new()];
    let mut elements = Vec::new();
    let mut labels = Vec::new();
    let mut kinds = BTreeMap::new();
    for kind in system
        .children
        .iter()
        .filter(|section| section.name == "KIND")
    {
        let label = single(&kind.arguments, kind.line)?;
        if kinds.insert(label.to_ascii_uppercase(), kind).is_some() {
            return Err(error(kind.line, format!("duplicate &KIND {label}")));
        }
    }
    for (line, row) in &coords.rows {
        let label = &row[0];
        if label.eq_ignore_ascii_case("UNIT") || label.eq_ignore_ascii_case("SCALED") {
            continue;
        }
        let kind = kinds.get(&label.to_ascii_uppercase());
        let symbol = kind
            .map(|kind| kind.keyword("ELEMENT"))
            .transpose()?
            .flatten()
            .map(|(line, values)| single(values, line))
            .transpose()?
            .unwrap_or(label);
        let element = Element::by_symbol(symbol)
            .ok_or_else(|| error(*line, format!("unknown element for atom kind {label}")))?;
        let xyz = triple(&row[1..], *line)?;
        for (axis, column) in columns.iter_mut().enumerate() {
            let value = if fractional {
                matrix[axis].iter().zip(xyz).map(|(a, b)| a * b).sum()
            } else {
                xyz[axis] * scale
            };
            column.push(value);
        }
        elements.push(element.symbol().to_string());
        labels.push(label.clone());
    }
    if elements.is_empty() {
        return Err(error(coords.line, "empty &COORD"));
    }

    let mut atoms = Block::new();
    for (name, values) in ["x", "y", "z"].into_iter().zip(columns) {
        insert_column(&mut atoms, name, values)?;
    }
    insert_column(&mut atoms, keys::ELEMENT, elements)?;
    insert_column(&mut atoms, keys::TYPE, labels)?;
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = Some(cell);
    check_read_frame(&frame)?;

    let mut settings = Map::new();
    settings.insert(
        "sections".into(),
        json!(
            root.children
                .iter()
                .map(Section::record)
                .collect::<Vec<_>>()
        ),
    );
    Ok(Cp2kInput { frame, settings })
}

fn parse_sections(text: &str) -> Result<Section> {
    let mut stack = vec![Section {
        line: 1,
        ..Section::default()
    }];
    for (index, input) in text.lines().enumerate() {
        let line = index + 1;
        let row = tokens(input, line)?;
        let Some(head) = row.first() else { continue };
        if head.starts_with('@') || row.last().is_some_and(|token| token.ends_with('\\')) {
            return Err(error(
                line,
                "CP2K preprocessing and line continuations are unsupported",
            ));
        }
        let section = stack.last_mut().expect("root remains on the stack");
        if head.eq_ignore_ascii_case("&END") {
            if stack.len() == 1 {
                return Err(error(line, "unmatched &END"));
            }
            let section = stack.pop().expect("the section stack is nonempty");
            if row.len() > 2
                || row
                    .get(1)
                    .is_some_and(|name| !section.name.eq_ignore_ascii_case(name))
            {
                return Err(error(line, format!("expected &END {}", section.name)));
            }
            stack
                .last_mut()
                .expect("parent remains on the stack")
                .children
                .push(section);
        } else if let Some(name) = head.strip_prefix('&') {
            if name.is_empty() {
                return Err(error(line, "missing section name"));
            }
            stack.push(Section {
                name: name.to_ascii_uppercase(),
                line,
                arguments: row[1..].to_vec(),
                ..Section::default()
            });
        } else {
            if section.name.is_empty() {
                return Err(error(line, "keyword outside a CP2K section"));
            }
            section.rows.push((line, row));
        }
    }
    if stack.len() != 1 {
        let section = stack.last().expect("section stack is nonempty");
        return Err(error(section.line, format!("unclosed &{}", section.name)));
    }
    Ok(stack.pop().expect("root remains on the stack"))
}

fn tokens(line: &str, line_no: usize) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if matches!(ch, '#' | '!') {
            break;
        }
        let mut token = String::new();
        if matches!(ch, '\'' | '"') {
            loop {
                let next = chars
                    .next()
                    .ok_or_else(|| error(line_no, "unclosed quoted value"))?;
                if next == ch {
                    break;
                }
                token.push(next);
            }
            if chars
                .peek()
                .is_some_and(|c| !c.is_whitespace() && !matches!(c, '#' | '!'))
            {
                return Err(error(line_no, "expected a separator after a quoted value"));
            }
        } else {
            token.push(ch);
            while let Some(next) = chars.next_if(|c| !c.is_whitespace() && !matches!(c, '#' | '!'))
            {
                token.push(next);
            }
        }
        tokens.push(token);
    }
    Ok(tokens)
}

fn read_cell(section: &Section) -> Result<SimBox> {
    let lengths = section.keyword("ABC")?;
    let angles = section.keyword("ALPHA_BETA_GAMMA")?;
    let vectors = [
        section.keyword("A")?,
        section.keyword("B")?,
        section.keyword("C")?,
    ];
    let cell = match (lengths, angles, vectors) {
        (Some((line, values)), angles, [None, None, None]) => {
            let lengths = length_triple(values, line)?;
            match angles {
                Some((line, values)) => Cell::LengthsAngles {
                    lengths,
                    angles: triple(values, line)?,
                },
                None => Cell::Lengths(lengths),
            }
        }
        (None, None, [Some((a_line, a)), Some((b_line, b)), Some((c_line, c))]) => {
            let rows = [
                length_triple(a, a_line)?,
                length_triple(b, b_line)?,
                length_triple(c, c_line)?,
            ];
            Cell::Matrix(std::array::from_fn(|i| std::array::from_fn(|j| rows[j][i])))
        }
        _ => {
            return Err(error(
                section.line,
                "expected ABC or three explicit lattice vectors A, B, C",
            ));
        }
    };
    let periodic = match section.keyword("PERIODIC")? {
        Some((line, values)) => {
            let axes = single(values, line)?.to_ascii_uppercase();
            match axes.as_str() {
                "NONE" => [false; 3],
                "X" | "Y" | "Z" | "XY" | "XZ" | "YZ" | "XYZ" => {
                    ['X', 'Y', 'Z'].map(|axis| axes.contains(axis))
                }
                _ => return Err(error(line, "invalid PERIODIC axes")),
            }
        }
        None => [true; 3],
    };
    SimBox::new(cell, [0.0; 3], periodic).map_err(invalid_data)
}

fn length_triple(values: &[String], line: usize) -> Result<[f64; 3]> {
    let (scale, values) = match values.split_first() {
        Some((first, rest)) if first.starts_with('[') => (length_scale(first, line)?, rest),
        _ => (1.0, values),
    };
    Ok(triple(values, line)?.map(|x| x * scale))
}

fn length_scale(unit: &str, line: usize) -> Result<f64> {
    match unit.trim_matches(['[', ']']).to_ascii_lowercase().as_str() {
        "angstrom" => Ok(1.0),
        "bohr" => Ok(BOHR_RADIUS / 1e-10),
        _ => Err(error(line, format!("unsupported length unit {unit}"))),
    }
}

fn triple(values: &[String], line: usize) -> Result<[f64; 3]> {
    let [a, b, c] = values else {
        return Err(error(line, "expected three numbers"));
    };
    let number = |s: &str| {
        s.replace(['d', 'D'], "e")
            .parse::<f64>()
            .ok()
            .filter(|v| v.is_finite())
            .ok_or_else(|| error(line, format!("invalid number {s}")))
    };
    Ok([number(a)?, number(b)?, number(c)?])
}

fn single(values: &[String], line: usize) -> Result<&str> {
    match values {
        [value] => Ok(value),
        _ => Err(error(line, "expected one value")),
    }
}

fn enabled(values: &[String], line: usize) -> Result<bool> {
    if values.is_empty() {
        return Ok(true);
    }
    match single(values, line)?.to_ascii_uppercase().as_str() {
        "T" | "TRUE" | ".TRUE." => Ok(true),
        "F" | "FALSE" | ".FALSE." => Ok(false),
        _ => Err(error(line, "expected a CP2K logical value")),
    }
}

fn error(line: usize, message: impl std::fmt::Display) -> std::io::Error {
    invalid_data(format!("CP2K input line {line}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: &str = "
&force_eval
  &dft
    BASIS_SET_FILE_NAME 'basis # sets'
    BASIS_SET_FILE_NAME other
  &end dft
  &subsys
    &cell
      A [bohr] 4 0 0
      B [bohr] 1 5 0
      C [bohr] 0 0 6
      PERIODIC xy
    &end
    &coord
      SCALED T
      H_site .25 .5 0 ! preserve order
      O 0 0 .25
    &end coord
    &kind H_site
      ELEMENT H
    &end kind
  &end subsys
&end force_eval
";

    #[test]
    fn reads_fractional_positions_kinds_and_repeated_keywords() {
        let input = read_cp2k_input(INPUT.as_bytes()).unwrap();
        let atoms = &input.frame["atoms"];
        let x = atoms["x"].get::<f64>().unwrap();
        assert!((x[[0]] - 1.5 * BOHR_RADIUS / 1e-10).abs() < 1e-12);
        assert_eq!(atoms[keys::ELEMENT].get::<String>().unwrap()[[0]], "H");
        assert_eq!(atoms[keys::TYPE].get::<String>().unwrap()[[0]], "H_site");
        assert_eq!(input.frame.simbox.unwrap().pbc(), [true, true, false]);
        assert_eq!(
            input.settings["sections"][0]["sections"][0]["keywords"]["BASIS_SET_FILE_NAME"],
            json!([["basis # sets"], ["other"]])
        );
    }

    #[test]
    fn refuses_malformed_sections_and_preprocessor_directives() {
        for input in [
            INPUT.replace("&end coord", "&end cell"),
            INPUT.replace("SCALED T", "@include atoms"),
            INPUT.replace("SCALED T", "SCALED T\n UNIT bohr"),
        ] {
            assert!(read_cp2k_input(input.as_bytes()).is_err());
        }
    }
}
