//! Quantum ESPRESSO pw.x input reader for explicit cells and Gamma sampling.

use std::collections::BTreeMap;
use std::io::Result;

use crate::core::constants::BOHR_RADIUS;
use crate::core::{Block, Cell, Element, Frame, SimBox, keys};
use crate::io::frame_columns::insert_column;
use crate::io::reader::check_read_frame;
use crate::io::{Source, invalid_data};

/// A declared QE atomic species and its pseudopotential file reference.
pub struct QeSpecies {
    pub label: String,
    pub mass_amu: f64,
    pub pseudopotential: String,
}

/// Structure and settings read from a pw.x input. Namelist values retain
/// their QE spelling, with string quotes removed.
pub struct QeInput {
    pub frame: Frame,
    pub namelists: BTreeMap<String, BTreeMap<String, String>>,
    pub species: Vec<QeSpecies>,
}

/// Read a pw.x input with `ibrav=0` and `K_POINTS gamma`. Explicit cells
/// accept angstrom and bohr, and positions also accept crystal coordinates.
/// The reader preserves calculator settings for the caller to interpret.
pub fn read_qe_input<'a>(source: impl Into<Source<'a>>) -> Result<QeInput> {
    let text = source.into().text()?;
    let lines: Vec<_> = text
        .lines()
        .enumerate()
        .map(|(i, line)| (i + 1, without_comment(line).trim()))
        .filter(|(_, line)| !line.is_empty())
        .collect();
    let mut cursor = 0;
    let mut namelists = BTreeMap::new();
    while let Some(&(line, value)) = lines.get(cursor).filter(|(_, s)| s.starts_with('&')) {
        let end = value.find(char::is_whitespace).unwrap_or(value.len());
        let name = value[1..end].to_ascii_lowercase();
        let mut tokens = namelist_tokens(&value[end..], line)?;
        cursor += 1;
        while !tokens.iter().any(|token| token == "/") {
            let (next_line, value) = take_line(&lines, &mut cursor, &name)?;
            tokens.extend(namelist_tokens(value, next_line)?);
        }
        if tokens.pop().as_deref() != Some("/") || tokens.iter().any(|token| token == "/") {
            return Err(qe_error(line, "unexpected text after namelist terminator"));
        }
        let settings = namelist_fields(&tokens, line)?;
        if namelists.insert(name.clone(), settings).is_some() {
            return Err(qe_error(line, format!("duplicate namelist &{name}")));
        }
    }

    let system = namelists
        .get("system")
        .ok_or_else(|| invalid_data("QE input requires &SYSTEM"))?;
    let count = |key: &str| {
        system
            .get(key)
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .ok_or_else(|| invalid_data(format!("QE &SYSTEM requires positive integer {key}")))
    };
    let n_atoms = count("nat")?;
    let n_species = count("ntyp")?;
    if system.get("ibrav").and_then(|v| v.parse::<i32>().ok()) != Some(0) {
        return Err(invalid_data(
            "QE input requires ibrav=0 and explicit CELL_PARAMETERS",
        ));
    }

    let mut cell = None;
    let mut positions = None;
    let mut species = None;
    let mut gamma = false;
    while cursor < lines.len() {
        let (line, header) = take_line(&lines, &mut cursor, "card")?;
        let mut parts = header.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or_default().to_ascii_uppercase();
        let unit = parts
            .next()
            .unwrap_or_default()
            .trim()
            .trim_matches(['(', ')', '{', '}'])
            .trim()
            .to_ascii_lowercase();
        match name.as_str() {
            "CELL_PARAMETERS" if cell.is_none() => {
                let scale = length_scale(&unit, line)?;
                let mut vectors = [[0.0; 3]; 3];
                for vector in &mut vectors {
                    let (row, value) = take_line(&lines, &mut cursor, &name)?;
                    *vector = triple(&value.split_whitespace().collect::<Vec<_>>(), row)?
                        .map(|value| value * scale);
                }
                cell = Some(vectors);
            }
            "ATOMIC_SPECIES" if species.is_none() && unit.is_empty() => {
                let mut entries: Vec<QeSpecies> = Vec::new();
                for _ in 0..n_species {
                    let (row, value) = take_line(&lines, &mut cursor, &name)?;
                    let tokens: Vec<_> = value.split_whitespace().collect();
                    let [label, mass, pseudopotential] = tokens.as_slice() else {
                        return Err(qe_error(
                            row,
                            "expected species label, mass and pseudopotential",
                        ));
                    };
                    let mass_amu = number(mass, row)?;
                    if mass_amu <= 0.0 || entries.iter().any(|entry| entry.label == *label) {
                        return Err(qe_error(row, "nonpositive mass or duplicate species label"));
                    }
                    entries.push(QeSpecies {
                        label: (*label).into(),
                        mass_amu,
                        pseudopotential: (*pseudopotential).into(),
                    });
                }
                species = Some(entries);
            }
            "ATOMIC_POSITIONS" if positions.is_none() => {
                let scale = if unit == "crystal" {
                    None
                } else {
                    Some(length_scale(&unit, line)?)
                };
                let mut entries = Vec::new();
                for _ in 0..n_atoms {
                    let (row, value) = take_line(&lines, &mut cursor, &name)?;
                    let tokens: Vec<_> = value.split_whitespace().collect();
                    let (label, coords) = tokens
                        .split_first()
                        .ok_or_else(|| qe_error(row, "missing atom label"))?;
                    entries.push((row, *label, triple(coords, row)?));
                }
                positions = Some((scale, entries));
            }
            "K_POINTS" if !gamma && unit == "gamma" => gamma = true,
            _ => {
                return Err(qe_error(
                    line,
                    format!("unsupported or duplicate card {header}"),
                ));
            }
        }
    }

    if !gamma {
        return Err(invalid_data("QE input requires K_POINTS gamma"));
    }
    let cell = cell.ok_or_else(|| invalid_data("missing QE CELL_PARAMETERS"))?;
    let species = species.ok_or_else(|| invalid_data("missing QE ATOMIC_SPECIES"))?;
    let (scale, positions) =
        positions.ok_or_else(|| invalid_data("missing QE ATOMIC_POSITIONS"))?;
    let mut frame = Frame::new();
    frame.simbox = Some(
        SimBox::new(
            Cell::Matrix(std::array::from_fn(|i| std::array::from_fn(|j| cell[j][i]))),
            [0.0; 3],
            [true; 3],
        )
        .map_err(invalid_data)?,
    );
    let mut atoms = Block::new();
    let mut coordinates = [Vec::new(), Vec::new(), Vec::new()];
    let mut labels = Vec::new();
    let mut symbols = Vec::new();
    let mut masses = Vec::new();
    for (line, label, xyz) in positions {
        let entry = species
            .iter()
            .find(|entry| entry.label == label)
            .ok_or_else(|| qe_error(line, format!("unknown species {label}")))?;
        // QE allows a digit or underscore suffix on an element symbol.
        let symbol = label
            .split(['_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            .next()
            .and_then(Element::by_symbol)
            .ok_or_else(|| qe_error(line, format!("invalid element in species {label}")))?;
        for (axis, column) in coordinates.iter_mut().enumerate() {
            let value = match scale {
                Some(scale) => xyz[axis] * scale,
                None => (0..3).map(|i| xyz[i] * cell[i][axis]).sum(),
            };
            column.push(value);
        }
        labels.push(label.to_string());
        symbols.push(symbol.symbol().to_string());
        masses.push(entry.mass_amu);
    }
    for (key, values) in ["x", "y", "z"].into_iter().zip(coordinates) {
        insert_column(&mut atoms, key, values)?;
    }
    insert_column(&mut atoms, keys::ELEMENT, symbols)?;
    insert_column(&mut atoms, keys::TYPE, labels)?;
    insert_column(&mut atoms, keys::MASS, masses)?;
    frame.insert("atoms", atoms);
    check_read_frame(&frame)?;
    Ok(QeInput {
        frame,
        namelists,
        species,
    })
}

fn qe_error(line: usize, message: impl std::fmt::Display) -> std::io::Error {
    invalid_data(format!("QE input line {line}: {message}"))
}

fn take_line<'a>(
    lines: &[(usize, &'a str)],
    cursor: &mut usize,
    context: &str,
) -> Result<(usize, &'a str)> {
    let row = lines
        .get(*cursor)
        .copied()
        .ok_or_else(|| invalid_data(format!("unexpected end of QE input in {context}")))?;
    *cursor += 1;
    Ok(row)
}

fn without_comment(line: &str) -> &str {
    let mut quote = None;
    for (index, ch) in line.char_indices() {
        match quote {
            Some(delimiter) if ch == delimiter => quote = None,
            None if matches!(ch, '\'' | '"') => quote = Some(ch),
            None if ch == '!' => return &line[..index],
            _ => {}
        }
    }
    line
}

fn namelist_tokens(line: &str, row: usize) -> Result<Vec<String>> {
    let mut tokens = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch.is_whitespace() || ch == ',' {
            continue;
        }
        let mut token = String::from(ch);
        if matches!(ch, '\'' | '"') {
            loop {
                let next = chars
                    .next()
                    .ok_or_else(|| qe_error(row, "unclosed namelist string"))?;
                token.push(next);
                if next == ch {
                    if chars.next_if_eq(&ch).is_none() {
                        break;
                    }
                    token.push(ch);
                }
            }
        } else if !matches!(ch, '=' | '/') {
            while let Some(next) =
                chars.next_if(|c| !c.is_whitespace() && !matches!(c, ',' | '=' | '/'))
            {
                token.push(next);
            }
        }
        tokens.push(token);
    }
    Ok(tokens)
}

fn namelist_fields(tokens: &[String], line: usize) -> Result<BTreeMap<String, String>> {
    let mut settings = BTreeMap::new();
    for field in tokens.chunks(3) {
        let [key, equals, value] = field else {
            return Err(qe_error(line, "expected namelist key=value"));
        };
        if equals != "=" || !key.starts_with(|c: char| c.is_ascii_alphabetic()) {
            return Err(qe_error(
                line,
                format!("expected namelist assignment near {key}"),
            ));
        }
        let value = if value.starts_with(['\'', '"']) {
            let quote = &value[..1];
            value[1..value.len() - 1].replace(&quote.repeat(2), quote)
        } else {
            value.clone()
        };
        if settings.insert(key.to_ascii_lowercase(), value).is_some() {
            return Err(qe_error(line, format!("duplicate namelist key {key}")));
        }
    }
    Ok(settings)
}

fn length_scale(unit: &str, line: usize) -> Result<f64> {
    match unit {
        "angstrom" => Ok(1.0),
        "bohr" => Ok(BOHR_RADIUS / 1e-10),
        _ => Err(qe_error(
            line,
            format!("unsupported coordinate unit {unit:?}"),
        )),
    }
}

fn number(value: &str, line: usize) -> Result<f64> {
    value
        .replace(['d', 'D'], "e")
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| qe_error(line, format!("invalid number {value:?}")))
}

fn triple(tokens: &[&str], line: usize) -> Result<[f64; 3]> {
    let [x, y, z] = tokens else {
        return Err(qe_error(line, "expected exactly three coordinates"));
    };
    Ok([number(x, line)?, number(y, line)?, number(z, line)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    const DECK: &str = "\
&control prefix='quoted / ! value', pseudo_dir='./' /
&system ibrav=0, nat=1, ntyp=1, ecutwfc=2.0d1 /
ATOMIC_SPECIES
Si1 28.0855 Si.upf
CELL_PARAMETERS {bohr}
4 0 0
1 5 0
0 0 6
ATOMIC_POSITIONS (crystal)
Si1 0.25 0.5 0
K_POINTS gamma
";

    #[test]
    fn preserves_namelists_species_and_cartesian_frame() {
        let input = read_qe_input(DECK.as_bytes()).unwrap();
        assert_eq!(input.namelists["control"]["prefix"], "quoted / ! value");
        assert_eq!(input.namelists["system"]["ecutwfc"], "2.0d1");
        assert_eq!(input.species[0].label, "Si1");
        assert_eq!(input.species[0].pseudopotential, "Si.upf");
        let atoms = &input.frame["atoms"];
        let x = atoms["x"].get::<f64>().unwrap();
        assert!((x[[0]] - 1.5 * BOHR_RADIUS / 1e-10).abs() < 1e-12);
        let cell = input.frame.simbox.as_ref().unwrap().matrix();
        assert_eq!(cell[0][1], BOHR_RADIUS / 1e-10);
        assert_eq!(cell[1][0], 0.0);
    }

    #[test]
    fn rejects_extra_cards_and_inconsistent_counts() {
        for deck in [
            DECK.replace("gamma", "automatic"),
            DECK.replace("nat=1", "nat=2"),
            DECK.replace("Si1 0.25", "Si2 0.25"),
            DECK.replace("ntyp=1", "ntyp=1, NTYP=1"),
        ] {
            assert!(read_qe_input(deck.as_bytes()).is_err(), "{deck}");
        }
    }
}
