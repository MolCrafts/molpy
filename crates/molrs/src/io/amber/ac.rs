//! Antechamber `.ac` file reader (ATOM / BOND sections → Frame).

use crate::io::frame_columns::insert_column;
use crate::io::{Source, invalid_data};
use std::io::Result;

use molrs::core::Block;
use molrs::core::Frame;
use molrs::op::{F, Idx};

/// Read an Antechamber `.ac` file into a Frame.
///
/// Atoms: `id` (1-based), `name`, `type`, `charge`, `xyz`.
/// Bonds: `atomi`/`atomj` (0-based), `type` as `"typei-typej"`.
///
/// An `.ac` ATOM record spells no species field, so the frame carries no
/// `element` / `atomic_number` column: species is not guessed from the atom
/// name or type. A consumer that needs one sets `element` itself and runs
/// `Element::fill_atomic_numbers`.
pub fn read_ac<'a>(src: impl Into<Source<'a>>) -> Result<Frame> {
    let frame = parse_ac(&src.into().text()?)?;
    crate::io::reader::check_read_frame(&frame)?;
    Ok(frame)
}

/// Parse Antechamber `.ac` text.
fn parse_ac(text: &str) -> Result<Frame> {
    let mut names: Vec<String> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut charges: Vec<F> = Vec::new();
    let mut xyz: Vec<[F; 3]> = Vec::new();
    let mut bond_i: Vec<Idx> = Vec::new();
    let mut bond_j: Vec<Idx> = Vec::new();
    let mut bond_type: Vec<String> = Vec::new();

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("ATOM") {
            // ATOM id name resname resid x y z charge type
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 10 {
                return Err(invalid_data(format!("short ATOM line: {line}")));
            }
            let name = parts[2].to_string();
            let x: F = parts[5].parse().map_err(invalid_data)?;
            let y: F = parts[6].parse().map_err(invalid_data)?;
            let z: F = parts[7].parse().map_err(invalid_data)?;
            let charge: F = parts[8].parse().map_err(invalid_data)?;
            let atype = parts[9].to_string();
            names.push(name);
            types.push(atype);
            charges.push(charge);
            xyz.push([x, y, z]);
        } else if line.starts_with("BOND") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // BOND id i j order [names...]  — i,j 1-based
            if parts.len() < 4 {
                return Err(invalid_data(format!("short BOND line: {line}")));
            }
            let i: usize = parts[2].parse().map_err(invalid_data)?;
            let j: usize = parts[3].parse().map_err(invalid_data)?;
            if i == 0 || j == 0 {
                return Err(invalid_data("BOND atom index 0"));
            }
            let atom_type = |k: usize| {
                types.get(k - 1).ok_or_else(|| {
                    invalid_data(format!(
                        "BOND names atom {k}, but only {} ATOM records precede it: {line}",
                        types.len()
                    ))
                })
            };
            bond_type.push(format!("{}-{}", atom_type(i)?, atom_type(j)?));
            bond_i.push((i - 1) as Idx);
            bond_j.push((j - 1) as Idx);
        }
    }

    let n = names.len();
    let mut frame = Frame::new();
    if n > 0 {
        let mut atoms = Block::new();
        insert_column(&mut atoms, "id", (1..=n as Idx).collect())?;
        insert_column(&mut atoms, "name", names)?;
        insert_column(&mut atoms, "type", types)?;
        insert_column(&mut atoms, "charge", charges)?;
        insert_column(&mut atoms, "x", xyz.iter().map(|r| r[0]).collect())?;
        insert_column(&mut atoms, "y", xyz.iter().map(|r| r[1]).collect())?;
        insert_column(&mut atoms, "z", xyz.iter().map(|r| r[2]).collect())?;
        frame.insert("atoms", atoms);
    }

    if !bond_i.is_empty() {
        let mut bonds = Block::new();
        let nb = bond_i.len();
        insert_column(&mut bonds, "atomi", bond_i)?;
        insert_column(&mut bonds, "atomj", bond_j)?;
        insert_column(&mut bonds, "type", bond_type)?;
        insert_column(&mut bonds, "id", (1..=nb as Idx).collect())?;
        frame.insert("bonds", bonds);
    }

    Ok(frame)
}

#[cfg(test)]
mod tests {
    use super::*;

    const AC: &str = "\
ATOM      1  C1  MOL     1       0.000   0.000   0.000 -0.100000        c3
ATOM      2  H1  MOL     1       1.090   0.000   0.000  0.100000        hc
BOND    1    1    2    1     C1   H1
";

    #[test]
    fn reads_atoms_and_bonds_without_guessing_species() {
        let frame = read_ac(AC.as_bytes()).unwrap();
        let atoms = frame.get("atoms").unwrap();
        assert_eq!(atoms.n_rows(), Some(2));
        assert!(atoms.get("element").is_none());
        assert!(atoms.get("atomic_number").is_none());
        let bonds = frame.get("bonds").unwrap();
        let ty = bonds.get("type").and_then(|c| c.get::<String>()).unwrap();
        assert_eq!(ty[[0]], "c3-hc");
    }

    #[test]
    fn bond_to_an_unread_atom_errors() {
        let text = "ATOM      1  C1  MOL     1       0.000   0.000   0.000 -0.1  c3\n\
                    BOND    1    1    2    1     C1   H1\n";
        assert!(read_ac(text.as_bytes()).is_err());
    }
}
