//! Antechamber `.ac` file reader (ATOM / BOND sections → Frame).

use crate::io::frame_columns::insert_column;
use crate::io::invalid_data;
use std::fs;
use std::io::Result;
use std::path::Path;

use ndarray::{Array1, IxDyn};

use molrs::core::Block;
use molrs::core::Element;
use molrs::core::Frame;
use molrs::op::{F, Idx};

/// Read an Antechamber `.ac` file into a Frame.
///
/// Atoms: `id` (1-based), `name`, `type`, `charge`, `xyz`, optional `element`.
/// Bonds: `atomi`/`atomj` (0-based), `type` as `"typei-typej"`.
pub fn read_ac(path: impl AsRef<Path>) -> Result<Frame> {
    let text = fs::read_to_string(path.as_ref())?;
    read_ac_str(&text)
}

/// Parse Antechamber `.ac` text.
pub fn read_ac_str(text: &str) -> Result<Frame> {
    let mut names: Vec<String> = Vec::new();
    let mut types: Vec<String> = Vec::new();
    let mut charges: Vec<F> = Vec::new();
    let mut xyz: Vec<[F; 3]> = Vec::new();
    let mut elements: Vec<String> = Vec::new();
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
            names.push(name.clone());
            types.push(atype.clone());
            charges.push(charge);
            xyz.push([x, y, z]);
            elements.push(guess_element(&name, &atype));
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
            bond_i.push((i - 1) as Idx);
            bond_j.push((j - 1) as Idx);
            let ti = types.get(i - 1).cloned().unwrap_or_default();
            let tj = types.get(j - 1).cloned().unwrap_or_default();
            bond_type.push(format!("{ti}-{tj}"));
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
        // xyz as (n, 3)
        let flat: Vec<F> = xyz.iter().flat_map(|r| r.iter().copied()).collect();
        let arr = Array1::from_vec(flat)
            .into_shape_with_order(IxDyn(&[n, 3]))
            .map_err(invalid_data)?
            .into_dyn();
        atoms.insert("xyz", arr).map_err(invalid_data)?;
        insert_column(&mut atoms, "x", xyz.iter().map(|r| r[0]).collect())?;
        insert_column(&mut atoms, "y", xyz.iter().map(|r| r[1]).collect())?;
        insert_column(&mut atoms, "z", xyz.iter().map(|r| r[2]).collect())?;
        if elements.iter().any(|e| !e.is_empty()) {
            insert_column(&mut atoms, "element", elements)?;
        }
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

fn guess_element(name: &str, atype: &str) -> String {
    for source in [name, atype] {
        let alpha: String = source.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if alpha.is_empty() {
            continue;
        }
        // Try 2-letter then 1-letter symbols.
        let candidates = [
            alpha.to_ascii_uppercase(),
            alpha
                .chars()
                .next()
                .map(|c| c.to_ascii_uppercase().to_string())
                .unwrap_or_default(),
        ];
        for cand in candidates {
            if cand.is_empty() {
                continue;
            }
            // Element::by_symbol is case-insensitive for common symbols via capitalize
            let title = {
                let mut cs = cand.chars();
                match cs.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + &cs.as_str().to_lowercase(),
                    None => String::new(),
                }
            };
            if Element::by_symbol(&title).is_some() {
                return title;
            }
        }
    }
    String::new()
}
