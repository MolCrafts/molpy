//! Amber prep (`.prepi` / `.prep`) residue template read/write.

use crate::io::invalid_data;
use std::fs;
use std::io::Result;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// One atom row in a prep residue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepAtom {
    pub index: i32,
    pub name: String,
    pub atom_type: String,
    pub tree_type: String,
    pub na: i32,
    pub nb: i32,
    pub nc: i32,
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
    pub charge: f64,
    #[serde(default)]
    pub element: String,
}

/// Residue definition in prep format.
///
/// The surface is minimal (molpy's prep surface): the residue name, the atom
/// Z-matrix rows and the optional `IMPROPER` section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrepResidue {
    pub name: String,
    pub atoms: Vec<PrepAtom>,
    #[serde(default)]
    pub head_atom: Option<String>,
    #[serde(default)]
    pub tail_atom: Option<String>,
    #[serde(default)]
    pub impropers: Vec<Vec<String>>,
}

/// Read a prep file from disk.
pub fn read_prep(path: impl AsRef<Path>) -> Result<PrepResidue> {
    let text = fs::read_to_string(path.as_ref())?;
    read_prep_str(&text)
}

/// Parse prep text into a residue.
pub fn read_prep_str(text: &str) -> Result<PrepResidue> {
    let lines: Vec<&str> = text.lines().collect();
    let mut name = String::new();
    let mut atoms = Vec::new();
    let mut impropers: Vec<Vec<String>> = Vec::new();
    let mut in_improper = false;
    let mut saw_correct = false;

    for (i, line) in lines.iter().enumerate() {
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        if s.starts_with("CORRECT") {
            saw_correct = true;
            continue;
        }
        if s == "IMPROPER" {
            in_improper = true;
            continue;
        }
        if s == "DONE" {
            break;
        }
        if in_improper {
            impropers.push(s.split_whitespace().map(|x| x.to_string()).collect());
            continue;
        }
        // Residue name is the first non-empty line that is not the 0 0 2 header
        // and appears before CORRECT, or the line right after "0 0 2".
        if !saw_correct {
            let toks: Vec<&str> = s.split_whitespace().collect();
            if toks.len() == 3 && toks.iter().all(|t| t.parse::<i32>().is_ok()) {
                // header 0 0 2
                continue;
            }
            if name.is_empty() && toks.len() == 1 {
                name = toks[0].to_string();
                continue;
            }
        }
        // Atom lines: index name type tree na nb nc r theta phi charge
        let parts: Vec<&str> = s.split_whitespace().collect();
        if parts.len() >= 11 && parts[0].parse::<i32>().is_ok() {
            let atom = PrepAtom {
                index: parts[0].parse().map_err(invalid_data)?,
                name: parts[1].to_string(),
                atom_type: parts[2].to_string(),
                tree_type: parts[3].to_string(),
                na: parts[4].parse().map_err(invalid_data)?,
                nb: parts[5].parse().map_err(invalid_data)?,
                nc: parts[6].parse().map_err(invalid_data)?,
                r: parts[7].parse().map_err(invalid_data)?,
                theta: parts[8].parse().map_err(invalid_data)?,
                phi: parts[9].parse().map_err(invalid_data)?,
                charge: parts[10].parse().map_err(invalid_data)?,
                element: String::new(),
            };
            atoms.push(atom);
        } else if name.is_empty() && i < 5 {
            name = s.to_string();
        }
    }

    if name.is_empty() {
        return Err(invalid_data("prep file missing residue name"));
    }
    Ok(PrepResidue {
        name,
        atoms,
        head_atom: None,
        tail_atom: None,
        impropers,
    })
}

/// Format a residue as prep text.
pub fn write_prep_str(res: &PrepResidue) -> String {
    let mut lines = vec![
        "    0    0    2".to_string(),
        res.name.clone(),
        String::new(),
        "CORRECT     OMIT DU   BEG".to_string(),
        String::new(),
    ];
    for a in &res.atoms {
        lines.push(format!(
            "{:5} {:4} {:4} {:1} {:5} {:5} {:5} {:10.5} {:10.5} {:10.5} {:10.6}",
            a.index,
            a.name,
            a.atom_type,
            a.tree_type,
            a.na,
            a.nb,
            a.nc,
            a.r,
            a.theta,
            a.phi,
            a.charge
        ));
    }
    lines.push(String::new());
    if !res.impropers.is_empty() {
        lines.push("IMPROPER".into());
        for imp in &res.impropers {
            lines.push(imp.join(" "));
        }
        lines.push(String::new());
    }
    lines.push("DONE".into());
    lines.push(String::new());
    lines.join("\n")
}

/// Write prep to disk.
pub fn write_prep(path: impl AsRef<Path>, res: &PrepResidue) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, write_prep_str(res))
}
