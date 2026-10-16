//! The labels the PDB writer gives each atom (name, residue name, chain,
//! residue number, insertion code) and the residues a PDB reader such as
//! OpenMM's `PDBFile` builds from them.
//!
//! The rules live here once. [`write_pdb`](crate::io::write_pdb) writes these
//! labels, and the OpenMM system writer
//! ([`write_openmm_system`](crate::io::write_openmm_system)) names its residue
//! templates with them, so the topology OpenMM reads from the PDB matches the
//! templates.
//!
//! A frame's own `name`, `res_name`, `res_id`, `chain` and `icode` columns
//! are written as given (trimmed, then cut to the PDB field width). When the
//! frame lacks one of them:
//!
//! - **residue number**: with no `res_id` column, each bonded molecule (a
//!   connected component of the `bonds` block, numbered from 1 by its first
//!   atom) is one residue;
//! - **residue name**: [`DEFAULT_RES_NAME`] (`MOL`). It is not `UNK`, which
//!   OpenMM treats as an unknown amino acid and bonds by backbone atom names;
//! - **atom name**: the element (else `X`) followed by its 1-based count
//!   among the residue's unnamed atoms of that element (`C1`, `C2`, `H1`, …).
//!   This keeps names unique within a residue while they fit the 4 columns
//!   of the field, then falls back to the bare element.
//!
//! Residue numbers wrap at 10 000 to fit the 4-column `resSeq` field.
//! Consecutive residues still get different numbers.

use std::collections::HashMap;
use std::ops::Range;

use crate::io::invalid_data;
use molrs::core::FrameAccess;
use molrs::op::Idx;

/// The PDB `resName` of an atom whose frame has no `res_name`.
pub(crate) const DEFAULT_RES_NAME: &str = "MOL";

/// `resSeq` is four columns wide.
const RES_SEQ_MODULUS: usize = 10_000;

/// `name` is four columns wide; `resName` three.
const NAME_WIDTH: usize = 4;
const RES_NAME_WIDTH: usize = 3;

/// The per-atom labels of one frame, as the PDB writer writes them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PdbLabels {
    /// Atom names, trimmed (what a reader strips them back to).
    pub names: Vec<String>,
    /// Residue names, trimmed.
    pub res_names: Vec<String>,
    /// Chain identifiers (`' '` for none).
    pub chains: Vec<char>,
    /// Residue numbers as written (`0..10_000`).
    pub res_seqs: Vec<usize>,
    /// Insertion codes (`' '` for none).
    pub icodes: Vec<char>,
}

impl PdbLabels {
    /// The labels of the frame's `n` atoms.
    ///
    /// # Errors
    ///
    /// A `bonds` row naming an atom past `n` (with no `res_id` column, the
    /// bonds decide the residues).
    pub(crate) fn of(frame: &impl FrameAccess, n: usize) -> std::io::Result<Self> {
        let strings = |key: &str| -> Vec<String> {
            frame
                .column("atoms", key)
                .and_then(|c| c.as_string())
                .map(|a| a.iter().map(|s| s.trim().to_string()).collect())
                .unwrap_or_default()
        };
        let one_char = |values: &[String], i: usize| -> char {
            values.get(i).and_then(|s| s.chars().next()).unwrap_or(' ')
        };
        let (chain_col, icode_col) = (strings("chain"), strings("icode"));
        let chains: Vec<char> = (0..n).map(|i| one_char(&chain_col, i)).collect();
        let icodes: Vec<char> = (0..n).map(|i| one_char(&icode_col, i)).collect();
        let res_name_col = strings("res_name");
        let res_names: Vec<String> = (0..n)
            .map(|i| {
                let given = res_name_col.get(i).map(String::as_str).unwrap_or("");
                let name = if given.is_empty() {
                    DEFAULT_RES_NAME
                } else {
                    given
                };
                name.chars().take(RES_NAME_WIDTH).collect()
            })
            .collect();
        let res_ids: Option<Vec<usize>> = frame
            .column("atoms", "res_id")
            .and_then(|c| c.as_uint())
            .map(|a| a.iter().map(|&r| r as usize).collect());
        let res_seqs: Vec<usize> = match res_ids {
            Some(ids) => ids.iter().map(|r| r % RES_SEQ_MODULUS).collect(),
            None => molecules(frame, n)?
                .iter()
                .map(|m| (m + 1) % RES_SEQ_MODULUS)
                .collect(),
        };
        let residues = runs(&chains, &res_seqs, &icodes, &res_names);
        let names = atom_names(&strings("name"), &strings("element"), &residues, n);
        Ok(Self {
            names,
            res_names,
            chains,
            res_seqs,
            icodes,
        })
    }

    /// The residues a PDB reader builds: maximal runs of consecutive atoms
    /// with the same chain, residue number, insertion code and residue name.
    pub(crate) fn residues(&self) -> Vec<Range<usize>> {
        runs(&self.chains, &self.res_seqs, &self.icodes, &self.res_names)
    }
}

/// Runs of consecutive atoms whose residue fields all agree.
fn runs(
    chains: &[char],
    res_seqs: &[usize],
    icodes: &[char],
    res_names: &[String],
) -> Vec<Range<usize>> {
    let key = |i: usize| (chains[i], res_seqs[i], icodes[i], &res_names[i]);
    let mut out: Vec<Range<usize>> = Vec::new();
    for i in 0..res_seqs.len() {
        match out.last_mut() {
            Some(run) if key(run.start) == key(i) => run.end = i + 1,
            _ => out.push(i..i + 1),
        }
    }
    out
}

/// Each atom's name: its own, else its element numbered within its residue.
fn atom_names(
    given: &[String],
    elements: &[String],
    residues: &[Range<usize>],
    n: usize,
) -> Vec<String> {
    let mut names = vec![String::new(); n];
    for residue in residues {
        let mut count: HashMap<&str, usize> = HashMap::new();
        for i in residue.clone() {
            let own = given.get(i).map(String::as_str).unwrap_or("");
            names[i] = if own.is_empty() {
                let element = match elements.get(i).map(String::as_str) {
                    Some(e) if !e.is_empty() => e,
                    _ => "X",
                };
                let k = count.entry(element).or_insert(0);
                *k += 1;
                let numbered = format!("{element}{k}");
                if numbered.chars().count() <= NAME_WIDTH {
                    numbered
                } else {
                    element.chars().take(NAME_WIDTH).collect()
                }
            } else {
                own.chars().take(NAME_WIDTH).collect()
            };
        }
    }
    names
}

/// The 0-based molecule of each atom: connected components of the `bonds`
/// block, numbered by their first atom. No bonds: every atom its own.
fn molecules(frame: &impl FrameAccess, n: usize) -> std::io::Result<Vec<usize>> {
    let column = |key: &str| -> Option<Vec<Idx>> {
        frame
            .column("bonds", key)
            .and_then(|c| c.as_uint())
            .map(|a| a.iter().copied().collect())
    };
    let mut adjacent: Vec<Vec<usize>> = vec![Vec::new(); n];
    if let (Some(ii), Some(jj)) = (column("atomi"), column("atomj")) {
        for (&i, &j) in ii.iter().zip(&jj) {
            let (i, j) = (i as usize, j as usize);
            if i >= n || j >= n {
                return Err(invalid_data(format!(
                    "bond ({i}, {j}) names an atom past the {n} atoms"
                )));
            }
            adjacent[i].push(j);
            adjacent[j].push(i);
        }
    }
    let mut molecule = vec![usize::MAX; n];
    let mut next = 0;
    for start in 0..n {
        if molecule[start] != usize::MAX {
            continue;
        }
        molecule[start] = next;
        let mut stack = vec![start];
        while let Some(a) = stack.pop() {
            for &b in &adjacent[a] {
                if molecule[b] == usize::MAX {
                    molecule[b] = next;
                    stack.push(b);
                }
            }
        }
        next += 1;
    }
    Ok(molecule)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, Frame};
    use ndarray::{Array1, IxDyn};

    fn strings(v: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(v.iter().map(|s| s.to_string()).collect())
            .into_shape_with_order(IxDyn(&[v.len()]))
            .unwrap()
    }

    fn uints(v: &[Idx]) -> ndarray::ArrayD<Idx> {
        Array1::from_vec(v.to_vec())
            .into_shape_with_order(IxDyn(&[v.len()]))
            .unwrap()
    }

    /// Water (O H H) then methane's C H, unnamed and without residues.
    fn two_molecules() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("element", strings(&["O", "H", "H", "C", "H"]))
            .unwrap();
        frame.insert("atoms", atoms);
        let mut bonds = Block::new();
        bonds.insert("atomi", uints(&[0, 0, 3])).unwrap();
        bonds.insert("atomj", uints(&[1, 2, 4])).unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    #[test]
    fn without_residue_columns_each_molecule_is_a_residue_with_numbered_names() {
        let labels = PdbLabels::of(&two_molecules(), 5).unwrap();
        assert_eq!(labels.res_seqs, [1, 1, 1, 2, 2]);
        assert_eq!(labels.res_names, ["MOL"; 5]);
        assert_eq!(labels.names, ["O1", "H1", "H2", "C1", "H1"]);
        assert_eq!(labels.residues(), [0..3, 3..5]);
    }

    #[test]
    fn the_frame_s_own_columns_win() {
        let mut frame = two_molecules();
        let atoms = frame.get_mut("atoms").unwrap();
        atoms
            .insert("name", strings(&["OW", "", "HW2", "CA", "HA"]))
            .unwrap();
        atoms
            .insert("res_name", strings(&["HOH", "HOH", "HOH", "METHANE", ""]))
            .unwrap();
        atoms
            .insert("res_id", uints(&[7, 7, 7, 10_003, 10_003]))
            .unwrap();
        let labels = PdbLabels::of(&frame, 5).unwrap();
        assert_eq!(labels.names, ["OW", "H1", "HW2", "CA", "HA"]);
        assert_eq!(labels.res_names, ["HOH", "HOH", "HOH", "MET", "MOL"]);
        assert_eq!(labels.res_seqs, [7, 7, 7, 3, 3]);
        // A residue-name change splits a residue, as a PDB reader does.
        assert_eq!(labels.residues(), [0..3, 3..4, 4..5]);
    }

    #[test]
    fn a_bond_past_the_atoms_is_an_error() {
        let mut frame = two_molecules();
        frame
            .get_mut("bonds")
            .unwrap()
            .insert("atomj", uints(&[1, 2, 9]))
            .unwrap();
        assert!(PdbLabels::of(&frame, 5).is_err());
    }
}
