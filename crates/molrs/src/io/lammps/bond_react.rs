//! LAMMPS `fix bond/react` (REACTER) templates and map files.

use crate::io::{Sink, invalid_data};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as _;
use std::io::Result;

use molrs::core::Frame;

use crate::core::keys::REACT_ID;

/// One `fix bond/react` reaction: the template before and after, and the
/// atoms the map file names, by `react_id`.
///
/// A `fix bond/react` reaction is a pre-reaction template, the same atoms
/// after the reaction, and a *map file* pairing them: which two atoms initiate
/// the reaction, which template atoms sit on the template's edge (bonded to
/// atoms the template leaves out), which atoms the reaction deletes, and the
/// pre ↔ post equivalence of every atom. [`BondReactTemplate`] holds one
/// reaction; [`write_lammps_bond_react_map`] writes its map file, and
/// [`write_lammps_bond_react_system`](crate::io::write_lammps_bond_react_system)
/// (feature `ff`) writes the whole file set a reactive run reads:
///
/// - `{stem}.data` — the system ([`write_lammps_data`](crate::io::write_lammps_data));
/// - `{stem}.ff` — the force field's styles and coefficients, numbered as the
///   data file is;
/// - `{name}_pre.mol` / `{name}_post.mol` — the templates, as LAMMPS molecule
///   files ([`write_lammps_molecule`](crate::io::write_lammps_molecule));
/// - `{name}.map` — the map file.
///
/// `fix bond/react` matches template atoms to system atoms by **type id**, so
/// the data file and every template must number types the same way. The
/// writer collects every type label the system and the templates use (the
/// string `type` columns) into one inventory per block, declares it in the
/// data file ([`TypeLabels::declare`](molrs::core::TypeLabels::declare)),
/// numbers the templates from it and writes the `*.ff` include with it, so
/// the include covers every type, template-only ones included.
///
/// Template atoms are paired by an integer `react_id` atom column, the same
/// value in the pre and the post template. A template's atom ids are its row
/// numbers (1-based): the map file speaks in them.
///
/// References: LAMMPS `fix bond/react`, <https://docs.lammps.org/fix_bond_react.html>;
/// Gissinger, Jensen & Wise, *Polymer* **128** (2017) 211;
/// *Macromolecules* **53** (2020) 9953.
#[derive(Debug, Clone)]
pub struct BondReactTemplate {
    /// Pre-reaction template; its `atoms` carry the integer `react_id`.
    pub pre: Frame,
    /// Post-reaction template: the same `react_id`s, the new topology.
    pub post: Frame,
    /// The two atoms that initiate the reaction (`InitiatorIDs`), in order.
    pub initiators: [i64; 2],
    /// Atoms bonded to atoms outside the template (`EdgeIDs`). One that is
    /// also an initiator, or is not in `pre`, is left out of the map.
    pub edges: Vec<i64>,
    /// Atoms the reaction deletes (`DeleteIDs`). One not in `pre` is left out.
    pub deleted: Vec<i64>,
}

/// The `react_id` of every atom row of `frame`, in row order.
fn react_ids(frame: &Frame, which: &str) -> Result<Vec<i64>> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| invalid_data(format!("the {which} template has no atoms")))?;
    let column = atoms.get(REACT_ID).ok_or_else(|| {
        invalid_data(format!(
            "the {which} template's atoms have no '{REACT_ID}' column pairing them"
        ))
    })?;
    if let Some(v) = column.get::<i32>() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else if let Some(v) = column.get::<u64>() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else if let Some(v) = column.get::<i64>() {
        Ok(v.iter().copied().collect())
    } else if let Some(v) = column.get::<u32>() {
        Ok(v.iter().map(|&x| x as i64).collect())
    } else {
        Err(invalid_data(format!(
            "the {which} template's '{REACT_ID}' column must hold integers"
        )))
    }
}

/// `react_id` → 1-based template atom id; a repeated `react_id` is an error.
fn index(ids: &[i64], which: &str) -> Result<HashMap<i64, usize>> {
    let mut out = HashMap::with_capacity(ids.len());
    for (row, &rid) in ids.iter().enumerate() {
        if out.insert(rid, row + 1).is_some() {
            return Err(invalid_data(format!(
                "react_id {rid} appears twice in the {which} template"
            )));
        }
    }
    Ok(out)
}

impl BondReactTemplate {
    /// The map file's text ([`write_lammps_bond_react_map`]).
    pub(crate) fn map_file_text(&self) -> Result<String> {
        let pre = react_ids(&self.pre, "pre")?;
        let post = react_ids(&self.post, "post")?;
        let pre_index = index(&pre, "pre")?;
        let post_index = index(&post, "post")?;

        let pre_set: BTreeSet<i64> = pre.iter().copied().collect();
        let post_set: BTreeSet<i64> = post.iter().copied().collect();
        if pre_set != post_set {
            return Err(invalid_data(format!(
                "the pre and post templates hold different atoms: missing in post \
                 {:?}, missing in pre {:?}",
                pre_set.difference(&post_set).collect::<Vec<_>>(),
                post_set.difference(&pre_set).collect::<Vec<_>>(),
            )));
        }

        let mut initiators = [0usize; 2];
        for (slot, rid) in initiators.iter_mut().zip(self.initiators) {
            *slot = *pre_index.get(&rid).ok_or_else(|| {
                invalid_data(format!(
                    "initiator atom (react_id={rid}) is not in the pre template; \
                     extract a wider local environment"
                ))
            })?;
        }
        let edges: Vec<usize> = self
            .edges
            .iter()
            .filter(|rid| !self.initiators.contains(rid))
            .filter_map(|rid| pre_index.get(rid).copied())
            .collect();
        let deleted: Vec<usize> = self
            .deleted
            .iter()
            .filter_map(|rid| pre_index.get(rid).copied())
            .collect();

        let mut text = String::new();
        // `write!` into a String cannot fail.
        let _ = writeln!(text, "# auto-generated map file for fix bond/react\n");
        let _ = writeln!(text, "{} equivalences", pre.len());
        let _ = writeln!(text, "{} edgeIDs", edges.len());
        let _ = writeln!(text, "{} deleteIDs\n", deleted.len());
        let _ = writeln!(text, "InitiatorIDs\n");
        for id in initiators {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nEdgeIDs\n");
        for id in &edges {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nDeleteIDs\n");
        for id in &deleted {
            let _ = writeln!(text, "{id}");
        }
        let _ = writeln!(text, "\nEquivalences\n");
        for (row, rid) in pre.iter().enumerate() {
            let _ = writeln!(text, "{}   {}", row + 1, post_index[rid]);
        }
        Ok(text)
    }
}

/// Write the map file of `template` into `sink`: a path (the file itself,
/// conventionally `{name}.map`, created or truncated) or a buffer (cleared,
/// then holding exactly the file's bytes). Nothing is written when the map is
/// refused.
///
/// ```
/// # use molrs::io::lammps::BondReactTemplate;
/// # fn map_of(template: &BondReactTemplate) -> std::io::Result<String> {
/// use molrs::io::write_lammps_bond_react_map;
///
/// let mut out = Vec::new();
/// write_lammps_bond_react_map(&mut out, template)?;
/// let text = String::from_utf8(out).expect("the map is ASCII");
/// assert!(text.contains("Equivalences"));
/// # Ok(text)
/// # }
/// ```
///
/// # Errors
///
/// [`ErrorKind::InvalidData`](std::io::ErrorKind::InvalidData) when either
/// template lacks an integer `react_id` column or repeats a value, when the
/// pre and post templates do not hold the same `react_id`s, or when an
/// initiator is not in the pre template; the sink's OS error.
pub fn write_lammps_bond_react_map<'a>(
    sink: impl Into<Sink<'a>>,
    template: &BondReactTemplate,
) -> Result<()> {
    let text = template.map_file_text()?;
    sink.into().put(text.as_bytes())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::op::{F, Idx};
    use ndarray::Array1;
    use std::io::ErrorKind;

    pub(crate) fn strings(values: &[&str]) -> ndarray::ArrayD<String> {
        Array1::from_vec(values.iter().map(|s| s.to_string()).collect()).into_dyn()
    }

    /// `atoms` with `react_id`, `type`, coordinates, and `bonds` among them.
    pub(crate) fn template(types: &[&str], rids: &[i64], bonds: &[(Idx, Idx, &str)]) -> Frame {
        let n = types.len();
        let mut atoms = Block::new();
        atoms.insert("type", strings(types), None).unwrap();
        atoms
            .insert(REACT_ID, Array1::from_vec(rids.to_vec()).into_dyn(), None)
            .unwrap();
        // Each label's mass is stated, as a data file needs it; a species
        // column would be no substitute (nothing derives mass from it).
        let masses: Vec<F> = types
            .iter()
            .map(|t| match *t {
                "oh" => 15.999,
                "hc" => 1.008,
                _ => 12.011,
            })
            .collect();
        atoms
            .insert("mass", Array1::from_vec(masses).into_dyn(), None)
            .unwrap();
        for axis in ["x", "y", "z"] {
            atoms
                .insert(axis, Array1::from_vec(vec![0.0 as F; n]).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "charge",
                Array1::from_vec(vec![0.0 as F; n]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        if !bonds.is_empty() {
            let mut b = Block::new();
            b.insert(
                "atomi",
                Array1::from_iter(bonds.iter().map(|t| t.0)).into_dyn(),
                None,
            )
            .unwrap();
            b.insert(
                "atomj",
                Array1::from_iter(bonds.iter().map(|t| t.1)).into_dyn(),
                None,
            )
            .unwrap();
            b.insert(
                "type",
                strings(&bonds.iter().map(|t| t.2).collect::<Vec<_>>()),
                None,
            )
            .unwrap();
            frame.insert("bonds", b);
        }
        frame
    }

    /// c3 + oh → c3-oh: the bond type exists only in the post template.
    pub(crate) fn coupling() -> BondReactTemplate {
        BondReactTemplate {
            pre: template(&["c3", "oh", "hc"], &[1, 2, 3], &[(1, 2, "hc-oh")]),
            post: template(&["oh", "c3", "hc"], &[2, 1, 3], &[(0, 1, "c3-oh")]),
            initiators: [1, 2],
            edges: vec![1, 3],
            deleted: vec![3, 99],
        }
    }

    /// The map file of `t`, written into a buffer.
    fn map(t: &BondReactTemplate) -> Result<String> {
        let mut out = Vec::new();
        write_lammps_bond_react_map(&mut out, t)?;
        Ok(String::from_utf8(out).expect("the map is ASCII"))
    }

    #[test]
    fn the_map_pairs_atoms_and_names_initiators_edges_and_deletions() {
        let text = map(&coupling()).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.contains(&"3 equivalences"));
        // edge 1 is an initiator and is left out.
        assert!(lines.contains(&"1 edgeIDs"));
        // deletion 99 is not in the template and is left out.
        assert!(lines.contains(&"1 deleteIDs"));
        let at = |section: &str| lines.iter().position(|l| *l == section).unwrap();
        assert!(at("InitiatorIDs") < at("EdgeIDs"));
        assert!(at("EdgeIDs") < at("DeleteIDs"));
        assert!(at("DeleteIDs") < at("Equivalences"));
        assert_eq!(
            &lines[at("InitiatorIDs") + 2..at("InitiatorIDs") + 4],
            ["1", "2"]
        );
        let eq: Vec<&str> = lines[at("Equivalences") + 2..].to_vec();
        assert_eq!(eq, ["1   2", "2   1", "3   3"]);
    }

    #[test]
    fn mismatched_templates_and_lost_initiators_are_refused() {
        let mut t = coupling();
        t.post = template(&["oh", "c3"], &[2, 1], &[]);
        assert_eq!(map(&t).unwrap_err().kind(), ErrorKind::InvalidData);
        let mut t = coupling();
        t.initiators = [1, 7];
        assert!(map(&t).unwrap_err().to_string().contains("react_id=7"));
        let mut t = coupling();
        t.pre = template(&["c3", "oh", "hc"], &[1, 1, 3], &[]);
        assert!(map(&t).unwrap_err().to_string().contains("twice"));
    }

    #[test]
    fn the_map_reaches_a_path_and_a_buffer_alike_and_a_refused_one_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rxn.map");
        write_lammps_bond_react_map(path.as_path(), &coupling()).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            map(&coupling()).unwrap()
        );
        let mut t = coupling();
        t.initiators = [1, 7];
        let refused = dir.path().join("refused.map");
        assert!(write_lammps_bond_react_map(refused.as_path(), &t).is_err());
        assert!(!refused.exists());
    }
}
