//! CMAP crossterms from a frame's dihedrals: [`assign_cmaps`].

use std::collections::{BTreeSet, HashMap};

use ndarray::Array1;

use crate::ff::forcefield::{ForceField, StyleDefs};
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOMJ, ATOMK, ATOML, ATOMM, TYPE};
use molrs::core::schema::block_names::{ATOMS, CMAPS, DIHEDRALS};
use molrs::op::Idx;

/// Build `frame`'s `cmaps` block from its dihedrals and `ff`'s `cmap` rows,
/// and return the number of crossterms.
///
/// A crossterm is five atoms `(a, b, c, d, e)` whose two dihedrals
/// `(a, b, c, d)` and `(b, c, d, e)` are both rows of the `dihedrals` block
/// (a dihedral row is a path, matched in either stored direction), and whose
/// atom types — the `atoms` block's `type` labels — equal a cmap row's
/// `(itom, jtom, ktom, ltom, mtom)` **in that order**. A row is never matched
/// reversed: CMAP prices φ = dihedral(a, b, c, d) on the grid's first axis
/// and ψ = dihedral(b, c, d, e) on its second, so reading the five atoms the
/// other way is a different term (CHARMM's C-NH1-CT1-C-NH1 backbone types
/// are not a palindrome, so a peptide's φ/ψ pairs are found once each).
///
/// The block holds `atomi` … `atomm` and `type` (the matched row's name),
/// one row per crossterm sorted by atom indices; it replaces any `cmaps`
/// block `frame` had, and is removed when nothing matches.
///
/// # Errors
///
/// - `ff` holds cmap rows in more than one style (a kernel prices a whole
///   `cmaps` block, so its rows must belong to one style);
/// - two rows name the same five atom types;
/// - a path matches a row read forward and a row read backward (two terms
///   for one pair of dihedrals, neither of which is the other's mistake);
/// - `atoms` lacks a string `type` column while there are dihedrals, or a
///   `dihedrals` endpoint is missing or out of range.
pub fn assign_cmaps(frame: &mut Frame, ff: &ForceField) -> Result<usize, String> {
    let mut by_types: HashMap<[&str; 5], &str> = HashMap::new();
    let mut style_of_rows: Option<&str> = None;
    for style in ff.get_styles("cmap") {
        let StyleDefs::Cmap(rows) = style.defs() else {
            continue;
        };
        if rows.is_empty() {
            continue;
        }
        if let Some(other) = style_of_rows {
            return Err(format!(
                "cmap rows in two styles (`{other}`, `{}`): a kernel prices a whole \
                 cmaps block, so they must be one style",
                style.name()
            ));
        }
        style_of_rows = Some(style.name());
        for row in rows {
            let key = [
                row.itom.as_str(),
                row.jtom.as_str(),
                row.ktom.as_str(),
                row.ltom.as_str(),
                row.mtom.as_str(),
            ];
            if let Some(first) = by_types.insert(key, row.name.as_str()) {
                return Err(format!(
                    "cmap rows `{first}` and `{}` both name the atom types {}",
                    row.name,
                    key.join("-")
                ));
            }
        }
    }

    let chains = if by_types.is_empty() {
        Vec::new()
    } else {
        match_chains(frame, &by_types)?
    };
    if chains.is_empty() {
        frame.remove(CMAPS);
        return Ok(0);
    }
    let n = chains.len();
    let mut block = Block::new();
    for (p, key) in [ATOMI, ATOMJ, ATOMK, ATOML, ATOMM].into_iter().enumerate() {
        let col: Vec<Idx> = chains.iter().map(|(atoms, _)| atoms[p] as Idx).collect();
        block
            .insert(key, Array1::from_vec(col).into_dyn(), None)
            .map_err(|e| e.to_string())?;
    }
    let labels: Vec<String> = chains.into_iter().map(|(_, name)| name).collect();
    block
        .insert(TYPE, Array1::from_vec(labels).into_dyn(), None)
        .map_err(|e| e.to_string())?;
    frame.insert(CMAPS, block);
    Ok(n)
}

/// Every five-atom path of two dihedrals whose types match a row forward,
/// with the row's name, sorted.
fn match_chains(
    frame: &Frame,
    by_types: &HashMap<[&str; 5], &str>,
) -> Result<Vec<([usize; 5], String)>, String> {
    let Some(dihedrals) = frame.get(DIHEDRALS) else {
        return Ok(Vec::new());
    };
    let n_dihedrals = dihedrals.n_rows().unwrap_or(0);
    if n_dihedrals == 0 {
        return Ok(Vec::new());
    }
    let types = frame
        .get(ATOMS)
        .and_then(|b| b.get(TYPE))
        .and_then(|c| c.get::<String>())
        .ok_or("assign_cmaps: atoms block missing string column \"type\"")?;
    let n_atoms = types.len();
    let column = |key: &str| {
        dihedrals
            .get(key)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| format!("assign_cmaps: dihedrals block missing u64 column {key:?}"))
    };
    let cols = [
        column(ATOMI)?,
        column(ATOMJ)?,
        column(ATOMK)?,
        column(ATOML)?,
    ];

    // Every dihedral in both directions, indexed by its first three atoms.
    let mut paths: BTreeSet<[usize; 4]> = BTreeSet::new();
    for r in 0..n_dihedrals {
        let d: [usize; 4] = std::array::from_fn(|p| cols[p][[r]] as usize);
        if let Some(&bad) = d.iter().find(|&&a| a >= n_atoms) {
            return Err(format!(
                "assign_cmaps: dihedral {r} names atom {bad} of {n_atoms}"
            ));
        }
        paths.insert(d);
        paths.insert([d[3], d[2], d[1], d[0]]);
    }
    let mut next: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for d in &paths {
        next.entry([d[0], d[1], d[2]]).or_default().push(d[3]);
    }

    let type_of = |a: usize| types[[a]].as_str();
    let mut found: HashMap<[usize; 5], String> = HashMap::new();
    for d in &paths {
        let Some(tails) = next.get(&[d[1], d[2], d[3]]) else {
            continue;
        };
        for &e in tails {
            if e == d[0] {
                continue;
            }
            let chain = [d[0], d[1], d[2], d[3], e];
            if let Some(name) = by_types.get(&chain.map(type_of)) {
                found.insert(chain, (*name).to_owned());
            }
        }
    }
    let mut chains: Vec<([usize; 5], String)> = found.into_iter().collect();
    chains.sort();
    for (chain, name) in &chains {
        let back = [chain[4], chain[3], chain[2], chain[1], chain[0]];
        if back < *chain
            && let Some((_, other)) = chains.iter().find(|(c, _)| *c == back)
        {
            return Err(format!(
                "assign_cmaps: atoms {chain:?} match cmap row `{name}` read one way and \
                 `{other}` read the other"
            ));
        }
    }
    Ok(chains)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use ndarray::ArrayD;

    /// Atoms of `types`, and the dihedrals `rows` (any direction).
    fn frame(types: &[&str], rows: &[[Idx; 4]]) -> Frame {
        let mut atoms = Block::new();
        atoms
            .insert(
                TYPE,
                Array1::from_vec(types.iter().map(|t| t.to_string()).collect::<Vec<_>>())
                    .into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        let mut dihedrals = Block::new();
        for (p, key) in [ATOMI, ATOMJ, ATOMK, ATOML].into_iter().enumerate() {
            let col: Vec<Idx> = rows.iter().map(|r| r[p]).collect();
            dihedrals
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        frame.insert(DIHEDRALS, dihedrals);
        frame
    }

    fn ff(rows: &[(&str, [&str; 5])]) -> ForceField {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        for (i, (name, ends)) in rows.iter().enumerate() {
            let mut params = Params::new();
            params.set("grid", ArrayD::from_elem(vec![24, 24], i as f64));
            style.def_type(name, ends, params).unwrap();
        }
        ff
    }

    fn rows(frame: &Frame) -> Vec<([u64; 5], String)> {
        let b = &frame[CMAPS];
        let types = b.get(TYPE).unwrap().get::<String>().unwrap();
        (0..b.n_rows().unwrap())
            .map(|r| {
                let atoms = [ATOMI, ATOMJ, ATOMK, ATOML, ATOMM]
                    .map(|k| b.get(k).unwrap().get::<u64>().unwrap()[[r]]);
                (atoms, types[[r]].clone())
            })
            .collect()
    }

    const ALA: [&str; 5] = ["C", "NH1", "CT1", "C", "NH1"];

    /// A dipeptide backbone C0-N1-CA2-C3-N4-CA5-C6-N7: two φ/ψ pairs, with
    /// the dihedrals stored in mixed directions, plus a side dihedral that
    /// shares atoms but makes no backbone path.
    #[test]
    fn a_backbone_gets_one_crossterm_per_residue_read_forward() {
        let types = ["C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1", "HB"];
        let dihedrals = [
            [0, 1, 2, 3],
            [4, 3, 2, 1], // stored backwards
            [2, 3, 4, 5],
            [3, 4, 5, 6],
            [7, 6, 5, 4], // stored backwards
            [8, 2, 3, 4],
        ];
        let mut f = frame(&types, &dihedrals);
        let n = assign_cmaps(&mut f, &ff(&[("ala", ALA)])).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            rows(&f),
            vec![
                ([0, 1, 2, 3, 4], "ala".to_owned()),
                ([3, 4, 5, 6, 7], "ala".to_owned())
            ]
        );
    }

    /// Only the forward reading matches: the reversed types are another row.
    #[test]
    fn a_row_is_not_matched_reversed() {
        let mut f = frame(
            &["NH1", "C", "CT1", "NH1", "C"],
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
        );
        assert_eq!(assign_cmaps(&mut f, &ff(&[("ala", ALA)])).unwrap(), 1);
        assert_eq!(rows(&f), vec![([4, 3, 2, 1, 0], "ala".to_owned())]);

        let mut f = frame(
            &["X", "C", "CT1", "NH1", "C"],
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
        );
        assert_eq!(assign_cmaps(&mut f, &ff(&[("ala", ALA)])).unwrap(), 0);
        assert!(f.get(CMAPS).is_none());
    }

    /// A stale block is replaced, and removed when nothing matches.
    #[test]
    fn the_block_is_rebuilt() {
        let mut f = frame(&ALA, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        assign_cmaps(&mut f, &ff(&[("ala", ALA)])).unwrap();
        assert_eq!(assign_cmaps(&mut f, &ff(&[("x", ["A"; 5])])).unwrap(), 0);
        assert!(f.get(CMAPS).is_none());
    }

    #[test]
    fn ambiguity_is_refused() {
        // A palindromic path matching a row both ways.
        let pal = ["A", "B", "C", "B", "A"];
        let mut f = frame(&pal, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let err = assign_cmaps(&mut f, &ff(&[("p", pal)])).unwrap_err();
        assert!(err.contains("read the other"), "{err}");

        // Two rows of one style on the same types are refused by the force
        // field itself; across two styles they are refused here.
        let mut two = ff(&[("a", ALA)]);
        let mut params = Params::new();
        params.set("grid", ArrayD::zeros(vec![24, 24]));
        two.def_style("cmap", "other", Params::new())
            .unwrap()
            .def_type("b", &ALA, params)
            .unwrap();
        let mut f = frame(&ALA, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let err = assign_cmaps(&mut f, &two).unwrap_err();
        assert!(err.contains("two styles"), "{err}");
    }
}
