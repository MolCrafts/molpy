//! CMAP crossterms of a typed molecule, from its dihedrals: [`assign_cmaps`].

use std::collections::{BTreeSet, HashMap};

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::{TypingError, name_atoms};
use crate::props;
use molrs::core::keys::TYPE;
use molrs::core::schema::block_names::CMAPS;
use molrs::core::{Atomistic, NodeId};

/// Replace `graph`'s `cmaps` terms by the crossterms its dihedrals and `ff`'s
/// `cmap` rows define, and return their number.
///
/// A crossterm is five atoms `(a, b, c, d, e)` whose two dihedrals
/// `(a, b, c, d)` and `(b, c, d, e)` are both dihedrals of `graph` (a
/// dihedral is a path, matched in either stored direction), and whose atom
/// types — each atom's `type` — equal a cmap row's
/// `(itom, jtom, ktom, ltom, mtom)` **in that order**. A row is never matched
/// reversed: CMAP prices φ = dihedral(a, b, c, d) on the grid's first axis
/// and ψ = dihedral(b, c, d, e) on its second, so reading the five atoms the
/// other way is a different term (CHARMM's C-NH1-CT1-C-NH1 backbone types
/// are not a palindrome, so a peptide's φ/ψ pairs are found once each).
///
/// Each crossterm is one `cmaps` term carrying `type` = the matched row's
/// name, added in atom order; every `cmaps` term `graph` had is deleted
/// first.
///
/// # Errors
///
/// - [`TypingError::Malformed`]: `ff` holds cmap rows in more than one style
///   (a kernel prices a whole `cmaps` block, so its rows must belong to one
///   style), or two rows name the same five atom types;
/// - [`TypingError::Conflict`]: a path matches a row read forward and a row
///   read backward (two terms for one pair of dihedrals, neither of which is
///   the other's mistake);
/// - [`TypingError::MissingAtomType`]: an atom of a dihedral has no `type`
///   while `ff` has cmap rows.
///
/// On an error `graph` is unchanged.
pub fn assign_cmaps(graph: &mut Atomistic, ff: &ForceField) -> Result<usize, TypingError> {
    let mut by_types: HashMap<[&str; 5], &str> = HashMap::new();
    let mut style_of_rows: Option<&str> = None;
    for style in ff.get_styles("cmap") {
        let rows = style.defs().types();
        if rows.is_empty() {
            continue;
        }
        if let Some(other) = style_of_rows {
            return Err(TypingError::malformed(format!(
                "cmap rows in two styles (`{other}`, `{}`): a kernel prices a whole \
                 cmaps block, so they must be one style",
                style.name()
            )));
        }
        style_of_rows = Some(style.name());
        for row in rows {
            let key = [
                row.endpoints[0].as_str(),
                row.endpoints[1].as_str(),
                row.endpoints[2].as_str(),
                row.endpoints[3].as_str(),
                row.endpoints[4].as_str(),
            ];
            if let Some(first) = by_types.insert(key, row.name.as_str()) {
                return Err(TypingError::malformed(format!(
                    "cmap rows `{first}` and `{}` both name the atom types {}",
                    row.name,
                    key.join("-")
                )));
            }
        }
    }

    let chains = if by_types.is_empty() {
        Vec::new()
    } else {
        match_chains(graph, &by_types)?
    };
    let stale: Vec<_> = graph
        .term_ids(CMAPS)
        .map_err(TypingError::malformed)?
        .collect();
    graph
        .del_terms(CMAPS, &stale)
        .map_err(TypingError::malformed)?;
    let n = chains.len();
    for (atoms, name) in chains {
        graph
            .add_term(CMAPS, &atoms, props! { TYPE => name })
            .map_err(TypingError::malformed)?;
    }
    Ok(n)
}

/// Every five-atom path of two dihedrals whose types match a row forward,
/// with the row's name, in atom order.
fn match_chains(
    graph: &Atomistic,
    by_types: &HashMap<[&str; 5], &str>,
) -> Result<Vec<([NodeId; 5], String)>, TypingError> {
    let table = graph.atom_table();
    let row = |id: NodeId| table.row(id).expect("a dihedral names live atoms");

    // Every dihedral in both directions, as atom rows, indexed by its first
    // three atoms.
    let mut paths: BTreeSet<[usize; 4]> = BTreeSet::new();
    for (_, d) in graph.dihedrals() {
        let d: [usize; 4] = std::array::from_fn(|p| row(d.nodes[p]));
        paths.insert(d);
        paths.insert([d[3], d[2], d[1], d[0]]);
    }
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let handles: Vec<NodeId> = table.handles().collect();
    let types: Vec<Option<&str>> = handles
        .iter()
        .map(|&id| table.get::<String>(id, TYPE).ok())
        .collect();
    let untyped: BTreeSet<usize> = paths
        .iter()
        .flatten()
        .copied()
        .filter(|&a| types[a].is_none())
        .collect();
    if !untyped.is_empty() {
        let ids: Vec<NodeId> = untyped.iter().map(|&a| handles[a]).collect();
        return Err(TypingError::MissingAtomType {
            table: "cmap".to_owned(),
            detail: name_atoms(graph, &ids),
        });
    }
    let type_of = |a: usize| types[a].expect("every dihedral atom is typed");

    let mut next: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
    for d in &paths {
        next.entry([d[0], d[1], d[2]]).or_default().push(d[3]);
    }
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
            return Err(TypingError::Conflict {
                detail: format!(
                    "atoms {chain:?} match cmap row `{name}` read one way and `{other}` read \
                     the other"
                ),
            });
        }
    }
    Ok(chains
        .into_iter()
        .map(|(chain, name)| (chain.map(|a| handles[a]), name))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::Params;
    use molrs::core::Element;
    use molrs::core::keys::ATOMIC_NUMBER;
    use ndarray::ArrayD;

    /// Atoms of `types`, and the dihedrals `rows` (any direction).
    fn graph(types: &[&str], rows: &[[usize; 4]]) -> Atomistic {
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = types
            .iter()
            .map(|t| {
                g.add_atom(props! { ATOMIC_NUMBER => Element::C, TYPE => *t })
                    .unwrap()
            })
            .collect();
        for r in rows {
            g.add_dihedral(ids[r[0]], ids[r[1]], ids[r[2]], ids[r[3]], props! {})
                .unwrap();
        }
        g
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

    /// The `cmaps` terms as atom rows and type.
    fn rows(g: &Atomistic) -> Vec<([usize; 5], String)> {
        let table = g.atom_table();
        g.term_ids(CMAPS)
            .unwrap()
            .map(|id| {
                let t = g.get_term(CMAPS, id).unwrap();
                let atoms = std::array::from_fn(|p| table.row(t.nodes[p]).unwrap());
                (atoms, t.props.get_str(TYPE).unwrap().to_owned())
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
        let mut g = graph(&types, &dihedrals);
        let n = assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap();
        assert_eq!(n, 2);
        assert_eq!(
            rows(&g),
            vec![
                ([0, 1, 2, 3, 4], "ala".to_owned()),
                ([3, 4, 5, 6, 7], "ala".to_owned())
            ]
        );
        let frame = g.to_frame().unwrap();
        assert_eq!(frame[CMAPS].n_rows(), Some(2));
    }

    /// Only the forward reading matches: the reversed types are another row.
    #[test]
    fn a_row_is_not_matched_reversed() {
        let mut g = graph(
            &["NH1", "C", "CT1", "NH1", "C"],
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
        );
        assert_eq!(assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap(), 1);
        assert_eq!(rows(&g), vec![([4, 3, 2, 1, 0], "ala".to_owned())]);

        let mut g = graph(
            &["X", "C", "CT1", "NH1", "C"],
            &[[0, 1, 2, 3], [1, 2, 3, 4]],
        );
        assert_eq!(assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap(), 0);
        assert_eq!(g.n_terms(CMAPS).unwrap(), 0);
    }

    /// Stale terms are replaced, and removed when nothing matches.
    #[test]
    fn the_terms_are_rebuilt() {
        let mut g = graph(&ALA, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap();
        assert_eq!(g.n_terms(CMAPS).unwrap(), 1);
        assert_eq!(assign_cmaps(&mut g, &ff(&[("x", ["A"; 5])])).unwrap(), 0);
        assert_eq!(g.n_terms(CMAPS).unwrap(), 0);
    }

    /// An untyped dihedral atom is named; the graph keeps its terms.
    #[test]
    fn an_untyped_dihedral_atom_is_refused() {
        let mut g = graph(&ALA, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap();
        let last = g.atom_table().handles().last().unwrap();
        g.clear_atom(last, TYPE).unwrap();
        let err = assign_cmaps(&mut g, &ff(&[("ala", ALA)])).unwrap_err();
        assert!(
            matches!(&err, TypingError::MissingAtomType { detail, .. } if detail == "atom 4 (C)"),
            "{err}"
        );
        assert_eq!(g.n_terms(CMAPS).unwrap(), 1);
    }

    #[test]
    fn ambiguity_is_refused() {
        // A palindromic path matching a row both ways.
        let pal = ["A", "B", "C", "B", "A"];
        let mut g = graph(&pal, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let err = assign_cmaps(&mut g, &ff(&[("p", pal)])).unwrap_err();
        assert!(matches!(err, TypingError::Conflict { .. }), "{err}");

        // Two rows of one style on the same types are refused by the force
        // field itself; across two styles they are refused here.
        let mut two = ff(&[("a", ALA)]);
        let mut params = Params::new();
        params.set("grid", ArrayD::zeros(vec![24, 24]));
        two.def_style("cmap", "other", Params::new())
            .unwrap()
            .def_type("b", &ALA, params)
            .unwrap();
        let mut g = graph(&ALA, &[[0, 1, 2, 3], [1, 2, 3, 4]]);
        let err = assign_cmaps(&mut g, &two).unwrap_err();
        assert!(err.to_string().contains("two styles"), "{err}");
    }
}
