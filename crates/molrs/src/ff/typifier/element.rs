//! Element typing: type labels derived from atomic numbers alone.

use std::collections::HashMap;

use indexmap::IndexMap;
use molrs::core::TypeName;
use molrs::core::keys;
use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS};
use molrs::core::{Atomistic, Element};
use molrs::core::{NodeId, PropValue, Relation};

use crate::ff::forcefield::ForceField;
use crate::ff::typifier::{Annotation, TypeAssignment, Typifier};

/// A typifier whose labels are the elements themselves; it defines no force
/// field.
///
/// # Labels
///
/// Every label is written under `type`:
///
/// - an atom: the symbol of the element its [`keys::ATOMIC_NUMBER`] names,
///   e.g. `"C"` for `Z = 6`;
/// - a bond, angle or dihedral: the elements of its endpoints, oriented to
///   the smaller (slot by slot) of the tuple read forward and reversed, then
///   [`TypeName::join`]ed. Bond O–H is `"H-O"`, angle H–C–C is `"C-C-H"`,
///   dihedral O–C–C–H is `"H-C-C-O"`.
///
/// Angles and dihedrals are labelled only when the graph has them; nothing is
/// generated.
///
/// # Stamp-only
///
/// The [`TypeAssignment`] carries only [`Annotation::Value`] entries and no styles or
/// pair rows, so [`Typing::forcefield`](crate::ff::typifier::Typing::forcefield)
/// stays empty. Masses and charges are left on the atoms as they are.
///
/// # Errors
///
/// - an atom without a valid atomic number, or a wildcard atom (`Z = 0`, which
///   names no element and so has no element type): `Err` naming its row;
/// - a graph with improper rows: `Err` naming their count. Improper labels
///   are not derived.
///
/// # Example
///
/// ```
/// use molrs::core::{Atomistic, Element, keys};
/// use molrs::ff::typifier::{ElementTypifier, Typing};
/// use molrs::props;
///
/// let mut mol = Atomistic::new();
/// let o = mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::O }).unwrap();
/// let h = mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::H }).unwrap();
/// mol.add_bond(o, h, props! {}).unwrap();
///
/// let mut typing = Typing::new(ElementTypifier::new());
/// let typed = typing.typify(&mol).unwrap();
///
/// let (_, bond) = typed.bonds().next().unwrap();
/// assert_eq!(bond.props["type"], "H-O".into());
/// assert!(typing.forcefield().styles().is_empty());
/// ```
///
/// [`ElementTypifier`] gives every atom and link a `type` label built from the
/// elements it touches, for writers that need type labels (LAMMPS data Type
/// Labels) on a molecule no force field has typed.
#[derive(Debug, Clone)]
pub struct ElementTypifier {
    library: ForceField,
}

impl ElementTypifier {
    /// A typifier with an empty library named `"element"`.
    pub fn new() -> Self {
        Self {
            library: ForceField::new("element"),
        }
    }
}

impl Default for ElementTypifier {
    fn default() -> Self {
        Self::new()
    }
}

/// One `type = label` annotation list.
fn type_value(label: &str) -> Vec<(String, Annotation)> {
    vec![(
        "type".to_owned(),
        Annotation::Value(PropValue::Str(label.to_owned())),
    )]
}

/// The label of every row `terms` of relation block `kind`, in row order: the
/// [`TypeName`] of its endpoint elements in [`TypeName::orient`]'s spelling,
/// cached per element tuple. Empty when there are no rows.
fn link_labels(
    kind: &str,
    terms: Vec<Relation>,
    elements: &HashMap<NodeId, &'static str>,
    cache: &mut HashMap<Vec<String>, String>,
) -> Result<Vec<Vec<(String, Annotation)>>, String> {
    let mut rows = Vec::new();
    for (position, term) in terms.iter().enumerate() {
        let tuple: Vec<&str> = term
            .nodes
            .iter()
            .map(|n| {
                elements
                    .get(n)
                    .copied()
                    .ok_or_else(|| format!("{kind} {position}: endpoint {n:?} is not an atom"))
            })
            .collect::<Result<_, _>>()?;
        let key: Vec<String> = tuple.iter().map(|e| (*e).to_owned()).collect();
        let label = match cache.get(&key) {
            Some(label) => label.clone(),
            None => {
                let label = TypeName::join(&TypeName::orient(&tuple))
                    .map_err(|e| format!("{kind} {position}: {e}"))?
                    .as_str()
                    .to_owned();
                cache.insert(key, label.clone());
                label
            }
        };
        rows.push(type_value(&label));
    }
    Ok(rows)
}

impl Typifier for ElementTypifier {
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, String> {
        if graph.n_impropers() > 0 {
            return Err(format!(
                "ElementTypifier derives no improper labels; the graph has {} impropers",
                graph.n_impropers()
            ));
        }
        let mut elements: HashMap<NodeId, &'static str> = HashMap::new();
        let mut nodes = Vec::new();
        for (row, (id, atom)) in graph.atoms().enumerate() {
            let z = atom.atomic_number().ok_or_else(|| {
                format!(
                    "ElementTypifier: atom {row} has no valid '{}'",
                    keys::ATOMIC_NUMBER
                )
            })?;
            let element = Element::by_number(z).ok_or_else(|| {
                format!(
                    "ElementTypifier: atom {row} is a wildcard ('{}' = 0), which has no element type",
                    keys::ATOMIC_NUMBER
                )
            })?;
            elements.insert(id, element.symbol());
            nodes.push(type_value(element.symbol()));
        }
        let terms: [(&str, Vec<Relation>); 3] = [
            (BONDS, graph.bonds().map(|(_, r)| r).collect()),
            (ANGLES, graph.angles().map(|(_, r)| r).collect()),
            (DIHEDRALS, graph.dihedrals().map(|(_, r)| r).collect()),
        ];

        let mut cache: HashMap<Vec<String>, String> = HashMap::new();
        let mut links = IndexMap::new();
        for (kind, terms) in terms {
            links.insert(
                kind.to_owned(),
                link_labels(kind, terms, &elements, &mut cache)?,
            );
        }
        Ok(TypeAssignment {
            nodes,
            links,
            ..TypeAssignment::default()
        })
    }

    fn source_forcefield(&self) -> &ForceField {
        &self.library
    }
}

#[cfg(test)]
mod tests {
    //! `ElementTypifier` through `Typing::typify` on hand-built graphs. Every
    //! expected label is written by hand.

    use crate::props;
    use molrs::core::Element;
    use molrs::core::keys;
    use molrs::core::{Atomistic, NodeId};
    use molrs::core::{PropValue, Props};

    use super::*;
    use crate::ff::typifier::Typing;

    fn str_value(s: &str) -> Option<PropValue> {
        Some(PropValue::Str(s.to_owned()))
    }

    fn atom_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.atoms().map(|(_, a)| a.get("type").cloned()).collect()
    }

    fn bond_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.bonds()
            .map(|(_, b)| b.props.get("type").cloned())
            .collect()
    }

    fn angle_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.angles()
            .map(|(_, a)| a.props.get("type").cloned())
            .collect()
    }

    fn dihedral_types(g: &Atomistic) -> Vec<Option<PropValue>> {
        g.dihedrals()
            .map(|(_, d)| d.props.get("type").cloned())
            .collect()
    }

    /// O bonded to two H: bonds written O–H and H–O.
    fn water() -> Atomistic {
        let mut g = Atomistic::new();
        let o = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let h1 = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        let h2 = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        g.add_bond(o, h1, props! {}).unwrap();
        g.add_bond(h2, o, props! {}).unwrap();
        g
    }

    /// C1–C2–H with bonds C1–C2, C2–H and one angle written H–C2–C1.
    fn cch() -> (Atomistic, [NodeId; 3]) {
        let mut g = Atomistic::new();
        let c1 = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c2 = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let h = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        g.add_bond(c1, c2, props! {}).unwrap();
        g.add_bond(c2, h, props! {}).unwrap();
        (g, [c1, c2, h])
    }

    #[test]
    fn water_atoms_get_elements_and_bonds_the_oriented_pair() {
        let mol = water();
        let atoms_before: Vec<Props> = mol.atoms().map(|(_, a)| a).collect();
        let bonds_before: Vec<_> = mol.bonds().map(|(_, b)| (b.nodes, b.props)).collect();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(
            atom_types(&typed),
            vec![str_value("O"), str_value("H"), str_value("H")]
        );
        assert_eq!(bond_types(&typed), vec![str_value("H-O"), str_value("H-O")]);
        assert!(
            typing.forcefield().styles().is_empty(),
            "{:?}",
            typing.forcefield().styles()
        );
        let atoms_after: Vec<Props> = mol.atoms().map(|(_, a)| a).collect();
        let bonds_after: Vec<_> = mol.bonds().map(|(_, b)| (b.nodes, b.props)).collect();
        assert_eq!(atoms_after, atoms_before);
        assert_eq!(bonds_after, bonds_before);
    }

    #[test]
    fn angle_label_is_the_oriented_element_sequence() {
        let (mut mol, [c1, c2, h]) = cch();
        mol.add_angle(h, c2, c1, props! {}).unwrap();
        let mut typing = Typing::new(ElementTypifier::default());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(bond_types(&typed), vec![str_value("C-C"), str_value("C-H")]);
        assert_eq!(angle_types(&typed), vec![str_value("C-C-H")]);
    }

    #[test]
    fn dihedral_label_is_the_oriented_element_sequence() {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        let c1 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let c2 = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let h = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
            .unwrap();
        mol.add_bond(o, c1, props! {}).unwrap();
        mol.add_bond(c1, c2, props! {}).unwrap();
        mol.add_bond(c2, h, props! {}).unwrap();
        mol.add_dihedral(o, c1, c2, h, props! {}).unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(dihedral_types(&typed), vec![str_value("H-C-C-O")]);
    }

    #[test]
    fn a_graph_without_angles_or_dihedrals_gets_no_such_labels() {
        let (mol, _) = cch();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert_eq!(typed.n_angles(), 0);
        assert_eq!(typed.n_dihedrals(), 0);
        assert_eq!(bond_types(&typed), vec![str_value("C-C"), str_value("C-H")]);
    }

    #[test]
    fn a_graph_with_an_improper_is_refused_naming_impropers() {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        let ids: Vec<NodeId> = (0..3)
            .map(|_| {
                mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                    .unwrap()
            })
            .collect();
        for &h in &ids {
            mol.add_bond(c, h, props! {}).unwrap();
        }
        mol.add_improper(c, ids[0], ids[1], ids[2], props! {})
            .unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let err = typing.typify(&mol).unwrap_err();

        assert!(err.contains("improper"), "{err}");
        assert!(err.contains('1'), "{err}");
    }

    #[test]
    fn a_wildcard_atom_is_refused_naming_its_row() {
        let mut mol = Atomistic::new();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => 0u8 }).unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let err = typing.typify(&mol).unwrap_err();

        assert!(err.contains("wildcard"), "{err}");
        assert!(err.contains("atom 1"), "{err}");
    }

    #[test]
    fn the_atom_label_is_derived_from_the_atomic_number_alone() {
        // No `element` column anywhere: the label is `Element::by_number(Z)`.
        let mut mol = Atomistic::new();
        mol.add_atom(props! { keys::ATOMIC_NUMBER => 17u8 })
            .unwrap();
        let mut typing = Typing::new(ElementTypifier::new());

        let typed = typing.typify(&mol).unwrap();

        assert!(
            typed
                .atoms()
                .all(|(_, a)| a.get_str(keys::ELEMENT).is_none())
        );
        assert_eq!(atom_types(&typed), vec![str_value("Cl")]);
    }
}
