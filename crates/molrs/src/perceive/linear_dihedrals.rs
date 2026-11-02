//! Torsions about a linear centre: the dihedrals whose angle is undefined.
//!
//! When the second or third atom of a dihedral `i-j-k-l` is linear, the planes
//! `i-j-k` and `j-k-l` degenerate and the dihedral angle carries no
//! information; a force field gives such a torsion no barrier. A linear centre
//! is an sp carbon (two connections) outside every 5- and 6-membered ring, or
//! a two-connected cationic nitrogen between two atoms that are not N-H — the
//! rule ByteFF writes as two SMARTS, which this module runs on molpy's own
//! matcher.
//!
//! Reference:
//!     Zheng, T. et al. "Data-driven parametrization of molecular mechanics
//!     force fields for expansive chemical space coverage." Chem. Sci. 2025.
//!     <https://arxiv.org/abs/2408.12817> (``byteff2`` ``match_linear_proper``).

use crate::core::{Atomistic, MolRsError, NodeId};

use super::smarts::{MatchOptions, SmartsPattern};

/// The linear centre at query atom 2, in either SMARTS.
const LINEAR_CENTRES: [&str; 2] = [
    "[*:1]~[#6X2;!r5;!r6:2]~[*:3]~[*:4]",
    "[!$([#7H1]):1]~[#7X2+:2]~[!$([#7H1]):3]~[*:4]",
];

/// Every dihedral `i-j-k-l` of the bond graph whose `j` or `k` is a linear centre.
///
/// A dihedral is listed once, in the orientation whose first atom comes first in
/// the atom table; the same four atoms reversed name the same dihedral.
///
/// # Errors
///
/// Only if a built-in pattern fails to parse, which a test rules out.
pub fn perceive_linear_dihedrals(mol: &Atomistic) -> Result<Vec<[NodeId; 4]>, MolRsError> {
    let rows = mol.atom_table();
    let row = |id: NodeId| rows.row(id).unwrap_or(usize::MAX);
    let mut found: Vec<[NodeId; 4]> = Vec::new();
    for smarts in LINEAR_CENTRES {
        let pattern = SmartsPattern::parse(smarts)?;
        for m in pattern.find(mol, MatchOptions::default()) {
            let a = m.atoms();
            let forward = [a[0], a[1], a[2], a[3]];
            let reversed = [a[3], a[2], a[1], a[0]];
            let canonical = if row(a[0]) <= row(a[3]) {
                forward
            } else {
                reversed
            };
            if !found.contains(&canonical) {
                found.push(canonical);
            }
        }
    }
    found.sort_by_key(|quad| quad.map(row));
    Ok(found)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Element, Props, keys};
    use crate::props;

    fn molecule(
        atoms: &[(Element, i32)],
        bonds: &[(usize, usize, i32)],
    ) -> (Atomistic, Vec<NodeId>) {
        let mut mol = Atomistic::new();
        let ids: Vec<NodeId> = atoms
            .iter()
            .map(|&(el, charge)| {
                let atom: Props = props! {
                    keys::ATOMIC_NUMBER => el, keys::FORMAL_CHARGE => charge, keys::H_COUNT => 0_i32
                };
                mol.add_atom(atom).unwrap()
            })
            .collect();
        for &(i, j, order) in bonds {
            mol.add_bond(ids[i], ids[j], props! { keys::BOND_NUMBER => order })
                .unwrap();
        }
        (mol, ids)
    }

    fn rows(mol: &Atomistic, quads: &[[NodeId; 4]]) -> Vec<[usize; 4]> {
        quads
            .iter()
            .map(|q| q.map(|id| mol.atom_table().row(id).unwrap()))
            .collect()
    }

    #[test]
    fn the_torsions_through_an_alkyne_carbon_are_linear_and_the_rest_are_not() {
        // 1-butyne H3C-CH2-C#CH: C0 C1 C2 C3, H4 on C3, H5-H7 on C0, H8-H9 on C1.
        let el = |e| (e, 0);
        let (mol, _) = molecule(
            &[
                el(Element::C),
                el(Element::C),
                el(Element::C),
                el(Element::C),
                el(Element::H),
                el(Element::H),
                el(Element::H),
                el(Element::H),
                el(Element::H),
                el(Element::H),
            ],
            &[
                (0, 1, 1),
                (1, 2, 1),
                (2, 3, 3),
                (3, 4, 1),
                (0, 5, 1),
                (0, 6, 1),
                (0, 7, 1),
                (1, 8, 1),
                (1, 9, 1),
            ],
        );
        let linear = rows(&mol, &perceive_linear_dihedrals(&mol).unwrap());
        // Through C2: x-C1-C2-C3 (x = 0, 8, 9); through C3: C1-C2-C3-H4.
        assert!(linear.contains(&[0, 1, 2, 3]) && linear.contains(&[1, 2, 3, 4]));
        assert!(linear.contains(&[3, 2, 1, 8]) || linear.contains(&[8, 1, 2, 3]));
        // H5-C0-C1-C2 turns about two sp3 carbons.
        assert!(!linear.iter().any(|q| q[1] == 0 || (q[1] == 1 && q[2] == 0)));
    }

    #[test]
    fn a_two_connected_cation_between_n_h_atoms_is_not_a_linear_centre() {
        // H-N=[N+]=N-H style: N1 (+, two connections) between two N-H nitrogens.
        let (mol, _) = molecule(
            &[
                (Element::N, 0),
                (Element::N, 1),
                (Element::N, 0),
                (Element::H, 0),
                (Element::H, 0),
            ],
            &[(0, 1, 2), (1, 2, 2), (0, 3, 1), (2, 4, 1)],
        );
        assert!(perceive_linear_dihedrals(&mol).unwrap().is_empty());
    }

    #[test]
    fn a_two_connected_cation_between_heavy_atoms_is_a_linear_centre() {
        // Methyl isocyanide CH3-[N+]#[C-]: C0 N1 C2, H3-H5 on C0.
        let (mol, _) = molecule(
            &[
                (Element::C, 0),
                (Element::N, 1),
                (Element::C, -1),
                (Element::H, 0),
                (Element::H, 0),
                (Element::H, 0),
            ],
            &[(0, 1, 1), (1, 2, 3), (0, 3, 1), (0, 4, 1), (0, 5, 1)],
        );
        // No fourth atom past C2, so no dihedral i-N1-C2-l; but H-C0-N1-C2 is the
        // torsion whose third atom (N1) is linear.
        let linear = rows(&mol, &perceive_linear_dihedrals(&mol).unwrap());
        assert_eq!(linear.len(), 3);
        assert!(linear.iter().all(|q| q.contains(&1) && q.contains(&2)));
    }
}
