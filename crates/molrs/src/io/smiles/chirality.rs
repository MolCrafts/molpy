//! Tetrahedral `@` / `@@` against a neighbor order that survives the graph.
//!
//! OpenSMILES defines the marker by the order the substituents are written,
//! looked at from the first toward the centre: `@` is anticlockwise, `@@`
//! clockwise. That order is the preceding atom (when the centre is not the
//! start of the string), then one implicit hydrogen as the first of the
//! remaining list, then ring closures in the order their digits are written,
//! then branches, then the atom that continues the chain.
//!
//! The graph stores the same configuration against a reference order that
//! does not depend on a traversal: the bond neighbors in bond order, with
//! that one implicit hydrogen (`h_count` 1 and three neighbors) placed
//! first. `"CW"` / `"CCW"` on an atom mean clockwise / anticlockwise in
//! that reference order, seen from the first substituent — the same reading
//! [`perceive_tetrahedral_stereo`](crate::perceive::perceive_tetrahedral_stereo)
//! uses for four bond neighbors. Writing SMILES permutes the marker into
//! the order that string actually emits. An odd permutation flips `@` and
//! `@@`; an even one keeps the glyph.

use crate::core::{Atomistic, NodeId};
use crate::line_notation::ast::Chirality;

pub(crate) use crate::perceive::{
    TetrahedralSite as Substituent, tetrahedral_reference as reference_order,
};

/// How many implicit hydrogens the OpenSMILES order inserts at `id`.
///
/// An absent count inserts none, the same as a stored zero. The reference
/// order does not use this: a missing `h_count` stays "not stated" there.
pub(crate) fn implicit_hydrogens(mol: &Atomistic, id: NodeId) -> i32 {
    crate::perceive::stated_hydrogen_count(mol, id).unwrap_or(0)
}

/// OpenSMILES neighbor order for `heavy` written with `has_from` and `h`
/// implicit hydrogens.
///
/// `heavy` is already the from-atom (if any), then ring-closure partners,
/// then branches, then the chain continuation. One implicit hydrogen is
/// inserted immediately after the from-atom, or first when the centre starts
/// the string: it is the first atom of the clockwise accounting.
pub(crate) fn opensmiles_order(
    has_from: bool,
    heavy: &[NodeId],
    h: i32,
) -> Option<[Substituent; 4]> {
    let mut subs: Vec<Substituent> = heavy.iter().copied().map(Substituent::Atom).collect();
    if h == 1 && subs.len() == 3 {
        subs.insert(usize::from(has_from), Substituent::Hydrogen);
    }
    if subs.len() == 4 {
        Some([subs[0], subs[1], subs[2], subs[3]])
    } else {
        None
    }
}

/// The OpenSMILES order a writer emits for `id`.
///
/// `ring_partners` is the order the ring digits are written on `id`, and
/// `children` is the order the branches and the chain continuation are
/// written. `parent` is the from-atom, absent when `id` starts the string.
pub(crate) fn emission_order(
    mol: &Atomistic,
    id: NodeId,
    parent: Option<NodeId>,
    ring_partners: &[NodeId],
    children: &[NodeId],
) -> Option<[Substituent; 4]> {
    let mut heavy = Vec::with_capacity(4);
    if let Some(parent) = parent {
        heavy.push(parent);
    }
    heavy.extend(ring_partners);
    heavy.extend(children);
    opensmiles_order(parent.is_some(), &heavy, implicit_hydrogens(mol, id))
}

/// `written`, which is the clock sense of `from`, restated in the order `to`.
///
/// `None` when `to` is not a permutation of `from`: the two orders do not
/// name the same four substituents, so there is no parity to apply.
pub(crate) fn chirality_in(
    written: Chirality,
    from: &[Substituent; 4],
    to: &[Substituent; 4],
) -> Option<Chirality> {
    let odd = odd_reorder(from, to)?;
    let clockwise = matches!(written, Chirality::Clockwise) ^ odd;
    Some(if clockwise {
        Chirality::Clockwise
    } else {
        Chirality::CounterClockwise
    })
}

/// Whether `to` is an odd permutation of `from`.
fn odd_reorder(from: &[Substituent; 4], to: &[Substituent; 4]) -> Option<bool> {
    let mut seen = [false; 4];
    let mut perm = [0usize; 4];
    for (i, item) in to.iter().enumerate() {
        let j = from
            .iter()
            .enumerate()
            .position(|(k, subst)| *subst == *item && !seen[k])?;
        seen[j] = true;
        perm[i] = j;
    }
    let mut inversions = 0usize;
    for i in 0..4 {
        for j in (i + 1)..4 {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    Some(inversions % 2 == 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::NeighborKinds;
    use crate::core::keys::STEREO;
    use crate::io::smiles::{HydrogenEmit, SmilesEmitOptions, SmilesIr};
    use std::collections::HashSet;

    fn parse(smiles: &str) -> Atomistic {
        SmilesIr::parse(smiles)
            .unwrap_or_else(|err| panic!("{smiles}: {err}"))
            .to_atomistic()
            .unwrap_or_else(|err| panic!("{smiles}: {err}"))
    }

    fn atom_z(mol: &Atomistic, id: NodeId) -> u8 {
        mol.get_atom(id)
            .ok()
            .and_then(|atom| atom.atomic_number())
            .unwrap_or(0)
    }

    /// Clock sense of the one tetrahedral centre, compared after both orders
    /// are ranked by constitution. Raw `"CW"` strings are not comparable
    /// across graphs whose bond orders differ.
    fn configuration_bit(mol: &Atomistic) -> bool {
        let mut centre = None;
        for (id, atom) in mol.atoms() {
            let clockwise = match atom.get_str(STEREO) {
                Some("CW") => true,
                Some("CCW") => false,
                _ => continue,
            };
            assert!(centre.is_none(), "two tetrahedral centres");
            centre = Some((id, clockwise));
        }
        let (id, clockwise) = centre.expect("no tetrahedral centre");
        let reference = reference_order(mol, id)
            .unwrap_or_else(|| panic!("centre {id:?} has no reference order"));
        let mut ranked: Vec<(Vec<Vec<u8>>, Substituent)> = reference
            .iter()
            .copied()
            .map(|subst| (substituent_key(mol, id, subst), subst))
            .collect();
        ranked.sort_by(|left, right| left.0.cmp(&right.0));
        for pair in ranked.windows(2) {
            assert_ne!(
                pair[0].0, pair[1].0,
                "substituents of {id:?} rank the same: {:?}",
                pair[0].0
            );
        }
        let order = [ranked[0].1, ranked[1].1, ranked[2].1, ranked[3].1];
        let odd = odd_reorder(&reference, &order).expect("ranking is not a permutation");
        clockwise ^ odd
    }

    /// Constitution of one substituent, not passing through the centre.
    ///
    /// Implicit hydrogen and an explicit hydrogen bonded only to the centre
    /// both rank as Z = 1. Every other atom is the BFS layers of atomic
    /// numbers around it. Distance to a marker such as fluorine splits the
    /// two ring carbons.
    fn substituent_key(mol: &Atomistic, centre: NodeId, subst: Substituent) -> Vec<Vec<u8>> {
        let Substituent::Atom(start) = subst else {
            return vec![vec![1]];
        };
        let only_the_centre = mol
            .neighbors(start, NeighborKinds::BONDS)
            .all(|(nb, _)| nb == centre);
        if atom_z(mol, start) == 1 && only_the_centre {
            return vec![vec![1]];
        }
        let mut visited = HashSet::new();
        visited.insert(centre);
        visited.insert(start);
        let mut layers = vec![vec![atom_z(mol, start)]];
        let mut frontier = vec![start];
        while !frontier.is_empty() {
            let mut next = Vec::new();
            let mut layer = Vec::new();
            for atom in frontier {
                for (nb, _) in mol.neighbors(atom, NeighborKinds::BONDS) {
                    if visited.insert(nb) {
                        next.push(nb);
                        layer.push(atom_z(mol, nb));
                    }
                }
            }
            if layer.is_empty() {
                break;
            }
            layer.sort_unstable();
            layers.push(layer);
            frontier = next;
        }
        layers
    }

    fn stereo_label(mol: &Atomistic) -> &'static str {
        let mut label = None;
        for (_, atom) in mol.atoms() {
            let stored = match atom.get_str(STEREO) {
                Some("CW") => "CW",
                Some("CCW") => "CCW",
                _ => continue,
            };
            assert!(label.is_none(), "two tetrahedral centres");
            label = Some(stored);
        }
        label.expect("no tetrahedral centre")
    }

    fn write(mol: &Atomistic, opts: &SmilesEmitOptions) -> String {
        SmilesIr::from_atomistic(mol, opts)
            .unwrap_or_else(|err| panic!("{err}"))
            .to_string()
    }

    fn rooted(root: NodeId) -> SmilesEmitOptions {
        SmilesEmitOptions {
            canonical: false,
            include_stereo: true,
            root: Some(root),
            ..SmilesEmitOptions::default()
        }
    }

    fn assert_same_configuration(smiles: &[&str]) {
        let bits: Vec<bool> = smiles
            .iter()
            .map(|src| configuration_bit(&parse(src)))
            .collect();
        assert!(
            bits.windows(2).all(|pair| pair[0] == pair[1]),
            "{smiles:?} -> {bits:?}"
        );
    }

    fn assert_configuration_round_trip(src: &str) {
        let mol = parse(src);
        let want = configuration_bit(&mol);
        for (id, atom) in mol.atoms() {
            if atom.atomic_number() == Some(1) {
                continue;
            }
            let text = write(&mol, &rooted(id));
            let again = parse(&text);
            assert_eq!(
                configuration_bit(&again),
                want,
                "{src} rooted at {id:?} wrote {text}"
            );
        }
        let canonical = SmilesEmitOptions {
            include_stereo: true,
            ..SmilesEmitOptions::default()
        };
        let text = write(&mol, &canonical);
        assert_eq!(
            configuration_bit(&parse(&text)),
            want,
            "{src} canonical wrote {text}"
        );
    }

    #[test]
    fn an_odd_reorder_flips_the_clock() {
        let n = Substituent::Atom(NodeId::from(1));
        let br = Substituent::Atom(NodeId::from(2));
        let o = Substituent::Atom(NodeId::from(3));
        let c = Substituent::Atom(NodeId::from(4));
        let h = Substituent::Hydrogen;
        // `N[C@](Br)(O)C` is anticlockwise on [N, Br, O, C]. Rooting at Br
        // writes [Br, N, O, C], one swap, so the glyph is `@@`.
        assert_eq!(
            chirality_in(Chirality::CounterClockwise, &[n, br, o, c], &[br, n, o, c]),
            Some(Chirality::Clockwise)
        );
        // `N[C@H](O)C` writes [N, H, O, C]. The reference puts H first.
        assert_eq!(
            chirality_in(Chirality::CounterClockwise, &[n, h, o, c], &[h, n, o, c]),
            Some(Chirality::Clockwise)
        );
        assert_eq!(
            opensmiles_order(
                true,
                &[NodeId::from(1), NodeId::from(3), NodeId::from(4)],
                1
            ),
            Some([n, h, o, c])
        );
        assert_eq!(
            opensmiles_order(
                false,
                &[NodeId::from(2), NodeId::from(3), NodeId::from(4)],
                1
            ),
            Some([
                h,
                Substituent::Atom(NodeId::from(2)),
                Substituent::Atom(NodeId::from(3)),
                Substituent::Atom(NodeId::from(4))
            ])
        );
    }

    #[test]
    fn opensmiles_examples_share_one_configuration() {
        let same = [
            "N[C@](Br)(O)C",
            "Br[C@](O)(N)C",
            "O[C@](Br)(C)N",
            "Br[C@](C)(O)N",
            "C[C@](Br)(N)O",
            "Br[C@](N)(C)O",
            "C[C@@](Br)(O)N",
            "Br[C@@](N)(O)C",
            "[C@@](C)(Br)(O)N",
            "[C@@](Br)(N)(O)C",
        ];
        assert_same_configuration(&same);
        let enantiomer = configuration_bit(&parse("N[C@@](Br)(O)C"));
        assert_ne!(enantiomer, configuration_bit(&parse(same[0])));
    }

    #[test]
    fn implicit_hydrogen_matches_the_explicit_hydrogen() {
        let implicit = parse("N[C@H](O)C");
        let explicit = parse("N[C@]([H])(O)C");
        let opposite = parse("N[C@@H](O)C");
        assert_eq!(stereo_label(&implicit), "CW");
        assert_eq!(stereo_label(&explicit), "CCW");
        assert_eq!(stereo_label(&parse("[C@@H](F)(Cl)Br")), "CW");
        assert_eq!(configuration_bit(&implicit), configuration_bit(&explicit));
        assert_ne!(configuration_bit(&implicit), configuration_bit(&opposite));
    }

    #[test]
    fn a_ring_digit_written_first_is_the_same_configuration() {
        let branched = parse("FC1C[C@](Br)(Cl)CCC1");
        let digit_first = parse("[C@]1(Br)(Cl)CCCC(F)C1");
        let opposite = parse("FC1C[C@@](Br)(Cl)CCC1");
        assert_eq!(stereo_label(&branched), "CCW");
        assert_eq!(stereo_label(&digit_first), "CW");
        assert_eq!(
            configuration_bit(&branched),
            configuration_bit(&digit_first)
        );
        assert_ne!(configuration_bit(&branched), configuration_bit(&opposite));
        assert_eq!(
            configuration_bit(&opposite),
            configuration_bit(&parse("[C@@]1(Br)(Cl)CCCC(F)C1"))
        );
    }

    #[test]
    fn an_incomplete_centre_is_not_labeled() {
        // The written substituents are not the reference four, so the marker
        // is not stored. Copying `@` onto that graph would name a clock the
        // substituents do not have.
        for src in ["[C@](F)(Cl)Br", "[C@@H2](F)C", "N[C@]C"] {
            let mol = parse(src);
            for (_, atom) in mol.atoms() {
                assert!(
                    atom.get_str(STEREO).is_none(),
                    "{src} stored {}",
                    atom.get_str(STEREO).unwrap_or("")
                );
            }
        }
    }

    #[test]
    fn re_rooting_rewrites_the_glyph() {
        let bromo = parse("N[C@](Br)(O)C");
        let br = bromo
            .atoms()
            .find(|(_, atom)| atom.atomic_number() == Some(35))
            .unwrap()
            .0;
        assert_eq!(write(&bromo, &rooted(br)), "Br[C@@](N)(O)C");

        let amino = parse("N[C@H](O)C");
        let nitrogen = amino
            .atoms()
            .find(|(_, atom)| atom.atomic_number() == Some(7))
            .unwrap()
            .0;
        assert_eq!(write(&amino, &rooted(nitrogen)), "N[C@H](O)C");
    }

    #[test]
    fn every_root_and_the_canonical_string_keep_the_configuration() {
        for src in [
            "N[C@](Br)(O)C",
            "N[C@@](Br)(O)C",
            "Br[C@@](N)(O)C",
            "[C@@](Br)(N)(O)C",
            "N[C@H](O)C",
            "N[C@@H](O)C",
            "[C@@H](F)(Cl)Br",
            "FC1C[C@](Br)(Cl)CCC1",
            "[C@]1(Br)(Cl)CCCC(F)C1",
            "FC1C[C@@](Br)(Cl)CCC1",
            "N[C@]([H])(O)C",
            "N[C@@]([H])(O)C",
        ] {
            assert_configuration_round_trip(src);
        }
    }

    #[test]
    fn an_omitted_explicit_hydrogen_is_written_as_the_bracket_hydrogen() {
        let explicit = parse("N[C@]([H])(O)C");
        let want = configuration_bit(&explicit);
        assert_eq!(want, configuration_bit(&parse("N[C@H](O)C")));
        let nitrogen = explicit
            .atoms()
            .find(|(_, atom)| atom.atomic_number() == Some(7))
            .unwrap()
            .0;
        assert_eq!(write(&explicit, &rooted(nitrogen)), "N[C@H](O)C");

        let opposite = parse("N[C@@]([H])(O)C");
        let nitrogen = opposite
            .atoms()
            .find(|(_, atom)| atom.atomic_number() == Some(7))
            .unwrap()
            .0;
        assert_eq!(write(&opposite, &rooted(nitrogen)), "N[C@@H](O)C");

        // `AsStored` also skips the hydrogen atom. The stored count is zero;
        // the fold writes `H`, not `H0`.
        let as_stored = SmilesEmitOptions {
            hydrogens: HydrogenEmit::AsStored,
            ..rooted(nitrogen)
        };
        assert_eq!(write(&opposite, &as_stored), "N[C@@H](O)C");

        // Stereo off still counts the omitted protium. The hydrogen atom is
        // not drawn, and the bracket keeps it.
        let silent = write(&explicit, &SmilesEmitOptions::default());
        assert!(!silent.contains('@'), "{silent}");
        assert!(silent.contains("[CH]"), "{silent}");
    }

    #[test]
    fn two_explicit_hydrogens_are_not_folded_into_one() {
        let mol = parse("N[C@]([H])([H])C");
        let nitrogen = mol
            .atoms()
            .find(|(_, atom)| atom.atomic_number() == Some(7))
            .unwrap()
            .0;
        let text = write(&mol, &rooted(nitrogen));
        assert!(!text.contains('@'), "{text}");
        assert!(text.contains("[CH2]"), "{text}");
        let reread = parse("N[C@]C");
        assert!(
            reread
                .atoms()
                .all(|(_, atom)| atom.get_str(STEREO).is_none())
        );
    }

    #[test]
    fn an_explicit_hydrogen_round_trips_when_it_is_written() {
        let mol = parse("N[C@]([H])(O)C");
        let want = configuration_bit(&mol);
        for (id, atom) in mol.atoms() {
            if atom.atomic_number() == Some(1) {
                continue;
            }
            let opts = SmilesEmitOptions {
                canonical: false,
                include_stereo: true,
                hydrogens: HydrogenEmit::ExplicitAll,
                root: Some(id),
                ..SmilesEmitOptions::default()
            };
            let text = write(&mol, &opts);
            let again = parse(&text);
            assert_eq!(
                again.n_atoms(),
                mol.n_atoms(),
                "root {id:?} dropped the explicit H: {text}"
            );
            assert_eq!(configuration_bit(&again), want, "root {id:?} wrote {text}");
        }
    }

    #[test]
    fn the_fragment_walk_stores_the_same_marker() {
        for src in ["[C@@H](F)(Cl)Br", "N[C@H](O)C", "[C@]1(Br)(Cl)CCCC(F)C1"] {
            let plain = parse(src);
            let (fragment, descriptors) = SmilesIr::from_fragment(src)
                .unwrap_or_else(|err| panic!("{src}: {err}"))
                .to_atomistic()
                .unwrap_or_else(|err| panic!("{src}: {err}"));
            assert!(descriptors.is_empty(), "{src}");
            assert_eq!(stereo_label(&fragment), stereo_label(&plain), "{src}");
        }
    }

    #[test]
    fn stereo_off_does_not_write_a_marker() {
        let mol = parse("N[C@](Br)(O)C");
        let text = write(&mol, &SmilesEmitOptions::default());
        assert!(!text.contains('@'), "{text}");
    }
}
