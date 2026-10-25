//! Stereochemistry support for molecular graphs.

use std::collections::HashMap;

use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::MolRsError;
use crate::core::keys;
use crate::core::{NodeId, RelationId};
use crate::op::vec3::{cross, dot, norm, scale, sub};

use super::valence::atomic_number;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// Tetrahedral stereochemistry at an atom.
///
/// `Clockwise` and `CounterClockwise` are the clock sense of the reference
/// order, seen from its first substituent.
/// That is the bond neighbors in bond order when all four are atoms. It is
/// not RDKit `CHI_TETRAHEDRAL_CW`, and it is not the SMILES `@@` glyph: `@`
/// and `@@` are this same clock after the string's own neighbor permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TetrahedralStereo {
    /// Clockwise in the reference order, seen from the first substituent.
    Clockwise,
    /// Counter-clockwise in the reference order, seen from the first substituent.
    CounterClockwise,
    /// No stereo information available or applicable.
    Unspecified,
}

/// One site of the tetrahedral reference order.
///
/// [`TetrahedralSite::Hydrogen`] is the implicit hydrogen of a stated
/// `h_count` of 1, not an atom of the graph. An explicit hydrogen atom is
/// [`TetrahedralSite::Atom`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TetrahedralSite {
    Atom(NodeId),
    Hydrogen,
}

/// Double-bond stereochemistry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondStereo {
    /// E (trans) — high-priority groups on opposite sides.
    E,
    /// Z (cis) — high-priority groups on the same side.
    Z,
    /// Double bond with unspecified stereo.
    Either,
    /// Not a stereo bond (no double bond or not applicable).
    None,
}

// ---------------------------------------------------------------------------
// Chiral volume
// ---------------------------------------------------------------------------

/// Compute the signed scalar triple product (chiral volume) at `center`.
///
/// `neighbor_order` should list the four substituents in the order that
/// determines the sign convention you want to test (typically CIP priority
/// order, lowest last).
///
/// * Positive return → CCW arrangement of n1→n2→n3 when viewed from n4 (S
///   configuration when the substituents are in CIP order).
/// * Negative return → CW arrangement (R configuration).
/// * Zero → the four atoms are coplanar (degenerate).
///
/// Returns `0.0` if any atom lacks `x`/`y`/`z` coordinates.
pub fn chiral_volume(mol: &Atomistic, center: NodeId, neighbor_order: &[NodeId; 4]) -> f64 {
    let pos = |id: NodeId| -> Option<[f64; 3]> {
        let a = mol.get_atom(id).ok()?;
        Some([a.get_f64("x")?, a.get_f64("y")?, a.get_f64("z")?])
    };

    let c = match pos(center) {
        Some(p) => p,
        None => return 0.0,
    };
    let p: Vec<[f64; 3]> = neighbor_order.iter().filter_map(|&id| pos(id)).collect();
    if p.len() < 4 {
        return 0.0;
    }

    // Vectors from center to each neighbour
    let v: Vec<[f64; 3]> = p.iter().map(|q| sub(*q, c)).collect();

    // Scalar triple product of v1, v2, v3  (v4 is the "viewing" direction)
    // volume = v[0] · (v[1] × v[2])
    let v12 = cross(v[1], v[2]);
    dot(v[0], v12)
}

// ---------------------------------------------------------------------------
// Chiral centre detection
// ---------------------------------------------------------------------------

/// Return the atom IDs that are potential tetrahedral stereocentres:
/// atoms with exactly 4 distinct neighbour atom IDs.
///
/// Note: this is a *topological* screen only.  Two neighbours may be
/// constitutionally identical.  CIP rank comparison is outside the scope
/// of this module.
pub fn perceive_chiral_centers(mol: &Atomistic) -> Vec<NodeId> {
    let mut centers = Vec::new();
    for (id, _atom) in mol.atoms() {
        let nbrs: Vec<NodeId> = mol
            .neighbors(id, NeighborKinds::BONDS)
            .map(|(nb, _)| nb)
            .collect();
        if nbrs.len() == 4 {
            // Check all four are distinct
            let mut unique = nbrs.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() == 4 {
                centers.push(id);
            }
        }
    }
    centers
}

// ---------------------------------------------------------------------------
// Stereo assignment from 3D coordinates
// ---------------------------------------------------------------------------

/// The `h_count` an atom states, or `None` when the key is absent.
///
/// Absent is not zero. A bracket atom that omitted the count stores zero; an
/// organic-subset atom stores nothing.
pub(crate) fn stated_hydrogen_count(mol: &Atomistic, id: NodeId) -> Option<i32> {
    mol.get_atom(id)
        .ok()
        .and_then(|atom| atom.get_int(keys::H_COUNT))
}

/// The order a tetrahedral `"CW"` / `"CCW"` is read against.
///
/// Four bond neighbors and a missing or zero `h_count` stand in bond order.
/// Three bond neighbors and `h_count` 1 put the implicit hydrogen first. Any
/// other count, including four neighbors together with a non-zero `h_count`,
/// is not a reference this function names.
pub(crate) fn tetrahedral_reference(mol: &Atomistic, id: NodeId) -> Option<[TetrahedralSite; 4]> {
    let heavy: Vec<NodeId> = mol
        .neighbors(id, NeighborKinds::BONDS)
        .map(|(nb, _)| nb)
        .collect();
    match (heavy.len(), stated_hydrogen_count(mol, id)) {
        (4, None) | (4, Some(0)) => Some([
            TetrahedralSite::Atom(heavy[0]),
            TetrahedralSite::Atom(heavy[1]),
            TetrahedralSite::Atom(heavy[2]),
            TetrahedralSite::Atom(heavy[3]),
        ]),
        (3, Some(1)) => Some([
            TetrahedralSite::Hydrogen,
            TetrahedralSite::Atom(heavy[0]),
            TetrahedralSite::Atom(heavy[1]),
            TetrahedralSite::Atom(heavy[2]),
        ]),
        _ => None,
    }
}

/// The substituent perception and SMILES both treat as the representative.
///
/// Highest atomic number, then the earliest atom in the molecule. One CIP
/// sphere: a wildcard (`Z = 0`) ranks below every element. The tie-break is
/// the atom's own position, not its place in `subs`, so the answer does not
/// follow bond-insertion order.
pub(crate) fn priority_substituent(mol: &Atomistic, subs: &[NodeId]) -> Option<NodeId> {
    subs.iter().copied().max_by_key(|&substituent| {
        let z = mol
            .get_atom(substituent)
            .ok()
            .map(|atom| atomic_number(&atom))
            .unwrap_or(0);
        let order = mol
            .atoms()
            .position(|(id, _)| id == substituent)
            .unwrap_or(usize::MAX);
        (z, std::cmp::Reverse(order))
    })
}

/// Infer tetrahedral chirality for every potential stereocentre from 3-D
/// coordinates.
///
/// The sign of the chiral volume uses the four atoms of the reference order.
/// For a centre whose four neighbors are atoms and
/// whose `h_count` is absent or zero, that is the bond-neighbor order. The
/// label is geometric, not CIP-ranked.
///
/// An implicit hydrogen is not an atom, and perception does not invent
/// coordinates for it, so a centre such as `[C@H]` (three neighbors and
/// `h_count` 1) stays unlabeled. [`perceive_chiral_centers`] only returns
/// atoms with four neighbors, which is the same limit: a stated hydrogen is
/// not a fourth neighbor. Four neighbors plus a non-zero `h_count` is an
/// inconsistent graph and stays unlabeled too.
///
/// Returns a map `NodeId → TetrahedralStereo`. Atoms without 3-D coordinates
/// receive `Unspecified`.
pub fn perceive_tetrahedral_stereo(mol: &Atomistic) -> HashMap<NodeId, TetrahedralStereo> {
    let mut result = HashMap::new();
    for center in perceive_chiral_centers(mol) {
        result.insert(center, tetrahedral_label(mol, center));
    }
    result
}

fn tetrahedral_label(mol: &Atomistic, center: NodeId) -> TetrahedralStereo {
    let Some(reference) = tetrahedral_reference(mol, center) else {
        return TetrahedralStereo::Unspecified;
    };
    let mut nbrs = Vec::with_capacity(4);
    for site in reference {
        match site {
            TetrahedralSite::Atom(id) => nbrs.push(id),
            TetrahedralSite::Hydrogen => return TetrahedralStereo::Unspecified,
        }
    }
    if nbrs.len() != 4 {
        return TetrahedralStereo::Unspecified;
    }
    let volume = chiral_volume(mol, center, &[nbrs[0], nbrs[1], nbrs[2], nbrs[3]]);
    if volume > 1e-9 {
        TetrahedralStereo::CounterClockwise
    } else if volume < -1e-9 {
        TetrahedralStereo::Clockwise
    } else {
        TetrahedralStereo::Unspecified
    }
}

/// Infer E/Z stereochemistry for every double bond from 3-D coordinates.
///
/// A bond A=B is considered to have E/Z stereo if:
/// * It is a double bond (`bond_type == Double`, or no class property and
///   degree rules suggest double).
/// * Both A and B have at least one other neighbour (substituents exist).
///
/// The dihedral angle φ between the highest-atomic-number substituent on A
/// and the highest-atomic-number substituent on B determines the label:
/// * |cos φ| < 0 (φ > 90°) → Z (same side, cis).
/// * |cos φ| > 0 (φ < 90°) → E (opposite sides, trans).
///
/// Returns a map `RelationId → BondStereo`.
pub fn perceive_bond_stereo(mol: &Atomistic) -> HashMap<RelationId, BondStereo> {
    let mut result = HashMap::new();

    for (bid, bond) in mol.bonds() {
        // A genuine double bond. An aromatic ring bond has no E/Z even when its
        // Kekulé phase is double — the ring is planar and the phase arbitrary.
        if BondOrder::from_prop(bond.props.get(keys::BOND_TYPE)) != BondOrder::Double {
            result.insert(bid, BondStereo::None);
            continue;
        }

        let (a, b) = (bond.nodes[0], bond.nodes[1]);

        // Substituents on A (excluding B) and on B (excluding A)
        let subs_a: Vec<NodeId> = mol
            .neighbors(a, NeighborKinds::BONDS)
            .map(|(nb, _)| nb)
            .filter(|&x| x != b)
            .collect();
        let subs_b: Vec<NodeId> = mol
            .neighbors(b, NeighborKinds::BONDS)
            .map(|(nb, _)| nb)
            .filter(|&x| x != a)
            .collect();

        if subs_a.is_empty() || subs_b.is_empty() {
            result.insert(bid, BondStereo::None);
            continue;
        }

        // Highest atomic number, then the earliest atom. See
        // [`priority_substituent`]: the tie-break is what keeps 3-methyl-2-pentene
        // from flipping when the two carbon substituents are inserted in the
        // other order. Still one CIP sphere.
        let Some(sa) = priority_substituent(mol, &subs_a) else {
            result.insert(bid, BondStereo::None);
            continue;
        };
        let Some(sb) = priority_substituent(mol, &subs_b) else {
            result.insert(bid, BondStereo::None);
            continue;
        };

        // Get 3-D positions
        let pos = |id: NodeId| -> Option<[f64; 3]> {
            let atom = mol.get_atom(id).ok()?;
            Some([atom.get_f64("x")?, atom.get_f64("y")?, atom.get_f64("z")?])
        };

        let (pa, pb, psa, psb) = match (pos(a), pos(b), pos(sa), pos(sb)) {
            (Some(pa), Some(pb), Some(psa), Some(psb)) => (pa, pb, psa, psb),
            _ => {
                result.insert(bid, BondStereo::Either);
                continue;
            }
        };

        // Vectors from double-bond axis ends to substituents
        let va = sub(psa, pa); // A → sub_a
        let vb = sub(psb, pb); // B → sub_b
        let ab = sub(pb, pa); // A → B (bond axis)

        // Project va and vb onto the plane perpendicular to ab
        let va_perp = sub(va, scale(ab, dot(va, ab) / dot(ab, ab)));
        let vb_perp = sub(vb, scale(ab, dot(vb, ab) / dot(ab, ab)));

        let cos_angle = dot(va_perp, vb_perp) / (norm(va_perp) * norm(vb_perp) + 1e-15);

        // cos > 0 → same side → Z; cos < 0 → opposite sides → E
        let stereo = if cos_angle > 1e-9 {
            BondStereo::Z
        } else if cos_angle < -1e-9 {
            BondStereo::E
        } else {
            BondStereo::Either
        };
        result.insert(bid, stereo);
    }

    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

use crate::core::NeighborKinds;
/// Atom / bond prop written by [`assign_stereo`]: the perceived descriptor.
use crate::core::keys::STEREO;

/// Perceive stereochemistry from 3-D coordinates and write it onto `mol`, in
/// place.
///
/// The [`perceive_tetrahedral_stereo`] and [`perceive_bond_stereo`] side
/// tables, projected as props. Both are **maps**, and the bond map is
/// *total*: it carries [`BondStereo::None`] for every bond that is not a
/// double bond. That variant is a sentinel meaning "not a stereo bond", not a
/// perceived fact, so it is **skipped** — as is
/// [`TetrahedralStereo::Unspecified`]. A `stereo` prop appears only where a real
/// descriptor was perceived: `"CW"` / `"CCW"` on atoms, `"E"` / `"Z"` /
/// `"either"` on bonds.
///
/// # Storage convention
///
/// Stereochemistry labels may be persisted in atom/bond properties:
/// * atom `"stereo"` → `"CW"` | `"CCW"` | `"unspecified"`
/// * bond `"stereo"` → `"E"` | `"Z"` | `"either"` | `"none"`
///
/// Two of those strings are *sentinels* rather than descriptors: `"unspecified"`
/// ([`TetrahedralStereo::Unspecified`]) means "this atom is not a stereocentre",
/// and `"none"` ([`BondStereo::None`]) means "this bond is not a stereo bond".
/// They record the **absence** of a stereochemical fact, not a perceived one.
/// Accordingly, this function writes **only** the real descriptors — `"CW"` /
/// `"CCW"` on atoms and `"E"` / `"Z"` / `"either"` on bonds — and omits the sentinels entirely, so a `"stereo"`
/// prop is present exactly where stereochemistry was perceived and absent
/// everywhere else. The full four-way vocabulary above still applies to graphs
/// written by other producers (a file reader, or a caller persisting the maps
/// returned by [`perceive_tetrahedral_stereo`] / [`perceive_bond_stereo`], both of
/// which are total and do carry the sentinel variants).
///
/// # Errors
///
/// [`MolRsError`] when a descriptor cannot be written (a column of `mol`
/// holding the key under a foreign dtype); `mol` may then be partly annotated.
pub fn assign_stereo(mol: &mut Atomistic) -> Result<(), MolRsError> {
    let atom_stereo = perceive_tetrahedral_stereo(mol);
    let bond_stereo = perceive_bond_stereo(mol);
    for (id, s) in atom_stereo {
        let label = match s {
            TetrahedralStereo::Clockwise => "CW",
            TetrahedralStereo::CounterClockwise => "CCW",
            TetrahedralStereo::Unspecified => continue,
        };
        mol.set_atom(id, STEREO, label)?;
    }
    for (bid, s) in bond_stereo {
        let label = match s {
            BondStereo::E => "E",
            BondStereo::Z => "Z",
            BondStereo::Either => "either",
            BondStereo::None => continue,
        };
        mol.set_bond(bid, STEREO, label)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondNumber, Element, Props};
    use crate::props;

    /// An atom of element `el` at `(x, y, z)`.
    fn atom_xyz(el: Element, x: f64, y: f64, z: f64) -> Props {
        props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
    }

    fn double() -> Props {
        props! { keys::BOND_TYPE => BondOrder::Double, keys::BOND_NUMBER => BondNumber::Double }
    }

    fn add_double_bond(mol: &mut Atomistic, a: NodeId, b: NodeId) {
        mol.add_bond(a, b, double()).expect("add double bond");
    }

    // --- Chiral volume tests ---

    #[test]
    fn test_chiral_volume_sign() {
        // Four atoms at known positions forming a right-handed tetrahedron.
        // center at origin, neighbours along +x, +y, +z, and -x-y-z.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let n1 = g.add_atom(atom_xyz(Element::H, 1.0, 0.0, 0.0)).unwrap();
        let n2 = g.add_atom(atom_xyz(Element::H, 0.0, 1.0, 0.0)).unwrap();
        let n3 = g.add_atom(atom_xyz(Element::H, 0.0, 0.0, 1.0)).unwrap();
        let n4 = g.add_atom(atom_xyz(Element::H, -1.0, -1.0, -1.0)).unwrap();
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n, props! {}).expect("add bond");
        }
        let vol = chiral_volume(&g, c, &[n1, n2, n3, n4]);
        assert!(vol > 0.0, "expected positive chiral volume, got {}", vol);
    }

    #[test]
    fn test_chiral_volume_opposite_sign() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let n1 = g.add_atom(atom_xyz(Element::H, 1.0, 0.0, 0.0)).unwrap();
        let n2 = g.add_atom(atom_xyz(Element::H, 0.0, 1.0, 0.0)).unwrap();
        let n3 = g.add_atom(atom_xyz(Element::H, 0.0, 0.0, 1.0)).unwrap();
        let n4 = g.add_atom(atom_xyz(Element::H, -1.0, -1.0, -1.0)).unwrap();
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n, props! {}).expect("add bond");
        }
        // Swapping two neighbours flips the sign
        let vol_swapped = chiral_volume(&g, c, &[n2, n1, n3, n4]);
        assert!(vol_swapped < 0.0);
    }

    // --- perceive_chiral_centers ---

    #[test]
    fn test_no_chiral_centers_in_ethane() {
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let c2 = g.add_atom(atom_xyz(Element::C, 1.5, 0.0, 0.0)).unwrap();
        g.add_bond(c1, c2, props! {}).expect("add bond");
        assert!(perceive_chiral_centers(&g).is_empty());
    }

    #[test]
    fn test_4_neighbor_atom_detected_as_center() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        for i in 0..4_usize {
            let h = g
                .add_atom(atom_xyz(Element::H, i as f64, 0.0, 0.0))
                .unwrap();
            g.add_bond(c, h, props! {}).expect("add bond");
        }
        // 4 neighbours, all with distinct IDs → detected
        let centers = perceive_chiral_centers(&g);
        assert_eq!(centers.len(), 1);
        assert_eq!(centers[0], c);
    }

    #[test]
    fn an_implicit_hydrogen_is_not_labeled_from_coordinates() {
        // Three neighbors and `h_count` 1 is a tetrahedral reference whose
        // first site is the implicit hydrogen. Perception does not invent
        // coordinates for that hydrogen, so the centre stays unlabeled.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        g.set_atom(c, keys::H_COUNT, 1).unwrap();
        for (el, x, y, z) in [
            (Element::F, 1.0, 0.0, 0.0),
            (Element::Cl, 0.0, 1.0, 0.0),
            (Element::Br, 0.0, 0.0, 1.0),
        ] {
            let neighbor = g.add_atom(atom_xyz(el, x, y, z)).unwrap();
            g.add_bond(c, neighbor, props! {}).expect("add bond");
        }
        assert!(perceive_chiral_centers(&g).is_empty());
        assert!(!perceive_tetrahedral_stereo(&g).contains_key(&c));
        let reference = tetrahedral_reference(&g, c).expect("three neighbors, one hydrogen");
        assert_eq!(reference[0], TetrahedralSite::Hydrogen);
    }

    // --- perceive_tetrahedral_stereo ---

    #[test]
    fn test_assign_stereo_returns_entry_for_center() {
        let mut g = Atomistic::new();
        let c = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let n1 = g.add_atom(atom_xyz(Element::F, 1.0, 0.0, 0.0)).unwrap();
        let n2 = g.add_atom(atom_xyz(Element::Cl, 0.0, 1.0, 0.0)).unwrap();
        let n3 = g.add_atom(atom_xyz(Element::Br, 0.0, 0.0, 1.0)).unwrap();
        let n4 = g.add_atom(atom_xyz(Element::H, -1.0, -1.0, -1.0)).unwrap();
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n, props! {}).expect("add bond");
        }
        let stereo = perceive_tetrahedral_stereo(&g);
        assert!(stereo.contains_key(&c));
        assert_ne!(stereo[&c], TetrahedralStereo::Unspecified);
    }

    // --- perceive_bond_stereo ---

    #[test]
    fn test_cis_2_butene_is_z() {
        // cis-2-butene: both methyl groups on same side.
        //   CH3      CH3
        //      \    /
        //       C=C
        // Place C1 at origin, C2 at (1.34, 0, 0).
        // Sub on C1 in +y direction, sub on C2 also in +y direction → Z.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let c2 = g.add_atom(atom_xyz(Element::C, 1.34, 0.0, 0.0)).unwrap();
        let sub1 = g.add_atom(atom_xyz(Element::C, -0.5, 1.0, 0.0)).unwrap(); // +y side
        let sub2 = g.add_atom(atom_xyz(Element::C, 1.84, 1.0, 0.0)).unwrap(); // +y side
        add_double_bond(&mut g, c1, c2);
        g.add_bond(c1, sub1, props! {}).expect("add bond");
        g.add_bond(c2, sub2, props! {}).expect("add bond");

        let stereo = perceive_bond_stereo(&g);
        let double_bid = stereo
            .iter()
            .find(|&(_, v)| *v == BondStereo::Z || *v == BondStereo::E)
            .map(|(&k, _)| k);
        assert!(double_bid.is_some(), "no E/Z bond found");
        assert_eq!(stereo[&double_bid.unwrap()], BondStereo::Z);
    }

    #[test]
    fn test_trans_2_butene_is_e() {
        // trans: sub on C1 in +y, sub on C2 in -y direction.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let c2 = g.add_atom(atom_xyz(Element::C, 1.34, 0.0, 0.0)).unwrap();
        let sub1 = g.add_atom(atom_xyz(Element::C, -0.5, 1.0, 0.0)).unwrap(); // +y side
        let sub2 = g.add_atom(atom_xyz(Element::C, 1.84, -1.0, 0.0)).unwrap(); // -y side
        add_double_bond(&mut g, c1, c2);
        g.add_bond(c1, sub1, props! {}).expect("add bond");
        g.add_bond(c2, sub2, props! {}).expect("add bond");

        let stereo = perceive_bond_stereo(&g);
        let double_bid = stereo
            .iter()
            .find(|&(_, v)| *v == BondStereo::E || *v == BondStereo::Z)
            .map(|(&k, _)| k);
        assert!(double_bid.is_some(), "no E/Z bond found");
        assert_eq!(stereo[&double_bid.unwrap()], BondStereo::E);
    }

    #[test]
    fn test_single_bond_has_no_stereo() {
        let mut g = Atomistic::new();
        let a = g.add_atom(atom_xyz(Element::C, 0.0, 0.0, 0.0)).unwrap();
        let b = g.add_atom(atom_xyz(Element::C, 1.5, 0.0, 0.0)).unwrap();
        g.add_bond(a, b, props! {}).expect("add bond"); // default single bond
        let stereo = perceive_bond_stereo(&g);
        let bid = g.bonds().next().unwrap().0;
        assert_eq!(stereo[&bid], BondStereo::None);
    }
    /// The E/Z label must be a function of the geometry, not of bond order.
    ///
    /// 3-methyl-2-pentene: both substituents on C3 are carbon, so the
    /// atomic-number key ties. `max_by_key` keeps the last maximum, so a
    /// ranking that relied on it would flip the label between E and Z when the
    /// two C3 bonds are swapped on identical coordinates.
    #[test]
    fn the_ez_label_does_not_depend_on_bond_insertion_order() {
        let build = |methyl_first: bool| {
            let mut mol = Atomistic::new();
            let c1 = mol.add_atom(atom_xyz(Element::C, -2.5, 0.6, 0.0)).unwrap();
            let c2 = mol.add_atom(atom_xyz(Element::C, -1.2, 0.0, 0.0)).unwrap();
            let c3 = mol.add_atom(atom_xyz(Element::C, 0.0, 0.6, 0.0)).unwrap();
            let me = mol.add_atom(atom_xyz(Element::C, 0.1, 2.1, 0.0)).unwrap();
            let et = mol.add_atom(atom_xyz(Element::C, 1.3, -0.1, 0.0)).unwrap();
            mol.add_bond(c1, c2, props! {}).unwrap();
            let d = mol.add_bond(c2, c3, double()).unwrap();
            if methyl_first {
                mol.add_bond(c3, me, props! {}).unwrap();
                mol.add_bond(c3, et, props! {}).unwrap();
            } else {
                mol.add_bond(c3, et, props! {}).unwrap();
                mol.add_bond(c3, me, props! {}).unwrap();
            }
            let stereo = perceive_bond_stereo(&mol);
            stereo.get(&d).copied()
        };

        assert_eq!(
            build(true),
            build(false),
            "the same molecule with the same coordinates got two different labels"
        );
    }
}
