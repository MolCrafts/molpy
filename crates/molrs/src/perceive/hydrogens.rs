//! Hydrogen addition for molecular graphs.

use std::collections::{HashMap, HashSet};

use super::valence::n_implicit_hydrogens;
use crate::core::Atomistic;
use crate::core::NeighborKinds;
use crate::core::NodeId;
use crate::core::{Props, keys};
use crate::op::vec3::{cross, norm};
use crate::props;
use molrs::core::Element;
use molrs::core::MolRsError;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Return a new [`Atomistic`] with explicit hydrogen atoms added to every
/// heavy atom that has unfilled valence.
///
/// Each heavy atom gets [`n_implicit_hydrogens`] hydrogens: its element's default
/// valences against the sum of its current bond orders, with the formal
/// charge folded into the element. Hydrogen atoms already present
/// (atomic number 1) are not modified, and a wildcard atom (atomic number 0)
/// has no valence model, so it gets none.
///
/// When a heavy atom has `x`/`y`/`z` components, each new H is placed at a
/// standard X–H length along a tetrahedral valence-completing direction
/// (initial geometry only — force fields may refine). When the heavy lacks
/// coordinates, H is added without `x`/`y`/`z` (topology-only path). Every
/// new H carries [`keys::ATOMIC_NUMBER`] 1 and its [`keys::MASS`].
///
/// Starts from [`Clone`] of `mol` (handles preserved on the skeleton); only
/// new H atoms and bonds are appended, so parent angles/dihedrals remain.
///
/// # Errors
///
/// [`MolRsError::Validation`] when an atom carries no valid
/// [`keys::ATOMIC_NUMBER`]; [`MolRsError`] when bonding a fresh hydrogen to
/// its heavy atom fails, which means the graph handed in holds a stale atom
/// handle. An [`Atomistic`] guarantees both away, so neither case is
/// reachable through its public constructors; the `Result` is here so that
/// the invariant is *returned* rather than asserted by a panic on a caller's
/// thread.
pub fn add_hydrogens(mol: &Atomistic) -> Result<Atomistic, MolRsError> {
    let mut new_mol = mol.clone();

    // Mass (amu) of the hydrogens this call appends, read once from the
    // periodic table: a literal is a second home for a number `Element` owns.
    let h_mass = f64::from(Element::H.atomic_mass());

    // Collect (atom_id, its atomic number, n_implicit_h) for all heavy atoms
    // up front so that we don't hold a borrow while mutating.
    let mut additions: Vec<(NodeId, u8, u32)> = Vec::new();
    let mut declared: Vec<NodeId> = Vec::new();
    for (id, atom) in new_mol.atoms() {
        let z = species(id, &atom)?;
        if z == Element::H.z() {
            continue; // existing hydrogens are left alone
        }
        if atom.get(keys::H_COUNT).is_some() {
            declared.push(id);
        }
        // `None` covers the wildcard (Z = 0) and every atom the valence
        // model does not fill: no hydrogens for them.
        match n_implicit_hydrogens(&new_mol, id) {
            Some(n) if n > 0 => additions.push((id, z, n)),
            _ => {}
        }
    }
    // A declared `h_count` counts the hydrogens an atom carries undrawn. Once
    // they are drawn below it is zero: left as it was, the next perception
    // would count every hydrogen twice — once as a bond, once as declared.
    for &id in &declared {
        new_mol.set_atom(id, keys::H_COUNT, 0)?;
    }

    for (heavy_id, heavy_z, n) in additions {
        // Hydrogen positions, when the heavy atom has coordinates to place
        // them from; `None` is the topology-only path.
        let positions: Option<Vec<[f64; 3]>> = new_mol.get_atom(heavy_id)?.position().map(|p| {
            let existing: Vec<[f64; 3]> = new_mol
                .neighbors(heavy_id, NeighborKinds::BONDS)
                .filter_map(|(nb, _)| new_mol.get_atom(nb).ok()?.position())
                .map(|q| unit([q[0] - p[0], q[1] - p[1], q[2] - p[2]]))
                .collect();
            let len = cap_length(heavy_z);
            cap_directions(&existing, n as usize)
                .into_iter()
                .map(|d| [p[0] + len * d[0], p[1] + len * d[1], p[2] + len * d[2]])
                .collect()
        });

        for k in 0..n as usize {
            let h = match &positions {
                Some(pos) => props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::MASS => h_mass,
                    keys::X => pos[k][0],
                    keys::Y => pos[k][1],
                    keys::Z => pos[k][2],
                },
                None => props! {
                    keys::ATOMIC_NUMBER => Element::H,
                    keys::MASS => h_mass,
                },
            };
            let h_id = new_mol.add_atom(h)?;
            // `Atomistic::add_bond` already stamps the bond Single/Single, so
            // there is nothing left to set. Both endpoints exist (one is the
            // atom just added), so the only error `add_bond` has — an unknown
            // endpoint — is a broken invariant; it travels out to the caller
            // rather than being swallowed, which would leave the new hydrogen
            // floating unbonded.
            new_mol.add_bond(heavy_id, h_id, props! {})?;
        }
    }

    Ok(new_mol)
}

// ---------------------------------------------------------------------------
// Initial X–H geometry (port of molpy.core.capping; values are starting
// guesses for downstream minimization, not equilibrium force-field lengths).
// ---------------------------------------------------------------------------

/// The atomic number of atom `id`, or the error naming it when it carries
/// none (a broken [`Atomistic`] invariant, returned rather than asserted).
fn species(id: NodeId, atom: &Props) -> Result<u8, MolRsError> {
    atom.atomic_number().ok_or_else(|| {
        MolRsError::validation(format!(
            "atom {id:?} has no valid '{}' (0..=118); convert an '{}' symbol \
             explicitly with Element::fill_atomic_numbers",
            keys::ATOMIC_NUMBER,
            keys::ELEMENT
        ))
    })
}

/// Cap X–H bond length (Å) keyed by the heavy atom's atomic number; 1.0 Å
/// for any element other than C, N, O and S.
fn cap_length(heavy_z: u8) -> f64 {
    match Element::by_number(heavy_z) {
        Some(Element::C) => 1.09,
        Some(Element::N) => 1.01,
        Some(Element::O) => 0.96,
        Some(Element::S) => 1.34,
        _ => 1.0,
    }
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    if n > 1e-9 {
        [v[0] / n, v[1] / n, v[2] / n]
    } else {
        v
    }
}

fn orthogonal(v: [f64; 3]) -> [f64; 3] {
    let seed = if v[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    unit(cross(v, seed))
}

/// `k` unit directions completing ~sp3 (tetrahedral) coordination.
fn cap_directions(existing: &[[f64; 3]], k: usize) -> Vec<[f64; 3]> {
    let n = existing.len();
    let mut caps: Vec<[f64; 3]> = if n >= 3 {
        let s = [
            existing[0][0] + existing[1][0] + existing[2][0],
            existing[0][1] + existing[1][1] + existing[2][1],
            existing[0][2] + existing[1][2] + existing[2][2],
        ];
        vec![unit([-s[0], -s[1], -s[2]])]
    } else if n == 2 {
        let u1 = existing[0];
        let u2 = existing[1];
        let bisector = unit([-(u1[0] + u2[0]), -(u1[1] + u2[1]), -(u1[2] + u2[2])]);
        let cr = cross(u1, u2);
        let normal = if norm(cr) < 1e-6 {
            orthogonal(u1)
        } else {
            unit(cr)
        };
        let half = 54.75_f64.to_radians();
        let (c, s) = (half.cos(), half.sin());
        vec![
            unit([
                c * bisector[0] + s * normal[0],
                c * bisector[1] + s * normal[1],
                c * bisector[2] + s * normal[2],
            ]),
            unit([
                c * bisector[0] - s * normal[0],
                c * bisector[1] - s * normal[1],
                c * bisector[2] - s * normal[2],
            ]),
        ]
    } else if n == 1 {
        let u = existing[0];
        let e1 = orthogonal(u);
        let e2 = unit(cross(u, e1));
        let theta = 109.47_f64.to_radians();
        let (ct, st) = (theta.cos(), theta.sin());
        [0.0_f64, 120.0, 240.0]
            .into_iter()
            .map(|phi_deg| {
                let phi = phi_deg.to_radians();
                let (cp, sp) = (phi.cos(), phi.sin());
                unit([
                    ct * u[0] + st * (cp * e1[0] + sp * e2[0]),
                    ct * u[1] + st * (cp * e1[1] + sp * e2[1]),
                    ct * u[2] + st * (cp * e1[2] + sp * e2[2]),
                ])
            })
            .collect()
    } else {
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ]
        .into_iter()
        .map(unit)
        .collect()
    };
    caps.truncate(k);
    // If fewer directions than k (shouldn't for tetrahedral cases), pad with +x-ish
    while caps.len() < k {
        caps.push([1.0, 0.0, 0.0]);
    }
    caps
}

/// Return a new [`Atomistic`] with all terminal explicit hydrogen atoms removed.
///
/// # The rule
///
/// An explicit hydrogen is stripped when **both** hold:
///
/// 1. its degree is exactly one, counted over the `bonds` kind alone
///    ([`Atomistic::neighbors`]) — the standard cheminformatics
///    convention for a "non-bridging" H;
/// 2. it is the anchor or the handle of no port ([`Atomistic::port_ids`]).
///
/// # Why the degree is counted over `bonds` only
///
/// [`MolGraph::neighbors`](crate::core::MolGraph::neighbors) is
/// kind-blind: it walks every arity-2 relation on the graph, so a hydrogen
/// that a caller also recorded in some other 2-ary kind reads as degree two
/// and is spared for a reason that has nothing to do with its bonding. The
/// question clause 1 asks is chemical — how many bonds does this hydrogen
/// have — so it is asked of the bond kind.
///
/// # Why a port handle is kept
///
/// A port is `(anchor, handle)`: the handle is a real bonded hydrogen that
/// *also* carries the attachment point where another fragment joins. Removing
/// it would cascade-delete the port row and leave the fragment with no
/// recorded join site, so the handle is kept however few bonds it has.
///
/// The two clauses are independent on purpose. Before they were separated,
/// handles survived only as a side effect of clause 1 being kind-blind: the
/// port relation inflated a handle's neighbour count to two. That made an
/// ordinary repletion hydrogen on a ported anchor and a port handle
/// indistinguishable to anyone reading the code, and tied the survival of
/// every port to the arity of the kind that records it.
///
/// Incident bonds, angles, and dihedrals of a removed hydrogen are
/// cascade-deleted by [`Atomistic::del_atoms`].
///
/// `mol` is never mutated; a clone is returned.
///
/// # Errors
///
/// [`MolRsError::Validation`] when an atom carries no valid
/// [`keys::ATOMIC_NUMBER`] (a broken [`Atomistic`] invariant), and the error of
/// [`Atomistic::get_port`] when a port's stored descriptor is malformed.
/// [`MolRsError`] when removing a hydrogen fails, which means the graph handed
/// in holds a stale atom handle: the handles stripped here were read out of
/// this same graph a moment ago, they are distinct, and removing one node
/// never despawns another. No public constructor of [`Atomistic`] can produce
/// such a graph, so the case is unreachable in practice; the `Result` is here
/// so that the invariant is *returned* rather than asserted by a panic on a
/// caller's thread.
pub fn remove_hydrogens(mol: &Atomistic) -> Result<Atomistic, MolRsError> {
    let mut new_mol = mol.clone();

    // Every atom a port names (anchor or handle), gathered before the first
    // removal.
    let mut port_nodes: HashSet<NodeId> = HashSet::new();
    for pid in new_mol.port_ids().collect::<Vec<_>>() {
        let port = new_mol.get_port(pid)?;
        port_nodes.insert(port.anchor);
        port_nodes.insert(port.handle);
    }

    let mut h_ids: Vec<NodeId> = Vec::new();
    for (id, atom) in new_mol.atoms() {
        if species(id, &atom)? == Element::H.z()
            && !port_nodes.contains(&id)
            && new_mol.neighbors(id, NeighborKinds::BONDS).count() == 1
        {
            h_ids.push(id);
        }
    }

    // A heavy atom that declares its undrawn hydrogens (`h_count`) keeps
    // declaring all of them: the ones removed here become undrawn too, so
    // drawing and removing hydrogens round-trips the count.
    let mut undrawn: HashMap<NodeId, i32> = HashMap::new();
    for &h in &h_ids {
        let (heavy, _) = new_mol
            .neighbors(h, NeighborKinds::BONDS)
            .next()
            .expect("a removed hydrogen has its one bond");
        if new_mol.get_atom(heavy)?.get(keys::H_COUNT).is_some() {
            *undrawn.entry(heavy).or_insert(0) += 1;
        }
    }
    for (heavy, removed) in undrawn {
        let before = new_mol.get_atom(heavy)?.get_int(keys::H_COUNT).unwrap_or(0);
        new_mol.set_atom(heavy, keys::H_COUNT, before + removed)?;
    }

    // `del_atoms`' only failure is an unknown handle, and every handle here is
    // live (see `# Errors`). Returning it keeps a broken node table a value
    // the caller can handle instead of a panic.
    new_mol.del_atoms(&h_ids)?;
    Ok(new_mol)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{BondNumber, BondOrder, PortKind};

    /// An atom of element `el`, with no other props.
    fn atom(el: Element) -> Props {
        props! { keys::ATOMIC_NUMBER => el }
    }

    /// An atom of element `el` carrying formal charge `fc`.
    fn charged(el: Element, fc: impl Into<crate::core::PropValue>) -> Props {
        props! { keys::ATOMIC_NUMBER => el, keys::FORMAL_CHARGE => fc }
    }

    /// An atom of element `el` at `[x, y, z]`.
    fn atom_at(el: Element, [x, y, z]: [f64; 3]) -> Props {
        props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
    }

    fn is_h(a: &Props) -> bool {
        a.atomic_number() == Some(Element::H.z())
    }

    fn bond_with_order(mol: &mut Atomistic, a: NodeId, b: NodeId, order: f64) {
        // The fixture's float order, split into the two facts it encodes:
        // 1.5 is "aromatic", every integer is a localized count.
        let (class, number) = if (order - 1.5).abs() < 1e-6 {
            (BondOrder::Aromatic, BondNumber::Unknown)
        } else {
            let number = BondNumber::from_code(order.round() as u32);
            (number.implied_type(), number)
        };
        mol.add_bond(
            a,
            b,
            props! { keys::BOND_TYPE => class, keys::BOND_NUMBER => number },
        )
        .expect("fixture bond");
    }

    #[test]
    fn test_methane_skeleton() {
        // Isolated C — should get 4 H.
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::C)).unwrap();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // original unchanged
        assert_eq!(g.n_atoms(), 1);
        // result has C + 4H
        assert_eq!(result.n_atoms(), 5);
        assert_eq!(result.n_bonds(), 4);
        let n_h = result.atoms().filter(|(_, a)| is_h(a)).count();
        assert_eq!(n_h, 4);
    }

    #[test]
    fn test_ethane_c_c() {
        // C-C single bond: each C needs 3 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 8); // 2C + 6H
    }

    #[test]
    fn test_ethylene_c_double_c() {
        // C=C double bond: each C needs 2 H.
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 2.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 6); // 2C + 4H
    }

    #[test]
    fn test_benzene_aromatic() {
        // 6-membered ring with bond order 1.5: each C should get 1 H.
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6)
            .map(|_| g.add_atom(atom(Element::C)).unwrap())
            .collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_benzene_kekule() {
        // Kekule benzene: alternating single/double bonds.
        // Each C has bond_order_sum = 1+2 = 3, needs 1 H. Total = 6 H.
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6)
            .map(|_| g.add_atom(atom(Element::C)).unwrap())
            .collect();
        let orders = [2.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], orders[i]);
        }
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        let n_h = result.atoms().filter(|(_, a)| is_h(a)).count();
        assert_eq!(n_h, 6, "Kekule benzene should get 6 H, got {}", n_h);
        assert_eq!(result.n_atoms(), 12); // 6C + 6H
    }

    #[test]
    fn test_ethylene_round_trip_frame() {
        // C=C → to_frame → from_frame → add_hydrogens should give 4H not 6H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 2.0);
        let frame = g.to_frame().expect("a schema-conforming graph converts");
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 6, "C=C round-trip should give 2C + 4H");
    }

    #[test]
    fn test_acetylene_round_trip_frame() {
        // C#C → to_frame → from_frame → add_hydrogens should give 2H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 3.0);
        let frame = g.to_frame().expect("a schema-conforming graph converts");
        let g2 = Atomistic::from_frame(&frame).unwrap();
        let result = add_hydrogens(&g2).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 4, "C#C round-trip should give 2C + 2H");
    }

    #[test]
    fn test_water() {
        // Isolated O → 2 H
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::O)).unwrap();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 3);
    }

    /// A wildcard atom (`atomic_number` 0) names no element, so it has no
    /// valence model and takes no hydrogens — it is never read as carbon.
    #[test]
    fn a_wildcard_atom_takes_no_hydrogens() {
        let mut g = Atomistic::new();
        let star = g.add_atom(props! { keys::ATOMIC_NUMBER => 0_u8 }).unwrap();
        assert_eq!(n_implicit_hydrogens(&g, star), None);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 1);
    }

    /// `remove_hydrogens` finds hydrogens by atomic number, so it strips the
    /// ones `add_hydrogens` wrote (which carry no element symbol).
    #[test]
    fn remove_hydrogens_strips_what_add_hydrogens_added() {
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::O)).unwrap();
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert!(
            with_h
                .atoms()
                .all(|(_, a)| a.get_str(keys::ELEMENT).is_none())
        );
        let stripped =
            remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 1);
    }

    #[test]
    fn test_ammonia_like() {
        // N with 1 bond → 2 H  (valence 3)
        let mut g = Atomistic::new();
        let n = g.add_atom(atom(Element::N)).unwrap();
        let c = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, n, c, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // N gets 2H, C gets 3H, total = 2C + 2H(on N) + 3H(on C) = 2+5 = 7
        assert_eq!(result.n_atoms(), 7);
    }

    #[test]
    fn test_nh4_plus() {
        // NH4+: formal_charge=1 on N → needs 4 H
        let mut g = Atomistic::new();
        let n = g.add_atom(charged(Element::N, 1.0_f64)).unwrap();
        let count = n_implicit_hydrogens(&g, n).unwrap();
        assert_eq!(count, 4);
    }

    /// Build a single charged heavy atom (no heavy neighbours) and return its
    /// implicit-H count.
    fn charged_atom_h(el: Element, fc: f64) -> u32 {
        let mut g = Atomistic::new();
        let id = g.add_atom(charged(el, fc)).unwrap();
        n_implicit_hydrogens(&g, id).unwrap_or(0)
    }

    #[test]
    fn charged_single_atoms_take_their_valence_hydrogens() {
        // Charged single-heavy-atom species (the cases a plain
        // bond_order_sum - formal_charge rule gets wrong for group-13/14):
        assert_eq!(charged_atom_h(Element::C, 1.0), 3, "[CH3+] -> 3 H");
        assert_eq!(charged_atom_h(Element::C, -1.0), 3, "[CH3-] -> 3 H");
        assert_eq!(charged_atom_h(Element::B, -1.0), 4, "[BH4-] -> 4 H");
        assert_eq!(charged_atom_h(Element::N, 1.0), 4, "[NH4+] -> 4 H");
        assert_eq!(charged_atom_h(Element::O, -1.0), 1, "[OH-] -> 1 H");
        assert_eq!(charged_atom_h(Element::N, -1.0), 2, "[NH2-] -> 2 H");

        // Neutral references (unchanged by the fix):
        assert_eq!(charged_atom_h(Element::C, 0.0), 4, "methane C -> 4 H");
        assert_eq!(charged_atom_h(Element::O, 0.0), 2, "water O -> 2 H");
        assert_eq!(charged_atom_h(Element::N, 0.0), 3, "ammonia N -> 3 H");
    }

    /// Like `charged_atom_h` but stores `formal_charge` as the canonical
    /// **integer** column (`PropValue::Int`) — what the parsers and the i32-typed
    /// graph schema actually emit. `n_implicit_hydrogens` must read an Int
    /// charge: reading it as F64 only would treat every charged atom as
    /// neutral — e.g. protonating the sulfonimide [N-] in TFSI/ANI and
    /// breaking antechamber's charge balance.
    fn charged_atom_h_int(el: Element, fc: i32) -> u32 {
        let mut g = Atomistic::new();
        // i32 == type alias `I` → PropValue::Int
        let id = g.add_atom(charged(el, fc)).unwrap();
        n_implicit_hydrogens(&g, id).unwrap_or(0)
    }

    #[test]
    fn test_int_formal_charge_parity() {
        // Same expectations as `charged_single_atoms_take_their_valence_hydrogens`, but the
        // charge is an Int prop (the real on-graph representation), not f64.
        assert_eq!(
            charged_atom_h_int(Element::N, -1),
            2,
            "[NH2-] (int fc) -> 2 H"
        );
        assert_eq!(
            charged_atom_h_int(Element::N, 1),
            4,
            "[NH4+] (int fc) -> 4 H"
        );
        assert_eq!(
            charged_atom_h_int(Element::O, -1),
            1,
            "[OH-] (int fc) -> 1 H"
        );
        assert_eq!(
            charged_atom_h_int(Element::C, 1),
            3,
            "[CH3+] (int fc) -> 3 H"
        );
        assert_eq!(charged_atom_h_int(Element::N, 0), 3, "neutral N -> 3 H");
    }

    #[test]
    fn test_sulfonimide_anion_not_protonated() {
        // The exact TFSI/ANI failure: a deprotonated sulfonimide N (two single
        // bonds, int formal_charge -1) must add NO hydrogen. Missing the Int
        // charge would read N as neutral (valence 3, demand 2) → 1 H.
        let mut g = Atomistic::new();
        let n = g.add_atom(charged(Element::N, -1_i32)).unwrap();
        let s1 = g.add_atom(atom(Element::S)).unwrap();
        let s2 = g.add_atom(atom(Element::S)).unwrap();
        bond_with_order(&mut g, n, s1, 1.0);
        bond_with_order(&mut g, n, s2, 1.0);
        assert_eq!(h_at(&g, n), 0, "sulfonimide [N-] with two bonds -> 0 H");
    }

    /// Helper: implicit-H on `atom_id` of a built graph.
    fn h_at(g: &Atomistic, id: NodeId) -> u32 {
        n_implicit_hydrogens(g, id).unwrap_or(0)
    }

    #[test]
    fn bonded_atoms_take_their_remaining_valence_hydrogens() {
        // ethane CC: each C has bos 1 -> 3 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 1.0);
        assert_eq!(h_at(&g, c1), 3, "ethane C -> 3 H");
        assert_eq!(h_at(&g, c2), 3, "ethane C -> 3 H");

        // ethylene C=C: each C has bos 2 -> 2 H
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 2.0);
        assert_eq!(h_at(&g, c1), 2, "ethylene C -> 2 H");

        // benzene (aromatic, bos 1.5+1.5=3): each C -> 1 H
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..6)
            .map(|_| g.add_atom(atom(Element::C)).unwrap())
            .collect();
        for i in 0..6 {
            bond_with_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
        }
        assert_eq!(h_at(&g, ids[0]), 1, "benzene C -> 1 H");

        // acetate CC(=O)[O-]: methyl C -> 3, carbonyl C -> 0,
        // carbonyl O (=O) -> 0, [O-] (single bond, fc -1) -> 0
        let mut g = Atomistic::new();
        let c_me = g.add_atom(atom(Element::C)).unwrap();
        let c_carb = g.add_atom(atom(Element::C)).unwrap();
        let o_dbl = g.add_atom(atom(Element::O)).unwrap();
        let o_minus = g.add_atom(charged(Element::O, -1.0_f64)).unwrap();
        bond_with_order(&mut g, c_me, c_carb, 1.0);
        bond_with_order(&mut g, c_carb, o_dbl, 2.0);
        bond_with_order(&mut g, c_carb, o_minus, 1.0);
        assert_eq!(h_at(&g, c_me), 3, "acetate methyl C -> 3 H");
        assert_eq!(h_at(&g, c_carb), 0, "acetate carbonyl C -> 0 H");
        assert_eq!(h_at(&g, o_dbl), 0, "acetate =O -> 0 H");
        assert_eq!(h_at(&g, o_minus), 0, "acetate [O-] -> 0 H");
    }

    #[test]
    fn test_no_double_h_on_existing_hydrogen() {
        // Existing H atoms should not get more H added.
        let mut g = Atomistic::new();
        let c = g.add_atom(atom(Element::C)).unwrap();
        let h = g.add_atom(atom(Element::H)).unwrap();
        bond_with_order(&mut g, c, h, 1.0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        // C had 1 bond, needs 3 more H; H should remain unchanged
        let n_h = result.atoms().filter(|(_, a)| is_h(a)).count();
        assert_eq!(n_h, 4); // 1 original + 3 new
    }

    // ── remove_hydrogens tests ──────────────────────────────────────────────

    #[test]
    fn test_remove_hydrogens_methane() {
        // C + 4H → remove → 1 atom (C only), 0 bonds
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::C)).unwrap();
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), 5);
        let stripped =
            remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
    }

    #[test]
    fn test_remove_hydrogens_ethane() {
        // 2C + 6H → remove → 2 atoms, 1 bond (C-C preserved)
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 1.0);
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), 8);
        let stripped =
            remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_immutable() {
        // Original graph must remain unchanged after remove_hydrogens
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::C)).unwrap();
        let with_h = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        let before = with_h.n_atoms();
        remove_hydrogens(&with_h).expect("stripping succeeds on a well-formed graph");
        assert_eq!(with_h.n_atoms(), before);
    }

    #[test]
    fn test_remove_hydrogens_no_h_present() {
        // C=C without any H → unchanged
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let c2 = g.add_atom(atom(Element::C)).unwrap();
        bond_with_order(&mut g, c1, c2, 2.0);
        let stripped = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 2);
        assert_eq!(stripped.n_bonds(), 1);
    }

    #[test]
    fn test_remove_hydrogens_cascades_angles() {
        // Build C with H and an angle involving H, then remove H → angle gone
        let mut g = Atomistic::new();
        let c = g.add_atom(atom(Element::C)).unwrap();
        let h1 = g.add_atom(atom(Element::H)).unwrap();
        let h2 = g.add_atom(atom(Element::H)).unwrap();
        bond_with_order(&mut g, c, h1, 1.0);
        bond_with_order(&mut g, c, h2, 1.0);
        g.add_angle(h1, c, h2, props! {}).expect("add angle");
        assert_eq!(g.n_angles(), 1);
        let stripped = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");
        assert_eq!(stripped.n_atoms(), 1);
        assert_eq!(stripped.n_bonds(), 0);
        assert_eq!(stripped.n_angles(), 0);
    }

    #[test]
    fn test_add_hydrogens_places_coords_when_heavy_has_xyz() {
        let mut g = Atomistic::new();
        g.add_atom(atom_at(Element::C, [0.0, 0.0, 0.0])).unwrap();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert_eq!(result.n_atoms(), 5);
        let mut n_h = 0;
        for (id, a) in result.atoms() {
            if !is_h(&a) {
                continue;
            }
            n_h += 1;
            let [x, y, z] = a.position().expect("H must have x/y/z");
            let dist = (x * x + y * y + z * z).sqrt();
            assert!(
                (dist - 1.09).abs() < 0.02,
                "C–H distance {dist} for H {id:?}"
            );
        }
        assert_eq!(n_h, 4);
    }

    #[test]
    fn test_add_hydrogens_no_xyz_when_heavy_lacks_coords() {
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::C)).unwrap();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        for (_, a) in result.atoms() {
            if is_h(&a) {
                assert!(a.get_f64(keys::X).is_none());
            }
        }
    }

    #[test]
    fn test_add_hydrogens_preserves_parent_angles() {
        let mut g = Atomistic::new();
        let c1 = g.add_atom(atom_at(Element::C, [0.0, 0.0, 0.0])).unwrap();
        let c2 = g.add_atom(atom_at(Element::C, [1.5, 0.0, 0.0])).unwrap();
        let c3 = g.add_atom(atom_at(Element::C, [3.0, 0.0, 0.0])).unwrap();
        bond_with_order(&mut g, c1, c2, 1.0);
        bond_with_order(&mut g, c2, c3, 1.0);
        g.generate_topology(true, false, false, false).unwrap();
        let n_ang = g.n_angles();
        assert!(n_ang > 0);
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");
        assert!(result.n_angles() >= n_ang);
    }

    #[test]
    fn test_explicit_h_count_is_authoritative() {
        // A declared H count (SMILES bracket atom) is exact — do not top it up
        // to the element's default valence.
        let mut g = Atomistic::new();
        let id = g
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C, keys::H_COUNT => 2 })
            .unwrap();
        assert_eq!(n_implicit_hydrogens(&g, id), Some(2));
    }

    /// An added hydrogen's mass is the element's mass, read from the periodic
    /// table rather than written out as a literal, so the two can never drift.
    #[test]
    fn added_hydrogen_carries_the_element_mass() {
        let mut g = Atomistic::new();
        g.add_atom(atom(Element::C)).unwrap();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");

        let expected = f64::from(Element::H.atomic_mass());
        let mut checked = 0;
        for (_, a) in result.atoms() {
            if is_h(&a) {
                assert_eq!(a.get_f64(keys::MASS), Some(expected));
                checked += 1;
            }
        }
        assert_eq!(checked, 4, "methane's four hydrogens were all checked");
    }

    // ---- ports ride along untouched ---------------------------------------

    #[test]
    fn add_hydrogens_keeps_the_ports_kind_and_caps_a_ported_c_c_o() {
        // C0-C1-O2 with two handle hydrogens: H3 on C0, H4 on O2. Each handle
        // is a real bonded H that is *additionally* recorded as a 2-ary `ports`
        // relation `[anchor, handle]`. No `h_count` / `formal_charge` is set,
        // so the valence model runs.
        //
        // Hand-derived atom count (every bond is single, and `valence_demand`
        // bills each bond at least 1, so the result does not depend on the
        // bond class):
        //   C0: C1 + H3 = 2 bonds, valence 4 -> 2 new H
        //   C1: C0 + O2 = 2 bonds, valence 4 -> 2 new H
        //   O2: C1 + H4 = 2 bonds, valence 2 -> 0 new H
        //   total = 3 heavy + 2 handles + 4 added = 9 atoms
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom(Element::C)).unwrap();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let o2 = g.add_atom(atom(Element::O)).unwrap();
        let h3 = g.add_atom(atom(Element::H)).unwrap();
        let h4 = g.add_atom(atom(Element::H)).unwrap();
        bond_with_order(&mut g, c0, c1, 1.0);
        bond_with_order(&mut g, c1, o2, 1.0);
        bond_with_order(&mut g, c0, h3, 1.0);
        bond_with_order(&mut g, o2, h4, 1.0);

        g.add_port(c0, h3, PortKind::Symmetric, "", BondNumber::Single)
            .expect("H3 is bonded to C0");
        g.add_port(o2, h4, PortKind::Symmetric, "", BondNumber::Single)
            .expect("H4 is bonded to O2");

        let before: Vec<NodeId> = g.atoms().map(|(id, _)| id).collect();
        let result = add_hydrogens(&g).expect("repletion succeeds on a well-formed graph");

        assert_eq!(result.n_atoms(), 9, "3 heavy + 2 handles + 4 added");

        assert_eq!(result.n_ports(), 2, "no added H bond was written as a port");

        let mut added = 0;
        for (id, a) in result.atoms() {
            if before.contains(&id) {
                continue;
            }
            assert_eq!(
                a.atomic_number(),
                Some(Element::H.z()),
                "add_hydrogens only appends hydrogens"
            );
            assert_eq!(
                result.neighbors(id, NeighborKinds::BONDS).count(),
                1,
                "each added H carries exactly one bond"
            );
            added += 1;
        }
        assert_eq!(added, 4, "two on C0, two on C1, none on the hydroxyl O");
    }

    /// The rule: degree is counted over the `bonds` kind **only**, and an H
    /// that is the handle of a port is kept whatever that degree says.
    ///
    /// C0-C1-O2 with two port handles (H3 on C0, H4 on O2 — real bonded H
    /// additionally recorded as 2-ary `ports` relations) and two plain
    /// repletion hydrogens (H5 on C0, H6 on C1). H5 shares its anchor with the
    /// handle H3, so the fixture separates the two halves of the rule: a
    /// kind-blind degree keeps H5 out of the removal set only by accident of
    /// the port relation, and a bonds-only degree without the port exemption
    /// strips H3 and H4 along with it. Only "bonds-only degree **plus** port
    /// exemption" leaves exactly the two handles.
    #[test]
    fn remove_hydrogens_keeps_port_handles_and_strips_a_plain_h_on_the_same_anchor() {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom(Element::C)).unwrap();
        let c1 = g.add_atom(atom(Element::C)).unwrap();
        let o2 = g.add_atom(atom(Element::O)).unwrap();
        let h3 = g.add_atom(atom(Element::H)).unwrap();
        let h4 = g.add_atom(atom(Element::H)).unwrap();
        let h5 = g.add_atom(atom(Element::H)).unwrap();
        let h6 = g.add_atom(atom(Element::H)).unwrap();
        bond_with_order(&mut g, c0, c1, 1.0);
        bond_with_order(&mut g, c1, o2, 1.0);
        bond_with_order(&mut g, c0, h3, 1.0);
        bond_with_order(&mut g, o2, h4, 1.0);
        bond_with_order(&mut g, c0, h5, 1.0);
        bond_with_order(&mut g, c1, h6, 1.0);

        g.add_port(c0, h3, PortKind::Symmetric, "", BondNumber::Single)
            .expect("H3 is bonded to C0");
        g.add_port(o2, h4, PortKind::Symmetric, "", BondNumber::Single)
            .expect("H4 is bonded to O2");

        let result = remove_hydrogens(&g).expect("stripping succeeds on a well-formed graph");

        assert_eq!(g.n_atoms(), 7, "the input graph is never mutated");
        assert_eq!(result.n_atoms(), 5, "3 heavy + 2 handles");
        assert_eq!(
            result.n_bonds(),
            4,
            "C0-C1, C1-O2 and the two handle bonds survive"
        );
        let n_h = result.atoms().filter(|(_, a)| is_h(a)).count();
        assert_eq!(n_h, 2, "both handles kept, both plain hydrogens removed");

        assert_eq!(
            result.n_ports(),
            2,
            "no port row was cascade-deleted with a removed handle"
        );
    }

    /// A port handle is a terminal hydrogen the pass must *not* remove. That
    /// exemption is a data condition, not a failure: the call returns `Ok`
    /// and the handle is still there.
    #[test]
    fn remove_hydrogens_returns_ok_when_a_port_handle_is_bonded_to_a_heavy_atom() {
        let mut g = Atomistic::new();
        let c0 = g.add_atom(atom(Element::C)).unwrap();
        let h1 = g.add_atom(atom(Element::H)).unwrap();
        bond_with_order(&mut g, c0, h1, 1.0);

        g.add_port(c0, h1, PortKind::Symmetric, "", BondNumber::Single)
            .expect("H1 is bonded to C0");

        let result = remove_hydrogens(&g).expect("an exempt handle is not an error");
        assert_eq!(result.n_atoms(), 2, "the port handle is kept");
    }

    #[test]
    fn n_implicit_hydrogens_on_a_long_alkane_reads_only_incident_bonds() {
        // Guards the O(degree) incident-bond read in `valence_demand`: on a
        // 2000-carbon chain the middle carbon has exactly two C-C bonds out of
        // 1999, so it must see a bond-order sum of 2 and take 2 H; each end
        // carbon sees one bond and takes 3.
        const N: usize = 2000;
        let mut g = Atomistic::new();
        let ids: Vec<NodeId> = (0..N)
            .map(|_| g.add_atom(atom(Element::C)).unwrap())
            .collect();
        for pair in ids.windows(2) {
            bond_with_order(&mut g, pair[0], pair[1], 1.0);
        }
        assert_eq!(g.n_bonds(), N - 1);

        assert_eq!(n_implicit_hydrogens(&g, ids[N / 2]), Some(2));
        assert_eq!(n_implicit_hydrogens(&g, ids[0]), Some(3));
        assert_eq!(n_implicit_hydrogens(&g, ids[N - 1]), Some(3));
    }
}

// ---------------------------------------------------------------------------
// Molecular-formula tests over real SMILES input.
//
// These are the end-to-end gate on the aromatic valence model: a hand-built
// graph can be given whatever bond orders make the arithmetic work, so only
// notation-driven fixtures can catch a parser that mis-declares aromaticity.
// ---------------------------------------------------------------------------
#[cfg(all(test, feature = "smiles"))]
mod smiles_formula_tests {
    use super::*;
    use crate::io::smiles::SmilesIr;
    use std::collections::BTreeMap;

    /// Element → count of the hydrogen-completed molecule.
    fn formula(smiles: &str) -> BTreeMap<String, usize> {
        let ir = SmilesIr::parse(smiles).unwrap_or_else(|e| panic!("parse {smiles:?}: {e}"));
        let mol = ir
            .to_atomistic()
            .unwrap_or_else(|e| panic!("to_atomistic {smiles:?}: {e}"));
        let with_h = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for (_, atom) in with_h.atoms() {
            let z = atom.atomic_number().expect("an atomistic atom has a Z");
            let el = Element::by_number(z).expect("no wildcard in these fixtures");
            *counts.entry(el.symbol().to_owned()).or_default() += 1;
        }
        counts
    }

    fn assert_formula(smiles: &str, expected: &[(&str, usize)]) {
        let got = formula(smiles);
        let want: BTreeMap<String, usize> = expected
            .iter()
            .map(|(s, n)| ((*s).to_owned(), *n))
            .collect();
        assert_eq!(got, want, "formula of {smiles:?}");
    }

    #[test]
    fn test_aspirin_formula() {
        // C9H8O4 — 21 atoms.
        assert_formula("CC(=O)Oc1ccccc1C(=O)O", &[("C", 9), ("H", 8), ("O", 4)]);
    }

    #[test]
    fn test_benzene_formula() {
        assert_formula("c1ccccc1", &[("C", 6), ("H", 6)]);
    }

    #[test]
    fn test_toluene_formula() {
        assert_formula("Cc1ccccc1", &[("C", 7), ("H", 8)]);
    }

    #[test]
    fn test_naphthalene_formula() {
        // Fusion carbons carry three aromatic bonds and take no hydrogen.
        assert_formula("c1ccc2ccccc2c1", &[("C", 10), ("H", 8)]);
    }

    #[test]
    fn test_pyridine_formula() {
        // One-electron donor N: three ring σ+π valences, no N–H.
        assert_formula("c1ccncc1", &[("C", 5), ("H", 5), ("N", 1)]);
    }

    #[test]
    fn test_pyrrole_formula() {
        // Lone-pair donor N — the H is declared by the bracket and must survive.
        assert_formula("c1cc[nH]c1", &[("C", 4), ("H", 5), ("N", 1)]);
    }

    #[test]
    fn test_furan_formula() {
        assert_formula("c1ccoc1", &[("C", 4), ("H", 4), ("O", 1)]);
    }

    #[test]
    fn test_thiophene_formula() {
        // S has valences [2,4,6]: the lone-pair donor must not reach for 4.
        assert_formula("c1ccsc1", &[("C", 4), ("H", 4), ("S", 1)]);
    }

    #[test]
    fn test_caffeine_formula() {
        assert_formula(
            "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
            &[("C", 8), ("H", 10), ("N", 4), ("O", 2)],
        );
    }

    #[test]
    fn test_ethanol_formula() {
        assert_formula("CCO", &[("C", 2), ("H", 6), ("O", 1)]);
    }

    #[test]
    fn test_acetate_anion_formula() {
        assert_formula("CC(=O)[O-]", &[("C", 2), ("H", 3), ("O", 2)]);
    }

    #[test]
    fn test_biphenyl_formula() {
        // The explicit single bond between the two rings is not aromatic.
        assert_formula("c1ccccc1-c1ccccc1", &[("C", 12), ("H", 10)]);
    }

    #[test]
    fn test_indole_formula() {
        assert_formula("c1ccc2[nH]ccc2c1", &[("C", 8), ("H", 7), ("N", 1)]);
    }

    /// A `Frame` round trip must not invent hydrogen counts.
    ///
    /// `CCO` is organic subset, so no atom declares an `h_count` and
    /// repletion is free to read the valence model. A column that spells an
    /// unset cell as the default `0` turns that freedom into a declared
    /// "this atom has no hydrogens", and the round-tripped molecule comes
    /// back bare. Ethanol is C2H6O either way.
    #[test]
    fn a_frame_round_trip_keeps_the_repletion_count_of_an_organic_subset_smiles() {
        let ir = SmilesIr::parse("CCO").expect("CCO parses");
        let mol = ir
            .to_atomistic()
            .expect("CCO converts to an atomistic graph");
        let direct = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        assert_eq!(direct.n_atoms(), 9, "C2H6O is nine atoms");

        let read_back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        assert_eq!(
            add_hydrogens(&read_back)
                .expect("repletion succeeds on a well-formed graph")
                .n_atoms(),
            9
        );
    }

    /// The same rule where the column really is partial: `[CH3]` declares
    /// three hydrogens and the plain `C` declares none, so `h_count` is set
    /// on one atom of two. Writing the unset cell as `0` tells repletion the
    /// plain carbon is already saturated, and ethane comes back as C2H3.
    #[test]
    fn a_frame_round_trip_keeps_an_undeclared_h_count_undeclared() {
        let ir = SmilesIr::parse("C[CH3]").expect("C[CH3] parses");
        let mol = ir
            .to_atomistic()
            .expect("C[CH3] converts to an atomistic graph");
        let direct = add_hydrogens(&mol).expect("repletion succeeds on a well-formed graph");
        assert_eq!(direct.n_atoms(), 8, "ethane is 2 C + 6 H");

        let read_back =
            Atomistic::from_frame(&mol.to_frame().expect("a schema-conforming graph converts"))
                .expect("an atomistic frame reads back");
        assert_eq!(
            add_hydrogens(&read_back)
                .expect("repletion succeeds on a well-formed graph")
                .n_atoms(),
            8
        );
    }

    /// A bracket atom's declared hydrogens are drawn once: the declared
    /// count drops to zero, so drawing again adds nothing, and removing the
    /// drawn hydrogens declares them again.
    #[test]
    fn declared_hydrogens_are_drawn_once_and_round_trip() {
        let pyrrole = crate::io::smiles::SmilesIr::parse("c1cc[nH]c1")
            .expect("parse")
            .to_atomistic()
            .expect("to_atomistic");
        let implicit = |mol: &Atomistic| -> Vec<u32> {
            mol.atom_ids()
                .filter_map(|id| n_implicit_hydrogens(mol, id))
                .collect()
        };

        let drawn = add_hydrogens(&pyrrole).unwrap();
        assert_eq!(drawn.n_atoms(), 10, "C4H4NH: five hydrogens drawn");
        assert!(implicit(&drawn).iter().all(|&n| n == 0));
        assert_eq!(add_hydrogens(&drawn).unwrap().n_atoms(), 10);

        let stripped = remove_hydrogens(&drawn).unwrap();
        assert_eq!(stripped.n_atoms(), 5);
        assert_eq!(implicit(&stripped), implicit(&pyrrole));
    }
}
