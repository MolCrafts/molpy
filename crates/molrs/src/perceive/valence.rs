//! The valence model: how many bonds an atom can carry, and so how many
//! hydrogens a graph implies but does not draw.
//!
//! One home for the facts every perception reads about valence — an
//! element's default valence ([`default_valence`]) and an atom's implicit
//! hydrogen count ([`n_implicit_hydrogens`]) — following RDKit's
//! `Atom::calcImplicitValence`. The formal charge both fold in is read by
//! [`Props::formal_charge`](crate::core::Props::formal_charge), which `core`
//! owns because the SMILES writer reads it too.
//!
//! The species every perception reads is the atom's atomic number
//! ([`atomic_number`]); the element symbol is never consulted.

use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::Element;
use crate::core::NeighborKinds;
use crate::core::NodeId;
use crate::core::Props;
use crate::core::keys;

/// The atomic number of `atom`, an atom of an [`Atomistic`] — `0` for the
/// wildcard `*`, which names no element and which every caller handles
/// explicitly.
///
/// # Panics
///
/// When `atom` carries no valid [`keys::ATOMIC_NUMBER`](crate::core::keys::ATOMIC_NUMBER).
/// An `Atomistic` guarantees one (0..=118) on every atom, so this is a broken
/// invariant, never a missing input to default.
pub(super) fn atomic_number(atom: &Props) -> u8 {
    atom.atomic_number()
        .expect("Atomistic invariant: every atom carries a valid atomic_number (0..=118)")
}

/// RDKit `PeriodicTable::getDefaultValence`: the first entry of the element's
/// valence list ([`Element::default_valences`]), or `-1` when the element has
/// none (noble gases) or `atno` names no element.
pub(crate) fn default_valence(atno: u8) -> i32 {
    Element::by_number(atno)
        .and_then(|e| e.default_valences().first())
        .map_or(-1, |&v| i32::from(v))
}

/// The number of hydrogens `atom_id` carries but the graph does not draw
/// (RDKit `getNumImplicitHs` + the declared count of a bracket atom).
///
/// A declared `h_count` (a SMILES bracket atom) is returned as is. Otherwise
/// the count is the smallest allowed valence of the charge-adjusted element
/// that is at least the atom's explicit valence, minus that explicit valence.
///
/// Returns `None` when `atom_id` names no atom, when the atom is the wildcard
/// (`atomic_number` 0, which has no valence model), when its charge-adjusted
/// element has no defined default valences (e.g. noble gases), or when its
/// explicit valence already exceeds every allowed valence.
///
/// # Panics
///
/// When the atom carries no valid atomic number — a broken [`Atomistic`]
/// invariant.
///
/// # Bond-order convention
///
/// Localized bond counts are read from the bond's `bond_number`, and
/// aromaticity from its `bond_type` — the two are separate questions; an
/// aromatic bond counts by its Kekulé `bond_number` like any other.
///
/// # Formal-charge correction
///
/// A formal charge is folded into the element identity, not into the bond
/// demand: the valence list of `Z − formal_charge` is used. This is RDKit's
/// `getEffectiveAtomicNum` rule and gets the group-13/14 cation case right
/// (e.g. `[CH3+]` → C(Z=6) − (+1) = B(Z=5), valence 3 → 3 H, rather than the
/// naive `bond_order_sum − formal_charge` which over-counts to 5 H). For the
/// late atoms N/O/F the two formulations happen to agree, but for early atoms
/// (B, C, Si, …) they diverge, which is exactly the bug this rule fixes.
pub fn n_implicit_hydrogens(mol: &Atomistic, atom_id: NodeId) -> Option<u32> {
    let atom = mol.get_atom(atom_id).ok()?;

    // A declared hydrogen count (SMILES bracket atom) is exact: `[nH]` has one
    // hydrogen and `[C]` has none, whatever the valence model would prefer.
    if let Some(h) = atom.get(keys::H_COUNT).and_then(|v| v.as_f64()) {
        return Some(h.max(0.0).round() as u32);
    }

    // The wildcard (Z = 0) names no element, so it has no valence model.
    let element = Element::by_number(atomic_number(&atom))?;

    // RDKit charged-atom valence rule (`getEffectiveAtomicNum` +
    // `calculateImplicitValence` in `Code/GraphMol/Atom.cpp`):
    //
    //   1. Z_eff = Z − formal_charge  (cation → element one place earlier;
    //      anion → one place later). The valence list is taken from Z_eff,
    //      NOT from the bare element with a charge-adjusted demand.
    //   2. demand = sum of incident bond orders (no charge term here).
    //   3. target = smallest Z_eff valence ≥ demand.
    //   4. implicit_h = target − demand.
    //
    // This is what makes early atoms (B, C, Si, …) and late atoms (N, O, F)
    // behave asymmetrically under charge:
    //   [CH3+]  Z 6−(+1)=5 (B), valences [3], demand 0 → 3 H
    //   [CH3-]  Z 6−(−1)=7 (N), valences [3,5], demand 0 → 3 H
    //   [NH4+]  Z 7−(+1)=6 (C), valences [4], demand 0 → 4 H
    //   [BH4-]  Z 5−(−1)=6 (C), valences [4], demand 0 → 4 H
    //   [OH-]   Z 8−(−1)=9 (F), valences [1], demand 0 → 1 H
    //   [NH2-]  Z 7−(−1)=8 (O), valences [2], demand 0 → 2 H
    let formal_charge = atom.formal_charge();

    // Fold the charge into the element identity, then read that element's
    // valence list. An out-of-range shift (or an element with no valence
    // model) means we add no hydrogens.
    let effective = element.effective_atomic_number(formal_charge)?;
    let valences = effective.default_valences();
    if valences.is_empty() {
        return None; // noble gas / effective element with no valence model
    }

    // Sum of bond orders connected to this atom (the explicit valence).
    let demand: f64 = valence_demand(mol, atom_id, valences[0]);

    // Select the smallest allowed valence ≥ the (un-charge-adjusted) demand.
    let target = valences
        .iter()
        .copied()
        .find(|&v| v as f64 >= demand - 1e-6);

    let target = target?; // if demand exceeds all valences, add nothing
    let n = target as f64 - demand;
    if n <= 0.5 {
        Some(0)
    } else {
        Some(n.round() as u32)
    }
}

/// Every hydrogen `atom_id` carries: its bonded hydrogen atoms plus the
/// [`n_implicit_hydrogens`] the graph does not draw (none for an atom the
/// valence model cannot fill, such as the wildcard).
///
/// The one total-hydrogen fact every perception reads — the SMARTS `H<n>`
/// primitive, the environment SMARTS a typifier writes, the snapshots of
/// aromaticity and hybridization — so an explicit-H graph and the same
/// molecule read from SMILES (hydrogens implicit) give the same answer.
pub fn n_hydrogens(mol: &Atomistic, atom_id: NodeId) -> u32 {
    let explicit = mol
        .neighbors(atom_id, NeighborKinds::BONDS)
        .filter(|(nb, _)| {
            mol.get_atom(*nb).ok().and_then(|atom| atom.atomic_number()) == Some(Element::H.z())
        })
        .count() as u32;
    explicit + n_implicit_hydrogens(mol, atom_id).unwrap_or(0)
}

/// Explicit valence of `atom_id` — the demand its existing bonds already place
/// on `lowest_valence`, the smallest valence its (charge-adjusted) element has.
///
/// # Aromatic bonds
///
/// An aromatic bond (`bond_type` aromatic) has order 1.5 in the delocalized
/// picture, but that number is a *bond* property and summing it does not give
/// an atom's valence: in every Kekulé
/// structure an aromatic atom has one σ bond per aromatic neighbour, plus at
/// most one π bond. Summing 1.5 per bond bills a ring atom with two aromatic
/// neighbours for two half-π bonds it does not both have, and bills a
/// lone-pair donor for a π bond it does not have at all — which is how
/// thiophene's S reaches 3.0, takes the S valence of 4, and grows a spurious
/// S–H.
///
/// So aromatic bonds are counted as the σ frame, and the π bond is added back
/// exactly once — only when the σ frame leaves room for it. That single test
/// separates the two donor classes without an element table:
///
/// | atom | σ | lowest valence | π? | demand | H |
/// |---|---|---|---|---|---|
/// | benzene C–H     | 2 | 4 | yes | 3 | 1 |
/// | substituted C   | 3 | 4 | yes | 4 | 0 |
/// | pyridine N      | 2 | 3 | yes | 3 | 0 |
/// | furan O         | 2 | 2 | no  | 2 | 0 |
/// | thiophene S     | 2 | 2 | no  | 2 | 0 |
/// | aromatic C=O    | 4 | 4 | no  | 4 | 0 |
///
/// (Pyrrole-type N reaches this path only when its H was *not* declared; a
/// declared `h_count` short-circuits in [`n_implicit_hydrogens`].)
///
/// A graph with integral Kekulé orders has no aromatic bonds and is summed
/// unchanged.
fn valence_demand(mol: &Atomistic, atom_id: NodeId, lowest_valence: u8) -> f64 {
    // The two facts are read from their own places: how many bonds this is
    // (the localized number) and whether it is delocalized (the class).
    let bonds: Vec<(BondOrder, f64)> = mol
        .neighbors(atom_id, NeighborKinds::BONDS)
        .map(|(_, bid)| {
            const LIVE: &str = "neighbors yields live bonds";
            let number = mol.bond_number(bid).expect(LIVE).count().max(1) as f64;
            (mol.bond_type(bid).expect(LIVE), number)
        })
        .collect();

    let n_aromatic = bonds.iter().filter(|(t, _)| t.is_aromatic()).count();
    // An aromatic bond contributes its sigma bond here; the extra pi bond is
    // added once below if the atom is still short of its lowest valence.
    let sigma: f64 = bonds
        .iter()
        .map(|(t, n)| if t.is_aromatic() { 1.0 } else { *n })
        .sum();

    if n_aromatic > 0 && sigma < lowest_valence as f64 - 1e-6 {
        sigma + 1.0
    } else {
        sigma
    }
}
