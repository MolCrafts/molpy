//! MMFF94 aromaticity perception, and the topology snapshot it runs on.
//!
//! [`MmffTopology`] is the immutable per-atom / per-bond snapshot every MMFF
//! pass reads (ported, data-flow only, from the per-atom queries RDKit's
//! `AtomTyper.cpp` makes on an `ROMol`: `getAtomicNum`, `getDegree`,
//! `getTotalDegree`, `getFormalCharge`, `getBondBetweenAtoms`,
//! `getValence(EXPLICIT) + getNumImplicitHs()`, plus the derived
//! Kekulé/SSSR/aromaticity flags and the hybridization [`super::perceive_hybridizations`]
//! derives; RDKit `Code/GraphMol/ForceFieldHelpers/MMFF/AtomTyper.cpp` and
//! `Code/GraphMol/ConjugHybrid.cpp`, BSD-3, RDKit contributors).
//!
//! [`set_mmff_aromaticity`] is a faithful port of
//! `RDKit::MolOps::setMMFFAromaticity` (`Code/GraphMol/Aromaticity.cpp`,
//! BSD-3, RDKit contributors). It walks every SSSR ring and counts pi
//! electrons (2 per ring double bond, +1 per exocyclic double bond to an
//! aromatic neighbour, +2 for a 5-ring carrying N/O/divalent-S with no
//! exocyclic double bond and an odd ring size). A ring whose pi count
//! satisfies the 4n+2 Hückel rule and contains only sp2 carbon / nitrogen is
//! flagged aromatic; its ring bonds are promoted to the aromatic bond class.
//! The outer `while` loop iterates until no further atoms are marked, so
//! fused systems converge. The output is a new snapshot (the input is left
//! untouched).
//!
//! MMFF's typifier (`ff::typifier`, MMFF94 / MMFF94s) is the consumer; the
//! perception itself is chemistry and lives here.

// The ring loop indexes several parallel per-ring arrays by ring index,
// mirroring the C++ `for (i = 0; i < atomRings.size(); ++i)` structure.
#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use crate::core::Element;
use crate::core::NeighborKinds;
use crate::core::{Atomistic, BondNumber, BondOrder, NodeId};
use crate::perceive::Hybridization;
use crate::perceive::{RingSet, perceive_rings};

use super::valence::atomic_number;

/// The bond class MMFF perceives against, from a localized bond number.
///
/// Aromaticity is *not* inferable from the number — it is the bond's class,
/// and the caller passes it separately. MMFF reads an unknown or quadruple
/// bond as the nearest of single / triple.
fn class_of_number(n: BondNumber) -> BondOrder {
    match n {
        BondNumber::Triple | BondNumber::Quadruple => BondOrder::Triple,
        BondNumber::Double => BondOrder::Double,
        BondNumber::Single | BondNumber::Unknown => BondOrder::Single,
    }
}

/// Immutable per-atom / per-bond snapshot used by all MMFF passes.
#[derive(Debug, Clone)]
pub(crate) struct MmffTopology {
    /// Atom ids in stable iteration order (this defines the public index).
    pub(crate) atom_ids: Vec<NodeId>,
    /// atom id -> dense index into `atom_ids`.
    idx_of: HashMap<NodeId, usize>,
    /// atomic number per atom index.
    pub(crate) atno: Vec<u8>,
    /// formal charge per atom index (from `"formal_charge"` prop, default 0).
    pub(crate) formal_charge: Vec<i32>,
    /// neighbor atom indices per atom index.
    pub(crate) nbrs: Vec<Vec<usize>>,
    /// bond order to each neighbor (parallel to `nbrs`); ring bonds become
    /// `Aromatic` after aromaticity perception.
    pub(crate) nbr_order: Vec<Vec<BondOrder>>,
    /// kekulé bond order to each neighbor (parallel to `nbrs`); never rewritten
    /// by aromaticity, so total-bond-order tests stay correct.
    pub(crate) nbr_kekule: Vec<Vec<BondOrder>>,
    /// SSSR ring info.
    pub(crate) rings: RingSet,
    /// rings expressed as dense atom indices (parallel to `rings.rings()`).
    pub(crate) ring_idx: Vec<Vec<usize>>,
    /// MMFF aromatic flag per atom (set by [`set_mmff_aromaticity`]).
    pub(crate) is_aromatic: Vec<bool>,
    /// MMFF aromatic flag per ring (parallel to `ring_idx`).
    pub(crate) ring_aromatic: Vec<bool>,
    /// RDKit hybridization per atom ([`crate::perceive::perceive_hybridizations`]),
    /// read by MMFF aromaticity to reject non-sp² ring carbon and nitrogen.
    pub(crate) hybridization: Vec<Hybridization>,
}

impl MmffTopology {
    /// Build the snapshot from an [`Atomistic`] molecule.
    ///
    /// MMFF types a molecule whose hydrogens are all drawn: the snapshot's
    /// degrees and valences count bonds only.
    ///
    /// Returns an error naming the atom for a wildcard atom (`atomic_number`
    /// 0), which names no element MMFF could type, and for one that carries
    /// hydrogens the graph does not draw ([`n_implicit_hydrogens`](crate::perceive::n_implicit_hydrogens)
    /// above zero): run `add_hydrogens` first.
    pub(crate) fn build(mol: &Atomistic) -> Result<Self, String> {
        let atom_ids: Vec<NodeId> = mol.atom_ids().collect();
        for (i, &id) in atom_ids.iter().enumerate() {
            if let Some(n) = crate::perceive::n_implicit_hydrogens(mol, id)
                && n > 0
            {
                return Err(format!(
                    "atom {i} carries {n} hydrogen(s) the graph does not draw; MMFF types \
                     explicit hydrogens only — add them first (perceive::add_hydrogens)"
                ));
            }
        }
        // A snapshot's own index (it outlives no edit of `mol`).
        let idx_of: HashMap<NodeId, usize> = atom_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let n = atom_ids.len();
        let mut atno = vec![0u8; n];
        let mut formal_charge = vec![0i32; n];
        let mut nbrs = vec![Vec::new(); n];
        let mut nbr_order = vec![Vec::new(); n];
        let mut nbr_kekule = vec![Vec::new(); n];

        for (i, &id) in atom_ids.iter().enumerate() {
            let atom = mol.get_atom(id).map_err(|e| e.to_string())?;
            let el = Element::by_number(atomic_number(&atom)).ok_or_else(|| {
                format!("atom {i} is the wildcard '*', which names no element MMFF types")
            })?;
            atno[i] = el.z();
            formal_charge[i] = atom.formal_charge();
            for (nbr_id, bid) in mol.neighbors(id, NeighborKinds::BONDS) {
                let j = idx_of[&nbr_id];
                // `nbr_order` is the chemical class MMFF perceives against;
                // `nbr_kekule` is the localized structure. They differ exactly
                // on aromatic bonds — which is the reason for two fields.
                let kekule = class_of_number(mol.bond_number(bid).map_err(|e| e.to_string())?);
                let class = if mol.bond_type(bid).map_err(|e| e.to_string())?.is_aromatic() {
                    BondOrder::Aromatic
                } else {
                    kekule
                };
                nbrs[i].push(j);
                nbr_order[i].push(class);
                nbr_kekule[i].push(kekule);
            }
        }

        let rings = perceive_rings(mol);
        let ring_idx: Vec<Vec<usize>> = rings
            .rings()
            .iter()
            .map(|r| r.iter().map(|id| idx_of[id]).collect())
            .collect();
        let ring_aromatic = vec![false; ring_idx.len()];

        Ok(Self {
            atom_ids,
            idx_of,
            atno,
            formal_charge,
            nbrs,
            nbr_order,
            nbr_kekule,
            rings,
            ring_idx,
            is_aromatic: vec![false; n],
            ring_aromatic,
            hybridization: crate::perceive::perceive_hybridizations(mol),
        })
    }

    /// Number of atoms.
    pub(crate) fn n_atoms(&self) -> usize {
        self.atom_ids.len()
    }

    /// `NodeId` for a dense atom index.
    pub(crate) fn id(&self, i: usize) -> NodeId {
        self.atom_ids[i]
    }

    /// RDKit `getDegree()`: number of explicit (heavy + H) neighbors.
    ///
    /// [`build`](Self::build) refuses undrawn hydrogens, so this equals
    /// `getTotalDegree()` as well.
    pub(crate) fn degree(&self, i: usize) -> usize {
        self.nbrs[i].len()
    }

    /// RDKit `getTotalDegree()`: `degree`, every hydrogen being drawn
    /// ([`build`](Self::build)).
    pub(crate) fn total_degree(&self, i: usize) -> usize {
        self.degree(i)
    }

    /// Bond order between atoms `i` and `j` if bonded.
    pub(crate) fn bond_order(&self, i: usize, j: usize) -> Option<BondOrder> {
        self.nbrs[i]
            .iter()
            .position(|&k| k == j)
            .map(|p| self.nbr_order[i][p])
    }

    /// RDKit `getValence(EXPLICIT) + getNumImplicitHs()` — the total bond
    /// order around the atom. [`build`](Self::build) refuses undrawn
    /// hydrogens, so `getNumImplicitHs() == 0` and this is the integer sum of
    /// Kekulé bond orders.
    pub(crate) fn total_bond_order(&self, i: usize) -> u32 {
        self.nbr_kekule[i]
            .iter()
            .map(|o| match o {
                BondOrder::Single | BondOrder::Aromatic | BondOrder::Unknown => 1,
                BondOrder::Double => 2,
                BondOrder::Triple => 3,
            })
            .sum()
    }

    /// Number of hydrogen neighbors.
    pub(crate) fn n_h_neighbors(&self, i: usize) -> usize {
        self.nbrs[i].iter().filter(|&&j| self.atno[j] == 1).count()
    }

    /// Is atom `i` in a ring of exactly `size` atoms?
    pub(crate) fn is_atom_in_ring_of_size(&self, i: usize, size: usize) -> bool {
        self.rings
            .rings_of_size(size)
            .iter()
            .any(|r| r.iter().any(|&id| self.idx_of[&id] == i))
    }

    /// Are all the listed atoms in the same ring of exactly `size`?
    pub(crate) fn atoms_in_same_ring_of_size(&self, size: usize, atoms: &[usize]) -> bool {
        self.ring_idx
            .iter()
            .filter(|r| r.len() == size)
            .any(|r| atoms.iter().all(|a| r.contains(a)))
    }

    /// Is atom `i` in an *aromatic* ring of exactly `size` atoms?
    /// (RDKit `RingMembershipSize::isAtomInAromaticRingOfSize`.)
    pub(crate) fn is_atom_in_aromatic_ring_of_size(&self, i: usize, size: usize) -> bool {
        if !self.is_aromatic[i] {
            return false;
        }
        self.ring_idx
            .iter()
            .zip(self.ring_aromatic.iter())
            .any(|(r, &arom)| arom && r.len() == size && r.contains(&i))
    }
}

/// Run MMFF aromaticity perception, returning an updated snapshot.
pub(crate) fn set_mmff_aromaticity(input: &MmffTopology) -> MmffTopology {
    let mut topo = input.clone();
    let n_rings = topo.ring_idx.len();
    if n_rings == 0 {
        return topo;
    }

    let n_atoms = topo.n_atoms();
    let mut arom_bit = vec![false; n_atoms];
    let mut arom_ring = vec![false; n_rings];

    let mut arom_rings_all_set = false;
    let mut n_arom_set: i64 = 0;
    let mut old_n_arom_set: i64 = -1;

    while !arom_rings_all_set && n_arom_set > old_n_arom_set {
        for ri in 0..n_rings {
            let ring = topo.ring_idx[ri].clone();
            let rsize = ring.len();
            let mut pi_e: u32 = 0;
            let mut move_to_next_ring = false;
            let mut is_nos_in_ring = false;
            let mut exo_double_bond = false;

            for j in 0..rsize {
                if move_to_next_ring {
                    break;
                }
                let a = ring[j];
                let atno = topo.atno[a];
                if atno == 7 || atno == 8 || (atno == 16 && topo.degree(a) == 2) {
                    is_nos_in_ring = true;
                }
                let next = if j == rsize - 1 { ring[0] } else { ring[j + 1] };
                if topo.bond_order(a, next) == Some(BondOrder::Double) {
                    pi_e += 2;
                } else {
                    // carbon, or nitrogen with total bond order == 4
                    let tbo = topo.total_bond_order(a);
                    if atno != 6 && !(atno == 7 && tbo == 4) {
                        continue;
                    }
                    for (p, &nbr) in topo.nbrs[a].iter().enumerate() {
                        // exocyclic only
                        if ring.contains(&nbr) {
                            continue;
                        }
                        let bo = topo.nbr_order[a][p];
                        if bo == BondOrder::Single {
                            continue;
                        }
                        // neighbor in a ring whose aromaticity is not yet set:
                        // defer this whole ring.
                        if topo.rings.is_atom_in_ring(topo.id(nbr)) && !arom_bit[nbr] {
                            move_to_next_ring = true;
                            break;
                        }
                        if bo == BondOrder::Double {
                            if arom_bit[nbr] {
                                pi_e += 1;
                            } else {
                                exo_double_bond = true;
                            }
                        }
                    }
                }
            }

            if move_to_next_ring {
                continue;
            }

            // mark perceived; reject non-sp2 C/N rings
            let mut can_be_aromatic = true;
            for &a in &ring {
                arom_bit[a] = true;
                let atno = topo.atno[a];
                if (atno == 6 || atno == 7) && topo.hybridization[a] != Hybridization::Sp2 {
                    can_be_aromatic = false;
                }
            }
            if !can_be_aromatic {
                continue;
            }

            if is_nos_in_ring && !exo_double_bond && (rsize % 2 == 1) {
                pi_e += 2;
            }

            if pi_e > 2 && (pi_e - 2).is_multiple_of(4) {
                arom_ring[ri] = true;
                for &a in &ring {
                    topo.is_aromatic[a] = true;
                }
            }
        }

        old_n_arom_set = n_arom_set;
        n_arom_set = 0;
        arom_rings_all_set = true;
        for ring in &topo.ring_idx {
            for &a in ring {
                if arom_bit[a] {
                    n_arom_set += 1;
                } else {
                    arom_rings_all_set = false;
                }
            }
        }
    }

    // Promote aromatic ring bonds to the aromatic bond order (so later passes
    // that test for Bond::AROMATIC behave like RDKit after setMMFFAromaticity).
    topo.ring_aromatic = arom_ring.clone();
    let arom_ring_bonds: Vec<(usize, usize)> = topo
        .ring_idx
        .iter()
        .enumerate()
        .filter(|(ri, _)| arom_ring[*ri])
        .flat_map(|(_, ring)| {
            let rsize = ring.len();
            (0..rsize).map(move |j| {
                let a = ring[j];
                let b = if j == rsize - 1 { ring[0] } else { ring[j + 1] };
                (a, b)
            })
        })
        .collect();
    for (a, b) in arom_ring_bonds {
        set_bond_aromatic(&mut topo, a, b);
    }

    topo
}

fn set_bond_aromatic(topo: &mut MmffTopology, a: usize, b: usize) {
    if let Some(p) = topo.nbrs[a].iter().position(|&k| k == b) {
        topo.nbr_order[a][p] = BondOrder::Aromatic;
    }
    if let Some(p) = topo.nbrs[b].iter().position(|&k| k == a) {
        topo.nbr_order[b][p] = BondOrder::Aromatic;
    }
}
