//! The molecule as the ATD rule language sees it.
//!
//! Every constraint an [`AtdRule`](crate::ff::params::AtdRule) can state is a
//! question about counts — atomic number, degree, attached hydrogens, ring
//! membership, how many single / double / aromatic / delocalized bonds an atom
//! carries. [`MolFacts`] answers all of them from **one** pass over the graph,
//! so a table with 200 rules costs one traversal, not 200.
//!
//! The facts are table-independent: `sb`/`db`/`ab`/`DL` mean the same thing in
//! `ATOMTYPE_BCC.DEF` and in `ATOMTYPE_GAS.DEF`. That is what lets one engine
//! walk every table. The one exception is antechamber's own: under the AM1-BCC
//! tables (BCC, ABCG2) `atomtype` perceives rings with the indole rule on.
//!
//! The ring facts (`RG*`, `NR`, `AR1` … `AR5`) are antechamber's rings and ring
//! classes ([`perceive_ring_classes`]): every chordless ring of up to ten
//! ring-capable atoms, not a smallest set, classed from connection counts and
//! the perceived bond types — so anthracene's middle ring is AR1 whichever
//! Kekulé structure it holds, and a ring through selenium is no ring.

use std::collections::HashMap;

use molrs::core::PropValue;
use molrs::core::keys;
use molrs::core::keys::BCC_BOND_TYPE;
use molrs::core::{Atomistic, NodeId, Relation, RelationId};
use molrs::perceive::perceive_ring_classes;

use crate::ff::params::AtomProp;

/// Pre-computed answers to every question an ATD rule can ask about an atom.
///
/// All vectors are indexed by the atom's position in `mol.atoms()` order;
/// [`MolFacts::index_of`] maps an [`NodeId`] onto that position.
#[derive(Debug, Clone)]
pub(super) struct MolFacts {
    /// Atom id -> row index into every vector below.
    pub(super) index: HashMap<NodeId, usize>,
    /// Atomic number, read from [`keys::ATOMIC_NUMBER`] (`0` for a wildcard
    /// atom, which only an element-free rule can match).
    pub(super) atomic_number: Vec<u8>,
    /// Number of bonded neighbours.
    pub(super) degree: Vec<usize>,
    /// Number of bonded hydrogens.
    pub(super) hydrogen_count: Vec<usize>,
    /// Whether the atom is electron-withdrawing (`EW`).
    pub(super) ewd: Vec<bool>,
    /// Residue name, or `*` when the graph carries none.
    pub(super) residue: Vec<String>,
    /// Ring / aromaticity / bond-order counts.
    pub(super) props: Vec<AtomPropertyFacts>,
    /// Neighbours as `(atom, antechamber bond type, bond)`.
    pub(super) neighbors: Vec<Vec<(NodeId, i32, RelationId)>>,
    /// Every bond as `(first atom row, second atom row, antechamber bond type)`,
    /// in graph bond order and with its endpoints in stored order — the bond
    /// list antechamber's post-typing passes sweep.
    pub(super) bonds: Vec<(usize, usize, i32)>,
}

impl MolFacts {
    /// Derive the facts of `mol`, whose bonds must already carry perceived
    /// antechamber bond types under [`BCC_BOND_TYPE`] (see
    /// [`AtdTypifier::perceive_bond_types`](super::AtdTypifier::perceive_bond_types)).
    ///
    /// `bcc` — the table is an AM1-BCC one (BCC, ABCG2), for which `atomtype`
    /// perceives rings with the indole rule on.
    pub(super) fn new(mol: &Atomistic, bcc: bool) -> Result<Self, String> {
        let atom_ids: Vec<_> = mol.atoms().map(|(aid, _)| aid).collect();
        let index: HashMap<NodeId, usize> = atom_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, aid)| (aid, i))
            .collect();
        let mut atomic_number = Vec::with_capacity(atom_ids.len());
        let mut residue = Vec::with_capacity(atom_ids.len());
        for aid in &atom_ids {
            let atom = mol.get_atom(*aid).map_err(|e| e.to_string())?;
            let z = atom.atomic_number().ok_or_else(|| {
                format!(
                    "ATD atom typing requires a valid `{}` for {aid:?}",
                    keys::ATOMIC_NUMBER
                )
            })?;
            atomic_number.push(z);
            residue.push(atom.get_str(keys::RES_NAME).unwrap_or("*").to_owned());
        }

        let mut neighbors = vec![Vec::new(); atom_ids.len()];
        let mut bonds = Vec::new();
        for (bid, bond) in mol.bonds() {
            let bond_type = antechamber_bond_type(&bond)?;
            let a = bond.nodes[0];
            let b = bond.nodes[1];
            let ia = index[&a];
            let ib = index[&b];
            neighbors[ia].push((b, bond_type, bid));
            neighbors[ib].push((a, bond_type, bid));
            bonds.push((ia, ib, bond_type));
        }
        let degree: Vec<usize> = neighbors.iter().map(Vec::len).collect();
        let mut hydrogen_count = vec![0; atom_ids.len()];
        for (i, nbs) in neighbors.iter().enumerate() {
            hydrogen_count[i] = nbs
                .iter()
                .filter(|(nb, _, _)| atomic_number[index[nb]] == 1)
                .count();
        }
        let ewd: Vec<bool> = atomic_number
            .iter()
            .map(|z| matches!(*z, 7 | 8 | 9 | 16 | 17 | 35 | 53))
            .collect();

        // antechamber's own rings and ring classes (`ring.c`), on the bond
        // types just perceived — as `atomtype` runs `ringdetect` on the bond
        // types `bondtype` wrote.
        let con: Vec<Vec<usize>> = neighbors
            .iter()
            .map(|nbs| nbs.iter().map(|(nb, _, _)| index[nb]).collect())
            .collect();
        let rings = perceive_ring_classes(&atomic_number, &con, &bonds, bcc);
        let mut props: Vec<AtomPropertyFacts> = rings
            .atoms
            .iter()
            .map(|f| AtomPropertyFacts {
                rg: f.rg,
                nr: f.nr,
                ar1: f.ar[0],
                ar2: f.ar[1],
                ar3: f.ar[2],
                ar4: f.ar[3],
                ar5: f.ar[4],
                ..AtomPropertyFacts::default()
            })
            .collect();
        for (i, nbs) in neighbors.iter().enumerate() {
            for (_, bond_type, _) in nbs {
                props[i].add_bond_type(*bond_type);
            }
        }

        Ok(Self {
            index,
            atomic_number,
            degree,
            hydrogen_count,
            ewd,
            residue,
            props,
            neighbors,
            bonds,
        })
    }

    /// The row index of `aid`.
    pub(super) fn index_of(&self, aid: NodeId) -> Result<usize, String> {
        self.index
            .get(&aid)
            .copied()
            .ok_or_else(|| format!("unknown atom id {aid:?}"))
    }

    /// Electron-withdrawing atoms around the atom `aid` hangs off.
    ///
    /// The `.DEF` column counts the EW neighbours of the *attachment point*, not
    /// of the candidate itself — that is how a hydrogen learns about the
    /// substituents of the carbon it sits on.
    pub(super) fn ewd_count_around_attachment(&self, aid: NodeId) -> Option<usize> {
        let i = self.index_of(aid).ok()?;
        let attached = self.neighbors[i].first()?.0;
        let j = self.index_of(attached).ok()?;
        Some(
            self.neighbors[j]
                .iter()
                .filter(|(nb, _, _)| self.ewd[self.index[nb]])
                .count(),
        )
    }
}

/// Ring, aromaticity and bond-order counts of a single atom.
///
/// The lowercase source tokens (`sb`, `db`, `tb`) count aromatic and delocalized
/// bonds too; the uppercase ones (`SB`, `DB`, `TB`) are strict. Both are counted
/// here, and [`AtomPropertyFacts::count`] hands out whichever the rule asked for.
#[derive(Debug, Clone, Default)]
pub(super) struct AtomPropertyFacts {
    /// `rg[0]` = rings of any size; `rg[n]` = rings of size `n` (antechamber's
    /// rings, see [`perceive_ring_classes`]).
    rg: [i32; 11],
    /// `NR` — in no ring.
    nr: i32,
    /// `AR1` — rings of this atom that are pure aromatic (benzene, pyridine).
    ar1: i32,
    /// `AR2` — planar rings of this atom with two continuous single bonds and at
    /// least two double bonds (imidazole, thiophene, pyrrole).
    ar2: i32,
    /// `AR3` — planar rings of this atom whose double bonds are formed between
    /// ring atoms and non-ring atoms (a quinone, a pyridone).
    ar3: i32,
    /// `AR4` — rings of this atom that are none of AR1, AR2, AR3 or AR5.
    ar4: i32,
    /// `AR5` — pure aliphatic rings of this atom, made of sp3 carbon.
    ar5: i32,
    /// `sb` — single bonds, aromatic and delocalized ones included.
    sb: i32,
    /// `SB` — single bonds, strictly.
    sb_strict: i32,
    /// `db` — double bonds, aromatic and delocalized ones included.
    db: i32,
    /// `DB` — double bonds, strictly.
    db_strict: i32,
    /// `tb` — triple bonds, aromatic ones included.
    tb: i32,
    /// `TB` — triple bonds, strictly.
    tb_strict: i32,
    /// `AB` — aromatic bonds.
    ab: i32,
    /// `DL` — delocalized bonds.
    dl: i32,
}

impl AtomPropertyFacts {
    /// Fold one incident bond into the counts.
    fn add_bond_type(&mut self, bond_type: i32) {
        match bond_type {
            1 => {
                self.sb += 1;
                self.sb_strict += 1;
            }
            2 => {
                self.db += 1;
                self.db_strict += 1;
            }
            3 => {
                self.tb += 1;
                self.tb_strict += 1;
            }
            7 => {
                self.ab += 1;
                self.sb += 1;
            }
            8 => {
                self.ab += 1;
                self.db += 1;
            }
            9 => {
                self.sb += 1;
                self.sb_strict += 1;
                self.dl += 1;
            }
            10 => {
                self.ab += 1;
            }
            _ => {}
        }
    }

    /// How many times this atom satisfies `prop` — negative only for an AR2
    /// count the AM1-BCC indole rule took below zero, as antechamber's does.
    pub(super) fn count(&self, prop: AtomProp) -> i32 {
        match prop {
            AtomProp::Rg => self.rg[0],
            AtomProp::Rg3 => self.rg[3],
            AtomProp::Rg4 => self.rg[4],
            AtomProp::Rg5 => self.rg[5],
            AtomProp::Rg6 => self.rg[6],
            AtomProp::Rg7 => self.rg[7],
            AtomProp::Rg8 => self.rg[8],
            AtomProp::Rg9 => self.rg[9],
            AtomProp::Rg10 => self.rg[10],
            AtomProp::Nr => self.nr,
            AtomProp::Ar1 => self.ar1,
            AtomProp::Ar2 => self.ar2,
            AtomProp::Ar3 => self.ar3,
            AtomProp::Ar4 => self.ar4,
            AtomProp::Ar5 => self.ar5,
            AtomProp::SbStrict => self.sb_strict,
            AtomProp::SbAny => self.sb,
            AtomProp::DbStrict => self.db_strict,
            AtomProp::DbAny => self.db,
            AtomProp::TbStrict => self.tb_strict,
            AtomProp::TbAny => self.tb,
            AtomProp::Ab => self.ab,
            AtomProp::Dl => self.dl,
        }
    }
}

/// The antechamber bond type a bond carries: 1 single, 2 double, 3 triple,
/// 7/8/10 aromatic, 9 delocalized.
///
/// Read from [`BCC_BOND_TYPE`] — the key bond-type perception
/// (`find_bond_types_from_connectivity`,
/// [`assign_bcc_bond_types`](molrs::perceive::assign_bcc_bond_types)) writes
/// it to — and **never** from the bond's `type`, which is the caller's and holds
/// their force-field bond-type *name*.
///
/// Shared with the BCC corrector, whose correction rows are keyed on the same
/// integer. A bond without one is an error, never a guessed single bond.
pub(crate) fn antechamber_bond_type(bond: &Relation) -> Result<i32, String> {
    match bond.props.get(BCC_BOND_TYPE) {
        Some(PropValue::Int(v)) => Ok(*v),
        Some(PropValue::F64(v)) if (*v - v.round()).abs() < 1.0e-6 => Ok(v.round() as i32),
        Some(PropValue::F64(v)) => Err(format!("`{BCC_BOND_TYPE}` must be integral, got {v}")),
        Some(PropValue::Str(s)) => s
            .parse::<i32>()
            .map_err(|e| format!("`{BCC_BOND_TYPE}` must be an integer string: {e}")),
        Some(PropValue::Bool(_)) => Err(format!("`{BCC_BOND_TYPE}` must be numeric, not bool")),
        None => Err(format!(
            "BCC correction requires a perceived `{BCC_BOND_TYPE}` on every bond"
        )),
    }
}
