//! Charge equivalencing — the topological-equivalence classes AM1 charges are
//! averaged over before the BCC stage (antechamber's `-eq`).

use std::collections::HashMap;

use crate::core::Atomistic;
use crate::core::BondOrder;
use crate::core::MolRsError;
use crate::core::NodeId;
use crate::core::keys;
use crate::op::vec3::dihedral;

use super::valence::atomic_number;

mod counts;
mod paths;

use paths::{PathHistograms, ScoreHistogram};

/// Weight of a path position, `0.11` (Antechamber Eq. (I)).
const POSITION_WEIGHT: f64 = 0.11;

/// Weight of a path atom's atomic number, `0.08` (Antechamber Eq. (I)).
const ELEMENT_WEIGHT: f64 = 0.08;

/// Per-position coefficient for a *trans* (E) arrangement at `-eq 2`.
const TRANS_COEF: f64 = 1.01;

/// Per-position coefficient for a *cis* (Z) arrangement at `-eq 2`.
const CIS_COEF: f64 = 0.99;

/// The width of antechamber's connectivity array (`ATOM.con[6]`).
///
/// Load-bearing, not an implementation detail: `scorepath()` emits a path's score
/// when it reaches the first **free** slot of `con[]`, so an atom with six or more
/// neighbours (`PF6⁻`, `SF6`) never terminates a scored path — paths still travel
/// *through* it, but no path *ending* at it is counted. Reproduced here so that
/// hypervalent species partition the way antechamber partitions them.
const MAX_CON: usize = 6;

/// antechamber's `-eq`: which topological-equivalence model to use.
///
/// | Level | antechamber | Meaning |
/// |---|---|---|
/// | [`EquivalenceLevel::Off`] | `-eq 0` | no equivalencing; every atom is its own class |
/// | [`EquivalenceLevel::Paths`] | `-eq 1` | the path score of [`perceive_equivalence_classes`] — **the default for `-c bcc` / `-c abcg2` / `-c resp`** (and *only* for those; every other charge method defaults to `0`) |
/// | [`EquivalenceLevel::PathsAndGeometry`] | `-eq 2` | the path score with an E/Z coefficient per position, making the partition **strictly finer** than level 1 — never coarser |
///
/// Because the default is per-charge-method, equivalencing is a **declaration of
/// the charge model**, not a global pipeline stage: see the `needs_equivalencing`
/// flag in the `ChargeModel` trait.
///
/// [`EquivalenceOptions::max_path_length`] is antechamber's `-pl`: paths longer
/// than the cap are not scored. The default is unlimited, matching `-pl -1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EquivalenceLevel {
    /// `-eq 0` — no equivalencing. Every atom is its own class, so averaging
    /// over the classes is a no-op.
    Off,
    /// `-eq 1` — equivalence by atomic paths. The default for AM1-BCC, ABCG2 and
    /// RESP, and the level pinned to the antechamber oracle.
    #[default]
    Paths,
    /// `-eq 2` — atomic paths refined by E/Z geometry: a position whose
    /// surrounding torsion is fixed by a rotation-restricted bond is weighted
    /// `1.01` (trans) or `0.99` (cis) instead of `1.0`, which can only *split*
    /// classes that level 1 merges. Strictly finer than [`Self::Paths`].
    ///
    /// The restricted bonds are perceived natively — C=C double bonds and amide
    /// C–N bonds, the native reading of the `c2/ce/cf/CM` and `c–n` / `C–N` pairs
    /// in antechamber's `ATOM_EQU.TYPE`, whose GAFF/Amber type names molrs does
    /// not assign (GAFF typing is delegated to AmberTools). Unlike level 1, this
    /// level is **not** pinned to the oracle: `-eq 2` is nobody's default.
    PathsAndGeometry,
}

/// How to compute the equivalence classes: antechamber's `-eq` and `-pl`.
///
/// [`Default`] is `-eq 1 -pl -1`, i.e. what `antechamber -c bcc` runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EquivalenceOptions {
    /// The equivalence model (`-eq`).
    pub level: EquivalenceLevel,
    /// `-pl` — the longest path, in atoms, that is scored. `None` (the default)
    /// is antechamber's `-pl -1`: no limit. A cap makes the enumeration cheaper
    /// on large or heavily fused molecules at the cost of merging atoms that a
    /// longer path would have separated.
    pub max_path_length: Option<usize>,
}

impl EquivalenceOptions {
    /// The options `antechamber -c bcc` runs with: `-eq 1`, no path-length cap.
    ///
    /// # Returns
    ///
    /// [`EquivalenceOptions::default()`].
    pub fn bcc() -> Self {
        Self::default()
    }

    /// Disable equivalencing (`-eq 0`).
    ///
    /// # Returns
    ///
    /// Options whose level is [`EquivalenceLevel::Off`].
    pub fn off() -> Self {
        Self {
            level: EquivalenceLevel::Off,
            max_path_length: None,
        }
    }
}

/// The topological-equivalence partition of a molecule's atoms.
///
/// Produced by [`perceive_equivalence_classes`]. Two atoms share a class iff their
/// sorted path-score arrays are **exactly** equal — the comparison is bit-for-bit,
/// as it is in `equatom.c`, and deliberately carries no tolerance: the scores are
/// sums of a handful of exactly-representable-ish terms, and two atoms that differ
/// chemically can produce scores that differ far below any tolerance one would dare
/// pick. A tolerance would merge them and silently smear their charges together.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EquivalenceClasses {
    /// Per atom: its class id.
    class_of: HashMap<NodeId, u32>,
    /// Per class: its members, in graph atom order.
    members: Vec<Vec<NodeId>>,
}

impl EquivalenceClasses {
    /// The class an atom belongs to.
    ///
    /// # Arguments
    ///
    /// * `atom` — the atom to look up.
    ///
    /// # Returns
    ///
    /// The 0-based class id, or `None` when the atom is not in the molecule the
    /// classes were computed from.
    pub fn class_of(&self, atom: NodeId) -> Option<u32> {
        self.class_of.get(&atom).copied()
    }

    /// The number of distinct classes.
    ///
    /// # Returns
    ///
    /// The class count; equal to the atom count exactly when no two atoms are
    /// equivalent (and always, at [`EquivalenceLevel::Off`]).
    pub fn n_classes(&self) -> usize {
        self.members.len()
    }

    /// The members of one class.
    ///
    /// # Arguments
    ///
    /// * `class` — a class id from [`Self::class_of`].
    ///
    /// # Returns
    ///
    /// The atoms in that class, in graph atom order, or `None` for an unknown id.
    pub fn members(&self, class: u32) -> Option<&[NodeId]> {
        self.members.get(class as usize).map(Vec::as_slice)
    }

    /// Iterate over every class's members.
    ///
    /// # Returns
    ///
    /// An iterator yielding each class's atoms, in class-id order.
    pub fn classes(&self) -> impl Iterator<Item = &[NodeId]> {
        self.members.iter().map(Vec::as_slice)
    }
}

/// Partition a molecule's atoms into charge-equivalence classes.
///
/// At [`EquivalenceLevel::Off`] every atom is placed in a class of its own, so
/// the caller can keep the pipeline shape and still opt out.
///
/// A semi-empirical calculation is done on **one conformer**, so its Mulliken
/// charges are not symmetric: methanol's three methyl hydrogens come out of `sqm`
/// as `(0.053, 0.098, 0.053)` purely because one of them eclipses the O–H. Feeding
/// that straight into a force field yields **conformer-dependent, symmetry-broken**
/// charges — the same molecule, re-embedded, would type differently. Averaging over
/// the topological-equivalence classes removes the artefact (all three become
/// `0.068`, the mean) and is what `antechamber -c bcc` does by default, *before*
/// applying any bond-charge correction.
///
/// # The algorithm: path scores, **not** automorphism orbits
///
/// For every atom, enumerate **all simple paths starting at it**, score each path,
/// sort the scores ascending, and compare. Two atoms are equivalent iff they have
/// the **same number of paths** and their sorted score arrays are **elementwise
/// exactly equal** (`f64` equality — there is no tolerance; see
/// [`EquivalenceClasses`]).
///
/// The score of a path is the Antechamber paper's Eq. (I) — position index `j`
/// (0-based) and atomic number `Z_j` of the atom at that position:
///
/// ```text
/// score = Σ_j [ (j + 1)·0.11 + Z_j·0.08 ]
/// ```
///
/// **This is not a graph-automorphism partition, and using one would be wrong.**
/// In exact arithmetic the sum collapses to
///
/// ```text
/// score = 0.11·L(L+1)/2 + 0.08·(Σ Z along the path)
/// ```
///
/// so a path enters the score only through its **length** and its **sum of atomic
/// numbers**. Two things follow, and both are load-bearing.
///
/// **It reads no bond orders and no formal charges.** A Kekulé carboxylate's two
/// oxygens — one `C=O`, one `C–O⁻` — are *the same atom* to this score, so
/// antechamber **merges** them and averages their charges. Any partition that
/// respects bond order or formal charge (Morgan / Weisfeiler-Leman /
/// [`Atomistic::structural_hash`], which folds both into its colours) **splits**
/// them. Orbits are therefore a **strictly finer** partition: the path score never
/// splits an orbit — an automorphism maps a path to a path with the same ordered
/// atomic numbers, hence the same score, bit for bit — but it does merge atoms that
/// lie in different orbits. Averaging by orbits leaves acetate's two oxygens at the
/// `sqm` values `-0.595 / -0.597` where antechamber returns `-0.596 / -0.596`: a
/// symmetry-broken carboxylate, and a `1e-3` e divergence from the oracle. A graph
/// hash may be used as a *pre-filter* (same orbit ⇒ same class) but never as the
/// class engine.
///
/// **It is order-blind only in exact arithmetic.** The score is accumulated
/// left to right along the path in `f64`, and the comparison is exact, so two paths
/// that are mathematically tied (same length, same `Σ Z`, different order — C–N–O
/// and C–O–N) can still land one ULP apart: `H–H–C` sums to `1.2999999999999998`
/// where `C–H–H` sums to `1.3`. The accumulation order is therefore part of the
/// contract, not an implementation detail — the scorer accumulates left to right
/// exactly as `scorepath()` does, so that the equality tested here is the equality
/// antechamber tested. (A *tolerant* comparison would merge such a pair. That is
/// the bug this exactness exists to prevent, and it is pinned by
/// `two_atoms_a_tolerance_would_merge_are_kept_apart`.)
///
/// # Arguments
///
/// * `mol` — the molecule to partition; left untouched.
/// * `opts` — the `-eq` level and `-pl` path-length cap.
///
/// # Returns
///
/// The partition, keyed by [`NodeId`].
///
/// # Performance
///
/// The paths are **counted, not walked**. A molecule with `k` rings in series has
/// `~2^k` simple paths per atom — a 10-unit conjugated polymer has ~10¹⁶ — so the
/// scores are accumulated as a histogram over the block-cut tree (see the `paths`
/// submodule): paths arriving at a cut atom are merged by their running sum and
/// length, and each biconnected block is walked once per such state set, through
/// its in-block paths merged by atomic-number sequence. Twin atoms (a methyl's
/// hydrogens) are computed once. The result is bit-for-bit the partition the walk
/// produces. What remains exponential is the enumeration *inside one block*, i.e.
/// in the size of a single fused ring system;
/// [`EquivalenceOptions::max_path_length`] (antechamber's `-pl`) still bounds
/// that. Atoms are processed in parallel under `rayon`.
///
/// # Provenance
///
/// A reimplementation of the perception in AmberTools' `antechamber/equatom.c`
/// (`scorepath()` / `equatom()`) and the class-mean in `charge.c::bccharge()`,
/// written by reading that source with the AmberTools developers' permission; see
/// `.claude/notes/notes.md` (2026-07-12) for the licensing posture.
pub fn perceive_equivalence_classes(
    mol: &Atomistic,
    opts: EquivalenceOptions,
) -> EquivalenceClasses {
    let flat = Flat::new(mol);
    let n = flat.ids.len();

    if opts.level == EquivalenceLevel::Off {
        return EquivalenceClasses {
            class_of: flat
                .ids
                .iter()
                .enumerate()
                .map(|(i, id)| (*id, i as u32))
                .collect(),
            members: flat.ids.iter().map(|id| vec![*id]).collect(),
        };
    }

    let geom = match opts.level {
        EquivalenceLevel::PathsAndGeometry => flat.restricted_torsions(),
        _ => Vec::new(),
    };
    let engine = PathHistograms::new(&flat, &geom, opts.max_path_length);
    // At `-eq 2` a twin swap does not preserve the geometry (the cis and the
    // trans hydrogen of a =CH₂), so every atom is computed.
    let twin_of = if geom.is_empty() {
        flat.twin_representatives()
    } else {
        (0..n).collect()
    };
    let histograms = histograms_of(&engine, &twin_of);

    // Group by the score histogram. A histogram keyed on the IEEE-754 bit
    // patterns is the sorted score array with its repeats counted, so equal
    // histograms are exactly `equatom.c`'s pairwise `!=` comparison finding no
    // difference — the scores are finite and positive, so bits agree iff the
    // values do — and it is hashable, which turns antechamber's O(N²) sweep
    // into one pass.
    let mut seen: HashMap<ScoreHistogram, u32> = HashMap::new();
    let mut class_of = HashMap::with_capacity(n);
    let mut class_of_row: Vec<u32> = Vec::with_capacity(n);
    let mut members: Vec<Vec<NodeId>> = Vec::new();
    for (i, key) in histograms.into_iter().enumerate() {
        let next = seen.len() as u32;
        // A twin shares its representative's histogram, hence its class; the
        // representative is its twin set's first row, so it is classed already.
        let class = match key {
            Some(key) => *seen.entry(key).or_insert(next),
            None => class_of_row[twin_of[i]],
        };
        class_of_row.push(class);
        if class as usize == members.len() {
            members.push(Vec::new());
        }
        members[class as usize].push(flat.ids[i]);
        class_of.insert(flat.ids[i], class);
    }

    EquivalenceClasses { class_of, members }
}

/// Every twin-set representative's score histogram, in graph atom order
/// (`None` for the other twins) — one independent computation per atom, so it
/// runs in parallel when `rayon` is on.
fn histograms_of(engine: &PathHistograms<'_>, twin_of: &[usize]) -> Vec<Option<ScoreHistogram>> {
    let one = |i: usize| (twin_of[i] == i).then(|| engine.histogram(i));
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        (0..twin_of.len()).into_par_iter().map(one).collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        (0..twin_of.len()).map(one).collect()
    }
}

/// One rotation-restricted torsion (antechamber's `GEOM`), used at `-eq 2`.
#[derive(Debug, Clone, Copy)]
struct Torsion {
    /// The four atoms, as flat indices.
    atoms: [usize; 4],
    /// Whether the arrangement is trans (`|φ| > 90°`).
    trans: bool,
}

/// The molecule flattened to the arrays the walk needs.
struct Flat {
    /// Per atom: its handle, in graph atom order.
    ids: Vec<NodeId>,
    /// Per atom: atomic number (`0` for a wildcard atom, which then scores as
    /// a path position that adds no Z).
    z: Vec<u8>,
    /// Per atom: neighbour indices, in bond order — antechamber's `con[]`.
    adj: Vec<Vec<usize>>,
    /// Per atom, per `adj` slot: the index into `bonds` of that bond.
    adj_bond: Vec<Vec<usize>>,
    /// Per atom: coordinates, for the `-eq 2` torsions.
    xyz: Vec<[f64; 3]>,
    /// Per bond: endpoints and chemical class.
    bonds: Vec<(usize, usize, BondOrder)>,
}

impl Flat {
    /// Flatten a molecule.
    fn new(mol: &Atomistic) -> Self {
        let mut ids = Vec::with_capacity(mol.n_atoms());
        let mut z = Vec::with_capacity(mol.n_atoms());
        let mut xyz = Vec::with_capacity(mol.n_atoms());
        for (id, atom) in mol.atoms() {
            ids.push(id);
            z.push(atomic_number(&atom));
            xyz.push([
                atom.get_f64(keys::X).unwrap_or(0.0),
                atom.get_f64(keys::Y).unwrap_or(0.0),
                atom.get_f64(keys::Z).unwrap_or(0.0),
            ]);
        }
        let rows = mol.atom_table();

        let mut adj = vec![Vec::new(); ids.len()];
        let mut adj_bond = vec![Vec::new(); ids.len()];
        let mut bonds = Vec::new();
        for (_, bond) in mol.bonds() {
            let (Some(i), Some(j)) = (rows.row(bond.nodes[0]), rows.row(bond.nodes[1])) else {
                continue;
            };
            adj[i].push(j);
            adj[j].push(i);
            adj_bond[i].push(bonds.len());
            adj_bond[j].push(bonds.len());
            // The *class*, not the number: "is this a double bond" must stay
            // false for an aromatic bond whose Kekulé phase happens to be 2.
            bonds.push((i, j, BondOrder::from_prop(bond.props.get(keys::BOND_TYPE))));
        }

        Self {
            ids,
            z,
            adj,
            adj_bond,
            xyz,
            bonds,
        }
    }

    /// Per atom, the first row of its **twin set**: atoms of one element with
    /// the same neighbours, as multisets (the hydrogens of a methyl, the
    /// fluorines of a CF₃). Swapping two twins maps the graph onto itself, and
    /// an automorphism maps every path to a path with the same atomic numbers
    /// in the same order, so twins have bit-identical score histograms and only
    /// one per set need be computed.
    ///
    /// That holds only while the walk explores every neighbour: an atom with
    /// more than [`MAX_CON`] neighbours is walked out of through its first six
    /// alone, in bond order, which a swap does not preserve. Twins touching such
    /// an atom — and atoms with a self-loop — are their own representatives.
    fn twin_representatives(&self) -> Vec<usize> {
        let n = self.adj.len();
        let mut first: HashMap<(u8, Vec<usize>), usize> = HashMap::new();
        (0..n)
            .map(|i| {
                let crowded = |a: usize| self.adj[a].len() > MAX_CON;
                if self.adj[i].contains(&i) || crowded(i) || self.adj[i].iter().any(|&a| crowded(a))
                {
                    return i;
                }
                let mut around = self.adj[i].clone();
                around.sort_unstable();
                *first.entry((self.z[i], around)).or_insert(i)
            })
            .collect()
    }

    /// The torsions whose E/Z is fixed, for `-eq 2`.
    ///
    /// antechamber gates these on pairs of GAFF/Amber atom-type names listed in
    /// `ATOM_EQU.TYPE` (`c2–c2`, `c2–ce`, `c2–cf`, `ce–cf`, `CM–CM`, `c–n`, `C–N`).
    /// molrs does not assign those names, so the gate is read natively: an sp2
    /// C=C double bond, or an amide C–N single bond (a nitrogen on a carbon that
    /// carries a double bond to O or S).
    fn restricted_torsions(&self) -> Vec<Torsion> {
        let mut out = Vec::new();
        for &(j, k, _) in &self.bonds {
            if !self.is_restricted(j, k) {
                continue;
            }
            for &i in self.adj[j].iter().take(MAX_CON) {
                if i == k {
                    continue;
                }
                for &l in self.adj[k].iter().take(MAX_CON) {
                    if l == j {
                        continue;
                    }
                    let phi =
                        dihedral(self.xyz[i], self.xyz[j], self.xyz[k], self.xyz[l]).to_degrees();
                    out.push(Torsion {
                        atoms: [i, j, k, l],
                        trans: !(-90.0..=90.0).contains(&phi),
                    });
                }
            }
        }
        out
    }

    /// Whether rotation about `j–k` is restricted: a C=C double bond, or an amide
    /// C–N bond.
    fn is_restricted(&self, j: usize, k: usize) -> bool {
        let (zj, zk) = (self.z[j], self.z[k]);
        let bond_type = self
            .bonds
            .iter()
            .find(|(a, b, _)| (*a == j && *b == k) || (*a == k && *b == j))
            .map_or(BondOrder::Unknown, |(_, _, t)| *t);

        // C=C.
        if zj == 6 && zk == 6 && bond_type == BondOrder::Double {
            return true;
        }
        // Amide C–N: the carbon carries a double bond to a chalcogen.
        let amide = |c: usize, n: usize| {
            self.z[c] == 6
                && self.z[n] == 7
                && self.bonds.iter().any(|(a, b, o)| {
                    let other = if *a == c {
                        *b
                    } else if *b == c {
                        *a
                    } else {
                        return false;
                    };
                    *o == BondOrder::Double && (self.z[other] == 8 || self.z[other] == 16)
                })
        };
        amide(j, k) || amide(k, j)
    }
}

/// Perceive charge-equivalence classes and write them onto `mol`, in place:
/// every atom receives an [`EQUIV_CLASS`](crate::core::keys::EQUIV_CLASS) id
/// (0-based), the [`perceive_equivalence_classes`] partition under `opts`
/// (`EquivalenceOptions::default()` is antechamber's `-eq 1`, the path-score
/// partition AM1-BCC averages its AM1 charges over).
///
/// Perception stops at the classes: the class-mean itself is a charge-model
/// step (`ff::charge`), taken by a model that declares it, because whether to
/// average is a property of the charge model and not of the graph.
///
/// # Errors
///
/// [`MolRsError`] when a class id cannot be written (an atom column holding
/// the key under a foreign dtype); `mol` may then be partly annotated.
pub fn assign_equivalence_classes(
    mol: &mut Atomistic,
    opts: EquivalenceOptions,
) -> Result<(), MolRsError> {
    let classes = perceive_equivalence_classes(mol, opts);
    for (class, members) in classes.classes().enumerate() {
        for id in members {
            mol.set_atom(*id, keys::EQUIV_CLASS, super::rings::saturating_i32(class))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
