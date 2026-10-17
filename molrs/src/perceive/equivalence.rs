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
/// The enumeration is exponential in the worst case — it walks every simple path
/// from every atom. Drug-sized organics are unproblematic (the antechamber oracle's
/// 37 molecules peak at 28 paths for a single atom), but a large fused-ring system
/// can blow up; [`EquivalenceOptions::max_path_length`] is the escape hatch, and is
/// why antechamber ships `-pl`.
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
    let mut scorer = PathScorer {
        flat: &flat,
        geom: &geom,
        max_path_length: opts.max_path_length,
        path: Vec::with_capacity(n),
        on_path: vec![false; n],
        scores: Vec::new(),
    };

    // Group by the sorted score array. Grouping on the IEEE-754 bit patterns is
    // exactly `equatom.c`'s pairwise `!=` comparison — the scores are finite and
    // positive, so bits agree iff the values do — and it is hashable, which turns
    // antechamber's O(N²) sweep into one pass.
    let mut seen: HashMap<Vec<u64>, u32> = HashMap::new();
    let mut class_of = HashMap::with_capacity(n);
    let mut members: Vec<Vec<NodeId>> = Vec::new();
    for i in 0..n {
        let key: Vec<u64> = scorer.scores_for(i).iter().map(|s| s.to_bits()).collect();
        let next = seen.len() as u32;
        let class = *seen.entry(key).or_insert(next);
        if class as usize == members.len() {
            members.push(Vec::new());
        }
        members[class as usize].push(flat.ids[i]);
        class_of.insert(flat.ids[i], class);
    }

    EquivalenceClasses { class_of, members }
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
        let index: HashMap<NodeId, usize> = ids
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();

        let mut adj = vec![Vec::new(); ids.len()];
        let mut bonds = Vec::new();
        for (_, bond) in mol.bonds() {
            let (Some(&i), Some(&j)) = (index.get(&bond.nodes[0]), index.get(&bond.nodes[1]))
            else {
                continue;
            };
            adj[i].push(j);
            adj[j].push(i);
            // The *class*, not the number: "is this a double bond" must stay
            // false for an aromatic bond whose Kekulé phase happens to be 2.
            bonds.push((i, j, BondOrder::from_prop(bond.props.get(keys::BOND_TYPE))));
        }

        Self {
            ids,
            z,
            adj,
            xyz,
            bonds,
        }
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

/// The depth-first walk that scores every simple path out of an atom.
struct PathScorer<'a> {
    flat: &'a Flat,
    geom: &'a [Torsion],
    max_path_length: Option<usize>,
    /// The path currently being walked (`equatom.c`'s `selectelement`).
    path: Vec<usize>,
    /// Membership of `path`, for an O(1) "have I visited this atom?" test.
    on_path: Vec<bool>,
    /// The scores collected for the atom being processed.
    scores: Vec<f64>,
}

impl PathScorer<'_> {
    /// Every path score for one atom, sorted ascending.
    fn scores_for(&mut self, atom: usize) -> &[f64] {
        self.scores.clear();
        self.path.clear();
        self.on_path.iter_mut().for_each(|v| *v = false);
        self.walk(atom);
        // Ascending, as `equatom.c::sort()` leaves them. No NaN can occur: every
        // term is finite and positive.
        self.scores.sort_by(f64::total_cmp);
        &self.scores
    }

    /// Extend the current path by `atom`, recurse into its unvisited neighbours,
    /// and score the path that ends here.
    fn walk(&mut self, atom: usize) {
        self.path.push(atom);
        self.on_path[atom] = true;

        // `-pl`: a path longer than the cap is abandoned *before* it is scored.
        let capped = self
            .max_path_length
            .is_some_and(|max| self.path.len() > max);
        if !capped {
            let degree = self.flat.adj[atom].len();
            for k in 0..degree.min(MAX_CON) {
                let next = self.flat.adj[atom][k];
                if !self.on_path[next] {
                    self.walk(next);
                }
            }
            // The path ending here is scored when the walk reaches a free slot of
            // `con[]` — so an atom with `MAX_CON` neighbours terminates no path.
            if degree < MAX_CON {
                let score = self.score();
                self.scores.push(score);
            }
        }

        self.on_path[atom] = false;
        self.path.pop();
    }

    /// Score the current path: Antechamber Eq. (I), accumulated left to right in
    /// the same order as `scorepath()` accumulates it, so that the exact equality
    /// the classification rests on is the same equality antechamber tests.
    fn score(&self) -> f64 {
        let mut score = 0.0;
        for (j, atom) in self.path.iter().enumerate() {
            let position = (j + 1) as f64 * POSITION_WEIGHT;
            let element = f64::from(self.flat.z[*atom]) * ELEMENT_WEIGHT;
            score += self.coefficient(j) * (position + element);
        }
        score
    }

    /// The E/Z coefficient of one path position (`1.0` unless `-eq 2` found a
    /// restricted torsion spanning it).
    ///
    /// `scorepath()` tests two windows — the four positions *ending* at `j` and the
    /// four *starting* at `j` — each in both directions, and lets the second
    /// override the first. Ported as written.
    fn coefficient(&self, j: usize) -> f64 {
        if self.geom.is_empty() {
            return 1.0;
        }
        let mut coef = 1.0;
        if j >= 3
            && let Some(c) = self.match_torsion(&self.path[j - 3..=j])
        {
            coef = c;
        }
        if self.path.len() - j > 3
            && let Some(c) = self.match_torsion(&self.path[j..=j + 3])
        {
            coef = c;
        }
        coef
    }

    /// The coefficient of the restricted torsion this four-atom window traverses,
    /// in either direction, if any.
    fn match_torsion(&self, window: &[usize]) -> Option<f64> {
        self.geom
            .iter()
            .find(|t| window == t.atoms || window.iter().rev().copied().eq(t.atoms.iter().copied()))
            .map(|t| if t.trans { TRANS_COEF } else { CIS_COEF })
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
mod tests {
    use super::*;
    use crate::core::{Element, Props};
    use crate::props;

    /// An atom of element `el` with no coordinates.
    fn atom(el: Element) -> Props {
        props! { keys::ATOMIC_NUMBER => el }
    }

    /// An atom of element `el` at `[x, y, z]`.
    fn atom_at(el: Element, [x, y, z]: [f64; 3]) -> Props {
        props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
    }

    /// Methanol, with the raw `sqm` Mulliken charges antechamber's own run
    /// produced (the methyl hydrogens are split 0.053 / 0.098 / 0.053 by the
    /// conformer).
    fn methanol() -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(atom_at(Element::C, [-0.371, -0.017, -0.004]))
            .unwrap();
        let o = mol
            .add_atom(atom_at(Element::O, [0.893, 0.621, -0.023]))
            .unwrap();
        let h1 = mol
            .add_atom(atom_at(Element::H, [-0.481, -0.582, 0.924]))
            .unwrap();
        let h2 = mol
            .add_atom(atom_at(Element::H, [-1.153, 0.744, -0.061]))
            .unwrap();
        let h3 = mol
            .add_atom(atom_at(Element::H, [-0.455, -0.687, -0.863]))
            .unwrap();
        let ho = mol
            .add_atom(atom_at(Element::H, [1.567, -0.079, 0.028]))
            .unwrap();
        for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
            mol.add_bond(a, b, props! {}).expect("bond");
        }
        for (a, q) in [
            (c, -0.073),
            (o, -0.326),
            (h1, 0.053),
            (h2, 0.098),
            (h3, 0.053),
            (ho, 0.195),
        ] {
            mol.set_atom(a, keys::CHARGE, q).expect("charge");
        }
        mol
    }

    #[test]
    fn methyl_hydrogens_are_one_class_and_the_hydroxyl_is_not() {
        let mol = methanol();
        let ids: Vec<NodeId> = mol.atoms().map(|(id, _)| id).collect();
        let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());

        assert_eq!(classes.n_classes(), 4, "C, O, 3×methyl H, hydroxyl H");
        let methyl = classes.class_of(ids[2]).expect("H class");
        assert_eq!(classes.class_of(ids[3]), Some(methyl));
        assert_eq!(classes.class_of(ids[4]), Some(methyl));
        assert_ne!(
            classes.class_of(ids[5]),
            Some(methyl),
            "hydroxyl H is not a methyl H"
        );
    }

    #[test]
    fn off_gives_every_atom_its_own_class() {
        let mol = methanol();
        let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::off());
        assert_eq!(classes.n_classes(), mol.n_atoms());

        assert!(classes.classes().all(|c| c.len() == 1));
    }

    #[test]
    fn path_score_is_blind_to_the_order_of_atoms_along_a_path() {
        // The score depends on a path only through its length and its sum of
        // atomic numbers, which is why automorphism orbits are the wrong engine.
        let mut mol = Atomistic::new();
        let c = mol.add_atom(atom(Element::C)).unwrap();
        let n = mol.add_atom(atom(Element::N)).unwrap();
        let o = mol.add_atom(atom(Element::O)).unwrap();
        mol.add_bond(c, n, props! {}).expect("bond");
        mol.add_bond(n, o, props! {}).expect("bond");
        let flat = Flat::new(&mol);
        let mut scorer = PathScorer {
            flat: &flat,
            geom: &[],
            max_path_length: None,
            path: Vec::new(),
            on_path: vec![false; 3],
            scores: Vec::new(),
        };
        scorer.path = vec![0, 1, 2]; // C–N–O
        let cno = scorer.score();
        scorer.path = vec![0, 2, 1]; // C–O–N (not a real path; the formula only)
        let con = scorer.score();
        assert_eq!(cno.to_bits(), con.to_bits());
    }

    /// ac-003 — the score comparison is **exact**, and must stay exact.
    ///
    /// Two mathematically-distinct scores are always at least `0.01` apart: a score
    /// is `(11·T + 8·S)/100` for integers `T` (positions) and `S` (Σ Z), and
    /// `gcd(11, 8) = 1`. So the *only* way two sorted score arrays can differ by
    /// less than a tolerance anyone would pick is the floating-point one: the arrays
    /// are mathematically **equal** and disagree in the last bit because the paths
    /// visit the same atomic numbers in a different order.
    ///
    /// This fixture is exactly that. Two fragments in one graph — antechamber
    /// equivalences across a whole file, so a salt's cation and anion are compared
    /// too. Both nitrogens see `{O, S}`; both have four paths; the mathematical
    /// profiles are identical, the length-3 path being `N–O–Cl` (7 + 8 + 17) in one
    /// and `N–S–F` (7 + 16 + 9) in the other — Σ Z = 32 either way. They differ by
    /// **4.4e-16**, and antechamber's `!=` keeps them apart.
    ///
    /// The fragments are contrived, deliberately: an exhaustive sweep of every
    /// valence-legal fragment up to five heavy atoms finds **no** ordinary molecule
    /// where two atoms tie mathematically but not in `f64`. That is a statement
    /// about how safe the exact comparison is in practice, not a licence to relax
    /// it — a tolerance here would be a silent, unbounded charge-smearing bug, and
    /// this test is what stops one being introduced.
    #[test]
    fn two_atoms_a_tolerance_would_merge_are_kept_apart() {
        let mut mol = Atomistic::new();
        // Fragment 1: Cl–O–N(–S).
        let n1 = mol.add_atom(atom(Element::N)).unwrap();
        let o1 = mol.add_atom(atom(Element::O)).unwrap();
        let cl = mol.add_atom(atom(Element::Cl)).unwrap();
        let s1 = mol.add_atom(atom(Element::S)).unwrap();
        // Fragment 2: F–S–N(–O).
        let n2 = mol.add_atom(atom(Element::N)).unwrap();
        let s2 = mol.add_atom(atom(Element::S)).unwrap();
        let f2 = mol.add_atom(atom(Element::F)).unwrap();
        let o2 = mol.add_atom(atom(Element::O)).unwrap();
        for (a, b) in [(n1, o1), (o1, cl), (n1, s1), (n2, s2), (s2, f2), (n2, o2)] {
            mol.add_bond(a, b, props! {}).expect("bond");
        }

        // The two score arrays are the same length and agree to 4.4e-16 …
        let flat = Flat::new(&mol);
        let mut scorer = PathScorer {
            flat: &flat,
            geom: &[],
            max_path_length: None,
            path: Vec::new(),
            on_path: vec![false; mol.n_atoms()],
            scores: Vec::new(),
        };
        let a: Vec<f64> = scorer.scores_for(0).to_vec();
        let b: Vec<f64> = scorer.scores_for(4).to_vec();
        assert_eq!(a.len(), b.len(), "same number of paths");
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max);
        assert!(worst > 0.0, "the arrays are not identical");
        assert!(
            worst < 1.0e-15,
            "the arrays differ by {worst:e} — below any tolerance one would pick"
        );

        // … and they are still NOT merged.
        let classes = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());
        assert_ne!(
            classes.class_of(n1),
            classes.class_of(n2),
            "a tolerance-based comparison merged two atoms antechamber keeps apart"
        );
    }

    #[test]
    fn a_path_length_cap_scores_no_longer_path() {
        let mol = methanol();
        let uncapped = perceive_equivalence_classes(&mol, EquivalenceOptions::bcc());
        let capped = perceive_equivalence_classes(
            &mol,
            EquivalenceOptions {
                level: EquivalenceLevel::Paths,
                max_path_length: Some(1),
            },
        );
        // With only the one-atom paths scored, every atom is scored by its element
        // alone: the three methyl H, the hydroxyl H all collapse into one class.
        assert!(capped.n_classes() < uncapped.n_classes());
        assert_eq!(capped.n_classes(), 3, "C | O | every H");
    }
}
