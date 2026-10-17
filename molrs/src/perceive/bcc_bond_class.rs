//! BCC bond-type perception — the bond typing AM1-BCC is keyed on.

use super::aromaticity::rewrite_aromatic_flags;
use super::kekule::{BondGraph, has_aromatic_marking, kekulize};
use crate::core::Atomistic;
use crate::core::MolRsError;
use crate::core::keys::BCC_BOND_TYPE;
use crate::perceive::perceive_bond_orders;
use crate::perceive::{RingClasses, perceive_ring_classes};

/// BCC bond type: a plain single bond.
pub(super) const SINGLE: i32 = 1;
/// BCC bond type: a plain double bond.
pub(super) const DOUBLE: i32 = 2;
/// BCC bond type: a triple bond.
pub(super) const TRIPLE: i32 = 3;
/// BCC bond type: N–O/S on a non-delocalized nitrogen bearing a second terminal
/// chalcogen (nitrite, and only its kin).
pub(super) const N_CHALCOGEN: i32 = 6;
/// BCC bond type: aromatic single.
pub(super) const AROMATIC_SINGLE: i32 = 7;
/// BCC bond type: aromatic double.
pub(super) const AROMATIC_DOUBLE: i32 = 8;
/// BCC bond type: delocalized (carboxylate, nitro, sulfonate, phosphate …).
pub(super) const DELOCALIZED: i32 = 9;
/// BCC bond type: the *unresolved* aromatic precursor (SYBYL `ar`). Accepted as an
/// input marking, never emitted.
pub(super) const AROMATIC_UNRESOLVED: i32 = 10;

/// Where [`assign_bcc_bond_types`] takes each bond's order from.
///
/// There is no default: the two sources answer differently on a molecule with
/// more than one Kekulé structure, and the caller who knows which program it
/// must agree with says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BondOrderSource {
    /// The orders the input states, its aromatic bonds kekulized by molrs and
    /// promoted on molrs's own aromaticity, with the type-6 rule repaired. The
    /// answer does not depend on the order of the bonds.
    Stated,
    /// The orders antechamber judges from the connectivity alone
    /// ([`perceive_bond_orders`], `bondtype -j full`), whatever orders the input
    /// states, then `bondtype`'s `finalize` — what `antechamber` does by default
    /// (`-j 4`, and always for `-c bcc`). On a molecule with more than one
    /// Kekulé structure (azulene, cyclooctatetraene) the structure — and every
    /// atom type that follows it — is the one antechamber's search settles on,
    /// and the graph's own atom and bond order decides it, as antechamber's
    /// input file's does. Every hydrogen must be drawn.
    ///
    /// Two things still read the input's bond types, because `bondtype` reads
    /// them from its file — taken here as the stated number, `ar` (10) for an
    /// aromatic bond that states none, and single otherwise:
    ///
    /// * which rings are aromatic for the 7 / 8 promotion (part 1), decided by
    ///   antechamber's ring classes ([`perceive_ring_classes`], with the AM1-BCC
    ///   indole rule) before any bond is judged — a ring whose input carries an
    ///   exocyclic double bond (a quinone drawn with its C=O) is AR3, not
    ///   aromatic;
    /// * a residue no valence state closes, where antechamber warns that "the
    ///   assigned bond types may be wrong" and keeps the file's types (an `ar`
    ///   bond stays 10).
    Connectivity,
}

/// Perceive the BCC bond type of every bond and write it onto `mol`, in place;
/// `source` says whether the bond orders are the ones the input states or the
/// ones antechamber judges from the connectivity (see [`BondOrderSource`]).
///
/// Every bond receives a [`BCC_BOND_TYPE`] prop holding the perceived type.
/// Bond `order` is *not* rewritten — the perceived Kekulé structure is consumed
/// internally and does not leak into the graph. Neither is the bond's
/// [`keys::TYPE`](crate::core::keys::TYPE), which is the caller's (see *The
/// perceived type is a perceived fact* below).
///
/// The type is always (re)derived from structure: a [`BCC_BOND_TYPE`] already on
/// the input is read only as an aromaticity *hint* (7, 8 and 10 mark an aromatic
/// bond), never trusted as an answer. That is what resolves the unresolved
/// aromatic precursor, type 10, into 7 or 8.
///
/// With [`BondOrderSource::Stated`], aromaticity is taken from the graph when it
/// carries any aromatic marking (a truthy `is_aromatic` bond prop, an `order` of
/// 1.5, or a [`BCC_BOND_TYPE`] of 7/8/10); when it carries none, aromaticity is
/// perceived first (as [`assign_aromaticity`](crate::perceive::assign_aromaticity)
/// perceives it) and its flags are written onto `mol` with the types.
///
/// AM1-BCC does not correct charges per *bond order*; it corrects them per **BCC
/// bond type**, a richer alphabet that distinguishes an aromatic single bond from
/// an aliphatic one and, crucially, marks *delocalized* bonds (a carboxylate's two
/// C–O bonds are neither "single" nor "double" — they are the same bond). Both the
/// `ATOMTYPE_BCC.DEF` atom-type rules (which count `sb`/`db`/`ab`/`DL` bonds) and
/// the `BCCPARM.DAT` correction table are keyed on it, so getting it wrong
/// silently corrupts every downstream charge.
///
/// # The alphabet
///
/// | Type | Meaning | Produced by |
/// |---|---|---|
/// | 1 | single | bond order 1 |
/// | 2 | double | bond order 2 |
/// | 3 | triple | bond order 3 |
/// | 6 | N–O/S on a **non-delocalized** N carrying a second terminal chalcogen (nitrite) | part 3 |
/// | 7 | aromatic single | part 1 — aromatic promotion |
/// | 8 | aromatic double | part 1 — aromatic promotion |
/// | 9 | delocalized (carboxylate, nitro, sulfonate, phosphate) | parts 2, 4 and 5 |
///
/// The part numbering is antechamber's own (`bondtype.c::finalize()`), kept so the
/// rules here can be read against the source they came from.
///
/// # Two order sources: the input's, or antechamber's
///
/// * [`BondOrderSource::Connectivity`] is `bondtype -j full`, what
///   antechamber runs by default (`-j 4`, and always for `-c bcc`): the bond
///   orders are judged from the connectivity alone ([`perceive_bond_orders`]),
///   the input's ignored; aromatic rings for part 1 are antechamber's own
///   ring classes ([`perceive_ring_classes`]) of the input's bond types; and part 3 is
///   written as `bondtype` writes it.
///   On the file antechamber reads, the bond types are antechamber's, bond
///   for bond — including the Kekulé structure of a molecule that has two
///   (azulene, cyclooctatetraene), which the order of the atoms and bonds
///   decides, as it does for antechamber.
/// * [`BondOrderSource::Stated`] keeps the input's orders: a localized bond is typed
///   by the number it states, an aromatic one by the Kekulé structure molrs
///   derives for it (below), promoted on molrs's own aromaticity, with part 3
///   repaired (below). Its answer does not depend on the order of the bonds.
///
/// The alphabet both emit is `{1, 2, 3, 6, 7, 8, 9}`. Two further values exist
/// in the tables:
///
/// * **10** is not a peer of 7/8 — it is the *unresolved* aromatic precursor (the
///   SYBYL `ar` input token). [`BondOrderSource::Stated`] resolves it into 7 or 8;
///   [`BondOrderSource::Connectivity`] keeps it only where antechamber
///   does: an `ar` bond of a molecule no valence state closes.
/// * **11** occupies 26 same-type diagonal rows of `BCCPARM.DAT`, all with a
///   correction of exactly `0.0000`, and no rule reaches it.
///
/// # Aromatic promotion is not "is the bond aromatic?"
///
/// A bond is promoted to 7/8 only when **both** endpoints are aromatic *and* they
/// share a ring of size **5 or 6 in which every ring atom is aromatic**. So
/// biphenyl's inter-ring bond and every bond of a 7-membered aromatic ring stay
/// 1/2 (part 1 of the perception). antechamber's "aromatic" is a ring class
/// (AR1 / AR2) rather than Hückel aromaticity: a quinone ring read from a
/// connectivity-only file is AR1 and promoted; the five-ring of an indole is
/// taken out by the AM1-BCC indole rule and is not.
///
/// # Kekulé structures, when the input's orders are kept
///
/// An aromatic input carries no Kekulé structure (order 1.5), so
/// [`BondOrderSource::Stated`] derives one by minimising the valence-state penalty
/// (`APS.DAT`) over the aromatic subsystem. Which of 7/8 a given ring bond ends
/// up with is *charge-invariant* — `BCCPARM` stores identical corrections for
/// types 7, 8 and 10 — but the **atom types** are not: a heteroaromatic ring
/// with two degenerate Kekulé structures (imidazolium) puts the N–C double bond
/// on a different nitrogen in each, and those two nitrogens type differently.
/// The tie-break is calibrated to AmberTools on the simple heteroaromatics; for
/// antechamber's own answer on any input, use
/// [`BondOrderSource::Connectivity`].
///
/// # Provenance
///
/// A reimplementation of the perception in AmberTools' `antechamber/bondtype.c`
/// (`finalize()` and the `conjatom[]` flags), written by reading that source with
/// the AmberTools developers' permission; see `.claude/notes/notes.md`
/// (2026-07-12) for the licensing posture.
///
/// # Type 6 — the order-dependence fix of [`BondOrderSource::Stated`]
///
/// `bondtype.c`'s type-6 rule (`/*part3*/`) has two defects that make its output
/// depend on the order the bonds appear in the input file:
///
/// 1. the neighbour scan's `break` is unbraced, so it stops after the *first*
///    non-partner neighbour — the answer depends on `con[]` order;
/// 2. the mirrored branch (partner stored first) assigns `type = 6`
///    *unconditionally*, above a loop whose result it then ignores.
///
/// Measured consequences, against AmberTools25: **nitrate**'s two topologically
/// identical single-bonded O⁻ receive *different* types (6 and 9), splitting their
/// final charges by 0.28 e; and **pyridine-N-oxide**'s N–O bond types as 6 or 9
/// purely according to whether the file wrote that bond as `O-N` or `N-O`.
///
/// [`BondOrderSource::Stated`] repairs both: the neighbour scan is **exhaustive**, and
/// the rule is **symmetric in the bond's endpoints** (nitrite stays 6/6,
/// nitromethane 9/9, nitrobenzene 9/9, TMAO 9).
/// [`BondOrderSource::Connectivity`] keeps both defects: its answer already
/// follows the input's order, as antechamber's does, and reproducing antechamber
/// means reproducing them.
///
/// # The perceived type is a *perceived fact*, and lives in its own key
///
/// The type is written to [`BCC_BOND_TYPE`] — never to the bond's [`keys::TYPE`](crate::core::keys::TYPE),
/// which belongs to the **caller**: it is where a bond's force-field type *name*
/// (`c3-c3`) or a reader's LAMMPS bond-type id lives, and it is what `to_frame` puts
/// in the `bonds` block's `type` column for every bonded kernel to resolve its
/// parameters by. Two facts, two keys — a
/// component column is typed by its first write and molrs (deliberately) refuses to
/// coerce it, so an `i32` BCC code sitting in `type` makes the molecule unusable for
/// the force field that must later put a `String` name there.
///
/// This is the same rule the charge models keep (`ff::charge`, ac-004): perception
/// neither reads nor writes `keys::TYPE`, so a molecule's own labels — GAFF names,
/// hostile LAMMPS ids, nothing at all — survive perception **byte-identical** and
/// cannot steer its answer.
///
/// # Arguments
///
/// * `mol` — the molecule to perceive; on return every bond carries a
///   [`BCC_BOND_TYPE`] prop in `{1, 2, 3, 6, 7, 8, 9}` (10 only as
///   [`BondOrderSource::Connectivity`] says).
/// * `source` — where the bond orders come from.
///
/// # Errors
///
/// [`MolRsError`] when a type or an aromatic flag cannot be written (a column
/// of `mol` holding the key under a foreign dtype); `mol` may then be partly
/// annotated.
///
/// # Determinism
///
/// With [`BondOrderSource::Stated`] the result is invariant under any
/// permutation of the input's bonds and under swapping any bond's endpoints. It
/// is keyed on atom order alone.
///
/// # Examples
///
/// ```
/// use molrs::core::Atomistic;
/// use molrs::perceive::{BondOrderSource, assign_bcc_bond_types};
/// use molrs::core::keys::BCC_BOND_TYPE;
/// use molrs::core::keys;
/// use molrs::core::{BondNumber, BondOrder, Element};
/// use molrs::props;
///
/// // Acetate: both C–O bonds are delocalized, so both oxygens must correct
/// // identically.
/// let mut mol = Atomistic::new();
/// let atom = |el: Element, [x, y, z]: [f64; 3]| {
///     props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
/// };
/// let c = mol.add_atom(atom(Element::C, [0.86, 0.12, 0.13])).unwrap();
/// let o1 = mol.add_atom(atom(Element::O, [1.18, 1.25, 0.59])).unwrap();
/// let o2 = mol.add_atom(atom(Element::O, [1.59, -0.86, -0.19])).unwrap();
/// let me = mol.add_atom(atom(Element::C, [-0.63, -0.09, -0.09])).unwrap();
/// let double = props! { keys::BOND_TYPE => BondOrder::Double, keys::BOND_NUMBER => BondNumber::Double };
/// let b1 = mol.add_bond(c, o1, double).unwrap();
/// let b2 = mol.add_bond(c, o2, props! {}).unwrap();
/// mol.add_bond(c, me, props! {}).unwrap();
///
/// assign_bcc_bond_types(&mut mol, BondOrderSource::Stated).unwrap();
/// let bond_type = |b| mol.get_bond(b).unwrap().props.get(BCC_BOND_TYPE).unwrap().as_f64();
/// assert_eq!(bond_type(b1), Some(9.0)); // was a double
/// assert_eq!(bond_type(b2), Some(9.0)); // was a single
/// ```
///
/// The caller's own bond labels are untouched — perception neither reads nor
/// writes [`keys::TYPE`](crate::core::keys::TYPE), so a molecule already carrying force-field bond-type
/// *names* (a `String` column an `i32` could never share) survives it unchanged
/// and is still usable to build a force field:
///
/// ```
/// use molrs::core::Atomistic;
/// use molrs::perceive::{BondOrderSource, assign_bcc_bond_types};
/// use molrs::core::keys::BCC_BOND_TYPE;
/// use molrs::core::keys;
/// use molrs::core::{BondNumber, BondOrder, Element};
/// use molrs::core::PropValue;
/// use molrs::props;
///
/// let mut mol = Atomistic::new();
/// let atom = |el: Element, [x, y, z]: [f64; 3]| {
///     props! { keys::ATOMIC_NUMBER => el, keys::X => x, keys::Y => y, keys::Z => z }
/// };
/// let c = mol.add_atom(atom(Element::C, [0.86, 0.12, 0.13])).unwrap();
/// let o1 = mol.add_atom(atom(Element::O, [1.18, 1.25, 0.59])).unwrap();
/// let o2 = mol.add_atom(atom(Element::O, [1.59, -0.86, -0.19])).unwrap();
/// let me = mol.add_atom(atom(Element::C, [-0.63, -0.09, -0.09])).unwrap();
/// let b1 = mol
///     .add_bond(c, o1, props! {
///         keys::BOND_TYPE => BondOrder::Double,
///         keys::BOND_NUMBER => BondNumber::Double,
///         keys::TYPE => "c-o", // the caller's FF bond type NAME
///     })
///     .unwrap();
/// let b2 = mol.add_bond(c, o2, props! { keys::TYPE => "c-o" }).unwrap();
/// mol.add_bond(c, me, props! { keys::TYPE => "c-c3" }).unwrap();
///
/// assign_bcc_bond_types(&mut mol, BondOrderSource::Stated).unwrap();
/// let props = |b| mol.get_bond(b).unwrap().props;
/// // The perceived fact went to its own key …
/// assert_eq!(props(b1).get(BCC_BOND_TYPE).unwrap().as_f64(), Some(9.0));
/// assert_eq!(props(b2).get(BCC_BOND_TYPE).unwrap().as_f64(), Some(9.0));
/// // … and the caller's name came through byte-identical.
/// let name = PropValue::Str("c-o".to_owned());
/// assert_eq!(props(b1).get(keys::TYPE), Some(&name));
/// assert_eq!(props(b2).get(keys::TYPE), Some(&name));
/// ```
pub fn assign_bcc_bond_types(
    mol: &mut Atomistic,
    source: BondOrderSource,
) -> Result<(), MolRsError> {
    if mol.n_bonds() == 0 {
        return Ok(());
    }
    let (bond_ids, types) = match source {
        BondOrderSource::Stated => {
            if !has_aromatic_marking(mol) {
                rewrite_aromatic_flags(mol)?;
            }
            let graph = BondGraph::new(mol);
            let types = graph.perceive(Seed::Input);
            (graph.bond_ids, types)
        }
        BondOrderSource::Connectivity => {
            let judged = perceive_bond_orders(mol);
            let graph = BondGraph::new(mol);

            // Which rings are aromatic is decided before any bond is judged,
            // from the bond types of the file antechamber reads: the stated
            // number, `ar` (10) for an aromatic bond that states none, single
            // otherwise.
            let stated: Vec<(usize, usize, i32)> = graph
                .ends
                .iter()
                .zip(&graph.bond_ids)
                .map(|((i, j), bid)| {
                    let stated = mol.bond_number(*bid)?.count();
                    let t = if (1..=3).contains(&stated) {
                        stated as i32
                    } else if graph.aromatic[graph.bond_index[bid]] {
                        AROMATIC_UNRESOLVED
                    } else {
                        SINGLE
                    };
                    Ok((*i, *j, t))
                })
                .collect::<Result<_, MolRsError>>()?;
            let rings = perceive_ring_classes(&graph.z, &graph.adj, &stated, true);

            // A residue no valence state closes keeps the file's types, as
            // antechamber keeps them.
            let kekule: Vec<i32> = judged
                .iter()
                .zip(&stated)
                .map(|(order, (_, _, t))| order.map_or(*t, i32::from))
                .collect();
            let types = graph.perceive(Seed::Judged {
                orders: &kekule,
                rings: &rings,
            });
            (graph.bond_ids, types)
        }
    };

    for (bid, ty) in bond_ids.into_iter().zip(types) {
        mol.set_bond(bid, BCC_BOND_TYPE, ty)?;
    }
    Ok(())
}

/// What [`BondGraph::perceive`] seeds the bond types from.
#[derive(Clone, Copy)]
enum Seed<'a> {
    /// The input's own orders, the aromatic bonds kekulized, promoted on
    /// molrs's aromaticity.
    Input,
    /// Orders antechamber's search judged, promoted on antechamber's rings.
    Judged {
        orders: &'a [i32],
        rings: &'a RingClasses,
    },
}

impl BondGraph {
    /// A terminal (one-bonded) oxygen or sulfur — the chalcogen every
    /// delocalization rule is written around.
    fn is_terminal_chalcogen(&self, atom: usize) -> bool {
        self.degree(atom) == 1 && matches!(self.z[atom], 8 | 16)
    }

    /// How many terminal O/S an atom carries.
    fn terminal_chalcogens(&self, atom: usize) -> usize {
        self.adj[atom]
            .iter()
            .filter(|nb| self.is_terminal_chalcogen(**nb))
            .count()
    }

    /// Run the whole perception, returning one BCC type per bond in bond order.
    fn perceive(&self, seed: Seed<'_>) -> Vec<i32> {
        let mut types = match seed {
            Seed::Input => self.seed_types(),
            Seed::Judged { orders, .. } => orders
                .iter()
                .map(|o| match *o {
                    AROMATIC_UNRESOLVED => AROMATIC_UNRESOLVED,
                    o => o.clamp(SINGLE, TRIPLE),
                })
                .collect(),
        };
        let conjugated = self.conjugated_centers();
        match seed {
            Seed::Input => self.promote_aromatic(&mut types),
            Seed::Judged { rings, .. } => self.promote_rings(&mut types, rings),
        }

        for k in 0..self.ends.len() {
            if self.delocalize_double(k, &mut types, &conjugated) {
                continue;
            }
            let claimed = match seed {
                Seed::Input => self.type_n_chalcogen(k, &mut types, &conjugated),
                Seed::Judged { .. } => self.type_n_chalcogen_as_bondtype(k, &mut types),
            };
            if claimed {
                continue;
            }
            self.delocalize_single(k, &mut types);
        }
        self.delocalize_hypervalent(&mut types);
        types
    }

    /// Seed every bond from its (Kekulé) bond order: 1/2/3 → 1/2/3.
    ///
    /// Aromatic bonds have no Kekulé order of their own, so they take the one
    /// [`kekulize`] derives. A non-integral, non-aromatic order (which no chemistry
    /// produces) rounds to its nearest neighbour rather than failing the whole
    /// molecule.
    fn seed_types(&self) -> Vec<i32> {
        let doubles = kekulize(self);
        self.ends
            .iter()
            .enumerate()
            .map(|(k, _)| {
                if self.aromatic[k] {
                    if doubles[k] { DOUBLE } else { SINGLE }
                } else {
                    (self.order[k].round() as i32).clamp(SINGLE, TRIPLE)
                }
            })
            .collect()
    }

    /// The `conjatom[]` flag: atoms whose bonds to terminal chalcogens are
    /// resonance-averaged rather than localized.
    ///
    /// A carboxylate carbon, a phosphate phosphorus, a sulfonate sulfur and a nitro
    /// nitrogen — each identified purely by degree and by how many terminal O/S it
    /// carries. (The thresholds are antechamber's, including its comment that
    /// `S` needs *three* terminal chalcogens so that SO₂'s S=O bonds are **not**
    /// delocalized.)
    fn conjugated_centers(&self) -> Vec<bool> {
        (0..self.z.len())
            .map(|a| {
                let chalcogens = self.terminal_chalcogens(a);
                match (self.z[a], self.degree(a)) {
                    (6, 3) => chalcogens >= 2,  // carboxylate / thiocarboxylate C
                    (15, 4) => chalcogens >= 2, // phosphate P
                    (16, 4) => chalcogens >= 3, // sulfonate S — not SO2
                    (7, 3) => chalcogens >= 2,  // nitro / nitrate N
                    _ => false,
                }
            })
            .collect()
    }

    /// **Part 1 — aromatic promotion.** 1 → 7 and 2 → 8, but only inside a genuine
    /// aromatic ring.
    ///
    /// The boundary is deliberately narrow: both endpoints must be aromatic **and**
    /// share a ring of size 5 or 6 **every** atom of which is aromatic. Biphenyl's
    /// inter-ring bond joins two aromatic atoms but lies in no ring, and a
    /// 7-membered aromatic ring is out of range — neither is promoted.
    fn promote_aromatic(&self, types: &mut [i32]) {
        let aromatic_rings: Vec<&Vec<usize>> = self
            .rings
            .iter()
            .filter(|ring| {
                matches!(ring.len(), 5 | 6) && ring.iter().all(|a| self.aromatic_atom[*a])
            })
            .collect();

        for (k, (i, j)) in self.ends.iter().copied().enumerate() {
            if !(self.aromatic_atom[i] && self.aromatic_atom[j]) {
                continue;
            }
            let shared = aromatic_rings
                .iter()
                .any(|ring| ring.contains(&i) && ring.contains(&j));
            if !shared {
                continue;
            }
            if types[k] == SINGLE {
                types[k] = AROMATIC_SINGLE;
            } else if types[k] == DOUBLE {
                types[k] = AROMATIC_DOUBLE;
            }
        }
    }

    /// **Part 1, as `bondtype` runs it** on a structure it judged: a bond whose
    /// ends are both AR1, or both AR2 (antechamber's ring classes,
    /// [`perceive_ring_classes`]), inside a five- or six-ring all of whose atoms are AR1
    /// or AR2, is promoted 1 → 7, 2 → 8.
    fn promote_rings(&self, types: &mut [i32], rings: &RingClasses) {
        let f = &rings.atoms;
        let aromatic = |a: usize| f[a].ar[0] > 0 || f[a].ar[1] > 0;
        for (k, (i, j)) in self.ends.iter().copied().enumerate() {
            if !((f[i].ar[0] > 0 && f[j].ar[0] > 0) || (f[i].ar[1] > 0 && f[j].ar[1] > 0)) {
                continue;
            }
            let shared = rings.rings.iter().any(|ring| {
                matches!(ring.num, 5 | 6)
                    && ring.members().iter().all(|a| aromatic(*a))
                    && ring.members().contains(&i)
                    && ring.members().contains(&j)
            });
            if shared {
                if types[k] == SINGLE {
                    types[k] = AROMATIC_SINGLE;
                } else if types[k] == DOUBLE {
                    types[k] = AROMATIC_DOUBLE;
                }
            }
        }
    }

    /// **Part 2 — delocalized double.** A double bond from a conjugated centre to a
    /// terminal O/S is delocalized: the carboxylate C=O, the nitro N=O.
    ///
    /// Returns whether the bond was claimed.
    fn delocalize_double(&self, k: usize, types: &mut [i32], conjugated: &[bool]) -> bool {
        if types[k] != DOUBLE {
            return false;
        }
        let (i, j) = self.ends[k];
        let claimed = (conjugated[i] && self.is_terminal_chalcogen(j))
            || (conjugated[j] && self.is_terminal_chalcogen(i));
        if claimed {
            types[k] = DELOCALIZED;
        }
        claimed
    }

    /// **Part 3 — type 6.** A bond from a nitrogen of degree 2 or 3 to a terminal
    /// O/S, where that nitrogen carries a *second* terminal chalcogen: nitrite, and
    /// only its kin.
    ///
    /// Two deliberate repairs to `bondtype.c` (see the module docs of `perceive::bcc_bond_class`):
    ///
    /// * the second-chalcogen scan is **exhaustive**, not "first non-partner
    ///   neighbour only" — antechamber's `break` is outside its `if`, which makes
    ///   the answer depend on the order the bonds were listed;
    /// * the rule is **symmetric in the endpoints** — antechamber's mirrored branch
    ///   assigns 6 unconditionally, so the same molecule types differently
    ///   depending on whether the file wrote `O-N` or `N-O`.
    ///
    /// The conjugated gate is what keeps the repair from over-firing: a *nitro*
    /// nitrogen also carries a second terminal chalcogen, but its bonds are
    /// delocalized (type 9), which is exactly what `conjatom[]` says and what parts
    /// 2 and 4 deliver. Without the gate, nitromethane's N–O would become 6 — a
    /// 0.28 e error on an oracle molecule.
    ///
    /// Returns whether the bond was claimed.
    fn type_n_chalcogen(&self, k: usize, types: &mut [i32], conjugated: &[bool]) -> bool {
        let (i, j) = self.ends[k];
        let Some((n, chalcogen)) = self.nitrogen_chalcogen_ends(i, j) else {
            return false;
        };
        if conjugated[n] {
            return false;
        }
        let second = self.adj[n]
            .iter()
            .any(|nb| *nb != chalcogen && self.is_terminal_chalcogen(*nb));
        if second {
            types[k] = N_CHALCOGEN;
        }
        second
    }

    /// **Part 3, as `bondtype` writes it** — for the structure it judged, where
    /// the answer already follows the input's atom and bond order, so the
    /// order-dependence the repaired [`type_n_chalcogen`](Self::type_n_chalcogen)
    /// removes is part of what is being reproduced.
    ///
    /// With the bond stored N → O/S, it is 6 when the nitrogen's first other
    /// neighbour (in `con[]` order) is itself a terminal O/S; stored O/S → N, it
    /// is 6 outright (pyridine N-oxide written `O-N`).
    fn type_n_chalcogen_as_bondtype(&self, k: usize, types: &mut [i32]) -> bool {
        let (i, j) = self.ends[k];
        let is_n = |a: usize| self.z[a] == 7 && matches!(self.degree(a), 2 | 3);
        if is_n(i) && self.is_terminal_chalcogen(j) {
            let first_other = self.adj[i].iter().find(|nb| **nb != j);
            if first_other.is_some_and(|nb| self.is_terminal_chalcogen(*nb)) {
                types[k] = N_CHALCOGEN;
                return true;
            }
        }
        if is_n(j) && self.is_terminal_chalcogen(i) {
            types[k] = N_CHALCOGEN;
            return true;
        }
        false
    }

    /// Orient a bond as `(nitrogen, terminal chalcogen)` if it is one, in whichever
    /// order its endpoints were stored.
    fn nitrogen_chalcogen_ends(&self, i: usize, j: usize) -> Option<(usize, usize)> {
        let is_n = |a: usize| self.z[a] == 7 && matches!(self.degree(a), 2 | 3);
        if is_n(i) && self.is_terminal_chalcogen(j) {
            Some((i, j))
        } else if is_n(j) && self.is_terminal_chalcogen(i) {
            Some((j, i))
        } else {
            None
        }
    }

    /// **Part 4 — delocalized single.** A single bond to a terminal O/S is
    /// delocalized: the carboxylate C–O⁻, the nitro N–O⁻.
    ///
    /// This is the partner of part 2, and together they are what make the type-9
    /// answer *resonance-invariant*: whichever of a carboxylate's two C–O bonds the
    /// Kekulé structure happened to call the double one, both come out 9.
    fn delocalize_single(&self, k: usize, types: &mut [i32]) {
        if types[k] != SINGLE {
            return;
        }
        let (i, j) = self.ends[k];
        if self.is_terminal_chalcogen(i) || self.is_terminal_chalcogen(j) {
            types[k] = DELOCALIZED;
        }
    }

    /// **Part 5 — hypervalent centres.** A sulfur of degree 3 with exactly two
    /// terminal chalcogens (a sulfone/sulfonamide S), and a phosphorus of degree 4
    /// with at least two (a phosphate P), delocalize *their* bonds to those
    /// chalcogens — whatever type the earlier parts gave them.
    fn delocalize_hypervalent(&self, types: &mut [i32]) {
        for a in 0..self.z.len() {
            let chalcogens = self.terminal_chalcogens(a);
            let hypervalent = match (self.z[a], self.degree(a)) {
                (16, 3) => chalcogens == 2,
                (15, 4) => chalcogens >= 2,
                _ => false,
            };
            if !hypervalent {
                continue;
            }
            for (k, (i, j)) in self.ends.iter().copied().enumerate() {
                let other = if i == a {
                    j
                } else if j == a {
                    i
                } else {
                    continue;
                };
                if self.is_terminal_chalcogen(other) {
                    types[k] = DELOCALIZED;
                }
            }
        }
    }
}
