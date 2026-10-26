//! Kernels: a force field's terms evaluated on coordinates — the
//! [`Potential`] traits, the generic form kernels and the built-in forms they
//! price (`bond`, `angle`, `dihedral`, `improper`, `pair`), PME's
//! reciprocal-space sum and `cmap`, and
//! the weights a non-bonded kernel takes on a frame's special pairs
//! ([`SpecialWeights`]).
//!
//! What a style *is* (its spec) is the IR's ([`crate::ff::ir`]); which kernel
//! prices a style is the style registry's ([`crate::ff::style_registry`]);
//! binding a force field's styles to a frame is the compiler's
//! ([`PotentialCompiler`](crate::ff::compile::PotentialCompiler)); which
//! integrator calls a kernel is `md`'s and `optimize`'s.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via
//! [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile).

pub(crate) mod flat_coords;

pub mod angle;
pub mod bond;
pub mod cmap;
pub mod dihedral;
mod error;
pub mod form_kernel;
pub mod improper;
pub mod kspace;
pub mod pair;
pub(crate) mod param_reads;
pub mod soft;

pub use error::CompileError;

use ndarray::{Array2, ArrayView2};

use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

/// `(lo, hi)` end-atom pairs of a topology block (bond ends, angle i–k,
/// dihedral i–l), one per row in row order. A missing block or column yields
/// none.
pub(crate) fn end_pairs(
    frame: &Frame,
    block: &str,
    col_a: &str,
    col_b: &str,
) -> Vec<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return Vec::new();
    };
    let (Some(a_col), Some(b_col)) = (
        b.get(col_a).and_then(|c| c.get::<u64>()),
        b.get(col_b).and_then(|c| c.get::<u64>()),
    ) else {
        return Vec::new();
    };
    a_col
        .iter()
        .zip(b_col.iter())
        .map(|(&i, &j)| {
            let (i, j) = (i as usize, j as usize);
            (i.min(j), i.max(j))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

/// What an evaluation adds into: the running energy, a force buffer, and —
/// when the caller asked for it — the virial `Σ f ⊗ r`.
///
/// Every evaluation primitive ([`Potential::evaluate`],
/// [`IndexedTerms::evaluate_terms`], [`PairDriven::evaluate_pairs`]) takes
/// one of these and *adds* to it, so energy-only, forces-only, energy and
/// forces, and energy-forces-virial are one call each: build the accumulator
/// for the channels wanted, call the primitive, read the channels back. An
/// aggregate passes the same accumulator to every member, which is what lets
/// a whole step own one force buffer instead of one allocation per member.
///
/// # The channels
///
/// * **energy** — always tallied. It is one add per term.
/// * **forces** — always tallied, into a buffer of `coords.len()` (`3·N`):
///   one the accumulator owns ([`new`](Self::new)), or the caller's
///   ([`adding_into`](Self::adding_into)). Not optional, because no kernel
///   has a cheaper energy-only path: `dE/dr` is the energy's own derivative,
///   evaluated in the same expression, and skipping it would save the stores
///   and nothing else. An energy-only caller reads [`energy`](Self::energy)
///   and drops the buffer.
/// * **virial** — only when requested ([`with_virial`](Self::with_virial)).
///   It is the one channel that costs real work per pair, so a kernel that
///   can tally one asks [`forces_with_virial`](Self::forces_with_virial)
///   whether to, once per call, and runs a loop monomorphised on the answer
///   — the inner loop never branches on it.
///
/// # A missing virial is `None`, never zero
///
/// A kernel adds forces through exactly one of two doors.
/// [`forces_with_virial`](Self::forces_with_virial) is for a kernel that
/// tallies the virial of what it adds; [`forces_without_virial`](Self::forces_without_virial)
/// is for every other one, and taking it marks a requested virial *lost* —
/// [`virial`](Self::virial) then answers `None` for the whole evaluation. A
/// virial missing a term is not a small error, it is a different quantity,
/// and a pressure computed from it would look entirely plausible. Because the
/// door is the declaration, an aggregate needs no bookkeeping of its own: a
/// bonded member, an external field or a compiled pair list loses the virial
/// by adding its forces, and a member that adds no force has no virial to
/// lose.
#[derive(Debug)]
pub struct Accumulator<'a> {
    energy: F,
    forces: ForceBuffer<'a>,
    virial: VirialChannel,
}

/// Where an [`Accumulator`]'s forces land.
#[derive(Debug)]
enum ForceBuffer<'a> {
    /// Zeroed at construction and handed back by [`Accumulator::into_forces`].
    Owned(Vec<F>),
    /// The caller's buffer, added into.
    Borrowed(&'a mut [F]),
}

/// The state of an [`Accumulator`]'s virial channel.
#[derive(Clone, Copy, Debug)]
enum VirialChannel {
    /// Not requested: kernels skip the tally.
    Off,
    /// Requested, and every force added so far came with its virial.
    Open(Virial),
    /// Requested, and some force was added without one.
    Lost,
}

impl Accumulator<'static> {
    /// Energy and forces into a zeroed buffer of `n_coords` (`3·N`)
    /// components the accumulator owns.
    pub fn new(n_coords: usize) -> Self {
        Self {
            energy: 0.0,
            forces: ForceBuffer::Owned(vec![0.0; n_coords]),
            virial: VirialChannel::Off,
        }
    }
}

impl<'a> Accumulator<'a> {
    /// Energy and forces added into `forces`, the caller's buffer — which
    /// keeps whatever it already held.
    pub fn adding_into(forces: &'a mut [F]) -> Self {
        Self {
            energy: 0.0,
            forces: ForceBuffer::Borrowed(forces),
            virial: VirialChannel::Off,
        }
    }

    /// Also tally the virial `Σ f ⊗ r` — see [`virial`](Self::virial).
    pub fn with_virial(mut self) -> Self {
        self.virial = VirialChannel::Open(Virial::ZERO);
        self
    }

    /// The energy added so far.
    pub fn energy(&self) -> F {
        self.energy
    }

    /// The force buffer: `-∇E` of everything added so far, plus whatever a
    /// borrowed buffer held before.
    pub fn forces(&self) -> &[F] {
        match &self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        }
    }

    /// The force buffer as a vector: an owned one moves out; a borrowed one is
    /// copied (its caller already holds it, so has no reason to ask).
    pub fn into_forces(self) -> Vec<F> {
        match self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s.to_vec(),
        }
    }

    /// The virial `Σ f ⊗ r` of everything added so far.
    ///
    /// `None` when it was not requested, or when some contribution added
    /// forces without one (see the type's documentation) — never a zero
    /// standing in for a term nobody tallied.
    pub fn virial(&self) -> Option<Virial> {
        match self.virial {
            VirialChannel::Open(w) => Some(w),
            VirialChannel::Off | VirialChannel::Lost => None,
        }
    }

    /// Kernel side: add `energy` to the running total.
    #[inline]
    pub fn add_energy(&mut self, energy: F) {
        self.energy += energy;
    }

    /// Kernel side: the force buffer, for a kernel that does **not** tally the
    /// virial of what it adds. A requested virial is lost from here on.
    #[inline]
    pub fn forces_without_virial(&mut self) -> &mut [F] {
        if let VirialChannel::Open(_) = self.virial {
            self.virial = VirialChannel::Lost;
        }
        self.forces_mut()
    }

    /// Kernel side: the force buffer and, when the virial was requested and
    /// is still whole, the running virial to add this kernel's `Σ f ⊗ r` to.
    ///
    /// Ask once per call and branch once on the answer, outside the loop.
    #[inline]
    pub fn forces_with_virial(&mut self) -> (&mut [F], Option<&mut Virial>) {
        let virial = match &mut self.virial {
            VirialChannel::Open(w) => Some(w),
            VirialChannel::Off | VirialChannel::Lost => None,
        };
        let forces: &mut [F] = match &mut self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        };
        (forces, virial)
    }

    fn forces_mut(&mut self) -> &mut [F] {
        match &mut self.forces {
            ForceBuffer::Owned(v) => v,
            ForceBuffer::Borrowed(s) => s,
        }
    }
}

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Energy and forces from coordinates alone.
///
/// A `Potential` is **molecule-bound**: its per-element parameters are expanded
/// against the molecule's topology once at [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)
/// (string type labels resolved to per-bond/angle/… arrays). Evaluation
/// therefore takes only coordinates — there is no per-call topology resolution.
///
/// This is the whole of what every potential can do. Two capabilities that only
/// some have are separate traits, so that a caller which needs one says so in a
/// type instead of asking at run time:
///
/// * [`IndexedTerms`] — the rows are named by an index table the caller may
///   replace. Every bonded kernel.
/// * [`PairDriven`] — the sum runs over whatever pairs a neighbour search turns
///   up. Every pair kernel.
///
/// Each of the three has exactly one evaluation primitive, and all three add
/// into an [`Accumulator`]; which channels come back is the accumulator's
/// business, not a family of methods'.
///
/// [`ForceTerm`] is the three of them as one value, chosen when the kernel is
/// built. It exists because a `Box<dyn Potential>` cannot be asked which of the
/// two it also is, and `terms()` is the wrong place to ask: its job is to
/// return a table, and it allocates one per member per call.
///
/// The geometry optimizer ([`crate::optimize::Lbfgs`]) depends on this trait —
/// not the other way around.
pub trait Potential: Send + Sync {
    /// Add this potential's energy and forces (`-∇E`) at `coords` — flat
    /// `[x0, y0, z0, x1, …]` — into `acc`.
    ///
    /// A kernel that tallies no virial adds its forces through
    /// [`Accumulator::forces_without_virial`]; one that does, through
    /// [`Accumulator::forces_with_virial`].
    ///
    /// Required: there is no default, so every implementor states how it
    /// adds its energy and forces.
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>);
}

/// A potential whose rows are named by an index table the caller can replace.
///
/// Implementing this is a declaration that *which atoms* is separable from
/// *what the parameters are*: row `r` of [`terms`](IndexedTerms::terms) belongs
/// with row `r` of the parameters, and a caller may hand back a different table
/// naming different atoms for the same rows. That is what lets a periodic
/// régime point a bond at a periodic copy without the kernel ever learning that
/// copies exist.
///
/// Every bonded kernel implements it; no pair kernel does, because a pair
/// kernel's rows are not fixed — they are whatever the neighbour search found.
pub trait IndexedTerms: Potential {
    /// The atom indices this kernel resolved at construction, `(n_terms,
    /// arity)` — arity 2 for a bond, 3 for an angle, 4 for a dihedral or an
    /// improper.
    fn terms(&self) -> Array2<u32>;

    /// [`Potential::evaluate`] with `terms` in place of the indices resolved
    /// at construction.
    ///
    /// `terms` must have the shape [`terms`](IndexedTerms::terms) returned: the
    /// row *set* is fixed — it is the terms the force field declared — and only
    /// which atoms each row names may differ.
    ///
    /// There is no default. A default that ignored `terms` would be a correct
    /// fallback for a kernel holding no indices and a silently wrong answer for
    /// one that does, and only the second kind is in this trait.
    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>);
}

/// A potential summed over whatever pairs a neighbour search turns up.
///
/// Every pair kernel implements it. Nothing else should: the two required
/// methods are the ones a caller maintaining a live neighbour table needs
/// answered honestly, and a potential that ignores the table cannot answer
/// them.
pub trait PairDriven: Potential {
    /// Add this kernel's contribution over the pair table `pairs` into `acc`,
    /// scaling each pair.
    ///
    /// * **`factor`** is one weight per pair, aligned with the table's rows, or
    ///   empty meaning all ones. A weight of exactly zero *skips* the pair
    ///   rather than multiplying it, because a bonded pair sits at bond length
    ///   where a repulsive term is enormous. Carrying the weights per pair is
    ///   what lets a caller stop rebuilding the pair table once per distinct
    ///   weight.
    /// * **the virial**, when `acc` asks for one, is tallied in the same loop
    ///   that makes the forces. It cannot be recovered from the forces
    ///   afterwards: under periodic boundaries `Σ_a f_a ⊗ x_a` over the stored
    ///   coordinates depends on where the cell's origin happens to be, and a
    ///   sum over separations does not — so a kernel tallies it here, where
    ///   both terms of each pair are still in hand. A kernel that cannot (one
    ///   bound to a compiled pair list) adds through
    ///   [`Accumulator::forces_without_virial`], and the virial reads `None`.
    ///
    /// No default: one that ignored `factor` would be right for a potential
    /// that does not sum over the pair table and would silently lose a force
    /// field's exclusions for one that does. Only the second kind is in this
    /// trait, so the question is asked of every implementor.
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    );

    /// Whether this kernel's parameters are bound to one fixed pair list.
    ///
    /// Such a kernel answers for **that** list and no other. Handed a
    /// neighbour table it does not read it — it returns the frozen sum — and
    /// that is not a cheaper route to the same number, it is a different
    /// question silently unanswered. Worse, the caller goes on maintaining and
    /// reporting a live list that provably does not enter the answer.
    ///
    /// A periodic force path refuses such a member at construction. No default:
    /// a kernel built from one pair list and a kernel keyed on the atoms are
    /// the same Rust type in molrs (the list is a private source variant), so
    /// the answer cannot be inferred and has to be given.
    fn binds_a_fixed_pair_list(&self) -> bool;

    /// Whether pricing this member needs a neighbour table handed to it each
    /// step: true for a kernel over a type table (`PairLjCut`, a
    /// `compile_typed` kernel), false for one bound to a fixed pair list. An
    /// aggregate asks its members ([`Potentials`]).
    fn reads_a_neighbor_table(&self) -> bool {
        !self.binds_a_fixed_pair_list()
    }
}

/// One member of a force evaluation, with its role fixed when it was built.
///
/// A force evaluation sums members that play different parts — a bonded term
/// reads an index table, a pair term reads a neighbour table, an external field
/// reads neither. Which part a member plays is settled by its constructor, and
/// this is where that answer is kept, so no loop has to re-derive it per step
/// and no method has to carry a default that is wrong for half its implementors.
pub enum ForceTerm {
    /// A bonded term: evaluated against an index table the caller supplies.
    Indexed(Box<dyn IndexedTerms>),
    /// A non-bonded term: evaluated against a neighbour table, with per-pair
    /// weights.
    Pair(Box<dyn PairDriven>),
    /// Everything else — an external field, a restraint, a constant force.
    /// Evaluated from coordinates alone.
    Plain(Box<dyn Potential>),
}

impl ForceTerm {
    /// A bonded term — one whose rows are named by an index table.
    pub fn indexed(p: impl IndexedTerms + 'static) -> Self {
        ForceTerm::Indexed(Box::new(p))
    }

    /// A non-bonded term — one summed over a neighbour table.
    pub fn pair(p: impl PairDriven + 'static) -> Self {
        ForceTerm::Pair(Box::new(p))
    }

    /// Anything else — evaluated from coordinates alone.
    pub fn plain(p: impl Potential + 'static) -> Self {
        ForceTerm::Plain(Box::new(p))
    }

    /// This member as a plain potential, whatever part it plays.
    pub fn as_potential(&self) -> &dyn Potential {
        match self {
            ForceTerm::Indexed(p) => &**p,
            ForceTerm::Pair(p) => &**p,
            ForceTerm::Plain(p) => &**p,
        }
    }

    /// The index table, for a bonded member.
    pub fn terms(&self) -> Option<Array2<u32>> {
        match self {
            ForceTerm::Indexed(p) => Some(p.terms()),
            _ => None,
        }
    }

    /// Whether this member is bound to one fixed pair list — see
    /// [`PairDriven::binds_a_fixed_pair_list`]. A member that reads no pair
    /// table is bound to none.
    pub fn binds_a_fixed_pair_list(&self) -> bool {
        match self {
            ForceTerm::Pair(p) => p.binds_a_fixed_pair_list(),
            _ => false,
        }
    }
}

impl std::fmt::Debug for ForceTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ForceTerm::Indexed(_) => "ForceTerm::Indexed",
            ForceTerm::Pair(_) => "ForceTerm::Pair",
            ForceTerm::Plain(_) => "ForceTerm::Plain",
        })
    }
}

impl Potential for ForceTerm {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        self.as_potential().evaluate(coords, acc);
    }
}

impl Potential for Box<dyn Potential> {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        (**self).evaluate(coords, acc);
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// A kernel built for a neighbour-driven evaluation, and which of a force
/// field's special-bonds weight sets scales it. `None` for a bonded kernel:
/// it *is* the bonded interaction, not a scaled copy of one.
pub type ScaledTerm = (ForceTerm, Option<crate::ff::ir::SpecialClass>);

/// The weights a force field puts on close non-bonded neighbours.
///
/// A bonded pair's non-bonded term is not wanted at full strength: 1-2 and 1-3
/// are normally excluded outright and 1-4 scaled, because the bonded terms
/// already describe those interactions. A compiled intramolecular list carries
/// that by *omitting* the excluded rows and baking the 1-4 factor into the
/// parameters — which works only while that exact list is the one being
/// evaluated. A neighbour table has no such memory: it finds every pair inside
/// the cutoff, bonded or not.
///
/// So the weights have to be applied at evaluation time, and this holds them:
/// the frame's special pairs (its `pairs` block) as a force field weighs them,
/// which [`PotentialCompiler::compile_typed`](crate::ff::compile::PotentialCompiler::compile_typed)
/// hands out per member.
///
/// # Why it splits the table rather than scaling in the kernel
///
/// Energy is a sum over pairs, so scaling a group of pairs and scaling their
/// contribution are the same number — which means the weights can be applied
/// *outside* the kernels, and no kernel has to learn that a force field has
/// exclusions.
///
/// The obvious cheaper trick — evaluate everything, then subtract what should
/// not have been counted — is not available. A 1-2 pair sits at bond length,
/// where a Lennard-Jones term is enormous; subtracting it from a total of
/// ordinary size cancels away the very digits the answer is made of.
#[derive(Clone, Debug, PartialEq)]
pub struct SpecialWeights {
    /// Per owned atom, its special partners sorted by index, with weights.
    per_atom: Vec<Vec<(u32, F)>>,
    /// Whether every list is empty. A fact about the table, not about a step.
    nothing_scaled: bool,
}

/// Nothing scaled — the right default, and not the one `derive` would give.
///
/// `nothing_scaled` is a cached answer, and a derived `bool` default is
/// `false`: an empty table would have claimed to scale something. It cost only
/// a slower path, because an empty lookup answers 1.0 for every pair — but a
/// cached fact that disagrees with the thing it caches is a trap whoever
/// trusts it next will fall into.
impl Default for SpecialWeights {
    fn default() -> Self {
        Self {
            per_atom: Vec::new(),
            nothing_scaled: true,
        }
    }
}

impl SpecialWeights {
    /// Take per-atom lists of `(partner, weight)`, each sorted by partner.
    pub fn new(special: &[Vec<(usize, F)>]) -> Self {
        let per_atom: Vec<Vec<(u32, F)>> = special
            .iter()
            .map(|l| l.iter().map(|&(p, w)| (p as u32, w)).collect())
            .collect();
        Self {
            nothing_scaled: per_atom.iter().all(|l| l.is_empty()),
            per_atom,
        }
    }

    /// The weight on the pair of atoms `(i, j)`. `1.0` when the two
    /// are far enough apart in the bond graph to interact normally.
    pub fn weight(&self, i: usize, j: usize) -> F {
        let Some(list) = self.per_atom.get(i) else {
            return 1.0;
        };
        match list.binary_search_by_key(&(j as u32), |&(p, _)| p) {
            Ok(k) => list[k].1,
            Err(_) => 1.0,
        }
    }

    /// Every scaled pair once, `(i, j, weight)` with `i < j`.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize, F)> + '_ {
        self.per_atom.iter().enumerate().flat_map(|(i, list)| {
            list.iter()
                .filter(move |&&(p, _)| p as usize > i)
                .map(move |&(p, w)| (i, p as usize, w))
        })
    }

    /// True when nothing is scaled, so a caller can skip the weights entirely.
    ///
    /// Answered from a flag set at construction: it is a property of the table
    /// and was being recomputed by scanning every atom, once per member, once
    /// per step.
    pub fn is_empty(&self) -> bool {
        self.nothing_scaled
    }

    /// Fill `out` with one weight per row of `pairs`.
    ///
    /// # Why a column and not a split
    ///
    /// Partitioning the table into a full-strength one and a group per
    /// distinct weight is also correct, because energy is a sum over pairs and
    /// scaling a group is the same as scaling its contribution. It costs the
    /// table being rebuilt — allocated, re-pushed column by column — once per
    /// weight per member per step. Measured at 4 096 atoms that is four times
    /// the kernel it prepares input for, and thirty megabytes a step.
    ///
    /// A weight is one number per pair. Handing the kernel that number is one
    /// pass over a buffer the caller keeps.
    /// The per-pair weights for `pairs`, or an empty slice when nothing is
    /// scaled.
    ///
    /// The empty slice is not a table of ones: a kernel reads it as "no weights
    /// apply" and skips the multiply entirely, which is the common case and the
    /// one worth not paying for. `scratch` is the caller's buffer, reused
    /// across steps — [`fill_factors`](Self::fill_factors) is what fills it.
    pub fn factors_for<'a>(&self, pairs: &Neighbors, scratch: &'a mut Vec<F>) -> &'a [F] {
        if self.is_empty() {
            return &[];
        }
        self.fill_factors(pairs, scratch);
        scratch
    }

    pub fn fill_factors(&self, pairs: &Neighbors, out: &mut Vec<F>) {
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        out.clear();
        out.reserve(i_col.len());
        for p in 0..i_col.len() {
            let (i, j) = (i_col[p] as usize, j_col[p] as usize);
            // An atom and its own periodic image are a real interaction, and
            // no bond-graph weight describes it: the walk is root-inclusive,
            // so `weight(i, i)` would answer 0 — the weight of an atom with
            // itself, a different question from the one this pair asks.
            out.push(if i == j { 1.0 } else { self.weight(i, j) });
        }
    }
}

/// Aggregates multiple potentials; energy/forces are summed.
///
/// Every member evaluates in one shared unit system. Unit conversion is the
/// caller's job (`UnitPreset`), never this type's.
pub struct Potentials {
    /// Each member with the part it plays, settled when it was built.
    ///
    /// The role is settled at construction, so it is kept as the member's
    /// [`ForceTerm`] variant rather than recovered per step from `terms()`.
    inner: Vec<ForceTerm>,
    /// Number of atoms the kernels were compiled against (`coords.len() / 3`).
    /// `0` when unknown (e.g. built incrementally via [`Potentials::push`]).
    n_atoms: usize,
}

impl std::fmt::Debug for Potentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potentials")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Potentials {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            n_atoms: 0,
        }
    }

    /// Add a member. Which part it plays is [`ForceTerm`]'s to say, and its
    /// constructor already said it.
    pub fn push(&mut self, member: ForceTerm) {
        self.inner.push(member);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether every pair member is bound to its pair list, so the
    /// collection can be priced from coordinates alone
    /// ([`Potential::evaluate`]). A pair kernel built over a type table —
    /// `PairLjCut`, a `compile_typed` kernel — reads a neighbour table each
    /// step and adds nothing to a coordinates-only evaluation; a door that
    /// evaluates from coordinates (a minimizer, `Potentials::evaluate`)
    /// asks here first.
    ///
    /// # Errors
    ///
    /// [`CompileError::Invalid`] naming how many members read a neighbour
    /// table, and where they are priced instead.
    pub fn require_pair_lists(&self) -> Result<(), CompileError> {
        let unbound = self
            .inner
            .iter()
            .filter(|m| matches!(m, ForceTerm::Pair(p) if p.reads_a_neighbor_table()))
            .count();
        if unbound == 0 {
            return Ok(());
        }
        Err(CompileError::Invalid(format!(
            "{unbound} pair member(s) read a neighbour table and add nothing to an evaluation \
             from coordinates alone: price them through an MD evaluator with a NeighborList \
             or VerletSkin, or compile the force field over the frame's pair list"
        )))
    }

    /// Number of atoms the kernels were compiled against, i.e. the expected
    /// `coords.len() / 3`. Returns `0` if unknown (built via [`push`]).
    ///
    /// [`push`]: Potentials::push
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Record the compiled atom count (used by [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)).
    pub fn set_n_atoms(&mut self, n_atoms: usize) {
        self.n_atoms = n_atoms;
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The members, in the order the force field's styles produced them.
    ///
    /// A caller that needs to treat members differently — a periodic régime
    /// rebinding the bonded terms onto copies while the nonbonded one reads a
    /// neighbour table — needs them one at a time, because the index table a
    /// member wants is the member's own. Summing them all is what
    /// [`Potential::evaluate`] on the aggregate is for.
    pub fn members(&self) -> &[ForceTerm] {
        &self.inner
    }

    /// Give up the members, for a caller that wants to own them individually.
    pub fn into_members(self) -> Vec<ForceTerm> {
        self.inner
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

/// Make the aggregate usable wherever a single [`Potential`] is expected (the
/// geometry optimizer in [`crate::optimize`], an MD integrator, another
/// [`Potentials`]): every member adds into the one accumulator it is handed —
/// one buffer for the whole aggregate, no allocation per member.
impl Potential for Potentials {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        for member in &self.inner {
            member.evaluate(coords, acc);
        }
    }
}

/// An aggregate is pair-driven when it is asked to be: it forwards to the
/// members that read a pair table and evaluates the rest the ordinary way.
///
/// The split is the member's own [`ForceTerm`] variant, which its
/// constructor chose.
impl PairDriven for Potentials {
    /// Every member adds into the same accumulator.
    ///
    /// A per-pair weight belongs to a member that reads the pair table. A
    /// bonded member ignores the table, and its own interaction is the thing
    /// the weights exist to avoid double-counting — scaling it would be
    /// scaling the wrong side of that.
    ///
    /// The virial needs no bookkeeping here: a pair member tallies its own,
    /// and every other member adds its forces through
    /// [`Accumulator::forces_without_virial`], which makes the aggregate's
    /// virial `None` — the honest answer when one member cannot report.
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        for member in &self.inner {
            match member {
                ForceTerm::Pair(p) => p.evaluate_pairs(coords, pairs, factor, acc),
                other => other.evaluate(coords, acc),
            }
        }
    }

    /// True if *any* member is. One compiled kernel is enough to make the
    /// aggregate's answer independent of the table it is handed.
    fn binds_a_fixed_pair_list(&self) -> bool {
        self.inner.iter().any(ForceTerm::binds_a_fixed_pair_list)
    }

    fn reads_a_neighbor_table(&self) -> bool {
        self.inner
            .iter()
            .any(|m| matches!(m, ForceTerm::Pair(p) if p.reads_a_neighbor_table()))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::ir::SpecialBonds;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::schema::block_names::PAIRS;

    /// The default scales nothing, and says so.
    ///
    /// `nothing_scaled` caches an answer, and a cached fact that disagrees with
    /// the thing it caches is worse than no cache: a derived `bool` default is
    /// `false`, so an empty table claimed to have weights and a provider with
    /// no neighbour list refused to build. The Python suite found it; this
    /// keeps it found.
    #[test]
    fn an_empty_weight_table_scales_nothing() {
        assert!(SpecialWeights::default().is_empty());
        assert!(SpecialWeights::new(&[]).is_empty());
        assert!(SpecialWeights::new(&[vec![], vec![]]).is_empty());
        assert!(!SpecialWeights::new(&[vec![(1_usize, 0.5_f64)], vec![]]).is_empty());
        // And an absent entry still answers full strength.
        assert_eq!(SpecialWeights::default().weight(0, 1), 1.0);
    }

    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::Params;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn evaluate(&self, _coords: &[F], acc: &mut Accumulator<'_>) {
            acc.add_energy(self.value);
            for f in acc.forces_without_virial() {
                *f += self.value;
            }
        }
    }

    fn make_atoms_only_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 2.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);
        frame
    }

    /// A four-atom frame carrying one term of every bonded arity, so a single
    /// force field produces a bond, an angle, a dihedral and an improper
    /// kernel and the rebinding contract can be checked on all four at once.
    fn make_all_bonded_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        // A non-planar, non-collinear arrangement: a dihedral and an improper
        // are both undefined on degenerate geometry, and a test that fed them
        // one would be asserting against NaN.
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.5, 2.1, 3.4]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 0.2, 1.4, 1.1]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.9, 0.3, 1.8]).into_dyn(),
                None,
            )
            .unwrap();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let idx = |v: Vec<Idx>| Array1::from_vec(v).into_dyn();
        let ty = |n: usize, name: &str| Array1::from_vec(vec![name.to_string(); n]).into_dyn();

        let mut bonds = Block::new();
        bonds.insert("atomi", idx(vec![0, 1, 2]), None).unwrap();
        bonds.insert("atomj", idx(vec![1, 2, 3]), None).unwrap();
        bonds.insert("type", ty(3, "A-A"), None).unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles.insert("atomi", idx(vec![0, 1]), None).unwrap();
        angles.insert("atomj", idx(vec![1, 2]), None).unwrap();
        angles.insert("atomk", idx(vec![2, 3]), None).unwrap();
        angles.insert("type", ty(2, "A-A-A"), None).unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals.insert("atomi", idx(vec![0]), None).unwrap();
        dihedrals.insert("atomj", idx(vec![1]), None).unwrap();
        dihedrals.insert("atomk", idx(vec![2]), None).unwrap();
        dihedrals.insert("atoml", idx(vec![3]), None).unwrap();
        dihedrals.insert("type", ty(1, "A-A-A-A"), None).unwrap();
        frame.insert("dihedrals", dihedrals);

        let mut impropers = Block::new();
        impropers.insert("atomi", idx(vec![1]), None).unwrap();
        impropers.insert("atomj", idx(vec![0]), None).unwrap();
        impropers.insert("atomk", idx(vec![2]), None).unwrap();
        impropers.insert("atoml", idx(vec![3]), None).unwrap();
        impropers.insert("type", ty(1, "A-A-A-A"), None).unwrap();
        frame.insert("impropers", impropers);

        frame
    }

    /// Handing a kernel back the very indices it resolved at construction must
    /// change nothing — bit for bit.
    ///
    /// This is the contract that makes a periodic régime possible: if the two
    /// entry points can disagree on identical input, they can disagree on
    /// remapped input too, and the difference would be indistinguishable from
    /// physics. Bit equality rather than a tolerance because it is meant to be
    /// the same arithmetic in the same order — a kernel that had grown a
    /// second copy of its force expression would show up here as a few ulp.
    #[test]
    fn rebinding_a_kernel_to_its_own_indices_changes_nothing() {
        let mut ff = ForceField::new("all-bonded");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A",
                &["A", "A"],
                Params::from_iter([("k", 300.0), ("r0", 1.5)]),
            )
            .unwrap();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A",
                &["A", "A", "A"],
                Params::from_iter([("k", 50.0), ("theta0", 109.5)]),
            )
            .unwrap();
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "A-A-A-A",
                &["A", "A", "A", "A"],
                Params::from_iter([("k1", 1.3), ("k2", -0.05), ("k3", 0.24), ("k4", 0.0)]),
            )
            .unwrap();
        ff.def_style("improper", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "A-A-A-A",
                &["A", "A", "A", "A"],
                Params::from_iter([("k", 10.0), ("chi0", 0.0)]),
            )
            .unwrap();

        let frame = make_all_bonded_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let mut checked = 0;
        for (m, member) in pots.members().iter().enumerate() {
            let Some(terms) = member.terms() else {
                continue;
            };
            let ForceTerm::Indexed(pot) = member else {
                unreachable!("only an indexed member answers with a table")
            };
            let (e0, f0) = pot.energy_forces_at(&coords);
            let (e1, f1) = pot.over_terms(&coords, terms.view());
            assert_eq!(
                e0.to_bits(),
                e1.to_bits(),
                "member {m}: energy {e0} vs {e1} on identical indices"
            );
            assert_eq!(f0.len(), f1.len(), "member {m}: force length");
            for (c, (a, b)) in f0.iter().zip(&f1).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "member {m}: force component {c}: {a} vs {b}"
                );
            }
            assert!(
                e0.is_finite() && e0 != 0.0,
                "member {m} contributed nothing, so it asserted nothing"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 4,
            "expected a bond, an angle, a dihedral and an improper to answer `terms()`"
        );
    }

    fn make_bond_frame() -> Frame {
        let mut frame = make_atoms_only_frame();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["A-A".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn make_lj_frame() -> Frame {
        // Two atoms 2 Å apart, each of per-atom `type` "A" (the kernel reads
        // per-atom LJ params keyed by it and Lorentz-Berthelot-combines).
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "A".to_string()]).into_dyn(),
                None,
            )
            .unwrap();

        // No bonds, so no special pair: (0, 1) is priced at full strength.
        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(ForceTerm::plain(DummyPotential { value: 1.0 }));
        pots.push(ForceTerm::plain(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.energy_at(&coords) - 3.0).abs() < 1e-5);

        let (_, forces) = pots.energy_forces_at(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    /// Prices each row of the table it is handed at 1, and nothing without one.
    struct PairCounting;

    impl Potential for PairCounting {
        fn evaluate(&self, _coords: &[F], _acc: &mut Accumulator<'_>) {}
    }

    impl PairDriven for PairCounting {
        fn evaluate_pairs(
            &self,
            _coords: &[F],
            pairs: &Neighbors,
            _factor: &[F],
            acc: &mut Accumulator<'_>,
        ) {
            // No force, so no virial to lose.
            acc.add_energy(pairs.query_point_indices().len() as F);
        }

        fn binds_a_fixed_pair_list(&self) -> bool {
            false
        }
    }

    #[test]
    fn one_pair_table_is_shared_with_every_member() {
        let pairs = Neighbors::from_pairs(
            [
                molrs::core::NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 1.0,
                    disp: [1.0, 0.0, 0.0],
                },
                molrs::core::NeighborPair {
                    i: 0,
                    j: 2,
                    dist_sq: 4.0,
                    disp: [2.0, 0.0, 0.0],
                },
            ],
            molrs::core::NeighborColumns::FULL,
            molrs::core::QueryMode::SelfQuery { n_points: 3 },
        );
        let mut pots = Potentials::new();
        pots.push(ForceTerm::pair(PairCounting));
        pots.push(ForceTerm::pair(PairCounting));
        let coords: Vec<F> = vec![0.0; 9];
        let (e, _, w) = pots.over_pairs_virial(&coords, &pairs);
        assert!((e - 4.0).abs() < 1e-12);
        assert_eq!(
            w,
            Some(Virial::ZERO),
            "every member tallied, so the sum is reported"
        );

        // A member that adds forces without a virial makes the aggregate's
        // `None` — not a total missing one term.
        pots.push(ForceTerm::plain(DummyPotential { value: 1.0 }));
        let (e, _, w) = pots.over_pairs_virial(&coords, &pairs);
        assert!((e - 5.0).abs() < 1e-12);
        assert_eq!(w, None);
    }

    /// The accumulator's channels: a borrowed buffer keeps what it held, an
    /// unrequested virial is `None`, and a requested one survives only while
    /// every contribution came with one.
    #[test]
    fn an_accumulator_reports_only_the_channels_it_was_given() {
        let mut buf = vec![1.0; 6];
        let mut acc = Accumulator::adding_into(&mut buf);
        DummyPotential { value: 2.0 }.evaluate(&[0.0; 6], &mut acc);
        assert_eq!(acc.energy(), 2.0);
        assert_eq!(acc.virial(), None, "not requested");
        assert_eq!(acc.into_forces(), vec![3.0; 6]);
        assert_eq!(buf, vec![3.0; 6], "added into, not overwritten");

        let mut acc = Accumulator::new(6).with_virial();
        assert_eq!(
            acc.virial(),
            Some(Virial::ZERO),
            "requested and nothing added"
        );
        if let (_, Some(w)) = acc.forces_with_virial() {
            w.add_outer([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
        }
        assert_eq!(acc.virial().map(|w| w.components[0]), Some(2.0));
        acc.forces_without_virial();
        assert_eq!(acc.virial(), None, "a contribution without one loses it");
        assert!(
            acc.forces_with_virial().1.is_none(),
            "and nobody tallies into a lost one"
        );
    }

    #[test]
    fn unknown_style_kernel_is_error() {
        // A style whose (category, name) has no kernel -> Err from PotentialCompiler::compile.
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "nonexistent", Params::new())
            .unwrap()
            .def_type("A-A", &["A", "A"], Params::from_iter([("k", 1.0)]))
            .unwrap();
        let frame = make_bond_frame();
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.to_string().contains("no kernel"), "{err}");
    }

    #[test]
    fn registering_a_style_extends_dispatch() {
        // A custom (category, name) with no built-in kernel becomes usable by
        // registering its constructor — no edit to PotentialCompiler required.
        fn my_constructor(
            _sp: &Params,
            _tp: &[(&str, &Params)],
            _f: &Frame,
            _coulomb: &SpecialWeights,
        ) -> Result<ForceTerm, crate::ff::potential::CompileError> {
            Ok(ForceTerm::plain(DummyPotential { value: 42.0 }))
        }
        crate::ff::style_registry::GlobalRegistry
            .register_style(
                crate::ff::ir::StyleSpec::new("pair", "test/custom"),
                Some(crate::ff::style_registry::Kernel::constructor(
                    my_constructor,
                )),
            )
            .unwrap();

        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "test/custom", Params::new())
            .unwrap()
            .def_type("A", &["A"], Params::new())
            .unwrap();
        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        // the custom kernel ran: DummyPotential yields its constant value.
        assert!((pots.energy_at(&coords) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn atom_style_is_skipped() {
        // Atom styles carry types/charges, not a pairwise kernel -> skipped.
        let mut ff = ForceField::new("test");
        ff.def_style("atom", "full", Params::new()).unwrap();
        let frame = make_atoms_only_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        assert_eq!(pots.len(), 0);
    }

    #[test]
    fn test_compile_requires_types() {
        let mut ff = ForceField::new("test");
        ff.def_style("bond", "harmonic", Params::new()).unwrap();
        let frame = make_bond_frame();
        let err = PotentialCompiler::new(&ff)
            .compile(&frame)
            .expect_err("expected compile to fail");
        assert!(err.to_string().contains("has no type definitions"));
    }

    #[test]
    fn test_compile_energy() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, _) = pots.energy_forces_at(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn a_pair_type_is_found_through_its_endpoints_not_its_name() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "anything",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, _) = pots.energy_forces_at(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compile_forces() {
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .unwrap();

        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (_, forces) = pots.energy_forces_at(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn lj_cut_combines_distinct_types_lorentz_berthelot() {
        // Two atoms 2.5 Å apart of distinct types A and B; the kernel must
        // Lorentz-Berthelot-combine their per-atom params.
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 2.5 as F]).into_dyn(),
                None,
            )
            .unwrap();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut ff = ForceField::new("test");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "lj/cut",
            Params::from_iter([("cutoff", F::INFINITY)]),
        )
        .unwrap()
        .def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
        )
        .unwrap()
        .def_type(
            "B",
            &["B"],
            Params::from_iter([("epsilon", 4.0), ("sigma", 3.0)]),
        )
        .unwrap();

        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let (energy, _) = pots.energy_forces_at(&coords);

        // ε = √(1·4) = 2, σ = (1+3)/2 = 2, at r = 2.5.
        let (eps, sigma, r) = (2.0_f64, 2.0_f64, 2.5_f64);
        let sr6 = (sigma / r).powi(6);
        let expected = 4.0 * eps * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-9,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn lj_cut_applies_special_bonds_14_scaling() {
        // A 4-atom chain 0-1-2-3 (bonds 0-1-2-3, angles, one dihedral): the
        // special pairs weigh 1-2/1-3 0 and the (0,3) 1-4 pair at
        // special_bonds lj 1-4 = 0.5, so its LJ energy is halved.
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.0, 2.0, 3.0]).into_dyn(),
                None,
            )
            .unwrap();
        for col in ["y", "z"] {
            atoms
                .insert(col, Array1::from_vec(vec![0.0 as F; 4]).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1, 2]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![1 as Idx, 2, 3]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1]).into_dyn(),
                None,
            )
            .unwrap();
        angles
            .insert(
                "atomk",
                Array1::from_vec(vec![2 as Idx, 3]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        dihedrals
            .insert("atoml", Array1::from_vec(vec![3 as Idx]).into_dyn(), None)
            .unwrap();
        frame.insert("dihedrals", dihedrals);

        // The special pairs: the three bonds, the two 1-3 pairs and (0,3).
        let pairs = molrs::core::Topology::from_frame(&frame)
            .unwrap()
            .pairs_block();
        assert_eq!(pairs.n_rows(), Some(6));
        frame.insert(PAIRS, pairs);

        let mut ff = ForceField::new("test");
        ff.def_style(
            "pair",
            "lj/cut",
            Params::from_iter([("cutoff", F::INFINITY)]),
        )
        .unwrap()
        .def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
        )
        .unwrap();
        let mut sb = crate::ff::ir::SpecialBonds::EXCLUDE_12_13;
        sb.lj[2] = 0.5;
        ff.set_special_bonds(sb);

        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let (energy, _) = pots.energy_forces_at(&coords);

        // (0,3) at r = 3, ε = σ = 1, scaled by the 0.5 1-4 weight.
        let sr6 = (1.0_f64 / 3.0).powi(6);
        let expected = 0.5 * 4.0 * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-12,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn test_compile_empty_ff() {
        let ff = ForceField::new("test");
        let frame = make_lj_frame();
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();

        let (energy, forces) = pots.energy_forces_at(&coords);
        assert!(energy.abs() < 1e-5);
        assert_eq!(forces.len(), 6);
        assert!(forces.iter().all(|x| x.abs() < 1e-5));
    }

    /// `compile` prices every pair at the weight the frame's special pairs
    /// give it: 1-2 and 1-3 out, 1-4 scaled by `special_bonds`, every other
    /// pair (an unbonded atom's included) at full strength. A frame with
    /// bonds and no special pairs is refused.
    #[test]
    fn compile_prices_every_pair_at_its_special_weight() {
        // Chain 0-1-2-3 along x, atom 4 unbonded beside it.
        let xs: [F; 5] = [0.0, 1.5, 3.0, 4.5, 2.0];
        let ys: [F; 5] = [0.0, 0.0, 0.0, 0.0, 3.0];
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(xs.to_vec()).into_dyn(), None)
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(ys.to_vec()).into_dyn(), None)
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F; 5]).into_dyn(), None)
            .unwrap();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 5]).into_dyn(),
                None,
            )
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                Array1::from_vec(vec![0 as Idx, 1, 2]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                "atomj",
                Array1::from_vec(vec![1 as Idx, 2, 3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);

        let (epsilon, sigma) = (0.2, 1.4);
        let mut ff = ForceField::new("derived");
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        ff.def_style(
            "pair",
            "lj/cut",
            Params::from_iter([("cutoff", F::INFINITY)]),
        )
        .unwrap()
        .def_type(
            "A",
            &["A"],
            Params::from_iter([("epsilon", epsilon), ("sigma", sigma)]),
        )
        .unwrap();

        let lj = |a: usize, b: usize| {
            let r = ((xs[a] - xs[b]).powi(2) + (ys[a] - ys[b]).powi(2)).sqrt();
            4.0 * epsilon * ((sigma / r).powi(12) - (sigma / r).powi(6))
        };
        let expected = 0.5 * lj(0, 3) + lj(0, 4) + lj(1, 4) + lj(2, 4) + lj(3, 4);
        let coords: Vec<F> = (0..5).flat_map(|k| [xs[k], ys[k], 0.0]).collect();

        // Bonds and no special pairs: the 1-2 terms would be priced at
        // full strength, so the frame is refused.
        let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
        assert!(err.to_string().contains("generate_topology"), "{err}");

        frame.insert(
            PAIRS,
            molrs::core::Topology::from_frame(&frame)
                .unwrap()
                .pairs_block(),
        );
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        assert!(
            (pots.energy_at(&coords) - expected).abs() < 1e-10,
            "{} != {expected}",
            pots.energy_at(&coords)
        );
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use super::{Accumulator, IndexedTerms, PairDriven, Potential};
    use molrs::core::{Neighbors, Virial};
    use molrs::op::F;
    use ndarray::ArrayView2;

    /// One evaluation in fresh storage, for a test that wants the totals:
    /// a new [`Accumulator`], one call of the primitive, its channels read
    /// back. Test-only — the API is the primitive and the accumulator.
    pub(crate) trait OneShot: Potential {
        /// [`Potential::evaluate`]: energy and forces.
        fn energy_forces_at(&self, coords: &[F]) -> (F, Vec<F>) {
            let mut acc = Accumulator::new(coords.len());
            self.evaluate(coords, &mut acc);
            (acc.energy(), acc.into_forces())
        }

        /// [`Potential::evaluate`]: energy.
        fn energy_at(&self, coords: &[F]) -> F {
            self.energy_forces_at(coords).0
        }

        /// [`PairDriven::evaluate_pairs`], unweighted: energy and forces.
        fn over_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>)
        where
            Self: PairDriven,
        {
            let (e, f, _) = self.over_pairs_virial(coords, pairs);
            (e, f)
        }

        /// [`PairDriven::evaluate_pairs`], unweighted, with the virial
        /// requested.
        fn over_pairs_virial(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>, Option<Virial>)
        where
            Self: PairDriven,
        {
            let mut acc = Accumulator::new(coords.len()).with_virial();
            self.evaluate_pairs(coords, pairs, &[], &mut acc);
            let (e, w) = (acc.energy(), acc.virial());
            (e, acc.into_forces(), w)
        }

        /// [`IndexedTerms::evaluate_terms`]: energy and forces.
        fn over_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>) -> (F, Vec<F>)
        where
            Self: IndexedTerms,
        {
            let mut acc = Accumulator::new(coords.len());
            self.evaluate_terms(coords, terms, &mut acc);
            (acc.energy(), acc.into_forces())
        }
    }

    impl<P: Potential + ?Sized> OneShot for P {}

    /// Central-difference check that every force component is `-dE/dx`.
    pub(crate) fn assert_forces_are_negative_gradient(pot: &dyn Potential, coords: &[F], tol: F) {
        let h: F = 1e-6;
        let (_, f) = pot.energy_forces_at(coords);
        let mut worst: F = 0.0;
        for i in 0..coords.len() {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let numeric = -(pot.energy_at(&plus) - pot.energy_at(&minus)) / (2.0 * h);
            worst = worst.max((f[i] - numeric).abs());
        }
        assert!(worst < tol, "max |F + dE/dx| = {worst:.3e}");
    }
}
