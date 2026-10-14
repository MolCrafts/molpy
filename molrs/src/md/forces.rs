use ndarray::{Array2, ArrayView2};

use molrs::core::VerletSkin;
use molrs::core::Virial;
use molrs::ff::potential::{ForceTerm, Potential};
use molrs::op::{F, Fnx3, Fnx3View, I};

use super::error::MdError;
use super::ghost_topology::BondedLists;
use super::state::ForceOutput;
use molrs::core::GhostHalo;
use molrs::ff::potential::SpecialWeights;

/// What an integrator asks of a force field.
///
/// Why the signature is what it is follows below.
///
/// The force-field seam: one trait between the integrator and everything that
/// makes a force.
///
/// An integrator asks exactly one question — *what are the energy, the forces
/// and the virial at this configuration?* — and [`ForceProvider`] is that
/// question. Everything a force evaluation needs on the way to answering it
/// (the potential, the neighbour bookkeeping, the periodic régime, the
/// fold-back of copies onto owners) lives behind the trait, and the integrator
/// sees none of it. That is what makes the engine force-field agnostic: a new
/// way to make a force is a new implementor, not a new enum variant and a new
/// arm in a `match` that every caller must be recompiled against.
///
/// # It is also the parallelisation boundary
///
/// One `compute` call is one unit of work an implementation may split across
/// threads however it likes. This is not incidental — it is the reason the
/// trait has the shape it has.
///
/// The per-pair fold that dominates a step happens *inside*
/// [`Potential::calc_energy_forces_with_pairs`], and `Potential` is a stable
/// interface with dozens of implementors that this module does not get to
/// change. So a parallel fold has to be owned by something above `Potential`
/// and below the integrator; this trait is that object.
///
/// Three consequences are load-bearing:
///
/// * **`&mut self`, and no interior mutability.** A provider is not stateless:
///   a Verlet skin holds its reference coordinates and its age, a halo holds
///   its copies and a generation counter, and all of it is bound to *one*
///   trajectory. Exclusive access for the duration of a call is what lets an
///   implementation own reusable scratch and mutate it from rayon with no
///   locks, and split its own data with `par_chunks_mut` with no
///   synchronisation at all. `&self` plus interior mutability would buy only
///   concurrent `compute` calls on one shared provider — meaningless here,
///   because two replicas at different positions cannot share a skin whose
///   held coordinates name one of them — at the price of a lock every step,
///   forever. One provider per replica is the right shape, and that needs
///   `Send`, not `&self`.
/// * **`Send + Sync` on the trait.** A boxed provider must be movable to a
///   worker thread; just as importantly, the bound *forbids* an implementor
///   from hiding an `Rc` or a `RefCell` and foreclosing that later.
/// * **An owned [`ForceOutput`] out, borrowing nothing.** No lifetime crosses
///   the boundary, so a provider may re-partition itself between calls —
///   rebuild its halo, change its domain decomposition, migrate atoms — with
///   no caller pinned to its internal layout.
pub trait ForceProvider: Send + Sync {
    /// Energy, forces on the **owned** atoms, and the virial, at `pos`.
    ///
    /// `forces` has exactly `pos.nrows()` rows. A provider that evaluates over
    /// a larger set — periodic copies, a padded domain — folds the result back
    /// before returning: that the extended set ever existed is not the
    /// caller's business, and an integrator that had to know would not be
    /// force-field agnostic.
    ///
    /// `wrap_shifts` is the lattice shift the caller's wrap just applied, one
    /// signed count per axis per atom in the schema's image integer type
    /// ([`I`], i32), all zeros when nothing folded. It is a
    /// parameter rather than something the provider re-derives because a fold
    /// *relabels* an atom without moving it: no displacement test can see one,
    /// and a provider holding copies must reconcile it in the same breath.
    ///
    /// The result lands in `out`. Its `forces` array is the caller's buffer:
    /// a provider that accumulates into its own array swaps the two, so a
    /// step that hands back the array it was given last step allocates
    /// nothing. Whatever shape `out.forces` arrives in, it leaves as
    /// `(pos.nrows(), 3)`.
    fn compute_into(
        &mut self,
        pos: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
        out: &mut ForceOutput,
    ) -> Result<(), MdError>;

    /// [`compute_into`](Self::compute_into) into a fresh [`ForceOutput`].
    fn compute(
        &mut self,
        pos: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<ForceOutput, MdError> {
        let mut out = ForceOutput {
            energy: 0.0,
            forces: Fnx3::zeros((0, 3)),
            virial: None,
        };
        self.compute_into(pos, wrap_shifts, &mut out)?;
        Ok(out)
    }

    /// Neighbour-bookkeeping counters, for logs and tests — never for physics.
    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats::default()
    }
}

impl ForceProvider for Box<dyn ForceProvider> {
    fn compute_into(
        &mut self,
        pos: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
        out: &mut ForceOutput,
    ) -> Result<(), MdError> {
        (**self).compute_into(pos, wrap_shifts, out)
    }

    fn compute(
        &mut self,
        pos: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<ForceOutput, MdError> {
        (**self).compute(pos, wrap_shifts)
    }

    fn neighbor_stats(&self) -> NeighborStats {
        (**self).neighbor_stats()
    }
}

/// What a provider's neighbour bookkeeping has been doing.
///
/// Every field is optional because a provider that keeps no list has no
/// answer, and zero would be an answer — the same distinction
/// `Neighbors::disp` draws, for the same reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighborStats {
    /// Pairs in the current table.
    pub edges: Option<usize>,
    /// Times the list has been rebuilt over the run.
    pub rebuilds: Option<usize>,
    /// Steps since the last rebuild.
    pub ago: Option<usize>,
}

// ---------------------------------------------------------------------------
// SelfPairedForces
// ---------------------------------------------------------------------------

/// No neighbour list: the potential is handed coordinates and enumerates its
/// own pairs.
pub struct SelfPairedForces {
    potential: Box<dyn Potential>,
}

impl SelfPairedForces {
    /// Evaluate `potential` at the raw coordinates, every step.
    pub fn new(potential: impl Potential + 'static) -> Self {
        Self {
            potential: Box::new(potential),
        }
    }
}

impl ForceProvider for SelfPairedForces {
    fn compute_into(
        &mut self,
        pos: Fnx3View<'_>,
        _wrap_shifts: ArrayView2<'_, I>,
        out: &mut ForceOutput,
    ) -> Result<(), MdError> {
        let (energy, forces) = match pos.as_slice() {
            Some(flat) => self.potential.calc_energy_forces(flat),
            None => {
                let flat: Vec<F> = pos.iter().copied().collect();
                self.potential.calc_energy_forces(&flat)
            }
        };
        // The potential hands back a fresh Vec, so this route allocates per
        // step by construction; the Vec becomes the array without a copy.
        *out = owned_output(energy, forces, pos.nrows())?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MicPairs
// ---------------------------------------------------------------------------

/// Refuse a member whose parameters are bound to a pair list nobody is
/// evaluating.
///
/// Such a kernel ignores the table it is handed and returns a frozen sum, so
/// the run would maintain a neighbour list, rebuild it, and report its
/// counters, while the answer depended on none of it. Only a
/// [`ForceTerm::Pair`] can be bound this way — a bonded member is never handed
/// the table — and [`ForceTerm::binds_a_fixed_pair_list`] already knows that, so
/// this does not ask each kernel twice.
///
/// It stays a run-time check rather than a type: a kernel built from a pair
/// list and the same kernel keyed on the atoms are one Rust type in molrs (the
/// list is a private source variant), so `ForceTerm::Pair` cannot distinguish
/// them and only the kernel can answer.
fn reject_frozen_members(members: &[(ForceTerm, SpecialWeights)]) -> Result<(), MdError> {
    for (m, (pot, _)) in members.iter().enumerate() {
        if pot.binds_a_fixed_pair_list() {
            return Err(MdError::Invalid(format!(
                "member {m} resolved its parameters against a fixed pair list, so it \
                 cannot be evaluated over a neighbour table — it would ignore the table \
                 and answer for the list it was built from. Build it from the atoms \
                 instead (PotentialCompiler::compile_typed)"
            )));
        }
    }
    Ok(())
}

/// Minimum-image pairs over the owned atoms.
///
/// The potential sees the owned atoms and a pair table whose displacements
/// have already been folded — it is never asked to know that a boundary
/// exists.
///
/// Members are evaluated one at a time, as under [`GhostPairs`], and for the
/// same reason: what each one needs differs. A member holding atom indices is
/// a bonded term and reads no pair table; the rest read the table, weighted by
/// the force field's exclusions.
///
/// # What this route cannot do
///
/// A bonded term here is measured on the **stored** coordinates, so a molecule
/// that straddles a face is measured across the cell. The minimum image fixes
/// up *pair* displacements and nothing else. A system whose molecules cross
/// boundaries wants [`GhostPairs`], which resolves the bonded indices onto
/// copies.
pub struct MicPairs {
    /// The force evaluation's members, in the order the caller gave them,
    /// each carrying the part it plays.
    members: Vec<ForceTerm>,
    /// Per member, the weights on its close non-bonded neighbours.
    special: Vec<SpecialWeights>,
    skin: VerletSkin,
    /// Force accumulator, reused across steps. Every member adds into it, so
    /// a step allocates one of these rather than one per member.
    acc: Fnx3,
    /// Per-pair weights for the member being evaluated, reused across steps.
    factors: Vec<F>,
    /// One bonded member's forces, reused across steps: its virial is tallied
    /// from them before they are folded into `acc`.
    term: Vec<F>,
}

impl MicPairs {
    /// Evaluate one member over the pairs `skin` maintains, with no
    /// special-bonds weights.
    pub fn new(member: ForceTerm, skin: VerletSkin) -> Result<Self, MdError> {
        Self::from_members(vec![(member, SpecialWeights::default())], skin)
    }

    /// Evaluate several members — what a force field compiles to — over the
    /// pairs `skin` maintains.
    ///
    /// Each member comes with the weights on its close non-bonded neighbours;
    /// a bonded member takes [`SpecialWeights::default`], which scales nothing.
    pub fn from_members(
        members: Vec<(ForceTerm, SpecialWeights)>,
        skin: VerletSkin,
    ) -> Result<Self, MdError> {
        reject_frozen_members(&members)?;
        let (members, special): (Vec<ForceTerm>, Vec<_>) = members.into_iter().unzip();
        Ok(Self {
            members,
            special,
            skin,
            acc: Fnx3::zeros((0, 3)),
            factors: Vec::new(),
            term: Vec::new(),
        })
    }
}

impl ForceProvider for MicPairs {
    fn compute_into(
        &mut self,
        pos: Fnx3View<'_>,
        _wrap_shifts: ArrayView2<'_, I>,
        out: &mut ForceOutput,
    ) -> Result<(), MdError> {
        let n_atoms = pos.nrows();
        let pairs = self.skin.pairs_at(pos)?;
        // Borrowed, not copied: a standard-layout `(N, 3)` view *is* the flat
        // `[x0, y0, z0, x1, …]` a kernel wants, so there is nothing to convert.
        let owned_copy: Vec<F>;
        let flat: &[F] = match pos.as_slice() {
            Some(sl) => sl,
            None => {
                owned_copy = pos.iter().copied().collect();
                &owned_copy
            }
        };

        // `acc` is whatever the caller handed back last step (see the swap
        // below), so its shape and layout are checked, not assumed.
        if self.acc.nrows() != n_atoms || !self.acc.is_standard_layout() {
            self.acc = Array2::zeros((n_atoms, 3));
        } else {
            self.acc.fill(0.0);
        }
        let acc = self
            .acc
            .as_slice_mut()
            .expect("a freshly shaped array is standard layout");

        let mut energy = 0.0;
        // `None` the moment any member declines to report one: a virial missing
        // a term is not a small error, it is a different quantity.
        let mut virial = Some(Virial::ZERO);
        for m in 0..self.members.len() {
            let member = &self.members[m];
            let (e, part) = match member {
                // A pair member reads the table, weighted by the force field's
                // exclusions. No copies here, so an index *is* its own owner.
                ForceTerm::Pair(pot) => {
                    let factors =
                        self.special[m].factors_for(pairs, n_atoms, &[], &mut self.factors);
                    pot.accumulate_pairs(flat, pairs, factors, acc)
                }
                // A bonded term reads its own indices, not the pair table —
                // and needs no kernel-side tally. Its forces sum to zero term
                // by term and no periodic image entered the geometry, so
                // `Σ_a f_a ⊗ x_a` over the stored coordinates *is* its virial,
                // and is independent of where the cell's origin falls.
                ForceTerm::Indexed(pot) => {
                    self.term.clear();
                    self.term.resize(acc.len(), 0.0);
                    let e = pot.accumulate(flat, &mut self.term);
                    let f = &self.term;
                    let mut w_term = Virial::ZERO;
                    for a in 0..n_atoms {
                        w_term.add_outer(
                            [f[a * 3], f[a * 3 + 1], f[a * 3 + 2]],
                            [flat[a * 3], flat[a * 3 + 1], flat[a * 3 + 2]],
                        );
                    }
                    for (dst, v) in acc.iter_mut().zip(f) {
                        *dst += v;
                    }
                    (e, Some(w_term))
                }
                // An external field, a restraint, a constant force. Its forces
                // do *not* sum to zero, so `Σ_a f_a ⊗ x_a` moves when the
                // cell's origin does and is not a virial. Reporting `None`
                // makes the whole step's virial `None`, which is the honest
                // answer: the pressure of a system being pushed on from outside
                // is not the sum of its pair terms.
                ForceTerm::Plain(pot) => (pot.accumulate(flat, acc), None),
            };
            energy += e;
            match (virial.as_mut(), part) {
                (Some(total), Some(p)) => {
                    for c in 0..6 {
                        total.components[c] += p.components[c];
                    }
                }
                (_, None) => virial = None,
                (None, _) => {}
            }
        }

        // Hand the accumulator over and keep the caller's array as next
        // step's accumulator: no clone, no allocation, in steady state.
        std::mem::swap(&mut out.forces, &mut self.acc);
        out.energy = energy;
        out.virial = virial;
        Ok(())
    }

    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats {
            edges: Some(self.skin.n_edges()),
            rebuilds: Some(self.skin.n_rebuilds()),
            ago: Some(self.skin.ago()),
        }
    }
}

// ---------------------------------------------------------------------------
// GhostPairs
// ---------------------------------------------------------------------------

/// Periodic copies: the potential sees an ordinary local cluster.
///
/// Owned atoms *and their copies* go in, with plain differences; the forces
/// come back over that extended set and are folded onto the owners before
/// anything physical is read off them.
///
/// Members are evaluated one at a time rather than through an aggregate,
/// because what each one needs differs: a member holding atom indices is
/// handed the indices resolved against the copies that exist now, and a member
/// reading geometry is handed the pair table. An aggregate would have to pick
/// one table for everybody, and the bonded ones are not interchangeable — a
/// bond list is not an angle list.
pub struct GhostPairs {
    halo: GhostHalo,
    /// The force evaluation's members, in the order the caller gave them,
    /// each carrying the part it plays.
    members: Vec<ForceTerm>,
    /// Per member, the weights on its close non-bonded neighbours.
    ///
    /// Per member and not one shared table, because a force field may scale
    /// van der Waals and electrostatics differently — Amber uses `1/2` and
    /// `1/1.2` — and in molrs those are separate kernels. One table would have
    /// to be wrong for one of them.
    special: Vec<SpecialWeights>,
    /// Which members keep atom indices, and what those indices are now.
    lists: BondedLists,
    /// The copy-list generation the members' per-atom state was gathered for,
    /// or `None` before the first gather.
    gathered: Option<u64>,
    /// Force accumulator over `[owned | ghost]`, reused across steps.
    acc: Fnx3,
    /// Per-pair weights for the member being evaluated, reused across steps.
    factors: Vec<F>,
}

impl GhostPairs {
    /// Evaluate one member over the copies `halo` maintains, with no
    /// special-bonds weights — right for a system with no bonded topology.
    pub fn new(member: ForceTerm, halo: GhostHalo) -> Result<Self, MdError> {
        Self::from_members(vec![(member, SpecialWeights::default())], halo)
    }

    /// Evaluate several members — what a force field compiles to — over the
    /// copies `halo` maintains.
    ///
    /// This takes no force field and no frame: each member already says which
    /// part it plays — [`ForceTerm::Indexed`] reads an index table, [`ForceTerm::Pair`]
    /// reads the neighbour table — and that is the whole of what MD needs to
    /// know about a force field.
    ///
    /// Each member comes with the weights on its close non-bonded neighbours.
    /// A neighbour table finds every pair inside the cutoff, bonded or not, so
    /// without them a bonded pair would be counted twice — once by the bonded
    /// term and once at full non-bonded strength. A bonded member takes
    /// [`SpecialWeights::default`], which scales nothing.
    pub fn from_members(
        members: Vec<(ForceTerm, SpecialWeights)>,
        halo: GhostHalo,
    ) -> Result<Self, MdError> {
        reject_frozen_members(&members)?;
        let (members, special): (Vec<_>, Vec<_>) = members.into_iter().unzip();
        let lists = BondedLists::new(&members);
        Ok(Self {
            halo,
            members,
            lists,
            special,
            gathered: None,
            acc: Fnx3::zeros((0, 3)),
            factors: Vec::new(),
        })
    }

    /// The halo, for tests that read its counters.
    pub fn halo(&self) -> &GhostHalo {
        &self.halo
    }
}

impl ForceProvider for GhostPairs {
    fn compute_into(
        &mut self,
        pos: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
        out: &mut ForceOutput,
    ) -> Result<(), MdError> {
        // Move the copies first, then re-resolve the indices against them, then
        // read the pairs. Each step depends on the one before it, and doing
        // them in one call would hide that.
        self.halo.advance(pos, wrap_shifts)?;
        // A member holding anything per atom — a charge, a type index — has to
        // cover the copies, because the pair table names them. That state is
        // *derived* from the owners', so a rebuild invalidates it and nothing
        // else does: a fold relabels an atom without changing what it is.
        let generation = self.halo.ghosts().generation();
        if self.gathered != Some(generation) {
            let owner = self.halo.ghosts().owner();
            for member in &mut self.members {
                member.gather_onto_copies(owner);
            }
            self.gathered = Some(generation);
        }
        self.lists.refresh(&self.halo, pos, wrap_shifts)?;

        // Both are reads: `advance` filled them.
        let pairs = self.halo.pairs();
        let all = self.halo.combined();
        let n_all = all.nrows();
        let owned_copy: Vec<F>;
        let flat: &[F] = match all.as_slice() {
            Some(sl) => sl,
            None => {
                owned_copy = all.iter().copied().collect();
                &owned_copy
            }
        };

        if self.acc.nrows() != n_all {
            self.acc = Array2::zeros((n_all, 3));
        } else {
            self.acc.fill(0.0);
        }
        let acc = self
            .acc
            .as_slice_mut()
            .expect("a freshly shaped array is standard layout");

        let set = self.halo.ghosts();
        let mut energy = 0.0;
        for m in 0..self.members.len() {
            let member = &self.members[m];
            let e = match member {
                // A bonded term reads the indices resolved against the copies
                // that exist now, not the pair table — so no weight applies to
                // it. `lists` holds one entry per `ForceTerm::Indexed`, so this
                // arm always finds its table.
                ForceTerm::Indexed(pot) => {
                    let terms = self.lists.current(m).ok_or_else(|| {
                        MdError::Invalid(format!("member {m} holds indices but has no list"))
                    })?;
                    pot.accumulate_with_terms(flat, terms, acc)
                }
                ForceTerm::Pair(pot) => {
                    let factors = self.special[m].factors_for(
                        pairs,
                        set.n_owned(),
                        set.owner(),
                        &mut self.factors,
                    );
                    pot.accumulate_pairs(flat, pairs, factors, acc).0
                }
                // Reads coordinates only — over `[owned | ghost]`, so its
                // forces land on the copies and the reverse pass folds them
                // back onto the owners like any other member's.
                ForceTerm::Plain(pot) => pot.accumulate(flat, acc),
            };
            energy += e;
        }

        // Reverse accumulation: a copy's force belongs to the atom it copies.
        // The virial is tallied from the copy forces *before* they are reduced
        // — afterwards the information it needs is gone. This covers every
        // member, bonded included, which is why the kernels' own tallies are
        // not summed here.
        let virial = self
            .halo
            .reverse_comm_with_virial(&mut self.acc, all.view())?;
        // The owned block is a prefix of `acc`, which also covers the copies,
        // so it is copied out — into the caller's array when that already has
        // the shape, so the copy is the only cost in steady state.
        let n_owned = pos.nrows();
        if out.forces.nrows() != n_owned || !out.forces.is_standard_layout() {
            out.forces = Array2::zeros((n_owned, 3));
        }
        out.forces
            .assign(&self.acc.slice(ndarray::s![..n_owned, ..]));
        out.energy = energy;
        out.virial = Some(virial);
        Ok(())
    }

    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats {
            edges: None,
            rebuilds: Some(self.halo.rebuilds()),
            ago: None,
        }
    }
}

/// Shape a flat force vector into the owned `(N, 3)` block, or say why it does
/// not fit.
fn owned_output(energy: F, forces: Vec<F>, n_atoms: usize) -> Result<ForceOutput, MdError> {
    let n_components = forces.len();
    let forces = Array2::from_shape_vec((n_atoms, 3), forces).map_err(|_| {
        MdError::Invalid(format!(
            "potential returned {n_components} force components for {n_atoms} atoms"
        ))
    })?;
    Ok(ForceOutput {
        energy,
        forces,
        virial: None,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use molrs::core::SimBox;
    use molrs::core::{NeighborList, NeighborPolicy};
    use molrs::ff::compile::PotentialCompiler;
    use molrs::ff::potential::Potentials;
    use molrs::ff::potential::pair::PairLjCut;

    use super::*;
    use molrs::core::Topology;
    use molrs::ff::potential::SpecialWeights;

    /// `n` atoms joined by `edges`.
    fn bonded(n: usize, edges: &[[usize; 2]]) -> Topology {
        let mut topo = Topology::with_atoms(n);
        topo.add_bonds(edges);
        topo
    }

    fn cell() -> SimBox {
        SimBox::cube(12.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap()
    }

    fn four_atoms() -> ndarray::Array2<F> {
        array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [3.0, 3.0, 11.0],
        ]
    }

    fn lj() -> PairLjCut {
        PairLjCut::new(0.3, 3.4, 5.0, 12, 6, false, false).unwrap()
    }

    fn skin(pos: ndarray::ArrayView2<'_, F>) -> molrs::core::VerletSkin {
        molrs::core::VerletSkin::new(
            NeighborList::new(5.0),
            5.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos,
            cell(),
        )
        .unwrap()
    }

    /// Every provider answers over the owned atoms, whatever it evaluated over.
    ///
    /// The ghost route is the one this can catch out: it evaluates over
    /// `owned + copies` and has to fold that back. An integrator handed the
    /// extended block would write forces onto rows that are not atoms, and the
    /// shapes would go on agreeing with each other for a while.
    #[test]
    fn a_provider_answers_with_exactly_the_owned_rows() {
        let pos = four_atoms();
        let n = pos.nrows();
        let no_fold = Array2::zeros((n, 3));

        let mut direct = SelfPairedForces::new(Potentials::new());
        assert_eq!(
            direct
                .compute(pos.view(), no_fold.view())
                .unwrap()
                .forces
                .nrows(),
            n
        );

        let mut mic = MicPairs::new(ForceTerm::pair(lj()), skin(pos.view())).unwrap();
        assert_eq!(
            mic.compute(pos.view(), no_fold.view())
                .unwrap()
                .forces
                .nrows(),
            n
        );

        let halo = GhostHalo::new(cell(), pos.view(), 5.0, 0.0).unwrap();
        let mut ghosts = GhostPairs::new(ForceTerm::pair(lj()), halo).unwrap();
        let out = ghosts.compute(pos.view(), no_fold.view()).unwrap();
        assert_eq!(out.forces.nrows(), n);
        assert!(
            !ghosts.halo().ghosts().is_empty(),
            "this cell must actually produce copies, or the fold-back is untested"
        );
    }

    /// A molecule that straddles a face is scored exactly as the same molecule
    /// sitting in the middle of the cell.
    ///
    /// This is the whole point of rebinding the indices, and it is the
    /// assertion that the kernels stayed ignorant of periodicity while it
    /// happened. A rigid translation of a periodic system changes nothing
    /// physical, so the bonded energy and every force component must come back
    /// the same — while the bookkeeping underneath is completely different:
    /// crossing, the bond and the angle are measured through copies; centred,
    /// they are measured between owned atoms.
    ///
    /// Without the rebinding the crossing case reads a bond stretched by the
    /// width of the box, which is not a small error — it is the failure the
    /// ghost régime exists to remove.
    #[test]
    fn a_molecule_across_a_face_scores_as_one_that_is_not() {
        use molrs::core::Block;
        use molrs::core::Frame;
        use molrs::ff::forcefield::ForceField;
        use molrs::ff::ir::Params;
        use molrs::op::Idx;
        use ndarray::Array1;

        let l = 20.0_f64;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // A bent three-atom chain: two bonds and one angle, so both arities are
        // exercised and the angle's anchoring matters.
        let shape = array![[-1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.9, 0.0]];

        let frame = |_: ()| {
            let mut f = Frame::new();
            let mut atoms = Block::new();
            atoms
                .insert("type", Array1::from(vec!["a".to_string(); 3]).into_dyn())
                .unwrap();
            f.insert("atoms", atoms);
            let mut bonds = Block::new();
            bonds
                .insert("atomi", Array1::from(vec![0 as Idx, 1]).into_dyn())
                .unwrap();
            bonds
                .insert("atomj", Array1::from(vec![1 as Idx, 2]).into_dyn())
                .unwrap();
            bonds
                .insert("type", Array1::from(vec!["a-a".to_string(); 2]).into_dyn())
                .unwrap();
            f.insert("bonds", bonds);
            let mut angles = Block::new();
            angles
                .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("atomk", Array1::from(vec![2 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("type", Array1::from(vec!["a-a-a".to_string()]).into_dyn())
                .unwrap();
            f.insert("angles", angles);
            f
        };

        // Rest lengths deliberately off the actual geometry, so both terms
        // carry real energy and real force; a molecule at rest would satisfy
        // this test by contributing nothing.
        let mut field = ForceField::new("probe");
        field
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "a-a",
                &["a", "a"],
                Params::from_pairs(&[("k", 100.0), ("r0", 1.2)]),
            )
            .unwrap();
        field
            .def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "a-a-a",
                &["a", "a", "a"],
                Params::from_pairs(&[("k", 40.0), ("theta0", 114.6)]),
            )
            .unwrap();

        let run = |origin: [F; 3]| {
            let mut pts = shape.clone();
            for i in 0..pts.nrows() {
                for k in 0..3 {
                    pts[[i, k]] += origin[k];
                }
            }
            let (wrapped, _) = bx.wrap_shifts(pts.view());
            let halo = GhostHalo::new(bx.clone(), wrapped.view(), 4.0, 0.0).unwrap();
            let members = PotentialCompiler::new(&field)
                .compile(&frame(()))
                .unwrap()
                .into_members();
            let members = members
                .into_iter()
                .map(|p| (p, SpecialWeights::default()))
                .collect();
            let mut provider = GhostPairs::from_members(members, halo).unwrap();
            // The halo was built from these coordinates, so from its point of
            // view nothing has folded. A fold is reported exactly once, to the
            // halo that existed before it.
            let no_fold = Array2::<I>::zeros((wrapped.nrows(), 3));
            let out = provider.compute(wrapped.view(), no_fold.view()).unwrap();
            (out.energy, out.forces)
        };

        // Straddling the low x face, and sitting in the middle.
        let (e_cross, f_cross) = run([0.2, 10.0, 10.0]);
        let (e_mid, f_mid) = run([10.0, 10.0, 10.0]);

        assert!(
            e_mid.abs() > 1.0,
            "the molecule must be strained for this to assert anything; got {e_mid}"
        );
        assert!(
            (e_cross - e_mid).abs() / e_mid.abs() < 1e-12,
            "energy {e_cross} across the face vs {e_mid} in the middle"
        );
        for i in 0..3 {
            for k in 0..3 {
                assert!(
                    (f_cross[[i, k]] - f_mid[[i, k]]).abs() < 1e-9,
                    "force on atom {i} component {k}: {} across the face vs {} in the middle",
                    f_cross[[i, k]],
                    f_mid[[i, k]]
                );
            }
        }
    }

    /// A kernel that reads per-atom types gives the same answer through copies
    /// as it does through the minimum image.
    ///
    /// The two régimes index differently: the minimum-image table names owned
    /// atoms, the ghost table names `[owned | copies]`. A typed kernel looks
    /// its parameters up *by index*, so the ghost route is only right if the
    /// copies carry their owners' types — and if they did not, the failure
    /// would be a wrong parameter rather than a crash: a silently different
    /// well depth for every pair that reaches through a face.
    ///
    /// Two unlike types, alternating, so picking up the wrong one is visible.
    #[test]
    fn a_typed_kernel_reads_the_same_types_through_copies_as_through_the_image() {
        use molrs::ff::ir::CombiningRule;

        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Eight atoms on a 4 Å cube whose every edge runs *through* a face, so
        // every interaction is periodic and every one of them needs a copy.
        let pos = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = pos.nrows();
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let lj = || {
            PairLjCut::typed(
                type_id.clone(),
                &per_type,
                CombiningRule::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap()
        };

        let no_fold = Array2::<I>::zeros((n, 3));

        // Zero skin on both sides: a skin is a caching policy, not a periodic
        // one, and a stale edge would be a difference this test is not about.
        let skin = molrs::core::VerletSkin::new(
            NeighborList::new(cutoff),
            cutoff,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx.clone(),
        )
        .unwrap();
        let mut mic = MicPairs::new(ForceTerm::pair(lj()), skin).unwrap();
        let mic_out = mic.compute(pos.view(), no_fold.view()).unwrap();

        let halo = GhostHalo::new(bx, pos.view(), cutoff, 0.0).unwrap();
        let mut ghosts = GhostPairs::new(ForceTerm::pair(lj()), halo).unwrap();
        let ghost_out = ghosts.compute(pos.view(), no_fold.view()).unwrap();

        assert!(
            ghosts.halo().ghosts().len() > n,
            "this cell must materialise copies, or the gather is untested"
        );
        assert!(
            mic_out.energy.abs() > 1.0,
            "the system must interact for this to assert anything"
        );
        assert!(
            (ghost_out.energy - mic_out.energy).abs() / mic_out.energy.abs() < 1e-12,
            "energy {} through copies vs {} through the image",
            ghost_out.energy,
            mic_out.energy
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (ghost_out.forces[[i, k]] - mic_out.forces[[i, k]]).abs() < 1e-10,
                    "force on atom {i} component {k}: {} vs {}",
                    ghost_out.forces[[i, k]],
                    mic_out.forces[[i, k]]
                );
            }
        }
    }

    /// An excluded pair contributes nothing — not a small number, nothing.
    ///
    /// A neighbour table finds every pair inside the cutoff, so a bonded pair
    /// turns up in it like any other. Left alone it would be counted twice:
    /// once by the bond term and once at full Lennard-Jones strength, at bond
    /// length, where that term is enormous. Here every pair of the chain is
    /// 1-2 or 1-3, so with an exclusion depth of 3 the non-bonded sum must be
    /// exactly zero.
    ///
    /// Exactly, because an exclusion drops the pair rather than scaling it.
    #[test]
    fn a_fully_excluded_molecule_has_no_non_bonded_energy() {
        use molrs::core::BondDistanceWeights;
        use molrs::ff::ir::CombiningRule;

        let bx = SimBox::cube(20.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // A bent chain, every atom well inside the 6 Å cutoff of the others.
        let pos = array![
            [9.0_f64, 10.0, 10.0],
            [10.5, 10.0, 10.0],
            [11.3, 11.2, 10.0]
        ];
        let n = pos.nrows();

        let topo = bonded(n, &[[0, 1], [1, 2]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let lj = PairLjCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            CombiningRule::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();

        let halo = GhostHalo::new(bx.clone(), pos.view(), 6.0, 0.0).unwrap();
        let no_fold = Array2::<I>::zeros((n, 3));
        let mut with =
            GhostPairs::from_members(vec![(ForceTerm::pair(lj), special)], halo).unwrap();
        let out = with.compute(pos.view(), no_fold.view()).unwrap();

        assert_eq!(
            out.energy, 0.0,
            "every pair of this chain is 1-2 or 1-3, so nothing may be left"
        );
        assert!(
            out.forces.iter().all(|&f| f == 0.0),
            "an excluded pair exerts no force either"
        );

        // Without the weights the same configuration is enormous — which is
        // what the exclusion is preventing, and what a silent failure would
        // have contributed instead.
        let lj = PairLjCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            CombiningRule::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let halo = GhostHalo::new(bx, pos.view(), 6.0, 0.0).unwrap();
        let mut without = GhostPairs::new(ForceTerm::pair(lj), halo).unwrap();
        let bare = without.compute(pos.view(), no_fold.view()).unwrap();
        assert!(
            bare.energy > 100.0,
            "the unexcluded sum should be large; got {}",
            bare.energy
        );
    }

    /// The exclusions follow a molecule through a face.
    ///
    /// A pair naming a copy has to be weighted as the atom the copy is of — a
    /// bond graph knows owners, and a copy is the same atom seen through a
    /// face. If the lookup used the copy's index instead, the weight would come
    /// back as 1.0 and a bonded pair would be scored at full strength, but only
    /// for molecules near a boundary: a bug that hides everywhere except where
    /// it matters.
    #[test]
    fn exclusions_follow_a_molecule_through_a_face() {
        use molrs::core::BondDistanceWeights;
        use molrs::ff::ir::CombiningRule;

        let l = 20.0_f64;
        let cutoff = 6.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Two bent chains: one straddles the low x face, the other does not.
        let shape = array![
            [-1.5_f64, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 1.2, 0.0],
            [4.6, 0.3, 0.4],
            [6.1, 0.3, 0.4],
            [6.9, 1.5, 0.4],
        ];
        let n = shape.nrows();
        let topo = bonded(n, &[[0, 1], [1, 2], [3, 4], [4, 5]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let run = |origin: F| {
            let mut pts = shape.clone();
            for i in 0..pts.nrows() {
                pts[[i, 0]] += origin;
                pts[[i, 1]] += 10.0;
                pts[[i, 2]] += 10.0;
            }
            let (wrapped, _) = bx.wrap_shifts(pts.view());
            let lj = PairLjCut::typed(
                vec![0_u32; n],
                &[(0.3_f64, 3.4_f64)],
                CombiningRule::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap();
            let halo = GhostHalo::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let mut provider =
                GhostPairs::from_members(vec![(ForceTerm::pair(lj), special.clone())], halo)
                    .unwrap();
            let no_fold = Array2::<I>::zeros((n, 3));
            let out = provider.compute(wrapped.view(), no_fold.view()).unwrap();
            (out.energy, out.forces)
        };

        // Straddling (atom 0 lands at −0.5 and wraps), and shifted a third of
        // a cell so nothing crosses.
        let (e_cross, f_cross) = run(1.0);
        let (e_mid, f_mid) = run(1.0 + l / 3.0);

        assert!(
            e_mid.abs() > 1e-3,
            "the two chains must see each other for this to assert anything; got {e_mid}"
        );
        assert!(
            (e_cross - e_mid).abs() / e_mid.abs() < 1e-12,
            "energy {e_cross} across the face vs {e_mid} away from it"
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (f_cross[[i, k]] - f_mid[[i, k]]).abs() < 1e-9,
                    "force on atom {i} component {k}: {} vs {}",
                    f_cross[[i, k]],
                    f_mid[[i, k]]
                );
            }
        }
    }

    /// A term that pushes on the system from outside makes the step's virial
    /// `None`; a bonded term does not.
    ///
    /// Both read coordinates only, so before [`ForceTerm`] told them apart they
    /// were the same case to this loop — and the case it took was the bonded
    /// one, which tallies `Σ_a f_a ⊗ x_a`. That sum *is* a virial for a term
    /// whose forces cancel pairwise, and moves with the cell's origin for one
    /// whose forces do not. A constant field would have reported a number that
    /// changed when the box was re-centred.
    #[test]
    fn an_external_field_refuses_a_virial_where_a_bonded_term_gives_one() {
        use molrs::ff::potential::bond::BondHarmonic;

        struct Push;
        impl Potential for Push {
            fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
                let mut f = vec![0.0; coords.len()];
                for a in 0..coords.len() / 3 {
                    f[a * 3] = 0.5;
                }
                (-0.5 * coords.len() as F / 3.0, f)
            }
        }

        let pos = four_atoms();
        let no_fold = Array2::<I>::zeros((pos.nrows(), 3));

        let bond = BondHarmonic::new(vec![0], vec![1], vec![100.0], vec![1.0]);
        let mut bonded =
            MicPairs::new(ForceTerm::indexed(bond), skin(pos.view())).expect("bonded provider");
        let out = bonded.compute(pos.view(), no_fold.view()).unwrap();
        assert!(
            out.virial.is_some(),
            "a bonded term's forces cancel term by term, so its tally is a virial"
        );

        let mut pushed = MicPairs::new(ForceTerm::plain(Push), skin(pos.view())).expect("plain");
        let out = pushed.compute(pos.view(), no_fold.view()).unwrap();
        assert!(
            out.virial.is_none(),
            "an external field has no origin-independent tally to report"
        );
        assert!(
            out.forces.iter().any(|v| *v != 0.0),
            "the field must actually push, or the test proves nothing"
        );
    }

    /// The compiled door and the neighbour-driven door agree on a force field
    /// that *keeps* its 1-3 neighbours.
    ///
    /// LAMMPS's `special_bonds fene` is `[0, 1, 1]`: 1-2 excluded, 1-3 at full
    /// strength — a bead-spring chain has nothing else holding it open. Both
    /// doors read the weights, so neither evaluates a different force field
    /// from the other.
    #[test]
    fn both_doors_keep_the_1_3_pairs_a_fene_field_asks_for() {
        use molrs::core::Block;
        use molrs::core::Frame;
        use molrs::ff::forcefield::ForceField;
        use molrs::ff::ir::{Params, SpecialBonds};
        use molrs::ff::potential::intramolecular_pairs;
        use molrs::op::Idx;
        use ndarray::Array1;

        // Three beads, 0-1-2: (0,1) and (1,2) are 1-2, (0,2) is 1-3.
        let pts = array![
            [9.0_f64, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [10.6, 10.9, 10.0]
        ];
        let n = pts.nrows();
        let bonds = [[0usize, 1], [1, 2]];
        let topo = bonded(n, &bonds);

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("type", Array1::from(vec!["a".to_string(); n]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        let mut blk = Block::new();
        blk.insert(
            "atomi",
            Array1::from(bonds.iter().map(|b| b[0] as Idx).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap();
        blk.insert(
            "atomj",
            Array1::from(bonds.iter().map(|b| b[1] as Idx).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap();
        frame.insert("bonds", blk);
        let mut ang = Block::new();
        ang.insert("atomi", Array1::from(vec![0 as Idx]).into_dyn())
            .unwrap();
        ang.insert("atomj", Array1::from(vec![1 as Idx]).into_dyn())
            .unwrap();
        ang.insert("atomk", Array1::from(vec![2 as Idx]).into_dyn())
            .unwrap();
        frame.insert("angles", ang);

        let mut field = ForceField::new("fene-probe");
        field
            .def_style("pair", "lj/cut", Params::from_pairs(&[("cutoff", 6.0_f64)]))
            .unwrap()
            .def_type(
                "a",
                &["a"],
                Params::from_pairs(&[("epsilon", 0.3), ("sigma", 3.4)]),
            )
            .unwrap();
        field.set_special_bonds(SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 1.0, 1.0],
        });

        // The compiled door.
        let pairs =
            intramolecular_pairs(&frame, field.special_bonds()).expect("fene is expressible");
        assert_eq!(
            pairs.n_rows(),
            Some(1),
            "the 1-3 pair (0,2) stays and the two 1-2 pairs go"
        );
        let mut compiled_frame = frame.clone();
        compiled_frame.insert("pairs", pairs);
        let pots = PotentialCompiler::new(&field)
            .compile(&compiled_frame)
            .unwrap();
        let mut compiled = SelfPairedForces::new(pots);
        let no_fold = Array2::<I>::zeros((n, 3));
        let a = compiled.compute(pts.view(), no_fold.view()).unwrap();
        assert!(a.energy.abs() > 1e-6, "the 1-3 pair must carry energy");

        // The neighbour-driven door, over a table holding *every* pair.
        let members: Vec<(ForceTerm, SpecialWeights)> = PotentialCompiler::new(&field)
            .compile_typed(&frame)
            .unwrap()
            .into_iter()
            .map(|(pot, weights)| {
                let special = weights
                    .map(|w| w.special_weights(&topo))
                    .unwrap_or_default();
                (pot, special)
            })
            .collect();
        let mut driven = MicPairs::from_members(members, skin(pts.view())).unwrap();
        let b = driven.compute(pts.view(), no_fold.view()).unwrap();

        assert!(
            (a.energy - b.energy).abs() < 1e-12,
            "compiled {} vs neighbour-driven {}",
            a.energy,
            b.energy
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (a.forces[[i, k]] - b.forces[[i, k]]).abs() < 1e-12,
                    "force on {i} component {k}: {} vs {}",
                    a.forces[[i, k]],
                    b.forces[[i, k]]
                );
            }
        }
    }

    /// The exclusions work the same without copies to map through.
    ///
    /// Under the minimum image an index is already its own owner, so the split
    /// runs with an empty owner map. That identity case is easy to get wrong in
    /// a way that only shows up here: a mapping that assumed copies exist would
    /// index past the end of an empty slice, or quietly weight the wrong pair.
    #[test]
    fn the_minimum_image_route_excludes_the_same_pairs() {
        use molrs::core::BondDistanceWeights;
        use molrs::ff::ir::CombiningRule;

        let bx = SimBox::cube(20.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let pos = array![
            [9.0_f64, 10.0, 10.0],
            [10.5, 10.0, 10.0],
            [11.3, 11.2, 10.0]
        ];
        let n = pos.nrows();
        let topo = bonded(n, &[[0, 1], [1, 2]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let lj = PairLjCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            CombiningRule::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let skin = molrs::core::VerletSkin::new(
            NeighborList::new(6.0),
            6.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx,
        )
        .unwrap();
        let mut mic = MicPairs::from_members(vec![(ForceTerm::pair(lj), special)], skin).unwrap();
        let out = mic
            .compute(pos.view(), Array2::<I>::zeros((n, 3)).view())
            .unwrap();

        assert_eq!(
            out.energy, 0.0,
            "every pair of this chain is 1-2 or 1-3, so nothing may be left"
        );
        assert!(out.forces.iter().all(|&f| f == 0.0));
        assert_eq!(
            out.virial.expect("a typed kernel tallies one").components,
            [0.0; 6],
            "no pairs, no virial"
        );
    }

    /// A kernel bound to a fixed pair list is refused, not quietly humoured.
    ///
    /// `PotentialCompiler::compile` builds exactly such kernels: their
    /// parameters were resolved against the frame's `pairs` block, and they
    /// answer for that list whatever table they are handed. Given one, a
    /// provider would maintain a neighbour list, rebuild it, and report its
    /// counters, while the energy depended on none of it — and with exclusions
    /// in play it would be evaluated once per split table and come back
    /// multiplied.
    ///
    /// The refusal is at construction because a run that has started is too
    /// late to find out.
    #[test]
    fn a_kernel_bound_to_a_fixed_pair_list_is_refused() {
        let pos = four_atoms();
        let compiled = PairLjCut::compiled(vec![0], vec![1], vec![0.3], vec![3.4]);
        assert!(
            ForceTerm::pair(PairLjCut::compiled(vec![0], vec![1], vec![0.3], vec![3.4]))
                .binds_a_fixed_pair_list(),
            "a compiled kernel must say so"
        );

        let Err(err) = MicPairs::new(ForceTerm::pair(compiled), skin(pos.view())) else {
            panic!("a compiled kernel cannot be evaluated over a neighbour table")
        };
        let msg = format!("{err}");
        assert!(msg.contains("fixed pair list"), "{msg}");
        assert!(msg.contains("PotentialCompiler::compile_typed"), "{msg}");

        let halo = GhostHalo::new(cell(), pos.view(), 5.0, 0.0).unwrap();
        let compiled = PairLjCut::compiled(vec![0], vec![1], vec![0.3], vec![3.4]);
        assert!(
            GhostPairs::new(ForceTerm::pair(compiled), halo).is_err(),
            "and so does the halo"
        );

        // The typed form of the same style is accepted.
        assert!(!ForceTerm::pair(lj()).binds_a_fixed_pair_list());
        assert!(MicPairs::new(ForceTerm::pair(lj()), skin(pos.view())).is_ok());
    }

    /// A provider that keeps no list reports no counters — not zeroes.
    ///
    /// Zero rebuilds is a fact about a list that exists; a provider without one
    /// has no such fact, and saying `0` would let a caller conclude the list is
    /// fresh.
    #[test]
    fn counters_are_absent_rather_than_zero_when_there_is_no_list() {
        let pos = four_atoms();

        assert_eq!(
            SelfPairedForces::new(Potentials::new()).neighbor_stats(),
            NeighborStats::default()
        );

        let mic = MicPairs::new(ForceTerm::pair(lj()), skin(pos.view())).unwrap();
        let stats = mic.neighbor_stats();
        assert!(stats.edges.is_some());
        assert!(stats.rebuilds.is_some());
        assert!(stats.ago.is_some());

        let halo = GhostHalo::new(cell(), pos.view(), 5.0, 0.0).unwrap();
        let stats = GhostPairs::new(ForceTerm::pair(lj()), halo)
            .unwrap()
            .neighbor_stats();
        assert!(stats.rebuilds.is_some(), "a halo counts its rebuilds");
        assert!(
            stats.edges.is_none(),
            "the halo keeps no persistent edge list, so it has no edge count to report"
        );
    }
}
