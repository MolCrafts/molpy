//! Bonded topology through the ghost halo: the bonded index lists of a force
//! evaluation, resolved against the periodic copies that exist now.
#![allow(dead_code)]

use ndarray::{Array2, ArrayView2};

use molrs::core::GhostHalo;
use molrs::ff::potential::ForceTerm;
use molrs::op::{F, Fnx3View, I};

use super::error::MdError;

/// The bonded index lists, resolved against the copies that exist now.
///
/// The copies themselves are [`GhostHalo`]'s; this type owns only the bonded
/// indices resolved against them. Which periodic régime an integrator runs is
/// not its question — that is the force provider's
/// identity, and it is open rather than enumerated.
///
/// A bonded kernel holds indices, not geometry: it is told *these two atoms are
/// bonded* and takes the plain difference of their coordinates. For a bond that
/// straddles a face those coordinates are a cell apart, so the kernel measures a
/// bond stretched by the width of the box and answers with a force to match.
/// Nothing raises — the number is simply wrong, and it is wrong by orders of
/// magnitude.
///
/// Pointing each term at the copies that make it compact fixes that without any
/// kernel learning that periodic boundaries exist, which is the whole point of
/// the ghost régime.
///
/// This holds **no force field and no frame**. A term is a row of indices, and
/// the parameters that belong with that row live in the kernel, aligned by row;
/// so rebinding a term is rewriting a row, not rebuilding a force field. LAMMPS
/// draws the line in the same place — `Comm` (here [`GhostHalo`]) moves the copies, the `NTopo`
/// classes rebuild the bonded lists against `Domain::closest_image`.
///
/// # Anchoring
///
/// Each term is resolved against **one** atom, not edge by edge:
///
/// * a bond `(i, j)` against `i`;
/// * an angle `(i, j, k)` against the vertex `j`, so both arms are placed
///   relative to the same point;
/// * a dihedral or improper `(i, j, k, l)` against `j`, the atom the others are
///   at most two bonds from.
///
/// Folding each edge against the previous atom instead can pick images that are
/// individually closest and jointly inconsistent — the three atoms of an angle
/// ending in three different cells — and the angle is then measured on a shape
/// that does not exist. An anchor cannot do that: every atom is placed relative
/// to a single point.
///
/// # When it refuses
///
/// A copy can only be chosen from the copies that exist. If the halo does not
/// reach as far as a bonded term does — a short pair cutoff with a long
/// molecule — the nearest available copy is still a cell away, and the term
/// stays stretched. That is reported as [`MdError::Invalid`] rather than
/// returned: a silently mismeasured bond is precisely the failure this type
/// exists to remove, and swapping it for a differently-silent one would be no
/// improvement.
#[derive(Debug)]
pub struct BondedLists {
    /// One entry per member of the force evaluation, in member order. `None`
    /// for a member that holds no atom indices — a pair style reading a
    /// neighbour table has nothing here to rebind.
    entries: Vec<Option<TermList>>,
    /// The copy-list generation `current` names, or `None` before the first
    /// resolution. A sentinel generation would be a lie a reader could not
    /// check; absence is one they cannot misread.
    generation: Option<u64>,
}

#[derive(Debug)]
struct TermList {
    /// The indices the kernel resolved at construction: owned atoms, the
    /// source of truth. Never rewritten, so every resolution runs
    /// owned → image and is idempotent.
    source: Array2<u32>,
    /// `source` resolved against the copies `generation` names.
    current: Array2<u32>,
    /// The column every other atom in a row is placed relative to.
    anchor: usize,
}

impl BondedLists {
    /// Record each member's index table.
    ///
    /// Only a [`ForceTerm::Indexed`] has one; every other member keeps no indices
    /// and is left alone.
    ///
    /// Nothing is resolved here: resolution needs the owned coordinates, and
    /// the first [`refresh`](Self::refresh) has them.
    pub fn new(members: &[ForceTerm]) -> Self {
        let entries = members
            .iter()
            .map(|m| {
                m.terms().map(|source| {
                    // Arity fixes the anchor: a bond is placed from its first
                    // atom, everything longer from the atom the rest are
                    // nearest to.
                    let anchor = if source.ncols() <= 2 { 0 } else { 1 };
                    TermList {
                        current: source.clone(),
                        source,
                        anchor,
                    }
                })
            })
            .collect();
        Self {
            entries,
            generation: None,
        }
    }

    /// Re-resolve if the copies have changed under it.
    ///
    /// Two things invalidate a resolved index, and only one is obvious.
    ///
    /// A **rebuild** replaces the copy list, and an index names a position in
    /// that list. This is checked rather than remembered: the resolution
    /// records the generation it was made against.
    ///
    /// A **fold** does not move an atom — it relabels it — so it costs no
    /// displacement and trips no rebuild. But "closest copy" is answered
    /// against the *stored* coordinate, and that has just jumped a lattice
    /// vector, so a bond resolved before the fold now reaches across the cell
    /// instead of through the face. Nothing is out of place; the index is
    /// answering an older question.
    pub fn refresh(
        &mut self,
        halo: &GhostHalo,
        owned: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<(), MdError> {
        let stale = self.generation != Some(halo.ghosts().generation());
        if stale {
            self.resolve(halo, owned, None)?;
            self.generation = Some(halo.ghosts().generation());
        } else if wrap_shifts.iter().any(|&m| m != 0) {
            let mask: Vec<bool> = (0..owned.nrows())
                .map(|a| {
                    wrap_shifts[[a, 0]] != 0 || wrap_shifts[[a, 1]] != 0 || wrap_shifts[[a, 2]] != 0
                })
                .collect();
            self.resolve(halo, owned, Some(&mask))?;
        }
        debug_assert_eq!(
            self.generation,
            Some(halo.ghosts().generation()),
            "bonded lists are resolved against a copy list that no longer exists"
        );
        Ok(())
    }

    /// ForceTerm `m`'s indices as they stand, or `None` if it keeps none.
    pub fn current(&self, m: usize) -> Option<ArrayView2<'_, u32>> {
        self.entries.get(m)?.as_ref().map(|e| e.current.view())
    }

    /// The copy-list generation the indices name, or `None` before the first
    /// resolution.
    pub fn generation(&self) -> Option<u64> {
        self.generation
    }

    /// How many members keep indices.
    pub fn bound(&self) -> usize {
        self.entries.iter().filter(|e| e.is_some()).count()
    }

    /// Re-resolve the terms, or only those that a fold can have disturbed.
    ///
    /// `folded` is `None` for a rebuild — every index names a position in a
    /// copy list that no longer exists, so every term has to be redone. For a
    /// fold it is the per-atom mask of who actually crossed: a fold relabels
    /// an atom without moving it, so a term none of whose atoms folded is
    /// resolved against exactly the geometry it was resolved against before,
    /// and redoing it would produce the same numbers at the cost of a copy
    /// search per partner. In a dense system something folds almost every
    /// step, so this is the difference between O(all terms) per step and
    /// O(terms touching the handful that crossed).
    fn resolve(
        &mut self,
        halo: &GhostHalo,
        owned: Fnx3View<'_>,
        folded: Option<&[bool]>,
    ) -> Result<(), MdError> {
        let set = halo.ghosts();
        // Beyond half the smallest plane spacing the minimum image is
        // ambiguous, so a term still that long after remapping has not been
        // resolved — it has been guessed.
        let min_width = halo
            .simbox()
            .nearest_plane_distance()
            .iter()
            .copied()
            .fold(F::INFINITY, F::min);
        let limit = (0.5 * min_width).min(set.reach());
        let all = halo.combined();
        let touched = |row: ndarray::ArrayView1<'_, u32>| match folded {
            None => true,
            Some(mask) => row.iter().any(|&a| mask[a as usize]),
        };

        for entry in self.entries.iter_mut().flatten() {
            let (n_terms, arity) = entry.source.dim();
            let anchor_col = entry.anchor;
            for t in 0..n_terms {
                if !touched(entry.source.row(t)) {
                    continue;
                }
                let anchor = entry.source[[t, anchor_col]] as usize;
                entry.current[[t, anchor_col]] = anchor as u32;
                for c in 0..arity {
                    if c == anchor_col {
                        continue;
                    }
                    let partner = entry.source[[t, c]] as usize;
                    let mapped = set
                        .closest_image(owned, anchor, partner)
                        .map_err(MdError::from)?;
                    entry.current[[t, c]] = mapped as u32;
                }
            }

            // Every term must now be compact. Checking it here, once per
            // rebuild, is cheaper than any kernel could and catches the case no
            // kernel can see: a halo that never reached far enough.
            for t in 0..n_terms {
                if !touched(entry.source.row(t)) {
                    continue;
                }
                let a = entry.current[[t, anchor_col]] as usize;
                for c in 0..arity {
                    if c == anchor_col {
                        continue;
                    }
                    let p = entry.current[[t, c]] as usize;
                    let d2: F = (0..3)
                        .map(|k| {
                            let d = all[[p, k]] - all[[a, k]];
                            d * d
                        })
                        .sum();
                    if d2.sqrt() > limit {
                        return Err(MdError::Invalid(format!(
                            "a bonded term still spans {:.3} Å after remapping, past the \
                             {limit:.3} Å at which the periodic image stops being decidable. \
                             The halo does not reach as far as this molecule: build it with \
                             a reach that covers the bonded extent, not only the pair cutoff",
                            d2.sqrt()
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod remap_tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::potential::fixtures::OneShot;

    use molrs::core::SimBox;
    use molrs::ff::potential::angle::AngleHarmonic;
    use molrs::ff::potential::bond::BondHarmonic;
    use ndarray::array;

    /// Resolve one kernel's indices against a halo and hand back the table.
    ///
    /// This goes through the same [`BondedLists`] the force path uses — there
    /// is no test-only door into the remapping, because a door tests take and
    /// production does not is a door that can be right while production is
    /// wrong.
    fn resolve_one(
        pot: ForceTerm,
        owned: Fnx3View<'_>,
        halo: &GhostHalo,
    ) -> Result<Array2<u32>, MdError> {
        let members = vec![pot];
        let mut lists = BondedLists::new(&members);
        let no_fold = Array2::<I>::zeros((owned.nrows(), 3));
        lists.refresh(halo, owned, no_fold.view())?;
        Ok(lists
            .current(0)
            .expect("this kernel keeps indices")
            .to_owned())
    }

    /// A cutoff past half the smallest perpendicular width is refused.
    ///
    /// Beyond it a pair has more than one image inside the cutoff, and the
    /// halo's pair table keeps one per owner pair — so it would not
    /// double-count, it would *under*-count, and nothing downstream could tell.
    /// `VerletSkin::new` has drawn this line for the minimum image all along;
    /// the copies obey the same physics.
    /// A bond whose two atoms sit on opposite sides of a face is rewritten to
    /// point at a copy, and the kernel that reads it — unchanged, and still
    /// ignorant of periodicity — then measures the bond that is actually there.
    #[test]
    fn a_crossing_bond_is_rewritten_to_its_closest_copy() {
        let bx = SimBox::new(Cell::Lengths([10.0; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[9.5_f64, 5.0, 5.0], [0.5, 5.0, 5.0]];
        let halo = GhostHalo::new(bx, owned.view(), 2.0, 0.0).unwrap();

        let pot = BondHarmonic::new(vec![0], vec![1], vec![100.0], vec![1.0]);
        let terms = resolve_one(ForceTerm::indexed(pot), owned.view(), &halo).unwrap();

        let j = terms[[0, 1]] as usize;
        assert!(j >= 2, "atomj must now name a copy, not atom 1; got {j}");
        assert_eq!(terms[[0, 0]], 0, "the anchor never moves");

        // Straight through the kernel, on the combined coordinates, with the
        // table it was handed.
        let all = halo.ghosts().combined(owned.view()).unwrap();
        let flat: Vec<F> = all.iter().copied().collect();
        let pot = BondHarmonic::new(vec![0], vec![1], vec![100.0], vec![1.0]);
        let (e, _) = pot.over_terms(&flat, terms.view());
        assert!(e.abs() < 1e-9, "the bond is at rest length; got E = {e}");

        // The unrewritten indices are what the bug looks like: same kernel,
        // same coordinates, a bond read as eight times its rest length.
        assert!((pot.energy_forces_at(&flat).0 - 6400.0).abs() < 1e-9);
    }

    /// Both arms of an angle are resolved against the vertex, so the three
    /// atoms end up in one cell rather than in three.
    #[test]
    fn an_angle_is_anchored_on_its_vertex() {
        let bx = SimBox::new(Cell::Lengths([10.0; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Vertex just inside the low face; the arms straddle it.
        let owned = array![[9.3_f64, 5.0, 5.0], [0.2, 5.0, 5.0], [1.2, 5.0, 5.0]];
        let halo = GhostHalo::new(bx, owned.view(), 3.0, 0.0).unwrap();

        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![50.0], vec![2.9]);
        let terms = resolve_one(ForceTerm::indexed(pot), owned.view(), &halo).unwrap();
        let (i, j, k) = (
            terms[[0, 0]] as usize,
            terms[[0, 1]] as usize,
            terms[[0, 2]] as usize,
        );
        assert_eq!(j, 1, "the vertex is the anchor and never moves");
        assert!(i >= 3, "the arm across the face must be a copy; got {i}");
        assert_eq!(k, 2, "the arm already beside the vertex is left alone");

        let all = halo.ghosts().combined(owned.view()).unwrap();
        let d = |a: usize, b: usize| {
            ((0..3)
                .map(|c| (all[[a, c]] - all[[b, c]]).powi(2))
                .sum::<F>())
            .sqrt()
        };
        assert!((d(i, j) - 0.9).abs() < 1e-9, "arm i-j is {}", d(i, j));
        assert!((d(k, j) - 1.0).abs() < 1e-9, "arm k-j is {}", d(k, j));
    }

    /// A molecule longer than the halo cannot be resolved, and saying so is the
    /// whole improvement: the alternative is a bond measured against whichever
    /// copy happened to exist.
    #[test]
    fn a_term_the_halo_cannot_reach_is_refused() {
        let bx = SimBox::new(Cell::Lengths([30.0; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Bonded, but 14 Å apart — beyond a halo built for a 2 Å pair cutoff.
        let owned = array![[1.0_f64, 5.0, 5.0], [15.0, 5.0, 5.0]];
        let halo = GhostHalo::new(bx, owned.view(), 2.0, 0.0).unwrap();

        let pot = BondHarmonic::new(vec![0], vec![1], vec![100.0], vec![1.0]);
        let err = resolve_one(ForceTerm::indexed(pot), owned.view(), &halo).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("still spans"), "{msg}");
        assert!(msg.contains("does not reach"), "{msg}");
    }

    /// A member that keeps no indices has nothing to rebind — the ghost path
    /// must not require a topology it was not given.
    #[test]
    fn a_member_without_indices_is_left_alone() {
        let bx = SimBox::new(Cell::Lengths([10.0; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[1.0_f64, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let halo = GhostHalo::new(bx, owned.view(), 2.0, 0.0).unwrap();

        let members: Vec<ForceTerm> =
            vec![ForceTerm::plain(molrs::ff::potential::Potentials::new())];
        let mut lists = BondedLists::new(&members);
        assert_eq!(lists.bound(), 0, "an aggregate keeps no atom indices");
        let no_fold = Array2::<I>::zeros((owned.nrows(), 3));
        lists.refresh(&halo, owned.view(), no_fold.view()).unwrap();
        assert!(lists.current(0).is_none());
    }
}

#[cfg(test)]
mod owned_potential_tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::core::Frame;
    use molrs::core::SimBox;
    use molrs::ff::compile::PotentialCompiler;
    use molrs::ff::forcefield::ForceField;
    use molrs::ff::ir::Params;
    use molrs::op::Idx;
    use ndarray::{Array1, array};

    /// Two atoms bonded across a face, drifting until the halo rebuilds.
    ///
    /// The bond is a pair of *indices*, and which copy those indices should
    /// name is re-decided at every rebuild. Left to the caller that is a
    /// cadence to get right and silent to get wrong: the kernels keep reading
    /// whatever indices they were built with. So the halo re-resolves them in
    /// the same call that rebuilt the copies, and this checks that it does —
    /// the bonded energy must stay smooth across a rebuild, not step.
    #[test]
    fn the_halo_re_resolves_its_topology_when_it_rebuilds() {
        let l = 20.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Bonded, 1 Å apart the short way round the +x face — and parked in a
        // corner, with two spectator atoms beside them. Drifting on all three
        // axes churns which copies survive the cull, so the ghost list is
        // reordered from one rebuild to the next and an index kept across one
        // names a different copy than it did. A pair alone in the middle of a
        // face would not show that: its copy list is stable, and a stale index
        // would keep working by luck.
        let mut owned = array![
            [19.5_f64, 19.4, 19.6],
            [0.5, 19.4, 19.6],
            [19.3, 0.4, 19.2],
            [0.7, 19.8, 0.3],
        ];

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from(vec!["a-a".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("bonds", bonds);
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from(vec!["a".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        // A real bond style, with a rest length equal to the bond's actual
        // length, so a correctly-resolved bond has zero energy and a
        // mis-resolved one does not.
        let mut field = ForceField::new("probe");
        field
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "a-a",
                &["a", "a"],
                Params::from_iter([("k", 100.0), ("r0", 1.0)]),
            )
            .unwrap();

        // A small skin, so drifting forces several rebuilds over the run.
        let mut halo = GhostHalo::new(bx.clone(), owned.view(), 3.0, 0.2).unwrap();
        // The force field compiles once, here. Nothing below names it again:
        // what MD carries forward is the kernels and their index lists.
        let members = PotentialCompiler::new(&field)
            .compile(&frame)
            .unwrap()
            .into_members();
        let mut lists = BondedLists::new(&members);

        let mut seen_rebuild = false;
        for step in 0..60 {
            // Both atoms drift together: the bond length never changes, so a
            // bonded energy that moves at all is the bookkeeping's doing.
            for i in 0..owned.nrows() {
                owned[[i, 0]] += 0.02;
                owned[[i, 1]] += 0.013;
                owned[[i, 2]] += 0.017;
            }
            let (wrapped, m) = bx.wrap(owned.view());
            owned = wrapped;

            let before = halo.rebuilds();
            halo.advance(owned.view(), m.view()).unwrap();
            lists.refresh(&halo, owned.view(), m.view()).unwrap();
            let all = halo.combined().clone();
            let flat: Vec<F> = all.iter().copied().collect();
            let mut e = 0.0;
            for (mi, member) in members.iter().enumerate() {
                let terms = lists.current(mi).expect("a bond style keeps indices");
                let ForceTerm::Indexed(pot) = member else {
                    unreachable!("a bond style is an indexed member")
                };
                e += pot.over_terms(&flat, terms).0;
            }
            let n_terms = lists.bound();
            let all_rows = all.nrows();

            // Without this the assertion below is satisfied by an empty
            // potential, which would prove nothing at all.
            assert!(
                n_terms > 0,
                "step {step}: the force field produced no bonded terms"
            );

            assert!(
                e.abs() < 1e-9,
                "step {step}: bonded energy {e}; the bond never changed length"
            );
            assert!(all_rows >= owned.nrows());
            if halo.rebuilds() > before {
                seen_rebuild = true;
            }
        }
        assert!(
            seen_rebuild,
            "the run must cross a rebuild, or it proves nothing"
        );
    }

    /// The invariant behind the re-resolution, asserted directly: a bonded
    /// topology is always resolved against the copy list that exists now.
    ///
    /// The re-resolution is keyed on the generation, not on a boolean the
    /// caller sets when it rebuilds (a thing to remember, which a test can only
    /// catch forgetting if the fixture happens to reorder the list). Keyed on
    /// the generation, the condition *is* the
    /// property, and this checks it every step rather than hoping a wrong
    /// answer shows up in an energy.
    #[test]
    fn the_topology_is_never_resolved_against_a_copy_list_that_is_gone() {
        let l = 20.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let mut owned = array![
            [19.5_f64, 19.4, 19.6],
            [0.5, 19.4, 19.6],
            [19.3, 0.4, 19.2],
            [0.7, 19.8, 0.3],
        ];

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from(vec!["a-a".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("bonds", bonds);
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from(vec!["a".to_string(); 4]).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut field = ForceField::new("probe");
        field
            .def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "a-a",
                &["a", "a"],
                Params::from_iter([("k", 100.0), ("r0", 1.0)]),
            )
            .unwrap();

        let mut halo = GhostHalo::new(bx.clone(), owned.view(), 3.0, 0.2).unwrap();
        let members = PotentialCompiler::new(&field)
            .compile(&frame)
            .unwrap()
            .into_members();
        let mut lists = BondedLists::new(&members);

        let mut generations: Vec<u64> = vec![halo.ghosts().generation()];
        for _ in 0..60 {
            for i in 0..owned.nrows() {
                owned[[i, 0]] += 0.02;
                owned[[i, 1]] += 0.013;
                owned[[i, 2]] += 0.017;
            }
            let (wrapped, m) = bx.wrap(owned.view());
            owned = wrapped;
            halo.advance(owned.view(), m.view()).unwrap();
            lists.refresh(&halo, owned.view(), m.view()).unwrap();
            assert_eq!(
                lists.generation(),
                Some(halo.ghosts().generation()),
                "the lists must name copies that exist"
            );
            let g = halo.ghosts().generation();
            if *generations.last().unwrap() != g {
                generations.push(g);
            }
        }
        assert!(
            generations.len() > 2,
            "the copy list must have been rebuilt more than once; saw {} generations",
            generations.len()
        );
    }
}
#[cfg(test)]
mod bonded_tests {
    use crate::core::Cell;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::GhostSet;
    use molrs::core::SimBox;
    use molrs::ff::potential::bond::BondHarmonic;
    use molrs::op::F;
    use ndarray::Array2;
    use ndarray::array;

    /// A bond across a periodic face, evaluated by a kernel that knows nothing
    /// about periodicity.
    ///
    /// The two atoms are 1 Å apart the short way round a 10 Å cell and 9 Å
    /// apart the long way. Handed their owned indices, the kernel measures the
    /// long way and reports a bond stretched to eight times its rest length;
    /// handed the partner's closest copy, it measures the short way and reports
    /// a bond at rest. Neither call raises — that is the point.
    #[test]
    fn a_bond_across_a_face_needs_the_closest_image() {
        let l = 10.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[9.5_f64, 5.0, 5.0], [0.5, 5.0, 5.0]];
        let r0 = 1.0;
        let k = 100.0;

        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let all = set.combined(owned.view()).unwrap();
        let flat: Vec<F> = all.iter().copied().collect();

        // The naive reading: atoms 0 and 1 as stored.
        let naive = BondHarmonic::new(vec![0], vec![1], vec![k], vec![r0]);
        let (e_naive, _) = naive.energy_forces_at(&flat);
        // r = 9, so dr = 8 and E = k·dr² = 100·64.
        assert!(
            (e_naive - 6400.0).abs() < 1e-9,
            "the naive reading measures the long way round: {e_naive}"
        );

        // The periodic reading: atom 1's closest copy.
        let j = set.closest_image(owned.view(), 0, 1).unwrap();
        assert!(
            j >= set.n_owned(),
            "the closest copy is a ghost, not atom 1"
        );
        let fixed = BondHarmonic::new(vec![0], vec![j], vec![k], vec![r0]);
        let (e_fixed, f_fixed) = fixed.energy_forces_at(&flat);
        assert!(
            e_fixed.abs() < 1e-9,
            "the bond is at its rest length: {e_fixed}"
        );

        // And the force it produces belongs to atom 1 once the copy is folded
        // back — the ghost row must not be left holding it.
        let mut f = Array2::from_shape_vec((all.nrows(), 3), f_fixed).unwrap();
        set.reverse_comm(&mut f).unwrap();
        for k in 0..3 {
            assert!(f[[0, k]].abs() < 1e-9, "no force at rest length");
            assert!(f[[1, k]].abs() < 1e-9, "no force at rest length");
        }
    }

    /// Stretch the same bond and check the force goes the right way: the atom
    /// on the low side is pulled *down*, through the face, not up across the
    /// whole cell.
    #[test]
    fn the_force_on_a_crossing_bond_points_the_short_way() {
        let l = 10.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // 1.5 Å apart the short way; rest length 1.0, so it pulls together.
        let owned = array![[9.5_f64, 5.0, 5.0], [1.0, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 3.0).unwrap();
        let all = set.combined(owned.view()).unwrap();
        let flat: Vec<F> = all.iter().copied().collect();

        let j = set.closest_image(owned.view(), 0, 1).unwrap();
        let pot = BondHarmonic::new(vec![0], vec![j], vec![100.0], vec![1.0]);
        let (_e, f) = pot.energy_forces_at(&flat);
        let mut f = Array2::from_shape_vec((all.nrows(), 3), f).unwrap();
        set.reverse_comm(&mut f).unwrap();

        // Atom 0 is at 9.5 and its partner is effectively at 10.5, so the pull
        // on atom 0 is toward +x — out through the face.
        assert!(
            f[[0, 0]] > 0.0,
            "atom 0 must be pulled toward the face it is bonded through, got {}",
            f[[0, 0]]
        );
        assert!(f[[1, 0]] < 0.0, "and atom 1 the other way");
        // Newton's third law survives the fold.
        for k in 0..3 {
            assert!((f[[0, k]] + f[[1, k]]).abs() < 1e-9);
        }
    }

    /// An angle must be built against one anchor. Resolving each partner
    /// against the centre keeps the three atoms in one cell; resolving each
    /// edge against the previous atom can place them in three.
    #[test]
    fn an_angle_resolves_both_arms_against_its_vertex() {
        let l = 10.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Vertex just inside the low face, arms on either side of it.
        let owned = array![[9.3_f64, 5.0, 5.0], [0.2, 5.0, 5.0], [1.2, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 3.0).unwrap();
        let all = set.combined(owned.view()).unwrap();

        // Anchored on atom 1, the vertex.
        let a = set.closest_image(owned.view(), 1, 0).unwrap();
        let c = set.closest_image(owned.view(), 1, 2).unwrap();
        let p = |idx: usize| [all[[idx, 0]], all[[idx, 1]], all[[idx, 2]]];
        let (pa, pv, pc) = (p(a), p(1), p(c));

        // Both arms are the short way round: 0.9 Å and 1.0 Å.
        let d = |x: [F; 3], y: [F; 3]| {
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        };
        assert!((d(pa, pv) - 0.9).abs() < 1e-9, "arm A is {}", d(pa, pv));
        assert!((d(pc, pv) - 1.0).abs() < 1e-9, "arm C is {}", d(pc, pv));

        // The three atoms end up collinear and adjacent, which is the shape
        // they actually have; the stored coordinates alone suggest otherwise.
        assert!((d(pa, pc) - 1.9).abs() < 1e-9);
        assert!(
            (owned[[0, 0]] - owned[[2, 0]]).abs() > 8.0,
            "as stored they look 8 Å apart"
        );
    }
}

#[cfg(test)]
mod force_path_tests {
    use crate::core::Cell;
    use crate::core::{NeighborList, NeighborPolicy, VerletSkin};
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::GhostSet;
    use molrs::core::SimBox;
    use molrs::ff::potential::pair::PairLjCut;
    use molrs::op::F;
    use ndarray::Array2;
    use ndarray::array;

    /// The end of the argument: a real potential evaluated through ghosts must
    /// agree with the same potential evaluated through the minimum image.
    ///
    /// Everything this layer adds — the copies, the tie-break, the reverse
    /// accumulation — exists so that a potential can be handed an ordinary
    /// cluster and still produce periodic physics. If the two routes disagree,
    /// one of them is wrong, and the whole construction is worth nothing.
    #[test]
    fn a_potential_through_ghosts_matches_the_same_potential_through_mic() {
        let l = 10.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let cutoff = 3.0;

        // Wrapped positions, deliberately crowded against the faces so most of
        // the interaction is found through copies rather than directly.
        // Separations are kept near the LJ minimum so the energy is O(1) and a
        // disagreement shows up as a disagreement, not as noise on a spike: two
        // atoms 0.3 A apart would put 10^11 kcal/mol on the scale and hide
        // everything else.
        let owned = array![
            [0.6_f64, 0.6, 0.6],
            [8.4, 0.6, 0.6],
            [0.6, 8.0, 5.0],
            [5.0, 5.0, 0.7],
            [5.0, 5.0, 8.5],
            [2.6, 2.6, 2.6],
            [8.2, 8.2, 8.2],
            [5.6, 1.0, 8.0],
        ];
        let n = owned.nrows();
        let flat_owned: Vec<F> = owned.iter().copied().collect();

        let lj = || PairLjCut::new(0.3, 3.4, cutoff, 12, 6, false, false).expect("lj");

        // --- route A: minimum image, through the existing skin ---
        let mut skin = VerletSkin::new(
            NeighborList::new(cutoff),
            cutoff,
            NeighborPolicy::default(),
            owned.view(),
            bx.clone(),
        )
        .expect("skin");
        let mic_pairs = skin.pairs_at(owned.view()).expect("mic pairs");
        let (e_mic, f_mic) = lj().over_pairs(&flat_owned, mic_pairs);

        // --- route B: ghosts, an ordinary non-periodic cluster ---
        let set = GhostSet::borders(&bx, owned.view(), cutoff).unwrap();
        let all = set.combined(owned.view()).unwrap();
        let flat_all: Vec<F> = all.iter().copied().collect();
        let ghost_pairs = set.pairs(owned.view(), cutoff).unwrap();
        let (e_ghost, f_ext) = lj().over_pairs(&flat_all, &ghost_pairs);

        // Ghost forces belong to the atoms they copy.
        let mut f_ext = Array2::from_shape_vec((all.nrows(), 3), f_ext).unwrap();
        set.reverse_comm(&mut f_ext).unwrap();

        // Relative, against the scale of the terms being summed: the two routes
        // add the same pairs in different orders, so they agree to rounding and
        // not to the bit.
        let scale = e_mic.abs().max(1.0);
        assert!(
            (e_ghost - e_mic).abs() / scale < 1e-12,
            "energy: ghosts {e_ghost} vs mic {e_mic}"
        );
        let fscale = f_mic.iter().fold(1.0_f64, |a, b| a.max(b.abs()));
        for i in 0..n {
            for k in 0..3 {
                let a = f_ext[[i, k]];
                let b = f_mic[3 * i + k];
                assert!(
                    (a - b).abs() / fscale < 1e-12,
                    "force on atom {i} axis {k}: ghosts {a} vs mic {b}"
                );
            }
        }

        // Nothing is left on the copies, and momentum is conserved.
        for g in 0..set.len() {
            for k in 0..3 {
                assert_eq!(f_ext[[n + g, k]], 0.0);
            }
        }
        for k in 0..3 {
            let net: F = (0..n).map(|i| f_ext[[i, k]]).sum();
            assert!(net.abs() / fscale < 1e-12, "net force on axis {k} is {net}");
        }
    }
}

#[cfg(test)]
mod virial_tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::{SimBox, Virial};
    use molrs::ff::potential::pair::PairLjCut;
    use molrs::op::F;
    use ndarray::{Array2, array};

    fn lj(cutoff: F) -> PairLjCut {
        PairLjCut::new(0.3, 3.4, cutoff, 12, 6, false, false).unwrap()
    }

    /// Two atoms, one pair, a virial that can be written down by hand.
    ///
    /// The force on atom 0 is `f` along `+x` and on atom 1 it is `−f`, so
    /// `Σ f ⊗ r` over the pair is `f·r_x` in the `xx` slot and nothing
    /// anywhere else.
    #[test]
    fn a_single_pair_gives_the_hand_computed_tensor() {
        let bx = SimBox::new(Cell::Lengths([30.0; 3]), [0.0_f64, 0.0, 0.0], [false; 3]).unwrap();
        let owned = array![[10.0_f64, 10.0, 10.0], [14.0, 10.0, 10.0]];
        let halo = GhostHalo::new(bx, owned.view(), 5.0, 0.0).unwrap();
        let all = halo.combined().clone();
        assert_eq!(all.nrows(), 2, "a free box makes no copies");

        let pairs = halo.pairs().clone();
        let flat: Vec<F> = all.iter().copied().collect();
        let (_e, f) = lj(5.0).over_pairs(&flat, &pairs);
        let mut f = Array2::from_shape_vec((2, 3), f).unwrap();

        let fx = f[[0, 0]];
        let w = halo.reverse_comm_with_virial(&mut f, all.view()).unwrap();

        // Σ f ⊗ r = f0⊗x0 + f1⊗x1 = fx·10 + (−fx)·14 = −4·fx.
        assert!(
            (w.components[0] - (-4.0 * fx)).abs() < 1e-12,
            "xx is {}, hand value {}",
            w.components[0],
            -4.0 * fx
        );
        for c in 1..6 {
            assert!(w.components[c].abs() < 1e-12, "component {c} should vanish");
        }
    }

    /// The virial cannot depend on where the box's origin is, and the tally
    /// order is the only thing standing between it and that dependence.
    ///
    /// Summing `f ⊗ x` over owned atoms *and copies*, before the copies' forces
    /// are folded away, gives `Σ f_ij ⊗ r_ij` — separations, not positions.
    /// Folding first and summing over owned atoms alone leaves
    /// `Σ f_ij ⊗ H·s` behind, which moves with the origin. This translates the
    /// whole system by a lattice vector and checks that nothing moves.
    #[test]
    fn the_virial_does_not_know_where_the_origin_is() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let base = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];

        let virial_of = |shift: F| {
            let mut pts = base.clone();
            for i in 0..pts.nrows() {
                for k in 0..3 {
                    pts[[i, k]] += shift;
                }
            }
            let (wrapped, _m) = bx.wrap(pts.view());
            let halo = GhostHalo::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let all = halo.combined().clone();
            let pairs = halo.pairs().clone();
            let flat: Vec<F> = all.iter().copied().collect();
            let (_e, f) = lj(cutoff).over_pairs(&flat, &pairs);
            let mut f = Array2::from_shape_vec((all.nrows(), 3), f).unwrap();
            halo.reverse_comm_with_virial(&mut f, all.view()).unwrap()
        };

        let a = virial_of(0.0);
        let b = virial_of(l); // a whole cell: the same physical system
        let scale = a.components.iter().fold(1.0_f64, |m, c| m.max(c.abs()));
        for c in 0..6 {
            assert!(
                (a.components[c] - b.components[c]).abs() / scale < 1e-10,
                "component {c}: {} vs {} after a lattice translation",
                a.components[c],
                b.components[c]
            );
        }
        assert!(
            scale > 1.0,
            "the virial must be non-trivial to mean anything"
        );
    }

    /// The order, shown rather than asserted: tallying after the fold gives a
    /// different number, and the difference is exactly the `Σ f ⊗ H·s` the
    /// derivation predicts.
    #[test]
    fn tallying_after_the_fold_loses_the_lattice_terms() {
        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
        ];
        let halo = GhostHalo::new(bx, owned.view(), cutoff, 0.0).unwrap();
        let all = halo.combined().clone();
        let pairs = halo.pairs().clone();
        let flat: Vec<F> = all.iter().copied().collect();
        let (_e, f0) = lj(cutoff).over_pairs(&flat, &pairs);

        // Right: tally over owned and copies, then fold.
        let mut f = Array2::from_shape_vec((all.nrows(), 3), f0.clone()).unwrap();
        let right = halo.reverse_comm_with_virial(&mut f, all.view()).unwrap();

        // Wrong: fold first, then tally over what is left.
        let mut f = Array2::from_shape_vec((all.nrows(), 3), f0).unwrap();
        halo.ghosts().reverse_comm(&mut f).unwrap();
        let mut wrong = Virial::ZERO;
        for a in 0..halo.ghosts().n_owned() {
            wrong.add_outer(
                [f[[a, 0]], f[[a, 1]], f[[a, 2]]],
                [all[[a, 0]], all[[a, 1]], all[[a, 2]]],
            );
        }

        let gap = (right.trace() - wrong.trace()).abs();
        assert!(
            gap / right.trace().abs().max(1.0) > 1e-3,
            "the two orders must differ, or this test is not watching anything: \
             {} vs {}",
            right.trace(),
            wrong.trace()
        );
    }
}
