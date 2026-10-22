//! [`GhostHalo`]: the ghost atoms of an MD run, kept current across a
//! trajectory.
//!
//! [`GhostSet`] is a snapshot of which periodic copies exist; the halo decides
//! when that snapshot has gone stale, rebuilds it, moves it every step, and
//! hands out the pair table over `[owned | ghost]` — the ghost régime's
//! counterpart to [`VerletSkin`](crate::core::VerletSkin).

use ndarray::{Array2, ArrayView2};

use crate::core::Neighbors;
use crate::core::SimBox;
use crate::core::Virial;
use crate::op::{F, Fnx3, Fnx3View, I};

use super::{GhostError, GhostSet};

/// Owns the ghost atoms of an MD run and keeps them current.
///
/// [`GhostSet`] is a **snapshot** — which copies exist and where they are right
/// now. This keeps one current across a trajectory: it decides when the
/// snapshot has gone stale, rebuilds it, moves it every step, and hands out the
/// pair table and the re-resolved topology that follow. It is the ghost
/// régime's counterpart to
/// [`VerletSkin`](molrs::core::VerletSkin) and answers the same
/// question: has anything moved far enough that what was frozen at the last
/// rebuild is no longer complete?
///
/// # The three operations
///
/// LAMMPS calls this object `Comm`; the name here says what it holds. The
/// operations keep LAMMPS's names, because the point is that a future domain decomposition changes
/// none of them:
///
/// | LAMMPS | here | what it does |
/// |---|---|---|
/// | `comm->borders()` | [`GhostSet::borders`] | decide *which* copies exist — a search, at rebuild |
/// | `comm->forward_comm()` | [`GhostSet::forward_comm`] | move them to follow their owners — every step |
/// | `comm->reverse_comm()` | [`GhostSet::reverse_comm`] | sum the forces on copies back onto owners |
///
/// In a single process a forward communication is a translation and a reverse
/// one is a scatter-add. Under MPI they become messages, an owner may live on
/// another rank, and nothing above this layer notices — which is the whole
/// reason to spell them this way now rather than invent a local vocabulary and
/// translate later.
#[derive(Debug)]
pub struct GhostHalo {
    set: GhostSet,
    bx: SimBox,
    cutoff: F,
    skin: F,
    /// Owned positions at the last halo build, for the displacement test.
    x_hold: Fnx3,
    rebuilds: usize,
    /// `[owned | ghost]` coordinates, refilled each step rather than rebuilt.
    all: Fnx3,
    /// Candidate pairs out to `cutoff + skin`, from the last halo rebuild.
    ///
    /// This is the halo's Verlet skin, and it is the whole reason the `skin`
    /// argument earns its place. Without it the halo's membership was stable
    /// across steps while its *pair table* was rediscovered from scratch every
    /// one — a spatial search, a hash of every pair and several allocations, to
    /// find a list that had not changed.
    edges: Vec<(u32, u32)>,
    /// The candidates inside the cutoff now, refilled from `edges`.
    table: Neighbors,
}

impl GhostHalo {
    /// Build the halo for `owned` at `cutoff`, with `skin` of margin.
    ///
    /// The halo reaches `cutoff + skin`, so it stays complete while no atom has
    /// moved more than `skin/2` since the build — the same bookkeeping a Verlet
    /// skin does, applied to copies instead of to a pair list.
    pub fn new(bx: SimBox, owned: Fnx3View<'_>, cutoff: F, skin: F) -> Result<Self, GhostError> {
        if cutoff.is_nan() || cutoff <= 0.0 {
            return Err(GhostError::InvalidCutoff(cutoff));
        }
        if skin.is_nan() || skin < 0.0 {
            return Err(GhostError::InvalidSkin(skin));
        }
        // Past half the smallest perpendicular width a pair has more than one
        // image inside the cutoff, and the halo keeps only the first it finds
        // (`GhostSet::pairs` dedupes on the owner pair). It would not
        // double-count; it would *under*-count, silently. `VerletSkin::new`
        // draws the same line for the minimum image — this is the same physics
        // and it belongs here too.
        let min_width = bx
            .nearest_plane_distance()
            .iter()
            .copied()
            .fold(F::INFINITY, F::min);
        let half_width = 0.5 * min_width;
        // The *search* radius, not the cutoff: the candidate list is built at
        // `cutoff + skin` and de-duplicated on owner pairs, so two images of
        // one pair inside that radius would leave whichever was found first
        // rather than whichever is closest when the table is filled.
        // `VerletSkin::new` bounds `cutoff + skin` for the same reason.
        if !bx.is_free() && cutoff + skin > half_width {
            return Err(GhostError::SearchBeyondHalfWidth {
                cutoff,
                skin,
                half_width,
            });
        }
        let set = GhostSet::borders(&bx, owned, cutoff + skin)?;
        let table = set.empty_table();
        let mut out = Self {
            all: set.combined(owned)?,
            set,
            bx,
            cutoff,
            skin,
            x_hold: owned.to_owned(),
            rebuilds: 0,
            edges: Vec::new(),
            table,
        };
        out.research()?;
        out.refill();
        Ok(out)
    }

    /// Redo the spatial search behind the pair table. Halo-rebuild cadence.
    fn research(&mut self) -> Result<(), GhostError> {
        self.edges = self
            .set
            .candidate_edges(self.all.view(), self.cutoff + self.skin)?;
        self.table = self.set.empty_table();
        Ok(())
    }

    /// Repair the candidate list after a fold. Fold cadence.
    ///
    /// A fold does not move an atom, it relabels it — so the *set* of owner
    /// pairs within reach is exactly what it was, and the search does not need
    /// to run again. What does change is which copy realises a pair: a folded
    /// atom's stored coordinate has jumped a lattice vector, so a pair that was
    /// direct is now a cell apart and its replacement is an image that was too
    /// far to be a candidate before.
    ///
    /// Re-resolving only the partners of the atoms that actually folded costs
    /// each of them its handful of copies. Researching costs a spatial build
    /// and a hash of every pair — and in a dense system something folds almost
    /// every step, so the difference is the difference between a cached table
    /// and no cache at all.
    fn reimage(
        &mut self,
        owned: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<(), GhostError> {
        let n_owned = self.set.n_owned();
        let folded: Vec<bool> = (0..n_owned)
            .map(|a| {
                wrap_shifts[[a, 0]] != 0 || wrap_shifts[[a, 1]] != 0 || wrap_shifts[[a, 2]] != 0
            })
            .collect();
        let set = &self.set;
        for e in &mut self.edges {
            let (i, j) = (e.0 as usize, e.1 as usize);
            let owner = if j < n_owned {
                j
            } else {
                set.owner()[j - n_owned] as usize
            };
            if !folded[i] && !folded[owner] {
                continue;
            }
            e.1 = set.closest_image(owned, i, owner)? as u32;
        }
        Ok(())
    }

    /// Recompute which candidates are inside the cutoff now. Per step.
    fn refill(&mut self) {
        self.set
            .fill_pairs(self.all.view(), &self.edges, self.cutoff, &mut self.table);
    }

    /// The cell the copies are generated from.
    pub fn simbox(&self) -> &SimBox {
        &self.bx
    }

    /// Move the copies to follow their owners, and rebuild the set if it has
    /// gone stale.
    ///
    /// `wrap_shifts` is the shift the caller's wrap just applied. It is
    /// required, not optional: a copy replaced relative to an owner that has
    /// folded moves a whole cell while the pair list still points at it.
    ///
    /// The set is rebuilt when an atom has drifted more than half the skin
    /// since the last one, which is the same completeness argument a Verlet
    /// skin makes. Displacements are minimum-image, so a fold reads as the step
    /// the atom took rather than as a jump of one cell.
    pub fn advance(
        &mut self,
        owned: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<(), GhostError> {
        self.set.forward_comm(&self.bx, owned, wrap_shifts)?;

        let mic = self.bx.mic();
        let half_skin_sq = (0.5 * self.skin) * (0.5 * self.skin);
        let mut max_d2 = 0.0;
        for i in 0..owned.nrows() {
            let d = mic.apply([
                owned[[i, 0]] - self.x_hold[[i, 0]],
                owned[[i, 1]] - self.x_hold[[i, 1]],
                owned[[i, 2]] - self.x_hold[[i, 2]],
            ]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > max_d2 {
                max_d2 = d2;
            }
        }
        let rebuilt = max_d2 > half_skin_sq;
        if rebuilt {
            self.set = GhostSet::borders(&self.bx, owned, self.cutoff + self.skin)?;
            self.x_hold = owned.to_owned();
            self.rebuilds += 1;
        }
        if rebuilt {
            // New copies, new indices: the candidate list names positions in a
            // list that no longer exists.
            self.all = self.set.combined(owned)?;
            self.research()?;
        } else if wrap_shifts.iter().any(|&m| m != 0) {
            self.reimage(owned, wrap_shifts)?;
            self.all = self.set.combined(owned)?;
        } else {
            self.all = self.set.combined(owned)?;
        }
        self.refill();
        Ok(())
    }

    /// The half-shell pair table over `[owned | ghost]`, centred on owned atoms.
    ///
    /// As of the last [`advance`](Self::advance) — this is a read, not a build.
    pub fn pairs(&self) -> &Neighbors {
        &self.table
    }

    /// The `[owned | ghost]` coordinates the pair table indexes.
    pub fn combined(&self) -> &Fnx3 {
        &self.all
    }

    /// How many times the halo has been rebuilt.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// Fold ghost forces onto their owners, and tally `Σ f ⊗ r` on the way.
    ///
    /// The two are one operation because the order between them is
    /// load-bearing and invisible when it is wrong.
    ///
    /// The sum runs over owned atoms **and copies**, each at its own position
    /// and carrying its own un-reduced force. That is what makes it the virial:
    ///
    /// ```text
    /// Σ_{a ∈ owned ∪ ghost} f_a ⊗ x_a
    ///   = Σ_pairs ( f_ij ⊗ x_i − f_ij ⊗ x_j' )
    ///   = Σ_pairs f_ij ⊗ r_ij
    /// ```
    ///
    /// — a sum over separations, independent of where the cell's origin is.
    ///
    /// Reduce first and sum over owned atoms only, and the copy's term becomes
    /// `−f_ij ⊗ x_j` instead of `−f_ij ⊗ (x_j + H·s)`. What is left over is
    /// `Σ_pairs f_ij ⊗ H·s`: non-zero for every pair that crosses a face, and
    /// dependent on the box origin, which no physical quantity may be. Nothing
    /// raises — the pressure is simply wrong, by an amount that grows with the
    /// number of crossing pairs.
    ///
    /// Reduction zeroes the copies' rows, so the mistake cannot be detected
    /// afterwards either: the same call would return a different number and
    /// look just as plausible. Hence one method, with the order inside it.
    ///
    /// `all` must be the `[owned | ghost]` coordinates the forces were computed
    /// at, i.e. what [`combined`](Self::combined) returned.
    pub fn reverse_comm_with_virial(
        &self,
        forces: &mut Array2<F>,
        all: Fnx3View<'_>,
    ) -> Result<Virial, GhostError> {
        let set = self.ghosts();
        let expected = set.n_owned() + set.len();
        if all.nrows() != expected {
            return Err(GhostError::Shape {
                expected,
                found: all.nrows(),
            });
        }
        if forces.nrows() != expected {
            return Err(GhostError::Shape {
                expected,
                found: forces.nrows(),
            });
        }
        let mut w = Virial::ZERO;
        for a in 0..expected {
            w.add_outer(
                [forces[[a, 0]], forces[[a, 1]], forces[[a, 2]]],
                [all[[a, 0]], all[[a, 1]], all[[a, 2]]],
            );
        }
        set.reverse_comm(forces)?;
        Ok(w)
    }

    /// The copies as they stand.
    pub fn ghosts(&self) -> &GhostSet {
        &self.set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    #[test]
    fn a_cutoff_past_half_the_cell_is_refused() {
        let l = 10.0_f64;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[1.0_f64, 1.0, 1.0], [2.0, 2.0, 2.0]];

        assert!(GhostHalo::new(bx.clone(), owned.view(), 4.9, 0.0).is_ok());
        let err = GhostHalo::new(bx, owned.view(), 5.1, 0.0)
            .expect_err("5.1 Å is past half of a 10 Å cell");
        let msg = format!("{err}");
        assert!(msg.contains("half the minimum perpendicular"), "{msg}");
    }
}
