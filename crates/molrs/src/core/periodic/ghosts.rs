//! Ghost atoms: periodic copies, with an owner and an integer shift.

use std::sync::atomic::AtomicU64;

use ndarray::{Array2, ArrayView2};

/// Source of [`GhostSet::generation`] values. Process-wide and monotonic, so
/// two sets are never confused even across boxes.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

use super::images::{GhostError, ImageRange};
use crate::core::SimBox;
use crate::core::{NeighborColumns, NeighborList, Neighbors, QueryMode};
use crate::op::{F, Fnx3, Fnx3View, I};

/// The periodic copies of one owned point set.
///
/// A ghost is `r_g = r_owner + H·s_g`: a copy of owned atom `owner(g)`,
/// translated by the integer lattice vector `s_g`. It is a **geometry object**
/// and nothing else — no velocity, no mass, no thermostat state, no image flag,
/// no identity that survives a rebuild. Only owned atoms are degrees of
/// freedom.
///
/// # `shift` is not an image flag
///
/// They are integer triples that both count cells, and they are different
/// quantities with different lifetimes:
///
/// * an **image flag** is the history of an owned atom — how many cells it has
///   crossed since the run began, monotone in nothing, owned by the MD state;
/// * a **shift** is which copy the *current* local geometry needs, owned by
///   this type and discarded at the next rebuild.
///
/// They move together at exactly one moment — a wrap — and
/// [`forward_comm`](Self::forward_comm) is where that happens.
///
/// # What this type does not hold
///
/// It stores **only ghost columns**. The owned coordinates stay where they
/// live, in the MD state, and arrive by view on every call. A ghost layer that
/// kept its own copy of them would be a second float array advancing alongside
/// the canonical one, which is precisely the arrangement the wrapped+image
/// design exists to avoid: two copies of one quantity drift apart, and the one
/// that drifts is whichever the reader did not check.
#[derive(Debug, Clone)]
pub struct GhostSet {
    /// Ghost positions `(n_ghost, 3)` in Å, rebuilt by `refresh`.
    positions: Fnx3,
    /// Owned index each ghost copies.
    owner: Vec<u32>,
    /// Integer lattice translation each ghost carries.
    shift: Vec<[i32; 3]>,
    /// Owned-atom count the topology was built against.
    n_owned: usize,
    /// Reach the halo was built for (Å).
    reach: F,
    /// Which membership this is. Bumped by [`borders`](Self::borders) and by
    /// nothing else, so an index into this set can be checked against the set
    /// it was taken from.
    generation: u64,
    /// CSR over `owner`: atom `a`'s copies are
    /// `by_owner[copies_start[a]..copies_start[a + 1]]`.
    ///
    /// "Which copies does this atom have" is asked once per bonded partner per
    /// resolution and once per folded edge; answering it by scanning the whole
    /// copy list makes those O(n_ghost) each, which is O(N^{5/3}) over a step
    /// and is what made the ghost régime unusable above a few thousand atoms.
    /// A counting sort at build time turns every one of them into a handful.
    copies_start: Vec<u32>,
    by_owner: Vec<u32>,
}

impl GhostSet {
    /// Build the halo: decide which ghosts exist, and place them.
    ///
    /// A copy is emitted only when it comes within `reach` of the primary cell,
    /// so an interior atom produces none and an atom near a corner produces the
    /// face, edge and corner copies it needs. The test is conservative — it may
    /// keep a copy slightly beyond `reach` — because over-inclusion costs a
    /// distance check downstream while under-inclusion loses a pair silently.
    ///
    /// `owned` is expected to be wrapped (inside the cell). Nothing breaks if
    /// it is not; the halo is simply built around wherever the atoms are.
    pub fn borders(bx: &SimBox, owned: Fnx3View<'_>, reach: F) -> Result<Self, GhostError> {
        let range = ImageRange::new(bx, reach)?;
        let d = bx.nearest_plane_distance();
        let frac = bx.to_frac(owned);
        let n_owned = owned.nrows();

        let mut owner: Vec<u32> = Vec::new();
        let mut shift: Vec<[i32; 3]> = Vec::new();
        for &s in range.shifts() {
            if s == [0, 0, 0] {
                continue; // the owned atoms themselves are not ghosts
            }
            for i in 0..n_owned {
                // Distance from the translated copy to the cell, measured
                // per-axis in plane-spacing units. `max(0, -g, g - 1) * d` is
                // how far outside the slab the copy sits on that axis; the true
                // point-to-cell distance is at least the largest of the three.
                let mut out_of_cell = 0.0_f64;
                for k in 0..3 {
                    let g = frac[[i, k]] + s[k] as F;
                    let past = (-g).max(g - 1.0).max(0.0);
                    let dist = past * d[k];
                    if dist > out_of_cell {
                        out_of_cell = dist;
                    }
                }
                if out_of_cell <= reach {
                    owner.push(i as u32);
                    shift.push(s);
                }
            }
        }

        let mut set = Self {
            positions: Fnx3::zeros((owner.len(), 3)),
            owner,
            shift,
            n_owned,
            reach,
            generation: NEXT_GENERATION.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            copies_start: Vec::new(),
            by_owner: Vec::new(),
        };
        set.index_by_owner();
        set.place(bx, owned)?;
        Ok(set)
    }

    /// Build the owner → copies index by counting sort. Once per membership.
    fn index_by_owner(&mut self) {
        let n = self.n_owned;
        self.copies_start = vec![0; n + 1];
        for &o in &self.owner {
            self.copies_start[o as usize + 1] += 1;
        }
        for a in 0..n {
            self.copies_start[a + 1] += self.copies_start[a];
        }
        let mut cursor = self.copies_start.clone();
        self.by_owner = vec![0; self.owner.len()];
        for (g, &o) in self.owner.iter().enumerate() {
            let slot = &mut cursor[o as usize];
            self.by_owner[*slot as usize] = g as u32;
            *slot += 1;
        }
    }

    /// The rows of this set that copy `atom`.
    pub fn copies_of(&self, atom: usize) -> &[u32] {
        let lo = self.copies_start[atom] as usize;
        let hi = self.copies_start[atom + 1] as usize;
        &self.by_owner[lo..hi]
    }

    /// Move the ghosts to follow their owners, reconciling a wrap in the same
    /// step.
    ///
    /// `wrap_shifts` is the per-owned-atom integer shift that the wrap just
    /// applied (the second half of [`SimBox::wrap`]), in the schema's
    /// image integer type ([`I`], i32) — the same type as a ghost's shift, so
    /// it adds in without a cast. It is not optional
    /// bookkeeping: when owner `j` is folded by `m`, its stored coordinate
    /// moves by `−H·m`, and a ghost replaced as `r_owner + H·s_g` would move by
    /// `−H·m` with it — a jump of at least one cell, while the neighbour list
    /// still points at it. Requiring `s_g += m` in the same call is what makes
    /// "refreshed but not reconciled" a state no caller can produce.
    ///
    /// Derivation of the sign, since it is easy to get backwards: holding the
    /// ghost's absolute position fixed across the wrap,
    /// `r_j − H·m + H·s_g^new = r_j + H·s_g^old`, hence `s_g^new = s_g^old + m`
    /// — the same sign the owner's image flag accumulates with.
    ///
    /// Topology is untouched: which ghosts exist was decided by
    /// [`borders`](Self::borders) and stays fixed until the next one. That split is
    /// the point — deciding the halo is a search, following it is a translation.
    pub fn forward_comm(
        &mut self,
        bx: &SimBox,
        owned: Fnx3View<'_>,
        wrap_shifts: ArrayView2<'_, I>,
    ) -> Result<(), GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        if wrap_shifts.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: wrap_shifts.nrows(),
            });
        }
        for (g, &o) in self.owner.iter().enumerate() {
            let o = o as usize;
            for k in 0..3 {
                self.shift[g][k] += wrap_shifts[[o, k]];
            }
        }
        self.place(bx, owned)
    }

    /// `r_g = r_owner + H·s_g` for every ghost.
    fn place(&mut self, bx: &SimBox, owned: Fnx3View<'_>) -> Result<(), GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        let a = [bx.lattice(0), bx.lattice(1), bx.lattice(2)];
        for g in 0..self.owner.len() {
            let o = self.owner[g] as usize;
            let s = self.shift[g];
            for k in 0..3 {
                self.positions[[g, k]] =
                    owned[[o, k]] + s[0] as F * a[0][k] + s[1] as F * a[1][k] + s[2] as F * a[2][k];
            }
        }
        Ok(())
    }

    /// Ghost positions `(n_ghost, 3)` in Å.
    pub fn positions(&self) -> Fnx3View<'_> {
        self.positions.view()
    }

    /// The owned index each ghost copies.
    pub fn owner(&self) -> &[u32] {
        &self.owner
    }

    /// The integer lattice translation each ghost carries.
    pub fn shift(&self) -> &[[i32; 3]] {
        &self.shift
    }

    /// How many ghosts exist.
    pub fn len(&self) -> usize {
        self.owner.len()
    }

    /// Whether the halo is empty (an interior-only configuration, or a free box).
    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    /// Owned-atom count the topology was built against.
    pub fn n_owned(&self) -> usize {
        self.n_owned
    }

    /// The reach (Å) the halo was built for.
    pub fn reach(&self) -> F {
        self.reach
    }

    /// Which membership this is.
    ///
    /// Every [`borders`](Self::borders) call produces a new one. An index into
    /// the combined view names a *position in this list*, so an index taken
    /// from one generation means something else in the next — which is why
    /// anything that stores such an index (a re-resolved bonded topology, say)
    /// has to record the generation beside it and check.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The combined `[owned | ghost]` view a force evaluation consumes.
    ///
    /// Rows `0..n_owned` are the owned atoms exactly as supplied; the rest are
    /// the ghosts. Only the first `n_owned` are degrees of freedom, which is
    /// why a force array over this layout has to be reduced before it means
    /// anything physical.
    pub fn combined(&self, owned: Fnx3View<'_>) -> Result<Fnx3, GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        let mut out = Array2::zeros((self.n_owned + self.owner.len(), 3));
        for i in 0..self.n_owned {
            for k in 0..3 {
                out[[i, k]] = owned[[i, k]];
            }
        }
        for g in 0..self.owner.len() {
            for k in 0..3 {
                out[[self.n_owned + g, k]] = self.positions[[g, k]];
            }
        }
        Ok(out)
    }

    /// The half-shell pair table over `[owned | ghost]`, centred on owned atoms.
    ///
    /// `i` indexes owned atoms, `j` indexes the combined view, and the
    /// displacement is the plain difference `r_j − r_i` — **no minimum image**.
    /// That is the point of the whole layer: the copies already carry the
    /// periodicity, so the geometry handed downstream is ordinary and local,
    /// and a potential that reads it never learns a boundary exists.
    ///
    /// The search itself runs against a **free** box over the combined points.
    /// Asking a periodic neighbour search to look at a set that already
    /// contains the images would find each pair through both routes.
    ///
    /// # Counting each pair once
    ///
    /// A pair of owned atoms `(i, o)` separated across a face is discovered
    /// twice — once as `i` with a copy of `o`, once as `o` with a copy of `i` —
    /// and the naive half-shell rule `i < j` does not break the tie, because
    /// every copy sorts above every owned atom and so *both* orderings satisfy
    /// it. Keeping both is a factor of two in energy, force and virial.
    ///
    /// The tie is broken by **identity**: a pair is recorded the first time it
    /// is seen and skipped afterwards, keyed on the two owner indices.
    ///
    /// A parity rule on those indices — LAMMPS's, keep when `i < o` and `i + o`
    /// is even — is cheaper and is what this originally used, but it is only
    /// sound when both discoveries are guaranteed to happen. LAMMPS can assume
    /// that: its ghost region is a slab, symmetric by construction. A *culled*
    /// halo is not. A copy is kept when it comes within reach of the cell, and
    /// after the owners have moved that test can pass for `o`'s copy and fail
    /// for `i`'s. The parity rule then rejects the one discovery that exists
    /// and the pair vanishes — silently, and only for atoms near a corner, and
    /// only after enough drift. Recording identity costs a hash per pair and
    /// cannot lose one.
    ///
    /// A copy of `i` itself is a genuine interaction — atom with own image —
    /// and is found twice too, as `+s` and `−s`. It is kept once, on the
    /// lexicographically positive translation.
    ///
    /// # Precondition
    ///
    /// `cutoff` must not exceed the halo's [`reach`](Self::reach), or a pair
    /// inside the cutoff may have no copy to be found through. It must also
    /// stay within half the smallest plane spacing for the *direct* and
    /// *imaged* separations of a pair not to both fall inside it, which would
    /// double-count in a way no tie-break can see.
    pub fn pairs(&self, owned: Fnx3View<'_>, cutoff: F) -> Result<Neighbors, GhostError> {
        let all = self.combined(owned)?;
        let edges = self.candidate_edges(all.view(), cutoff)?;
        let mut table = self.empty_table();
        self.fill_pairs(all.view(), &edges, cutoff, &mut table);
        Ok(table)
    }

    /// An empty pair table shaped for this halo, ready to be filled.
    pub fn empty_table(&self) -> Neighbors {
        Neighbors::empty(
            QueryMode::CrossQuery {
                n_query_points: self.n_owned,
                n_points: self.n_owned + self.len(),
            },
            NeighborColumns::FULL,
        )
    }

    /// Candidate pairs out to `reach`, as indices into the combined view.
    ///
    /// This is the search and the de-duplication — everything that depends on
    /// *which copies exist* rather than on where the atoms are right now. A
    /// caller that keeps the result across steps pays the search once per halo
    /// rebuild and only recomputes displacements in between, which is exactly
    /// what a Verlet skin does for the minimum image. Recomputing it every step
    /// is a spatial search, a hash of every pair, and several allocations, to
    /// rediscover a list that did not change.
    ///
    /// # Precondition
    ///
    /// `reach` must not exceed the halo's [`reach`](Self::reach), **or half the
    /// smallest plane spacing**. Past the latter a pair of owners can have two
    /// images inside `reach`, and the de-duplication below — which keys on the
    /// owners — would keep whichever was found first rather than whichever is
    /// closest when the table is next filled.
    pub fn candidate_edges(
        &self,
        all: Fnx3View<'_>,
        reach: F,
    ) -> Result<Vec<(u32, u32)>, GhostError> {
        if reach > self.reach {
            return Err(GhostError::InvalidReach(reach));
        }
        let n_owned = self.n_owned;
        let n_all = all.nrows();
        let mut edges = Vec::new();
        if n_all == 0 {
            return Ok(edges);
        }

        // A free box: the copies are the periodicity, so the search must not
        // add any of its own.
        let free = SimBox::bounding(all, [reach + 1.0; 3], [false; 3])
            .expect("free box over finite points");
        let mut nl = NeighborList::new(reach);
        nl.build(all, &free);

        // Keyed on owner indices. With `reach` inside half the smallest plane
        // spacing a pair has at most one image in range, so the pair of owners
        // identifies it completely.
        //
        // This also absorbs a case that looks like it needs its own guard and
        // does not: reconciling a fold can leave a copy with translation
        // `(0,0,0)`, sitting exactly on the atom it copies. Keyed on owners, it
        // collides with the direct pair and is dropped.
        let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
        nl.for_each_pair(|p| {
            // The centre must be an owned atom. Copies sort above owned atoms
            // and the backend yields `i < j`, so `i >= n_owned` means both ends
            // are copies — a shadow of a pair that is counted elsewhere.
            let (i, j) = (p.i as usize, p.j as usize);
            if i >= n_owned {
                return;
            }
            if j >= n_owned {
                let g = j - n_owned;
                let o = self.owner[g] as usize;
                if o == i {
                    // Own image: keep the `+s` half of the `±s` pair.
                    if self.shift[g] <= [0, 0, 0] {
                        return;
                    }
                } else if !seen.insert((i.min(o), i.max(o))) {
                    return;
                }
            } else if !seen.insert((i.min(j), i.max(j))) {
                return;
            }
            edges.push((i as u32, j as u32));
        });
        Ok(edges)
    }

    /// Fill `out` with the candidate pairs that are inside `cutoff` now.
    ///
    /// The displacements are plain differences: a copy already carries its
    /// translation, so there is no minimum image to apply and nothing here
    /// knows that periodic boundaries exist.
    pub fn fill_pairs(
        &self,
        all: Fnx3View<'_>,
        edges: &[(u32, u32)],
        cutoff: F,
        out: &mut Neighbors,
    ) {
        out.set_mode(QueryMode::CrossQuery {
            n_query_points: self.n_owned,
            n_points: self.n_owned + self.len(),
        });
        out.clear();
        let cutoff2 = cutoff * cutoff;
        for &(i, j) in edges {
            let (ia, ja) = (i as usize, j as usize);
            let dr = [
                all[[ja, 0]] - all[[ia, 0]],
                all[[ja, 1]] - all[[ia, 1]],
                all[[ja, 2]] - all[[ia, 2]],
            ];
            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            if d2 <= cutoff2 {
                out.push(i, j, d2, dr);
            }
        }
    }

    /// The index, in the combined view, of the copy of `partner` that lies
    /// closest to `centre`.
    ///
    /// Returns `partner` itself when the direct separation is already the
    /// shortest, otherwise `n_owned + g` for the copy that is.
    ///
    /// # Why bonded terms need this
    ///
    /// A non-bonded kernel is handed a pair table and reads the displacement
    /// out of it. A bonded kernel is not: it holds the topology — *these* two
    /// atoms are bonded — and takes the plain difference of their coordinates.
    /// For a bond that straddles a face those coordinates are a cell apart, so
    /// the kernel sees a bond stretched by the width of the box and answers
    /// with a force to match. Nothing raises; the number is simply wrong.
    ///
    /// Remapping the partner to its closest copy makes the plain difference the
    /// right one, which is what lets the kernels stay ignorant of periodicity.
    /// It is the same resolution LAMMPS reaches with `Domain::closest_image`,
    /// and it is chosen once when the halo is built rather than per evaluation:
    /// an image that changed under the atoms would make the bond a
    /// discontinuous function of their positions.
    ///
    /// # Anchoring, not per-edge folding
    ///
    /// Every partner in an angle or a dihedral must be resolved against the
    /// **same centre**, not each against the previous one. Folding each edge
    /// independently can pick images that are individually closest and jointly
    /// inconsistent — the three atoms of an angle ending up in three different
    /// cells — and the angle is then measured on a shape that does not exist.
    /// Resolving against one anchor cannot do that: the atoms are placed
    /// relative to a single point.
    pub fn closest_image(
        &self,
        owned: Fnx3View<'_>,
        centre: usize,
        partner: usize,
    ) -> Result<usize, GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        if centre >= self.n_owned || partner >= self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: centre.max(partner) + 1,
            });
        }
        let c = [owned[[centre, 0]], owned[[centre, 1]], owned[[centre, 2]]];
        let d2 = |p: [F; 3]| {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        };

        let mut best = partner;
        let mut best_d2 = d2([
            owned[[partner, 0]],
            owned[[partner, 1]],
            owned[[partner, 2]],
        ]);
        for &g in self.copies_of(partner) {
            let g = g as usize;
            let cand = d2([
                self.positions[[g, 0]],
                self.positions[[g, 1]],
                self.positions[[g, 2]],
            ]);
            if cand < best_d2 {
                best_d2 = cand;
                best = self.n_owned + g;
            }
        }
        Ok(best)
    }

    /// Fold ghost forces onto the atoms that own them.
    ///
    /// `forces` is over the `[owned | ghost]` layout; the ghost rows are added
    /// into their owners and then cleared, so what remains in the first
    /// `n_owned` rows is the physical force. This is the single-process form of
    /// LAMMPS reverse communication, and it is the step that makes ghosts free:
    /// a copy contributes to the force on the atom it copies, never to a degree
    /// of freedom of its own.
    pub fn reverse_comm(&self, forces: &mut Array2<F>) -> Result<(), GhostError> {
        let expected = self.n_owned + self.owner.len();
        if forces.nrows() != expected {
            return Err(GhostError::Shape {
                expected,
                found: forces.nrows(),
            });
        }
        for g in 0..self.owner.len() {
            let row = self.n_owned + g;
            let o = self.owner[g] as usize;
            for k in 0..3 {
                forces[[o, k]] += forces[[row, k]];
                forces[[row, k]] = 0.0;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    fn cube(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap()
    }

    /// An atom in the middle of the cell needs no copies; one in a corner needs
    /// seven — three faces, three edges, one corner.
    #[test]
    fn a_corner_atom_gets_face_edge_and_corner_copies() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();

        assert_eq!(set.len(), 7, "3 faces + 3 edges + 1 corner");
        let mut got: Vec<[i32; 3]> = set.shift().to_vec();
        got.sort_unstable();
        let mut want = vec![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        want.sort_unstable();
        assert_eq!(got, want);

        // Each copy really is the owner plus that lattice vector.
        for g in 0..set.len() {
            let s = set.shift()[g];
            for k in 0..3 {
                let want = owned[[0, k]] + s[k] as F * 10.0;
                assert!((set.positions()[[g, k]] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn an_interior_atom_gets_none() {
        let bx = cube(10.0);
        let owned = array![[5.0_f64, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        assert!(set.is_empty());
        // ...while the range still enumerates the full 27 translations: which
        // copies *could* exist and which are *kept* are separate questions.
        assert_eq!(ImageRange::new(&bx, 2.0).unwrap().shifts().len(), 27);
    }

    #[test]
    fn a_non_periodic_axis_contributes_no_copies() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0_f64, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        let owned = array![[0.5_f64, 0.5, 0.5]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        assert!(!set.is_empty());
        for s in set.shift() {
            assert_eq!(s[1], 0, "the free axis must not be tiled");
        }
    }

    /// The reconciliation that makes wrapped coordinates safe: when an owner is
    /// folded, its ghosts must not move in absolute terms.
    ///
    /// Without `s_g += m` the ghost is replaced relative to the *new* owner
    /// position and jumps by a whole cell, while the neighbour list still
    /// points at it — a displacement of 10 Å here, against a skin of well under
    /// 1 Å.
    #[test]
    fn folding_an_owner_leaves_its_ghosts_where_they_were() {
        let bx = cube(10.0);
        // Atom 0 sits just inside the +x face, so it has a copy at x = -0.3.
        let owned = array![[9.7_f64, 5.0, 5.0], [5.0, 5.0, 5.0]];
        let mut set = GhostSet::borders(&bx, owned.view(), 1.0).unwrap();

        let g = set
            .shift()
            .iter()
            .position(|s| *s == [-1, 0, 0])
            .expect("a copy across the +x face");
        let before = set.positions()[[g, 0]];
        assert!((before - (-0.3)).abs() < 1e-12, "copy at {before}");

        // One step of +0.7 Å carries the owner out through the face; the wrap
        // brings it back to 0.4 and reports the shift it applied.
        let stepped = array![[10.4_f64, 5.0, 5.0], [5.0, 5.0, 5.0]];
        let (wrapped, m) = bx.wrap(stepped.view());
        assert!((wrapped[[0, 0]] - 0.4).abs() < 1e-12);
        assert_eq!(m[[0, 0]], 1);

        set.forward_comm(&bx, wrapped.view(), m.view()).unwrap();

        // The shift absorbed the fold...
        assert_eq!(set.shift()[g], [0, 0, 0]);
        // ...so the copy moved by the atom's actual step, not by a cell.
        let after = set.positions()[[g, 0]];
        assert!(
            (after - 0.4).abs() < 1e-12,
            "copy is at {after}; it moved {} Å, and the step was 0.7",
            after - before
        );
        // The other atom's copies, if any, are untouched by atom 0's fold.
        for h in 0..set.len() {
            if set.owner()[h] == 1 {
                let s = set.shift()[h];
                for k in 0..3 {
                    let want = wrapped[[1, k]] + s[k] as F * 10.0;
                    assert!((set.positions()[[h, k]] - want).abs() < 1e-12);
                }
            }
        }
    }

    /// Between rebuilds the halo's membership is fixed: `refresh` translates,
    /// it does not re-decide.
    #[test]
    fn refresh_keeps_the_topology_and_only_moves_things() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5]];
        let mut set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let owners = set.owner().to_vec();
        let n = set.len();

        // Drift well inside the cell: nothing wraps, and nothing leaves.
        let moved = array![[4.9_f64, 4.9, 4.9]];
        let zero = Array2::<I>::zeros((1, 3));
        set.forward_comm(&bx, moved.view(), zero.view()).unwrap();

        assert_eq!(set.len(), n, "membership is decided by build, not refresh");
        assert_eq!(set.owner(), owners.as_slice());
        for g in 0..set.len() {
            let s = set.shift()[g];
            for k in 0..3 {
                let want = moved[[0, k]] + s[k] as F * 10.0;
                assert!((set.positions()[[g, k]] - want).abs() < 1e-12);
            }
        }
    }

    /// Ghost forces belong to the atoms they copy, and nothing else survives.
    #[test]
    fn forces_on_copies_are_folded_onto_their_owners() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5], [9.5, 9.5, 9.5]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        assert!(set.len() >= 2);

        let mut f = Array2::<F>::zeros((set.n_owned() + set.len(), 3));
        f[[0, 0]] = 1.0; // a direct force on atom 0
        for g in 0..set.len() {
            f[[set.n_owned() + g, 0]] = 0.5; // and one on every copy
        }
        let owned_copies = set.owner().iter().filter(|&&o| o == 0).count() as F;

        set.reverse_comm(&mut f).unwrap();

        assert!((f[[0, 0]] - (1.0 + 0.5 * owned_copies)).abs() < 1e-12);
        for g in 0..set.len() {
            assert_eq!(f[[set.n_owned() + g, 0]], 0.0, "copies keep no force");
        }
    }

    /// The combined view is owned-then-ghost, and the owned half is the input
    /// unchanged — a caller indexing below `n_owned` is indexing real atoms.
    #[test]
    fn the_combined_view_is_owned_then_ghost() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5], [5.0, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let all = set.combined(owned.view()).unwrap();

        assert_eq!(all.nrows(), set.n_owned() + set.len());
        for i in 0..set.n_owned() {
            for k in 0..3 {
                assert_eq!(all[[i, k]], owned[[i, k]]);
            }
        }
        for g in 0..set.len() {
            for k in 0..3 {
                assert_eq!(all[[set.n_owned() + g, k]], set.positions()[[g, k]]);
            }
        }
    }

    /// The pair table is the minimum-image table, found a different way.
    ///
    /// This is what the whole layer has to earn: for every owned pair inside
    /// the cutoff, the ghost route must report the same separation the
    /// minimum-image convention reports, exactly once. If it reports a pair
    /// twice the energy doubles; if it misses one the force is wrong and
    /// nothing says so.
    #[test]
    fn the_ghost_table_is_the_minimum_image_table() {
        let l = 10.0_f64;
        let bx = cube(l);
        let cutoff = 3.0;

        // Atoms at and near every face, so most pairs are found through a copy.
        let owned = array![
            [0.4_f64, 0.5, 0.6],
            [9.6, 0.5, 0.6],
            [0.5, 9.7, 5.0],
            [5.0, 5.0, 0.3],
            [5.0, 5.0, 9.8],
            [2.0, 2.0, 2.0],
            [9.9, 9.9, 9.9],
            [0.1, 0.1, 0.1],
        ];

        let set = GhostSet::borders(&bx, owned.view(), cutoff).unwrap();
        let table = set.pairs(owned.view(), cutoff).unwrap();

        // What the ghost route found, as owned-index pairs with a separation.
        let mut got: Vec<(usize, usize, F)> = Vec::new();
        for row in 0..table.query_point_indices().len() {
            let i = table.query_point_indices()[row] as usize;
            let j = table.point_indices()[row] as usize;
            let o = if j < set.n_owned() {
                j
            } else {
                set.owner()[j - set.n_owned()] as usize
            };
            got.push((i.min(o), i.max(o), table.dist_sq().unwrap()[row]));
        }
        got.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        // What the minimum-image convention says, computed independently.
        let mut want: Vec<(usize, usize, F)> = Vec::new();
        for i in 0..owned.nrows() {
            for j in (i + 1)..owned.nrows() {
                let pi = [owned[[i, 0]], owned[[i, 1]], owned[[i, 2]]];
                let pj = [owned[[j, 0]], owned[[j, 1]], owned[[j, 2]]];
                let dr = bx.mic().apply(crate::op::vec3::sub(pj, pi));
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff * cutoff {
                    want.push((i, j, d2));
                }
            }
        }
        want.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(
            got.len(),
            want.len(),
            "pair counts differ: ghost {got:?} vs mic {want:?}"
        );
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!((g.0, g.1), (w.0, w.1), "pair identity");
            assert!(
                (g.2 - w.2).abs() < 1e-12,
                "pair ({},{}) separation: ghost {} vs mic {}",
                g.0,
                g.1,
                g.2,
                w.2
            );
        }
    }

    /// The displacement handed downstream is the plain difference of the two
    /// coordinates in the combined view — no minimum image anywhere in it.
    /// That is the property a potential relies on when it reads positions
    /// rather than edge vectors.
    #[test]
    fn displacements_are_plain_differences() {
        let bx = cube(10.0);
        let owned = array![[0.4_f64, 5.0, 5.0], [9.6, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let all = set.combined(owned.view()).unwrap();
        let table = set.pairs(owned.view(), 2.0).unwrap();

        assert_eq!(
            table.query_point_indices().len(),
            1,
            "one pair, across the face"
        );
        let row = 0;
        let i = table.query_point_indices()[row] as usize;
        let j = table.point_indices()[row] as usize;
        let disp = table.disp().unwrap();
        for k in 0..3 {
            let plain = all[[j, k]] - all[[i, k]];
            assert!(
                (disp[[row, k]] - plain).abs() < 1e-12,
                "component {k}: table {} vs plain difference {plain}",
                disp[[row, k]]
            );
        }
        // And it is the short way round: 0.8 Å, not 9.2.
        assert!((table.dist_sq().unwrap()[row] - 0.64).abs() < 1e-12);
    }

    /// A pair straddling a face is reported once, not twice. The tie-break is
    /// the only thing standing between this design and a factor of two.
    #[test]
    fn a_pair_across_a_face_is_reported_once() {
        let bx = cube(10.0);
        // Every pairing here is short only through a copy.
        let owned = array![
            [0.2_f64, 0.2, 0.2],
            [9.8, 9.8, 9.8],
            [0.2, 9.8, 0.2],
            [9.8, 0.2, 9.8],
        ];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let table = set.pairs(owned.view(), 2.0).unwrap();

        let mut seen: Vec<(usize, usize)> = Vec::new();
        for row in 0..table.query_point_indices().len() {
            let i = table.query_point_indices()[row] as usize;
            let j = table.point_indices()[row] as usize;
            let o = if j < set.n_owned() {
                j
            } else {
                set.owner()[j - set.n_owned()] as usize
            };
            seen.push((i.min(o), i.max(o)));
        }
        let mut uniq = seen.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            seen.len(),
            "a pair was counted more than once: {seen:?}"
        );
    }

    /// A reach the cell cannot express is refused with the numbers in hand,
    /// never quietly truncated.
    #[test]
    fn an_unrepresentable_reach_is_refused() {
        let bx = cube(1.0);
        let err = ImageRange::new(&bx, 1_000.0).unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, GhostError::TooManyImages { .. }), "{msg}");
        assert!(msg.contains("nothing is truncated"), "{msg}");

        assert!(matches!(
            ImageRange::new(&bx, -1.0).unwrap_err(),
            GhostError::InvalidReach(_)
        ));
        assert!(matches!(
            ImageRange::new(&bx, F::NAN).unwrap_err(),
            GhostError::InvalidReach(_)
        ));
    }

    /// A tilted cell sizes its range from the plane spacings. The cell here has
    /// three 10 Å edges but spacings of 8, 8 and 10, so edge-length sizing
    /// would enumerate 27 translations where 75 are needed.
    #[test]
    fn a_tilted_cell_sizes_from_plane_spacings() {
        let h = [[10.0_f64, 6.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]];
        let bx = SimBox::new(Cell::Matrix(h), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let d = bx.nearest_plane_distance();
        assert!((d[0] - 8.0).abs() < 1e-12 && (d[1] - 8.0).abs() < 1e-12);
        assert!((bx.lengths()[1] - 10.0).abs() < 1e-12);

        let range = ImageRange::new(&bx, 9.0).unwrap();
        assert_eq!(range.per_axis(), [2, 2, 1]);
        assert_eq!(range.shifts().len(), 75);
    }
}
