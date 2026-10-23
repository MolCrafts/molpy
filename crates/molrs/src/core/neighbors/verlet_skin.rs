//! A [`NeighborList`] with Verlet skin: [`VerletSkin`].

use std::fmt;

use ndarray::{Array2, ArrayView2};

use crate::core::SimBox;
use crate::op::{F, Fnx3};

use super::{NeighborColumns, NeighborList, Neighbors, QueryMode};

/// Failures from Verlet-skin construction or the unwrapped-position guard.
#[derive(Debug)]
pub enum SkinError {
    /// A constructor or update argument is out of domain.
    Invalid(String),
    /// Completeness / unwrapped-position guard failed.
    Guard(String),
}

impl fmt::Display for SkinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Guard(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for SkinError {}

/// Rebuild policy for [`VerletSkin`].
///
/// The search backend is **not** selected here — pass a ready [`NeighborList`]
/// (`NeighborList::new` or [`NeighborList::brute_force`]) into
/// [`VerletSkin::new`].
#[derive(Clone, Copy, Debug)]
pub struct NeighborPolicy {
    /// Verlet skin `s` in Å. The list is built at `cutoff + s`.
    pub skin: F,
    /// Attempt a rebuild only when `ago` is a multiple of this (steps).
    pub every: usize,
    /// Attempt no rebuild until at least this many steps since the last build.
    /// Must be a multiple of `every`. `0` is always legal.
    pub delay: usize,
    /// Rebuild only when the max displacement exceeds `skin/2`.
    /// `false` rebuilds on cadence alone and skips the unwrapped-position guard.
    pub check: bool,
}

impl Default for NeighborPolicy {
    fn default() -> Self {
        Self {
            skin: 0.0,
            every: 1,
            delay: 0,
            check: true,
        }
    }
}

/// One half-shell neighbour edge (backend `i < j` convention).
#[derive(Clone, Copy, Debug)]
pub struct SkinPair {
    /// Source atom index.
    pub i: u32,
    /// Target atom index.
    pub j: u32,
}

/// Minimum-image Verlet-skin neighbour list over a core [`NeighborList`].
///
/// Two entry points: [`rebuild`](Self::rebuild) forces a build,
/// [`update`](Self::update) applies the `every` / `delay` / `check` policy.
///
/// This **is** a neighbour list: same job as [`NeighborList`] (find pairs under
/// a cutoff), plus the LAMMPS `neigh_modify every/delay/check` gate and a skin
/// margin so the index can stay valid across several steps. Pair *search* is
/// delegated to the wrapped [`NeighborList`] (LinkCell, BruteForce, …).
///
/// Constructed as `VerletSkin::new(NeighborList::new(cutoff + skin)?, …)`.
///
/// The list is built at `r_build = cutoff + skin` and stays complete out to
/// `cutoff` while no atom has moved more than `skin/2` since the last build
/// (strict `>`).
///
/// **Positions may be wrapped.** Displacements are measured as the
/// *minimum-image* difference from the held coordinates, which is exact here
/// rather than merely convenient: the constructor already refuses a
/// `r_build` above half the smallest perpendicular cell width, and the rebuild
/// threshold is `skin/2 < r_build/2`, so a displacement the skin tolerates is
/// always far inside the range where the minimum image is unambiguous. An atom
/// that crosses a face therefore reads as the small step it actually took, not
/// as a jump of one cell.
///
/// A displacement that *does* reach half the smallest perpendicular width still
/// raises, and means something else: the minimum image cannot tell
/// which copy it came from, so the number is not a displacement at all. That is
/// a blow-up or a cell change, not a wrap.
///
/// Force / analysis callers must use [`for_each_pair_at`](VerletSkin::for_each_pair_at)
/// (or the stored `(i, j)` edges plus a fresh MIC). Never stream the inner
/// [`NeighborList::for_each_pair`] for live forces — that reads geometry
/// frozen at the last rebuild.
///
/// Edge *order* is not part of the contract. The *set* of half-shell `(i, j)`
/// pairs (`i < j` from the backend) plus the live count is.
///
/// # References
///
/// LAMMPS `neigh_modify` / `Neighbor::decide` / `Neighbor::check_distance` /
/// `Neighbor::init`. Nordlund lecture notes for the two-atom skin criterion.
/// Allen & Tildesley for cell lists.
pub struct VerletSkin {
    cutoff: F,
    skin: F,
    every: usize,
    delay: usize,
    check: bool,
    r_build: F,
    simbox: SimBox,
    search: NeighborList,
    edges: Vec<SkinPair>,
    n_edges: usize,
    n_rebuilds: usize,
    ago: usize,
    ndanger: usize,
    half_skin_sq: F,
    /// Half the minimum perpendicular cell width, squared: beyond it the
    /// minimum image is ambiguous and a displacement cannot be read.
    ambiguity_guard_sq: F,
    danger_ago: usize,
    x_hold: Fnx3,
    pairs_buf: Neighbors,
    /// Per-edge squared distance and displacement, reused across steps.
    ///
    /// The displacement pass is the expensive half of building a table — a
    /// random gather into the coordinates for every stored edge — and it is the
    /// half that parallelises without touching determinism, because every entry
    /// is written by exactly one thread at a fixed index. Keeping the scratch
    /// here is what lets that happen without an allocation per step.
    scratch_r2: Vec<F>,
    scratch_disp: Vec<F>,
}

impl VerletSkin {
    /// Wrap `search` (must be built at `cutoff + policy.skin`) with Verlet policy.
    ///
    /// The initial build is not counted in [`Self::n_rebuilds`].
    pub fn new(
        mut search: NeighborList,
        cutoff: F,
        policy: NeighborPolicy,
        positions: ArrayView2<'_, F>,
        simbox: SimBox,
    ) -> Result<Self, SkinError> {
        if !(cutoff > 0.0 && cutoff.is_finite()) {
            return Err(SkinError::Invalid(format!(
                "cutoff must be a positive finite length, got {cutoff} Å"
            )));
        }
        if policy.skin < 0.0 {
            return Err(SkinError::Invalid(format!(
                "skin must be >= 0 Å, got {} Å",
                policy.skin
            )));
        }
        if policy.every < 1 {
            return Err(SkinError::Invalid(format!(
                "every must be an integer >= 1 step, got {}",
                policy.every
            )));
        }
        if !policy.delay.is_multiple_of(policy.every) {
            return Err(SkinError::Invalid(format!(
                "delay {} steps must be a multiple of every {} steps; LAMMPS Neighbor::init \
                 rejects the same pair, because the danger threshold max(every, delay) would \
                 be an ago the gate never permits, leaving ndanger silently dead.",
                policy.delay, policy.every
            )));
        }
        if positions.ncols() != 3 {
            return Err(SkinError::Invalid(format!(
                "positions must have shape (N, 3), got {:?}",
                positions.shape()
            )));
        }
        if !simbox.is_cell_defined() {
            return Err(SkinError::Invalid(
                "VerletSkin requires a geometrically defined cell".into(),
            ));
        }

        let r_build = cutoff + policy.skin;
        let want = search.cutoff();
        if (want - r_build).abs() > 1e-12 {
            return Err(SkinError::Invalid(format!(
                "NeighborList cutoff {want} Å must equal cutoff + skin = {r_build} Å \
                 (cutoff {cutoff} Å + skin {} Å)",
                policy.skin
            )));
        }

        let widths = simbox.nearest_plane_distance();
        let min_width = widths.iter().copied().fold(F::INFINITY, F::min);
        let half_width = 0.5 * min_width;
        if !simbox.is_free() && r_build > half_width {
            return Err(SkinError::Invalid(format!(
                "r_build {r_build} Å (cutoff {cutoff} Å + skin {} Å) exceeds half the \
                 minimum perpendicular cell width ({half_width:.3} Å); the minimum-image \
                 reduction would silently drop pairs inside the cutoff.",
                policy.skin
            )));
        }

        let n_atoms = positions.nrows();
        search.build(positions, &simbox);

        let mut skin = Self {
            cutoff,
            skin: policy.skin,
            every: policy.every,
            delay: policy.delay,
            check: policy.check,
            r_build,
            simbox,
            search,
            edges: Vec::new(),
            n_edges: 0,
            n_rebuilds: 0,
            ago: 0,
            ndanger: 0,
            half_skin_sq: (0.5 * policy.skin) * (0.5 * policy.skin),
            ambiguity_guard_sq: half_width * half_width,
            danger_ago: policy.every.max(policy.delay),
            x_hold: Array2::zeros((n_atoms, 3)),
            pairs_buf: Neighbors::empty(
                QueryMode::SelfQuery { n_points: n_atoms },
                NeighborColumns::FULL,
            ),
            scratch_r2: Vec::new(),
            scratch_disp: Vec::new(),
        };
        skin.pairs_buf.simbox = Some(skin.simbox.clone());
        skin.rebuild_edges();
        skin.hold(positions);
        Ok(skin)
    }

    /// Interaction cutoff `r_cut` in Å (fixed at construction).
    pub fn cutoff(&self) -> F {
        self.cutoff
    }

    /// Verlet skin in Å (from [`NeighborPolicy`], fixed at construction).
    pub fn skin(&self) -> F {
        self.skin
    }

    /// `every` gate in steps (fixed at construction).
    pub fn every(&self) -> usize {
        self.every
    }

    /// `delay` gate in steps (fixed at construction).
    pub fn delay(&self) -> usize {
        self.delay
    }

    /// Displacement check switch (fixed at construction).
    pub fn check(&self) -> bool {
        self.check
    }

    /// Live edge count (`== edges().len()`).
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }

    /// Rebuilds since construction; the initial build is not one of them.
    pub fn n_rebuilds(&self) -> usize {
        self.n_rebuilds
    }

    /// Steps since the last build (LAMMPS `ago`).
    pub fn ago(&self) -> usize {
        self.ago
    }

    /// Rebuilds that fired at the first permitted opportunity.
    pub fn ndanger(&self) -> usize {
        self.ndanger
    }

    /// Restore cadence counters from a trusted persistence snapshot.
    ///
    /// Geometry, edges, and the held positions must first be reconstructed via
    /// [`new`](Self::new). This only restores progress through the rebuild
    /// policy so a resumed run makes the same next cadence decision.
    pub fn restore_progress(&mut self, ago: usize, n_rebuilds: usize, ndanger: usize) {
        self.ago = ago;
        self.n_rebuilds = n_rebuilds;
        self.ndanger = ndanger;
    }

    /// Build radius `cutoff + skin` in Å (derived, never settable).
    pub fn r_build(&self) -> F {
        self.r_build
    }

    /// Half-shell edges from the last rebuild.
    pub fn edges(&self) -> &[SkinPair] {
        &self.edges
    }

    /// The cell this list was built against.
    pub fn simbox(&self) -> &SimBox {
        &self.simbox
    }

    /// Core pair search, built at [`Self::r_build`]. Interaction cutoff is
    /// the caller's (the pair potential still masks the skin shell).
    pub fn search(&self) -> &NeighborList {
        &self.search
    }

    /// Hand the pair search back, to wrap again for another cell.
    pub fn into_search(self) -> NeighborList {
        self.search
    }

    fn rebuild_edges(&mut self) {
        let r_build_sq = self.r_build * self.r_build;
        self.edges.clear();
        self.search.for_each_pair(|pair| {
            if pair.dist_sq <= 0.0 || pair.dist_sq > r_build_sq {
                return;
            }
            self.edges.push(SkinPair {
                i: pair.i,
                j: pair.j,
            });
        });
        self.n_edges = self.edges.len();
    }

    fn hold(&mut self, positions: ArrayView2<'_, F>) {
        self.ago = 0;
        self.x_hold.assign(&positions);
    }

    fn build_at(&mut self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        self.require_shape(positions)?;
        self.search.build(positions, &self.simbox);
        self.rebuild_edges();
        self.hold(positions);
        Ok(())
    }

    fn require_shape(&self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        let n = self.x_hold.nrows();
        if positions.shape() != [n, 3] {
            return Err(SkinError::Invalid(format!(
                "positions must have shape ({n}, 3) — this list was constructed for {n} atoms \
                 — but got {:?}",
                positions.shape()
            )));
        }
        Ok(())
    }

    /// Recompute the neighbour list at `positions` (unconditional).
    pub fn rebuild(&mut self, positions: ArrayView2<'_, F>) -> Result<(), SkinError> {
        self.build_at(positions)?;
        self.n_rebuilds += 1;
        Ok(())
    }

    /// Rebuild at `positions` if the `every`/`delay`/`check` gate says so.
    ///
    /// Call once per force evaluation, at the positions being evaluated.
    /// Returns whether a rebuild happened.
    pub fn update(&mut self, positions: ArrayView2<'_, F>) -> Result<bool, SkinError> {
        self.require_shape(positions)?;
        self.ago += 1;
        let permitted = self.ago >= self.delay && self.ago.is_multiple_of(self.every);
        if !permitted {
            return Ok(false);
        }
        if !self.check {
            self.rebuild(positions)?;
            return Ok(true);
        }
        // Minimum-image displacement, so a wrapped crossing reads as the step
        // the atom took rather than as a jump of one cell. Exact in every
        // configuration this type accepts — see the module documentation.
        let mic = self.simbox.mic();
        let mut max_d2 = 0.0;
        for i in 0..positions.nrows() {
            let d = mic.apply([
                positions[[i, 0]] - self.x_hold[[i, 0]],
                positions[[i, 1]] - self.x_hold[[i, 1]],
                positions[[i, 2]] - self.x_hold[[i, 2]],
            ]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > max_d2 {
                max_d2 = d2;
            }
        }
        if max_d2 >= self.ambiguity_guard_sq && !self.simbox.is_free() {
            return Err(SkinError::Guard(format!(
                "displacement {:.3} Å >= half the minimum perpendicular cell \
                 width ({:.3} Å): the minimum image can no longer identify which \
                 copy this atom came from, so the number is not a displacement. \
                 A cell change or a blow-up, not a wrap — wrapped coordinates are \
                 expected and handled.",
                max_d2.sqrt(),
                self.ambiguity_guard_sq.sqrt()
            )));
        }
        if max_d2 > self.half_skin_sq {
            if self.ago == self.danger_ago {
                self.ndanger += 1;
            }
            self.rebuild(positions)?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Stream stored edges with **current** minimum-image geometry.
    ///
    /// `disp` is MIC `r_j − r_i` at `positions`; `r2 = |disp|²`. Pairs outside
    /// the interaction `cutoff` are still visited (caller may mask).
    pub fn for_each_pair_at<G>(&self, positions: ArrayView2<'_, F>, mut f: G)
    where
        G: FnMut(u32, u32, F, [F; 3]),
    {
        debug_assert_eq!(positions.ncols(), 3);
        let mic = self.simbox.mic();
        let Some(pos) = positions.as_slice() else {
            // Fall back for non-contiguous views (rare in MD).
            for edge in &self.edges {
                let i = edge.i as usize;
                let j = edge.j as usize;
                let pi = [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
                let pj = [positions[[j, 0]], positions[[j, 1]], positions[[j, 2]]];
                let disp = mic.apply([pj[0] - pi[0], pj[1] - pi[1], pj[2] - pi[2]]);
                let r2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
                f(edge.i, edge.j, r2, disp);
            }
            return;
        };
        for edge in &self.edges {
            let i = edge.i as usize;
            let j = edge.j as usize;
            let bi = 3 * i;
            let bj = 3 * j;
            let disp = mic.apply([
                pos[bj] - pos[bi],
                pos[bj + 1] - pos[bi + 1],
                pos[bj + 2] - pos[bi + 2],
            ]);
            let r2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
            f(edge.i, edge.j, r2, disp);
        }
    }

    /// Run the update policy, then materialize `(i, j, disp, dist_sq)` for the
    /// current geometry into a reusable FULL buffer. This is the only place
    /// in the MD loop that computes a minimum image.
    pub fn pairs_at(&mut self, positions: ArrayView2<'_, F>) -> Result<&Neighbors, SkinError> {
        self.update(positions)?;
        let n_edges = self.edges.len();
        self.scratch_r2.resize(n_edges, 0.0);
        self.scratch_disp.resize(n_edges * 3, 0.0);

        // Pass one: the geometry. A random gather per stored edge, which is
        // what this costs, and every result written at its own index — so this
        // is the half that takes threads without taking the answer's
        // reproducibility with it.
        let owned_copy: Vec<F>;
        let pos: &[F] = match positions.as_slice() {
            Some(sl) => sl,
            None => {
                owned_copy = positions.iter().copied().collect();
                &owned_copy
            }
        };
        let mic = self.simbox.mic();
        let edges = &self.edges;
        let r2s = &mut self.scratch_r2;
        let disps = &mut self.scratch_disp;
        let one = |edge: &SkinPair, r2: &mut F, d: &mut [F]| {
            let bi = 3 * edge.i as usize;
            let bj = 3 * edge.j as usize;
            let dr = mic.apply([
                pos[bj] - pos[bi],
                pos[bj + 1] - pos[bi + 1],
                pos[bj + 2] - pos[bi + 2],
            ]);
            *r2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            d.copy_from_slice(&dr);
        };

        // Below this the fork/join costs more than the work it splits.
        #[cfg(feature = "rayon")]
        let parallel = n_edges >= 4_096;
        #[cfg(not(feature = "rayon"))]
        let parallel = false;
        if parallel {
            #[cfg(feature = "rayon")]
            {
                use rayon::prelude::*;
                r2s.par_iter_mut()
                    .zip(disps.par_chunks_mut(3))
                    .zip(edges.par_iter())
                    .for_each(|((r2, d), edge)| one(edge, r2, d));
            }
        } else {
            for ((r2, d), edge) in r2s.iter_mut().zip(disps.chunks_mut(3)).zip(edges.iter()) {
                one(edge, r2, d);
            }
        }

        // Pass two: the selection, in edge order. The stored edges reach
        // `cutoff + skin`; the table must not. The skin is a *caching* policy —
        // it decides how often the search runs, and it may not decide what
        // interacts. A kernel without a cutoff of its own would otherwise score
        // the shell, and the energy would step every time the list rebuilt.
        //
        // Serial, and deliberately: the rows come out in the order of the
        // edges, and a parallel append would let scheduling decide it.
        self.pairs_buf.set_mode(QueryMode::SelfQuery {
            n_points: positions.nrows(),
        });
        self.pairs_buf.clear();
        let cutoff2 = self.cutoff * self.cutoff;
        for (k, edge) in self.edges.iter().enumerate() {
            let r2 = self.scratch_r2[k];
            if r2 <= cutoff2 {
                let d = &self.scratch_disp[k * 3..k * 3 + 3];
                self.pairs_buf.push(edge.i, edge.j, r2, [d[0], d[1], d[2]]);
            }
        }
        Ok(&self.pairs_buf)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::Cell;

    /// The table is the same, bit for bit, however many threads built it.
    ///
    /// The displacement pass is parallel. That is safe only because every entry
    /// is written at a fixed index by exactly one thread and the rows are then
    /// selected in edge order — no summation, no append, nothing for a
    /// scheduler to decide. This asserts it rather than trusting the argument:
    /// a reordering would change the order of `energy += e` in every kernel
    /// downstream, and the last bits of an MD trajectory with it.
    #[cfg(feature = "rayon")]
    #[test]
    fn the_pair_table_does_not_depend_on_the_thread_count() {
        let n = 24_usize;
        let bx = cube(20.0);
        let pos = Array2::from_shape_fn((n, 3), |(a, k)| {
            // Deterministic scatter, not a lattice: equal distances would let a
            // reordering go unnoticed.
            let x = (a * 7 + k * 13) as F;
            (x * 0.618).fract() * 20.0
        });
        let build = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut skin = VerletSkin::new(
                    NeighborList::new(8.0).unwrap(),
                    8.0,
                    NeighborPolicy {
                        skin: 0.0,
                        ..NeighborPolicy::default()
                    },
                    pos.view(),
                    bx.clone(),
                )
                .unwrap();
                let t = skin.pairs_at(pos.view()).unwrap();
                (
                    t.query_point_indices().to_vec(),
                    t.point_indices().to_vec(),
                    t.dist_sq().unwrap().to_vec(),
                    t.disp().unwrap().to_owned(),
                )
            })
        };

        let one = build(1);
        assert!(!one.0.is_empty(), "the fixture must produce pairs");
        for threads in [2, 4, 8] {
            let many = build(threads);
            assert_eq!(one.0, many.0, "{threads} threads reordered the i column");
            assert_eq!(one.1, many.1, "{threads} threads reordered the j column");
            for (k, (a, b)) in one.2.iter().zip(&many.2).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: dist_sq[{k}]");
            }
            for (k, (a, b)) in one.3.iter().zip(many.3.iter()).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: disp[{k}]");
            }
        }
    }

    use ndarray::{Array2, array};

    use super::*;

    fn cube(a: F) -> SimBox {
        SimBox::new(Cell::Lengths([a; 3]), [0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn two_atoms(dx: F) -> Array2<F> {
        array![[0.0, 0.0, 0.0], [dx, 0.0, 0.0]]
    }

    fn skin_link(
        cutoff: F,
        policy: NeighborPolicy,
        pos: ArrayView2<'_, F>,
        box_a: F,
    ) -> VerletSkin {
        let search = NeighborList::new(cutoff + policy.skin).unwrap();
        VerletSkin::new(search, cutoff, policy, pos, cube(box_a)).unwrap()
    }

    #[test]
    fn delay_must_be_multiple_of_every() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.0).unwrap(),
            2.0,
            NeighborPolicy {
                every: 2,
                delay: 3,
                ..NeighborPolicy::default()
            },
            two_atoms(1.0).view(),
            cube(20.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("multiple"));
    }

    #[test]
    fn search_cutoff_must_match_r_build() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.0).unwrap(),
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            two_atoms(1.0).view(),
            cube(20.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("cutoff + skin"));
    }

    #[test]
    fn r_build_beyond_half_width_is_rejected() {
        let Err(err) = VerletSkin::new(
            NeighborList::new(2.5).unwrap(),
            2.5,
            NeighborPolicy::default(),
            two_atoms(1.0).view(),
            cube(4.0),
        ) else {
            panic!("expected error");
        };
        assert!(format!("{err}").contains("r_build"));
    }

    #[test]
    fn pair_inside_cutoff_is_half_shell() {
        let nl = skin_link(2.5, NeighborPolicy::default(), two_atoms(1.0).view(), 20.0);
        assert_eq!(nl.n_edges(), 1);
        assert_eq!(nl.edges()[0].i, 0);
        assert_eq!(nl.edges()[0].j, 1);
    }

    #[test]
    fn pair_outside_cutoff_is_absent() {
        let nl = skin_link(1.0, NeighborPolicy::default(), two_atoms(1.5).view(), 20.0);
        assert_eq!(nl.n_edges(), 0);
    }

    #[test]
    fn half_skin_exactly_does_not_rebuild() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(!rebuilt);
        assert_eq!(nl.n_rebuilds(), 0);
    }

    #[test]
    fn half_skin_beyond_does_rebuild() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(rebuilt);
        assert_eq!(nl.n_rebuilds(), 1);
    }

    #[test]
    fn for_each_pair_at_uses_live_geometry() {
        let pos0 = two_atoms(1.0);
        let nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            20.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let mut r2_live = 0.0;
        nl.for_each_pair_at(pos1.view(), |_i, _j, r2, disp| {
            r2_live = r2;
            assert!((disp[0] - 1.4).abs() < 1e-12);
        });
        assert!((r2_live - 1.4 * 1.4).abs() < 1e-12);
    }

    /// An atom that crosses a face is displaced by the step it took, not by a
    /// cell. This is the invariant that lets MD keep wrapped coordinates: the
    /// raw difference here is 9.8 Å, the true motion is 0.2 Å, and the skin
    /// must see the latter and stay quiet.
    #[test]
    fn a_wrapped_crossing_is_not_a_displacement() {
        let pos0 = array![[0.1_f64, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 1.0,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            10.0,
        );
        // Atom 0 steps -0.2 Å through the origin face and comes back at 9.9.
        let pos1 = array![[9.9_f64, 0.0, 0.0], [5.0, 0.0, 0.0]];
        let rebuilt = nl
            .update(pos1.view())
            .expect("a wrapped crossing must not be an error");
        assert!(
            !rebuilt,
            "0.2 Å of motion is far inside skin/2 = 0.5 Å; no rebuild is owed"
        );
    }

    /// The guard survives, with a different meaning: past half the minimum
    /// perpendicular width the minimum image cannot say which copy an atom came
    /// from, so the number is not a displacement. That is a blow-up, and it
    /// still raises.
    #[test]
    fn ambiguous_displacement_still_raises() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 0.5,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            10.0,
        );
        // 5.0 Å on a 10 Å cube is exactly half the perpendicular width.
        let pos1 = array![[5.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let err = nl.update(pos1.view()).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("minimum image can no longer identify"),
            "{msg}"
        );
        assert!(
            !msg.contains("unwrapped"),
            "wrapped coordinates are expected: {msg}"
        );
    }

    #[test]
    fn check_false_skips_the_guard_and_rebuilds_on_cadence() {
        let pos0 = two_atoms(1.0);
        let mut nl = skin_link(
            2.0,
            NeighborPolicy {
                skin: 0.5,
                check: false,
                ..NeighborPolicy::default()
            },
            pos0.view(),
            10.0,
        );
        let pos1 = array![[0.0, 0.0, 0.0], [7.0, 0.0, 0.0]];
        let rebuilt = nl.update(pos1.view()).unwrap();
        assert!(rebuilt);
    }

    #[test]
    fn linkcell_and_bruteforce_agree_on_the_edge_set() {
        let mut pos = Array2::<F>::zeros((8, 3));
        for i in 0..8 {
            pos[[i, 0]] = (i % 2) as F * 2.0;
            pos[[i, 1]] = ((i / 2) % 2) as F * 2.0;
            pos[[i, 2]] = (i / 4) as F * 2.0;
        }
        let policy = NeighborPolicy {
            skin: 0.5,
            ..NeighborPolicy::default()
        };
        let cutoff = 3.5;
        let a = VerletSkin::new(
            NeighborList::new(cutoff + policy.skin).unwrap(),
            cutoff,
            policy,
            pos.view(),
            cube(20.0),
        )
        .unwrap();
        let b = VerletSkin::new(
            NeighborList::brute_force(cutoff + policy.skin).unwrap(),
            cutoff,
            policy,
            pos.view(),
            cube(20.0),
        )
        .unwrap();
        assert_eq!(edge_set(&a), edge_set(&b));
    }

    fn edge_set(nl: &VerletSkin) -> std::collections::BTreeSet<(u32, u32)> {
        nl.edges().iter().map(|e| (e.i, e.j)).collect()
    }
}
