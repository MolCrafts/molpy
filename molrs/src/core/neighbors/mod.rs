//! Neighbor-list algorithms for pairwise distance queries; [`NeighborList`] is the entry point.

use crate::core::SimBox;
use crate::op::{F, Fnx3, Fnx3View};
use ndarray::ArrayView2;

mod aabb;
mod bruteforce;
mod filter;
mod grid;
mod linkcell;
mod query;
mod verlet_skin;

pub use aabb::AabbQuery;
pub use bruteforce::BruteForce;
pub use filter::{filter_rad, filter_sann};
pub use grid::CellGrid;
pub use linkcell::LinkCell;
pub use query::NeighborQuery;
pub use verlet_skin::{NeighborPolicy, SkinError, SkinPair, VerletSkin};

// NeighborColumns and NeighborPair are defined below next to Neighbors.

// ---------------------------------------------------------------------------
// QueryMode — which point sets a table indexes, and how large they are
// ---------------------------------------------------------------------------

/// Which point sets a [`Neighbors`] table indexes, plus the size of each.
///
/// A search is one of two shapes. Either one point set is searched against
/// itself (*self-query*), in which case there is a single population and a
/// single count; or a set of query points is searched against a separate
/// reference set (*cross-query*), in which case there are two populations and
/// two counts.
///
/// The counts live inside the mode rather than beside it, so the illegal state
/// "self-query whose two counts disagree" cannot be written down at all. The
/// counts matter downstream: number densities (and therefore radial
/// distribution function normalization) are computed from them, and a
/// half-shell self-query contributes each pair once where a full-shell list
/// would contribute it twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// Self-query: pairs within the same point set, half-shell.
    ///
    /// Both `i` and `j` index that one set, and every pair satisfies `i < j`,
    /// so each unordered pair is present exactly once and no pair has `i == j`.
    SelfQuery {
        /// Size of the single point set both `i` and `j` index.
        n_points: usize,
    },
    /// Cross-query: `i` indexes the query points, `j` indexes the reference
    /// points.
    ///
    /// Directed and full-shell: there is no `i < j` constraint, and every query
    /// point reports all of its reference neighbors. If the caller passes the
    /// same coordinates as both sets, a point is its own neighbor at distance
    /// zero and both orderings of a pair appear — that is the defined
    /// behavior, not an accident, and it is why a cross-query must never be
    /// relabelled as a self-query.
    CrossQuery {
        /// Size of the query point set (`i` indices).
        n_query_points: usize,
        /// Size of the reference point set (`j` indices).
        n_points: usize,
    },
}

// ---------------------------------------------------------------------------
// PairVisitor -- zero-allocation callback for on-demand pair traversal
// ---------------------------------------------------------------------------

/// Callback a [`Backend`] hands each discovered pair to, without storing it.
///
/// A visitor always receives the complete physics of a pair — the same two
/// quantities a [`NeighborPair`] carries — because nothing has been dropped by
/// a column policy yet. Crate-internal: the public streaming API is
/// [`NeighborList::for_each_pair`], which packs these four arguments into a
/// [`NeighborPair`].
pub(crate) trait PairVisitor {
    /// Called for each pair `(i, j)` within the cutoff.
    ///
    /// * `i`, `j` — original particle indices (indices into the point set the
    ///   search was built from, not into any sorted internal order)
    /// * `dist_sq` — squared minimum-image distance, Å²
    /// * `diff` — minimum-image displacement `r_j - r_i`, Å, unnormalized.
    ///   This is the same quantity the table calls `disp`
    ///   ([`NeighborPair::disp`]).
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]);
}

/// Blanket impl: any `FnMut(u32, u32, F, [F; 3])` is a `PairVisitor`.
impl<T: FnMut(u32, u32, F, [F; 3])> PairVisitor for T {
    #[inline(always)]
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]) {
        self(i, j, dist_sq, diff)
    }
}

// ---------------------------------------------------------------------------
// Backend trait — the search algorithm behind a NeighborList
// ---------------------------------------------------------------------------

/// A search algorithm a [`NeighborList`] can run on: index the points, then
/// enumerate the pairs.
///
/// Crate-internal. Callers choose a backend by constructor
/// ([`NeighborList::new`] for the cell list, [`NeighborList::brute_force`] for
/// the O(N²) reference), not by supplying their own implementation, so this
/// trait is not part of the public surface.
///
/// Every implementor searches one point set against itself, so what it produces
/// is always a half-shell self-query: `mode()` is [`QueryMode::SelfQuery`] and
/// every pair satisfies `i < j`. Cross-queries against a second point set go
/// through [`NeighborQuery`] instead.
///
/// The two halves are separate on purpose: `build_index` / `update_index` place
/// the points in space and allocate no pair storage, and `visit_pairs` walks
/// that index and hands out pairs. Deciding what to do with those pairs —
/// reduce them on the fly, or write them into a table — belongs to
/// [`NeighborList`], not here.
pub(crate) trait Backend: std::fmt::Debug {
    /// Build the spatial index only — NO pair enumeration.
    ///
    /// `points` is an `N × 3` view of Cartesian coordinates (Å) and `bx` is the
    /// simulation box supplying the periodicity used for minimum-image
    /// distances.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive or `points` does not have exactly
    /// 3 columns.
    fn build_index(&mut self, points: Fnx3View<'_>, bx: &SimBox);

    /// The interaction cutoff (Å) this backend indexes at.
    ///
    /// Single authority for the value — the engine reads it back rather than
    /// carrying a second copy that could drift.
    fn cutoff(&self) -> F;

    /// Rebuild the spatial index for new positions — NO pair enumeration.
    ///
    /// Implementations may reuse internal buffers. There is no skin/Verlet
    /// shortcut: this re-indexes, it does not decide for you whether a rebuild
    /// was necessary.
    fn update_index(&mut self, points: Fnx3View<'_>, bx: &SimBox);

    /// Build the spatial index from three coordinate columns (Å) — NO pair
    /// enumeration. Column sibling of [`build_index`](Self::build_index).
    ///
    /// The three slices are already known to have equal length: the check
    /// belongs to [`NeighborList::build_columns`], the public boundary.
    ///
    /// The default interleaves the columns into an owned `N × 3` array and
    /// defers to `build_index`. That copy is the price of not having a column
    /// path, and an implementor that indexes columns natively — [`LinkCell`]
    /// does — overrides this to avoid it; for an O(N²) backend the copy is
    /// dominated by the search itself, so correctness beats avoiding it.
    fn build_index_columns(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        let mut points = Fnx3::zeros((xs.len(), 3));
        for i in 0..xs.len() {
            points[[i, 0]] = xs[i];
            points[[i, 1]] = ys[i];
            points[[i, 2]] = zs[i];
        }
        self.build_index(points.view(), bx);
    }

    /// Traverse pairs on demand, calling the visitor once per pair within the
    /// cutoff. Zero allocation — no [`Neighbors`] table is built.
    ///
    /// The visitor is handed real physics: `dist_sq` (Å²) and the minimum-image
    /// displacement `r_j - r_i` (Å) from the same evaluation. Without a prior
    /// index the visitor is simply never called.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor);

    /// Append every pair within the cutoff to `out`, keeping the columns `out`'s
    /// column policy asks for.
    ///
    /// The default drives [`visit_pairs`](Self::visit_pairs) and pushes each
    /// pair as it arrives, so a backend gets materialization for free and the
    /// two paths cannot disagree about *which* pairs exist. An implementor whose
    /// search parallelizes overrides this to write the table faster —
    /// [`LinkCell`] folds over occupied cells with rayon — and then owes three
    /// things, none of which the default can enforce for it:
    ///
    /// 1. the same pair *set* the visitor would have produced, half-shell with
    ///    `i < j`;
    /// 2. `out`'s column policy honored, so a column the caller did not ask for
    ///    is never written;
    /// 3. rows appended, never overwritten — `out` may already hold pairs.
    ///
    /// Pair *order* is not owed: a parallel fold emits in whatever order the
    /// work split produced, which is why nothing downstream may depend on the
    /// row order of a materialized table.
    fn materialize_into(&self, out: &mut Neighbors) {
        self.visit_pairs(&mut |i, j, d2, dr| out.push(i, j, d2, dr));
    }
}

// ---------------------------------------------------------------------------
// NeighborList — the engine: coordinates in, pairs out
// ---------------------------------------------------------------------------

/// Neighbor search over one point set: index the coordinates, then read pairs.
///
/// A **neighbor list** is the table of particle pairs that lie closer together
/// than a fixed **cutoff** distance. Almost every pairwise quantity in
/// molecular simulation is computed by walking such a table: short-range
/// forces; the **radial distribution function** (**RDF**), the average number
/// of neighbors found at each separation `r`; **bond-order parameters**, which
/// describe how the *directions* to a particle's neighbors are arranged;
/// clustering. Building the table by testing every pair costs `O(N²)` distance
/// evaluations for `N` particles. The algorithms here bucket space so that the
/// cost falls to `O(N)`.
///
/// This is the door to every self search in molrs. It owns the cutoff, the
/// backend that indexes space, and the box the index was built for, and it
/// keeps the two halves of the job apart:
///
/// - [`build`](Self::build), [`build_columns`](Self::build_columns) and
///   [`update`](Self::update) own the spatial index. None of the three
///   enumerates a pair or allocates a [`Neighbors`] table.
/// - [`for_each_pair`](Self::for_each_pair) streams the pairs;
///   [`neighbors`](Self::neighbors) materializes them into a table. These two
///   are the only way pairs ever leave the engine.
///
/// The pairs are a **half-shell self list**: every unordered pair is produced
/// exactly once, with `i < j`, each carrying its squared minimum-image distance
/// (Å²) and minimum-image displacement `r_j - r_i` (Å). A directed search
/// against a *second* point set is a different question — see
/// [`NeighborQuery::query`].
///
/// # Streaming versus materializing
///
/// An analysis that consumes each pair once and keeps only a reduction — a
/// radial distribution histogram, an energy sum — streams and allocates
/// nothing:
///
/// ```
/// use molrs::core::NeighborList;
/// use molrs::core::SimBox;
/// use ndarray::array;
///
/// let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
/// // Only the first two are within 3 Å of each other; the third is 4.5 Å away
/// // (5.5 Å the other way round the box), so it contributes no pair.
/// let points = array![[1.0, 1.0, 1.0], [1.4, 1.0, 1.0], [5.5, 1.0, 1.0]];
///
/// let mut nl = NeighborList::new(3.0);
/// nl.build(points.view(), &bx);
///
/// let bin_width = 0.5;
/// let mut histogram = vec![0usize; 6];
/// nl.for_each_pair(|pair| {
///     histogram[(pair.dist_sq.sqrt() / bin_width) as usize] += 1;
/// });
/// assert_eq!(histogram[0], 1); // the 0.4 Å pair
/// assert_eq!(histogram.iter().sum::<usize>(), 1);
/// ```
///
/// An analysis that walks the same pairs repeatedly, or needs them as columns —
/// a bond-order parameter needs the direction to each neighbor, not just the
/// distance — materializes once, naming the columns it will actually read:
///
/// ```
/// use molrs::core::{NeighborList, NeighborColumns};
/// use molrs::core::SimBox;
/// use ndarray::array;
///
/// let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
/// let points = array![[1.0, 1.0, 1.0], [1.4, 1.0, 1.0]];
///
/// let mut nl = NeighborList::new(3.0);
/// nl.build(points.view(), &bx);
///
/// let table = nl.neighbors(NeighborColumns::DISP);
/// let disp = table.disp().expect("DISP keeps the displacement column");
/// assert_eq!(table.n_pairs(), 1);
/// assert!((disp[[0, 0]] - 0.4).abs() < 1e-12);
/// assert!(table.dist_sq().is_none(), "DISP drops the distance column");
/// ```
///
/// [`NeighborColumns::FULL`] selects every *column*; it never means a
/// bidirectional pair list, and nothing here falls back to it implicitly —
/// [`neighbors`](Self::neighbors) takes the policy as an argument.
///
/// # Backends
///
/// [`NeighborList`] is the entry point: it owns the cutoff and the **backend**
/// that indexes space, and turns coordinates into pairs. Two backends:
///
/// - [`LinkCell`] (via [`NeighborList::new`]) — O(N) **cell list**: space is
///   cut into a grid of cells at least one cutoff wide, so a particle can only
///   have neighbors in its own cell and the 26 cells touching it. The default
///   backend and the right choice in production. Indexing runs on one thread;
///   [`NeighborList::neighbors`] then materializes the table by folding over
///   occupied cells with rayon (default feature, above a small-system
///   threshold), which is 2–4× faster at molecular-dynamics sizes and is why
///   the row order of a materialized table is unspecified.
///   [`NeighborList::for_each_pair`] stays single-threaded: it hands pairs to
///   an `FnMut` visitor, which cannot be shared across threads.
/// - [`AabbQuery`] (via [`NeighborList::aabb`]) — a **bounding-volume
///   hierarchy**: a binary tree of axis-aligned boxes, descended once per
///   lattice image, pruning any subtree whose box is out of reach. `O(N log N)`
///   to build. It earns that over the cell list where a cell cannot be sized
///   well — markedly non-uniform density, or particle sizes spread wide enough
///   that one cutoff makes the cells coarse and the scan mostly empty.
/// - [`BruteForce`] (via [`NeighborList::brute_force`]) — O(N²) all-pairs
///   reference: a test oracle, and adequate for very small systems.
///
/// [`NeighborQuery`] runs a **cross-query**, searching one point set against a
/// separate reference set. [`AabbQuery`] answers the one question a cutoff
/// cannot express — *which `k` points are closest*, which has no radius — via
/// [`AabbQuery::query_knn`]; it is the same tree that serves as the `Aabb`
/// backend above, so one index answers both.
///
/// # API at a glance
///
/// ```ignore
/// let mut nl = NeighborList::new(3.0);       // cell-list backend, cutoff 3 Å
/// nl.build(points.view(), &simbox);          // index only — no pair table
/// nl.build_columns(&xs, &ys, &zs, &simbox);  // ...same, from x/y/z columns
/// nl.update(points.view());                  // re-index, reusing that box
///
/// nl.for_each_pair(|pair| { /* ... */ });    // stream half-shell self pairs
/// let table = nl.neighbors(NeighborColumns::FULL);  // ...or materialize them
///
/// let nq = NeighborQuery::new(&simbox, points.view(), 3.0);
/// let cross = nq.query(query_points.view()); // cross-query, directed
///
/// table.query_point_indices()   // &[u32]           — i of each pair
/// table.point_indices()         // &[u32]           — j of each pair
/// table.dist_sq()               // Option<&[F]>     — Å², None if not stored
/// table.disp()                  // Option<Fnx3View> — Å,  None if not stored
/// ```
///
/// # Name mapping from freud
///
/// The API mirrors [freud-analysis](https://freud.readthedocs.io/) closely
/// enough that freud's documentation transfers, with two deliberate
/// differences: molrs stores the *squared* distance, and molrs makes the
/// physical columns opt-in.
///
/// | freud | molrs | difference |
/// |---|---|---|
/// | `NeighborList.query_point_indices` | [`Neighbors::query_point_indices()`] | — |
/// | `NeighborList.point_indices` | [`Neighbors::point_indices()`] | — |
/// | `NeighborList.distances` (r, Å) | [`Neighbors::dist_sq()`] (r², Å²) | molrs stores the square and never hides a square root inside an accessor; take `.sqrt()` at the call site |
/// | `NeighborList.vectors` | [`Neighbors::disp()`] | same unnormalized MIC vector `r_j - r_i` |
/// | (both always present) | [`NeighborColumns`] | freud always carries distances and vectors; molrs returns `None` for a column the search was told not to store |
/// | `freud.locality.LinkCell` | [`NeighborList`] (its [`LinkCell`] backend) | — |
/// | `freud.locality.AABBQuery` | [`AabbQuery`] | one tree, both questions: the `Aabb` backend of [`NeighborList`] for cutoff searches, [`AabbQuery::query_knn`] for k-nearest |
/// | `freud.locality.FilterSANN` / `FilterRAD` | [`filter_sann`] / [`filter_rad`] | — |
#[derive(Debug)]
pub struct NeighborList {
    /// The algorithm that indexes space and enumerates pairs. Also the single
    /// owner of the cutoff — read back via [`cutoff`](Self::cutoff).
    backend: Box<dyn Backend + Send + Sync>,
    /// Box fixed by the last [`build`](Self::build) or
    /// [`build_columns`](Self::build_columns). `None` until then, which is what
    /// makes an `update` before a `build` a loud error instead of a guess.
    bx: Option<SimBox>,
    /// Point count of the last build/update — the `n_points` a materialized
    /// table is tagged with.
    n_points: usize,
}

impl NeighborList {
    /// A search with the O(N) cell-list backend — the production choice.
    ///
    /// `cutoff` is the interaction radius in Å. It is fixed here rather than
    /// passed per query because it determines the cell width of the index.
    ///
    /// # Panics
    /// Panics if `cutoff` is not positive.
    pub fn new(cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        Self::with_backend(Box::new(LinkCell::new().cutoff(cutoff)))
    }

    /// A search with the bounding-volume-hierarchy backend.
    ///
    /// Same pair set as [`new`](Self::new) — a backend changes which pairs are
    /// *visited*, never which exist. The tree costs `O(N log N)` to build
    /// against the cell list's `O(N)`, and earns that back when a cell list
    /// cannot be sized well: a cell must be at least one cutoff wide, so a
    /// system with markedly non-uniform density, or with particle sizes spread
    /// wide enough that one cutoff makes the cells coarse, leaves a cell list
    /// scanning mostly-empty volume. `LinkCell` remains the default.
    ///
    /// # Panics
    /// Panics if `cutoff` is not positive.
    pub fn aabb(cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        Self::with_backend(Box::new(AabbQuery::new(cutoff)))
    }

    /// A search with the O(N²) all-pairs backend.
    ///
    /// Same results as [`new`](Self::new) — that is what makes it useful as a
    /// test oracle — at a cost that grows with the square of the particle
    /// count. Prefer it only for very small systems, or to check the cell list.
    ///
    /// # Panics
    /// Panics if `cutoff` is not positive.
    pub fn brute_force(cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        Self::with_backend(Box::new(BruteForce::new(cutoff)))
    }

    fn with_backend(backend: Box<dyn Backend + Send + Sync>) -> Self {
        Self {
            backend,
            bx: None,
            n_points: 0,
        }
    }

    /// The cutoff distance (Å) fixed at construction.
    #[inline]
    pub fn cutoff(&self) -> F {
        self.backend.cutoff()
    }

    /// Whether a [`build`](Self::build) or [`build_columns`](Self::build_columns)
    /// has happened — the precondition [`update`](Self::update) panics on.
    ///
    /// Binders that must not let a panic cross an FFI seam check this and raise
    /// their own error instead of shadowing the state with a flag of their own.
    #[inline]
    pub fn is_built(&self) -> bool {
        self.bx.is_some()
    }

    /// Box captured by the last [`build`](Self::build), if any.
    #[inline]
    pub fn simbox(&self) -> Option<&SimBox> {
        self.bx.as_ref()
    }

    /// Index `points` in `bx` — coordinates **and** box.
    ///
    /// `points` is an `N × 3` array of Cartesian coordinates in Å, one row per
    /// particle. `bx` is the simulation box, whose periodicity decides which
    /// periodic image of a pair counts (the minimum-image convention).
    ///
    /// This builds the spatial index and nothing else: no pairs are enumerated
    /// and no [`Neighbors`] table is allocated, so the cost is the indexing
    /// cost alone. Pairs come afterwards, from
    /// [`for_each_pair`](Self::for_each_pair) or
    /// [`neighbors`](Self::neighbors), and either may be called any number of
    /// times on one build.
    ///
    /// `bx` is retained, so a later [`update`](Self::update) can re-index new
    /// coordinates in the same box. **A simulation whose box changes from step
    /// to step must call `build` every step**, not `update`. That is the case
    /// whenever a *barostat* is running — the algorithm that holds the pressure
    /// constant by rescaling the box, as in constant-pressure NPT dynamics
    /// (fixed particle number `N`, pressure `P` and temperature `T`) — because
    /// `update` reuses the box captured here and would fold minimum images
    /// against a stale cell.
    ///
    /// # Panics
    /// Panics if `points` does not have exactly 3 columns.
    pub fn build(&mut self, points: Fnx3View<'_>, bx: &SimBox) {
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");
        self.backend.build_index(points, bx);
        self.bx = Some(bx.clone());
        self.n_points = points.nrows();
    }

    /// Index three coordinate columns in `bx` — the column sibling of
    /// [`build`](Self::build).
    ///
    /// `xs`, `ys` and `zs` are the Cartesian coordinates in Å, one slice per
    /// axis, all of length `N`; point `i` is `(xs[i], ys[i], zs[i])`. This is
    /// the shape a column store hands out (`atoms.x` / `atoms.y` / `atoms.z`),
    /// so such a caller reaches the engine without transposing into an `N × 3`
    /// array first.
    ///
    /// Semantics are `build`'s exactly: the spatial index is built and nothing
    /// else — no pairs enumerated, no [`Neighbors`] table allocated — `bx` is
    /// retained for a later [`update`](Self::update), and the pairs that
    /// [`for_each_pair`](Self::for_each_pair) and [`neighbors`](Self::neighbors)
    /// then produce are the same ones `build` would have produced from the same
    /// coordinates. The box caveat carries over too: when the box changes from
    /// step to step, re-index with this method (or `build`) each step rather
    /// than with `update`, which would keep the box captured here.
    ///
    /// # Panics
    /// Panics if the three columns do not have equal length.
    pub fn build_columns(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        assert!(
            xs.len() == ys.len() && ys.len() == zs.len(),
            "x/y/z columns must have equal length, got {} / {} / {}",
            xs.len(),
            ys.len(),
            zs.len()
        );
        self.backend.build_index_columns(xs, ys, zs, bx);
        self.bx = Some(bx.clone());
        self.n_points = xs.len();
    }

    /// Re-index new coordinates (`N × 3`, Å) in the box captured by the last
    /// [`build`](Self::build) or [`build_columns`](Self::build_columns).
    ///
    /// The natural per-step call in a fixed-volume simulation: the positions
    /// move, the box does not. Index-only like `build` — no pairs are
    /// enumerated and no [`Neighbors`] table is allocated. **If the box itself
    /// changed, call `build` again instead**: `update` silently keeps the old
    /// box, which under a barostat (constant-pressure NPT dynamics, where the
    /// box is rescaled every step) means folding minimum images against a stale
    /// cell.
    ///
    /// There is no *skin* here — no margin added to the cutoff so that one list
    /// stays usable for several steps, which is the trick a Verlet list plays.
    /// `update` re-indexes every time it is called rather than deciding for you
    /// that the previous index was still good enough.
    ///
    /// The point count may change between calls; a materialized table is tagged
    /// with the count from the most recent index.
    ///
    /// # Panics
    /// Panics if neither [`build`](Self::build) nor
    /// [`build_columns`](Self::build_columns) has run — the box is then unknown,
    /// and the panic message says to call `build` first. Panics too if `points`
    /// does not have exactly 3 columns.
    pub fn update(&mut self, points: Fnx3View<'_>) {
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");
        let bx = self
            .bx
            .as_ref()
            .expect("NeighborList::update reuses the box of the previous build: call build(points, bx) first");
        self.backend.update_index(points, bx);
        self.n_points = points.nrows();
    }

    /// Stream every pair within the cutoff, in whatever order the backend
    /// discovers them.
    ///
    /// This is a search of the point set against itself, so the stream is
    /// *half-shell*: each unordered pair arrives exactly once, with `i < j`, and
    /// no pair has `i == j`.
    ///
    /// Allocates nothing: each [`NeighborPair`] is handed to `f` and dropped.
    /// A pair always carries both physical quantities — `dist_sq` in Å² and the
    /// displacement `disp` in Å — taken from the same minimum-image evaluation,
    /// so `pair.dist_sq == |pair.disp|²` holds to floating-point rounding.
    ///
    /// Pair order is deliberately unspecified — a cell list walks cells, not
    /// particles — so a caller that needs a fixed order sorts, or materializes
    /// with [`neighbors`](Self::neighbors) and sorts the table. Without a prior
    /// [`build`](Self::build) the callback is simply never invoked.
    pub fn for_each_pair(&self, mut f: impl FnMut(NeighborPair)) {
        self.backend.visit_pairs(&mut |i, j, dist_sq, disp| {
            f(NeighborPair {
                i,
                j,
                dist_sq,
                disp,
            })
        });
    }

    /// Materialize the pairs into a [`Neighbors`] table keeping the columns
    /// `columns` asks for.
    ///
    /// The same pair *set* [`for_each_pair`](Self::for_each_pair) streams, so
    /// the result is `Neighbors::from_pairs(collected_stream, columns, mode)` up
    /// to row order. Row order is **not** part of the contract: the cell-list
    /// backend materializes by folding over occupied cells in parallel (rayon
    /// feature, on by default), which is 2–4× faster at molecular-dynamics
    /// sizes but hands back the pairs in whatever order the work split produced.
    /// A caller that needs a fixed order sorts the table. Streaming stays
    /// single-threaded — a visitor is `FnMut`, so `for_each_pair` cannot be
    /// parallelized behind the caller's back.
    ///
    /// The table is tagged `SelfQuery { n_points }` with the point count of
    /// the last index.
    ///
    /// A column `columns` leaves out is not written, and its accessor reports
    /// `None` rather than a fabricated zero; it cannot be added afterwards
    /// (see [`Neighbors::repack`]). `columns` is an argument rather than a
    /// default, so no call site ever materializes
    /// [`NeighborColumns::FULL`] by accident — the columns a table carries are
    /// always something the caller asked for by name.
    pub fn neighbors(&self, columns: NeighborColumns) -> Neighbors {
        let mut out = Neighbors::empty(
            QueryMode::SelfQuery {
                n_points: self.n_points,
            },
            columns,
        );
        self.backend.materialize_into(&mut out);
        out.simbox = self.bx.clone();
        out
    }
}

/// A set of neighbor pairs a pair kernel can stream: a materialized
/// [`Neighbors`] table, or a built [`NeighborList`] that streams pairs straight
/// out of its spatial index without materializing a table.
///
/// Both answer the same questions, so a kernel that only needs to see every
/// pair once takes `&dyn PairSource` and serves both — the table when the
/// caller already holds one, the list when the table would not fit in memory.
pub trait PairSource {
    /// Pair direction: a half-shell self-query or a directed cross-query.
    fn mode(&self) -> QueryMode;

    /// The box the pairs were searched in, if a search produced them.
    fn simbox(&self) -> Option<&SimBox>;

    /// The physical columns [`visit_pairs`](Self::visit_pairs) reports.
    fn columns(&self) -> NeighborColumns;

    /// Visit every pair once as `(i, j, dist_sq, disp)`. A column outside
    /// [`columns`](Self::columns) arrives as `None`, never as a zero.
    fn visit_pairs(&self, visitor: &mut dyn FnMut(u32, u32, Option<F>, Option<[F; 3]>));
}

impl PairSource for Neighbors {
    fn mode(&self) -> QueryMode {
        self.mode
    }

    fn simbox(&self) -> Option<&SimBox> {
        self.simbox.as_ref()
    }

    fn columns(&self) -> NeighborColumns {
        self.columns
    }

    fn visit_pairs(&self, visitor: &mut dyn FnMut(u32, u32, Option<F>, Option<[F; 3]>)) {
        self.for_each_pair(visitor);
    }
}

impl PairSource for NeighborList {
    fn mode(&self) -> QueryMode {
        QueryMode::SelfQuery {
            n_points: self.n_points,
        }
    }

    fn simbox(&self) -> Option<&SimBox> {
        self.bx.as_ref()
    }

    /// A streaming search computes both physical columns for every pair.
    fn columns(&self) -> NeighborColumns {
        NeighborColumns::FULL
    }

    fn visit_pairs(&self, visitor: &mut dyn FnMut(u32, u32, Option<F>, Option<[F; 3]>)) {
        self.backend
            .visit_pairs(&mut |i, j, dist_sq, disp| visitor(i, j, Some(dist_sq), Some(disp)));
    }
}

// ---------------------------------------------------------------------------
// Neighbors — the materialized pair table
// ---------------------------------------------------------------------------

/// Which per-pair physical columns a materialized [`Neighbors`] table keeps.
///
/// The pair indices `(i, j)` are always stored — two `u32`s, 8 bytes per pair.
/// The two physical columns are optional:
///
/// - [`dist_sq`](NeighborColumns::dist_sq) — squared minimum-image distance in
///   Å², one `f64` (8 B) per pair.
/// - [`disp`](NeighborColumns::disp) — minimum-image displacement
///   `r_j - r_i` in Å, three `f64`s (24 B) per pair.
///
/// # `FULL` is about columns, not about direction
///
/// [`FULL`](Self::FULL) means **every column is present**. It says nothing
/// about pair direction: a self-query stays half-shell (each unordered pair
/// once, `i < j`) under `FULL`, and an indices-only cross-query stays directed
/// and bidirectional. Column policy and pair direction are orthogonal, and the
/// word "full" here never means "both `(i, j)` and `(j, i)` are stored".
///
/// # Choosing a policy
///
/// Full storage costs 40 B/pair. An analysis that only needs distances — a
/// radial distribution function histogram built from a materialized table, for
/// instance — should use [`DIST_SQ`](Self::DIST_SQ) at 16 B/pair; one that only
/// needs directions, such as a bond-order parameter, should use
/// [`DISP`](Self::DISP) at 32 B/pair. When pair counts are large, prefer not
/// materializing at all: [`NeighborList::for_each_pair`] streams every pair
/// with both quantities and allocates nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeighborColumns {
    /// Store squared minimum-image distances (Å²).
    pub dist_sq: bool,
    /// Store minimum-image displacement vectors `r_j - r_i` as `(dx, dy, dz)`
    /// (Å, unnormalized).
    pub disp: bool,
}

/// The default is [`FULL`](NeighborColumns::FULL): unless a caller says
/// otherwise, a materialized table carries both physical columns, so no
/// downstream consumer is surprised by a missing one.
impl Default for NeighborColumns {
    fn default() -> Self {
        Self::FULL
    }
}

impl NeighborColumns {
    /// Indices only — lightest materialization (8 B/pair).
    ///
    /// Enough for connectivity questions (clustering, percolation) that never
    /// look at a distance or a direction.
    pub const INDICES_ONLY: Self = Self {
        dist_sq: false,
        disp: false,
    };
    /// Indices + squared distance `d²` (16 B/pair).
    ///
    /// Enough for any purely radial analysis, such as a radial distribution
    /// function histogram.
    pub const DIST_SQ: Self = Self {
        dist_sq: true,
        disp: false,
    };
    /// Indices + minimum-image displacement (32 B/pair).
    ///
    /// Enough for direction-dependent analyses (bond-order parameters), which
    /// can recover `d²` from the displacement themselves.
    pub const DISP: Self = Self {
        dist_sq: false,
        disp: true,
    };
    /// Every column present (40 B/pair) — *not* a bidirectional pair list.
    pub const FULL: Self = Self {
        dist_sq: true,
        disp: true,
    };
}

/// One neighbor pair carrying its full physics — the value a search emits.
///
/// A pair is never short a column at enumeration time. Writing `r_i` and `r_j`
/// for the two particle positions and `MIC(·)` for the minimum-image convention
/// (the shortest separation over all periodic copies of the box), the two
/// physical fields are
///
/// ```text
/// disp    = MIC(r_j - r_i)
/// dist_sq = |disp|^2
/// ```
///
/// Three properties follow and are worth stating outright, because each has
/// been a source of silent errors elsewhere:
///
/// - `disp` is **not normalized**. It is not divided by the distance and is not
///   a unit direction; its length is the pair distance itself. A caller that
///   wants a direction divides by `dist_sq.sqrt()`.
/// - `disp` points **from `i` to `j`**, so swapping the two indices flips its
///   sign.
/// - `dist_sq` is taken from the *same* minimum image as `disp`, in the same
///   evaluation, so `dist_sq == |disp|²` holds to floating-point rounding and
///   the two can never describe different periodic copies.
///
/// Both quantities come from the single call [`SimBox::shortest_vector_impl`].
/// Lengths carry the unit of the input coordinates, which is Å (ångström)
/// everywhere in molrs; `disp` is therefore in Å and `dist_sq` in Å².
///
/// Columns become optional only on the way into a [`Neighbors`] table; a value
/// of this type always has all of them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeighborPair {
    /// Query point index (`i`).
    pub i: u32,
    /// Reference point index (`j`).
    pub j: u32,
    /// Squared minimum-image distance (Å²).
    pub dist_sq: F,
    /// Minimum-image displacement `r_j - r_i` (Å), unnormalized.
    pub disp: [F; 3],
}

/// Materialized neighbor pair table (freud-style).
///
/// Column store of every pair within the cutoff: two index columns that are
/// always present, plus whichever physical columns [`NeighborColumns`] asked
/// for. Row `k` of the table is one pair, and all columns are indexed by the
/// same `k`.
///
/// # Pair direction
///
/// [`mode()`](Neighbors::mode()) records which point sets the indices refer to
/// and how large those sets are:
///
/// - **Self-query** — both indices refer to the same point set, and the table
///   is half-shell: every pair satisfies `i < j`, so each unordered pair
///   appears exactly once and never with `i == j`.
/// - **Cross-query** — `i` indexes the query points, `j` indexes the reference
///   points. Directed, with no ordering constraint between the two.
///
/// # Optional columns: `None` is not "empty"
///
/// [`dist_sq()`](Neighbors::dist_sq()) and [`disp()`](Neighbors::disp()) return
/// `Option`s. `None` means *this table never stored that quantity*, and it is
/// deliberately distinct from a stored-but-empty column: a table with a hundred
/// pairs and no `disp` column returns `None`, never a hundred rows of zeros.
/// Zeros would be a physically meaningful displacement (two coincident
/// particles), so fabricating them would turn a missing column into a silent
/// wrong answer. If an analysis needs a column, rerun the search with that
/// column enabled — it cannot be reconstructed afterwards (see
/// [`repack`](Neighbors::repack)).
///
/// # Building one
///
/// [`from_pairs`](Neighbors::from_pairs) is the public constructor: it consumes
/// a stream of [`NeighborPair`] values and keeps the columns the policy asks
/// for. [`NeighborList::neighbors`] is the same construction driven straight
/// from a search, without the intermediate stream.
#[derive(Debug, Clone)]
pub struct Neighbors {
    /// Which optional columns `push` writes; the rest stay empty.
    pub(crate) columns: NeighborColumns,
    /// Query point index per pair (i).
    pub(crate) idx_i: Vec<u32>,
    /// Reference point index per pair (j).
    pub(crate) idx_j: Vec<u32>,
    /// Squared distances, one per pair (empty when `!columns.dist_sq`).
    pub(crate) dist_sq: Vec<F>,
    /// Flat MIC displacements `[dx0,dy0,dz0, …]` (empty when `!columns.disp`).
    pub(crate) disp_flat: Vec<F>,
    /// Query identity: self vs cross, plus the point-set sizes.
    pub(crate) mode: QueryMode,
    /// The box the pairs were searched in. `None` for a table nothing searched
    /// in a box — hand-built rows, ghost-image tables, topology pair lists.
    pub(crate) simbox: Option<SimBox>,
}

impl Neighbors {
    /// Materialize a pair stream into a table, keeping the columns `columns`
    /// asks for.
    ///
    /// This is the only public way to build a table from pairs. Each
    /// [`NeighborPair`] arrives with both physical quantities; `columns` decides
    /// which of them survive into the table. A column the policy drops is simply
    /// not written — it is never zero-filled, and the corresponding accessor
    /// will report `None`. Rows keep the order in which the iterator yielded
    /// them, so a search that emits pairs in a deterministic order produces a
    /// deterministic table.
    ///
    /// `mode` states what the indices mean and how many points each set holds;
    /// it is not inferred from the pairs, because an empty stream carries no
    /// evidence either way.
    ///
    /// # Panics
    /// In debug builds, panics if `mode` is [`QueryMode::SelfQuery`] and a pair
    /// violates the half-shell contract `i < j`. Mislabelling a full-shell
    /// stream as a self-query would silently double every count downstream (a
    /// radial distribution function multiplies self-query pair counts by two to
    /// recover both orderings), so the contract is checked rather than trusted.
    /// The check is a `debug_assert!` and is compiled out of release builds; it
    /// is a guard against a programming error, not against untrusted input.
    /// [`QueryMode::CrossQuery`] imposes no ordering constraint and is never
    /// checked.
    pub fn from_pairs(
        pairs: impl IntoIterator<Item = NeighborPair>,
        columns: NeighborColumns,
        mode: QueryMode,
    ) -> Self {
        let mut out = Self::empty(mode, columns);
        let half_shell = matches!(mode, QueryMode::SelfQuery { .. });
        for p in pairs {
            debug_assert!(
                !half_shell || p.i < p.j,
                "self-query pairs are half-shell: expected i < j, got i = {}, j = {}",
                p.i,
                p.j
            );
            out.push(p.i, p.j, p.dist_sq, p.disp);
        }
        out
    }

    /// An empty table with the given query identity and column policy.
    pub(crate) fn empty(mode: QueryMode, columns: NeighborColumns) -> Self {
        Self {
            columns,
            idx_i: Vec::new(),
            idx_j: Vec::new(),
            dist_sq: Vec::new(),
            disp_flat: Vec::new(),
            mode,
            simbox: None,
        }
    }

    /// The box the pairs were searched in, or `None` if no search produced
    /// this table. A pair kernel that needs a volume reads it from here and
    /// rejects `None`; it never invents one.
    #[inline]
    pub fn simbox(&self) -> Option<&SimBox> {
        self.simbox.as_ref()
    }

    /// The same table, recording `simbox` as the box its pairs live in — for a
    /// table built by hand with [`from_pairs`](Self::from_pairs).
    pub fn with_simbox(mut self, simbox: SimBox) -> Self {
        self.simbox = Some(simbox);
        self
    }

    pub(crate) fn clear(&mut self) {
        self.idx_i.clear();
        self.idx_j.clear();
        self.dist_sq.clear();
        self.disp_flat.clear();
    }

    pub(crate) fn set_mode(&mut self, mode: QueryMode) {
        self.mode = mode;
    }

    /// Column policy for optional columns — equivalently, which of
    /// [`dist_sq()`](Self::dist_sq()) and [`disp()`](Self::disp()) will return
    /// `Some` for this table.
    #[inline]
    pub fn columns(&self) -> NeighborColumns {
        self.columns
    }

    /// Push a neighbor pair into the table.
    ///
    /// `d2` is the squared minimum-image distance (Å²) and `dr` the
    /// minimum-image displacement `r_j - r_i` (Å); each is retained only when
    /// the corresponding [`NeighborColumns`] flag is set, and silently dropped
    /// otherwise. Callers are responsible for the mode's ordering contract —
    /// this is the crate-internal fast path that engines use to write straight
    /// into the table, so unlike `from_pairs` it does not check `i < j`.
    #[inline(always)]
    pub(crate) fn push(&mut self, i: u32, j: u32, d2: F, dr: [F; 3]) {
        self.idx_i.push(i);
        self.idx_j.push(j);
        if self.columns.dist_sq {
            self.dist_sq.push(d2);
        }
        if self.columns.disp {
            self.disp_flat.extend(dr);
        }
    }

    /// Move every row of `other` onto the end of this table.
    ///
    /// The merge step of a parallel materialization: each worker folds its share
    /// of the search into a table of its own, and those partial tables are then
    /// concatenated. `other` is left empty. That is the *only* reason this
    /// exists, hence the feature gate — without `rayon` there is one worker, and
    /// it pushes straight into the caller's table.
    ///
    /// # Column alignment is the invariant this method exists to hold
    ///
    /// The columns are separate `Vec`s, so "row `k`" is a convention, not a type
    /// — appending `idx_i` without appending `disp_flat` would leave a table
    /// whose distances belong to the wrong pairs, and nothing downstream could
    /// notice. Concatenating *all* the columns in one place, from tables that by
    /// construction share a column policy, is what keeps them aligned: a column
    /// the policy drops is empty in both tables, so appending it is a no-op, and
    /// a column the policy keeps is appended in both. `disp_flat` holds three
    /// values per pair rather than one, and is appended whole for the same
    /// reason.
    ///
    /// # Panics
    /// In debug builds, panics if the two tables disagree about their columns
    /// policy — that is exactly the case where the concatenation would misalign
    /// the columns — or if the merged columns do not come out to one entry per
    /// pair.
    #[cfg(feature = "rayon")]
    pub(crate) fn append(&mut self, other: &mut Self) {
        debug_assert_eq!(
            self.columns, other.columns,
            "merging neighbor tables with different column policies would \
             misalign the columns"
        );
        if self.idx_i.is_empty() {
            // Receiving into an empty table — the final merge into the caller's
            // fresh `out` — is a buffer handover, not a copy: up to 40 B/pair
            // saved on multi-million-pair materializations.
            std::mem::swap(&mut self.idx_i, &mut other.idx_i);
            std::mem::swap(&mut self.idx_j, &mut other.idx_j);
            std::mem::swap(&mut self.dist_sq, &mut other.dist_sq);
            std::mem::swap(&mut self.disp_flat, &mut other.disp_flat);
        } else {
            self.idx_i.append(&mut other.idx_i);
            self.idx_j.append(&mut other.idx_j);
            self.dist_sq.append(&mut other.dist_sq);
            self.disp_flat.append(&mut other.disp_flat);
        }
        debug_assert_eq!(self.idx_i.len(), self.idx_j.len(), "index columns");
        debug_assert!(
            !self.columns.dist_sq || self.dist_sq.len() == self.idx_i.len(),
            "dist_sq column lost alignment with the index columns"
        );
        debug_assert!(
            !self.columns.disp || self.disp_flat.len() == 3 * self.idx_i.len(),
            "disp column lost alignment with the index columns"
        );
    }

    /// Iterate the materialized pairs without allocating.
    ///
    /// `callback` is invoked once per pair, in table order, with four
    /// arguments:
    ///
    /// 1. `i: u32` — the query point index of the pair.
    /// 2. `j: u32` — the reference point index of the pair.
    /// 3. `Option<F>` — the squared minimum-image distance in Å².
    /// 4. `Option<[F; 3]>` — the minimum-image displacement `r_j - r_i` in Å,
    ///    unnormalized.
    ///
    /// A column this table did not store arrives as `None` — never as a
    /// fabricated `0.0` or `[0, 0, 0]`, both of which are legal physical values
    /// (two coincident particles) and would therefore be indistinguishable from
    /// real data. The `Option`s are decided once by the table's column policy,
    /// not per pair: either every call sees `Some` for a given argument or every
    /// call sees `None`. Only a [`NeighborPair`] stream straight out of a search
    /// is guaranteed to carry every physical quantity.
    ///
    /// A consumer that requires a column should reject `None` loudly rather than
    /// substitute a value — see [`filter_rad`] and [`filter_sann`] for the
    /// in-tree precedent.
    pub fn for_each_pair<C>(&self, mut callback: C)
    where
        C: FnMut(u32, u32, Option<F>, Option<[F; 3]>),
    {
        for k in 0..self.n_pairs() {
            let d2 = self.columns.dist_sq.then(|| self.dist_sq[k]);
            let dr = self.columns.disp.then(|| {
                let b = k * 3;
                [
                    self.disp_flat[b],
                    self.disp_flat[b + 1],
                    self.disp_flat[b + 2],
                ]
            });
            callback(self.idx_i[k], self.idx_j[k], d2, dr);
        }
    }

    /// Re-materialize this table under a leaner column policy.
    ///
    /// **Downgrade only.** Columns kept by both the old and the new policy are
    /// copied verbatim; columns the new policy drops are discarded. Indices and
    /// [`mode()`](Self::mode()) are preserved, so the pair set and its ordering
    /// contract are unchanged; asking for the policy the table already has is a
    /// plain clone. The source table is left untouched.
    ///
    /// An upgrade is impossible by construction, not merely unimplemented: a
    /// column that was never stored cannot be recovered from the indices, since
    /// the table does not keep the coordinates or the box the distances came
    /// from. Rerun the search with that column enabled instead.
    ///
    /// # Panics
    /// Panics when `columns` asks for a column this table never stored, with a
    /// message naming the column. It never invents values, and in particular
    /// never fills the new column with zeros — a zero displacement or distance
    /// is physically meaningful (coincident particles) and would be
    /// indistinguishable from real data downstream.
    pub fn repack(&self, columns: NeighborColumns) -> Self {
        if columns.dist_sq && !self.columns.dist_sq {
            panic!(
                "repack cannot add a dist_sq column: physical quantities cannot be \
                 fabricated from an indices-only / lean list — rerun the neighbor \
                 search with NeighborColumns::dist_sq set"
            );
        }
        if columns.disp && !self.columns.disp {
            panic!(
                "repack cannot add a disp column: physical quantities cannot be \
                 fabricated from an indices-only / lean list — rerun the neighbor \
                 search with NeighborColumns::disp set"
            );
        }
        Self {
            columns,
            idx_i: self.idx_i.clone(),
            idx_j: self.idx_j.clone(),
            dist_sq: if columns.dist_sq {
                self.dist_sq.clone()
            } else {
                Vec::new()
            },
            disp_flat: if columns.disp {
                self.disp_flat.clone()
            } else {
                Vec::new()
            },
            mode: self.mode,
            simbox: self.simbox.clone(),
        }
    }

    // -----------------------------------------------------------------------
    // Public accessors (freud-style names)
    // -----------------------------------------------------------------------

    /// Number of neighbor pairs — the number of rows every column shares.
    #[inline]
    pub fn n_pairs(&self) -> usize {
        self.idx_i.len()
    }

    /// Query identity: self vs cross, plus the point-set sizes.
    #[inline]
    pub fn mode(&self) -> QueryMode {
        self.mode
    }

    /// Number of reference points (the point set used to build the spatial index).
    #[inline]
    pub fn n_points(&self) -> usize {
        match self.mode {
            QueryMode::SelfQuery { n_points } => n_points,
            QueryMode::CrossQuery { n_points, .. } => n_points,
        }
    }

    /// Number of query points.
    /// - Self-query: same as `n_points`.
    /// - Cross-query: the number of points passed to `query()`.
    #[inline]
    pub fn n_query_points(&self) -> usize {
        match self.mode {
            QueryMode::SelfQuery { n_points } => n_points,
            QueryMode::CrossQuery { n_query_points, .. } => n_query_points,
        }
    }

    /// Query point indices (i), one per pair.
    ///
    /// - Self-query: indices into the single point set.
    /// - Cross-query: indices into `query_points`.
    #[inline]
    pub fn query_point_indices(&self) -> &[u32] {
        &self.idx_i
    }

    /// Reference point indices (j), one per pair.
    ///
    /// - Self-query: indices into the single point set (always > i, half-shell).
    /// - Cross-query: indices into the reference `points`.
    #[inline]
    pub fn point_indices(&self) -> &[u32] {
        &self.idx_j
    }

    /// Squared minimum-image distances (Å²), one per pair.
    ///
    /// `None` when [`NeighborColumns::dist_sq`] was not set for this table —
    /// which is different from an empty slice, and must not be treated as
    /// "distances are zero". When `Some`, the slice has exactly
    /// [`n_pairs()`](Self::n_pairs) elements, aligned row-for-row with the index
    /// columns.
    ///
    /// The square root is deliberately left to the caller: most consumers
    /// compare against a squared cutoff and never need it, and an accessor that
    /// hid a per-pair `sqrt` would make that cost invisible.
    #[inline]
    pub fn dist_sq(&self) -> Option<&[F]> {
        self.columns.dist_sq.then_some(self.dist_sq.as_slice())
    }

    /// Minimum-image displacements `r_j - r_i` (Å) as an `n_pairs × 3` view.
    ///
    /// Row `k` holds `(dx, dy, dz)` for pair `k`, pointing from `i` to `j`, and
    /// is **not** normalized — its length is the pair distance, so a caller who
    /// wants a direction divides the row by that length. When `Some`, the view
    /// has exactly [`n_pairs()`](Self::n_pairs) rows.
    ///
    /// `None` when [`NeighborColumns::disp`] was not set for this table. That
    /// is not the same as a zero vector, which would mean two coincident
    /// particles.
    #[inline]
    pub fn disp(&self) -> Option<Fnx3View<'_>> {
        if !self.columns.disp {
            return None;
        }
        Some(
            ArrayView2::from_shape((self.n_pairs(), 3), &self.disp_flat)
                .expect("disp_flat shape mismatch"),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests — spec neighborlist-01-types: Neighbors / from_pairs / optional columns
// ---------------------------------------------------------------------------

#[cfg(test)]
mod test_fixtures {
    //! Fixtures and projections shared by the neighbor test modules —
    //! `from_pairs_tests`, `engine_tests`, and `linkcell::tests` — so one
    //! hand-derived MIC golden set has one home.
    use super::{Neighbors, SimBox};
    use crate::op::F;
    use ndarray::{Array2, array};

    /// Orthorhombic 10 × 8 × 6 Å box, fully periodic.
    pub(crate) fn small_box() -> SimBox {
        SimBox::ortho(
            array![10.0, 8.0, 6.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("valid orthorhombic box")
    }

    /// Four atoms in [`small_box`]; at cutoff 1.5 Å exactly three pairs survive.
    ///
    /// | pair | `disp` = MIC(r_j − r_i) | `dist_sq` |
    /// |------|--------------------------|-----------|
    /// | 0-1  | (−0.9, 0.0, 0.0)         | 0.81      |
    /// | 0-2  | ( 0.0, −0.8, 0.0)        | 0.64      |
    /// | 1-2  | ( 0.9, −0.8, 0.0)        | 1.45      |
    ///
    /// Atom 3 sits at the box centre, further than 1.5 Å from every other atom.
    pub(crate) fn small_points() -> Array2<F> {
        array![
            [0.5, 0.5, 0.5],
            [9.6, 0.5, 0.5],
            [0.5, 7.7, 0.5],
            [5.0, 4.0, 3.0],
        ]
    }

    /// The three goldens of [`small_points`], keyed by `(i, j)`.
    ///
    /// Derived by hand from the minimum-image convention: 0-1 wraps in x
    /// (9.6 → −0.4 relative to 0.5, i.e. 0.9 Å apart), 0-2 wraps in y (7.7 →
    /// −0.3, i.e. 0.8 Å apart), and 1-2 wraps in both, giving
    /// 0.9² + 0.8² = 1.45 Å².
    pub(crate) const SMALL_GOLDEN: [(u32, u32, F, [F; 3]); 3] = [
        (0, 1, 0.81, [-0.9, 0.0, 0.0]),
        (0, 2, 0.64, [0.0, -0.8, 0.0]),
        (1, 2, 1.45, [0.9, -0.8, 0.0]),
    ];

    /// Rows of a FULL materialized table as `(i, j, dist_sq, disp)`, sorted by
    /// `(i, j)` so that row order — which is not part of the contract — cannot
    /// decide a comparison.
    pub(crate) fn table_rows_sorted(nb: &Neighbors) -> Vec<(u32, u32, F, [F; 3])> {
        let d2 = nb.dist_sq().expect("FULL storage materializes dist_sq");
        let disp = nb.disp().expect("FULL storage materializes disp");
        let mut rows: Vec<(u32, u32, F, [F; 3])> = (0..nb.n_pairs())
            .map(|k| {
                (
                    nb.query_point_indices()[k],
                    nb.point_indices()[k],
                    d2[k],
                    [disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]],
                )
            })
            .collect();
        rows.sort_by_key(|r| (r.0, r.1));
        rows
    }
}

#[cfg(test)]
mod from_pairs_tests {
    use super::*;

    /// Two hard-coded pairs shared by the column-flag tests.
    ///
    /// | k | i | j | `dist_sq` | `disp`        | ‖disp‖² |
    /// |---|---|---|-----------|---------------|---------|
    /// | 0 | 0 | 1 | 2.0       | (1.0,1.0,0.0) | 2.0     |
    /// | 1 | 1 | 3 | 9.0       | (0.0,0.0,3.0) | 9.0     |
    ///
    /// Both satisfy the half-shell contract `i < j`, so they are legal under
    /// `QueryMode::SelfQuery { n_points: 4 }`.
    fn two_pairs() -> [NeighborPair; 2] {
        [
            NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 2.0,
                disp: [1.0, 1.0, 0.0],
            },
            NeighborPair {
                i: 1,
                j: 3,
                dist_sq: 9.0,
                disp: [0.0, 0.0, 3.0],
            },
        ]
    }

    /// AC-003: `FULL` materializes indices + both physical columns verbatim.
    #[test]
    fn from_pairs_full_roundtrip() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );

        assert_eq!(nb.n_pairs(), 2);
        assert!(matches!(nb.mode(), QueryMode::SelfQuery { n_points: 4 }));
        assert_eq!(nb.query_point_indices(), &[0u32, 1u32][..]);
        assert_eq!(nb.point_indices(), &[1u32, 3u32][..]);

        let d2 = nb.dist_sq().expect("FULL storage must materialize dist_sq");
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);

        let disp = nb.disp().expect("FULL storage must materialize disp");
        assert_eq!(disp.nrows(), 2);
        assert_eq!(disp.ncols(), 3);

        let expected: [[F; 3]; 2] = [[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]];
        for k in 0..2 {
            for c in 0..3 {
                assert!(
                    (disp[[k, c]] - expected[k][c]).abs() <= 1e-15,
                    "disp[{k}][{c}] = {} != {}",
                    disp[[k, c]],
                    expected[k][c]
                );
            }
        }
    }

    /// AC-004: no column requested ⇒ `None`, never a fabricated zero column.
    #[test]
    fn from_pairs_indices_only_gives_none_columns() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: 4 },
        );

        assert_eq!(nb.n_pairs(), 2);
        assert!(
            nb.dist_sq().is_none(),
            "INDICES_ONLY must not fabricate a zero dist_sq column"
        );
        assert!(
            nb.disp().is_none(),
            "INDICES_ONLY must not fabricate a zero disp column"
        );
    }

    /// AC-005: `DIST_SQ` keeps d² only.
    #[test]
    fn from_pairs_dist_sq_only() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::DIST_SQ,
            QueryMode::SelfQuery { n_points: 4 },
        );

        let d2 = nb.dist_sq().expect("DIST_SQ must materialize dist_sq");
        assert_eq!(d2.len(), nb.n_pairs());
        assert_eq!(d2, &[2.0_f64, 9.0_f64][..]);
        assert!(
            nb.disp().is_none(),
            "DIST_SQ must leave the disp column absent"
        );
    }

    /// AC-005: `DISP` keeps the MIC displacement column only.
    #[test]
    fn from_pairs_disp_only() {
        let nb = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::DISP,
            QueryMode::SelfQuery { n_points: 4 },
        );

        assert!(
            nb.dist_sq().is_none(),
            "DISP must leave the dist_sq column absent"
        );
        let disp = nb.disp().expect("DISP must materialize disp");
        assert_eq!(disp.nrows(), nb.n_pairs());
        assert_eq!(disp.ncols(), 3);
        assert!((disp[[0, 0]] - 1.0).abs() <= 1e-15);
        assert!((disp[[1, 2]] - 3.0).abs() <= 1e-15);
    }

    /// AC-006 + AC-007: a materialized self-query is half-shell (`i < j`) and
    /// its two physical columns agree, `d² == ‖disp‖²`, under PBC.
    ///
    /// Driven through the O(N²) reference backend
    /// ([`NeighborList::brute_force`]), whose all-pairs scan shares no
    /// cell-assignment code with the production cell list.
    ///
    /// Hard-coded orthorhombic box 10 × 8 × 6 Å, fully periodic, cutoff 1.5 Å:
    ///
    /// | pair | separation                    | d²   |
    /// |------|-------------------------------|------|
    /// | 0-1  | x-image, 0.9 Å                | 0.81 |
    /// | 0-2  | y-image, 0.8 Å                | 0.64 |
    /// | 1-2  | x- and y-image, (0.9, 0.8, 0) | 1.45 |
    ///
    /// Atom 3 sits at the box centre, beyond 1.5 Å of every other atom.
    #[test]
    fn bruteforce_self_half_shell_and_consistency() {
        let bx = super::test_fixtures::small_box();
        let pts = super::test_fixtures::small_points();

        let mut bf = NeighborList::brute_force(1.5);
        bf.build(pts.view(), &bx);
        let nb: Neighbors = bf.neighbors(NeighborColumns::FULL);

        assert_eq!(nb.n_pairs(), 3);
        assert!(matches!(nb.mode(), QueryMode::SelfQuery { .. }));

        let idx_i = nb.query_point_indices();
        let idx_j = nb.point_indices();
        let d2 = nb.dist_sq().expect("FULL storage materializes dist_sq");
        let disp = nb.disp().expect("FULL storage materializes disp");

        for k in 0..nb.n_pairs() {
            assert!(
                idx_i[k] < idx_j[k],
                "self-query pair {k} violates half-shell: {} !< {}",
                idx_i[k],
                idx_j[k]
            );
            let norm_sq = disp[[k, 0]] * disp[[k, 0]]
                + disp[[k, 1]] * disp[[k, 1]]
                + disp[[k, 2]] * disp[[k, 2]];
            assert!(
                (d2[k] - norm_sq).abs() <= 1e-12,
                "pair {k}: dist_sq {} != ||disp||^2 {}",
                d2[k],
                norm_sq
            );
        }
    }

    /// AC-009: dropping a column copies what survives, exactly.
    #[test]
    fn repack_downgrade_drops_columns() {
        let full = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );
        let lean = full.repack(NeighborColumns::DIST_SQ);

        assert_eq!(lean.n_pairs(), 2);
        assert_eq!(lean.query_point_indices(), &[0u32, 1u32][..]);
        assert_eq!(lean.point_indices(), &[1u32, 3u32][..]);
        assert_eq!(
            lean.dist_sq().expect("downgrade to DIST_SQ keeps dist_sq"),
            &[2.0_f64, 9.0_f64][..]
        );
        assert!(
            lean.disp().is_none(),
            "downgrade must drop the disp column, not zero it"
        );
        // Source table is untouched by the downgrade.
        assert!(full.disp().is_some());
    }

    /// AC-009: an upgrade cannot invent physics — it fails loudly.
    #[test]
    #[should_panic]
    fn repack_upgrade_panics() {
        let lean = Neighbors::from_pairs(
            two_pairs(),
            NeighborColumns::INDICES_ONLY,
            QueryMode::SelfQuery { n_points: 4 },
        );
        let _upgraded = lean.repack(NeighborColumns::FULL);
    }

    /// Spec §from_pairs trust boundary: `SelfQuery` with `i > j` is a
    /// programming error and must trip the debug assertion.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn self_query_from_pairs_debug_asserts_half_shell() {
        let bad = [NeighborPair {
            i: 3,
            j: 1,
            dist_sq: 4.0,
            disp: [2.0, 0.0, 0.0],
        }];
        let _nb = Neighbors::from_pairs(
            bad,
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );
    }
}

// ---------------------------------------------------------------------------
// Tests — spec neighborlist-02-engine: the NeighborList engine
// ---------------------------------------------------------------------------

/// The `NeighborList` engine: index-only `build`/`update`, a streaming
/// `for_each_pair`, and `neighbors(columns)` as the materializing sugar.
///
/// Every fixture here is hard-coded — lattices are generated by an explicit
/// deterministic loop, never by a random number generator — so a failure names
/// a specific pair rather than a seed.
#[cfg(test)]
mod engine_tests {
    use super::test_fixtures::{SMALL_GOLDEN, small_box, small_points, table_rows_sorted};
    use super::*;
    use ndarray::{Array2, array};

    /// Collect the engine's pair stream, sorted by `(i, j)`.
    ///
    /// The engine is free to emit pairs in any order (a cell-list walks cells,
    /// not particles), so every comparison in this module sorts first.
    fn collect_sorted(nl: &NeighborList) -> Vec<NeighborPair> {
        let mut pairs: Vec<NeighborPair> = Vec::new();
        nl.for_each_pair(|p| pairs.push(p));
        pairs.sort_by_key(|p| (p.i, p.j));
        pairs
    }

    /// 60 atoms on a 5 × 4 × 3 lattice (2 Å pitch) inside a 10 × 8 × 6 Å box,
    /// each site pushed off its node by a deterministic, index-derived offset.
    ///
    /// No random number generator: the offsets cycle with periods 3, 5 and 7 in
    /// x, y and z, which are coprime with each other and with the 5 × 4 × 3
    /// lattice, so the configuration is disordered enough to exercise the cell
    /// stencil in every direction while remaining reproducible byte-for-byte.
    /// The largest offset is 0.3 Å, so every atom stays strictly inside the box
    /// (extents 0.8–9.2, 0.7–7.3, 0.7–5.3 Å) and no coordinate needs wrapping.
    fn lattice_60() -> Array2<F> {
        let mut pts = Array2::<F>::zeros((60, 3));
        let mut idx: usize = 0;
        for ix in 0..5 {
            for iy in 0..4 {
                for iz in 0..3 {
                    let ox = 0.20 * (((idx % 3) as F) - 1.0);
                    let oy = 0.15 * (((idx % 5) as F) - 2.0);
                    let oz = 0.10 * (((idx % 7) as F) - 3.0);
                    pts[[idx, 0]] = 2.0 * (ix as F) + 1.0 + ox;
                    pts[[idx, 1]] = 2.0 * (iy as F) + 1.0 + oy;
                    pts[[idx, 2]] = 2.0 * (iz as F) + 1.0 + oz;
                    idx += 1;
                }
            }
        }
        pts
    }

    /// Pair count of [`lattice_60`] in [`small_box`] at cutoff 2.3 Å.
    ///
    /// Hard-coded golden, obtained by an independent scalar recomputation of the
    /// minimum-image convention (`d -= L·round(d/L)` per axis, legitimate here
    /// because 2.3 Å < min(10, 8, 6)/2 = 3 Å). The closest pair distance to the
    /// cutoff is 0.037 Å away from it, so no pair sits on the `d² <= r_c²`
    /// knife-edge and the count cannot flip on a last-bit difference.
    const LATTICE_60_PAIRS: usize = 163;

    /// Cutoff for [`lattice_60`]: below half the shortest box length (so the
    /// minimum image is unique) and above the 2 Å lattice pitch (so each atom
    /// has neighbors along all three axes).
    const LATTICE_60_CUTOFF: F = 2.3;

    /// Sites per axis of [`lattice_512`].
    const LATTICE_512_SIDE: usize = 8;

    /// Total sites of [`lattice_512`]: 8³.
    const LATTICE_512_N: usize = 512;

    /// Lattice pitch (Å) of [`lattice_512`] — and, at
    /// [`LATTICE_512_CUTOFF`], its only pair distance.
    const LATTICE_512_PITCH: F = 3.5;

    /// Cubic box edge (Å) of [`lattice_512`]: exactly eight pitches, so the
    /// lattice tiles the box seamlessly and every site has the same
    /// environment.
    const LATTICE_512_BOX: F = 28.0;

    /// Cutoff (Å) for [`lattice_512`]: above the 3.5 Å first shell, below the
    /// 3.5·√2 = 4.9497 Å second one, with 0.5 Å and 0.95 Å of margin — no pair
    /// sits on the `d² <= r_c²` knife-edge.
    const LATTICE_512_CUTOFF: F = 4.0;

    /// Coordination number of every site of [`lattice_512`] at
    /// [`LATTICE_512_CUTOFF`]: ±x, ±y, ±z.
    const LATTICE_512_COORD: usize = 6;

    /// Golden pair count of [`lattice_512`]: 512 sites × 6 neighbors / 2, each
    /// unordered pair counted once by the half-shell contract.
    const LATTICE_512_PAIRS: usize = 1536;

    /// Occupied cells of [`lattice_512`] at [`LATTICE_512_CUTOFF`]: 7³.
    ///
    /// `celldim = floor(28 / 4) = 7` per axis, and the eight lattice planes per
    /// axis (0.25, 3.75, … 24.75 Å) fall into cells
    /// `floor(pos / 4) = 0, 0, 1, 2, 3, 4, 5, 6` — seven distinct columns, all
    /// of them occupied.
    const LATTICE_512_OCCUPIED_CELLS: usize = 343;

    /// 512 sites on a simple-cubic 8 × 8 × 8 lattice, 3.5 Å pitch, offset
    /// 0.25 Å from the origin, inside a fully periodic 28 Å cube.
    ///
    /// The box edge is exactly eight pitches, so the lattice is seamlessly
    /// periodic: **every** site has six neighbors at exactly 3.5 Å (±x, ±y, ±z,
    /// wrapping included) and twelve at 4.9497 Å. The 0.25 Å offset keeps every
    /// coordinate off a box face and off a cell boundary; all coordinates
    /// (0.25, 3.5, 28, 24.75) are exact binary fractions, so the minimum-image
    /// arithmetic is exact and `dist_sq` comes out as exactly 3.5² = 12.25.
    ///
    /// No random number generator: the loop below is the whole fixture.
    fn lattice_512() -> Array2<F> {
        let mut pts = Array2::<F>::zeros((LATTICE_512_N, 3));
        let mut idx: usize = 0;
        for ix in 0..LATTICE_512_SIDE {
            for iy in 0..LATTICE_512_SIDE {
                for iz in 0..LATTICE_512_SIDE {
                    pts[[idx, 0]] = LATTICE_512_PITCH * (ix as F) + 0.25;
                    pts[[idx, 1]] = LATTICE_512_PITCH * (iy as F) + 0.25;
                    pts[[idx, 2]] = LATTICE_512_PITCH * (iz as F) + 0.25;
                    idx += 1;
                }
            }
        }
        pts
    }

    /// ac-003: the streamed pairs are the half-shell self list, and each one
    /// carries both physical quantities consistently.
    #[test]
    fn engine_for_each_pair_yields_full_pairs() {
        let bx = small_box();
        let pts = small_points();

        let mut nl = NeighborList::new(1.5);
        nl.build(pts.view(), &bx);
        let pairs = collect_sorted(&nl);

        assert_eq!(
            pairs.len(),
            SMALL_GOLDEN.len(),
            "expected exactly the three golden pairs, got {pairs:?}"
        );

        for (pair, &(gi, gj, gd2, gdisp)) in pairs.iter().zip(SMALL_GOLDEN.iter()) {
            assert!(
                pair.i < pair.j,
                "self stream is half-shell: expected i < j, got {} !< {}",
                pair.i,
                pair.j
            );
            assert_eq!((pair.i, pair.j), (gi, gj), "pair identity");
            assert!(
                (pair.dist_sq - gd2).abs() <= 1e-12,
                "pair ({gi},{gj}): dist_sq {} != golden {gd2}",
                pair.dist_sq
            );
            for (d, &g) in gdisp.iter().enumerate() {
                assert!(
                    (pair.disp[d] - g).abs() <= 1e-12,
                    "pair ({gi},{gj}): disp[{d}] {} != golden {g}",
                    pair.disp[d]
                );
            }
            // The two columns must come from the same periodic image.
            let norm_sq = pair.disp[0] * pair.disp[0]
                + pair.disp[1] * pair.disp[1]
                + pair.disp[2] * pair.disp[2];
            assert!(
                (pair.dist_sq - norm_sq).abs() <= 1e-12,
                "pair ({gi},{gj}): dist_sq {} != ||disp||^2 {norm_sq}",
                pair.dist_sq
            );
        }
    }

    /// ac-004: `neighbors(FULL)` is the materialization of the pair stream —
    /// the internal `push` path must not diverge from `Neighbors::from_pairs`.
    #[test]
    fn engine_neighbors_equals_from_pairs() {
        let bx = small_box();
        let pts = small_points();

        let mut nl = NeighborList::new(1.5);
        nl.build(pts.view(), &bx);

        let streamed = collect_sorted(&nl);
        let manual = Neighbors::from_pairs(
            streamed.iter().copied(),
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );
        let engine = nl.neighbors(NeighborColumns::FULL);

        assert_eq!(
            engine.mode(),
            QueryMode::SelfQuery { n_points: 4 },
            "engine table must be tagged as a self-query over all four points"
        );
        assert_eq!(engine.n_pairs(), manual.n_pairs(), "pair count");

        let got = table_rows_sorted(&engine);
        let want = table_rows_sorted(&manual);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!((g.0, g.1), (w.0, w.1), "pair indices");
            assert!(
                (g.2 - w.2).abs() <= 1e-12,
                "pair ({},{}) dist_sq: engine {} vs from_pairs {}",
                g.0,
                g.1,
                g.2,
                w.2
            );
            for d in 0..3 {
                assert!(
                    (g.3[d] - w.3[d]).abs() <= 1e-12,
                    "pair ({},{}) disp[{d}]: engine {} vs from_pairs {}",
                    g.0,
                    g.1,
                    g.3[d],
                    w.3[d]
                );
            }
        }
    }

    /// ac-006: `build_columns` is the SoA sibling of `build` — same engine, same
    /// semantics, column input.
    ///
    /// A column store (`atoms.x` / `atoms.y` / `atoms.z`) holds coordinates as
    /// three independent columns, never as an N × 3 matrix. Without this entry
    /// point such a caller has to transpose into an `Array2` — or, as RDF does
    /// today, bypass the engine entirely and reach for a backend. So the two
    /// paths are held to *identity*, not merely to agreement: same pairs, same
    /// numbers, same query tagging.
    ///
    /// Both paths evaluate the same minimum-image arithmetic on the same
    /// coordinates, so bitwise equality is the expectation; the 1e-15 escape
    /// admits a last-bit reassociation only — three orders of magnitude tighter
    /// than the 1e-12 the golden comparisons above use, and far too tight for
    /// any difference that could move a pair across the cutoff.
    #[test]
    fn engine_build_columns_matches_build() {
        let bx = small_box();
        let pts = small_points();

        // The very same four coordinates, split into three columns.
        let xs: Vec<F> = pts.column(0).to_vec();
        let ys: Vec<F> = pts.column(1).to_vec();
        let zs: Vec<F> = pts.column(2).to_vec();

        let mut aos = NeighborList::new(1.5);
        aos.build(pts.view(), &bx);

        let mut soa = NeighborList::new(1.5);
        soa.build_columns(&xs, &ys, &zs, &bx);

        let want = collect_sorted(&aos);
        let got = collect_sorted(&soa);

        assert_eq!(
            want.len(),
            SMALL_GOLDEN.len(),
            "the AoS reference must still produce the three golden pairs, got {want:?}"
        );
        assert_eq!(
            got.len(),
            want.len(),
            "build_columns emitted {} pairs, build emitted {}",
            got.len(),
            want.len()
        );

        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(
                (g.i, g.j),
                (w.i, w.j),
                "pair identity diverges: columns ({},{}) vs points ({},{})",
                g.i,
                g.j,
                w.i,
                w.j
            );
            assert!(
                g.dist_sq == w.dist_sq || (g.dist_sq - w.dist_sq).abs() <= 1e-15,
                "pair ({},{}): dist_sq columns {} vs points {}",
                g.i,
                g.j,
                g.dist_sq,
                w.dist_sq
            );
            for d in 0..3 {
                assert!(
                    g.disp[d] == w.disp[d] || (g.disp[d] - w.disp[d]).abs() <= 1e-15,
                    "pair ({},{}): disp[{d}] columns {} vs points {}",
                    g.i,
                    g.j,
                    g.disp[d],
                    w.disp[d]
                );
            }
        }

        // A column-fed build is still a self-query over every point handed in:
        // the SoA path must not lose the point count the table is tagged with.
        assert_eq!(
            soa.neighbors(NeighborColumns::FULL).mode(),
            QueryMode::SelfQuery { n_points: 4 },
            "a table built from columns must be tagged as a self-query over all \
             four points"
        );
    }

    /// ac-005: the cell-list backend and the O(N²) reference backend produce the
    /// same unordered pair multiset with the same distances.
    ///
    /// The comparison keeps duplicates (a `Vec`, not a set) so that a backend
    /// emitting a pair twice fails here instead of being silently deduplicated.
    #[test]
    fn engine_linkcell_vs_bruteforce_multiset() {
        let bx = small_box();
        let pts = lattice_60();

        let mut lc = NeighborList::new(LATTICE_60_CUTOFF);
        lc.build(pts.view(), &bx);
        let lc_pairs = collect_sorted(&lc);

        let mut bf = NeighborList::brute_force(LATTICE_60_CUTOFF);
        bf.build(pts.view(), &bx);
        let bf_pairs = collect_sorted(&bf);

        assert_eq!(
            bf_pairs.len(),
            LATTICE_60_PAIRS,
            "reference backend disagrees with the hard-coded golden pair count"
        );
        assert_eq!(
            lc_pairs.len(),
            bf_pairs.len(),
            "cell-list emitted {} pairs, reference emitted {}",
            lc_pairs.len(),
            bf_pairs.len()
        );

        for (a, b) in lc_pairs.iter().zip(bf_pairs.iter()) {
            assert_eq!(
                (a.i, a.j),
                (b.i, b.j),
                "pair sets diverge: cell-list ({},{}) vs reference ({},{})",
                a.i,
                a.j,
                b.i,
                b.j
            );
            assert!(
                (a.dist_sq - b.dist_sq).abs() <= 1e-12,
                "pair ({},{}): cell-list dist_sq {} vs reference {}",
                a.i,
                a.j,
                a.dist_sq,
                b.dist_sq
            );
        }
    }

    /// The **parallel** materialization of the cell-list backend reproduces the
    /// O(N²) reference, pair for pair, and honors the column policy through the
    /// merge.
    ///
    /// [`NeighborList::neighbors`] folds over occupied cells with rayon above 64
    /// of them and concatenates the workers' partial tables; below that it falls
    /// back to the serial visitor walk. Every other fixture in this module is a
    /// handful of cells, so without this test the parallel branch — the fold,
    /// the `Neighbors::append` merge, and the column alignment that merge has to
    /// preserve — is never executed at all.
    ///
    /// The fixture ([`lattice_512`]) is sized to cross that threshold with room
    /// to spare, and the premise is *asserted* rather than asserted-in-prose:
    /// the occupied-cell count is recomputed here from the public [`CellGrid`]
    /// API, so a later change to the cutoff or the box that dropped the fixture
    /// back onto the serial path fails here instead of silently testing nothing.
    ///
    /// # Goldens
    ///
    /// | quantity | value | derivation |
    /// |---|---|---|
    /// | pairs | 1536 | 512 sites × 6 neighbors / 2 |
    /// | `dist_sq` | 12.25 Å² | (3.5 Å)², exact in binary |
    /// | coordination | 6 | ±x, ±y, ±z at one pitch |
    /// | occupied cells | 343 | 7³, from `floor(28 / 4) = 7` per axis |
    ///
    /// Without the `rayon` feature the same assertions run against the serial
    /// fallback, which is the point: both branches owe the identical pair set.
    #[test]
    fn engine_parallel_materialize_matches_brute_force() {
        #[cfg(feature = "rayon")]
        crate::core::test_thread_pool::ensure();
        #[cfg(feature = "rayon")]
        assert!(
            rayon::current_num_threads() >= 2,
            "parallel materialization must run on a multithreaded pool \
             (set MOLRS_TEST_THREADS>=2)"
        );
        use std::collections::BTreeSet;

        /// Index pairs of a table, sorted — row order is not part of the
        /// contract, least of all under a parallel fold.
        fn sorted_index_pairs(nb: &Neighbors) -> Vec<(u32, u32)> {
            let mut pairs: Vec<(u32, u32)> = (0..nb.n_pairs())
                .map(|k| (nb.query_point_indices()[k], nb.point_indices()[k]))
                .collect();
            pairs.sort_unstable();
            pairs
        }

        let bx = SimBox::cube(LATTICE_512_BOX, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("valid cubic box");
        let pts = lattice_512();

        // Premise: the fixture must actually reach the parallel branch.
        let grid = CellGrid::for_cutoff(&bx, LATTICE_512_CUTOFF);
        assert_eq!(
            grid.celldim(),
            [7, 7, 7],
            "cell partition changed shape: floor(28 / 4) = 7 per axis"
        );
        let occupied: BTreeSet<usize> = (0..pts.nrows())
            .map(|i| grid.cell_of(&bx, [pts[[i, 0]], pts[[i, 1]], pts[[i, 2]]]))
            .collect();
        assert_eq!(
            occupied.len(),
            LATTICE_512_OCCUPIED_CELLS,
            "occupied-cell count changed"
        );
        assert!(
            occupied.len() >= 64,
            "fixture must stay above the 64-occupied-cell threshold at which the \
             cell-list backend switches to the rayon fold; got {}",
            occupied.len()
        );

        let mut nl = NeighborList::new(LATTICE_512_CUTOFF);
        nl.build(pts.view(), &bx);
        let full = nl.neighbors(NeighborColumns::FULL);

        let mut bf = NeighborList::brute_force(LATTICE_512_CUTOFF);
        bf.build(pts.view(), &bx);
        let oracle = bf.neighbors(NeighborColumns::FULL);

        assert_eq!(
            full.mode(),
            QueryMode::SelfQuery {
                n_points: LATTICE_512_N
            },
            "a parallel materialization must still be tagged with the point count"
        );
        assert_eq!(
            oracle.n_pairs(),
            LATTICE_512_PAIRS,
            "the O(N²) reference disagrees with the hard-coded golden pair count"
        );
        assert_eq!(
            full.n_pairs(),
            LATTICE_512_PAIRS,
            "the parallel fold emitted {} pairs, golden is {LATTICE_512_PAIRS}",
            full.n_pairs()
        );
        assert_eq!(
            sorted_index_pairs(&full),
            sorted_index_pairs(&oracle),
            "parallel pair set diverges from the reference backend"
        );

        // Every pair: half-shell, both columns from the same periodic image, and
        // the one distance this lattice can produce.
        let d2 = full.dist_sq().expect("FULL storage materializes dist_sq");
        let disp = full.disp().expect("FULL storage materializes disp");
        let golden_d2 = LATTICE_512_PITCH * LATTICE_512_PITCH;
        let mut coordination = vec![0usize; LATTICE_512_N];
        for k in 0..full.n_pairs() {
            let (i, j) = (full.query_point_indices()[k], full.point_indices()[k]);
            assert!(
                i < j,
                "a parallel worker emitted a non-half-shell pair: {i} !< {j}"
            );
            coordination[i as usize] += 1;
            coordination[j as usize] += 1;

            let norm_sq = disp[[k, 0]] * disp[[k, 0]]
                + disp[[k, 1]] * disp[[k, 1]]
                + disp[[k, 2]] * disp[[k, 2]];
            assert!(
                (d2[k] - norm_sq).abs() <= 1e-12,
                "pair ({i},{j}): dist_sq {} != ||disp||^2 {norm_sq} — the merge \
                 misaligned the columns",
                d2[k]
            );
            assert!(
                (d2[k] - golden_d2).abs() <= 1e-10,
                "pair ({i},{j}): dist_sq {} != golden {golden_d2}",
                d2[k]
            );
        }
        for (site, &n) in coordination.iter().enumerate() {
            assert_eq!(
                n, LATTICE_512_COORD,
                "site {site} has {n} neighbors, every site of this lattice has \
                 {LATTICE_512_COORD}"
            );
        }

        // The column policy survives the parallel merge: a column the caller
        // did not ask for is absent, never a fabricated zero, and the pair set
        // is unchanged.
        let dist_only = nl.neighbors(NeighborColumns::DIST_SQ);
        assert_eq!(dist_only.n_pairs(), LATTICE_512_PAIRS);
        assert_eq!(
            dist_only
                .dist_sq()
                .expect("DIST_SQ keeps the distance column")
                .len(),
            LATTICE_512_PAIRS,
            "the dist_sq column lost alignment with the index columns"
        );
        assert!(
            dist_only.disp().is_none(),
            "DIST_SQ must not fabricate a disp column in the parallel path"
        );
        assert_eq!(
            sorted_index_pairs(&dist_only),
            sorted_index_pairs(&full),
            "column policy selects columns, not pairs"
        );

        let indices_only = nl.neighbors(NeighborColumns::INDICES_ONLY);
        assert_eq!(indices_only.n_pairs(), LATTICE_512_PAIRS);
        assert!(
            indices_only.dist_sq().is_none(),
            "INDICES_ONLY must not fabricate a dist_sq column in the parallel path"
        );
        assert!(
            indices_only.disp().is_none(),
            "INDICES_ONLY must not fabricate a disp column in the parallel path"
        );
        assert_eq!(
            sorted_index_pairs(&indices_only),
            sorted_index_pairs(&full),
            "column policy selects columns, not pairs"
        );
    }

    /// ac-009 (second half): `update` re-indexes the new coordinates; the pair
    /// stream follows them in both directions.
    #[test]
    fn engine_update_follows_new_coordinates() {
        let bx = SimBox::ortho(
            array![20.0, 20.0, 20.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("valid orthorhombic box");

        // Two atoms 1.0 Å apart — inside the 1.5 Å cutoff.
        let near = array![[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]];
        // Same two atoms 5.0 Å apart; the shortest periodic image is 5.0 Å
        // (the wrapped one is 15.0 Å), so the pair is gone.
        let far = array![[1.0, 1.0, 1.0], [6.0, 1.0, 1.0]];

        let mut nl = NeighborList::new(1.5);
        nl.build(near.view(), &bx);
        assert_eq!(
            collect_sorted(&nl).len(),
            1,
            "built configuration: one pair"
        );

        nl.update(far.view());
        assert_eq!(
            collect_sorted(&nl).len(),
            0,
            "after moving the atoms apart the pair must disappear"
        );

        nl.update(near.view());
        let back = collect_sorted(&nl);
        assert_eq!(back.len(), 1, "after moving them back the pair must return");
        assert_eq!((back[0].i, back[0].j), (0, 1));
        assert!(
            (back[0].dist_sq - 1.0).abs() <= 1e-12,
            "dist_sq {} != 1.0",
            back[0].dist_sq
        );
    }

    /// ac-009 (first half): updating an engine that was never built is a
    /// programming error and must fail loudly, naming `build`.
    #[test]
    #[should_panic(expected = "build")]
    fn engine_update_before_build_panics() {
        let pts = small_points();
        let mut nl = NeighborList::new(1.5);
        nl.update(pts.view());
    }

    /// ac-002 / ac-004: the column policy changes the columns, never the pairs.
    #[test]
    fn engine_neighbors_lean_storage() {
        let bx = small_box();
        let pts = small_points();

        let mut nl = NeighborList::new(1.5);
        nl.build(pts.view(), &bx);

        let lean = nl.neighbors(NeighborColumns::INDICES_ONLY);
        assert!(
            lean.dist_sq().is_none(),
            "INDICES_ONLY must not fabricate a dist_sq column"
        );
        assert!(
            lean.disp().is_none(),
            "INDICES_ONLY must not fabricate a disp column"
        );

        let full = nl.neighbors(NeighborColumns::FULL);
        assert_eq!(
            lean.n_pairs(),
            full.n_pairs(),
            "column policy selects columns, not pairs"
        );
        assert_eq!(lean.n_pairs(), SMALL_GOLDEN.len());
        assert_eq!(lean.query_point_indices(), full.query_point_indices());
        assert_eq!(lean.point_indices(), full.point_indices());
    }

    /// ac-007: a cross-query is reachable through the public surface, and it
    /// is directed — no `i < j` is imposed.
    ///
    /// `NeighborQuery::query` is the path for cross searches; `NeighborList`
    /// owns the self path. The fixture is built so
    /// that one pair has `i > j`, which a half-shell contract would forbid.
    ///
    /// Box: 20 Å cube, fully periodic, cutoff 1.0 Å.
    ///
    /// | query point | nearest reference | separation | pair    |
    /// |-------------|-------------------|------------|---------|
    /// | q0 (9.2,9,9)| r2 (9,9,9)        | 0.2 Å      | (0, 2)  |
    /// | q1 (5.1,5,5)| r1 (5,5,5)        | 0.1 Å      | (1, 1)  |
    /// | q2 (1.1,1,1)| r0 (1,1,1)        | 0.1 Å      | (2, 0)  |
    ///
    /// Every other query-reference separation is at least 6.9 Å, so exactly
    /// three pairs survive the cutoff.
    #[test]
    fn engine_cross_query_reachable() {
        let bx = SimBox::ortho(
            array![20.0, 20.0, 20.0],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .expect("valid orthorhombic box");
        let refs = array![[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let query = array![[9.2, 9.0, 9.0], [5.1, 5.0, 5.0], [1.1, 1.0, 1.0]];

        let nq = NeighborQuery::new(&bx, refs.view(), 1.0);
        let nb = nq.query(query.view());

        assert_eq!(
            nb.mode(),
            QueryMode::CrossQuery {
                n_query_points: 3,
                n_points: 3,
            },
            "a cross-query must stay tagged as one"
        );
        assert_eq!(nb.n_query_points(), 3);
        assert_eq!(nb.n_points(), 3);
        assert_eq!(nb.n_pairs(), 3);

        let mut got: Vec<(u32, u32)> = (0..nb.n_pairs())
            .map(|k| (nb.query_point_indices()[k], nb.point_indices()[k]))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(0u32, 2u32), (1, 1), (2, 0)]);
        assert!(
            got.iter().any(|&(i, j)| i > j),
            "fixture must contain a pair with i > j to prove no half-shell \
             constraint is imposed on a cross-query"
        );
    }
}

#[cfg(test)]
mod geometry_tests {
    //! A table remembers the box it was searched in, and a materialized table
    //! and a built [`NeighborList`] answer the same [`PairSource`] questions.
    use super::test_fixtures::{SMALL_GOLDEN, small_box, small_points};
    use super::*;
    use crate::core::filter_rad;

    fn same_box(a: &SimBox, b: &SimBox) -> bool {
        a.matrix() == b.matrix() && a.volume() == b.volume()
    }

    /// One visited pair: `(i, j, dist_sq, disp)`.
    type Row = (u32, u32, Option<F>, Option<[F; 3]>);

    /// Visit a source and return its rows sorted by `(i, j)`.
    fn visited(source: &dyn PairSource) -> Vec<Row> {
        let mut rows = Vec::new();
        source.visit_pairs(&mut |i, j, d2, dr| rows.push((i, j, d2, dr)));
        rows.sort_by_key(|r| (r.0, r.1));
        rows
    }

    #[test]
    fn neighbor_list_table_carries_the_build_box() {
        let bx = small_box();
        let mut nl = NeighborList::new(1.5);
        nl.build(small_points().view(), &bx);

        let nb = nl.neighbors(NeighborColumns::FULL);

        assert!(same_box(
            nb.simbox().expect("a searched table has a box"),
            &bx
        ));
    }

    #[test]
    fn query_tables_carry_the_query_box() {
        let bx = small_box();
        let pts = small_points();
        let q = NeighborQuery::new(&bx, pts.view(), 1.5);

        let self_table = q.query_self();
        let cross_table = q.query(pts.view());

        assert!(same_box(self_table.simbox().expect("self query"), &bx));
        assert!(same_box(cross_table.simbox().expect("cross query"), &bx));
    }

    #[test]
    fn hand_built_table_has_no_box_until_given_one() {
        let pairs = SMALL_GOLDEN.map(|(i, j, dist_sq, disp)| NeighborPair {
            i,
            j,
            dist_sq,
            disp,
        });
        let nb = Neighbors::from_pairs(
            pairs,
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );
        assert!(nb.simbox().is_none());

        let bx = small_box();
        let boxed = nb.with_simbox(bx.clone());
        assert!(same_box(boxed.simbox().expect("box was attached"), &bx));
    }

    #[test]
    fn repack_and_filters_keep_the_box() {
        let bx = small_box();
        let mut nl = NeighborList::new(1.5);
        nl.build(small_points().view(), &bx);
        let nb = nl.neighbors(NeighborColumns::FULL);

        assert!(same_box(
            nb.repack(NeighborColumns::DIST_SQ).simbox().unwrap(),
            &bx
        ));
        assert!(same_box(filter_rad(&nb, 1.0).simbox().unwrap(), &bx));
    }

    /// The streamed list and its materialized table are one pair set: same
    /// rows, same physical values, same mode and box.
    #[test]
    fn table_and_list_are_the_same_pair_source() {
        let bx = small_box();
        let mut nl = NeighborList::new(1.5);
        nl.build(small_points().view(), &bx);
        let nb = nl.neighbors(NeighborColumns::FULL);

        let from_list = visited(&nl);
        let from_table = visited(&nb);

        assert_eq!(from_list.len(), SMALL_GOLDEN.len());
        for (row, (gi, gj, gd2, gdr)) in from_list.iter().zip(SMALL_GOLDEN) {
            assert_eq!((row.0, row.1), (gi, gj));
            assert!((row.2.unwrap() - gd2).abs() <= 1e-12);
            for (got, want) in row.3.unwrap().iter().zip(gdr) {
                assert!((got - want).abs() <= 1e-12);
            }
        }
        assert_eq!(from_list, from_table);
        assert_eq!(PairSource::mode(&nl), PairSource::mode(&nb));
        assert_eq!(PairSource::columns(&nl), NeighborColumns::FULL);
        assert!(same_box(
            PairSource::simbox(&nl).unwrap(),
            PairSource::simbox(&nb).unwrap()
        ));
    }

    /// A lean table reports its absent columns as absent: `None`, never zero.
    #[test]
    fn lean_table_visits_absent_columns_as_none() {
        let bx = small_box();
        let mut nl = NeighborList::new(1.5);
        nl.build(small_points().view(), &bx);
        let lean = nl.neighbors(NeighborColumns::INDICES_ONLY);

        assert_eq!(PairSource::columns(&lean), NeighborColumns::INDICES_ONLY);
        let rows = visited(&lean);
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.2.is_none() && r.3.is_none()));
    }
}
