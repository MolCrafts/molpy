//! Every atom's path-score histogram, computed over the block-cut tree instead
//! of by walking every simple path.
//!
//! # Why the walk is not needed
//!
//! The classes compare *multisets* of path scores, so a histogram
//! `score bits → number of paths` carries exactly the same information as the
//! sorted score array. What makes the walk exponential is that a molecule with
//! `k` rings in series has `~2^k` simple paths; what makes a histogram cheap is
//! two facts about those paths:
//!
//! 1. **A simple path decomposes over the biconnected blocks.** Blocks meet only
//!    at cut atoms, and the block-cut graph is a tree, so a path that leaves a
//!    block through a cut atom can never come back. Inside each block it is a
//!    simple path of that block alone, and the set of blocks it can still enter
//!    depends only on the cut atom it is standing on — not on how it got there.
//! 2. **A path's future score depends on its past only through a small state.**
//!    `scorepath()` accumulates left to right, so what a prefix contributes to
//!    every extension is its running sum (as exact `f64` bits) and its length —
//!    plus, at `-eq 2`, the last six atoms, which the E/Z windows still need.
//!
//! So the paths arriving at a cut atom are merged into a histogram over that
//! state, and each block is walked once per entry state set instead of once per
//! path. Across blocks the cost is the number of distinct states, which is
//! bounded by the distinct (length, running-sum) pairs rather than by the number
//! of paths.
//!
//! 3. **At `-eq 1`, a path acts on a state only through its atomic numbers.** So
//!    the simple paths *inside* a block out of an entry atom are enumerated once
//!    per molecule and merged by Z sequence into a [`SeqTrie`] with path counts as
//!    weights; a fused carbon ring system has thousands of in-block paths but
//!    little more distinct sequences than lengths. (`-eq 2` reads atom identities
//!    — the torsion windows — so it walks the in-block paths themselves.) The
//!    in-block enumeration remains exponential in the size of one fused ring
//!    system, as the whole walk used to be in the size of the molecule.
//!
//! # Exactness
//!
//! Nothing is approximated. Every state transition performs the same `f64`
//! operations, in the same order, as the path walk in `equatom.c::scorepath()`
//! (and as the reference walk kept in this module's tests), and states are merged
//! only when bit-identical. Path counts are exact (see [`PathCount`]).

use std::collections::HashMap;

use super::counts::{Acc, Count, PathCount, accumulate};
use super::{CIS_COEF, ELEMENT_WEIGHT, Flat, MAX_CON, POSITION_WEIGHT, TRANS_COEF, Torsion};

/// "No atom" in a state's tail and "no block" for a self-loop slot.
const NONE: u32 = u32::MAX;

/// How many trailing atoms an `-eq 2` state remembers.
///
/// Position `p`'s E/Z coefficient reads the windows `p-3..=p` and `p..=p+3`, so
/// it is final once the path is three atoms past it; finalizing it then needs the
/// seven atoms `p-3..=p+3`, i.e. the six already on the path plus the new one.
const TAIL: usize = 6;

/// An `-eq 1` prefix state: the bits of its running sum and its length. Every
/// term is final the moment it is added, so nothing else matters.
type Eager = (u64, u32);

/// What a path prefix passes on to its extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct State {
    /// Bits of the running sum of the *finalized* positions' terms.
    sum: u64,
    /// Path length in atoms.
    len: u32,
    /// `-eq 2` only: the last [`TAIL`] atoms, oldest first, [`NONE`] before the
    /// path start. All [`NONE`] at `-eq 1`, where every term is final at once.
    tail: [u32; TAIL],
}

impl State {
    /// The empty path.
    const EMPTY: Self = Self {
        sum: 0,
        len: 0,
        tail: [NONE; TAIL],
    };
}

/// A histogram of path prefixes over their states.
type Hist<S = State, C = PathCount> = Vec<(S, C)>;

/// Per cut atom reached in a block, the prefixes arriving there, merged.
type Exits<S, C> = Vec<(usize, Acc<S, C>)>;

/// The exit accumulator for `cut`, created on first arrival.
fn exit_for<S, C>(exits: &mut Exits<S, C>, cut: usize) -> &mut Acc<S, C> {
    let at = match exits.iter().position(|(c, _)| *c == cut) {
        Some(at) => at,
        None => {
            exits.push((cut, Acc::default()));
            exits.len() - 1
        }
    };
    &mut exits[at].1
}

/// One atom's path-score histogram: `(score bits, count)`, sorted by bits.
pub(super) type ScoreHistogram = Vec<(u64, PathCount)>;

/// The biconnected blocks, as the walk needs them.
struct Blocks {
    /// Per atom, per adjacency slot: the block of that bond ([`NONE`] for a
    /// self-loop, which no walk can take).
    slot_block: Vec<Vec<u32>>,
    /// Per atom: every block it belongs to (more than one iff it is a cut atom).
    blocks_of: Vec<Vec<u32>>,
}

impl Blocks {
    /// Tarjan's biconnected components over bonds, iteratively (no recursion,
    /// so a long chain cannot overflow the stack).
    fn of(flat: &Flat) -> Self {
        let n = flat.adj.len();
        let mut disc = vec![NONE; n];
        let mut low = vec![0u32; n];
        let mut bond_block = vec![NONE; flat.bonds.len()];
        let mut n_blocks = 0u32;
        let mut clock = 0u32;
        let mut edges: Vec<usize> = Vec::new();
        // (atom, bond it was entered by, next adjacency slot to look at)
        let mut frames: Vec<(usize, usize, usize)> = Vec::new();

        for root in 0..n {
            if disc[root] != NONE {
                continue;
            }
            disc[root] = clock;
            low[root] = clock;
            clock += 1;
            frames.push((root, usize::MAX, 0));
            while let Some(frame) = frames.last_mut() {
                let (u, via, slot) = *frame;
                if slot < flat.adj[u].len() {
                    frame.2 += 1;
                    let (v, bond) = (flat.adj[u][slot], flat.adj_bond[u][slot]);
                    if v == u || bond == via {
                        continue;
                    }
                    if disc[v] == NONE {
                        edges.push(bond);
                        disc[v] = clock;
                        low[v] = clock;
                        clock += 1;
                        frames.push((v, bond, 0));
                    } else if disc[v] < disc[u] {
                        edges.push(bond);
                        low[u] = low[u].min(disc[v]);
                    }
                    continue;
                }
                frames.pop();
                let Some(&(parent, _, _)) = frames.last() else {
                    continue;
                };
                low[parent] = low[parent].min(low[u]);
                if low[u] >= disc[parent] {
                    // `parent` separates `u`'s subtree: everything stacked since
                    // the tree bond into `u` is one block.
                    while let Some(bond) = edges.pop() {
                        bond_block[bond] = n_blocks;
                        if bond == via {
                            break;
                        }
                    }
                    n_blocks += 1;
                }
            }
        }

        let slot_block: Vec<Vec<u32>> = flat
            .adj_bond
            .iter()
            .zip(&flat.adj)
            .enumerate()
            .map(|(u, (bonds, nbrs))| {
                bonds
                    .iter()
                    .zip(nbrs)
                    .map(|(&b, &v)| if v == u { NONE } else { bond_block[b] })
                    .collect()
            })
            .collect();
        let blocks_of = slot_block
            .iter()
            .map(|slots| {
                let mut own: Vec<u32> = slots.iter().copied().filter(|&b| b != NONE).collect();
                own.sort_unstable();
                own.dedup();
                own
            })
            .collect();
        Self {
            slot_block,
            blocks_of,
        }
    }
}

/// Computes path-score histograms for the atoms of one molecule.
pub(super) struct PathHistograms<'a> {
    flat: &'a Flat,
    /// `-eq 2` torsions keyed by atom window (both directions), first torsion in
    /// `geom` order winning, exactly as a linear `find` over `geom` would.
    torsions: HashMap<[u32; 4], f64>,
    /// `-eq 2` is on: terms are finalized lazily, three atoms late.
    lazy: bool,
    max_path_length: Option<usize>,
    blocks: Blocks,
    /// `-eq 1` only: per `(block, entry atom)`, the block's in-block paths out of
    /// that atom merged by their atomic-number sequence (see [`SeqTrie`]).
    tries: HashMap<(u32, usize), SeqTrie>,
}

/// The simple paths inside one block out of one entry atom, merged by the
/// sequence of atomic numbers they visit.
///
/// At `-eq 1` a path changes a prefix's state through its atomic numbers alone —
/// each step adds `(j + 1)·0.11 + Z·0.08` — so two in-block paths with the same
/// Z sequence carry every prefix to bit-identical states. Walking the trie of
/// distinct sequences, with path counts as weights, does the same work as walking
/// every path; in a fused carbon ring system the distinct sequences are little
/// more than the distinct lengths, against thousands of paths.
#[derive(Debug, Default)]
struct SeqTrie {
    /// Node 0 is the entry atom itself (already on the path); every other node
    /// is one more atom.
    nodes: Vec<SeqNode>,
}

/// One distinct Z sequence of [`SeqTrie`].
#[derive(Debug, Default)]
struct SeqNode {
    /// Atomic number of the atom this node appends.
    z: u8,
    /// Child nodes (one per distinct next Z).
    children: Vec<u32>,
    /// How many in-block paths with this sequence end at an atom that
    /// terminates paths (fewer than [`MAX_CON`] neighbours), i.e. are scored.
    scored: u64,
    /// How many end at each cut atom, from which they continue into the cut
    /// atom's other blocks.
    exits: Vec<(usize, u64)>,
}

impl SeqTrie {
    /// Enumerate the in-block paths out of `entry` — by the walk's own rules:
    /// the first [`MAX_CON`] neighbours only, no atom twice — into a trie.
    fn build(flat: &Flat, blocks: &Blocks, block: u32, entry: usize) -> Self {
        let mut trie = Self {
            nodes: vec![SeqNode::default()],
        };
        let mut on_path = vec![false; flat.adj.len()];
        on_path[entry] = true;
        trie.grow(flat, blocks, block, entry, 0, &mut on_path);
        trie
    }

    fn grow(
        &mut self,
        flat: &Flat,
        blocks: &Blocks,
        block: u32,
        atom: usize,
        node: usize,
        on_path: &mut [bool],
    ) {
        let slots = flat.adj[atom]
            .iter()
            .zip(&blocks.slot_block[atom])
            .take(MAX_CON);
        for (&next, &slot_block) in slots {
            if slot_block != block || on_path[next] {
                continue;
            }
            let z = flat.z[next];
            let child = match self.nodes[node]
                .children
                .iter()
                .find(|&&c| self.nodes[c as usize].z == z)
            {
                Some(&c) => c as usize,
                None => {
                    self.nodes.push(SeqNode {
                        z,
                        ..SeqNode::default()
                    });
                    let c = self.nodes.len() - 1;
                    self.nodes[node].children.push(c as u32);
                    c
                }
            };
            let here = &mut self.nodes[child];
            if flat.adj[next].len() < MAX_CON {
                here.scored += 1;
            }
            if blocks.blocks_of[next].len() > 1 {
                match here.exits.iter_mut().find(|(cut, _)| *cut == next) {
                    Some((_, n)) => *n += 1,
                    None => here.exits.push((next, 1)),
                }
            }
            on_path[next] = true;
            self.grow(flat, blocks, block, next, child, on_path);
            on_path[next] = false;
        }
    }
}

impl<'a> PathHistograms<'a> {
    /// Prepare the walk: block decomposition and torsion lookup.
    pub(super) fn new(flat: &'a Flat, geom: &[Torsion], max_path_length: Option<usize>) -> Self {
        let mut torsions = HashMap::new();
        for t in geom {
            let coef = if t.trans { TRANS_COEF } else { CIS_COEF };
            let fwd = t.atoms.map(|a| a as u32);
            let mut rev = fwd;
            rev.reverse();
            torsions.entry(fwd).or_insert(coef);
            torsions.entry(rev).or_insert(coef);
        }
        let lazy = !geom.is_empty();
        let blocks = Blocks::of(flat);
        // `-eq 2` reads atom identities (the torsion windows), which a Z
        // sequence forgets: it walks the paths themselves.
        let tries = if lazy {
            HashMap::new()
        } else {
            blocks
                .blocks_of
                .iter()
                .enumerate()
                .flat_map(|(atom, own)| own.iter().map(move |&b| (b, atom)))
                .map(|(b, atom)| ((b, atom), SeqTrie::build(flat, &blocks, b, atom)))
                .collect()
        };
        Self {
            flat,
            lazy,
            torsions,
            max_path_length,
            blocks,
            tries,
        }
    }

    /// The histogram of every simple path starting at `root`, as the path walk
    /// would score it: sorted by score bits, one entry per distinct score.
    pub(super) fn histogram(&self, root: usize) -> ScoreHistogram {
        if self.lazy {
            return self.lazy_histogram(root);
        }
        // Counts fit a u128 unless one score is reached by more than 2^128
        // paths; only then is the root redone with unbounded counts.
        match self.eager_histogram::<u128>(root) {
            Some(h) => h
                .into_iter()
                .map(|(b, c)| (b, c.into_path_count()))
                .collect(),
            None => self
                .eager_histogram::<PathCount>(root)
                .expect("unbounded counts cannot overflow"),
        }
    }

    /// [`Self::histogram`] at `-eq 1`: blocks walked through their [`SeqTrie`]s.
    fn eager_histogram<C: Count>(&self, root: usize) -> Option<Vec<(u64, C)>> {
        let Some(start) = self.step_eager((0, 0), self.flat.z[root]) else {
            return Some(Vec::new()); // `-pl 0`: not even the one-atom path is scored.
        };
        self.drive(root, start, start.0, |block, entry, hist, exits, out| {
            self.walk_trie(&self.tries[&(block, entry)], 0, hist, exits, out)
        })
    }

    /// [`Self::histogram`] at `-eq 2`: blocks walked path by path.
    fn lazy_histogram(&self, root: usize) -> ScoreHistogram {
        let Some(start) = self.step(&State::EMPTY, root) else {
            return Vec::new();
        };
        let mut on_path = vec![false; self.flat.adj.len()];
        let score = self.finish(&start).to_bits();
        self.drive(root, start, score, |block, entry, hist, exits, out| {
            on_path[entry] = true;
            self.walk_block(block, entry, hist, &mut on_path, exits, out);
            on_path[entry] = false;
            Some(())
        })
        .expect("unbounded counts cannot overflow")
    }

    /// Walk the block-cut tree out of `root`: every block is entered once per
    /// cut atom with the merged histogram of the prefixes arriving there, and
    /// `walk` extends them through it. `None` when a count overflows.
    fn drive<S: Copy + Eq + std::hash::Hash, C: Count>(
        &self,
        root: usize,
        start: S,
        start_score: u64,
        mut walk: impl FnMut(u32, usize, &[(S, C)], &mut Exits<S, C>, &mut Acc<u64, C>) -> Option<()>,
    ) -> Option<Vec<(u64, C)>> {
        let mut out: Acc<u64, C> = Acc::default();
        if self.flat.adj[root].len() < MAX_CON {
            out.insert(start_score, C::one());
        }
        let hist: Hist<S, C> = vec![(start, C::one())];
        let mut work: Vec<(u32, usize, Hist<S, C>)> = self.blocks.blocks_of[root]
            .iter()
            .map(|&b| (b, root, hist.clone()))
            .collect();
        while let Some((block, entry, hist)) = work.pop() {
            let mut exits: Exits<S, C> = Vec::new();
            walk(block, entry, &hist, &mut exits, &mut out)?;
            for (cut, arrived) in exits {
                let hist: Hist<S, C> = arrived.into_iter().collect();
                for &next in &self.blocks.blocks_of[cut] {
                    if next != block {
                        work.push((next, cut, hist.clone()));
                    }
                }
            }
        }
        let mut out: Vec<(u64, C)> = out.into_iter().collect();
        out.sort_unstable_by_key(|(bits, _)| *bits);
        Some(out)
    }

    /// Enumerate the simple paths inside `block` that extend the prefixes in
    /// `hist` (all ending at `atom`), emitting each one's score and collecting,
    /// per cut atom reached, the prefixes that may continue into its other blocks.
    fn walk_block(
        &self,
        block: u32,
        atom: usize,
        hist: &[(State, PathCount)],
        on_path: &mut [bool],
        exits: &mut Exits<State, PathCount>,
        out: &mut Acc<u64, PathCount>,
    ) {
        // `con[]` holds six neighbours: the walk never leaves through a seventh.
        let slots = self.flat.adj[atom]
            .iter()
            .zip(&self.blocks.slot_block[atom])
            .take(MAX_CON);
        for (&next, &slot_block) in slots {
            if slot_block != block || on_path[next] {
                continue;
            }
            let extended = self.advance(hist, next);
            if extended.is_empty() {
                continue; // every prefix hit the `-pl` cap
            }
            self.emit(next, &extended, out);
            if self.blocks.blocks_of[next].len() > 1 {
                let acc = exit_for(exits, next);
                for (state, count) in &extended {
                    accumulate(acc, *state, count.clone());
                }
            }
            on_path[next] = true;
            self.walk_block(block, next, &extended, on_path, exits, out);
            on_path[next] = false;
        }
    }

    /// [`Self::walk_block`] at `-eq 1`: the same paths, walked once per distinct
    /// Z sequence and weighted by how many paths share it.
    ///
    /// Nothing is merged on the way down: an `-eq 1` step maps distinct states to
    /// distinct states (apart from a rare rounding collision, which the
    /// accumulators at the block exits and of the scores absorb).
    fn walk_trie<C: Count>(
        &self,
        trie: &SeqTrie,
        node: usize,
        hist: &[(Eager, C)],
        exits: &mut Exits<Eager, C>,
        out: &mut Acc<u64, C>,
    ) -> Option<()> {
        for &child in &trie.nodes[node].children {
            let here = &trie.nodes[child as usize];
            let extended: Hist<Eager, C> = hist
                .iter()
                .filter_map(|(state, count)| {
                    self.step_eager(*state, here.z).map(|s| (s, count.clone()))
                })
                .collect();
            if extended.is_empty() {
                continue; // every prefix hit the `-pl` cap
            }
            if here.scored > 0 {
                for (state, count) in &extended {
                    accumulate(out, state.0, count.scaled_checked(here.scored)?)?;
                }
            }
            for &(cut, paths) in &here.exits {
                let acc = exit_for(exits, cut);
                for (state, count) in &extended {
                    accumulate(acc, *state, count.scaled_checked(paths)?)?;
                }
            }
            self.walk_trie(trie, child as usize, &extended, exits, out)?;
        }
        Some(())
    }

    /// Every prefix in `hist`, extended by `atom`; merged.
    fn advance(&self, hist: &[(State, PathCount)], atom: usize) -> Hist {
        let mut next: Hist = hist
            .iter()
            .filter_map(|(state, count)| self.step(state, atom).map(|s| (s, count.clone())))
            .collect();
        merge(&mut next).expect("unbounded counts cannot overflow");
        next
    }

    /// Score the paths ending at `atom` — unless it has [`MAX_CON`] neighbours,
    /// which terminate no path in `scorepath()`.
    fn emit(&self, atom: usize, hist: &[(State, PathCount)], out: &mut Acc<u64, PathCount>) {
        if self.flat.adj[atom].len() >= MAX_CON {
            return;
        }
        for (state, count) in hist {
            accumulate(out, self.finish(state).to_bits(), count.clone());
        }
    }

    /// Append `atom` to an `-eq 2` prefix, or `None` when that would exceed `-pl`.
    fn step(&self, state: &State, atom: usize) -> Option<State> {
        let len = state.len as usize + 1;
        if self.max_path_length.is_some_and(|max| len > max) {
            return None;
        }
        let mut sum = f64::from_bits(state.sum);
        // The seven atoms at positions len-7 ..= len-1.
        let mut window = [NONE; TAIL + 1];
        window[..TAIL].copy_from_slice(&state.tail);
        window[TAIL] = atom as u32;
        if len >= 4 {
            // Position len-4 now has both of its windows; finalize it.
            let p = len - 4;
            let mut coef = 1.0;
            if p >= 3
                && let Some(&c) = self.torsions.get(&quad(&window[0..4]))
            {
                coef = c;
            }
            if let Some(&c) = self.torsions.get(&quad(&window[3..7])) {
                coef = c;
            }
            sum += self.term(p, window[3] as usize, coef);
        }
        let mut tail = [NONE; TAIL];
        tail.copy_from_slice(&window[1..]);
        Some(State {
            sum: sum.to_bits(),
            len: len as u32,
            tail,
        })
    }

    /// [`Self::step`] at `-eq 1`, where only the appended atom's Z matters:
    /// every term is final the moment it is added.
    fn step_eager(&self, (sum, len): Eager, z: u8) -> Option<Eager> {
        let len = len as usize + 1;
        if self.max_path_length.is_some_and(|max| len > max) {
            return None;
        }
        let sum = f64::from_bits(sum) + term(len - 1, z, 1.0);
        Some((sum.to_bits(), len as u32))
    }

    /// The score of the path an `-eq 2` state ends: its finalized sum plus the
    /// last (up to) three positions, whose forward window runs off the end.
    fn finish(&self, state: &State) -> f64 {
        let mut sum = f64::from_bits(state.sum);
        let len = state.len as usize;
        for p in len.saturating_sub(3)..len {
            let at = p + TAIL - len; // index of position p in the tail
            let mut coef = 1.0;
            if p >= 3
                && let Some(&c) = self.torsions.get(&quad(&state.tail[at - 3..=at]))
            {
                coef = c;
            }
            sum += self.term(p, state.tail[at] as usize, coef);
        }
        sum
    }

    /// The term of one path position, for the atom at it.
    fn term(&self, position: usize, atom: usize, coef: f64) -> f64 {
        term(position, self.flat.z[atom], coef)
    }
}

/// The term of one path position — the same expression, evaluated in the same
/// order, as `scorepath()`.
fn term(position: usize, z: u8, coef: f64) -> f64 {
    let position = (position + 1) as f64 * POSITION_WEIGHT;
    let element = f64::from(z) * ELEMENT_WEIGHT;
    coef * (position + element)
}

/// A four-atom window as a lookup key.
fn quad(window: &[u32]) -> [u32; 4] {
    [window[0], window[1], window[2], window[3]]
}

/// Sort a histogram by key and add up the counts of equal keys; `None` when a
/// sum overflows.
fn merge<K: Copy + Ord, C: Count>(hist: &mut Vec<(K, C)>) -> Option<()> {
    hist.sort_unstable_by_key(|(key, _)| *key);
    let mut overflow = false;
    hist.dedup_by(|(key, count), (kept_key, kept)| {
        if key == kept_key {
            overflow |= kept.add_checked(count).is_none();
            true
        } else {
            false
        }
    });
    (!overflow).then_some(())
}
