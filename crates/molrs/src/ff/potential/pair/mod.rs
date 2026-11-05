//! Pair styles: each built-in pair style's energy as a [`ScalarForm`] of
//! the distance, priced by [`ScalarPair`] at both compile doors; the
//! one-type kernel a neighbour loop feeds ([`PairLjCut`]); the 1-4
//! exceptions kernel ([`PairExceptions`]); and what every pair kernel
//! shares — the `pairs` block's columns, the type-pair index, the
//! thread-count-independent fold.
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`ScalarPair`]: crate::ff::potential::form_kernel::ScalarPair

use crate::ff::ir::SpecialClass;
use molrs::core::schema::block_names::ATOMS;

use crate::ff::potential::Accumulator;
use molrs::core::Frame;
use molrs::core::Virial;
use molrs::op::F;

/// Pair kernel: already-reduced geometry in, energy and the force on `j` out.
///
/// One primitive, [`pair_energy_force`](PairPotential::pair_energy_force):
/// the energy and the force of a pair come out of one expression, so asking
/// for either alone is asking for both and dropping one.
pub trait PairPotential: Send + Sync {
    /// Energy and force on `j` of the pair at squared distance `r2` with
    /// displacement `disp` (`x_j - x_i`), or `None` when it is outside the
    /// cutoff or degenerate.
    fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])>;
}

/// Run a per-pair fold into `acc`, tallying the virial only when `acc` asks
/// for one.
///
/// A kernel hands its row fold twice, monomorphised on whether it tallies:
/// `plain` with the tally compiled out, `tallied` with it in. Which one runs
/// is decided here, once per call, so the per-pair loop never branches on a
/// channel. The rows are grouped by [`fold_chunks`] either way.
pub(crate) fn fold_pairs<P, T>(acc: &mut Accumulator<'_>, n_pairs: usize, plain: P, tallied: T)
where
    P: Fn(&mut [F], std::ops::Range<usize>) -> (F, Virial) + Sync,
    T: Fn(&mut [F], std::ops::Range<usize>) -> (F, Virial) + Sync,
{
    let (out, virial) = acc.forces_with_virial();
    let energy = match virial {
        Some(total) => {
            let (e, w) = fold_chunks(out, n_pairs, tallied);
            *total += w;
            e
        }
        None => fold_chunks(out, n_pairs, plain).0,
    };
    acc.add_energy(energy);
}

/// The pairs a compiled pair kernel prices, each with its weight in the
/// van-der-Waals and in the electrostatic styles (0: not priced).
///
/// The compiler builds it from the atoms and the frame's special pairs
/// ([`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile));
/// [`ExplicitTerms`](crate::ff::compile::ExplicitTerms) from the pairs it was
/// handed, at full strength.
#[derive(Clone, Debug, Default)]
pub(crate) struct CandidatePairs {
    pub(crate) i: Vec<usize>,
    pub(crate) j: Vec<usize>,
    pub(crate) lj: Vec<F>,
    pub(crate) coul: Vec<F>,
}

impl CandidatePairs {
    /// The weights the styles of `class` take.
    pub(crate) fn weights(&self, class: SpecialClass) -> &[F] {
        match class {
            SpecialClass::Vdw => &self.lj,
            SpecialClass::Coulomb => &self.coul,
        }
    }

    pub(crate) fn push(&mut self, i: usize, j: usize, lj: F, coul: F) {
        self.i.push(i);
        self.j.push(j);
        self.lj.push(lj);
        self.coul.push(coul);
    }

    pub(crate) fn len(&self) -> usize {
        self.i.len()
    }
}

/// Map each atom to a dense type index, and hand back the labels in that order.
///
/// A neighbour-driven kernel keys its parameters on the atoms, so it needs the
/// types as small integers it can index a table with — not as the strings the
/// frame carries. The labels come back so the caller can look each type's
/// parameters up once, rather than once per atom.
pub(crate) fn atom_type_index(frame: &Frame) -> Result<(Vec<u32>, Vec<String>), String> {
    let atoms = frame
        .get(ATOMS)
        .ok_or_else(|| "typed pair kernel: frame missing \"atoms\" block".to_string())?;
    let types = atoms
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "typed pair kernel: atoms block missing \"type\" column".to_string())?;
    let mut labels: Vec<String> = Vec::new();
    let mut index = std::collections::HashMap::new();
    let mut type_id = Vec::with_capacity(types.len());
    for t in types.iter() {
        let next = labels.len() as u32;
        let id = *index.entry(t.clone()).or_insert_with(|| {
            labels.push(t.clone());
            next
        });
        type_id.push(id);
    }
    Ok((type_id, labels))
}

/// Most groups a fold is split into: each holds a force buffer of its own.
const MAX_CHUNKS: usize = 64;
/// Fewest pairs a group is worth: below it the buffer costs more than the
/// rows it folds.
const MIN_CHUNK: usize = 4_096;
/// Most bytes the groups' force buffers hold together.
const BUFFER_BUDGET: usize = 256 << 20;

/// How many groups a fold of `n_pairs` rows into `out_len` force components
/// is split into — decided by nothing but those two numbers, because the
/// grouping of a floating-point sum is part of its answer.
///
/// One group when the table is small, or when merging a buffer (`out_len`
/// adds) would cost about what folding its rows does; otherwise as many
/// groups of at least [`MIN_CHUNK`] rows as fit [`MAX_CHUNKS`] and
/// [`BUFFER_BUDGET`].
fn chunk_count(n_pairs: usize, out_len: usize) -> usize {
    if n_pairs < 2 * MIN_CHUNK || n_pairs < 3 * out_len {
        return 1;
    }
    let by_memory = BUFFER_BUDGET / (out_len.max(1) * std::mem::size_of::<F>());
    (n_pairs / MIN_CHUNK).min(MAX_CHUNKS).min(by_memory).max(1)
}

/// Split a per-pair fold into a fixed set of groups, run in parallel when
/// that is worth it, with a result that does not depend on the thread count.
///
/// `run(acc, rows)` folds one contiguous range of pairs into `acc`.
///
/// # Why the grouping is fixed, and why the serial path groups too
///
/// Floating-point addition is not associative, so the answer depends on how
/// the pairs are grouped. Grouping by "one chunk per thread" makes it depend
/// on the thread count — which is what the first version of this did, and the
/// test caught it between one thread and two. The grouping here is decided
/// by nothing but the table and buffer sizes ([`chunk_count`]), each group
/// folds into a zeroed buffer of its own, and every force component sums the
/// groups' partials in group order. The serial path walks the same groups
/// and merges them the same way; it has to, or running on one thread would
/// give a different number from running on four.
///
/// The group count is bounded ([`MAX_CHUNKS`], [`BUFFER_BUDGET`]), so the
/// buffers held at once are too: a group per few thousand pairs would hold
/// one `3·N` buffer per few thousand pairs — gigabytes for a large system.
pub(crate) fn fold_chunks<R>(out: &mut [F], n_pairs: usize, run: R) -> (F, molrs::core::Virial)
where
    R: Fn(&mut [F], std::ops::Range<usize>) -> (F, molrs::core::Virial) + Sync,
{
    use molrs::core::Virial;
    let nchunks = chunk_count(n_pairs, out.len());
    // One group is the whole fold, and both paths run it straight into
    // `out`. Small tables therefore pay nothing at all for any of this.
    if nchunks == 1 {
        return run(out, 0..n_pairs);
    }
    let len = n_pairs.div_ceil(nchunks);
    let bounds = |c: usize| (c * len).min(n_pairs)..((c + 1) * len).min(n_pairs);

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if rayon::current_num_threads() > 1 {
            let parts: Vec<(Vec<F>, F, Virial)> = (0..nchunks)
                .into_par_iter()
                .map(|c| {
                    let mut local = vec![0.0; out.len()];
                    let (e, w) = run(&mut local, bounds(c));
                    (local, e, w)
                })
                .collect();
            let mut energy = 0.0;
            let mut virial = Virial::ZERO;
            for (_, e, w) in &parts {
                energy += e;
                virial += *w;
            }
            // Each component sums the partials in group order, whichever
            // thread takes its block.
            const BLOCK: usize = 4_096;
            out.par_chunks_mut(BLOCK).enumerate().for_each(|(b, dst)| {
                let at = b * BLOCK;
                let n = dst.len();
                for (local, _, _) in &parts {
                    for (d, v) in dst.iter_mut().zip(&local[at..at + n]) {
                        *d += v;
                    }
                }
            });
            return (energy, virial);
        }
    }

    // Serial, over the same groups and merged the same way — including the
    // per-group buffer, which is not an optimisation here but a requirement:
    // a group that accumulated straight into `out` would add its terms to
    // whatever was already there, where a parallel group sums them from zero
    // first. Those are different sums, and one thread would disagree with four.
    let mut local = vec![0.0; out.len()];
    let mut energy = 0.0;
    let mut virial = Virial::ZERO;
    for c in 0..nchunks {
        local.fill(0.0);
        let (e, w) = run(&mut local, bounds(c));
        energy += e;
        virial += w;
        for (dst, v) in out.iter_mut().zip(&local) {
            *dst += v;
        }
    }
    (energy, virial)
}

/// Index into a type-pair parameter table laid out `ti * ntypes + tj`.
///
/// This is LAMMPS's `pair_coeff i j` model, and it is what a neighbour-driven
/// evaluation needs: which pairs exist is re-decided at every rebuild, so a
/// pair's parameters have to be findable from the atoms it names rather than
/// from a row of a previous list.
#[inline]
pub(crate) fn type_pair(ti: u32, tj: u32, ntypes: usize) -> usize {
    ti as usize * ntypes + tj as usize
}

pub(crate) mod buck;
pub(crate) mod charmm;
pub(crate) mod coul_cut;
pub(crate) mod coul_long;
pub(crate) mod exceptions;
pub(crate) mod lj_class2;
pub(crate) mod lj_cut;
pub(crate) mod lj_rows;
pub(crate) mod mmff;
pub(crate) mod morse;
pub(crate) mod tang_toennies;
pub(crate) mod thole;
pub(crate) mod uff;

pub use buck::Buck;
pub use charmm::{CoulCharmm, LjCharmm};
pub use coul_cut::CoulCut;
pub use coul_long::CoulLong;
pub use exceptions::PairExceptions;
pub use lj_class2::LjClass2;
pub use lj_cut::{LjCut, PairLjCut};
pub use lj_rows::lj_ab_to_sigma_epsilon;
pub use mmff::MmffVdw;
pub use morse::Morse;
pub use tang_toennies::TangToennies;
pub use thole::Thole;
pub use uff::UffLj;

#[cfg(test)]
pub(crate) mod fixtures {
    use molrs::core::{Block, NeighborColumns, NeighborPair, Neighbors, QueryMode};
    use molrs::op::{F, Idx};
    use ndarray::Array1;

    /// The `pairs` block under which the pair styles price exactly `links`
    /// of `n_atoms` atoms: each link flagged in `is_14` (empty: none) as a
    /// 1-4 pair (`n_bonds` 3, weighed by `special_bonds`), every other link
    /// at full strength, and every pair not linked weighed 0.
    pub(crate) fn pricing_only(n_atoms: usize, links: &[(usize, usize)], is_14: &[bool]) -> Block {
        let listed = |i: usize, j: usize| {
            links
                .iter()
                .position(|&(a, b)| (a.min(b), a.max(b)) == (i, j))
        };
        let (mut ai, mut aj, mut hops, mut excluded) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n_atoms {
            for j in (i + 1)..n_atoms {
                let row = match listed(i, j) {
                    Some(k) if is_14.get(k) == Some(&true) => Some((3, false)),
                    Some(_) => None,
                    None => Some((0, true)),
                };
                if let Some((d, gone)) = row {
                    ai.push(i as Idx);
                    aj.push(j as Idx);
                    hops.push(d as Idx);
                    excluded.push(gone);
                }
            }
        }
        let mut block = Block::new();
        for (key, col) in [("atomi", ai), ("atomj", aj), ("n_bonds", hops)] {
            block
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        if excluded.iter().any(|&e| e) {
            for key in ["lj_scale", "coul_scale"] {
                block
                    .insert(
                        key,
                        Array1::from_vec(vec![0.0; excluded.len()]).into_dyn(),
                        Some(excluded.clone()),
                    )
                    .unwrap();
            }
        }
        block
    }

    /// A neighbour table over exactly `links`, with the displacements a
    /// neighbour engine would have computed for them.
    ///
    /// Every typed kernel is checked against its compiled twin on the same
    /// pairs, and "the same pairs" has to mean the same arithmetic too: the
    /// compiled path takes plain coordinate differences, so the table must
    /// carry those and not a re-derived approximation of them.
    pub(crate) fn table_over(coords: &[F], links: &[(usize, usize)]) -> Neighbors {
        let n_points = coords.len() / 3;
        let pairs: Vec<NeighborPair> = links
            .iter()
            .map(|&(i, j)| {
                let d = [
                    coords[j * 3] - coords[i * 3],
                    coords[j * 3 + 1] - coords[i * 3 + 1],
                    coords[j * 3 + 2] - coords[i * 3 + 2],
                ];
                NeighborPair {
                    i: i as u32,
                    j: j as u32,
                    dist_sq: d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                    disp: d,
                }
            })
            .collect();
        Neighbors::from_pairs(
            pairs,
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points },
        )
    }

    /// A kernel's own virial must equal `Σ_a f_a ⊗ x_a` over the coordinates
    /// it was handed — whenever the pair displacements *are* coordinate
    /// differences, which on a free boundary they are.
    ///
    /// Under periodic boundaries the two differ, and that difference is the
    /// whole reason a kernel tallies its own: the atom-weighted sum depends on
    /// where the cell's origin falls once a pair reaches through a face. On a
    /// free-boundary configuration there is no such pair, so they must agree —
    /// which makes this the cheapest possible check that a kernel got its sign
    /// convention right, and it is one a kernel accumulating forces the other
    /// way round (UFF does) is easy to fail.
    pub(crate) fn assert_virial_matches_forces(
        label: &str,
        coords: &[F],
        out: (F, Vec<F>, Option<molrs::core::Virial>),
    ) {
        let (_, forces, virial) = out;
        let virial = virial.unwrap_or_else(|| panic!("{label}: this kernel must tally a virial"));
        let mut from_forces = molrs::core::Virial::ZERO;
        for a in 0..coords.len() / 3 {
            from_forces.add_outer(
                [forces[a * 3], forces[a * 3 + 1], forces[a * 3 + 2]],
                [coords[a * 3], coords[a * 3 + 1], coords[a * 3 + 2]],
            );
        }
        let scale = from_forces
            .components
            .iter()
            .fold(1.0_f64, |m, c| m.max(c.abs()));
        assert!(
            scale > 1e-6,
            "{label}: the virial must be non-trivial for this to assert anything"
        );
        for c in 0..6 {
            assert!(
                (virial.components[c] - from_forces.components[c]).abs() / scale < 1e-12,
                "{label}: component {c}: kernel says {}, the forces say {}",
                virial.components[c],
                from_forces.components[c]
            );
        }
    }

    /// Assert two evaluations agree bit for bit, and that they said something.
    pub(crate) fn assert_same(label: &str, a: (F, Vec<F>), b: (F, Vec<F>)) {
        let (e_a, f_a) = a;
        let (e_b, f_b) = b;
        assert!(
            e_a.abs() > 1e-9,
            "{label}: the configuration must interact for this to assert anything; got {e_a}"
        );
        assert_eq!(
            e_a.to_bits(),
            e_b.to_bits(),
            "{label}: energy {e_a} vs {e_b}"
        );
        assert_eq!(f_a.len(), f_b.len(), "{label}: force length");
        for (c, (x, y)) in f_a.iter().zip(&f_b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: force component {c}: {x} vs {y}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fold whose rows each add `1/(k+1)` to every force component: the
    /// sum depends on how the rows are grouped, so grouping that moved with
    /// the thread count would show.
    fn fold_counting(n_pairs: usize, n_out: usize, groups: &AtomicUsize) -> (F, Vec<F>) {
        let mut out = vec![0.0; n_out];
        let (e, _) = fold_chunks(&mut out, n_pairs, |local, rows| {
            groups.fetch_add(1, Ordering::Relaxed);
            let mut e = 0.0;
            for k in rows {
                let v = 1.0 / (k as F + 1.0);
                e += v;
                for f in local.iter_mut() {
                    *f += v;
                }
            }
            (e, Virial::ZERO)
        });
        (e, out)
    }

    /// A table of a million pairs is folded in at most 64 groups — one
    /// force buffer each — however many threads run, and the answer is the
    /// same bit for bit on one thread and on several.
    #[test]
    fn a_large_table_folds_in_a_bounded_number_of_groups() {
        let n_pairs = 1_000_000;
        let run = |threads: usize| {
            let groups = AtomicUsize::new(0);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (e, out) = pool.install(|| fold_counting(n_pairs, 6, &groups));
            (e, out, groups.into_inner())
        };
        let (e1, f1, g1) = run(1);
        assert!(g1 > 1, "the table is split");
        assert!(g1 <= 64, "{g1} groups, each with its own force buffer");
        for threads in [2, 4, 8] {
            let (e, f, g) = run(threads);
            assert_eq!(g, g1, "{threads} threads: the grouping");
            assert_eq!(e.to_bits(), e1.to_bits(), "{threads} threads: energy");
            for (a, b) in f.iter().zip(&f1) {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: forces");
            }
        }
    }

    /// The groups' force buffers together stay within a fixed budget: a
    /// system of a million atoms gets fewer groups than a small one.
    #[test]
    fn the_group_buffers_stay_within_their_budget() {
        for n_out in [3, 3_000, 300_000, 3_000_000, 30_000_000] {
            let n = chunk_count(100_000_000, n_out);
            assert!((1..=64).contains(&n), "{n_out}: {n} groups");
            assert!(
                n == 1 || n * n_out * std::mem::size_of::<F>() <= 256 << 20,
                "{n_out}: {n} buffers of {n_out}"
            );
        }
    }
}
