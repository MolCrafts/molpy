//! Pair styles: each built-in pair style's energy as a [`ScalarForm`] of
//! the distance, priced by [`ScalarPair`] at both compile doors; the
//! one-type kernel a neighbour loop feeds ([`PairLjCut`]); the 1-4
//! exceptions kernel ([`PairExceptions`]); and what every pair kernel
//! shares — the `pairs` block's columns, the type-pair index, the
//! thread-count-independent fold.
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`ScalarPair`]: crate::ff::potential::form_kernel::ScalarPair

use molrs::core::schema::block_names::{ATOMS, PAIRS};

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

/// The `pairs` block's `atomi`, `atomj` and (optional) `is_14` columns — the
/// list every compiled pair kernel reads, parsed once. `who` names the kernel
/// in the error.
///
/// # Errors
///
/// A frame without a `pairs` block, or a block without a u64 `atomi` /
/// `atomj` column.
pub(crate) type PairColumns<'f> = (
    &'f ndarray::ArrayD<u64>,
    &'f ndarray::ArrayD<u64>,
    Option<&'f ndarray::ArrayD<bool>>,
);

/// See [`PairColumns`].
pub(crate) fn pair_columns<'f>(frame: &'f Frame, who: &str) -> Result<PairColumns<'f>, String> {
    let block = frame
        .get(PAIRS)
        .ok_or_else(|| format!("{who}: frame missing \"pairs\" block"))?;
    let column = |key: &str| {
        block
            .get(key)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| format!("{who}: pairs block missing \"{key}\" column"))
    };
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());
    Ok((column("atomi")?, column("atomj")?, is_14))
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

/// Split a per-pair fold into fixed chunks, run in parallel when that is
/// worth it, with a result that does not depend on the thread count.
///
/// `run(acc, rows)` folds one contiguous range of pairs into `acc`.
///
/// # Why the chunk *size* is fixed, and why the serial path chunks too
///
/// Floating-point addition is not associative, so the answer depends on how
/// the pairs are grouped. Grouping by "one chunk per thread" makes it depend
/// on the thread count — which is what the first version of this did, and the
/// test caught it between one thread and two. The grouping here is a fixed
/// number of pairs per chunk, decided by nothing but `n_pairs`, and the
/// partials are merged in chunk order. The serial path walks the same chunks
/// and merges them the same way; it has to, or running on one thread would
/// give a different number from running on four.
///
/// A scatter needs one accumulator per chunk, and below the threshold those
/// cost more to allocate and merge than the fold costs to run — so a small
/// table stays serial and touches no pool.
pub(crate) fn fold_chunks<R>(out: &mut [F], n_pairs: usize, run: R) -> (F, molrs::core::Virial)
where
    R: Fn(&mut [F], std::ops::Range<usize>) -> (F, molrs::core::Virial) + Sync,
{
    use molrs::core::Virial;
    /// Pairs per chunk. Fixed, because it decides the grouping of a
    /// floating-point sum and so is part of the answer.
    const CHUNK: usize = 4_096;
    // Splitting costs one accumulator per chunk, and merging one costs
    // `out.len()` adds — a number that has nothing to do with how many pairs
    // there are, so a table can be big enough to want splitting and still lose
    // to the merge. Both are properties of the configuration and not of the
    // machine, so the grouping stays the same however many threads run.
    let split = n_pairs >= CHUNK && n_pairs >= 3 * out.len();
    let nchunks = if split { n_pairs.div_ceil(CHUNK) } else { 1 };
    let bounds = |c: usize| (c * CHUNK)..((c + 1) * CHUNK).min(n_pairs);

    // One chunk is the whole fold, and both paths run it straight into `out`.
    // Small tables therefore pay nothing at all for any of this.
    if nchunks == 1 {
        return run(out, 0..n_pairs);
    }

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        if nchunks > 1 && rayon::current_num_threads() > 1 {
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
            for (local, e, w) in &parts {
                energy += e;
                for k in 0..6 {
                    virial.components[k] += w.components[k];
                }
                for (dst, v) in out.iter_mut().zip(local) {
                    *dst += v;
                }
            }
            return (energy, virial);
        }
    }

    // Serial, over the same chunks and merged the same way — including the
    // per-chunk buffer, which is not an optimisation here but a requirement:
    // a chunk that accumulated straight into `out` would add its terms to
    // whatever was already there, where a parallel chunk sums them from zero
    // first. Those are different sums, and one thread would disagree with four.
    let mut local = vec![0.0; out.len()];
    let mut energy = 0.0;
    let mut virial = Virial::ZERO;
    for c in 0..nchunks {
        local.fill(0.0);
        let (e, w) = run(&mut local, bounds(c));
        energy += e;
        for k in 0..6 {
            virial.components[k] += w.components[k];
        }
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
    use molrs::core::{NeighborColumns, NeighborPair, Neighbors, QueryMode};
    use molrs::op::F;

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
