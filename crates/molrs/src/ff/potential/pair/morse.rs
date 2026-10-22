//! Morse pair potential (LAMMPS `pair_style morse`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::type_pair;
use crate::ff::potential::pair::{fold_pairs, fold_pairs_untallied};
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

/// Where a pair's Morse `(D₀, α, r₀)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
    },
    /// A type-pair table, keyed by the types of the two atoms.
    ///
    /// The row label a compiled kernel keys on belongs to a pair list that a
    /// neighbour engine rebuilds from scratch; the atoms' types survive it.
    /// This is LAMMPS's `pair_coeff i j` model. Under a ghost régime `type_id`
    /// covers the copies too, each carrying its owner's type.
    Typed {
        type_id: Vec<u32>,
        ntypes: usize,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// Morse pair potential (LAMMPS `pair_style morse`):
/// E = d0 * ((1 - exp(-alpha*(r-r0)))^2 - 1) = d0 [e^{-2α(r-r0)} - 2e^{-α(r-r0)}]
///
/// Morse non-bonded form (note the `-1` offset vs the Morse bond, so the well
/// minimum is `-d0` at `r = r0`). Parameters per pair type, as LAMMPS names
/// them: `d0` (LAMMPS `D0`), `alpha`, `r0`. The compiled and the
/// neighbour-driven constructors read the same keys, so one field prices
/// under both.
pub struct PairMorse {
    source: Source,
    /// `cutoff²` (`r < cutoff`, as LAMMPS), at both compile doors; infinite
    /// for a style that states no cutoff.
    cutoff2: F,
}

impl PairMorse {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(d0.len(), n);
        assert_eq!(alpha.len(), n);
        assert_eq!(r0.len(), n);
        Self {
            cutoff2: F::INFINITY,
            source: Source::Compiled {
                atom_i,
                atom_j,
                d0,
                alpha,
                r0,
            },
        }
    }

    /// A kernel that finds its parameters from the types of the two atoms.
    ///
    /// `type_id` is one type index per atom; each table is `ntypes × ntypes`
    /// laid out `ti * ntypes + tj`. This is the form a neighbour-driven
    /// evaluation needs.
    ///
    /// A neighbour table is a different list of pairs every rebuild, so a
    /// parameter resolved against an older one belongs to different atoms.
    /// Keyed on the atoms instead, it can be found for whatever pair turns up
    /// — including one that names a periodic copy.
    pub fn typed(type_id: Vec<u32>, ntypes: usize, d0: Vec<F>, alpha: Vec<F>, r0: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, d0.len(), "d0 must cover every type pair");
        assert_eq!(n_cells, alpha.len(), "alpha must cover every type pair");
        assert_eq!(n_cells, r0.len(), "r0 must cover every type pair");
        debug_assert!(
            type_id.iter().all(|&t| (t as usize) < ntypes),
            "an atom has a type with no parameters"
        );
        let n_owned = type_id.len();
        Self {
            cutoff2: F::INFINITY,
            source: Source::Typed {
                type_id,
                ntypes,
                d0,
                alpha,
                r0,
                n_owned,
            },
        }
    }

    /// Price only pairs closer than `cutoff`: the style's, as LAMMPS
    /// truncates, at both compile doors.
    pub fn with_cutoff(mut self, cutoff: F) -> Self {
        self.cutoff2 = cutoff * cutoff;
        self
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], d0: F, alpha: F, r0: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let y = (-alpha * (r - r0)).exp();
        let one_my = 1.0 - y;
        let energy = d0 * (one_my * one_my - 1.0);

        // d/dr[(1-y)^2 - 1] = 2 alpha y (1 - y)  -> dE/dr = 2 D0 alpha y (1-y)
        let dedr = 2.0 * d0 * alpha * y * one_my;
        let factor = -dedr / r;
        Some((
            energy,
            [factor * disp[0], factor * disp[1], factor * disp[2]],
        ))
    }

    /// The accumulation, once, into `acc`, scaling each pair. Only where the
    /// pairs and the parameters come from differs between the two entry
    /// points; `tally` is whether this source can report a virial (a
    /// compiled list cannot).
    fn fold_into(
        &self,
        acc: &mut Accumulator<'_>,
        factor: &[F],
        n_pairs: usize,
        tally: bool,
        pair: impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F) + Sync,
    ) {
        let plain = |out: &mut [F], rows| self.fold_rows::<false>(out, factor, rows, &pair);
        if tally {
            fold_pairs(acc, n_pairs, plain, |out: &mut [F], rows| {
                self.fold_rows::<true>(out, factor, rows, &pair)
            });
        } else {
            fold_pairs_untallied(acc, n_pairs, plain);
        }
    }

    /// One contiguous range of pairs, into `out`. The whole fold when it runs
    /// serially; one chunk of it when it does not.
    fn fold_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        rows: std::ops::Range<usize>,
        pair: &(impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F) + Sync),
    ) -> (F, Virial) {
        let mut energy: F = 0.0;
        let mut virial = Virial::ZERO;
        for idx in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[idx] };
            // Exactly zero *skips*: a bonded pair sits at bond length,
            // where a repulsive term is enormous, and scaling it by zero
            // would be arithmetic on a number that should never have been
            // computed.
            if w == 0.0 {
                continue;
            }
            let (i, j, (d0, alpha, r0), disp, r2) = pair(idx);
            if r2 >= self.cutoff2 {
                continue;
            }
            let Some((e, f)) = self.pair_kernel(r2, disp, d0, alpha, r0) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            if VIRIAL {
                virial.add_outer(f, disp);
            }
            out[j * 3] += f[0];
            out[j * 3 + 1] += f[1];
            out[j * 3 + 2] += f[2];
            out[i * 3] -= f[0];
            out[i * 3 + 1] -= f[1];
            out[i * 3 + 2] -= f[2];
        }
        (energy, virial)
    }
}

impl Potential for PairMorse {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            d0,
            alpha,
            r0,
        } = &self.source
        else {
            // A type table needs a pair table, and nobody handed one over.
            return;
        };
        self.fold_into(acc, &[], atom_i.len(), false, |idx| {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (i, j, (d0[idx], alpha[idx], r0[idx]), d, r2)
        });
    }
}

impl PairDriven for PairMorse {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        let Source::Typed {
            type_id,
            ntypes,
            d0,
            alpha,
            r0,
            ..
        } = &self.source
        else {
            // A compiled kernel answers for its own list, not for this one.
            // A compiled kernel cannot read the table, so it cannot read a
            // per-pair weight either. Both providers refuse one, so this is
            // the free-boundary path and `factor` is empty.
            debug_assert!(factor.is_empty());
            self.evaluate(coords, acc);
            return;
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            // No displacements, nothing priced — and no virial claimed for it.
            acc.forces_without_virial();
            return;
        };
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        self.fold_into(acc, factor, i_col.len(), true, |p| {
            let i = i_col[p] as usize;
            let j = j_col[p] as usize;
            debug_assert!(
                i < type_id.len() && j < type_id.len(),
                "a pair names an atom the type table does not cover"
            );
            let t = type_pair(type_id[i], type_id[j], *ntypes);
            (
                i,
                j,
                (d0[t], alpha[t], r0[t]),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::Typed {
            type_id, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(type_id, *n_owned, owner);
    }
}

/// Construct a [`PairMorse`] from style params, type params, and Frame topology.
///
/// A pair's row is found from its two atoms' types — the self row, else the
/// cross row (`morse` does not mix: neither is [`IrError::NoMixing`]) — as
/// `pair_morse_typed_constructor` finds it and LAMMPS's `pair_coeff i j` states it.
///
/// [`IrError::NoMixing`]: crate::ff::ir::IrError::NoMixing
pub fn pair_morse_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    // `PotentialCompiler::compile` projects the force field's `special_bonds` 1-4
    // weight here. The energy is linear in this parameter, so scaling it is
    // exactly scaling the pair.
    let scale_14 = style_params
        .get("lj14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;

    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairMorse: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairMorse: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairMorse: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());
    let atom_types = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "PairMorse: atoms block missing \"type\" column".to_string())?;

    let (mut ai, mut aj) = (Vec::new(), Vec::new());
    let (mut dv, mut av, mut rv) = (Vec::new(), Vec::new(), Vec::new());
    let need = |p: &Params, key: &str, label: &str| param_reads::type_num("morse", label, p, key);
    for idx in 0..i_col.len() {
        let (ta, tb) = (
            atom_types[i_col[idx] as usize].as_str(),
            atom_types[j_col[idx] as usize].as_str(),
        );
        let p = param_reads::unmixed_row("morse", "d0", &type_map, ta, tb)?;
        let key = pair_key(ta, tb)?;
        let label = &key;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        dv.push(if is_14.is_some_and(|b| b[idx]) {
            need(p, "d0", label)? * scale_14
        } else {
            need(p, "d0", label)?
        });
        av.push(need(p, "alpha", label)?);
        rv.push(need(p, "r0", label)?);
    }

    // The style's `cutoff` (`r < cutoff`, as LAMMPS truncates every pair,
    // 1-4 ones included; ∞ when it states none).
    let cutoff = param_reads::pair_cutoff("morse", style_params)?;
    Ok(ForceTerm::pair(
        PairMorse::new(ai, aj, dv, av, rv).with_cutoff(cutoff),
    ))
}

/// Construct a neighbour-driven [`PairMorse`] from per-atom parameters.
///
/// The counterpart of [`pair_morse_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_morse_typed_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut d0 = vec![0.0 as F; ntypes * ntypes];
    let mut alpha = vec![0.0 as F; ntypes * ntypes];
    let mut r0 = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            // Keyed in either order alike; a self-pair by the atom type alone.
            let (ta, tb) = (labels[ti].as_str(), labels[tj].as_str());
            let p = param_reads::unmixed_row("morse", "d0", &type_map, ta, tb)?;
            let key = pair_key(ta, tb)?;
            let t = type_pair(ti as u32, tj as u32, ntypes);
            d0[t] = param_reads::type_num("morse", &key, p, "d0")?;
            alpha[t] = param_reads::type_num("morse", &key, p, "alpha")?;
            r0[t] = param_reads::type_num("morse", &key, p, "r0")?;
        }
    }
    let kernel = PairMorse::typed(type_id, ntypes, d0, alpha, r0);
    // The style's `cutoff` (`r < cutoff`, as LAMMPS): finite, for a
    // neighbour sum is not finite without one.
    let cutoff = param_reads::neighbour_cutoff("morse", style_params)?;
    Ok(ForceTerm::pair(kernel.with_cutoff(cutoff)))
}

#[cfg(test)]
mod tests {

    /// A type-pair table gives the same number as a per-row label resolved
    /// earlier against a fixed list — bit for bit, on the same pairs.
    ///
    /// The row label belongs to a pair list a neighbour engine rebuilds from
    /// scratch; the atoms' types survive it. That is the whole difference.
    #[test]
    fn a_type_table_scores_a_pair_exactly_as_compiled_rows() {
        use crate::ff::potential::pair::fixtures::{
            assert_same, assert_virial_matches_forces, table_over,
        };

        let ntypes = 2_usize;
        let type_id = vec![0_u32, 1, 0, 1];
        let d0: Vec<F> = vec![0.25, 0.18, 0.18, 0.31];
        let alpha: Vec<F> = vec![1.9, 2.1, 2.1, 1.7];
        let r0: Vec<F> = vec![3.4, 3.1, 3.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let d0_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| d0[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let alpha_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| alpha[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let r0_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| r0[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairMorse::new(ai, aj, d0_c, alpha_c, r0_c);
        let typed = PairMorse::typed(type_id, ntypes, d0, alpha, r0);

        let table = table_over(&coords, &links);
        assert_same(
            "morse",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces("morse", &coords, typed.over_pairs_virial(&coords, &table));
    }
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &PairMorse, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.energy_at(&cp) - pot.energy_at(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn well_minimum_is_minus_d0() {
        // At r = r0: E = D0 (0 - 1) = -D0, force zero.
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&coords);
        assert!((e - (-5.0)).abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
