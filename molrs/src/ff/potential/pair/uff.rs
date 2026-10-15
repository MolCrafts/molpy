//! UFF van der Waals (RDKit `vdWContrib`).

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::{fold_pairs, fold_pairs_untallied};
use crate::ff::potential::param_reads;
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use molrs::op::F;

/// Where a pair's `(xᵢⱼ, Dᵢⱼ)` comes from.
enum Source {
    /// Combined against one fixed pair list at construction.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        xij: Vec<F>,
        dij: Vec<F>,
    },
    /// Per-atom `x1`/`D1`, combined when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild, so parameters combined against
    /// an older one belong to different atoms. Under a ghost régime the
    /// vectors cover the copies too, each carrying its owner's values.
    PerAtom {
        x1: Vec<F>,
        d1: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// UFF van der Waals: E = D · [(x/r)¹² − 2·(x/r)⁶] (RDKit `vdWContrib`).
///
/// Per-instance columns on the `pairs` block: `xij`, `Dij` (baked by the
/// typifier after the neighbour list is built — but the typifier only sets
/// atom `x1`/`D1`; this ctor combines them geometrically like RDKit).
pub struct PairUffVdw {
    source: Source,
}

impl PairUffVdw {
    /// Parameters combined against a fixed pair list.
    pub fn compiled(atom_i: Vec<usize>, atom_j: Vec<usize>, xij: Vec<F>, dij: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), xij.len());
        assert_eq!(atom_i.len(), dij.len());
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                xij,
                dij,
            },
        }
    }

    /// Per-atom `x1`/`D1`, combined geometrically when a pair turns up — the
    /// same rule [`pair_uff_vdw_constructor`] applies, and RDKit with it.
    pub fn typed(x1: Vec<F>, d1: Vec<F>) -> Self {
        assert_eq!(x1.len(), d1.len());
        let n_owned = x1.len();
        Self {
            source: Source::PerAtom { x1, d1, n_owned },
        }
    }

    /// The pair term for one already-reduced separation.
    ///
    /// Returns the force **on `i`** — UFF's sign convention, kept rather than
    /// normalised so this stays a transcription of RDKit's `vdWContrib`.
    fn pair_kernel(&self, r2: F, disp: [F; 3], xij: F, dij: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 as F {
            return None;
        }
        let r = r2.sqrt();
        // cutoff at 2 * xij (RDKit default threshMultiplier ≈ 2 when used that way)
        if r > 2.0 * xij {
            return None;
        }
        let inv_r = 1.0 / r;
        let rr = xij * inv_r;
        let r2u = rr * rr;
        let r6 = r2u * r2u * r2u;
        let r12 = r6 * r6;
        let energy = dij * (r12 - 2.0 * r6);

        // RDKit: preFactor = 12·D/x · ((x/r)⁷ − (x/r)¹³);
        // dGrad on atom1 (i) = preFactor · (i−j)/r  (gradient contrib);
        // force = −gradient, so F_i = −preFactor · (i−j)/r = preFactor · (j−i)/r
        let r7 = r6 * rr;
        let r13 = r12 * rr;
        let pref = 12.0 * dij / xij * (r7 - r13);
        let factor = pref / r;
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
        pair: impl Fn(usize) -> (usize, usize, F, F, [F; 3], F) + Sync,
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
        pair: &(impl Fn(usize) -> (usize, usize, F, F, [F; 3], F) + Sync),
    ) -> (F, Virial) {
        let mut energy = 0.0 as F;
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
            let (i, j, xij, dij, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, xij, dij) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            if VIRIAL {
                virial.add_outer(f, [-disp[0], -disp[1], -disp[2]]);
            }
            out[i * 3] += f[0];
            out[i * 3 + 1] += f[1];
            out[i * 3 + 2] += f[2];
            out[j * 3] -= f[0];
            out[j * 3 + 1] -= f[1];
            out[j * 3 + 2] -= f[2];
        }
        (energy, virial)
    }
}

impl Potential for PairUffVdw {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let _n = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            xij,
            dij,
        } = &self.source
        else {
            // Per-atom parameters need a pair table, and nobody handed one over.
            return;
        };
        self.fold_into(acc, &[], atom_i.len(), false, |idx| {
            let (i, j) = (atom_i[idx], atom_j[idx]);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (i, j, xij[idx], dij[idx], d, r2)
        });
    }
}

impl PairDriven for PairUffVdw {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        let Source::PerAtom { x1, d1, .. } = &self.source else {
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
                i < x1.len() && j < x1.len(),
                "a pair names an atom the per-atom parameters do not cover"
            );
            (
                i,
                j,
                ((x1[i] * x1[j]) as F).sqrt(),
                ((d1[i] * d1[j]) as F).sqrt(),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::PerAtom {
            x1, d1, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(x1, *n_owned, owner);
        gather_copies(d1, *n_owned, owner);
    }
}

pub fn pair_uff_vdw_constructor(
    style_params: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let atoms = frame.get(ATOMS).ok_or("uff_lj: missing atoms")?;
    let x1 = param_reads::instance_col("uff_lj", atoms, "x1")?;
    let d1 = param_reads::instance_col("uff_lj", atoms, "D1")?;
    // `PotentialCompiler::compile` projects the force field's `special_bonds` 1-4
    // weight here. `E = D·((x/r)¹² − 2(x/r)⁶)` is linear in `D`, so scaling
    // the well depth is exactly scaling the pair.
    let scale_14 = style_params
        .get("lj14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;

    let pairs = frame
        .get(PAIRS)
        .ok_or("uff_lj: missing pairs (call intramolecular_pairs first)")?;
    let is_14 = pairs.get("is_14").and_then(|c| c.get::<bool>());
    if pairs.n_rows().unwrap_or(0) == 0 {
        return Ok(ForceTerm::pair(PairUffVdw::compiled(
            vec![],
            vec![],
            vec![],
            vec![],
        )));
    }
    let pi = pairs
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_lj: pairs missing atomi")?;
    let pj = pairs
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("uff_lj: pairs missing atomj")?;
    let n = pi.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut xij = Vec::with_capacity(n);
    let mut dij = Vec::with_capacity(n);
    for t in 0..n {
        let i = pi[t] as usize;
        let j = pj[t] as usize;
        atom_i.push(i);
        atom_j.push(j);
        xij.push(((x1[i] * x1[j]) as F).sqrt());
        let d = ((d1[i] * d1[j]) as F).sqrt();
        dij.push(if is_14.is_some_and(|b| b[t]) {
            d * scale_14
        } else {
            d
        });
    }
    Ok(ForceTerm::pair(PairUffVdw::compiled(
        atom_i, atom_j, xij, dij,
    )))
}

/// Construct a neighbour-driven [`PairUffVdw`] from per-atom parameters.
///
/// The counterpart of [`pair_uff_vdw_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_uff_vdw_typed_constructor(
    _style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let atoms = frame.get(ATOMS).ok_or("uff_lj: missing atoms")?;
    let x1 = param_reads::instance_col("uff_lj", atoms, "x1")?;
    let d1 = param_reads::instance_col("uff_lj", atoms, "D1")?;
    Ok(ForceTerm::pair(PairUffVdw::typed(
        x1.iter().map(|&v| v as F).collect(),
        d1.iter().map(|&v| v as F).collect(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::pair::fixtures::{
        assert_same, assert_virial_matches_forces, table_over,
    };

    /// Combining `x1`/`D1` when a pair turns up is the same number as having
    /// combined them earlier against a fixed list — bit for bit, on the same
    /// pairs.
    ///
    /// The difference is only *when*, and it matters because a neighbour table
    /// is a different list of pairs every rebuild: parameters combined against
    /// an older one belong to different atoms, silently.
    #[test]
    fn per_atom_parameters_score_a_pair_exactly_as_compiled_ones() {
        let x1 = vec![3.5_f64, 3.1, 3.8, 2.9];
        let d1 = vec![0.06_f64, 0.09, 0.05, 0.12];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let xij: Vec<F> = links
            .iter()
            .map(|&(i, j)| ((x1[i] * x1[j]) as F).sqrt())
            .collect();
        let dij: Vec<F> = links
            .iter()
            .map(|&(i, j)| ((d1[i] * d1[j]) as F).sqrt())
            .collect();
        let compiled = PairUffVdw::compiled(ai, aj, xij, dij);
        let typed = PairUffVdw::typed(x1, d1);

        let table = table_over(&coords, &links);
        assert_same(
            "uff_lj",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces("uff_lj", &coords, typed.over_pairs_virial(&coords, &table));
    }
}
