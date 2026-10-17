//! Buckingham pair potential (LAMMPS `pair_style buck`).

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

/// Where a pair's Buckingham `(A, ρ, C)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        a: Vec<F>,
        rho: Vec<F>,
        c: Vec<F>,
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
        a: Vec<F>,
        rho: Vec<F>,
        c: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// Buckingham pair potential: E = A * exp(-r/rho) - C / r^6
///
/// The exp-6 form used for repulsion/dispersion (e.g. CL&Pol non-bonded cores).
/// Parameters per pair type: `a` (energy), `rho` (length), `c` (energy*length^6).
/// Lowercase is the canonical spelling (spec ff-params-01) and matches molpy;
/// GROMACS spells the middle one `B = 1/rho`, normalized at that reader.
pub struct PairBuck {
    source: Source,
    /// `cutoff²` (`r < cutoff`, as LAMMPS), at both compile doors; infinite
    /// for a style that states no cutoff.
    cutoff2: F,
}

impl PairBuck {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, a: Vec<F>, rho: Vec<F>, c: Vec<F>) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(a.len(), n);
        assert_eq!(rho.len(), n);
        assert_eq!(c.len(), n);
        Self {
            cutoff2: F::INFINITY,
            source: Source::Compiled {
                atom_i,
                atom_j,
                a,
                rho,
                c,
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
    pub fn typed(type_id: Vec<u32>, ntypes: usize, a: Vec<F>, rho: Vec<F>, c: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, a.len(), "a must cover every type pair");
        assert_eq!(n_cells, rho.len(), "rho must cover every type pair");
        assert_eq!(n_cells, c.len(), "c must cover every type pair");
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
                a,
                rho,
                c,
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
    fn pair_kernel(&self, r2: F, disp: [F; 3], a: F, rho: F, c: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        // A zero `rho` is what an unparameterised entry of a type-pair table
        // looks like. `A·exp(-r/0)` is zero and `0/(0·r)` is NaN, so the pair
        // would contribute no energy and an undefined force — the worst of the
        // two possible failures. It contributes nothing instead.
        if rho <= 0.0 {
            return None;
        }
        let r = r2.sqrt();
        let exp_term = a * (-r / rho).exp();
        let r6 = r2 * r2 * r2;
        let energy = exp_term - c / r6;

        // E = A exp(-r/rho) - C r^-6
        // dE/dr = -(A/rho) exp(-r/rho) + 6 C r^-7
        // factor = -(1/r) dE/dr = (A/(rho r)) exp(-r/rho) - 6 C r^-8
        let factor = exp_term / (rho * r) - 6.0 * c / (r6 * r2);
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
            let (i, j, (a, rho, c), disp, r2) = pair(idx);
            if r2 >= self.cutoff2 {
                continue;
            }
            let Some((e, f)) = self.pair_kernel(r2, disp, a, rho, c) else {
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

impl Potential for PairBuck {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            a,
            rho,
            c,
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
            (i, j, (a[idx], rho[idx], c[idx]), d, r2)
        });
    }
}

impl PairDriven for PairBuck {
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
            a,
            rho,
            c,
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
                (a[t], rho[t], c[t]),
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

/// Construct a [`PairBuck`] from style params, type params, and Frame topology.
///
/// A pair's row is found from its two atoms' types — the self row, else the
/// cross row (`buck` does not mix: neither is [`IrError::NoMixing`]) — as
/// `pair_buck_typed_constructor` finds it and LAMMPS's `pair_coeff i j` states it.
///
/// [`IrError::NoMixing`]: crate::ff::ir::IrError::NoMixing
pub fn pair_buck_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    // `PotentialCompiler::compile` projects the force field's `special_bonds` 1-4
    // weight here. `E = A·exp(-r/rho) - C/r⁶` is linear in **both** `A` and
    // `C`, so scaling the two is exactly scaling the pair; scaling only `A`
    // would leave the dispersion term at full strength.
    let scale_14 = style_params
        .get("lj14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;

    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairBuck: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairBuck: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairBuck: pairs block missing \"atomj\" column".to_string())?;
    let atom_types = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "PairBuck: atoms block missing \"type\" column".to_string())?;
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut a_vec = Vec::with_capacity(i_col.len());
    let mut rho_vec = Vec::with_capacity(i_col.len());
    let mut c_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let (ta, tb) = (
            atom_types[i_col[idx] as usize].as_str(),
            atom_types[j_col[idx] as usize].as_str(),
        );
        let params = param_reads::unmixed_row("buck", "a", &type_map, ta, tb)?;
        let label = pair_key(ta, tb)?;
        let a = param_reads::type_num("buck", &label, params, "a")?;
        let rho = param_reads::type_num("buck", &label, params, "rho")?;
        let c = param_reads::type_num("buck", &label, params, "c")?;

        let w = if is_14.is_some_and(|b| b[idx]) {
            scale_14
        } else {
            1.0
        };
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        a_vec.push(a * w);
        rho_vec.push(rho);
        c_vec.push(c * w);
    }

    // The style's `cutoff` (`r < cutoff`, as LAMMPS truncates every pair,
    // 1-4 ones included; ∞ when it states none).
    let cutoff = param_reads::pair_cutoff("buck", style_params)?;
    Ok(ForceTerm::pair(
        PairBuck::new(atom_i, atom_j, a_vec, rho_vec, c_vec).with_cutoff(cutoff),
    ))
}

/// Construct a neighbour-driven [`PairBuck`] from per-atom parameters.
///
/// The counterpart of [`pair_buck_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_buck_typed_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut a = vec![0.0 as F; ntypes * ntypes];
    let mut rho = vec![0.0 as F; ntypes * ntypes];
    let mut c = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            // Keyed in either order alike; a self-pair by the atom type alone.
            let (ta, tb) = (labels[ti].as_str(), labels[tj].as_str());
            let p = param_reads::unmixed_row("buck", "a", &type_map, ta, tb)?;
            let key = pair_key(ta, tb)?;
            let t = type_pair(ti as u32, tj as u32, ntypes);
            a[t] = param_reads::type_num("buck", &key, p, "a")?;
            rho[t] = param_reads::type_num("buck", &key, p, "rho")?;
            c[t] = param_reads::type_num("buck", &key, p, "c")?;
        }
    }
    let kernel = PairBuck::typed(type_id, ntypes, a, rho, c);
    // The style's `cutoff` (`r < cutoff`, as LAMMPS): finite, for a
    // neighbour sum is not finite without one.
    let cutoff = param_reads::neighbour_cutoff("buck", style_params)?;
    Ok(ForceTerm::pair(kernel.with_cutoff(cutoff)))
}

#[cfg(test)]
mod tests {

    /// A flagged 1-4 pair is scaled by `lj14scale`, and scaled in **both**
    /// terms.
    ///
    /// Buckingham is `A·exp(-r/rho) − C/r⁶`. The energy is linear in `A` and
    /// in `C` separately, so scaling the pair means scaling both; scaling only
    /// the repulsive `A` — the obvious half-fix — leaves the dispersion term
    /// at full strength and gives a different number at every separation.
    /// Halving the weight must halve the energy exactly, whatever `r` is.
    #[test]
    fn a_1_4_pair_is_scaled_in_both_buckingham_terms() {
        use crate::ff::ir::Params;
        use molrs::core::Block;
        use molrs::core::Frame;
        use molrs::op::Idx;
        use ndarray::Array1;

        let build = |scale: f64| {
            let mut frame = Frame::new();
            let mut atoms = Block::new();
            atoms
                .insert(
                    "type",
                    Array1::from(vec!["a".to_string(), "a".to_string()]).into_dyn(),
                    None,
                )
                .unwrap();
            frame.insert("atoms", atoms);
            let mut pairs = Block::new();
            pairs
                .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn(), None)
                .unwrap();
            pairs
                .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn(), None)
                .unwrap();
            pairs
                .insert("is_14", Array1::from(vec![true]).into_dyn(), None)
                .unwrap();
            frame.insert("pairs", pairs);

            let mut sp = Params::new();
            sp.set("lj14scale", scale);
            let mut tp = Params::new();
            tp.set("a", 12000.0);
            tp.set("rho", 0.31);
            tp.set("c", 280.0);
            pair_buck_constructor(&sp, &[("a", &tp)], &frame).expect("buck kernel")
        };

        // Two separations: one where repulsion dominates, one where dispersion
        // does. A half-fix passes at neither, and passing at both is what makes
        // this a statement about the pair rather than about one term.
        for r in [1.5_f64, 4.5] {
            let coords = vec![0.0, 0.0, 0.0, r, 0.0, 0.0];
            let full = build(1.0).energy_forces_at(&coords);
            let half = build(0.5).energy_forces_at(&coords);
            assert!(
                full.0.abs() > 1e-9,
                "r = {r}: the pair must carry energy, or this proves nothing"
            );
            assert!(
                (half.0 - 0.5 * full.0).abs() < 1e-12 * full.0.abs().max(1.0),
                "r = {r}: scaled {} vs half of {}",
                half.0,
                full.0
            );
            for k in 0..full.1.len() {
                assert!(
                    (half.1[k] - 0.5 * full.1[k]).abs() < 1e-12 * full.1[k].abs().max(1.0),
                    "r = {r}: force component {k}: {} vs half of {}",
                    half.1[k],
                    full.1[k]
                );
            }
        }
    }

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
        let a: Vec<F> = vec![12000.0, 9000.0, 9000.0, 7000.0];
        let rho: Vec<F> = vec![0.31, 0.29, 0.29, 0.27];
        let c: Vec<F> = vec![280.0, 190.0, 190.0, 140.0];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let a_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| a[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let rho_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| rho[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let c_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| c[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairBuck::new(ai, aj, a_c, rho_c, c_c);
        let typed = PairBuck::typed(type_id, ntypes, a, rho, c);

        let table = table_over(&coords, &links);
        assert_same(
            "buck",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces("buck", &coords, typed.over_pairs_virial(&coords, &table));
    }
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &PairBuck, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            let ep = pot.energy_at(&cp);
            let em = pot.energy_at(&cm);
            num[k] = -(ep - em) / (2.0 * h); // force = -dE/dx
        }
        num
    }

    #[test]
    fn energy_matches_closed_form() {
        // A=1000, rho=0.3, C=100 at r=2.0:
        // E = 1000*exp(-2/0.3) - 100/64
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let e = pot.energy_at(&coords);
        let expected = 1000.0 * (-2.0f64 / 0.3).exp() - 100.0 / 64.0;
        assert!((e - expected).abs() < 1e-9, "energy {e} vs {expected}");
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        // off-axis geometry exercising all three components
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.energy_forces_at(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} analytical={} numerical={}",
                analytical[k],
                numerical[k]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
