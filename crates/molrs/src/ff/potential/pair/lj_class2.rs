//! Class2 (9-6) Lennard-Jones pair potential (LAMMPS `pair_style lj/class2`).

use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::lj_cut::{lj_pair_params, mixing_of};
use crate::ff::potential::pair::type_pair;
use crate::ff::potential::pair::{fold_pairs, fold_pairs_untallied};
use crate::ff::potential::param_reads;
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

/// Where a pair's class2 `(ε, σ)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
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
        epsilon: Vec<F>,
        sigma: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// Class2 (9-6) Lennard-Jones pair potential:
/// E = epsilon * (2*(sigma/r)^9 - 3*(sigma/r)^6)
///
/// The COMPASS/class2 non-bonded form. Parameters per pair type: `epsilon`
/// (energy), `sigma` (length).
pub struct PairLjClass2 {
    source: Source,
    /// `cutoff²` (`r < cutoff`, as LAMMPS), at both compile doors; infinite
    /// for a style that states no cutoff.
    cutoff2: F,
}

impl PairLjClass2 {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(epsilon.len(), n);
        assert_eq!(sigma.len(), n);
        Self {
            cutoff2: F::INFINITY,
            source: Source::Compiled {
                atom_i,
                atom_j,
                epsilon,
                sigma,
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
    pub fn typed(type_id: Vec<u32>, ntypes: usize, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, epsilon.len(), "epsilon must cover every type pair");
        assert_eq!(n_cells, sigma.len(), "sigma must cover every type pair");
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
                epsilon,
                sigma,
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
    fn pair_kernel(&self, r2: F, disp: [F; 3], eps: F, sigma: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let u = sigma / r; // sigma/r
        let u3 = u * u * u;
        let u6 = u3 * u3;
        let u9 = u6 * u3;
        let energy = eps * (2.0 * u9 - 3.0 * u6);

        // E = eps(2 u^9 - 3 u^6), u = sigma/r, du/dr = -u/r
        // dE/dr = eps(18 u^8 - 18 u^5)(-u/r) = -18 eps (u^9 - u^6)/r
        // factor = -(1/r) dE/dr = 18 eps (u^9 - u^6)/r^2
        let factor = 18.0 * eps * (u9 - u6) / r2;
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
        pair: impl Fn(usize) -> (usize, usize, (F, F), [F; 3], F) + Sync,
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
        pair: &(impl Fn(usize) -> (usize, usize, (F, F), [F; 3], F) + Sync),
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
            let (i, j, (eps, sigma), disp, r2) = pair(idx);
            if r2 >= self.cutoff2 {
                continue;
            }
            let Some((e, f)) = self.pair_kernel(r2, disp, eps, sigma) else {
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

impl Potential for PairLjClass2 {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            epsilon,
            sigma,
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
            (i, j, (epsilon[idx], sigma[idx]), d, r2)
        });
    }
}

impl PairDriven for PairLjClass2 {
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
            epsilon,
            sigma,
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
                (epsilon[t], sigma[t]),
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

/// Construct a [`PairLjClass2`] from style params, type params, and Frame topology.
pub fn pair_lj_class2_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let mixing = mixing_of("lj/class2", style_params)?;
    // `PotentialCompiler::compile` projects the force field's `special_bonds` 1-4
    // weight here. The energy is linear in this parameter, so scaling it is
    // exactly scaling the pair.
    let scale_14 = style_params
        .get("lj14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;

    let atom_types = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "PairLjClass2: atoms block missing \"type\" column".to_string())?;
    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairLjClass2: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairLjClass2: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairLjClass2: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut eps_vec = Vec::with_capacity(n);
    let mut sig_vec = Vec::with_capacity(n);
    for idx in 0..n {
        let (i, j) = (i_col[idx] as usize, j_col[idx] as usize);
        let (eps, sigma) = lj_pair_params(
            "lj/class2",
            &type_map,
            mixing,
            &atom_types[i],
            &atom_types[j],
        )?;
        atom_i.push(i);
        atom_j.push(j);
        eps_vec.push(if is_14.is_some_and(|b| b[idx]) {
            eps * scale_14
        } else {
            eps
        });
        sig_vec.push(sigma);
    }

    // The style's `cutoff` (`r < cutoff`, as LAMMPS truncates every pair,
    // 1-4 ones included; ∞ when it states none).
    let cutoff = param_reads::pair_cutoff("lj/class2", style_params)?;
    Ok(ForceTerm::pair(
        PairLjClass2::new(atom_i, atom_j, eps_vec, sig_vec).with_cutoff(cutoff),
    ))
}

/// Construct a neighbour-driven [`PairLjClass2`] from per-atom parameters.
///
/// The counterpart of [`pair_lj_class2_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_lj_class2_typed_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let mixing = mixing_of("lj/class2", style_params)?;
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut epsilon = vec![0.0 as F; ntypes * ntypes];
    let mut sigma = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            let t = type_pair(ti as u32, tj as u32, ntypes);
            (epsilon[t], sigma[t]) =
                lj_pair_params("lj/class2", &type_map, mixing, &labels[ti], &labels[tj])?;
        }
    }
    let kernel = PairLjClass2::typed(type_id, ntypes, epsilon, sigma);
    // The style's `cutoff` (`r < cutoff`, as LAMMPS): finite, for a
    // neighbour sum is not finite without one.
    let cutoff = param_reads::neighbour_cutoff("lj/class2", style_params)?;
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
        let epsilon: Vec<F> = vec![0.12, 0.09, 0.09, 0.17];
        let sigma: Vec<F> = vec![3.4, 3.1, 3.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let epsilon_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| epsilon[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let sigma_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| sigma[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairLjClass2::new(ai, aj, epsilon_c, sigma_c);
        let typed = PairLjClass2::typed(type_id, ntypes, epsilon, sigma);

        let table = table_over(&coords, &links);
        assert_same(
            "lj/class2",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "lj/class2",
            &coords,
            typed.over_pairs_virial(&coords, &table),
        );
    }
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &PairLjClass2, coords: &[F]) -> Vec<F> {
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
    fn energy_at_sigma_is_negative_eps() {
        // At r = sigma: E = eps(2 - 3) = -eps.
        let pot = PairLjClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((pot.energy_at(&coords) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn force_vanishes_at_minimum() {
        // Minimum of 2u^9 - 3u^6 is at u=1 (r=sigma); force is zero there.
        let pot = PairLjClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (_, f) = pot.energy_forces_at(&coords);
        for fi in f {
            assert!(fi.abs() < 1e-9, "force {fi}");
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairLjClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
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
        let pot = PairLjClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }

    /// An unlike pair with no cross row is mixed by the style's `mixing`,
    /// absent `sixthpower` — LAMMPS's rule for `lj/class2` — at both doors.
    #[test]
    fn an_unlike_pair_mixes_sixthpower_by_default() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::ir::CombiningRule;
        use crate::ff::ir::Params;
        use molrs::core::Block;
        use molrs::op::Idx;
        use ndarray::Array1;

        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/class2", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            )
            .unwrap()
            .def_type(
                "B",
                &["B"],
                Params::from_iter([("epsilon", 0.4), ("sigma", 3.6)]),
            )
            .unwrap();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        for (key, v) in [("x", [0.0, 3.4]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])] {
            atoms
                .insert(key, Array1::from_vec(v.to_vec()).into_dyn(), None)
                .unwrap();
        }
        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);

        let (eps, sigma) = CombiningRule::SixthPower.combine((0.1, 3.0), (0.4, 3.6));
        let s = sigma / 3.4;
        let want = eps * (2.0 * s.powi(9) - 3.0 * s.powi(6));
        let e = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&[0.0, 0.0, 0.0, 3.4, 0.0, 0.0]);
        assert!((e - want).abs() < 1e-12, "{e} vs {want}");
    }
}
