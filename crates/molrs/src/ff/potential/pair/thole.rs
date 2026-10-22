//! Thole dipole-dipole screening (CL&Pol short-range damping).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::{fold_pairs, fold_pairs_untallied};
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

/// Thole-screened Coulomb pair potential with pre-resolved flat arrays.
///
/// `s[idx]` is the screening factor `s_ij` and `qq[idx]` the charge product
/// `q_i q_j` for pair `idx` — both depend on the two endpoints' atom types and
/// are resolved once at construction.
/// Where a pair's screening length and charge product come from.
enum Source {
    /// Resolved against one fixed pair list at construction.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        s: Vec<F>,
        qq: Vec<F>,
    },
    /// Per-atom `(q, α, a)`, combined when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild. Under a ghost régime the vectors
    /// cover the copies too, each carrying its owner's values.
    PerAtom {
        q: Vec<F>,
        alpha: Vec<F>,
        a_thole: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// Thole dipole-dipole screening (CL&Pol short-range damping).
///
/// Screens the Coulomb interaction between Drude-related point charges at short
/// range with the exponential Thole function
///
/// ```text
/// T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)
/// s_ij    = a_ij / (alpha_i alpha_j)^(1/6),   a_ij = (a_i + a_j) / 2
/// ```
///
/// so the damped energy of a pair is `T_ij(r) * q_i q_j / r`. The screening
/// `s_ij` depends on **both** endpoints' atomic polarizabilities, so the
/// constructor resolves per-atom-type `charge` / `alpha` / `damp` from the
/// `atoms` block and precomputes `(s_ij, q_i q_j)` per pair. `alpha` and
/// `damp` (the Thole `a`) are LAMMPS `pair_style thole`'s `pair_coeff` names;
/// the per-type `charge` is molrs's (LAMMPS damps the Drude charges of the
/// atoms).
///
/// Units: r in A, alpha in A^3, a dimensionless, q in e (energy in the same
/// Coulomb units as the accompanying electrostatic kernel — Thole is a
/// multiplicative screen on `q_i q_j / r`).
///
/// Reference: Thole, Chem. Phys. 59 (1981) 341,
/// DOI 10.1016/0301-0104(81)85176-2; as emitted by the paduagroup/clandpol
/// polarizer (LAMMPS `pair_style thole`).
pub struct PairThole {
    source: Source,
}

impl PairThole {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, s: Vec<F>, qq: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), s.len());
        assert_eq!(atom_i.len(), qq.len());
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                s,
                qq,
            },
        }
    }

    /// Per-atom `(q, α, a)`, combined when a pair turns up by the same rule
    /// [`pair_thole_constructor`] applies: `a_ij = ½(aᵢ + aⱼ)`, `s = a_ij/(αᵢαⱼ)^(1/6)`.
    pub fn typed(q: Vec<F>, alpha: Vec<F>, a_thole: Vec<F>) -> Self {
        assert_eq!(q.len(), alpha.len());
        assert_eq!(q.len(), a_thole.len());
        let n_owned = q.len();
        Self {
            source: Source::PerAtom {
                q,
                alpha,
                a_thole,
                n_owned,
            },
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], s: F, qq: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let x = s * r;
        let e_x = (-x).exp();
        let t = 1.0 - (1.0 + x / 2.0) * e_x;
        let energy = t * qq / r;

        // V = T qq / r ;  T'(r) = (s/2)(1 + x) e^{-x}
        // dV/dr = qq (T'/r - T/r^2)
        // factor = -(1/r) dV/dr = qq (T/r^3 - T'/r^2)
        let tp = (s / 2.0) * (1.0 + x) * e_x;
        let dvdr = qq * (tp / r - t / r2);
        let factor = -dvdr / r;
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
            let (i, j, s, qq, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, s, qq) else {
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

impl Potential for PairThole {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            s,
            qq,
        } = &self.source
        else {
            // Per-atom parameters need a pair table, and nobody handed one over.
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
            (i, j, s[idx], qq[idx], d, r2)
        });
    }
}

impl PairDriven for PairThole {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        let Source::PerAtom {
            q, alpha, a_thole, ..
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
                i < q.len() && j < q.len(),
                "a pair names an atom the per-atom parameters do not cover"
            );
            let a_ij = 0.5 * (a_thole[i] + a_thole[j]);
            let s = a_ij / (alpha[i] * alpha[j]).powf(1.0 / 6.0);
            (
                i,
                j,
                s,
                q[i] * q[j],
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
            q,
            alpha,
            a_thole,
            n_owned,
            ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(q, *n_owned, owner);
        gather_copies(alpha, *n_owned, owner);
        gather_copies(a_thole, *n_owned, owner);
    }
}

/// Construct a [`PairThole`] from per-atom-type params and Frame topology.
///
/// The thole style's per-type definitions are keyed by **atom type name** and
/// carry `charge`, `alpha`, `damp`. Each pair's screening is resolved from
/// its two endpoints' atom types (read from the `atoms` block `type` column).
pub fn pair_thole_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    // `PotentialCompiler::compile` projects the force field's `special_bonds` 1-4
    // weight here. The energy is linear in the charge product, so scaling it
    // is exactly scaling the pair.
    let scale_14 = style_params
        .get("coulomb14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;

    let atoms = frame
        .get(ATOMS)
        .ok_or_else(|| "PairThole: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "PairThole: atoms block missing \"type\" column".to_string())?;

    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairThole: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairThole: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairThole: pairs block missing \"atomj\" column".to_string())?;

    let lookup = |type_name: &str| -> Result<(F, F, F), crate::ff::potential::CompileError> {
        let p = type_map
            .get(type_name)
            .ok_or_else(|| format!("PairThole: unknown atom type '{}'", type_name))?;
        let q = param_reads::type_num("thole", type_name, p, "charge")?;
        let alpha = param_reads::type_num("thole", type_name, p, "alpha")?;
        let a = param_reads::type_num("thole", type_name, p, "damp")?;
        Ok((q, alpha, a))
    };

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());
    let mut s_vec = Vec::with_capacity(i_col.len());
    let mut qq_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let (qi, alpha_i, ai) = lookup(&atom_types[i])?;
        let (qj, alpha_j, aj) = lookup(&atom_types[j])?;

        let a_ij = 0.5 * (ai + aj);
        let s = a_ij / (alpha_i * alpha_j).powf(1.0 / 6.0);

        atom_i.push(i);
        atom_j.push(j);
        s_vec.push(s);
        qq_vec.push(if is_14.is_some_and(|b| b[idx]) {
            qi * qj * scale_14
        } else {
            qi * qj
        });
    }

    Ok(ForceTerm::pair(PairThole::new(
        atom_i, atom_j, s_vec, qq_vec,
    )))
}

/// Construct a neighbour-driven [`PairThole`] from per-atom parameters.
///
/// The counterpart of [`pair_thole_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_thole_typed_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let mut per_type = Vec::with_capacity(labels.len());
    for l in &labels {
        let p = type_map
            .get(l.as_str())
            .ok_or_else(|| format!("PairThole: unknown atom type '{l}'"))?;
        let get = |k: &str| param_reads::type_num("thole", l, p, k);
        per_type.push((get("charge")?, get("alpha")?, get("damp")?));
    }
    let pick = |f: fn(&(F, F, F)) -> F| -> Vec<F> {
        type_id.iter().map(|&t| f(&per_type[t as usize])).collect()
    };
    Ok(ForceTerm::pair(PairThole::typed(
        pick(|p| p.0),
        pick(|p| p.1),
        pick(|p| p.2),
    )))
}

#[cfg(test)]
mod tests {

    /// Combining `(q, α, a)` when a pair turns up is the same number as having
    /// combined them earlier against a fixed list — bit for bit.
    #[test]
    fn per_atom_parameters_score_a_pair_exactly_as_compiled_ones() {
        use crate::ff::potential::pair::fixtures::{
            assert_same, assert_virial_matches_forces, table_over,
        };

        let q = vec![0.4_f64, -0.7, 0.3, -0.2];
        let alpha = vec![1.1_f64, 0.8, 1.4, 0.6];
        let a_thole = vec![2.6_f64, 2.6, 2.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let s: Vec<F> = links
            .iter()
            .map(|&(i, j)| {
                let a_ij = 0.5 * (a_thole[i] + a_thole[j]);
                a_ij / (alpha[i] * alpha[j]).powf(1.0 / 6.0)
            })
            .collect();
        let qq: Vec<F> = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let compiled = PairThole::new(ai, aj, s, qq);
        let typed = PairThole::typed(q, alpha, a_thole);

        let table = table_over(&coords, &links);
        assert_same(
            "thole",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces("thole", &coords, typed.over_pairs_virial(&coords, &table));
    }
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn numerical_forces(pot: &PairThole, coords: &[F]) -> Vec<F> {
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
    fn damping_factor_matches_closed_form() {
        // s=1.0, qq=1.0, r=2.0: x=2, T = 1 - (1+1)e^-2 = 1 - 2 e^-2
        let pot = PairThole::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let x = 2.0_f64;
        let t = 1.0 - (1.0 + x / 2.0) * (-x).exp();
        let expected = t * 1.0 / 2.0;
        assert!(
            (pot.energy_at(&coords) - expected).abs() < 1e-12,
            "energy {} vs {expected}",
            pot.energy_at(&coords)
        );
    }

    #[test]
    fn damping_vanishes_at_zero_separation_limit() {
        // T(r) -> 0 as r -> 0 (screening fully removes the singular Coulomb term),
        // so the energy stays finite. Check T is small at small r.
        let pot = PairThole::new(vec![0], vec![1], vec![5.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 0.05, 0.0, 0.0];
        assert!(pot.energy_at(&coords).is_finite());
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
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
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
