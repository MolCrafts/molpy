//! Buffered Coulomb pair potential with a hard distance cutoff (`pair coul/cut`).

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

/// Below this squared separation a pair's force has no direction, so only the
/// (buffered) energy is accumulated. Unchanged from the unbuffered kernel.
const R_MIN2: F = 1e-24;

/// Buffered Coulomb with cutoff, over pre-resolved flat arrays.
///
/// `E = k·qᵢqⱼ / (D·(r + δ))`. All four scalars come from the force field.
///
/// Buffered Coulomb pair potential with a hard distance cutoff — `pair/coul/cut`:
///
/// ```text
/// E(r) = k · qᵢqⱼ / (D · (r + δ))      r < r_cut,  else 0
/// ```
///
/// Every constant in that formula is **force-field data, read from the style**:
/// the Coulomb constant `k`, the dielectric `D` and the buffering distance `δ`.
/// The kernel holds none of them.
///
/// # One kernel, two force fields
///
/// `δ = 0` degenerates the buffered form into the textbook Coulomb `k·qᵢqⱼ/r`,
/// which is what OPLS, LAMMPS and every other non-buffered force field mean. MMFF's
/// electrostatics (Halgren, *MMFF.I* eq. 5) is the *same kernel* at
/// `k = 332.0716`, `D = 1.0`, `δ = 0.05 Å` — a parameterization, not a kernel of
/// its own. The buffer is what keeps `E` finite at `r = 0`, where MMFF's charges
/// sit on the nuclei.
///
/// # Why `k` is a style param and not a shared constant
///
/// MMFF uses Halgren's **332.0716**; OPLS/LAMMPS use CODATA's
/// [`COULOMB_REAL`](molrs::core::constants::COULOMB_REAL) = **332.06371**. The 2.4e-5
/// relative difference is worth 0.0036 kcal/mol on caffeine's −150.48 kcal/mol
/// electrostatic term — **above** the 1e-3 RDKit parity tolerance. Both values are
/// correct; the force field decides. A kernel holding either one would be choosing a
/// force field for its caller, so there is no default: a style that does not say is
/// an [`Err`].
///
/// The kernel is topology-blind: it consumes a pre-resolved pair list whose per-pair
/// charge products `qᵢqⱼ` already include any exclusion / 1-4 scaling.
pub struct PairCoulCut {
    charges: Charges,
    /// Coulomb constant `k` in kcal·Å·mol⁻¹·e⁻² — the force field's, not the kernel's.
    coulomb: F,
    /// Dielectric constant `D` of the medium the force field was parameterised in.
    dielectric: F,
    /// Buffering distance δ in Å; `0.0` is the unbuffered, textbook Coulomb.
    delta: F,
    /// Distance cutoff in Å; `f64::INFINITY` disables it.
    cutoff: F,
}

/// Where a pair's charge product comes from.
enum Charges {
    /// Products resolved against one fixed pair list at construction, already
    /// carrying any exclusion or 1-4 scaling.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
    },
    /// Per-atom charges; the product is formed when a pair turns up.
    ///
    /// This is what a neighbour-driven evaluation needs — a neighbour table is
    /// a different list of pairs every rebuild, so a product resolved against
    /// an older one names the wrong atoms. Under a ghost régime the vector
    /// covers the copies too, each carrying its owner's charge.
    ///
    /// It carries **no** 1-4 scaling: there is nothing on a neighbour table to
    /// carry it. Scaling close neighbours is the caller's, through the
    /// special-bonds weights.
    PerAtom {
        q: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

impl PairCoulCut {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
        coulomb: F,
        dielectric: F,
        delta: F,
        cutoff: F,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qiqj.len());
        Self {
            charges: Charges::Compiled {
                atom_i,
                atom_j,
                qiqj,
            },
            coulomb,
            dielectric,
            delta,
            cutoff,
        }
    }

    /// A kernel that forms `qᵢqⱼ` from the atoms a pair names.
    ///
    /// The form a neighbour-driven evaluation needs.
    ///
    /// A neighbour table is a different list of pairs every rebuild, so a
    /// parameter resolved against an older one belongs to different atoms.
    /// Keyed on the atoms instead, it can be found for whatever pair turns up
    /// — including one that names a periodic copy.
    pub fn typed(q: Vec<F>, coulomb: F, dielectric: F, delta: F, cutoff: F) -> Self {
        let n_owned = q.len();
        Self {
            charges: Charges::PerAtom { q, n_owned },
            coulomb,
            dielectric,
            delta,
            cutoff,
        }
    }

    /// The pair term: buffered Coulomb for one already-reduced separation.
    ///
    /// `qq` is `k·qᵢqⱼ/D` — the style constants are hoisted by the caller, once
    /// per evaluation rather than once per pair. Returns the energy always and
    /// the force only when the separation has a direction.
    fn pair_kernel(&self, r2: F, disp: [F; 3], qq: F) -> Option<(F, [F; 3])> {
        if r2 >= self.cutoff * self.cutoff {
            return None;
        }
        let r = r2.sqrt();
        let r_buf = r + self.delta;
        // Unbuffered (δ = 0) coincident charges: `k·qᵢqⱼ/0` is undefined, so
        // skip the pair entirely. With a buffer this cannot trigger — keeping
        // the term finite at r = 0 is the buffer's whole purpose.
        if r_buf <= 0.0 {
            return None;
        }
        let energy = qq / r_buf;
        // The force is `−dE/dr · r̂` with `dE/dr = −k·qᵢqⱼ/(D·(r+δ)²)`, i.e. the
        // BUFFERED distance squared in the denominator — not `r²`. At r = 0 it
        // has no direction (the energy above is still real), so leave it zero.
        if r2 < R_MIN2 {
            return Some((energy, [0.0, 0.0, 0.0]));
        }
        let factor = qq / (r_buf * r_buf * r);
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
        pair: impl Fn(usize) -> (usize, usize, F, [F; 3], F) + Sync,
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
        pair: &(impl Fn(usize) -> (usize, usize, F, [F; 3], F) + Sync),
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
            let (i, j, qq, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, qq) else {
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

impl Potential for PairCoulCut {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let n_atoms = validate_coords(coords);
        // `k/D` is a style-level constant: hoist it out of the pair loop. At
        // D = 1 this is exactly `k` (IEEE: `x / 1.0 == x`), so the δ = 0, D = 1
        // path stays bit-for-bit identical to the unbuffered kernel this
        // generalizes.
        let k_over_d = self.coulomb / self.dielectric;
        let Charges::Compiled {
            atom_i,
            atom_j,
            qiqj,
        } = &self.charges
        else {
            // Per-atom charges need a pair table, and nobody handed one over.
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
            (i, j, k_over_d * qiqj[idx], d, r2)
        });
    }
}

impl PairDriven for PairCoulCut {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        let k_over_d = self.coulomb / self.dielectric;
        let Charges::PerAtom { q, .. } = &self.charges else {
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
                "a pair names an atom the charge vector does not cover"
            );
            // Parenthesised the way the compiled path forms it: `k/D · (qᵢqⱼ)`
            // and `(k/D · qᵢ) · qⱼ` differ in the last bit, and the two paths
            // are meant to be the same number.
            let qq = k_over_d * (q[i] * q[j]);
            (i, j, qq, [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]], d2[p])
        });
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.charges, Charges::Compiled { .. })
    }
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Charges::PerAtom { q, n_owned, .. } = &mut self.charges else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(q, *n_owned, owner);
    }
}

/// `(coulomb, dielectric, delta)` of a gathered `coul/cut` style.
fn coul_style(style_params: &Params) -> Result<(F, F, F), crate::ff::ir::IrError> {
    let get = |key: &str| param_reads::style_num("coul/cut", style_params, key);
    Ok((get("coulomb")?, get("dielectric")?, get("delta")?))
}

/// Construct a [`PairCoulCut`] from **per-atom charges** + a neighbour list.
///
/// Reads per-atom `charge` from the atoms block (the molecule carries its own
/// charges — RESP/AM1-BCC for GAFF, MMFF's bond-increment charges, …), forms `qᵢqⱼ`
/// for each interacting pair, and scales 1-4-flagged pairs by the force field's 1-4
/// Coulomb weight.
///
/// # Style params
///
/// They arrive gathered ([`StyleSpec::gather`](crate::ff::ir::StyleSpec::gather)):
/// the spec's defaults are in place and this kernel states none.
///
/// | param | meaning | absent |
/// |---|---|---|
/// | `coulomb` | Coulomb constant `k` | [`IrError::MissingParam`] — the force field's to choose |
/// | `dielectric` | dielectric `D` | the spec's default, 1 (LAMMPS's `dielectric`) |
/// | `delta` | buffering distance δ (Å) | the spec's default, 0: no buffer, the textbook Coulomb |
/// | `cutoff` | cutoff (Å): a pair prices at `r < cutoff`, as LAMMPS | the spec's default, ∞: do not truncate |
/// | `coulomb14scale` | 1-4 weight | projected from `special_bonds` by `PotentialCompiler::compile` |
///
/// [`IrError::MissingParam`]: crate::ff::ir::IrError::MissingParam
///
/// The `pairs` block is the consumer-built neighbour list (`atomi`/`atomj`/`is_14`)
/// from `intramolecular_pairs`; 1-2/1-3 are already excluded. Charge-free pair types
/// are not consulted — this kernel is per-atom
/// ([`ParamSource::PerInstance`](crate::ff::ir::ParamSource::PerInstance)).
pub fn pair_coul_cut_constructor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let (coulomb, dielectric, delta) = coul_style(style_params)?;
    let cutoff = param_reads::pair_cutoff("coul/cut", style_params)?;
    let scale_14 = param_reads::style_num("coul/cut", style_params, "coulomb14scale")?;

    let atoms = frame
        .get(ATOMS)
        .ok_or_else(|| "PairCoulCut: frame missing \"atoms\" block".to_string())?;
    let charges = atoms
        .get("charge")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| "PairCoulCut: atoms block missing \"charge\" column".to_string())?;
    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairCoulCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut qiqj = Vec::with_capacity(n);

    for idx in 0..n {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let mut qq = charges[i] as F * charges[j] as F;
        if is_14.is_some_and(|b| b[idx]) {
            qq *= scale_14;
        }
        atom_i.push(i);
        atom_j.push(j);
        qiqj.push(qq);
    }

    Ok(ForceTerm::pair(PairCoulCut::new(
        atom_i, atom_j, qiqj, coulomb, dielectric, delta, cutoff,
    )))
}

/// Construct a neighbour-driven [`PairCoulCut`] from per-atom parameters.
///
/// The counterpart of [`pair_coul_cut_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_coul_cut_typed_constructor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let (coulomb, dielectric, delta) = coul_style(style_params)?;
    let cutoff = param_reads::neighbour_cutoff("coul/cut", style_params)?;
    // `coulomb14scale` is deliberately not read here. On this path the 1-4
    // weight is applied to the pair table, not baked into a charge product, so
    // requiring the kernel to know it would be asking it a question it no
    // longer answers.
    let atoms = frame
        .get(ATOMS)
        .ok_or_else(|| "PairCoulCut: frame missing \"atoms\" block".to_string())?;
    let charge = atoms
        .get("charge")
        .and_then(|c| c.get::<f64>())
        .ok_or_else(|| "PairCoulCut: atoms block missing \"charge\" column".to_string())?;
    let q: Vec<F> = charge.iter().map(|&c| c as F).collect();
    Ok(ForceTerm::pair(PairCoulCut::typed(
        q, coulomb, dielectric, delta, cutoff,
    )))
}

#[cfg(test)]
mod tests {
    use crate::ff::potential::pair::fixtures::{
        assert_same, assert_virial_matches_forces, table_over,
    };

    /// Forming `qᵢqⱼ` from the atoms is the same number as having formed it
    /// earlier against a fixed list — bit for bit, on the same pairs.
    ///
    /// The difference is only *when*, and that matters because a neighbour
    /// table is a different list of pairs every rebuild: a product resolved
    /// against an older one names the wrong atoms, silently.
    #[test]
    fn per_atom_charges_score_a_pair_exactly_as_compiled_products() {
        let q = vec![0.4_f64, -0.7, 0.3, -0.2];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let (k, d, delta, cutoff) = (332.0716_f64, 1.0_f64, 0.05_f64, 10.0_f64);

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let qiqj: Vec<F> = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let compiled = PairCoulCut::new(ai, aj, qiqj, k, d, delta, cutoff);
        let typed = PairCoulCut::typed(q, k, d, delta, cutoff);

        let table = table_over(&coords, &links);
        assert_same(
            "coul/cut",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "coul/cut",
            &coords,
            typed.over_pairs_virial(&coords, &table),
        );
    }

    use super::*;

    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::constants::COULOMB_REAL;

    /// MMFF's parameterization: Halgren's constant, unit dielectric, 0.05 Å buffer.
    const MMFF: (F, F, F) = (332.0716, 1.0, 0.05);

    fn unbuffered(qiqj: F, cutoff: F) -> PairCoulCut {
        PairCoulCut::new(vec![0], vec![1], vec![qiqj], COULOMB_REAL, 1.0, 0.0, cutoff)
    }

    #[test]
    fn energy_and_sign() {
        // Unit positive charges 2 Å apart: E = k_e / 2.
        let pot = unbuffered(1.0, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!((e - COULOMB_REAL / 2.0).abs() < 1e-9, "E got {e}");
        // Like charges repel: force on j (+x atom) points +x (away from i).
        assert!(
            forces[3] > 0.0,
            "like charges should repel, fxj={}",
            forces[3]
        );
        // Unlike charges attract.
        let pot2 = unbuffered(-1.0, F::INFINITY);
        let (_, f2) = pot2.energy_forces_at(&coords);
        assert!(f2[3] < 0.0, "unlike charges should attract, fxj={}", f2[3]);
    }

    #[test]
    fn cutoff_and_zero_distance() {
        let pot = unbuffered(1.0, 1.5);
        // 2 Å apart, cutoff 1.5 -> excluded, E=0.
        let far: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert_eq!(pot.energy_forces_at(&far).0, 0.0);
        // Coincident atoms, UNBUFFERED -> skipped, finite.
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&zero);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    /// The buffer's reason to exist: at r = 0 the energy is `k·qᵢqⱼ/(D·δ)`, finite.
    #[test]
    fn buffered_zero_distance_is_finite() {
        let (k, d, delta) = MMFF;
        let pot = PairCoulCut::new(vec![0], vec![1], vec![-0.25], k, d, delta, F::INFINITY);
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.energy_forces_at(&zero);
        assert!((e - k * -0.25 / (d * delta)).abs() < 1e-9, "E got {e}");
        assert!(f.iter().all(|x| x.is_finite()), "forces {f:?}");
    }

    /// `δ = 0` reproduces the unbuffered kernel bit-for-bit (r = 2 is exact in f64).
    #[test]
    fn delta_zero_is_the_unbuffered_kernel() {
        let pot = unbuffered(-0.48, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, _) = pot.energy_forces_at(&coords);
        assert_eq!(e, COULOMB_REAL * -0.48 / 2.0);
    }

    /// The force must use `(r + δ)²`, not `r²` — buffering the energy and forgetting
    /// the gradient passes every energy assertion and fails here.
    #[test]
    fn numerical_gradient() {
        for (k, d, delta) in [(COULOMB_REAL, 1.0, 0.0), MMFF] {
            let pot = PairCoulCut::new(vec![0], vec![1], vec![0.8], k, d, delta, F::INFINITY);
            let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.7, 0.3, -0.4];
            let (_, forces) = pot.energy_forces_at(&coords);
            let h = 1e-6;
            for dim in 0..coords.len() {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[dim] += h;
                cm[dim] -= h;
                let fd = -(pot.energy_forces_at(&cp).0 - pot.energy_forces_at(&cm).0) / (2.0 * h);
                assert!(
                    (forces[dim] - fd).abs() < 1e-5,
                    "delta {delta}, comp {dim}: analytic {} vs fd {fd}",
                    forces[dim]
                );
            }
            // Newton's third law.
            for dim in 0..3 {
                assert!((forces[dim] + forces[3 + dim]).abs() < 1e-9);
            }
        }
    }

    /// `coulomb` is the force field's to state: a style without it is
    /// [`IrError::MissingParam`](crate::ff::ir::IrError::MissingParam). The
    /// spec's defaults — `dielectric` 1 (LAMMPS's), `delta` 0, `cutoff` ∞ —
    /// apply when the style leaves them out, exactly as stating them does.
    #[test]
    fn coulomb_is_required_and_the_spec_defaults_apply() {
        use crate::ff::compile::ExplicitTerms;
        use crate::ff::ir::IrError;
        let terms = |style: &[(&str, f64)]| {
            ExplicitTerms::new("pair", "coul/cut")
                .style_params(Params::from_iter(style.iter().copied()))
                .atoms([vec![0, 1]])
                .charges(vec![0.5, -0.4])
                .compile()
        };
        let err = terms(&[]).map(|_| ()).unwrap_err();
        assert_eq!(
            err.ir(),
            Some(&IrError::MissingParam {
                style: "coul/cut".into(),
                type_: String::new(),
                param: "coulomb".into(),
            })
        );
        let coords = [0.0, 0.0, 0.0, 2.5, 0.0, 0.0];
        let bare = terms(&[("coulomb", COULOMB_REAL)]).unwrap();
        let stated = terms(&[
            ("coulomb", COULOMB_REAL),
            ("dielectric", 1.0),
            ("delta", 0.0),
            ("cutoff", F::INFINITY),
        ])
        .unwrap();
        let want = COULOMB_REAL * 0.5 * -0.4 / 2.5;
        assert!((bare.energy_at(&coords) - want).abs() < 1e-12);
        assert_eq!(bare.energy_at(&coords), stated.energy_at(&coords));
    }
}
