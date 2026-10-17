//! MMFF94's Van der Waals kernel — the buffered 14-7 potential.
//!
//! The style is **per-type**: its 95 rows (one per MMFF atom type) carry the
//! polarizabilities the combining rules need, and the style-level block
//! (`<VdWParams B= Beta= DARAD= DAEPS=>`) shapes those rules.
//!
//! # MMFF's electrostatics is not here, and owns no kernel
//!
//! It is a **buffered Coulomb** — `E = k·qᵢqⱼ / (D·(r + δ))` — which is the *generic*
//! [`PairCoulCut`](super::coul_cut::PairCoulCut) kernel at `k = 332.0716`, `D = 1.0`,
//! `δ = 0.05 Å`: a parameterization, not a kernel of its own. The numbers live in
//! [`MMFF_ELE_STYLE`](crate::ff::params::mmff::MMFF_ELE_STYLE) and reach the kernel
//! through the style. vdW is the opposite case — a genuine per-type table — which is
//! why it stays.
//!
//! # 1-4 scaling (house convention)
//!
//! MMFF scales the 1-4 **electrostatic** interaction by 0.75 and does **not**
//! scale the 1-4 **van der Waals** interaction at all (Halgren 1996; RDKit
//! `Nonbonded.cpp`). Both weights arrive through [`SpecialBonds`], which
//! [`PotentialCompiler::compile`] projects into the pair params as `coulomb14scale` /
//! `lj14scale` — so neither kernel hardcodes a scale factor.
//!
//! [`SpecialBonds`]: crate::ff::ir::SpecialBonds
//! [`PotentialCompiler::compile`]: crate::ff::compile::PotentialCompiler::compile

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::Params;
#[cfg(test)]
use crate::ff::params::mmff::DA_NEITHER;
use crate::ff::params::mmff::{DA_ACCEPTOR, DA_DONOR};
use crate::ff::potential::flat_coords::{sub3, validate_coords};
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::{fold_pairs, fold_pairs_untallied};
use crate::ff::potential::{Accumulator, ForceTerm, PairDriven, Potential};
use crate::op::vec3::norm;
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

// ---------------------------------------------------------------------------
// PairMmffVdw: Buffered 14-7 potential
// ---------------------------------------------------------------------------

/// Buffered 14-7 van der Waals (MMFF.I eq. 8), one row per non-excluded pair.
///
/// `r_star` is in Å and `epsilon` in kcal·mol⁻¹; both are the *combined* pair
/// values produced by [`pair_mmff_vdw_constructor`]'s combining rules (including the
/// donor/acceptor corrections), with the 1-4 weight (`lj14scale`, 1.0 for MMFF)
/// already folded into `epsilon`.
/// Where a pair's `(R*, ε)` comes from.
enum Source {
    /// Combined against one fixed pair list at construction, already carrying
    /// any 1-4 scaling.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        r_star: Vec<F>,
        epsilon: Vec<F>,
    },
    /// Per-atom vdW parameters, combined when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild. Under a ghost régime the vector
    /// covers the copies too, each carrying its owner's parameters.
    ///
    /// It carries **no** 1-4 scaling: there is nothing on a neighbour table to
    /// carry it. MMFF's 1-4 vdW weight is exactly 1.0, so nothing is lost here
    /// today — but a style that set one would need the caller's special-bonds
    /// weights.
    PerAtom {
        atoms: Vec<PairMmffVdwAtomParams>,
        style: PairMmffVdwStyleParams,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct PairMmffVdw {
    source: Source,
}

impl PairMmffVdw {
    /// Parameters combined against a fixed pair list.
    pub fn compiled(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        r_star: Vec<F>,
        epsilon: Vec<F>,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), r_star.len());
        assert_eq!(atom_i.len(), epsilon.len());
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                r_star,
                epsilon,
            },
        }
    }

    /// Per-atom vdW parameters, combined when a pair turns up by the same
    /// rule [`pair_mmff_vdw_constructor`] applies.
    pub fn typed(atoms: Vec<PairMmffVdwAtomParams>, style: PairMmffVdwStyleParams) -> Self {
        let n_owned = atoms.len();
        Self {
            source: Source::PerAtom {
                atoms,
                style,
                n_owned,
            },
        }
    }

    /// The pair term for one already-reduced separation (Halgren buffered 14-7).
    fn pair_kernel(&self, d: [F; 3], rs: F, eps: F) -> Option<(F, [F; 3])> {
        let r = norm(d);
        if r < 1e-12 as F {
            return None;
        }
        let rho = r + 0.07 * rs;
        let u = 1.07 * rs / rho;
        let u6 = u * u * u * u * u * u;
        let u7 = u6 * u;
        let r7 = r * r * r * r * r * r * r;
        let rs7 = rs * rs * rs * rs * rs * rs * rs;
        let v = 1.12 * rs7 / (r7 + 0.12 * rs7);

        let energy = eps * u7 * (v - 2.0);

        let du_dr = -u / rho;
        let dv_dr = -7.0 * r * r * r * r * r * r * v / (r7 + 0.12 * rs7);
        // `7·u⁷/u` rather than `7·u⁶` was `0/0` for a zero vdW radius — a dummy
        // site, a lone pair, an unparameterised type — and one NaN force
        // poisons the whole trajectory in a single step. Same value, exactly,
        // one multiplication cheaper, and defined everywhere.
        let de_dr = eps * (7.0 * u6 * du_dr * (v - 2.0) + u7 * dv_dr);

        let factor = -de_dr / r;
        Some((energy, [factor * d[0], factor * d[1], factor * d[2]]))
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
        pair: impl Fn(usize) -> (usize, usize, F, F, [F; 3]) + Sync,
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
        pair: &(impl Fn(usize) -> (usize, usize, F, F, [F; 3]) + Sync),
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
            let (i, j, rs, eps, d) = pair(idx);
            let Some((e, f)) = self.pair_kernel(d, rs, eps) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            if VIRIAL {
                virial.add_outer(f, d);
            }
            for dim in 0..3 {
                out[j * 3 + dim] += f[dim];
                out[i * 3 + dim] -= f[dim];
            }
        }
        (energy, virial)
    }
}

impl Potential for PairMmffVdw {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let _n = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            r_star,
            epsilon,
        } = &self.source
        else {
            // Per-atom parameters need a pair table, and nobody handed one over.
            return;
        };
        self.fold_into(acc, &[], atom_i.len(), false, |idx| {
            let (i, j) = (atom_i[idx], atom_j[idx]);
            (i, j, r_star[idx], epsilon[idx], sub3(coords, j, coords, i))
        });
    }
}

impl PairDriven for PairMmffVdw {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        let Source::PerAtom { atoms, style, .. } = &self.source else {
            // A compiled kernel answers for its own list, not for this one.
            // A compiled kernel cannot read the table, so it cannot read a
            // per-pair weight either. Both providers refuse one, so this is
            // the free-boundary path and `factor` is empty.
            debug_assert!(factor.is_empty());
            self.evaluate(coords, acc);
            return;
        };
        let Some(disp) = pairs.disp() else {
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
                i < atoms.len() && j < atoms.len(),
                "a pair names an atom the per-atom parameters do not cover"
            );
            let (rs, eps) = vdw_combining(&atoms[i], &atoms[j], style);
            (i, j, rs, eps, [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]])
        });
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::PerAtom { atoms, n_owned, .. } = &mut self.source else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(atoms, *n_owned, owner);
    }
}

/// Per-atom VdW parameters for combining rules.
///
/// `alpha` is the atomic polarizability α (Å³), `n_eff` the Slater-Kirkwood
/// effective electron number N, `a_i` / `g_i` the MMFF scale factors A and G, and
/// `da` the hydrogen-bond role ([`DA_NEITHER`](crate::ff::params::mmff::DA_NEITHER) / [`DA_DONOR`] / [`DA_ACCEPTOR`]).
#[derive(Clone, Debug)]
pub struct PairMmffVdwAtomParams {
    /// Atomic polarizability α (Å³).
    pub alpha: f64,
    /// Slater-Kirkwood effective electron number N.
    pub n_eff: f64,
    /// MMFF scale factor A.
    pub a_i: f64,
    /// MMFF scale factor G.
    pub g_i: f64,
    /// Hydrogen-bond role: [`DA_NEITHER`](crate::ff::params::mmff::DA_NEITHER), [`DA_DONOR`] or [`DA_ACCEPTOR`].
    pub da: u8,
}

/// Style-level VdW constants (MMFF94.PAR's global vdW line, `<VdWParams>`).
///
/// `B` and `Beta` shape the R* combining rule; `DARAD` and `DAEPS` are the
/// donor-acceptor corrections. They are **data**, not constants of this kernel —
/// the defaults below are MMFF94's own values and exist only so a hand-built
/// style without the section still evaluates the standard force field.
#[derive(Clone, Debug)]
pub struct PairMmffVdwStyleParams {
    /// `B` in the R* combining rule.
    pub b: f64,
    /// `Beta` in the R* combining rule.
    pub beta: f64,
    /// Donor-acceptor radius correction `DARAD`.
    pub darad: f64,
    /// Donor-acceptor well-depth correction `DAEPS`.
    pub daeps: f64,
}

impl PairMmffVdwStyleParams {
    /// Read the global vdW line from a gathered style: a hand-built style
    /// that omits the section has MMFF94's own values, the spec's defaults.
    pub fn from_style(sp: &Params) -> Result<Self, crate::ff::ir::IrError> {
        let get = |key: &str| param_reads::style_num("mmff_vdw", sp, key);
        Ok(Self {
            b: get("B")?,
            beta: get("Beta")?,
            darad: get("DARAD")?,
            daeps: get("DAEPS")?,
        })
    }
}

/// MMFF's van der Waals combining rules (Halgren, *MMFF.II*, J. Comput. Chem.
/// **17**, 520–552 (1996), eqs. 5–7), returning `(R*_ij [Å], eps_ij [kcal/mol])`.
///
/// ```text
/// R*_ii = A_i * alpha_i^(1/4)
/// gamma = (R*_ii - R*_jj) / (R*_ii + R*_jj)
/// R*_ij = 0.5 * (R*_ii + R*_jj) * (1 + B * (1 - exp(-Beta * gamma^2)))
/// eps_ij = 181.16 * G_i * G_j * alpha_i * alpha_j
///          / ((sqrt(alpha_i/N_i) + sqrt(alpha_j/N_j)) * R*_ij^6)
/// ```
///
/// # The two donor/acceptor rules
///
/// Both are part of the *definition* of MMFF's vdW term, not an optional
/// refinement, and omitting them costs (measured, vs RDKit): urea +0.588, e_big
/// +1.261, N-methylacetamide +0.116 kcal/mol.
///
/// 1. **Donor R\* suppression** — if either partner is a donor, the `B` shape
///    term is dropped (`R*_ij` is the plain arithmetic mean). A polar hydrogen
///    approaches its acceptor far closer than the empirical expansion allows.
/// 2. **Donor-acceptor scaling** — for a donor *paired with* an acceptor,
///    `R*_ij *= DARAD` (0.8) and `eps_ij *= DAEPS` (0.5): the hydrogen bond is
///    shorter and softer than dispersion alone would give.
///
/// `eps_ij` is evaluated at the **unscaled** `R*_ij`, then scaled — the order is
/// load-bearing (`R*^-6` would otherwise pick up a spurious `DARAD^-6` = 3.8x).
fn vdw_combining(
    pi: &PairMmffVdwAtomParams,
    pj: &PairMmffVdwAtomParams,
    sp: &PairMmffVdwStyleParams,
) -> (F, F) {
    let rs_i = pi.a_i * pi.alpha.powf(0.25);
    let rs_j = pj.a_i * pj.alpha.powf(0.25);
    let gamma = (rs_i - rs_j) / (rs_i + rs_j);

    // Rule 1: a donor suppresses the B shape term entirely.
    let donor = pi.da == DA_DONOR || pj.da == DA_DONOR;
    let shape = if donor {
        0.0
    } else {
        sp.b * (1.0 - (-sp.beta * gamma * gamma).exp())
    };
    let mut r_star = 0.5 * (rs_i + rs_j) * (1.0 + shape);

    let denom = (pi.alpha / pi.n_eff).sqrt() + (pj.alpha / pj.n_eff).sqrt();
    let mut epsilon = 181.16 * pi.g_i * pj.g_i * pi.alpha * pj.alpha / (denom * r_star.powi(6));

    // Rule 2: donor-acceptor pairs get a shorter, softer minimum.
    let da_pair =
        (pi.da == DA_DONOR && pj.da == DA_ACCEPTOR) || (pi.da == DA_ACCEPTOR && pj.da == DA_DONOR);
    if da_pair {
        r_star *= sp.darad;
        epsilon *= sp.daeps;
    }
    (r_star as F, epsilon as F)
}

/// Build the buffered-14-7 van der Waals potential.
///
/// Style params (`sp`, from `<VdWParams>`): `B`, `Beta`, `DARAD`, `DAEPS`, plus
/// the `lj14scale` weight [`PotentialCompiler::compile`] projects out of the force
/// field's [`SpecialBonds`]. Type params (`tp`, from `<VdW>`): `alpha`, `n_eff`,
/// `a_i`, `g_i`, `da`.
///
/// # MMFF does not scale 1-4 van der Waals
///
/// `lj14scale` is **1.0** for MMFF (set by the reader) and applying it is a
/// deliberate no-op: MMFF's 1-4 rule touches electrostatics only (0.75), and its
/// torsion parameters were fitted against *unscaled* 1-4 vdW. Do not "fix" this
/// to 0.5 by analogy with Amber — it would corrupt every MMFF vdW energy with a
/// 1-4 pair. The weight is read (rather than ignored) so that a force field which
/// *does* scale can reuse this kernel, and so the 1.0 is visible as a choice.
///
/// [`SpecialBonds`]: crate::ff::ir::SpecialBonds
/// [`PotentialCompiler::compile`]: crate::ff::compile::PotentialCompiler::compile
pub fn pair_mmff_vdw_constructor(
    sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let style = PairMmffVdwStyleParams::from_style(sp)?;
    let lj_14 = sp.get("lj14scale").and_then(|v| v.as_num()).unwrap_or(1.0) as F;
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let atoms = frame.get(ATOMS).ok_or("mmff_vdw: missing \"atoms\"")?;
    let atom_types = atoms
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or("mmff_vdw: missing atom \"type\"")?;
    let pairs = frame.get(PAIRS).ok_or("mmff_vdw: missing \"pairs\"")?;
    let ic = pairs
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomi")?;
    let jc = pairs
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomj")?;
    let is_14 = pairs.get("is_14").and_then(|c| c.get::<bool>());

    let n = ic.len();
    let (mut ai, mut aj, mut rs_vec, mut eps_vec) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let ti = &atom_types[ic[idx] as usize];
        let tj = &atom_types[jc[idx] as usize];
        let pi = type_map
            .get(ti.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", ti))?;
        let pj = type_map
            .get(tj.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", tj))?;

        let to_vdw =
            |p: &Params, label: &str| -> Result<PairMmffVdwAtomParams, crate::ff::ir::IrError> {
                let get = |k: &str| param_reads::type_num("mmff_vdw", label, p, k);
                Ok(PairMmffVdwAtomParams {
                    alpha: get("alpha")?,
                    n_eff: get("n_eff")?,
                    a_i: get("a_i")?,
                    g_i: get("g_i")?,
                    da: get("da")? as u8,
                })
            };
        let (rs, eps) = vdw_combining(&to_vdw(pi, ti)?, &to_vdw(pj, tj)?, &style);
        // The 1-4 weight scales the well depth, hence the whole pair energy
        // (E is linear in eps). For MMFF this multiplies by exactly 1.0.
        let scale = if is_14.is_some_and(|b| b[idx]) {
            lj_14
        } else {
            1.0
        };
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        rs_vec.push(rs);
        eps_vec.push(eps * scale);
    }
    Ok(ForceTerm::pair(PairMmffVdw::compiled(
        ai, aj, rs_vec, eps_vec,
    )))
}

/// Construct a neighbour-driven [`PairMmffVdw`] from per-atom parameters.
///
/// The counterpart of [`pair_mmff_vdw_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_mmff_vdw_typed_constructor(
    sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let style = PairMmffVdwStyleParams::from_style(sp)?;
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let mut per_type = Vec::with_capacity(labels.len());
    for l in &labels {
        let p = type_map
            .get(l.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{l}'"))?;
        let get = |k: &str| param_reads::type_num("mmff_vdw", l, p, k);
        per_type.push(PairMmffVdwAtomParams {
            alpha: get("alpha")?,
            n_eff: get("n_eff")?,
            a_i: get("a_i")?,
            g_i: get("g_i")?,
            da: get("da")? as u8,
        });
    }
    let atoms = type_id
        .iter()
        .map(|&t| per_type[t as usize].clone())
        .collect();
    Ok(ForceTerm::pair(PairMmffVdw::typed(atoms, style)))
}

#[cfg(test)]
mod tests {

    /// Running Halgren's combining rules when a pair turns up is the same
    /// number as having run them earlier against a fixed list — bit for bit.
    #[test]
    fn per_atom_parameters_score_a_pair_exactly_as_compiled_ones() {
        use crate::ff::potential::pair::fixtures::{
            assert_same, assert_virial_matches_forces, table_over,
        };

        let style = PairMmffVdwStyleParams {
            b: 0.2,
            beta: 12.0,
            darad: 0.8,
            daeps: 0.5,
        };
        let atoms: Vec<PairMmffVdwAtomParams> = [
            (1.05, 2.490, 3.890, 1.282),
            (0.25, 0.800, 4.200, 1.209),
            (1.35, 2.490, 3.890, 1.282),
            (0.70, 3.150, 3.890, 1.282),
        ]
        .iter()
        .map(|&(alpha, n_eff, a_i, g_i)| PairMmffVdwAtomParams {
            alpha,
            n_eff,
            a_i,
            g_i,
            da: DA_NEITHER,
        })
        .collect();
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let mut rs = Vec::new();
        let mut eps = Vec::new();
        for &(i, j) in &links {
            let (r, e) = vdw_combining(&atoms[i], &atoms[j], &style);
            rs.push(r);
            eps.push(e);
        }
        let compiled = PairMmffVdw::compiled(ai, aj, rs, eps);
        let typed = PairMmffVdw::typed(atoms, style);

        let table = table_over(&coords, &links);
        assert_same(
            "mmff_vdw",
            compiled.energy_forces_at(&coords),
            typed.over_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "mmff_vdw",
            &coords,
            typed.over_pairs_virial(&coords, &table),
        );
    }
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    /// MMFF94's global vdW line (`<VdWParams B="0.2" Beta="12.0" DARAD="0.8"
    /// DAEPS="0.5">`), as the reader hands it to the kernel.
    fn mmff94_vdw_style() -> PairMmffVdwStyleParams {
        PairMmffVdwStyleParams::from_style(&Params::from_iter([
            ("B", 0.2),
            ("Beta", 12.0),
            ("DARAD", 0.8),
            ("DAEPS", 0.5),
        ]))
        .unwrap()
    }

    #[test]
    fn test_mmff_vdw_combining() {
        let p = PairMmffVdwAtomParams {
            alpha: 1.050,
            n_eff: 2.490,
            a_i: 3.890,
            g_i: 1.282,
            da: DA_NEITHER,
        };
        let (rs, eps) = vdw_combining(&p, &p, &mmff94_vdw_style());
        assert!(rs > 0.0, "r_star should be positive");
        assert!(eps > 0.0, "epsilon should be positive");
    }

    #[test]
    fn test_mmff_vdw_energy() {
        let pot = PairMmffVdw::compiled(vec![0], vec![1], vec![1.94], vec![0.02]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "dim {}: sum = {}", dim, sum);
        }
    }
}
