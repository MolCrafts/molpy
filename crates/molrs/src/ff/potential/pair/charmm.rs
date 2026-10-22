//! The two halves of LAMMPS's `pair_style lj/charmm/coul/charmm`: `lj/charmm` and `coul/charmm`.

use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::IrError;
use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::{atom_type_index, fold_pairs, type_pair};
use crate::ff::potential::{
    Accumulator, CompileError, ForceTerm, PairDriven, Potential, param_reads,
};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

const MIN_R2: F = 1e-24;

/// CHARMM's energy switch between `inner` and `cutoff`.
#[derive(Clone, Copy, Debug)]
struct Switch {
    inner2: F,
    outer2: F,
    denom: F,
}

impl Switch {
    fn new(inner: F, outer: F, style: &str) -> Result<Self, IrError> {
        if !(inner.is_finite() && outer.is_finite() && inner > 0.0) {
            return Err(param_reads::bad(
                style,
                "",
                "inner",
                format!("= {inner} with cutoff = {outer}: both must be finite and positive"),
            ));
        }
        // LAMMPS: "Pair inner cutoff >= Pair outer cutoff".
        if inner >= outer {
            return Err(param_reads::bad(
                style,
                "",
                "inner",
                format!("= {inner} is not below the outer cutoff {outer}"),
            ));
        }
        let (inner2, outer2) = (inner * inner, outer * outer);
        let span = outer2 - inner2;
        Ok(Self {
            inner2,
            outer2,
            denom: span * span * span,
        })
    }

    /// LAMMPS's `switch1`.
    #[inline]
    fn s1(&self, rsq: F) -> F {
        (self.outer2 - rsq) * (self.outer2 - rsq) * (self.outer2 + 2.0 * rsq - 3.0 * self.inner2)
            / self.denom
    }

    /// LAMMPS's `switch2` (= −r·dS/dr).
    #[inline]
    fn s2(&self, rsq: F) -> F {
        12.0 * rsq * (self.outer2 - rsq) * (rsq - self.inner2) / self.denom
    }
}

/// `inner` and `cutoff` of a CHARMM style; both are required (LAMMPS's
/// inner and outer switching cutoffs are part of the CHARMM energy).
fn switch_of(params: &Params, style: &str) -> Result<Switch, IrError> {
    let get = |key: &str| param_reads::style_num(style, params, key);
    Switch::new(get("inner")?, get("cutoff")?, style)
}

/// LAMMPS's `lj1..lj4` for one `(ε, σ)`: `48εσ¹²`, `24εσ⁶`, `4εσ¹²`, `4εσ⁶`.
pub(crate) fn lj_coeffs(epsilon: F, sigma: F) -> [F; 4] {
    let s12 = sigma.powf(12.0);
    let s6 = sigma.powf(6.0);
    [
        48.0 * epsilon * s12,
        24.0 * epsilon * s6,
        4.0 * epsilon * s12,
        4.0 * epsilon * s6,
    ]
}

/// The mixing rule of a CHARMM style: declared, or the IR's (and
/// LAMMPS's) default [`CombiningRule::UNDECLARED`], `arithmetic`.
pub(crate) fn charmm_mixing(style: &Params) -> Result<CombiningRule, IrError> {
    match style.get("mixing").and_then(|v| v.as_text()) {
        Some(name) => {
            CombiningRule::parse(name).map_err(|e| param_reads::bad("lj/charmm", "", "mixing", e))
        }
        None => Ok(CombiningRule::UNDECLARED),
    }
}

/// `((ε, σ), (ε₁₄, σ₁₄))`: a pair's regular and 1-4 Lennard-Jones parameters.
pub(crate) type CharmmParams = ((F, F), (F, F));

/// `((ε, σ), (ε₁₄, σ₁₄))` of one `lj/charmm` row.
fn charmm_row(p: &Params, key: &str) -> Result<CharmmParams, IrError> {
    let need = |k: &str| param_reads::type_num("lj/charmm", key, p, k);
    let (eps, sigma) = (need("epsilon")?, need("sigma")?);
    let eps14 = p
        .get("epsilon14")
        .and_then(|v| v.as_num())
        .map(|v| v as F)
        .unwrap_or(eps);
    let sigma14 = p
        .get("sigma14")
        .and_then(|v| v.as_num())
        .map(|v| v as F)
        .unwrap_or(sigma);
    Ok(((eps, sigma), (eps14, sigma14)))
}

/// The regular and 1-4 `(ε, σ)` of the atom-type pair `(a, b)` under an
/// `lj/charmm` style: its explicit cross row when it has one, the two self
/// rows mixed by `mixing` otherwise — LAMMPS's `init_one`.
pub(crate) fn charmm_pair_params(
    rows: &HashMap<&str, &Params>,
    mixing: CombiningRule,
    a: &str,
    b: &str,
) -> Result<CharmmParams, CompileError> {
    if a != b {
        let key = pair_key(a, b)?;
        if let Some(p) = rows.get(key.as_str()) {
            return Ok(charmm_row(p, &key)?);
        }
    }
    let own = |t: &str| -> Result<CharmmParams, CompileError> {
        let p = rows
            .get(t)
            .ok_or_else(|| format!("lj/charmm: unknown atom type '{t}'"))?;
        Ok(charmm_row(p, t)?)
    };
    let (ra, r14a) = own(a)?;
    let (rb, r14b) = own(b)?;
    Ok((mixing.combine(ra, rb), mixing.combine(r14a, r14b)))
}

/// Where a `lj/charmm` pair's coefficients come from.
#[derive(Clone, Debug)]
enum LjSource {
    /// Resolved against one fixed pair list, with each row's weight.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        coeffs: Vec<[F; 4]>,
        weight: Vec<F>,
    },
    /// A type-pair table of `lj1..lj4`, keyed by the atoms' types.
    Typed {
        type_id: Vec<u32>,
        ntypes: usize,
        coeffs: Vec<[F; 4]>,
        n_owned: usize,
    },
}

/// LAMMPS `pair_style lj/charmm/coul/charmm`, van-der-Waals half.
///
/// The two halves of LAMMPS's `pair_style lj/charmm/coul/charmm`, as molrs
/// splits `lj/cut/coul/cut` into `lj/cut` + `coul/cut`:
///
/// * `pair lj/charmm` — Lennard-Jones 12-6 with CHARMM's energy switch,
///   per type `epsilon`, `sigma`, `epsilon14`, `sigma14`;
/// * `pair coul/charmm` — Coulomb with the same switch.
///
/// Both follow `pair_lj_charmm_coul_charmm.cpp` term for term. With
/// `r_in` = `inner`, `r_c` = `cutoff` and
///
/// ```text
/// S(r)  = (r_c² − r²)² (r_c² + 2r² − 3r_in²) / (r_c² − r_in²)³     r_in < r < r_c
/// ```
///
/// (1 below `r_in`, 0 from `r_c` on):
///
/// ```text
/// E_lj   = 4ε[(σ/r)¹² − (σ/r)⁶] · S(r)        F_lj = −dE_lj/dr  (consistent)
/// E_coul = C qᵢqⱼ / r · S(r)                   F_coul = C qᵢqⱼ / r² · S(r)
/// ```
///
/// `epsilon14` / `sigma14` (absent → `epsilon` / `sigma`, as a two-number
/// `pair_coeff` line in LAMMPS) are not used by this kernel: LAMMPS prices
/// them only inside `dihedral_style charmm`, which molrs routes to the 1-4
/// exceptions kernel ([`PairExceptions`](super::PairExceptions)). A 1-4 pair this style meets on
/// the `pairs` list is weighted by `special_bonds` with the regular `epsilon`
/// / `sigma`, as LAMMPS's pair style does.
///
/// Cross pairs: an explicit cross row, else the style's `mixing` (LAMMPS's
/// default for the CHARMM styles is `arithmetic`); `epsilon14` / `sigma14`
/// mix the same way, as LAMMPS's `init_one` does.
///
/// The switch is part of the style's energy, so **both** compile doors apply
/// it: the compiled (pair-list) form prices a pair at or beyond `cutoff` at
/// zero, as LAMMPS does — as every pair style's compiled form truncates at
/// its `cutoff`.
#[derive(Clone, Debug)]
pub struct PairLjCharmm {
    switch: Switch,
    source: LjSource,
}

impl PairLjCharmm {
    /// A kernel over a fixed pair list: one `[ε, σ]` and one weight per row.
    pub fn compiled(
        inner: F,
        cutoff: F,
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        eps_sigma: &[(F, F)],
        weight: Vec<F>,
    ) -> Result<Self, CompileError> {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), eps_sigma.len());
        assert_eq!(atom_i.len(), weight.len());
        Ok(Self {
            switch: Switch::new(inner, cutoff, "lj/charmm")?,
            source: LjSource::Compiled {
                atom_i,
                atom_j,
                coeffs: eps_sigma.iter().map(|&(e, s)| lj_coeffs(e, s)).collect(),
                weight,
            },
        })
    }

    /// A kernel keyed on the atoms: `type_id` per atom and a full
    /// `ntypes × ntypes` table of `(ε, σ)`, laid out `ti * ntypes + tj`.
    pub fn typed(
        inner: F,
        cutoff: F,
        type_id: Vec<u32>,
        table: &[(F, F)],
    ) -> Result<Self, CompileError> {
        let ntypes = (table.len() as f64).sqrt().round() as usize;
        if ntypes * ntypes != table.len() || ntypes == 0 {
            return Err("lj/charmm: the type-pair table must be square and non-empty".into());
        }
        if type_id.iter().any(|&t| t as usize >= ntypes) {
            return Err(
                format!("lj/charmm: an atom type is outside the {ntypes} tabulated").into(),
            );
        }
        let n_owned = type_id.len();
        Ok(Self {
            switch: Switch::new(inner, cutoff, "lj/charmm")?,
            source: LjSource::Typed {
                type_id,
                ntypes,
                coeffs: table.iter().map(|&(e, s)| lj_coeffs(e, s)).collect(),
                n_owned,
            },
        })
    }

    /// `(energy, fpair)` of one pair, `fpair` the LAMMPS force factor (force
    /// on `j` is `fpair · (xⱼ − xᵢ)`).
    #[inline]
    fn eval(&self, rsq: F, c: &[F; 4]) -> Option<(F, F)> {
        if !(MIN_R2..self.switch.outer2).contains(&rsq) {
            return None;
        }
        let r2inv = 1.0 / rsq;
        let r6inv = r2inv * r2inv * r2inv;
        let mut forcelj = r6inv * (c[0] * r6inv - c[1]);
        let mut philj = r6inv * (c[2] * r6inv - c[3]);
        if rsq > self.switch.inner2 {
            let s1 = self.switch.s1(rsq);
            forcelj = forcelj * s1 + philj * self.switch.s2(rsq);
            philj *= s1;
        }
        Some((philj, forcelj * r2inv))
    }

    fn fold_compiled(&self, coords: &[F], out: &mut [F]) -> F {
        let LjSource::Compiled {
            atom_i,
            atom_j,
            coeffs,
            weight,
        } = &self.source
        else {
            return 0.0;
        };
        validate_coords(coords);
        let mut energy = 0.0;
        for idx in 0..atom_i.len() {
            let w = weight[idx];
            if w == 0.0 {
                continue;
            }
            let (i, j) = (atom_i[idx], atom_j[idx]);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let Some((e, fpair)) = self.eval(rsq, &coeffs[idx]) else {
                continue;
            };
            energy += w * e;
            scatter(out, i, j, w * fpair, d);
        }
        energy
    }

    fn fold_typed(&self, acc: &mut Accumulator<'_>, factor: &[F], pairs: &Neighbors) {
        if pairs.disp().is_none() || pairs.dist_sq().is_none() {
            // No displacements, nothing priced — and no virial claimed for it.
            acc.forces_without_virial();
            return;
        }
        fold_pairs(
            acc,
            pairs.query_point_indices().len(),
            |out: &mut [F], rows| self.fold_typed_rows::<false>(out, factor, pairs, rows),
            |out: &mut [F], rows| self.fold_typed_rows::<true>(out, factor, pairs, rows),
        );
    }

    /// One contiguous range of the table's pairs, into `out`.
    fn fold_typed_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut energy = 0.0;
        let mut virial = Virial::ZERO;
        let LjSource::Typed {
            type_id,
            ntypes,
            coeffs,
            ..
        } = &self.source
        else {
            return (energy, virial);
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (energy, virial);
        };
        let (ic, jc) = (pairs.query_point_indices(), pairs.point_indices());
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            if w == 0.0 {
                continue;
            }
            let (i, j) = (ic[p] as usize, jc[p] as usize);
            let t = type_pair(type_id[i], type_id[j], *ntypes);
            let Some((e, fpair)) = self.eval(d2[p], &coeffs[t]) else {
                continue;
            };
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            energy += w * e;
            let f = w * fpair;
            if VIRIAL {
                virial.add_outer([f * d[0], f * d[1], f * d[2]], d);
            }
            scatter(out, i, j, f, d);
        }
        (energy, virial)
    }
}

/// Add the pair force `f·d` on `j` and its reaction on `i`.
#[inline]
fn scatter(out: &mut [F], i: usize, j: usize, f: F, d: [F; 3]) {
    for k in 0..3 {
        out[j * 3 + k] += f * d[k];
        out[i * 3 + k] -= f * d[k];
    }
}

/// Where a `coul/charmm` pair's charge product comes from.
#[derive(Clone, Debug)]
enum CoulSource {
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
        weight: Vec<F>,
    },
    PerAtom {
        q: Vec<F>,
        n_owned: usize,
    },
}

/// LAMMPS `pair_style lj/charmm/coul/charmm`, Coulomb half.
///
/// The Coulomb force is LAMMPS's: the switched force, **not** the gradient of
/// the switched energy (LAMMPS's `forcecoul *= switch1` drops the `E·S′`
/// term). Inside `r_in` the two agree; between `r_in` and `r_c` they differ,
/// exactly as in LAMMPS. The Lennard-Jones force is the true gradient
/// (LAMMPS adds `philj · switch2`).
///
/// The switch `S(r)` and the cross-pair rules are [`PairLjCharmm`]'s.
#[derive(Clone, Debug)]
pub struct PairCoulCharmm {
    switch: Switch,
    /// `coulomb / dielectric` — LAMMPS's `qqrd2e`.
    k: F,
    source: CoulSource,
}

impl PairCoulCharmm {
    /// A kernel over a fixed pair list, one `qᵢqⱼ` and one weight per row.
    pub fn compiled(
        inner: F,
        cutoff: F,
        k: F,
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
        weight: Vec<F>,
    ) -> Result<Self, CompileError> {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qiqj.len());
        assert_eq!(atom_i.len(), weight.len());
        Ok(Self {
            switch: Switch::new(inner, cutoff, "coul/charmm")?,
            k,
            source: CoulSource::Compiled {
                atom_i,
                atom_j,
                qiqj,
                weight,
            },
        })
    }

    /// A kernel that forms `qᵢqⱼ` from per-atom charges.
    pub fn typed(inner: F, cutoff: F, k: F, q: Vec<F>) -> Result<Self, CompileError> {
        let n_owned = q.len();
        Ok(Self {
            switch: Switch::new(inner, cutoff, "coul/charmm")?,
            k,
            source: CoulSource::PerAtom { q, n_owned },
        })
    }

    #[inline]
    fn eval(&self, rsq: F, qiqj: F) -> Option<(F, F)> {
        if !(MIN_R2..self.switch.outer2).contains(&rsq) {
            return None;
        }
        let r2inv = 1.0 / rsq;
        let mut forcecoul = self.k * qiqj * r2inv.sqrt();
        let mut ecoul = forcecoul;
        if rsq > self.switch.inner2 {
            // LAMMPS switches the force as it switches the energy; the
            // `E·S′` term of the energy's gradient is not in it.
            let s1 = self.switch.s1(rsq);
            forcecoul *= s1;
            ecoul *= s1;
        }
        Some((ecoul, forcecoul * r2inv))
    }

    fn fold_compiled(&self, coords: &[F], out: &mut [F]) -> F {
        let CoulSource::Compiled {
            atom_i,
            atom_j,
            qiqj,
            weight,
        } = &self.source
        else {
            return 0.0;
        };
        validate_coords(coords);
        let mut energy = 0.0;
        for idx in 0..atom_i.len() {
            let w = weight[idx];
            if w == 0.0 {
                continue;
            }
            let (i, j) = (atom_i[idx], atom_j[idx]);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let Some((e, fpair)) = self.eval(rsq, qiqj[idx]) else {
                continue;
            };
            energy += w * e;
            scatter(out, i, j, w * fpair, d);
        }
        energy
    }

    fn fold_typed(&self, acc: &mut Accumulator<'_>, factor: &[F], pairs: &Neighbors) {
        if pairs.disp().is_none() || pairs.dist_sq().is_none() {
            // No displacements, nothing priced — and no virial claimed for it.
            acc.forces_without_virial();
            return;
        }
        fold_pairs(
            acc,
            pairs.query_point_indices().len(),
            |out: &mut [F], rows| self.fold_typed_rows::<false>(out, factor, pairs, rows),
            |out: &mut [F], rows| self.fold_typed_rows::<true>(out, factor, pairs, rows),
        );
    }

    /// One contiguous range of the table's pairs, into `out`.
    fn fold_typed_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut energy = 0.0;
        let mut virial = Virial::ZERO;
        let CoulSource::PerAtom { q, .. } = &self.source else {
            return (energy, virial);
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (energy, virial);
        };
        let (ic, jc) = (pairs.query_point_indices(), pairs.point_indices());
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            if w == 0.0 {
                continue;
            }
            let (i, j) = (ic[p] as usize, jc[p] as usize);
            let Some((e, fpair)) = self.eval(d2[p], q[i] * q[j]) else {
                continue;
            };
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            energy += w * e;
            let f = w * fpair;
            if VIRIAL {
                virial.add_outer([f * d[0], f * d[1], f * d[2]], d);
            }
            scatter(out, i, j, f, d);
        }
        (energy, virial)
    }
}

macro_rules! pair_member_impls {
    ($ty:ty, $compiled:path) => {
        impl Potential for $ty {
            fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
                // A typed kernel needs a pair table nobody handed over.
                let energy = self.fold_compiled(coords, acc.forces_without_virial());
                acc.add_energy(energy);
            }
        }

        impl PairDriven for $ty {
            fn evaluate_pairs(
                &self,
                coords: &[F],
                pairs: &Neighbors,
                factor: &[F],
                acc: &mut Accumulator<'_>,
            ) {
                if self.binds_a_fixed_pair_list() {
                    // A compiled list cannot read a per-pair weight, and a
                    // virial from its raw differences is about nothing.
                    debug_assert!(factor.is_empty());
                    self.evaluate(coords, acc);
                    return;
                }
                self.fold_typed(acc, factor, pairs);
            }

            fn binds_a_fixed_pair_list(&self) -> bool {
                matches!(self.source, $compiled { .. })
            }

            fn gather_onto_copies(&mut self, owner: &[u32]) {
                self.gather(owner);
            }
        }
    };
}

pair_member_impls!(PairLjCharmm, LjSource::Compiled);
pair_member_impls!(PairCoulCharmm, CoulSource::Compiled);

impl PairLjCharmm {
    fn gather(&mut self, owner: &[u32]) {
        if let LjSource::Typed {
            type_id, n_owned, ..
        } = &mut self.source
        {
            gather_copies(type_id, *n_owned, owner);
        }
    }
}

impl PairCoulCharmm {
    fn gather(&mut self, owner: &[u32]) {
        if let CoulSource::PerAtom { q, n_owned } = &mut self.source {
            gather_copies(q, *n_owned, owner);
        }
    }
}

/// A `pairs` block's `atomi`, `atomj` and `is_14` columns.
type PairRows = (Vec<usize>, Vec<usize>, Option<Vec<bool>>);

/// The `pairs` block's `(atomi, atomj, is_14)`.
fn pair_rows(frame: &Frame, who: &str) -> Result<PairRows, String> {
    let block = frame
        .get(PAIRS)
        .ok_or_else(|| format!("{who}: frame missing \"pairs\" block"))?;
    let col = |k: &str| {
        block
            .get(k)
            .and_then(|c| c.get::<u64>())
            .map(|c| c.iter().map(|&v| v as usize).collect::<Vec<_>>())
            .ok_or_else(|| format!("{who}: pairs block missing \"{k}\" column"))
    };
    let is_14 = block
        .get("is_14")
        .and_then(|c| c.get::<bool>())
        .map(|c| c.iter().copied().collect());
    Ok((col("atomi")?, col("atomj")?, is_14))
}

/// The row weight a compiled list carries: the projected 1-4 weight on an
/// `is_14` row, 1 elsewhere (1-2 / 1-3 rows are present only at weight 1).
fn row_weights(n: usize, is_14: &Option<Vec<bool>>, w14: F) -> Vec<F> {
    (0..n)
        .map(|r| match is_14 {
            Some(f) if f[r] => w14,
            _ => 1.0,
        })
        .collect()
}

fn atom_types(frame: &Frame, who: &str) -> Result<Vec<String>, String> {
    frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
        .map(|c| c.iter().cloned().collect())
        .ok_or_else(|| format!("{who}: atoms block missing \"type\" column"))
}

/// Construct a compiled `lj/charmm` from the frame's `pairs` block.
pub fn pair_lj_charmm_constructor(
    style: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let switch = switch_of(style, "lj/charmm")?;
    let mixing = charmm_mixing(style)?;
    let rows: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let types = atom_types(frame, "lj/charmm")?;
    let (ai, aj, is_14) = pair_rows(frame, "lj/charmm")?;
    let mut eps_sigma = Vec::with_capacity(ai.len());
    for (&i, &j) in ai.iter().zip(&aj) {
        eps_sigma.push(charmm_pair_params(&rows, mixing, &types[i], &types[j])?.0);
    }
    let weight = row_weights(
        ai.len(),
        &is_14,
        style
            .get("lj14scale")
            .and_then(|v| v.as_num())
            .unwrap_or(1.0),
    );
    let kernel = PairLjCharmm {
        switch,
        source: LjSource::Compiled {
            atom_i: ai,
            atom_j: aj,
            coeffs: eps_sigma.iter().map(|&(e, s)| lj_coeffs(e, s)).collect(),
            weight,
        },
    };
    Ok(ForceTerm::pair(kernel))
}

/// Construct a neighbour-driven `lj/charmm`, keyed on the atoms' types.
pub fn pair_lj_charmm_typed_constructor(
    style: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let switch = switch_of(style, "lj/charmm")?;
    let mixing = charmm_mixing(style)?;
    let rows: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut coeffs = Vec::with_capacity(ntypes * ntypes);
    for a in &labels {
        for b in &labels {
            let (e, s) = charmm_pair_params(&rows, mixing, a, b)?.0;
            coeffs.push(lj_coeffs(e, s));
        }
    }
    let n_owned = type_id.len();
    let kernel = PairLjCharmm {
        switch,
        source: LjSource::Typed {
            type_id,
            ntypes,
            coeffs,
            n_owned,
        },
    };
    Ok(ForceTerm::pair(kernel))
}

/// `coulomb / dielectric`: `coulomb` the force field's to state, `dielectric`
/// gathered with its declared default (1, LAMMPS's).
fn coulomb_constant(style: &Params) -> Result<F, IrError> {
    let need = |k: &str| param_reads::style_num("coul/charmm", style, k);
    Ok(need("coulomb")? / need("dielectric")?)
}

fn charges(frame: &Frame) -> Result<Vec<F>, String> {
    frame
        .get(ATOMS)
        .and_then(|b| b.get("charge"))
        .and_then(|c| c.get::<f64>())
        .map(|c| c.iter().map(|&v| v as F).collect())
        .ok_or_else(|| "coul/charmm: atoms block missing \"charge\" column".to_string())
}

/// Construct a compiled `coul/charmm` from per-atom charges and `pairs`.
pub fn pair_coul_charmm_constructor(
    style: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let switch = switch_of(style, "coul/charmm")?;
    let k = coulomb_constant(style)?;
    let q = charges(frame)?;
    let (ai, aj, is_14) = pair_rows(frame, "coul/charmm")?;
    let qiqj = ai.iter().zip(&aj).map(|(&i, &j)| q[i] * q[j]).collect();
    let weight = row_weights(
        ai.len(),
        &is_14,
        style
            .get("coulomb14scale")
            .and_then(|v| v.as_num())
            .unwrap_or(1.0),
    );
    let kernel = PairCoulCharmm {
        switch,
        k,
        source: CoulSource::Compiled {
            atom_i: ai,
            atom_j: aj,
            qiqj,
            weight,
        },
    };
    Ok(ForceTerm::pair(kernel))
}

/// Construct a neighbour-driven `coul/charmm` from per-atom charges.
pub fn pair_coul_charmm_typed_constructor(
    style: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let switch = switch_of(style, "coul/charmm")?;
    let k = coulomb_constant(style)?;
    let q = charges(frame)?;
    let n_owned = q.len();
    let kernel = PairCoulCharmm {
        switch,
        k,
        source: CoulSource::PerAtom { q, n_owned },
    };
    Ok(ForceTerm::pair(kernel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::pair::fixtures::{
        assert_same, assert_virial_matches_forces, table_over,
    };

    /// LAMMPS's switch, written out from the manual's formula.
    fn s(r: F, ri: F, rc: F) -> F {
        if r <= ri {
            1.0
        } else if r >= rc {
            0.0
        } else {
            let (r2, ri2, rc2) = (r * r, ri * ri, rc * rc);
            (rc2 - r2).powi(2) * (rc2 + 2.0 * r2 - 3.0 * ri2) / (rc2 - ri2).powi(3)
        }
    }

    fn lj(eps: F, sigma: F, r: F) -> F {
        let x = (sigma / r).powi(6);
        4.0 * eps * (x * x - x)
    }

    fn two(r: F) -> Vec<F> {
        vec![0.0, 0.0, 0.0, r, 0.0, 0.0]
    }

    #[test]
    fn lj_hand_values_inside_across_and_beyond_the_switch() {
        let (eps, sigma, ri, rc) = (0.2, 3.1, 4.0, 6.0);
        let k =
            PairLjCharmm::compiled(ri, rc, vec![0], vec![1], &[(eps, sigma)], vec![1.0]).unwrap();
        for r in [3.0, 3.9, 4.5, 5.2, 5.99, 6.0, 7.0] {
            let want = lj(eps, sigma, r) * s(r, ri, rc);
            let got = k.energy_at(&two(r));
            assert!(
                (got - want).abs() <= 1e-13 * want.abs().max(1e-12),
                "r = {r}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn lj_force_is_the_gradient_through_the_switch() {
        let k =
            PairLjCharmm::compiled(4.0, 6.0, vec![0], vec![1], &[(0.2, 3.1)], vec![0.7]).unwrap();
        for r in [3.3, 4.6, 5.5] {
            let (_, f) = k.energy_forces_at(&two(r));
            let h = 1e-6;
            let fd = -(k.energy_at(&two(r + h)) - k.energy_at(&two(r - h))) / (2.0 * h);
            assert!((f[3] - fd).abs() < 1e-7, "r = {r}: {} vs {fd}", f[3]);
            assert!((f[0] + f[3]).abs() < 1e-12);
        }
    }

    #[test]
    fn coulomb_hand_values_and_lammps_switched_force() {
        let (ri, rc, k, qq) = (4.0, 6.0, 332.06371, -0.3);
        let pot =
            PairCoulCharmm::compiled(ri, rc, k, vec![0], vec![1], vec![qq], vec![1.0]).unwrap();
        for r in [2.5, 4.5, 5.5, 6.5] {
            let (e, f) = pot.energy_forces_at(&two(r));
            let sw = s(r, ri, rc);
            let want = k * qq / r * sw;
            assert!(
                (e - want).abs() <= 1e-13 * want.abs().max(1e-12),
                "{e} {want}"
            );
            // LAMMPS's force: k qq / r² · S, the switched force.
            let want_f = k * qq / (r * r) * sw;
            assert!((f[3] - want_f).abs() <= 1e-12 * want_f.abs().max(1e-12));
        }
        // Inside `inner` the force is the gradient.
        let h = 1e-6;
        let fd = -(pot.energy_at(&two(3.0 + h)) - pot.energy_at(&two(3.0 - h))) / (2.0 * h);
        assert!((pot.energy_forces_at(&two(3.0)).1[3] - fd).abs() < 1e-7);
    }

    #[test]
    fn inner_must_be_below_the_cutoff() {
        assert!(PairCoulCharmm::typed(6.0, 6.0, 1.0, vec![]).is_err());
        assert!(PairLjCharmm::typed(7.0, 6.0, vec![], &[(0.1, 3.0)]).is_err());
        let style = Params::from_iter([("cutoff", 10.0)]);
        let err = switch_of(&style, "lj/charmm").unwrap_err();
        assert_eq!(
            err,
            IrError::MissingParam {
                style: "lj/charmm".into(),
                type_: String::new(),
                param: "inner".into()
            }
        );
    }

    /// The compiled list and the type-pair table are the same numbers on the
    /// same pairs, bit for bit, weights included.
    #[test]
    fn typed_scores_a_pair_exactly_as_compiled() {
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            3.6, 0.4, 0.2, //
            1.2, 4.9, 0.7, //
            4.0, 3.3, 1.1,
        ];
        let type_id = vec![0_u32, 1, 0, 1];
        let per = [(0.3, 3.4), (0.1, 2.6)];
        let mut table = Vec::new();
        for a in per {
            for b in per {
                table.push(CombiningRule::Arithmetic.combine(a, b));
            }
        }
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3), (0, 3)];
        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let es: Vec<(F, F)> = links
            .iter()
            .map(|&(i, j)| table[type_id[i] as usize * 2 + type_id[j] as usize])
            .collect();
        let q = vec![0.4, -0.7, 0.3, -0.2];
        let qq = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let ones = vec![1.0; links.len()];
        let lj_c =
            PairLjCharmm::compiled(3.5, 5.0, ai.clone(), aj.clone(), &es, ones.clone()).unwrap();
        let lj_t = PairLjCharmm::typed(3.5, 5.0, type_id, &table).unwrap();
        let co_c = PairCoulCharmm::compiled(3.5, 5.0, 332.0, ai, aj, qq, ones).unwrap();
        let co_t = PairCoulCharmm::typed(3.5, 5.0, 332.0, q).unwrap();
        let nb = table_over(&coords, &links);
        assert_same(
            "lj/charmm",
            lj_c.energy_forces_at(&coords),
            lj_t.over_pairs(&coords, &nb),
        );
        assert_same(
            "coul/charmm",
            co_c.energy_forces_at(&coords),
            co_t.over_pairs(&coords, &nb),
        );
        assert_virial_matches_forces("lj/charmm", &coords, lj_t.over_pairs_virial(&coords, &nb));
    }

    #[test]
    fn epsilon14_mixes_like_epsilon_and_a_cross_row_wins() {
        let a = Params::from_iter([
            ("epsilon", 0.1),
            ("sigma", 3.0),
            ("epsilon14", 0.05),
            ("sigma14", 2.8),
        ]);
        let b = Params::from_iter([("epsilon", 0.4), ("sigma", 3.6)]);
        let ab = Params::from_iter([("epsilon", 0.9), ("sigma", 2.0)]);
        let key = pair_key("A", "B").unwrap();
        let mut rows: HashMap<&str, &Params> = HashMap::new();
        rows.insert("A", &a);
        rows.insert("B", &b);
        let (reg, r14) = charmm_pair_params(&rows, CombiningRule::Arithmetic, "A", "B").unwrap();
        assert_eq!(reg, ((0.1_f64 * 0.4).sqrt(), 3.3));
        assert_eq!(r14, ((0.05_f64 * 0.4).sqrt(), 0.5 * (2.8 + 3.6)));
        rows.insert(key.as_str(), &ab);
        let (reg, r14) = charmm_pair_params(&rows, CombiningRule::Arithmetic, "B", "A").unwrap();
        assert_eq!(reg, (0.9, 2.0));
        assert_eq!(r14, (0.9, 2.0), "a 2-number cross row is its own 1-4 row");
    }
}
