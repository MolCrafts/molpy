//! Unique LJ / Mie pair kernel (`lj/cut`).

use molrs::core::schema::block_names::{ATOMS, PAIRS};
use std::collections::HashMap;

use crate::ff::ir::CombiningRule;
use crate::ff::ir::IrError;
use crate::ff::ir::{Params, pair_key};
use crate::ff::potential::flat_coords::validate_coords;
use crate::ff::potential::gather_copies;
use crate::ff::potential::pair::PairPotential;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::fold_pairs;
use crate::ff::potential::{
    Accumulator, CompileError, ForceTerm, PairDriven, Potential, param_reads,
};
use molrs::core::Frame;
use molrs::core::Neighbors;
use molrs::core::Virial;
use molrs::op::F;

const MIN_R2: F = 1e-24;

#[derive(Clone, Debug)]
enum PairSource {
    Loop,
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
        /// Per pair, the energy at the cutoff a shifted style subtracts
        /// (LAMMPS's `offset`); zero unshifted.
        e0: Vec<F>,
    },
    /// Per-atom types plus a type-pair table.
    ///
    /// The difference from `Compiled` is *when* a parameter is chosen. A
    /// compiled kernel resolved its parameters against a pair list at
    /// construction, which is only meaningful while that exact list is the one
    /// being evaluated — and a neighbour table is rebuilt from scratch every
    /// few steps, with different rows in a different order. Keyed on the atoms
    /// instead, a parameter can be found for whatever pair turns up, including
    /// a pair that involves a periodic copy.
    Typed {
        /// Type index per atom. Under a ghost régime this covers the copies
        /// too, gathered from their owners.
        type_id: Vec<u32>,
        ntypes: usize,
        /// Flattened `ti * ntypes + tj`, each already mixed and each carrying
        /// the cutoff-dependent constants that go with it.
        sigma: Vec<F>,
        ceps: Vec<F>,
        e0: Vec<F>,
        f_rc: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// The cutoff-dependent constants for one mixed `(ε, σ)`.
///
/// Shared by the style-level constructor and the type-pair table so the two
/// cannot disagree about what shifting means.
fn shift_constants(
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
) -> (F, F, F) {
    let ceps = mie_c(n, m) * epsilon;
    let sr_c = sigma / cutoff;
    let sr_n_c = sr_c.powi(n);
    let sr_m_c = sr_c.powi(m);
    let u_c = ceps * (sr_n_c - sr_m_c);
    let fac_c = ceps * ((n as F) * sr_n_c - (m as F) * sr_m_c) / (cutoff * cutoff);
    let shift_energy = shifted || smeared;
    (
        ceps,
        if shift_energy { u_c } else { 0.0 },
        if smeared { fac_c * cutoff } else { 0.0 },
    )
}

/// LAMMPS `pair_style lj/cut`.
///
/// Pair source is fixed at construction:
/// - [`PairLjCut::new`] / [`PairLjCut::lj126`] — uniform ε/σ, loop-fed pairs (MD)
/// - [`PairLjCut::compiled`] — per-pair ε/σ from a ForceField `pairs` block,
///   truncated (and shifted) at the style's `cutoff` by [`PairLjCut::truncated`]
///
/// Every source prices a pair only at `r < cutoff`, as LAMMPS does.
/// Arithmetic uses `inv_r2 = 1/r2`. Degenerate pairs `r2 < 1e-24` are skipped.
#[derive(Clone, Debug)]
pub struct PairLjCut {
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
    cutoff2: F,
    ceps: F,
    e0: F,
    f_rc: F,
    source: PairSource,
}

fn mie_c(n: i32, m: i32) -> F {
    let n = n as F;
    let m = m as F;
    (n / (n - m)) * (n / m).powf(m / (n - m))
}

impl PairLjCut {
    pub fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
            return Err("PairLjCut requires epsilon, sigma, cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "PairLjCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        let cutoff2 = cutoff * cutoff;
        let (ceps, e0, f_rc) = shift_constants(epsilon, sigma, cutoff, n, m, shifted, smeared);
        Ok(Self {
            epsilon,
            sigma,
            cutoff,
            n,
            m,
            shifted: shifted || smeared,
            smeared,
            cutoff2,
            ceps,
            e0,
            f_rc,
            source: PairSource::Loop,
        })
    }

    pub fn lj126(epsilon: F, sigma: F, cutoff: F) -> Result<Self, String> {
        Self::new(epsilon, sigma, cutoff, 12, 6, true, false)
    }

    pub fn compiled(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), epsilon.len());
        assert_eq!(atom_i.len(), sigma.len());
        let e0 = vec![0.0; atom_i.len()];
        Self {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: F::INFINITY,
            n: 12,
            m: 6,
            shifted: false,
            smeared: false,
            cutoff2: F::INFINITY,
            ceps: 4.0,
            e0: 0.0,
            f_rc: 0.0,
            source: PairSource::Compiled {
                atom_i,
                atom_j,
                epsilon,
                sigma,
                e0,
            },
        }
    }

    /// The compiled kernel truncated at `cutoff` (`r < cutoff`, as LAMMPS's
    /// `lj/cut` truncates every pair, special ones included) and, when
    /// `shifted`, shifted to zero there (`pair_modify shift yes`) — the
    /// constants the typed kernel computes, pair by pair. `cutoff` = ∞ is a
    /// style that states none: every listed pair prices, unshifted.
    ///
    /// Call it after [`with_exponents`](Self::with_exponents): the shift is
    /// the energy of the kernel's own exponents at the cutoff.
    ///
    /// # Errors
    /// The kernel is not compiled, or `cutoff` is not positive.
    pub fn truncated(mut self, cutoff: F, shifted: bool) -> Result<Self, String> {
        if cutoff.is_nan() || cutoff <= 0.0 {
            return Err(format!("PairLjCut requires cutoff > 0, got {cutoff}"));
        }
        let (n, m) = (self.n, self.m);
        let PairSource::Compiled {
            epsilon, sigma, e0, ..
        } = &mut self.source
        else {
            return Err("PairLjCut::truncated: only a compiled kernel is truncated here".into());
        };
        if shifted {
            for (e, (&eps, &sig)) in e0.iter_mut().zip(epsilon.iter().zip(sigma.iter())) {
                *e = shift_constants(eps, sig, cutoff, n, m, true, false).1;
            }
        }
        self.cutoff = cutoff;
        self.cutoff2 = cutoff * cutoff;
        self.shifted = shifted;
        Ok(self)
    }

    /// The compiled kernel with the Mie exponents `n > m > 0` of the style
    /// (`n`, `m`; LAMMPS's `lj/mie/cut` generalisation, 12-6 by default):
    /// `C ε [(σ/r)ⁿ − (σ/r)ᵐ]`, `C = n/(n − m) (n/m)^(m/(n − m))`.
    ///
    /// # Errors
    /// The kernel is not compiled, or the exponents do not satisfy
    /// `n > m > 0`.
    pub fn with_exponents(mut self, n: i32, m: i32) -> Result<Self, String> {
        if !matches!(self.source, PairSource::Compiled { .. }) {
            return Err("PairLjCut::with_exponents: only a compiled kernel takes them here".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "PairLjCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        self.n = n;
        self.m = m;
        self.ceps = mie_c(n, m);
        Ok(self)
    }

    /// A kernel that finds its parameters from the atoms a pair names.
    ///
    /// `type_id` is one type index per atom — including, under a ghost régime,
    /// the copies, which carry their owners'. `per_type` is `(ε, σ)` per type,
    /// combined by `mixing` into the type-pair table.
    ///
    /// This is the form a neighbour-driven evaluation needs.
    /// [`compiled`](Self::compiled) resolved its parameters against one fixed
    /// pair list, so it can only answer for that list; a neighbour table is a
    /// different list every rebuild.
    #[allow(clippy::too_many_arguments)]
    pub fn typed(
        type_id: Vec<u32>,
        per_type: &[(F, F)],
        mixing: CombiningRule,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if cutoff <= 0.0 {
            return Err("PairLjCut requires cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "PairLjCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        let ntypes = per_type.len();
        if ntypes == 0 {
            return Err("PairLjCut::typed needs at least one type".into());
        }
        if let Some(&t) = type_id.iter().max()
            && t as usize >= ntypes
        {
            return Err(format!(
                "PairLjCut::typed: atom type {t} has no parameters (only {ntypes} types)"
            ));
        }
        let mut sigma = vec![0.0; ntypes * ntypes];
        let mut ceps = vec![0.0; ntypes * ntypes];
        let mut e0 = vec![0.0; ntypes * ntypes];
        let mut f_rc = vec![0.0; ntypes * ntypes];
        for ti in 0..ntypes {
            for tj in 0..ntypes {
                let (eps_ij, sig_ij) = mixing.combine(per_type[ti], per_type[tj]);
                let (c, e, fr) = shift_constants(eps_ij, sig_ij, cutoff, n, m, shifted, smeared);
                let t = ti * ntypes + tj;
                sigma[t] = sig_ij;
                ceps[t] = c;
                e0[t] = e;
                f_rc[t] = fr;
            }
        }
        let n_owned = type_id.len();
        Ok(Self {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff,
            n,
            m,
            shifted: shifted || smeared,
            smeared,
            cutoff2: cutoff * cutoff,
            ceps: 0.0,
            e0: 0.0,
            f_rc: 0.0,
            source: PairSource::Typed {
                type_id,
                ntypes,
                sigma,
                ceps,
                e0,
                f_rc,
                n_owned,
            },
        })
    }

    /// Price the type pair `(ti, tj)` with its own `(ε, σ)` instead of the mixed
    /// value — an explicit cross row (CHARMM NBFIX, AMBER ACOEF/BCOEF, GROMACS
    /// `nonbond_params`). Both orders are set; the pair is symmetric.
    ///
    /// # Errors
    /// The kernel is not a type-pair table, or a type index is out of range.
    pub fn with_type_pair(
        mut self,
        ti: usize,
        tj: usize,
        epsilon: F,
        sigma: F,
    ) -> Result<Self, String> {
        let (cutoff, n, m, shifted, smeared) =
            (self.cutoff, self.n, self.m, self.shifted, self.smeared);
        let PairSource::Typed {
            ntypes,
            sigma: table_sigma,
            ceps,
            e0,
            f_rc,
            ..
        } = &mut self.source
        else {
            return Err(
                "PairLjCut::with_type_pair: only a typed kernel has a type-pair table".into(),
            );
        };
        let nt = *ntypes;
        if ti >= nt || tj >= nt {
            return Err(format!(
                "PairLjCut::with_type_pair: type pair ({ti}, {tj}) out of range ({nt} types)"
            ));
        }
        let (c, e, fr) = shift_constants(epsilon, sigma, cutoff, n, m, shifted, smeared);
        for t in [ti * nt + tj, tj * nt + ti] {
            table_sigma[t] = sigma;
            ceps[t] = c;
            e0[t] = e;
            f_rc[t] = fr;
        }
        Ok(self)
    }

    pub fn epsilon(&self) -> F {
        self.epsilon
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }
    pub fn cutoff(&self) -> F {
        self.cutoff
    }
    pub fn n(&self) -> i32 {
        self.n
    }
    pub fn m(&self) -> i32 {
        self.m
    }
    pub fn shifted(&self) -> bool {
        self.shifted
    }
    pub fn smeared(&self) -> bool {
        self.smeared
    }

    #[allow(clippy::too_many_arguments)]
    fn pair_kernel_params(
        &self,
        r2: F,
        disp: [F; 3],
        sigma: F,
        ceps: F,
        e0: F,
        f_rc: F,
        cutoff2: F,
        n: i32,
        m: i32,
    ) -> Option<(F, [F; 3])> {
        // LAMMPS: `if (rsq < cutsq[itype][jtype])`.
        if r2 < MIN_R2 || r2 >= cutoff2 {
            return None;
        }
        if n == 12 && m == 6 {
            let inv_r2 = 1.0 / r2;
            let sr2 = sigma * sigma * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let mut energy = ceps * (sr12 - sr6) - e0;
            let mut fac = ceps * (12.0 * sr12 - 6.0 * sr6) * inv_r2;
            if f_rc != 0.0 {
                let r = r2.sqrt();
                energy += (r - self.cutoff) * f_rc;
                fac -= f_rc / r;
            }
            return Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]));
        }
        let sr = sigma / r2.sqrt();
        let sr_n = sr.powi(n);
        let sr_m = sr.powi(m);
        let mut energy = ceps * (sr_n - sr_m) - e0;
        let mut fac = ceps * ((n as F) * sr_n - (m as F) * sr_m) / r2;
        if f_rc != 0.0 {
            let r = r2.sqrt();
            energy += (r - self.cutoff) * f_rc;
            fac -= f_rc / r;
        }
        Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]))
    }

    fn pair_kernel(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel_params(
            r2,
            disp,
            self.sigma,
            self.ceps,
            self.e0,
            self.f_rc,
            self.cutoff2,
            self.n,
            self.m,
        )
    }

    /// Fold the compiled intramolecular list into `forces`, returning the energy.
    fn fold_compiled(&self, coords: &[F], forces: &mut [F]) -> F {
        let PairSource::Compiled {
            atom_i,
            atom_j,
            epsilon,
            sigma,
            e0,
        } = &self.source
        else {
            return 0.0;
        };
        let n_atoms = validate_coords(coords);
        let mut energy = 0.0;
        for idx in 0..atom_i.len() {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            // `ceps` is the Mie prefactor of the exponents (4 at 12-6).
            let ceps = self.ceps * epsilon[idx];
            let Some((e, f)) = self.pair_kernel_params(
                r2,
                [dx, dy, dz],
                sigma[idx],
                ceps,
                e0[idx],
                0.0,
                self.cutoff2,
                self.n,
                self.m,
            ) else {
                continue;
            };
            energy += e;
            forces[j * 3] += f[0];
            forces[j * 3 + 1] += f[1];
            forces[j * 3 + 2] += f[2];
            forces[i * 3] -= f[0];
            forces[i * 3 + 1] -= f[1];
            forces[i * 3 + 2] -= f[2];
        }
        energy
    }

    /// Fold a neighbour table with the parameters the atoms' types select.
    ///
    /// The only difference from [`fold_neighbors`](Self::fold_neighbors) is
    /// where `(σ, ε)` comes from: there, one style-level pair; here, the
    /// type-pair table. The geometry is already reduced either way — this
    /// kernel never learns whether a neighbour is an owned atom or a copy.
    fn fold_typed(&self, acc: &mut Accumulator<'_>, factor: &[F], pairs: &Neighbors) {
        fold_pairs(
            acc,
            pairs.query_point_indices().len(),
            |out: &mut [F], rows| self.fold_typed_rows::<false>(out, factor, pairs, rows),
            |out: &mut [F], rows| self.fold_typed_rows::<true>(out, factor, pairs, rows),
        );
    }

    fn fold_typed_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut virial = Virial::ZERO;
        let PairSource::Typed {
            type_id,
            ntypes,
            sigma,
            ceps,
            e0,
            f_rc,
            ..
        } = &self.source
        else {
            return (0.0, virial);
        };
        let Some(disp) = pairs.disp() else {
            return (0.0, virial);
        };
        let i = pairs.query_point_indices();
        let j = pairs.point_indices();
        let d2 = pairs.dist_sq();
        let mut energy = 0.0;
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            // Exactly zero skips: a bonded pair sits at bond length,
            // where this term is enormous.
            if w == 0.0 {
                continue;
            }
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            let (Some(&ti), Some(&tj)) = (type_id.get(ia), type_id.get(ja)) else {
                // A pair naming an atom the type table does not cover cannot be
                // scored. Under a ghost régime that means the copies were not
                // gathered, which is a wiring fault and not a zero.
                debug_assert!(
                    false,
                    "pair ({ia}, {ja}) is outside the {} type ids",
                    type_id.len()
                );
                continue;
            };
            let t = ti as usize * ntypes + tj as usize;
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match d2 {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel_params(
                r2,
                d,
                sigma[t],
                ceps[t],
                e0[t],
                f_rc[t],
                self.cutoff2,
                self.n,
                self.m,
            ) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            if VIRIAL {
                virial.add_outer(f, d);
            }
            let (bj, bi) = (3 * ja, 3 * ia);
            out[bj] += f[0];
            out[bj + 1] += f[1];
            out[bj + 2] += f[2];
            out[bi] -= f[0];
            out[bi + 1] -= f[1];
            out[bi + 2] -= f[2];
        }
        (energy, virial)
    }

    fn fold_neighbors(&self, acc: &mut Accumulator<'_>, factor: &[F], pairs: &Neighbors) {
        fold_pairs(
            acc,
            pairs.query_point_indices().len(),
            |out: &mut [F], rows| self.fold_neighbors_rows::<false>(out, factor, pairs, rows),
            |out: &mut [F], rows| self.fold_neighbors_rows::<true>(out, factor, pairs, rows),
        );
    }

    fn fold_neighbors_rows<const VIRIAL: bool>(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut virial = Virial::ZERO;
        let Some(disp) = pairs.disp() else {
            return (0.0, virial);
        };
        let i = pairs.query_point_indices();
        let j = pairs.point_indices();
        let d2 = pairs.dist_sq();
        let mut energy = 0.0;
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            // Exactly zero skips: a bonded pair sits at bond length,
            // where this term is enormous.
            if w == 0.0 {
                continue;
            }
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match d2 {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel(r2, d) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            if VIRIAL {
                virial.add_outer(f, d);
            }
            let bj = 3 * ja;
            let bi = 3 * ia;
            out[bj] += f[0];
            out[bj + 1] += f[1];
            out[bj + 2] += f[2];
            out[bi] -= f[0];
            out[bi + 1] -= f[1];
            out[bi + 2] -= f[2];
        }
        (energy, virial)
    }
}

impl PairPotential for PairLjCut {
    fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel(r2, disp)
    }
}

impl Potential for PairLjCut {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        match &self.source {
            PairSource::Compiled { .. } => {
                let energy = self.fold_compiled(coords, acc.forces_without_virial());
                acc.add_energy(energy);
            }
            // Both need a pair table nobody handed over: nothing to add.
            PairSource::Loop | PairSource::Typed { .. } => {}
        }
    }
}

impl PairDriven for PairLjCut {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        match &self.source {
            // A compiled list is an intramolecular sum with no periodicity;
            // a virial from its raw differences would be a number
            // about nothing, and it cannot read a per-pair weight either.
            PairSource::Compiled { .. } => {
                debug_assert!(factor.is_empty());
                self.evaluate(coords, acc);
            }
            PairSource::Loop => self.fold_neighbors(acc, factor, pairs),
            PairSource::Typed { .. } => self.fold_typed(acc, factor, pairs),
        }
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, PairSource::Compiled { .. })
    }
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let PairSource::Typed {
            type_id, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(type_id, *n_owned, owner);
    }
}

/// `(ε, σ)` of the row keyed *key* of the Lennard-Jones style `style`, or
/// `None` when there is none.
fn lj_row(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    key: &str,
) -> Result<Option<(F, F)>, IrError> {
    let Some(p) = type_map.get(key) else {
        return Ok(None);
    };
    let eps = param_reads::type_num(style, key, p, "epsilon")?;
    let sigma = param_reads::type_num(style, key, p, "sigma")?;
    Ok(Some((eps, sigma)))
}

/// The explicit cross row between atom types `a` and `b`, if the style defines
/// one — keyed by [`pair_key`], as `kernel_type_params` keys it, so it answers
/// in either order. A self pair has none.
fn lj_cross_row(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    a: &str,
    b: &str,
) -> Result<Option<(F, F)>, CompileError> {
    if a == b {
        return Ok(None);
    }
    Ok(lj_row(style, type_map, &pair_key(a, b)?)?)
}

/// `(ε, σ)` of the atom-type pair `(a, b)` under the Lennard-Jones style
/// `style` (`lj/cut`, `lj/class2`): its explicit cross row, else the two self
/// rows mixed by `mixing` — what every such kernel prices it with.
pub(crate) fn lj_pair_params(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    mixing: CombiningRule,
    a: &str,
    b: &str,
) -> Result<(F, F), CompileError> {
    if let Some(row) = lj_cross_row(style, type_map, a, b)? {
        return Ok(row);
    }
    let own = |t: &str| -> Result<(F, F), CompileError> {
        lj_row(style, type_map, t)?
            .ok_or_else(|| format!("{style}: unknown atom type '{t}'").into())
    };
    Ok(mixing.combine(own(a)?, own(b)?))
}

/// The per-type `(ε, σ)` of every label, then each explicit cross row as
/// `(ti, tj, ε, σ)`: what a neighbour-driven Lennard-Jones table is built
/// from.
pub(crate) type LjTable = (Vec<(F, F)>, Vec<(usize, usize, F, F)>);

/// [`LjTable`] of `labels` under the style `style`.
pub(crate) fn lj_table(
    style: &str,
    type_map: &HashMap<&str, &Params>,
    labels: &[String],
) -> Result<LjTable, CompileError> {
    let mut per_type = Vec::with_capacity(labels.len());
    for l in labels {
        per_type.push(
            lj_row(style, type_map, l)?
                .ok_or_else(|| format!("{style}: unknown atom type '{l}'"))?,
        );
    }
    let mut cross = Vec::new();
    for (ti, a) in labels.iter().enumerate() {
        for (tj, b) in labels.iter().enumerate().skip(ti + 1) {
            if let Some((eps, sigma)) = lj_cross_row(style, type_map, a, b)? {
                cross.push((ti, tj, eps, sigma));
            }
        }
    }
    Ok((per_type, cross))
}

/// The `mixing` rule of the Lennard-Jones style `style` (gathered: declared,
/// else the spec's default).
pub(crate) fn mixing_of(style: &str, style_params: &Params) -> Result<CombiningRule, IrError> {
    let name = param_reads::style_text(style, style_params, "mixing")?;
    CombiningRule::parse(name).map_err(|e| param_reads::bad(style, "", "mixing", e))
}

/// The style's exponents `(n, m)`: integers with `n > m > 0`.
fn exponents(style_params: &Params) -> Result<(i32, i32), IrError> {
    let get = |key: &str| -> Result<i32, IrError> {
        let v = param_reads::style_num("lj/cut", style_params, key)?;
        if v.fract() != 0.0 {
            return Err(param_reads::bad(
                "lj/cut",
                "",
                key,
                format!("= {v} is not an integer"),
            ));
        }
        Ok(v as i32)
    };
    let (n, m) = (get("n")?, get("m")?);
    if m <= 0 || n <= m {
        return Err(param_reads::bad(
            "lj/cut",
            "",
            "n",
            format!("= {n} with m = {m}: the exponents satisfy n > m > 0"),
        ));
    }
    Ok((n, m))
}

/// Construct a compiled [`PairLjCut`] from per-atom-type params + a neighbour list.
///
/// A pair whose types have an explicit cross row is priced with it; every other
/// pair is mixed from the two self rows by the style's `mixing`. A pair at or
/// beyond the style's `cutoff` prices nothing, shifted by `shift` — exactly
/// as the typed kernel and LAMMPS price it.
pub fn pair_lj_cut_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let scale_14 = style_params
        .get("lj14scale")
        .and_then(|v| v.as_num())
        .unwrap_or(1.0) as F;
    let mixing = mixing_of("lj/cut", style_params)?;
    let (n_exp, m_exp) = exponents(style_params)?;
    let cutoff = param_reads::pair_cutoff("lj/cut", style_params)?;
    let shifted = param_reads::style_num("lj/cut", style_params, "shift")? != 0.0;

    let atoms = frame
        .get(ATOMS)
        .ok_or_else(|| "PairLjCut: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "PairLjCut: atoms block missing \"type\" column".to_string())?;
    let block = frame
        .get(PAIRS)
        .ok_or_else(|| "PairLjCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairLjCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or_else(|| "PairLjCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get("is_14").and_then(|c| c.get::<bool>());

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut eps_vec = Vec::with_capacity(n);
    let mut sig_vec = Vec::with_capacity(n);

    for idx in 0..n {
        let (ti, tj) = (
            atom_types[i_col[idx] as usize].as_str(),
            atom_types[j_col[idx] as usize].as_str(),
        );
        let (mut eps, sigma) = lj_pair_params("lj/cut", &type_map, mixing, ti, tj)?;
        if is_14.is_some_and(|b| b[idx]) {
            eps *= scale_14;
        }
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(ForceTerm::pair(
        PairLjCut::compiled(atom_i, atom_j, eps_vec, sig_vec)
            .with_exponents(n_exp, m_exp)?
            .truncated(cutoff, shifted)?,
    ))
}

/// Construct a neighbour-driven [`PairLjCut`] from per-atom parameters.
///
/// The counterpart of [`pair_lj_cut_constructor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_lj_cut_typed_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let mixing = mixing_of("lj/cut", style_params)?;
    // Required finite, where the compiled form takes ∞ (a style stating no
    // cutoff): an intramolecular list is finite by construction, a periodic
    // neighbour sum is not.
    let cutoff = param_reads::neighbour_cutoff("lj/cut", style_params)?;
    let (n, m) = exponents(style_params)?;
    let shifted = param_reads::style_num("lj/cut", style_params, "shift")? != 0.0;

    let (type_id, labels) = atom_type_index(frame)?;
    let (per_type, cross) = lj_table("lj/cut", &type_map, &labels)?;
    let mut kernel = PairLjCut::typed(type_id, &per_type, mixing, cutoff, n, m, shifted, false)?;
    // Explicit cross rows replace the mixed entries of the type-pair table.
    for (ti, tj, eps, sigma) in cross {
        kernel = kernel.with_type_pair(ti, tj, eps, sigma)?;
    }
    Ok(ForceTerm::pair(kernel))
}

/// Lennard-Jones `A`/`B` coefficients to `(σ, ε)`.
///
/// The same 12-6 interaction is written two ways: Amber files carry
/// `E = A/r¹² − B/r⁶`, and this kernel takes `E = 4ε[(σ/r)¹² − (σ/r)⁶]`.
///
/// `r_min = (2A/B)^{1/6}`, `ε = B²/(4A)`, `σ = 2^{−1/6} r_min`. A row with
/// either coefficient zero is a non-interacting type — Amber writes those for
/// types that carry no vdW term — and yields `(1.0, 0.0)`: ε of zero switches
/// the term off, and σ is given a benign non-zero value so that no consumer
/// divides by it.
///
/// ```
/// # use molrs::ff::potential::pair::lj_ab_to_sigma_epsilon;
/// // GAFF c3: R* = 1.9080 Å, ε = 0.1094 kcal/mol.
/// let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
/// assert!((epsilon - 0.1094).abs() < 1e-6);
/// assert!((sigma - 3.399669).abs() < 1e-5);
/// assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
/// ```
#[must_use]
pub fn lj_ab_to_sigma_epsilon(a: F, b: F) -> (F, F) {
    if a == 0.0 || b == 0.0 {
        return (1.0, 0.0);
    }
    let r_min = (2.0 * a / b).powf(1.0 / 6.0);
    let epsilon = 0.25 * b * b / a;
    let sigma = (2.0 as F).powf(-1.0 / 6.0) * r_min;
    (sigma, epsilon)
}

#[cfg(test)]
mod tests {
    use crate::core::Cell;

    /// The fold gives the same number, bit for bit, however many threads ran.
    ///
    /// Above a pair-count threshold the fold is split across threads, and a
    /// sum of floats is not associative — so the split has to be by a fixed
    /// arithmetic boundary and the partials merged in chunk order. rayon's own
    /// adaptive split would reorder the additions and move the last bits of
    /// every energy, force and virial with them. This asserts the property
    /// rather than the intent: the table here is deliberately larger than the
    /// threshold, so the parallel path is the one being measured.
    #[cfg(feature = "rayon")]
    #[test]
    fn the_fold_does_not_depend_on_the_thread_count() {
        use molrs::core::SimBox;
        use molrs::core::{NeighborList, NeighborPolicy, VerletSkin};

        // 10³ atoms at 3 Å with a 6 Å cutoff clears 8192 pairs comfortably.
        let side = 10_usize;
        let n = side * side * side;
        let l = side as F * 3.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
        let pos = ndarray::Array2::from_shape_fn((n, 3), |(a, k)| {
            let (i, j, m) = (a % side, (a / side) % side, a / (side * side));
            let base = [i, j, m][k] as F * 3.0;
            // A deterministic jitter, so no two separations coincide and a
            // reordering cannot be masked by equal terms.
            base + ((a * 7 + k * 13) as F * 0.618).fract() * 0.4
        });
        let mut skin = VerletSkin::new(
            NeighborList::new(6.0),
            6.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx,
        )
        .unwrap();
        let table = skin.pairs_at(pos.view()).unwrap().clone();
        assert!(
            table.query_point_indices().len() > 8_192,
            "the fixture must cross the parallel threshold"
        );

        let kernel = PairLjCut::typed(
            (0..n).map(|i| (i % 2) as u32).collect(),
            &[(0.3, 3.4), (0.5, 3.0)],
            CombiningRule::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let flat: Vec<F> = pos.iter().copied().collect();

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let (e, out, w) = kernel.over_pairs_virial(&flat, &table);
                (e, out, w.unwrap())
            })
        };

        let (e1, f1, w1) = run(1);
        assert!(e1.abs() > 1.0, "the fixture must interact");
        for threads in [2, 4, 8] {
            let (e, f, w) = run(threads);
            assert_eq!(e1.to_bits(), e.to_bits(), "{threads} threads: energy");
            for (k, (a, b)) in f1.iter().zip(&f).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: force {k}");
            }
            for c in 0..6 {
                assert_eq!(
                    w1.components[c].to_bits(),
                    w.components[c].to_bits(),
                    "{threads} threads: virial {c}"
                );
            }
        }
    }

    use super::*;

    use crate::ff::potential::fixtures::OneShot;

    #[test]
    fn mie_c_is_four_for_12_6() {
        assert!((mie_c(12, 6) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn loop_without_pairs_is_explicit_zero() {
        let lj = PairLjCut::lj126(1.0, 1.0, 2.5).unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0];
        let (e, f) = lj.energy_forces_at(&coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn compiled_newton_third_law() {
        let pot = PairLjCut::compiled(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            assert!((forces[dim] + forces[3 + dim]).abs() < 1e-12);
        }
    }

    #[test]
    fn compiled_ignores_loop_pairs() {
        let pot = PairLjCut::compiled(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e0, _) = pot.energy_forces_at(&coords);
        let extra = Neighbors::from_pairs(
            [molrs::core::NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            molrs::core::NeighborColumns::FULL,
            molrs::core::QueryMode::SelfQuery { n_points: 2 },
        );
        let (e1, _) = pot.over_pairs(&coords, &extra);
        assert_eq!(e0, e1);
    }

    /// The compiled door prices the style's Mie exponents, as the typed door
    /// and the spec's expression do (it priced 12-6 whatever `n`, `m` said).
    #[test]
    fn the_compiled_door_takes_the_style_exponents() {
        use molrs::core::Block;
        use ndarray::Array1;
        let rows = [(
            "A".to_string(),
            Params::from_iter([("epsilon", 0.2), ("sigma", 3.0)]),
        )];
        let tp: Vec<(&str, &Params)> = rows.iter().map(|(k, p)| (k.as_str(), p)).collect();
        let mut frame = ab_frame();
        frame.insert(ATOMS, {
            let mut atoms = Block::new();
            atoms
                .insert(
                    "type",
                    Array1::from_vec(vec!["A".to_string(), "A".to_string()]).into_dyn(),
                    None,
                )
                .unwrap();
            atoms
        });
        let coords = [0.0, 0.0, 0.0, 3.7, 0.0, 0.0];
        for (n, m) in [(12.0, 6.0), (9.0, 6.0), (10.0, 4.0)] {
            let style = gathered(Params::from_iter([("cutoff", 100.0), ("n", n), ("m", m)]));
            let compiled = pair_lj_cut_constructor(&style, &tp, &frame).unwrap();
            let typed = pair_lj_cut_typed_constructor(&style, &tp, &frame).unwrap();
            let c = (n / (n - m)) * (n / m).powf(m / (n - m));
            let want = c * 0.2 * ((3.0_f64 / 3.7).powf(n) - (3.0_f64 / 3.7).powf(m));
            let e_c = compiled.as_potential().energy_at(&coords);
            assert!((e_c - want).abs() < 1e-12, "{n}-{m}: {e_c} vs {want}");
            let pair = Neighbors::from_pairs(
                vec![molrs::core::NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 3.7 * 3.7,
                    disp: [3.7, 0.0, 0.0],
                }],
                molrs::core::NeighborColumns::FULL,
                molrs::core::QueryMode::SelfQuery { n_points: 2 },
            );
            let ForceTerm::Pair(typed) = typed else {
                panic!("a pair member")
            };
            let e_t = typed.over_pairs(&coords, &pair).0;
            assert!((e_t - want).abs() < 1e-12, "{n}-{m} typed: {e_t} vs {want}");
        }
    }

    /// The typed table and the compiled list are two ways of finding the same
    /// number, and on the same pairs they must find it bit for bit.
    ///
    /// This is the whole claim of the typed form: nothing about the physics
    /// changed, only *when* a parameter is chosen. Anything else that moved
    /// would show up here, and it is checked on identical arithmetic — same
    /// mixing, same exponents, no cutoff, no shift — so bit equality is the
    /// right bar rather than a tolerance.
    ///
    /// Free boundary on purpose: this is about the lookup, not periodicity.
    #[test]
    fn a_typed_kernel_scores_a_pair_exactly_as_a_compiled_one() {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};

        // Two types, deliberately unlike each other, so a table indexed the
        // wrong way round would give a different answer.
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id = vec![0_u32, 1, 0, 1];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            3.1, 0.4, 0.2, //
            1.2, 2.9, 0.7, //
            4.0, 3.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let mixing = CombiningRule::Arithmetic;

        let mut ai = Vec::new();
        let mut aj = Vec::new();
        let mut eps = Vec::new();
        let mut sig = Vec::new();
        let mut table = Vec::new();
        for &(i, j) in &links {
            let (e, sg) =
                mixing.combine(per_type[type_id[i] as usize], per_type[type_id[j] as usize]);
            ai.push(i);
            aj.push(j);
            eps.push(e);
            sig.push(sg);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            table.push(NeighborPair {
                i: i as u32,
                j: j as u32,
                dist_sq: d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                disp: d,
            });
        }

        let compiled = PairLjCut::compiled(ai, aj, eps, sig);
        let typed =
            PairLjCut::typed(type_id, &per_type, mixing, F::INFINITY, 12, 6, false, false).unwrap();

        let neighbors = Neighbors::from_pairs(
            table,
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 4 },
        );

        let (e_c, f_c) = compiled.energy_forces_at(&coords);
        let (e_t, f_t) = typed.over_pairs(&coords, &neighbors);

        assert!(
            e_c.abs() > 1e-6,
            "the configuration must interact for this to assert anything; got {e_c}"
        );
        assert_eq!(e_c.to_bits(), e_t.to_bits(), "energy {e_c} vs {e_t}");
        assert_eq!(f_c.len(), f_t.len());
        for (c, (a, b)) in f_c.iter().zip(&f_t).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "force component {c}: {a} vs {b}");
        }

        crate::ff::potential::pair::fixtures::assert_virial_matches_forces(
            "lj/cut",
            &coords,
            typed.over_pairs_virial(&coords, &neighbors),
        );
    }

    /// A typed kernel stops at its cutoff (`r < cutoff`, as LAMMPS): every
    /// pair inside it interacts and nothing outside it does, which is what
    /// makes the sum finite in a periodic system. The compiled kernel stops
    /// at the same cutoff ([`PairLjCut::truncated`]).
    #[test]
    fn a_typed_kernel_stops_at_its_cutoff() {
        let typed = PairLjCut::typed(
            vec![0_u32, 0],
            &[(1.0, 1.0)],
            CombiningRule::Arithmetic,
            2.5,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        assert!(typed.pair_energy_force(4.0, [2.0, 0.0, 0.0]).is_some());
        assert!(
            typed.pair_energy_force(9.0, [3.0, 0.0, 0.0]).is_none(),
            "3 Å is past the 2.5 Å cutoff"
        );
        assert!(
            typed.pair_energy_force(6.25, [2.5, 0.0, 0.0]).is_none(),
            "LAMMPS prices `rsq < cutsq`: nothing at the cutoff itself"
        );
    }

    /// The compiled kernel truncated and shifted prices a pair as the typed
    /// one does, bit for bit, on each side of the cutoff.
    #[test]
    fn a_truncated_compiled_kernel_is_the_typed_one() {
        use crate::ff::potential::pair::fixtures::{assert_same, table_over};
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.2, 0.0, 2.9, -0.4, 0.3];
        let links = [(0_usize, 1_usize), (0, 2), (1, 2)];
        for shifted in [false, true] {
            let compiled =
                PairLjCut::compiled(vec![0, 0, 1], vec![1, 2, 2], vec![1.0; 3], vec![1.0; 3])
                    .truncated(2.5, shifted)
                    .unwrap();
            let typed = PairLjCut::typed(
                vec![0_u32; 3],
                &[(1.0, 1.0)],
                CombiningRule::Arithmetic,
                2.5,
                12,
                6,
                shifted,
                false,
            )
            .unwrap();
            assert_same(
                &format!("shift {shifted}"),
                compiled.energy_forces_at(&coords),
                typed.over_pairs(&coords, &table_over(&coords, &links)),
            );
        }
    }

    #[test]
    fn unshifted_energy_at_sigma_is_zero() {
        let lj = PairLjCut::new(1.5, 2.0, 5.0, 12, 6, false, false).unwrap();
        let (e, f) = lj.pair_energy_force(4.0, [2.0, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0] > 0.0);
    }

    // -- explicit cross rows (NBFIX) -------------------------------------------

    /// Two atoms, types `A` and `B`, joined by one pair row.
    fn ab_frame() -> Frame {
        use molrs::core::Block;
        use molrs::op::Idx;
        use ndarray::Array1;
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut pairs = Block::new();
        pairs
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        frame.insert(PAIRS, pairs);
        frame
    }

    /// Self rows A (0.1, 3.0) and B (0.4, 3.6), plus — when given — one cross row
    /// keyed the way `kernel_type_params` keys it, between `cross.0` and `cross.1`.
    fn ab_rows(cross: Option<(&str, &str)>) -> Vec<(String, Params)> {
        let mut rows = vec![
            (
                "A".to_string(),
                Params::from_iter([("epsilon", 0.1), ("sigma", 3.0)]),
            ),
            (
                "B".to_string(),
                Params::from_iter([("epsilon", 0.4), ("sigma", 3.6)]),
            ),
        ];
        if let Some((a, b)) = cross {
            rows.push((
                pair_key(a, b).unwrap(),
                Params::from_iter([("epsilon", 0.9), ("sigma", 2.0)]),
            ));
        }
        rows
    }

    /// `style` as a compile hands it to the kernel: the `lj/cut` spec's
    /// defaults filled in.
    fn gathered(style: Params) -> Params {
        crate::ff::style_registry::with_global_registry(|r| {
            r.style("pair", "lj/cut").unwrap().0.gather(&style, &[])
        })
        .unwrap()
        .0
    }

    fn lj(eps: F, sigma: F, r: F) -> F {
        let s6 = (sigma / r).powi(6);
        4.0 * eps * (s6 * s6 - s6)
    }

    const R_AB: F = 2.5;

    fn compiled_energy(rows: &[(String, Params)]) -> F {
        let mut style = Params::new();
        style.set("mixing", "geometric");
        let style = gathered(style);
        let refs: Vec<(&str, &Params)> = rows.iter().map(|(k, p)| (k.as_str(), p)).collect();
        let member = pair_lj_cut_constructor(&style, &refs, &ab_frame()).unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, R_AB, 0.0, 0.0];
        member.as_potential().energy_forces_at(&coords).0
    }

    fn typed_energy(rows: &[(String, Params)]) -> F {
        use molrs::core::{NeighborColumns, NeighborPair, QueryMode};
        let mut style = Params::from_iter([("cutoff", 10.0)]);
        style.set("mixing", "geometric");
        let style = gathered(style);
        let refs: Vec<(&str, &Params)> = rows.iter().map(|(k, p)| (k.as_str(), p)).collect();
        let ForceTerm::Pair(kernel) =
            pair_lj_cut_typed_constructor(&style, &refs, &ab_frame()).unwrap()
        else {
            panic!("lj/cut is a pair member");
        };
        let neighbors = Neighbors::from_pairs(
            vec![NeighborPair {
                i: 0,
                j: 1,
                dist_sq: R_AB * R_AB,
                disp: [R_AB, 0.0, 0.0],
            }],
            NeighborColumns::FULL,
            QueryMode::SelfQuery { n_points: 2 },
        );
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, R_AB, 0.0, 0.0];
        kernel.over_pairs(&coords, &neighbors).0
    }

    /// A pair with an explicit cross row is priced with that row, not with the
    /// mixing rule. The row reached the kernel and was ignored, so an NBFIX was
    /// silently replaced by the mixed value.
    #[test]
    fn an_explicit_cross_row_overrides_the_mixing_rule() {
        let want = lj(0.9, 2.0, R_AB);
        let rows = ab_rows(Some(("A", "B")));
        assert!((compiled_energy(&rows) - want).abs() < 1e-12);
        assert!((typed_energy(&rows) - want).abs() < 1e-12);
    }

    /// A cross row defined as `B-A` answers for the pair `(A, B)` too: the
    /// force field keys it by its endpoints in byte order.
    #[test]
    fn a_cross_row_is_found_in_either_order() {
        use crate::ff::forcefield::ForceField;
        let lj_params =
            |eps: f64, sigma: f64| Params::from_iter([("epsilon", eps), ("sigma", sigma)]);
        let mut ff = ForceField::new("t");
        let style = ff.def_style("pair", "lj/cut", Params::new()).unwrap();
        style
            .def_type("A", &["A"], lj_params(0.1, 3.0))
            .unwrap()
            .def_type("B", &["B"], lj_params(0.4, 3.6))
            .unwrap()
            .def_type("nbfix", &["B", "A"], lj_params(0.9, 2.0))
            .unwrap();
        let rows = style.defs().kernel_type_params().unwrap();
        let want = lj(0.9, 2.0, R_AB);
        assert!((compiled_energy(&rows) - want).abs() < 1e-12);
        assert!((typed_energy(&rows) - want).abs() < 1e-12);
    }

    /// Without a cross row the pair is mixed, geometrically here.
    #[test]
    fn without_a_cross_row_the_pair_is_mixed() {
        let want = lj((0.1_f64 * 0.4).sqrt(), (3.0_f64 * 3.6).sqrt(), R_AB);
        let rows = ab_rows(None);
        assert!((compiled_energy(&rows) - want).abs() < 1e-12);
        assert!((typed_energy(&rows) - want).abs() < 1e-12);
    }

    #[test]
    fn zero_coefficient_is_a_non_interacting_type() {
        assert_eq!(lj_ab_to_sigma_epsilon(0.0, 1.0), (1.0, 0.0));
        assert_eq!(lj_ab_to_sigma_epsilon(1.0, 0.0), (1.0, 0.0));
    }

    #[test]
    fn round_trips_the_gaff_c3_row() {
        let (sigma, epsilon) = lj_ab_to_sigma_epsilon(1.043080230e6, 6.75612248e2);
        assert!((epsilon - 0.1094).abs() < 1e-6, "{epsilon}");
        // sigma = 2^(-1/6) * R* * 2, with R* the Amber half-distance 1.9080 Å.
        assert!((sigma - 3.399669).abs() < 1e-5, "{sigma}");
    }

    #[test]
    fn inverts_the_molrs_form() {
        // Build A/B from a known (sigma, epsilon) and recover it.
        let (sigma, epsilon) = (3.4_f64, 0.238_f64);
        let a = 4.0 * epsilon * sigma.powi(12);
        let b = 4.0 * epsilon * sigma.powi(6);
        let (got_sigma, got_epsilon) = lj_ab_to_sigma_epsilon(a, b);
        assert!((got_sigma - sigma).abs() < 1e-12, "{got_sigma}");
        assert!((got_epsilon - epsilon).abs() < 1e-12, "{got_epsilon}");
    }
}
