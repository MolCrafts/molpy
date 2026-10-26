//! The 1-4 exceptions kernel: every non-bonded pair whose pricing is not the
//! pair styles' own, as one list.

use ndarray::{Array2, ArrayView2};

use crate::ff::potential::flat_coords::{term_table, validate_coords};
use crate::ff::potential::{Accumulator, IndexedTerms, Potential};
use molrs::op::F;

const MIN_R2: F = 1e-24;

/// LAMMPS's `lj1..lj4` for one `(ε, σ)`: `48εσ¹²`, `24εσ⁶`, `4εσ¹²`, `4εσ⁶`.
fn lj_coeffs(epsilon: F, sigma: F) -> [F; 4] {
    let s12 = sigma.powf(12.0);
    let s6 = sigma.powf(6.0);
    [
        48.0 * epsilon * s12,
        24.0 * epsilon * s6,
        4.0 * epsilon * s12,
        4.0 * epsilon * s6,
    ]
}

/// The 1-4 exceptions of one molecule: LJ 12-6 plus Coulomb, no cutoff.
///
/// LAMMPS has three ways to price a close (1-4) pair, and molrs represents
/// each with LAMMPS's parameters (the conventions guide, "1-4 interactions"):
///
/// 1. **`special_bonds`** — the pair styles price the pair at their own
///    parameters, scaled by the force field's 1-4 weights. Nothing here.
/// 2. **`dihedral_style charmm` `w`** — each dihedral prices the pair of its
///    end atoms, `w·[LJ(ε₁₄, σ₁₄) + C qᵢqⱼ/r]` (`dihedral_charmm.cpp`), with
///    the `epsilon14` / `sigma14` of the `lj/charmm` pair style mixed as
///    LAMMPS's `init_one` mixes them, no cutoff and no switch. LAMMPS refuses
///    `w > 0` beside non-zero `special_bonds` 1-4 weights, and so does molrs.
///    A pair at the ends of several dihedrals takes the sum of their `w`.
/// 3. **Per-pair overrides** — columns on the frame's `pairs` block for what
///    LAMMPS cannot express (a GROMACS `[ pairs ]` row with parameters, an
///    OpenMM exception, an AMBER dihedral's own SCEE / SCNB):
///    [`PAIR_OVERRIDE_COLUMNS`](molrs::core::schema::PAIR_OVERRIDE_COLUMNS) = `epsilon`, `sigma`, `charge_product`,
///    `lj_scale`, `coul_scale`.
///
/// **Precedence**, per pair and per quantity: a per-pair override cell is
/// final; a null cell takes what the pair would have without the row — the
/// dihedral's `w` pricing when its ends carry `w > 0`, the pair style's
/// parameters at the `special_bonds` weight of the pair's bond-distance class
/// otherwise. So `lj_scale` / `coul_scale` replace `w` or the global weight,
/// and `epsilon` / `sigma` / `charge_product` replace the style's (or the
/// dihedral's 1-4) values. A cell is priced only by the style it belongs to:
/// `epsilon` / `sigma` / `lj_scale` under a Lennard-Jones style,
/// `charge_product` / `coul_scale` under a Coulomb style. A field without
/// that style ignores them, so a bonded-only field on a frame with
/// materialized 1-4 cells prices no pair at all.
///
/// Every such pair is priced here, once:
///
/// ```text
/// E = lj_w · 4ε[(σ/r)¹² − (σ/r)⁶]  +  coul_w · C qᵢqⱼ / r
/// ```
///
/// with no cutoff (as LAMMPS's dihedral 1-4 term, and an OpenMM exception),
/// `C` the Coulomb style's `coulomb / dielectric`. The regular pair kernels
/// price an **override** pair at weight 0, at both compile doors (its
/// [`SpecialWeights`](crate::ff::potential::SpecialWeights) weight is
/// zeroed). A `w` pair needs no such step: `special_bonds` 1-4 is 0 for it,
/// as LAMMPS requires.
///
/// The kernel is a fixed list of atom pairs, so it is an indexed (bond-like)
/// member at both compile doors: a periodic régime rebinds it like a bond.
pub struct PairExceptions {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    /// LAMMPS's `lj1..lj4` of each pair.
    lj: Vec<[F; 4]>,
    lj_w: Vec<F>,
    /// `C·qᵢqⱼ`, LAMMPS's `qqrd2e·qᵢ·qⱼ`.
    qq: Vec<F>,
    coul_w: Vec<F>,
}

impl PairExceptions {
    /// One pair per row: `(ε, σ)` with its weight and `C·qᵢqⱼ` with its
    /// weight.
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        eps_sigma: &[(F, F)],
        lj_w: Vec<F>,
        qq: Vec<F>,
        coul_w: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert!(
            [
                atom_j.len(),
                eps_sigma.len(),
                lj_w.len(),
                qq.len(),
                coul_w.len()
            ]
            .iter()
            .all(|&m| m == n)
        );
        Self {
            atom_i,
            atom_j,
            lj: eps_sigma.iter().map(|&(e, s)| lj_coeffs(e, s)).collect(),
            lj_w,
            qq,
            coul_w,
        }
    }

    /// How many pairs this kernel prices.
    pub fn len(&self) -> usize {
        self.atom_i.len()
    }

    /// Whether it prices none.
    pub fn is_empty(&self) -> bool {
        self.atom_i.is_empty()
    }

    /// The van-der-Waals and Coulomb parts of the energy, apart — LAMMPS
    /// tallies the dihedral's 1-4 pair into `evdwl` and `ecoul`.
    pub fn energy_terms(&self, coords: &[F]) -> (F, F) {
        let mut out = vec![0.0; coords.len()];
        self.fold(coords, &mut out, self.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        })
    }

    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> (F, F) {
        validate_coords(coords);
        let (mut evdwl, mut ecoul) = (0.0, 0.0);
        for t in 0..n {
            let (i, j) = atoms(t);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let rsq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if rsq < MIN_R2 {
                continue;
            }
            // dihedral_charmm.cpp, with the one weight split in two.
            let r2inv = 1.0 / rsq;
            let r6inv = r2inv * r2inv * r2inv;
            let c = &self.lj[t];
            let forcecoul = self.qq[t] * r2inv.sqrt();
            let forcelj = r6inv * (c[0] * r6inv - c[1]);
            let fpair = (self.lj_w[t] * forcelj + self.coul_w[t] * forcecoul) * r2inv;
            evdwl += self.lj_w[t] * (r6inv * (c[2] * r6inv - c[3]));
            ecoul += self.coul_w[t] * forcecoul;
            for k in 0..3 {
                out[j * 3 + k] += fpair * d[k];
                out[i * 3 + k] -= fpair * d[k];
            }
        }
        (evdwl, ecoul)
    }
}

impl Potential for PairExceptions {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let (v, c) = self.fold(coords, acc.forces_without_virial(), self.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        });
        acc.add_energy(v + c);
    }
}

impl IndexedTerms for PairExceptions {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j])
    }

    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(terms.nrows(), self.len());
        let (v, c) = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (terms[[t, 0]] as usize, terms[[t, 1]] as usize)
        });
        acc.add_energy(v + c);
    }
}
