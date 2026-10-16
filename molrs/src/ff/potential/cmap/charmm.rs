//! CHARMM CMAP crossterm (LAMMPS `fix cmap`).

use crate::ff::potential::param_reads;
use std::collections::HashMap;

use ndarray::{Array2, ArrayD, ArrayView2};

use crate::ff::ir::CMAP_GRID;
use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{compute_dihedral, sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::{cross, dot, scale};
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOMJ, ATOMK, ATOML, ATOMM, TYPE};
use molrs::core::schema::block_names::CMAPS;
use molrs::op::F;

/// Lower grid edge, degrees (LAMMPS `CMAPXMIN2`).
const ORIGIN: F = -180.0;

/// Below this squared cross-product norm (Å⁴) a crossterm is skipped, as
/// `fix cmap` skips it.
const DEGENERATE: F = 0.0001;

/// The bicubic weight matrix of LAMMPS `FixCMAP::bc_coeff` (Numerical Recipes
/// `bcucof`).
const WT: [[i8; 16]; 16] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
    [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
    [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
    [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
    [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
    [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
    [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1],
];

/// One map with the node derivatives LAMMPS precomputes from it.
#[derive(Debug, Clone, PartialEq)]
pub struct CmapGrid {
    n: usize,
    /// Grid spacing Δ = 360°/N, degrees.
    dx: F,
    /// Energies, φ-major: `[i·N + j]` at (−180° + iΔ, −180° + jΔ).
    e: Vec<F>,
    /// ∂E/∂φ at the nodes, energy/degree.
    d1: Vec<F>,
    /// ∂E/∂ψ at the nodes, energy/degree.
    d2: Vec<F>,
    /// ∂²E/∂φ∂ψ at the nodes, energy/degree².
    d12: Vec<F>,
}

impl CmapGrid {
    /// Precompute `grid`'s node derivatives (LAMMPS
    /// `FixCMAP::set_map_derivatives`).
    ///
    /// `Err` unless `grid` is a square, finite N×N array with N ≥ 2.
    pub fn new(grid: &ArrayD<f64>) -> Result<Self, String> {
        let shape = grid.shape();
        let n = match shape {
            [a, b] if a == b && *a >= 2 => *a,
            _ => {
                return Err(format!(
                    "is not a square N×N array with N >= 2: its shape is {shape:?}"
                ));
            }
        };
        if let Some(bad) = grid.iter().find(|v| !v.is_finite()) {
            return Err(format!("holds {bad}; a cmap grid's values are finite"));
        }
        let e: Vec<F> = grid.iter().copied().collect();
        let dx = 360.0 / n as F;
        let xm = n / 2;
        let two = 2 * n;

        // The map extended periodically: row/column `i` of `tmap` is grid
        // index (i − xm) mod N, angle −180° + (i − xm)Δ.
        let wrap = |i: usize| (i + n - xm) % n;
        let mut tmap = vec![vec![0.0; two]; two];
        for (i, row) in tmap.iter_mut().enumerate() {
            for (j, v) in row.iter_mut().enumerate() {
                *v = e[wrap(i) * n + wrap(j)];
            }
        }
        let tddmap: Vec<Vec<F>> = tmap.iter().map(|row| spline(row, dx)).collect();

        // LAMMPS evaluates its spline formulas at the node itself, where its
        // a, b are exactly 1 and 0 (Δ = 15° makes every operand an integer).
        let (a, b): (F, F) = (1.0, 0.0);
        let a1 = a * a * a - a;
        let b1 = b * b * b - b;
        let a2 = 3.0 * a * a - 1.0;
        let b2 = 3.0 * b * b - 1.0;

        let mut d1 = vec![0.0; n * n];
        let mut d2 = vec![0.0; n * n];
        let mut d12 = vec![0.0; n * n];
        let mut tmp_y = vec![0.0; two];
        let mut tmp_dy = vec![0.0; two];
        for j in xm..n + xm {
            // ψ node j: E and ∂E/∂ψ of every φ row of the doubled map.
            let ix = j;
            for k in 0..two {
                let (row, dd) = (&tmap[k], &tddmap[k]);
                tmp_y[k] = a * row[ix]
                    + b * row[ix + 1]
                    + (a1 * dd[ix] + b1 * dd[ix + 1]) * (dx * dx) / 6.0;
                tmp_dy[k] = (row[ix + 1] - row[ix]) / dx - (a2 / 6.0 * dx * dd[ix])
                    + (b2 / 6.0 * dx * dd[ix + 1]);
            }
            let ddy = spline(&tmp_y, dx);
            let dddy = spline(&tmp_dy, dx);
            for i in xm..n + xm {
                let ix = i;
                let g = (i - xm) * n + (j - xm);
                d1[g] = (tmp_y[ix + 1] - tmp_y[ix]) / dx - a2 / 6.0 * dx * ddy[ix]
                    + b2 / 6.0 * dx * ddy[ix + 1];
                d2[g] = a * tmp_dy[ix]
                    + b * tmp_dy[ix + 1]
                    + (a1 * dddy[ix] + b1 * dddy[ix + 1]) * (dx * dx) / 6.0;
                d12[g] = (tmp_dy[ix + 1] - tmp_dy[ix]) / dx - a2 / 6.0 * dx * dddy[ix]
                    + b2 / 6.0 * dx * dddy[ix + 1];
            }
        }
        Ok(Self {
            n,
            dx,
            e,
            d1,
            d2,
            d12,
        })
    }

    /// The map's size N.
    pub fn n(&self) -> usize {
        self.n
    }

    /// `(E, ∂E/∂φ, ∂E/∂ψ)` at (φ, ψ) in **degrees**, the slopes per
    /// **radian** (LAMMPS `FixCMAP::bc_interpol` on the cell `post_force`
    /// picks). φ and ψ are in [−180°, 180°]; 180° reads as −180°.
    pub fn eval(&self, phi: F, psi: F) -> (F, F, F) {
        let phi = if phi == 180.0 { -180.0 } else { phi };
        let psi = if psi == 180.0 { -180.0 } else { psi };
        let (n, dx) = (self.n, self.dx);
        let cell = |x: F| (((x - ORIGIN) / dx) as usize).min(n - 1);
        let (li3, li4) = (cell(phi), cell(psi));
        let (i0, i1) = (li3 % n, (li3 + 1) % n);
        let (j0, j1) = (li4 % n, (li4 + 1) % n);
        // Corners counter-clockwise from (φ_lo, ψ_lo), as `post_force` lists
        // them.
        let corners = [(i0, j0), (i1, j0), (i1, j1), (i0, j1)];
        let mut x = [0.0; 16];
        for (c, &(i, j)) in corners.iter().enumerate() {
            let g = i * n + j;
            x[c] = self.e[g];
            x[c + 4] = self.d1[g] * dx;
            x[c + 8] = self.d2[g] * dx;
            x[c + 12] = self.d12[g] * dx * dx;
        }
        let mut cij = [[0.0; 4]; 4];
        for (row, w) in WT.iter().enumerate() {
            let mut xx = 0.0;
            for (k, &wk) in w.iter().enumerate() {
                xx += F::from(wk) * x[k];
            }
            cij[row / 4][row % 4] = xx;
        }

        let t = (phi - (ORIGIN + li3 as F * dx)) / dx;
        let u = (psi - (ORIGIN + li4 as F * dx)) / dx;
        let (mut e, mut de_dphi, mut de_dpsi) = (0.0, 0.0, 0.0);
        for i in (0..4).rev() {
            e = t * e + ((cij[i][3] * u + cij[i][2]) * u + cij[i][1]) * u + cij[i][0];
            de_dphi = u * de_dphi + (3.0 * cij[3][i] * t + 2.0 * cij[2][i]) * t + cij[1][i];
            de_dpsi = t * de_dpsi + (3.0 * cij[i][3] * u + 2.0 * cij[i][2]) * u + cij[i][1];
        }
        let per_radian = (1.0 / dx).to_degrees();
        (e, de_dphi * per_radian, de_dpsi * per_radian)
    }
}

/// Second derivatives of a natural cubic spline through `y` at spacing `dx`
/// (LAMMPS `FixCMAP::spline`).
fn spline(y: &[F], dx: F) -> Vec<F> {
    let n = y.len();
    let mut ddy = vec![0.0; n];
    let mut u = vec![0.0; n - 1];
    for i in 1..=n - 2 {
        let p = 1.0 / (ddy[i - 1] + 4.0);
        ddy[i] = -p;
        u[i] = ((((6.0 * y[i + 1]) - (12.0 * y[i]) + (6.0 * y[i - 1])) / (dx * dx)) - u[i - 1]) * p;
    }
    ddy[n - 1] = 0.0;
    for j in (0..=n - 2).rev() {
        ddy[j] = ddy[j] * ddy[j + 1] + u[j];
    }
    ddy
}

/// The CHARMM CMAP crossterm over pre-resolved atoms and maps.
///
/// A crossterm names five atoms `(a, b, c, d, e)` and is priced from an N×N
/// energy grid over the two dihedrals it spans:
///
/// ```text
/// φ = dihedral(a, b, c, d),   ψ = dihedral(b, c, d, e),   E = map(φ, ψ)
/// ```
///
/// # The grid
///
/// A type's `grid` array param is the map, **φ-major**: element `[i][j]` is
/// the energy at φ = −180° + i·Δ, ψ = −180° + j·Δ, Δ = 360°/N (CHARMM:
/// N = 24, Δ = 15°) — the order of a LAMMPS / CHARMM `.cmap` file.
///
/// # The interpolation is LAMMPS's
///
/// This is a port of LAMMPS `src/MOLECULE/fix_cmap.cpp`, step for step, so a
/// molrs energy is the `fix cmap` energy (molrs-python docs, "Force-field
/// conventions", CMAP):
///
/// 1. **Node derivatives** (`set_map_derivatives`). The map is extended
///    periodically to 2N × 2N (φ, ψ ∈ [−360°, 360°)), each φ row gets a
///    *natural* cubic spline along ψ; at every node, the row splines give
///    `E` and `∂E/∂ψ` down the 2N φ column, which are splined along φ again.
///    The node's `∂E/∂φ`, `∂E/∂ψ` and `∂²E/∂φ∂ψ` (per degree) are those
///    splines' values and slopes. The doubled map keeps the spline's natural
///    end conditions half a period away from every node it is read at.
/// 2. **Patch** (`bc_coeff`, `bc_interpol`). The cell holding (φ, ψ) gets the
///    16 bicubic coefficients from its four corners' values and derivatives
///    (Numerical Recipes' `bcucof` weight matrix), and E, ∂E/∂φ, ∂E/∂ψ are
///    the bicubic polynomial and its slopes at the point; the slopes are
///    converted from per degree to per radian.
/// 3. **Angles and forces** (`post_force`). φ and ψ are LAMMPS's
///    `atan2` dihedrals in degrees, in [−180°, 180°) (180° reads as −180°),
///    and the forces are LAMMPS's `dφ/dr`, `dψ/dr` expressions, so the
///    crossterm is distributed onto the five atoms exactly as LAMMPS does.
///
/// Two LAMMPS behaviours are kept on purpose, as they change energies:
///
/// - a crossterm whose dihedral planes are degenerate — any of the four
///   cross products `|b_ij × b_jk|²` below 10⁻⁴ Å⁴ — contributes **nothing**
///   (`fix cmap` skips it);
/// - the dihedrals are [`op::vec3::dihedral`](crate::op::vec3::dihedral) in
///   degrees, which is LAMMPS's `atan2` dihedral (the IUPAC sign) to rounding.
///
/// One is not: LAMMPS fixes N = 24 and at most six maps (`CMAPDIM`,
/// `CMAPMAX`); this kernel takes any N ≥ 2 and any number of maps, with
/// Δ = 360°/N. LAMMPS stores the derivative grids rotated by N/2 and finds
/// their cell from φ wrapped to [0°, 360°); here they are stored unrotated
/// and read at the value cell — the same numbers, except where a float tie on
/// a cell edge sent LAMMPS's two lookups to neighbouring cells.
pub struct CmapCharmm {
    atoms: [Vec<usize>; 5],
    /// Map index of each crossterm into `maps`.
    map: Vec<usize>,
    maps: Vec<CmapGrid>,
}

impl CmapCharmm {
    /// Crossterms `atoms[t]` priced from `maps[map[t]]`.
    ///
    /// # Panics
    ///
    /// If `atoms` and `map` differ in length or a map index is out of range.
    pub fn new(atoms: Vec<[usize; 5]>, map: Vec<usize>, maps: Vec<CmapGrid>) -> Self {
        assert_eq!(atoms.len(), map.len(), "one map per crossterm");
        assert!(
            map.iter().all(|&m| m < maps.len()),
            "a crossterm names a map that does not exist"
        );
        let column = |p: usize| atoms.iter().map(|a| a[p]).collect();
        Self {
            atoms: [column(0), column(1), column(2), column(3), column(4)],
            map,
            maps,
        }
    }

    /// The physics, once (LAMMPS `FixCMAP::post_force`); only which atoms a
    /// term names differs between the two entry points.
    fn fold(&self, x: &[F], f: &mut [F], n_terms: usize, atoms: impl Fn(usize) -> [usize; 5]) -> F {
        let _ = validate_coords(x);
        let mut energy = 0.0;
        for t in 0..n_terms {
            let [i1, i2, i3, i4, i5] = atoms(t);
            let vb21 = sub3(x, i2, x, i1);
            let vb12 = scale(vb21, -1.0);
            let vb32 = sub3(x, i3, x, i2);
            let vb23 = scale(vb32, -1.0);
            let vb34 = sub3(x, i3, x, i4);
            let vb43 = scale(vb34, -1.0);
            let vb45 = sub3(x, i4, x, i5);

            let a1 = cross(vb12, vb23);
            let b1 = cross(vb43, vb23);
            let a2 = cross(vb23, vb34);
            let b2 = cross(vb45, vb43);

            let r32 = dot(vb32, vb32).sqrt();
            let a1sq = dot(a1, a1);
            let b1sq = dot(b1, b1);
            let r43 = dot(vb43, vb43).sqrt();
            let a2sq = dot(a2, a2);
            let b2sq = dot(b2, b2);
            if a1sq < DEGENERATE || b1sq < DEGENERATE || a2sq < DEGENERATE || b2sq < DEGENERATE {
                continue;
            }
            let dpr21r32 = dot(vb21, vb32);
            let dpr34r32 = dot(vb34, vb32);
            let dpr32r43 = dot(vb32, vb43);
            let dpr45r43 = dot(vb45, vb43);

            // LAMMPS's `FixCMAP::dihedral_angle_atan2` is this angle: the
            // IUPAC sign, `atan2(|b₂| b₁·n₂, n₁·n₂)`. Its −180° is the +180°
            // here, which `CmapGrid::eval` reads as −180°.
            let phi = compute_dihedral(x, i1, i2, i3, i4).to_degrees();
            let psi = compute_dihedral(x, i2, i3, i4, i5).to_degrees();
            let (e, de_dphi, de_dpsi) = self.maps[self.map[t]].eval(phi, psi);
            energy += e;

            for d in 0..3 {
                let dphidr1 = r32 / a1sq * a1[d];
                let dphidr2 = -r32 / a1sq * a1[d] - dpr21r32 / a1sq / r32 * a1[d]
                    + dpr34r32 / b1sq / r32 * b1[d];
                let dphidr3 = dpr34r32 / b1sq / r32 * b1[d]
                    - dpr21r32 / a1sq / r32 * a1[d]
                    - r32 / b1sq * b1[d];
                let dphidr4 = r32 / b1sq * b1[d];

                let dpsidr1 = r43 / a2sq * a2[d];
                let dpsidr2 = r43 / a2sq * a2[d] + dpr32r43 / a2sq / r43 * a2[d]
                    - dpr45r43 / b2sq / r43 * b2[d];
                let dpsidr3 = dpr45r43 / b2sq / r43 * b2[d]
                    - dpr32r43 / a2sq / r43 * a2[d]
                    - r43 / b2sq * b2[d];
                let dpsidr4 = r43 / b2sq * b2[d];

                f[3 * i1 + d] += de_dphi * dphidr1;
                f[3 * i2 + d] += de_dphi * dphidr2 + de_dpsi * dpsidr1;
                f[3 * i3 + d] += -de_dphi * dphidr3 - de_dpsi * dpsidr2;
                f[3 * i4 + d] += -de_dphi * dphidr4 - de_dpsi * dpsidr3;
                f[3 * i5 + d] += -de_dpsi * dpsidr4;
            }
        }
        energy
    }
}

impl Potential for CmapCharmm {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let a = &self.atoms;
        let energy = self.fold(coords, acc.forces_without_virial(), self.map.len(), |t| {
            [a[0][t], a[1][t], a[2][t], a[3][t], a[4][t]]
        });
        acc.add_energy(energy);
    }
}

impl IndexedTerms for CmapCharmm {
    fn terms(&self) -> Array2<u32> {
        let a = &self.atoms;
        term_table(&[&a[0], &a[1], &a[2], &a[3], &a[4]])
    }

    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.map.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            std::array::from_fn(|p| terms[[t, p]] as usize)
        });
        acc.add_energy(energy);
    }
}

/// Construct a [`CmapCharmm`] from per-type params (each a `grid`) and a
/// Frame's `cmaps` block (`atomi` … `atomm`, `type`).
///
/// Only the maps a crossterm uses are prepared. `Err` on a missing column, an
/// unknown type label, a type without a `grid`, or a grid [`CmapGrid::new`]
/// refuses.
pub fn cmap_charmm_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(CMAPS)
        .ok_or("cmap_charmm: missing \"cmaps\" block")?;
    let endpoint = |key: &str| {
        block
            .get(key)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| format!("cmap_charmm: cmaps block missing u64 column {key:?}"))
    };
    let cols = [
        endpoint(ATOMI)?,
        endpoint(ATOMJ)?,
        endpoint(ATOMK)?,
        endpoint(ATOML)?,
        endpoint(ATOMM)?,
    ];
    let types = block
        .get(TYPE)
        .and_then(|c| c.get::<String>())
        .ok_or("cmap_charmm: cmaps block missing string column \"type\"")?;

    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut maps = Vec::new();
    let mut atoms = Vec::with_capacity(types.len());
    let mut map = Vec::with_capacity(types.len());
    for (row, label) in types.iter().enumerate() {
        let label = label.as_str();
        let m = match index.get(label) {
            Some(&m) => m,
            None => {
                let params = type_map
                    .get(label)
                    .ok_or_else(|| format!("cmap_charmm: unknown type '{label}'"))?;
                let grid = params
                    .get(CMAP_GRID)
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| param_reads::missing("charmm", label, CMAP_GRID))?;
                maps.push(
                    CmapGrid::new(grid)
                        .map_err(|e| param_reads::bad("charmm", label, CMAP_GRID, e))?,
                );
                index.insert(label, maps.len() - 1);
                maps.len() - 1
            }
        };
        atoms.push(std::array::from_fn(|p| cols[p][[row]] as usize));
        map.push(m);
    }
    Ok(ForceTerm::indexed(CmapCharmm::new(atoms, map, maps)))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::flat_coords::compute_dihedral;
    use crate::io::lammps::forcefield_reader::parse_cmap_file;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    /// The CHARMM36 alanine map (LAMMPS `potentials/charmm36.cmap`, type 1).
    pub(crate) fn alanine() -> ArrayD<f64> {
        parse_cmap_file(include_str!("testdata/charmm36_alanine.cmap"))
            .unwrap()
            .maps
            .remove(0)
    }

    /// The atom bonded to `c` at `bond` Å, angle `b-c-new` = `angle`°, and
    /// dihedral `a-b-c-new` = `tors`° (natural extension of reference frame).
    pub(crate) fn place(a: [F; 3], b: [F; 3], c: [F; 3], bond: F, angle: F, tors: F) -> [F; 3] {
        let unit = |v: [F; 3]| {
            let n = dot(v, v).sqrt();
            [v[0] / n, v[1] / n, v[2] / n]
        };
        let bc = unit([c[0] - b[0], c[1] - b[1], c[2] - b[2]]);
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let n = unit(cross(ab, bc));
        let m = cross(n, bc);
        let (th, ph) = (angle.to_radians(), tors.to_radians());
        let d = [
            -bond * th.cos(),
            bond * th.sin() * ph.cos(),
            bond * th.sin() * ph.sin(),
        ];
        std::array::from_fn(|k| c[k] + d[0] * bc[k] + d[1] * m[k] + d[2] * n[k])
    }

    /// Five backbone-like atoms with dihedral(0,1,2,3) = φ and
    /// dihedral(1,2,3,4) = ψ, degrees.
    pub(crate) fn chain(phi: F, psi: F) -> Vec<F> {
        let a = [0.3, 1.4, 0.2];
        let b = [0.0, 0.0, 0.0];
        let c = [1.46, 0.0, 0.0];
        let d = place(a, b, c, 1.52, 111.0, phi);
        let e = place(b, c, d, 1.33, 116.0, psi);
        [a, b, c, d, e].concat()
    }

    fn one(grid: &ArrayD<f64>) -> CmapCharmm {
        CmapCharmm::new(
            vec![[0, 1, 2, 3, 4]],
            vec![0],
            vec![CmapGrid::new(grid).unwrap()],
        )
    }

    #[test]
    fn chain_has_the_dihedrals_it_was_built_with() {
        for (phi, psi) in [(-63.0, -41.0), (170.0, -179.0), (0.0, 90.0)] {
            let x = chain(phi, psi);
            let got = (
                compute_dihedral(&x, 0, 1, 2, 3).to_degrees(),
                compute_dihedral(&x, 1, 2, 3, 4).to_degrees(),
            );
            assert!((got.0 - phi).abs() < 1e-10 && (got.1 - psi).abs() < 1e-10);
        }
    }

    /// At a node the bicubic patch is its corner value, so E is the grid.
    #[test]
    fn energy_at_the_nodes_is_the_grid() {
        let grid = alanine();
        let map = CmapGrid::new(&grid).unwrap();
        let pot = one(&grid);
        for i in 0..24 {
            for j in 0..24 {
                let (phi, psi) = (-180.0 + 15.0 * i as F, -180.0 + 15.0 * j as F);
                let want = grid[[i, j]];
                let (e, _, _) = map.eval(phi, psi);
                assert!((e - want).abs() <= 1e-12, "({phi}, {psi}): {e} vs {want}");
                let e = pot.energy_at(&chain(phi, psi));
                assert!(
                    (e - want).abs() <= 1e-12,
                    "xyz ({phi}, {psi}): {e} vs {want}"
                );
            }
        }
    }

    /// E and its slopes are continuous across the ±180° seam of both angles
    /// and across interior cell edges.
    #[test]
    fn continuous_across_the_seam_and_cell_edges() {
        let map = CmapGrid::new(&alanine()).unwrap();
        let eps = 1e-9;
        for k in 0..48 {
            let other = -180.0 + 7.5 * k as F + 1.3;
            for (lo, hi) in [(180.0 - eps, -180.0), (-165.0 - eps, -165.0), (-eps, 0.0)] {
                for (a, b) in [
                    (map.eval(lo, other), map.eval(hi, other)),
                    (map.eval(other, lo), map.eval(other, hi)),
                ] {
                    assert!(
                        (a.0 - b.0).abs() < 1e-7,
                        "E at {lo}|{hi}, {other}: {a:?} {b:?}"
                    );
                    assert!((a.1 - b.1).abs() < 1e-5, "dφ at {lo}|{hi}: {a:?} {b:?}");
                    assert!((a.2 - b.2).abs() < 1e-5, "dψ at {lo}|{hi}: {a:?} {b:?}");
                }
            }
            // 180° itself reads as −180°.
            assert_eq!(map.eval(180.0, other), map.eval(-180.0, other));
        }
    }

    /// A geometry off the ideal chain, through every quadrant of (φ, ψ).
    fn bent(phi: F, psi: F) -> Vec<F> {
        let mut x = chain(phi, psi);
        for (k, v) in x.iter_mut().enumerate() {
            *v += 0.03 * ((k * 7 % 11) as F - 5.0) / 5.0;
        }
        x
    }

    #[test]
    fn forces_are_minus_the_finite_difference_gradient() {
        let pot = one(&alanine());
        for (phi, psi) in [
            (-63.0, -41.0),
            (-120.0, 130.0),
            (60.0, 30.0),
            (178.0, -177.0),
        ] {
            let x = bent(phi, psi);
            let (_, f) = pot.energy_forces_at(&x);
            let h = 1e-6;
            for d in 0..x.len() {
                let (mut p, mut m) = (x.clone(), x.clone());
                p[d] += h;
                m[d] -= h;
                let fd = -(pot.energy_at(&p) - pot.energy_at(&m)) / (2.0 * h);
                assert!(
                    (f[d] - fd).abs() <= 1e-6 * (1.0 + fd.abs()),
                    "({phi}, {psi}) component {d}: analytic {} vs fd {fd}",
                    f[d]
                );
            }
        }
    }

    /// The forces sum to zero and exert no torque (Newton's third law; E is
    /// invariant under rigid motion).
    #[test]
    fn newtons_third_law() {
        let pot = one(&alanine());
        for (phi, psi) in [(-63.0, -41.0), (100.0, -100.0)] {
            let x = bent(phi, psi);
            let (_, f) = pot.energy_forces_at(&x);
            let scale: F = f.iter().map(|v| v.abs()).sum();
            assert!(scale > 1e-3, "nothing to balance");
            let mut net = [0.0; 3];
            let mut torque = [0.0; 3];
            for a in 0..5 {
                let r = [x[3 * a], x[3 * a + 1], x[3 * a + 2]];
                let fa = [f[3 * a], f[3 * a + 1], f[3 * a + 2]];
                let t = cross(r, fa);
                for k in 0..3 {
                    net[k] += fa[k];
                    torque[k] += t[k];
                }
            }
            for k in 0..3 {
                assert!(net[k].abs() < 1e-12 * scale, "net force {net:?}");
                assert!(torque[k].abs() < 1e-11 * scale, "net torque {torque:?}");
            }
        }
    }

    /// The rectangle defect E(a,c) + E(b,d) − E(a,d) − E(b,c), which is zero
    /// for a sum of a function of φ and one of ψ.
    fn defect(map: &CmapGrid, (a, b): (F, F), (c, d): (F, F)) -> F {
        map.eval(a, c).0 + map.eval(b, d).0 - map.eval(a, d).0 - map.eval(b, c).0
    }

    /// A separable grid f(φ) + g(ψ) interpolates separably (the spline
    /// cross-derivatives vanish, so the patch is a sum of two 1-D cubics),
    /// and tracks the analytic f + g and its slopes; CHARMM's own map is not
    /// separable — no sum of 1-D torsions prices it.
    #[test]
    fn a_separable_grid_stays_separable_and_tracks_its_function() {
        let f = |p: F| 1.5 * p.to_radians().cos();
        let g = |q: F| 0.5 * (2.0 * q.to_radians()).sin();
        let grid = ArrayD::from_shape_fn(vec![24, 24], |ix| {
            f(-180.0 + 15.0 * ix[0] as F) + g(-180.0 + 15.0 * ix[1] as F)
        });
        let map = CmapGrid::new(&grid).unwrap();
        assert!(map.d12.iter().all(|v| v.abs() < 1e-13), "{:?}", map.d12);
        for (a, b, c, d) in [(-170.0, 33.3, -12.0, 151.0), (5.0, 95.5, -100.0, 179.9)] {
            assert!(defect(&map, (a, b), (c, d)).abs() < 1e-12);
        }
        for k in 0..100 {
            let (p, q) = (-180.0 + 3.61 * k as F, 179.0 - 3.37 * k as F);
            let (e, dp, dq) = map.eval(p, q);
            assert!((e - (f(p) + g(q))).abs() < 2e-3, "E({p}, {q}) = {e}");
            let (dp_want, dq_want) = (-1.5 * p.to_radians().sin(), (2.0 * q.to_radians()).cos());
            assert!(
                (dp - dp_want).abs() < 2e-2,
                "dE/dφ({p}) = {dp} vs {dp_want}"
            );
            assert!(
                (dq - dq_want).abs() < 2e-2,
                "dE/dψ({q}) = {dq} vs {dq_want}"
            );
        }
        let charmm = CmapGrid::new(&alanine()).unwrap();
        assert!(defect(&charmm, (-63.0, -120.0), (-41.0, 130.0)).abs() > 0.1);
    }

    /// `fix cmap` skips a crossterm with a degenerate dihedral plane.
    #[test]
    fn a_collinear_crossterm_contributes_nothing() {
        let pot = one(&alanine());
        let x = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.5, 1.0, 0.0,
        ];
        assert_eq!(pot.energy_forces_at(&x), (0.0, vec![0.0; 15]));
    }

    /// Two crossterms on a six-atom chain, typed `ala` and `flat`.
    fn two_crossterm_frame() -> (ForceField, Frame, Vec<F>) {
        let mut ff = ForceField::new("t");
        let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
        let mut ala = Params::new();
        ala.set(CMAP_GRID, alanine());
        style
            .def_type("ala", &["C", "NH1", "CT1", "C", "NH1"], ala)
            .unwrap();
        let mut flat = Params::new();
        flat.set(CMAP_GRID, alanine().mapv(|v| 0.5 * v + 0.1));
        style
            .def_type("flat", &["NH1", "CT1", "C", "NH1", "CT1"], flat)
            .unwrap();

        let mut x = chain(-63.0, -41.0);
        let n = x.len() / 3;
        let next = place(
            [x[3 * (n - 3)], x[3 * (n - 3) + 1], x[3 * (n - 3) + 2]],
            [x[3 * (n - 2)], x[3 * (n - 2) + 1], x[3 * (n - 2) + 2]],
            [x[3 * (n - 1)], x[3 * (n - 1) + 1], x[3 * (n - 1) + 2]],
            1.45,
            121.0,
            -150.0,
        );
        x.extend(next);
        let mut atoms = Block::new();
        for (k, key) in ["x", "y", "z"].into_iter().enumerate() {
            let col: Vec<F> = x.iter().skip(k).step_by(3).copied().collect();
            atoms
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let mut cmaps = Block::new();
        for (p, key) in [ATOMI, ATOMJ, ATOMK, ATOML, ATOMM].into_iter().enumerate() {
            let col: Vec<Idx> = vec![p as Idx, p as Idx + 1];
            cmaps
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        cmaps
            .insert(
                TYPE,
                Array1::from_vec(vec!["ala".to_owned(), "flat".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert(CMAPS, cmaps);
        (ff, frame, x)
    }

    /// Both compile doors build the same crossterms, bit for bit, and a
    /// kernel handed back its own term table answers identically.
    #[test]
    fn compile_and_compile_typed_agree_and_terms_rebind() {
        let (ff, frame, x) = two_crossterm_frame();
        let compiler = PotentialCompiler::new(&ff);
        let pots = compiler.compile(&frame).unwrap();
        let typed = compiler.compile_typed(&frame).unwrap();
        assert_eq!((pots.members().len(), typed.len()), (1, 1));
        let (e0, f0) = pots.energy_forces_at(&x);
        let (e1, f1) = typed[0].0.energy_forces_at(&x);
        assert_eq!(e0.to_bits(), e1.to_bits());
        assert_eq!(f0, f1);
        assert!(e0 != 0.0);

        let ForceTerm::Indexed(k) = &pots.members()[0] else {
            panic!("a cmap member is indexed")
        };
        let terms = k.terms();
        assert_eq!(terms.shape(), &[2, 5]);
        let (e2, f2) = k.over_terms(&x, terms.view());
        assert_eq!(e0.to_bits(), e2.to_bits());
        assert_eq!(f0, f2);
    }

    #[test]
    fn the_ctor_refuses_what_it_cannot_price() {
        let (_, frame, _) = two_crossterm_frame();
        let mut bare = ForceField::new("t");
        let style = bare.def_style("cmap", "charmm", Params::new()).unwrap();
        style.def_type("ala", &["C"; 5], Params::new()).unwrap();
        let err = PotentialCompiler::new(&bare)
            .compile(&frame)
            .map(|_| ())
            .unwrap_err();
        assert!(
            err.to_string().contains("grid") || err.to_string().contains("unknown type"),
            "{err}"
        );
        let mut odd = Params::new();
        odd.set(CMAP_GRID, ArrayD::zeros(vec![3, 4]));
        assert!(CmapGrid::new(odd.get(CMAP_GRID).and_then(|v| v.as_array()).unwrap()).is_err());
    }
}
