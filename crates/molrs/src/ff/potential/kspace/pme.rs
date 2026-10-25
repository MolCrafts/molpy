//! Particle Mesh Ewald (PME) electrostatic potential.
//!
//! The real-space erfc term is the pair form
//! [`CoulLong`](crate::ff::potential::pair::CoulLong). This kernel is the
//! rest: the self energy, the exclusion correction and the reciprocal-space
//! sum. A compiled frame with no `pairs` block still prices the real-space
//! term here, minimum-imaged, because there is no list to hand the form.

use crate::ff::potential::form_kernel::ScalarPair;
use crate::ff::potential::pair::CoulLong;
use crate::ff::potential::pair::coul_long::real_space_pair;
use crate::ff::potential::param_reads;
use crate::ff::potential::{Accumulator, CompileError, ForceTerm, PairDriven, Potential};
use molrs::core::Neighbors;
use molrs::core::schema::block_names::{ATOMS, EXCLUSIONS, PAIRS};
use std::borrow::Cow;
use std::collections::HashSet;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::ff::ir::Params;
use molrs::core::Frame;
use molrs::core::Mic;
use molrs::op::F;

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn erf_f(x: F) -> F {
    libm::erf(x)
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// PME configuration parameters.
#[derive(Debug, Clone)]
pub struct PairCoulLongPmeParams {
    /// Ewald splitting parameter (1/length).
    pub alpha: F,
    /// Real-space cutoff distance.
    pub cutoff: F,
    /// FFT grid dimensions `[Kx, Ky, Kz]`.
    pub grid_size: [usize; 3],
    /// B-spline interpolation order (typically 4 or 5).
    pub order: usize,
    /// Coulomb constant (e.g. 332.0636 for kcal/mol units, or 1.0).
    pub coulomb: F,
}

// ---------------------------------------------------------------------------
// FFT plan cache
// ---------------------------------------------------------------------------

struct FftPlans {
    fwd: [Arc<dyn Fft<F>>; 3],
    inv: [Arc<dyn Fft<F>>; 3],
}

// ---------------------------------------------------------------------------
// Scratch buffers (interior mutability for `&self`)
// ---------------------------------------------------------------------------

struct PmeScratch {
    grid: Vec<Complex<F>>,        // Kx*Ky*Kz complex grid
    buf: Vec<Complex<F>>,         // temp for 1D FFT rows
    fft_scratch: Vec<Complex<F>>, // rustfft scratch
    /// One atom's B-spline weights (`order` entries), reused across atoms.
    spline: Vec<[F; 3]>,
    /// Their derivatives, same shape.
    dspline: Vec<[F; 3]>,
}

// ---------------------------------------------------------------------------
// PairCoulLongPme
// ---------------------------------------------------------------------------

/// PME electrostatic potential implementing [`Potential`]: self-energy,
/// exclusion correction, reciprocal space (a 3D FFT built from 1D
/// `rustfft`), and — when this kernel was built to price it — the
/// real-space erfc term.
///
/// Registered as the built-in kernel `("pair", "coul/long/pme")`. The
/// constructor reads charges from `frame["atoms"]["charge"]` (float), the
/// periodic cell from the frame's own box ([`Frame::simbox`]) — frame data,
/// as LAMMPS's kspace reads its simulation box, never a force-field
/// parameter — and exclusion pairs from `frame["exclusions"]` (`atomi`,
/// `atomj` columns).
///
/// The real-space term is the pair form
/// [`CoulLong`](crate::ff::potential::pair::CoulLong) whenever the compiled
/// frame has a `pairs` block, and on the neighbour-driven door always. This
/// kernel then prices only the reciprocal part (`direct` off), so the erfc
/// term is not counted twice. With no `pairs` block the compiled door keeps
/// the real-space loop here.
pub struct PairCoulLongPme {
    params: PairCoulLongPmeParams,
    n_atoms: usize,
    charges: Vec<F>,
    h: [[F; 3]; 3],       // box matrix (row-major, lower-triangular)
    recip_h: [[F; 3]; 3], // inverse box matrix
    volume: F,
    exclusions: Vec<[usize; 2]>,
    self_energy: F,
    bspline_moduli: [Vec<F>; 3],
    fft_plans: FftPlans,
    scratch: Mutex<PmeScratch>,
    /// Minimum-image convention of the box, the same kernel every other
    /// pair loop in molrs uses (`SimBox::mic`).
    mic: Mic,
    /// Whether [`evaluate`](Potential::evaluate) adds the real-space erfc
    /// loop. Off when that term is the pair form beside this kernel.
    direct: bool,
}

impl PairCoulLongPme {
    /// Construct a PME potential that prices the real-space erfc loop.
    ///
    /// * `charges` — per-atom partial charges (length `n_atoms`).
    /// * `box_vectors` — 3×3 box matrix, row-major, lower-triangular.
    /// * `exclusions` — pairs `[i, j]` with `i < j` whose reciprocal-space
    ///   interaction must be subtracted.
    pub fn new(
        params: PairCoulLongPmeParams,
        charges: Vec<F>,
        box_vectors: [[F; 3]; 3],
        exclusions: Vec<[usize; 2]>,
    ) -> Self {
        Self::build(params, charges, box_vectors, exclusions, true)
    }

    /// The reciprocal sum, the self energy and the exclusion correction.
    /// The real-space erfc term is the pair form beside this kernel.
    #[cfg(test)]
    pub(crate) fn reciprocal(
        params: PairCoulLongPmeParams,
        charges: Vec<F>,
        box_vectors: [[F; 3]; 3],
        exclusions: Vec<[usize; 2]>,
    ) -> Self {
        Self::build(params, charges, box_vectors, exclusions, false)
    }

    fn build(
        params: PairCoulLongPmeParams,
        charges: Vec<F>,
        box_vectors: [[F; 3]; 3],
        exclusions: Vec<[usize; 2]>,
        direct: bool,
    ) -> Self {
        let n_atoms = charges.len();
        let h = box_vectors;
        let recip_h = invert_box_vectors(&h);
        let volume = h[0][0] * h[1][1] * h[2][2]; // lower-triangular determinant
        // `h` keeps the lattice vectors as rows; `Mic` wants them as columns.
        let cell = std::array::from_fn(|i| std::array::from_fn(|j| h[j][i]));
        let mic = Mic::new(cell, [true; 3]).expect("PME requires a valid periodic cell");

        // Self energy: -α/√π * C * Σq²
        let sum_q2: F = charges.iter().map(|q| q * q).sum();
        let self_energy = -(params.alpha / PI.sqrt()) * params.coulomb * sum_q2;

        // B-spline moduli
        let bspline_moduli = [
            compute_bspline_moduli(params.grid_size[0], params.order),
            compute_bspline_moduli(params.grid_size[1], params.order),
            compute_bspline_moduli(params.grid_size[2], params.order),
        ];

        // FFT plans
        let [kx, ky, kz] = params.grid_size;
        let mut planner = FftPlanner::<F>::new();
        let fft_plans = FftPlans {
            fwd: [
                planner.plan_fft_forward(kx),
                planner.plan_fft_forward(ky),
                planner.plan_fft_forward(kz),
            ],
            inv: [
                planner.plan_fft_inverse(kx),
                planner.plan_fft_inverse(ky),
                planner.plan_fft_inverse(kz),
            ],
        };

        // Scratch
        let grid_len = kx * ky * kz;
        let max_dim = kx.max(ky).max(kz);
        let max_fft_scratch = fft_plans
            .fwd
            .iter()
            .chain(fft_plans.inv.iter())
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let zero = Complex::new(0.0, 0.0);
        let scratch = Mutex::new(PmeScratch {
            grid: vec![zero; grid_len],
            buf: vec![zero; max_dim],
            fft_scratch: vec![zero; max_fft_scratch],
            spline: vec![[0.0; 3]; params.order],
            dspline: vec![[0.0; 3]; params.order],
        });

        Self {
            params,
            n_atoms,
            charges,
            h,
            recip_h,
            volume,
            exclusions,
            self_energy,
            bspline_moduli,
            fft_plans,
            scratch,
            mic,
            direct,
        }
    }

    // -----------------------------------------------------------------------
    // Direct space energy + gradient
    // -----------------------------------------------------------------------

    fn direct_energy(&self, coords: &[F]) -> F {
        let alpha = self.params.alpha;
        let cutoff = self.params.cutoff;
        let coulomb = self.params.coulomb;
        let cutoff2 = cutoff * cutoff;
        let mut energy: F = 0.0;

        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                if self.is_excluded(i, j) {
                    continue;
                }
                let (dx, dy, dz) = self.min_image_delta(coords, i, j);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 {
                    continue;
                }
                let r = r2.sqrt();
                let (e, _) = real_space_pair(coulomb, alpha, self.charges[i], self.charges[j], r);
                energy += e;
            }
        }
        energy
    }

    fn direct_gradient(&self, coords: &[F], grad: &mut [F]) {
        let alpha = self.params.alpha;
        let cutoff = self.params.cutoff;
        let coulomb = self.params.coulomb;
        let cutoff2 = cutoff * cutoff;

        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                if self.is_excluded(i, j) {
                    continue;
                }
                let (dx, dy, dz) = self.min_image_delta(coords, i, j);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 {
                    continue;
                }
                let r = r2.sqrt();
                let (_, de) = real_space_pair(coulomb, alpha, self.charges[i], self.charges[j], r);
                // `de` is dE/dr. The gradient stores ∂E/∂x; evaluate turns it into a force.
                let factor = de / r;
                grad[j * 3] += factor * dx;
                grad[j * 3 + 1] += factor * dy;
                grad[j * 3 + 2] += factor * dz;
                grad[i * 3] -= factor * dx;
                grad[i * 3 + 1] -= factor * dy;
                grad[i * 3 + 2] -= factor * dz;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Exclusion correction energy + gradient
    // -----------------------------------------------------------------------

    fn exclusion_energy(&self, coords: &[F]) -> F {
        let alpha = self.params.alpha;
        let coulomb = self.params.coulomb;
        let mut energy: F = 0.0;

        for &[i, j] in &self.exclusions {
            let (dx, dy, dz) = self.delta(coords, i, j);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-15 {
                continue;
            }
            let alpha_r = alpha * r;
            energy -= coulomb * self.charges[i] * self.charges[j] * erf_f(alpha_r) / r;
        }
        energy
    }

    fn exclusion_gradient(&self, coords: &[F], grad: &mut [F]) {
        let alpha = self.params.alpha;
        let coulomb = self.params.coulomb;
        let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();

        for &[i, j] in &self.exclusions {
            let (dx, dy, dz) = self.delta(coords, i, j);
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-30 {
                continue;
            }
            let r = r2.sqrt();
            let alpha_r = alpha * r;
            let qi_qj = self.charges[i] * self.charges[j];
            let factor = coulomb
                * qi_qj
                * (erf_f(alpha_r) - two_alpha_over_sqrt_pi * r * (-alpha_r * alpha_r).exp())
                / (r2 * r);
            grad[j * 3] += factor * dx;
            grad[j * 3 + 1] += factor * dy;
            grad[j * 3 + 2] += factor * dz;
            grad[i * 3] -= factor * dx;
            grad[i * 3 + 1] -= factor * dy;
            grad[i * 3 + 2] -= factor * dz;
        }
    }

    // -----------------------------------------------------------------------
    // Reciprocal space energy + gradient
    // -----------------------------------------------------------------------

    fn reciprocal_energy(&self, coords: &[F]) -> F {
        let mut scratch = self.scratch.lock().unwrap();
        let sqrt_coulomb = self.params.coulomb.sqrt();

        scratch.grid.fill(Complex::new(0.0, 0.0));

        // Spread charges onto grid
        {
            let PmeScratch { grid, spline, .. } = &mut *scratch;
            self.spread_charges(coords, grid, spline, sqrt_coulomb);
        }

        // Forward 3D FFT — destructure to satisfy borrow checker
        let PmeScratch {
            ref mut grid,
            ref mut buf,
            ref mut fft_scratch,
            ..
        } = *scratch;
        self.fft_3d_forward(grid, buf, fft_scratch);

        // Convolution + energy accumulation
        let energy = self.reciprocal_convolution(&mut scratch.grid);
        0.5 * energy
    }

    fn reciprocal_gradient(&self, coords: &[F], grad: &mut [F]) {
        let mut scratch = self.scratch.lock().unwrap();
        let sqrt_coulomb = self.params.coulomb.sqrt();

        scratch.grid.fill(Complex::new(0.0, 0.0));

        // Spread charges
        {
            let PmeScratch { grid, spline, .. } = &mut *scratch;
            self.spread_charges(coords, grid, spline, sqrt_coulomb);
        }

        // Forward FFT
        {
            let PmeScratch {
                ref mut grid,
                ref mut buf,
                ref mut fft_scratch,
                ..
            } = *scratch;
            self.fft_3d_forward(grid, buf, fft_scratch);
        }

        // Convolution (modifies grid in-place)
        let _ = self.reciprocal_convolution(&mut scratch.grid);

        // Inverse FFT (unnormalized, matching C++ irfftn with norm="forward")
        {
            let PmeScratch {
                ref mut grid,
                ref mut buf,
                ref mut fft_scratch,
                ..
            } = *scratch;
            self.fft_3d_inverse(grid, buf, fft_scratch);
        }

        // Interpolate forces from grid
        let PmeScratch {
            grid,
            spline,
            dspline,
            ..
        } = &mut *scratch;
        self.interpolate_forces(coords, grid, spline, dspline, sqrt_coulomb, grad);
    }

    // -----------------------------------------------------------------------
    // Charge spreading
    // -----------------------------------------------------------------------

    fn spread_charges(
        &self,
        coords: &[F],
        grid: &mut [Complex<F>],
        data: &mut [[F; 3]],
        sqrt_coulomb: F,
    ) {
        let [kx, ky, kz] = self.params.grid_size;
        let order = self.params.order;

        for atom in 0..self.n_atoms {
            let grid_index = self.compute_spline(coords, atom, data);

            for ix in 0..order {
                let xindex = (grid_index[0] + ix) % kx;
                let dx = self.charges[atom] * sqrt_coulomb * data[ix][0];
                for iy in 0..order {
                    let yindex = (grid_index[1] + iy) % ky;
                    let dxdy = dx * data[iy][1];
                    for (iz, spline_z) in data.iter().enumerate().take(order) {
                        let zindex = (grid_index[2] + iz) % kz;
                        let index = xindex * ky * kz + yindex * kz + zindex;
                        grid[index].re += dxdy * spline_z[2];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Force interpolation from reciprocal grid
    // -----------------------------------------------------------------------

    fn interpolate_forces(
        &self,
        coords: &[F],
        grid: &[Complex<F>],
        data: &mut [[F; 3]],
        ddata: &mut [[F; 3]],
        sqrt_coulomb: F,
        grad: &mut [F],
    ) {
        let [kx, ky, kz] = self.params.grid_size;
        let order = self.params.order;

        for atom in 0..self.n_atoms {
            let grid_index = self.compute_spline_with_deriv(coords, atom, data, ddata);

            let mut dpos = [0.0 as F; 3];
            for ix in 0..order {
                let xindex = (grid_index[0] + ix) % kx;
                let dx = data[ix][0];
                let ddx = ddata[ix][0];
                for iy in 0..order {
                    let yindex = (grid_index[1] + iy) % ky;
                    let dy = data[iy][1];
                    let ddy = ddata[iy][1];
                    for iz in 0..order {
                        let zindex = (grid_index[2] + iz) % kz;
                        let dz = data[iz][2];
                        let ddz = ddata[iz][2];
                        let g = grid[xindex * ky * kz + yindex * kz + zindex].re;
                        dpos[0] += ddx * dy * dz * g;
                        dpos[1] += dx * ddy * dz * g;
                        dpos[2] += dx * dy * ddz * g;
                    }
                }
            }

            let scale = self.charges[atom] * sqrt_coulomb;
            let rh = &self.recip_h;
            let gs = self.params.grid_size;
            // Chain rule: fractional → Cartesian
            grad[atom * 3] += scale * (dpos[0] * gs[0] as F * rh[0][0]);
            grad[atom * 3 + 1] +=
                scale * (dpos[0] * gs[0] as F * rh[1][0] + dpos[1] * gs[1] as F * rh[1][1]);
            grad[atom * 3 + 2] += scale
                * (dpos[0] * gs[0] as F * rh[2][0]
                    + dpos[1] * gs[1] as F * rh[2][1]
                    + dpos[2] * gs[2] as F * rh[2][2]);
        }
    }

    // -----------------------------------------------------------------------
    // Reciprocal convolution — returns raw energy (before ×0.5)
    // -----------------------------------------------------------------------

    fn reciprocal_convolution(&self, grid: &mut [Complex<F>]) -> F {
        let [kx, ky, kz] = self.params.grid_size;
        let alpha = self.params.alpha;
        let recip_exp_factor = PI * PI / (alpha * alpha);
        let rh = &self.recip_h;
        let scale_factor = PI * self.volume;
        let xmod = &self.bspline_moduli[0];
        let ymod = &self.bspline_moduli[1];
        let zmod = &self.bspline_moduli[2];

        let mut energy: F = 0.0;

        for (ikx, &xmod_k) in xmod.iter().enumerate().take(kx) {
            let mx = if ikx < kx.div_ceil(2) {
                ikx as i64
            } else {
                ikx as i64 - kx as i64
            };
            let mhx = mx as F * rh[0][0];
            let bx = scale_factor * xmod_k;

            for (iky, &ymod_k) in ymod.iter().enumerate().take(ky) {
                let my = if iky < ky.div_ceil(2) {
                    iky as i64
                } else {
                    iky as i64 - ky as i64
                };
                let mhy = mx as F * rh[1][0] + my as F * rh[1][1];
                let mhx2y2 = mhx * mhx + mhy * mhy;
                let bxby = bx * ymod_k;

                for (ikz, &bz) in zmod.iter().enumerate().take(kz) {
                    let index = ikx * ky * kz + iky * kz + ikz;
                    let mz = if ikz < kz.div_ceil(2) {
                        ikz as i64
                    } else {
                        ikz as i64 - kz as i64
                    };
                    let mhz = mx as F * rh[2][0] + my as F * rh[2][1] + mz as F * rh[2][2];
                    let m2 = mhx2y2 + mhz * mhz;
                    let denom = m2 * bxby * bz;
                    let eterm = if index == 0 {
                        0.0
                    } else {
                        (-recip_exp_factor * m2).exp() / denom
                    };

                    let g = grid[index];
                    energy += eterm * (g.re * g.re + g.im * g.im);
                    grid[index] = g * eterm;
                }
            }
        }
        energy
    }

    // -----------------------------------------------------------------------
    // 3D FFT via three rounds of 1D FFT
    // -----------------------------------------------------------------------

    fn fft_3d_forward(
        &self,
        grid: &mut [Complex<F>],
        buf: &mut [Complex<F>],
        fft_scratch: &mut [Complex<F>],
    ) {
        let [kx, ky, kz] = self.params.grid_size;

        // Round 1: along z (contiguous) — Kx*Ky batches of length Kz
        for i in 0..(kx * ky) {
            let start = i * kz;
            self.fft_plans.fwd[2].process_with_scratch(&mut grid[start..start + kz], fft_scratch);
        }

        // Round 2: along y (stride Kz) — Kx*Kz batches of length Ky
        for ix in 0..kx {
            for iz in 0..kz {
                // Gather
                for iy in 0..ky {
                    buf[iy] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.fwd[1].process_with_scratch(&mut buf[..ky], fft_scratch);
                // Scatter
                for iy in 0..ky {
                    grid[ix * ky * kz + iy * kz + iz] = buf[iy];
                }
            }
        }

        // Round 3: along x (stride Ky*Kz) — Ky*Kz batches of length Kx
        for iy in 0..ky {
            for iz in 0..kz {
                // Gather
                for ix in 0..kx {
                    buf[ix] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.fwd[0].process_with_scratch(&mut buf[..kx], fft_scratch);
                // Scatter
                for ix in 0..kx {
                    grid[ix * ky * kz + iy * kz + iz] = buf[ix];
                }
            }
        }
    }

    fn fft_3d_inverse(
        &self,
        grid: &mut [Complex<F>],
        buf: &mut [Complex<F>],
        fft_scratch: &mut [Complex<F>],
    ) {
        let [kx, ky, kz] = self.params.grid_size;

        // Round 1: along x
        for iy in 0..ky {
            for iz in 0..kz {
                for ix in 0..kx {
                    buf[ix] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.inv[0].process_with_scratch(&mut buf[..kx], fft_scratch);
                for ix in 0..kx {
                    grid[ix * ky * kz + iy * kz + iz] = buf[ix];
                }
            }
        }

        // Round 2: along y
        for ix in 0..kx {
            for iz in 0..kz {
                for iy in 0..ky {
                    buf[iy] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.inv[1].process_with_scratch(&mut buf[..ky], fft_scratch);
                for iy in 0..ky {
                    grid[ix * ky * kz + iy * kz + iz] = buf[iy];
                }
            }
        }

        // Round 3: along z (contiguous)
        for i in 0..(kx * ky) {
            let start = i * kz;
            self.fft_plans.inv[2].process_with_scratch(&mut grid[start..start + kz], fft_scratch);
        }
    }

    // -----------------------------------------------------------------------
    // B-spline computation
    // -----------------------------------------------------------------------

    /// Fill `data` (`order` rows) with an atom's B-spline coefficients and
    /// return its grid index.
    fn compute_spline(&self, coords: &[F], atom: usize, data: &mut [[F; 3]]) -> [usize; 3] {
        let order = self.params.order;
        let gs = self.params.grid_size;
        let pos = [coords[atom * 3], coords[atom * 3 + 1], coords[atom * 3 + 2]];

        // Wrap position into box
        let mut pos_in_box = pos;
        for i in (0..3).rev() {
            let s = (pos_in_box[i] * self.recip_h[i][i]).floor();
            for (j, pos_j) in pos_in_box.iter_mut().enumerate() {
                *pos_j -= s * self.h[i][j];
            }
        }

        // Fractional coordinates → grid coordinates
        let mut grid_index = [0usize; 3];
        let mut dr = [0.0 as F; 3];
        for i in 0..3 {
            let mut t = pos_in_box[0] * self.recip_h[0][i]
                + pos_in_box[1] * self.recip_h[1][i]
                + pos_in_box[2] * self.recip_h[2][i];
            t = (t - t.floor()) * gs[i] as F;
            let ti = t as usize;
            dr[i] = t - ti as F;
            grid_index[i] = ti % gs[i];
        }

        bspline_fill(data, &dr, order);
        grid_index
    }

    /// Like [`compute_spline`](Self::compute_spline), also filling `ddata`
    /// with the derivatives.
    fn compute_spline_with_deriv(
        &self,
        coords: &[F],
        atom: usize,
        data: &mut [[F; 3]],
        ddata: &mut [[F; 3]],
    ) -> [usize; 3] {
        let order = self.params.order;
        let gs = self.params.grid_size;
        let pos = [coords[atom * 3], coords[atom * 3 + 1], coords[atom * 3 + 2]];

        let mut pos_in_box = pos;
        for i in (0..3).rev() {
            let s = (pos_in_box[i] * self.recip_h[i][i]).floor();
            for (j, pos_j) in pos_in_box.iter_mut().enumerate() {
                *pos_j -= s * self.h[i][j];
            }
        }

        let mut grid_index = [0usize; 3];
        let mut dr = [0.0 as F; 3];
        for i in 0..3 {
            let mut t = pos_in_box[0] * self.recip_h[0][i]
                + pos_in_box[1] * self.recip_h[1][i]
                + pos_in_box[2] * self.recip_h[2][i];
            t = (t - t.floor()) * gs[i] as F;
            let ti = t as usize;
            dr[i] = t - ti as F;
            grid_index[i] = ti % gs[i];
        }

        bspline_fill_with_deriv(data, ddata, &dr, order);
        grid_index
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimum-image displacement vector from atom i to atom j.
    fn min_image_delta(&self, coords: &[F], i: usize, j: usize) -> (F, F, F) {
        let d = self.mic.apply([
            coords[j * 3] - coords[i * 3],
            coords[j * 3 + 1] - coords[i * 3 + 1],
            coords[j * 3 + 2] - coords[i * 3 + 2],
        ]);
        (d[0], d[1], d[2])
    }

    /// Raw displacement from atom i to atom j (no minimum image).
    fn delta(&self, coords: &[F], i: usize, j: usize) -> (F, F, F) {
        (
            coords[j * 3] - coords[i * 3],
            coords[j * 3 + 1] - coords[i * 3 + 1],
            coords[j * 3 + 2] - coords[i * 3 + 2],
        )
    }

    fn is_excluded(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.exclusions.iter().any(|&[a, b]| a == lo && b == hi)
    }
}

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Energy and forces (`-∇E`) over a flat coordinate vector. The Ewald sum
/// is not a sum over a pair table's separations, so it tallies no virial.
impl Potential for PairCoulLongPme {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let direct = if self.direct {
            self.direct_energy(coords)
        } else {
            0.0
        };
        acc.add_energy(
            self.self_energy
                + direct
                + self.exclusion_energy(coords)
                + self.reciprocal_energy(coords),
        );
        let mut grad = vec![0.0; coords.len()];
        // Self energy has zero gradient (constant).
        if self.direct {
            self.direct_gradient(coords, &mut grad);
        }
        self.exclusion_gradient(coords, &mut grad);
        self.reciprocal_gradient(coords, &mut grad);
        for (f, g) in acc.forces_without_virial().iter_mut().zip(&grad) {
            *f -= g;
        }
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Invert a lower-triangular 3×3 box matrix.
fn invert_box_vectors(h: &[[F; 3]; 3]) -> [[F; 3]; 3] {
    let det = h[0][0] * h[1][1] * h[2][2];
    let s = 1.0 / det;
    [
        [h[1][1] * h[2][2] * s, 0.0, 0.0],
        [-h[1][0] * h[2][2] * s, h[0][0] * h[2][2] * s, 0.0],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * s,
            -h[0][0] * h[2][1] * s,
            h[0][0] * h[1][1] * s,
        ],
    ]
}

/// Fill B-spline coefficients (no derivatives). Matches the C++ `computeSpline`.
fn bspline_fill(data: &mut [[F; 3]], dr: &[F; 3], order: usize) {
    let scale = 1.0 / (order - 1) as F;
    for i in 0..3 {
        data[order - 1][i] = 0.0;
        data[1][i] = dr[i];
        data[0][i] = 1.0 - dr[i];
        for j in 3..order {
            let div = 1.0 / (j - 1) as F;
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for k in 1..(j - 1) {
                data[j - k - 1][i] = div
                    * ((dr[i] + k as F) * data[j - k - 2][i]
                        + (j as F - k as F - dr[i]) * data[j - k - 1][i]);
            }
            data[0][i] *= div * (1.0 - dr[i]);
        }
        // Final scaling pass
        data[order - 1][i] = scale * dr[i] * data[order - 2][i];
        for j in 1..(order - 1) {
            data[order - j - 1][i] = scale
                * ((dr[i] + j as F) * data[order - j - 2][i]
                    + (order as F - j as F - dr[i]) * data[order - j - 1][i]);
        }
        data[0][i] *= scale * (1.0 - dr[i]);
    }
}

/// Fill B-spline coefficients AND derivatives. Derivatives are computed just
/// before the final scaling pass (matching the C++ reference).
fn bspline_fill_with_deriv(data: &mut [[F; 3]], ddata: &mut [[F; 3]], dr: &[F; 3], order: usize) {
    let scale = 1.0 / (order - 1) as F;
    for i in 0..3 {
        data[order - 1][i] = 0.0;
        data[1][i] = dr[i];
        data[0][i] = 1.0 - dr[i];
        for j in 3..order {
            let div = 1.0 / (j - 1) as F;
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for k in 1..(j - 1) {
                data[j - k - 1][i] = div
                    * ((dr[i] + k as F) * data[j - k - 2][i]
                        + (j as F - k as F - dr[i]) * data[j - k - 1][i]);
            }
            data[0][i] *= div * (1.0 - dr[i]);
        }
        // Derivatives (before final scaling)
        ddata[0][i] = -data[0][i];
        for j in 1..order {
            ddata[j][i] = data[j - 1][i] - data[j][i];
        }
        // Final scaling pass
        data[order - 1][i] = scale * dr[i] * data[order - 2][i];
        for j in 1..(order - 1) {
            data[order - j - 1][i] = scale
                * ((dr[i] + j as F) * data[order - j - 2][i]
                    + (order as F - j as F - dr[i]) * data[order - j - 1][i]);
        }
        data[0][i] *= scale * (1.0 - dr[i]);
    }
}

/// Precompute B-spline moduli for reciprocal-space convolution.
fn compute_bspline_moduli(grid_size: usize, order: usize) -> Vec<F> {
    // Compute the B-spline values at uniform intervals
    let mut bspline = vec![0.0 as F; grid_size];
    let mut data = vec![[0.0 as F; 3]; order];
    let dr = [0.0 as F; 3];
    bspline_fill(&mut data, &dr, order);
    // data[j][0] gives the B-spline value at u=0 for the j-th support point
    for j in 0..order {
        bspline[j] = data[j][0];
    }

    // Compute moduli via DFT of bspline values
    let two_pi_over_n = 2.0 * PI / grid_size as F;
    let mut moduli = vec![0.0 as F; grid_size];
    for (k, moduli_k) in moduli.iter_mut().enumerate().take(grid_size) {
        let mut sum_cos: F = 0.0;
        let mut sum_sin: F = 0.0;
        for (j, bspline_j) in bspline.iter().enumerate().take(order) {
            let arg = two_pi_over_n * k as F * j as F;
            sum_cos += *bspline_j * arg.cos();
            sum_sin += *bspline_j * arg.sin();
        }
        *moduli_k = sum_cos * sum_cos + sum_sin * sum_sin;
    }
    // Avoid division by zero at k=0
    if moduli[0] < 1e-30 {
        moduli[0] = 1e-30;
    }
    moduli
}

// ---------------------------------------------------------------------------
// BuiltinKernels constructor
// ---------------------------------------------------------------------------

/// Real-space erfc ([`CoulLong`]) beside the reciprocal sum. `long` has its
/// direct-space loop off, so the erfc term is priced once.
struct PmeTerms {
    real: ScalarPair<CoulLong>,
    long: PairCoulLongPme,
}

impl Potential for PmeTerms {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        self.real.evaluate(coords, acc);
        self.long.evaluate(coords, acc);
    }
}

impl PairDriven for PmeTerms {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        // The pair weight scales the real-space term. The reciprocal sum
        // includes every pair and takes no weight: a 1-4 scale is not a
        // `(1 − s) erf` correction. Binary exclusions remain that list.
        self.real.evaluate_pairs(coords, pairs, factor, acc);
        self.long.evaluate(coords, acc);
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        self.real.binds_a_fixed_pair_list()
    }
}

/// Constructor for the kernel registry.
///
/// **`style_params`** keys — exactly the spec's: `alpha`, `cutoff`,
/// `grid_x`, `grid_y`, `grid_z`, `order`, `coulomb`, `dielectric`.
///
/// **`frame`**:
/// - its box ([`Frame::simbox`]): periodic in all three directions, the cell
///   in LAMMPS's restricted triclinic form (`a` along x, `b` in the xy
///   plane). Anything else is [`CompileError::NoBox`].
/// - `"atoms"` with `"charge"` column (f64) — per-atom charges.
/// - `"exclusions"` with `"atomi"`, `"atomj"` columns (u64) — the erf
///   correction list.
/// - `"pairs"`, when present: the real-space erfc list. Rows that are also
///   exclusions are omitted. The distances are raw coordinate differences,
///   as for every other compiled pair style, so a periodic frame whose list
///   is only intramolecular does not price intermolecular pairs on this
///   door — `pair_coul_long_pme_typed` does, over the neighbour table.
///   With no `pairs` block the real-space term stays the minimum-image loop
///   in this kernel, so the energy is not dropped.
///
/// [`CompileError::NoBox`]: crate::ff::potential::CompileError::NoBox
pub fn pair_coul_long_pme_constructor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, CompileError> {
    // A present `pairs` block, even an empty one, replaces the minimum-image
    // loop. An absent block keeps it.
    let prices_here = frame.get(PAIRS).is_none();
    let long = long_range(style_params, frame, prices_here)?;
    if prices_here {
        return Ok(ForceTerm::plain(long));
    }
    let real_frame = real_space_frame(frame)?;
    let nrows = real_frame.get(PAIRS).and_then(|b| b.n_rows()).unwrap_or(0);
    if nrows == 0 {
        return Ok(ForceTerm::plain(long));
    }
    let spec = pme_spec();
    let real = ScalarPair::compiled(
        Arc::new(CoulLong),
        &spec,
        style_params,
        type_params,
        &real_frame,
    )?
    .fused();
    Ok(ForceTerm::plain(PmeTerms { real, long }))
}

/// Neighbour-driven door: the real-space term is [`CoulLong`] over the
/// neighbour table, and this kernel prices the reciprocal sum, the self
/// energy and the exclusion correction. The minimum-image loop stays off.
/// The box, the charges and the Ewald parameters are still required.
pub(crate) fn pair_coul_long_pme_typed(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, CompileError> {
    let long = long_range(style_params, frame, false)?;
    let spec = pme_spec();
    let real =
        ScalarPair::typed(Arc::new(CoulLong), &spec, style_params, type_params, frame)?.fused();
    Ok(ForceTerm::pair(PmeTerms { real, long }))
}

fn pme_spec() -> crate::ff::ir::StyleSpec {
    crate::ff::ir::builtin_styles()
        .into_iter()
        .find(|spec| spec.category == "pair" && spec.name == "coul/long/pme")
        .expect("pair coul/long/pme is a built-in style")
}

fn long_range(
    style_params: &Params,
    frame: &Frame,
    direct: bool,
) -> Result<PairCoulLongPme, CompileError> {
    let get = |key: &str| param_reads::style_num("coul/long/pme", style_params, key);
    let count = |key: &str| -> Result<usize, crate::ff::ir::IrError> {
        let v = get(key)?;
        if v < 1.0 || v.fract() != 0.0 {
            return Err(param_reads::bad(
                "coul/long/pme",
                "",
                key,
                format!("= {v} is not a positive integer"),
            ));
        }
        Ok(v as usize)
    };
    let alpha = get("alpha")?;
    let cutoff = get("cutoff")?;
    let grid_x = count("grid_x")?;
    let grid_y = count("grid_y")?;
    let grid_z = count("grid_z")?;
    let order = count("order")?;
    // The prefactor every PME term is priced in: `coulomb / dielectric`,
    // LAMMPS's `qqrd2e`. The pair form divides the same two parameters.
    let coulomb = get("coulomb")? / get("dielectric")?;

    let atoms = frame
        .get(ATOMS)
        .ok_or("PME: Frame missing \"atoms\" block")?;
    let charges: Vec<F> = if let Some(charge_col) = atoms.get("charge").and_then(|c| c.get::<f64>())
    {
        charge_col.iter().copied().collect()
    } else {
        return Err("PME: atoms block missing \"charge\" float column".into());
    };

    let box_vectors = box_vectors(frame)?;

    let mut exclusions = Vec::new();
    if let Some(block) = frame.get(EXCLUSIONS)
        && let (Some(i_col), Some(j_col)) = (
            block.get("atomi").and_then(|c| c.get::<u64>()),
            block.get("atomj").and_then(|c| c.get::<u64>()),
        )
    {
        for (&a, &b) in i_col.iter().zip(j_col.iter()) {
            let (lo, hi) = ordered(a, b);
            exclusions.push([lo as usize, hi as usize]);
        }
    }

    let params = PairCoulLongPmeParams {
        alpha,
        cutoff,
        grid_size: [grid_x, grid_y, grid_z],
        order,
        coulomb,
    };
    Ok(PairCoulLongPme::build(
        params,
        charges,
        box_vectors,
        exclusions,
        direct,
    ))
}

/// `frame` with exclusion rows dropped from `pairs`. Those pairs are the
/// erf correction and must not also be priced as erfc. Cloned only when a
/// row is dropped; a block whose endpoints cannot be read is passed through
/// so the pair form reports that.
fn real_space_frame(frame: &Frame) -> Result<Cow<'_, Frame>, CompileError> {
    let Some(pairs) = frame.get(PAIRS) else {
        return Ok(Cow::Borrowed(frame));
    };
    let (Some(i_col), Some(j_col)) = (
        pairs.get("atomi").and_then(|c| c.get::<u64>()),
        pairs.get("atomj").and_then(|c| c.get::<u64>()),
    ) else {
        return Ok(Cow::Borrowed(frame));
    };
    let excluded = exclusion_keys(frame);
    if excluded.is_empty() {
        return Ok(Cow::Borrowed(frame));
    }
    let mut keep = Vec::new();
    for (idx, (&a, &b)) in i_col.iter().zip(j_col.iter()).enumerate() {
        if !excluded.contains(&ordered(a, b)) {
            keep.push(idx);
        }
    }
    if keep.len() == i_col.len() {
        return Ok(Cow::Borrowed(frame));
    }
    let filtered = pairs
        .select_rows(&keep)
        .map_err(|err| CompileError::from(err.to_string()))?;
    let mut owned = frame.clone();
    owned.insert(PAIRS, filtered);
    Ok(Cow::Owned(owned))
}

fn exclusion_keys(frame: &Frame) -> HashSet<(u64, u64)> {
    let mut keys = HashSet::new();
    if let Some(block) = frame.get(EXCLUSIONS)
        && let (Some(i_col), Some(j_col)) = (
            block.get("atomi").and_then(|c| c.get::<u64>()),
            block.get("atomj").and_then(|c| c.get::<u64>()),
        )
    {
        for (&a, &b) in i_col.iter().zip(j_col.iter()) {
            keys.insert(ordered(a, b));
        }
    }
    keys
}

fn ordered(a: u64, b: u64) -> (u64, u64) {
    if a < b { (a, b) } else { (b, a) }
}

/// The frame's periodic cell as [`PairCoulLongPme::new`] takes it: the lattice
/// vectors as **rows** (lower-triangular), the transpose of
/// [`SimBox::matrix`](molrs::core::SimBox::matrix), whose columns
/// they are.
///
/// # Errors
/// [`CompileError::NoBox`](crate::ff::potential::CompileError::NoBox): the
/// frame has no box, a box not periodic in x, y and z (LAMMPS: "Cannot use
/// nonperiodic boundaries with PPPM"), an undefined cell, or a cell outside
/// LAMMPS's restricted triclinic form (`a` along x, `b` in the xy plane,
/// positive lengths) — a general cell's coordinates rotate with it, which
/// is the caller's to do.
fn box_vectors(frame: &Frame) -> Result<[[F; 3]; 3], crate::ff::potential::CompileError> {
    let refuse = |reason: &str| crate::ff::potential::CompileError::NoBox {
        category: "pair".into(),
        style: "coul/long/pme".into(),
        reason: reason.into(),
    };
    let bx = frame
        .simbox
        .as_ref()
        .ok_or_else(|| refuse("the frame has no box"))?;
    if !bx.is_cell_defined() {
        return Err(refuse("the frame's box has no cell"));
    }
    if bx.pbc() != [true; 3] {
        return Err(refuse(&format!(
            "the box is periodic in {:?} (x, y, z); Ewald sums need all three",
            bx.pbc()
        )));
    }
    let h = bx.matrix();
    // Columns are the lattice vectors: a = (h00, h10, h20), b = (h01, h11, h21).
    if h[1][0] != 0.0
        || h[2][0] != 0.0
        || h[2][1] != 0.0
        || (0..3).any(|i| h[i][i].is_nan() || h[i][i] <= 0.0)
    {
        return Err(refuse(&format!(
            "the cell {h:?} is not in LAMMPS's restricted triclinic form \
             (a along x, b in the xy plane, positive lengths)"
        )));
    }
    Ok(std::array::from_fn(|r| std::array::from_fn(|c| h[c][r])))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::SpecialBonds;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::pair::fixtures::table_over;
    use molrs::core::schema::block_names::{EXCLUSIONS, PAIRS};
    use molrs::core::{Block, SimBox};
    use ndarray::Array1;

    fn cubic_box(l: F) -> [[F; 3]; 3] {
        [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
    }

    // --- B-spline unit tests ---

    #[test]
    fn test_bspline_partition_of_unity() {
        // For any fractional offset, the B-spline values must sum to 1.
        let order = 4;
        for &u in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.99] {
            let u: F = u as F;
            let dr = [u, u, u];
            let mut data = vec![[0.0 as F; 3]; order];
            bspline_fill(&mut data, &dr, order);
            let sum: F = data.iter().map(|d| d[0]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "partition of unity failed for u={}: sum={}",
                u,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_deriv_sum_zero() {
        let order = 4;
        for &u in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let u: F = u as F;
            let dr = [u, u, u];
            let mut data = vec![[0.0 as F; 3]; order];
            let mut ddata = vec![[0.0 as F; 3]; order];
            bspline_fill_with_deriv(&mut data, &mut ddata, &dr, order);
            let sum: F = ddata.iter().map(|d| d[0]).sum();
            assert!(
                sum.abs() < 1e-5,
                "derivative sum should be zero for u={}: sum={}",
                u,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_order5() {
        let order = 5;
        let dr: [F; 3] = [0.3, 0.7, 0.5];
        let mut data = vec![[0.0 as F; 3]; order];
        bspline_fill(&mut data, &dr, order);
        for dim in 0..3 {
            let sum: F = data.iter().map(|d| d[dim]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "dim={}: sum={}", dim, sum);
        }
    }

    // --- FFT round-trip test ---

    #[test]
    fn test_fft_roundtrip() {
        let params = PairCoulLongPmeParams {
            alpha: 0.3,
            cutoff: 5.0,
            grid_size: [4, 4, 4],
            order: 4,
            coulomb: 1.0,
        };
        let pme = PairCoulLongPme::new(params, vec![1.0], cubic_box(10.0), vec![]);
        let n = 4 * 4 * 4;
        let mut grid: Vec<Complex<F>> = (0..n).map(|i| Complex::new(i as F, 0.0)).collect();
        let original = grid.clone();
        let max_dim = 4;
        let zero = Complex::new(0.0 as F, 0.0);
        let mut buf = vec![zero; max_dim];
        let max_scratch = pme
            .fft_plans
            .fwd
            .iter()
            .chain(pme.fft_plans.inv.iter())
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let mut fft_scratch = vec![zero; max_scratch];

        pme.fft_3d_forward(&mut grid, &mut buf, &mut fft_scratch);
        pme.fft_3d_inverse(&mut grid, &mut buf, &mut fft_scratch);

        // Normalize
        let inv_n: F = 1.0 / n as F;
        for c in grid.iter_mut() {
            *c *= inv_n;
        }

        for i in 0..n {
            assert!(
                (grid[i].re - original[i].re).abs() < 1e-3,
                "FFT round-trip failed at {}: got {}, expected {}",
                i,
                grid[i].re,
                original[i].re,
            );
            assert!(
                grid[i].im.abs() < 1e-3,
                "FFT round-trip imaginary part at {}: {}",
                i,
                grid[i].im,
            );
        }
    }

    // --- Two-ion test ---

    #[test]
    fn test_two_ions_energy() {
        let box_l: F = 20.0;
        let r: F = 3.0;
        let alpha: F = 0.3;
        let coulomb: F = 1.0;
        let params = PairCoulLongPmeParams {
            alpha,
            cutoff: 9.0,
            grid_size: [32, 32, 32],
            order: 5,
            coulomb,
        };
        let charges = vec![1.0, -1.0];
        let exclusions = vec![];
        let pme = PairCoulLongPme::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0 + r,
            box_l / 2.0,
            box_l / 2.0,
        ];
        let e = pme.energy_at(&coords);

        // Coulomb energy in vacuum: -1/r (with coulomb=1, q=+1,-1)
        let e_vacuum: F = -coulomb / r;
        // With periodic images the energy differs slightly, but should be close
        assert!(
            (e - e_vacuum).abs() < 0.05,
            "PME energy={}, vacuum={}, diff={}",
            e,
            e_vacuum,
            (e - e_vacuum).abs()
        );
    }

    // --- Numerical forces test ---

    #[test]
    fn test_numerical_forces() {
        let box_l: F = 10.0;
        let params = PairCoulLongPmeParams {
            alpha: 0.4,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.3, 0.2];
        let exclusions = vec![[0, 1]]; // exclude the 0-1 pair
        let pme = PairCoulLongPme::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![2.0, 3.0, 4.0, 5.0, 3.5, 4.5, 7.0, 6.0, 5.0];

        let forces = pme.energy_forces_at(&coords).1;

        let eps: F = 1e-3;
        for idx in 0..9 {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[idx] += eps;
            cm[idx] -= eps;
            let numerical_force = -(pme.energy_at(&cp) - pme.energy_at(&cm)) / (2.0 * eps);
            assert!(
                (forces[idx] - numerical_force).abs() < 1.0,
                "idx={}: analytical={:.6}, numerical={:.6}, diff={:.2e}",
                idx,
                forces[idx],
                numerical_force,
                (forces[idx] - numerical_force).abs()
            );
        }
    }

    // --- Newton's third law ---

    #[test]
    fn test_newton_third_law() {
        let box_l: F = 10.0;
        let params = PairCoulLongPmeParams {
            alpha: 0.35,
            cutoff: 4.5,
            grid_size: [32, 32, 32],
            order: 5,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.3, 0.4, -0.6];
        let exclusions = vec![[0, 1], [2, 3]];
        let pme = PairCoulLongPme::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 6.0, 7.0, 2.0, 8.0, 7.5, 2.5];
        let forces = pme.energy_forces_at(&coords).1;

        for dim in 0..3 {
            let sum: F = (0..4).map(|a| forces[a * 3 + dim]).sum();
            assert!(sum.abs() < 0.1, "dim={}: total force sum={:.2e}", dim, sum);
        }
    }

    // --- Integration test: PME + a compiled lj/cut in Potentials ---

    #[test]
    fn test_pme_in_potentials_collection() {
        use crate::ff::compile::ExplicitTerms;
        use crate::ff::ir::Params;
        use crate::ff::potential::Potentials;

        let box_l: F = 10.0;
        let params = PairCoulLongPmeParams {
            alpha: 0.3,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.5];
        let pme = PairCoulLongPme::new(params, charges, cubic_box(box_l), vec![]);

        let lj = ExplicitTerms::new("pair", "lj/cut")
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", 1.0), ("sigma", 1.0)]),
            )
            .compile()
            .unwrap();

        let mut pots = Potentials::new();
        pots.push(ForceTerm::plain(pme));
        pots.push(ForceTerm::pair(lj));

        let coords: Vec<F> = vec![
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0 + 2.0,
            box_l / 2.0,
            box_l / 2.0,
        ];

        let e = pots.energy_at(&coords);
        assert!(e.is_finite(), "energy should be finite, got {}", e);
        assert!(e.abs() > 1e-10, "energy should be non-zero");

        let (_, forces) = pots.energy_forces_at(&coords);
        for (i, &f) in forces.iter().enumerate() {
            assert!(F::is_finite(f), "forces[{}] should be finite", i);
        }
    }

    // --- The box is the frame's ---

    /// `pair coul/long/pme` compiles against the frame's own periodic box,
    /// as LAMMPS's kspace reads its simulation box, and prices what the
    /// kernel built on that cell prices — a restricted triclinic cell too;
    /// a frame without a usable box is refused by name.
    #[test]
    fn the_cell_is_the_frames_box_and_none_is_refused_by_name() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::forcefield::ForceField;
        use crate::ff::potential::CompileError;
        use molrs::core::Block;
        use molrs::core::SimBox;
        use ndarray::Array1;

        let coords: Vec<F> = vec![4.0, 5.0, 5.0, 6.5, 5.3, 4.6, 5.1, 3.8, 6.2];
        let charges = vec![0.6, -0.4, -0.2];
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].iter().enumerate() {
            let col: Vec<F> = coords.iter().skip(d).step_by(3).copied().collect();
            atoms
                .insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert("charge", Array1::from_vec(charges.clone()).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        let mut ff = ForceField::new("pme");
        ff.set_special_bonds(crate::ff::ir::SpecialBonds::EXCLUDE_12_13);
        ff.def_style(
            "pair",
            "coul/long/pme",
            Params::from_iter([
                ("coulomb", 332.06371),
                ("dielectric", 1.0),
                ("cutoff", 4.5),
                ("alpha", 0.35),
                ("order", 4.0),
                ("grid_x", 16.0),
                ("grid_y", 16.0),
                ("grid_z", 16.0),
            ]),
        )
        .unwrap();
        let params = PairCoulLongPmeParams {
            alpha: 0.35,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 332.06371,
        };
        let compile = |frame: &Frame| PotentialCompiler::new(&ff).compile(frame);

        // The columns of `SimBox::matrix` are the lattice vectors; the
        // kernel takes them as rows.
        let tilted = [[10.0, 1.5, -0.5], [0.0, 9.0, 2.0], [0.0, 0.0, 11.0]];
        let rows = [[10.0, 0.0, 0.0], [1.5, 9.0, 0.0], [-0.5, 2.0, 11.0]];
        for (h, h_rows) in [(cubic_box(10.0), cubic_box(10.0)), (tilted, rows)] {
            let mut boxed = frame.clone();
            boxed.simbox = Some(SimBox::new(Cell::Matrix(h), [0.0; 3], [true; 3]).unwrap());
            let want = PairCoulLongPme::new(params.clone(), charges.clone(), h_rows, vec![])
                .energy_at(&coords);
            let got = compile(&boxed).unwrap().energy_at(&coords);
            assert_eq!(got.to_bits(), want.to_bits(), "{h:?}: {got} vs {want}");
        }

        let refused = |frame: &Frame| match compile(frame) {
            Err(CompileError::NoBox {
                category, style, ..
            }) => assert_eq!(
                (category.as_str(), style.as_str()),
                ("pair", "coul/long/pme")
            ),
            other => panic!("NoBox expected, got {other:?}", other = other.map(|_| ())),
        };
        refused(&frame);
        let mut slab = frame.clone();
        slab.simbox = Some(
            SimBox::new(
                Cell::Lengths([10.0; 3]),
                [0.0, 0.0, 0.0],
                [true, true, false],
            )
            .unwrap(),
        );
        refused(&slab);
        let mut general = frame.clone();
        general.simbox = Some(
            SimBox::new(
                Cell::Matrix([[10.0, 0.0, 0.0], [1.0, 9.0, 0.0], [0.0, 0.0, 11.0]]),
                [0.0; 3],
                [true; 3],
            )
            .unwrap(),
        );
        refused(&general);
    }

    const IONS: [F; 9] = [4.0, 5.0, 5.0, 6.5, 5.3, 4.6, 5.1, 3.8, 6.2];
    const ION_CHARGES: [F; 3] = [0.6, -0.4, -0.2];

    fn style_params() -> Params {
        Params::from_iter([
            ("coulomb", 332.06371),
            ("dielectric", 1.0),
            ("cutoff", 4.5),
            ("alpha", 0.35),
            ("order", 4.0),
            ("grid_x", 16.0),
            ("grid_y", 16.0),
            ("grid_z", 16.0),
        ])
    }

    fn field(special: SpecialBonds) -> ForceField {
        let mut ff = ForceField::new("pme");
        ff.set_special_bonds(special);
        ff.def_style("pair", "coul/long/pme", style_params())
            .unwrap();
        ff
    }

    fn cubic(box_l: F) -> SimBox {
        SimBox::new(
            Cell::Matrix([[box_l, 0.0, 0.0], [0.0, box_l, 0.0], [0.0, 0.0, box_l]]),
            [0.0; 3],
            [true; 3],
        )
        .unwrap()
    }

    fn ions(box_l: F) -> Frame {
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].iter().enumerate() {
            let col: Vec<F> = IONS.iter().skip(d).step_by(3).copied().collect();
            atoms
                .insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "charge",
                Array1::from_vec(ION_CHARGES.to_vec()).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        frame.simbox = Some(cubic(box_l));
        frame
    }

    fn put_pairs(frame: &mut Frame, rows: &[(u64, u64)], is_14: &[bool]) {
        let mut pairs = Block::new();
        pairs
            .insert(
                "atomi",
                Array1::from_vec(rows.iter().map(|r| r.0).collect()).into_dyn(),
                None,
            )
            .unwrap();
        pairs
            .insert(
                "atomj",
                Array1::from_vec(rows.iter().map(|r| r.1).collect()).into_dyn(),
                None,
            )
            .unwrap();
        if !is_14.is_empty() {
            pairs
                .insert("is_14", Array1::from_vec(is_14.to_vec()).into_dyn(), None)
                .unwrap();
        }
        frame.insert(PAIRS, pairs);
    }

    fn put_exclusions(frame: &mut Frame, rows: &[(u64, u64)]) {
        let mut block = Block::new();
        block
            .insert(
                "atomi",
                Array1::from_vec(rows.iter().map(|r| r.0).collect()).into_dyn(),
                None,
            )
            .unwrap();
        block
            .insert(
                "atomj",
                Array1::from_vec(rows.iter().map(|r| r.1).collect()).into_dyn(),
                None,
            )
            .unwrap();
        frame.insert(EXCLUSIONS, block);
    }

    fn separation(coords: &[F], i: usize, j: usize) -> F {
        (0..3)
            .map(|c| coords[3 * j + c] - coords[3 * i + c])
            .map(|d| d * d)
            .sum::<F>()
            .sqrt()
    }

    fn kernel_params() -> PairCoulLongPmeParams {
        PairCoulLongPmeParams {
            alpha: 0.35,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 332.06371,
        }
    }

    /// A `pairs` block that lists every non-excluded pair inside the cutoff
    /// prices the same energy as the minimum-image loop. The two must not
    /// both run: that would count the erfc term twice.
    #[test]
    fn a_pairs_block_prices_the_real_space_term_once() {
        let links = [(0, 1), (0, 2), (1, 2)];
        let mut frame = ions(10.0);
        put_pairs(&mut frame, &links, &[]);
        let compiled = PotentialCompiler::new(&field(SpecialBonds::EXCLUDE_12_13))
            .compile(&frame)
            .unwrap();
        let full = PairCoulLongPme::new(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        );
        let (got_e, got_f) = compiled.energy_forces_at(&IONS);
        let (want_e, want_f) = full.energy_forces_at(&IONS);
        assert!((got_e - want_e).abs() < 1e-8, "energy {got_e} vs {want_e}");
        for (i, (g, w)) in got_f.iter().zip(&want_f).enumerate() {
            assert!((g - w).abs() < 1e-6, "force {i}: {g} vs {w}");
        }
    }

    /// An empty `pairs` block is a list, so the real-space loop stays off.
    /// The reciprocal sum, the self energy and the exclusion correction remain.
    #[test]
    fn an_empty_pairs_block_prices_reciprocal_space_only() {
        let mut frame = ions(10.0);
        frame.insert(PAIRS, Block::new());
        let got = PotentialCompiler::new(&field(SpecialBonds::EXCLUDE_12_13))
            .compile(&frame)
            .unwrap()
            .energy_at(&IONS);
        let recip = PairCoulLongPme::reciprocal(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        )
        .energy_at(&IONS);
        let full = PairCoulLongPme::new(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        )
        .energy_at(&IONS);
        assert_eq!(got.to_bits(), recip.to_bits(), "{got} vs {recip}");
        assert!(
            (full - got).abs() > 1.0,
            "real space must be the difference"
        );
    }

    /// A 1-4 row takes `coulomb14scale` on the erfc term. The reciprocal sum
    /// is unchanged, so the total is not `s/r`.
    #[test]
    fn a_1_4_row_scales_the_real_space_term_only() {
        let coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let charges = [1.0, -1.0];
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].iter().enumerate() {
            let col: Vec<F> = coords.iter().skip(d).step_by(3).copied().collect();
            atoms
                .insert(*key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        atoms
            .insert(
                "charge",
                Array1::from_vec(charges.to_vec()).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        frame.simbox = Some(cubic(20.0));
        put_pairs(&mut frame, &[(0, 1)], &[true]);
        let compile = |coul_14: F| {
            let special = SpecialBonds {
                lj: [0.0, 0.0, 1.0],
                coul: [0.0, 0.0, coul_14],
            };
            PotentialCompiler::new(&field(special))
                .compile(&frame)
                .unwrap()
                .energy_forces_at(&coords)
        };
        let (full_e, full_f) = compile(1.0);
        let (half_e, half_f) = compile(0.5);
        let (real_e, real_de) = real_space_pair(332.06371, 0.35, 1.0, -1.0, 2.0);
        assert!(
            (full_e - half_e - 0.5 * real_e).abs() < 1e-8,
            "delta {} vs half erfc {}",
            full_e - half_e,
            0.5 * real_e
        );
        // Δx / r = 1, so the unscaled real-space force on the +x atom is
        // −dE/dr. The two compiles share the reciprocal gradient.
        assert!(
            (full_f[3] - half_f[3] - (-0.5 * real_de)).abs() < 1e-6,
            "force delta {} vs {}",
            full_f[3] - half_f[3],
            -0.5 * real_de
        );
    }

    /// A pair listed in both `pairs` and `exclusions` is not priced as erfc.
    /// The exclusion still subtracts the erf correction.
    #[test]
    fn an_excluded_pair_on_the_list_is_not_priced_as_erfc() {
        let links = [(0_u64, 1_u64), (0, 2), (1, 2)];
        let mut frame = ions(10.0);
        put_pairs(&mut frame, &links, &[]);
        // Stored reversed: the filter orders the endpoints.
        put_exclusions(&mut frame, &[(1, 0)]);
        let got = PotentialCompiler::new(&field(SpecialBonds::EXCLUDE_12_13))
            .compile(&frame)
            .unwrap()
            .energy_at(&IONS);
        let with = PairCoulLongPme::new(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![[0, 1]],
        )
        .energy_at(&IONS);
        let without = PairCoulLongPme::new(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        )
        .energy_at(&IONS);
        assert!((got - with).abs() < 1e-8, "{got} vs {with}");
        assert!(
            (got - without).abs() > 1.0,
            "dropping the erfc must move the energy"
        );
    }

    /// The neighbour door prices erfc over the table it is handed and adds
    /// the reciprocal sum. Without a table, `evaluate` is the reciprocal
    /// part. A frame without a box is refused by name.
    #[test]
    fn compile_typed_prices_erfc_over_the_neighbour_table() {
        let frame = ions(10.0);
        let compiled = PotentialCompiler::new(&field(SpecialBonds::EXCLUDE_12_13))
            .compile_typed(&frame)
            .unwrap();
        assert_eq!(compiled.len(), 1);
        let (term, weights) = &compiled[0];
        let ForceTerm::Pair(kernel) = term else {
            panic!("neighbour-driven member, got {term:?}");
        };
        assert!(!kernel.binds_a_fixed_pair_list());
        let weights = weights.as_ref().expect("Coulomb pair weights");
        assert_eq!(
            weights.by_distance(),
            &SpecialBonds::EXCLUDE_12_13.coul_weights()
        );

        let recip = PairCoulLongPme::reciprocal(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        )
        .energy_at(&IONS);
        assert_eq!(kernel.energy_at(&IONS).to_bits(), recip.to_bits());

        let all = table_over(&IONS, &[(0, 1), (0, 2), (1, 2)]);
        let (got_e, got_f) = kernel.over_pairs(&IONS, &all);
        let (want_e, want_f) = PairCoulLongPme::new(
            kernel_params(),
            ION_CHARGES.to_vec(),
            cubic_box(10.0),
            vec![],
        )
        .energy_forces_at(&IONS);
        assert!((got_e - want_e).abs() < 1e-8, "{got_e} vs {want_e}");
        for (i, (g, w)) in got_f.iter().zip(&want_f).enumerate() {
            assert!((g - w).abs() < 1e-6, "force {i}: {g} vs {w}");
        }

        let one = table_over(&IONS, &[(0, 1)]);
        let (e_one, _) = kernel.over_pairs(&IONS, &one);
        let r = separation(&IONS, 0, 1);
        let (erfc, _) = real_space_pair(332.06371, 0.35, ION_CHARGES[0], ION_CHARGES[1], r);
        assert!(
            (e_one - recip - erfc).abs() < 1e-8,
            "{e_one} vs recip {recip} + erfc {erfc}"
        );

        let mut unboxed = frame.clone();
        unboxed.simbox = None;
        match PotentialCompiler::new(&field(SpecialBonds::EXCLUDE_12_13)).compile_typed(&unboxed) {
            Err(CompileError::NoBox {
                category, style, ..
            }) => {
                assert_eq!(
                    (category.as_str(), style.as_str()),
                    ("pair", "coul/long/pme")
                );
            }
            other => panic!("NoBox expected, got {other:?}"),
        }
    }

    // --- Box inversion test ---

    #[test]
    fn test_invert_box_vectors() {
        let h: [[F; 3]; 3] = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let inv = invert_box_vectors(&h);
        assert!((inv[0][0] - 0.1).abs() < 1e-5);
        assert!((inv[1][1] - 0.1).abs() < 1e-5);
        assert!((inv[2][2] - 0.1).abs() < 1e-5);
        assert!(inv[0][1].abs() < 1e-5);
        assert!(inv[1][0].abs() < 1e-5);
    }

    #[test]
    fn test_invert_box_vectors_triclinic() {
        let h: [[F; 3]; 3] = [[10.0, 0.0, 0.0], [2.0, 8.0, 0.0], [1.0, 3.0, 6.0]];
        let inv = invert_box_vectors(&h);
        // Verify H * H^{-1} = I (row-by-row dot product)
        for (row, h_row) in h.iter().enumerate() {
            for (col, _) in inv[0].iter().enumerate() {
                let mut dot: F = 0.0;
                for (k, h_row_k) in h_row.iter().enumerate() {
                    dot += h_row_k * inv[k][col];
                }
                let expected: F = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "H*Hinv[{}][{}]={}, expected {}",
                    row,
                    col,
                    dot,
                    expected
                );
            }
        }
    }
}
