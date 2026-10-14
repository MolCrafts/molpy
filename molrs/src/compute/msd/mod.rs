//! Mean squared displacement analysis (stateless).

mod accumulator;
mod result;

pub use accumulator::MsdAccumulator;
pub use result::{MsdResult, MsdTimeSeries};

use molrs::core::FrameAccess;
use molrs::op::F;
use ndarray::Array1;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex as RfComplex;

use crate::compute::ComputeError;
use crate::compute::LegacyCompute;
use crate::compute::positions::get_positions_ref_any_dim;

/// Mode of MSD computation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MsdMode {
    /// `Msd(t) = ⟨|r(t) − r(0)|²⟩`, single time origin (frame 0).
    #[default]
    Direct,
    /// `Msd(t) = ⟨|r(τ+t) − r(τ)|²⟩` averaged over all time origins,
    /// computed via FFT autocorrelation. O(T log T) per particle.
    Window,
}

/// Mean squared displacement analysis.
///
/// # Hard precondition — unwrapped coordinates
///
/// Positions must be **continuous / unwrapped** across the trajectory (or
/// unwrapped by the caller with image flags + box **before** MSD). This
/// kernel does **not** apply the minimum-image convention: MIC inside MSD
/// re-wraps jumps and destroys the Einstein diffusion signal. Wrapped PBC
/// dumps silently saturate MSD ~ \(L^2\) and yield meaningless \(D\).
/// Same contract as freud `Msd` / MDAnalysis EinsteinMSD.
///
/// Given a slice of frames, produces one [`MsdResult`] per lag-time. Two
/// modes are supported, matching the conventions in `freud.msd`:
///
/// - [`MsdMode::Direct`] — `Msd(t) = ⟨|r(t) − r(0)|²⟩_i` with frame 0 as
///   the single time origin.
/// - [`MsdMode::Window`] — `Msd(t) = ⟨|r(τ+t) − r(τ)|²⟩_{i, τ}` averaged
///   over all time origins τ. Implemented in O(N log N) via the
///   Wiener–Khinchin identity (zero-padded autocorrelation through
///   `rustfft`) — the nMoldyn / Allen-Tildesley algorithm.
///
/// Both modes produce the same `MsdTimeSeries` output shape; callers select
/// via [`Msd::with_mode`] (default is `Direct`).
#[derive(Debug, Clone, Copy, Default)]
pub struct Msd {
    mode: MsdMode,
}

impl Msd {
    /// New MSD analyzer in direct (single-reference) mode.
    pub fn new() -> Self {
        Self {
            mode: MsdMode::Direct,
        }
    }

    /// New MSD analyzer with the given mode.
    pub fn with_mode(mode: MsdMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> MsdMode {
        self.mode
    }
}

fn msd_vs_reference<FA: FrameAccess>(
    frame: &FA,
    ref_x: &[F],
    ref_y: &[F],
    ref_z: &[F],
) -> Result<MsdResult, ComputeError> {
    let (xs_p, ys_p, zs_p) = get_positions_ref_any_dim(frame)?;
    let xs = xs_p.slice();
    let ys = ys_p.slice();
    let zs = zs_p.slice();
    let n = xs.len();
    if n != ref_x.len() {
        return Err(ComputeError::DimensionMismatch {
            expected: ref_x.len(),
            got: n,
            what: "Msd particle count",
        });
    }
    let mut per_particle = Array1::<F>::zeros(n);
    let mut total: F = 0.0;
    let pp = per_particle.as_slice_mut().expect("zeros is contiguous");
    // Tight scalar loop — autovectorizes on AArch64/NEON and x86-64/AVX.
    for i in 0..n {
        let dx = xs[i] - ref_x[i];
        let dy = ys[i] - ref_y[i];
        let dz = zs[i] - ref_z[i];
        let d2 = dx * dx + dy * dy + dz * dz;
        pp[i] = d2;
        total += d2;
    }
    let mean = if n > 0 { total / n as F } else { 0.0 };
    Ok(MsdResult { per_particle, mean })
}

/// Unnormalised linear autocorrelation of a length-`T` real series via FFT.
///
/// Result `out[t] = Σ_{τ=0}^{T-1-t} x[τ] · x[τ+t]`, length `T`. The series
/// is zero-padded to length `2T` for linear (not circular) autocorrelation.
fn autocorrelate_fft(planner: &mut FftPlanner<F>, x: &[F]) -> Vec<F> {
    let t = x.len();
    let n = (2 * t).next_power_of_two();
    let mut buf: Vec<RfComplex<F>> = (0..n)
        .map(|i| {
            if i < t {
                RfComplex::new(x[i], 0.0)
            } else {
                RfComplex::new(0.0, 0.0)
            }
        })
        .collect();
    let fwd = planner.plan_fft_forward(n);
    fwd.process(&mut buf);
    for c in buf.iter_mut() {
        *c = RfComplex::new(c.norm_sqr(), 0.0);
    }
    let inv = planner.plan_fft_inverse(n);
    inv.process(&mut buf);
    let scale = 1.0 / n as F;
    buf.iter().take(t).map(|c| c.re * scale).collect()
}

/// Windowed MSD via Wiener–Khinchin.
///
/// Computes, for each particle i and lag t in [0, T):
///
/// ```text
///   MSD_i(t) = (1/(T-t)) Σ_τ |r_i(τ+t) - r_i(τ)|²
/// ```
///
/// using the identity
/// `|a-b|² = |a|² + |b|² - 2 a·b` and the FFT autocorrelation of each
/// coordinate component.
fn msd_windowed<FA: FrameAccess + Sync>(frames: &[&FA]) -> Result<MsdTimeSeries, ComputeError> {
    let t = frames.len();
    if t == 0 {
        return Err(ComputeError::EmptyInput);
    }
    // Gather positions: positions[k] = (x: Vec<F>, y, z) at frame k.
    let (xs0_p, _, _) = get_positions_ref_any_dim(frames[0])?;
    let n = xs0_p.slice().len();
    let mut x = vec![vec![0.0; t]; n];
    let mut y = vec![vec![0.0; t]; n];
    let mut z = vec![vec![0.0; t]; n];
    let mut r2 = vec![vec![0.0; t]; n];

    for (k, frame) in frames.iter().enumerate() {
        let (xp, yp, zp) = get_positions_ref_any_dim(*frame)?;
        let xs = xp.slice();
        let ys = yp.slice();
        let zs = zp.slice();
        if xs.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: xs.len(),
                what: "Msd particle count",
            });
        }
        for i in 0..n {
            x[i][k] = xs[i];
            y[i][k] = ys[i];
            z[i][k] = zs[i];
            r2[i][k] = xs[i] * xs[i] + ys[i] * ys[i] + zs[i] * zs[i];
        }
    }

    let mut planner = FftPlanner::<F>::new();
    // Per-particle MSD time series.
    let mut per_particle_per_lag = vec![vec![0.0_f64; n]; t];

    for i in 0..n {
        // Autocorrelations of each component (unnormalised).
        let ac_x = autocorrelate_fft(&mut planner, &x[i]);
        let ac_y = autocorrelate_fft(&mut planner, &y[i]);
        let ac_z = autocorrelate_fft(&mut planner, &z[i]);

        // S[t] = Σ_τ (|r(τ+t)|² + |r(τ)|²), τ ∈ [0, T-1-t]
        // Recurrence: S[0] = 2 Σ_τ |r(τ)|²;
        //              S[t] = S[t-1] − |r(t-1)|² − |r(T-t)|²
        let mut s: F = 2.0 * r2[i].iter().sum::<F>();
        for lag in 0..t {
            let denom = (t - lag) as F;
            let ac = ac_x[lag] + ac_y[lag] + ac_z[lag];
            per_particle_per_lag[lag][i] = (s - 2.0 * ac) / denom;
            // Update S for next lag.
            if lag + 1 < t {
                s -= r2[i][lag];
                s -= r2[i][t - 1 - lag];
            }
        }
    }

    let mut data: Vec<MsdResult> = Vec::with_capacity(t);
    for per_particle in per_particle_per_lag.into_iter() {
        let pp = Array1::from_vec(per_particle);
        let mean = if n > 0 {
            pp.iter().sum::<F>() / n as F
        } else {
            0.0
        };
        data.push(MsdResult {
            per_particle: pp,
            mean,
        });
    }
    Ok(MsdTimeSeries::new(data))
}

impl LegacyCompute for Msd {
    type Args<'a> = ();
    type Output = MsdTimeSeries;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        _args: (),
    ) -> Result<MsdTimeSeries, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        match self.mode {
            MsdMode::Direct => self.compute_direct(frames),
            MsdMode::Window => msd_windowed(frames),
        }
    }
}

impl Msd {
    fn compute_direct<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
    ) -> Result<MsdTimeSeries, ComputeError> {
        // Own the reference positions so downstream frames can be processed
        // in parallel. Contiguous views on the first frame become slice
        // copies into owned Vecs once.
        let (rx_p, ry_p, rz_p) = get_positions_ref_any_dim(frames[0])?;
        let ref_x: Vec<F> = rx_p.slice().to_vec();
        let ref_y: Vec<F> = ry_p.slice().to_vec();
        let ref_z: Vec<F> = rz_p.slice().to_vec();

        // Per-frame MSD is a single pass through N atoms with trivial
        // arithmetic, so rayon's task-submission overhead (~5-10 µs)
        // dominates until we have ~8+ frames. Fall back to serial below
        // that threshold.
        let serial = || {
            frames
                .iter()
                .map(|frame| msd_vs_reference(*frame, &ref_x, &ref_y, &ref_z))
                .collect::<Result<Vec<_>, _>>()
        };
        #[cfg(feature = "rayon")]
        let results: Vec<MsdResult> = if frames.len() >= 8 {
            use rayon::prelude::*;
            frames
                .par_iter()
                .map(|frame| msd_vs_reference(*frame, &ref_x, &ref_y, &ref_z))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            serial()?
        };
        #[cfg(not(feature = "rayon"))]
        let results: Vec<MsdResult> = serial()?;

        Ok(MsdTimeSeries::new(results))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use ndarray::Array1 as A1;

    fn make_frame(x: &[F], y: &[F], z: &[F]) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", A1::from_vec(x.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("y", A1::from_vec(y.to_vec()).into_dyn())
            .unwrap();
        block
            .insert("z", A1::from_vec(z.to_vec()).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    #[test]
    fn reference_frame_msd_is_zero() {
        let f0 = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let f1 = make_frame(&[1.0, 1.0], &[0.0, 0.0], &[0.0, 0.0]);
        let series = Msd::new().compute(&[&f0, &f1], ()).unwrap();
        assert_eq!(series.len(), 2);
        assert!(series.per_frame[0].mean.abs() < 1e-12);
        assert!((series.per_frame[1].mean - 1.0).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_calls() {
        let f0 = make_frame(&[0.0, 0.0, 0.0], &[0.0; 3], &[0.0; 3]);
        let f1 = make_frame(&[1.0, 1.0, 1.0], &[0.0; 3], &[0.0; 3]);
        let f2 = make_frame(&[2.0, 2.0, 2.0], &[0.0; 3], &[0.0; 3]);
        let msd = Msd::new();
        let a = msd.compute(&[&f0, &f1, &f2], ()).unwrap();
        let b = msd.compute(&[&f0, &f1, &f2], ()).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a.per_frame[i].mean - b.per_frame[i].mean).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_progression() {
        // Frame i: each particle at (i, 0, 0). MSD(i) = i² per particle.
        let frames_owned: Vec<Frame> = (0..4)
            .map(|i| make_frame(&[i as F; 3], &[0.0; 3], &[0.0; 3]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();
        let series = Msd::new().compute(&frames, ()).unwrap();
        for i in 0..4 {
            let expected = (i as F) * (i as F);
            assert!(
                (series.per_frame[i].mean - expected).abs() < 1e-12,
                "MSD[{i}] = {}, expected {expected}",
                series.per_frame[i].mean
            );
        }
    }

    #[test]
    fn empty_input_errors() {
        let frames: Vec<&Frame> = Vec::new();
        let err = Msd::new().compute(&frames, ()).unwrap_err();
        assert!(matches!(err, ComputeError::EmptyInput));
    }

    #[test]
    fn mismatched_particle_count_errors() {
        let f0 = make_frame(&[0.0, 0.0], &[0.0, 0.0], &[0.0, 0.0]);
        let f1 = make_frame(&[1.0], &[0.0], &[0.0]); // one particle only
        let err = Msd::new().compute(&[&f0, &f1], ()).unwrap_err();
        assert!(matches!(
            err,
            ComputeError::DimensionMismatch {
                expected: 2,
                got: 1,
                ..
            }
        ));
    }

    // --- Windowed mode (Wiener-Khinchin / FFT) ---

    /// Reference implementation: nested O(T²) loop. Returns one MSD value per
    /// lag time, averaged over all valid time origins.
    fn windowed_reference(frames: &[&Frame]) -> Vec<F> {
        let t = frames.len();
        let n = {
            let (xp, _, _) = get_positions_ref_any_dim(frames[0]).unwrap();
            xp.slice().len()
        };
        // Gather positions.
        let mut x = vec![vec![0.0; n]; t];
        let mut y = vec![vec![0.0; n]; t];
        let mut z = vec![vec![0.0; n]; t];
        for (k, f) in frames.iter().enumerate() {
            let (xp, yp, zp) = get_positions_ref_any_dim(*f).unwrap();
            x[k] = xp.slice().to_vec();
            y[k] = yp.slice().to_vec();
            z[k] = zp.slice().to_vec();
        }
        let mut out = vec![0.0_f64; t];
        for lag in 0..t {
            let n_origins = t - lag;
            let mut acc: F = 0.0;
            for tau in 0..n_origins {
                for i in 0..n {
                    let dx = x[tau + lag][i] - x[tau][i];
                    let dy = y[tau + lag][i] - y[tau][i];
                    let dz = z[tau + lag][i] - z[tau][i];
                    acc += dx * dx + dy * dy + dz * dz;
                }
            }
            out[lag] = acc / (n_origins as F * n as F);
        }
        out
    }

    #[test]
    fn windowed_matches_nested_loop_reference() {
        // Two particles drifting along x with a small random component in y.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys_origins = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames_owned: Vec<Frame> = (0..6)
            .map(|t| {
                make_frame(
                    &[xs[t], xs[t] + 0.5],
                    &[ys_origins[t], ys_origins[t] + 0.4],
                    &[0.0, 0.0],
                )
            })
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();

        let reference = windowed_reference(&frames);
        let computed = Msd::with_mode(MsdMode::Window)
            .compute(&frames, ())
            .unwrap();

        assert_eq!(computed.len(), 6);
        for (lag, &ref_val) in reference.iter().enumerate() {
            assert!(
                (computed.per_frame[lag].mean - ref_val).abs() < 1e-9,
                "lag {lag}: window={}, ref={ref_val}",
                computed.per_frame[lag].mean,
            );
        }
    }

    #[test]
    fn windowed_lag_zero_is_zero() {
        let f0 = make_frame(&[0.0, 1.0], &[0.0; 2], &[0.0; 2]);
        let f1 = make_frame(&[1.0, 2.0], &[0.0; 2], &[0.0; 2]);
        let f2 = make_frame(&[2.0, 3.0], &[0.0; 2], &[0.0; 2]);
        let series = Msd::with_mode(MsdMode::Window)
            .compute(&[&f0, &f1, &f2], ())
            .unwrap();
        assert!(series.per_frame[0].mean.abs() < 1e-10);
    }

    #[test]
    fn windowed_diffusive_signature_linear_in_lag() {
        // Pure ballistic drift along +x: r_i(t) = (t·v_i, 0, 0). Windowed MSD
        // for lag t equals t² v_i² averaged over particles AND time origins,
        // which still gives t² · ⟨v²⟩ since each (τ+t − τ) = t.
        let v = [1.0, 2.0];
        let frames_owned: Vec<Frame> = (0..8)
            .map(|t| make_frame(&[t as F * v[0], t as F * v[1]], &[0.0; 2], &[0.0; 2]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();
        let series = Msd::with_mode(MsdMode::Window)
            .compute(&frames, ())
            .unwrap();
        let mean_v2 = (v[0] * v[0] + v[1] * v[1]) / 2.0;
        for lag in 0..8 {
            let expected = (lag as F) * (lag as F) * mean_v2;
            assert!(
                (series.per_frame[lag].mean - expected).abs() < 1e-10,
                "lag {lag}: got {}, expected {expected}",
                series.per_frame[lag].mean,
            );
        }
    }

    #[test]
    fn msd_is_dimension_agnostic() {
        // A 1-D series: the frame carries only `x`. Missing axes read as zero,
        // so the answer is the 1-D MSD rather than a MissingColumn error.
        let xs: Vec<F> = vec![0.0, 1.0, 3.0];
        let frames: Vec<Frame> = xs
            .iter()
            .map(|&x| {
                let mut f = Frame::new();
                let mut b = Block::new();
                b.insert("x", A1::from_vec(vec![x]).into_dyn()).unwrap();
                f.insert("atoms", b);
                f
            })
            .collect();
        let refs: Vec<&Frame> = frames.iter().collect();

        let direct = Msd::new().compute(&refs, ()).unwrap();
        let means: Vec<F> = direct.per_frame.iter().map(|r| r.mean).collect();
        assert_eq!(means, vec![0.0, 1.0, 9.0]);

        // and the windowed estimator agrees with the nested loop in 1-D too
        let windowed = Msd::with_mode(MsdMode::Window).compute(&refs, ()).unwrap();
        let w: Vec<F> = windowed.per_frame.iter().map(|r| r.mean).collect();
        assert!((w[1] - ((1.0 + 4.0) / 2.0)).abs() < 1e-12, "{w:?}");
    }
}
