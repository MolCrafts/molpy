//! Einstein diffusion raw compute — the windowed-MSD route to D.

use molrs::core::FrameAccess;
use ndarray::Array1;

use crate::compute::ComputeError;
use crate::compute::ComputeResult;
use crate::compute::LegacyCompute;
use crate::compute::{Fit, LinearFit, Msd, MsdMode};

/// Raw self-MSD for the Einstein diffusion route.
#[derive(Debug, Clone)]
pub struct EinsteinDiffusionResult {
    /// Lag times τ = i·dt, length `n_frames`. Units: `[dt]`.
    pub lag_times: Array1<f64>,
    /// System-average MSD per lag, identical to
    /// [`Msd`] in `Window` mode's per-lag mean. Units: `[length]²`.
    pub msd: Array1<f64>,
}

impl ComputeResult for EinsteinDiffusionResult {}

impl EinsteinDiffusionResult {
    /// The self-diffusion coefficient by the Einstein relation,
    /// `D = slope / (2 · n_dims)`, where `slope` is the [`LinearFit`] of
    /// [`msd`](Self::msd) against [`lag_times`](Self::lag_times) over `window`
    /// (fractions of the last lag, as [`LinearFit::window`]). Units:
    /// `[length]² / [dt]`.
    ///
    /// This is the one place molrs turns an MSD slope into `D`; every binding
    /// calls it.
    ///
    /// # Errors
    ///
    /// [`ComputeError::OutOfRange`] for `n_dims == 0`; otherwise whatever the
    /// [`LinearFit`] refuses (a bad window, too few points).
    pub fn diffusion_coefficient(
        &self,
        n_dims: usize,
        window: (f64, f64),
    ) -> Result<f64, ComputeError> {
        if n_dims == 0 {
            return Err(ComputeError::OutOfRange {
                field: "n_dims",
                value: n_dims.to_string(),
            });
        }
        let fit = LinearFit { window }.fit((&self.lag_times, &self.msd))?;
        Ok(fit.slope / (2.0 * n_dims as f64))
    }
}

/// Raw self-MSD compute. Delegates to
/// [`Msd`] in `Window` mode — MSD math is **not** re-derived here.
/// `D = slope/(2d)` is then
/// [`EinsteinDiffusionResult::diffusion_coefficient`].
#[derive(Debug, Clone, Copy, Default)]
pub struct EinsteinDiffusion;

/// `EinsteinDiffusion` argument bundle: the frame spacing for the lag axis.
#[derive(Debug, Clone, Copy)]
pub struct EinsteinDiffusionArgs {
    /// Frame spacing for the lag-time axis. Units: time.
    pub dt: f64,
}

impl LegacyCompute for EinsteinDiffusion {
    /// Frame spacing only; positions come from `frames`.
    type Args<'a> = EinsteinDiffusionArgs;
    type Output = EinsteinDiffusionResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        if frames.is_empty() {
            return Err(ComputeError::EmptyInput);
        }
        if args.dt <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "dt",
                value: args.dt.to_string(),
            });
        }
        let series = Msd::with_mode(MsdMode::Window).compute(frames, ())?;
        let msd = Array1::from_iter(series.per_frame.iter().map(|r| r.mean));
        let lag_times = Array1::from_iter((0..series.per_frame.len()).map(|i| i as f64 * args.dt));
        Ok(EinsteinDiffusionResult { lag_times, msd })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Block;
    use molrs::core::Frame;
    use ndarray::Array1 as A1;

    fn make_frame(x: &[f64], y: &[f64], z: &[f64]) -> Frame {
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
    fn einstein_diffusion_delegates_to_msd_windowed() {
        // ac-012: EinsteinDiffusion.msd == MSD::with_mode(MsdMode::Window).compute means.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = [0.1, -0.2, 0.3, -0.1, 0.05, 0.0];
        let frames_owned: Vec<Frame> = (0..6)
            .map(|t| make_frame(&[xs[t], xs[t] + 0.5], &[ys[t], ys[t] + 0.4], &[0.0, 0.0]))
            .collect();
        let frames: Vec<&Frame> = frames_owned.iter().collect();

        let series = Msd::with_mode(MsdMode::Window)
            .compute(&frames, ())
            .unwrap();
        let raw = EinsteinDiffusion
            .compute(&frames, EinsteinDiffusionArgs { dt: 2.0 })
            .unwrap();
        assert_eq!(raw.msd.len(), series.per_frame.len());
        for i in 0..raw.msd.len() {
            assert!(
                (raw.msd[i] - series.per_frame[i].mean).abs() < 1e-12,
                "i={i}"
            );
            assert!((raw.lag_times[i] - i as f64 * 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn diffusion_coefficient_is_the_einstein_relation() {
        // MSD = 6 D t in 3-D with D = 0.25: slope 1.5, so D = 1.5 / 6.
        let lag_times = ndarray::Array1::from_iter((0..10).map(|i| i as f64 * 0.5));
        let msd = lag_times.mapv(|t| 1.5 * t);
        let result = EinsteinDiffusionResult { lag_times, msd };
        let d = result.diffusion_coefficient(3, (0.0, 1.0)).unwrap();
        assert!((d - 0.25).abs() < 1e-12, "{d}");
        assert!(result.diffusion_coefficient(0, (0.0, 1.0)).is_err());
    }
}
