//! Resonance-Raman iso/aniso ACF raw compute — Raman machinery over a resonant
//! polarizability series.

use molrs::core::FrameAccess;
use ndarray::Array2;

use super::raman_tensor::{RamanTensor, RamanTensorResult};
use crate::compute::ComputeError;
use crate::compute::LegacyCompute;

/// Raw resonance-Raman iso/aniso ACFs — identical machinery to [`RamanTensor`]
/// but consuming a caller-supplied **resonant** (excitation-frequency-dependent)
/// polarizability series instead of the static one.
///
/// molrs does not compute the excited-state response; the caller supplies the
/// resonant polarizability time series and this compute produces the same raw
/// iso/aniso ACFs (Voigt `[xx, yy, zz, xy, xz, yz]`) that [`RamanTensor`] /
/// `raman.cpp` produce, so the resonance-Raman spectrum reuses the Raman fit.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResonanceRamanTensor;

/// `(resonant_polarizabilities (n,6), dt, resolution)` for
/// [`ResonanceRamanTensor`] — same shape/convention as
/// [`RamanTensorArgs`](super::raman_tensor::RamanTensorArgs).
pub type ResonanceRamanTensorArgs<'a> = (&'a Array2<f64>, f64, usize);

impl LegacyCompute for ResonanceRamanTensor {
    type Args<'a> = ResonanceRamanTensorArgs<'a>;
    type Output = RamanTensorResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        // Structurally identical to the static Raman tensor; only the input
        // polarizability differs (resonant vs static). Reuse, do not duplicate.
        RamanTensor.compute(frames, args)
    }
}
