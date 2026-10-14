//! Mean squared displacement — WASM face of `molrs::compute::Msd`.

use crate::core::frame::Frame;
use crate::core::nd_array::JsFloatArray;
use molrs::compute::LegacyCompute;
use molrs::compute::{Msd as RsMsd, MsdResult as RsMsdResult};
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Mean squared displacement (MSD) analysis.
///
/// Computes MSD = |r(t) - r(0)|^2 for each particle and the system
/// average. The first frame fed is automatically used as the reference.
/// Useful for measuring diffusion coefficients via D = MSD / (6t).
///
/// All distances are in angstrom (A), so MSD is in A^2.
///
/// # Example (JavaScript)
///
/// ```js
/// const msd = new Msd();
/// for (const frame of trajectory) {
///   msd.feed(frame);         // first frame = reference
/// }
/// const results = msd.results();  // MsdResult[] per frame
/// console.log(results[10].mean);  // MSD at frame 10 in A^2
/// ```
///
/// # References
///
/// - Einstein, A. (1905). *Annalen der Physik*, 322(8), 549-560.
#[wasm_bindgen(js_name = Msd)]
pub struct Msd {
    frames: Vec<molrs::core::Frame>,
}

#[allow(clippy::new_without_default)]
#[wasm_bindgen(js_class = Msd)]
impl Msd {
    /// Create an empty MSD analysis.
    ///
    /// The first frame passed to [`feed`] becomes the reference
    /// configuration (t=0).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new Msd();
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { frames: Vec::new() }
    }

    /// Feed a frame into the MSD analysis.
    ///
    /// Internally clones the frame's core data so subsequent mutations on
    /// the JS side (e.g. trajectory playback overwriting buffers) do not
    /// race against pending [`results`](Self::results) calls. The first
    /// frame sets the reference configuration.
    ///
    /// # Arguments
    ///
    /// * `frame` - Frame with `"atoms"` block containing
    ///   `x`, `y`, `z` (F) columns
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const msd = new Msd();
    /// msd.feed(frame0);  // sets reference
    /// msd.feed(frame1);  // added to trajectory
    /// const series = msd.results();
    /// ```
    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    /// Run the stateless [`molrs::compute::Msd`] over every fed frame and
    /// return the per-frame time series.
    ///
    /// The first frame is always the reference, so `results()[0].mean ≈ 0`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const results = msd.results();
    /// results.forEach((r, t) => console.log(`t=${t}: Msd=${r.mean}`));
    /// ```
    pub fn results(&self) -> Result<Vec<MsdResult>, JsValue> {
        if self.frames.is_empty() {
            return Ok(Vec::new());
        }
        let refs: Vec<&molrs::core::Frame> = self.frames.iter().collect();
        let series = RsMsd::new()
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("Msd results: {e}")))?;
        Ok(series
            .per_frame
            .iter()
            .map(|r| MsdResult { inner: r.clone() })
            .collect())
    }

    /// Number of frames accumulated.
    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.frames.len()
    }

    /// Reset the analysis, clearing the trajectory buffer.
    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Result of a mean squared displacement computation.
///
/// # Example (JavaScript)
///
/// ```js
/// const result = msd.compute(frame);
/// console.log(result.mean);              // number (A^2)
/// console.log(result.perParticle());     // Float64Array (A^2)
/// ```
#[wasm_bindgen(js_name = MsdResult)]
pub struct MsdResult {
    inner: RsMsdResult,
}

#[wasm_bindgen(js_class = MsdResult)]
impl MsdResult {
    /// System-average mean squared displacement in A^2.
    ///
    /// This is the arithmetic mean of all per-particle squared
    /// displacements: `mean = sum(|r_i(t) - r_i(0)|^2) / N`.
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> F {
        self.inner.mean
    }

    /// Zero-copy `Float64Array` view of per-particle squared displacements
    /// in A². `perParticle()[i]` is `|r_i(t) - r_i(0)|²` for particle `i`.
    /// **Invalidated** on WASM memory growth; copy in JS if needed.
    #[wasm_bindgen(js_name = perParticle)]
    pub fn per_particle(&self) -> JsFloatArray {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { JsFloatArray::view(self.inner.per_particle.as_slice().unwrap()) }
    }
}
