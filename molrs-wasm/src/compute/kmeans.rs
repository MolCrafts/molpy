//! k-means over a PCA projection — the WASM face of `molrs::compute`'s
//! `kmeans` ([`molrs::compute::Kmeans`]).

use js_sys::Int32Array;
use molrs::compute::Kmeans as RsKmeans;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Wrapper for [`molrs::compute::Kmeans`].
///
/// # Example (JavaScript)
///
/// ```js
/// const km = new Kmeans(3, 100, 42);
/// const labels = km.fit(coords, nRows, 2);   // Int32Array
/// ```
#[wasm_bindgen]
pub struct Kmeans {
    inner: RsKmeans,
}

#[wasm_bindgen(js_class = Kmeans)]
impl Kmeans {
    /// Create a new k-means configuration.
    ///
    /// # Arguments
    ///
    /// * `k` — number of clusters (>= 1).
    /// * `max_iter` — maximum Lloyd iterations (>= 1).
    /// * `seed` — RNG seed for k-means++ initialization. Cast to `u64`
    ///   internally (JS numbers are `f64`; integers up to 2^53 pass
    ///   through losslessly).
    ///
    /// # Errors
    ///
    /// Throws if `k == 0` or `max_iter == 0`.
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize, max_iter: usize, seed: f64) -> Result<Kmeans, JsValue> {
        let seed_u64 = seed as u64;
        RsKmeans::new(k, max_iter, seed_u64)
            .map(|inner| Kmeans { inner })
            .map_err(|e| JsValue::from_str(&format!("Kmeans: {e}")))
    }

    /// Cluster a row-major `n_rows × n_dims` coordinate matrix.
    ///
    /// # Returns
    ///
    /// Cluster labels in `0..k` as an owned `Int32Array`, one per row.
    ///
    /// # Errors
    ///
    /// Throws if `k > n_rows`, `n_dims == 0`, the length does not match
    /// `n_rows * n_dims`, or any element is non-finite.
    pub fn fit(&self, coords: &[F], n_rows: usize, n_dims: usize) -> Result<Int32Array, JsValue> {
        if n_dims != 2 {
            return Err(JsValue::from_str(
                "Kmeans wasm binding supports n_dims=2 only (PCA-score input)",
            ));
        }
        if coords.len() != n_rows * n_dims {
            return Err(JsValue::from_str(&format!(
                "Kmeans: coords length {} != n_rows * n_dims = {} * {}",
                coords.len(),
                n_rows,
                n_dims
            )));
        }
        let pca = molrs::compute::PcaResult {
            coords: coords.to_vec(),
            variance: [0.0 as F, 0.0 as F],
        };
        let dummy = molrs::core::Frame::new();
        let labels = self
            .inner
            .compute(&[&dummy], &pca)
            .map_err(|e| JsValue::from_str(&format!("Kmeans fit: {e}")))?;
        let out = Int32Array::new_with_length(labels.0.len() as u32);
        out.copy_from(&labels.0);
        Ok(out)
    }
}
