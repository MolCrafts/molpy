//! Structure builders — the WASM face of `molrs::builder`.

use molrs::builder::{CarbonTubeBuilder as RsCarbonTubeBuilder, TubeExtent as RsTubeExtent};
use wasm_bindgen::prelude::*;

use crate::core::Frame;

/// How far a [`CarbonTubeBuilder`] tube extends along its axis — the WASM
/// face of `molrs::builder::TubeExtent`, one static constructor per variant.
///
/// # Example (JavaScript)
///
/// ```js
/// builder.setExtent(TubeExtent.cells(4));      // four translational cells
/// builder.setExtent(TubeExtent.length(25.0));  // ≥ 25 Å, whole cells
/// ```
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct TubeExtent {
    inner: RsTubeExtent,
}

#[wasm_bindgen]
impl TubeExtent {
    /// A fixed number (≥ 1) of axial translational cells.
    pub fn cells(cells: u32) -> TubeExtent {
        TubeExtent {
            inner: RsTubeExtent::Cells(cells as usize),
        }
    }

    /// An axial length in angstrom (> 0), rounded up to complete
    /// translational cells.
    pub fn length(length: f64) -> TubeExtent {
        TubeExtent {
            inner: RsTubeExtent::Length(length),
        }
    }
}

/// Exact single-wall carbon nanotube builder.
#[wasm_bindgen]
pub struct CarbonTubeBuilder {
    inner: RsCarbonTubeBuilder,
}

#[wasm_bindgen]
impl CarbonTubeBuilder {
    /// Start a builder for chiral indices `(n, m)`.
    #[wasm_bindgen(constructor)]
    pub fn new(n: u32, m: u32) -> Result<CarbonTubeBuilder, JsValue> {
        Ok(Self {
            inner: RsCarbonTubeBuilder::new(n, m).map_err(js_error)?,
        })
    }

    /// Set how far the tube extends along its axis — `with_extent`.
    /// Unset, the tube is one translational cell long (two when periodic).
    ///
    /// # Errors
    ///
    /// Throws for zero cells or a non-positive length.
    #[wasm_bindgen(js_name = setExtent)]
    pub fn set_extent(&mut self, extent: &TubeExtent) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_extent(extent.inner)
            .map_err(js_error)?;
        Ok(())
    }

    /// Set the carbon-carbon bond length in angstrom.
    #[wasm_bindgen(js_name = setBondLength)]
    pub fn set_bond_length(&mut self, bond_length: f64) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_bond_length(bond_length)
            .map_err(js_error)?;
        Ok(())
    }

    /// Enable or disable periodic closure along the tube axis.
    #[wasm_bindgen(js_name = setPeriodic)]
    pub fn set_periodic(&mut self, periodic: bool) {
        self.inner = self.inner.clone().with_periodic(periodic);
    }

    /// Set transverse vacuum padding in angstrom.
    #[wasm_bindgen(js_name = setVacuum)]
    pub fn set_vacuum(&mut self, vacuum: f64) -> Result<(), JsValue> {
        self.inner = self.inner.clone().with_vacuum(vacuum).map_err(js_error)?;
        Ok(())
    }

    /// Build a fresh frame containing atoms, bonds, and the simulation box.
    pub fn build(&self) -> Result<Frame, JsValue> {
        Frame::from_rs(self.inner.build().map_err(js_error)?)
    }

    /// Validate the exact lattice and periodic topology.
    pub fn validate(&self) -> Result<(), JsValue> {
        self.inner.validate().map_err(js_error)
    }
}

fn js_error(error: molrs::builder::CarbonTubeError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn builds_frame_for_javascript() {
        let mut builder = CarbonTubeBuilder::new(6, 0).unwrap();
        builder.set_extent(&TubeExtent::cells(2)).unwrap();
        let frame = builder.build().unwrap();
        assert!(frame.has("atoms").unwrap());
        assert!(frame.has("bonds").unwrap());
        assert!(builder.set_extent(&TubeExtent::cells(0)).is_err());
        assert!(builder.set_extent(&TubeExtent::length(-1.0)).is_err());
        builder.set_extent(&TubeExtent::length(10.0)).unwrap();
        assert!(builder.build().is_ok());
    }
}
