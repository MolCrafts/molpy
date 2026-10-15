//! Configuration values shared by several computes: the [`Axis`] and
//! [`RegularGrid`] classes, and the reader of the configuration kind that is
//! not a plain number (`Units`).
//!
//! A constructor argument the caller leaves out is never filled in here: an
//! optional parameter's setter runs only when the value is given
//! ([`Given::given`]), so every default is the core's.

use molrs::compute::{self as mc, ComputeError};
use molrs::core::{UnitPreset, lookup_unit_preset};
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Apply an optional constructor argument through its core setter.
pub(crate) trait Given: Sized {
    /// `set(self, value)` when `value` is given, `self` otherwise.
    fn given<V>(
        self,
        value: Option<V>,
        set: impl FnOnce(Self, V) -> Result<Self, ComputeError>,
    ) -> Result<Self, ComputeError> {
        match value {
            Some(value) => set(self, value),
            None => Ok(self),
        }
    }
}

impl<T> Given for T {}

/// The unit preset registered as `name` (`ParamKind::Units`).
pub(crate) fn unit_preset(name: &str) -> Result<UnitPreset, ComputeError> {
    lookup_unit_preset(name).ok_or_else(|| ComputeError::OutOfRange {
        field: "units (a registered unit preset name)",
        value: name.to_string(),
    })
}

/// `values` as a fixed-length array, or the error naming `what`.
pub(crate) fn fixed<T: Copy, const N: usize>(
    values: &[T],
    what: &'static str,
) -> Result<[T; N], ComputeError> {
    values
        .try_into()
        .map_err(|_| ComputeError::DimensionMismatch {
            expected: N,
            got: values.len(),
            what,
        })
}

/// A binning axis: `nBins` equal bins over `[min, max)` (`ParamKind::Axis`).
///
/// ```js
/// const axis = new Axis(0.0, 5.0, 100);
/// const df = new DistributionFunction("distance", axis);
/// ```
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Axis {
    pub(crate) inner: mc::Axis,
}

#[wasm_bindgen]
impl Axis {
    /// `n_bins` equal bins over `[min, max)`.
    #[wasm_bindgen(constructor)]
    pub fn new(min: F, max: F, n_bins: usize) -> Result<Axis, JsError> {
        Ok(Self {
            inner: mc::Axis::new(min, max, n_bins)?,
        })
    }

    /// Lower bound.
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> F {
        self.inner.min()
    }

    /// Upper bound.
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> F {
        self.inner.max()
    }

    /// Number of bins.
    #[wasm_bindgen(getter, js_name = nBins)]
    pub fn n_bins(&self) -> usize {
        self.inner.n_bins()
    }

    /// An independent copy of this axis.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> Axis {
        *self
    }
}

/// A regular 3-D grid: `dims[d]` voxels with edge vectors `basis[d]` from
/// `origin`, periodic along the axes `pbc` marks (`ParamKind::Grid`,
/// `InputKind::Grid`).
///
/// ```js
/// const grid = new RegularGrid([0, 0, 0], [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5],
///                              [20, 20, 20], [true, true, true]);
/// ```
#[wasm_bindgen]
#[derive(Clone)]
pub struct RegularGrid {
    pub(crate) inner: mc::RegularGrid<3>,
}

#[wasm_bindgen]
impl RegularGrid {
    /// `origin` (3), `basis` (9, row `d` is the voxel edge vector of axis
    /// `d`), `dims` (3) and `pbc` (3).
    #[wasm_bindgen(constructor)]
    pub fn new(
        origin: Vec<F>,
        basis: Vec<F>,
        dims: Vec<usize>,
        pbc: Vec<u8>,
    ) -> Result<RegularGrid, JsError> {
        let origin = fixed::<F, 3>(&origin, "RegularGrid origin (3 values)")?;
        let basis = fixed::<F, 9>(&basis, "RegularGrid basis (9 values, row-major)")?;
        let dims = fixed::<usize, 3>(&dims, "RegularGrid dims (3 values)")?;
        let pbc = fixed::<u8, 3>(&pbc, "RegularGrid pbc (3 flags)")?;
        let rows = basis.as_chunks::<3>().0;
        let inner = mc::RegularGrid::new(
            origin,
            [rows[0], rows[1], rows[2]],
            dims,
            pbc.map(|flag| flag != 0),
        )?;
        Ok(Self { inner })
    }

    /// An independent copy of this grid.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> RegularGrid {
        self.clone()
    }

    /// The grid as a plain object.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsError> {
        super::wire::to_js(&self.inner)
    }
}
