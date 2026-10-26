//! Distributions of internal coordinates and the spatial distribution
//! function — the WASM face of `molrs::compute::{distribution,
//! density::spatial_distribution}`.
//!
//! All three are frame-form computes: `compute` takes an array of per-frame
//! objects and folds them one at a time.
//!
//! ```js
//! const adf = new DistributionFunction("angle", new Axis(0, Math.PI, 90));
//! const result = adf.compute([{ positions, simbox, tuples: triples }]);
//!
//! const joint = new CombinedDistribution(["distance", "angle"],
//!     [new Axis(0, 4, 40), new Axis(0, Math.PI, 36)]);
//! joint.compute([{ positions, tuples: [pairs, triples] }]).marginal(0);
//! ```

use std::rc::Rc;

use js_sys::Array;
use molrs::compute::{self as mc, Accumulate, ComputeError, Observable, Stream};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::{Axis, Given, RegularGrid, fixed};
use super::handle::{Bound, described, output};
use super::wire::{self, Fields, records};
use crate::core::NDArray;

output! {
    /// A 1-D distribution: the `histogram`, its `density` (and the sin θ
    /// corrected density for an angle), sample and frame counts.
    DistributionResult => mc::DistributionResult
}

output! {
    /// The spatial distribution function on its grid: `counts`, `density`,
    /// `g_sdf`, the optional orientation histogram, `voxel_volume`.
    SpatialDistributionResult => mc::SpatialDistributionResult
}

/// The distribution of one internal coordinate over item tuples.
#[wasm_bindgen]
pub struct DistributionFunction {
    inner: mc::DistributionFunction<mc::InternalCoordinate>,
}

#[wasm_bindgen]
impl DistributionFunction {
    /// `observable` is an [`InternalCoordinate`](mc::InternalCoordinate) name.
    #[wasm_bindgen(constructor)]
    pub fn new(observable: &str, axis: &Axis) -> Result<DistributionFunction, JsError> {
        Ok(Self {
            inner: mc::DistributionFunction::new(observable.parse()?, axis.inner),
        })
    }

    /// The form binned over the observable's natural range.
    #[wasm_bindgen(js_name = overNaturalRange)]
    pub fn over_natural_range(
        observable: &str,
        n_bins: usize,
    ) -> Result<DistributionFunction, JsError> {
        Ok(Self {
            inner: mc::DistributionFunction::over_natural_range(observable.parse()?, n_bins)?,
        })
    }

    /// `[{ positions, simbox?, tuples }, …]`: `tuples` flat, the
    /// observable's arity per tuple.
    pub fn compute(&self, frames: &JsValue) -> Result<DistributionResult, JsError> {
        let arity = self.inner.observable().arity();
        let mut accumulator = self.inner.accumulator();
        for frame in records(frames, "DistributionFunction")? {
            let data = Fields::new(&frame, "DistributionFunction")?;
            let positions = data.matrix("positions", Some(3))?;
            let simbox = data.opt_simbox("simbox")?;
            let tuples = data.tuples("tuples", arity)?;
            accumulator.push(mc::DistributionFrame {
                positions: positions.view(),
                simbox: simbox.as_ref(),
                tuples: tuples.view(),
            })?;
        }
        Ok(accumulator.finish()?.into())
    }
}

/// A joint distribution of two or three internal coordinates.
#[wasm_bindgen]
pub struct CombinedDistribution {
    inner: Joint,
}

/// The core distribution, by its number of axes.
enum Joint {
    Two(mc::CombinedDistribution<2>),
    Three(mc::CombinedDistribution<3>),
}

/// The core distribution over `N` axes, sin θ weighted on `sin_weight`.
fn joint<const N: usize>(
    observables: &[mc::InternalCoordinate],
    axes: &[mc::Axis],
    sin_weight: Option<Vec<usize>>,
) -> Result<mc::CombinedDistribution<N>, ComputeError> {
    let observables = fixed::<_, N>(observables, "CombinedDistribution observables")?;
    let axes = fixed::<_, N>(axes, "CombinedDistribution axes (one per observable)")?;
    mc::CombinedDistribution::new(observables, axes)?.given(sin_weight, |c, weighted| {
        weighted
            .into_iter()
            .try_fold(c, |c, axis| c.with_sin_weight(axis))
    })
}

/// Fold `frames` through a distribution over `N` axes.
fn fold_joint<const N: usize>(
    distribution: &mc::CombinedDistribution<N>,
    frames: &JsValue,
) -> Result<mc::CombinedDistributionResult<N>, ComputeError> {
    let arities = distribution.observables().map(|o| o.arity());
    let mut accumulator = distribution.accumulator();
    for frame in records(frames, "CombinedDistribution")? {
        let data = Fields::new(&frame, "CombinedDistribution")?;
        let positions = data.matrix("positions", Some(3))?;
        let simbox = data.opt_simbox("simbox")?;
        let tables = data.tuples_list("tuples", &arities)?;
        let tuples: [_; N] = std::array::from_fn(|axis| tables[axis].view());
        accumulator.push(mc::CombinedDistributionFrame {
            positions: positions.view(),
            simbox: simbox.as_ref(),
            tuples,
        })?;
    }
    accumulator.finish()
}

#[wasm_bindgen]
impl CombinedDistribution {
    /// `observables` are [`InternalCoordinate`](mc::InternalCoordinate) names,
    /// one `Axis` each (two or three); `sinWeight` lists the angle axes to
    /// weight by sin θ.
    #[wasm_bindgen(constructor)]
    pub fn new(
        observables: Vec<String>,
        axes: Array,
        sin_weight: Option<Vec<usize>>,
    ) -> Result<CombinedDistribution, JsError> {
        let observables = observables
            .iter()
            .map(|name| name.parse())
            .collect::<Result<Vec<mc::InternalCoordinate>, _>>()?;
        let axes: Vec<mc::Axis> =
            wire::handles::<Axis>(&axes, "CombinedDistribution axes", "Axis")?
                .into_iter()
                .map(|axis| axis.inner)
                .collect();
        let inner = match observables.len() {
            2 => Joint::Two(joint(&observables, &axes, sin_weight)?),
            3 => Joint::Three(joint(&observables, &axes, sin_weight)?),
            n => {
                return Err(ComputeError::OutOfRange {
                    field: "CombinedDistribution observables (2 or 3)",
                    value: n.to_string(),
                }
                .into());
            }
        };
        Ok(Self { inner })
    }

    /// `[{ positions, simbox?, tuples }, …]`: `tuples` holds one flat table
    /// per axis, each of its observable's arity.
    pub fn compute(&self, frames: &JsValue) -> Result<CombinedDistributionResult, JsError> {
        let inner = match &self.inner {
            Joint::Two(d) => JointResult::Two(Rc::new(fold_joint(d, frames)?)),
            Joint::Three(d) => JointResult::Three(Rc::new(fold_joint(d, frames)?)),
        };
        Ok(CombinedDistributionResult { inner })
    }

    /// The molrs descriptor of this compute (one for every number of axes).
    pub fn spec() -> Result<JsValue, JsError> {
        wire::to_js(&<Self as Bound>::spec())
    }

    /// `[name, value]` for every parameter that has a default.
    pub fn tunables(&self) -> Result<JsValue, JsError> {
        wire::to_js(&Bound::tunables(self))
    }
}

impl Bound for CombinedDistribution {
    fn spec() -> mc::ComputeSpec {
        <mc::CombinedDistribution<2> as mc::Described>::spec()
    }

    fn tunables(&self) -> Vec<(&'static str, mc::ParamValue)> {
        match &self.inner {
            Joint::Two(d) => mc::Described::tunables(d),
            Joint::Three(d) => mc::Described::tunables(d),
        }
    }
}

/// The result of a joint distribution, by its number of axes.
enum JointResult {
    Two(Rc<mc::CombinedDistributionResult<2>>),
    Three(Rc<mc::CombinedDistributionResult<3>>),
}

/// A joint distribution: the N-D `histogram`, its `density`, sample and
/// frame counts, and which axes are sin θ corrected.
#[wasm_bindgen]
pub struct CombinedDistributionResult {
    inner: JointResult,
}

#[wasm_bindgen]
impl CombinedDistributionResult {
    /// The result as a plain object.
    #[wasm_bindgen(js_name = toJSON)]
    pub fn to_json(&self) -> Result<JsValue, JsError> {
        match &self.inner {
            JointResult::Two(r) => wire::to_js(&**r),
            JointResult::Three(r) => wire::to_js(&**r),
        }
    }

    /// A second handle to the same result.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> CombinedDistributionResult {
        let inner = match &self.inner {
            JointResult::Two(r) => JointResult::Two(Rc::clone(r)),
            JointResult::Three(r) => JointResult::Three(Rc::clone(r)),
        };
        Self { inner }
    }

    /// The 1-D distribution along `axis`, summed over the others.
    pub fn marginal(&self, axis: usize) -> Result<DistributionResult, JsError> {
        let marginal = match &self.inner {
            JointResult::Two(r) => r.marginal(axis)?,
            JointResult::Three(r) => r.marginal(axis)?,
        };
        Ok(marginal.into())
    }

    /// `−kT ln p` of the density at `temperature` (K).
    #[wasm_bindgen(js_name = freeEnergy)]
    pub fn free_energy(&self, temperature: F) -> NDArray {
        let energy = match &self.inner {
            JointResult::Two(r) => r.free_energy(temperature),
            JointResult::Three(r) => r.free_energy(temperature),
        };
        NDArray::from_ndarray(&energy)
    }
}

/// The density of target items around oriented reference fragments.
#[wasm_bindgen]
pub struct SpatialDistribution {
    inner: mc::SpatialDistribution,
}

#[wasm_bindgen]
impl SpatialDistribution {
    #[wasm_bindgen(constructor)]
    pub fn new(
        grid: &RegularGrid,
        bulk_density: Option<F>,
    ) -> Result<SpatialDistribution, JsError> {
        let inner = mc::SpatialDistribution::new(grid.inner.clone())
            .given(bulk_density, |c, v| c.with_bulk_density(v))?;
        Ok(Self { inner })
    }

    /// `[{ positions, simbox?, reference, template, target, orientation? }, …]`.
    pub fn compute(&self, frames: &JsValue) -> Result<SpatialDistributionResult, JsError> {
        let mut accumulator = self.inner.accumulator();
        for frame in records(frames, "SpatialDistribution")? {
            let data = Fields::new(&frame, "SpatialDistribution")?;
            let positions = data.matrix("positions", Some(3))?;
            let simbox = data.opt_simbox("simbox")?;
            let reference = data.indices("reference")?;
            let template = data.matrix("template", Some(3))?;
            let target = data.indices("target")?;
            let orientation = data.opt_index_pairs("orientation")?;
            accumulator.push(mc::SpatialDistributionFrame {
                positions: positions.view(),
                simbox: simbox.as_ref(),
                reference: &reference,
                template: template.view(),
                target: &target,
                orientation: orientation.as_deref(),
            })?;
        }
        Ok(accumulator.finish()?.into())
    }
}

described!(
    DistributionFunction => mc::DistributionFunction<mc::InternalCoordinate>,
    SpatialDistribution => mc::SpatialDistribution,
);
