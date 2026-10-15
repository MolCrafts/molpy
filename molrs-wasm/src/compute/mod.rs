//! WASM bindings for trajectory analysis — the face of `molrs::compute`.
//!
//! Every entry of `ComputeSpec::all()` is a class here under exactly its molrs
//! name, and nothing else is: no free functions, no binding-only analyses.
//! Each class follows the core contract:
//!
//! - **the constructor takes configuration only**, in the order the spec
//!   lists its parameters. An optional parameter is `undefined` unless given,
//!   and only a given one reaches its core setter, so every default is the
//!   core's. A choice is one of its canonical names, parsed by the core;
//! - **`compute(data)` takes the data**: one object whose properties are the
//!   fields of the core `Input` struct, under the names `spec().inputs` lists
//!   (see [`wire`] for how each kind is spelled). A frame-form compute
//!   (`Rdf`, the distributions) takes an array of such per-frame objects;
//! - **the result is a handle** of the core output type, under its molrs name
//!   (`TimeCorrelation`, `RdfResult`, …): `toJSON()` gives its data, and the
//!   handle itself feeds a downstream compute (`TimeCorrelation` → `LinearFit`,
//!   `ClusterResult` → `CenterOfMass`, `PcaResult` → `Kmeans`);
//! - **errors** are the core `ComputeError`, thrown as a JavaScript `Error`;
//! - `spec()` (static) and `tunables()` read the core descriptor, and
//!   [`ComputeSpec.all()`](ComputeSpec::all) lists every one of them.
//!
//! No binding reads a frame: positions, velocities, charges, orientations and
//! neighbour tables are data the caller passes.
//!
//! # Example (JavaScript)
//!
//! ```js
//! const nl = new NeighborList(2.0);
//! nl.build(frame);
//! const pairs = nl.neighbors();
//!
//! const clusters = new Cluster(2).compute({ pairs });
//! const com = new CenterOfMass().compute({
//!     positions, simbox, partition: clusters, masses,
//! });
//!
//! const gr = new Rdf(100, 5.0, 0.0).compute([{ pairs }]);
//! console.log(gr.toJSON().rdf.data);
//! ```

mod config;
mod dielectric;
mod distribution;
mod dynamics;
mod environment;
mod fitting;
mod groups;
mod handle;
mod order;
mod pmft;
mod spectroscopy;
mod structure;
mod time_series;
mod upstream;
#[cfg(feature = "voronoi")]
mod voronoi;
mod wire;

#[cfg(test)]
mod tests;

pub use config::{Axis, RegularGrid};
pub use dielectric::*;
pub use distribution::*;
pub use dynamics::*;
pub use environment::*;
pub use fitting::*;
pub use groups::*;
pub use order::*;
pub use pmft::*;
pub use spectroscopy::*;
pub use structure::*;
pub use time_series::*;
#[cfg(feature = "voronoi")]
pub use voronoi::*;

use molrs::compute as mc;
use wasm_bindgen::prelude::*;

/// The registry of every compute type this build binds: the core
/// `ComputeSpec::all()`, serialised.
///
/// Each entry is `{ name, params, inputs, form }`:
///
/// - `name` — the class exported under that name;
/// - `params` — `{ name, kind, default }` per constructor argument / setter,
///   in constructor order; `default` is `"Required"`, `{ RequiredIn: ctor }`,
///   `{ Value: v }`, `{ SameAs: param }` or `{ Scaled: { of, factor } }`;
/// - `inputs` — `{ name, kind, optional }` per `compute` data field;
/// - `form` — `"Single"` (one data object) or `"Frames"` (an array of them).
///
/// Labels, units, categories and slider ranges are presentation and live with
/// the UI that presents them, keyed by `(name, param.name)`.
///
/// ```js
/// for (const spec of ComputeSpec.all()) console.log(spec.name, spec.params);
/// ```
#[wasm_bindgen]
pub struct ComputeSpec {}

#[wasm_bindgen]
impl ComputeSpec {
    /// The descriptor of every compute type, one per class.
    pub fn all() -> Result<JsValue, JsError> {
        wire::to_js(mc::ComputeSpec::all())
    }

    /// The descriptor named `name`, or `undefined`.
    pub fn find(name: &str) -> Result<JsValue, JsError> {
        match mc::ComputeSpec::find(name) {
            Some(spec) => wire::to_js(spec),
            None => Ok(JsValue::UNDEFINED),
        }
    }
}
