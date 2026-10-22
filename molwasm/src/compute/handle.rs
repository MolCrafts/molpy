//! The two macros every binding in this module is built from.
//!
//! - [`output!`](output) binds a core **output** type as a handle class of the
//!   same name. A `compute` returns one; `toJSON()` gives its data as a plain
//!   object, and the handle itself is what a downstream compute's `Upstream`,
//!   `Partition` or `FeatureMatrix` field takes. The core value lives behind
//!   an `Rc`, so `clone()` — how a data argument borrows it — copies nothing.
//! - [`described!`](described) gives a compute class its `spec()` (static) and
//!   `tunables()`, both read from the core [`Described`] impl, and the
//!   crate-internal [`Bound`] the parity tests enumerate.

use molrs::compute::{ComputeSpec, ParamValue};

/// A compute class bound over one core compute type.
pub(crate) trait Bound {
    /// The core descriptor of the bound type.
    fn spec() -> ComputeSpec;

    /// The value of every defaulted parameter this instance holds.
    fn tunables(&self) -> Vec<(&'static str, ParamValue)>;
}

/// Bind a core output type as a handle class.
///
/// ```ignore
/// output! {
///     /// A time-correlation curve.
///     TimeCorrelation => mc::TimeCorrelation
/// }
/// ```
macro_rules! output {
    ($(#[$meta:meta])* $name:ident => $core:ty) => {
        $(#[$meta])*
        ///
        /// `toJSON()` returns the data as a plain object with the molrs field
        /// names; `clone()` returns a second handle to the same value.
        #[wasm_bindgen::prelude::wasm_bindgen]
        pub struct $name {
            pub(crate) inner: std::rc::Rc<$core>,
        }

        #[wasm_bindgen::prelude::wasm_bindgen]
        impl $name {
            /// The result as a plain object: the molrs field names, every
            /// array as `{ v, dim, data }`.
            #[wasm_bindgen(js_name = toJSON)]
            pub fn to_json(&self) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsError> {
                $crate::compute::wire::to_js(&*self.inner)
            }

            /// A second handle to the same result; nothing is copied.
            #[wasm_bindgen(js_name = clone)]
            pub fn duplicate(&self) -> $name {
                $name {
                    inner: std::rc::Rc::clone(&self.inner),
                }
            }
        }

        impl From<$core> for $name {
            fn from(value: $core) -> Self {
                Self {
                    inner: std::rc::Rc::new(value),
                }
            }
        }
    };
}

/// Give a compute class (a struct whose `inner` is the core compute) its static
/// `spec()`, its `tunables()` and its [`Bound`] impl.
///
/// ```ignore
/// described!(Msd => mc::Msd, Vacf => mc::Vacf);
/// ```
macro_rules! described {
    ($($name:ident => $core:ty),+ $(,)?) => {$(
        #[wasm_bindgen::prelude::wasm_bindgen]
        impl $name {
            /// The molrs descriptor of this compute: its parameters with their
            /// defaults and the fields `compute` takes.
            pub fn spec() -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsError> {
                $crate::compute::wire::to_js(&<Self as $crate::compute::handle::Bound>::spec())
            }

            /// `[name, value]` for every parameter that has a default, as this
            /// instance holds it.
            pub fn tunables(&self) -> Result<wasm_bindgen::JsValue, wasm_bindgen::JsError> {
                $crate::compute::wire::to_js(
                    &<Self as $crate::compute::handle::Bound>::tunables(self),
                )
            }
        }

        impl $crate::compute::handle::Bound for $name {
            fn spec() -> molrs::compute::ComputeSpec {
                <$core as molrs::compute::Described>::spec()
            }

            fn tunables(&self) -> Vec<(&'static str, molrs::compute::ParamValue)> {
                molrs::compute::Described::tunables(&self.inner)
            }
        }
    )+};
}

pub(crate) use {described, output};
