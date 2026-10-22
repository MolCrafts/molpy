//! Potentials of mean force and torque — the WASM face of
//! `molrs::compute::Pmft<C>`, one class per coordinate system.
//!
//! ```js
//! const pmft = new PmftXy(3.0, 3.0, 60, 60);
//! const result = pmft.compute({ pairs, orientations: angles }).toJSON();
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// PMFT in (r, θ₁, θ₂): `density`, `raw_counts`, `pmf`, bin `edges`.
    PmftR12Result => mc::PmftResult<mc::R12>
}

output! {
    /// PMFT in (x, y): `density`, `raw_counts`, `pmf`, bin `edges`.
    PmftXyResult => mc::PmftResult<mc::Xy>
}

output! {
    /// PMFT in (x, y, θ): `density`, `raw_counts`, `pmf`, bin `edges`.
    PmftXytResult => mc::PmftResult<mc::Xyt>
}

output! {
    /// PMFT in (x, y, z): `density`, `raw_counts`, `pmf`, bin `edges`.
    PmftXyzResult => mc::PmftResult<mc::Xyz>
}

/// One PMFT class: its constructor arguments, and how its orientations are
/// read (planar angles, or `N × 4` quaternions).
macro_rules! pmft {
    (
        $(#[$meta:meta])*
        $name:ident, $coordinate:ident, $result:ident,
        ($($arg:ident: $ty:ty),+),
        |$data:ident| $orientations:expr
    ) => {
        $(#[$meta])*
        #[wasm_bindgen]
        pub struct $name {
            inner: mc::Pmft<mc::$coordinate>,
        }

        #[wasm_bindgen]
        impl $name {
            #[wasm_bindgen(constructor)]
            pub fn new($($arg: $ty),+) -> Result<$name, JsError> {
                Ok(Self {
                    inner: mc::Pmft::<mc::$coordinate>::new($($arg),+)?,
                })
            }

            /// `{ pairs, orientations? }`.
            pub fn compute(&self, data: &JsValue) -> Result<$result, JsError> {
                let $data = Fields::new(data, stringify!($name))?;
                let pairs = $data.neighbors("pairs")?;
                let orientations = $orientations?;
                let input = mc::PmftInput {
                    pairs: &pairs,
                    orientations: orientations.as_ref().map(|o| o.view()),
                };
                Ok(self.inner.compute(input)?.into())
            }
        }
    };
}

pmft! {
    /// PMFT over (r, θ₁, θ₂); `orientations` are planar angles.
    PmftR12, R12, PmftR12Result,
    (r_max: F, n_r: usize, n_t1: usize, n_t2: usize),
    |data| data.opt_reals("orientations")
}

pmft! {
    /// PMFT over (x, y); `orientations` are planar angles.
    PmftXy, Xy, PmftXyResult,
    (x_max: F, y_max: F, n_x: usize, n_y: usize),
    |data| data.opt_reals("orientations")
}

pmft! {
    /// PMFT over (x, y, θ); `orientations` are planar angles.
    PmftXyt, Xyt, PmftXytResult,
    (x_max: F, y_max: F, n_x: usize, n_y: usize, n_t: usize),
    |data| data.opt_reals("orientations")
}

pmft! {
    /// PMFT over (x, y, z); `orientations` are `N × 4` quaternions.
    PmftXyz, Xyz, PmftXyzResult,
    (x_max: F, y_max: F, z_max: F, n_x: usize, n_y: usize, n_z: usize),
    |data| data.opt_matrix("orientations", Some(4))
}

described!(
    PmftR12 => mc::Pmft<mc::R12>,
    PmftXy => mc::Pmft<mc::Xy>,
    PmftXyt => mc::Pmft<mc::Xyt>,
    PmftXyz => mc::Pmft<mc::Xyz>,
);
