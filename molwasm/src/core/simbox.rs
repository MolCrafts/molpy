//! WASM bindings for the simulation box ([`Box`]).
//!
//! The simulation box (exported as `Box` in JavaScript) defines the
//! parallelepiped that contains the molecular system and controls
//! periodic boundary conditions (PBC). It supports:
//!
//! - Cubic, orthorhombic, and triclinic (general parallelepiped) geometries
//! - Independent PBC flags per axis (x, y, z)
//! - Cartesian <--> fractional coordinate conversion
//! - Minimum-image displacement vectors
//! - Coordinate wrapping into the primary image
//!
//! # Box matrix convention
//!
//! The box is defined by a 3x3 upper-triangular matrix `h` (row-major)
//! where the columns are the three box edge vectors:
//!
//! ```text
//! h = | lx  xy  xz |
//!     |  0  ly  yz |
//!     |  0   0  lz |
//! ```
//!
//! All lengths are in angstrom (A).

use crate::core::nd_array::NDArray;
use molrs::core::{Cell as RsCell, SimBox};
use molrs::op::F;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

/// The geometry a [`Box`] (or a `Parallelepiped`) is built from — the WASM
/// face of `molrs::core::Cell`, the one argument of `new Box(cell, …)` that
/// says what the cell is.
///
/// One static constructor per variant of the core enum, each a different
/// description of a cell: `Cell.matrix(h)`, `Cell.lengths(l)`,
/// `Cell.lengthsAngles(l, angles)`, `Cell.lengthsTilts(l, tilts)` and
/// `Cell.undefined()`. A `Cell` is a value: passing it to a constructor does
/// not consume it.
///
/// # Example (JavaScript)
///
/// ```js
/// const cube    = new Box(Cell.lengths([10, 10, 10]), [0, 0, 0], [true, true, true]);
/// const tric    = new Box(Cell.lengthsTilts([10, 10, 10], [2, 0, 0]), [0, 0, 0], [true, true, true]);
/// const crystal = new Box(Cell.lengthsAngles([5, 5, 7], [90, 90, 120]), [0, 0, 0], [true, true, true]);
/// const none    = new Box(Cell.undefined(), [0, 0, 0], [false, false, false]);
/// ```
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub struct Cell {
    pub(crate) inner: RsCell,
}

#[wasm_bindgen]
impl Cell {
    /// The cell matrix `h`: 9 values, row-major `h[row][col]`, lattice
    /// vectors as **columns** (Å) — `Cell::Matrix`.
    ///
    /// # Errors
    ///
    /// Throws if `h` does not hold 9 values. A singular matrix is refused
    /// by the constructor that takes the cell.
    pub fn matrix(h: &[F]) -> Result<Cell, JsValue> {
        let h: [F; 9] = h.try_into().map_err(|_| {
            JsValue::from_str(&format!(
                "Cell.matrix: h must be 9 values (row-major 3x3), got {}",
                h.len()
            ))
        })?;
        Ok(Cell {
            inner: RsCell::Matrix([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], h[8]]]),
        })
    }

    /// An orthorhombic cell with edge lengths `[a, b, c]` (Å) —
    /// `Cell::Lengths`. A cube is `Cell.lengths([a, a, a])`.
    ///
    /// # Errors
    ///
    /// Throws if `lengths` does not hold 3 values. A non-positive edge is
    /// refused by the constructor that takes the cell.
    pub fn lengths(lengths: &[F]) -> Result<Cell, JsValue> {
        Ok(Cell {
            inner: RsCell::Lengths(triple("Cell.lengths", "lengths", lengths)?),
        })
    }

    /// A restricted-triclinic cell from edge lengths `[a, b, c]` (Å) and the
    /// angles `[alpha, beta, gamma]` between them, in degrees —
    /// `Cell::LengthsAngles`.
    ///
    /// # Errors
    ///
    /// Throws if either array does not hold 3 values.
    #[wasm_bindgen(js_name = lengthsAngles)]
    pub fn lengths_angles(lengths: &[F], angles: &[F]) -> Result<Cell, JsValue> {
        Ok(Cell {
            inner: RsCell::LengthsAngles {
                lengths: triple("Cell.lengthsAngles", "lengths", lengths)?,
                angles: triple("Cell.lengthsAngles", "angles", angles)?,
            },
        })
    }

    /// A restricted-triclinic cell from the diagonal sizes `[lx, ly, lz]` (Å)
    /// and the tilts `[xy, xz, yz]` (Å), the LAMMPS form —
    /// `Cell::LengthsTilts`.
    ///
    /// # Errors
    ///
    /// Throws if either array does not hold 3 values.
    #[wasm_bindgen(js_name = lengthsTilts)]
    pub fn lengths_tilts(lengths: &[F], tilts: &[F]) -> Result<Cell, JsValue> {
        Ok(Cell {
            inner: RsCell::LengthsTilts {
                lengths: triple("Cell.lengthsTilts", "lengths", lengths)?,
                tilts: triple("Cell.lengthsTilts", "tilts", tilts)?,
            },
        })
    }

    /// No geometric cell: a box of undefined, zero volume — `Cell::Undefined`.
    /// Distinct from periodicity: a defined non-periodic box has a cell.
    #[wasm_bindgen(js_name = undefined)]
    pub fn undefined_cell() -> Cell {
        Cell {
            inner: RsCell::Undefined,
        }
    }
}

#[wasm_bindgen(typescript_custom_section)]
const PBC_TYPE: &'static str = r#"
/** Per-axis periodicity `[x, y, z]`, as `new Box` and `Box.bounding` take it. */
export type Pbc = [boolean, boolean, boolean];

/** What `Box.wrap` returns: the wrapped points and the image shift applied to each. */
export interface Wrapped {
    /** Wrapped coordinates, shape `[N, 3]` (Å). */
    xyz: NDArray;
    /** Integer image shift `m` per point, flat `[N * 3]`: `xyz = input - H·m`. */
    shifts: Int32Array;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// `[boolean, boolean, boolean]`: see the `Pbc` TypeScript type.
    #[wasm_bindgen(typescript_type = "Pbc")]
    pub type JsPbc;

    /// `{ xyz, shifts }`: see the `Wrapped` TypeScript type.
    #[wasm_bindgen(typescript_type = "Wrapped")]
    pub type JsWrapped;
}

/// Exactly three values of `v`, named in the error.
pub(crate) fn triple(op: &str, name: &str, v: &[F]) -> Result<[F; 3], JsValue> {
    v.try_into()
        .map_err(|_| JsValue::from_str(&format!("{op}: {name} must be 3 values, got {}", v.len())))
}

/// Three flags as the `Pbc` array `new Box` takes: `[boolean, boolean, boolean]`.
pub(crate) fn js_pbc(flags: [bool; 3]) -> JsPbc {
    flags
        .iter()
        .map(|&f| JsValue::from_bool(f))
        .collect::<js_sys::Array>()
        .unchecked_into()
}

/// The `[x, y, z]` periodicity flags of a `Pbc` array: three booleans, no
/// coercion.
pub(crate) fn pbc3(op: &str, pbc: &JsPbc) -> Result<[bool; 3], JsValue> {
    let refuse = || JsValue::from_str(&format!("{op}: pbc must be [boolean, boolean, boolean]"));
    let array = pbc.dyn_ref::<js_sys::Array>().ok_or_else(refuse)?;
    if array.length() != 3 {
        return Err(refuse());
    }
    let flag = |i: u32| array.get(i).as_bool().ok_or_else(refuse);
    Ok([flag(0)?, flag(1)?, flag(2)?])
}

/// The `[N, 3]` view of `coords`, `name` in the error.
fn rows<'a>(name: &str, coords: &'a NDArray) -> Result<ndarray::ArrayView2<'a, F>, JsValue> {
    let shape = coords.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(JsValue::from_str(&format!("{name} must have shape [N, 3]")));
    }
    coords
        .as_array2(shape[0], 3)
        .map_err(|e| JsValue::from_str(&e))
}

/// Simulation box defining periodic boundary conditions and coordinate
/// transformations — the WASM face of `molrs::core::SimBox`.
///
/// A parallelepiped (its [`Cell`]) placed at an origin, periodic
/// independently along x, y and z. One constructor, `new Box(cell, origin,
/// pbc)`, as `SimBox::new`; `Box.bounding(points, padding, pbc)` is the box
/// around a point cloud, as `SimBox::bounding`.
///
/// Exported as `Box` in JavaScript.
///
/// # Example (JavaScript)
///
/// ```js
/// const box = new Box(Cell.lengths([10, 10, 10]), [0, 0, 0], [true, true, true]);
/// console.log(box.volume()); // 1000.0
/// console.log(box.lengths().toCopy()); // [10, 10, 10]
/// ```
#[wasm_bindgen]
pub struct Box {
    pub(crate) inner: SimBox,
}

#[wasm_bindgen]
impl Box {
    /// A box from its `cell`, its `origin` `[x, y, z]` (Å) and per-axis
    /// periodicity `pbc` — `SimBox::new`.
    ///
    /// # Errors
    ///
    /// Throws if `origin` does not hold 3 values or `pbc` is not three
    /// booleans; for a singular `Cell.matrix`, a non-positive or non-finite
    /// edge, or lengths/angles that describe no cell.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const h = new Float64Array([10, 2, 0, 0, 10, 0, 0, 0, 10]);
    /// const box = new Box(Cell.matrix(h), [0, 0, 0], [true, true, true]);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(cell: &Cell, origin: &[F], pbc: &JsPbc) -> Result<Box, JsValue> {
        let origin = triple("new Box", "origin", origin)?;
        let pbc = pbc3("new Box", pbc)?;
        SimBox::new(cell.inner, origin, pbc)
            .map(|inner| Box { inner })
            .map_err(|e| JsValue::from_str(&format!("new Box: {e}")))
    }

    /// The orthorhombic box around `points` (`NDArray` of shape `[N, 3]`, Å),
    /// each side padded by `padding[d]` (Å, ≥ 0) — `SimBox::bounding`.
    ///
    /// # Errors
    ///
    /// Throws if `points` is not `[N, 3]` or is empty, if `padding` is not 3
    /// finite non-negative values, if an axis has zero extent (every point
    /// shares that coordinate and its padding is 0), or if `pbc` is not three
    /// booleans.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const box = Box.bounding(NDArray.from(xyz, [n, 3]), [2, 2, 2], [false, false, false]);
    /// ```
    pub fn bounding(points: &NDArray, padding: &[F], pbc: &JsPbc) -> Result<Box, JsValue> {
        let points = rows("points", points)?;
        let padding = triple("Box.bounding", "padding", padding)?;
        let pbc = pbc3("Box.bounding", pbc)?;
        SimBox::bounding(points, padding, pbc)
            .map(|inner| Box { inner })
            .map_err(|e| JsValue::from_str(&format!("Box.bounding: {e}")))
    }

    /// Return the box volume in cubic angstrom (A^3).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(box.volume()); // e.g., 1000.0
    /// ```
    pub fn volume(&self) -> F {
        self.inner.volume()
    }

    /// An independent copy of this box.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> Box {
        Box {
            inner: self.inner.clone(),
        }
    }

    /// Return the box origin as a `NDArray` with shape `[3]`.
    ///
    /// The origin is the lower-left corner of the box in angstrom (A).
    ///
    /// # Returns
    ///
    /// `NDArray` containing `[ox, oy, oz]` in angstrom.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const o = box.origin().toCopy(); // Float64Array [0, 0, 0]
    /// ```
    pub fn origin(&self) -> NDArray {
        NDArray::from_vec(self.inner.origin().to_vec(), std::boxed::Box::new([3]))
    }

    /// The box's 3×3 cell matrix `h` as a `NDArray` of shape `[9]`,
    /// **row-major** `h[row][col]` with the lattice vectors as columns — the
    /// layout `Cell.matrix` takes, so `Cell.matrix(box.h().toCopy())` rebuilds
    /// the cell. A consumer that wants the vectors contiguous (a column-major
    /// `Float64Array`) transposes on its side.
    ///
    /// # Returns
    ///
    /// `NDArray` of length 9: `[h00, h01, h02, h10, h11, h12, h20, h21, h22]`,
    /// in angstrom (A).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const h = box.h().toCopy(); // Float64Array, length 9, row-major
    /// const a = [h[0], h[3], h[6]]; // the first lattice vector is column 0
    /// ```
    pub fn h(&self) -> NDArray {
        let flat: Vec<F> = self.inner.matrix().iter().flatten().copied().collect();
        NDArray::from_vec(flat, std::boxed::Box::new([9]))
    }

    /// Return the box edge lengths as a `NDArray` with shape `[3]`.
    ///
    /// For orthorhombic boxes these are `[lx, ly, lz]`. For triclinic
    /// boxes these are the lengths of the three cell vectors.
    ///
    /// # Returns
    ///
    /// `NDArray` containing `[lx, ly, lz]` in angstrom (A).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const L = box.lengths().toCopy(); // Float64Array [10, 10, 10]
    /// ```
    /// The per-axis periodic boundary flags, `[x, y, z]` booleans: the `Pbc`
    /// `new Box` takes, so a box rebuilds from its own `pbc()`.
    ///
    /// Renderers that respect PBC (isosurface marching cubes, bond
    /// minimum-image, atom wrap-around) read this to decide whether to wrap
    /// coordinates across the cell.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const pbc = box.pbc(); // [true, true, true] for fully periodic
    /// const fullyPeriodic = pbc.every(Boolean);
    /// ```
    #[wasm_bindgen(js_name = pbc)]
    pub fn pbc(&self) -> JsPbc {
        js_pbc(self.inner.pbc())
    }

    pub fn lengths(&self) -> NDArray {
        let l = self.inner.lengths();
        NDArray::from_vec(l.to_vec(), std::boxed::Box::new([3]))
    }

    /// Return the box tilt factors as a `NDArray` with shape `[3]`.
    ///
    /// Tilt factors `[xy, xz, yz]` define the off-diagonal elements
    /// of the cell matrix (the LAMMPS standard). For orthorhombic boxes
    /// all tilts are zero.
    ///
    /// # Returns
    ///
    /// `NDArray` containing `[xy, xz, yz]` (dimensionless ratios
    /// multiplied by the corresponding box length, so effectively in A).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const t = box.tilts().toCopy(); // Float64Array [0, 0, 0]
    /// ```
    pub fn tilts(&self) -> NDArray {
        let t = self.inner.tilts();
        NDArray::from_vec(t.to_vec(), std::boxed::Box::new([3]))
    }

    /// Fractional coordinates of Cartesian `coords` (`NDArray` `[N, 3]`, Å):
    /// `H⁻¹ (r − origin)`, in `[0, 1)` inside the primary image.
    ///
    /// One door per transform, returning the result: to keep it as a column,
    /// `block.set(key, box.toFrac(xyz).toCopy(), [n, 3])`.
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frac = box.toFrac(NDArray.from(coords, [1, 3]));
    /// console.log(frac.toCopy()); // [0.5, 0.5, 0.5] for a 10x10x10 box
    /// ```
    #[wasm_bindgen(js_name = toFrac)]
    pub fn to_frac(&self, coords: &NDArray) -> Result<NDArray, JsValue> {
        Ok(NDArray::from_array2(
            self.inner.to_frac(rows("coords", coords)?),
        ))
    }

    /// Cartesian coordinates (Å) of fractional `coords` (`NDArray` `[N, 3]`):
    /// `origin + H f`.
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    #[wasm_bindgen(js_name = toCart)]
    pub fn to_cart(&self, coords: &NDArray) -> Result<NDArray, JsValue> {
        Ok(NDArray::from_array2(
            self.inner.to_cart(rows("coords", coords)?),
        ))
    }

    /// Wrap Cartesian `coords` (`NDArray` `[N, 3]`, Å) into the primary image
    /// along the periodic axes — `SimBox::wrap`.
    ///
    /// Returns `{ xyz, shifts }`: the wrapped points, and the integer image
    /// shift `m` actually applied to each (`xyz = coords − H·m`, flat
    /// `Int32Array` of `3N`), which is what an `ix`/`iy`/`iz` image column
    /// accumulates.
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const { xyz, shifts } = box.wrap(NDArray.from(positions, [n, 3]));
    /// ```
    pub fn wrap(&self, coords: &NDArray) -> Result<JsWrapped, JsValue> {
        let (xyz, shifts) = self.inner.wrap(rows("coords", coords)?);
        let (shifts, _offset) = shifts.into_raw_vec_and_offset();
        let out = js_sys::Object::new();
        js_sys::Reflect::set(
            &out,
            &JsValue::from_str("xyz"),
            &JsValue::from(NDArray::from_array2(xyz)),
        )?;
        js_sys::Reflect::set(
            &out,
            &JsValue::from_str("shifts"),
            &js_sys::Int32Array::from(shifts.as_slice()).into(),
        )?;
        Ok(out.unchecked_into())
    }

    /// Displacements `b − a` row by row (`NDArray`s `[N, 3]`, Å), under the
    /// minimum-image convention along the periodic axes when `minimumImage`
    /// — `SimBox::delta`.
    ///
    /// # Errors
    ///
    /// Throws if `a` or `b` does not have shape `[N, 3]`, or if their row
    /// counts differ.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const d = box.delta(a, b, true); // minimum-image displacement
    /// ```
    pub fn delta(&self, a: &NDArray, b: &NDArray, minimum_image: bool) -> Result<NDArray, JsValue> {
        let a = rows("a", a)?;
        let b = rows("b", b)?;
        if a.nrows() != b.nrows() {
            return Err(JsValue::from_str(
                "a and b must have the same number of rows",
            ));
        }
        Ok(NDArray::from_array2(self.inner.delta(a, b, minimum_image)))
    }

    /// Return the 8 corner vertices of the parallelepiped.
    ///
    /// # Returns
    ///
    /// `NDArray` with shape `[8, 3]` containing the corner
    /// coordinates in angstrom (A). The flat array has 24 elements.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const corners = box.corners();
    /// console.log(corners.len()); // 24 (8 corners x 3 coords)
    /// ```
    pub fn corners(&self) -> NDArray {
        let corners = self.inner.corners();
        NDArray::from_array2(corners)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::{Box as WasmBox, Cell, JsPbc};
    use crate::NDArray;
    use crate::core::nd_array::JsFloatArray;
    use molrs::op::F;
    use wasm_bindgen::{JsCast, JsValue};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn float_array(values: &[F]) -> JsFloatArray {
        JsFloatArray::from(values)
    }

    pub(crate) fn pbc(flags: [bool; 3]) -> JsPbc {
        super::js_pbc(flags)
    }

    fn assert_eq_array(actual: &JsFloatArray, expected: &[F]) {
        assert_eq!(actual.length() as usize, expected.len());
        for (i, value) in expected.iter().enumerate() {
            let got = actual.get_index(i as u32);
            assert!((got - value).abs() < 1.0e-5, "index {i}: {got} != {value}");
        }
    }

    fn rows(values: &[F]) -> NDArray {
        NDArray::from_js(&float_array(values), Some(Box::new([values.len() / 3, 3]))).expect("rows")
    }

    #[wasm_bindgen_test]
    fn box_coordinate_ops() {
        let view = rows(&[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
        let h = [2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        let sim_box =
            WasmBox::new(&Cell::matrix(&h).unwrap(), &[0.0; 3], &pbc([true; 3])).expect("box new");

        let frac = sim_box.to_frac(&view).expect("to_frac failed");
        assert_eq_array(&frac.to_copy(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let cart = sim_box.to_cart(&frac).expect("to_cart failed");
        assert_eq_array(&cart.to_copy(), &[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);

        let wrapped = sim_box.wrap(&rows(&[2.5, 3.5, 4.5])).expect("wrap failed");
        let xyz: NDArray = wasm_bindgen::convert::TryFromJsValue::try_from_js_value(
            js_sys::Reflect::get(&wrapped, &"xyz".into()).unwrap(),
        )
        .expect("xyz is an NDArray");
        assert_eq_array(&xyz.to_copy(), &[0.5, 0.5, 0.5]);
        let shifts: js_sys::Int32Array = js_sys::Reflect::get(&wrapped, &"shifts".into())
            .unwrap()
            .unchecked_into();
        assert_eq!(shifts.to_vec(), vec![1, 1, 1]);

        assert_eq!(sim_box.corners().len(), 24);

        let cube = WasmBox::new(
            &Cell::lengths(&[10.0; 3]).unwrap(),
            &[0.0; 3],
            &pbc([true; 3]),
        )
        .expect("cube");
        assert!((cube.volume() - 1000.0).abs() < 1.0e-5);
        assert_eq_array(&cube.lengths().to_copy(), &[10.0, 10.0, 10.0]);
        assert_eq_array(&cube.tilts().to_copy(), &[0.0, 0.0, 0.0]);

        let a = rows(&[1.0, 1.0, 1.0]);
        let b = rows(&[9.0, 9.0, 9.0]);
        let delta_mi = cube.delta(&a, &b, true).expect("delta mi").to_copy();
        let delta_no = cube.delta(&a, &b, false).expect("delta no").to_copy();
        assert!(delta_mi.get_index(0).abs() < delta_no.get_index(0).abs());
    }

    /// `h()` is the layout `Cell.matrix` takes and `pbc()` is what `new Box`
    /// takes: a box rebuilds from its own accessors, triclinic included.
    #[wasm_bindgen_test]
    fn a_box_rebuilds_from_its_own_accessors() {
        // Row-major: `b` tilted along x.
        let h = [10.0, 2.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
        let sim_box = WasmBox::new(
            &Cell::matrix(&h).unwrap(),
            &[0.0; 3],
            &pbc([true, true, false]),
        )
        .unwrap();
        let flat = sim_box.h().to_copy();
        assert_eq_array(&flat, &h);
        let flags: JsPbc = JsValue::from(sim_box.pbc()).unchecked_into();
        let rebuilt = WasmBox::new(&Cell::matrix(&flat.to_vec()).unwrap(), &[0.0; 3], &flags)
            .expect("a box rebuilds from h() and pbc()");
        assert_eq_array(&rebuilt.h().to_copy(), &h);
        let flags: js_sys::Array = JsValue::from(sim_box.pbc()).unchecked_into();
        assert_eq!(flags.get(2).as_bool(), Some(false));
    }

    #[wasm_bindgen_test]
    fn every_cell_variant_builds_and_bad_input_throws() {
        let o = [0.0; 3];
        let p = pbc([true; 3]);
        let tilted = WasmBox::new(
            &Cell::lengths_tilts(&[10.0; 3], &[2.0, 0.0, 0.0]).unwrap(),
            &o,
            &p,
        )
        .unwrap();
        assert_eq_array(&tilted.tilts().to_copy(), &[2.0, 0.0, 0.0]);
        let hex = WasmBox::new(
            &Cell::lengths_angles(&[5.0, 5.0, 7.0], &[90.0, 90.0, 120.0]).unwrap(),
            &o,
            &p,
        )
        .unwrap();
        assert!(hex.volume() > 0.0);
        let none = WasmBox::new(&Cell::undefined_cell(), &o, &pbc([false; 3])).unwrap();
        assert_eq!(none.volume(), 0.0);
        assert!(Cell::lengths(&[1.0, 2.0]).is_err());
        assert!(Cell::matrix(&[1.0; 8]).is_err());
        assert!(WasmBox::new(&Cell::lengths(&[-1.0, 1.0, 1.0]).unwrap(), &o, &p).is_err());
        let not_flags: JsPbc = JsValue::from(js_sys::Array::of3(
            &JsValue::from_f64(1.0),
            &JsValue::TRUE,
            &JsValue::TRUE,
        ))
        .unchecked_into();
        assert!(WasmBox::new(&Cell::lengths(&[1.0; 3]).unwrap(), &o, &not_flags).is_err());
    }

    #[wasm_bindgen_test]
    fn bounding_pads_the_point_cloud_and_refuses_zero_extent() {
        let points = rows(&[0.0, 0.0, 0.0, 2.0, 1.0, 3.0]);
        let bx = WasmBox::bounding(&points, &[1.0; 3], &pbc([false; 3])).unwrap();
        assert_eq_array(&bx.origin().to_copy(), &[-1.0, -1.0, -1.0]);
        assert_eq_array(&bx.lengths().to_copy(), &[4.0, 3.0, 5.0]);
        let flat = rows(&[0.0, 0.0, 0.0, 2.0, 0.0, 3.0]);
        assert!(WasmBox::bounding(&flat, &[0.0; 3], &pbc([false; 3])).is_err());
    }
}
