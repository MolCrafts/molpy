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

use crate::core::block::Block;
use crate::core::nd_array::{JsFloatArray, NDArray};
use molrs::core::SimBox;
use molrs::op::F;
use wasm_bindgen::prelude::*;

/// Simulation box defining periodic boundary conditions and coordinate
/// transformations.
///
/// Represents a parallelepiped defined by a 3x3 matrix `h` and an
/// origin point. Supports periodic boundary conditions (PBC)
/// independently in x, y, z directions.
///
/// Exported as `Box` in JavaScript.
///
/// # Example (JavaScript)
///
/// ```js
/// const h = floatArrayH;
/// const origin = floatArrayOrigin;
/// const box = new Box(h, origin, true, true, true);
/// console.log(box.volume()); // 1000.0
/// console.log(box.lengths().toCopy()); // [10, 10, 10]
/// ```
#[wasm_bindgen]
pub struct Box {
    pub(crate) inner: SimBox,
}

#[inline]
fn vec3_from_slice(v: &[F]) -> ndarray::Array1<F> {
    ndarray::arr1(&[v[0], v[1], v[2]])
}

#[inline]
fn mat3_from_slice(v: &[F]) -> ndarray::Array2<F> {
    ndarray::arr2(&[[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]])
}

#[wasm_bindgen]
impl Box {
    /// Create a new box from a 3x3 cell matrix and origin.
    ///
    /// # Arguments
    ///
    /// * `h` - 3x3 cell matrix as a float typed array with 9 elements in
    ///   row-major order: `[h00, h01, h02, h10, h11, h12, h20, h21, h22]`.
    ///   All values in angstrom (A).
    /// * `origin` - 3D origin vector as a float typed array with 3 elements
    ///   `[x, y, z]` in angstrom.
    /// * `pbc_x` - Enable periodic boundary in x direction
    /// * `pbc_y` - Enable periodic boundary in y direction
    /// * `pbc_z` - Enable periodic boundary in z direction
    ///
    /// # Returns
    ///
    /// A new `Box` instance.
    ///
    /// # Errors
    ///
    /// Throws if `h` does not have 9 elements or `origin` does not have
    /// 3 elements, or if the matrix is singular.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// // Triclinic box
    /// const h = new Float64Array([10, 2, 0, 0, 10, 0, 0, 0, 10]);
    /// const origin = new Float64Array([0, 0, 0]);
    /// const box = new Box(h, origin, true, true, true);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        h: &JsFloatArray,
        origin: &JsFloatArray,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if h.length() != 9 {
            return Err(JsValue::from_str("h must have 9 elements (3x3 matrix)"));
        }
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }

        let h_vec = h.to_vec();
        let origin_vec = origin.to_vec();

        let h_mat = mat3_from_slice(&h_vec);
        let origin_arr = vec3_from_slice(&origin_vec);
        let inner = SimBox::new(h_mat, origin_arr, [pbc_x, pbc_y, pbc_z])
            .map_err(|e| JsValue::from_str(&format!("box new error: {:?}", e)))?;
        Ok(Box { inner })
    }

    /// Create a cubic box with equal side lengths.
    ///
    /// # Arguments
    ///
    /// * `a` - Side length of the cube in angstrom (A)
    /// * `origin` - 3D origin vector as a float typed array with 3 elements
    ///   `[x, y, z]` in angstrom
    /// * `pbc_x` - Enable periodic boundary in x direction
    /// * `pbc_y` - Enable periodic boundary in y direction
    /// * `pbc_z` - Enable periodic boundary in z direction
    ///
    /// # Returns
    ///
    /// A new cubic `Box` with side length `a`.
    ///
    /// # Errors
    ///
    /// Throws if `origin` does not have 3 elements.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const origin = originVec;
    /// const box = Box.cube(10.0, origin, true, true, true);
    /// console.log(box.volume()); // 1000.0
    /// ```
    pub fn cube(
        a: F,
        origin: &JsFloatArray,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }
        let origin_vec = origin.to_vec();
        let inner = SimBox::cube(a, vec3_from_slice(&origin_vec), [pbc_x, pbc_y, pbc_z])
            .map_err(|e| JsValue::from_str(&format!("box cube error: {:?}", e)))?;
        Ok(Box { inner })
    }

    /// Create an orthorhombic (rectangular) box with axis-aligned edges.
    ///
    /// # Arguments
    ///
    /// * `lengths` - Box dimensions as a float typed array with 3 elements
    ///   `[lx, ly, lz]` in angstrom (A)
    /// * `origin` - 3D origin vector as a float typed array with 3 elements
    ///   `[x, y, z]` in angstrom
    /// * `pbc_x` - Enable periodic boundary in x direction
    /// * `pbc_y` - Enable periodic boundary in y direction
    /// * `pbc_z` - Enable periodic boundary in z direction
    ///
    /// # Returns
    ///
    /// A new orthorhombic `Box`.
    ///
    /// # Errors
    ///
    /// Throws if `lengths` or `origin` does not have 3 elements.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const origin = originVec;
    /// const box = Box.ortho(lengthsVec, origin, true, true, true);
    /// console.log(box.volume()); // 6000.0
    /// ```
    pub fn ortho(
        lengths: &JsFloatArray,
        origin: &JsFloatArray,
        pbc_x: bool,
        pbc_y: bool,
        pbc_z: bool,
    ) -> Result<Box, JsValue> {
        if lengths.length() != 3 {
            return Err(JsValue::from_str("lengths must have 3 elements"));
        }
        if origin.length() != 3 {
            return Err(JsValue::from_str("origin must have 3 elements"));
        }
        let lengths_vec = lengths.to_vec();
        let origin_vec = origin.to_vec();
        let inner = SimBox::ortho(
            vec3_from_slice(&lengths_vec),
            vec3_from_slice(&origin_vec),
            [pbc_x, pbc_y, pbc_z],
        )
        .map_err(|e| JsValue::from_str(&format!("box ortho error: {:?}", e)))?;
        Ok(Box { inner })
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
        let o = self.inner.origin_view();
        NDArray::from_vec(o.to_vec(), std::boxed::Box::new([3]))
    }

    /// Return the box's 3×3 cell matrix `h` as a `NDArray` with shape
    /// `[9]`, **column-major** (column j of `h` is the j-th lattice
    /// vector). Suitable for direct consumption by `marchingCubes` and
    /// other code that expects the cell as a flat 3×3 column-major
    /// `Float64Array`.
    ///
    /// # Returns
    ///
    /// `NDArray` of length 9: `[h00, h10, h20, h01, h11, h21, h02, h12, h22]`,
    /// in angstrom (A).
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const cell = box.h().toCopy(); // Float64Array, length 9
    /// // col0 = a-vector = cell[0..3]
    /// ```
    pub fn h(&self) -> NDArray {
        let h = self.inner.h_view();
        // h_view is a 3×3 ndarray; emit it column-major to match the
        // marching-cubes consumer's expectation. Iterate (col, row) so
        // the flat output is [h[0,0], h[1,0], h[2,0], h[0,1], …].
        let mut flat = Vec::with_capacity(9);
        for j in 0..3 {
            for i in 0..3 {
                flat.push(h[[i, j]]);
            }
        }
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
    /// Return the per-axis periodic boundary flags as a `Uint8Array`
    /// of length 3 (`[px, py, pz]`, 1 = periodic, 0 = open).
    ///
    /// Renderers that respect PBC (isosurface marching cubes,
    /// bond minimum-image, atom wrap-around) need this to decide
    /// whether to wrap coordinates across the cell.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const pbc = box.pbc(); // Uint8Array [1, 1, 1] for fully periodic
    /// const fullyPeriodic = pbc[0] === 1 && pbc[1] === 1 && pbc[2] === 1;
    /// ```
    #[wasm_bindgen(js_name = pbc)]
    pub fn pbc(&self) -> js_sys::Uint8Array {
        let p = self.inner.pbc();
        let arr = js_sys::Uint8Array::new_with_length(3);
        arr.set_index(0, p[0] as u8);
        arr.set_index(1, p[1] as u8);
        arr.set_index(2, p[2] as u8);
        arr
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

    /// Convert Cartesian coordinates to fractional coordinates.
    ///
    /// Fractional coordinates are in the range [0, 1) for atoms
    /// inside the primary image of the box.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` containing
    ///   Cartesian coordinates in angstrom (A)
    ///
    /// # Returns
    ///
    /// `NDArray` with shape `[N, 3]` containing fractional coordinates
    /// (dimensionless).
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const cart = NDArray.from(coords, [1, 3]);
    /// const frac = box.toFrac(cart);
    /// console.log(frac.toCopy()); // [0.5, 0.5, 0.5] for a 10x10x10 box
    /// ```
    #[wasm_bindgen(js_name = toFrac)]
    pub fn to_frac(&self, coords: &NDArray) -> Result<NDArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.to_frac(coords_arr);
        Ok(NDArray::from_array2(result))
    }

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` containing
    ///   fractional coordinates (dimensionless)
    ///
    /// # Returns
    ///
    /// `NDArray` with shape `[N, 3]` containing Cartesian coordinates
    /// in angstrom (A).
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frac = NDArray.from(fracCoords, [1, 3]);
    /// const cart = box.toCart(frac);
    /// console.log(cart.toCopy()); // [5, 5, 5] for a 10x10x10 box
    /// ```
    #[wasm_bindgen(js_name = toCart)]
    pub fn to_cart(&self, coords: &NDArray) -> Result<NDArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.to_cart(coords_arr);
        Ok(NDArray::from_array2(result))
    }

    /// Wrap Cartesian coordinates into the primary image of the box.
    ///
    /// Atoms outside the box are translated back into the primary image
    /// using the periodic boundary conditions. Only axes with PBC
    /// enabled are wrapped.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` containing
    ///   Cartesian coordinates in angstrom (A)
    ///
    /// # Returns
    ///
    /// `NDArray` with shape `[N, 3]` containing wrapped coordinates
    /// in angstrom (A).
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const pos = NDArray.from(positions, [1, 3]);
    /// const wrapped = box.wrap(pos); // wraps into [0, lx) x [0, ly) x [0, lz)
    /// ```
    pub fn wrap(&self, coords: &NDArray) -> Result<NDArray, JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.wrap(coords_arr);
        Ok(NDArray::from_array2(result))
    }

    /// Wrap coordinates and write the result directly into a [`Block`] column.
    ///
    /// This is an allocation-efficient alternative to [`wrap`](Box::wrap)
    /// that avoids creating an intermediate `NDArray`.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` containing
    ///   Cartesian coordinates in angstrom (A)
    /// * `out_block` - Target [`Block`] to write the result into
    /// * `out_key` - Column name for the result (float, shape `[N, 3]`)
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]` or if the
    /// block write fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// box.wrapToBlock(coords, outBlock, "wrapped_pos");
    /// ```
    #[wasm_bindgen(js_name = wrapToBlock)]
    pub fn wrap_to_block(
        &self,
        coords: &NDArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.wrap(coords_arr);
        out_block.insert_array(out_key, result.into_dyn())
    }

    /// Calculate displacement vectors between two sets of coordinates.
    ///
    /// Computes `delta = b - a` for each pair of points. When
    /// `minimum_image` is `true`, the minimum-image convention is
    /// applied so that the displacement uses the shortest vector
    /// under periodic boundary conditions.
    ///
    /// # Arguments
    ///
    /// * `a` - `NDArray` with shape `[N, 3]` (reference positions in A)
    /// * `b` - `NDArray` with shape `[N, 3]` (target positions in A)
    /// * `minimum_image` - If `true`, apply minimum image convention
    ///   for PBC-enabled axes
    ///
    /// # Returns
    ///
    /// `NDArray` with shape `[N, 3]` containing displacement vectors
    /// `(b - a)` in angstrom (A).
    ///
    /// # Errors
    ///
    /// Throws if `a` or `b` does not have shape `[N, 3]`, or if the
    /// two arrays have different numbers of rows.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const a = NDArray.from(aCoords, [1, 3]);
    /// const b = NDArray.from(bCoords, [1, 3]);
    /// const d = box.delta(a, b, true); // minimum-image displacement
    /// ```
    pub fn delta(&self, a: &NDArray, b: &NDArray, minimum_image: bool) -> Result<NDArray, JsValue> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        if shape_a.len() != 2 || shape_a[1] != 3 {
            return Err(JsValue::from_str("a must have shape [N, 3]"));
        }
        if shape_b.len() != 2 || shape_b[1] != 3 {
            return Err(JsValue::from_str("b must have shape [N, 3]"));
        }
        if shape_a[0] != shape_b[0] {
            return Err(JsValue::from_str(
                "a and b must have the same number of atoms",
            ));
        }

        let n_atoms = shape_a[0];
        let a_arr = a.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let b_arr = b.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;

        let result = self.inner.delta(a_arr, b_arr, minimum_image);
        Ok(NDArray::from_array2(result))
    }

    /// Calculate displacement vectors and write the result directly into
    /// a [`Block`] column.
    ///
    /// This is an allocation-efficient alternative to [`delta`](Box::delta).
    ///
    /// # Arguments
    ///
    /// * `a` - `NDArray` with shape `[N, 3]` (reference positions in A)
    /// * `b` - `NDArray` with shape `[N, 3]` (target positions in A)
    /// * `minimum_image` - If `true`, apply minimum image convention
    /// * `out_block` - Target [`Block`] to write the result into
    /// * `out_key` - Column name for the result (float, shape `[N, 3]`)
    ///
    /// # Errors
    ///
    /// Throws if shapes are invalid or if the block write fails.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// box.deltaToBlock(a, b, true, outBlock, "displacements");
    /// ```
    #[wasm_bindgen(js_name = deltaToBlock)]
    pub fn delta_to_block(
        &self,
        a: &NDArray,
        b: &NDArray,
        minimum_image: bool,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape_a = a.shape();
        let shape_b = b.shape();
        if shape_a.len() != 2 || shape_a[1] != 3 {
            return Err(JsValue::from_str("a must have shape [N, 3]"));
        }
        if shape_b.len() != 2 || shape_b[1] != 3 {
            return Err(JsValue::from_str("b must have shape [N, 3]"));
        }
        if shape_a[0] != shape_b[0] {
            return Err(JsValue::from_str(
                "a and b must have the same number of atoms",
            ));
        }
        let n_atoms = shape_a[0];
        let a_arr = a.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let b_arr = b.as_array2(n_atoms, 3).map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.delta(a_arr, b_arr, minimum_image);
        out_block.insert_array(out_key, result.into_dyn())
    }

    /// Convert Cartesian to fractional coordinates and write the result
    /// directly into a [`Block`] column.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` (Cartesian, A)
    /// * `out_block` - Target [`Block`]
    /// * `out_key` - Column name for the result (float, shape `[N, 3]`)
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// box.toFracToBlock(cartCoords, outBlock, "frac_coords");
    /// ```
    #[wasm_bindgen(js_name = toFracToBlock)]
    pub fn to_frac_to_block(
        &self,
        coords: &NDArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.to_frac(coords_arr);
        out_block.insert_array(out_key, result.into_dyn())
    }

    /// Convert fractional to Cartesian coordinates and write the result
    /// directly into a [`Block`] column.
    ///
    /// # Arguments
    ///
    /// * `coords` - `NDArray` with shape `[N, 3]` (fractional, dimensionless)
    /// * `out_block` - Target [`Block`]
    /// * `out_key` - Column name for the result (float, shape `[N, 3]`)
    ///
    /// # Errors
    ///
    /// Throws if `coords` does not have shape `[N, 3]`.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// box.toCartToBlock(fracCoords, outBlock, "cart_coords");
    /// ```
    #[wasm_bindgen(js_name = toCartToBlock)]
    pub fn to_cart_to_block(
        &self,
        coords: &NDArray,
        out_block: &mut Block,
        out_key: &str,
    ) -> Result<(), JsValue> {
        let shape = coords.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(JsValue::from_str("coords must have shape [N, 3]"));
        }
        let n_atoms = shape[0];
        let coords_arr = coords
            .as_array2(n_atoms, 3)
            .map_err(|e| JsValue::from_str(&e))?;
        let result = self.inner.to_cart(coords_arr);
        out_block.insert_array(out_key, result.into_dyn())
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
mod tests {
    use super::Box as WasmBox;
    use crate::core::nd_array::JsFloatArray;
    use crate::{Frame, NDArray};
    use molrs::op::F;
    #[allow(unused_imports)]
    use wasm_bindgen::JsCast;
    use wasm_bindgen_test::wasm_bindgen_test;

    fn float_array(values: &[F]) -> JsFloatArray {
        JsFloatArray::from(values)
    }

    fn assert_eq_array(actual: &JsFloatArray, expected: &[F]) {
        assert_eq!(actual.length() as usize, expected.len());
        for (i, value) in expected.iter().enumerate() {
            let got = actual.get_index(i as u32);
            let expected = *value;
            assert!(
                (got - expected).abs() < 1.0e-5,
                "index {}: {} != {}",
                i,
                got,
                expected
            );
        }
    }

    #[wasm_bindgen_test]
    fn box_coordinate_ops() {
        let mut view = NDArray::new(Box::new([2_usize, 3_usize]));
        let view_data = float_array(&[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);
        view.copy_from(&view_data).expect("copy_from failed");

        let h = float_array(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
        let origin = float_array(&[0.0, 0.0, 0.0]);
        let sim_box = WasmBox::new(&h, &origin, true, true, true).expect("box new");

        let frac = sim_box.to_frac(&view).expect("to_frac failed");
        let frac_js = frac.to_copy();
        assert_eq_array(&frac_js, &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let cart = sim_box.to_cart(&frac).expect("to_cart failed");
        let cart_js = cart.to_copy();
        assert_eq_array(&cart_js, &[0.0, 0.0, 0.0, 2.0, 3.0, 4.0]);

        let mut wrap_view = NDArray::new(Box::new([1_usize, 3_usize]));
        wrap_view
            .copy_from(&float_array(&[2.5, 3.5, 4.5]))
            .expect("write wrap_view");
        let wrapped = sim_box.wrap(&wrap_view).expect("wrap failed");
        let wrapped_js = wrapped.to_copy();
        assert_eq_array(&wrapped_js, &[0.5, 0.5, 0.5]);

        let corners = sim_box.corners();
        assert_eq!(corners.len(), 24);

        let cube = WasmBox::cube(10.0, &origin, true, true, true).expect("cube");
        assert!((cube.volume() - 1000.0).abs() < 1.0e-5);

        let lengths = cube.lengths().to_copy();
        assert_eq_array(&lengths, &[10.0, 10.0, 10.0]);

        let tilts = cube.tilts().to_copy();
        assert_eq_array(&tilts, &[0.0, 0.0, 0.0]);

        let ortho_lengths = float_array(&[3.0, 4.0, 5.0]);
        let ortho = WasmBox::ortho(&ortho_lengths, &origin, true, true, true).expect("ortho");
        let ortho_lengths_out = ortho.lengths().to_copy();
        assert_eq_array(&ortho_lengths_out, &[3.0, 4.0, 5.0]);

        let mut a = NDArray::new(Box::new([1_usize, 3_usize]));
        let mut b = NDArray::new(Box::new([1_usize, 3_usize]));
        a.copy_from(&float_array(&[1.0, 1.0, 1.0]))
            .expect("write a");
        b.copy_from(&float_array(&[9.0, 9.0, 9.0]))
            .expect("write b");

        let delta_mi = cube.delta(&a, &b, true).expect("delta mi").to_copy();
        let delta_no = cube.delta(&a, &b, false).expect("delta no").to_copy();
        assert!(delta_mi.get_index(0).abs() < delta_no.get_index(0).abs());

        let frame = Frame::new();
        let mut out = frame.create_block("out").expect("create out");
        cube.wrap_to_block(&wrap_view, &mut out, "wrapped")
            .expect("wrapToBlock");
        let wrapped_out: JsFloatArray =
            wasm_bindgen::JsValue::from(out.copy("wrapped", None).expect("copy")).unchecked_into();
        assert_eq_array(&wrapped_out, &[2.5, 3.5, 4.5]);
    }
}
