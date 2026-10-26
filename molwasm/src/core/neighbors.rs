//! Neighbor search — the WASM face of `molrs::core::NeighborList`.
//!
//! Two doors, one per question:
//!
//! - [`NeighborList`] — a **self** search over one point set. `build` /
//!   `update` index the coordinates and enumerate nothing; `neighbors()`
//!   materializes the half-shell pairs (`i < j`) into a [`Neighbors`] table.
//! - [`NeighborQuery`] — a **cross** search: index a reference frame once, then
//!   ask which of its atoms lie within the cutoff of another frame's atoms.
//!
//! Both produce the same [`Neighbors`] column table, which the analysis classes
//! (`Rdf`, `Cluster`, the order parameters, …) and `PotentialCompiler.compileOver`
//! consume.
//!
//! All distances are in angstrom (Å).

use std::rc::Rc;

use js_sys::Uint32Array;
use molrs::core::{
    Cell, NeighborColumns as RsNeighborColumns, NeighborList as RsNeighborList,
    NeighborQuery as RsNeighborQuery, Neighbors as RsNeighbors, QueryMode, SimBox,
};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;
use crate::core::frame_coords;
use crate::core::nd_array::JsFloatArray;

/// Neighbor search over one point set: index the coordinates, then read pairs.
///
/// This is the door to every self search. It owns the cutoff and the backend
/// that indexes space, and keeps the two halves of the job apart: `build` and
/// `update` place the atoms in space and enumerate **nothing**, while
/// `neighbors()` materializes the pairs into a [`Neighbors`] table.
///
/// The pairs are a **half-shell** self list: every unordered pair appears
/// exactly once, with `i < j`, and never as `i == j`. A directed search against
/// a *second* point set is a different question — see [`NeighborQuery`].
///
/// All distances are in angstrom (Å).
///
/// # Example (JavaScript)
///
/// ```js
/// const nl = new NeighborList(3.0);      // O(N) cell list, cutoff 3 Å
/// nl.build(frame);                       // index only — no pair table
/// const neigh = nl.neighbors();          // both columns (the default)
/// const lean  = nl.neighbors({ disp: false });   // indices + d² only
///
/// nl.update(movedFrame);                 // re-index in the box from `build`
/// ```
#[wasm_bindgen(js_name = NeighborList)]
pub struct NeighborList {
    inner: RsNeighborList,
}

#[wasm_bindgen(js_class = NeighborList)]
impl NeighborList {
    /// Create a search with the O(N) cell-list backend — the production choice.
    ///
    /// `cutoff` is the interaction radius in angstrom (Å). It is fixed here
    /// rather than passed per query because it sets the cell width of the
    /// index.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    #[wasm_bindgen(constructor)]
    pub fn new(cutoff: F) -> Result<NeighborList, JsValue> {
        Ok(NeighborList {
            inner: RsNeighborList::new(cutoff).map_err(cutoff_err)?,
        })
    }

    /// Create a search with the O(N²) all-pairs backend.
    ///
    /// Finds exactly the same pairs as the cell list — that is what makes it
    /// useful as a reference — at a cost that grows with the square of the
    /// particle count. Prefer it only for very small systems.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const nl = NeighborList.bruteForce(12.5);
    /// ```
    #[wasm_bindgen(js_name = bruteForce)]
    pub fn brute_force(cutoff: F) -> Result<NeighborList, JsValue> {
        Ok(NeighborList {
            inner: RsNeighborList::brute_force(cutoff).map_err(cutoff_err)?,
        })
    }

    /// The cutoff distance (Å) fixed at construction.
    #[wasm_bindgen(getter)]
    pub fn cutoff(&self) -> F {
        self.inner.cutoff()
    }

    /// Index a [`Frame`]'s atom positions — coordinates **and** box.
    ///
    /// Builds the spatial index and nothing else: no pairs are enumerated and
    /// no table is allocated. A frame without a `simbox` is treated as a free
    /// (non-periodic) system and gets a bounding box padded by the cutoff, so
    /// no pair can wrap around an edge.
    ///
    /// The box is retained, so a later [`update`](Self::update) can re-index
    /// new coordinates in the same box.
    ///
    /// # Errors
    ///
    /// Throws if the frame has no `atoms` block with `x` / `y` / `z` columns,
    /// or if a free-boundary box cannot be derived from the coordinates.
    pub fn build(&mut self, frame: &Frame) -> Result<(), JsValue> {
        let cutoff = self.inner.cutoff();
        frame.with_frame(|rs_frame| {
            let pos = frame_coords(rs_frame)?;
            let simbox;
            let bx_ref = match rs_frame.simbox.as_ref() {
                Some(sb) => sb,
                None => {
                    // As `NeighborQuery::unbounded`: the cutoff-padded
                    // bounding box, or for no points a cutoff cube at the
                    // origin (no bounds to pad; no pairs to find).
                    let free = [false; 3];
                    simbox = if pos.nrows() == 0 {
                        SimBox::new(Cell::Lengths([cutoff; 3]), [0.0; 3], free)
                    } else {
                        SimBox::bounding(pos.view(), [cutoff; 3], free)
                    }
                    .map_err(|e| JsValue::from_str(&format!("free-boundary box: {e}")))?;
                    &simbox
                }
            };
            self.inner.build(pos.view(), bx_ref);
            Ok(())
        })?;
        Ok(())
    }

    /// Re-index new coordinates in the box captured by the last
    /// [`build`](Self::build).
    ///
    /// The natural per-step call at fixed volume: the positions move, the box
    /// does not. Index-only, like `build`. There is no skin — this re-indexes
    /// every time rather than deciding for you that the previous index was
    /// still good enough. **If the box itself changed** (a barostat), call
    /// `build` again: `update` keeps the old box and would fold minimum images
    /// against a stale cell.
    ///
    /// # Errors
    ///
    /// Throws if no `build` has run yet — the box is then unknown, and guessing
    /// one silently changes every minimum-image distance. Throws too if the
    /// frame carries no readable positions.
    pub fn update(&mut self, frame: &Frame) -> Result<(), JsValue> {
        // Core panics on update-before-build; a panic aborts wasm, so check
        // the engine's own state and throw instead.
        if !self.inner.is_built() {
            return Err(JsValue::from_str(
                "NeighborList.update reuses the box of the previous build: \
                 call build(frame) first",
            ));
        }
        frame.with_frame(|rs_frame| {
            let pos = frame_coords(rs_frame)?;
            self.inner.update(pos.view());
            Ok(())
        })
    }

    /// Materialize the pairs into a [`Neighbors`] table.
    ///
    /// `columns` is an optional `{ distSq?: boolean, disp?: boolean }`. Both
    /// columns are kept by default, so no analysis is surprised by a missing
    /// one; pass `false` for a column this call site will not read. A dropped
    /// column cannot be added afterwards — materialize again instead.
    ///
    /// Keeping both columns names the *columns*, not the pair direction: a self
    /// search stays half-shell (`i < j`) either way.
    ///
    /// Row order is unspecified — the cell-list backend materializes in
    /// parallel and the work split decides the order.
    ///
    /// # Errors
    ///
    /// Throws if `columns` is neither nullish nor an object, or if one of its
    /// two fields is present but not a boolean.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const neigh = nl.neighbors();                  // distSq + disp
    /// const lean  = nl.neighbors({ disp: false });   // indices + d²
    /// ```
    pub fn neighbors(&self, columns: Option<NeighborColumnsOptions>) -> Result<Neighbors, JsValue> {
        let policy = match columns.as_deref() {
            // Omitted, `undefined` or `null`: keep every column — the safe
            // default, and the one an order parameter needs.
            None => RsNeighborColumns::FULL,
            Some(options) if options.is_undefined() || options.is_null() => RsNeighborColumns::FULL,
            Some(options) if !options.is_object() => {
                return Err(JsValue::from_str(
                    "neighbors(columns) expects an object like { distSq: true, disp: true }",
                ));
            }
            Some(options) => RsNeighborColumns {
                dist_sq: column_flag(options, "distSq")?,
                disp: column_flag(options, "disp")?,
            },
        };
        Ok(Neighbors {
            inner: Rc::new(self.inner.neighbors(policy)),
        })
    }
}

/// The core's refusal of a cutoff, as a thrown JS error.
fn cutoff_err(err: molrs::core::CutoffError) -> JsValue {
    JsValue::from_str(&err.to_string())
}

#[wasm_bindgen(typescript_custom_section)]
const NEIGHBORS_STORAGE_OPTIONS: &'static str = r#"
/**
 * Which optional columns `NeighborList.neighbors` keeps. Both default to
 * `true`: a column omitted here cannot be read back, and cannot be added
 * afterwards without materializing again.
 */
export interface NeighborColumnsOptions {
    /** Keep squared minimum-image distances (Å²), 8 B/pair. */
    distSq?: boolean;
    /** Keep minimum-image displacement vectors (Å), 24 B/pair. */
    disp?: boolean;
}
"#;

#[wasm_bindgen]
extern "C" {
    /// The `{ distSq?, disp? }` object accepted by
    /// [`NeighborList::neighbors`] — typed on the JS side rather than `any`,
    /// so a misspelt flag is a compile error for a TypeScript caller.
    #[wasm_bindgen(typescript_type = "NeighborColumnsOptions")]
    pub type NeighborColumnsOptions;
}

/// One boolean field of the column options object; absent means `true`.
///
/// A field that is present but not a boolean is an error rather than a
/// coercion: `{ disp: 0 }` silently dropping the displacement column is exactly
/// the failure this surface exists to prevent.
fn column_flag(columns: &JsValue, key: &str) -> Result<bool, JsValue> {
    let value = js_sys::Reflect::get(columns, &JsValue::from_str(key))?;
    if value.is_undefined() || value.is_null() {
        return Ok(true);
    }
    value
        .as_bool()
        .ok_or_else(|| JsValue::from_str(&format!("neighbors(columns): '{key}' must be a boolean")))
}

// ===========================================================================
// Neighbors — the materialized pair table
// ===========================================================================

/// Materialized neighbor pair table: every atom pair within the cutoff.
///
/// A column store — two index columns that are always present, plus whichever
/// physical columns the search was told to keep. Row `k` of every column
/// describes the same pair. Produced by [`NeighborList::neighbors`] and
/// consumed by the analysis classes (`Rdf`, `Cluster`, `Steinhardt`, …) and
/// `PotentialCompiler.compileOver`.
///
/// A **self** search is half-shell: each unordered pair appears exactly once,
/// with `i < j`. A cross search ([`NeighborQuery::query`]) is directed and has no
/// such rule.
///
/// # Properties
///
/// | Property | Type | Description |
/// |----------|------|-------------|
/// | `nPairs` | `number` | Number of pairs — the row count every column shares |
/// | `nPoints` | `number` | Number of reference points |
/// | `nQueryPoints` | `number` | Number of query points (= `nPoints` for a self search) |
/// | `isSelfQuery` | `boolean` | Whether both index columns address the same point set |
///
/// # Optional columns
///
/// `distSq()` and `disp()` return `undefined` when the search did not store
/// that column. That is deliberately different from an empty or zero-filled
/// array: a zero displacement is a physically meaningful value (two coincident
/// particles), so fabricating one would turn a missing column into a wrong
/// answer.
///
/// # Example (JavaScript)
///
/// ```js
/// const neigh = nl.neighbors();
/// console.log(neigh.nPairs);
///
/// const i  = neigh.queryPointIndices(); // Uint32Array
/// const j  = neigh.pointIndices();      // Uint32Array
/// const d2 = neigh.distSq();            // Float64Array (Å²) or undefined
/// const dr = neigh.disp();              // Float64Array (Å), 3 per pair
/// ```
#[wasm_bindgen(js_name = Neighbors)]
pub struct Neighbors {
    /// Crate-visible so the force-field optimizer and the analyses can read
    /// the table. Shared, so [`clone`](Self::duplicate) — the handle a
    /// compute's data argument borrows — copies no pairs.
    pub(crate) inner: Rc<RsNeighbors>,
}

#[wasm_bindgen(js_class = Neighbors)]
impl Neighbors {
    /// A second handle to the same table; no pair is copied.
    #[wasm_bindgen(js_name = clone)]
    pub fn duplicate(&self) -> Neighbors {
        Neighbors {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Number of neighbor pairs — the row count every column shares.
    #[wasm_bindgen(getter, js_name = nPairs)]
    pub fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference (target) points the search indexed.
    #[wasm_bindgen(getter, js_name = nPoints)]
    pub fn n_points(&self) -> usize {
        self.inner.n_points()
    }

    /// Number of query points; equal to `nPoints` for a self search.
    #[wasm_bindgen(getter, js_name = nQueryPoints)]
    pub fn n_query_points(&self) -> usize {
        self.inner.n_query_points()
    }

    /// Whether both index columns address the same point set (half-shell,
    /// `i < j`).
    #[wasm_bindgen(getter, js_name = isSelfQuery)]
    pub fn is_self_query(&self) -> bool {
        matches!(self.inner.mode(), QueryMode::SelfQuery { .. })
    }

    /// Zero-copy `Uint32Array` view of query point indices (`i`) over
    /// WASM memory. **Invalidated** on any WASM memory growth — copy
    /// in JS (`new Uint32Array(view)`) if it needs to outlive later calls.
    #[wasm_bindgen(js_name = queryPointIndices)]
    pub fn query_point_indices(&self) -> Uint32Array {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { Uint32Array::view(self.inner.query_point_indices()) }
    }

    /// Zero-copy `Uint32Array` view of reference point indices (`j`).
    /// Same invalidation caveat as [`queryPointIndices`](Self::query_point_indices).
    #[wasm_bindgen(js_name = pointIndices)]
    pub fn point_indices(&self) -> Uint32Array {
        // SAFETY: view borrows wasm memory; short-lived use only.
        unsafe { Uint32Array::view(self.inner.point_indices()) }
    }

    /// Squared minimum-image distances in Å², one per pair, or `undefined`
    /// when this table never stored the column.
    ///
    /// Zero-copy view; same invalidation caveat as
    /// [`queryPointIndices`](Self::query_point_indices). The square root is
    /// left to the caller — most consumers compare against a squared cutoff,
    /// and an accessor hiding a per-pair `sqrt` would make that cost invisible.
    #[wasm_bindgen(js_name = distSq)]
    pub fn dist_sq(&self) -> Option<JsFloatArray> {
        // SAFETY: view borrows wasm memory; short-lived use only.
        self.inner
            .dist_sq()
            .map(|column| unsafe { JsFloatArray::view(column) })
    }

    /// Minimum-image displacements `r_j - r_i` in Å, flattened as
    /// `[dx0, dy0, dz0, dx1, …]` — three values per pair, `3 * nPairs` long.
    /// `undefined` when this table never stored the column.
    ///
    /// The vector is **not** normalized: its length is the pair distance, and
    /// it points from `i` to `j`, so swapping the indices flips its sign. Both
    /// physical columns come from the same minimum-image evaluation, so
    /// `distSq[k]` is exactly the squared length of row `k`.
    ///
    /// Zero-copy view; same invalidation caveat as
    /// [`queryPointIndices`](Self::query_point_indices).
    ///
    /// # Errors
    ///
    /// Throws if the stored column is not contiguous, which would make a flat
    /// view silently misaligned rather than merely slow.
    pub fn disp(&self) -> Result<Option<JsFloatArray>, JsValue> {
        let Some(view) = self.inner.disp() else {
            return Ok(None);
        };
        let flat = view.as_slice().ok_or_else(|| {
            JsValue::from_str("disp column is not contiguous; cannot expose a flat view")
        })?;
        // SAFETY: view borrows wasm memory; short-lived use only.
        Ok(Some(unsafe { JsFloatArray::view(flat) }))
    }
}

// ===========================================================================
// NeighborQuery — the cross search
// ===========================================================================

/// Cross neighbor search: index a reference point set once, then ask which of
/// its points lie within the cutoff of **other** points.
///
/// This is the door for the cross question — the atoms of one frame against
/// the atoms of another. A search of one frame against itself belongs to
/// [`NeighborList`], which also re-indexes moved coordinates and lets a call
/// choose which columns a table keeps. Every table returned here carries both
/// physical columns.
///
/// All distances are in angstrom (Å).
///
/// # Example (JavaScript)
///
/// ```js
/// const nq = new NeighborQuery(refFrame, 3.0);   // index the reference atoms
/// const cross = nq.query(otherFrame);            // directed, no i < j rule
/// console.log(cross.nPairs, cross.isSelfQuery); // …, false
/// ```
#[wasm_bindgen(js_name = NeighborQuery)]
pub struct NeighborQuery {
    inner: RsNeighborQuery,
}

#[wasm_bindgen(js_class = NeighborQuery)]
impl NeighborQuery {
    /// Index the atom positions of `frame` as the reference set.
    ///
    /// A frame with a `simbox` is searched under its periodic boundaries; one
    /// without is a free (non-periodic) system whose bounding box is derived
    /// from the coordinates.
    ///
    /// # Errors
    ///
    /// Throws if `cutoff` is not a positive length, or if the frame has no
    /// `atoms` block with `x` / `y` / `z` columns.
    #[wasm_bindgen(constructor)]
    pub fn new(frame: &Frame, cutoff: F) -> Result<NeighborQuery, JsValue> {
        frame.with_frame(|rs_frame| {
            let pos = frame_coords(rs_frame)?;
            let inner = match rs_frame.simbox.as_ref() {
                Some(sb) => RsNeighborQuery::new(sb, pos.view(), cutoff),
                None => RsNeighborQuery::unbounded(pos.view(), cutoff),
            }
            .map_err(cutoff_err)?;
            Ok(NeighborQuery { inner })
        })
    }

    /// The cutoff distance (Å) fixed at construction.
    #[wasm_bindgen(getter)]
    pub fn cutoff(&self) -> F {
        self.inner.cutoff()
    }

    /// All pairs where `i` indexes `frame`'s atoms and `j` indexes the
    /// reference atoms.
    ///
    /// Directed and full-shell: every query point reports all of its reference
    /// neighbors, with no `i < j` rule, so the table is tagged
    /// `isSelfQuery === false`.
    ///
    /// # Errors
    ///
    /// Throws if `frame` carries no readable positions.
    pub fn query(&self, frame: &Frame) -> Result<Neighbors, JsValue> {
        frame.with_frame(|rs_frame| {
            let pos = frame_coords(rs_frame)?;
            Ok(Neighbors {
                inner: Rc::new(self.inner.query(pos.view())),
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use wasm_bindgen_test::*;

    fn frame_at(positions: &[[F; 3]]) -> Frame {
        let coords = Array2::from_shape_fn((positions.len(), 3), |(i, k)| positions[i][k]);
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.set_coords(coords.view()).unwrap();
        rs_frame.simbox =
            Some(SimBox::new(Cell::Lengths([20.0; 3]), [0.0; 3], [false; 3]).unwrap());
        Frame::from_rs(rs_frame).unwrap()
    }

    #[wasm_bindgen_test]
    fn neighbor_list_finds_half_shell_pairs() {
        let frame = frame_at(&[[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [8.0, 8.0, 8.0]]);
        let mut nl = NeighborList::new(2.0).unwrap();
        nl.build(&frame).unwrap();
        let pairs = nl.neighbors(None).unwrap();
        assert_eq!(pairs.n_pairs(), 1);
        assert!(pairs.is_self_query());
    }

    #[wasm_bindgen_test]
    fn neighbor_query_is_directed() {
        let reference = frame_at(&[[1.0, 1.0, 1.0], [8.0, 8.0, 8.0]]);
        let other = frame_at(&[[1.5, 1.0, 1.0], [8.5, 8.0, 8.0], [15.0, 15.0, 15.0]]);
        let cross = NeighborQuery::new(&reference, 2.0)
            .unwrap()
            .query(&other)
            .unwrap();
        assert_eq!(cross.n_pairs(), 2);
        assert_eq!(cross.n_query_points(), 3);
        assert_eq!(cross.n_points(), 2);
        assert!(!cross.is_self_query());
    }
}
