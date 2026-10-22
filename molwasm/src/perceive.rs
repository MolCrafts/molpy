//! Chemical perception for the WASM / JS surface — the face of
//! [`molrs::perceive`], with the core's two semantics:
//!
//! - **`assign*` mutate.** As `assign_*(&mut Atomistic)` in molrs, each writes
//!   the perceived fact onto the frame it is given, in place, and returns no
//!   frame: the perceived columns are added to (or overwrite) the frame's
//!   `atoms` / `bonds` blocks; every other column, block, the box and the meta
//!   are kept. Block handles taken from the frame before the call are
//!   invalidated — take them again (`frame.get("atoms")`).
//! - **`addHydrogens` / `removeHydrogens` return a new frame**, as their core
//!   functions return a new molecule: they change the graph itself (rows
//!   appear or disappear), so the input is left untouched.
//!
//! Every function reads species from `atoms.atomic_number` (`uint`) only. A
//! frame that spells symbols (`atoms.element`, e.g. from `readXyz`) is
//! not converted implicitly: call `fillAtomicNumbers(frame)` first, or the
//! function throws, naming the atom without a valid atomic number.
//!
//! # Example (JavaScript)
//!
//! ```js
//! const withH = addHydrogens(frame);   // a new frame
//! const heavy = removeHydrogens(withH); // a new frame
//! assignRings(frame);                   // frame's atoms/bonds gain is_in_ring, n_rings
//! ```

use wasm_bindgen::prelude::*;

use molrs::core::{Atomistic, MolRsError};

use crate::core::frame::Frame;

/// Run the in-place `assign` on the molecule `frame` holds and write what it
/// perceived back onto `frame`: every column of the molecule's blocks is
/// inserted into the frame's block of that name (overwriting a column of the
/// same name), and a block the frame lacks is added. The topology is
/// unchanged by an `assign`, so the rows correspond one to one.
fn assign_in_place<R>(
    frame: &Frame,
    op: &str,
    assign: impl FnOnce(&mut Atomistic) -> Result<R, MolRsError>,
) -> Result<R, JsValue> {
    let mut mol = frame_to_atomistic(frame)?;
    let result = assign(&mut mol).map_err(|e| JsValue::from_str(&format!("{op}: {e}")))?;
    let perceived = mol
        .to_frame()
        .map_err(|e| JsValue::from_str(&format!("{op}: toFrame: {e}")))?;
    frame
        .inner
        .with_mut(|rs| -> Result<(), JsValue> {
            for (key, block) in perceived.iter() {
                let Some(target) = rs.get_mut(key) else {
                    rs.insert(key, block.clone());
                    continue;
                };
                for (name, column) in block.iter() {
                    target
                        .insert(
                            name,
                            column.clone(),
                            block.validity(name).map(<[bool]>::to_vec),
                        )
                        .map_err(|e| JsValue::from_str(&format!("{op}: {key}.{name}: {e}")))?;
                }
            }
            Ok(())
        })
        .map_err(crate::core::js_err)??;
    Ok(result)
}

/// Run `edit` on the molecule `frame` holds and hand back the edited
/// molecule as a new frame; `frame` is not touched.
fn frame_out(
    frame: &Frame,
    op: &str,
    edit: impl FnOnce(&Atomistic) -> Result<Atomistic, MolRsError>,
) -> Result<Frame, JsValue> {
    let mol = frame_to_atomistic(frame)?;
    let out = edit(&mol).map_err(|e| JsValue::from_str(&format!("{op}: {e}")))?;
    Frame::from_rs(
        out.to_frame()
            .map_err(|e| JsValue::from_str(&format!("{op}: toFrame: {e}")))?,
    )
}

/// Perceive rings (SSSR) and record them on `frame`, in place
/// ([`molrs::perceive::assign_rings`]).
///
/// Every atom and every bond receives `is_in_ring` (`0` / `1`) and `n_rings`,
/// the number of SSSR rings it belongs to — acyclic ones included, flagged
/// `0` rather than left unset. Input needs `"atoms"` and `"bonds"`
/// (`atomi` / `atomj`).
///
/// # Errors
///
/// Throws, changing nothing, if the frame cannot be read as an atomistic
/// molecule; throws if a perceived column contradicts the Frame schema on the
/// way back.
///
/// # Example (JavaScript)
///
/// ```js
/// assignRings(frame);
/// frame.get("atoms").copy("is_in_ring"); // Int32Array, 1 for ring atoms
/// ```
#[wasm_bindgen(js_name = assignRings)]
pub fn assign_rings(frame: &Frame) -> Result<(), JsValue> {
    assign_in_place(frame, "assignRings", molrs::perceive::assign_rings)
}

/// A new frame: `frame`'s molecule with explicit hydrogens added for
/// unfilled heavy-atom valence ([`molrs::perceive::add_hydrogens`]). Input
/// needs `"atoms"` (`atomic_number`, optional `x`/`y`/`z`) and preferably
/// `"bonds"`; the added hydrogens carry `atomic_number` 1. `frame` itself is
/// not changed.
///
/// When coordinates are present, H is placed at standard X–H lengths along
/// tetrahedral valence-completing directions (geometry only — force fields
/// may refine).
///
/// # Errors
///
/// Throws if the frame cannot be read as an atomistic molecule, if
/// repletion reports a stale atom handle on the graph it built, or if a
/// property of the result contradicts the Frame schema on the way out.
#[wasm_bindgen(js_name = addHydrogens)]
pub fn add_hydrogens(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, "addHydrogens", molrs::perceive::add_hydrogens)
}

/// Assign a localized (Kekulé) `bond_number` to every aromatic bond of
/// `frame`, in place ([`molrs::perceive::assign_kekule_bond_orders`]).
/// Kekulization and nothing else — a frame whose aromatic bonds are not
/// marked yet is left unchanged, because deciding *which* bonds are aromatic
/// belongs to `assignAromaticity`.
///
/// Returns whether any bond was aromatic (and so received a `bond_number`).
///
/// # Errors
///
/// Throws, changing nothing, if the frame cannot be read as an atomistic
/// molecule or the aromatic system has no Kekulé structure.
#[wasm_bindgen(js_name = assignKekuleBondOrders)]
pub fn assign_kekule_bond_orders(frame: &Frame) -> Result<bool, JsValue> {
    assign_in_place(
        frame,
        "assignKekuleBondOrders",
        molrs::perceive::assign_kekule_bond_orders,
    )
}

/// Bring `frame` to the standard aromatic representation, in place
/// ([`molrs::perceive::assign_aromaticity`]).
///
/// On return every aromatic atom carries `is_aromatic`, every aromatic bond
/// carries `bond_type = 4`, and every bond carries an integer `bond_number`
/// — the localized Lewis structure. Nothing carries a fractional order.
///
/// A renderer reads `bond_type` **first**: `4` is aromatic and may be drawn
/// either as a uniform aromatic style or, in Kekulé mode, using
/// `bond_number`. Reading `bond_number` alone and calling anything above 1
/// a double bond is what drew benzene as six double bonds.
///
/// # Errors
///
/// Throws, changing nothing, if the frame cannot be read as an atomistic
/// molecule or perception fails; throws if a perceived column contradicts the
/// Frame schema on the way back.
#[wasm_bindgen(js_name = assignAromaticity)]
pub fn assign_aromaticity(frame: &Frame) -> Result<(), JsValue> {
    assign_in_place(
        frame,
        "assignAromaticity",
        molrs::perceive::assign_aromaticity,
    )
}

/// A new frame: `frame`'s molecule without its terminal (degree-1) explicit
/// hydrogen atoms ([`molrs::perceive::remove_hydrogens`]); non-terminal H is
/// left in place. `frame` itself is not changed.
///
/// # Errors
///
/// Throws if the frame cannot be read as an atomistic molecule, if
/// stripping reports a stale atom handle on the graph it built, or if a
/// property of the result contradicts the Frame schema on the way out.
#[wasm_bindgen(js_name = removeHydrogens)]
pub fn remove_hydrogens(frame: &Frame) -> Result<Frame, JsValue> {
    frame_out(frame, "removeHydrogens", molrs::perceive::remove_hydrogens)
}

/// The molecule `frame` holds. Species comes from `atoms.atomic_number` only;
/// a frame without it (or with an invalid one) is an error that propagates to
/// JS — there is no implicit element → atomic-number conversion here.
fn frame_to_atomistic(frame: &Frame) -> Result<Atomistic, JsValue> {
    frame.with_frame(|rs_frame| {
        Atomistic::from_frame(rs_frame)
            .map_err(|e| JsValue::from_str(&format!("Frame → Atomistic: {e}")))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, Element, keys};
    use molrs::op::Idx;
    use ndarray::{Array1, ArrayD};
    use wasm_bindgen::JsCast;
    use wasm_bindgen_test::*;

    /// Cyclopropane carbons plus one pendant carbon on atom 0.
    fn ring_with_tail() -> Frame {
        let mut atoms = Block::new();
        let carbons: ArrayD<Idx> = Array1::from_vec(vec![Idx::from(Element::C.z()); 4]).into_dyn();
        atoms.insert(keys::ATOMIC_NUMBER, carbons, None).unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                keys::ATOMI,
                Array1::<Idx>::from_vec(vec![0, 1, 2, 0]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                keys::ATOMJ,
                Array1::<Idx>::from_vec(vec![1, 2, 0, 3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        rs_frame.insert("bonds", bonds);
        Frame::from_rs(rs_frame).unwrap()
    }

    /// `assignRings` writes onto the frame it is given and keeps what the
    /// frame held besides (here a meta key and a box).
    #[wasm_bindgen_test]
    fn assign_rings_flags_ring_atoms_in_place() {
        let frame = ring_with_tail();
        frame
            .set_meta("note", JsValue::from_str("kept").unchecked_into())
            .unwrap();
        assign_rings(&frame).unwrap();
        let flags: Vec<i32> = frame
            .with_frame(|f| {
                let column = f.get("atoms").and_then(|a| a.get("is_in_ring"));
                Ok(column
                    .and_then(|c| c.get::<i32>())
                    .map(|a| a.iter().copied().collect())
                    .expect("is_in_ring written onto the input frame"))
            })
            .unwrap();
        assert_eq!(flags, vec![1, 1, 1, 0]);
        let note: JsValue = frame.get_meta("note").unwrap().into();
        assert_eq!(note.as_string().as_deref(), Some("kept"));
    }

    /// `addHydrogens` returns a new frame and leaves its input alone.
    #[wasm_bindgen_test]
    fn add_hydrogens_returns_a_new_frame() {
        let frame = ring_with_tail();
        let with_h = add_hydrogens(&frame).unwrap();
        assert_eq!(frame.get("atoms").unwrap().n_rows().unwrap(), 4);
        assert!(with_h.get("atoms").unwrap().n_rows().unwrap() > 4);
    }

    /// A symbol-only frame is not converted implicitly: perception refuses it
    /// until `fillAtomicNumbers` has run.
    #[wasm_bindgen_test]
    fn perception_refuses_a_symbol_only_frame_until_converted() {
        let mut atoms = Block::new();
        let elements: ArrayD<String> = Array1::from_vec(vec!["C".to_string(); 2]).into_dyn();
        atoms.insert(keys::ELEMENT, elements, None).unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                keys::ATOMI,
                Array1::<Idx>::from_vec(vec![0]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                keys::ATOMJ,
                Array1::<Idx>::from_vec(vec![1]).into_dyn(),
                None,
            )
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        rs_frame.insert("bonds", bonds);
        let frame = Frame::from_rs(rs_frame).unwrap();
        assert!(assign_rings(&frame).is_err());
        crate::core::fill_atomic_numbers(&frame).unwrap();
        assert!(assign_rings(&frame).is_ok());
    }
}
