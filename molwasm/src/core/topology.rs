//! Bond-graph topology — the WASM face of `molrs::core::Topology`.
//!
//! The graph a frame's `bonds` block spells out, with the angles, dihedrals,
//! impropers and connected components derived from it. Ring perception is not
//! here: it is chemistry, and belongs to `assignRings`.

use molrs::core::Topology as RsTopology;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Graph-based molecular topology with automated detection of angles,
/// dihedrals, impropers and connected components.
///
/// API mirrors igraph / molpy conventions.
///
/// # Example (JavaScript)
///
/// ```js
/// const topo = Topology.fromFrame(frame);
/// console.log(topo.nAtoms, topo.nBonds);
///
/// const angles = topo.angles();       // Uint32Array [i,j,k, ...]
/// const dihedrals = topo.dihedrals(); // Uint32Array [i,j,k,l, ...]
/// const cc = topo.connectedComponents(); // Int32Array per-atom labels
/// ```
#[wasm_bindgen(js_name = Topology)]
pub struct Topology {
    inner: RsTopology,
}

#[wasm_bindgen(js_class = Topology)]
impl Topology {
    /// Create a topology with `n` atoms and no bonds.
    #[wasm_bindgen(constructor)]
    pub fn new(n_atoms: usize) -> Self {
        Self {
            inner: RsTopology::with_atoms(n_atoms),
        }
    }

    /// Build a topology from a Frame's `bonds` block.
    ///
    /// The atom count is the `atoms` block's row count; the edges are the
    /// `bonds` block's `atomi` / `atomj` columns, in row order. A frame with no
    /// `bonds` block (or an empty one) is a graph with no edges.
    ///
    /// # Errors
    ///
    /// Throws if the frame has no `atoms` block, if a non-empty `bonds` block
    /// lacks `atomi` / `atomj`, or if a bond names an atom outside the frame.
    #[wasm_bindgen(js_name = fromFrame)]
    pub fn from_frame(frame: &Frame) -> Result<Topology, JsValue> {
        frame.with_frame(|rs_frame| {
            RsTopology::from_frame(rs_frame)
                .map(|inner| Self { inner })
                .map_err(|e| JsValue::from_str(&format!("Topology.fromFrame: {e}")))
        })
    }

    /// Number of atoms (vertices).
    #[wasm_bindgen(getter, js_name = nAtoms)]
    pub fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds (edges).
    #[wasm_bindgen(getter, js_name = nBonds)]
    pub fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Number of unique angles.
    #[wasm_bindgen(getter, js_name = nAngles)]
    pub fn n_angles(&self) -> usize {
        self.inner.n_angles()
    }

    /// Number of unique proper dihedrals.
    #[wasm_bindgen(getter, js_name = nDihedrals)]
    pub fn n_dihedrals(&self) -> usize {
        self.inner.n_dihedrals()
    }

    /// Number of connected components.
    #[wasm_bindgen(getter, js_name = nComponents)]
    pub fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// All bond pairs as flat `Uint32Array` `[i0,j0, i1,j1, ...]`.
    pub fn bonds(&self) -> Vec<u32> {
        self.inner
            .bonds()
            .iter()
            .flat_map(|b| [b[0] as u32, b[1] as u32])
            .collect()
    }

    /// All angle triplets as flat `Uint32Array` `[i,j,k, ...]`.
    pub fn angles(&self) -> Vec<u32> {
        self.inner
            .angles()
            .iter()
            .flat_map(|a| [a[0] as u32, a[1] as u32, a[2] as u32])
            .collect()
    }

    /// All proper dihedral quartets as flat `Uint32Array` `[i,j,k,l, ...]`.
    pub fn dihedrals(&self) -> Vec<u32> {
        self.inner
            .dihedrals()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// One improper quartet per atom with exactly three neighbours, as flat
    /// `Uint32Array` `[center,i,j,k, ...]` (legs sorted).
    pub fn impropers(&self) -> Vec<u32> {
        self.inner
            .impropers()
            .iter()
            .flat_map(|d| [d[0] as u32, d[1] as u32, d[2] as u32, d[3] as u32])
            .collect()
    }

    /// Per-atom connected component labels as `Int32Array`.
    ///
    /// Labels are 0-based and contiguous. Each atom gets a component ID.
    /// Atoms in the same connected subgraph share the same label.
    #[wasm_bindgen(js_name = connectedComponents)]
    pub fn connected_components(&self) -> Vec<i32> {
        self.inner
            .connected_components()
            .iter()
            .map(|&c| c as i32)
            .collect()
    }

    /// Neighbor atom indices of atom `idx` as `Uint32Array`.
    pub fn neighbors(&self, idx: usize) -> Vec<u32> {
        self.inner
            .neighbors(idx)
            .iter()
            .map(|&n| n as u32)
            .collect()
    }

    /// Degree (number of bonds) of atom `idx`.
    pub fn degree(&self, idx: usize) -> usize {
        self.inner.degree(idx)
    }

    /// Whether atoms `i` and `j` are directly bonded.
    #[wasm_bindgen(js_name = areBonded)]
    pub fn are_bonded(&self, i: usize, j: usize) -> bool {
        self.inner.are_bonded(i, j)
    }

    /// Append `n` unbonded atoms, indexed after the existing ones.
    #[wasm_bindgen(js_name = addAtoms)]
    pub fn add_atoms(&mut self, n: usize) {
        self.inner.add_atoms(n);
    }

    /// Add the bonds of a flat `Uint32Array` `[i0,j0, i1,j1, ...]` (the
    /// layout [`bonds`](Self::bonds) returns), in order; a bond already
    /// present, in either orientation, is skipped. One bond is
    /// `addBonds(new Uint32Array([i, j]))`.
    ///
    /// # Errors
    ///
    /// Throws, adding nothing, when the array has an odd length or names an
    /// atom outside `0..nAtoms`.
    #[wasm_bindgen(js_name = addBonds)]
    pub fn add_bonds(&mut self, pairs: &[u32]) -> Result<(), JsValue> {
        if !pairs.len().is_multiple_of(2) {
            return Err(JsValue::from_str(&format!(
                "Topology.addBonds: {} indices is not a list of [i, j] pairs",
                pairs.len()
            )));
        }
        let n = self.inner.n_atoms();
        let edges = pairs
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(k, &[i, j])| {
                let (i, j) = (i as usize, j as usize);
                if i < n && j < n {
                    Ok([i, j])
                } else {
                    Err(JsValue::from_str(&format!(
                        "Topology.addBonds: pair {k} ({i}, {j}) names an atom outside 0..{n}"
                    )))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.inner.add_bonds(&edges);
        Ok(())
    }

    /// Delete atom `idx` and every bond incident to it.
    ///
    /// Swap-remove: the last atom moves into slot `idx`, so that one atom's
    /// index changes. To delete several atoms, use
    /// [`delAtoms`](Self::del_atoms) — deleting them one by one would make the
    /// later indices name relabelled atoms.
    ///
    /// # Errors
    ///
    /// Throws, deleting nothing, when `idx` is not in `0..nAtoms`.
    #[wasm_bindgen(js_name = delAtom)]
    pub fn del_atom(&mut self, idx: usize) -> Result<(), JsValue> {
        check_range("delAtom", "atom", &[idx], self.inner.n_atoms())?;
        self.inner.del_atom(idx);
        Ok(())
    }

    /// Delete every atom of `idxs` (indices *before* the call; duplicates
    /// name one atom) and every bond incident to any of them.
    ///
    /// # Errors
    ///
    /// Throws, deleting nothing, when an index is not in `0..nAtoms`.
    #[wasm_bindgen(js_name = delAtoms)]
    pub fn del_atoms(&mut self, idxs: &[u32]) -> Result<(), JsValue> {
        let idxs: Vec<usize> = idxs.iter().map(|&i| i as usize).collect();
        check_range("delAtoms", "atom", &idxs, self.inner.n_atoms())?;
        self.inner.del_atoms(&idxs);
        Ok(())
    }

    /// Delete bond `idx` (an index into [`bonds`](Self::bonds)).
    ///
    /// Swap-remove: the last bond moves into slot `idx`. To delete several
    /// bonds, use [`delBonds`](Self::del_bonds).
    ///
    /// # Errors
    ///
    /// Throws, deleting nothing, when `idx` is not in `0..nBonds`.
    #[wasm_bindgen(js_name = delBond)]
    pub fn del_bond(&mut self, idx: usize) -> Result<(), JsValue> {
        check_range("delBond", "bond", &[idx], self.inner.n_bonds())?;
        self.inner.del_bond(idx);
        Ok(())
    }

    /// Delete every bond of `idxs` (indices into [`bonds`](Self::bonds)
    /// *before* the call; duplicates name one bond).
    ///
    /// # Errors
    ///
    /// Throws, deleting nothing, when an index is not in `0..nBonds`.
    #[wasm_bindgen(js_name = delBonds)]
    pub fn del_bonds(&mut self, idxs: &[u32]) -> Result<(), JsValue> {
        let idxs: Vec<usize> = idxs.iter().map(|&i| i as usize).collect();
        check_range("delBonds", "bond", &idxs, self.inner.n_bonds())?;
        self.inner.del_bonds(&idxs);
        Ok(())
    }
}

/// Refuse any index of `idxs` outside `0..n`, before the core method (which
/// panics — a trap in wasm) sees it.
fn check_range(op: &str, what: &str, idxs: &[usize], n: usize) -> Result<(), JsValue> {
    match idxs.iter().find(|&&i| i >= n) {
        Some(i) => Err(JsValue::from_str(&format!(
            "Topology.{op}: {what} {i} is outside 0..{n}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Block, keys};
    use molrs::op::Idx;
    use ndarray::Array1;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn from_frame_reads_the_canonical_bond_columns() {
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::<f64>::zeros(3).into_dyn(), None)
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                keys::ATOMI,
                Array1::<Idx>::from_vec(vec![0, 1]).into_dyn(),
                None,
            )
            .unwrap();
        bonds
            .insert(
                keys::ATOMJ,
                Array1::<Idx>::from_vec(vec![1, 2]).into_dyn(),
                None,
            )
            .unwrap();
        let mut rs_frame = molrs::core::Frame::new();
        rs_frame.insert("atoms", atoms);
        rs_frame.insert("bonds", bonds);
        let frame = Frame::from_rs(rs_frame).unwrap();

        let topo = Topology::from_frame(&frame).unwrap();
        assert_eq!(topo.n_atoms(), 3);
        assert_eq!(topo.n_bonds(), 2);
        assert_eq!(topo.n_angles(), 1);
    }

    #[wasm_bindgen_test]
    fn del_single_and_batch_are_distinct_and_range_checked() {
        let mut topo = Topology::new(4);
        topo.add_bonds(&[0, 1, 1, 2, 2, 3]).unwrap();
        assert!(topo.del_bond(3).is_err());
        assert_eq!(topo.n_bonds(), 3);
        topo.del_bond(0).unwrap();
        assert_eq!(topo.n_bonds(), 2);
        topo.del_bonds(&[0, 1]).unwrap();
        assert_eq!(topo.n_bonds(), 0);
        assert!(topo.del_atoms(&[0, 4]).is_err());
        assert_eq!(topo.n_atoms(), 4);
        topo.del_atoms(&[0, 3]).unwrap();
        assert_eq!(topo.n_atoms(), 2);
        topo.del_atom(1).unwrap();
        assert_eq!(topo.n_atoms(), 1);
        assert!(topo.del_atom(1).is_err());
    }
}
