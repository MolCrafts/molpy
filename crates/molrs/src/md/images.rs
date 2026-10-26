//! Bonded terms across periodic faces: each term placed compactly around one
//! atom, the images it needs appended as copies of their owners.
//!
//! A bonded kernel holds indices, not geometry: it is told *these two atoms
//! are bonded* and takes the plain difference of their coordinates. For a bond
//! that straddles a face those coordinates are a cell apart, so the kernel
//! measures a bond stretched by the width of the box and answers with a force
//! to match. Nothing raises — the number is simply wrong.
//!
//! [`BondedImages`] fixes that without any kernel learning that periodic
//! boundaries exist. Every term is resolved against **one** anchor atom — a
//! bond against its first atom, anything longer against its second (the
//! vertex of an angle, the atom a dihedral's others are nearest) — and each
//! other atom is placed at its minimum image from the anchor. A partner whose
//! nearest image is not where it is stored becomes an appended row: a copy of
//! its owner at the image position. The kernel reads `[base | copies]` through
//! the remapped indices, and the copies' forces are folded back onto their
//! owners after the virial `Σ f ⊗ x` has been taken over every row.
//!
//! Resolving against one anchor rather than edge by edge matters: individually
//! closest images can be jointly inconsistent — the three atoms of an angle in
//! three different cells — and the angle is then measured on a shape that does
//! not exist.

use ndarray::Array2;

use molrs::core::SimBox;
use molrs::op::F;

use super::error::MdError;

/// The index tables of a force evaluation's bonded members, resolved against
/// the minimum image, and the copies they reach.
#[derive(Debug, Default)]
pub(crate) struct BondedImages {
    /// Per member, the remapped table; `None` for a member that holds no
    /// indices.
    current: Vec<Option<Array2<u32>>>,
    /// Owner of each appended copy, in append order.
    owner: Vec<u32>,
    /// Flat `[x, y, z, …]` of the appended copies.
    copies: Vec<F>,
}

impl BondedImages {
    /// Resolve every table of `sources` (one entry per member) against `base`
    /// (flat coordinates whose first `n_owned` rows are the owned atoms the
    /// tables index) in `simbox`. With no box, or a box periodic on no axis,
    /// every table is taken as written.
    ///
    /// # Errors
    ///
    /// [`MdError::Invalid`] when a term still spans more than half the
    /// smallest perpendicular width of the cell after placement: past that the
    /// minimum image is ambiguous, so the term would have been guessed, not
    /// resolved.
    pub(crate) fn resolve(
        &mut self,
        sources: &[Option<Array2<u32>>],
        base: &[F],
        simbox: Option<&SimBox>,
    ) -> Result<(), MdError> {
        self.current.clear();
        self.owner.clear();
        self.copies.clear();
        let periodic = simbox.filter(|b| b.pbc().iter().any(|&p| p));
        let Some(simbox) = periodic else {
            self.current.extend(sources.iter().cloned());
            return Ok(());
        };
        let mic = simbox.mic();
        let limit = 0.5
            * simbox
                .nearest_plane_distance()
                .iter()
                .copied()
                .fold(F::INFINITY, F::min);
        let n_base = (base.len() / 3) as u32;
        let at = |a: usize| [base[3 * a], base[3 * a + 1], base[3 * a + 2]];
        for source in sources {
            let Some(source) = source else {
                self.current.push(None);
                continue;
            };
            let mut current = source.clone();
            let anchor_col = if source.ncols() <= 2 { 0 } else { 1 };
            for (t, row) in source.rows().into_iter().enumerate() {
                let anchor = row[anchor_col] as usize;
                let x_anchor = at(anchor);
                for (c, &partner) in row.iter().enumerate() {
                    if c == anchor_col {
                        continue;
                    }
                    let x = at(partner as usize);
                    let raw = [x[0] - x_anchor[0], x[1] - x_anchor[1], x[2] - x_anchor[2]];
                    let d = mic.apply(raw);
                    let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    if length > limit {
                        return Err(MdError::Invalid(format!(
                            "a bonded term spans {length:.3} between atoms {anchor} and \
                             {partner}, past half the cell's smallest width ({limit:.3}); \
                             its periodic image is not decidable"
                        )));
                    }
                    if d != raw {
                        current[[t, c]] = n_base + self.owner.len() as u32;
                        self.owner.push(partner);
                        self.copies.extend((0..3).map(|k| x_anchor[k] + d[k]));
                    }
                }
            }
            self.current.push(Some(current));
        }
        Ok(())
    }

    /// Member `m`'s table as resolved.
    pub(crate) fn current(&self, m: usize) -> Option<&Array2<u32>> {
        self.current.get(m)?.as_ref()
    }

    /// The appended copies, flat.
    pub(crate) fn copies(&self) -> &[F] {
        &self.copies
    }

    /// Fold the copies' forces (rows `n_base..` of `forces`) onto their
    /// owners and clear those rows.
    pub(crate) fn fold(&self, forces: &mut [F], n_base: usize) {
        for (c, &owner) in self.owner.iter().enumerate() {
            let row = n_base + c;
            for k in 0..3 {
                forces[3 * owner as usize + k] += forces[3 * row + k];
                forces[3 * row + k] = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Cell;

    fn cubic(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0; 3], [true; 3]).unwrap()
    }

    /// A bond through the x face is placed as its short image, through one
    /// appended copy whose force folds back onto its owner.
    #[test]
    fn a_bond_across_a_face_reaches_through_one_copy() {
        let base = [0.5, 1.0, 1.0, 9.5, 1.0, 1.0];
        let table = Array2::from_shape_vec((1, 2), vec![0_u32, 1]).unwrap();
        let mut images = BondedImages::default();
        images
            .resolve(&[Some(table)], &base, Some(&cubic(10.0)))
            .unwrap();
        assert_eq!(images.current(0).unwrap().row(0).to_vec(), [0, 2]);
        assert_eq!(images.copies(), [-0.5, 1.0, 1.0]);

        let mut forces = vec![0.0; 9];
        forces[6] = 2.0;
        images.fold(&mut forces, 2);
        assert_eq!(&forces[3..6], [2.0, 0.0, 0.0]);
        assert_eq!(&forces[6..], [0.0; 3]);
    }

    /// An angle is placed around its vertex: both arms reach the vertex's
    /// side of the cell, so the shape measured is the real one.
    #[test]
    fn an_angle_is_placed_around_its_vertex() {
        let base = [9.8, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2, 1.0, 0.0];
        let table = Array2::from_shape_vec((1, 3), vec![0_u32, 1, 2]).unwrap();
        let mut images = BondedImages::default();
        images
            .resolve(&[Some(table)], &base, Some(&cubic(10.0)))
            .unwrap();
        assert_eq!(images.current(0).unwrap().row(0).to_vec(), [3, 1, 2]);
        let x = images.copies()[0];
        assert!((x + 0.2).abs() < 1e-12, "{x}");
    }

    /// Without a periodic box the tables are used as written, and a term
    /// longer than half the cell is refused, not guessed.
    #[test]
    fn free_boxes_keep_tables_and_overlong_terms_are_refused() {
        let base = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0];
        let table = Array2::from_shape_vec((1, 2), vec![0_u32, 1]).unwrap();
        let mut images = BondedImages::default();
        images.resolve(&[Some(table.clone())], &base, None).unwrap();
        assert_eq!(images.current(0), Some(&table));
        assert!(images.copies().is_empty());

        let base = [0.0, 0.0, 0.0, 5.0, 5.0, 5.0];
        let err = images
            .resolve(&[Some(table)], &base, Some(&cubic(10.0)))
            .unwrap_err();
        assert!(err.to_string().contains("not decidable"), "{err}");
    }
}
