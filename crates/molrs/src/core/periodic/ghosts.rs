//! Ghost atoms: periodic copies, with an owner and an integer shift.

use ndarray::Array2;

use super::images::{GhostError, ImageRange};
use crate::core::SimBox;
use crate::op::{F, Fnx3, Fnx3View};

/// The periodic copies of one owned point set.
///
/// A ghost is `r_g = r_owner + H·s_g`: a copy of owned atom `owner(g)`,
/// translated by the integer lattice vector `s_g`. It is a **geometry object**
/// and nothing else — no velocity, no mass, no thermostat state, no image flag,
/// no identity that survives a rebuild.
///
/// A **shift** is not an image flag: an image flag is an atom's history (how
/// many cells it has crossed), a shift only says which copy this set holds.
///
/// The set stores **only ghost columns**; the owned coordinates arrive by
/// view, so no second copy of them can drift from the first.
#[derive(Debug, Clone)]
pub struct GhostSet {
    /// Ghost positions `(n_ghost, 3)` in Å, placed by [`borders`](Self::borders).
    positions: Fnx3,
    /// Owned index each ghost copies.
    owner: Vec<u32>,
    /// Integer lattice translation each ghost carries.
    shift: Vec<[i32; 3]>,
    /// Owned-atom count the topology was built against.
    n_owned: usize,
    /// Reach the halo was built for (Å).
    reach: F,
}

impl GhostSet {
    /// Build the halo: decide which ghosts exist, and place them.
    ///
    /// A copy is emitted only when it comes within `reach` of the primary cell,
    /// so an interior atom produces none and an atom near a corner produces the
    /// face, edge and corner copies it needs. The test is conservative — it may
    /// keep a copy slightly beyond `reach` — because over-inclusion costs a
    /// distance check downstream while under-inclusion loses a pair silently.
    ///
    /// `owned` is expected to be wrapped (inside the cell). Nothing breaks if
    /// it is not; the halo is simply built around wherever the atoms are.
    pub fn borders(bx: &SimBox, owned: Fnx3View<'_>, reach: F) -> Result<Self, GhostError> {
        let range = ImageRange::new(bx, reach)?;
        let d = bx.nearest_plane_distance();
        let frac = bx.to_frac(owned);
        let n_owned = owned.nrows();

        let mut owner: Vec<u32> = Vec::new();
        let mut shift: Vec<[i32; 3]> = Vec::new();
        for &s in range.shifts() {
            if s == [0, 0, 0] {
                continue; // the owned atoms themselves are not ghosts
            }
            for i in 0..n_owned {
                // Distance from the translated copy to the cell, measured
                // per-axis in plane-spacing units. `max(0, -g, g - 1) * d` is
                // how far outside the slab the copy sits on that axis; the true
                // point-to-cell distance is at least the largest of the three.
                let mut out_of_cell = 0.0_f64;
                for k in 0..3 {
                    let g = frac[[i, k]] + s[k] as F;
                    let past = (-g).max(g - 1.0).max(0.0);
                    let dist = past * d[k];
                    if dist > out_of_cell {
                        out_of_cell = dist;
                    }
                }
                if out_of_cell <= reach {
                    owner.push(i as u32);
                    shift.push(s);
                }
            }
        }

        let mut set = Self {
            positions: Fnx3::zeros((owner.len(), 3)),
            owner,
            shift,
            n_owned,
            reach,
        };
        set.place(bx, owned)?;
        Ok(set)
    }

    /// `r_g = r_owner + H·s_g` for every ghost.
    fn place(&mut self, bx: &SimBox, owned: Fnx3View<'_>) -> Result<(), GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        let a = [bx.lattice(0), bx.lattice(1), bx.lattice(2)];
        for g in 0..self.owner.len() {
            let o = self.owner[g] as usize;
            let s = self.shift[g];
            for k in 0..3 {
                self.positions[[g, k]] =
                    owned[[o, k]] + s[0] as F * a[0][k] + s[1] as F * a[1][k] + s[2] as F * a[2][k];
            }
        }
        Ok(())
    }

    /// Ghost positions `(n_ghost, 3)` in Å.
    pub fn positions(&self) -> Fnx3View<'_> {
        self.positions.view()
    }

    /// The owned index each ghost copies.
    pub fn owner(&self) -> &[u32] {
        &self.owner
    }

    /// The integer lattice translation each ghost carries.
    pub fn shift(&self) -> &[[i32; 3]] {
        &self.shift
    }

    /// How many ghosts exist.
    pub fn len(&self) -> usize {
        self.owner.len()
    }

    /// Whether the halo is empty (an interior-only configuration, or a free box).
    pub fn is_empty(&self) -> bool {
        self.owner.is_empty()
    }

    /// Owned-atom count the topology was built against.
    pub fn n_owned(&self) -> usize {
        self.n_owned
    }

    /// The reach (Å) the halo was built for.
    pub fn reach(&self) -> F {
        self.reach
    }

    /// The combined `[owned | ghost]` view a force evaluation consumes.
    ///
    /// Rows `0..n_owned` are the owned atoms exactly as supplied; the rest are
    /// the ghosts. Only the first `n_owned` are degrees of freedom, which is
    /// why a force array over this layout has to be reduced before it means
    /// anything physical.
    pub fn combined(&self, owned: Fnx3View<'_>) -> Result<Fnx3, GhostError> {
        if owned.nrows() != self.n_owned {
            return Err(GhostError::Shape {
                expected: self.n_owned,
                found: owned.nrows(),
            });
        }
        let mut out = Array2::zeros((self.n_owned + self.owner.len(), 3));
        for i in 0..self.n_owned {
            for k in 0..3 {
                out[[i, k]] = owned[[i, k]];
            }
        }
        for g in 0..self.owner.len() {
            for k in 0..3 {
                out[[self.n_owned + g, k]] = self.positions[[g, k]];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use ndarray::array;

    fn cube(l: F) -> SimBox {
        SimBox::new(Cell::Lengths([l; 3]), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap()
    }

    /// An atom in the middle of the cell needs no copies; one in a corner needs
    /// seven — three faces, three edges, one corner.
    #[test]
    fn a_corner_atom_gets_face_edge_and_corner_copies() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();

        assert_eq!(set.len(), 7, "3 faces + 3 edges + 1 corner");
        let mut got: Vec<[i32; 3]> = set.shift().to_vec();
        got.sort_unstable();
        let mut want = vec![
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ];
        want.sort_unstable();
        assert_eq!(got, want);

        // Each copy really is the owner plus that lattice vector.
        for g in 0..set.len() {
            let s = set.shift()[g];
            for k in 0..3 {
                let want = owned[[0, k]] + s[k] as F * 10.0;
                assert!((set.positions()[[g, k]] - want).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn an_interior_atom_gets_none() {
        let bx = cube(10.0);
        let owned = array![[5.0_f64, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        assert!(set.is_empty());
        // ...while the range still enumerates the full 27 translations: which
        // copies *could* exist and which are *kept* are separate questions.
        assert_eq!(ImageRange::new(&bx, 2.0).unwrap().shifts().len(), 27);
    }

    #[test]
    fn a_non_periodic_axis_contributes_no_copies() {
        let bx = SimBox::new(
            Cell::Lengths([10.0; 3]),
            [0.0_f64, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        let owned = array![[0.5_f64, 0.5, 0.5]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        assert!(!set.is_empty());
        for s in set.shift() {
            assert_eq!(s[1], 0, "the free axis must not be tiled");
        }
    }

    /// The combined view is owned-then-ghost, and the owned half is the input
    /// unchanged — a caller indexing below `n_owned` is indexing real atoms.
    #[test]
    fn the_combined_view_is_owned_then_ghost() {
        let bx = cube(10.0);
        let owned = array![[0.5_f64, 0.5, 0.5], [5.0, 5.0, 5.0]];
        let set = GhostSet::borders(&bx, owned.view(), 2.0).unwrap();
        let all = set.combined(owned.view()).unwrap();

        assert_eq!(all.nrows(), set.n_owned() + set.len());
        for i in 0..set.n_owned() {
            for k in 0..3 {
                assert_eq!(all[[i, k]], owned[[i, k]]);
            }
        }
        for g in 0..set.len() {
            for k in 0..3 {
                assert_eq!(all[[set.n_owned() + g, k]], set.positions()[[g, k]]);
            }
        }
    }

    /// A reach the cell cannot express is refused with the numbers in hand,
    /// never quietly truncated.
    #[test]
    fn an_unrepresentable_reach_is_refused() {
        let bx = cube(1.0);
        let err = ImageRange::new(&bx, 1_000.0).unwrap_err();
        let msg = format!("{err}");
        assert!(matches!(err, GhostError::TooManyImages { .. }), "{msg}");
        assert!(msg.contains("nothing is truncated"), "{msg}");

        assert!(matches!(
            ImageRange::new(&bx, -1.0).unwrap_err(),
            GhostError::InvalidReach(_)
        ));
        assert!(matches!(
            ImageRange::new(&bx, F::NAN).unwrap_err(),
            GhostError::InvalidReach(_)
        ));
    }

    /// A tilted cell sizes its range from the plane spacings. The cell here has
    /// three 10 Å edges but spacings of 8, 8 and 10, so edge-length sizing
    /// would enumerate 27 translations where 75 are needed.
    #[test]
    fn a_tilted_cell_sizes_from_plane_spacings() {
        let h = [[10.0_f64, 6.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]];
        let bx = SimBox::new(Cell::Matrix(h), [0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let d = bx.nearest_plane_distance();
        assert!((d[0] - 8.0).abs() < 1e-12 && (d[1] - 8.0).abs() < 1e-12);
        assert!((bx.lengths()[1] - 10.0).abs() < 1e-12);

        let range = ImageRange::new(&bx, 9.0).unwrap();
        assert_eq!(range.per_axis(), [2, 2, 1]);
        assert_eq!(range.shifts().len(), 75);
    }
}
