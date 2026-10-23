//! Crystal lattices: a unit cell and its basis sites, tiled over a range of
//! cells and optionally clipped to a region.
//!
//! The cell is a [`SimBox`], which owns fractional ↔ Cartesian conversion; the
//! space groups are [`SpaceGroup`]'s.
//!
//! ```
//! use molrs::builder::Lattice;
//!
//! let nickel = Lattice::fcc(3.52, "Ni").unwrap();
//! let crystal = nickel.build(None, Some([2, 2, 2])).unwrap();
//! assert_eq!(crystal.n_atoms(), 32);
//! ```

use std::collections::BTreeSet;
use std::fmt;

use ndarray::Array2;

use crate::core::schema::block_names::ATOMS;
use crate::core::{Atomistic, Cell, Element, Frame, Props, Region, SimBox, SpaceGroup, keys};
use crate::op::F;

/// Fractional slack on the covering tile, so an atom on the region's
/// boundary is not lost to rounding in `to_frac`.
const FRAC_SLACK: F = 1e-6;
/// An occupancy this close below 1 still counts as a fully occupied site.
const OCCUPANCY_TOL: F = 1e-6;
/// The atoms column a built crystal names each atom's basis site in.
const LABEL: &str = "label";

/// A basis site in fractional coordinates of the cell.
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    /// Site identifier (`"A"`, `"B1"`).
    pub label: String,
    /// Element symbol of the atom placed there.
    pub species: String,
    /// Fractional coordinates `(u, v, w)`.
    pub frac: [F; 3],
    /// Partial charge.
    pub charge: F,
    /// Further per-atom fields every built atom of this site carries. Every
    /// site of a lattice declares the same keys.
    pub attrs: Props,
}

impl Site {
    /// A neutral site with no further fields.
    pub fn new(label: impl Into<String>, species: impl Into<String>, frac: [F; 3]) -> Self {
        Self {
            label: label.into(),
            species: species.into(),
            frac,
            charge: 0.0,
            attrs: Props::new(),
        }
    }
}

/// Why a lattice cannot be made or built.
#[derive(Debug, Clone, PartialEq)]
pub enum LatticeError {
    /// The box has no cell.
    NoCell,
    /// A repeat count is zero.
    Repeats([usize; 3]),
    /// Neither a region nor repeats chooses the tile.
    NoTile,
    /// The region is unbounded and no repeats choose the tile.
    Unbounded,
    /// A site lacks a field other sites declare.
    MissingAttrs {
        /// The site.
        site: String,
        /// The fields it lacks.
        keys: Vec<String>,
    },
    /// A site's species is no element symbol.
    Species {
        /// The site.
        site: String,
        /// Its species.
        species: String,
    },
    /// A CIF frame lacks what a lattice is read from.
    Cif(String),
    /// Sites a CIF states as partially occupied.
    PartialOccupancy(Vec<String>),
    /// The box or the built structure refused.
    Core(String),
}

impl fmt::Display for LatticeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoCell => write!(f, "a lattice needs a box with a defined cell"),
            Self::Repeats(r) => write!(f, "repeats must be three positive integers, got {r:?}"),
            Self::NoTile => write!(f, "provide a region, repeats, or both"),
            Self::Unbounded => write!(
                f,
                "the region is unbounded; pass repeats to choose the tile"
            ),
            Self::MissingAttrs { site, keys } => write!(
                f,
                "site {site:?} lacks field(s) {keys:?} that other sites declare"
            ),
            Self::Species { site, species } => {
                write!(f, "site {site:?}: species {species:?} is no element symbol")
            }
            Self::Cif(why) => write!(f, "{why}"),
            Self::PartialOccupancy(sites) => write!(
                f,
                "site(s) {sites:?} have partial occupancy (< 1); a lattice places one atom \
                 per site and cannot represent disorder. Resolve the disorder in the CIF \
                 (keep one component at occupancy 1) before building"
            ),
            Self::Core(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for LatticeError {}

/// A Bravais lattice: a unit cell and the basis sites in it.
#[derive(Debug, Clone)]
pub struct Lattice {
    cell: SimBox,
    basis: Vec<Site>,
}

impl Lattice {
    /// A lattice on `cell` with `basis`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::NoCell`] for a box without a cell.
    pub fn new(cell: SimBox, basis: Vec<Site>) -> Result<Self, LatticeError> {
        if !cell.is_cell_defined() {
            return Err(LatticeError::NoCell);
        }
        Ok(Self { cell, basis })
    }

    /// The unit cell.
    pub fn cell(&self) -> &SimBox {
        &self.cell
    }

    /// The basis sites.
    pub fn basis(&self) -> &[Site] {
        &self.basis
    }

    /// This lattice with `site` appended to its basis.
    pub fn with_site(&self, site: Site) -> Self {
        let mut basis = self.basis.clone();
        basis.push(site);
        Self {
            cell: self.cell.clone(),
            basis,
        }
    }

    /// The box `repeats` unit cells span: each lattice vector scaled by its
    /// repeat, the origin and periodicity the unit cell's.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Repeats`] for a zero repeat.
    pub fn supercell(&self, repeats: [usize; 3]) -> Result<SimBox, LatticeError> {
        positive(repeats)?;
        let scale = Array2::from_diag(&ndarray::arr1(&repeats.map(|r| r as F)));
        self.cell
            .transformed(&scale)
            .map_err(|e| LatticeError::Core(e.to_string()))
    }

    /// A cubic cell of edge `a` (Å), periodic on every axis, with `basis`.
    fn cubic(a: F, basis: Vec<Site>) -> Result<Self, LatticeError> {
        let cell = SimBox::new(Cell::Lengths([a; 3]), [0.0; 3], [true; 3])
            .map_err(|e| LatticeError::Core(e.to_string()))?;
        Self::new(cell, basis)
    }

    /// Simple cubic: one atom per cell.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Core`] for an edge that is not a positive length.
    pub fn sc(a: F, species: &str) -> Result<Self, LatticeError> {
        Self::cubic(a, vec![Site::new("A", species, [0.0; 3])])
    }

    /// Body-centred cubic: two atoms per cell.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Core`] for an edge that is not a positive length.
    pub fn bcc(a: F, species: &str) -> Result<Self, LatticeError> {
        Self::cubic(
            a,
            vec![
                Site::new("A", species, [0.0; 3]),
                Site::new("B", species, [0.5; 3]),
            ],
        )
    }

    /// Face-centred cubic: four atoms per cell.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Core`] for an edge that is not a positive length.
    pub fn fcc(a: F, species: &str) -> Result<Self, LatticeError> {
        Self::cubic(
            a,
            vec![
                Site::new("A", species, [0.0, 0.0, 0.0]),
                Site::new("B", species, [0.5, 0.5, 0.0]),
                Site::new("C", species, [0.5, 0.0, 0.5]),
                Site::new("D", species, [0.0, 0.5, 0.5]),
            ],
        )
    }

    /// Rocksalt (NaCl): two interpenetrating face-centred cubic sublattices.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Core`] for an edge that is not a positive length.
    pub fn rocksalt(a: F, species_a: &str, species_b: &str) -> Result<Self, LatticeError> {
        let corner = [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ];
        let edge = [
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5],
        ];
        let basis = corner
            .iter()
            .enumerate()
            .map(|(k, f)| Site::new(format!("A{}", k + 1), species_a, *f))
            .chain(
                edge.iter()
                    .enumerate()
                    .map(|(k, f)| Site::new(format!("B{}", k + 1), species_b, *f)),
            )
            .collect();
        Self::cubic(a, basis)
    }

    /// The full-cell lattice of an asymmetric unit: every symmetry image of
    /// every site in `group`, images on a special position collapsed within
    /// `symprec` (fractional). Image `k` of site `s` is labelled `"s_k"` and
    /// keeps the site's species, charge and fields.
    ///
    /// # Errors
    ///
    /// [`LatticeError::NoCell`] for a box without a cell.
    pub fn from_spacegroup(
        cell: SimBox,
        sites: &[Site],
        group: &SpaceGroup,
        symprec: F,
    ) -> Result<Self, LatticeError> {
        let basis = sites
            .iter()
            .flat_map(|site| {
                group
                    .equivalent_positions(site.frac, symprec)
                    .into_iter()
                    .enumerate()
                    .map(move |(k, frac)| Site {
                        label: format!("{}_{k}", site.label),
                        frac,
                        ..site.clone()
                    })
            })
            .collect();
        Self::new(cell, basis)
    }

    /// The full-cell lattice of a frame read from a CIF: the cell is the
    /// frame's box, the asymmetric unit its `atoms` (label from `name`,
    /// species from `element`, fractional coordinates from the box), the
    /// space group the operator loop the reader keeps in `meta["symops"]`.
    ///
    /// # Errors
    ///
    /// [`LatticeError::Cif`] for a frame without a cell, operators, or the
    /// `name` / `element` / coordinate columns; [`LatticeError::PartialOccupancy`]
    /// for a site occupied less than fully — the lattice places one atom per
    /// site, so a disordered structure would be built fully occupied.
    pub fn from_cif(frame: &Frame, symprec: F) -> Result<Self, LatticeError> {
        let cell = frame
            .simbox
            .clone()
            .filter(SimBox::is_cell_defined)
            .ok_or_else(|| {
                LatticeError::Cif(
                    "the frame has no box with a cell (the CIF states no _cell_length_* / \
                     _cell_angle_*); a lattice needs the unit cell"
                        .into(),
                )
            })?;
        let group = SpaceGroup::from_frame(frame).map_err(LatticeError::Cif)?;
        let atoms = frame
            .get(ATOMS)
            .ok_or_else(|| LatticeError::Cif("the frame has no atoms block".into()))?;
        let strings = |key: &str| -> Result<Vec<String>, LatticeError> {
            let col = atoms
                .get(key)
                .and_then(|c| c.get::<String>())
                .ok_or_else(|| LatticeError::Cif(format!("CIF atoms block lacks `{key}`")))?;
            Ok(col.iter().cloned().collect())
        };
        let names = strings(keys::NAME)?;
        let species = strings(keys::ELEMENT)?;
        if let Some(occupancy) = atoms.get(keys::OCCUPANCY).and_then(|c| c.get::<F>()) {
            let partial: Vec<String> = names
                .iter()
                .zip(occupancy.iter())
                .filter(|(_, occ)| **occ < 1.0 - OCCUPANCY_TOL)
                .map(|(n, _)| n.clone())
                .collect();
            if !partial.is_empty() {
                return Err(LatticeError::PartialOccupancy(partial));
            }
        }
        let xyz = frame
            .coords()
            .map_err(|e| LatticeError::Cif(format!("CIF atoms coordinates: {e}")))?;
        let frac = cell.to_frac(xyz.view());
        let sites: Vec<Site> = names
            .into_iter()
            .zip(species)
            .zip(frac.rows())
            .map(|((name, element), f)| Site::new(name, element, [f[0], f[1], f[2]]))
            .collect();
        Self::from_spacegroup(cell, &sites, &group, symprec)
    }

    /// Tile the lattice and keep the atoms inside `region`.
    ///
    /// `repeats` tiles cells `0 ≤ (i, j, k) < repeats`; without it the tile is
    /// every cell — negative indices included — whose atoms can fall inside
    /// the region's bounds. Each atom carries `x`/`y`/`z`, `element`,
    /// `atomic_number`, `charge`, `label` (its site) and its site's fields.
    /// The tiled cell is [`supercell`](Self::supercell), which the caller
    /// sets on a frame when the structure becomes a simulation.
    ///
    /// # Errors
    ///
    /// [`LatticeError::NoTile`] with neither argument, [`LatticeError::Repeats`]
    /// for a zero repeat, [`LatticeError::Unbounded`] for an unbounded region
    /// without repeats, [`LatticeError::MissingAttrs`] when the sites declare
    /// different fields, [`LatticeError::Species`] for a species that is no
    /// element.
    pub fn build(
        &self,
        region: Option<&dyn Region>,
        repeats: Option<[usize; 3]>,
    ) -> Result<Atomistic, LatticeError> {
        let (lo, hi) = match (repeats, region) {
            (Some(r), _) => {
                positive(r)?;
                ([0; 3], r.map(|n| n as i64))
            }
            (None, Some(region)) => self.covering_cells(region)?,
            (None, None) => return Err(LatticeError::NoTile),
        };
        let fields = self.shared_fields()?;
        let elements = self
            .basis
            .iter()
            .map(|site| {
                Element::by_symbol(&site.species).ok_or_else(|| LatticeError::Species {
                    site: site.label.clone(),
                    species: site.species.clone(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut fracs = Vec::new();
        let mut owners = Vec::new();
        for i in lo[0]..hi[0] {
            for j in lo[1]..hi[1] {
                for k in lo[2]..hi[2] {
                    for (s, site) in self.basis.iter().enumerate() {
                        fracs.extend([
                            i as F + site.frac[0],
                            j as F + site.frac[1],
                            k as F + site.frac[2],
                        ]);
                        owners.push(s);
                    }
                }
            }
        }
        let fracs = Array2::from_shape_vec((owners.len(), 3), fracs)
            .map_err(|e| LatticeError::Core(e.to_string()))?;
        let carts = self.cell.to_cart(fracs.view());
        let keep = match region {
            Some(region) => region.contains(&carts).to_vec(),
            None => vec![true; owners.len()],
        };

        let mut out = Atomistic::new();
        for (row, &s) in owners.iter().enumerate() {
            if !keep[row] {
                continue;
            }
            let site = &self.basis[s];
            let mut props = Props::new();
            for key in &fields {
                if let Some(value) = site.attrs.get(key) {
                    props.set(key, value.clone());
                }
            }
            props.set(keys::X, carts[[row, 0]]);
            props.set(keys::Y, carts[[row, 1]]);
            props.set(keys::Z, carts[[row, 2]]);
            props.set(keys::ELEMENT, site.species.clone());
            props.set(keys::ATOMIC_NUMBER, elements[s]);
            props.set(keys::CHARGE, site.charge);
            props.set(LABEL, site.label.clone());
            out.add_atom(props)
                .map_err(|e| LatticeError::Core(e.to_string()))?;
        }
        Ok(out)
    }

    /// The field keys every site declares; an error naming the first site
    /// that lacks one another declares.
    fn shared_fields(&self) -> Result<BTreeSet<String>, LatticeError> {
        let all: BTreeSet<String> = self
            .basis
            .iter()
            .flat_map(|site| site.attrs.keys().map(str::to_owned))
            .collect();
        for site in &self.basis {
            let missing: Vec<String> = all
                .iter()
                .filter(|key| !site.attrs.contains_key(key))
                .cloned()
                .collect();
            if !missing.is_empty() {
                return Err(LatticeError::MissingAttrs {
                    site: site.label.clone(),
                    keys: missing,
                });
            }
        }
        Ok(all)
    }

    /// Cell range `[lo, hi)` whose atoms cover `region`'s bounding box. An
    /// atom of cell `n` sits at fractional `n + b` for a basis offset `b`, so
    /// the cells that reach the box's fractional extent `[f_min, f_max]` are
    /// `floor(f_min − max b) ≤ n ≤ floor(f_max − min b)` on every axis.
    fn covering_cells(&self, region: &dyn Region) -> Result<([i64; 3], [i64; 3]), LatticeError> {
        let bounds = region.bounds();
        if bounds.iter().any(|v| !v.is_finite()) {
            return Err(LatticeError::Unbounded);
        }
        let mut corners = Array2::<F>::zeros((8, 3));
        for c in 0..8 {
            for axis in 0..3 {
                corners[[c, axis]] = bounds[[axis, (c >> axis) & 1]];
            }
        }
        let frac = self.cell.to_frac(corners.view());
        let mut lo = [0; 3];
        let mut hi = [0; 3];
        for axis in 0..3 {
            let column = frac.column(axis);
            let f_min = column.iter().copied().fold(F::INFINITY, F::min);
            let f_max = column.iter().copied().fold(F::NEG_INFINITY, F::max);
            let b_min = self
                .basis
                .iter()
                .map(|s| s.frac[axis])
                .fold(F::INFINITY, F::min);
            let b_max = self
                .basis
                .iter()
                .map(|s| s.frac[axis])
                .fold(F::NEG_INFINITY, F::max);
            lo[axis] = (f_min - b_max - FRAC_SLACK).floor() as i64;
            hi[axis] = (f_max - b_min + FRAC_SLACK).floor() as i64 + 1;
        }
        Ok((lo, hi))
    }
}

/// `repeats` when every count is positive.
fn positive(repeats: [usize; 3]) -> Result<(), LatticeError> {
    if repeats.contains(&0) {
        return Err(LatticeError::Repeats(repeats));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Cuboid, Sphere};

    #[test]
    fn repeats_tile_every_basis_site_of_every_cell() {
        let fcc = Lattice::fcc(3.52, "Ni").unwrap();
        let crystal = fcc.build(None, Some([2, 3, 1])).unwrap();
        assert_eq!(crystal.n_atoms(), 4 * 6);
        let xs: Vec<F> = crystal
            .atoms()
            .map(|(_, a)| a.position().expect("placed")[0])
            .collect();
        // Cells 0 and 1 along a: fractional 0, 0.5, 1, 1.5 → 0, 1.76, 3.52, 5.28 Å.
        for x in [0.0, 1.76, 3.52, 5.28] {
            assert!(xs.iter().any(|v| (v - x).abs() < 1e-12), "{x} in {xs:?}");
        }
        let atom = crystal.atoms().next().unwrap().1;
        assert_eq!(atom.get_str(keys::ELEMENT), Some("Ni"));
        assert_eq!(atom.atomic_number(), Some(28));
    }

    #[test]
    fn a_region_alone_tiles_the_cells_that_reach_it_and_keeps_its_inside() {
        let sc = Lattice::sc(1.0, "Ar").unwrap();
        // A cube from −1.5 to 1.5 Å holds the points −1, 0, 1 on each axis.
        let cube = Cuboid::new(ndarray::arr1(&[-1.5; 3]), ndarray::arr1(&[3.0; 3]));
        let crystal = sc.build(Some(&cube), None).unwrap();
        assert_eq!(crystal.n_atoms(), 27);
        let ball = Sphere::new(ndarray::arr1(&[0.0; 3]), 1.01);
        // The origin and its six face neighbours.
        assert_eq!(sc.build(Some(&ball), None).unwrap().n_atoms(), 7);
    }

    #[test]
    fn what_cannot_be_tiled_is_refused() {
        let sc = Lattice::sc(1.0, "Ar").unwrap();
        assert_eq!(sc.build(None, None).unwrap_err(), LatticeError::NoTile);
        assert_eq!(
            sc.build(None, Some([1, 0, 1])).unwrap_err(),
            LatticeError::Repeats([1, 0, 1])
        );
        let mut odd = Site::new("B", "Ar", [0.5; 3]);
        odd.attrs.set("tag", 1);
        let err = sc.with_site(odd).build(None, Some([1, 1, 1])).unwrap_err();
        assert!(matches!(err, LatticeError::MissingAttrs { site, .. } if site == "A"));
        let bad = Lattice::sc(1.0, "Xx").unwrap();
        assert!(matches!(
            bad.build(None, Some([1, 1, 1])).unwrap_err(),
            LatticeError::Species { .. }
        ));
    }

    #[test]
    fn the_supercell_scales_each_lattice_vector() {
        let fcc = Lattice::fcc(2.0, "Cu").unwrap();
        let big = fcc.supercell([2, 3, 4]).unwrap();
        assert_eq!(big.lengths().to_vec(), vec![4.0, 6.0, 8.0]);
    }

    #[test]
    fn a_space_group_expands_the_asymmetric_unit() {
        // Fm-3m's centring translations make one site the four of fcc.
        let group = SpaceGroup::from_generators([
            "x,y,z",
            "x,y+1/2,z+1/2",
            "x+1/2,y,z+1/2",
            "x+1/2,y+1/2,z",
        ])
        .unwrap();
        let cell = Lattice::sc(3.6, "Cu").unwrap().cell().clone();
        let lattice =
            Lattice::from_spacegroup(cell, &[Site::new("Cu", "Cu", [0.0; 3])], &group, 1e-5)
                .unwrap();
        assert_eq!(lattice.basis().len(), 4);
        assert!(lattice.basis().iter().all(|s| s.label.starts_with("Cu_")));
    }
}
