//! Exact single-wall carbon nanotubes built from rolled graphene topology.

use crate::core::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

use crate::core::Atomistic;
use crate::core::{Element, Props};

use crate::core::Frame;
use crate::core::SimBox;
use crate::core::keys;
use crate::op::F;

type SiteKey = (i64, i64, u8);

/// How far a [`CarbonTubeBuilder`] tube extends along its axis.
///
/// Unset, the tube is one translational cell long (two when periodic, the
/// shortest periodic graph without parallel edges for most chiralities).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TubeExtent {
    /// A fixed number (≥ 1) of axial translational cells.
    Cells(usize),
    /// An axial length in angstrom (> 0), rounded up to complete
    /// translational cells.
    Length(f64),
}

/// Error returned when nanotube parameters cannot describe a valid graph.
#[derive(Debug, Clone, PartialEq)]
pub enum CarbonTubeError {
    /// The chiral indices are zero, too small, or impractically large.
    InvalidChirality,
    /// A scalar parameter is non-finite or outside its allowed range.
    InvalidParameter(&'static str),
    /// The exact graphene quotient did not enumerate the expected sites.
    IncompleteLattice,
    /// The axial periodic cell would require parallel graph edges.
    PeriodicCellTooShort,
    /// A generated bond could not be added to the graph.
    Graph(String),
    /// The generated simulation cell is invalid.
    Cell(String),
}

impl fmt::Display for CarbonTubeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidChirality => write!(
                f,
                "n and m must be non-negative, not both zero, and describe distinct graphene neighbours"
            ),
            Self::InvalidParameter(name) => write!(f, "{name} is invalid"),
            Self::IncompleteLattice => {
                write!(f, "internal nanotube lattice enumeration is incomplete")
            }
            Self::PeriodicCellTooShort => write!(
                f,
                "the periodic cell is too short to represent distinct bonds; increase cells or length"
            ),
            Self::Graph(message) => write!(f, "could not build nanotube graph: {message}"),
            Self::Cell(message) => write!(f, "could not build nanotube cell: {message}"),
        }
    }
}

impl Error for CarbonTubeError {}

#[derive(Debug)]
struct CompiledTube {
    coordinates: Vec<[f64; 3]>,
    bonds: Vec<(usize, usize)>,
    radius: f64,
    axial_length: f64,
}

/// Builder for an exact single-wall carbon nanotube [`Frame`].
///
/// Sites and bonds are inherited from an integer graphene lattice quotient;
/// connectivity is never guessed from rolled Cartesian distances. The built
/// frame contains `atoms` and `bonds` blocks plus an orthorhombic simulation
/// box with periodicity enabled only along the tube axis when requested.
#[derive(Debug, Clone)]
pub struct CarbonTubeBuilder {
    n: u32,
    m: u32,
    extent: Option<TubeExtent>,
    bond_length: f64,
    periodic: bool,
    vacuum: f64,
    atom_type: Option<String>,
    charge: f64,
}

impl CarbonTubeBuilder {
    /// Start a nanotube builder for chiral indices `(n, m)`.
    pub fn new(n: u32, m: u32) -> Result<Self, CarbonTubeError> {
        // The upper bound prevents accidental multi-terabyte enumeration while
        // remaining far beyond physically useful interactive nanotubes.
        if (n == 0 && m == 0)
            || u64::from(n) * u64::from(n)
                + u64::from(n) * u64::from(m)
                + u64::from(m) * u64::from(m)
                < 4
            || n > 10_000
            || m > 10_000
        {
            return Err(CarbonTubeError::InvalidChirality);
        }
        Ok(Self {
            n,
            m,
            extent: None,
            bond_length: 1.42,
            periodic: false,
            vacuum: 10.0,
            atom_type: None,
            charge: 0.0,
        })
    }

    /// Set the axial extent: a cell count or a length rounded up to complete
    /// cells.
    ///
    /// # Errors
    ///
    /// [`CarbonTubeError::InvalidParameter`] for zero cells or a length that
    /// is not finite and positive.
    pub fn with_extent(mut self, extent: TubeExtent) -> Result<Self, CarbonTubeError> {
        match extent {
            TubeExtent::Cells(0) => return Err(CarbonTubeError::InvalidParameter("cells")),
            TubeExtent::Cells(_) => {}
            TubeExtent::Length(length) => positive("length", length)?,
        }
        self.extent = Some(extent);
        Ok(self)
    }

    /// Set the carbon-carbon bond length in angstrom.
    pub fn with_bond_length(mut self, bond_length: f64) -> Result<Self, CarbonTubeError> {
        positive("bond_length", bond_length)?;
        self.bond_length = bond_length;
        Ok(self)
    }

    /// Close the graph along the tube axis.
    pub fn with_periodic(mut self, periodic: bool) -> Self {
        self.periodic = periodic;
        self
    }

    /// Set the empty transverse padding around the tube wall in angstrom.
    pub fn with_vacuum(mut self, vacuum: f64) -> Result<Self, CarbonTubeError> {
        nonnegative("vacuum", vacuum)?;
        self.vacuum = vacuum;
        Ok(self)
    }

    /// Add a force-field atom type to every generated carbon atom.
    pub fn with_atom_type(mut self, atom_type: impl Into<String>) -> Result<Self, CarbonTubeError> {
        let atom_type = atom_type.into();
        if atom_type.is_empty() {
            return Err(CarbonTubeError::InvalidParameter("atom_type"));
        }
        self.atom_type = Some(atom_type);
        Ok(self)
    }

    /// Add a finite partial charge to every generated carbon atom.
    pub fn with_charge(mut self, charge: f64) -> Result<Self, CarbonTubeError> {
        if !charge.is_finite() {
            return Err(CarbonTubeError::InvalidParameter("charge"));
        }
        self.charge = charge;
        Ok(self)
    }

    /// Chiral index `n`.
    pub fn n(&self) -> u32 {
        self.n
    }

    /// Chiral index `m`.
    pub fn m(&self) -> u32 {
        self.m
    }

    /// Carbon-carbon bond length in angstrom.
    pub fn bond_length(&self) -> f64 {
        self.bond_length
    }

    /// Whether the axial graph and cell are periodic.
    pub fn periodic(&self) -> bool {
        self.periodic
    }

    /// Resolved number of complete axial translational cells.
    pub fn cells(&self) -> usize {
        match self.extent {
            None => {
                if self.periodic {
                    2
                } else {
                    1
                }
            }
            Some(TubeExtent::Cells(cells)) => cells,
            Some(TubeExtent::Length(length)) => {
                (length / self.axial_unit_length()).ceil().max(1.0) as usize
            }
        }
    }

    /// Validate the full lattice and periodic topology without retaining it.
    pub fn validate(&self) -> Result<(), CarbonTubeError> {
        self.compile().map(|_| ())
    }

    /// Build a fresh molecular [`Frame`] with atoms, bonds, and simulation box.
    pub fn build(&self) -> Result<Frame, CarbonTubeError> {
        let compiled = self.compile()?;
        let mut graph = Atomistic::new();
        let mut atoms = Vec::with_capacity(compiled.coordinates.len());
        for [x, y, z] in compiled.coordinates {
            let mut props = Props::new()
                .with(keys::ATOMIC_NUMBER, Element::C)
                .with(keys::X, x)
                .with(keys::Y, y)
                .with(keys::Z, z)
                .with(keys::CHARGE, self.charge);
            if let Some(atom_type) = &self.atom_type {
                props.set(keys::TYPE, atom_type.as_str());
            }
            atoms.push(
                graph
                    .add_atom(props)
                    .map_err(|error| CarbonTubeError::Graph(error.to_string()))?,
            );
        }
        for (i, j) in compiled.bonds {
            graph
                .add_bond(atoms[i], atoms[j], Props::new())
                .map_err(|error| CarbonTubeError::Graph(error.to_string()))?;
        }

        let mut frame = graph
            .to_frame()
            .map_err(|error| CarbonTubeError::Graph(error.to_string()))?;
        frame.simbox = Some(self.cell_from_geometry(compiled.radius, compiled.axial_length)?);
        Ok(frame)
    }

    /// Simulation cell matching the generated coordinates.
    pub fn cell(&self) -> Result<SimBox, CarbonTubeError> {
        let compiled = self.compile()?;
        self.cell_from_geometry(compiled.radius, compiled.axial_length)
    }

    fn cell_from_geometry(
        &self,
        radius: f64,
        axial_length: f64,
    ) -> Result<SimBox, CarbonTubeError> {
        let transverse = 2.0 * (radius + self.vacuum);
        SimBox::new(
            Cell::Lengths([transverse as F, transverse as F, axial_length as F]),
            [
                (-radius - self.vacuum) as F,
                (-radius - self.vacuum) as F,
                0.0 as F,
            ],
            [false, false, self.periodic],
        )
        .map_err(|error| CarbonTubeError::Cell(format!("{error:?}")))
    }

    fn translation(&self) -> (i64, i64) {
        let n = i64::from(self.n);
        let m = i64::from(self.m);
        let divisor = gcd(2 * m + n, 2 * n + m);
        ((2 * m + n) / divisor, -(2 * n + m) / divisor)
    }

    fn axial_unit_length(&self) -> f64 {
        let (t1, t2) = self.translation();
        let a1x = 3.0_f64.sqrt() * self.bond_length;
        let a2x = 0.5 * a1x;
        let a2y = 1.5 * self.bond_length;
        ((t1 as f64 * a1x + t2 as f64 * a2x).powi(2) + (t2 as f64 * a2y).powi(2)).sqrt()
    }

    fn compile(&self) -> Result<CompiledTube, CarbonTubeError> {
        let n = i64::from(self.n);
        let m = i64::from(self.m);
        let cells = self.cells();
        let (t1, t2) = self.translation();
        let determinant = n * t2 - m * t1;
        let orientation = determinant.signum();
        let denominator = 3 * determinant.abs();

        let projected = |i: i64, j: i64, sublattice: u8| -> (i64, i64) {
            let sublattice = i64::from(sublattice);
            let qx = 3 * i + sublattice;
            let qy = 3 * j + sublattice;
            (
                orientation * (qx * t2 - qy * t1),
                orientation * (n * qy - m * qx),
            )
        };

        let corners = [(0, 0), (n, m), (t1, t2), (n + t1, m + t2)];
        let i_min = corners.iter().map(|(i, _)| *i).min().unwrap() - 2;
        let i_max = corners.iter().map(|(i, _)| *i).max().unwrap() + 2;
        let j_min = corners.iter().map(|(_, j)| *j).min().unwrap() - 2;
        let j_max = corners.iter().map(|(_, j)| *j).max().unwrap() + 2;

        let mut unit_sites = BTreeMap::new();
        for i in i_min..=i_max {
            for j in j_min..=j_max {
                for sublattice in [0, 1] {
                    let (u_num, v_num) = projected(i, j, sublattice);
                    if (0..denominator).contains(&u_num) && (0..denominator).contains(&v_num) {
                        unit_sites.insert((u_num, v_num, sublattice), (i, j));
                    }
                }
            }
        }

        let mut sites = BTreeMap::new();
        for cell in 0..cells {
            let cell =
                i64::try_from(cell).map_err(|_| CarbonTubeError::InvalidParameter("cells"))?;
            for (&(u_num, v_num, sublattice), &(i, j)) in &unit_sites {
                sites.insert(
                    (u_num, v_num + cell * denominator, sublattice),
                    (i + cell * t1, j + cell * t2),
                );
            }
        }

        let expected_per_cell = 4 * (n * n + n * m + m * m) / gcd(2 * m + n, 2 * n + m);
        if sites.len() != expected_per_cell as usize * cells {
            return Err(CarbonTubeError::IncompleteLattice);
        }

        let mut ordered_keys: Vec<SiteKey> = sites.keys().copied().collect();
        ordered_keys.sort_by_key(|&(u_num, v_num, sublattice)| (v_num, u_num, sublattice));
        let indices: BTreeMap<SiteKey, usize> = ordered_keys
            .iter()
            .copied()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect();
        let axial_denominator = i64::try_from(cells)
            .map_err(|_| CarbonTubeError::InvalidParameter("cells"))?
            * denominator;

        let canonical_key = |i: i64, j: i64, sublattice: u8| -> Option<SiteKey> {
            let (u_num, mut v_num) = projected(i, j, sublattice);
            if self.periodic {
                v_num = v_num.rem_euclid(axial_denominator);
            } else if !(0..axial_denominator).contains(&v_num) {
                return None;
            }
            Some((u_num.rem_euclid(denominator), v_num, sublattice))
        };

        let mut bonds = BTreeSet::new();
        for (&key, &(i, j)) in &sites {
            if key.2 != 0 {
                continue;
            }
            for (neighbour_i, neighbour_j) in [(i, j), (i - 1, j), (i, j - 1)] {
                let Some(neighbour) = canonical_key(neighbour_i, neighbour_j, 1) else {
                    continue;
                };
                let Some(&neighbour_index) = indices.get(&neighbour) else {
                    return Err(CarbonTubeError::IncompleteLattice);
                };
                let index = indices[&key];
                bonds.insert(if index < neighbour_index {
                    (index, neighbour_index)
                } else {
                    (neighbour_index, index)
                });
            }
        }

        if self.periodic && bonds.len() != 3 * sites.len() / 2 {
            return Err(CarbonTubeError::PeriodicCellTooShort);
        }

        let a1x = 3.0_f64.sqrt() * self.bond_length;
        let a2x = 0.5 * a1x;
        let a2y = 1.5 * self.bond_length;
        let circumference =
            ((n as f64 * a1x + m as f64 * a2x).powi(2) + (m as f64 * a2y).powi(2)).sqrt();
        let radius = circumference / (2.0 * PI);
        let axial_unit = self.axial_unit_length();
        let coordinates = ordered_keys
            .iter()
            .map(|&(u_num, v_num, _)| {
                let theta = 2.0 * PI * u_num as f64 / denominator as f64;
                [
                    radius * theta.cos(),
                    radius * theta.sin(),
                    axial_unit * v_num as f64 / denominator as f64,
                ]
            })
            .collect();

        Ok(CompiledTube {
            coordinates,
            bonds: bonds.into_iter().collect(),
            radius,
            axial_length: cells as f64 * axial_unit,
        })
    }
}

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

fn positive(name: &'static str, value: f64) -> Result<(), CarbonTubeError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CarbonTubeError::InvalidParameter(name))
    }
}

fn nonnegative(name: &'static str, value: f64) -> Result<(), CarbonTubeError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(CarbonTubeError::InvalidParameter(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_tube_has_exact_graphene_degree() {
        let frame = CarbonTubeBuilder::new(4, 2)
            .unwrap()
            .with_extent(TubeExtent::Cells(2))
            .unwrap()
            .with_periodic(true)
            .build()
            .unwrap();
        let atoms = frame.get("atoms").unwrap();
        let bonds = frame.get("bonds").unwrap();
        let atom_count = atoms.n_rows().unwrap();
        assert_eq!(atom_count, 112);
        assert_eq!(bonds.n_rows(), Some(3 * atom_count / 2));

        let mut degree = vec![0; atom_count];
        for &index in bonds.get("atomi").and_then(|c| c.get::<u64>()).unwrap() {
            degree[index as usize] += 1;
        }
        for &index in bonds.get("atomj").and_then(|c| c.get::<u64>()).unwrap() {
            degree[index as usize] += 1;
        }
        assert!(degree.into_iter().all(|value| value == 3));
    }

    #[test]
    fn build_returns_independent_frames_with_axial_cell() {
        let builder = CarbonTubeBuilder::new(5, 5)
            .unwrap()
            .with_extent(TubeExtent::Cells(2))
            .unwrap()
            .with_periodic(true)
            .with_vacuum(4.0)
            .unwrap();
        let first = builder.build().unwrap();
        let second = builder.build().unwrap();
        assert_eq!(
            first.get("atoms").unwrap().n_rows(),
            second.get("atoms").unwrap().n_rows()
        );
        let simbox = first.simbox.as_ref().unwrap();
        assert_eq!(simbox.pbc(), [false, false, true]);
    }

    #[test]
    fn one_cell_armchair_periodic_graph_is_rejected() {
        let result = CarbonTubeBuilder::new(5, 5)
            .unwrap()
            .with_extent(TubeExtent::Cells(1))
            .unwrap()
            .with_periodic(true)
            .build();
        assert_eq!(result.unwrap_err(), CarbonTubeError::PeriodicCellTooShort);
    }

    #[test]
    fn with_extent_refuses_zero_cells_and_a_non_positive_length() {
        let builder = CarbonTubeBuilder::new(6, 0).unwrap();
        for extent in [
            TubeExtent::Cells(0),
            TubeExtent::Length(0.0),
            TubeExtent::Length(-1.0),
            TubeExtent::Length(f64::NAN),
        ] {
            assert!(
                matches!(
                    builder.clone().with_extent(extent),
                    Err(CarbonTubeError::InvalidParameter(_))
                ),
                "{extent:?}"
            );
        }
        assert_eq!(
            builder.with_extent(TubeExtent::Cells(3)).unwrap().cells(),
            3
        );
    }

    #[test]
    fn requested_length_rounds_up_to_axial_units() {
        let unit = CarbonTubeBuilder::new(6, 0)
            .unwrap()
            .cell()
            .unwrap()
            .lengths()[2];
        let cell = CarbonTubeBuilder::new(6, 0)
            .unwrap()
            .with_extent(TubeExtent::Length(2.2 * unit))
            .unwrap()
            .cell()
            .unwrap();
        assert!(cell.lengths()[2] >= 2.2 * unit);
        assert!(cell.lengths()[2] < 3.2 * unit);
    }
}
