//! Flat graphene (honeycomb) sheet builder.

use crate::core::Cell;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::core::Atomistic;
use crate::core::{Element, Props};

use crate::core::Frame;
use crate::core::SimBox;
use crate::core::keys;
use crate::op::F;

/// Error returned when graphene sheet parameters are invalid.
#[derive(Debug, Clone, PartialEq)]
pub enum GrapheneError {
    /// A dimension is zero or impractically large.
    InvalidSize,
    /// A scalar parameter is non-finite or outside its allowed range.
    InvalidParameter(&'static str),
    /// A bond or atom property could not be written.
    Graph(String),
    /// The simulation cell could not be built.
    Cell(String),
}

impl fmt::Display for GrapheneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize => {
                write!(f, "nx and ny must be positive and not larger than 10_000")
            }
            Self::InvalidParameter(name) => write!(f, "{name} is invalid"),
            Self::Graph(message) => write!(f, "could not build graphene graph: {message}"),
            Self::Cell(message) => write!(f, "could not build graphene cell: {message}"),
        }
    }
}

impl Error for GrapheneError {}

/// Builder for a rectangular graphene sheet [`Frame`].
///
/// Honeycomb lattice with bond length `a` and lattice vectors
///
/// ```text
/// a₁ = (√3 a, 0)
/// a₂ = (√3 a / 2, 3 a / 2)
/// ```
///
/// Two carbon sites (A, B) per unit cell. `nx × ny` cells → `2·nx·ny` atoms.
/// In-plane bonds wrap when [`with_periodic_xy`](Self::with_periodic_xy) is true.
#[derive(Debug, Clone)]
pub struct GrapheneBuilder {
    nx: u32,
    ny: u32,
    bond_length: f64,
    vacuum: f64,
    periodic_xy: bool,
    atom_type: Option<String>,
    charge: f64,
}

impl GrapheneBuilder {
    /// Start a sheet builder for `nx × ny` honeycomb unit cells.
    pub fn new(nx: u32, ny: u32) -> Result<Self, GrapheneError> {
        if nx == 0 || ny == 0 || nx > 10_000 || ny > 10_000 {
            return Err(GrapheneError::InvalidSize);
        }
        Ok(Self {
            nx,
            ny,
            bond_length: 1.42,
            vacuum: 10.0,
            periodic_xy: true,
            atom_type: None,
            charge: 0.0,
        })
    }

    /// Carbon–carbon bond length in ångström (default 1.42).
    pub fn with_bond_length(mut self, bond_length: f64) -> Result<Self, GrapheneError> {
        if !bond_length.is_finite() || bond_length <= 0.0 {
            return Err(GrapheneError::InvalidParameter("bond_length"));
        }
        self.bond_length = bond_length;
        Ok(self)
    }

    /// Vacuum padding along *z* (default 10 Å).
    pub fn with_vacuum(mut self, vacuum: f64) -> Result<Self, GrapheneError> {
        if !vacuum.is_finite() || vacuum < 0.0 {
            return Err(GrapheneError::InvalidParameter("vacuum"));
        }
        self.vacuum = vacuum;
        Ok(self)
    }

    /// Close bonds across the *xy* periodic images (default `true`).
    pub fn with_periodic_xy(mut self, periodic_xy: bool) -> Self {
        self.periodic_xy = periodic_xy;
        self
    }

    /// Optional force-field atom type for every carbon.
    pub fn with_atom_type(mut self, atom_type: impl Into<String>) -> Result<Self, GrapheneError> {
        let atom_type = atom_type.into();
        if atom_type.is_empty() {
            return Err(GrapheneError::InvalidParameter("atom_type"));
        }
        self.atom_type = Some(atom_type);
        Ok(self)
    }

    /// Finite partial charge on every carbon.
    pub fn with_charge(mut self, charge: f64) -> Result<Self, GrapheneError> {
        if !charge.is_finite() {
            return Err(GrapheneError::InvalidParameter("charge"));
        }
        self.charge = charge;
        Ok(self)
    }

    /// Number of unit cells along **a₁**.
    pub fn nx(&self) -> u32 {
        self.nx
    }

    /// Number of unit cells along **a₂**.
    pub fn ny(&self) -> u32 {
        self.ny
    }

    /// Carbon–carbon bond length in ångström.
    pub fn bond_length(&self) -> f64 {
        self.bond_length
    }

    /// Whether *xy* bonds wrap across the cell.
    pub fn periodic_xy(&self) -> bool {
        self.periodic_xy
    }

    /// Build a fresh molecular [`Frame`] (atoms, bonds, orthorhombic box).
    pub fn build(&self) -> Result<Frame, GrapheneError> {
        let a = self.bond_length;
        let a1x = 3.0_f64.sqrt() * a;
        let a2x = 0.5 * a1x;
        let a2y = 1.5 * a;
        let nx = self.nx as usize;
        let ny = self.ny as usize;

        let idx = |i: usize, j: usize, s: usize| -> usize { 2 * (i + j * nx) + s };
        let wrap = |i: isize, n: usize| -> Option<usize> {
            if self.periodic_xy {
                Some(i.rem_euclid(n as isize) as usize)
            } else if (0..n as isize).contains(&i) {
                Some(i as usize)
            } else {
                None
            }
        };

        let mut graph = Atomistic::new();
        let mut atoms = Vec::with_capacity(2 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let ox = i as f64 * a1x + j as f64 * a2x;
                let oy = j as f64 * a2y;
                for (dx, dy) in [(0.0, 0.0), (a1x / 3.0 + a2x / 3.0, a2y / 3.0)] {
                    let mut props = Props::new()
                        .with(keys::ATOMIC_NUMBER, Element::C)
                        .with(keys::X, ox + dx)
                        .with(keys::Y, oy + dy)
                        .with(keys::Z, 0.0)
                        .with(keys::CHARGE, self.charge);
                    if let Some(ref t) = self.atom_type {
                        props.set(keys::TYPE, t.as_str());
                    }
                    atoms.push(
                        graph
                            .add_atom(props)
                            .map_err(|e| GrapheneError::Graph(e.to_string()))?,
                    );
                }
            }
        }

        // A (s=0) bonds to three B (s=1) sites: (i,j), (i-1,j), (i,j-1).
        let mut bonds = BTreeSet::new();
        for j in 0..ny {
            for i in 0..nx {
                let a0 = idx(i, j, 0);
                let partners = [
                    Some(idx(i, j, 1)),
                    wrap(i as isize - 1, nx).map(|ii| idx(ii, j, 1)),
                    wrap(j as isize - 1, ny).map(|jj| idx(i, jj, 1)),
                ];
                for partner in partners.into_iter().flatten() {
                    bonds.insert(if a0 < partner {
                        (a0, partner)
                    } else {
                        (partner, a0)
                    });
                }
            }
        }
        for (u, v) in bonds {
            graph
                .add_bond(atoms[u], atoms[v], Props::new())
                .map_err(|e| GrapheneError::Graph(e.to_string()))?;
        }

        let mut frame = graph
            .to_frame()
            .map_err(|e| GrapheneError::Graph(e.to_string()))?;
        frame.simbox = Some(self.cell()?);
        Ok(frame)
    }

    /// Simulation cell matching the generated sheet.
    pub fn cell(&self) -> Result<SimBox, GrapheneError> {
        let a = self.bond_length;
        let a1x = 3.0_f64.sqrt() * a;
        let a2y = 1.5 * a;
        let lx = self.nx as f64 * a1x;
        let ly = self.ny as f64 * a2y;
        let lz = self.vacuum.max(a);
        SimBox::new(
            Cell::Lengths([lx as F, ly as F, lz as F]),
            [0.0 as F, 0.0 as F, -lz * 0.5 as F],
            [self.periodic_xy, self.periodic_xy, false],
        )
        .map_err(|e| GrapheneError::Cell(format!("{e:?}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_periodic_sheet_has_three_bonds_per_atom() {
        let frame = GrapheneBuilder::new(4, 4)
            .unwrap()
            .with_periodic_xy(true)
            .build()
            .unwrap();
        let n = frame.get("atoms").unwrap().n_rows().unwrap();
        assert_eq!(n, 32);
        let bonds = frame.get("bonds").unwrap();
        assert_eq!(bonds.n_rows(), Some(3 * n / 2));

        let mut degree = vec![0; n];
        for &i in bonds.get("atomi").and_then(|c| c.get::<u64>()).unwrap() {
            degree[i as usize] += 1;
        }
        for &j in bonds.get("atomj").and_then(|c| c.get::<u64>()).unwrap() {
            degree[j as usize] += 1;
        }
        assert!(degree.into_iter().all(|d| d == 3));
    }

    #[test]
    fn open_sheet_has_fewer_bonds_than_periodic() {
        let open = GrapheneBuilder::new(3, 3)
            .unwrap()
            .with_periodic_xy(false)
            .build()
            .unwrap();
        let closed = GrapheneBuilder::new(3, 3)
            .unwrap()
            .with_periodic_xy(true)
            .build()
            .unwrap();
        assert!(
            open.get("bonds").unwrap().n_rows().unwrap()
                < closed.get("bonds").unwrap().n_rows().unwrap()
        );
    }
}
