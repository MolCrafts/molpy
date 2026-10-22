//! Orthogonal / triclinic box bounds → [`SimBox`].

use crate::core::Cell;
use crate::io::invalid_data;
use molrs::core::SimBox;
use molrs::op::Pbc3;

/// Simulation-box extents as written in LAMMPS data headers or dump
/// `ITEM: BOX BOUNDS` blocks.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoxBounds {
    pub xlo: f64,
    pub xhi: f64,
    pub ylo: f64,
    pub yhi: f64,
    pub zlo: f64,
    pub zhi: f64,
    pub xy: Option<f64>,
    pub xz: Option<f64>,
    pub yz: Option<f64>,
    /// Which axes were present in the file header (`xlo xhi` / …).
    pub has_x: bool,
    pub has_y: bool,
    pub has_z: bool,
}

/// Collapse LAMMPS boundary tokens (`pp`, `ff`, `ss`, `fs`, …) to a per-axis
/// periodic flag: periodic iff the first character is `p`.
pub(crate) fn pbc_from_boundary_tokens(tokens: &[String; 3]) -> Pbc3 {
    [
        tokens[0].starts_with('p'),
        tokens[1].starts_with('p'),
        tokens[2].starts_with('p'),
    ]
}

/// Build a [`SimBox`] from extents + PBC. Returns `Ok(None)` when any edge
/// length is non-positive.
pub(crate) fn simbox_from_bounds(bounds: &BoxBounds, pbc: Pbc3) -> std::io::Result<Option<SimBox>> {
    let lx = bounds.xhi - bounds.xlo;
    let ly = bounds.yhi - bounds.ylo;
    let lz = bounds.zhi - bounds.zlo;
    if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
        return Ok(None);
    }
    let origin = [bounds.xlo, bounds.ylo, bounds.zlo];
    let simbox = if let (Some(xy), Some(xz), Some(yz)) = (bounds.xy, bounds.xz, bounds.yz) {
        let h = [[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]];
        SimBox::new(Cell::Matrix(h), origin, pbc).map_err(|e| invalid_data(format!("{:?}", e)))?
    } else {
        SimBox::new(Cell::Lengths([lx, ly, lz]), origin, pbc)
            .map_err(|e| invalid_data(format!("{:?}", e)))?
    };
    Ok(Some(simbox))
}
