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

/// LAMMPS's own bounds for an axis a data file does not state
/// (`read_data`: `xlo xhi` defaults to `-0.5 0.5`).
const DEFAULT_AXIS: (f64, f64) = (-0.5, 0.5);

/// Build a [`SimBox`] from extents + PBC, as LAMMPS reads them: an axis the
/// header does not state takes LAMMPS's default bounds ([`DEFAULT_AXIS`]),
/// and absent tilt factors are 0. `Ok(None)` only when no axis is stated at
/// all.
///
/// # Errors
///
/// `InvalidData` naming the axis whose `hi` is not above its `lo` — a box
/// LAMMPS refuses, never dropped.
pub(crate) fn simbox_from_bounds(bounds: &BoxBounds, pbc: Pbc3) -> std::io::Result<Option<SimBox>> {
    if !(bounds.has_x || bounds.has_y || bounds.has_z) {
        return Ok(None);
    }
    let axis = |name: &str, present: bool, lo: f64, hi: f64| {
        let (lo, hi) = if present { (lo, hi) } else { DEFAULT_AXIS };
        if hi > lo {
            Ok((lo, hi - lo))
        } else {
            Err(invalid_data(format!(
                "box {name}lo {name}hi is {lo} {hi}: {name}hi must be above {name}lo"
            )))
        }
    };
    let (xlo, lx) = axis("x", bounds.has_x, bounds.xlo, bounds.xhi)?;
    let (ylo, ly) = axis("y", bounds.has_y, bounds.ylo, bounds.yhi)?;
    let (zlo, lz) = axis("z", bounds.has_z, bounds.zlo, bounds.zhi)?;
    let origin = [xlo, ylo, zlo];
    let cell = if bounds.xy.is_some() || bounds.xz.is_some() || bounds.yz.is_some() {
        let (xy, xz, yz) = (
            bounds.xy.unwrap_or(0.0),
            bounds.xz.unwrap_or(0.0),
            bounds.yz.unwrap_or(0.0),
        );
        Cell::Matrix([[lx, xy, xz], [0.0, ly, yz], [0.0, 0.0, lz]])
    } else {
        Cell::Lengths([lx, ly, lz])
    };
    SimBox::new(cell, origin, pbc)
        .map(Some)
        .map_err(|e| invalid_data(format!("{e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stated(lo_hi: [(f64, f64); 3], present: [bool; 3]) -> BoxBounds {
        BoxBounds {
            xlo: lo_hi[0].0,
            xhi: lo_hi[0].1,
            ylo: lo_hi[1].0,
            yhi: lo_hi[1].1,
            zlo: lo_hi[2].0,
            zhi: lo_hi[2].1,
            has_x: present[0],
            has_y: present[1],
            has_z: present[2],
            ..BoxBounds::default()
        }
    }

    /// An axis the header leaves out is LAMMPS's `-0.5 0.5`, not a reason to
    /// drop the other two.
    #[test]
    fn an_absent_axis_takes_the_lammps_default() {
        let b = stated([(0.0, 10.0), (0.0, 0.0), (2.0, 5.0)], [true, false, true]);
        let sb = simbox_from_bounds(&b, [true; 3]).unwrap().expect("a box");
        assert_eq!(sb.lengths().to_vec(), [10.0, 1.0, 3.0]);
        assert_eq!(sb.origin().to_vec(), [0.0, -0.5, 2.0]);
    }

    /// An axis whose hi is not above its lo is refused by name.
    #[test]
    fn a_reversed_or_empty_axis_is_refused() {
        for (lo, hi) in [(5.0, 1.0), (3.0, 3.0)] {
            let b = stated([(0.0, 1.0), (lo, hi), (0.0, 1.0)], [true; 3]);
            let err = simbox_from_bounds(&b, [true; 3]).unwrap_err();
            assert!(err.to_string().contains("yhi must be above ylo"), "{err}");
        }
        let none = stated([(0.0, 0.0); 3], [false; 3]);
        assert!(simbox_from_bounds(&none, [true; 3]).unwrap().is_none());
    }
}
