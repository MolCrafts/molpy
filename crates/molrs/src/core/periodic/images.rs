//! Which periodic images of a cell can reach a given distance.
//!
//! A ghost layer materialises copies of atoms across periodic faces, and the
//! first question is *which* copies: a translation by `n₁a₁ + n₂a₂ + n₃a₃` for
//! integer `n`. [`ImageRange`] answers it for a reach `R`, and refuses rather
//! than truncates when it cannot.

use crate::core::SimBox;
use crate::op::F;

/// Why an image enumeration could not be produced.
///
/// Every variant carries the numbers that produced it: a periodic-geometry
/// failure is a quantitative statement, and a message that only says "invalid"
/// sends the reader back to a debugger.
#[derive(Debug, Clone, PartialEq)]
pub enum GhostError {
    /// The cell has a zero or non-finite plane spacing — no lattice to speak of.
    DegenerateCell {
        /// Axis whose spacing is unusable.
        axis: usize,
        /// The spacing found.
        width: F,
    },
    /// The reach is negative or non-finite.
    InvalidReach(F),
    /// The enumeration would exceed [`ImageRange::MAX_IMAGES`].
    ///
    /// Raised rather than clamped: a silently truncated image set drops pairs
    /// with nothing said, which is the failure this whole layer exists to make
    /// impossible.
    TooManyImages {
        /// Per-axis half-range that produced the count.
        per_axis: [i32; 3],
        /// Total translations the range would enumerate.
        count: u128,
        /// The ceiling.
        limit: u128,
    },
    /// A ghost halo's pair cutoff is not a positive distance.
    InvalidCutoff(F),
    /// A ghost halo's skin is negative or not a number.
    InvalidSkin(F),
    /// A ghost halo's search radius `cutoff + skin` passes half the smallest
    /// perpendicular cell width, so a pair would have more than one image
    /// inside it and the halo keeps only one.
    SearchBeyondHalfWidth {
        /// Pair cutoff (Å).
        cutoff: F,
        /// Verlet skin (Å).
        skin: F,
        /// Half the smallest perpendicular cell width (Å).
        half_width: F,
    },
    /// A caller handed columns whose row counts disagree.
    Shape {
        /// Rows expected.
        expected: usize,
        /// Rows supplied.
        found: usize,
    },
}

impl std::fmt::Display for GhostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DegenerateCell { axis, width } => write!(
                f,
                "cell is degenerate along axis {axis}: plane spacing {width} Å is not a usable width"
            ),
            Self::InvalidReach(r) => {
                write!(f, "ghost reach must be finite and >= 0 Å, got {r} Å")
            }
            Self::TooManyImages {
                per_axis,
                count,
                limit,
            } => write!(
                f,
                "image enumeration needs {count} lattice translations \
                 (half-range {per_axis:?}), above the {limit} ceiling. \
                 The reach is large against this cell; nothing is truncated \
                 because a missing image is a missing pair"
            ),
            Self::InvalidCutoff(c) => write!(f, "cutoff must be > 0 Å, got {c}"),
            Self::InvalidSkin(s) => write!(f, "skin must be >= 0 Å, got {s}"),
            Self::SearchBeyondHalfWidth {
                cutoff,
                skin,
                half_width,
            } => write!(
                f,
                "cutoff {cutoff} Å + skin {skin} Å exceeds half the minimum perpendicular \
                 cell width ({half_width:.3} Å); a pair would have more than one image \
                 inside the search radius and the halo keeps only one"
            ),
            Self::Shape { expected, found } => {
                write!(f, "expected {expected} rows, got {found}")
            }
        }
    }
}

impl std::error::Error for GhostError {}

/// The lattice translations that can bring a copy within `reach` of the cell.
///
/// The half-range on periodic axis `k` is `ceil(reach / d_k)`, where `d_k` is
/// the **perpendicular plane spacing** ([`SimBox::nearest_plane_distance`]).
/// A non-periodic axis contributes only the zero translation.
///
/// Sizing from `d_k` and not from the lattice-vector length `‖a_k‖` is the
/// whole correctness content: `d_k ≤ ‖a_k‖` for every cell, with equality only
/// when the cell is orthogonal along `k`, so dividing the reach by `‖a_k‖`
/// gives a *smaller* range and drops images that were needed. Nothing raises
/// when that happens — the pairs are simply not there.
#[derive(Debug, Clone)]
pub struct ImageRange {
    shifts: Vec<[i32; 3]>,
    per_axis: [i32; 3],
    reach: F,
}

impl ImageRange {
    /// Most translations an enumeration may produce before it is refused.
    ///
    /// `35937 = 33³`, i.e. a half-range of 16 on every axis. A cell needing
    /// more is being asked for a reach sixteen cells deep, which is a modelling
    /// error rather than a big system.
    pub const MAX_IMAGES: u128 = 35_937;

    /// Enumerate the translations for `reach` (Å) in `bx`.
    ///
    /// Returns `Err` — never a clamped range — when the cell is degenerate, the
    /// reach is not a usable distance, or the count exceeds
    /// [`MAX_IMAGES`](Self::MAX_IMAGES).
    pub fn new(bx: &SimBox, reach: F) -> Result<Self, GhostError> {
        if !reach.is_finite() || reach < 0.0 {
            return Err(GhostError::InvalidReach(reach));
        }
        let pbc = bx.pbc();
        let d = bx.nearest_plane_distance();
        let mut per_axis = [0_i32; 3];
        for k in 0..3 {
            if !pbc[k] {
                continue;
            }
            if !d[k].is_finite() || d[k] <= 0.0 {
                return Err(GhostError::DegenerateCell {
                    axis: k,
                    width: d[k],
                });
            }
            let n = (reach / d[k]).ceil();
            if !n.is_finite() || n > i32::MAX as F {
                return Err(GhostError::TooManyImages {
                    per_axis: [n as i32, 0, 0],
                    count: u128::MAX,
                    limit: Self::MAX_IMAGES,
                });
            }
            per_axis[k] = n as i32;
        }
        let count: u128 = per_axis
            .iter()
            .map(|&n| (2 * n as i64 + 1) as u128)
            .product();
        if count > Self::MAX_IMAGES {
            return Err(GhostError::TooManyImages {
                per_axis,
                count,
                limit: Self::MAX_IMAGES,
            });
        }

        let mut shifts = Vec::with_capacity(count as usize);
        for ix in -per_axis[0]..=per_axis[0] {
            for iy in -per_axis[1]..=per_axis[1] {
                for iz in -per_axis[2]..=per_axis[2] {
                    shifts.push([ix, iy, iz]);
                }
            }
        }
        Ok(Self {
            shifts,
            per_axis,
            reach,
        })
    }

    /// Every translation in the range, including the identity `(0, 0, 0)`.
    pub fn shifts(&self) -> &[[i32; 3]] {
        &self.shifts
    }

    /// Per-axis half-range `n_k`.
    pub fn per_axis(&self) -> [i32; 3] {
        self.per_axis
    }

    /// The reach (Å) this range was built for.
    pub fn reach(&self) -> F {
        self.reach
    }
}
