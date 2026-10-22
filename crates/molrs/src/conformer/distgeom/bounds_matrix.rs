//! Symmetric distance-bounds matrix container.
//!
//! Faithful port of RDKit's `DistGeom::BoundsMatrix`
//! (`$RDBASE/Code/DistGeom/BoundsMatrix.h`, BSD-3-licensed,
//! Copyright (C) 2004-2006 Greg Landrum and other RDKit contributors).
//!
//! RDKit packs the bounds into a single `N×N` square matrix: the **upper**
//! distance bound between `i` and `j` lives in the upper triangle
//! (`getVal(min, max)`) and the **lower** bound in the lower triangle
//! (`getVal(max, min)`). We reproduce that exact storage so the matrix we
//! emit is byte-for-byte comparable with
//! `rdDistGeom.GetMoleculeBoundsMatrix`.

/// Square distance-bounds matrix with lower bounds in the lower triangle and
/// upper bounds in the upper triangle (RDKit `DistGeom::BoundsMatrix` layout).
#[derive(Debug, Clone)]
pub struct BoundsMatrix {
    n: usize,
    /// Row-major `n*n` storage, mirroring RDKit's `SquareMatrix<double>`.
    data: Vec<f64>,
}

impl BoundsMatrix {
    /// Create an `n×n` matrix initialised to `fill` everywhere.
    pub fn new(n: usize, fill: f64) -> Self {
        Self {
            n,
            data: vec![fill; n * n],
        }
    }

    /// Number of points (atoms).
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        i * self.n + j
    }

    /// Raw value at `(i, j)` (RDKit `getValUnchecked`).
    #[inline]
    pub fn raw(&self, i: usize, j: usize) -> f64 {
        self.data[self.idx(i, j)]
    }

    /// Set raw value at `(i, j)` (RDKit `setValUnchecked`).
    #[inline]
    pub fn set_raw(&mut self, i: usize, j: usize, v: f64) {
        let k = self.idx(i, j);
        self.data[k] = v;
    }

    /// Upper bound between `i` and `j` (stored in the upper triangle).
    #[inline]
    pub fn upper(&self, i: usize, j: usize) -> f64 {
        if i < j {
            self.raw(i, j)
        } else {
            self.raw(j, i)
        }
    }

    /// Lower bound between `i` and `j` (stored in the lower triangle).
    #[inline]
    pub fn lower(&self, i: usize, j: usize) -> f64 {
        if i < j {
            self.raw(j, i)
        } else {
            self.raw(i, j)
        }
    }

    /// Set the upper bound between `i` and `j`.
    #[inline]
    pub fn set_upper(&mut self, i: usize, j: usize, v: f64) {
        if i < j {
            self.set_raw(i, j, v);
        } else {
            self.set_raw(j, i, v);
        }
    }

    /// Set the lower bound between `i` and `j`.
    #[inline]
    pub fn set_lower(&mut self, i: usize, j: usize, v: f64) {
        if i < j {
            self.set_raw(j, i, v);
        } else {
            self.set_raw(i, j, v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_lives_above_the_diagonal_and_lower_below() {
        let mut b = BoundsMatrix::new(3, 0.0);
        b.set_upper(2, 0, 4.0);
        b.set_lower(2, 0, 1.0);
        assert_eq!(b.raw(0, 2), 4.0);
        assert_eq!(b.raw(2, 0), 1.0);
        assert_eq!(b.upper(0, 2), b.upper(2, 0));
        assert_eq!(b.lower(0, 2), b.lower(2, 0));
    }
}
