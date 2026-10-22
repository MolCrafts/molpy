//! Dimensionless 1-2 / 1-3 / … / 1-N scale weights indexed by bond distance: [`BondDistanceWeights`].

use crate::core::MolRsError;
use crate::op::F;

/// Bond-distance scale weights: slot 0 is 1-2, last slot is the 1-N tail.
///
/// Weights are dimensionless in `[0, 1]`: `0.0` fully exempts that distance,
/// `1.0` leaves it at full strength, and a fraction (Amber 1-4 `0.5`) is legal.
/// There is no [`Default`] — callers write the table;
/// [`from_exclusion_depth`](Self::from_exclusion_depth) is the named one-liner
/// for "exempt up to this hop, then 1".
///
/// The table is the Cassandra `Intra_Scaling` shape: slot `k` (0-based) is the
/// weight at bond distance `k + 1`, and the last slot is the 1-N tail for every
/// farther hop. LAMMPS `special_bonds` writes only three slots (1-2 / 1-3 / 1-4)
/// with an implicit tail of `1`; that triple is **not** a legal construction
/// here until the tail is appended.
///
/// This type is not `molrs::ff::ir::SpecialBonds`. That force-field
/// type is gated, holds separate LJ/Coulomb triples, and has no 1-N tail.
/// There is no `From` / `Into` between them.
///
/// A length-3 table is **not** a LAMMPS `special_bonds` triple. LAMMPS charmm
/// `0 0 0` is `[0, 0, 0, 1]` here; `new(vec![0.0, 0.0, 0.0])` is a zero tail
/// that exempts the rest of the connected component.
///
/// Reference: LAMMPS `special_bonds`
/// (<https://docs.lammps.org/special_bonds.html>); Cassandra `Intra_Scaling`
/// tail convention. Zero-weight distances are the hops a bonded prior already
/// owns (Boon 2017, arXiv:1710.03256, doi:10.1063/1.5029566).
///
/// # Examples
///
/// ```
/// use molrs::core::BondDistanceWeights;
///
/// let table = BondDistanceWeights::from_exclusion_depth(3);
/// assert_eq!(table.as_slice(), &[0.0, 0.0, 0.0, 1.0]);
/// assert_eq!(table.weight(0), 0.0);
/// assert_eq!(table.weight(3), 0.0);
/// assert_eq!(table.weight(4), 1.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct BondDistanceWeights(Vec<F>);

impl BondDistanceWeights {
    /// Construct a table from explicit per-distance weights.
    ///
    /// `weights[0]` is bond-distance 1 (1-2). The last entry is the 1-N tail.
    /// A zero tail is legal.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] if `weights` is empty or any entry is
    /// outside `[0, 1]` or not finite.
    pub fn new(weights: Vec<F>) -> Result<Self, MolRsError> {
        if weights.is_empty() {
            return Err(MolRsError::validation(
                "bond-distance weights must be non-empty",
            ));
        }
        for (i, &w) in weights.iter().enumerate() {
            if !(0.0..=1.0).contains(&w) {
                return Err(MolRsError::validation(format!(
                    "bond-distance weight[{i}] = {w} is outside [0, 1] or not finite"
                )));
            }
        }
        Ok(Self(weights))
    }

    /// Exempt bond distances `1..=depth` and set the 1-N tail to `1.0`.
    ///
    /// Equivalent to `vec![0.0; depth]` followed by `1.0`. Depth 3 is the
    /// Cassandra / first-order-torsion table `[0, 0, 0, 1]` (1-2 / 1-3 / 1-4
    /// exempt, 1-5+ full strength).
    pub fn from_exclusion_depth(depth: usize) -> Self {
        let mut weights = vec![0.0; depth];
        weights.push(1.0);
        Self(weights)
    }

    /// Scale weight at graph distance `distance`.
    ///
    /// Distance 0 is the atom itself and is always `0.0`. Distances past the
    /// last slot return the 1-N tail (the last entry).
    pub fn weight(&self, distance: usize) -> F {
        if distance == 0 {
            return 0.0;
        }
        self.0[(distance - 1).min(self.0.len() - 1)]
    }

    /// The stored table: index 0 is 1-2, last index is the 1-N tail.
    pub fn as_slice(&self) -> &[F] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MolRsError;

    #[test]
    fn test_new_empty_is_validation() {
        assert!(matches!(
            BondDistanceWeights::new(vec![]),
            Err(MolRsError::Validation { .. })
        ));
    }

    #[test]
    fn test_new_out_of_range_is_validation() {
        assert!(matches!(
            BondDistanceWeights::new(vec![0.0, 1.5]),
            Err(MolRsError::Validation { .. })
        ));
    }

    #[test]
    fn test_new_nan_is_validation() {
        assert!(matches!(
            BondDistanceWeights::new(vec![0.0, f64::NAN]),
            Err(MolRsError::Validation { .. })
        ));
    }

    #[test]
    fn test_new_negative_is_validation() {
        assert!(matches!(
            BondDistanceWeights::new(vec![-0.1]),
            Err(MolRsError::Validation { .. })
        ));
    }

    #[test]
    fn test_new_infinity_is_validation() {
        assert!(matches!(
            BondDistanceWeights::new(vec![0.0, f64::INFINITY]),
            Err(MolRsError::Validation { .. })
        ));
    }

    #[test]
    fn test_new_single_zero_succeeds_with_zero_tail() {
        let table = BondDistanceWeights::new(vec![0.0]).expect("zero tail is legal");
        assert_eq!(table.weight(10), 0.0);
    }

    #[test]
    fn test_new_length_three_zeros_succeeds_with_zero_tail() {
        // Zero tail: every farther hop is exempt. Not LAMMPS charmm `0 0 0`
        // (that table is [0, 0, 0, 1] here).
        let table = BondDistanceWeights::new(vec![0.0, 0.0, 0.0]).expect("zero tail is legal");
        assert_eq!(table.weight(10), 0.0);
    }

    #[test]
    fn test_new_one_then_zero_tail_succeeds() {
        let table = BondDistanceWeights::new(vec![1.0, 0.0]).expect("zero tail is legal");
        assert_eq!(table.weight(97), 0.0);
    }

    #[test]
    fn test_from_exclusion_depth_three_is_cassandra_table() {
        // Cassandra Intra_Scaling tail: last entry is 1-N. depth 3 ≡ 1-2/1-3/1-4
        // exempt, 1-5+ full strength (Boon 2017; LAMMPS charmm `0 0 0` plus the
        // implicit 1).
        assert_eq!(
            BondDistanceWeights::from_exclusion_depth(3).as_slice(),
            &[0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn test_weight_on_exclusion_depth_three_table() {
        let table = BondDistanceWeights::from_exclusion_depth(3);
        assert_eq!(table.weight(0), 0.0);
        assert_eq!(table.weight(1), 0.0);
        assert_eq!(table.weight(3), 0.0);
        assert_eq!(table.weight(4), 1.0);
        assert_eq!(table.weight(97), 1.0);
    }

    #[test]
    fn test_weight_zero_is_self_not_first_slot() {
        // weights[0] is 1-2; distance 0 is the atom itself and is always 0.
        let table = BondDistanceWeights::new(vec![1.0]).expect("unit weight is legal");
        assert_eq!(table.weight(0), 0.0);
        assert_eq!(table.weight(1), 1.0);
    }

    #[test]
    fn test_as_slice_matches_constructed_weights() {
        let table =
            BondDistanceWeights::new(vec![0.0, 0.5, 1.0]).expect("fractional 1-3 weight is legal");
        assert_eq!(table.as_slice(), &[0.0, 0.5, 1.0]);
    }
}
