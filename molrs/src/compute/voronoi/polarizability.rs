//! Finite-field molecular polarizability from Voronoi dipoles.

use molrs::op::F;
use ndarray::Array2;

use super::integrate::MolecularMoments;
use crate::compute::{Compute, ComputeError};

/// Per-molecule polarizability column `∂μ/∂E_j` (e·Å per field unit), shape
/// `(n_mol, 3)`, by central difference of the dipoles at `+E` and `−E`.
///
/// Configuration: the field magnitude `E` the two runs were made at. Data: the
/// [`MolecularMoments`] of the `+E ê_j` and `−E ê_j` runs.
///
/// Ported from the reference implementation's polarizability workflow (`src/dpol.cpp`): three external
/// field directions, each a ±E pair of Voronoi-integrated dipole sets, combined
/// by central difference. The `2.0` denominator here is the same central-finite
/// difference reference implementation uses (`dpol.cpp`, the `... / estrength ... * 2.0 ...`
/// prefactor, stripped of the reference implementation's Debye/SI unit conversions — molrs stays in
/// `e·Å` / `e·Å²·V⁻¹`-style natural units).
///
/// # Definition
///
/// `α_ij = ∂μ_i/∂E_j ≈ (μ_i(+E ê_j) − μ_i(−E ê_j)) / (2E)`.
///
/// One call handles **one** field direction `j` and returns the column
/// `∂μ/∂E_j` (a 3-vector per molecule); assemble the full 3×3 tensor from three
/// orthogonal field runs (see the test). For a linear response
/// `μ(E) = μ₀ + α E`, the central difference is exact, so the field-off run is
/// not needed.
#[derive(Debug, Clone, Copy)]
pub struct FiniteFieldPolarizability {
    field: F,
}

/// The two runs a [`FiniteFieldPolarizability`] differences.
#[derive(Clone, Copy)]
pub struct FiniteFieldPolarizabilityInput<'a> {
    /// Moments at field `+E ê_j`.
    pub plus: &'a MolecularMoments,
    /// Moments at field `−E ê_j`.
    pub minus: &'a MolecularMoments,
}

impl FiniteFieldPolarizability {
    /// Difference runs made at field magnitude `field`.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a zero or non-finite field.
    pub fn new(field: F) -> Result<Self, ComputeError> {
        if field == 0.0 || !field.is_finite() {
            return Err(ComputeError::OutOfRange {
                field: "FiniteFieldPolarizability::field",
                value: field.to_string(),
            });
        }
        Ok(Self { field })
    }

    /// The configured field magnitude.
    pub fn field(&self) -> F {
        self.field
    }
}

impl Compute for FiniteFieldPolarizability {
    type Input<'a> = FiniteFieldPolarizabilityInput<'a>;
    type Output = Array2<F>;

    /// # Errors
    /// [`ComputeError::BadShape`] when the two runs do not report the same
    /// molecules (group keys) or their dipoles are not one `3`-vector per
    /// molecule.
    fn compute(
        &self,
        input: FiniteFieldPolarizabilityInput<'_>,
    ) -> Result<Array2<F>, ComputeError> {
        let FiniteFieldPolarizabilityInput { plus, minus } = input;
        if plus.groups != minus.groups {
            return Err(ComputeError::BadShape {
                expected: format!("the same molecules in both runs, {:?}", plus.groups),
                got: format!("{:?}", minus.groups),
            });
        }
        let n = plus.groups.len();
        for dipoles in [&plus.dipoles, &minus.dipoles] {
            if dipoles.dim() != (n, 3) {
                return Err(ComputeError::BadShape {
                    expected: format!("dipoles ({n}, 3)"),
                    got: format!("{:?}", dipoles.dim()),
                });
            }
        }
        let mut col = Array2::<F>::zeros((n, 3));
        let inv = 1.0 / (2.0 * self.field);
        for m in 0..n {
            for c in 0..3 {
                col[[m, c]] = (plus.dipoles[[m, c]] - minus.dipoles[[m, c]]) * inv;
            }
        }
        Ok(col)
    }
}

crate::compute::spec::compute_spec! {
    FiniteFieldPolarizability as "FiniteFieldPolarizability";
    params |s| {
        field: Float;
    }
    inputs {
        plus: Upstream("MolecularMoments");
        minus: Upstream("MolecularMoments");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn moments(dipoles: Array2<F>) -> MolecularMoments {
        let n = dipoles.nrows();
        MolecularMoments {
            groups: (0..n as u64).collect(),
            charges: vec![0.0; n],
            dipoles,
            references: Array2::zeros((n, 3)),
        }
    }

    #[test]
    fn recovers_known_alpha_tensor() {
        // Known α (one molecule), anisotropic + with off-diagonal coupling.
        let alpha = [[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]];
        let mu0 = [0.4, -0.2, 0.05];
        let e = 1e-3;
        let polarizability = FiniteFieldPolarizability::new(e).unwrap();

        let mut recovered = [[0.0; 3]; 3];
        for j in 0..3 {
            // μ(±E ê_j) = μ0 ± E α_{:,j}
            let plus_v: Vec<F> = (0..3).map(|i| mu0[i] + e * alpha[i][j]).collect();
            let minus_v: Vec<F> = (0..3).map(|i| mu0[i] - e * alpha[i][j]).collect();
            let plus = moments(Array2::from_shape_vec((1, 3), plus_v).unwrap());
            let minus = moments(Array2::from_shape_vec((1, 3), minus_v).unwrap());
            let col = polarizability
                .compute(FiniteFieldPolarizabilityInput {
                    plus: &plus,
                    minus: &minus,
                })
                .unwrap();
            for i in 0..3 {
                recovered[i][j] = col[[0, i]];
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (recovered[i][j] - alpha[i][j]).abs() < 1e-9,
                    "α[{i}][{j}] = {} expected {}",
                    recovered[i][j],
                    alpha[i][j]
                );
            }
        }
    }

    #[test]
    fn zero_field_is_error() {
        for field in [0.0, F::NAN, F::INFINITY] {
            assert!(matches!(
                FiniteFieldPolarizability::new(field),
                Err(ComputeError::OutOfRange { .. })
            ));
        }
    }

    #[test]
    fn runs_over_different_molecules_are_an_error() {
        let plus = moments(Array2::zeros((2, 3)));
        let minus = moments(Array2::zeros((1, 3)));
        let err = FiniteFieldPolarizability::new(1e-3)
            .unwrap()
            .compute(FiniteFieldPolarizabilityInput {
                plus: &plus,
                minus: &minus,
            })
            .unwrap_err();
        assert!(matches!(err, ComputeError::BadShape { .. }));
    }
}
